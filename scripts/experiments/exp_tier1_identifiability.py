#!/usr/bin/env python3
"""Tier 1: E1, E2, E4, E6 - one streaming pass over the corpus.

E1  Does the isolated fraction follow p = exp(-M_d/d)?
E2  Is d* set by the host application rather than the target? The quantity that
    decides this is the occupancy ratio rho = |V_target|/|V_filtered|, not a
    regression of d* on M, which is circular because d* is defined as M/ln(1/p).
E4  Do lower-entropy initial colourings need a smaller d?
E6  Are the two necessary conditions separable: dimension bound (d >~ d*) and
    identifier stability (signature recurrence across samples)?

Parsing dominates the cost, so every measurement that needs a parsed graph is
taken in the same pass. Output is JSONL and the pass is resumable: rerunning
skips apps already recorded.

Env: PER_FAM (default all), COLOURINGS (default labeled,fuzzy,api),
     CORPUS_ROOT, MEM_GB (default 12)
"""
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import gen_wl_vectors as G

E.cap_memory(float(os.environ.get("MEM_GB", "12")))

# ID_TAG / ID_TRUTH (opt-in, #1756): ID_TRUTH=label_audit replaces the SDK targets with exp_label_audit.SDK (the
# reverse-engineered namespaces of all eight corpus-v3 classes), and ID_TAG names separate output files, so
# the default run and its outputs (used by the robustness paper) are unchanged
if os.environ.get("ID_TAG"):
    _TAG = "_" + os.environ["ID_TAG"]
else:
    _TAG = ""
if os.environ.get("ID_TRUTH") == "label_audit":
    import exp_label_audit as _A
    _sdk_prefixes = {}
    for _f in _A.FAMS_V3:
        _sdk_prefixes[_f] = tuple(_A.SDK[_f])
    E.SDK_PREFIXES = _sdk_prefixes
OUT_JSONL = E.out_path(f"exp_tier1_identifiability{_TAG}.jsonl")
OUT_SUMMARY = E.out_path(f"exp_tier1_identifiability{_TAG}.json")
# E6 recurrence needs a corpus-wide signature counter. Holding it in the JSONL
# would add gigabytes of duplicated method names, so it lives beside the record
# file (per dataset) and is checkpointed so a resumed run does not lose it.
OUT_SIGS = E.out_path(f"exp_tier1_sigcounts{_TAG}.json")
DIMS = []
for _k in range(10, 22):
    DIMS.append(2 ** _k)
ITERS = 2
if os.environ.get("PER_FAM"):
    PER_FAM = int(os.environ["PER_FAM"])
else:
    PER_FAM = None
COLOURINGS = os.environ.get("COLOURINGS", "labeled,fuzzy,api").split(",")

# extra namespace targets, so the proxy SDKs sit on the same axes as ordinary
# third-party libraries rather than being measured in isolation
try:
    from tpl_filter_experiment import TPL_EXTRA
    TPL_TARGETS = list(TPL_EXTRA)
except Exception:
    TPL_TARGETS = []


def main():
    # apps already recorded (resume)
    done = set()
    if os.path.isfile(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done.add(json.loads(line)["sha"])
                except Exception:
                    continue
        print(f"[*] resuming, {len(done)} apps already recorded", flush=True)

    samples = E.list_samples(per_family=PER_FAM, require_opcodes=False)
    todo = []
    for fam, sha, path in samples:
        if sha not in done:
            todo.append((fam, sha, path))
    print(f"[*] {len(samples)} samples, {len(todo)} to process", flush=True)

    sig_counts = defaultdict(Counter)
    if os.path.isfile(OUT_SIGS):
        raw = json.load(open(OUT_SIGS))
        for tname in raw:
            for h, c in raw[tname].items():
                sig_counts[tname][int(h)] = c
        print(f"[*] restored signature counts for {len(sig_counts)} targets", flush=True)

    # one streaming pass: parse each app once, measure every target and colouring
    sink = open(OUT_JSONL, "a")
    t0 = time.time()
    n = 0
    for fam, sha, path in todo:
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception as exc:
            print(f"    [skip] {sha[:12]} {exc}", flush=True)
            continue

        labels = E.kept_labels(nl, "fcg")
        del nl
        if not labels:
            continue
        n_kept = len(labels)

        opmap = None
        if "combined" in COLOURINGS or "opcode" in COLOURINGS:
            opf = os.path.join(os.path.dirname(path), sha + ".method_opcodes.json")
            if os.path.isfile(opf):
                opmap = json.load(open(opf))

        # every namespace target present in this app, its own SDK included
        targets = []
        if fam in E.SDK_PREFIXES:
            targets.append(("SDK:" + fam, E.SDK_PREFIXES.get(fam, ())))
        for t in TPL_TARGETS:
            targets.append((t, (t,)))

        rec = {"sha": sha, "family": fam, "n_kept": n_kept, "targets": {}}

        for tname, prefixes in targets:
            tgt_nodes = E.target_nodes(labels, prefixes)
            if len(tgt_nodes) < 5:
                continue
            entry = {
                "n_target": len(tgt_nodes),
                "rho": round(len(tgt_nodes) / n_kept, 6),
                "colourings": {},
            }
            # E6: count each distinct target signature once per app it appears in
            seen_here = set()
            for nid in tgt_nodes:
                seen_here.add(labels[nid])
            for sig in seen_here:
                sig_counts[tname][G.stable_hash_u32(sig)] += 1

            for mode in COLOURINGS:
                if mode in ("opcode", "combined") and opmap is None:
                    continue
                try:
                    colors = E.recolour(labels, mode, su, opmap, "fcg")
                except Exception:
                    continue
                tgt_h, non_h, n_inst = E.token_provenance(colors, su, pr, tgt_nodes, ITERS)
                del colors
                if not tgt_h:
                    continue
                curve = E.isolation_curve(tgt_h, non_h, DIMS)
                m_d = len(non_h)
                pred = {}
                for d in DIMS:
                    pred[str(d)] = round(math.exp(-m_d / d), 6)
                measured = {}
                for d, v in curve.items():
                    measured[str(d)] = round(v, 6)
                entry["colourings"][mode] = {
                    "M_distinct": m_d,
                    "M_instances": n_inst,
                    "n_target_tokens": len(tgt_h),
                    "shared_tokens": len(tgt_h & non_h),
                    "d_star_90": int(m_d / math.log(1 / 0.9)),
                    "measured": measured,
                    "predicted": pred,
                }
                del tgt_h, non_h
            if entry["colourings"]:
                rec["targets"][tname] = entry

        sink.write(json.dumps(rec) + "\n")
        sink.flush()
        del labels, su, pr
        n += 1
        if n % 200 == 0:
            flush_sigs(sig_counts)
        if n % 25 == 0:
            rate = n / (time.time() - t0)
            print(f"    {n}/{len(todo)}  {rate:.2f} app/s  "
                  f"eta {(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    sink.close()
    flush_sigs(sig_counts)
    summarize()


def flush_sigs(sig_counts):
    dump = {}
    for tname in sig_counts:
        inner = {}
        for h, c in sig_counts[tname].items():
            inner[str(h)] = c
        dump[tname] = inner
    tmp = OUT_SIGS + ".tmp"
    json.dump(dump, open(tmp, "w"))
    os.replace(tmp, OUT_SIGS)


def summarize():
    """Reduce the JSONL to the four experiment answers."""
    per_target_sigs = defaultdict(Counter)
    if os.path.isfile(OUT_SIGS):
        raw = json.load(open(OUT_SIGS))
        for tname in raw:
            for h, c in raw[tname].items():
                per_target_sigs[tname][int(h)] = c
    per_target_apps = Counter()
    rows = []
    with open(OUT_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for tname, entry in rec["targets"].items():
                per_target_apps[tname] += 1
                rows.append((rec["family"], rec["n_kept"], tname, entry))

    # E1: measured versus predicted isolation, pooled by dimension
    e1 = {}
    for d in DIMS:
        errs = []
        for fam, n_kept, tname, entry in rows:
            c = entry["colourings"].get("labeled")
            if not c:
                continue
            m = c["measured"].get(str(d))
            p = c["predicted"].get(str(d))
            if m is None or p is None:
                continue
            errs.append(m - p)
        if errs:
            errs.sort()
            mean = sum(errs) / len(errs)
            abs_errs = []
            for x in errs:
                abs_errs.append(abs(x))
            e1[str(d)] = {
                "n": len(errs),
                "mean_signed_error": round(mean, 5),
                "median_signed_error": round(errs[len(errs) // 2], 5),
                "max_abs_error": round(max(abs_errs), 5),
            }

    # E2: occupancy ratio, the only route by which the target moves d*
    rhos = []
    for fam, n_kept, tname, entry in rows:
        rhos.append(entry["rho"])
    rhos.sort()
    e2 = {}
    if rhos:
        if rhos[-1] < 1:
            spread = round(1.0 / (1.0 - rhos[-1]), 4)
        else:
            spread = None
        e2 = {
            "n": len(rhos),
            "rho_median": round(rhos[len(rhos) // 2], 5),
            "rho_p95": round(rhos[int(0.95 * (len(rhos) - 1))], 5),
            "rho_max": round(rhos[-1], 5),
            "max_dstar_spread_factor": spread,
        }

    # E4: does a lower-entropy colouring reduce M_distinct, hence d*?
    e4 = {}
    by_mode = defaultdict(list)
    for fam, n_kept, tname, entry in rows:
        for mode, c in entry["colourings"].items():
            by_mode[mode].append(c["M_distinct"])
    base = by_mode.get("labeled")
    for mode in by_mode:
        vals = sorted(by_mode[mode])
        med = vals[len(vals) // 2]
        e4[mode] = {"n": len(vals), "M_distinct_median": med,
                    "d_star_90_median": int(med / math.log(1 / 0.9))}
    if base:
        b = sorted(base)[len(base) // 2]
        for mode in e4:
            e4[mode]["ratio_to_labeled"] = round(e4[mode]["M_distinct_median"] / b, 4)

    # E6: recurrence per target, the second necessary condition
    e6 = {}
    for tname in per_target_apps:
        counts = per_target_sigs[tname]
        if not counts:
            continue
        rec_n = 0
        for s in counts:
            if counts[s] >= 2:
                rec_n += 1
        e6[tname] = {"apps": per_target_apps[tname],
                     "distinct_sigs": len(counts),
                     "recurrence": round(rec_n / len(counts), 4)}

    # write summary and print the four answers
    out = {"dims": DIMS, "iters": ITERS, "colourings": COLOURINGS,
           "E1_isolation_vs_prediction": e1,
           "E2_occupancy_ratio": e2,
           "E4_colouring_vs_M_distinct": e4,
           "E6_recurrence_by_target": e6}
    json.dump(out, open(OUT_SUMMARY, "w"), indent=2)

    print(f"\n[E1] measured minus predicted isolation, labeled colouring")
    print(f"  {'dim':>9}{'n':>7}{'mean err':>11}{'max |err|':>11}")
    for d in DIMS:
        if str(d) in e1:
            r = e1[str(d)]
            print(f"  {d:>9}{r['n']:>7}{r['mean_signed_error']:>11.4f}{r['max_abs_error']:>11.4f}")
    print(f"\n[E2] occupancy ratio rho over {e2.get('n', 0)} (app, target) pairs")
    print(f"  median {e2.get('rho_median')}  p95 {e2.get('rho_p95')}  max {e2.get('rho_max')}")
    print(f"  worst-case d* spread across all targets: {e2.get('max_dstar_spread_factor')}x")
    print(f"\n[E4] median M_distinct by initial colouring")
    for mode in e4:
        print(f"  {mode:10} {e4[mode]['M_distinct_median']:>9}  "
              f"d*90 {e4[mode]['d_star_90_median']:>9}  "
              f"ratio {e4[mode].get('ratio_to_labeled')}")
    print(f"\n[E6] recurrence spread over {len(e6)} targets")
    vals = []
    for t in e6:
        vals.append(e6[t]["recurrence"])
    vals.sort()
    if vals:
        print(f"  min {vals[0]:.3f}  median {vals[len(vals)//2]:.3f}  max {vals[-1]:.3f}")
    print(f"\nsaved {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
