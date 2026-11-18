#!/usr/bin/env python3
"""R4: pipeline sensitivity to semantics-preserving structural perturbation, by locale.

For each staged proxy app, apply each (op, locale, budget, seed) from
graph_perturb, re-vectorize the perturbed FCG in four WL colourings, and measure
against a per-colouring linear model trained on the clean packed vectors:
  disp     ||phi(G') - phi(G)||_2   (L2-normalized vectors)
  margin   min_{c != y} pairwise score gap for the true class y
  retention margin(G') / margin(G)   (clean-correct apps only)
  flip     argmax(G') != argmax(G)

Colourings span the accuracy/robustness axis: labeled (name-based, most accurate),
fuzzy and api (renaming-invariant), structure (topology only). The certificate
(theory.tex) predicts labeled is aimable and fragile once tokens can be steered,
invariant colourings deny aim; structural ops move all colourings.

Question Q2: for a fixed budget, does a perturbation inside the SDK subgraph move
the prediction more than one in the host? Locales sdk / host / boundary answer it.
`flatten` (locale sdk/host/all) is the structural-destruction bound.

Output: results/tier4/r4_perturb.jsonl (resume-safe per app), summary via --summarize.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD
import gen_wl_structure as ST
import graph_perturb as GP
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "r4_perturb.jsonl")
COLOURS = ["labeled", "fuzzy", "api", "structure"]
PACK = {"labeled": "fcg_wl_1024", "fuzzy": "fcg_wlfz_1024", "api": "fcg_wlapi_1024", "structure": "fcg_wls_1024"}
OPS = ["wrapper", "indirection", "dead_node", "phantom_edge"]
LOCALES = ["sdk", "host", "boundary"]
BUDGETS = [10, 50, 200]
SEEDS = [0, 1]


def vec(colour, nl, su, pr):
    if colour == "labeled":
        return G.wl_hashed_vector(nl, su, pr, "fcg", 1024, iters=2, drop_lib=True, use_direction=True)
    if colour == "structure":
        return ST.wl_structure_vector(nl, su, pr, 1024)
    lbl = {}
    for n, r in nl.items():
        lbl[n] = G.canonical_label(r, "fcg")
    cols = {}
    if colour == "fuzzy":
        for n in lbl:
            cols[n] = HD.fuzzy_label(lbl[n])
    else:
        for n in lbl:
            cols[n] = HD.api_label(n, lbl, su)
    return HD.wl_vector(cols, su, pr, 1024, 2)


def margin(model, x, y):
    W, b, classes = model
    c = classes.index(y)
    s = (W[c] - W) @ x + (b[c] - b)
    s[c] = np.inf
    pred = int(np.argmax(W @ x + b))
    return float(s.min()), pred == c, pred


def worker(task):
    fam, sha = task
    models = _MODELS
    t0 = time.time()
    try:
        nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
    except Exception as e:
        return [{"family": fam, "sha": sha, "error": repr(e)[:120]}]
    if fam in P.SDK_PREFIXES:
        sdk = P.SDK_PREFIXES[fam]
    else:
        sdk = ()
    part = GP.partition(nl, sdk)
    clean = {}
    for c in COLOURS:
        clean[c] = vec(c, nl, su, pr)
    base = {}
    for c in COLOURS:
        mgn, correct, pred = margin(models[c], clean[c], fam)
        base[c] = {"margin": mgn, "correct": correct, "pred": pred}
    out = []
    # perturbation grid: op x locale x budget x seed, then the three flatten locales
    grid = []
    for op in OPS:
        for loc in LOCALES:
            for bud in BUDGETS:
                for sd in SEEDS:
                    grid.append((op, loc, bud, sd))
    for loc in ("sdk", "host", "all"):
        grid.append(("flatten", loc, 0, 0))
    for op, loc, bud, sd in grid:
        try:
            L, S, Pp = GP.apply(nl, su, pr, sdk, op, loc, bud, sd)
        except Exception as e:
            out.append({"family": fam, "sha": sha, "op": op, "locale": loc, "budget": bud,
                        "seed": sd, "error": repr(e)[:100]})
            continue
        added = len(L) - len(nl)
        rec = {"family": fam, "sha": sha, "op": op, "locale": loc, "budget": bud, "seed": sd,
               "nodes": len(nl), "sdk_nodes": len(part["sdk"]), "host_nodes": len(part["host"]),
               "added_nodes": added, "col": {}}
        for c in COLOURS:
            v2 = vec(c, L, S, Pp)
            mgn, correct, pred = margin(models[c], v2, fam)
            if base[c]["correct"] and base[c]["margin"] > 0:
                retention = mgn / base[c]["margin"]
            else:
                retention = None
            rec["col"][c] = {"disp": float(np.linalg.norm(v2 - clean[c])),
                             "margin": mgn, "flip": bool(pred != base[c]["pred"]),
                             "lost_correct": bool(base[c]["correct"] and not correct),
                             "retention": retention}
        out.append(rec)
        del L, S, Pp, v2  # otherwise the previous perturbed copy stays alive while the next one is built
    out.append({"family": fam, "sha": sha, "baseline": base, "seconds": round(time.time() - t0, 1)})
    return out


_MODELS = None


def _init(staged):
    """One linear SGD per colouring, on clean packed proxy vectors minus staged apps."""
    global _MODELS
    staged_shas = set(staged)
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    models = {}
    for colour in COLOURS:
        z = np.load(os.path.join(H.VEC, PACK[colour] + ".npz"))
        keep_list = []
        for i, f in zip(z["ids"], z["family"]):
            keep_list.append(f in H.PROXY and i not in staged_shas)
        keep = np.array(keep_list)
        X = z["X"][keep]
        fam = z["family"][keep]
        le = LabelEncoder()
        le.fit(fam)
        y = le.transform(fam)
        # simple upsample to the majority count
        import random
        rng = random.Random(42)
        by = defaultdict(list)
        for j, c in enumerate(y):
            by[c].append(j)
        sizes = []
        for v in by.values():
            sizes.append(len(v))
        top = max(sizes)
        idx = []
        for v in by.values():
            idx += v
            for _ in range(top - len(v)):
                idx.append(rng.choice(v))
        idx = np.array(idx)
        m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
        m.fit(X[idx], y[idx])
        models[colour] = (m.coef_.astype(np.float64), m.intercept_.astype(np.float64), list(le.classes_))
    _MODELS = models


def summarize():
    rows = []
    for l in open(OUT):
        rows.append(json.loads(l))
    per = []
    for r in rows:
        if "op" in r and "error" not in r:
            per.append(r)
    # Q1: margin retention by op x colour, averaged over apps/locales/budgets
    print("\n=== retention (margin kept) by op x colour, budget=50, locale=sdk ===")
    hdr = f"{'op':13}"
    for c in COLOURS:
        hdr += f"{c:>11}"
    print(hdr)
    for op in OPS:
        cells = []
        for c in COLOURS:
            vals = []
            for r in per:
                if r["op"] == op and r["locale"] == "sdk" and r["budget"] == 50 and r["col"][c]["retention"] is not None:
                    vals.append(r["col"][c]["retention"])
            if vals:
                cells.append(f"{np.mean(vals):10.3f}")
            else:
                cells.append(f"{'-':>10} ")
        print(f"{op:13}" + "".join(cells))
    # Q2: locale effect. retention by locale at fixed budget, per colour
    print("\n=== SDK vs host vs boundary: mean retention (budget=50, avg over ops) ===")
    hdr = f"{'colour':11}"
    for l in LOCALES:
        hdr += f"{l:>10}"
    print(hdr)
    for c in COLOURS:
        cells = []
        for loc in LOCALES:
            vals = []
            for r in per:
                if r["locale"] == loc and r["budget"] == 50 and r["op"] in OPS and r["col"][c]["retention"] is not None:
                    vals.append(r["col"][c]["retention"])
            if vals:
                cells.append(f"{np.mean(vals):9.3f}")
            else:
                cells.append(f"{'-':>9} ")
        print(f"{c:11}" + "".join(cells))
    # flip rate vs budget by colour (sdk locale)
    print("\n=== flip rate vs budget, locale=sdk, avg over ops ===")
    hdr = f"{'colour':11}"
    for b in BUDGETS:
        hdr += f"{b:>8}"
    print(hdr)
    for c in COLOURS:
        cells = []
        for b in BUDGETS:
            vals = []
            for r in per:
                if r["locale"] == "sdk" and r["budget"] == b and r["op"] in OPS:
                    vals.append(r["col"][c]["flip"])
            if vals:
                cells.append(f"{np.mean(vals):7.2f}")
            else:
                cells.append(f"{'-':>7} ")
        print(f"{c:11}" + "".join(cells))
    # flatten
    print("\n=== flatten (structural destruction): mean retention & flip ===")
    for loc in ("sdk", "host", "all"):
        for c in COLOURS:
            vals = []
            for r in per:
                if r["op"] == "flatten" and r["locale"] == loc:
                    vals.append(r["col"][c])
            ret = []
            fl = []
            disps = []
            for x in vals:
                if x["retention"] is not None:
                    ret.append(x["retention"])
                fl.append(x["flip"])
                disps.append(x["disp"])
            if fl:
                if ret:
                    mean_ret = np.mean(ret)
                else:
                    mean_ret = float('nan')
                print(f"  flatten/{loc:4} {c:10} retention={mean_ret:.3f} flip={np.mean(fl):.2f} disp={np.mean(disps):.3f}")
    summ = {"per_op_colour_retention": {}, "locale_retention": {}, "flip_vs_budget": {}}
    json.dump(summ, open(OUT.replace(".jsonl", "_summary.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    import random
    apps = []
    for fam in H.PROXY:
        shas = H.staged(fam)
        random.Random(42).shuffle(shas)
        for s in shas[:a.per_family]:
            apps.append((fam, s))
    staged = []
    for _, s in apps:
        staged.append(s)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if r.get("seconds") is not None:
                done.add((r["family"], r["sha"]))
    remaining = []
    for t in apps:
        if t not in done:
            remaining.append(t)
    apps = remaining
    print(f"{len(apps)} apps to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(staged,)) as ex, open(OUT, "a") as f:
        futs = {}
        for t in apps:
            futs[ex.submit(worker, t)] = t
        for k, fu in enumerate(as_completed(futs), 1):
            for rec in fu.result():
                f.write(json.dumps(rec) + "\n")
            f.flush()
            t = futs[fu]
            print(f"[{k}/{len(apps)}] {t[0]} {t[1][:10]} done", flush=True)
    summarize()


if __name__ == "__main__":
    main()
