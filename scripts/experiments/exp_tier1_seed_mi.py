#!/usr/bin/env python3
"""Tier 1 / E3: does the a priori seed criterion predict the ablation ordering?

The criterion selects an initial colouring without fitting a classifier:

    J(l_0) = min over T in the threat set of  I( l_0(V_{T(G)}) ; Y )

with T = {identity, per-app identifier renaming}. If J ranks the colourings the
same way measured macro-F1 does, the choice of l_0 is justifiable before any
model is trained; if it does not, the criterion is wrong and should be dropped.

Mutual information over alphabets this large is upward biased, so a permutation
null is computed by shuffling the family label across apps and recomputing.
The reported signal is I minus that null; raw I is kept for inspection but is
not the quantity to compare across colourings, because the bias grows with the
alphabet and would flatter the identifier colouring exactly where it is weakest.

Env: PER_FAM (default 60), CORPUS_ROOT, MEM_GB (default 12)
"""
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import gen_wl_vectors as G
import gen_wl_renamed as REN
import gen_wl_hardened as HD

E.cap_memory(float(os.environ.get("MEM_GB", "12")))

OUT = E.out_path("exp_tier1_seed_mi.json")
PER_FAM = int(os.environ.get("PER_FAM", "60"))
MODES = ["labeled", "fuzzy", "api", "opcode", "combined"]
TRANSFORMS = ["identity", "rename"]
N_PERM = 5


def entropy_from_counts(counts):
    total = 0
    for c in counts:
        total += c
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def mutual_information(cells, n_families):
    """I(X;Y) from a {colour: [count per family]} style flat counter.

    cells maps (colour_hash * n_families + family_index) -> count, which keeps
    one integer key per observed pair rather than a list per colour.
    """
    per_colour = {}
    per_family = [0] * n_families
    total = 0
    for key, cnt in cells.items():
        colour = key // n_families
        fam = key % n_families
        if colour not in per_colour:
            per_colour[colour] = [0] * n_families
        per_colour[colour][fam] += cnt
        per_family[fam] += cnt
        total += cnt
    if total == 0:
        return 0.0, 0
    h_y = entropy_from_counts(per_family)
    h_y_given_x = 0.0
    for colour in per_colour:
        row = per_colour[colour]
        n_c = 0
        for v in row:
            n_c += v
        h_y_given_x += (n_c / total) * entropy_from_counts(row)
    return h_y - h_y_given_x, len(per_colour)


def main():
    samples = E.list_samples(per_family=PER_FAM, require_opcodes=True)
    fam_set = set()
    for s in samples:
        fam_set.add(s[0])
    fams = sorted(fam_set)
    fam_idx = {}
    for i, f in enumerate(fams):
        fam_idx[f] = i
    n_fam = len(fams)
    print(f"[*] {len(samples)} apps over {n_fam} families: {fams}", flush=True)

    # per (mode, transform), a flat colour-by-family counter; plus, per app, the
    # colour histogram needed to recompute MI under a permuted label assignment
    cells = {}
    app_hists = {}
    for mode in MODES:
        for tf in TRANSFORMS:
            cells[(mode, tf)] = Counter()
            app_hists[(mode, tf)] = []

    for n, (fam, sha, path) in enumerate(samples):
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception as exc:
            print(f"    [skip] {sha[:12]} {exc}", flush=True)
            continue
        opf = os.path.join(os.path.dirname(path), sha + ".method_opcodes.json")
        if os.path.isfile(opf):
            opmap = json.load(open(opf))
        else:
            opmap = None

        raw_labels = E.kept_labels(nl, "fcg")
        del nl
        if not raw_labels:
            continue

        for tf in TRANSFORMS:
            if tf == "identity":
                labels = raw_labels
                use_opmap = opmap
            else:
                seed = sha[:16]
                labels = {}
                for nid in raw_labels:
                    labels[nid] = REN.rename_label(raw_labels[nid], seed)
                # Re-key the opcode map to the renamed signatures: identifier
                # renaming changes method names, not method bodies, so opcode
                # histograms are invariant. Without this the lookup misses on the
                # renamed name and opcode/combined spuriously collapse.
                if opmap is not None:
                    use_opmap = {}
                    for nid in raw_labels:
                        ok = HD._method_of(G.canonical_label(raw_labels[nid], "fcg"))
                        if ok in opmap:
                            use_opmap[HD._method_of(G.canonical_label(labels[nid], "fcg"))] = opmap[ok]
                else:
                    use_opmap = None

            for mode in MODES:
                if mode in ("opcode", "combined") and opmap is None:
                    continue
                try:
                    colors = E.recolour(labels, mode, su, use_opmap, "fcg")
                except Exception:
                    continue
                hist = Counter()
                for nid in colors:
                    hist[G.stable_hash_u32(colors[nid])] += 1
                del colors
                key = (mode, tf)
                for ch, cnt in hist.items():
                    cells[key][ch * n_fam + fam_idx[fam]] += cnt
                app_hists[key].append((fam, hist))
            if tf != "identity":
                del labels

        del raw_labels, su, pr
        if (n + 1) % 25 == 0:
            print(f"    {n + 1}/{len(samples)}", flush=True)

    # MI per colouring and transform, against a label-permutation null
    rng = random.Random(42)
    result = {"families": fams, "apps": len(samples), "n_permutations": N_PERM,
              "modes": {}}

    for mode in MODES:
        entry = {}
        for tf in TRANSFORMS:
            key = (mode, tf)
            if not cells[key]:
                continue
            info, n_colours = mutual_information(cells[key], n_fam)

            # permutation null: reassign each app's family, keep its histogram
            null_vals = []
            hists = app_hists[key]
            fam_list = []
            for h in hists:
                fam_list.append(h[0])
            for _ in range(N_PERM):
                shuffled = fam_list[:]
                rng.shuffle(shuffled)
                perm_cells = Counter()
                for idx in range(min(len(hists), len(shuffled))):
                    orig_fam, hist = hists[idx]
                    new_fam = shuffled[idx]
                    fi = fam_idx[new_fam]
                    for ch, cnt in hist.items():
                        perm_cells[ch * n_fam + fi] += cnt
                nv, _ = mutual_information(perm_cells, n_fam)
                null_vals.append(nv)
                del perm_cells
            null_mean = sum(null_vals) / len(null_vals)
            entry[tf] = {
                "I_bits": round(info, 4),
                "I_null_bits": round(null_mean, 4),
                "I_signal_bits": round(info - null_mean, 4),
                "distinct_colours": n_colours,
            }
        if entry:
            sig = []
            for tf in entry:
                sig.append(entry[tf]["I_signal_bits"])
            entry["J_maximin_bits"] = round(min(sig), 4)
            result["modes"][mode] = entry

    # save and print the ranking
    json.dump(result, open(OUT, "w"), indent=2)

    print(f"\n[E3] seed criterion, {len(samples)} apps")
    print(f"  {'colouring':11}{'I id':>9}{'I ren':>9}{'null':>9}{'J':>9}{'colours':>11}")
    ranked = []
    for mode in result["modes"]:
        m = result["modes"][mode]
        ident = m.get("identity", {})
        ren = m.get("rename", {})
        print(f"  {mode:11}{ident.get('I_signal_bits', 0):>9.4f}"
              f"{ren.get('I_signal_bits', 0):>9.4f}"
              f"{ident.get('I_null_bits', 0):>9.4f}"
              f"{m['J_maximin_bits']:>9.4f}{ident.get('distinct_colours', 0):>11d}")
        ranked.append((m["J_maximin_bits"], mode))
    ranked.sort(reverse=True)
    order = []
    for _, mode in ranked:
        order.append(mode)
    print(f"\n  predicted ordering by J: {' > '.join(order)}")
    print(f"  compare against measured macro-F1 in results/ablation_hardened_cv.json")
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
