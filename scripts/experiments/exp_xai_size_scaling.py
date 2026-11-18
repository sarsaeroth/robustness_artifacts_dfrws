#!/usr/bin/env python3
"""Size scaling of GIN node attribution (exact decomposition vs GNNExplainer).

Recomputes, from the saved 133-graph per-graph precisions of exp_xai_gnnexplainer.py,
how localization varies with graph size: Spearman rank correlation of precision with
log(n) for each method, and the exact/GNNExplainer advantage by size band. Graph order
is reproduced exactly as in exp_xai_gnnexplainer.py's "use all" path: proxy families,
>=5 target nodes, sorted by (family, sha). The tier-3 graph cache was purged, so n and
the target count are recomputed from FCGs staged by stage_graphs_from_bundle.py (same
seeded 60-per-family selection, same kept-label and SDK-prefix rules as
exp_tier3_gnn.prep). Alignment is validated per graph: target/n must equal the saved
per-graph base rate.

Usage: CORPUS_ROOT=work/stage_tier3 exp_xai_size_scaling.py
Output: results/exp_xai_size_scaling.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import gen_wl_vectors as G
import exp_tier3_gnn as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "exp_xai_gnnexplainer.json")
OUT = os.path.join(ROOT, "results", "exp_xai_size_scaling.json")
BANDS = [(0, 5000), (5000, 20000), (20000, 80000), (80000, None)]


def graph_meta():
    meta = []
    for fam, sha, path in E.list_samples(per_family=M.PER_FAM):
        if fam == "non_proxy":
            continue
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception:
            continue
        labels = E.kept_labels(nl, "fcg")
        del nl, su, pr
        if len(labels) < 50:
            continue
        prefixes = E.SDK_PREFIXES.get(fam, ())
        prefix_tuple = tuple(prefixes)
        tgt = 0
        for l in labels.values():
            if l.startswith(prefix_tuple):
                tgt += 1
        if tgt >= 5:
            meta.append({"sha": sha, "family": fam, "n": len(labels), "target": tgt})
    meta.sort(key=lambda g: (g["family"], g["sha"]))
    return meta


def main():
    src = json.load(open(SRC))
    pg = src["per_graph"]
    meta = graph_meta()
    if len(meta) != len(pg["exact"]):
        sys.exit(f"order mismatch: {len(meta)} cached graphs vs {len(pg['exact'])} results")
    sizes = []
    for g in meta:
        sizes.append(g["n"])
    n = np.array(sizes, dtype=float)
    ex = np.array(pg["exact"], dtype=float)
    gx = np.array(pg["gnnexplainer"], dtype=float)
    base = np.array(pg["base"], dtype=float)
    rates = []
    for g in meta:
        rates.append(g["target"] / g["n"])
    recomputed = np.array(rates)
    err = np.abs(recomputed - base).max()
    if err > 1e-4:
        sys.exit(f"alignment failed: max |target/n - saved base| = {err:.4g}")
    print(f"alignment OK: target/n matches saved base rate on all {len(meta)} graphs "
          f"(max err {err:.1e})")

    res = {"n_graphs": len(meta),
           "nodes_min": int(n.min()), "nodes_median": int(np.median(n)), "nodes_max": int(n.max())}
    for name, v in (("exact", ex), ("gnnexplainer", gx), ("base", base)):
        r = spearmanr(np.log(n), v)
        res[f"spearman_logn_{name}"] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    # advantage relative to chance, per size band (ratio of means, robust to zeros)
    res["bands"] = []
    for lo, hi in BANDS:
        # band mask; an open upper bound (hi None) keeps everything >= lo
        if hi:
            m = (n >= lo) & (n < hi)
        else:
            m = (n >= lo) & True
        if m.sum() == 0:
            continue
        row = {"lo": lo, "hi": hi, "count": int(m.sum()),
               "exact_mean": float(ex[m].mean()), "gnnexplainer_mean": float(gx[m].mean()),
               "base_mean": float(base[m].mean()),
               "exact_over_base": float(ex[m].mean() / base[m].mean()),
               "gnnexplainer_over_base": float(gx[m].mean() / base[m].mean())}
        if gx[m].mean() > 0:
            row["exact_over_gnnexplainer"] = float(ex[m].mean() / gx[m].mean())
        else:
            row["exact_over_gnnexplainer"] = None
        res["bands"].append(row)
    json.dump(res, open(OUT, "w"), indent=1)

    print(f"{res['n_graphs']} graphs, nodes {res['nodes_min']}..{res['nodes_max']} "
          f"(median {res['nodes_median']})")
    for name in ("exact", "gnnexplainer", "base"):
        s = res[f"spearman_logn_{name}"]
        print(f"  spearman(log n, {name:12}) = {s['rho']:+.3f}  p={s['p']:.1e}")
    print(f"  {'band':>16} {'k':>4} {'exact':>7} {'gnnx':>7} {'base':>7} {'ex/base':>8} {'gx/base':>8} {'ex/gx':>7}")
    for b in res["bands"]:
        if b["hi"]:
            hi_text = b["hi"]
        else:
            hi_text = "inf"
        lab = f"{b['lo']}-{hi_text}"
        if b["exact_over_gnnexplainer"]:
            exgx = f"{b['exact_over_gnnexplainer']:.1f}"
        else:
            exgx = "inf"
        print(f"  {lab:>16} {b['count']:>4} {b['exact_mean']:7.3f} {b['gnnexplainer_mean']:7.3f} "
              f"{b['base_mean']:7.3f} {b['exact_over_base']:8.2f} {b['gnnexplainer_over_base']:8.2f} {exgx:>7}")


if __name__ == "__main__":
    main()
