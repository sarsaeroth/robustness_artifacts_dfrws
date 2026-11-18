#!/usr/bin/env python3
"""Plan v3, block D: localization against known SDK ground truth, and its stability.

  D18  Exact-token method localization vs the known SDK method set (PL.SDK_PREFIXES).
       Train logistic on exact identifier tokens; for each held-out app rank its methods by
       their contribution to the correct-vs-rival score gap and score the ranking against
       SDK membership with precision@|SDK|, average precision, and AUROC (above base rate).
       Turns the abstract "localization ceiling" into localization vs a labelled target.
  D16  Pair the collision-free ceiling (exact tokens, X3) with the deployed hashed-explainer
       precision (colour_localize.json, labeled|clean) per family: how much of the gap is
       hashing collisions versus signal that is simply not in the SDK.
  D19  Attribution rank agreement across training seeds: Spearman of per-method contribution
       vectors between seeds, median over held-out apps. Whether localization is stable
       enough to report at all.

Output: results/tier4/v3/localize_truth.json
"""
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_tok_analyses as T

OUT = H.out_path("tier4", "v3", "localize_truth.json")
N_SEED = 5


def method_contrib(W, cls, Xj_dense, tok_cols, tid, morg, n_meth):
    """Per-method contribution to the correct-vs-rival gap for one app."""
    scores = W @ Xj_dense
    scores[cls] = -1e9
    rival = int(np.argmax(scores))
    u = W[cls] - W[rival]
    contrib = np.zeros(len(tid))
    keep = tok_cols >= 0
    contrib[keep] = u[tok_cols[keep]]
    mc = np.zeros(n_meth)
    np.add.at(mc, morg[keep], contrib[keep])
    return mc


def run(apps, data):
    fam_set = set()
    for f, _ in apps:
        fam_set.add(f)
    fam_names = sorted(fam_set)
    shalist = []
    for _, s in apps:
        if s in data:
            shalist.append(s)
    fam_list = []
    for s in shalist:
        fam_list.append(data[s][0])
    fam = np.array(fam_list)
    y_list = []
    for f in fam:
        y_list.append(fam_names.index(f))
    y = np.array(y_list)
    g = H.groups_for(np.array(shalist))
    X, vocab = T.count_matrix(shalist, data)
    Xl = X.copy()
    Xl.data = np.log1p(Xl.data)
    tok_pos = {}
    for i, t in enumerate(vocab):
        tok_pos[int(t)] = i
    # precompute per-app token->column maps (-1: token not in vocabulary)
    tcol = {}
    for s in shalist:
        tid = data[s][1][0]
        cols = []
        for t in tid:
            if int(t) in tok_pos:
                cols.append(tok_pos[int(t)])
            else:
                cols.append(-1)
        tcol[s] = np.array(cols)

    d18 = {}
    for f in fam_names:
        d18[f] = defaultdict(list)
    agree = []
    for r, k, tr, te in H.folds(y, g, 1):
        per_seed_mc = {}
        for j in te:
            per_seed_mc[j] = []
        for seed in range(N_SEED):
            up = H.upsample(tr, y, 1000 + seed)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=3000, tol=1e-5,
                              random_state=seed).fit(Xl[up], y[up])
            W = m.coef_
            for j in te:
                s = shalist[j]
                tid, rnd, morg, msdk = data[s][1]
                xj = Xl[j].toarray().ravel()
                mc = method_contrib(W, int(y[j]), xj, tcol[s], tid, morg, len(msdk))
                per_seed_mc[j].append(mc)
                if seed == 0:
                    nt = int(msdk.sum())
                    if 0 < nt and nt < len(msdk):
                        order = np.argsort(-mc)
                        f = fam_names[y[j]]
                        d18[f]["p_at"].append(float(msdk[order[:nt]].mean()))
                        d18[f]["ap"].append(float(average_precision_score(msdk, mc)))
                        d18[f]["auroc"].append(float(roc_auc_score(msdk, mc)))
                        d18[f]["base"].append(float(msdk.mean()))
        for j in te:
            mcs = per_seed_mc[j]
            # pairwise seed agreement, finite values only
            rhos = []
            for a in range(len(mcs)):
                for b in range(a + 1, len(mcs)):
                    rho = spearmanr(mcs[a], mcs[b]).statistic
                    if np.isfinite(rho):
                        rhos.append(rho)
            if rhos:
                agree.append(float(np.median(rhos)))

    d18_out = {}
    for f, d in d18.items():
        if d["p_at"]:
            d18_out[f] = {"n": len(d["p_at"]),
                          "precision_at_sdk": float(np.median(d["p_at"])),
                          "average_precision": float(np.median(d["ap"])),
                          "auroc": float(np.median(d["auroc"])),
                          "base_rate": float(np.median(d["base"]))}
    d19_out = {"median_spearman_across_seeds": float(np.median(agree)),
               "p10": float(np.percentile(agree, 10)), "n": len(agree)}
    return d18_out, d19_out


def d16():
    out = {}
    tok = json.load(open(H.out_path("tier4", "v2", "tok_analyses.json")))
    ceil = {}
    if "identifier" in tok and "X3_localization_ceiling" in tok["identifier"]:
        ceil = tok["identifier"]["X3_localization_ceiling"]
    cl = json.load(open(H.out_path("tier4", "p3", "colour_localize.json")))
    dep = {}
    if "labeled|clean" in cl and "1024" in cl["labeled|clean"]:
        dep = cl["labeled|clean"]["1024"]
    for f in H.PROXY:
        row = {"collision_free_ceiling": ceil.get(f)}
        if f in dep:
            row["deployed_hashed_p_at_target"] = dep[f]["p_at_target"]
            row["deployed_base_rate"] = dep[f]["base_rate"]
        out[f] = row
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    apps = T.staged()
    print("building identifier tokens (cached)", flush=True)
    data = T.build_cached("identifier", apps, 4)
    print("D18 + D19: localization vs SDK ground truth and seed agreement", flush=True)
    d18, d19 = run(apps, data)
    res["D18"] = d18
    res["D19"] = d19
    print("D16: collision-free ceiling vs deployed hashed explainer", flush=True)
    res["D16"] = d16()
    json.dump(res, open(OUT, "w"), indent=1)
    for f, v in d18.items():
        print(f"  D18 {f:12} p@sdk={v['precision_at_sdk']:.3f} AP={v['average_precision']:.3f} "
              f"AUROC={v['auroc']:.3f} base={v['base_rate']:.3f}", flush=True)
    print(f"  D19 median seed-agreement Spearman={d19['median_spearman_across_seeds']:.3f}", flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
