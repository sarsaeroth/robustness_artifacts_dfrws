#!/usr/bin/env python3
"""E4d: does attribution EVENNESS predict the CERTIFIED robustness radius?

Melis et al. (IJMLC 2022) showed attribution EVENNESS (E1/E2) correlates with EMPIRICAL
adversarial robustness (evasion cost) on Drebin. We extend this to CERTIFIED bounds on
hashed WL features: for each app compute the per-bucket linear-SHAP attribution
r = w_y (x - mu), its evenness E1/E2, and the additive certified radius k_cert (R2), then
correlate. Prediction: evenness correlates POSITIVELY with k_cert (a diffuse attribution
has small ||w||_inf relative to mass, so more tokens must be added to flip).

Melis evenness (on |r|, m features):
  E1 = (2/(m-1)) [ m - sum_k F(k) ],  F(k)=share of total |r| in the top-k  (0 concentrated .. 1 uniform)
  E2 = (1/m) ||r||_1 / ||r||_inf
Output: results/tier4/e4d_evenness.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P
import exp_r2_certificate as R2

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "e4d_evenness.json")


def main():
    ids, fam = H.sample_ids("four")
    y_all, le = H.labels(fam)
    Xn = H.matrix("labeled_fcg", ids)
    up = H.upsample(np.arange(len(ids)), y_all, 0)
    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
    m.fit(Xn[up], y_all[up])
    W = m.coef_.astype(np.float64)
    b = m.intercept_.astype(np.float64)
    ids5, fam5 = H.sample_ids("five")
    mu = H.matrix("labeled_fcg", ids5)[fam5 == H.BENIGN].mean(0).astype(np.float64)

    rows = defaultdict(list)
    id_set = set(ids.tolist())
    staged = []
    for f in H.PROXY:
        for sha in H.staged(f):
            if sha in id_set:
                staged.append((f, sha))
    n = 0
    for f, sha in staged:
        try:
            nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
            xn, prov, nrm = P.provenance_vector(nl, su, pr)
        except Exception:
            continue
        if xn is None:
            continue
        phi = (xn * nrm).astype(np.float64)
        yy = int(le.transform([f])[0])
        xnorm = phi / np.linalg.norm(phi)
        if int(np.argmax(W @ xnorm + b)) != yy:
            continue
        r = W[yy] * (xnorm - mu)                # per-bucket linear SHAP for true class
        # Melis evenness E1/E2 of |r|
        a = np.sort(np.abs(r))[::-1]
        n_feat = len(a)
        tot = a.sum()
        if tot == 0 or n_feat < 2:
            e1 = None
            e2 = None
        else:
            F = np.cumsum(a) / tot
            e1 = float((2.0 / (n_feat - 1)) * (n_feat - F.sum()))
            e2 = float((a.sum() / a.max()) / n_feat)
        kc, _ = R2.k_cert(phi, W, b, yy)
        if e1 is None:
            continue
        rows[f].append({"e1": e1, "e2": e2, "k_cert": kc})
        n += 1
        del nl, su, pr
        if n % 20 == 0:
            print(f"  {n} apps", flush=True)

    allr = []
    for rs in rows.values():
        for r in rs:
            allr.append(r)

    def corr(xs, ys):
        if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
            return None
        return {"spearman": float(spearmanr(xs, ys).statistic),
                "spearman_p": float(spearmanr(xs, ys).pvalue),
                "kendall": float(kendalltau(xs, ys).statistic)}
    e1s = []
    e2s = []
    kcs = []
    for r in allr:
        e1s.append(r["e1"])
        e2s.append(r["e2"])
        kcs.append(r["k_cert"])
    res = {"n": len(allr),
           "e1_vs_kcert": corr(e1s, kcs), "e2_vs_kcert": corr(e2s, kcs),
           "e1_median": float(np.median(e1s)), "e2_median": float(np.median(e2s)),
           "kcert_median": float(np.median(kcs)),
           "per_family": {}}
    for f, rs in rows.items():
        if len(rs) >= 5:
            fe1 = []
            fkc = []
            for r in rs:
                fe1.append(r["e1"])
                fkc.append(r["k_cert"])
            res["per_family"][f] = {"n": len(rs),
                                    "e1_vs_kcert": corr(fe1, fkc),
                                    "e1_median": float(np.median(fe1)),
                                    "kcert_median": float(np.median(fkc))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n=== E4d: attribution evenness vs certified k ===")
    print(f"n={res['n']}  E1 median={res['e1_median']:.3f}  kcert median={res['kcert_median']:.0f}")
    print(f"E1 vs kcert: {res['e1_vs_kcert']}")
    print(f"E2 vs kcert: {res['e2_vs_kcert']}")
    for f, r in res["per_family"].items():
        print(f"  {f:11} n={r['n']} E1={r['e1_median']:.3f} kcert={r['kcert_median']:.0f} E1-kcert={r['e1_vs_kcert']}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
