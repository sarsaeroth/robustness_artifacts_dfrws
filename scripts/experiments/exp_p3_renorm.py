#!/usr/bin/env python3
"""X5: error of Proposition 1 when the normalizer is not held fixed.

Proposition 1 gives the method-level Shapley value psi_m = w.tok(m) / ||phi(G)||
for the game with the normalizer held at ||phi(G)||. The deployed model sees the
renormalized vector, so the "true" game is v(S) = w_c . phi_S / ||phi_S|| with
phi_S the tokens emitted by the methods in S (v(empty) = 0). Its Shapley values
are estimated here by Monte Carlo over permutations, incrementally, and compared
with psi_m: rank agreement, relative L2 error, top-|target| overlap and
precision against the SDK namespace. Model: linear SGD on labeled FCG WL, trained
on the DEX-grouped fold that holds the app out. Apps: the smallest staged apps
with SDK ground truth (ipninja, luminati), where exact-in-expectation sampling is
affordable.

Output: results/tier4/p3/renorm.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P
import graph_io as GI

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "p3", "renorm.json")
N_PERM = 200
PER_FAM = 8
# stable-identifier families
if H.DATASET == "proxy":
    FAMS = ["ipninja", "luminati"]
else:
    FAMS = list(H.PROXY)


def main():
    # data, DEX-grouped folds, and the fold that holds out each app
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    X = H.matrix("labeled_fcg", ids)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    models = {}
    rows = []
    for fa in FAMS:
        # smallest staged apps first
        dd = os.path.join(GRAPHS, fa)
        shas = []
        for s in sorted(H.staged(fa), key=lambda s: GI.size(os.path.join(dd, s, f"{s}.fcg"))):
            if s in fold_of:
                shas.append(s)
        done = 0
        for sha in shas:
            if done >= PER_FAM:
                break
            nl, su, pr = G.parse_gml_lightweight(os.path.join(dd, sha, f"{sha}.fcg"))
            x, prov, nrm = P.provenance_vector(nl, su, pr)
            if x is None:
                continue
            k = fold_of[sha]
            if k not in models:
                _, _, tr, _ = F[k]
                up = H.upsample(tr, y, k)
                models[k] = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5,
                                          random_state=42).fit(X[up], y[up])
            c = int(le.transform([fa])[0])
            w = models[k].coef_[c].astype(float)
            # tokens per method
            toks = defaultdict(list)
            for b, s, m in prov:
                toks[m].append((b, s))
            methods = list(toks)
            truth_list = []
            for m in methods:
                truth_list.append(m.startswith(P.SDK_PREFIXES.get(fa, ())))
            truth = np.array(truth_list)
            if truth.sum() == 0:
                continue
            # Proposition 1 values with the normalizer held fixed
            psi_list = []
            for m in methods:
                total = 0
                for b, s in toks[m]:
                    total += w[b] * s
                psi_list.append(total / nrm)
            psi = np.array(psi_list)
            # Monte Carlo Shapley of the renormalized game
            rng = np.random.default_rng(0)
            sh = np.zeros(len(methods))
            sh2 = np.zeros(len(methods))
            for _ in range(N_PERM):
                phi = np.zeros(len(w))
                num = 0.0
                sq = 0.0
                prev = 0.0
                for i in rng.permutation(len(methods)):
                    for b, s in toks[methods[i]]:
                        num += w[b] * s
                        sq += 2 * phi[b] * s + 1.0
                        phi[b] += s
                    if sq > 0:
                        val = num / np.sqrt(sq)
                    else:
                        val = 0.0
                    d = val - prev
                    sh[i] += d
                    sh2[i] += d * d
                    prev = val
            sh /= N_PERM
            se = np.sqrt(np.maximum(sh2 / N_PERM - sh ** 2, 0) / N_PERM)
            # compare the two attributions
            nt = int(truth.sum())
            top_psi = set(np.argsort(-psi)[:nt])
            top_sh = set(np.argsort(-sh)[:nt])
            rec = {"family": fa, "sha": sha, "n_methods": len(methods), "n_target": nt,
                   "spearman": float(stats.spearmanr(psi, sh)[0]),
                   "pearson": float(np.corrcoef(psi, sh)[0, 1]),
                   "rel_l2": float(np.linalg.norm(sh - psi) / np.linalg.norm(psi)),
                   "rel_l2_mc_noise": float(np.linalg.norm(se) / np.linalg.norm(psi)),
                   "top_target_jaccard": len(top_psi & top_sh) / len(top_psi | top_sh),
                   "p_at_target_psi": float(truth[list(top_psi)].mean()),
                   "p_at_target_renorm": float(truth[list(top_sh)].mean()),
                   "efficiency_check": float(abs(sh.sum() - w @ x))}
            rows.append(rec)
            done += 1
            print(rec, flush=True)

    # medians over apps
    summ = {}
    for k in ("spearman", "pearson", "rel_l2", "rel_l2_mc_noise", "top_target_jaccard",
              "p_at_target_psi", "p_at_target_renorm", "n_methods"):
        values = []
        for r in rows:
            values.append(r[k])
        summ[k] = float(np.median(values))
    res = {"n_apps": len(rows), "n_perm": N_PERM, "median": summ, "apps": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
