#!/usr/bin/env python3
"""Per-sample soundness check of the first-order additive certificate
(exp_p3_frontier_post.k_cert_grad): for every certified staged app, the greedy
aimed attack (all tokens free) must need strictly more than k_cert_grad tokens,
and so must random padding. Any violation means the certificate is unsound.
Same folds, models and raw vectors as the frontier run.

Output: results/tier4/p3/cert_soundness.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR
import exp_p3_frontier_post as FP


def steer_tokens(phi, W, b, y, menu0=None, max_t=FR.MAX_T):
    """Token-level greedy (menu applied to every third token for structure)."""
    phi = phi.astype(float).copy()
    for t in range(1, max_t + 1):
        if menu0 and (t - 1) % 3 == 0:
            allowed = menu0
        else:
            allowed = None
        j, s = FR.greedy_token(phi, W, b, y, allowed)
        phi[j] += s
        if FR.pred(W, b, phi) != y:
            return t
    return None


def main():
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    # test fold of every app
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    apps, mats = FR.staged_raw(ids)
    apps_t = []
    for a in apps:
        apps_t.append(tuple(a))
    apps = apps_t
    res = {}
    for c in FR.COLOURS:
        idx, _, XP = H._load(FR.PACK[c])
        rows = []
        for i in ids:
            rows.append(idx[i])
        X = XP[rows].astype(np.float64)
        viol = 0
        n = 0
        ratios = []
        free_vs_menu = []
        # refit per fold, then attack every certified test app
        for r, k, tr, te in F:
            W, b = FR.fit_sgd(X, y, tr, 1000 * r + k)
            for i, (fa, sha) in enumerate(apps):
                if fold_of[sha] != k:
                    continue
                phi = mats[c][i]
                yy = int(le.transform([fa])[0])
                if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != yy:
                    continue
                kc = FP.k_cert_grad(phi, W, b, yy)
                ks = steer_tokens(phi, W, b, yy)
                n += 1
                if ks is not None and ks <= kc:
                    viol += 1
                if ks:
                    ratios.append(ks / max(kc, 1))
                if c == "structure":
                    km = steer_tokens(phi, W, b, yy, FR.STRUCT_MENU)
                    if ks and km:
                        free_vs_menu.append(km / ks)
        med = float(np.median(ratios))
        qs = []
        for q in np.quantile(ratios, [.1, .9]):
            qs.append(float(q))
        res[c] = {"n": n, "violations": viol, "steer_over_cert_median": med,
                  "steer_over_cert_q": qs}
        if free_vs_menu:
            res[c]["menu_over_free_median"] = float(np.median(free_vs_menu))
        print(c, res[c], flush=True)
    json.dump(res, open(os.path.join(FR.P3, "cert_soundness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
