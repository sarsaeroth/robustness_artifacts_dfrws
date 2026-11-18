#!/usr/bin/env python3
"""Frontier follow-ups (after exp_p3_frontier.py).

1. Tighter additive certificate. The norm term of Theorem certadd used
   ||phi + Delta|| <= ||phi|| + T. Since ||Delta||_1 = T, also
   ||phi + Delta||^2 <= ||phi||^2 + 2 T ||phi||_inf + T^2, which is much smaller for
   T << ||phi||. Certified T is the largest T with
       u_c.phi - T ||u_c||_inf > max(0, -beta_c) sqrt(||phi||^2 + 2T||phi||_inf + T^2)
   for all c != y (found by bisection; the left side decreases and the right side
   increases in T). Recomputed for every SGD colouring and l_inf bound, with the
   same folds and models as the frontier run.
2. Monotone detectors at higher dimension. At d=1,024 unsigned buckets are all
   occupied (~100 tokens each) and monotone detectors degenerate; here unsigned
   log1p counts at d = 2^14 and 2^16 from the cached labeled tokens.

Output: results/tier4/p3/frontier_post.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR

OUT = os.path.join(FR.P3, "frontier_post.json")
TOKDIR = os.path.join(FR.P3, "colour_tokens")


def k_cert_tight(phi, W, b, y, t_max=10 ** 6):
    nrm2 = float(phi @ phi)
    pinf = float(np.abs(phi).max())
    best = np.inf
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = W[y] - W[c]
        beta = b[y] - b[c]
        nb = max(0.0, -beta)
        ui = float(np.abs(u).max())
        a = float(u @ phi)
        ok = lambda T: a - T * ui > nb * np.sqrt(nrm2 + 2 * T * pinf + T * T)
        if not ok(0):
            return 0
        # exponential then binary search for the largest certified T
        lo = 0
        hi = 1
        while hi < t_max and ok(hi):
            lo = hi
            hi = hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        best = min(best, lo)
    return int(best)


def k_cert_grad(phi, W, b, y):
    """First-order certificate with curvature correction."""
    # With g = u.phi + beta||phi|| and G = u + beta phi/||phi||, for any Delta with ||Delta||_1 = T:
    #   beta >= 0: ||phi+Delta|| >= ||phi|| + Delta.phi/||phi||  (convexity)
    #   beta <  0: ||phi+Delta|| <= ||phi|| + Delta.phi/||phi|| + ||Delta||^2/(2||phi||)
    # so g(phi+Delta) >= g(phi) - T||G||_inf - max(0,-beta) T^2 / (2||phi||).
    nrm = float(np.linalg.norm(phi))
    best = np.inf
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = W[y] - W[c]
        beta = b[y] - b[c]
        g0 = float(u @ phi + beta * nrm)
        Gi = float(np.abs(u + beta * phi / nrm).max())
        q = max(0.0, -beta) / (2 * nrm)
        if q > 0:
            T = (-Gi + np.sqrt(Gi * Gi + 4 * q * g0)) / (2 * q)
        elif Gi > 0:
            T = g0 / Gi
        else:
            T = np.inf
        best = min(best, T)
    if not np.isfinite(best):
        return 10 ** 9
    return max(0, int(np.ceil(best)) - 1)


def recert(X, y, F, apps_by_fold, raw, le, fitter):
    out = []
    for r, k, tr, te in F:
        W, b = fitter(X, y, tr, 1000 * r + k)
        for idx, (fam, sha) in apps_by_fold.get(k, []):
            phi = raw[idx]
            yy = int(le.transform([fam])[0])
            if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != yy:
                continue
            out.append({"family": fam, "k_old": FR.k_cert(phi, W, b, yy), "k_tight": k_cert_tight(phi, W, b, yy),
                        "k_grad": k_cert_grad(phi, W, b, yy)})
    # summary (keys in the same order as before)
    ko_list = []
    kt_list = []
    kg_list = []
    for o in out:
        ko_list.append(o["k_old"])
        kt_list.append(o["k_tight"])
        kg_list.append(o["k_grad"])
    ko = np.array(ko_list)
    kt = np.array(kt_list)
    kg = np.array(kg_list)
    summ = {"k_cert_grad_median": float(np.median(kg))}
    grad_q = []
    for q in np.quantile(kg, [.25, .75]):
        grad_q.append(float(q))
    summ["k_cert_grad_q"] = grad_q
    summ["zero_frac_grad"] = float((kg == 0).mean())
    per_family_grad = {}
    for f in H.PROXY:
        fam_grad = []
        for o in out:
            if o["family"] == f:
                fam_grad.append(o["k_grad"])
        if fam_grad:
            per_family_grad[f] = float(np.median(fam_grad))
    summ["per_family_grad"] = per_family_grad
    summ["n"] = len(out)
    summ["k_cert_old_median"] = float(np.median(ko))
    summ["k_cert_tight_median"] = float(np.median(kt))
    tight_q = []
    for q in np.quantile(kt, [.25, .75]):
        tight_q.append(float(q))
    summ["k_cert_tight_q"] = tight_q
    summ["zero_frac_old"] = float((ko == 0).mean())
    summ["zero_frac_tight"] = float((kt == 0).mean())
    per_family = {}
    for f in H.PROXY:
        fam_tight = []
        for o in out:
            if o["family"] == f:
                fam_tight.append(o["k_tight"])
        if fam_tight:
            per_family[f] = float(np.median(fam_tight))
    summ["per_family"] = per_family
    return summ


def monotone(d):
    ids, _ = H.sample_ids("four")
    # unsigned log1p bucket counts of the cached labeled tokens at dimension d
    id_set = set(ids)
    staged = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in id_set:
                staged.append((f, s))
    Xs = []
    fams = []
    shas = []
    for f, s in staged:
        z = np.load(os.path.join(TOKDIR, f"{s}.npz"))
        h = z["labeled|clean|h"]
        v = np.bincount((h % d).astype(np.int64), minlength=d).astype(float)
        Xs.append(np.log1p(v))
        fams.append(f)
        shas.append(s)
    X = np.array(Xs)
    fams = np.array(fams)
    groups = H.groups_for(np.array(shas))
    res = {}
    # one-vs-rest logistic regression, weights constrained >= 0 (monotone) or free
    for fam in H.PROXY:
        yb = (fams == fam).astype(int)
        s_mon = np.zeros(len(yb))
        s_free = np.zeros(len(yb))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yb, groups):
            mu = X[tr].mean(0)
            sd = X[tr].std(0) + 1e-6       # positive scaling keeps monotonicity
            Xt = (X[tr] - mu) / sd
            yt = yb[tr]
            n, dd = Xt.shape
            sw = np.where(yt == 1, n / (2 * yt.sum()), n / (2 * (n - yt.sum())))
            for nonneg, store in ((True, s_mon), (False, s_free)):
                def fg(th):
                    w = th[:dd]
                    b0 = th[dd]
                    z = np.clip(Xt @ w + b0, -30, 30)
                    p = 1 / (1 + np.exp(-z))
                    loss = -np.mean(sw * (yt * np.log(p + 1e-12) + (1 - yt) * np.log(1 - p + 1e-12))) + 5e-3 * (w @ w)
                    gz = sw * (p - yt) / n
                    return loss, np.concatenate([Xt.T @ gz + 1e-2 * w, [gz.sum()]])
                if nonneg:
                    w_bound = (0, None)
                else:
                    w_bound = (None, None)
                bnds = [w_bound] * dd + [(None, None)]
                opt = minimize(fg, np.zeros(dd + 1), jac=True, method="L-BFGS-B", bounds=bnds,
                               options={"maxiter": 1000})
                th = opt.x
                store[te] = ((X[te] - mu) / sd) @ th[:dd] + th[dd]
        res[fam] = {"auc_monotone": float(roc_auc_score(yb, s_mon)), "auc_unconstrained": float(roc_auc_score(yb, s_free)),
                    "f1_monotone": float(f1_score(yb, s_mon > 0)), "f1_unconstrained": float(f1_score(yb, s_free > 0))}
        print("monotone", d, fam, res[fam], flush=True)
    return res


def main():
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    # staged apps grouped by their test fold
    apps, mats = FR.staged_raw(ids)
    apps_t = []
    for a in apps:
        apps_t.append(tuple(a))
    apps = apps_t
    apps_by_fold = {}
    for i, (fa, sha) in enumerate(apps):
        if fold_of[sha] not in apps_by_fold:
            apps_by_fold[fold_of[sha]] = []
        apps_by_fold[fold_of[sha]].append((i, (fa, sha)))
    Xs = {}
    for c in FR.COLOURS:
        idx, _, XP = H._load(FR.PACK[c])
        rows = []
        for i in ids:
            rows.append(idx[i])
        Xs[c] = XP[rows].astype(np.float64)
    for c in FR.COLOURS:
        if f"sgd_{c}" not in res:
            res[f"sgd_{c}"] = recert(Xs[c], y, F, apps_by_fold, mats[c], le, FR.fit_sgd)
            print(c, res[f"sgd_{c}"], flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    wmax = json.load(open(FR.OUT))["linf"]["wmax_unconstrained"]
    for q in (1.0, 0.5, 0.25, 0.1, 0.05):
        key = f"linf_{q}"
        if key not in res:
            res[key] = recert(Xs["labeled"], y, F, apps_by_fold, mats["labeled"], le,
                              lambda X, yy, tr, s, t=q * wmax: FR.fit_linf(X, yy, tr, s, t))
            print(key, res[key], flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    for d in (2 ** 14, 2 ** 16):
        key = f"monotone_{d}"
        if key not in res:
            res[key] = monotone(d)
            json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
