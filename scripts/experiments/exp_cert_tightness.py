#!/usr/bin/env python3
"""Plan v3, block C: how tight and how reproducible is the per-application certificate.

  C11/C12  Per staged app, held out of its fold, compute four quantities on the deployed
           (full-corpus-trained) linear model:
             k_cert       loose additive lower bound (Thm certadd, ||phi+D|| <= ||phi||+T)
             k_cert_tight tight additive lower bound (quadratic norm bound, bisection)
             k_cert_grad  gradient bound with curvature correction
             k_steer      strongest realizable aimed attack (empirical upper bound), capped
           The certified floor is max(k_cert, k_cert_tight, k_cert_grad); the empirical
           ceiling is k_steer. Report per-family medians and the ceiling/floor ratio so the
           conservativeness of the published certificate is quantified, and how much the
           tight and gradient bounds improve on the loose one.
  C14      Certificate rank stability across the hash seed. Re-hash each staged app's exact
           WL tokens into d=1,024 signed vectors under several salts, train staged-only
           grouped models, recompute every held-out app's k_cert, and report Spearman rank
           correlation between salts (held fixed: tokens, folds, training set; varied: seed).

Output: results/tier4/v3/cert_tightness.json
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR
import exp_p3_frontier_post as FP

OUT = H.out_path("tier4", "v3", "cert_tightness.json")
COLOURS = ["labeled", "fuzzy", "api", "structure"]
STEER_CAP = 4000      # bound aimed-attack runtime; records censoring


def setup():
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    # staged apps, grouped by the fold that holds them out
    apps, mats = FR.staged_raw(ids)
    app_tuples = []
    for a in apps:
        app_tuples.append(tuple(a))
    apps = app_tuples
    by_fold = {}
    for i, (fa, sha) in enumerate(apps):
        by_fold.setdefault(fold_of[sha], []).append((i, fa, sha))
    return ids, y, g, le, F, by_fold, apps, mats


def c11_c12(ids, y, g, le, F, by_fold, apps, mats):
    out = {}
    for c in COLOURS:
        idx, _, XP = H._load(FR.PACK[c])
        rows_of_ids = []
        for i in ids:
            rows_of_ids.append(idx[i])
        X = XP[rows_of_ids].astype(np.float64)
        if c == "structure":
            menu0 = FR.STRUCT_MENU
        else:
            menu0 = None
        # per fold: certificates and aimed attack on correctly classified held-out apps
        rows = []
        for r, k, tr, te in F:
            W, b = FR.fit_sgd(X, y, tr, 1000 * r + k)
            for i, fa, sha in by_fold.get(k, []):
                phi = mats[c][i]
                yy = int(le.transform([fa])[0])
                if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != yy:
                    continue
                kc = FR.k_cert(phi, W, b, yy)
                kt = FP.k_cert_tight(phi, W, b, yy)
                kg = FP.k_cert_grad(phi, W, b, yy)
                ks, _ = FR.k_steer(phi, W, b, yy, menu0, max_t=STEER_CAP)
                rows.append({"family": fa, "k_cert": kc, "k_tight": kt, "k_grad": kg,
                             "k_steer": ks, "censored": ks is None})
            print(f"  {c} fold {k}: n={len(rows)}", flush=True)
        out[c] = summarize(rows)
        print(f"  {c}: {json.dumps(out[c]['overall'])}", flush=True)
    return out


def summarize(rows):
    def med(vals):
        v = []
        for x in vals:
            if x is not None:
                v.append(x)
        if v:
            return float(np.median(v))
        else:
            return None

    # overall: floor = best certified bound, ceiling = aimed attack
    floor = []
    ratios = []
    tight_gain = []
    grad_gain = []
    k_cert_values = []
    k_tight_values = []
    k_grad_values = []
    k_steer_values = []
    censored = []
    for r in rows:
        fl = max(r["k_cert"], r["k_tight"], r["k_grad"])
        floor.append(fl)
        if r["k_steer"] is not None:
            ratios.append(r["k_steer"] / max(fl, 1))
        tight_gain.append(r["k_tight"] / max(r["k_cert"], 1))
        grad_gain.append(r["k_grad"] / max(r["k_cert"], 1))
        k_cert_values.append(r["k_cert"])
        k_tight_values.append(r["k_tight"])
        k_grad_values.append(r["k_grad"])
        k_steer_values.append(r["k_steer"])
        censored.append(r["censored"])
    out = {"overall": {"n": len(rows),
                       "k_cert_median": med(k_cert_values),
                       "k_tight_median": med(k_tight_values),
                       "k_grad_median": med(k_grad_values),
                       "floor_median": med(floor),
                       "k_steer_median": med(k_steer_values),
                       "ceiling_over_floor_median": med(ratios),
                       "tight_over_loose_median": med(tight_gain),
                       "grad_over_loose_median": med(grad_gain),
                       "censored_frac": float(np.mean(censored))},
           "per_family": {}}

    # per family
    for fam in H.PROXY:
        rr = []
        for r in rows:
            if r["family"] == fam:
                rr.append(r)
        if rr:
            fl = []
            rt = []
            steer = []
            for r in rr:
                f = max(r["k_cert"], r["k_tight"], r["k_grad"])
                fl.append(f)
                if r["k_steer"] is not None:
                    rt.append(r["k_steer"] / max(f, 1))
                steer.append(r["k_steer"])
            out["per_family"][fam] = {"n": len(rr), "floor_median": med(fl),
                                      "k_steer_median": med(steer),
                                      "ceiling_over_floor_median": med(rt)}
    return out


def c14(salts=(1, 2, 3, 4, 5)):
    """Rank stability of k_cert across hash seed, from exact tokens, staged-trained."""
    import exp_tok_analyses as Tk
    # exact identifier tokens of the staged apps
    apps = Tk.staged()
    fam_set = set()
    for f, _ in apps:
        fam_set.add(f)
    fam_names = sorted(fam_set)
    data = Tk.build_cached("identifier", apps, 4)
    shalist = []
    for _, s in apps:
        if s in data:
            shalist.append(s)
    y_list = []
    for s in shalist:
        y_list.append(fam_names.index(data[s][0]))
    y = np.array(y_list)
    g = H.groups_for(np.array(shalist))
    D = 1024
    toks = {}
    for s in shalist:
        toks[s] = data[s][1][0]
    P = np.uint64(0x9E3779B97F4A7C15)

    F = H.folds(y, g, 1)
    percert = {}
    for salt in salts:
        # re-hash every app's tokens under this salt
        vectors = []
        for s in shalist:
            t = toks[s]
            mixed = (t * P) ^ np.uint64(salt * 0x100000001B3)
            bucket = (mixed % np.uint64(D)).astype(np.int64)
            sign = np.where((mixed >> np.uint64(63)) == np.uint64(0), 1.0, -1.0)
            v = np.zeros(D)
            np.add.at(v, bucket, sign)
            vectors.append(v)
        V = np.array(vectors)
        # held-out certificates per fold
        kc = {}
        for r, k, tr, te in F:
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            up = H.upsample(tr, y, 1000 * r + k)
            m.fit(V[up], y[up])
            W = m.coef_.astype(float)
            b = m.intercept_.astype(float)
            for j in te:
                phi = V[j]
                if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != y[j]:
                    continue
                kc[shalist[j]] = FR.k_cert(phi, W, b, int(y[j]))
        percert[salt] = kc
        print(f"  C14 salt={salt}: certified {len(kc)}", flush=True)

    # pairwise rank correlation on apps certified under every salt
    salt_sets = []
    for s in salts:
        salt_sets.append(set(percert[s]))
    common = set.intersection(*salt_sets)
    common = sorted(common)
    mat_rows = []
    for s in salts:
        row = []
        for sha in common:
            row.append(percert[s][sha])
        mat_rows.append(row)
    mat = np.array(mat_rows)
    rho = []
    for i in range(len(salts)):
        for j in range(i + 1, len(salts)):
            rho.append(float(spearmanr(mat[i], mat[j]).statistic))
    median_k_by_salt = {}
    for s in salts:
        median_k_by_salt[str(s)] = float(np.median(list(percert[s].values())))
    return {"n_common": len(common), "n_salts": len(salts),
            "spearman_median": float(np.median(rho)), "spearman_min": float(np.min(rho)),
            "median_k_by_salt": median_k_by_salt}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["C11", "C14"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    if "C11" in a.parts:
        print("C11/C12: certificate tightness vs empirical attack", flush=True)
        ids, y, g, le, F, by_fold, apps, mats = setup()
        res["C11_C12"] = c11_c12(ids, y, g, le, F, by_fold, apps, mats)
        json.dump(res, open(OUT, "w"), indent=1)
    if "C14" in a.parts:
        print("C14: certificate rank stability across hash seed", flush=True)
        res["C14"] = c14()
        json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
