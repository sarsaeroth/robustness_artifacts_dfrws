#!/usr/bin/env python3
"""R3a: real per-build identifier renaming against the certified l2 region.

For a linear model on L2-normalized vectors, the prediction for x (class y) is
certified against any perturbation with
    ||x' - x||_2 < r(x) = min_{c != y} s_c(x) / ||u_c||_2,
    u_c = w_y - w_c,  s_c(x) = u_c . x + (b_y - b_c).
The paper's certificates (theory.tex, eq:cert_norm) are this bound composed with
a raw-to-normalized conversion; here the normalized perturbation is measured
directly, because the renamed vectors (per-app random renaming of developer
identifiers, gen_wl_renamed.py) exist for every app.

Per test app (5 repeats x StratifiedGroupKFold(5), model trained on the clean
representation of the training fold):
  r        certified radius at the clean vector
  delta    ||x_renamed - x||_2
  flipped  prediction on x_renamed differs from prediction on x
Checks: soundness (no flip with delta < r), and the ratio delta / r.

Representations: labeled (identifier-seeded, renaming moves it) and the
invariant colourings, whose renamed vectors should equal the clean ones.
Output: results/tier4/r3a_renaming.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "r3a_renaming.json")
REPEATS = 5


def radius(W, b, x, yhat):
    s = (W[yhat] - W) @ x + (b[yhat] - b)
    u = np.linalg.norm(W[yhat] - W, axis=1)
    # the predicted class itself is excluded from the minimum
    s[yhat] = np.inf
    u[yhat] = 1.0
    return float(np.min(s / u))


def run(clean_rep, renamed_rep):
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    X = H.matrix(clean_rep, ids)
    Xr = H.matrix(renamed_rep, ids)
    g = H.groups_for(ids)
    rec = {"r": [], "delta": [], "flip": [], "correct": [], "family": []}
    for r, f, tr, te in H.folds(y, g, REPEATS):
        up = H.upsample(tr, y, 1000 * r + f)
        m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
        m.fit(X[up], y[up])
        W = m.coef_
        b = m.intercept_
        p0 = m.predict(X[te])
        p1 = m.predict(Xr[te])
        for k, j in enumerate(te):
            rec["r"].append(radius(W, b, X[j], p0[k]))
            rec["delta"].append(float(np.linalg.norm(Xr[j] - X[j])))
            rec["flip"].append(bool(p0[k] != p1[k]))
            rec["correct"].append(bool(p0[k] == y[j]))
            rec["family"].append(str(fam[j]))
    R = np.array(rec["r"])
    D = np.array(rec["delta"])
    F = np.array(rec["flip"])
    C = np.array(rec["correct"])
    inside = D < R
    fams = np.array(rec["family"])
    # summary values
    n_evals = int(len(R))
    clean_accuracy = float(C.mean())
    flip_rate = float(F.mean())
    renamed_accuracy_proxy = float(np.mean(np.where(F, ~C, C)))  # approx; dropped below
    inside_frac = float(inside.mean())
    unsound = int((F & inside).sum())
    delta_median = float(np.median(D))
    radius_median = float(np.median(R))
    ratio = D / np.maximum(R, 1e-12)
    ratio_median = float(np.median(ratio))
    ratio_q = []
    for q in np.quantile(ratio, [.1, .25, .75, .9]):
        ratio_q.append(float(q))
    if (~inside).any():
        flip_outside = float(F[~inside].mean())
    else:
        flip_outside = None
    per_family = {}
    for fa in sorted(set(fams)):
        in_fam = fams == fa
        per_family[fa] = {"n": int(in_fam.sum()),
                          "flip_rate": float(F[in_fam].mean()),
                          "inside_frac": float(inside[in_fam].mean()),
                          "delta_over_radius_median": float(np.median(ratio[in_fam]))}
    out = {
        "n_evals": n_evals,
        "clean_accuracy": clean_accuracy,
        "flip_rate": flip_rate,
        "renamed_accuracy_proxy": renamed_accuracy_proxy,
        "inside_certified_frac": inside_frac,
        "unsound_flips_inside": unsound,
        "delta_median": delta_median, "radius_median": radius_median,
        "delta_over_radius_median": ratio_median,
        "delta_over_radius_q": ratio_q,
        "flip_rate_given_outside": flip_outside,
        "per_family": per_family,
    }
    del out["renamed_accuracy_proxy"]
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    if "labeled" not in res:
        res["labeled"] = run("labeled", "renamed")
        json.dump(res, open(OUT, "w"), indent=1)
    # Invariant colourings have no separate renamed vectors: renaming leaves them
    # unchanged (fuzzy/api measured at 0.00 vector difference on renamed graphs in
    # the obfuscation ablation; opcode/combined invariant by construction). Report
    # their certified radii for comparison with the labeled model.
    for rep in ["combined", "fuzzy", "structure"]:
        if rep in res:
            continue
        ids, fam = H.sample_ids("four")
        y, _ = H.labels(fam)
        try:
            X = H.matrix(rep, ids)
        except KeyError as e:  # e.g. combined before a dataset's opcode maps exist
            print(f"{rep}: SKIP ({e})", flush=True)
            continue
        g = H.groups_for(ids)
        radii = []
        acc = []
        for r, f, tr, te in H.folds(y, g, REPEATS):
            up = H.upsample(tr, y, 1000 * r + f)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            m.fit(X[up], y[up])
            p = m.predict(X[te])
            for k, j in enumerate(te):
                radii.append(radius(m.coef_, m.intercept_, X[j], p[k]))
            acc += list(p == y[te])
        res[rep] = {"clean_accuracy": float(np.mean(acc)), "radius_median": float(np.median(radii)),
                    "renaming_delta": 0.0, "note": "renaming-invariant (see comment)"}
        json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
