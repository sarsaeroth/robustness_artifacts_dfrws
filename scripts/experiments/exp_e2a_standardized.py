#!/usr/bin/env python3
"""E2a: fair representation comparison with STANDARDIZED embeddings.

Corrects the paradigm sweep, which fed learned embeddings RAW to a linear model
(node2vec linear 0.668) while WL was L2-normalized - an unfair preprocessing
mismatch. Here every representation is compared under matched preprocessing and the
per-representation best-linear AND best-nonlinear are both reported, with diagnostics
that separate "representation" from "classifier".

Representations (all at matched dim 1024 per graph, CFG+FCG concat where applicable):
  labeled WL, combined WL, fuzzy WL, structure WL, node2vec, dex.
Preprocessing arms per representation: raw, L2-normalized, z-scored (StandardScaler).
Classifiers: linear SGD (log loss), RandomForest(200).
Diagnostics: linear/nonlinear macro-F1 ratio; hard-margin LinearSVC training error
(is the point set linearly separable?); anisotropy = mean pairwise cosine of a random
sample (how "curled"/low-rank the space is).

4-class, 3 repeats of shuffled StratifiedGroupKFold(5) on DEX groups, in-fold upsample.
Output: results/tier4/e2a_standardized.json
"""
import gc
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "e2a_standardized.json")
REPS = ["labeled", "combined", "fuzzy", "structure", "node2vec", "dex", "mama_family", "mama_package"]  # Drebin is sparse; see exp_cert_drebin.py
REPEATS = 3


def prep(Xtr, Xte, mode):
    if mode == "raw":
        return Xtr, Xte
    if mode == "l2":
        n = Normalizer().fit(Xtr)
        return n.transform(Xtr), n.transform(Xte)
    s = StandardScaler().fit(Xtr)
    return s.transform(Xtr), s.transform(Xte)


def anisotropy(X, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    A = X[idx]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    # mean cosine of random pairs
    p = rng.permutation(len(A))
    return float(np.mean(np.sum(A * A[p], axis=1)))


def available(rep, ids):
    """ids that every packed file of rep covers (Drebin/MaMaDroid can miss failed extractions)."""
    ok = np.ones(len(ids), dtype=bool)
    for ext in H.REPS[rep]:
        idx, _, _ = H._load(ext)
        covered = []
        for i in ids:
            covered.append(i in idx)
        ok &= np.array(covered)
    return ok


def main():
    ids0, fam0 = H.sample_ids("four")
    g0 = H.groups_for(ids0)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    for rep in REPS:
        # skip finished representations; load data
        if rep in res and "error" not in res[rep]:
            continue
        try:
            ok = available(rep, ids0)
            if ok.mean() < 0.9:
                raise KeyError(f"{rep}: only {ok.sum()} of {len(ok)} apps have this representation")
            ids = ids0[ok]
            fam = fam0[ok]
            g = g0[ok]
            y, le = H.labels(fam)
            folds = H.folds(y, g, REPEATS)
            X = np.asarray(H.matrix(rep, ids), dtype=np.float32)  # float32: halves peak memory (MaMaDroid package is 14,884-d)
        except KeyError as e:
            res[rep] = {"error": str(e)}
            print(f"{rep}: SKIP ({e})", flush=True)
            continue
        row = {"dim": int(X.shape[1]), "n": int(len(X)), "anisotropy_raw": anisotropy(X)}
        # linear under 3 preprocessings + RF (raw) + hard-margin separability
        for mode in ("raw", "l2", "std"):
            f1 = []
            for r, f, tr, te in folds:
                up = H.upsample(tr, y, 1000 * r + f)
                Xtr, Xte = prep(X[up], X[te], mode)
                m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
                m.fit(Xtr, y[up])
                f1.append(f1_score(y[te], m.predict(Xte), average="macro"))
                del Xtr, Xte, m
                gc.collect()
            row[f"lin_{mode}"] = float(np.mean(f1))
        # RF (raw; trees are scale-invariant)
        f1 = []
        for r, f, tr, te in folds:
            up = H.upsample(tr, y, 1000 * r + f)
            m = RandomForestClassifier(n_estimators=200, n_jobs=6, random_state=42)
            m.fit(X[up], y[up])
            f1.append(f1_score(y[te], m.predict(X[te]), average="macro"))
            del m
            gc.collect()
        row["rf"] = float(np.mean(f1))
        # hard-margin linear separability: LinearSVC training error on std features, big C
        Xs = StandardScaler().fit_transform(X)
        svc = LinearSVC(C=1e4, dual=True, max_iter=5000)
        try:
            svc.fit(Xs, y)
            row["hardmargin_train_err"] = float(1 - svc.score(Xs, y))
        except Exception as e:
            row["hardmargin_train_err"] = None
        del Xs
        gc.collect()
        # best linear vs RF
        best_lin = max(row["lin_raw"], row["lin_l2"], row["lin_std"])
        row["best_linear"] = float(best_lin)
        if row["rf"]:
            row["lin_over_rf"] = float(best_lin / row["rf"])
        else:
            row["lin_over_rf"] = None
        res[rep] = row
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{rep:10} dim={row['dim']:4} lin_raw={row['lin_raw']:.3f} lin_l2={row['lin_l2']:.3f} "
              f"lin_std={row['lin_std']:.3f} rf={row['rf']:.3f} bestlin/rf={row['lin_over_rf']:.3f} "
              f"hm_err={row['hardmargin_train_err']} aniso={row['anisotropy_raw']:.3f}", flush=True)
    # final table
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n=== E2a fair comparison (macro-F1, 4-class, DEX-grouped) ===")
    print(f"{'rep':10} {'lin_raw':>8}{'lin_l2':>8}{'lin_std':>8}{'rf':>8}{'bestlin/rf':>11}")
    for rep in REPS:
        r = res.get(rep, {})
        if "error" in r:
            continue
        print(f"{rep:10} {r['lin_raw']:8.3f}{r['lin_l2']:8.3f}{r['lin_std']:8.3f}{r['rf']:8.3f}{r['lin_over_rf']:11.3f}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
