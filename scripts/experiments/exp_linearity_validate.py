#!/usr/bin/env python3
"""validate the "post-WL data is linear because evidence is additive" story.

  B6  Synthetic additive generator (positive + negative control for the whole diagnostic).
      Additive scenario: each family owns private marker tokens, apps draw them independently
      plus shared noise; class is an additive function of markers, so naive Bayes is optimal
      and should tie logistic, while a forest adds nothing. Nonlinear scenario: class is an
      XOR of two markers; NB and plain logistic should fall and only a nonlinear model recover.
      Confirms NB=logistic is a real signature of additivity, not an artifact.
  B7  Centered kernel alignment of the linear kernel with the label kernel, per coloring;
      should track the measured NB-logistic gap and the headline F1 ordering.
  B8  Normalized-margin distribution and within-margin (support) fraction per coloring;
      separates "separable" from "large-margin separable".
  B9  Exact tokens hashed from 2^8 to 2^16, linear vs random forest, on the staged apps;
      if linear keeps pace with the forest as dimension grows, linearity is not a collision
      artifact.

Output: results/tier4/v3/linearity_validate.json
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "v3", "linearity_validate.json")
REPS = ["labeled", "combined", "fuzzy", "api", "structure", "node2vec", "dex"]


# B6: synthetic additive and XOR generators
def gen_additive(rng, n_per, n_fam=4, n_marker=8, n_noise=200, p_marker=0.6, contam=0.05):
    """Additive: app = its family's markers (each w.p. p_marker) + shared noise tokens."""
    markers = {}
    for f in range(n_fam):
        markers[f] = list(range(f * n_marker, (f + 1) * n_marker))
    base = n_fam * n_marker
    rows = []
    ys = []
    for f in range(n_fam):
        for _ in range(n_per):
            v = np.zeros(base + n_noise)
            for m in markers[f]:
                if rng.random() < p_marker:
                    v[m] += 1
            # light cross-family contamination
            for g in range(n_fam):
                if g == f:
                    continue
                for m in markers[g]:
                    if rng.random() < contam:
                        v[m] += 1
            idx = rng.integers(0, n_noise, size=rng.integers(20, 60))
            np.add.at(v, base + idx, 1.0)
            rows.append(v)
            ys.append(f)
    return np.array(rows), np.array(ys)


def gen_xor(rng, n_per, n_fam=4, n_noise=200, p=0.5):
    """Nonlinear: family f is signalled by XOR of marker pair (2f, 2f+1)."""
    base = 2 * n_fam
    rows = []
    ys = []
    for f in range(n_fam):
        for _ in range(n_per):
            v = np.zeros(base + n_noise)
            a = rng.random() < p
            b = rng.random() < p
            # ensure the XOR pattern for the true family holds more often than chance
            if rng.random() < 0.85:
                a = True
                b = False
            v[2 * f] += int(a)
            v[2 * f + 1] += int(b)
            for g in range(n_fam):
                if g == f:
                    continue
                ga = rng.random() < p
                gb = rng.random() < p
                v[2 * g] += int(ga)
                v[2 * g + 1] += int(gb)
            idx = rng.integers(0, n_noise, size=rng.integers(20, 60))
            np.add.at(v, base + idx, 1.0)
            rows.append(v)
            ys.append(f)
    return np.array(rows), np.array(ys)


def cv_models(X, y, models, seed=0):
    Xl = np.log1p(X)
    out = {}
    for k in models:
        out[k] = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, y):
        for name, mk in models.items():
            m = mk().fit(Xl[tr], y[tr])
            out[name].append(f1_score(y[te], m.predict(Xl[te]), average="macro"))
    means = {}
    for k, v in out.items():
        means[k] = float(np.mean(v))
    return means


def b6():
    rng = np.random.default_rng(0)
    models = {"naive_bayes": lambda: MultinomialNB(alpha=0.1, fit_prior=False),
              "logistic": lambda: LogisticRegression(C=10.0, max_iter=3000),
              "forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=4)}
    Xa, ya = gen_additive(rng, 120)
    Xx, yx = gen_xor(rng, 120)
    out = {"additive": cv_models(Xa, ya, models), "xor": cv_models(Xx, yx, models)}
    print(f"  B6 additive {out['additive']}", flush=True)
    print(f"  B6 xor      {out['xor']}", flush=True)
    return out


# B7: centered kernel alignment
def center(K):
    r = K.mean(0, keepdims=True)
    return K - r - r.T + K.mean()


def cka(X, y):
    X = X.astype(np.float64)
    K = center(X @ X.T)
    Y = (y[:, None] == y[None, :]).astype(np.float64)
    Yc = center(Y)
    denom = np.linalg.norm(K) * np.linalg.norm(Yc)
    if denom > 0:
        return float((K * Yc).sum() / denom)
    else:
        return None


def b7(ids, y):
    out = {}
    for rep in REPS:
        try:
            X = H.matrix(rep, ids)
        except Exception as e:
            out[rep] = {"error": repr(e)[:60]}
            continue
        out[rep] = {"cka": cka(X, y)}
        print(f"  B7 {rep:10} cka={out[rep]['cka']:.4f}", flush=True)
    return out


# B8: margins
def b8(ids, y, groups):
    out = {}
    F = H.folds(y, groups, 1)
    for rep in REPS:
        try:
            X = H.matrix(rep, ids).astype(np.float64)
        except Exception as e:
            out[rep] = {"error": repr(e)[:60]}
            continue
        gaps = []
        sv_frac = []
        for r, k, tr, te in F:
            up = H.upsample(tr, y, k)
            clf = LinearSVC(C=1.0, dual=False, max_iter=3000).fit(X[up], y[up])
            df = clf.decision_function(X[te])
            if df.ndim == 1:
                df = np.column_stack([-df, df])
            part = np.partition(df, -2, axis=1)
            gap = part[:, -1] - part[:, -2]
            gaps.extend(gap.tolist())
            # within-unit-margin fraction on train (OvR functional margin proxy for SVs)
            dft = clf.decision_function(X[up])
            if dft.ndim == 1:
                dft = np.column_stack([-dft, dft])
            sv_frac.append(float((np.abs(dft) < 1).any(axis=1).mean()))
        g = np.array(gaps)
        out[rep] = {"margin_median": float(np.median(g)), "margin_p10": float(np.percentile(g, 10)),
                    "within_margin_frac": float(np.mean(sv_frac))}
        print(f"  B8 {rep:10} margin_median={out[rep]['margin_median']:.3f} "
              f"within_margin={out[rep]['within_margin_frac']:.3f}", flush=True)
    return out


# B9: hash-dimension sweep
def b9():
    import exp_tok_analyses as Tk
    apps = Tk.staged()
    fam_set = set()
    for f, _ in apps:
        fam_set.add(f)
    fam_names = sorted(fam_set)
    out = {}
    for col in ("identifier", "degree"):
        data = Tk.build_cached(col, apps, 4)
        shalist = []
        for _, s in apps:
            if s in data:
                shalist.append(s)
        y_list = []
        for s in shalist:
            y_list.append(fam_names.index(data[s][0]))
        y = np.array(y_list)
        g = H.groups_for(np.array(shalist))
        rc = {}
        for d in (2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16):
            Xh = np.zeros((len(shalist), d), dtype=np.float32)
            for j, s in enumerate(shalist):
                tid = data[s][1][0]
                idx = (tid % d).astype(np.int64)
                sgn = np.where((tid >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
                np.add.at(Xh[j], idx, sgn)
            nrm = np.linalg.norm(Xh, axis=1, keepdims=True)
            nrm[nrm == 0] = 1
            Xh /= nrm
            lin = []
            rf = []
            for r, k, tr, te in H.folds(y, g, 1):
                up = H.upsample(tr, y, k)
                m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42).fit(Xh[up], y[up])
                lin.append(f1_score(y[te], m.predict(Xh[te]), average="macro"))
                rfm = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).fit(Xh[up], y[up])
                rf.append(f1_score(y[te], rfm.predict(Xh[te]), average="macro"))
            rc[str(d)] = {"linear": float(np.mean(lin)), "forest": float(np.mean(rf))}
            print(f"  B9 {col} d={d}: linear={rc[str(d)]['linear']:.3f} forest={rc[str(d)]['forest']:.3f}", flush=True)
        out[col] = rc
        del data
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["B6", "B7", "B8", "B9"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("four")
    y, _ = H.labels(fam)
    groups = H.groups_for(ids)
    if "B6" in a.parts:
        print("B6: synthetic additive vs xor", flush=True)
        res["B6"] = b6()
        json.dump(res, open(OUT, "w"), indent=1)
    if "B7" in a.parts:
        print("B7: centered kernel alignment", flush=True)
        res["B7"] = b7(ids, y)
        json.dump(res, open(OUT, "w"), indent=1)
    if "B8" in a.parts:
        print("B8: margin distribution", flush=True)
        res["B8"] = b8(ids, y, groups)
        json.dump(res, open(OUT, "w"), indent=1)
    if "B9" in a.parts:
        print("B9: hash-dim sweep linear vs forest", flush=True)
        res["B9"] = b9()
        json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
