#!/usr/bin/env python3
"""Drebin features: vectorization, accuracy and the exact certificate against feature addition.

Features are the Drebin sets written by fetch_worker.py ({sha}.drebin.json): S1 hardware,
S2 requested permissions, S3 components, S4 intent actions/categories, S5 framework API calls,
S7 suspicious API calls, S8 network addresses. Binary indicator per feature, vocabulary = features
present in at least MIN_APPS applications. A linear model (the ladder's tuned linear SGD, and a
linear SVM as in Drebin) is fitted on DEX-grouped / near-duplicate-grouped folds (harness).

Certificate. The adversary adds features (adds code, manifest entries or strings without
removing anything), so x_j can move from 0 to 1 only. For the margin over rival c with
u = w_y - w_c, adding a set A of absent features changes the margin by sum_{j in A} u_j, so the
minimum number of additions that flips the decision against c is the smallest k with
M + (sum of the k most negative u_j over absent j) <= 0, and the certificate min_c k_c is exact
(it equals the strongest attack). Raw binary features are used without normalization so that
the bound is the classical one (Demontis et al., TDSC 2019, study the same threat model on
Drebin). If the negative mass is insufficient, the decision is unconditionally robust.

Output: vectors drebin.npz (full sets; datasets with APKs) and drebin_code.npz (S5, S7, S8; all
datasets, proxy from drebin_code.py), and tier4/v4/cert_drebin.json (dataset-aware).
Usage: DATASET=... exp_cert_drebin.py [vectorize|evaluate]
"""
import json
import os
import sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "v4", "cert_drebin.json")
MIN_APPS = 5
SETS = ("S1", "S2", "S3", "S4", "S5", "S7", "S8")


CODE_SETS = ("S5", "S7", "S8")


def vectorize():
    for name, sets in (("drebin", SETS), ("drebin_code", CODE_SETS)):
        if name == "drebin" and (H.DATASET == "proxy" or H.V3):
            continue
        _vectorize(name, sets)


def _vectorize(name, sets):
    """Two passes over the per-app JSON records so that no more than one app's feature set is held
    at a time: count document frequencies, then emit the sparse matrix."""
    import scipy.sparse as sp
    ids, fam = H.sample_ids("five")
    # per-app Drebin record path; apps without a record are dropped
    apps = []
    for f, s in zip(fam, ids):
        if H.DATASET == "proxy":  # no APKs: code-only records from drebin_code.py
            p = os.path.join(H.ROOT, "results", "robust2", "drebin_code", f"{s}.drebin.json")
        elif H.V3:  # new apps: staged full records (S5/S7/S8 agree); v2 apps: drebin_code.py
            p = os.path.join(H.GRAPHS_DIR, f, s, f"{s}.drebin.json")
            if not os.path.exists(p):
                p = os.path.join(H.ROOT, "results", "robust2", "drebin_code", f"{s}.drebin.json")
        else:
            if H.SPEC["full_graphs"][0] == "dir":
                root = H.SPEC["full_graphs"][1]
            else:
                root = H.GRAPHS_DIR
            p = os.path.join(root, f, s, f"{s}.drebin.json")
        if os.path.exists(p):
            apps.append((s, f, p))

    def feats(p):
        d = json.load(open(p))
        out = set()
        for k in sets:
            for v in d.get(k, []):
                out.add(f"{k}::{v}")
        return out

    # pass 1: document frequencies -> vocabulary
    cnt = Counter()
    for _, _, p in apps:
        cnt.update(feats(p))
    vocab = []
    for t, c in cnt.items():
        if c >= MIN_APPS:
            vocab.append(t)
    vocab.sort()
    vi = {}
    for i, t in enumerate(vocab):
        vi[t] = i
    del cnt
    # pass 2: sparse rows
    indptr = [0]
    indices = []
    for _, _, p in apps:
        row = []
        for t in feats(p):
            if t in vi:
                row.append(vi[t])
        row.sort()
        indices.extend(row)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    X = sp.csr_matrix((data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
                      shape=(len(apps), len(vocab)))
    if H.DATASET != "proxy":
        out = H.VEC
    else:
        out = os.path.join(H.ROOT, "results", "robust2", "vectors")
    sp.save_npz(os.path.join(out, f"{name}_X.npz"), X)
    app_ids = []
    app_fams = []
    for s, f, _ in apps:
        app_ids.append(s)
        app_fams.append(f)
    np.savez(os.path.join(out, f"{name}_meta.npz"), ids=np.array(app_ids),
             family=np.array(app_fams), vocab=np.array(vocab))
    print(f"{name}: {X.shape}, nnz {X.nnz} from {len(apps)} apps", flush=True)


def cert_one(W, b, x, y):
    k_best = np.inf
    unconditional = True
    absent = x == 0
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = W[y] - W[c]
        M = float(u @ x + b[y] - b[c])
        # fewest additions: running sum of the most negative u_j over absent features
        neg = np.sort(u[absent])
        neg = neg[neg < 0]
        cs = M + np.cumsum(neg)
        hit = np.nonzero(cs <= 0)[0]
        if len(hit):
            unconditional = False
            k_best = min(k_best, int(hit[0]) + 1)
    if unconditional:
        return None
    return int(k_best)


def evaluate():
    res = {}
    for name in ("drebin", "drebin_code"):
        if os.path.exists(os.path.join(H.VEC, f"{name}_X.npz")):
            res[name] = _evaluate(name)
            json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


def _evaluate(rep):
    import exp_l1_ladder as L
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC
    import scipy.sparse as sp
    from sklearn.ensemble import RandomForestClassifier
    z = np.load(os.path.join(H.VEC, f"{rep}_meta.npz"))
    XA = sp.load_npz(os.path.join(H.VEC, f"{rep}_X.npz")).tocsr()
    idx = {}
    for i, s in enumerate(z["ids"]):
        idx[s] = i
    ids, fam = H.sample_ids("four")
    keep_list = []
    for s in ids:
        keep_list.append(s in idx)
    keep = np.array(keep_list)
    ids = ids[keep]
    fam = fam[keep]
    row_idx = []
    for s in ids:
        row_idx.append(idx[s])
    X = XA[row_idx].astype(np.float64)
    y, _ = H.labels(fam)
    g = H.groups_for(ids)
    res = {"dataset": H.DATASET, "groups": H.GROUPS, "n": int(len(ids)), "dim": int(X.shape[1])}
    f1 = []
    for r, k, tr, te in H.folds(y, g, 1):  # nonlinear reference on the same folds (no certificate)
        up = H.upsample(tr, y, k)
        rf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=42)
        rf.fit(X[up], y[up])
        f1.append(f1_score(y[te], rf.predict(X[te]), average="macro"))
    res["rf"] = {"f1": float(np.mean(f1))}
    print(rep, "rf", res["rf"], flush=True)
    for name in ("linear_sgd", "linear_svm"):
        f1 = []
        rows = []
        for r, k, tr, te in H.folds(y, g, 1):
            if name == "linear_sgd":
                m, _ = L.fit_linear(X, y, g, tr, k)
            else:
                up = H.upsample(tr, y, k)
                m = LinearSVC(C=1.0, max_iter=5000)
                m.fit(X[up], y[up])
            W = np.atleast_2d(m.coef_).astype(np.float64)
            b = np.atleast_1d(m.intercept_).astype(np.float64)
            if W.shape[0] == 1:  # binary: expand to two rows
                W = np.vstack([-W[0], W[0]])
                b = np.array([-b[0], b[0]])
            pred = np.asarray(X[te] @ W.T + b).argmax(1)
            f1.append(f1_score(y[te], pred, average="macro"))
            # certificate for every correctly classified test app
            for j, i in enumerate(te):
                if pred[j] == y[i]:
                    rows.append({"family": str(fam[i]), "k": cert_one(W, b, X[i].toarray().ravel(), int(y[i]))})
        # summary, evaluated in the same order as the original dict literal
        ks = []
        for row in rows:
            if row["k"] is not None:
                ks.append(row["k"])
        f1_mean = float(np.mean(f1))
        n_certified = len(rows)
        none_flags = []
        for row in rows:
            none_flags.append(row["k"] is None)
        unconditional_frac = float(np.mean(none_flags))
        if ks:
            k_median = float(np.median(ks))
        else:
            k_median = None
        if ks:
            k_q10_q90 = [float(np.percentile(ks, 10)), float(np.percentile(ks, 90))]
        else:
            k_q10_q90 = None
        families = set()
        for row in rows:
            families.add(row["family"])
        per_family = {}
        for f in sorted(families):
            n_fam = 0
            fam_ks = []
            for row in rows:
                if row["family"] == f:
                    n_fam += 1
                    if row["k"] is not None:
                        fam_ks.append(row["k"])
            if fam_ks:
                fam_k_median = float(np.median(fam_ks))
            else:
                fam_k_median = None
            per_family[f] = {"n": n_fam, "k_median": fam_k_median}
        res[name] = {"f1": f1_mean, "n_certified": n_certified,
                     "unconditional": unconditional_frac,
                     "k_median": k_median,
                     "k_q10_q90": k_q10_q90,
                     "per_family": per_family}
        summary = {}
        for key, v in res[name].items():
            if key != "per_family":
                summary[key] = v
        print(rep, name, summary, flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "vectorize":
        vectorize()
    else:
        evaluate()
