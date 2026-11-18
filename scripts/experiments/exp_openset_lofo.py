#!/usr/bin/env python3
"""Open-set and generalization evaluations on vectors, for any dataset (DATASET env).

  lofo       leave-one-family-out detection: a binary target-vs-benign linear model trained on
             all target families but one (grouped folds for the benign class) is applied to the
             held-out family; TPR at the threshold giving 1% FPR on held-out benign apps.
  novelty    open-set scoring of applications outside the training families (the dataset's
             benign class, and the AndroZoo Play sample where pilot vectors exist): maximum
             softmax probability, distance to the nearest class mean (cosine), and a one-class
             SVM per family (max decision value). AUROC known-vs-unknown, and PPV against
             prevalence at the threshold keeping 95% of known applications:
             PPV(pi) = TPR pi / (TPR pi + FPR (1 - pi)).
  temporal   train on applications dated before the median, test on the later (DEX date; corpus v3: first-seen date, proxy_v3_meta.py)
             half (where dates exist; AndroZoo's 1980 placeholder is treated as missing).
Representation: labeled (identifier WL; GRAPHS env selects blocks).
Output: tier4/v4/openset_lofo.json (dataset-aware). ONLY=temporal recomputes only the temporal split.
"""
import csv
import glob
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_l1_ladder as L

OUT = H.out_path("tier4", "v4", "openset_lofo.json")
REP = "labeled"


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def wild():
    pilot = os.path.join(H.ROOT, "results", "androzoo_pilot")
    ok = set()
    for f in glob.glob(os.path.join(pilot, "results_5class_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["status"] == "ok":
                ok.add(r["sha256"])
    ok -= H.wild_flagged()
    if not ok:
        return None
    vecs = []
    for s in sorted(ok):
        vecs.append(np.load(os.path.join(pilot, "vectors", f"{s}.npy")))
    W = np.stack(vecs).astype(np.float32)
    blocks = {"cfg": W[:, :1024], "fcg": W[:, 1024:]}
    parts = []
    for k in H.GRAPHS:
        parts.append(blocks[k])
    return np.hstack(parts)


def novelty(X, y, g, names, benign_idx, Xw):
    from sklearn.svm import OneClassSVM
    known = np.where(y != benign_idx)[0]
    unk_sets = {"benign_class": np.where(y == benign_idx)[0]}
    res = {}
    F = H.folds(y[known], g[known], 1)
    scores = {}
    for k in ("msp", "centroid", "ocsvm"):
        scores[k] = {"known": [], "benign_class": [], "wild": []}
    for r, k, tr, te in F:
        tr = known[tr]
        te = known[te]
        yk = y[tr]
        m, _ = L.fit_linear(X, y, g, tr, k)
        cls = m.classes_
        # class centroids (cosine) and one one-class SVM per class
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        means = []
        for c in cls:
            means.append(Xn[tr][yk == c].mean(0))
        mu = np.stack(means)
        mu /= np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
        oc = []
        for c in cls:
            oc.append(OneClassSVM(nu=0.05, gamma="scale").fit(X[tr][yk == c]))
        evals = {"known": X[te], "benign_class": X[unk_sets["benign_class"]]}
        if Xw is not None:
            evals["wild"] = Xw
        for nm, A in evals.items():
            An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
            scores["msp"][nm].append(softmax(m.decision_function(A)).max(1))
            scores["centroid"][nm].append((An @ mu.T).max(1))
            oc_scores = []
            for o in oc:
                oc_scores.append(o.decision_function(A))
            scores["ocsvm"][nm].append(np.max(np.stack(oc_scores), 0))
    for sname, d in scores.items():
        kn = np.concatenate(d["known"])
        thr = float(np.quantile(kn, 0.05))
        res[sname] = {}
        for unk in ("benign_class", "wild"):
            if not d[unk]:
                continue
            u = np.concatenate(d[unk])
            fpr = float((u > thr).mean())
            tpr = 0.95
            ppv = {}
            for pi in (1e-4, 1e-3, 1e-2, 1e-1):
                if fpr > 0:
                    ppv[str(pi)] = float(tpr * pi / (tpr * pi + fpr * (1 - pi)))
                else:
                    ppv[str(pi)] = 1.0
            res[sname][unk] = {"auroc": float(roc_auc_score(np.r_[np.ones(len(kn)), np.zeros(len(u))], np.r_[kn, u])),
                               "fpr_at_tpr95": fpr,
                               "ppv": ppv}
        print("  novelty", sname, res[sname], flush=True)
    return res


def temporal(X, y, names, ids):
    if H.V3:
        meta_ds = "proxy_v3"
    else:
        meta_ds = H.DATASET
    meta_p = os.path.join(H.ROOT, "results", "datasets", meta_ds, "meta.json")
    if not os.path.exists(meta_p):
        return {"skipped": "no dates for this dataset"}
    meta = json.load(open(meta_p))
    if H.V3:
        field = "first_seen"  # see proxy_v3_meta.py
    else:
        field = "dex_date"
    # date string per app; an app missing from meta gets ""
    dates = []
    for s in ids:
        if s in meta:
            dates.append(meta[s].get(field, "")[:10])
        else:
            dates.append("")
    d = np.array(dates)
    ok_list = []
    for x in d:
        ok_list.append(len(x) == 10 and not x.startswith("1980"))
    ok = np.array(ok_list)
    if ok.mean() < 0.5:
        return {"skipped": f"only {ok.mean():.2f} of apps dated"}
    med = np.sort(d[ok])[ok.sum() // 2]
    tr = np.where(ok & (d < med))[0]
    te = np.where(ok & (d >= med))[0]
    g = np.arange(len(y))
    m, _ = L.fit_linear(X, y, g, tr, 0)
    present = sorted(set(y[tr]) & set(y[te]))
    msk = np.isin(y[te], present)
    return {"date_field": field, "split_date": str(med), "n_train": int(len(tr)), "n_test": int(msk.sum()),
            "f1_temporal": float(f1_score(y[te][msk], m.predict(X[te][msk]), average="macro", labels=present))}


def main():
    if os.environ.get("ONLY") == "temporal" and os.path.exists(OUT):  # add the split to an existing result
        res = json.load(open(OUT))
        ids4, fam4 = H.sample_ids("four")
        y4, names4 = H.labels(fam4)
        res["temporal"] = temporal(H.matrix(REP, ids4).astype(np.float64), y4, names4, ids4)
        print("  temporal", res["temporal"], flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        return
    ids, fam = H.sample_ids("five")
    names = sorted(set(fam.tolist()))
    y_list = []
    for f in fam:
        y_list.append(names.index(f))
    y = np.array(y_list)
    benign_idx = names.index(H.BENIGN)
    X = H.matrix(REP, ids).astype(np.float64)
    g = H.groups_for(ids)
    res = {"dataset": H.DATASET, "groups": H.GROUPS, "graphs": list(H.GRAPHS)}
    if len(names) - 1 < 2:  # e.g. piggyback: one malicious class, nothing to hold out
        skipped = {"skipped": "fewer than two non-benign classes"}
        res["lofo"] = skipped
        res["novelty"] = skipped
    else:
        # leave-one-family-out detection; benign groups split once into train/test
        lofo_out = {}
        tgt = []
        for c in range(len(names)):
            if c != benign_idx:
                tgt.append(c)
        ben = np.where(y == benign_idx)[0]
        rng = np.random.default_rng(0)
        gb = np.unique(g[ben])
        rng.shuffle(gb)
        test_g = set(gb[: max(1, len(gb) // 5)])
        ben_te_list = []
        ben_tr_list = []
        for i in ben:
            if g[i] in test_g:
                ben_te_list.append(i)
        for i in ben:
            if g[i] not in test_g:
                ben_tr_list.append(i)
        ben_te = np.array(ben_te_list)
        ben_tr = np.array(ben_tr_list)
        for f in tgt:
            tr = np.concatenate([np.where((y != benign_idx) & (y != f))[0], ben_tr])
            yb = (y != benign_idx).astype(int)
            m, _ = L.fit_linear(X, yb, g, tr, f)
            s = m.decision_function(X)
            t = float(np.quantile(s[ben_te], 1 - 0.01))  # threshold at 1% FPR on held-out benign
            held = np.where(y == f)[0]
            lofo_out[names[f]] = {"n": int(len(held)), "tpr_at_fpr1": float((s[held] > t).mean()),
                                  "auc_vs_benign": float(roc_auc_score(np.r_[np.ones(len(held)), np.zeros(len(ben_te))],
                                                                       np.r_[s[held], s[ben_te]]))}
            print("  lofo", names[f], lofo_out[names[f]], flush=True)
        res["lofo"] = lofo_out
        json.dump(res, open(OUT, "w"), indent=1)
        res["novelty"] = novelty(X, y, g, names, benign_idx, wild())
    json.dump(res, open(OUT, "w"), indent=1)
    ids4, fam4 = H.sample_ids("four")
    y4, _ = H.labels(fam4)
    res["temporal"] = temporal(H.matrix(REP, ids4).astype(np.float64), y4, names, ids4)
    print("  temporal", res["temporal"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
