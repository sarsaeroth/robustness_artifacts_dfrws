#!/usr/bin/env python3
"""L1: interaction ladder across representations (robustness paper, sec:lindiag D1, D3).

Question: when the linear model wins, is that because the attribution problem is
linear, or because the representation made it so? For each representation, fit
on identical samples and folds:

  linear    SGD log loss, fit to convergence, alpha tuned by inner grouped CV
  additive  gradient boosting with depth-1 trees (nonlinear main effects, no interactions)
  pairwise  gradient boosting with depth-2 trees (pairwise interactions)
  gbdt      gradient boosting, unrestricted depth (31 leaves)
  rf        random forest, 500 trees
  svm_rbf   RBF-kernel SVM
  mlp       one hidden layer of 512

Four-class (the four proxy families, n=3,364), upsampling inside each training fold,
R repeats of shuffled StratifiedGroupKFold(5) on DEX groups. Flexible models also
report D3, linear-surrogate fidelity: a ridge regression on x is fitted to the
model's training-set log-odds and scored (R^2, argmax agreement) on the held-out fold.

Prediction (EXPERIMENTS.md L1): labeled WL flat (linear ~ additive ~ pairwise);
learned embeddings jump; invariant colourings in between.

Output: results/tier4/l1_ladder.jsonl (resume-safe), summary via --summarize.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "l1_ladder.jsonl")
MODELS = ["linear", "additive", "pairwise", "gbdt", "rf", "svm_rbf", "mlp"]
FLEXIBLE = {"additive", "pairwise", "gbdt", "rf", "svm_rbf", "mlp"}
REP_ORDER = ["labeled", "combined", "fuzzy", "api", "structure", "node2vec", "dex"]
SEED = 42
# SGD convergence tolerance for fit_linear; env LINEAR_TOL relaxes it for slow refits (40-class
# standardized scaling checks). Default 1e-5 reproduces every earlier result.
TOL = float(os.environ.get("LINEAR_TOL", "1e-5"))


def make(name, jobs):
    import gpu_models as G
    if name == "mlp" and G.ENABLED:
        return G.TorchMLP(hidden=512, alpha=1e-4, max_iter=300, seed=SEED)
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    if name == "additive" or name == "pairwise" or name == "gbdt":
        if name == "additive":
            depth = 1
        elif name == "pairwise":
            depth = 2
        else:
            depth = None
        return HistGradientBoostingClassifier(
            max_depth=depth, max_leaf_nodes=31, learning_rate=0.1, max_iter=500,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, n_jobs=jobs, random_state=SEED)
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=10.0, gamma="scale")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, early_stopping=True,
                             max_iter=300, random_state=SEED)
    raise ValueError(name)


def fit_linear(X, y, groups, tr_idx, seed):
    """SGD log loss to convergence; alpha chosen by inner 3-fold grouped CV.
    With GPU=1 (and a torch device), the same objective solved exactly on the GPU (gpu_models)."""
    import gpu_models as G
    if G.ENABLED:
        return G.fit_linear_gpu(X, y, groups, tr_idx, seed, H.upsample)
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    best, best_a = -1, None
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    for a in (1e-5, 1e-4, 1e-3):
        sc = []
        for itr, ite in inner.split(X[tr_idx], y[tr_idx], groups[tr_idx]):
            itr, ite = tr_idx[itr], tr_idx[ite]
            up = H.upsample(itr, y, seed)
            m = SGDClassifier(loss="log_loss", alpha=a, max_iter=2000, tol=TOL, random_state=SEED)
            m.fit(X[up], y[up])
            sc.append(f1_score(y[ite], m.predict(X[ite]), average="macro"))
        if np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    up = H.upsample(tr_idx, y, seed)
    m = SGDClassifier(loss="log_loss", alpha=best_a, max_iter=2000, tol=TOL, random_state=SEED)
    m.fit(X[up], y[up])
    return m, {"alpha": best_a}


def logodds(model, X):
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        return model.decision_function(X)
    if model.__class__.__name__ == "SVC":
        return model.decision_function(X)
    p = np.clip(model.predict_proba(X), 1e-6, 1.0)
    lp = np.log(p)
    return lp - lp.mean(1, keepdims=True)


_D = {}


def run(task, jobs):
    import time
    from sklearn.metrics import f1_score
    rep, r, f, model_name = task
    # per-representation data and folds, cached per worker process
    if rep not in _D:
        all_ids, all_fam = H.sample_ids("four")
        ids, fam = H.available(rep, all_ids, all_fam)
        X = H.matrix(rep, ids)
        y, _ = H.labels(fam)
        g = H.groups_for(ids)
        fold_map = {}
        for a, b, tr, te in H.folds(y, g, 5):
            fold_map[(a, b)] = (tr, te)
        _D[rep] = (X, y, g, fold_map)
    X, y, g, F = _D[rep]
    tr, te = F[(r, f)]
    t0 = time.time()
    extra = {}
    if model_name == "linear":
        m, extra = fit_linear(X, y, g, tr, 1000 * r + f)
    else:
        m = make(model_name, jobs)
        up = H.upsample(tr, y, 1000 * r + f)
        m.fit(X[up], y[up])
        # D3: ridge on x fitted to the model's training log-odds, scored on the test fold
        from sklearn.linear_model import Ridge
        ftr = logodds(m, X[tr])
        fte = logodds(m, X[te])
        ridge = Ridge(alpha=1.0).fit(X[tr], ftr)
        spred = ridge.predict(X[te])
        ss_res = ((fte - spred) ** 2).sum()
        ss_tot = ((fte - fte.mean(0)) ** 2).sum()
        if fte.ndim == 1:  # binary decision_function: one score per app; agreement = same predicted class (sign)
            agree = float(((spred > 0) == (fte > 0)).mean())
        else:
            agree = float((spred.argmax(1) == fte.argmax(1)).mean())
        extra = {"surrogate_r2": float(1 - ss_res / ss_tot), "surrogate_argmax_agree": agree}
    pred = m.predict(X[te])
    row = dict(rep=rep, r=r, f=f, model=model_name,
               f1=float(f1_score(y[te], pred, average="macro")),
               n_train=int(len(tr)), n_test=int(len(te)),
               seconds=round(time.time() - t0, 1))
    for k in extra:
        row[k] = extra[k]
    return row


def summarize():
    rows = []
    for l in open(OUT):
        rows.append(json.loads(l))
    by = {}
    for r in rows:
        by.setdefault((r["rep"], r["model"]), {})[(r["r"], r["f"])] = r
    out = {}
    header = []
    for m in MODELS:
        header.append(f"{m:>9}")
    print(f"{'rep':10} " + " ".join(header))
    for rep in REP_ORDER:
        if (rep, "linear") not in by:
            continue
        lin = by[(rep, "linear")]
        cells = []
        out[rep] = {}
        for m in MODELS:
            d = by.get((rep, m), {})
            keys = sorted(set(d) & set(lin))
            if not keys:
                cells.append(f"{'-':>9}")
                continue
            f1_list = []
            diff = []
            for k in keys:
                f1_list.append(d[k]["f1"])
                diff.append(d[k]["f1"] - lin[k]["f1"])
            f1 = np.array(f1_list)
            n_tr = d[keys[0]]["n_train"]
            n_te = d[keys[0]]["n_test"]
            md, t, p = H.corrected_t(diff, n_tr, n_te)
            rec = {"f1": float(f1.mean()), "sd": float(f1.std()), "n_folds": len(keys),
                   "diff_vs_linear": md, "t": t, "p": p}
            if m in FLEXIBLE:
                r2s = []
                agrees = []
                for k in keys:
                    r2s.append(d[k]["surrogate_r2"])
                    agrees.append(d[k]["surrogate_argmax_agree"])
                rec["surrogate_r2"] = float(np.mean(r2s))
                rec["surrogate_agree"] = float(np.mean(agrees))
            out[rep][m] = rec
            if m == "linear":
                star = ""
            elif p < 0.05:
                star = "*"
            else:
                star = " "
            cells.append(f"{f1.mean():8.3f}{star}")
        print(f"{rep:10} " + " ".join(cells))
    print("\n* = differs from linear, corrected resampled t-test p < 0.05")
    print("\nsurrogate R^2 (argmax agreement) of flexible models:")
    for rep, d in out.items():
        parts = []
        for m in MODELS:
            if m in d and m in FLEXIBLE:
                parts.append(f"{m}={d[m]['surrogate_r2']:.2f}({d[m]['surrogate_agree']:.2f})")
        print(f"{rep:10} " + " ".join(parts))
    json.dump(out, open(OUT.replace(".jsonl", "_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=REP_ORDER)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeats-labeled", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["rep"], r["r"], r["f"], r["model"]))
    # pending (rep, repeat, fold, model) tasks
    tasks = []
    for rep in a.reps:
        if rep == "labeled":
            n_rep = a.repeats_labeled
        else:
            n_rep = a.repeats
        for r in range(n_rep):
            for f in range(5):
                for m in a.models:
                    t = (rep, r, f, m)
                    if t not in done:
                        tasks.append(t)
    # slow models first; group by representation so each worker reuses its cache
    order = {"svm_rbf": 0, "gbdt": 1, "pairwise": 2, "additive": 3, "mlp": 4, "rf": 5, "linear": 6}
    tasks.sort(key=lambda t: (order[t[3]], t[0]))
    print(f"{len(tasks)} tasks ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(OUT, "a") as out:
        futs = {}
        for t in tasks:
            futs[ex.submit(run, t, a.jobs)] = t
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"[{k}/{len(tasks)}] {r['rep']} r{r['r']}f{r['f']} {r['model']} f1={r['f1']:.4f} {r['seconds']}s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
