#!/usr/bin/env python3
"""Explainer metrics M3-M6 (robustness paper, sec:method_explainers).

X2 (exp_x2_method_level.py) reports M1 precision and M2 agreement. This adds, for
the identifier-stable families (ipninja, luminati) at d=1,024, FCG labeled WL:

  M3 faithfulness   delete the top-k% attributed methods from the FCG (real graph
                    edit, full re-vectorization), record the explained model's
                    margin retention; random deletion of the same number of
                    methods is the control. k in {1, 5, 20}%.
  M4 stability      retrain each model on 3 bootstrap resamples of its training
                    fold; top-100 method Jaccard between the attributions.
  M5 sanity         retrain each model on permuted labels; precision@target and
                    Spearman correlation with the trained model's attribution.
  M6 cost           wall-clock seconds per explanation.

Explainers: lin_exact (Proposition 1), rf_path and xgb_path (TreeSHAP), mlp_ig
(Integrated Gradients, 64 steps), mlp_gxi. Interventional TreeSHAP is omitted
here for cost; X2 shows it tracks path-dependent TreeSHAP at d=1,024.

Output: results/tier4/p3/explainer_mx.json
"""
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_x2_method_level as X2
import gen_wl_vectors as G
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "p3", "explainer_mx.json")
# stable-identifier families
if H.DATASET == "proxy":
    FAMS = ["ipninja", "luminati"]
else:
    FAMS = list(H.PROXY)
EXPL = ["lin_exact", "rf_path", "xgb_path", "mlp_ig", "mlp_gxi"]
KFRAC = [0.01, 0.05, 0.20]
N_BOOT = 3


def fit_all(X, y, idx, seed, shuffle=False):
    import random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    up = H.upsample(idx, y, seed)
    yy = y.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        yy[idx] = rng.permutation(y[idx])
        up = H.upsample(idx, yy, seed)
    lin = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=seed).fit(X[up], yy[up])
    rf = RandomForestClassifier(n_estimators=200, n_jobs=6, random_state=seed).fit(X[up], yy[up])
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=6,
                        random_state=seed, verbosity=0).fit(X[up], yy[up])
    mlp = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, early_stopping=True,
                        max_iter=300, random_state=seed).fit(X[up], yy[up])
    return {"lin": lin, "rf": rf, "xgb": xgb, "mlp": mlp}


def margin(model, x, c):
    xr = x.reshape(1, -1)
    if hasattr(model, "decision_function") and model.__class__.__name__ == "SGDClassifier":
        s = model.decision_function(xr)[0]
        if np.ndim(s) == 0:  # binary: one score, the class-1 margin
            if c == 1:
                return float(s)
            return float(-s)
    else:
        s = np.log(np.clip(model.predict_proba(xr)[0], 1e-9, 1))
    o = np.delete(s, c)
    return float(s[c] - o.max())


def explain(models, x, prov, nrm, mu, c):
    import shap
    import shap_compat  # noqa: F401  additivity-check fallback for xgboost 3.x
    out = {}
    cost = {}
    t = time.time()
    w = X2.lin_w(models["lin"], c)
    out["lin_exact"] = P.split_bucket_credit(prov, w, mu, nrm, w * (x - mu), rule="exact")
    cost["lin_exact"] = time.time() - t
    for name in ("rf", "xgb"):
        t = time.time()
        v = np.asarray(shap.TreeExplainer(models[name]).shap_values(x.reshape(1, -1)))
        phi = X2.shap_phi(v, c)
        out[f"{name}_path"] = X2.to_methods(phi, prov)
        cost[f"{name}_path"] = time.time() - t
    t = time.time()
    ig = X2.mlp_ig(models["mlp"], x, mu, c)
    out["mlp_ig"] = X2.to_methods(ig, prov)
    cost["mlp_ig"] = time.time() - t
    t = time.time()
    grad = X2.mlp_grad(models["mlp"], x, c)
    out["mlp_gxi"] = X2.to_methods(x * grad, prov)
    cost["mlp_gxi"] = time.time() - t
    return out, cost


def ranked(credit):
    ms = list(credit)
    scores = []
    for m in ms:
        scores.append(credit[m])
    sc = np.array(scores)
    # methods by descending credit, and method -> credit
    order = []
    for i in np.argsort(-sc):
        order.append(ms[i])
    by_method = {}
    for i in range(len(ms)):
        by_method[ms[i]] = sc[i]
    return order, by_method


def delete_and_vectorize(nl, su, pr, drop_methods):
    keep = set()
    for n, r in nl.items():
        if G.canonical_label(r, "fcg") not in drop_methods:
            keep.add(n)
    # induced subgraph on the kept nodes
    nl2 = {}
    for n in keep:
        nl2[n] = nl[n]
    su2 = {}
    for n in keep:
        kept_succ = []
        if n in su:
            for v in su[n]:
                if v in keep:
                    kept_succ.append(v)
        su2[n] = kept_succ
    pr2 = {}
    for n in keep:
        kept_pred = []
        if n in pr:
            for v in pr[n]:
                if v in keep:
                    kept_pred.append(v)
        pr2[n] = kept_pred
    vec = G.wl_hashed_vector(nl2, su2, pr2, "fcg", 1024, iters=2, drop_lib=True, use_direction=True)
    return vec.astype(float)


def main():
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    X = H.matrix("labeled_fcg", ids)
    g = H.groups_for(ids)
    ids5, fam5 = H.sample_ids("five")
    mu = H.matrix("labeled_fcg", ids5)[fam5 == H.BENIGN].astype(float).mean(0)
    F = H.folds(y, g, 1)
    # app id -> test fold
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    staged = []
    for f in FAMS:
        for s in H.staged(f):
            if s in fold_of:
                staged.append((f, s))
    rows = []
    if os.path.exists(OUT):  # resume after a memory abort from the per-fold partial dump
        prev = json.load(open(OUT))
        if "summary" not in prev:
            if "apps" in prev:
                rows = prev["apps"]
            else:
                rows = []
    done_shas = set()
    for a in rows:
        done_shas.add(a.get("sha"))
    for r, k, tr, te in F:
        todo = []
        for f, s in staged:
            if fold_of[s] == k:
                todo.append((f, s))
        if not todo:
            continue
        all_done = True
        for _, s in todo:
            if s not in done_shas:
                all_done = False
                break
        if all_done:
            print(f"fold {k}: resumed from partial results", flush=True)
            continue
        # resume within a fold after a memory abort
        remaining = []
        for f, s in todo:
            if s not in done_shas:
                remaining.append((f, s))
        todo = remaining
        t0 = time.time()
        # fitted model sets are cached per fold, so a run killed by the memory guard during the long
        # training phase resumes with the sets it already finished (same seeds, same models)
        import joblib
        mdir = os.path.join(os.path.dirname(OUT), "explainer_mx_models")
        os.makedirs(mdir, exist_ok=True)

        def cached(name, fn):
            pth = os.path.join(mdir, f"fold{k}_{name}.joblib")
            if os.path.exists(pth):
                return joblib.load(pth)
            m = fn()
            joblib.dump(m, pth + ".tmp")
            os.replace(pth + ".tmp", pth)
            return m

        base = cached("base", lambda: fit_all(X, y, tr, 0))
        rng = np.random.default_rng(k)
        boots = []
        for b in range(N_BOOT):
            bi = rng.choice(tr, len(tr), replace=True)  # drawn every time, so the RNG stream is unchanged on resume
            boots.append(cached(f"boot{b}", lambda bi=bi, b=b: fit_all(X, y, np.unique(bi), 100 + b)))
        rand_m = cached("rand", lambda: fit_all(X, y, tr, 7, shuffle=True))
        print(f"fold {k}: models in {time.time()-t0:.0f}s, {len(todo)} apps", flush=True)
        for f, sha in todo:
            nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
            x, prov, nrm = P.provenance_vector(nl, su, pr)
            if x is None:
                continue
            c = int(le.transform([f])[0])
            att, cost = explain(base, x, prov, nrm, mu, c)
            rank = {}
            for e in EXPL:
                rank[e] = ranked(att[e])
            methods = rank["lin_exact"][0]
            truth = {}
            for m in methods:
                truth[m] = m.startswith(P.SDK_PREFIXES.get(f, ()))
            nt = sum(truth.values())
            rec = {"family": f, "sha": sha, "n_methods": len(methods), "n_target": nt, "cost": cost,
                   "M3": {}, "M4": {}, "M5": {}}
            # M4
            batt = []
            for bm in boots:
                boot_att, _ = explain(bm, x, prov, nrm, mu, c)
                batt.append(boot_att)
            for e in EXPL:
                tops = []
                for a in batt:
                    tops.append(set(ranked(a[e])[0][:100]))
                # pairwise Jaccard of the bootstrap top-100 sets
                js = []
                for i in range(len(tops)):
                    for j in range(i + 1, len(tops)):
                        js.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
                rec["M4"][e] = float(np.mean(js))
            # M5
            ratt, _ = explain(rand_m, x, prov, nrm, mu, c)
            for e in EXPL:
                rr, rs = ranked(ratt[e])
                common = list(rank[e][1])
                trained_scores = []
                random_scores = []
                for m in common:
                    trained_scores.append(rank[e][1][m])
                for m in common:
                    random_scores.append(rs.get(m, 0.0))
                rho = stats.spearmanr(trained_scores, random_scores)[0]
                if nt:
                    hits = []
                    for m in rr[:nt]:
                        hits.append(truth.get(m, False))
                    p_rand = float(np.mean(hits))
                    hits = []
                    for m in rank[e][0][:nt]:
                        hits.append(truth.get(m, False))
                    p_true = float(np.mean(hits))
                else:
                    p_rand = None
                    p_true = None
                rec["M5"][e] = {"spearman_vs_trained": float(rho), "p_at_target_random": p_rand,
                                "p_at_target_trained": p_true}
            # M3
            mdl = {"lin_exact": base["lin"], "rf_path": base["rf"], "xgb_path": base["xgb"],
                   "mlp_ig": base["mlp"], "mlp_gxi": base["mlp"]}
            m0 = {}
            for e in EXPL:
                m0[e] = margin(mdl[e], x, c)
            rr_rng = np.random.default_rng(hash(sha) % 2 ** 32)
            for kf in KFRAC:
                kk = max(1, int(kf * len(methods)))
                drawn = rr_rng.choice(methods, kk, replace=False)
                rand_drop = set(drawn.tolist())
                xr = delete_and_vectorize(nl, su, pr, rand_drop)
                for e in EXPL:
                    xd = delete_and_vectorize(nl, su, pr, set(rank[e][0][:kk]))
                    if m0[e]:
                        retention = margin(mdl[e], xd, c) / m0[e]
                        retention_random = margin(mdl[e], xr, c) / m0[e]
                    else:
                        retention = None
                        retention_random = None
                    flip = bool(margin(mdl[e], xd, c) < 0)
                    if e not in rec["M3"]:
                        rec["M3"][e] = {}
                    rec["M3"][e][str(kf)] = {"retention": retention, "retention_random": retention_random,
                                             "flip": flip}
            rows.append(rec)
            json.dump({"apps": rows}, open(OUT, "w"))  # per-app checkpoint
            m4_parts = []
            for e in EXPL:
                m4_parts.append(f"{e}={rec['M4'][e]:.2f}")
            print(f"  {f} {sha[:10]} M4 " + " ".join(m4_parts), flush=True)
            del nl, su, pr, prov
        json.dump({"apps": rows}, open(OUT, "w"))
        for fn_ in os.listdir(mdir):  # fold finished: its cached models are no longer needed (disk)
            if fn_.startswith(f"fold{k}_"):
                os.remove(os.path.join(mdir, fn_))
    summ = {}
    rows_all = rows
    # summaries over apps with payload ground truth (piggyback originals have none)
    rows = []
    for r in rows_all:
        if r.get("n_target"):
            rows.append(r)
    for e in EXPL:
        m4 = []
        m5_rho = []
        m5_rand = []
        m5_true = []
        m6 = []
        for r in rows:
            m4.append(r["M4"][e])
            m5_rho.append(r["M5"][e]["spearman_vs_trained"])
            m5_rand.append(r["M5"][e]["p_at_target_random"])
            m5_true.append(r["M5"][e]["p_at_target_trained"])
            m6.append(r["cost"][e])
        s = {"M4_top100_jaccard": float(np.mean(m4)),
             "M5_spearman": float(np.nanmean(m5_rho)),
             "M5_p_random": float(np.mean(m5_rand)),
             "M5_p_trained": float(np.mean(m5_true)),
             "M6_seconds": float(np.median(m6))}
        for kf in KFRAC:
            v = []
            for r in rows:
                if r["M3"][e][str(kf)]["retention"] is not None:
                    v.append(r["M3"][e][str(kf)])
            ret = []
            ret_rand = []
            flips = []
            for x in v:
                ret.append(x["retention"])
            for x in v:
                ret_rand.append(x["retention_random"])
            for x in v:
                flips.append(x["flip"])
            s[f"M3_retention_{kf}"] = float(np.median(ret))
            s[f"M3_retention_random_{kf}"] = float(np.median(ret_rand))
            s[f"M3_flip_{kf}"] = float(np.mean(flips))
        summ[e] = s
    ratios = []
    for r in rows:
        ratios.append(r["n_target"] / r["n_methods"])
    base_rate = float(np.mean(ratios))
    json.dump({"apps": rows_all, "summary": summ, "base_rate": base_rate, "n_summarized": len(rows)},
              open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
