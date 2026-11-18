#!/usr/bin/env python3
"""X2: method-level explainer comparison against SDK namespace ground truth.

For each staged proxy app (work/graphs40), the FCG is re-vectorized with token
provenance (provenance_localization.provenance_vector), and every explainer's
bucket-level attribution for the app's own family is mapped to methods by one
common rule: bucket j's score is split over the tokens in j in proportion to
occupancy |s_t| / sum |s|, and each method sums its tokens. The linear model also
gets the exact decomposition of Proposition 1 (lin_exact).

Models (FCG-only labeled WL, d=1,024; trained on the harness fold in which the app
is held out, repeat 0): SGD log loss (converged), RF 200, XGB 200, MLP (512).
Explainers:
  lin_exact  Prop. 1, background = non_proxy mean
  lin_int    w (.) (x - mu_nonproxy), common rule
  lin_zero   w (.) x, common rule (= gradient x input = IG(0) on a linear model)
  rf_path, rf_int, xgb_path, xgb_int   TreeSHAP, interventional background = 100 non_proxy apps
  mlp_ig     Integrated Gradients, baseline mu_nonproxy, 64 steps
  mlp_gxi    gradient x input
Metrics per app: precision at |target| and at 25, average precision, with the
base rate; pairwise top-100 method Jaccard between explainers.
Ground truth is reliable for ipninja and luminati; monkeysocks and oxylabs are
reported separately (their namespaces are renamed or unstable per build).

Output: results/tier4/x2_method_level.json
"""
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "x2_method_level.json")
EXPLAINERS = ["lin_exact", "lin_int", "lin_zero", "rf_path", "rf_int",
              "xgb_path", "xgb_int", "mlp_ig", "mlp_gxi"]


def mlp_grad(mlp, x, c):
    """Gradient of class-c logit wrt x for a one-hidden-layer ReLU MLP."""
    # binary MLPs have one output, the class-1 logit; class 0's evidence is its negative
    W1, W2 = mlp.coefs_
    b1 = mlp.intercepts_[0]
    mask = (x @ W1 + b1) > 0
    if W2.shape[1] == 1:
        g = W1 @ (mask * W2[:, 0])
        if c == 1:
            return g
        return -g
    return W1 @ (mask * W2[:, c])


def lin_w(model, c):
    """Class-c weight vector of a linear model; binary models have one row (class-1 evidence)."""
    W = np.asarray(model.coef_, dtype=np.float64)
    if W.shape[0] == 1:
        if c == 1:
            return W[0]
        return -W[0]
    return W[c]


def shap_phi(v, c):
    """Class-c SHAP attribution for one sample, from any shap output layout."""
    # layouts: per-class (1, d, K), a list of per-class arrays, or a single binary output (1, d)
    # giving class-1 evidence
    v = np.asarray(v)
    if v.ndim == 3:
        return v[0, :, c]
    if v.ndim == 2 and v.shape[0] == 1:
        if c == 1:
            return v[0]
        return -v[0]
    return np.asarray(v[c])[0]


def mlp_ig(mlp, x, base, c, steps=64):
    alphas = (np.arange(steps) + 0.5) / steps
    g = np.zeros_like(x)
    for a in alphas:
        g += mlp_grad(mlp, base + a * (x - base), c)
    return (x - base) * g / steps


def to_methods(phi, prov):
    """Common rule: split bucket score by token occupancy, sum per method."""
    tot = defaultdict(float)
    for b, s, _ in prov:
        tot[b] += abs(s)
    out = defaultdict(float)
    for b, s, m in prov:
        if tot[b] > 0:
            out[m] += phi[b] * abs(s) / tot[b]
    return out


def main():
    import shap
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    X = H.matrix("labeled_fcg", ids)
    g = H.groups_for(ids)
    pos = {}
    for k, i in enumerate(ids):
        pos[i] = k
    # non_proxy background: mean, and 100 sampled rows for interventional TreeSHAP
    ids5, fam5 = H.sample_ids("five")
    Xnp = H.matrix("labeled_fcg", ids5)[fam5 == H.BENIGN].astype(np.float64)
    mu = Xnp.mean(0)
    bg_rng = np.random.default_rng(0)
    bg_rows = bg_rng.choice(len(Xnp), 100, replace=False)
    bg = Xnp[bg_rows]

    staged = []
    for f in H.PROXY:
        for sha in H.staged(f):
            if sha in pos:
                staged.append((f, sha))
    fold_of = {}
    folds = H.folds(y, g, 1)
    for r, k, tr, te in folds:
        for j in te:
            fold_of[ids[j]] = k
    res = {"apps": [], "explainers": EXPLAINERS}
    agree = defaultdict(list)
    # resume after a memory abort: the per-fold partial dump keeps apps and agreement lists
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        if "agreement_top100_jaccard" not in prev:  # key written only by the final dump
            res["apps"] = prev.get("apps", [])
            for kk, vv in prev.get("agree_partial", {}).items():
                agree[kk] = list(vv)
    done_shas = set()
    for a in res["apps"]:
        done_shas.add(a["sha"])
    for r, k, tr, te in folds:
        fold_apps = []
        for f, s in staged:
            if fold_of[s] == k:
                fold_apps.append((f, s))
        if not fold_apps:
            continue
        all_done = True
        for _, s in fold_apps:
            if s not in done_shas:
                all_done = False
                break
        if all_done:
            print(f"fold {k}: resumed from partial results", flush=True)
            continue
        # resume within a fold after a memory abort
        todo = []
        for f, s in fold_apps:
            if s not in done_shas:
                todo.append((f, s))
        t0 = time.time()
        # the four models on the upsampled training fold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        up = H.upsample(tr, y, k)
        lin = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
        lin.fit(X[up], y[up])
        rf = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=42)
        rf.fit(X[up], y[up])
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=8,
                            random_state=42, verbosity=0)
        xgb.fit(X[up], y[up])
        mlp = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, early_stopping=True,
                            max_iter=300, random_state=42)
        mlp.fit(X[up], y[up])
        ex = {"rf_path": shap.TreeExplainer(rf), "xgb_path": shap.TreeExplainer(xgb),
              "rf_int": shap.TreeExplainer(rf, data=bg, feature_perturbation="interventional"),
              "xgb_int": shap.TreeExplainer(xgb, data=bg, feature_perturbation="interventional")}
        print(f"fold {k}: trained in {time.time()-t0:.0f}s, {len(todo)} apps", flush=True)
        for f, sha in todo:
            fcg = os.path.join(GRAPHS, f, sha, f"{sha}.fcg")
            try:
                nl, su, pr = G.parse_gml_lightweight(fcg)
                x, prov, nrm = P.provenance_vector(nl, su, pr)
            except Exception as e:
                res["apps"].append({"family": f, "sha": sha, "error": repr(e)[:120]})
                continue
            del nl, su, pr
            if x is None:
                continue
            c = int(le.transform([f])[0])
            cos = float(np.dot(x, X[pos[sha]]) / np.linalg.norm(X[pos[sha]]))
            w = lin_w(lin, c)
            phis = {"lin_int": w * (x - mu), "lin_zero": w * x}
            xr = x.reshape(1, -1)
            additivity_failed = []
            for name, e in ex.items():
                try:
                    v = np.asarray(e.shap_values(xr))
                except Exception as err:  # newer shap/xgboost: float32 rounding trips the additivity check
                    if "Additivity" not in str(err):
                        raise
                    v = np.asarray(e.shap_values(xr, check_additivity=False))
                    additivity_failed.append(name)
                phis[name] = shap_phi(v, c)
            phis["mlp_ig"] = mlp_ig(mlp, x, mu, c)
            phis["mlp_gxi"] = x * mlp_grad(mlp, x, c)
            credits = {}
            for n, p in phis.items():
                credits[n] = to_methods(p, prov)
            credits["lin_exact"] = P.split_bucket_credit(prov, w, mu, nrm, w * (x - mu), rule="exact")
            rec = {"family": f, "sha": sha, "cos_vs_packed": cos, "shap_additivity_failed": additivity_failed,
                   "pred": {"lin": int(lin.predict(xr)[0] == c), "rf": int(rf.predict(xr)[0] == c),
                            "xgb": int(xgb.predict(xr)[0] == c), "mlp": int(mlp.predict(xr)[0] == c)}}
            tops = {}
            for n in EXPLAINERS:
                # score this explainer's method credits against the family's SDK namespaces
                credit = credits[n]
                ms = list(credit)
                sc_list = []
                truth_list = []
                for m in ms:
                    sc_list.append(credit[m])
                    truth_list.append(m.startswith(P.SDK_PREFIXES.get(f, ())))
                sc = np.array(sc_list)
                truth = np.array(truth_list)
                n_t = int(truth.sum())
                if n_t == 0:
                    s_ = None
                else:
                    order = np.argsort(-sc)
                    s_ = {"n_methods": len(ms), "n_target": n_t, "base_rate": n_t / len(ms),
                          "p_at_target": float(truth[order[:n_t]].mean()),
                          "p_at_25": float(truth[order[:25]].mean()),
                          "ap": float(average_precision_score(truth, sc))}
                rec[n] = s_
                # top-100 methods by credit
                top_idx = np.argsort(-sc)[:100]
                top_methods = np.array(ms)[top_idx]
                tops[n] = set(top_methods.tolist())
            # pairwise top-100 Jaccard between explainers
            for a in range(len(EXPLAINERS)):
                for b in range(a + 1, len(EXPLAINERS)):
                    A = tops[EXPLAINERS[a]]
                    B = tops[EXPLAINERS[b]]
                    agree[f"{EXPLAINERS[a]}|{EXPLAINERS[b]}"].append(len(A & B) / len(A | B))
            res["apps"].append(rec)
            snap = dict(res)
            snap["agree_partial"] = agree
            json.dump(snap, open(OUT, "w"))  # per-app checkpoint
            shown = []
            for n in EXPLAINERS:
                if rec[n]:
                    shown.append(f"{n}={rec[n]['p_at_target']:.2f}")
            print(f"  {f} {sha[:10]} cos={cos:.4f} " + " ".join(shown), flush=True)
            del prov, credits
        snap = dict(res)
        snap["agree_partial"] = agree
        json.dump(snap, open(OUT, "w"))
    # per-family means of each explainer's metrics
    summary = {}
    for f in H.PROXY:
        rows = []
        for a in res["apps"]:
            if a.get("family") == f and "error" not in a:
                rows.append(a)
        summary[f] = {"n_apps": len(rows)}
        for n in EXPLAINERS:
            v = []
            for a in rows:
                if a.get(n):
                    v.append(a[n])
            if v:
                summary[f][n] = {}
                for k in ("p_at_target", "p_at_25", "ap", "base_rate"):
                    vals = []
                    for x in v:
                        vals.append(x[k])
                    summary[f][n][k] = float(np.mean(vals))
    res["summary"] = summary
    agreement = {}
    for k, v in agree.items():
        agreement[k] = float(np.mean(v))
    res["agreement_top100_jaccard"] = agreement
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
