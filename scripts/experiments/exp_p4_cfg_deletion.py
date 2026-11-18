#!/usr/bin/env python3
"""Deletion faithfulness for CFG-only and for the deployed CFG+FCG classifier
(robustness paper, CFG extension; the FCG-only version is exp_p3_explainer_mx).

For each IPNinja and Luminati staged app: explain the model's decision with the
exact decomposition (Proposition 1 summed over the blocks), linear w.x and
TreeSHAP on RF and XGB (common occupancy rule), and Integrated Gradients and
gradient x input on an MLP; delete the top-ranked 1/5/20% of methods (each
method's FCG node and all of its CFG blocks) or a random set of the same size,
re-extract BOTH graphs with the real vectorizer, and record whether the
explained model's decision changes. Attributions use the token cache of
exp_p4_cfg_tokens.py (identifier coloring, d = 1,024 per graph).

Output: results/tier4/p4/cfg_deletion.json
"""
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_x2_method_level as _X2  # binary-safe class weights / SHAP output (lin_w, shap_phi)
import exp_harness as H
import exp_x2_method_level as X2
import gen_wl_vectors as G

GRAPHS = H.GRAPHS_DIR
P4 = H.out_dir("tier4", "p4")
OUT = os.path.join(P4, "cfg_deletion.json")
# stable-identifier families
if H.DATASET == "proxy":
    FAMS = ["ipninja", "luminati"]
else:
    FAMS = list(H.PROXY)
EXPL = ["lin_exact", "lin_zero", "rf_path", "xgb_path", "mlp_ig", "mlp_gxi"]
KFRAC = [0.01, 0.05, 0.20]
D = 1024
SETTINGS = ("concat", "cfg")
_M = {}


def _init(m):
    _M.update(m)


def raw(h):
    v = np.zeros(D)
    # signed feature hashing: bucket h % D, sign from the top bit
    buckets = (h % D).astype(np.int64)
    signs = np.where(h & 0x80000000, -1.0, 1.0)
    np.add.at(v, buckets, signs)
    return v


def margin(model, x, c):
    xr = x.reshape(1, -1)
    if model.__class__.__name__ == "SGDClassifier":
        s = model.decision_function(xr)[0]
    else:
        s = np.log(np.clip(model.predict_proba(xr)[0], 1e-9, 1))
    others = np.delete(s, c)
    return float(s[c] - others.max())


def vectorize(nlc, suc, prc, nlf, suf, prf, drop, setting):
    def sub(nl, su, pr, kind):
        # induced subgraph without the dropped methods
        keep = set()
        for n, r in nl.items():
            if G.canonical_label(r, kind) not in drop:
                keep.add(n)
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
        return nl2, su2, pr2
    cfg_nl, cfg_su, cfg_pr = sub(nlc, suc, prc, "cfg")
    vc = G.wl_hashed_vector(cfg_nl, cfg_su, cfg_pr, "cfg", D, iters=2, drop_lib=True, use_direction=True)
    if setting == "cfg":
        return vc.astype(float)
    fcg_nl, fcg_su, fcg_pr = sub(nlf, suf, prf, "fcg")
    vf = G.wl_hashed_vector(fcg_nl, fcg_su, fcg_pr, "fcg", D, iters=2, drop_lib=True, use_direction=True)
    return np.concatenate([vc, vf]).astype(float)


def credits(models, blocks, x, nmeth, c, mu):
    """Per-method credit for each explainer, from the token cache."""
    w = _X2.lin_w(models["lin"], c)
    out = {}
    ex = np.zeros(nmeth)
    for k, (h, m) in enumerate(blocks):
        nrm = np.linalg.norm(raw(h))
        keep = m >= 0
        sg = np.where(h & 0x80000000, -1.0, 1.0)
        w_block = w[k * D:(k + 1) * D]
        kept_buckets = (h[keep] % D).astype(np.int64)
        np.add.at(ex, m[keep], w_block[kept_buckets] * sg[keep] / nrm)
    out["lin_exact"] = ex
    import shap
    import shap_compat  # noqa: F401  additivity-check fallback for xgboost 3.x
    phis = {"lin_zero": w * x}
    for nm, key in (("rf_path", "rf"), ("xgb_path", "xgb")):
        v = np.asarray(shap.TreeExplainer(models[key]).shap_values(x.reshape(1, -1)))
        phis[nm] = _X2.shap_phi(v, c)
    phis["mlp_ig"] = X2.mlp_ig(models["mlp"], x, mu, c)
    grad = X2.mlp_grad(models["mlp"], x, c)
    phis["mlp_gxi"] = x * grad
    for nm, ph in phis.items():
        cr = np.zeros(nmeth)
        for k, (h, m) in enumerate(blocks):
            idx = (h % D).astype(np.int64)
            tot = np.bincount(idx, minlength=D).astype(float)
            keep = m >= 0
            # each token's share: bucket value split evenly over the tokens hashed to the bucket
            share = ph[k * D:(k + 1) * D][idx] / np.maximum(tot[idx], 1)
            np.add.at(cr, m[keep], share[keep])
        out[nm] = cr
    return out


def run_app(args):
    setting, fam, sha, c = args
    models, mu = _M[setting]
    z = np.load(os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{sha}.npz"))
    methods = list(z["methods"])
    truth = z["methods_truth"]
    if setting == "cfg":
        gs = ["cfg"]
    else:
        gs = ["cfg", "fcg"]
    blocks = []
    for g in gs:
        blocks.append((z[f"{g}|identifier|clean|h"], z[f"{g}|identifier|clean|m"]))
    # L2-normalized raw vector per graph, concatenated
    normed = []
    for h, _ in blocks:
        normed.append(raw(h) / np.linalg.norm(raw(h)))
    x = np.concatenate(normed)
    t0 = time.time()
    cr = credits(models, blocks, x, len(methods), c, mu)
    cost = (time.time() - t0) / len(EXPL)
    d = os.path.join(GRAPHS, fam, sha)
    nlc, suc, prc = G.parse_gml_lightweight(os.path.join(d, f"{sha}.cfg"))
    if setting == "concat":
        nlf, suf, prf = G.parse_gml_lightweight(os.path.join(d, f"{sha}.fcg"))
    else:
        nlf, suf, prf = ({}, {}, {})
    mdl = {"lin_exact": models["lin"], "lin_zero": models["lin"], "rf_path": models["rf"],
           "xgb_path": models["xgb"], "mlp_ig": models["mlp"], "mlp_gxi": models["mlp"]}
    m0 = {}
    for e in EXPL:
        m0[e] = margin(mdl[e], x, c)
    rng = np.random.default_rng(int(sha[:8], 16))
    m3 = {}
    for e in EXPL:
        m3[e] = {}
    rec = {"family": fam, "sha": sha, "n_methods": len(methods), "n_target": int(truth.sum()),
           "explain_seconds": cost, "M3": m3}
    for kf in KFRAC:
        kk = max(1, int(kf * len(methods)))
        rand_pick = rng.choice(len(methods), kk, replace=False)
        rand = set(np.array(methods)[rand_pick].tolist())
        xr = vectorize(nlc, suc, prc, nlf, suf, prf, rand, setting)
        for e in EXPL:
            top_pick = np.argsort(-cr[e])[:kk]
            top = set(np.array(methods)[top_pick].tolist())
            xd = vectorize(nlc, suc, prc, nlf, suf, prf, top, setting)
            md = margin(mdl[e], xd, c)
            mr = margin(mdl[e], xr, c)
            if m0[e]:
                retention = md / m0[e]
                retention_random = mr / m0[e]
            else:
                retention = None
                retention_random = None
            rec["M3"][e][str(kf)] = {"flip": bool(md < 0), "flip_random": bool(mr < 0),
                                    "retention": retention,
                                    "retention_random": retention_random}
    return rec


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("four")
    ids, fam = H.with_cfg_fcg(ids, fam)
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    # app id -> test fold
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    ids5, fam5 = H.sample_ids("five")
    # staged apps that are in a fold and have a token cache
    staged = []
    for f in FAMS:
        for s in H.staged(f):
            if s in fold_of and os.path.exists(os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{s}.npz")):
                staged.append((f, s))
    for setting in SETTINGS:
        if setting in res and "summary" in res[setting]:
            continue
        def mat(ii, which):
            parts = []
            if setting == "cfg":
                kinds = ["cfg"]
            else:
                kinds = ["cfg", "fcg"]
            for kind in kinds:
                idx, _, XP = H._load(f"{kind}_wl_1024")
                rows_of = []
                for i in ii:
                    rows_of.append(idx[i])
                parts.append(XP[rows_of])
            return np.hstack(parts).astype(float)
        X = mat(ids, None)
        mu = mat(ids5[fam5 == H.BENIGN], None).mean(0)
        rows = []
        if setting in res and "apps" in res[setting]:
            rows = res[setting]["apps"]
        done = set()
        for r in rows:
            done.add(r["sha"])
        for r, k, tr, te in F:
            todo = []
            for f, s in staged:
                if fold_of[s] == k and s not in done:
                    todo.append((f, s))
            if not todo:
                continue
            up = H.upsample(tr, y, 0)
            models = {"lin": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42).fit(X[up], y[up]),
                      "rf": RandomForestClassifier(n_estimators=200, n_jobs=6, random_state=42).fit(X[up], y[up]),
                      "xgb": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=6, random_state=42, verbosity=0).fit(X[up], y[up]),
                      "mlp": MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, early_stopping=True, max_iter=300, random_state=42).fit(X[up], y[up])}
            _M[setting] = (models, mu)
            tasks = []
            for f, s in todo:
                tasks.append((setting, f, s, int(le.transform([f])[0])))
            # spawn: the fitted models (XGBoost probes the CUDA driver when loaded) need workers that did
            # not inherit a CUDA-initialized process image
            with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn"), initializer=_init,
                                     initargs=({setting: _M[setting]},)) as ex:
                for rec in ex.map(run_app, tasks):
                    rows.append(rec)
                    flips = {}
                    for e in EXPL:
                        flips[e] = rec["M3"][e]["0.05"]["flip"]
                    print(setting, rec["family"], rec["sha"][:10], "flips@5%", flips, flush=True)
            res[setting] = {"apps": rows}
            json.dump(res, open(OUT, "w"))
        summ = {}
        for e in EXPL:
            s = {}
            for kf in KFRAC:
                v = []
                for a in rows:
                    v.append(a["M3"][e][str(kf)])
                flip = []
                flip_random = []
                rr = []
                for x in v:
                    flip.append(x["flip"])
                for x in v:
                    flip_random.append(x["flip_random"])
                for x in v:
                    if x["retention"] is not None:
                        rr.append(x["retention"])
                s[f"flip_{kf}"] = float(np.mean(flip))
                s[f"flip_random_{kf}"] = float(np.mean(flip_random))
                if rr:
                    s[f"retention_{kf}"] = float(np.median(rr))
                else:
                    s[f"retention_{kf}"] = None
            seconds = []
            for a in rows:
                seconds.append(a["explain_seconds"])
            s["explain_seconds"] = float(np.median(seconds))
            summ[e] = s
        res[setting] = {"apps": rows, "summary": summ}
        json.dump(res, open(OUT, "w"))
        print(setting, json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
