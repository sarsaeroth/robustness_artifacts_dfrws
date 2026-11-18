#!/usr/bin/env python3
"""Localization on CFG features and on the deployed CFG+FCG classifier
(robustness paper, CFG extension). Everything runs from the token cache
written by exp_p4_cfg_tokens.py.

  exact      exact method-level decomposition (Proposition 1, summed over the
             blocks of the concatenated vector) for every coloring, clean and
             under per-application renaming, at d = 2^10, 2^16, 2^20 per graph;
             5-fold DEX-grouped linear SGD over the staged apps.
  explainers identifier coloring, clean: exact vs. linear w.x (common occupancy
             rule) vs. path-dependent TreeSHAP on RF and XGBoost, at the same d.
  granularity fraction of SDK methods whose round-0 CFG color also occurs on a
             host method (blocks inherit their method's color).
  monotone   per-family monotone detectors on unsigned CFG identifier counts.

Output: results/tier4/p4/cfg_localize.json
"""
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_x2_method_level as _X2  # binary-safe class weights / SHAP output (lin_w, shap_phi)
import exp_harness as H

P4 = H.out_dir("tier4", "p4")
TOK = os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens")
OUT = os.path.join(P4, "cfg_localize.json")
COLOURS = ["identifier", "fuzzy", "api", "structure", "combined"]
DIMS = [2 ** 10, 2 ** 16, 2 ** 20]
# stable-identifier families
if H.DATASET == "proxy":
    STABLE = ["ipninja", "luminati"]
else:
    STABLE = list(H.PROXY)


def vec(h, d):
    v = np.zeros(d)
    np.add.at(v, (h % d).astype(np.int64), np.where(h & 0x80000000, -1.0, 1.0))
    n = np.linalg.norm(v)
    if n > 0:
        v_out = v / n
    else:
        v_out = v
    if n:
        n_out = n
    else:
        n_out = 1.0
    return v_out, n_out


def blocks_for(z, graph, colour, cond):
    if graph == "cfg":
        gs = ["cfg"]
    else:
        gs = ["cfg", "fcg"]
    out = []
    for g in gs:
        k = f"{g}|{colour}|{cond}"
        if f"{k}|h" not in z.files:
            return None
        out.append((z[f"{k}|h"], z[f"{k}|m"]))
    return out


class LazyBlocks:
    """An app's token blocks, read from its npz on each access: every app's CFG tokens together are GBs."""
    def __init__(self, Z, graph, colour, cond):
        self.Z = Z
        self.args = (graph, colour, cond)

    def __getitem__(self, s):
        return blocks_for(self.Z[s], *self.args)

    def has(self, s):
        graph, colour, cond = self.args
        if graph == "cfg":
            gs = ["cfg"]
        else:
            gs = ["cfg", "fcg"]
        for g in gs:
            if f"{g}|{colour}|{cond}|h" not in self.Z[s].files:
                return False
        return True


def design(cache, shas, d):
    rows = []
    for s in shas:
        parts = []
        for h, _ in cache[s]:
            parts.append(vec(h, d)[0])
        rows.append(sparse.csr_matrix(np.concatenate(parts)))
    return sparse.vstack(rows).tocsr()


def exact_credit(blocks, w, d, nmeth):
    credit = np.zeros(nmeth)
    for k, (h, m) in enumerate(blocks):
        wk = w[k * d:(k + 1) * d]
        _, nrm = vec(h, d)
        keep = m >= 0
        sg = np.where(h & 0x80000000, -1.0, 1.0)
        np.add.at(credit, m[keep], wk[(h[keep] % d).astype(np.int64)] * sg[keep] / nrm)
    return credit


def rule_credit(blocks, phi, d, nmeth):
    """Common occupancy rule: bucket attribution split over its tokens."""
    credit = np.zeros(nmeth)
    for k, (h, m) in enumerate(blocks):
        pk = phi[k * d:(k + 1) * d]
        idx = (h % d).astype(np.int64)
        tot = np.bincount(idx, minlength=d).astype(float)
        keep = m >= 0
        np.add.at(credit, m[keep], (pk[idx] / np.maximum(tot[idx], 1))[keep])
    return credit


def precision(credit, truth, seed):
    nt = int(truth.sum())
    if nt == 0:
        return None
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(credit)), -credit))
    return float(truth[order[:nt]].mean()), nt / len(truth)


def main():
    import shap
    import shap_compat  # noqa: F401  additivity-check fallback for xgboost 3.x
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}

    # staged apps with a token cache
    ids, fam = H.sample_ids("four")
    ids, fam = H.with_cfg_fcg(ids, fam)
    pos = set(ids)
    staged = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos and os.path.exists(os.path.join(TOK, f"{s}.npz")):
                staged.append((f, s))
    shas = []
    fam_list = []
    for f, s in staged:
        shas.append(s)
        fam_list.append(f)
    fams = np.array(fam_list)
    le = LabelEncoder().fit(fams)
    y = le.transform(fams)
    g = H.groups_for(np.array(shas))
    Z = {}
    for s in shas:
        Z[s] = np.load(os.path.join(TOK, f"{s}.npz"))
    truth = {}
    for s in shas:
        truth[s] = Z[s]["methods_truth"]

    # granularity of CFG colorings
    if "granularity" not in res:
        gran = {}
        for c in COLOURS:
            per = defaultdict(list)
            for f, s in staged:
                k = f"cfg|{c}|clean"
                if f"{k}|h" not in Z[s].files:
                    continue
                h = Z[s][f"{k}|h"]
                m = Z[s][f"{k}|m"]
                n0 = len(h) // 3
                h0 = h[:n0]
                m0 = m[:n0]
                tr = truth[s]
                # round-0 colors on host methods (or on blocks of no method)
                host_method_colors = set(h0[(m0 >= 0) & ~tr[np.maximum(m0, 0)]].tolist())
                no_method_colors = set(h0[m0 < 0].tolist())
                host = host_method_colors | no_method_colors
                sdk_m = defaultdict(set)
                h_sel = h0[m0 >= 0]
                m_sel = m0[m0 >= 0]
                for idx in range(min(len(h_sel), len(m_sel))):
                    hh = h_sel[idx]
                    mm = m_sel[idx]
                    if tr[mm]:
                        sdk_m[mm].add(int(hh))
                if sdk_m:
                    shared = []
                    for v in sdk_m.values():
                        on_host = False
                        for x in v:
                            if x in host:
                                on_host = True
                                break
                        shared.append(on_host)
                    per[f].append(np.mean(shared))
            gran[c] = {}
            for f, v in per.items():
                gran[c][f] = float(np.mean(v))
        res["granularity"] = gran
        json.dump(res, open(OUT, "w"), indent=1)

    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g))
    for graph in ("cfg", "concat"):
        for c in COLOURS:
            for cond in ("clean", "renamed"):
                key = f"{graph}|{c}|{cond}"
                if key in res:  # resume: only complete keys are skipped
                    complete = True
                    for d in DIMS:
                        if str(d) not in res[key]:
                            complete = False
                            break
                    if complete:
                        continue
                cache = LazyBlocks(Z, graph, c, cond)
                ok = []
                for i, s in enumerate(shas):
                    if cache.has(s):
                        ok.append(i)
                if len(ok) < len(shas):
                    print(key, "missing", len(shas) - len(ok))
                res.setdefault(key, {})
                for d in DIMS:
                    if str(d) in res[key]:  # dimension already finished before an abort
                        continue
                    ok_shas = []
                    for i in ok:
                        ok_shas.append(shas[i])
                    X = design(cache, ok_shas, d)
                    yy = y[ok]
                    gg = g[ok]
                    prec = defaultdict(list)
                    accs = []
                    # tree-model SHAP explainers up to 2^16 dimensions: RF/XGB on 2^20 features exhausts memory here;
                    # the linear localization itself runs at every dimension
                    expl = (c == "identifier" and cond == "clean" and d <= 2 ** 16)
                    if expl:
                        eprec = {}
                        for e in ("lin_zero", "rf_path", "xgb_path"):
                            eprec[e] = defaultdict(list)
                    else:
                        eprec = None
                    for tr_, te_ in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yy)), yy, gg):
                        up = H.upsample(tr_, yy, 0)
                        m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-4, random_state=42).fit(X[up], yy[up])
                        accs.append(float((m.predict(X[te_]) == yy[te_]).mean()))
                        if expl:
                            rf = RandomForestClassifier(n_estimators=150, n_jobs=6, random_state=42).fit(X[up], yy[up])
                            xg = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, n_jobs=6,
                                               tree_method="hist", random_state=42, verbosity=0).fit(X[up], yy[up])
                            rfe = shap.TreeExplainer(rf)
                            xge = shap.TreeExplainer(xg)
                        # exact credit (and explainer credit) per held-out app
                        for j in te_:
                            s = shas[ok[j]]
                            f = fams[ok[j]]
                            tr = truth[s]
                            if tr.sum() == 0:
                                continue
                            cls = yy[j]
                            w = _X2.lin_w(m, cls)
                            cr = exact_credit(cache[s], w, d, len(tr))
                            p = precision(cr, tr, j)
                            prec[f].append(p)
                            if expl and f in STABLE:
                                dense = X[j].toarray().ravel()
                                phis = {"lin_zero": w * dense}
                                for nm, ex in (("rf_path", rfe), ("xgb_path", xge)):
                                    v = np.asarray(ex.shap_values(dense.reshape(1, -1)))
                                    phis[nm] = _X2.shap_phi(v, cls)
                                for nm, ph in phis.items():
                                    rule = rule_credit(cache[s], ph, d, len(tr))
                                    eprec[nm][f].append(precision(rule, tr, j)[0])

                    # per-family precision at |target|
                    row = {"accuracy": float(np.mean(accs))}
                    for f in H.PROXY:
                        if prec.get(f):
                            pa = np.array(prec[f])
                            row[f] = {"p_at_target": float(pa[:, 0].mean()), "base_rate": float(pa[:, 1].mean()), "n": len(pa)}
                    if expl:
                        explainers = {}
                        for nm, per_fam in eprec.items():
                            explainers[nm] = {}
                            for f, v in per_fam.items():
                                explainers[nm][f] = float(np.mean(v))
                        row["explainers"] = explainers
                    res[key][str(d)] = row
                    shown = {}
                    for k, v in row.items():
                        if not isinstance(v, dict) or k == "explainers":
                            shown[k] = v
                        else:
                            shown[k] = round(v.get("p_at_target", 0), 3)
                    print(key, d, json.dumps(shown), flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)

    # monotone per-family detectors on unsigned CFG identifier counts
    if "monotone_cfg_65536" not in res:
        from scipy.optimize import minimize
        from sklearn.metrics import roc_auc_score
        d = 2 ** 16
        count_rows = []
        for s in shas:
            count_rows.append(np.log1p(np.bincount((Z[s]["cfg|identifier|clean|h"] % d).astype(np.int64), minlength=d)))
        Xu = np.array(count_rows)
        out = {}
        for f in H.PROXY:
            yb = (fams == f).astype(int)
            sm = np.zeros(len(yb))
            sf = np.zeros(len(yb))
            for tr_, te_ in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xu, yb, g):
                mu = Xu[tr_].mean(0)
                sd = Xu[tr_].std(0) + 1e-6
                Xt = (Xu[tr_] - mu) / sd
                yt = yb[tr_]
                n, dd = Xt.shape
                sw = np.where(yt == 1, n / (2 * yt.sum()), n / (2 * (n - yt.sum())))
                # weighted L2 logistic regression, with and without non-negative weights
                for nonneg, store in ((True, sm), (False, sf)):
                    def fg(th):
                        w, b0 = th[:dd], th[dd]
                        zz = np.clip(Xt @ w + b0, -30, 30)
                        pp = 1 / (1 + np.exp(-zz))
                        loss = -np.mean(sw * (yt * np.log(pp + 1e-12) + (1 - yt) * np.log(1 - pp + 1e-12))) + 5e-3 * (w @ w)
                        gz = sw * (pp - yt) / n
                        return loss, np.concatenate([Xt.T @ gz + 1e-2 * w, [gz.sum()]])
                    if nonneg:
                        weight_bound = (0, None)
                    else:
                        weight_bound = (None, None)
                    th = minimize(fg, np.zeros(dd + 1), jac=True, method="L-BFGS-B",
                                  bounds=[weight_bound] * dd + [(None, None)],
                                  options={"maxiter": 1000}).x
                    store[te_] = ((Xu[te_] - mu) / sd) @ th[:dd] + th[dd]
            out[f] = {"auc_monotone": float(roc_auc_score(yb, sm)), "auc_unconstrained": float(roc_auc_score(yb, sf))}
            print("monotone cfg", f, out[f], flush=True)
        res["monotone_cfg_65536"] = out
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
