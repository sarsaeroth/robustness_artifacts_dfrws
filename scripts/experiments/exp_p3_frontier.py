#!/usr/bin/env python3
"""Accuracy-robustness frontier (robustness paper, sec:method_frontier / res_frontier).

Per classifier: clean macro-F1 (FCG-only, d=1,024, 4-class, 5-fold DEX-grouped on
the full proxy corpus) and, per staged proxy app held out of that fold, the
additive certificate and the strongest realizable aimed attack, in added tokens.

  k_cert   largest T satisfying Theorem certadd:
             u_c.phi - T||u_c||_inf > max(0, -beta_c)(||phi|| + T)  for all c != y
  k_steer  greedy aimed attack: each added token is placed in the (bucket, sign)
           that most reduces the nearest score gap. Realizability per colouring:
             labeled, fuzzy, api  every token aimable (method names, parameter
                                  lists and framework-call sets of an inserted stub
                                  are adversary-chosen and can be searched)
             structure            iteration-0 token restricted to the degree
                                  signatures of a stub in an attached tree
                                  (in-degree 1..3, out-degree 0..8); iteration-1
                                  and -2 tokens aimable by choice of tree shape
           Attacks add methods of h+1 = 3 tokens each; k_steer_methods counts them.
  k_rand   unaimed padding baseline (random bucket/sign), capped at MAX_T.

Classifiers: linear SGD on labeled / fuzzy / api / structure (and combined, F1
only: the staged graphs lack opcode maps so combined raw vectors cannot be
rebuilt); l_inf-bounded multinomial logistic regression on labeled at several
bounds; monotone per-family detectors (unsigned hashing, nonnegative weights) on
the staged apps only, since unsigned vectors need the graphs.

Output: results/tier4/p3/frontier.json (+ staged_raw.npz cache)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD

GRAPHS = H.GRAPHS_DIR
P3 = H.out_dir("tier4", "p3")
OUT = os.path.join(P3, "frontier.json")
CACHE = os.path.join(P3, "staged_raw.npz")
D = 1024
COLOURS = ["labeled", "fuzzy", "api", "structure"]
PACK = {"labeled": "fcg_wl_1024", "fuzzy": "fcg_wlfz_1024", "api": "fcg_wlapi_1024",
        "structure": "fcg_wls_1024", "combined": "fcg_wlall_1024"}
MAX_T = 20000


# vectorizing
def _tok(tok):
    h = G.stable_hash_u32(tok)
    if (h & 0x80000000) == 0:
        sign = 1.0
    else:
        sign = -1.0
    return h % D, sign


def raw_wl(colors, su, pr, unsigned=False):
    vec = np.zeros(D)

    def add(t):
        b, s = _tok(t)
        if unsigned:
            vec[b] += 1.0
        else:
            vec[b] += s

    for c in colors.values():
        add(f"0|{c}")
    cur = colors
    for i in (1, 2):
        new = {}
        for n in cur:
            inl = []
            for p in pr.get(n, []):
                if p in cur:
                    inl.append(cur[p])
            inl.sort()
            out = []
            for s in su.get(n, []):
                if s in cur:
                    out.append(cur[s])
            out.sort()
            new[n] = wl_digest(cur[n], inl, out)
        cur = new
        for c in cur.values():
            add(f"{i}|{c}")
    return vec


def vectorize_app(args):
    f, sha = args
    try:
        nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
    except Exception as e:
        return f, sha, None, repr(e)[:80]
    out = {}
    for c in COLOURS:
        # node colours for this colouring
        if c == "labeled":
            cols = {}
            for n, r in nl.items():
                cols[n] = G.canonical_label(r, "fcg")
            drop = []
            for n, l in cols.items():
                if l.startswith(G.LIB_PREFIXES):
                    drop.append(n)
            if drop and len(drop) < len(cols):
                for n in drop:
                    del cols[n]
        elif c == "structure":
            cols = {}
            for n in nl:
                cols[n] = f"d{len(pr.get(n, []))}_{len(su.get(n, []))}"
        else:
            lbl = {}
            for n, r in nl.items():
                lbl[n] = G.canonical_label(r, "fcg")
            cols = {}
            if c == "fuzzy":
                for n in lbl:
                    cols[n] = HD.fuzzy_label(lbl[n])
            else:
                for n in lbl:
                    cols[n] = HD.api_label(n, lbl, su)
        out[c] = raw_wl(cols, su, pr)
        if c == "labeled":
            out["labeled_unsigned"] = raw_wl(cols, su, pr, unsigned=True)
    op = H.load_opcodes(f, sha)  # combined coloring where the opcode map exists (bundle or dataset)
    if op:
        out["combined"] = raw_wl(HD.build_colors("combined", nl, su, "fcg", op), su, pr)
    return f, sha, out, None


def staged_raw(ids):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        cached = {}
        for k in z.files:
            if k != "apps":
                cached[k] = z[k]
        return list(z["apps"]), cached
    pos = set(ids)
    todo = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos:
                todo.append((f, s))
    apps = []
    mats = {}
    for c in COLOURS + ["labeled_unsigned"]:
        mats[c] = []
    comb = []
    with ProcessPoolExecutor(4) as ex:
        for f, sha, out, err in ex.map(vectorize_app, todo):
            if out is None:
                print("skip", sha[:10], err, flush=True)
                continue
            apps.append((f, sha))
            for c in mats:
                mats[c].append(out[c])
            comb.append(out.get("combined"))
            print(f"vectorized {len(apps)}/{len(todo)}", flush=True)
    for c in mats:
        mats[c] = np.array(mats[c])
    all_comb = True
    for v in comb:
        if v is None:
            all_comb = False
            break
    if comb and all_comb:
        mats["combined"] = np.array(comb)
    np.savez_compressed(CACHE, apps=np.array(apps, dtype=object), **mats)
    return apps, mats


# certificates
def k_cert(phi, W, b, y):
    nrm = np.linalg.norm(phi)
    best = np.inf
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = W[y] - W[c]
        beta = b[y] - b[c]
        nb = max(0.0, -beta)
        num = u @ phi - nb * nrm
        den = np.abs(u).max() + nb
        if den > 0:
            best = min(best, num / den)
        else:
            best = min(best, np.inf)
    return int(np.floor(max(0.0, best)))


def pred(W, b, phi):
    return int(np.argmax(W @ (phi / np.linalg.norm(phi)) + b))


def greedy_token(phi, W, b, y, allowed=None):
    """Best (bucket, sign) for one added token; allowed = list of (bucket, sign)."""
    nrm = np.linalg.norm(phi)
    g = (W[y] - W) @ phi + (b[y] - b) * nrm
    g[y] = np.inf
    c = int(np.argmin(g))
    grad = (W[y] - W[c]) + (b[y] - b[c]) * (phi / nrm)
    if allowed is None:
        j = int(np.argmax(np.abs(grad)))
        s = -np.sign(grad[j])
        if not s:  # zero gradient: default to a positive token
            s = 1.0
        return j, s
    vals = []
    for j, s in allowed:
        vals.append(s * grad[j])
    k = int(np.argmin(vals))
    return allowed[k]


def k_steer(phi, W, b, y, menu0=None, max_t=MAX_T):
    """Aimed attack in methods of 3 tokens; token 0 restricted to menu0 if given."""
    phi = phi.astype(float).copy()
    t = 0
    while t < max_t:
        for slot in range(3):
            if slot == 0 and menu0:
                allowed = menu0
            else:
                allowed = None
            j, s = greedy_token(phi, W, b, y, allowed)
            phi[j] += s
            t += 1
            if pred(W, b, phi) != y:
                return t, (t + 2) // 3
    return None, None


def k_random(phi, W, b, y, seed, max_t=MAX_T):
    rng = np.random.default_rng(seed)
    phi = phi.astype(float).copy()
    for t in range(1, max_t + 1):
        # draw the bucket before the sign (original evaluation order)
        j = int(rng.integers(D))
        if rng.random() < 0.5:
            phi[j] += 1.0
        else:
            phi[j] += -1.0
        if pred(W, b, phi) != y:
            return t
    return None


_struct_menu_set = set()
for _i in range(1, 4):
    for _o in range(0, 9):
        _struct_menu_set.add(_tok(f"0|d{_i}_{_o}"))
STRUCT_MENU = sorted(_struct_menu_set)
del _struct_menu_set, _i, _o


# classifiers
def fit_sgd(X, y, tr, seed):
    up = H.upsample(tr, y, seed)
    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
    m.fit(X[up], y[up])
    return m.coef_.astype(float), m.intercept_.astype(float)


def fit_linf(X, y, tr, seed, tau, lam=1e-4):
    """Multinomial logistic regression, L2 penalty lam, box constraint |W| <= tau."""
    up = H.upsample(tr, y, seed)
    Xu = X[up].astype(float)
    yu = y[up]
    n, d = Xu.shape
    C = int(y.max()) + 1
    Y = np.eye(C)[yu]

    def fg(theta):
        W = theta[:C * d].reshape(C, d)
        b = theta[C * d:]
        Z = Xu @ W.T + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -np.mean(np.log(P[np.arange(n), yu] + 1e-12)) + 0.5 * lam * (W ** 2).sum()
        G_ = (P - Y) / n
        gW = G_.T @ Xu + lam * W
        return loss, np.concatenate([gW.ravel(), G_.sum(0)])

    bounds = [(-tau, tau)] * (C * d) + [(None, None)] * C
    r = minimize(fg, np.zeros(C * d + C), jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 500})
    return r.x[:C * d].reshape(C, d), r.x[C * d:]


def f1_of(W, b, X, y, te):
    return float(f1_score(y[te], np.argmax(X[te] @ W.T + b, 1), average="macro"))


def certify(name, fitter, X, y, F, apps_by_fold, raw, le, menu0=None, rand=True):
    f1s = []
    rows = []
    for r, k, tr, te in F:
        W, b = fitter(X, y, tr, 1000 * r + k)
        f1s.append(f1_of(W, b, X, y, te))
        for idx, (fam, sha) in apps_by_fold.get(k, []):
            phi = raw[idx]
            yy = int(le.transform([fam])[0])
            if np.linalg.norm(phi) == 0 or pred(W, b, phi) != yy:
                continue
            kc = k_cert(phi, W, b, yy)
            ks_t, ks_m = k_steer(phi, W, b, yy, menu0)
            kr = []
            if rand:
                for s in range(2):
                    kr.append(k_random(phi, W, b, yy, s))
            kr_ok = []
            for v in kr:
                if v is not None:
                    kr_ok.append(v)
            if kr_ok:
                k_rand = float(np.median(kr_ok))
            else:
                k_rand = None
            rows.append({"family": fam, "sha": sha, "k_cert": kc, "k_steer": ks_t, "k_steer_methods": ks_m,
                         "k_rand": k_rand,
                         "rand_never": bool(rand and len(kr_ok) < len(kr)),
                         "norm": float(np.linalg.norm(phi))})
        print(f"  {name} fold {k}: f1={f1s[-1]:.3f} certified so far {len(rows)}", flush=True)

    # summary over certified apps (key order is the JSON output order)
    kc_list = []
    ks_list = []
    for x in rows:
        kc_list.append(x["k_cert"])
        if x["k_steer"]:
            ks_list.append(x["k_steer"])
    kc = np.array(kc_list)
    ks = np.array(ks_list)
    out = {"f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)), "n_cert": len(rows)}
    if len(kc):
        out["k_cert_median"] = float(np.median(kc))
        q_list = []
        for q in np.quantile(kc, [.25, .75]):
            q_list.append(float(q))
        out["k_cert_q"] = q_list
    else:
        out["k_cert_median"] = None
        out["k_cert_q"] = None
    if len(ks):
        out["k_steer_median"] = float(np.median(ks))
        methods = []
        ratios = []
        for x in rows:
            if x["k_steer_methods"]:
                methods.append(x["k_steer_methods"])
            if x["k_steer"]:
                ratios.append(x["k_steer"] / max(x["k_cert"], 1))
        out["k_steer_methods_median"] = float(np.median(methods))
        out["steer_over_cert_median"] = float(np.median(ratios))
    else:
        out["k_steer_median"] = None
        out["k_steer_methods_median"] = None
        out["steer_over_cert_median"] = None
    if rows:
        never = []
        for x in rows:
            never.append(x["k_steer"] is None)
        out["steer_never_frac"] = float(np.mean(never))
    else:
        out["steer_never_frac"] = None
    if rows and rand:
        rnever = []
        for x in rows:
            rnever.append(x["rand_never"])
        out["rand_never_frac"] = float(np.mean(rnever))
    else:
        out["rand_never_frac"] = None
    out["per_family"] = {}
    out["rows"] = rows

    # per-family medians
    for fam in H.PROXY:
        rr = []
        for x in rows:
            if x["family"] == fam:
                rr.append(x)
        if rr:
            kc_fam = []
            ks_fam = []
            for x in rr:
                kc_fam.append(x["k_cert"])
                if x["k_steer"]:
                    ks_fam.append(x["k_steer"])
            if ks_fam:
                ks_med = float(np.median(ks_fam))
            else:
                ks_med = None
            out["per_family"][fam] = {"n": len(rr), "k_cert_median": float(np.median(kc_fam)),
                                      "k_steer_median": ks_med}
    return out


def main():
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    g_of = {}
    for j in range(len(ids)):
        g_of[ids[j]] = g[j]
    apps, mats = staged_raw(ids)
    app_tuples = []
    for a in apps:
        app_tuples.append(tuple(a))
    apps = app_tuples
    apps_by_fold = {}
    for i, (fa, sha) in enumerate(apps):
        apps_by_fold.setdefault(fold_of[sha], []).append((i, (fa, sha)))

    # re-vectorization check against the packed (normalized) vectors
    chk = {}
    has_comb_pack = os.path.exists(os.path.join(H.VEC, PACK["combined"] + ".npz"))
    chk_colours = list(COLOURS)
    if "combined" in mats and has_comb_pack:
        chk_colours.append("combined")
    for c in chk_colours:
        idx, _, XP = H._load(PACK[c])
        cs = []
        for i, (fa, sha) in enumerate(apps):
            v = mats[c][i]
            if sha in idx and np.linalg.norm(v) > 0:
                p = XP[idx[sha]]
                cs.append(float(v @ p / (np.linalg.norm(v) * np.linalg.norm(p) + 1e-12)))
        chk[c] = {"cos_median": float(np.median(cs)), "cos_min": float(np.min(cs))}
    res["revectorization_cos"] = chk
    print("revectorization", chk, flush=True)

    # per coloring: matrix, and (labels, folds) restricted to apps with that vector
    Xs = {}
    YF = {}
    has_comb = os.path.exists(os.path.join(H.VEC, PACK["combined"] + ".npz"))
    pack_colours = list(COLOURS)
    if has_comb:
        pack_colours.append("combined")
    for c in pack_colours:
        idx, _, XP = H._load(PACK[c])
        keep_list = []
        for i in ids:
            keep_list.append(i in idx)
        keep = np.array(keep_list)
        if not keep.all():  # same folds, minus the apps without this vector (e.g. failed CFG extraction)
            print(f"  {c}: {int((~keep).sum())} of {len(ids)} apps lack a {c} vector; dropped for this coloring",
                  flush=True)
        new = np.cumsum(keep) - 1
        rows_idx = []
        for i in ids[keep]:
            rows_idx.append(idx[i])
        Xs[c] = XP[rows_idx].astype(np.float64)
        Fk = []
        for r, k, tr, te in F:
            Fk.append((r, k, new[tr[keep[tr]]], new[te[keep[te]]]))
        YF[c] = (y[keep], Fk)

    # linear SGD per coloring
    for c in COLOURS:
        key = f"sgd_{c}"
        if key in res:
            continue
        menu0 = None
        if c == "structure":
            menu0 = STRUCT_MENU
        yc, Fc = YF[c]
        res[key] = certify(key, fit_sgd, Xs[c], yc, Fc, apps_by_fold, mats[c], le, menu0=menu0)
        json.dump(res, open(OUT, "w"), indent=1)
    if has_comb and "combined" in mats and ("sgd_combined" not in res or "note" in res["sgd_combined"]):
        yc, Fc = YF["combined"]
        res["sgd_combined"] = certify("sgd_combined", fit_sgd, Xs["combined"], yc, Fc, apps_by_fold, mats["combined"], le)
        json.dump(res, open(OUT, "w"), indent=1)
    elif "sgd_combined" not in res and has_comb:
        yc, Fc = YF["combined"]
        f1s = []
        for r, k, tr, te in Fc:
            W, b = fit_sgd(Xs["combined"], yc, tr, 1000 * r + k)
            f1s.append(f1_of(W, b, Xs["combined"], yc, te))
        res["sgd_combined"] = {"f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                               "note": "no certificate: staged graphs lack opcode maps"}
        json.dump(res, open(OUT, "w"), indent=1)

    # l_inf-bounded: tau as fraction of the unconstrained solution's max |w|
    if "linf" not in res:
        yl, Fl = YF["labeled"]
        W0, _ = fit_linf(Xs["labeled"], yl, Fl[0][2], 0, tau=1e6)
        wmax = float(np.abs(W0).max())
        res["linf"] = {"wmax_unconstrained": wmax}
        for q in (1.0, 0.5, 0.25, 0.1, 0.05):
            tau = q * wmax
            res["linf"][str(q)] = certify(f"linf_{q}", lambda X, yy, tr, s, t=tau: fit_linf(X, yy, tr, s, t),
                                          Xs["labeled"], yl, Fl, apps_by_fold, mats["labeled"], le, rand=False)
            json.dump(res, open(OUT, "w"), indent=1)

    # monotone detectors: per-family one-vs-rest on the staged apps, log1p(unsigned counts),
    # logistic regression with and without w >= 0, 5-fold DEX-grouped
    if "monotone" not in res:
        from sklearn.model_selection import StratifiedGroupKFold
        fams_list = []
        groups_list = []
        for a in apps:
            fams_list.append(a[0])
            groups_list.append(g_of[a[1]])
        fams = np.array(fams_list)
        X = np.log1p(mats["labeled_unsigned"])
        groups = np.array(groups_list)
        mono = {}
        for mfam in H.PROXY:
            yb = (fams == mfam).astype(int)
            s_mon = np.zeros(len(yb))
            s_free = np.zeros(len(yb))
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
            for tr, te in cv.split(X, yb, groups):
                for nonneg, store in ((True, s_mon), (False, s_free)):
                    n, d = len(tr), X.shape[1]
                    Xt = X[tr]
                    yt = yb[tr]
                    wpos = len(yt) / (2 * max(yt.sum(), 1))
                    wneg = len(yt) / (2 * max((1 - yt).sum(), 1))
                    sw = np.where(yt == 1, wpos, wneg)

                    def fg(th):
                        w, b0 = th[:d], th[d]
                        z = Xt @ w + b0
                        p = 1 / (1 + np.exp(-z))
                        loss = -np.mean(sw * (yt * np.log(p + 1e-12) + (1 - yt) * np.log(1 - p + 1e-12))) + 5e-4 * (w ** 2).sum()
                        gz = sw * (p - yt) / n
                        return loss, np.concatenate([Xt.T @ gz + 1e-3 * w, [gz.sum()]])

                    if nonneg:
                        bnds = [(0, None)] * d + [(None, None)]
                    else:
                        bnds = [(None, None)] * d + [(None, None)]
                    r = minimize(fg, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bnds,
                                 options={"maxiter": 500})
                    store[te] = X[te] @ r.x[:d] + r.x[d]
            mono[mfam] = {"n_pos": int(yb.sum()),
                          "auc_monotone": float(roc_auc_score(yb, s_mon)),
                          "auc_unconstrained": float(roc_auc_score(yb, s_free)),
                          "f1_monotone": float(f1_score(yb, s_mon > 0)),
                          "f1_unconstrained": float(f1_score(yb, s_free > 0))}
            print("monotone", mfam, mono[mfam], flush=True)
        res["monotone"] = mono
        json.dump(res, open(OUT, "w"), indent=1)

    # summary without per-family blocks
    summary = {}
    for k, v in res.items():
        if not isinstance(v, dict):
            summary[k] = v
        else:
            sub = {}
            for kk, vv in v.items():
                if kk != "per_family":
                    sub[kk] = vv
            summary[k] = sub
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
