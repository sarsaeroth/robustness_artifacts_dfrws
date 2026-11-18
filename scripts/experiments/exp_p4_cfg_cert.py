#!/usr/bin/env python3
"""Certificates and aimed attacks for CFG-only and for the deployed CFG+FCG
classifier (robustness paper, CFG extension).

Block-normalized certificate (Theorem block). The deployed classifier sees
x = (phi_1/||phi_1||, ..., phi_K/||phi_K||). For block k with a = ||phi_k||,
h = u_k.phi_k / a and G = u_k - h phi_k/a, any addition with ||Delta||_1 = t < a
satisfies
    h(phi+Delta) >= h - [t ||G||_inf + max(0,h) t^2 / (2a)] / (a - t).
The prediction is certified against T added tokens if
    beta_c + sum_k L_k(t_k) > 0  for every split t_1 + ... + t_K <= T,  all c != y.
CFG-only models use Theorem certgrad (single block, raw space) as in the FCG
results; the block theorem is also evaluated on FCG-only for comparison.

Attack: token-level greedy on the actual normalized prediction; each step adds
one token to the block and bucket with the largest |G_k,j| / a_k for the nearest
wrong class (the direction of steepest decrease of the score gap).

Colorings: identifier, combined, fuzzy, api, structure. Raw vectors come from
the token caches (CFG: p4/cfg_tokens; FCG: p3/colour_tokens, FCG combined from
p4/cfg_tokens), all at d = 1,024 per graph; models are linear SGD trained on the
packed corpus vectors of the fold that holds each staged app out.

Output: results/tier4/p4/cfg_cert.json
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR
import exp_p3_frontier_post as FP
import exp_p3_cert_soundness as CS

P3 = H.out_dir("tier4", "p3")
P4 = H.out_dir("tier4", "p4")
OUT = os.path.join(P4, "cfg_cert.json")
D = 1024
COL = ["identifier", "combined", "fuzzy", "api", "structure"]
PACK = {"cfg": {"identifier": "cfg_wl_1024", "combined": "cfg_wlall_1024", "fuzzy": "cfg_wlfz_1024",
                "api": "cfg_wlapi_1024", "structure": "cfg_wls_1024"},
        "fcg": {"identifier": "fcg_wl_1024", "combined": "fcg_wlall_1024", "fuzzy": "fcg_wlfz_1024",
                "api": "fcg_wlapi_1024", "structure": "fcg_wls_1024"}}
FCG_CACHE_KEY = {"identifier": "labeled", "fuzzy": "fuzzy", "api": "api", "structure": "structure"}
MAX_T = 20000


def raw(h):
    v = np.zeros(D)
    # bucket = h mod D, sign from the top bit
    buckets = (h % D).astype(np.int64)
    signs = np.where(h & 0x80000000, -1.0, 1.0)
    np.add.at(v, buckets, signs)
    return v


def raw_vectors(sha, colour):
    p = os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{sha}.npz")
    if not os.path.exists(p) and H.V3:
        # token caches depend only on the app, so views of corpus v3 (proxy_v3_four, ...) share proxy_v3's
        p = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "tier4", "p4", "cfg_tokens", f"{sha}.npz")
    if not os.path.exists(p):
        return None, None  # no token cache for this app; callers skip apps with a missing block
    z = np.load(p)
    kc = f"cfg|{colour}|clean|h"
    kf = f"fcg|{colour}|clean|h"
    if kc in z.files:
        cfg = raw(z[kc])
    else:
        cfg = None
    if kf in z.files:
        fcg = raw(z[kf])
    else:
        fcg = None
    return cfg, fcg


# block theorem
def block_L(phi, u, t):
    a = float(np.linalg.norm(phi))
    if t >= a:
        return -np.inf
    h = float(u @ phi) / a
    G = u - h * phi / a
    return h - (t * float(np.abs(G).max()) + max(0.0, h) * t * t / (2 * a)) / (a - t)


def block_consts(phi, u):
    a = float(np.linalg.norm(phi))
    n = float(u @ phi)
    h = n / a
    ph = phi / a
    G = u - h * ph                       # orthogonal to ph
    return a, n, h, G, ph


def block_L_vec(consts, t):
    """Vertex-form lower bound on h(phi + Delta) over ||Delta||_1 = t (Theorem block).

    With x = G.Delta and y = ph.Delta, h(phi+Delta) - h = (a x - n e) / (a a') where
    a' >= a + y and 0 <= e <= t^2/(2a). The reachable (x, y) form t * conv{+-(G_j, ph_j)},
    and the bound min(0, a x - max(0,n) t^2/(2a)) / (a (a + y)) is linear-fractional in
    (x, y), so its minimum is attained at a vertex +-t (G_j, ph_j)."""
    a, n, h, G, ph = consts
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.full(t.shape, -np.inf)
    ok = t < a
    if not ok.any():
        return out
    tt = t[ok][:, None]
    K = max(0.0, n) * tt[:, 0] ** 2 / (2 * a)
    best = np.zeros(tt.shape[0])
    for sg in (1.0, -1.0):
        den = a * (a + sg * tt * ph[None, :])
        val = (a * sg * tt * G[None, :] - K[:, None]) / den
        best = np.minimum(best, val.min(1))
    out[ok] = h + best
    return out


def block_cert(phis, W, b, y, t_max=MAX_T):
    """Largest T such that every split of T tokens over the blocks keeps all score
    gaps positive. phis: list of raw block vectors; W, b: model on the
    concatenation of normalized blocks."""
    dims = []
    for p in phis:
        dims.append(len(p))
    offs = np.cumsum([0] + dims)
    # one (beta, per-block constants) entry per wrong class
    cons = []
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = W[y] - W[c]
        block_constants = []
        for k in range(len(phis)):
            block_constants.append(block_consts(phis[k], u[offs[k]:offs[k + 1]]))
        cons.append((b[y] - b[c], block_constants))

    def worst(T):
        best = np.inf
        for beta, bc in cons:
            if len(bc) == 1:
                val = beta + float(block_L_vec(bc[0], [T])[0])
            else:
                t = np.arange(T + 1)
                val = beta + float(np.min(block_L_vec(bc[0], t) + block_L_vec(bc[1], T - t)))
            best = min(best, val)
        return best

    if worst(0) <= 0:
        return 0
    # doubling, then bisection on the largest certified T
    lo = 0
    hi = 1
    while hi < t_max and worst(hi) > 0:
        lo = hi
        hi = hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if worst(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def pred_blocks(phis, W, b):
    normalized = []
    for p in phis:
        normalized.append(p / np.linalg.norm(p))
    x = np.concatenate(normalized)
    return int(np.argmax(W @ x + b))


def attack_blocks(phis, W, b, y, max_t=MAX_T):
    copies = []
    for p in phis:
        copies.append(p.astype(float).copy())
    phis = copies
    dims = []
    for p in phis:
        dims.append(len(p))
    offs = np.cumsum([0] + dims)
    for t in range(1, max_t + 1):
        normalized = []
        for p in phis:
            normalized.append(p / np.linalg.norm(p))
        x = np.concatenate(normalized)
        s = (W[y] - W) @ x + (b[y] - b)
        s[y] = np.inf
        c = int(np.argmin(s))
        u = W[y] - W[c]
        best = (0.0, 0, 0, 1.0)
        for k, p in enumerate(phis):
            a = np.linalg.norm(p)
            uk = u[offs[k]:offs[k + 1]]
            g = (uk - (uk @ p / a) * p / a) / a
            j = int(np.argmax(np.abs(g)))
            if abs(g[j]) > best[0]:
                sg = -np.sign(g[j])
                if not sg:
                    sg = 1.0
                best = (abs(g[j]), k, j, sg)
        _, k, j, sg = best
        phis[k][j] += sg
        if pred_blocks(phis, W, b) != y:
            return t
    return None


def fit(X, y, tr, seed):
    up = H.upsample(tr, y, seed)
    import gpu_models as GM
    if GM.ENABLED:  # same objective (log loss, alpha 1e-5) solved exactly on the GPU
        m = GM.TorchLinear(1e-5).fit(X[up], y[up])
        return m.coef_, m.intercept_
    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
    m.fit(X[up], y[up])
    return m.coef_.astype(float), m.intercept_.astype(float)


def main():
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("four")
    ids, fam = H.with_cfg_fcg(ids, fam)
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    pos = set(ids)
    # staged apps in the sample that have a CFG token cache
    staged = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos and os.path.exists(os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{s}.npz")):
                staged.append((f, s))
    for colour in COL:
        for setting in ("cfg", "concat", "fcg_block"):
            key = f"{setting}|{colour}"
            if key in res:
                continue
            Xc = H._load(PACK["cfg"][colour])
            Xf = H._load(PACK["fcg"][colour])
            cfg_rows = []
            for i in ids:
                cfg_rows.append(Xc[0][i])
            Xcfg = Xc[2][cfg_rows].astype(float)
            fcg_rows = []
            for i in ids:
                fcg_rows.append(Xf[0][i])
            Xfcg = Xf[2][fcg_rows].astype(float)
            if setting == "cfg":
                X = Xcfg
            elif setting == "concat":
                X = np.hstack([Xcfg, Xfcg])
            else:
                X = Xfcg
            rows = []
            f1s = []
            ck = OUT + f".partial_{key.replace('|', '_')}.json"  # per-fold checkpoint within a key
            done_f = {}
            if os.path.exists(ck):
                for a, v in json.load(open(ck)).items():
                    done_f[int(a)] = v
            for r, k, tr, te in F:
                if k in done_f:
                    f1s.append(done_f[k]["f1"])
                    rows += done_f[k]["rows"]
                    print(f"  {key} fold {k}: resumed from checkpoint", flush=True)
                    continue
                n0 = len(rows)
                W, b = fit(X, y, tr, 1000 * r + k)
                y_pred = np.argmax(X[te] @ W.T + b, 1)
                f1s.append(float(f1_score(y[te], y_pred, average="macro")))
                for fa, sha in staged:
                    if fold_of[sha] != k:
                        continue
                    pc, pf = raw_vectors(sha, colour)
                    if pc is None or (setting != "cfg" and pf is None):
                        continue
                    if setting == "cfg":
                        phis = [pc]
                    elif setting == "concat":
                        phis = [pc, pf]
                    else:
                        phis = [pf]
                    has_empty_block = False
                    for p in phis:
                        if np.linalg.norm(p) == 0:
                            has_empty_block = True
                            break
                    if has_empty_block:
                        continue
                    yy = int(le.transform([fa])[0])
                    if pred_blocks(phis, W, b) != yy:
                        continue
                    if setting == "cfg":
                        kc = FP.k_cert_grad(pc, W, b, yy)          # Theorem certgrad, single block
                        kb = block_cert([pc], W, b, yy)            # block theorem, for comparison
                        ka = CS.steer_tokens(pc, W, b, yy)
                    elif setting == "fcg_block":
                        kc = FP.k_cert_grad(pf, W, b, yy)
                        kb = block_cert([pf], W, b, yy)
                        ka = None
                    else:
                        kc = None
                        kb = block_cert([pc, pf], W, b, yy)
                        ka = attack_blocks([pc, pf], W, b, yy)
                    rows.append({"family": fa, "sha": sha, "k_certgrad": kc, "k_block": kb, "k_attack": ka})
                done_f[k] = {"f1": f1s[-1], "rows": rows[n0:]}
                json.dump(done_f, open(ck, "w"))
                print(f"  {key} fold {k}: f1={f1s[-1]:.3f} rows={len(rows)}", flush=True)
            def med(field):
                v = []
                for x in rows:
                    if x[field] is not None:
                        v.append(x[field])
                if v:
                    return float(np.median(v))
                return None
            if setting != "cfg":
                main_k = "k_block"
            else:
                main_k = "k_certgrad"
            # attacks that succeeded at or below the certified budget
            viol = 0
            for x in rows:
                if x["k_attack"] is not None and x[main_k] is not None and x["k_attack"] <= x[main_k]:
                    viol += 1
            f1_mean = float(np.mean(f1s))
            n_rows = len(rows)
            k_certgrad_median = med("k_certgrad")
            k_block_median = med("k_block")
            k_attack_median = med("k_attack")
            any_attack = False
            for x in rows:
                if x["k_attack"]:
                    any_attack = True
                    break
            if any_attack:
                ratios = []
                for x in rows:
                    if x["k_attack"] and x[main_k] is not None:
                        ratios.append(x["k_attack"] / max(x[main_k], 1))
                attack_over_cert = float(np.median(ratios))
            else:
                attack_over_cert = None
            res[key] = {"f1": f1_mean, "n": n_rows, "k_certgrad_median": k_certgrad_median,
                        "k_block_median": k_block_median, "k_attack_median": k_attack_median, "violations": viol,
                        "attack_over_cert_median": attack_over_cert,
                        "rows": rows}
            summary = {}
            for k2, v in res[key].items():
                if k2 != "rows":
                    summary[k2] = v
            print(key, summary, flush=True)
            json.dump(res, open(OUT, "w"))
            if os.path.exists(ck):
                os.remove(ck)


if __name__ == "__main__":
    main()
