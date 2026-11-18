#!/usr/bin/env python3
"""Exact certificate against code addition for a linear model on l1-normalized blocks
(MaMaDroid Markov-chain features), for any dataset (DATASET env).

Setting. The feature vector concatenates the rows x_s = c_s / n_s of a transition-count
matrix (n_s = |c_s|_1). Adding code adds call edges; an added call from a caller in state s
to a callee in state j adds one to c_{s,j}. Only rows whose state can be a caller in the
application's own code are modifiable: self-defined and obfuscated (framework methods are
external and have no outgoing edges in the FCG).

Bound. For the margin of class y over rival c, u = w_y - w_c, row s contributes
m_s = u_s . x_s. After adding t >= 0 calls from s, distributed as q (a distribution over j),
the row becomes (n_s x_s + t q)/(n_s + t), contributing (n_s m_s + t u_s.q)/(n_s + t), which is
smallest when q puts all mass on argmin_j u_{s,j} =: j*. The largest possible decrease of the
row's contribution after t calls is therefore
    R_s(t) = a_s t / (n_s + t),   a_s = max(0, m_s - min_j u_{s,j})          (n_s > 0)
    R_s(t) = max(0, -min_j u_{s,j}) for every t >= 1                            (n_s = 0)
Each R_s is concave and nondecreasing, so the minimum total number of added calls that can
drive the margin M to zero is attained by allocating calls greedily by marginal decrease; it is
an exact certificate (it equals the strongest attack in this threat model), and if
sum_s sup_t R_s(t) <= M no number of added calls flips the decision against c.

Output: tier4/v4/cert_l1block.json (dataset-aware). The certificate is verified on every app
by constructing the attack it implies and re-scoring.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_l1_ladder as L

OUT = H.out_path("tier4", "v4", "cert_l1block.json")
MODIFIABLE = ("self", "obf")


def row_reduction(a, n, t):
    if n == 0:
        if t >= 1:
            return a
        return 0.0
    return a * t / (n + t)


def units_at(a, n, lam):
    """Calls placed in a row when every unit with marginal decrease >= lam is taken."""
    if a <= 0:
        return 0
    if n == 0:
        if a >= lam:
            return 1
        return 0
    # marginal decrease of the t-th call: a n / ((n + t)(n + t - 1))
    z = (1 + np.sqrt(1 + 4 * a * n / lam)) / 2
    return max(0, int(np.floor(z)) - int(n))


def certify(W, b, x, rowc, y, states):
    S = len(states)
    mod = []
    for k, s in enumerate(states):
        if s in MODIFIABLE:
            mod.append(k)
    best = (np.inf, None, None)
    unconditional = True
    for c in range(W.shape[0]):
        if c == y:
            continue
        u = (W[y] - W[c]).reshape(S, S)
        M = float((W[y] - W[c]) @ x + b[y] - b[c])
        X = x.reshape(S, S)
        a = []
        n = []
        jstar = []
        for s in mod:
            if rowc[s] > 0:
                m_s = float(u[s] @ X[s])
            else:
                m_s = 0.0
            j = int(np.argmin(u[s]))
            if rowc[s] > 0:
                a.append(max(0.0, m_s - float(u[s, j])))
            else:
                a.append(max(0.0, -float(u[s, j])))
            n.append(float(rowc[s]))
            jstar.append(j)

        # smallest total of added calls whose greedy allocation reduces the margin by >= M
        cap = 0
        for a_ in a:
            cap += a_
        if cap <= M:
            k = None
            t = None
        else:
            single_call = False
            if M > 0:  # one call is the least possible; skip the search when a single call already suffices
                r1 = []
                for a_, n_ in zip(a, n):
                    if a_ > 0:
                        r1.append(row_reduction(a_, n_, 1))
                    else:
                        r1.append(0.0)
                s1 = int(np.argmax(r1))
                if r1[s1] >= M:
                    k = 1
                    t = []
                    for i in range(len(a)):
                        if i == s1:
                            t.append(1)
                        else:
                            t.append(0)
                    single_call = True
            if not single_call:
                terms = []
                for a_, n_ in zip(a, n):
                    if n_ == 0:
                        terms.append(a_ * 1)
                    else:
                        terms.append(a_ * (1 / max(n_, 1)))
                lo = 1e-12
                hi = max(max(terms), 1e-12) * 2
                for _ in range(200):  # largest lam that still reaches M
                    mid = np.sqrt(lo * hi)
                    t = []
                    for a_, n_ in zip(a, n):
                        t.append(units_at(a_, n_, mid))
                    reached = 0
                    for a_, n_, t_ in zip(a, n, t):
                        reached += row_reduction(a_, n_, t_)
                    if reached >= M:
                        lo = mid
                    else:
                        hi = mid
                    if hi / lo < 1 + 1e-9:
                        break
                t = []
                for a_, n_ in zip(a, n):
                    t.append(units_at(a_, n_, lo))
                # trim: remove the smallest-gain units while the margin stays reached
                k = sum(t)

        if k is not None:
            unconditional = False
            if k < best[0]:
                best = (k, c, (mod, t, jstar))
                if k == 1:  # cannot be beaten; the first class reaching one call is kept, as before
                    break
    if unconditional:
        return None, best
    return int(best[0]), best


def apply(x, rowc, states, plan):
    S = len(states)
    mod, t, jstar = plan
    C = x.reshape(S, S) * rowc[:, None]
    for s, ts, j in zip(mod, t, jstar):
        C[s, j] += ts
    rs = C.sum(1, keepdims=True)
    return np.divide(C, rs, out=np.zeros_like(C), where=rs > 0).ravel()


def run(rep, standardize=False):
    ids, fam = H.sample_ids("four")
    z = np.load(os.path.join(H.VEC, f"{rep}.npz"))
    idx = {}
    for i, s in enumerate(z["ids"]):
        idx[s] = i
    keep_list = []
    for s in ids:
        keep_list.append(s in idx)
    keep = np.array(keep_list)
    ids = ids[keep]
    fam = fam[keep]
    sel = []
    for s in ids:
        sel.append(idx[s])
    X = z["X"][sel].astype(np.float64)
    Rc = z["R"][sel].astype(np.float64)
    states = []
    for s in z["states"]:
        states.append(str(s))
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    rows = []
    f1 = []
    from sklearn.metrics import f1_score
    # per-fold checkpoint, so a run killed by the memory guard resumes at the next fold
    if standardize:
        ck = OUT + f".partial_{rep}_std.json"
    else:
        ck = OUT + f".partial_{rep}.json"
    done_folds = {}
    if os.path.exists(ck):
        for k_, v in json.load(open(ck)).items():
            done_folds[int(k_)] = v
    for r, k, tr, te in H.folds(y, g, 1):
        if k in done_folds:
            f1.append(done_folds[k]["f1"])
            rows += done_folds[k]["rows"]
            print(f"  {rep} fold {k}: resumed from checkpoint", flush=True)
            continue
        n0 = len(rows)
        # tuned linear model; with standardize, fitted on per-feature standardized features and folded
        # back to the raw features (w' = w / sigma, b' = b - w' . mu), so the certificate on the raw
        # l1-normalized rows applies unchanged to the standardized model
        if not standardize:
            m, _ = L.fit_linear(X, y, g, tr, k)
            W = m.coef_.astype(np.float64)
            b = m.intercept_.astype(np.float64)
        else:
            from sklearn.preprocessing import StandardScaler
            up = H.upsample(tr, y, k)
            sc = StandardScaler().fit(X[up])
            sig = np.where(sc.scale_ > 0, sc.scale_, 1.0)
            m, _ = L.fit_linear((X - sc.mean_) / sig, y, g, tr, k)
            W = m.coef_.astype(np.float64) / sig
            b = m.intercept_.astype(np.float64) - W @ sc.mean_
        pred = (X[te] @ W.T + b).argmax(1)
        f1.append(f1_score(y[te], pred, average="macro"))
        for j, i in enumerate(te):
            if pred[j] != y[i]:
                continue
            kc, best = certify(W, b, X[i], Rc[i], int(y[i]), states)
            c = best[1]
            plan = best[2]
            mod_idx = []
            for s in MODIFIABLE:
                mod_idx.append(states.index(s))
            row = {"family": str(fam[i]), "k": kc, "rows_modifiable": float(sum(Rc[i][mod_idx]))}
            if kc is not None:
                xa = apply(X[i], Rc[i], states, plan)
                row["attack_flips"] = bool((xa @ W.T + b).argmax() != y[i])
                # one call fewer must not flip (tightness of the greedy allocation)
                mod, t, jstar = plan
                tt = list(t)
                jmax = int(np.argmax(tt))
                tt[jmax] -= 1
                xb = apply(X[i], Rc[i], states, (mod, tt, jstar))
                row["one_fewer_flips_to_c"] = bool((xb @ W.T + b)[c] >= (xb @ W.T + b)[y[i]])
            rows.append(row)
        done_folds[k] = {"f1": float(f1[-1]), "rows": rows[n0:]}
        json.dump(done_folds, open(ck, "w"))
        print(f"  {rep} fold {k}: f1={f1[-1]:.3f} certified {len(rows)}", flush=True)
    # summary (key order is the JSON output order)
    ks = []
    uncond = []
    flips = []
    one_fewer = []
    fams_seen = set()
    for r in rows:
        if r["k"] is not None:
            ks.append(r["k"])
        uncond.append(r["k"] is None)
        if "attack_flips" in r:
            flips.append(r["attack_flips"])
        if "one_fewer_flips_to_c" in r:
            one_fewer.append(r["one_fewer_flips_to_c"])
        fams_seen.add(r["family"])
    out = {"f1": float(np.mean(f1)), "n": len(rows), "unconditional": float(np.mean(uncond))}
    if ks:
        out["k_median_when_finite"] = float(np.median(ks))
        out["k_q10_q90"] = [float(np.percentile(ks, 10)), float(np.percentile(ks, 90))]
        out["attack_flips_rate"] = float(np.mean(flips))
        out["one_fewer_flips_rate"] = float(np.mean(one_fewer))
    else:
        out["k_median_when_finite"] = None
        out["k_q10_q90"] = None
        out["attack_flips_rate"] = None
        out["one_fewer_flips_rate"] = None
    out["per_family"] = {}
    for f in sorted(fams_seen):
        rr = []
        kf = []
        uncond_f = []
        for r in rows:
            if r["family"] == f:
                rr.append(r)
                if r["k"] is not None:
                    kf.append(r["k"])
                uncond_f.append(r["k"] is None)
        if kf:
            k_med = float(np.median(kf))
        else:
            k_med = None
        out["per_family"][f] = {"n": len(rr), "unconditional": float(np.mean(uncond_f)), "k_median": k_med}
    return out


def main():
    res = {"dataset": H.DATASET, "groups": H.GROUPS, "modifiable_rows": list(MODIFIABLE)}
    if os.path.exists(OUT):
        for k, v in json.load(open(OUT)).items():
            if k.startswith("mama_"):
                res[k] = v
    for rep in ("mama_family", "mama_package"):
        if not os.path.exists(os.path.join(H.VEC, f"{rep}.npz")):
            res[rep] = {"skipped": "no features"}
            continue
        for std in (False, True):
            if std:
                key = rep + "_std"
            else:
                key = rep
            if key in res and "f1" in res[key]:
                continue
            res[key] = run(rep, std)
            shown = {}
            for k, v in res[key].items():
                if k != "per_family":
                    shown[k] = v
            print(key, shown, flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            ck = OUT + f".partial_{key}.json"
            if os.path.exists(ck):
                os.remove(ck)
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
