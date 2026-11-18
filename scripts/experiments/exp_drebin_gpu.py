#!/usr/bin/env python3
"""Drebin features on the GPU: linear models, exact additive certificate, attack verification.

GPU counterpart of exp_cert_drebin.py's evaluate step (same folds, upsampling and output format),
for vocabularies of ~1e5 features where CPU training of a dense copy does not fit next to the other
jobs. Run with the ROCm environment: .venv-rocm/bin/python scripts/exp_drebin_gpu.py

  linear_sgd  multinomial logistic regression, L2 strength chosen by inner 3-fold grouped CV over
              the same grid as exp_l1_ladder.fit_linear (alpha 1e-5, 1e-4, 1e-3), full-batch L-BFGS
  linear_svm  one-vs-rest squared-hinge linear SVM (C = 1), full-batch L-BFGS
  rf          random forest (300 trees) on the sparse matrix, CPU; accuracy reference only

Certificate (Proposition certbin): for a correctly classified application x and rival c, the
smallest number of absent features whose addition flips the decision against c is the smallest k
with M_c + (sum of the k most negative entries of u_c = w_y - w_c over absent features) <= 0. It is
exact; we verify it by adding those features and re-scoring (must flip) and by adding one fewer
(must not flip against that rival).

Output: tier4/v4/cert_drebin.json (dataset-aware), keys drebin / drebin_code.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "v4", "cert_drebin.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_dev(X):
    """Dense float32 copy on the device."""
    # sparse input is sent as CSR and densified on the GPU, so the host never holds the dense matrix
    # (MalNet Drebin: 6,494 x 176,628 = 4.6 GB dense)
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
        T = torch.sparse_csr_tensor(torch.as_tensor(X.indptr, dtype=torch.int64), torch.as_tensor(X.indices, dtype=torch.int64),
                                    torch.as_tensor(X.data), size=X.shape, device=DEV)
        return T.to_dense()
    return torch.tensor(X, dtype=torch.float32, device=DEV)


def train(Xt, yt, K, alpha, loss="logistic", iters=200):
    n, d = Xt.shape
    W = torch.zeros(K, d, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    Y = torch.nn.functional.one_hot(yt, K).float() * 2 - 1

    def closure():
        opt.zero_grad()
        Z = Xt @ W.T + b
        if loss == "logistic":
            L = torch.nn.functional.cross_entropy(Z, yt) + alpha * (W * W).sum()
        else:  # one-vs-rest squared hinge, C = 1  <=>  alpha = 1 / (2 n)
            L = torch.clamp(1 - Y * Z, min=0).pow(2).sum() / n + (W * W).sum() / (2 * n)
        L.backward()
        return L

    opt.step(closure)
    return W.detach(), b.detach()


def macro_f1(y, p, K):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=list(range(K))))


def evaluate(rep):
    from sklearn.ensemble import RandomForestClassifier
    # Drebin rows of the evaluated apps
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
    rows_x = []
    for s in ids:
        rows_x.append(idx[s])
    X = XA[rows_x]
    y, le = H.labels(fam)
    K = len(le.classes_)
    g = H.groups_for(ids)
    XD = to_dev(X)
    F = H.folds(y, g, 1)
    res = {"dataset": H.DATASET, "groups": H.GROUPS, "n": int(len(ids)), "dim": int(X.shape[1]), "device": DEV}
    # random forest accuracy reference (CPU)
    f1 = []
    for r, k, tr, te in F:
        up = H.upsample(tr, y, k)
        rf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=42)
        rf.fit(X[up], y[up])
        f1.append(macro_f1(y[te], rf.predict(X[te]), K))
    res["rf"] = {"f1": float(np.mean(f1))}
    print(rep, "rf", res["rf"], flush=True)
    for name in ("linear_sgd", "linear_svm"):
        f1 = []
        rows = []
        for r, k, tr, te in F:
            if name == "linear_sgd":
                # L2 strength by inner 3-fold grouped CV, then refit on the whole training fold
                from sklearn.model_selection import StratifiedGroupKFold
                best = -1
                best_a = None
                inner = StratifiedGroupKFold(3, shuffle=True, random_state=k)
                for a in (1e-5, 1e-4, 1e-3):
                    sc = []
                    for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
                        itr = tr[itr]
                        ite = tr[ite]
                        up = H.upsample(itr, y, k)
                        W, b = train(XD[up], torch.tensor(y[up], device=DEV), K, a)
                        sc.append(macro_f1(y[ite], (XD[ite] @ W.T + b).argmax(1).cpu().numpy(), K))
                    if np.mean(sc) > best:
                        best = np.mean(sc)
                        best_a = a
                up = H.upsample(tr, y, k)
                W, b = train(XD[up], torch.tensor(y[up], device=DEV), K, best_a)
            else:
                up = H.upsample(tr, y, k)
                W, b = train(XD[up], torch.tensor(y[up], device=DEV), K, 0.0, loss="hinge")
            pred = (XD[te] @ W.T + b).argmax(1).cpu().numpy()
            f1.append(macro_f1(y[te], pred, K))
            for j, i in enumerate(te):
                if pred[j] != y[i]:
                    continue
                x = XD[i]
                yy = int(y[i])
                # exact minimal number of added features to flip x (a 0/1 row on the device) against any rival
                absent = x == 0
                kk = None
                plan = None
                for c in range(W.shape[0]):
                    if c == yy:
                        continue
                    u = W[yy] - W[c]
                    M = float(u @ x + b[yy] - b[c])
                    v = torch.where(absent & (u < 0), u, torch.zeros_like(u))
                    vals, order = torch.sort(v)
                    cs = M + torch.cumsum(vals, 0)
                    hit = torch.nonzero(cs <= 0)
                    if len(hit):
                        kc = int(hit[0]) + 1
                        if kk is None or kc < kk:
                            kk = kc
                            plan = (c, order[:kc])
                row = {"family": str(fam[i]), "k": kk}
                if kk is not None:
                    # verify: adding the planned features must flip; one fewer must not flip against that rival
                    c_plan, add = plan
                    xa = x.clone()
                    xa[add] = 1
                    flips = bool((xa @ W.T + b).argmax() != yy)
                    xb = x.clone()
                    xb[add[:-1]] = 1
                    s = xb @ W.T + b
                    row["attack_flips"] = flips
                    row["one_fewer_flips"] = bool(s[c_plan] >= s[yy])
                rows.append(row)
                del x  # x is a view of XD; drop it so del XD below still frees the device matrix
            print(f"  {rep} {name} fold {k}: f1={f1[-1]:.3f}", flush=True)
        # summary (keys in the same order as before)
        ks = []
        for r_ in rows:
            if r_["k"] is not None:
                ks.append(r_["k"])
        fl = []
        for r_ in rows:
            if "attack_flips" in r_:
                fl.append(r_)
        uncond = []
        for r_ in rows:
            uncond.append(r_["k"] is None)
        summ = {"f1": float(np.mean(f1)), "n_certified": len(rows), "unconditional": float(np.mean(uncond))}
        if ks:
            summ["k_median"] = float(np.median(ks))
            summ["k_q10_q90"] = [float(np.percentile(ks, 10)), float(np.percentile(ks, 90))]
        else:
            summ["k_median"] = None
            summ["k_q10_q90"] = None
        if fl:
            attack = []
            fewer = []
            for r_ in fl:
                attack.append(r_["attack_flips"])
                fewer.append(r_["one_fewer_flips"])
            summ["attack_flips_rate"] = float(np.mean(attack))
            summ["one_fewer_flips_rate"] = float(np.mean(fewer))
        else:
            summ["attack_flips_rate"] = None
            summ["one_fewer_flips_rate"] = None
        fam_set = set()
        for r_ in rows:
            fam_set.add(r_["family"])
        per_family = {}
        for f in sorted(fam_set):
            n_f = 0
            ks_f = []
            for r_ in rows:
                if r_["family"] == f:
                    n_f += 1
                    if r_["k"] is not None:
                        ks_f.append(r_["k"])
            if ks_f:
                k_med = float(np.median(ks_f))
            else:
                k_med = None
            per_family[f] = {"n": n_f, "k_median": k_med}
        summ["per_family"] = per_family
        res[name] = summ
        shown = {}
        for k_, v_ in res[name].items():
            if k_ != "per_family":
                shown[k_] = v_
        print(rep, name, shown, flush=True)
    del XD
    torch.cuda.empty_cache()
    return res


def main():
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    else:
        out = {}
    for rep in ("drebin", "drebin_code"):
        if os.path.exists(os.path.join(H.VEC, f"{rep}_X.npz")):
            out[rep] = evaluate(rep)
            json.dump(out, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
