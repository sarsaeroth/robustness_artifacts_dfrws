#!/usr/bin/env python3
"""GNNExplainer over the numpy GIN, scored against SDK ground truth.

GNNExplainer (Ying et al.) learns a soft mask over edges that keeps the model's
prediction while being sparse: the surviving subgraph is offered as the
explanation. It is the most cited GNN explainer and the natural comparison for
our exact mean-pool decomposition, which is not a learned approximation at all.

Implemented directly rather than imported, for two reasons. The classifier is a
hand-written numpy GIN, so its gradients are already available and adding a deep
learning framework to obtain one gradient would be disproportionate. And a
hand-derived mask gradient can be checked against finite differences, which an
imported implementation cannot be.

Mask gradient. With A masked elementwise by sigmoid(m), layer l computes
Z_l = (A o sigmoid(m)) H_l + (1+eps) H_l, so for edge e = (i, j),

    dL/dm_e = sigmoid'(m_e) * sum_l < dZ_l[i], H_l[j] >

which the existing backward pass supplies once it returns dZ per layer.

Objective: maximise the log-softmax of the predicted class under the mask,
penalised by mask mass and mask entropy, which is GNNExplainer's formulation.

Node scores are obtained by summing the learned mask over each node's incident
edges, then scored by precision@|target| against the SDK namespace, the same
metric used for the exact decomposition.

Env: CORPUS_ROOT, N_GRAPHS (default 25), ITERS (default 60), MEM_GB (default 12)
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import exp_tier3_gnn as M

OUT = E.out_path("exp_xai_gnnexplainer.json")
CKPT = E.out_path("exp_tier3_gnn_params.npz")  # per dataset (exp_tier3_gnn.py)
N_GRAPHS = int(os.environ.get("N_GRAPHS", "25"))
ITERS = int(os.environ.get("ITERS", "60"))
LR = float(os.environ.get("LR", "0.10"))
L1 = float(os.environ.get("L1", "0.005"))
LENT = float(os.environ.get("LENT", "0.10"))


def masked_forward(p, feat, rows, cols, n, mask):
    """GIN forward with edge weights, caching what the mask gradient needs."""
    A = sp.csr_matrix((mask, (rows, cols)), shape=(n, n))
    H = p["emb"][feat]
    hs = [H]
    cache = []
    for l in range(M.LAYERS):
        Z = A.dot(H) + (1.0 + p[f"eps_{l}"][0]) * H
        U = Z @ p[f"W1_{l}"] + p[f"b1_{l}"]
        R = np.maximum(U, 0.0)
        Hn = R @ p[f"W2_{l}"] + p[f"b2_{l}"]
        cache.append((H, Z, U, R))
        H = Hn
        hs.append(H)
    pooled = []
    for h in hs:
        pooled.append(h.mean(axis=0))
    g_vec = np.concatenate(pooled)
    logits = g_vec @ p["Wo"] + p["bo"]
    return logits, g_vec, hs, cache, A


def mask_gradient(p, hs, cache, g_vec, dlogits, rows, cols, n, A):
    """dL/dA restricted to existing edges, as a vector over the edge list."""
    dg = p["Wo"] @ dlogits
    dh = []
    for l in range(M.LAYERS + 1):
        seg = dg[l * M.HID:(l + 1) * M.HID]
        dh.append(np.tile(seg / n, (n, 1)))
    grad_e = np.zeros(len(rows))
    for l in range(M.LAYERS - 1, -1, -1):
        Hin, Z, U, R = cache[l]
        dH_out = dh[l + 1]
        dR = dH_out @ p[f"W2_{l}"].T
        dU = dR * (U > 0)
        dZ = dU @ p[f"W1_{l}"].T
        # edge (i, j) carries H_l[j] into Z_l[i]
        grad_e += np.einsum("ek,ek->e", dZ[rows], Hin[cols])
        dh[l] += A.T.dot(dZ) + (1.0 + p[f"eps_{l}"][0]) * dZ
    return grad_e


def explain(p, g, cls, iters=ITERS):
    """Learn an edge mask; return per-node scores from incident mask mass."""
    A0 = g["A"].tocoo()
    rows = A0.row.astype(np.int64)
    cols = A0.col.astype(np.int64)
    n = g["n"]
    feat = g["feat"]
    m = np.full(len(rows), 0.5)          # mask logits, sigmoid(0.5) ~ 0.62
    for it in range(iters):
        w = 1.0 / (1.0 + np.exp(-m))
        logits, g_vec, hs, cache, A = masked_forward(p, feat, rows, cols, n, w)
        # d(log softmax at cls)/d(logits)
        z = logits - logits.max()
        soft = np.exp(z) / np.exp(z).sum()
        dlogits = -soft.copy()
        dlogits[cls] += 1.0
        ge = mask_gradient(p, hs, cache, g_vec, dlogits, rows, cols, n, A)
        dsig = w * (1.0 - w)
        grad = ge * dsig
        grad -= L1 * dsig                                    # sparsity
        ent = np.log(w + 1e-12) - np.log(1.0 - w + 1e-12)
        grad -= LENT * (-ent) * dsig                         # entropy
        m += LR * grad / (np.abs(grad).max() + 1e-12)
    w = 1.0 / (1.0 + np.exp(-m))
    score = np.zeros(n)
    np.add.at(score, rows, w)
    np.add.at(score, cols, w)
    return score


def main():
    z = np.load(CKPT)
    p = {}
    for k in z.files:
        p[k] = z[k]
    graphs = M.load_cached(use_renamed=False)
    families = set()
    for g in graphs:
        families.add(g["family"])
    classes = sorted(families)
    cls_idx = {}
    for i, c in enumerate(classes):
        cls_idx[c] = i

    usable = []
    for g in graphs:
        if g["family"] != "non_proxy" and g["tgt"].sum() >= 5:
            usable.append(g)
    # A smallest-first sample is badly unrepresentative here: small graphs carry a
    # proportionally larger SDK, so the base rate rises and the comparison flatters
    # every method equally. Sample at random across families instead, stratified so
    # each family contributes in proportion to what the corpus holds.
    import collections
    import random
    by_fam = collections.defaultdict(list)
    for g in usable:
        by_fam[g["family"]].append(g)
    rng_pick = random.Random(42)
    picked = []
    if N_GRAPHS >= len(usable):
        # asking for everything: no sampling question to answer, use the population
        usable.sort(key=lambda g: (g["family"], g["sha"]))
        print(f"[*] using all {len(usable)} evaluable graphs", flush=True)
        return_all = True
    else:
        return_all = False
    fams = sorted(by_fam)
    per = max(1, N_GRAPHS // len(fams))
    for fam in fams:
        lst = sorted(by_fam[fam], key=lambda g: g["sha"])
        rng_pick.shuffle(lst)
        picked.extend(lst[:per])
    if not return_all:
        usable = picked[:N_GRAPHS]
    print(f"[*] {len(usable)} graphs, {ITERS} mask iterations each", flush=True)

    rows = {"gnnexplainer": [], "exact": [], "degree": [], "base": []}
    for i, g in enumerate(usable):
        c = cls_idx[g["family"]]
        k = int(g["tgt"].sum())
        rows["base"].append(k / g["n"])

        s = explain(p, g, c)
        top = np.argsort(-s)[:k]
        rows["gnnexplainer"].append(float(g["tgt"][top].sum()) / k)

        ex, _ = M.node_contributions(p, g, c)
        top = np.argsort(-ex)[:k]
        rows["exact"].append(float(g["tgt"][top].sum()) / k)

        out_deg = np.asarray(g["A"].sum(axis=1)).ravel()
        in_deg = np.asarray(g["A"].sum(axis=0)).ravel()
        deg = out_deg + in_deg
        top = np.argsort(-deg)[:k]
        rows["degree"].append(float(g["tgt"][top].sum()) / k)

        print(f"    [{i+1}/{len(usable)}] {g['family']:12} n={g['n']:6} "
              f"gnnexp={rows['gnnexplainer'][-1]:.3f} exact={rows['exact'][-1]:.3f}",
              flush=True)

    result = {"graphs": len(usable), "iters": ITERS,
              "l1": L1, "entropy": LENT, "lr": LR, "precision": {},
              "per_graph": {}}
    for k2 in rows:
        result["precision"][k2] = round(float(np.mean(rows[k2])), 5)
        rounded = []
        for x in rows[k2]:
            rounded.append(round(float(x), 6))
        result["per_graph"][k2] = rounded

    # Is the learned explainer actually distinguishable from a degree ranking?
    # Paired over graphs, since the same graph gives every method its score.
    from scipy.stats import wilcoxon
    a = np.asarray(rows["gnnexplainer"])
    d = np.asarray(rows["degree"])
    x = np.asarray(rows["exact"])
    tests = {}
    for name, u, v in (("gnnexplainer_vs_degree", a, d),
                       ("exact_vs_gnnexplainer", x, a),
                       ("exact_vs_degree", x, d)):
        diff = u - v
        if np.allclose(diff, 0):
            tests[name] = {"median_diff": 0.0, "p": 1.0, "note": "identical"}
            continue
        try:
            st = wilcoxon(u, v)
            tests[name] = {"median_diff": round(float(np.median(diff)), 5),
                           "mean_diff": round(float(diff.mean()), 5),
                           "p": round(float(st.pvalue), 5),
                           "wins": int((diff > 0).sum()),
                           "losses": int((diff < 0).sum())}
        except Exception as exc:
            tests[name] = {"error": str(exc)[:60]}
    result["paired_tests"] = tests
    json.dump(result, open(OUT, "w"), indent=2)

    b = result["precision"]["base"]
    print(f"\n[O1] precision@|target|, {len(usable)} graphs, base rate {b:.4f}")
    for k2 in ("gnnexplainer", "exact", "degree"):
        v = result["precision"][k2]
        print(f"  {k2:16}{v:8.4f}   {v/b:5.2f}x chance")
    print(f"\n  paired comparisons (Wilcoxon, n={len(usable)}):")
    for name, t in result["paired_tests"].items():
        if "p" in t:
            print(f"    {name:26} mean diff {t.get('mean_diff', 0):+.4f}  "
                  f"p={t['p']:.4f}  W/L {t.get('wins','-')}/{t.get('losses','-')}")
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
