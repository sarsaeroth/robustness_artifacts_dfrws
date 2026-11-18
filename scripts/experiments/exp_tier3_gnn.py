#!/usr/bin/env python3
"""Tier 3: E12, E13, E15 - a GIN implemented directly in numpy.

E12  Expressiveness ceiling. Message passing is bounded above by 1-WL, so a GIN
     on the same graphs with the same initial colouring should not beat the WL
     kernel. Any excess should trace to node features rather than to expressive
     power, and a clear win would mean the WL colouring is discarding signal.
E13  Attribution. For a mean-pooling readout, node i's contribution to a logit
     is exact: sum_l (dlogit/dg_l) . H_l[i] / n. That is a faithful attribution
     at the readout, unlike a learned explainer, which makes it the fair
     comparison against provenance ranking. Run on renamed graphs too, where
     provenance has nothing to aggregate on.
E15  Inflation. Padding with k inert nodes dilutes mean pooling by n/(n+k),
     which is polynomial. The kernel's attribution decays as exp(-k(H+1)/d),
     which is exponential. Same attack, two different laws; measuring both is
     the point.

No autograd: the backward pass is written out. GIN layer is
    H' = MLP( (A + (1+eps)I) H ),  MLP = Linear -> ReLU -> Linear
readout is per-layer mean pooling concatenated, classifier is linear.

Stages:  prep   parse and cache graphs as npz
         train  fit the GIN, report macro-F1
         attrib E13 and E15 on the trained model
Run with no argument to do all three.

Env: PER_FAM (default 60), EPOCHS (default 30), HID (default 64),
     VOCAB (default 4096), LAYERS (default 3), CORPUS_ROOT, MEM_GB (default 14)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import gen_wl_vectors as G
import gen_wl_renamed as REN

if os.environ.get("GPU") != "1":  # an address-space cap blocks ROCm's virtual-memory reservation; run_guarded.sh still limits resident memory
    E.cap_memory(float(os.environ.get("MEM_GB", "14")))

CACHE = E.out_path("exp_tier3_graphs")
OUT = E.out_path("exp_tier3_gnn.json")
CKPT = E.out_path("exp_tier3_gnn_params.npz")
PER_FAM = int(os.environ.get("PER_FAM", "60"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
HID = int(os.environ.get("HID", "64"))
VOCAB = int(os.environ.get("VOCAB", "4096"))
LAYERS = int(os.environ.get("LAYERS", "3"))
LR = 2e-3
BATCH = 8
K_SWEEP = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]
RNG = np.random.default_rng(0)


def prep():
    os.makedirs(CACHE, exist_ok=True)
    samples = E.list_samples(per_family=PER_FAM)
    print(f"[*] prep {len(samples)} graphs", flush=True)
    n_done = 0
    t0 = time.time()
    for fam, sha, path in samples:
        out = os.path.join(CACHE, f"{sha}.npz")
        if os.path.isfile(out):
            continue
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception:
            continue
        labels = E.kept_labels(nl, "fcg")
        del nl
        if len(labels) < 50:
            del labels, su, pr
            continue

        order = sorted(labels)
        index = {}
        for i, nid in enumerate(order):
            index[nid] = i
        n = len(order)

        rows, cols = [], []
        for nid in order:
            i = index[nid]
            # successor list, empty when the node has none
            succ = su.get(nid)
            if not succ:
                succ = ()
            for s in succ:
                if s in index:
                    rows.append(i)
                    cols.append(index[s])
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)

        # two feature vocabularies: identifier colouring, and the same graph
        # after per-app renaming, which is where E13 separates the methods
        seed = sha[:16]
        feat = np.empty(n, dtype=np.int32)
        feat_ren = np.empty(n, dtype=np.int32)
        tgt = np.zeros(n, dtype=np.int8)
        prefixes = E.SDK_PREFIXES.get(fam, ())
        for nid in order:
            i = index[nid]
            lbl = labels[nid]
            feat[i] = G.stable_hash_u32(lbl) % VOCAB
            feat_ren[i] = G.stable_hash_u32(REN.rename_label(lbl, seed)) % VOCAB
            for p in prefixes:
                if lbl.startswith(p):
                    tgt[i] = 1
                    break

        np.savez_compressed(out, rows=rows, cols=cols, feat=feat,
                            feat_ren=feat_ren, tgt=tgt, n=np.int32(n),
                            family=np.bytes_(fam))
        del labels, su, pr
        n_done += 1
        if n_done % 20 == 0:
            print(f"    {n_done} cached, {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[*] cache holds {len(glob.glob(os.path.join(CACHE, '*.npz')))} graphs",
          flush=True)


def load_cached(use_renamed=False):
    graphs = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        n = int(z["n"])
        rows = z["rows"]
        cols = z["cols"]
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n, n))
        if use_renamed:
            feat = z["feat_ren"]
        else:
            feat = z["feat"]
        graphs.append({
            "sha": os.path.basename(f)[:-4],
            "A": A,
            "feat": feat,
            "tgt": z["tgt"],
            "n": n,
            "family": str(z["family"].item().decode()),
        })
    return graphs


def init_params(n_classes):
    p = {}
    p["emb"] = (RNG.standard_normal((VOCAB, HID)) * 0.1).astype(np.float32)
    for l in range(LAYERS):
        p[f"W1_{l}"] = (RNG.standard_normal((HID, HID)) * (1.0 / np.sqrt(HID))).astype(np.float32)
        p[f"b1_{l}"] = np.zeros(HID, dtype=np.float32)
        p[f"W2_{l}"] = (RNG.standard_normal((HID, HID)) * (1.0 / np.sqrt(HID))).astype(np.float32)
        p[f"b2_{l}"] = np.zeros(HID, dtype=np.float32)
        p[f"eps_{l}"] = np.zeros(1, dtype=np.float32)
    p["Wo"] = (RNG.standard_normal(((LAYERS + 1) * HID, n_classes)) * 0.1).astype(np.float32)
    p["bo"] = np.zeros(n_classes, dtype=np.float32)
    return p


def forward(p, g, keep_cache=False):
    """Returns logits, pooled features per layer, and the cache for backward."""
    A = g["A"]
    H = p["emb"][g["feat"]]
    hs = [H]
    cache = []
    for l in range(LAYERS):
        Z = A.dot(H) + (1.0 + p[f"eps_{l}"][0]) * H
        U = Z @ p[f"W1_{l}"] + p[f"b1_{l}"]
        R = np.maximum(U, 0.0)
        Hn = R @ p[f"W2_{l}"] + p[f"b2_{l}"]
        if keep_cache:
            cache.append((H, Z, U, R))
        H = Hn
        hs.append(H)
    pooled = []
    for h in hs:
        pooled.append(h.mean(axis=0))
    g_vec = np.concatenate(pooled)
    logits = g_vec @ p["Wo"] + p["bo"]
    return logits, g_vec, hs, cache


def backward(p, g, hs, cache, g_vec, dlogits):
    grads = {}
    for k in p:
        grads[k] = np.zeros_like(p[k])
    grads["Wo"] = np.outer(g_vec, dlogits)
    grads["bo"] = dlogits
    dg = p["Wo"] @ dlogits

    n = g["n"]
    dh = []
    for l in range(LAYERS + 1):
        seg = dg[l * HID:(l + 1) * HID]
        dh.append(np.tile(seg / n, (n, 1)).astype(np.float32))

    A = g["A"]
    for l in range(LAYERS - 1, -1, -1):
        Hin, Z, U, R = cache[l]
        dH_out = dh[l + 1]
        grads[f"W2_{l}"] += R.T @ dH_out
        grads[f"b2_{l}"] += dH_out.sum(axis=0)
        dR = dH_out @ p[f"W2_{l}"].T
        dU = dR * (U > 0)
        grads[f"W1_{l}"] += Z.T @ dU
        grads[f"b1_{l}"] += dU.sum(axis=0)
        dZ = dU @ p[f"W1_{l}"].T
        grads[f"eps_{l}"] += np.array([float((dZ * Hin).sum())], dtype=np.float32)
        dh[l] += A.T.dot(dZ) + (1.0 + p[f"eps_{l}"][0]) * dZ

    np.add.at(grads["emb"], g["feat"], dh[0])
    return grads


def train(graphs, classes):
    cls_idx = {}
    for i, c in enumerate(classes):
        cls_idx[c] = i
    p = init_params(len(classes))
    m, v = {}, {}
    for k in p:
        m[k] = np.zeros_like(p[k])
        v[k] = np.zeros_like(p[k])
    step = 0

    order = list(range(len(graphs)))
    for ep in range(EPOCHS):
        RNG.shuffle(order)
        tot_loss = 0.0
        acc = {}
        for k in p:
            acc[k] = np.zeros_like(p[k])
        in_batch = 0
        for gi in order:
            g = graphs[gi]
            y = cls_idx[g["family"]]
            logits, g_vec, hs, cache = forward(p, g, keep_cache=True)
            # softmax
            x = logits - logits.max()
            e = np.exp(x)
            probs = e / e.sum()
            tot_loss += -np.log(max(probs[y], 1e-12))
            dlogits = probs.copy()
            dlogits[y] -= 1.0
            grads = backward(p, g, hs, cache, g_vec, dlogits.astype(np.float32))
            for k in acc:
                acc[k] += grads[k]
            in_batch += 1
            if in_batch == BATCH:
                step += 1
                for k in p:
                    gk = acc[k] / in_batch
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v[k] = 0.999 * v[k] + 0.001 * (gk * gk)
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    p[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
                    acc[k][:] = 0
                in_batch = 0
        print(f"    epoch {ep+1:>3}  loss {tot_loss/len(graphs):.4f}", flush=True)
    return p, cls_idx


def train_torch(graphs, classes):
    """train() on the GPU (GPU=1): same initialization (same RNG draws), forward pass, per-graph
    gradients averaged over BATCH graphs and Adam update as the numpy version, with autograd for
    the backward pass. Returns numpy parameters in the same format."""
    import torch
    dev = "cuda"
    cls_idx = {}
    for i, c in enumerate(classes):
        cls_idx[c] = i
    p0 = init_params(len(classes))
    P = {}
    for k, v in p0.items():
        P[k] = torch.tensor(v, device=dev, requires_grad=True)
    m = {}
    for k, v in P.items():
        m[k] = torch.zeros_like(v)
    v_ = {}
    for k, v in P.items():
        v_[k] = torch.zeros_like(v)
    G = []
    for g in graphs:
        A = sp.csr_matrix(g["A"], dtype=np.float32)
        crow = torch.as_tensor(A.indptr, dtype=torch.int64)
        col = torch.as_tensor(A.indices, dtype=torch.int64)
        val = torch.as_tensor(A.data)
        At = torch.sparse_csr_tensor(crow, col, val, size=A.shape, device=dev)
        G.append((At, torch.as_tensor(g["feat"], device=dev), cls_idx[g["family"]]))

    def fwd(At, feat):
        H = P["emb"][feat]
        pooled = [H.mean(0)]
        for l in range(LAYERS):
            Z = At @ H + (1.0 + P[f"eps_{l}"][0]) * H
            R = torch.relu(Z @ P[f"W1_{l}"] + P[f"b1_{l}"])
            H = R @ P[f"W2_{l}"] + P[f"b2_{l}"]
            pooled.append(H.mean(0))
        return torch.cat(pooled) @ P["Wo"] + P["bo"]

    step = 0
    order = list(range(len(graphs)))
    for ep in range(EPOCHS):
        RNG.shuffle(order)
        tot_loss = 0.0
        in_batch = 0
        for k in P:
            P[k].grad = None
        for gi in order:
            At, feat, y = G[gi]
            logits = fwd(At, feat)
            target = torch.tensor([y], device=dev)
            loss = torch.nn.functional.cross_entropy(logits[None], target)
            loss.backward()  # gradients accumulate over the batch, as in the numpy version
            tot_loss += float(loss)
            in_batch += 1
            if in_batch == BATCH:
                step += 1
                with torch.no_grad():
                    for k in P:
                        gk = P[k].grad / in_batch
                        m[k].mul_(0.9).add_(0.1 * gk)
                        v_[k].mul_(0.999).add_(0.001 * gk * gk)
                        mh = m[k] / (1 - 0.9 ** step)
                        vh = v_[k] / (1 - 0.999 ** step)
                        P[k] -= LR * mh / (torch.sqrt(vh) + 1e-8)
                        P[k].grad = None
                in_batch = 0
        print(f"    epoch {ep+1:>3}  loss {tot_loss/len(graphs):.4f}", flush=True)
    out = {}
    for k, v in P.items():
        out[k] = v.detach().cpu().numpy()
    del G, P, m, v_
    torch.cuda.empty_cache()
    return out, cls_idx


def node_contributions(p, g, cls):
    """Exact contribution of each node to one logit, given mean-pooling readout."""
    logits, g_vec, hs, _ = forward(p, g)
    w = p["Wo"][:, cls]
    n = g["n"]
    contrib = np.zeros(n, dtype=np.float64)
    for l in range(LAYERS + 1):
        seg = w[l * HID:(l + 1) * HID]
        contrib += (hs[l] @ seg) / n
    return contrib, logits


def main():
    graphs = load_cached(use_renamed=False)
    if not graphs:
        sys.exit("no cached graphs; run: exp_tier3_gnn.py prep")
    family_set = set()
    for g in graphs:
        family_set.add(g["family"])
    classes = sorted(family_set)
    print(f"[*] {len(graphs)} graphs, classes {classes}", flush=True)

    # grouped split so near-identical builds cannot straddle train and test
    gp_path = os.path.join(E.ROOT, "results", "full_sweep", "dex_groups.json")
    if os.path.isfile(gp_path):
        groups = json.load(open(gp_path))
    else:
        groups = {}
    keyed = {}
    for i, g in enumerate(graphs):
        if g["sha"] in groups:
            key = str(groups[g["sha"]])  # ids are ints, fallback is str
        else:
            key = str("solo_" + g["sha"])
        if key not in keyed:
            keyed[key] = []
        keyed[key].append(i)
    keys = sorted(keyed)
    RNG.shuffle(keys)
    cut = int(0.75 * len(keys))
    tr = []
    te = []
    for k in keys[:cut]:
        tr.extend(keyed[k])
    for k in keys[cut:]:
        te.extend(keyed[k])
    print(f"[*] {len(tr)} train / {len(te)} test (DEX-grouped)", flush=True)

    train_graphs = []
    for i in tr:
        train_graphs.append(graphs[i])
    test_graphs = []
    for i in te:
        test_graphs.append(graphs[i])
    cls_idx = {}
    for i, c in enumerate(classes):
        cls_idx[c] = i
    if os.path.isfile(CKPT) and os.environ.get("RETRAIN", "0") != "1":
        z = np.load(CKPT)
        p = {}
        for k in z.files:
            p[k] = z[k]
        print(f"[*] loaded trained parameters from {os.path.basename(CKPT)}", flush=True)
    else:
        use_gpu = False
        if os.environ.get("GPU") == "1":
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                pass
        if use_gpu:
            p, cls_idx = train_torch(train_graphs, classes)
        else:
            p, cls_idx = train(train_graphs, classes)
        np.savez(CKPT, **p)
        print(f"[*] saved parameters to {os.path.basename(CKPT)}", flush=True)
    # macro-F1 on the test graphs
    y_true = []
    y_pred = []
    for g in test_graphs:
        logits, _, _, _ = forward(p, g)
        y_true.append(cls_idx[g["family"]])
        y_pred.append(int(np.argmax(logits)))
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n[E12] GIN macro-F1 (DEX-grouped test): {f1:.4f}", flush=True)

    # E13: attribution against the SDK ground truth
    def attrib_scores(gs):
        precs = []
        for g in gs:
            if g["tgt"].sum() < 5 or g["family"] == "non_proxy":
                continue
            contrib, _ = node_contributions(p, g, cls_idx[g["family"]])
            k = int(g["tgt"].sum())
            top = np.argsort(-contrib)[:k]
            precs.append(float(g["tgt"][top].sum()) / k)
        return precs

    pr_id = attrib_scores(test_graphs)
    ren_graphs = load_cached(use_renamed=True)
    ren_test = []
    test_shas = set()
    for g in test_graphs:
        test_shas.add(g["sha"])
    for g in ren_graphs:
        if g["sha"] in test_shas:
            ren_test.append(g)
    pr_ren = attrib_scores(ren_test)

    print(f"[E13] precision@|target|, identifier features: "
          f"{np.mean(pr_id):.4f} (n={len(pr_id)})")
    print(f"[E13] precision@|target|, renamed features:    "
          f"{np.mean(pr_ren):.4f} (n={len(pr_ren)})", flush=True)

    if pr_id:
        e13_identifier = round(float(np.mean(pr_id)), 5)
    else:
        e13_identifier = None
    if pr_ren:
        e13_renamed = round(float(np.mean(pr_ren)), 5)
    else:
        e13_renamed = None
    partial = {
        "graphs": len(graphs), "classes": classes,
        "train": len(tr), "test": len(te),
        "hid": HID, "layers": LAYERS, "vocab": VOCAB, "epochs": EPOCHS,
        "E12_gin_macro_f1": round(float(f1), 5),
        "E13_precision_identifier": e13_identifier,
        "E13_precision_renamed": e13_renamed,
    }
    json.dump(partial, open(OUT, "w"), indent=2)

    # E15: inflation, polynomial dilution rather than exponential
    infl = {}
    probe = []
    for g in test_graphs:
        if g["tgt"].sum() >= 5 and g["family"] != "non_proxy":
            probe.append(g)
        if len(probe) >= 20:
            break
    for k in K_SWEEP:
        vals = []
        for g in probe:
            n = g["n"]
            # inert nodes carry no edges, so data/indices are untouched and only
            # indptr is extended; slice-assigning into a CSR builds a dense
            # index array instead and exhausts memory on real graphs
            A0 = g["A"]
            pad_indptr = np.concatenate([
                A0.indptr,
                np.full(k, A0.indptr[-1], dtype=A0.indptr.dtype)])
            pad_A = sp.csr_matrix((A0.data, A0.indices, pad_indptr),
                                  shape=(n + k, n + k))
            pad_feat = np.concatenate([g["feat"],
                                       RNG.integers(0, VOCAB, size=k).astype(np.int32)])
            pad_tgt = np.concatenate([g["tgt"], np.zeros(k, dtype=np.int8)])
            pg = {"A": pad_A, "feat": pad_feat, "tgt": pad_tgt,
                  "n": n + k, "family": g["family"], "sha": g["sha"]}
            contrib, _ = node_contributions(p, pg, cls_idx[g["family"]])
            kk = int(pad_tgt.sum())
            top = np.argsort(-contrib)[:kk]
            vals.append(float(pad_tgt[top].sum()) / kk)
        infl[str(k)] = round(float(np.mean(vals)), 5)
    print(f"\n[E15] GNN attribution under inflation (n={len(probe)} apps)")
    print(f"  {'k':>6}{'precision':>12}{'n/(n+k) ref':>14}")
    probe_sizes = []
    for g in probe:
        probe_sizes.append(g["n"])
    n0 = float(np.mean(probe_sizes))
    for k in K_SWEEP:
        print(f"  {k:>6}{infl[str(k)]:>12.4f}{n0/(n0+k):>14.4f}")

    json.dump({
        "graphs": len(graphs), "classes": classes,
        "train": len(tr), "test": len(te),
        "hid": HID, "layers": LAYERS, "vocab": VOCAB, "epochs": EPOCHS,
        "E12_gin_macro_f1": round(float(f1), 5),
        "E13_precision_identifier": e13_identifier,
        "E13_precision_renamed": e13_renamed,
        "E15_inflation_precision": infl,
        "E15_mean_nodes": round(n0, 1),
    }, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stage = sys.argv[1]
    else:
        stage = "all"
    if stage in ("prep", "all"):
        prep()
    if stage in ("train", "all"):
        main()
