#!/usr/bin/env python3
"""G2: can a TRAINED GNN localize to SDK code? (exact GIN logit decomposition vs WL/SGC)

Completes the identity-vs-invariance / classification-vs-localization frontier with a real
end-to-end trained message-passing GNN. E1 showed a DECOUPLED linear GNN (SGC) classifies
(.87-.90) but localizes ~0 on renaming-INVARIANT node features. G2 asks whether a heavier,
fully trained GIN buys localization it could not.

Exact attribution (no heuristic explainer needed): our GIN is sum-pool + linear head, so
  logit_y = w_y . BN_pool(sum_i h_i) + b = sum_i (w' . h_i) + const,  w' = w_y (*) gamma/sigma
folding the post-pool BatchNorm affine into the head. Node i contribution r_i = w' . h_i is
EXACT (the graph analog of E1's Linear-SHAP-on-SGC / graph-CAM, Pope CVPR'19). Rank methods
by r_i, precision@(#SDK) vs SDK ground truth - the SAME metric as E1/WL.

Node features = E1's continuous renaming-invariant descriptor (matched to E1/SGC and G1).
4-class, staged FCGs, DEX-grouped CV, GPU (ROCm / RX 7900 XTX). Memory-safe: localize one
app at a time. Prediction: trained GIN localizes ~0 too -> identity in the features, not
model expressiveness, is what enables localization. Output: results/tier4/g2_gnn_localize.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P
import exp_e1_sgc as E1

GRAPHS = E1.GRAPHS
OUT = H.out_path("tier4", "g2_gnn_localize.json")


def load(fam, sha):
    import torch
    from torch_geometric.data import Data
    nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
    ids, idx, X = E1.node_features(nl, su)
    if len(ids) < 10:
        return None
    truth_list = []
    for i in ids:
        truth_list.append(str(nl[i]).startswith(P.SDK_PREFIXES.get(fam, ())))
    truth = np.array(truth_list)
    if not truth.any():
        return None
    # directed call edges between kept nodes, then both directions
    src = []
    dst = []
    for u in ids:
        for v in su.get(u, []):
            if v in idx:
                src.append(idx[u])
                dst.append(idx[v])
    if not src:
        return None
    ei = torch.tensor([src, dst], dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    d = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=ei)
    d.num_nodes = len(ids)
    del nl, su, pr
    return d, truth


def make_gin(in_dim, hidden, n_classes):
    import torch
    from torch import nn
    from torch_geometric.nn import GINConv, global_add_pool

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Linear(o, o))

    class GIN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inorm = nn.BatchNorm1d(in_dim)
            self.c1 = GINConv(mlp(in_dim, hidden))
            self.b1 = nn.BatchNorm1d(hidden)
            self.c2 = GINConv(mlp(hidden, hidden))
            self.b2 = nn.BatchNorm1d(hidden)
            self.c3 = GINConv(mlp(hidden, hidden))
            self.b3 = nn.BatchNorm1d(hidden)
            self.pool = nn.BatchNorm1d(hidden)
            self.lin = nn.Linear(hidden, n_classes)

        def node_repr(self, x, ei):
            x = self.inorm(x)
            x = self.b1(self.c1(x, ei)).relu()
            x = self.b2(self.c2(x, ei)).relu()
            return self.b3(self.c3(x, ei))          # pre-pool node embeddings h_i

        def forward(self, x, ei, batch):
            return self.lin(self.pool(global_add_pool(self.node_repr(x, ei), batch)))

        def node_contrib(self, h, cls):
            # exact per-node share of logit[cls]: w' = w_cls (*) gamma / sqrt(var+eps)
            eps = self.pool.eps
            wprime = self.lin.weight[cls] * self.pool.weight / torch.sqrt(self.pool.running_var + eps)
            return h @ wprime                        # (n_nodes,)

    return GIN()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    print(f"device={dev}", flush=True)

    ids_all, _ = H.sample_ids("four")
    pos = set(ids_all.tolist())
    # staged apps (optionally capped per family) with SDK ground truth
    staged = []
    for f in H.PROXY:
        for sha in H.staged(f):
            if sha in pos:
                staged.append((f, sha))
    if a.limit:
        byf = defaultdict(list)
        for f, s in staged:
            byf[f].append((f, s))
        staged = []
        for f in byf:
            for t in byf[f][:a.limit]:
                staged.append(t)
    print(f"loading {len(staged)} graphs ...", flush=True)
    data = []
    truths = []
    fams = []
    shas = []
    for f, sha in staged:
        try:
            r = load(f, sha)
        except Exception:
            continue
        if r is None:
            continue
        d, tr = r
        data.append(d)
        truths.append(tr)
        fams.append(f)
        shas.append(sha)
    if len(set(fams)) < 2:  # e.g. CIC-AndMal: no graph carries SDK/payload ground truth, nothing to localize
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump({"graphs": len(data), "skipped": "fewer than two families with SDK/payload ground-truth graphs"},
                  open(OUT, "w"), indent=1)
        print(f"skipped: {len(data)} graphs with SDK/payload ground truth", flush=True)
        return
    le = LabelEncoder().fit(fams)
    y = le.transform(fams)
    for i in range(min(len(data), len(y))):
        data[i].y = torch.tensor([int(y[i])], dtype=torch.long)
    grp = H.groups_for(np.array(shas))
    base = {}
    for f in le.classes_:
        rates = []
        for i in range(len(data)):
            if fams[i] == f:
                rates.append(truths[i].mean())
        base[f] = float(np.mean(rates))
    base_parts = []
    for f in le.classes_:
        base_parts.append(f"{f}={base[f]:.3f}")
    print(f"{len(data)} graphs; SDK base-rate per family: " +
          " ".join(base_parts), flush=True)

    import time
    prec = defaultdict(list)          # family -> [precision@target per test app]
    f1s = []
    for rep in range(a.reps):
        fold_i = 0
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, grp):
            _t0 = time.time()
            m = make_gin(data[0].num_features, a.hidden, len(le.classes_)).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
            # class-balanced training
            inv_counts = []
            for c in range(len(le.classes_)):
                inv_counts.append(1.0 / max(1, (y[tr] == c).sum()))
            cw = torch.tensor(inv_counts, device=dev, dtype=torch.float32)
            cw = cw / cw.sum() * len(le.classes_)
            train_graphs = []
            for i in tr:
                train_graphs.append(data[i])
            dl = DataLoader(train_graphs, batch_size=8, shuffle=True, drop_last=True)
            m.train()
            for _ in range(a.epochs):
                for b in dl:
                    b = b.to(dev)
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(m(b.x, b.edge_index, b.batch), b.y, weight=cw)
                    loss.backward()
                    opt.step()
            m.eval()
            # classification sanity
            preds = []
            ys = []
            test_graphs = []
            for i in te:
                test_graphs.append(data[i])
            with torch.no_grad():
                for b in DataLoader(test_graphs, batch_size=8):
                    b = b.to(dev)
                    preds.append(m(b.x, b.edge_index, b.batch).argmax(1).cpu())
                    ys.append(b.y.cpu())
            f1s.append(f1_score(torch.cat(ys).numpy(), torch.cat(preds).numpy(), average="macro"))
            # localization: exact per-node contribution to TRUE class, one app at a time.
            # move only COPIES of x/edge_index to GPU - never mutate data[i] (PyG .to() is
            # in-place, which would leave that graph on GPU and break the next fold's collate)
            with torch.no_grad():
                for i in te:
                    x = data[i].x.to(dev)
                    ei = data[i].edge_index.to(dev)
                    h = m.node_repr(x, ei)
                    r = m.node_contrib(h, int(y[i])).cpu().numpy()
                    t = truths[i]
                    nt = int(t.sum())
                    order = np.argsort(-r)
                    prec[fams[i]].append(float(t[order[:nt]].mean()))
                    del x, ei, h, r
                    torch.cuda.empty_cache()
            fold_i += 1
            print(f"  rep{rep} fold{fold_i}/5 done in {time.time()-_t0:.0f}s "
                  f"(running macroF1={np.mean(f1s):.3f})", flush=True)
        print(f"rep {rep} done; macroF1={np.mean(f1s):.3f}", flush=True)

    # summary
    localize_p = {}
    for f in le.classes_:
        if prec.get(f):
            localize_p[f] = float(np.mean(prec[f]))
    localize_n = {}
    for f in le.classes_:
        if prec.get(f):
            localize_n[f] = len(prec[f])
    res = {"n": len(data), "epochs": a.epochs, "reps": a.reps,
           "macro_f1": float(np.mean(f1s)), "device": dev,
           "base_rate": base,
           "localize_p_at_target": localize_p,
           "localize_n": localize_n}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n=== G2: trained-GIN exact localization (precision@#SDK) ===")
    print(f"classification macroF1={res['macro_f1']:.3f}")
    for f in le.classes_:
        if prec.get(f):
            print(f"  {f:12} p@target={res['localize_p_at_target'][f]:.3f} "
                  f"(base {base[f]:.3f}, n={len(prec[f])})")
    print("vs E1 invariant-SGC ~0.00-0.02; WL labeled ipninja .20 luminati .65")
    print("done", flush=True)


if __name__ == "__main__":
    main()
