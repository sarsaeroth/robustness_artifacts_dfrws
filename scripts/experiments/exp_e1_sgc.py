#!/usr/bin/env python3
"""E1: continuous-yet-explainable localization - SGC/SIGN + exact Linear SHAP vs WL.

The flagship: first Linear-SHAP-on-SGC code-localization result on real malware,
quantifying the continuity <-> sharpness frontier.

Memory-safe design: NEVER caches all apps'
per-node propagated matrices. Pass 1 keeps only the small pooled graph feature per app;
localization RE-PARSES and re-propagates one app at a time inside the CV loop. Address
space is hard-capped (RLIMIT_AS) so a runaway dies instead of taking the box down.

Pipeline (decoupled GNN = finite GNTK analog, DnX/AISTATS'23 precedent):
  node x = continuous, renaming-invariant [structural + hashed framework-call counts];
  H = S^K X, S = D^-1/2 (A+I) D^-1/2 (SGC); SIGN = [X,SX,S2X,S3X]; SUM-pool over nodes.
  Linear head: logit_y = w_y . sum_i H_i + b => node i contribution = w_y . H_i = exact
  per-node Linear SHAP (analytic, deterministic). Rank methods by it; precision vs SDK GT.
Output: results/tier4/e1_sgc.json
"""
import json
import os
import resource
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "e1_sgc.json")
FRAME = ("Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/", "Lkotlinx/",
         "Lcom/google/", "Lorg/apache/", "Lorg/json/", "Lorg/xml/", "Lorg/w3c/",
         "Lsun/", "Ldalvik/")
FHASH = 64
KS = [1, 2, 3]
STRUCT = 6


def is_frame(sig):
    if sig:
        text = str(sig)
    else:
        text = ""
    return text.split("->", 1)[0].startswith(FRAME)


def node_features(labels, succ):
    ids = list(labels)
    idx = {}
    for k, n in enumerate(ids):
        idx[n] = k
    pred_ct = defaultdict(int)
    for u in ids:
        for v in succ.get(u, []):
            if v in idx:
                pred_ct[v] += 1
    X = np.zeros((len(ids), STRUCT + FHASH), dtype=np.float64)
    for n in ids:
        i = idx[n]
        ind = pred_ct[n]
        outd = len(succ.get(n, []))
        if is_frame(labels[n]):
            frame_flag = 1.0
        else:
            frame_flag = 0.0
        X[i, :6] = [ind, outd, np.log1p(ind), np.log1p(outd), ind + outd, frame_flag]
        for s in succ.get(n, []):
            sl = labels.get(s)
            if sl is not None and is_frame(sl):
                # owner package: first three path components of the callee's class
                o = str(sl).split("->", 1)[0]
                parts = o.split("/")
                if len(parts) >= 3:
                    pkg = "/".join(parts[:3])
                else:
                    pkg = o
                X[i, STRUCT + (G.stable_hash_u32("F|" + pkg) % FHASH)] += 1.0
    return ids, idx, X


def app_features(fam, sha):
    """Parse one FCG -> (ids, labels_list, X, {K: S^K X}, truth). One app in memory."""
    nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
    ids, idx, X = node_features(nl, su)
    if len(ids) < 10:
        return None
    # normalized adjacency S = D^-1/2 (A+I) D^-1/2
    n = len(ids)
    rows = []
    cols = []
    for u in ids:  # self-loops
        rows.append(idx[u])
        cols.append(idx[u])
    for u in ids:
        for v in su.get(u, []):
            if v in idx:
                rows.append(idx[v])
                rows.append(idx[u])
                cols.append(idx[u])
                cols.append(idx[v])
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    dinv = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(1)).ravel(), 1e-9))
    D = sparse.diags(dinv)
    S = D @ A @ D
    del rows, cols, A, dinv, D  # freed here, as when this was a separate function
    # propagate H_k = S^k X
    Hk = {}
    cur = X
    for k in range(1, max(KS) + 1):
        cur = S @ cur
        Hk[k] = cur
    truth_list = []
    for i in ids:
        truth_list.append(str(nl[i]).startswith(P.SDK_PREFIXES.get(fam, ())))
    truth = np.array(truth_list)
    del nl, su, pr, S
    return ids, X, Hk, truth


def main():
    if not (os.environ.get("GPU") == "1" or ".venv-rocm" in sys.prefix):  # ROCm needs its address-space reservation
        resource.setrlimit(resource.RLIMIT_AS, (12 * 2**30, 12 * 2**30))   # hard 12 GB cap
    ids_all, fam_all = H.sample_ids("four")
    pos = set(ids_all.tolist())
    staged = []
    for f in H.PROXY:
        for sha in H.staged(f):
            if sha in pos:
                staged.append((f, sha))
    print(f"{len(staged)} staged proxy apps; pass 1 (pooled features, one app at a time) ...", flush=True)

    graph_feats = {}
    for k in KS + ["sign"]:
        graph_feats[k] = {}
    samples = []
    for f, sha in staged:
        try:
            r = app_features(f, sha)
        except Exception as e:
            print(f"  skip {sha[:10]}: {repr(e)[:60]}", flush=True)
            continue
        if r is None:
            continue
        ids, X, Hk, truth = r
        if not truth.any():
            continue
        # sum-pooled graph features per K, and the SIGN concatenation
        for k in KS:
            graph_feats[k][sha] = Hk[k].sum(0)
        sign_parts = [X.sum(0)]
        for k in KS:
            sign_parts.append(Hk[k].sum(0))
        graph_feats["sign"][sha] = np.concatenate(sign_parts)
        del sign_parts
        samples.append((f, sha))
        del ids, X, Hk, truth
    shas = []
    fam_list = []
    for f, s in samples:
        shas.append(s)
        fam_list.append(f)
    fams = np.array(fam_list)
    print(f"pass 1 done: {len(shas)} apps with SDK ground truth", flush=True)

    le = LabelEncoder().fit(fams)
    y = le.transform(fams)
    if len(np.unique(y)) < 2:  # e.g. piggyback: only one family has payload ground truth, nothing to compare
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump({"skipped": "fewer than two families with SDK/payload ground truth",
                   "n_apps": len(shas), "families": sorted(set(fams.tolist()))}, open(OUT, "w"), indent=1)
        print("skipped: fewer than two families with SDK/payload ground truth", flush=True)
        return
    g = H.groups_for(np.array(shas))

    res = {}
    for key in KS + ["sign"]:
        feats = []
        for s in shas:
            feats.append(graph_feats[key][s])
        Xg = np.vstack(feats)
        sc = StandardScaler().fit(Xg)
        Xs = sc.transform(Xg)
        f1s = []
        prec = defaultdict(list)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xs, y, g):
            up = H.upsample(tr, y, 0)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            m.fit(Xs[up], y[up])
            f1s.append(f1_score(y[te], m.predict(Xs[te]), average="macro"))
            for j in te:
                f = fams[j]
                sha = shas[j]
                c = list(le.classes_).index(f)
                r = app_features(f, sha)            # re-parse one app for localization
                if r is None:
                    continue
                ids, X, Hk, truth = r
                if not truth.any():
                    del ids, X, Hk, truth
                    continue
                if key == "sign":
                    blocks = [X]
                    for k in KS:
                        blocks.append(Hk[k])
                    Hcat = np.hstack(blocks)
                    del blocks
                else:
                    Hcat = Hk[key]
                contrib = (Hcat / sc.scale_) @ m.coef_[c]
                nt = int(truth.sum())
                order = np.argsort(-contrib)
                prec[f].append({"p_at_target": float(truth[order[:nt]].mean()),
                                "p_at_25": float(truth[order[:25]].mean()),
                                "ap": float(average_precision_score(truth, contrib)),
                                "base": nt / len(truth)})
                del ids, X, Hk, truth, contrib
        row = {"macro_f1": float(np.mean(f1s))}
        for f in H.PROXY:
            rs = prec.get(f, [])
            if rs:
                means = {}
                for kk in ("p_at_target", "p_at_25", "ap", "base"):
                    vals = []
                    for r in rs:
                        vals.append(r[kk])
                    means[kk] = float(np.mean(vals))
                row[f] = means
        res[key] = row
        cells = []
        for f in H.PROXY:
            if f in row and "p_at_target" in row[f]:
                pt = row[f]["p_at_target"]
            else:
                pt = float("nan")
            cells.append(f"{f[:4]}={pt:.3f}")
        print(f"K={key!s:5} macroF1={row['macro_f1']:.3f}  " + " ".join(cells), flush=True)

    out = {"n_apps": len(shas), "struct_dim": STRUCT, "fhash": FHASH,
           "note": "continuous renaming-invariant node feats; sum-pool; exact per-node linear-SHAP; memory-safe",
           "results": res,
           "wl_baseline_d1024": {"ipninja": .088, "luminati": .138, "monkeysocks": .012, "oxylabs": .033},
           "wl_baseline_2^20": {"luminati": .65, "ipninja": .20, "monkeysocks": .18, "oxylabs": .00}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nWL labeled baseline p@target d=1024: ip .088 lum .138 monkey .012 oxy .033; at 2^20 lum .65", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
