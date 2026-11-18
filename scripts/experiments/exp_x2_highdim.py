#!/usr/bin/env python3
"""X2 high-dimension arm: do explainers separate once hash collisions clear?

At the deployed d=1,024, ~100 tokens share each bucket, so every bucket-level
explainer is capped near the base rate (X2 main result). This sweeps d up to 2^20.
One WL pass per staged app yields every token's 32-bit hash + origin method; the
bucket (h % d) and sign (top bit) are recomputed at each d, so re-vectorization is
free. At each d a linear SGD is trained on the staged proxy apps with 5-fold
DEX-grouped CV, and the exact Proposition-1 method decomposition is scored against
SDK namespaces on the held-out apps.

Prediction (EXPLAINER_BENCHMARK H1, second half): method-level precision rises
with d for the identifier-stable families (ipninja, luminati) as buckets become
SDK-pure, approaching the d* isolation dimension; monkeysocks/oxylabs stay flat
(renamed / unstable namespaces).

Output: results/tier4/x2_highdim.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "x2_highdim.json")
DIMS = []
for _e in (10, 12, 14, 16, 18, 20):
    DIMS.append(2 ** _e)
del _e
ITERS = 2
STABLE = {"ipninja", "luminati"}


def tokens(nl, su, pr):
    """One WL pass: every emitted token's 32-bit hash and origin method, as arrays.
    Returns (hashes uint32, origin int32, methods): origin indexes methods, which are numbered in order
    of first emission."""
    lab = {}
    for n, raw in nl.items():
        lab[n] = G.canonical_label(raw, "fcg")
    drop = []
    for n, l in lab.items():
        if l.startswith(G.LIB_PREFIXES):
            drop.append(n)
    if drop and len(drop) < len(lab):
        for n in drop:
            del lab[n]
    if not lab:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32), []
    # number methods in order of first appearance
    mid = {}
    node_m = {}
    for n, l in lab.items():
        if l not in mid:
            mid[l] = len(mid)
        node_m[n] = mid[l]
    hs = []
    org = []

    def emit(tok, node):
        hs.append(G.stable_hash_u32(tok))
        org.append(node_m[node])
    for n, l in lab.items():
        emit(f"0|{l}", n)
    cur = lab
    for it in range(1, ITERS + 1):
        nxt = {}
        for n in cur:
            i_l = []
            for p in pr.get(n, []):
                if p in cur:
                    i_l.append(cur[p])
            i_l.sort()
            o_l = []
            for s in su.get(n, []):
                if s in cur:
                    o_l.append(cur[s])
            o_l.sort()
            nxt[n] = wl_digest(cur[n], i_l, o_l)
        cur = nxt
        for n in cur:
            emit(f"{it}|{cur[n]}", n)
    return np.array(hs, np.uint32), np.array(org, np.int32), list(mid)


def load_tokens(path):
    z = np.load(path)
    if z["m"].size:
        ms = z["m"].tobytes().decode("utf-8", "surrogatepass").split("\n")
    else:
        ms = []
    return z["h"], z["o"], ms


def _bucket_sign(h, d):
    return (h % d).astype(np.int64), np.where(h & 0x80000000, -1.0, 1.0)


def main():
    ids, fam = H.sample_ids("four")
    pos = {}
    for k, i in enumerate(ids):
        pos[i] = k
    staged = []
    for f in H.PROXY:
        for sha in H.staged(f):
            if sha in pos:
                staged.append((f, sha))
    print(f"{len(staged)} staged proxy apps; extracting WL tokens ...", flush=True)
    # tokens are spilled to one npz per app and read back when needed: all apps' tokens do not fit in memory
    tokdir = H.data_dir("tier4", "x2_highdim_tokens")
    os.makedirs(tokdir, exist_ok=True)
    tok_path = {}
    for f, sha in staged:
        p = os.path.join(tokdir, f"{sha}.npz")
        if not os.path.exists(p):
            try:
                nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
                toks = tokens(nl, su, pr)
                del nl, su, pr
                # hashes, origins, and method names as one utf-8 byte array
                th, to, tms = toks
                np.savez(p[:-4] + ".tmp.npz", h=th, o=to,
                         m=np.frombuffer("\n".join(tms).encode("utf-8", "surrogatepass"), np.uint8))
                os.replace(p[:-4] + ".tmp.npz", p)
            except Exception as e:
                print(f"  skip {sha[:10]}: {e}", flush=True)
                continue
        tok_path[sha] = (f, p)
    shas = []
    for _, s in staged:
        if s in tok_path:
            shas.append(s)
    fam_list = []
    for s in shas:
        fam_list.append(tok_path[s][0])
    fams = np.array(fam_list)
    le = LabelEncoder().fit(fams)
    y = le.transform(fams)
    g = H.groups_for(np.array(shas))
    print(f"cached {len(shas)} apps; sweeping d ...", flush=True)

    res = {"dims": DIMS, "n_apps": len(shas), "per_dim": {}}
    for d in DIMS:
        # L2-normalized sparse vectors at dimension d from cached token hashes
        rows = []
        for s in shas:
            toks = load_tokens(tok_path[s][1])
            idx, sgn = _bucket_sign(toks[0], d)
            v = np.zeros(d, dtype=np.float64)
            np.add.at(v, idx, sgn)
            n = np.linalg.norm(v)
            if n > 0:
                rows.append(sparse.csr_matrix(v / n))
            else:
                rows.append(sparse.csr_matrix(v))
        X = sparse.vstack(rows).tocsr()
        del rows
        prec = defaultdict(list)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        for tr, te in cv.split(np.zeros(len(y)), y, g):
            up = H.upsample(tr, y, 0)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-4, random_state=42)
            m.fit(X[up], y[up])
            for j in te:
                f = fams[j]
                c = list(le.classes_).index(f)
                w = m.coef_[c].astype(np.float64)
                # exact Prop-1 attribution at dim d: psi_m = sum_tok w[h%d]*sign / ||phi||
                h, org, ms = load_tokens(tok_path[shas[j]][1])
                v = np.zeros(d)
                idx, sgn = _bucket_sign(h, d)
                np.add.at(v, idx, sgn)
                nrm = np.linalg.norm(v)
                if not nrm:
                    nrm = 1.0
                sc = np.zeros(len(ms))
                np.add.at(sc, org, w[idx] * sgn / nrm)  # sequential, in token order, like the per-token loop
                truth_list = []
                for meth in ms:
                    truth_list.append(meth.startswith(P.SDK_PREFIXES.get(f, ())))
                truth = np.array(truth_list, dtype=bool)
                nt = int(truth.sum())
                if nt == 0:
                    r = None
                else:
                    order = np.argsort(-sc)
                    r = {"p_at_target": float(truth[order[:nt]].mean()),
                         "p_at_25": float(truth[order[:25]].mean()),
                         "n_target": nt, "base_rate": nt / len(ms)}
                if r:
                    prec[f].append(r)
        row = {}
        for f in H.PROXY:
            rs = prec.get(f, [])
            if rs:
                pt = []
                p25 = []
                base = []
                enr = []
                for x in rs:
                    pt.append(x["p_at_target"])
                    p25.append(x["p_at_25"])
                    base.append(x["base_rate"])
                    if x["base_rate"] > 0:
                        enr.append(x["p_at_target"] / x["base_rate"])
                row[f] = {"p_at_target": float(np.mean(pt)),
                          "p_at_25": float(np.mean(p25)),
                          "base_rate": float(np.mean(base)),
                          "enrichment": float(np.mean(enr)),
                          "n": len(rs)}
        res["per_dim"][str(d)] = row
        parts = []
        for f in H.PROXY:
            if f in row:
                parts.append(f"{f[:4]}={row[f]['p_at_target']:.2f}({row[f]['enrichment']:.0f}x)")
        line = " ".join(parts)
        print(f"d=2^{int(np.log2(d)):<2} {line}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("\nprecision@target vs d (stable families should rise):")
    for f in ("ipninja", "luminati", "monkeysocks", "oxylabs"):
        cells = []
        for d in DIMS:
            per_dim = res["per_dim"][str(d)]
            if f in per_dim and "p_at_target" in per_dim[f]:
                pt = per_dim[f]["p_at_target"]
            else:
                pt = float("nan")
            cells.append(f"2^{int(np.log2(d))}={pt:.2f}")
        print(f"  {f:11} " + " ".join(cells))


if __name__ == "__main__":
    main()
