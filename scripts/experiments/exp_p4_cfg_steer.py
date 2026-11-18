#!/usr/bin/env python3
"""Graph-level steered attack against CFG features and the deployed CFG+FCG
classifier, with real name synthesis and full re-extraction (robustness paper,
CFG extension; FCG-only version is exp_p3_steer_graph).

Each synthesized stub method appears in both graphs: one FCG node attached as a
leaf under a host call site, and a straight-line chain of b basic blocks in the
CFG, every block labelled with the stub's method signature (as the extractor
does). The stub's round-0 token string '0|<name>' is the same in both graphs, so
grinding the name aims it at the same bucket index in both halves of the
deployed vector, and b times in the CFG half; the stub's later-round tokens are
coupled to its neighbours and scatter. Targets are chosen from the clean
vector's score-gap gradient, combining both graphs with the CFG multiplicity.
k (stubs) is found by exponential then binary search, re-extracting the graphs
for every trial.

Settings: deployed CFG+FCG with b in {1, 16}; CFG-only with b = 16.
Output: results/tier4/p4/cfg_steer.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import graph_io as GI

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "p4", "cfg_steer.json")
D = 1024
PER_FAM = 4
MAX_K = 4096
N_TARGETS = 4
SETTINGS = [("concat", 1), ("concat", 16), ("cfg", 16)]


def vectors(nlc, suc, prc, nlf, suf, prf, setting):
    vc = G.wl_hashed_vector(nlc, suc, prc, "cfg", D, iters=2, drop_lib=True, use_direction=True)
    if setting == "cfg":
        return vc.astype(float)
    vf = G.wl_hashed_vector(nlf, suf, prf, "fcg", D, iters=2, drop_lib=True, use_direction=True)
    return np.concatenate([vc, vf]).astype(float)


def attacked(G0, site_f, site_c, tg, k, b, setting, W, bb, y):
    # private copies of both graphs (adjacency lists copied too)
    nlc0, suc0, prc0, nlf0, suf0, prf0 = G0
    nlc = dict(nlc0)
    suc = {}
    for n, v in suc0.items():
        suc[n] = list(v)
    prc = {}
    for n, v in prc0.items():
        prc[n] = list(v)
    nlf = dict(nlf0)
    suf = {}
    for n, v in suf0.items():
        suf[n] = list(v)
    prf = {}
    for n, v in prf0.items():
        prf[n] = list(v)
    used = set()
    nid = -1
    for i in range(k):
        j, sg = tg[i % len(tg)]
        # grind a fresh stub name whose round-0 token hashes to bucket j with sign sg
        if sg > 0:
            want = 0
        else:
            want = 0x80000000
        name = None
        n_try = 0
        while n_try < 16 * D:
            cand = f"Lp/s{j}x{n_try};->m()V"
            h = G.stable_hash_u32(f"0|{cand}")
            if h % D == j and (h & 0x80000000) == want and cand not in used:
                used.add(cand)
                name = cand
                break
            n_try += 1
        if name is None:
            continue
        # FCG: one node, leaf under the host call site
        if setting == "concat":
            nlf[nid] = name
            if site_f not in suf:
                suf[site_f] = []
            suf[site_f].append(nid)
            prf[nid] = [site_f]
            suf[nid] = []
        # CFG: a straight-line chain of b blocks labelled with the stub's signature
        prev = None
        for _ in range(b):
            nlc[nid] = name
            suc[nid] = []
            prc[nid] = []
            if prev is not None:
                suc[prev].append(nid)
                prc[nid].append(prev)
            prev = nid
            nid -= 1
    v = vectors(nlc, suc, prc, nlf, suf, prf, setting)
    return int(np.argmax(W @ v + bb)) != y


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
    # PER_FAM smallest staged apps per provider (by CFG file size)
    apps = []
    for f in H.PROXY:
        dd = os.path.join(GRAPHS, f)
        in_folds = []
        for s in H.staged(f):
            if s in fold_of:
                in_folds.append(s)
        shas = sorted(in_folds, key=lambda s: GI.size(os.path.join(dd, s, f"{s}.cfg")))
        for s in shas[:PER_FAM]:
            apps.append((f, s))
    for setting, bk in SETTINGS:
        key = f"{setting}|b{bk}"
        if key in res:
            continue
        # training matrix: CFG half, plus the FCG half for the deployed setting
        if setting == "cfg":
            kinds = ["cfg"]
        else:
            kinds = ["cfg", "fcg"]
        parts = []
        for kind in kinds:
            idx, _, XP = H._load(f"{kind}_wl_1024")
            sel_rows = []
            for i in ids:
                sel_rows.append(idx[i])
            parts.append(XP[sel_rows])
        X = np.hstack(parts).astype(float)
        models = {}
        rows = []
        for f, sha in apps:
            k0 = fold_of[sha]
            if k0 not in models:
                _, _, tr, _ = F[k0]
                up = H.upsample(tr, y, k0)
                m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
                m.fit(X[up], y[up])
                models[k0] = (m.coef_.astype(float), m.intercept_.astype(float))
            W, bb = models[k0]
            d = os.path.join(GRAPHS, f, sha)
            nlc, suc, prc = G.parse_gml_lightweight(os.path.join(d, f"{sha}.cfg"))
            nlf, suf, prf = G.parse_gml_lightweight(os.path.join(d, f"{sha}.fcg"))
            G0 = (nlc, dict(suc), dict(prc), nlf, dict(suf), dict(prf))
            x = vectors(G0[0], G0[1], G0[2], G0[3], G0[4], G0[5], setting)
            yy = int(le.transform([f])[0])
            if int(np.argmax(W @ x + bb)) != yy:
                continue
            # host call site: the developer FCG node with the most edges (first one on ties)
            dev = []
            for n, l in nlf.items():
                if not G.canonical_label(l, "fcg").startswith(G.LIB_PREFIXES):
                    dev.append(n)
            site_f = None
            best_deg = None
            for n in dev:
                deg = len(suf.get(n, [])) + len(prf.get(n, []))
                if best_deg is None or deg > best_deg:
                    site_f = n
                    best_deg = deg
            # target buckets from the clean vector's score gap to the closest class
            s = (W[yy] - W) @ x + (bb[yy] - bb)
            s[yy] = np.inf
            c = int(np.argmin(s))
            u = W[yy] - W[c]
            if setting == "cfg":
                gsum = u[:D] * bk
            else:   # same bucket index in both halves; CFG copies weighted by b
                gsum = u[:D] * bk + u[D:]
            order = np.argsort(-np.abs(gsum))[:N_TARGETS]
            tg = []
            for j in order:
                if gsum[j] > 0:
                    tg.append((int(j), -1.0))
                else:
                    tg.append((int(j), 1.0))
            # exponential search for a flipping k, then binary search for the smallest
            lo = 0
            hi = 1
            hit = None
            while hi <= MAX_K:
                if attacked(G0, site_f, None, tg, hi, bk, setting, W, bb, yy):
                    hit = hi
                    break
                lo = hi
                hi = hi * 2
            kmin = None
            if hit is not None:
                a = lo
                cc = hit
                while cc - a > 1:
                    mid = (a + cc) // 2
                    if attacked(G0, site_f, None, tg, mid, bk, setting, W, bb, yy):
                        cc = mid
                    else:
                        a = mid
                kmin = cc
            rows.append({"family": f, "sha": sha, "k_stubs": kmin})
            print(key, f, sha[:10], "k_stubs", kmin, flush=True)
        # summary for this setting
        ks = []
        for r in rows:
            if r["k_stubs"]:
                ks.append(r["k_stubs"])
        summ = {"n": len(rows), "flipped": len(ks)}
        if ks:
            summ["k_median_flipped"] = float(np.median(ks))
        else:
            summ["k_median_flipped"] = None
        if rows:
            never = []
            for r in rows:
                never.append(r["k_stubs"] is None)
            summ["never_frac"] = float(np.mean(never))
        else:
            summ["never_frac"] = None
        per_family = {}
        for f in H.PROXY:
            per_family[f] = []
            for r in rows:
                if r["family"] == f:
                    per_family[f].append(r["k_stubs"])
        summ["per_family"] = per_family
        summ["rows"] = rows
        res[key] = summ
        json.dump(res, open(OUT, "w"), indent=1)
        shown = {}
        for k, v in res[key].items():
            if k != "rows":
                shown[k] = v
        print(key, shown, flush=True)


if __name__ == "__main__":
    main()
