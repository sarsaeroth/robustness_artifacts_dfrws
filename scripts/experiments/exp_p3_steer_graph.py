"""Graph-level steered attack with real name synthesis (robustness paper,
sec:method_steering / res_frontier). The feature-space attacks of exp_r2/frontier
add abstract +/-1 tokens to the count vector. This closes most of the gap to a
real program: it synthesizes actual method names, splices stub methods into the
real FCG, and re-runs the ACTUAL vectorizer (gen_wl_vectors.wl_hashed_vector:
real WL refinement, real signed hashing, real library dropping, real L2
normalization) on the modified graph, then checks whether the deployed linear
model flips. The only remaining step to a built APK is compiling each (empty,
never-called) stub method into DEX, which is mechanical; no APK is produced here.

Attack: for the true class y and nearest wrong class c, compute the score-gap
gradient on the clean vector and pick the top buckets with their evasive sign.
Synthesize k stub methods whose iteration-0 token f"0|<name>" is ground (by
varying the name and re-hashing, ~2d hashes each) to land in those buckets with
those signs, round-robin; attach each as a fresh leaf under one real host call
site. Their iteration-1/2 tokens and the shifted labels of the attach site are
left to fall where they may and are captured by full re-vectorization, so the
measured k is an honest problem-space cost, not the idealized feature-space one.
k is found by exponential then binary search, re-vectorizing once per trial.

Output: results/tier4/p3/steer_graph.json
"""
import hashlib
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
OUT = H.out_path("tier4", "p3", "steer_graph.json")
D = 1024
PER_FAM = 6
MAX_K = 8192
N_BUCKETS = 4           # concentrate on the top-|gradient| buckets


def grind_name(bucket, sign, used):
    """A method label whose iteration-0 FCG token hashes to (bucket, sign)."""
    if sign > 0:
        wsign = 0
    else:
        wsign = 0x80000000
    i = 0
    while True:
        name = f"Lp/s{bucket}x{i};->m()V"      # non-library owner Lp/ so it is kept
        h = G.stable_hash_u32(f"0|{name}")
        if h % D == bucket and (h & 0x80000000) == wsign and name not in used:
            used.add(name)
            return name
        i += 1
        if i > 8 * D:
            return None


def build_targets(phi, W, b, y):
    nrm = np.linalg.norm(phi)
    g = (W[y] - W) @ phi + (b[y] - b) * nrm
    g[y] = np.inf
    c = int(np.argmin(g))
    grad = (W[y] - W[c]) + (b[y] - b[c]) * (phi / nrm)   # increasing a bucket raises g_c by grad[j]
    order = np.argsort(-np.abs(grad))[:N_BUCKETS]
    # (bucket, evasive sign): push each bucket against its gradient
    targets = []
    for j in order:
        if grad[j] > 0:
            targets.append((int(j), -1.0))
        else:
            targets.append((int(j), 1.0))
    return targets


def attack_graph(nl, su, pr, site, targets, k, W, b, y):
    """Add k grinded stubs as leaves under `site`; return True if prediction flips."""
    nl2 = dict(nl)
    su2 = {}
    for n, v in su.items():
        su2[n] = list(v)
    pr2 = {}
    for n, v in pr.items():
        pr2[n] = list(v)
    used = set()
    nid = -1
    for i in range(k):
        j, s = targets[i % len(targets)]
        name = grind_name(j, s, used)
        if name is None:
            continue
        nl2[nid] = name
        if site not in su2:
            su2[site] = []
        su2[site].append(nid)
        pr2[nid] = [site]
        su2[nid] = []
        nid -= 1
    v = G.wl_hashed_vector(nl2, su2, pr2, "fcg", D, iters=2, drop_lib=True, use_direction=True)
    flipped = int(np.argmax(W @ v + b)) != y
    return flipped, v


def main():
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    idx, _, XP = H._load("fcg_wl_1024")
    rows_of_ids = []
    for i in ids:
        rows_of_ids.append(idx[i])
    X = XP[rows_of_ids].astype(np.float64)
    F = H.folds(y, g, 1)
    # app id -> test fold
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    Wb = {}
    apps = []
    id_set = set(ids)
    for f in H.PROXY:
        dd = os.path.join(GRAPHS, f)
        cands = []
        for s in H.staged(f):
            if s in id_set:
                cands.append(s)
        # smallest FCG files first
        shas = sorted(cands, key=lambda s: GI.size(os.path.join(dd, s, f"{s}.fcg")))
        for s in shas[:PER_FAM]:
            apps.append((f, s))
    rows = []
    for f, sha in apps:
        k = fold_of[sha]
        if k not in Wb:
            _, _, tr, _ = F[k]
            up = H.upsample(tr, y, k)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42).fit(X[up], y[up])
            Wb[k] = (m.coef_.astype(np.float64), m.intercept_.astype(np.float64))
        W, b = Wb[k]
        nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
        su = dict(su)
        pr = dict(pr)
        v0 = G.wl_hashed_vector(nl, su, pr, "fcg", D, iters=2, drop_lib=True, use_direction=True)
        yy = int(le.transform([f])[0])
        if int(np.argmax(W @ v0 + b)) != yy:
            continue
        # targets from the normalized vector's gradient (sign/order unaffected by positive scale)
        targets = build_targets(v0, W, b, yy)
        # host attach site: highest-degree developer (non-library) node
        dev = []
        for n, l in nl.items():
            if not G.canonical_label(l, "fcg").startswith(G.LIB_PREFIXES):
                dev.append(n)
        if not dev:
            continue
        # first node of maximal in+out degree (same tie-break as max())
        site = None
        site_deg = None
        for n in dev:
            deg = 0
            if n in su:
                deg += len(su[n])
            if n in pr:
                deg += len(pr[n])
            if site_deg is None or deg > site_deg:
                site = n
                site_deg = deg
        # exponential search then binary search on k
        lo = 0
        hi = 1
        hit = None
        while hi <= MAX_K:
            flip, _ = attack_graph(nl, su, pr, site, targets, hi, W, b, yy)
            if flip:
                hit = hi
                break
            lo = hi
            hi = hi * 2
        kmin = None
        if hit is not None:
            a = lo
            c = hit
            while c - a > 1:
                mid = (a + c) // 2
                flip, _ = attack_graph(nl, su, pr, site, targets, mid, W, b, yy)
                if flip:
                    c = mid
                else:
                    a = mid
            kmin = c
        rows.append({"family": f, "sha": sha, "k_graph": kmin, "site_deg": site_deg,
                     "n_nodes": len(nl)})
        print(f"{f} {sha[:10]} k_graph={kmin} nodes={len(nl)}", flush=True)
    def med(sub=None):
        v = []
        for r in rows:
            if r["k_graph"] is not None and (sub is None or r["family"] == sub):
                v.append(r["k_graph"])
        if v:
            return float(np.median(v))
        return None
    k_median = med()
    if rows:
        never = []
        for r in rows:
            never.append(r["k_graph"] is None)
        never_flipped = float(np.mean(never))
    else:
        never_flipped = None
    per_family = {}
    for f in H.PROXY:
        n_fam = 0
        for r in rows:
            n_fam += r["family"] == f
        per_family[f] = {"n": n_fam, "k_graph_median": med(f)}
    res = {"n": len(rows), "note": "k in synthesized stub methods; each emits ~3 WL tokens; full pipeline re-vectorization",
           "k_graph_median": k_median,
           "never_flipped_frac": never_flipped,
           "per_family": per_family,
           "rows": rows}
    json.dump(res, open(OUT, "w"), indent=1)
    summary = {}
    for k, v in res.items():
        if k != "rows":
            summary[k] = v
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
