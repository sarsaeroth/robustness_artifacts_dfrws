#!/usr/bin/env python3
"""Recompute the block-normalized certificate (Theorem certblock, vertex form)
for the deployed CFG+FCG classifier and for FCG-only (comparison), reusing the
attack results already stored by exp_p4_cfg_cert.py.

L_k(t) depends on t only, so it is tabulated once per (class, block) for
t = 0..T_MAX; the certified budget is then the largest T such that
min_{t1+t2=T} beta + L_1(t1) + L_2(T-t1) > 0 for every competing class.

Output: updates results/tier4/p4/cfg_cert.json (keys concat|* and fcg_block|*:
k_block becomes the vertex-form certificate; the earlier bound is kept as
k_block_v1).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p4_cfg_cert as C

OUT = C.OUT
# this step runs with GRAPHS=cfg,fcg (to include the CFG block), which tags the output path; it post-processes
# the cfg_cert.json written by exp_p4_cfg_cert under the dataset's default graphs, so fall back to that file
if not os.path.exists(OUT):
    _untagged = OUT.replace(f"{H.DATASET}_cfg+fcg", H.DATASET)
    if os.path.exists(_untagged):
        OUT = _untagged
T_MAX = 16384
_W = {}


def _init(w):
    _W.update(w)


def cert_app(args):
    key, sha, fam, k = args
    setting, colour = key.split("|")
    W, b = _W[(key, k)]
    pc, pf = C.raw_vectors(sha, colour)
    if setting == "concat":
        phis = [pc, pf]
    else:
        phis = [pf]
    yy = _W["le"][fam]
    # block offsets of each raw vector inside the concatenated weight row
    dims = []
    for p in phis:
        dims.append(len(p))
    offs = np.cumsum([0] + dims)
    worst = np.full(T_MAX + 1, np.inf)
    for c in range(W.shape[0]):
        if c == yy:
            continue
        u = W[yy] - W[c]
        beta = b[yy] - b[c]
        # L_k(t) tabulated for t = 0..T_MAX, one table per block
        tabs = []
        for q in range(len(phis)):
            tabs.append(C.block_L_vec(C.block_consts(phis[q], u[offs[q]:offs[q + 1]]), np.arange(T_MAX + 1)))
        if len(tabs) == 1:
            val = beta + tabs[0]
        else:
            # val[T] = beta + min_{t<=T} L0[t] + L1[T-t]; computed for all T at once in chunks
            L0, L1 = tabs
            val = np.empty(T_MAX + 1)
            for T in range(T_MAX + 1):
                val[T] = beta + np.min(L0[:T + 1] + L1[T::-1])
                if val[T] <= 0:
                    val[T + 1:] = -np.inf
                    break
        worst = np.minimum(worst, val)
    bad = np.where(worst <= 0)[0]
    if len(bad):
        return key, sha, int(bad[0] - 1)
    return key, sha, T_MAX


def main():
    res = json.load(open(OUT))
    ids, fam = H.sample_ids("four")
    ids, fam = H.with_cfg_fcg(ids, fam)
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    le_map = {}
    for f in H.PROXY:
        le_map[f] = int(le.transform([f])[0])
    W = {"le": le_map}
    # refit the models of every setting not yet in vertex form, and queue its apps
    block_keys = []
    for k in res:
        if k.startswith(("concat|", "fcg_block|")):
            block_keys.append(k)
    tasks = []
    for key in block_keys:
        if res[key].get("vertex_form"):
            continue
        setting, colour = key.split("|")
        Xc = H._load(C.PACK["cfg"][colour])
        Xf = H._load(C.PACK["fcg"][colour])
        rows_c = []
        for i in ids:
            rows_c.append(Xc[0][i])
        Xcfg = Xc[2][rows_c].astype(float)
        rows_f = []
        for i in ids:
            rows_f.append(Xf[0][i])
        Xfcg = Xf[2][rows_f].astype(float)
        if setting == "concat":
            X = np.hstack([Xcfg, Xfcg])
        else:
            X = Xfcg
        for r, k, tr, te in F:
            W[(key, k)] = C.fit(X, y, tr, 1000 * r + k)       # identical models to exp_p4_cfg_cert
        for row in res[key]["rows"]:
            tasks.append((key, row["sha"], row["family"], fold_of[row["sha"]]))
    print(len(tasks), "certificates to recompute", flush=True)
    new = {}
    with ProcessPoolExecutor(4, initializer=_init, initargs=(W,)) as ex:
        for n, (key, sha, kb) in enumerate(ex.map(cert_app, tasks, chunksize=4), 1):
            new[(key, sha)] = kb
            if n % 50 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    # write the new certificates back, keeping the earlier bound as *_v1
    new_keys = set()
    for k, _ in new:
        new_keys.add(k)
    for key in new_keys:
        rows = res[key]["rows"]
        for row in rows:
            row["k_block_v1"] = row["k_block"]
            row["k_block"] = new[(key, row["sha"])]
        kb = []
        for x in rows:
            kb.append(x["k_block"])
        res[key]["k_block_v1_median"] = res[key]["k_block_median"]
        res[key]["k_block_median"] = float(np.median(kb))
        viol = 0
        for x in rows:
            if x["k_attack"] is not None and x["k_attack"] <= x["k_block"]:
                viol += 1
        res[key]["violations"] = viol
        att = []
        for x in rows:
            if x["k_attack"]:
                att.append(x["k_attack"] / max(x["k_block"], 1))
        if att:
            res[key]["attack_over_cert_median"] = float(np.median(att))
        else:
            res[key]["attack_over_cert_median"] = None
        res[key]["vertex_form"] = True
        print(key, "block", res[key]["k_block_median"], "(v1", res[key]["k_block_v1_median"], ") viol",
              res[key]["violations"], "ratio", res[key]["attack_over_cert_median"], flush=True)
    json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()
