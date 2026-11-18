#!/usr/bin/env python3
"""Random-forest addition certificate and attack on CFG and on the deployed
CFG+FCG vector (robustness paper, CFG extension; FCG version exp_p3_tree_cert).

The interval box of exp_p3_tree_cert is built per block, each block with its
own norm; giving every block the full budget T over-approximates every split of
T tokens across blocks, so the certificate remains sound. The greedy attack adds
tokens to whichever block and bucket most lowers the forest's probability for
the true class (re-aimed every 20 tokens).

Output: results/tier4/p4/cfg_tree.json
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# scores CFG+FCG vectors against models trained on the deployed [cfg | fcg] matrix, so it needs both graph blocks;
# datasets whose default is FCG-only get GRAPHS=cfg,fcg (no change for the proxy corpus)
os.environ.setdefault("GRAPHS", "cfg,fcg")
import exp_harness as H
import exp_p3_tree_cert as TC

P4 = H.out_dir("tier4", "p4")
OUT = os.path.join(P4, "cfg_tree.json")
D = 1024
PER_FAM = 12
ATTACK_PER_FAM = 6


def raw(h):
    v = np.zeros(D)
    np.add.at(v, (h % D).astype(np.int64), np.where(h & 0x80000000, -1.0, 1.0))
    return v


def norm_cat(phis):
    blocks = []
    for p in phis:
        blocks.append(p / np.linalg.norm(p))
    return np.concatenate(blocks)


def cert(rf, phis, y, max_T=20000):
    # vote margin of the true class over the runner-up
    trees = []
    for e in rf.estimators_:
        trees.append(e.tree_)
    cls = rf.classes_.tolist()
    yi = cls.index(y)
    x0 = norm_cat(phis).astype(np.float32).reshape(1, -1)
    votes = np.zeros(len(cls))
    for tr in trees:
        votes[int(np.argmax(tr.value[tr.apply(x0)][0][0]))] += 1
    order = np.argsort(-votes)
    if order[0] == yi:
        ru = order[1]
    else:
        ru = order[0]
    margin = votes[yi] - votes[ru]
    if margin <= 0:
        return 0

    # certified at budget T: trees that can leave the true class cannot overturn the margin
    def ok(T):
        los = []
        his = []
        for p in phis:
            lo_p, hi_p = TC.box(p, T, np.linalg.norm(p))
            los.append(lo_p)
            his.append(hi_p)
        lo = np.concatenate(los)
        hi = np.concatenate(his)
        n = 0
        for tr in trees:
            rc = TC.tree_reachable_classes(tr, lo, hi)
            if yi not in rc or len(rc) > 1:
                n += 1
        return 2 * n < margin
    if not ok(0):
        return 0

    # doubling, then bisection on the largest certified budget
    lo_T = 0
    hi_T = 1
    while hi_T < max_T and ok(hi_T):
        lo_T = hi_T
        hi_T = hi_T * 2
    while hi_T - lo_T > 1:
        mid = (lo_T + hi_T) // 2
        if ok(mid):
            lo_T = mid
        else:
            hi_T = mid
    return lo_T


def attack(rf, phis, y, max_T=2000, batch=20):
    cls = rf.classes_.tolist()
    yi = cls.index(y)
    copies = []
    for p in phis:
        copies.append(p.astype(float).copy())
    phis = copies
    rng = np.random.default_rng(0)
    t = 0
    while t < max_T:
        # pick the block, bucket and sign that most lowers the true-class probability
        best = (2.0, 0, 0, 1.0)
        for k in range(len(phis)):
            for j in rng.choice(D, 12, replace=False):
                for s in (1.0, -1.0):
                    phis[k][j] += s
                    p = rf.predict_proba(norm_cat(phis).reshape(1, -1))[0][yi]
                    phis[k][j] -= s
                    if p < best[0]:
                        best = (p, k, j, s)
        _, k, j, s = best
        # add tokens there until the prediction flips or the batch ends
        for _ in range(batch):
            phis[k][j] += s
            t += 1
            if rf.predict(norm_cat(phis).reshape(1, -1))[0] != y:
                return t
    return None


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
    Xfull = H.matrix("labeled", ids)
    for setting in ("cfg", "concat"):
        if setting in res:
            continue
        if setting == "cfg":
            X = Xfull[:, :D]
        else:
            X = Xfull

        # one forest per fold
        f1s = []
        models = {}
        for r, k, tr, te in F:
            up = H.upsample(tr, y, k)
            models[k] = RandomForestClassifier(n_estimators=60, n_jobs=4, random_state=42).fit(X[up], y[up])
            f1s.append(float(f1_score(y[te], models[k].predict(X[te]), average="macro")))

        # certify and attack correctly classified held-out apps
        rows = []
        cap = {}
        for f in H.PROXY:
            cap[f] = 0
        acap = {}
        for f in H.PROXY:
            acap[f] = 0
        for f in H.PROXY:
            for sha in H.staged(f):
                if sha not in fold_of or cap[f] >= PER_FAM or not os.path.exists(os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{sha}.npz")):
                    continue
                z = np.load(os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens", f"{sha}.npz"))
                phis = [raw(z["cfg|identifier|clean|h"])]
                if setting == "concat":
                    phis.append(raw(z["fcg|identifier|clean|h"]))
                rf = models[fold_of[sha]]
                yy = int(le.transform([f])[0])
                if rf.predict(norm_cat(phis).reshape(1, -1))[0] != yy:
                    continue
                cap[f] += 1
                kc = cert(rf, phis, yy)
                ka = None
                attacked = False
                if acap[f] < ATTACK_PER_FAM:
                    ka = attack(rf, phis, yy)
                    acap[f] += 1
                    attacked = True
                rows.append({"family": f, "k_cert": kc, "k_attack": ka, "attacked": attacked})
                print(setting, f, sha[:10], "cert", kc, "attack", ka, flush=True)

        # summary
        kc = []
        for x in rows:
            kc.append(x["k_cert"])
        ka = []
        for x in rows:
            if x["k_attack"]:
                ka.append(x["k_attack"])
        is_zero = []
        for v in kc:
            is_zero.append(v == 0)
        attack_never = 0
        for x in rows:
            if x["attacked"] and x["k_attack"] is None:
                attack_never += 1
        if ka:
            k_attack_median = float(np.median(ka))
        else:
            k_attack_median = None
        res[setting] = {"f1": float(np.mean(f1s)), "n": len(rows), "k_cert_median": float(np.median(kc)),
                        "zero_frac": float(np.mean(is_zero)),
                        "k_attack_median": k_attack_median,
                        "attack_never": int(attack_never), "rows": rows}
        json.dump(res, open(OUT, "w"), indent=1)
        shown = {}
        for k, v in res[setting].items():
            if k != "rows":
                shown[k] = v
        print(setting, shown, flush=True)


if __name__ == "__main__":
    main()
