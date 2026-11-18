#!/usr/bin/env python3
"""realistic attack threat models beyond the white-box upper bound.

The white-box aimed attack assumes the adversary holds the deployed weights. Two weaker,
more realistic models:

  F25  Transfer across retraining. Craft the aimed perturbation on one model, then test
       whether it flips a model retrained on the same fold with a different seed and a group
       bootstrap. A real adversary never has the exact deployed weights, so this is the
       honest white-box threat.
  F24  Grey-box surrogate. The SDK developer knows the coloring and the public family
       labels but not the weights, so they train their own surrogate on disjoint public data
       and craft on it, then transfer to the deployed model. This is the corrected grey-box
       model, and it sits between random padding and full white-box.

Per family, at the fixed budget B=200 added tokens (matching the earlier strategy result),
flip rate under: random padding, grey-box surrogate (disjoint data), transfer across reseed,
and white-box. Colorings labeled (fully aimable) and structure (iteration-0 token
constrained to stub degree signatures).

Output: results/tier4/v3/attack_realism.json
"""
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR

OUT = H.out_path("tier4", "v3", "attack_realism.json")
COLOURS = ["labeled", "structure"]
B = 200
MAXT = 600


def fit(X, y, idx, seed):
    up = H.upsample(idx, y, seed)
    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=seed % 100000)
    m.fit(X[up], y[up])
    return m.coef_.astype(float), m.intercept_.astype(float)


def steer_delta(phi, W, b, y, menu0=None, max_t=MAXT):
    """Greedy aimed attack on (W,b); returns (perturbed phi, budget) or (phi, None)."""
    p = phi.astype(float).copy()
    t = 0
    while t < max_t:
        for slot in range(3):
            if slot == 0 and menu0:
                allowed = menu0
            else:
                allowed = None
            j, s = FR.greedy_token(p, W, b, y, allowed)
            p[j] += s
            t += 1
            if FR.pred(W, b, p) != y:
                return p, t
    return p, None


def tallies_push(d, name, val):
    d[name].append(bool(val))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    apps, mats = FR.staged_raw(ids)
    app_tuples = []
    for a in apps:
        app_tuples.append(tuple(a))
    apps = app_tuples
    by_fold = {}
    for i, (fa, sha) in enumerate(apps):
        by_fold.setdefault(fold_of[sha], []).append((i, fa, sha))

    out = {}
    for c in COLOURS:
        idx, _, XP = H._load(FR.PACK[c])
        rows_idx = []
        for i in ids:
            rows_idx.append(idx[i])
        X = XP[rows_idx].astype(np.float64)
        if c == "structure":
            menu0 = FR.STRUCT_MENU
        else:
            menu0 = None
        tallies = defaultdict(lambda: defaultdict(list))  # family -> scheme -> flip flags
        for r, k, tr, te in F:
            W, b = fit(X, y, tr, 1000 * r + k)                      # deployed model
            Wr, br = fit(X, y, tr, 200 + k)                         # reseed surrogate
            # bootstrap surrogate: resample whole groups of the training set with replacement
            rng = np.random.default_rng(300 + k)
            ug = np.unique(g[tr])
            members = {}
            for gid in ug:
                members[gid] = tr[g[tr] == gid]
            boot_parts = []
            for gid in rng.choice(ug, len(ug), replace=True):
                boot_parts.append(members[gid])
            Wb, bb = fit(X, y, np.concatenate(boot_parts), 300 + k)
            # disjoint-data grey-box surrogate: half of the fold's training set
            rng = np.random.default_rng(400 + k)
            half = rng.permutation(tr)[: len(tr) // 2]
            Wg, bg = fit(X, y, half, 400 + k)
            for i, fa, sha in by_fold.get(k, []):
                phi = mats[c][i]
                yy = int(le.transform([fa])[0])
                if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != yy:
                    continue
                # white-box on deployed
                pw, tw = steer_delta(phi, W, b, yy, menu0)
                tallies_push(tallies[fa], "white_box", tw is not None and tw <= B)
                # transfer across reseed and bootstrap: craft on surrogate, test deployed
                for (Ws, bs, name) in ((Wr, br, "transfer_reseed"), (Wb, bb, "transfer_bootstrap"),
                                       (Wg, bg, "greybox_disjoint")):
                    ps, ts = steer_delta(phi, Ws, bs, yy, menu0)
                    flipped = ts is not None and ts <= B and FR.pred(W, b, ps) != yy
                    tallies_push(tallies[fa], name, flipped)
                # random padding baseline on deployed
                kr = FR.k_random(phi, W, b, yy, seed=k, max_t=B)
                tallies_push(tallies[fa], "random", kr is not None)
            print(f"  {c} fold {k} done", flush=True)
        # flip rate per family and scheme, then pooled over families
        per_fam = {}
        for fa, d in tallies.items():
            rates = {}
            for sch, v in d.items():
                rates[sch] = float(np.mean(v))
            per_fam[fa] = rates
        out[c] = per_fam
        first = next(iter(tallies.values()))
        overall = {}
        for sch in first:
            flags = []
            for d in tallies.values():
                for f in d[sch]:
                    flags.append(f)
            overall[sch] = float(np.mean(flags))
        out[c]["_overall"] = overall
        print(f"  {c} overall: {out[c]['_overall']}", flush=True)

    res = {"budget": B, "results": out}
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
