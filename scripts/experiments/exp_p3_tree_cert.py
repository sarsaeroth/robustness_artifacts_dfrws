"""Tree-ensemble addition certificate on normalized features (robustness paper,
sec:method_certificates, the case exp_p3_frontier left to future work).

A random forest predicts on x = phi/||phi||. Adding T tokens sends phi -> phi+Delta
with ||Delta||_1 = T, so each raw coordinate moves within [phi_j - T, phi_j + T]
and ||phi+Delta||_2 within [max(eps, ||phi|| - T), ||phi|| + T]. The induced box on
each normalized coordinate is therefore
    x_j in [ (phi_j - T)/den_lo(j), (phi_j + T)/den_hi(j) ],
taking the denominator that minimizes/maximizes each endpoint by its sign; this is
a sound over-approximation of the reachable set (it ignores the l_1 coupling
across coordinates, which only enlarges the box). Interval-propagating this box
through every decision tree gives the set of classes each tree could still vote
for; a tree is "flippable" if it could stop voting for the predicted class. The
forest prediction is certified against every T-token addition while the number of
flippable trees is below the one-vs-runner-up vote margin, and the certified
budget is the largest such T (found by bisection). This is a sound lower bound on
the additions needed to change the forest, the tree analogue of Theorem certgrad;
the greedy aimed attack (same as the linear frontier, on the forest's own predict)
is the matching upper bound.

Output: results/tier4/p3/tree_cert.json
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p3_frontier as FR


def tree_reachable_classes(tree, lo, hi):
    """Classes reachable at any leaf consistent with the box [lo, hi]. DFS over
    nodes whose split feature interval straddles the threshold."""
    cl = tree.children_left
    cr = tree.children_right
    feat = tree.feature
    thr = tree.threshold
    val = tree.value  # (n_nodes, 1, n_classes)
    out = set()
    stack = [0]
    while stack:
        n = stack.pop()
        if cl[n] == -1:
            out.add(int(np.argmax(val[n][0])))
            continue
        j = feat[n]
        t = thr[n]
        if lo[j] <= t:
            stack.append(cl[n])
        if hi[j] > t:
            stack.append(cr[n])
    return out


def box(phi, T, nrm):
    den_hi = nrm + T
    den_lo = max(1e-9, nrm - T)
    num_lo = phi - T
    num_hi = phi + T
    lo = np.where(num_lo >= 0, num_lo / den_hi, num_lo / den_lo)
    hi = np.where(num_hi >= 0, num_hi / den_lo, num_hi / den_hi)
    return lo, hi


def rf_cert(rf, phi, nrm, y, max_T=20000):
    trees = []
    for e in rf.estimators_:
        trees.append(e.tree_)
    classes = rf.classes_.tolist()
    yi = classes.index(y)
    # runner-up by clean vote
    votes = np.zeros(len(classes))
    x_clean = (phi.reshape(1, -1) / nrm).astype(np.float32)
    for tr in trees:
        leaf = tr.apply(x_clean)
        voted = int(np.argmax(tr.value[leaf][0][0]))
        votes[voted] += 1
    order = np.argsort(-votes)
    if order[0] == yi:
        ru = order[1]
    else:
        ru = order[0]
    margin = votes[yi] - votes[ru]
    if margin <= 0:
        return 0

    def flippable(T):
        lo, hi = box(phi, T, nrm)
        cnt = 0
        for tr in trees:
            rc = tree_reachable_classes(tr, lo, hi)
            # a tree endangers the margin if it could drop y or could add a vote to ru
            if yi not in rc or len(rc) > 1:
                # could it stop voting y AND/OR start voting ru? worst case: -1 to y, +1 to ru
                cnt += 1
        return cnt

    # certified while flippable trees can't erase the margin: worst case each
    # flippable tree shifts the (y - ru) gap by 2 (loses a y vote, gives ru one)
    lo_T = 0
    hi_T = 1
    if not 2 * flippable(0) < margin:
        return 0
    # doubling, then bisection on the largest certified T
    while hi_T < max_T and 2 * flippable(hi_T) < margin:
        lo_T = hi_T
        hi_T = hi_T * 2
    while hi_T - lo_T > 1:
        mid = (lo_T + hi_T) // 2
        if 2 * flippable(mid) < margin:
            lo_T = mid
        else:
            hi_T = mid
    return lo_T


def rf_attack(rf, phi, nrm, y, max_T=2000, batch=20):
    """Greedy aimed addition against the forest's own predict (upper bound).
    Re-aims every `batch` tokens: evaluate a random candidate set once, then add
    that many tokens at the best (bucket, sign) before re-aiming. Cheap enough to
    pair with the certificate; it is an upper bound, so coarse aiming is fine."""
    classes = rf.classes_.tolist()
    yi = classes.index(y)
    phi = phi.astype(float).copy()
    rng = np.random.default_rng(0)
    t = 0
    while t < max_T:
        best_j = None
        best_s = 1.0
        best_p = 2.0
        # aim: the (bucket, sign) among 24 random buckets that lowers p(y) most
        for j in rng.choice(len(phi), 24, replace=False):
            for s in (1.0, -1.0):
                phi[j] += s
                x = (phi / np.linalg.norm(phi)).reshape(1, -1)
                p = rf.predict_proba(x)[0][yi]
                phi[j] -= s
                if p < best_p:
                    best_p = p
                    best_j = j
                    best_s = s
        for _ in range(batch):
            phi[best_j] += best_s
            t += 1
            x = (phi / np.linalg.norm(phi)).reshape(1, -1)
            if rf.predict(x)[0] != y:
                return t
    return None


def main():
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    idx, _, XP = H._load("fcg_wl_1024")
    row_idx = []
    for i in ids:
        row_idx.append(idx[i])
    X = XP[row_idx].astype(np.float64)
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
    models = {}
    rows = []
    percap = {}
    for f in H.PROXY:
        percap[f] = 0
    attack_cap = {}
    for i, (f, sha) in enumerate(apps):
        if percap[f] >= 12:
            continue
        percap[f] += 1
        k = fold_of[sha]
        if k not in models:
            _, _, tr, _ = F[k]
            up = H.upsample(tr, y, k)
            rf_model = RandomForestClassifier(n_estimators=60, n_jobs=6, random_state=42)
            rf_model.fit(X[up], y[up])
            models[k] = rf_model
        rf = models[k]
        phi = mats["labeled"][i]
        nrm = np.linalg.norm(phi)
        yy = int(le.transform([f])[0])
        # only apps with a non-empty vector that the forest classifies correctly
        if nrm == 0:
            continue
        if rf.predict((phi / nrm).reshape(1, -1))[0] != yy:
            continue
        kc = rf_cert(rf, phi, nrm, yy)
        if f not in attack_cap:
            attack_cap[f] = 0
        if attack_cap[f] < 6:
            ka = rf_attack(rf, phi, nrm, yy, max_T=2000)
            attack_cap[f] += 1
        else:
            ka = None
        rows.append({"family": f, "k_cert": kc, "k_attack": ka})
        print(f"{len(rows)} {f} cert={kc} attack={ka}", flush=True)
    def med(key, sub=None):
        v = []
        for r in rows:
            if r[key] is not None and (sub is None or r["family"] == sub):
                v.append(r[key])
        if v:
            return float(np.median(v))
        return None
    # summary values
    n_rows = len(rows)
    k_cert_median = med("k_cert")
    k_attack_median = med("k_attack")
    if rows:
        zero_flags = []
        for r in rows:
            zero_flags.append(r["k_cert"] == 0)
        zero_cert_frac = float(np.mean(zero_flags))
    else:
        zero_cert_frac = None
    if rows:
        ratios = []
        for r in rows:
            if r["k_attack"]:
                ratios.append(r["k_attack"] / max(r["k_cert"], 1))
        attack_over_cert = float(np.median(ratios))
    else:
        attack_over_cert = None
    per_family = {}
    for f in H.PROXY:
        n_fam = 0
        for r in rows:
            if r["family"] == f:
                n_fam += 1
        per_family[f] = {"n": n_fam, "k_cert": med("k_cert", f), "k_attack": med("k_attack", f)}
    res = {"n": n_rows, "model": "RandomForest(60) on normalized FCG identifier WL",
           "k_cert_median": k_cert_median, "k_attack_median": k_attack_median,
           "zero_cert_frac": zero_cert_frac,
           "attack_over_cert_median": attack_over_cert,
           "per_family": per_family}
    json.dump({"summary": res, "rows": rows}, open(H.out_path("tier4", "p3", "tree_cert.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
