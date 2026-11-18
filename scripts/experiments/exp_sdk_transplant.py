#!/usr/bin/env python3
"""is the SDK signal portable across hosts?

A1 showed accuracy falls under host-cluster holdout, and the SDK-only figures of A4/G28
included apps with no SDK ground truth (an empty SDK vector is itself a family cue, mostly
Oxylabs). Everything here is restricted to apps with SDK tokens, in the three families with
enough of them (IPNinja, Luminati, Monkeysocks; Oxylabs has only five).

  A2a  Full / SDK-only / host-only macro-F1 under DEX-grouped folds, with corrected
       resampled t-tests (SDK vs host, SDK vs full).
  A2b  The same three feature sets under host-cluster holdout (host-token Jaccard
       components, as A1).
  A2c  Transplant. Train the full-token model; for each test app, combine its SDK tokens
       with the host tokens of an app from a different family and record whether the
       prediction follows the SDK or the host. Hosts drawn from the test fold (unseen) or
       from the training fold (seen). Feature-space construction: tokens are assigned by
       origin method, so the 14-23 boundary methods per graph keep their original neighbors.

Identifier coloring, exact tokens. Output: results/tier4/v3/sdk_transplant.json
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_parts_cache as PC

OUT = H.out_path("tier4", "v3", "sdk_transplant.json")
# proxy: the three families with namespace ground truth (Oxylabs has five apps); other datasets:
# the staged payload families (run with STAGED_SET=payload)
if H.DATASET == "proxy":
    FAMS = ["ipninja", "luminati", "monkeysocks"]
else:
    FAMS = list(H.PROXY)
REPEATS = 2
K_PARTNERS = 5
CLF = lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-3, random_state=42)


class Vocab:
    """Columns = tokens seen (in either part) in at least two apps."""
    def __init__(self, shalist, P):
        per_app = []
        for s in shalist:
            per_app.append(np.unique(np.concatenate([P[s][0], P[s][2]])))
        allt = np.concatenate(per_app)
        u, n = np.unique(allt, return_counts=True)
        self.v = u[n >= 2]

    def cols(self, ids):
        i = np.searchsorted(self.v, ids)
        i[i >= len(self.v)] = 0
        ok = self.v[i] == ids
        return i, ok

    def row(self, pieces):
        """pieces: list of (ids, counts); returns log1p count row (1 x |V|)."""
        cs = []
        vs = []
        for ids, c in pieces:
            i, ok = self.cols(ids)
            cs.append(i[ok])
            vs.append(c[ok])
        cs = np.concatenate(cs)
        vs = np.concatenate(vs)
        r = sparse.csr_matrix((vs, (np.zeros(len(cs), dtype=np.int64), cs)), shape=(1, len(self.v)))
        r.sum_duplicates()
        r.data = np.log1p(r.data)
        return r


def matrices(shalist, P, V):
    rows = {"full": [], "sdk": [], "host": []}
    for s in shalist:
        su, sc, hu, hc = P[s]
        rows["full"].append(V.row([(su, sc), (hu, hc)]))
        rows["sdk"].append(V.row([(su, sc)]))
        rows["host"].append(V.row([(hu, hc)]))
    mats = {}
    for k, v in rows.items():
        mats[k] = sparse.vstack(v).tocsr()
    return mats


def cv_compare(X, y, groups, label):
    """Per-fold macro-F1 for each feature set on identical folds, plus corrected t-tests."""
    F = H.folds(y, groups, REPEATS)
    t0 = time.time()
    sc = {}
    for k in X:
        sc[k] = []
    ntr = 0
    nte = 0
    for r, k, tr, te in F:
        up = H.upsample(tr, y, k)
        ntr = len(tr)
        nte = len(te)
        for name, M in X.items():
            m = CLF()
            m.fit(M[up], y[up])
            sc[name].append(f1_score(y[te], m.predict(M[te]), average="macro"))
    out = {}
    for name, v in sc.items():
        out[name] = {"macro_f1": float(np.mean(v)), "sd": float(np.std(v))}
    for a, b in (("sdk", "host"), ("full", "sdk"), ("full", "host")):
        md, t, p = H.corrected_t(np.array(sc[a]) - np.array(sc[b]), ntr, nte)
        out[f"{a}_minus_{b}"] = {"mean_gap": md, "t": t, "p": p}
    print(f"  {label}: full={out['full']['macro_f1']:.3f} sdk={out['sdk']['macro_f1']:.3f} "
          f"host={out['host']['macro_f1']:.3f} sdk-host p={out['sdk_minus_host']['p']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    # apps with SDK ground truth present
    raw = PC.load("fcg", "identifier", FAMS)
    shalist = []
    for s, r in raw.items():
        if len(r[1]) > 0:
            shalist.append(s)
    P = {}
    for s in shalist:
        P[s] = raw[s][1:]
    y_list = []
    for s in shalist:
        y_list.append(FAMS.index(raw[s][0]))
    y = np.array(y_list, dtype=int)
    del raw
    n_by_fam = dict(zip(FAMS, np.bincount(y, minlength=len(FAMS)).tolist()))
    print(f"apps with SDK tokens: {n_by_fam}", flush=True)
    if len(np.unique(y)) < 2:  # e.g. piggyback: only one family carries payload tokens, so there is nothing to compare
        res["families"] = FAMS
        res["skipped"] = "fewer than two families with SDK/payload tokens"
        res["n_by_family"] = n_by_fam
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
        print("skipped: fewer than two families with SDK/payload tokens", flush=True)
        return
    V = Vocab(shalist, P)
    X = matrices(shalist, P, V)
    dex = H.groups_for(np.array(shalist))
    res["families"] = FAMS
    res["n_by_family"] = dict(zip(FAMS, np.bincount(y).tolist()))
    # median share of SDK tokens per family
    frac = {}
    for i, f in enumerate(FAMS):
        fam_fracs = []
        for s, yy in zip(shalist, y):
            if yy == i:
                fam_fracs.append(P[s][1].sum() / (P[s][1].sum() + P[s][3].sum()))
        frac[f] = float(np.median(fam_fracs))
    res["median_sdk_token_frac"] = frac
    res["vocab"] = int(len(V.v))
    res["repeats"] = REPEATS
    save = lambda: json.dump(res, open(OUT, "w"), indent=1)
    # A2a: DEX-grouped folds
    if "A2a" not in res:
        print("A2a: DEX-grouped full / SDK-only / host-only", flush=True)
        res["A2a"] = cv_compare(X, y, dex, "dex-grouped")
        save()
    # A2b: host-cluster holdout (connected components with an edge when host-token Jaccard >= thr)
    if "A2b" not in res:
        res["A2b"] = {}
    clusters = {}
    for thr in (0.3, 0.5, 0.7):
        n = len(shalist)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        hs = []
        for s in shalist:
            hs.append(P[s][2])
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(np.intersect1d(hs[i], hs[j], assume_unique=True))
                if inter and inter / (len(hs[i]) + len(hs[j]) - inter) >= thr:
                    parent[find(i)] = find(j)
        # number the components in order of first appearance
        roots = {}
        comp = []
        for i in range(n):
            root = find(i)
            if root not in roots:
                roots[root] = len(roots)
            comp.append(roots[root])
        clusters[thr] = np.array(comp, dtype=np.int64)
        if str(thr) in res["A2b"]:
            continue
        print(f"A2b: host-cluster holdout thr={thr}", flush=True)
        r = cv_compare(X, y, clusters[thr], f"host-cluster thr={thr}")
        r["n_clusters"] = int(len(np.unique(clusters[thr])))
        res["A2b"][str(thr)] = r
        save()
    # A2c: transplant SDK tokens of one family into hosts of other families (full-token model)
    if "A2c" not in res:
        res["A2c"] = {}
    for name, g in (("dex_grouped", dex), ("host_cluster_0.3", clusters[0.3])):
        if name not in res["A2c"]:
            print(f"A2c: transplant {name}", flush=True)
            label = f"transplant {name}"
            Xfull = X["full"]
            rng = np.random.default_rng(0)
            F = H.folds(y, g, REPEATS)
            tally = {}
            for src in ("all",) + tuple(FAMS):
                tally[src] = {"unseen": [0, 0, 0], "seen": [0, 0, 0]}
            clean_ok = 0
            clean_n = 0
            for r, k, tr, te in F:
                up = H.upsample(tr, y, k)
                m = CLF()
                m.fit(Xfull[up], y[up])
                pred_clean = m.predict(Xfull[te])
                clean_ok += int((pred_clean == y[te]).sum())
                clean_n += len(te)
                for i in te:
                    su, sc, _, _ = P[shalist[i]]
                    for pool_name, pool in (("unseen", te), ("seen", tr)):
                        cand = []
                        for j in pool:
                            if y[j] != y[i]:
                                cand.append(j)
                        if not cand:
                            continue
                        for j in rng.choice(cand, size=min(K_PARTNERS, len(cand)), replace=False):
                            _, _, hu, hc = P[shalist[j]]
                            p = m.predict(V.row([(su, sc), (hu, hc)]))[0]
                            # slot 0: follows the SDK, 1: follows the host, 2: neither
                            if p == y[i]:
                                slot = 0
                            elif p == y[j]:
                                slot = 1
                            else:
                                slot = 2
                            for src in ("all", FAMS[y[i]]):
                                tally[src][pool_name][slot] += 1
            out = {"clean_accuracy": clean_ok / clean_n}
            for src, d in tally.items():
                out[src] = {}
                for pool_name, (a, b, c) in d.items():
                    n = a + b + c
                    if n:
                        out[src][pool_name] = {"n": n, "follows_sdk": a / n,
                                               "follows_host": b / n, "neither": c / n}
                    else:
                        out[src][pool_name] = {"n": n, "follows_sdk": None,
                                               "follows_host": None, "neither": None}
            a = out["all"]
            print(f"  {label}: clean acc={out['clean_accuracy']:.3f}  unseen hosts: sdk={a['unseen']['follows_sdk']:.3f} "
                  f"host={a['unseen']['follows_host']:.3f}  seen hosts: sdk={a['seen']['follows_sdk']:.3f} "
                  f"host={a['seen']['follows_host']:.3f}", flush=True)
            res["A2c"][name] = out
            save()
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
