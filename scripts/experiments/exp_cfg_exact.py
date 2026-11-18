#!/usr/bin/env python3
"""the exact-token controls on control-flow graphs.

CFG nodes are basic blocks labeled "method:block"; stripping the block index gives the
method signature, so the FCG exact-token builder applies with SDK membership by method
prefix. Re-runs on CFG, with the FCG side recomputed on identical folds:

  D1a  Naive Bayes vs logistic on exact tokens (additivity signature), four families,
       identifier and degree colorings, corrected resampled t-test.
  D1b  Full / SDK-only / host-only on the apps with SDK ground truth (as A2a), identifier.

Output: results/tier4/v3/cfg_exact.json
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import exp_tok_analyses as T
import exp_sdk_transplant as A2
import exp_parts_cache as PC
import gen_wl_vectors as G
import provenance_localization as PL

OUT = H.out_path("tier4", "v3", "cfg_exact.json")
REPEATS = 2
SGD = lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-3, random_state=42)


def cfg_tokens(fam, sha, coloring):
    """As T.app_tokens, on the CFG; 'methods' here are blocks, each flagged by its method's SDK origin."""
    nl, su, pr = G.parse_gml_lightweight(os.path.join(T.GRAPHS, fam, sha, f"{sha}.cfg"))
    # canonical labels; drop library blocks unless that would drop everything
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "cfg")
    drop = []
    for n, l in lab.items():
        if l.startswith(G.LIB_PREFIXES):
            drop.append(n)
    if drop and len(drop) < len(lab):
        for n in drop:
            del lab[n]
    keep = set(lab)
    pref = tuple(PL.SDK_PREFIXES.get(fam, ()))

    # initial colors
    if coloring == "identifier":
        c0 = dict(lab)
    else:
        c0 = {}
        for n in lab:
            n_in = 0
            for p in pr.get(n, []):
                if p in keep:
                    n_in += 1
            n_out = 0
            for s in su.get(n, []):
                if s in keep:
                    n_out += 1
            c0[n] = f"deg|{n_in}|{n_out}"
    nodes = list(lab)
    midx = {}
    for i, n in enumerate(nodes):
        midx[n] = i
    blk_sdk_list = []
    for n in nodes:
        blk_sdk_list.append(lab[n].startswith(pref))
    blk_sdk = np.array(blk_sdk_list, dtype=bool)

    # round-0 tokens, then WL rounds
    tid = []
    rnd = []
    morg = []
    for n in nodes:
        tid.append(T._h("0|" + c0[n]))
        rnd.append(0)
        morg.append(midx[n])
    cur = c0
    for it in range(1, T.ITERS + 1):
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
        for n in nodes:
            tid.append(T._h(f"{it}|{cur[n]}"))
            rnd.append(it)
            morg.append(midx[n])
    del nl, su, pr
    return (np.array(tid, dtype=np.uint64), np.array(rnd, dtype=np.int8),
            np.array(morg, dtype=np.int32), blk_sdk)


def nb_vs_logistic(graph, col, label):
    # exact-token matrices for every cached app
    raw = PC.load(graph, col)
    fam_set = set()
    for r in raw.values():
        fam_set.add(r[0])
    fam_names = sorted(fam_set)
    shalist = list(raw)
    y_list = []
    for s in shalist:
        y_list.append(fam_names.index(raw[s][0]))
    y = np.array(y_list)
    P = {}
    for s in shalist:
        P[s] = raw[s][1:]
    del raw
    X = A2.matrices(shalist, P, A2.Vocab(shalist, P))["full"]
    del P

    # Naive Bayes vs logistic on identical folds
    g = H.groups_for(np.array(shalist))
    sc = {"nb": [], "logistic": []}
    F = H.folds(y, g, REPEATS)
    for r, k, tr, te in F:
        up = H.upsample(tr, y, k)
        nb = MultinomialNB(alpha=0.1, fit_prior=False).fit(X[up], y[up])
        sc["nb"].append(f1_score(y[te], nb.predict(X[te]), average="macro"))
        logistic = SGD().fit(X[up], y[up])
        sc["logistic"].append(f1_score(y[te], logistic.predict(X[te]), average="macro"))
    train_sizes = []
    test_sizes = []
    for f in F:
        train_sizes.append(len(f[2]))
        test_sizes.append(len(f[3]))
    ntr = float(np.mean(train_sizes))
    nte = float(np.mean(test_sizes))
    md, t, p = H.corrected_t(np.array(sc["nb"]) - np.array(sc["logistic"]), ntr, nte)
    out = {"n": len(shalist), "nb": float(np.mean(sc["nb"])), "logistic": float(np.mean(sc["logistic"])),
           "gap": md, "t": t, "p": p}
    print(f"  {label}: nb={out['nb']:.3f} logistic={out['logistic']:.3f} p={p:.3f}", flush=True)
    return out


def sdk_host(graph, label):
    # apps with SDK tokens
    raw = PC.load(graph, "identifier", A2.FAMS)
    shalist = []
    for s, r in raw.items():
        if len(r[1]) > 0:
            shalist.append(s)
    P = {}
    for s in shalist:
        P[s] = raw[s][1:]
    y_list = []
    for s in shalist:
        y_list.append(A2.FAMS.index(raw[s][0]))
    y = np.array(y_list, dtype=int)
    del raw
    if len(np.unique(y)) < 2:  # e.g. piggyback: only one family carries SDK/payload tokens, nothing to compare
        counts = np.bincount(y, minlength=len(A2.FAMS)).tolist()
        n_by_family = {}
        for i, f in enumerate(A2.FAMS):
            n_by_family[f] = counts[i]
        return {"skipped": "fewer than two families with SDK/payload tokens",
                "n_by_family": n_by_family}

    # full / SDK-only / host-only
    X = A2.matrices(shalist, P, A2.Vocab(shalist, P))
    out = A2.cv_compare(X, y, H.groups_for(np.array(shalist)), label)
    counts = np.bincount(y, minlength=len(A2.FAMS)).tolist()
    n_by_family = {}
    for i, f in enumerate(A2.FAMS):
        n_by_family[f] = counts[i]
    out["n_by_family"] = n_by_family
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    save = lambda: json.dump(res, open(OUT, "w"), indent=1)
    res["repeats"] = REPEATS

    # D1a: NB vs logistic, per coloring and graph (resumable)
    for col in ("identifier", "degree"):
        for graph in ("cfg", "fcg"):
            d1a = res.setdefault("D1a", {})
            d1a_col = d1a.setdefault(col, {})
            if graph in d1a_col:
                continue
            res["D1a"][col][graph] = nb_vs_logistic(graph, col, f"D1a {graph} {col}")
            save()

    # D1b: SDK / host split on CFG
    d1b = res.setdefault("D1b", {})
    if "cfg" not in d1b:
        res["D1b"]["cfg"] = sdk_host("cfg", "D1b cfg")
        save()
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
