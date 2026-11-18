#!/usr/bin/env python3
"""Refinement depth ablation at full scale, for any dataset (DATASET env).

What does WL refinement add over the initial coloring? For every closed-world application
(proxy: streamed from the bundle; MalNet: datasets/malnet), the FCG is colored with each
coloring and the per-round token vectors (rounds 0, 1, 2) are kept, so the h = 0, 1, 2
representations are sums of rounds on identical graphs. Colorings use the generators of the
deployed vectors: identifier (gen_wl_vectors, library nodes dropped), fuzzy and api
(gen_wl_hardened), degree (in/out degree within the library-dropped graph,
gen_wl_structure_libdrop) and combined (gen_wl_hardened, where opcode maps exist). The h = 2
sums equal the packed vectors (checked on a sample).

Classifier: the ladder's tuned linear model on L2-normalized vectors, 2 repeats of grouped
5-fold CV (harness folds), with the corrected resampled t-test for h = 2 against h = 0.

Output: tier4/v4/h_ablation.json (dataset-aware path); per-app rounds cached in
tier4/v4/h_rounds/{sha}.npz.
Usage: exp_h_ablation.py vectorize [WORKERS] | evaluate
"""
import hashlib
import json
import os
import sys
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD

OUT = H.out_path("tier4", "v4", "h_ablation.json")
CACHE = H.data_dir("tier4", "v4", "h_rounds")
COLS = ["identifier", "fuzzy", "api", "degree", "combined"]
D = 1024
ITERS = 2
REPEATS = 2


def colors(col, nl, su, pr, opmap):
    if col in ("identifier", "degree"):
        lab = {}
        for n, r in nl.items():
            lab[n] = G.canonical_label(r, "fcg")
        # drop library nodes, unless that would drop everything
        drop = []
        for n, l in lab.items():
            if l.startswith(G.LIB_PREFIXES):
                drop.append(n)
        if drop and len(drop) < len(lab):
            for n in drop:
                del lab[n]
        if col == "identifier":
            return lab
        # degree colour: in/out degree within the kept graph
        keep = set(lab)
        degree_cols = {}
        for n in lab:
            n_in = 0
            if n in pr:
                for p in pr[n]:
                    if p in keep:
                        n_in += 1
            n_out = 0
            if n in su:
                for s in su[n]:
                    if s in keep:
                        n_out += 1
            degree_cols[n] = f"d{n_in}_{n_out}"
        return degree_cols
    if col == "combined" and opmap is None:
        return None
    return HD.build_colors(col, nl, su, "fcg", opmap)


def rounds(cols, su, pr):
    """Raw signed token vectors per WL round, tokens and refinement exactly as the generators."""
    out = np.zeros((ITERS + 1, D), dtype=np.float32)

    def add(r, tok):
        h = G.stable_hash_u32(tok)
        # signed hashing: the top bit picks the sign
        if (h & 0x80000000) == 0:
            out[r, h % D] += 1.0
        else:
            out[r, h % D] += -1.0

    for c in cols.values():
        add(0, f"0|{c}")
    cur = cols
    for i in range(1, ITERS + 1):
        new = {}
        for n in cur:
            inl = []
            if n in pr:
                for p in pr[n]:
                    if p in cur:
                        inl.append(cur[p])
            inl = sorted(inl)
            outl = []
            if n in su:
                for s in su[n]:
                    if s in cur:
                        outl.append(cur[s])
            outl = sorted(outl)
            new[n] = wl_digest(cur[n], inl, outl)
        cur = new
        for c in cur.values():
            add(i, f"{i}|{c}")
    return out


def _init():
    H._zip = None


def _job(t):
    fam, sha = t
    dst = os.path.join(CACHE, f"{sha}.npz")
    if os.path.exists(dst):
        return "cached"
    g = H.load_graph(fam, sha, "fcg")
    if g is None:
        return "missing"
    nl, su, pr = g
    op = H.load_opcodes(fam, sha)
    res = {}
    for col in COLS:
        c = colors(col, nl, su, pr, op)
        if c:
            res[col] = rounds(c, su, pr)
    np.savez_compressed(dst + ".tmp.npz", **res)
    os.replace(dst + ".tmp.npz", dst)
    return "ok"


def vectorize(workers):
    ids, fam = H.sample_ids("four")
    jobs = []
    for f, s in zip(fam, ids):
        if not os.path.exists(os.path.join(CACHE, f"{s}.npz")):
            jobs.append((f, s))
    print(f"{len(jobs)} apps to vectorize ({len(ids)} total)", flush=True)
    counts = {}
    with get_context("fork").Pool(workers, initializer=_init, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            if r in counts:
                counts[r] = counts[r] + 1
            else:
                counts[r] = 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} {counts}", flush=True)
    print(f"done {counts}", flush=True)


def evaluate():
    import exp_l1_ladder as L
    from sklearn.metrics import f1_score
    ids, fam = H.sample_ids("four")
    cached = []
    for s in ids:
        cached.append(os.path.exists(os.path.join(CACHE, f"{s}.npz")))
    keep = np.array(cached)
    ids = ids[keep]
    fam = fam[keep]
    y, _ = H.labels(fam)
    g = H.groups_for(ids)
    R = []
    for s in ids:
        R.append(np.load(os.path.join(CACHE, f"{s}.npz")))
    # check h = 2 sums against the packed identifier vectors (first 200 apps)
    idx, _, XP = H._load("fcg_wl_1024")
    chk = []
    for j in range(min(200, len(ids), len(R))):
        s = ids[j]
        z = R[j]
        v = z["identifier"].sum(0)
        p = XP[idx[s]]
        chk.append(float(v @ p / (np.linalg.norm(v) * np.linalg.norm(p) + 1e-12)))
    res = {"dataset": H.DATASET, "groups": H.GROUPS, "n": int(len(ids)), "classes": int(len(set(fam))),
           "packed_identifier_cos_min": float(np.min(chk))}
    print(res, flush=True)
    F = H.folds(y, g, REPEATS)
    for col in COLS:
        has_list = []
        for z in R:
            has_list.append(col in z.files)
        has = np.array(has_list)
        if has.mean() < 0.99:  # more than 1% missing: skip rather than change the sample materially
            res[col] = {"skipped": f"{int((~has).sum())} apps lack this coloring"}
            continue
        yc = y
        gc = g
        Fc = F
        if not has.all():  # a handful missing (e.g. no opcode map): drop them, refold, and record it
            if "dropped" not in res:
                res["dropped"] = {}
            res["dropped"][col] = int((~has).sum())
            yc = y[has]
            gc = g[has]
            Fc = H.folds(yc, gc, REPEATS)
        stacked = []
        for z, k in zip(R, has):
            if k:
                stacked.append(z[col])
        A = np.stack(stacked)  # n x rounds x d
        del stacked  # A is a copy; release the per-app arrays
        f1 = {}
        for h in range(ITERS + 1):
            X = A[:, :h + 1].sum(1)
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            f1[h] = []
            for r, k, tr, te in Fc:
                m, _ = L.fit_linear(X, yc, gc, tr, 1000 * r + k)
                f1[h].append(f1_score(yc[te], m.predict(X[te]), average="macro"))
        md, t, p = H.corrected_t(np.array(f1[2]) - np.array(f1[0]), len(Fc[0][2]), len(Fc[0][3]))
        res[col] = {}
        for h, v in f1.items():
            res[col][f"h{h}"] = float(np.mean(v))
        res[col]["gain_h2_minus_h0"] = md
        res[col]["p"] = p
        print(col, res[col], flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "vectorize":
        if len(sys.argv) > 2:
            workers = int(sys.argv[2])
        else:
            workers = 3
        vectorize(workers)
    else:
        evaluate()
