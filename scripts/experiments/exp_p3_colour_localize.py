#!/usr/bin/env python3
"""Localization and identifiability across initial colourings.

The localization results of the prior papers and of X2/E5a use the identifier
colouring only. This repeats the exact Proposition-1 decomposition for the
renaming-invariant colourings, clean and under per-app identifier renaming, at
d = 2^10, 2^16, 2^20, on the staged proxy apps (5-fold DEX-grouped linear SGD per
d, as in exp_x2_highdim). Methods are always identified by their original
signature for scoring, and only non-library methods are ranked, so base rates are
identical across colourings; what changes is which tokens a method emits.

Also per colouring and app: granularity = fraction of SDK methods whose initial
colour also occurs on a host method (such methods emit an identical round-0 token
and cannot be told apart by it), and the number of distinct SDK round-0 colours.

Output: results/tier4/p3/colour_localize.json
"""
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD
import gen_wl_renamed as REN
import provenance_localization as P

GRAPHS = H.GRAPHS_DIR
P3 = H.out_dir("tier4", "p3")
TOKDIR = os.path.join(P3, "colour_tokens")
OUT = os.path.join(P3, "colour_localize.json")
COLOURS = ["labeled", "fuzzy", "api", "structure"]
DIMS = [2 ** 10, 2 ** 16, 2 ** 20]


def colour_map(colour, lab_all, su, pr):
    """lab_all: node -> canonical label (possibly renamed), all nodes."""
    cols = {}
    if colour == "labeled":
        for n, l in lab_all.items():
            if not l.startswith(G.LIB_PREFIXES):
                cols[n] = l
        return cols
    if colour == "structure":
        # in-degree / out-degree
        for n in lab_all:
            if n in pr:
                n_in = len(pr[n])
            else:
                n_in = 0
            if n in su:
                n_out = len(su[n])
            else:
                n_out = 0
            cols[n] = f"d{n_in}_{n_out}"
        return cols
    if colour == "fuzzy":
        for n in lab_all:
            cols[n] = HD.fuzzy_label(lab_all[n])
        return cols
    for n in lab_all:
        cols[n] = HD.api_label(n, lab_all, su)
    return cols


def wl_tokens(cols, su, pr):
    hs = []
    nodes = []
    for n, c in cols.items():
        hs.append(G.stable_hash_u32(f"0|{c}"))
        nodes.append(n)
    cur = cols
    for it in (1, 2):
        nxt = {}
        for n in cur:
            # sorted colours of in- and out-neighbours
            i_l = []
            if n in pr:
                for p in pr[n]:
                    if p in cur:
                        i_l.append(cur[p])
            i_l = sorted(i_l)
            o_l = []
            if n in su:
                for s in su[n]:
                    if s in cur:
                        o_l.append(cur[s])
            o_l = sorted(o_l)
            nxt[n] = wl_digest(cur[n], i_l, o_l)
        cur = nxt
        for n, c in cur.items():
            hs.append(G.stable_hash_u32(f"{it}|{c}"))
            nodes.append(n)
    return hs, nodes


def extract(args):
    fam, sha = args
    path = os.path.join(TOKDIR, f"{sha}.npz")
    if os.path.exists(path):
        return fam, sha, True
    nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    ren = {}
    for n, l in lab.items():
        ren[n] = REN.rename_label(l, sha[:16])
    # scored methods: non-library nodes, keyed by ORIGINAL signature
    dev = []
    for n, l in lab.items():
        if not l.startswith(G.LIB_PREFIXES):
            dev.append(n)
    mid = {}
    for i, n in enumerate(dev):
        mid[n] = i
    is_sdk = []
    for n in dev:
        is_sdk.append(lab[n].startswith(P.SDK_PREFIXES.get(fam, ())))
    truth = np.array(is_sdk)
    out = {"truth": truth}
    for cond, L in (("clean", lab), ("renamed", ren)):
        for c in COLOURS:
            cols = colour_map(c, L, su, pr)
            hs, nodes = wl_tokens(cols, su, pr)
            out[f"{c}|{cond}|h"] = np.array(hs, dtype=np.uint32)
            # token -> scored-method index (-1 for library nodes)
            method_of = []
            for n in nodes:
                if n in mid:
                    method_of.append(mid[n])
                else:
                    method_of.append(-1)
            out[f"{c}|{cond}|m"] = np.array(method_of, dtype=np.int32)
            if cond == "clean":
                sdk_cols = set()
                host_cols = set()
                sdk_nodes = []
                for n in dev:
                    if truth[mid[n]] and n in cols:
                        sdk_cols.add(cols[n])
                for n in dev:
                    if not truth[mid[n]] and n in cols:
                        host_cols.add(cols[n])
                for n in dev:
                    if truth[mid[n]] and n in cols:
                        sdk_nodes.append(n)
                if sdk_nodes:
                    in_host = []
                    for n in sdk_nodes:
                        in_host.append(cols[n] in host_cols)
                    shared = np.mean(in_host)
                else:
                    shared = np.nan
                out[f"{c}|gran"] = np.array([shared, len(sdk_cols), len(sdk_nodes)], dtype=float)
    os.makedirs(TOKDIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return fam, sha, True


def vec(h, d):
    v = np.zeros(d)
    buckets = (h % d).astype(np.int64)
    signs = np.where(h & 0x80000000, -1.0, 1.0)
    np.add.at(v, buckets, signs)
    n = np.linalg.norm(v)
    # (normalized vector, norm used for scaling; a zero norm reports 1.0)
    if n > 0:
        normed = v / n
    else:
        normed = v
    if n:
        scale = n
    else:
        scale = 1.0
    return normed, scale


def main():
    ids, fam = H.sample_ids("four")
    pos = set(ids)
    staged = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos:
                staged.append((f, s))
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for k, _ in enumerate(ex.map(extract, staged), 1):
            if k % 10 == 0:
                print(f"extracted {k}/{len(staged)}", flush=True)
    shas = []
    fam_list = []
    for f, s in staged:
        shas.append(s)
        fam_list.append(f)
    fams = np.array(fam_list)
    le = LabelEncoder().fit(fams)
    y = le.transform(fams)
    g = H.groups_for(np.array(shas))
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    # granularity
    gran = {}
    for c in COLOURS:
        per = defaultdict(list)
        for f, s in staged:
            z = np.load(os.path.join(TOKDIR, f"{s}.npz"))
            gr = z[f"{c}|gran"]
            if not np.isnan(gr[0]):
                per[f].append(gr)
        gran[c] = {}
        for f, vs in per.items():
            shared = []
            distinct = []
            n_sdk = []
            for v in vs:
                shared.append(v[0])
                distinct.append(v[1])
                n_sdk.append(v[2])
            gran[c][f] = {"sdk_colour_shared_with_host": float(np.mean(shared)),
                          "distinct_sdk_colours_median": float(np.median(distinct)),
                          "sdk_methods_median": float(np.median(n_sdk))}
    res["granularity"] = gran
    for c in COLOURS:
        for cond in ("clean", "renamed"):
            key = f"{c}|{cond}"
            if key in res:
                continue
            toks = {}
            for f, s in staged:
                z = np.load(os.path.join(TOKDIR, f"{s}.npz"))
                toks[s] = (z[f"{c}|{cond}|h"], z[f"{c}|{cond}|m"], z["truth"])
            res[key] = {}
            for d in DIMS:
                row_mats = []
                for s in shas:
                    v_norm, _ = vec(toks[s][0], d)
                    row_mats.append(sparse.csr_matrix(v_norm))
                X = sparse.vstack(row_mats).tocsr()
                del row_mats, v_norm  # X is a copy; release the per-app rows as the original temporary list did
                prec = defaultdict(list)
                f1s = []
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
                for tr, te in cv.split(np.zeros(len(y)), y, g):
                    up = H.upsample(tr, y, 0)
                    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-4,
                                      random_state=42).fit(X[up], y[up])
                    f1s.append(float((m.predict(X[te]) == y[te]).mean()))
                    for j in te:
                        h, mi, truth = toks[shas[j]]
                        if truth.sum() == 0:
                            continue
                        w = m.coef_[y[j]].astype(float)
                        _, nrm = vec(h, d)
                        sgn = np.where(h & 0x80000000, -1.0, 1.0)
                        keep = mi >= 0
                        credit = np.zeros(len(truth))
                        kept_buckets = (h[keep] % d).astype(np.int64)
                        np.add.at(credit, mi[keep], w[kept_buckets] * sgn[keep] / nrm)
                        nt = int(truth.sum())
                        # ties broken at random so shared-colour methods get no free credit
                        rng = np.random.default_rng(j)
                        order = np.lexsort((rng.random(len(credit)), -credit))
                        prec[fams[j]].append((truth[order[:nt]].mean(), nt / len(truth)))
                row = {"accuracy": float(np.mean(f1s))}
                for f in H.PROXY:
                    if f in prec and prec[f]:
                        pa = np.array(prec[f])
                        row[f] = {"p_at_target": float(pa[:, 0].mean()), "base_rate": float(pa[:, 1].mean()),
                                  "n": len(pa)}
                res[key][str(d)] = row
                print(key, d, json.dumps(row), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
