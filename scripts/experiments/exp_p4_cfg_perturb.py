#!/usr/bin/env python3
"""Structural perturbation of CFGs (robustness paper, CFG extension; FCG
versions are exp_p3_thm_bounds, exp_p3_wrapper_bound and exp_r4_perturb).

CFG edges are intra-method only, so semantics-preserving CFG edits act inside a
method: a forwarding (goto) block inserted on an edge, a two-block indirection,
a dead block hanging off an existing block, and an opaque-predicate (phantom)
edge between blocks of the same method. Inserted blocks carry the label of the
method they are inserted into. For n_ops in {1, 10, 50} (bounds) and
{10, 50, 200} (flips), in the SDK or the host, under the identifier, fuzzy and
degree colorings:
  L, l1, l2       unit token events and raw-vector change (d = 1,024)
  bounds          classical 2(h+1) B Delta^h (global and local Delta);
                  Theorem wrapper 7 + 2(deg u + deg v) on single block wrappers
  flips           decision changes of the CFG-only and the deployed CFG+FCG
                  linear classifiers (identifier), FCG unchanged.
Output: results/tier4/p4/cfg_perturb.json
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# this experiment compares the CFG-only model with the deployed CFG+FCG model, so it always needs both graph
# blocks ([cfg | fcg]); datasets whose default is FCG-only get GRAPHS=cfg,fcg (no change for the proxy corpus)
os.environ.setdefault("GRAPHS", "cfg,fcg")
import exp_harness as H
import exp_p3_thm_bounds as T
import gen_wl_vectors as G
import gen_wl_hardened as HD
import graph_perturb as GP
import provenance_localization as P
import graph_io as GI

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "p4", "cfg_perturb.json")
OPS = {"wrapper": 2, "indirection": 3, "dead_node": 2, "phantom_edge": 1}
COLOURS = ["identifier", "fuzzy", "structure"]
D = 1024
_MODELS = {}


def _init(models):
    _MODELS.update(models)


def colours(colour, nl, su, pr):
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "cfg")
    if colour == "identifier":
        # drop library blocks unless that would drop every block
        keep = {}
        for n, l in lab.items():
            if not l.startswith(G.LIB_PREFIXES):
                keep[n] = l
        if keep:
            return keep
        return lab
    if colour == "structure":
        degree_cols = {}
        for n in nl:
            degree_cols[n] = f"d{len(pr.get(n, []))}_{len(su.get(n, []))}"
        return degree_cols
    fuzzy_cols = {}
    for n in lab:
        fuzzy_cols[n] = HD.fuzzy_label(lab[n])
    return fuzzy_cols


def apply_cfg(nl, su, pr, prefixes, op, loc, k, seed):
    """graph_perturb on a CFG, then give inserted blocks their method's label and
    keep phantom edges inside one method."""
    if op == "phantom_edge":
        L = dict(nl)
        S = {}
        for a, b in su.items():
            S[a] = list(b)
        Pp = {}
        for a, b in pr.items():
            Pp[a] = list(b)
        rng_seed = hashlib.sha1(f"{op}|{loc}|{k}|{seed}".encode()).hexdigest()
        rng = random.Random(rng_seed)
        part = GP.partition(L, prefixes)[loc]
        # blocks grouped by method label; only methods with 2+ blocks can take a phantom edge
        by_m = {}
        for n in part:
            if L[n] not in by_m:
                by_m[L[n]] = []
            by_m[L[n]].append(n)
        multi = []
        for v in by_m.values():
            if len(v) > 1:
                multi.append(v)
        added = 0
        tries = 0
        while added < k and tries < 20 * k and multi:
            grp = rng.choice(multi)
            u, v = rng.sample(grp, 2)
            tries += 1
            if v not in S.get(u, []):
                if u not in S:
                    S[u] = []
                S[u].append(v)
                if v not in Pp:
                    Pp[v] = []
                Pp[v].append(u)
                added += 1
        return L, S, Pp
    L, S, Pp = GP.apply(nl, su, pr, prefixes, op, loc, k, seed)
    # inserted blocks take the label of the nearest original block, following predecessors
    inserted = []
    for n in L:
        if n not in nl:
            inserted.append(n)
    for n in inserted:
        nb = Pp.get(n, [])
        if not nb:
            nb = S.get(n, [])
        while nb and nb[0] not in nl and Pp.get(nb[0]):
            nb = Pp.get(nb[0])
        if nb and nb[0] in nl:
            L[n] = nl[nb[0]]
    return L, S, Pp


def run_app(args):
    fam, sha, c = args
    Wc, bc, Wd, bd = _MODELS[sha]
    d = os.path.join(GRAPHS, fam, sha)
    nl, su, pr = G.parse_gml_lightweight(os.path.join(d, f"{sha}.cfg"))
    su = dict(su)
    pr = dict(pr)
    fcg_nl, fcg_su, fcg_pr = G.parse_gml_lightweight(os.path.join(d, f"{sha}.fcg"))
    fcg_vec = G.wl_hashed_vector(fcg_nl, fcg_su, fcg_pr, "fcg", D, iters=2,
                                 drop_lib=True, use_direction=True)
    del fcg_nl, fcg_su, fcg_pr  # the FCG is only needed for its vector
    degrees = []
    for n in nl:
        degrees.append(len(su.get(n, [])) + len(pr.get(n, [])))
    dmax = max(degrees)
    base = {}
    for cl in COLOURS:
        base[cl] = T.wl_rounds(colours(cl, nl, su, pr), su, pr)
    v0 = G.wl_hashed_vector(nl, su, pr, "cfg", D, iters=2, drop_lib=True, use_direction=True)
    p0c = int(np.argmax(Wc @ v0 + bc))
    p0d = int(np.argmax(Wd @ np.concatenate([v0, fcg_vec]) + bd))
    rows = []
    flips = []
    wraps = []
    for op, cost in OPS.items():
        for loc in ("host", "sdk"):
            for k in (1, 10, 50, 200):
                L1, S1, P1 = apply_cfg(nl, su, pr, P.SDK_PREFIXES.get(fam, ()), op, loc, k, 0)
                # token-event counts and bounds for the small budgets
                if k <= 50:
                    t = T.touched(nl, su, L1, S1)
                    if t:
                        # 2-hop neighbourhood of the touched blocks
                        nb = set(t)
                        for _ in range(2):
                            reached = set()
                            for node in list(nb):
                                for nbr in S1.get(node, []) + P1.get(node, []):
                                    reached.add(nbr)
                            nb |= reached
                        local_degrees = []
                        for n in nb:
                            local_degrees.append(len(S1.get(n, [])) + len(P1.get(n, [])))
                        dloc = max(local_degrees)
                        for cl in COLOURS:
                            r0, t0 = base[cl]
                            r1, t1 = T.wl_rounds(colours(cl, L1, S1, P1), S1, P1)
                            sym_diff = (t0 - t1) + (t1 - t0)
                            Lev = sum(sym_diff.values())
                            dv = T.hashed(t1) - T.hashed(t0)
                            rows.append({"op": op, "locale": loc, "n_ops": k, "colour": cl, "L": int(Lev),
                                         "l2": float(np.linalg.norm(dv)), "bound_global": 2 * 3 * cost * k * dmax ** 2,
                                         "bound_local": 2 * 3 * cost * k * dloc ** 2})
                # decision flips for the larger budgets
                if k >= 10:
                    v1 = G.wl_hashed_vector(L1, S1, P1, "cfg", D, iters=2, drop_lib=True, use_direction=True)
                    pred_cfg = int(np.argmax(Wc @ v1 + bc))
                    pred_deployed = int(np.argmax(Wd @ np.concatenate([v1, fcg_vec]) + bd))
                    flips.append({"op": op, "locale": loc, "n_ops": k,
                                  "flip_cfg": pred_cfg != p0c,
                                  "flip_deployed": pred_deployed != p0d})
    # single block wrappers for the corrected wrapper bound
    rng = random.Random(sha)
    edges = []
    for u, vs in su.items():
        for v in vs:
            if u != v:
                edges.append((u, v))
    for u, v in rng.sample(edges, min(25, len(edges))):
        S = {}
        for a, b in su.items():
            S[a] = list(b)
        Pp = {}
        for a, b in pr.items():
            Pp[a] = list(b)
        L = dict(nl)
        # wrapper block w on edge u -> v
        w = -1
        L[w] = nl[u]
        S[u].remove(v)
        Pp[v].remove(u)
        if u not in S:
            S[u] = []
        S[u].append(w)
        Pp[w] = [u]
        S[w] = [v]
        Pp[v].append(w)
        for cl in COLOURS:
            cols = colours(cl, L, S, Pp)
            if u not in cols or v not in cols:
                continue
            t1 = T.wl_rounds(cols, S, Pp)[1]
            sym_diff = (base[cl][1] - t1) + (t1 - base[cl][1])
            Lev = sum(sym_diff.values())
            keep = base[cl][0][0]
            du = 0
            for x in su.get(u, []) + pr.get(u, []):
                if x in keep:
                    du += 1
            dv_ = 0
            for x in su.get(v, []) + pr.get(v, []):
                if x in keep:
                    dv_ += 1
            wraps.append({"colour": cl, "L": int(Lev), "bound": 7 + 2 * (du + dv_)})
    return {"rows": rows, "flips": flips, "wraps": wraps, "dmax": dmax}


def main():
    ids, fam = H.sample_ids("four")
    ids, fam = H.with_cfg_fcg(ids, fam)
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    Xc = H.matrix("labeled", ids)[:, :D]          # cfg_wl_1024 is the first block of 'labeled'
    Xd = H.matrix("labeled", ids)                  # deployed CFG+FCG
    # the five smallest-CFG staged apps per provider
    apps = []
    for f in H.PROXY:
        dd = os.path.join(GRAPHS, f)
        cand = []
        for s in H.staged(f):
            if s in fold_of:
                cand.append(s)
        cfg_size = {}
        for s in cand:
            cfg_size[s] = GI.size(os.path.join(dd, s, f"{s}.cfg"))
        shas = sorted(cand, key=lambda s: cfg_size[s])
        for s in shas[:5]:
            apps.append((f, s))
    # one CFG-only and one deployed model per fold that holds a selected app
    models = {}
    for f, sha in apps:
        k0 = fold_of[sha]
        if k0 not in models:
            _, _, tr, _ = F[k0]
            up = H.upsample(tr, y, k0)
            mc = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            mc.fit(Xc[up], y[up])
            md = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            md.fit(Xd[up], y[up])
            models[k0] = (mc.coef_, mc.intercept_, md.coef_, md.intercept_)
        _MODELS[sha] = models[k0]
    out = {"rows": [], "flips": [], "wraps": [], "dmax": []}
    tasks = []
    for f, s in apps:
        tasks.append((f, s, int(le.transform([f])[0])))
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3)), initializer=_init, initargs=(dict(_MODELS),)) as ex:
        for r in ex.map(run_app, tasks):
            for k in ("rows", "flips", "wraps"):
                out[k] += r[k]
            out["dmax"].append(r["dmax"])
            print("app done", len(out["rows"]), flush=True)
    # summaries per colour and op
    summ = {"dmax_median": float(np.median(out["dmax"]))}
    for cl in COLOURS:
        for op in OPS:
            rr = []
            for r in out["rows"]:
                if r["colour"] == cl and r["op"] == op:
                    rr.append(r)
            if rr:
                per_op = []
                holds = []
                slack = []
                for r in rr:
                    per_op.append(r["L"] / r["n_ops"])
                    holds.append(r["L"] <= r["bound_global"])
                    slack.append(np.log10(r["bound_local"] / max(r["L"], 1)))
                summ[f"{cl}|{op}"] = {"n": len(rr), "L_per_op_median": float(np.median(per_op)),
                                      "bound_holds": float(np.mean(holds)),
                                      "log10_slack_local": float(np.median(slack))}
        ww = []
        for w in out["wraps"]:
            if w["colour"] == cl:
                ww.append(w)
        if ww:
            w_holds = []
            w_tight = []
            w_old = []
            for w in ww:
                w_holds.append(w["L"] <= w["bound"])
                w_tight.append(w["L"] == w["bound"])
                w_old.append(w["L"] <= 3)
            summ[f"wrapper|{cl}"] = {"n": len(ww), "holds": float(np.mean(w_holds)),
                                     "tight": float(np.mean(w_tight)),
                                     "old_thm3_holds": float(np.mean(w_old))}
    fl = out["flips"]
    flip_cfg = []
    flip_deployed = []
    for x in fl:
        flip_cfg.append(x["flip_cfg"])
        flip_deployed.append(x["flip_deployed"])
    summ["flip_cfg"] = float(np.mean(flip_cfg))
    summ["flip_deployed"] = float(np.mean(flip_deployed))
    summ["flip_n"] = len(fl)
    out["summary"] = summ
    json.dump(out, open(OUT, "w"))
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
