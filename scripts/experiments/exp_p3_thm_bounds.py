#!/usr/bin/env python3
"""Theorems thm:radius, thm:perturb and thm:wrapper against measured perturbations
on real FCGs (robustness paper, sec:method_certificates).

For staged proxy apps, apply graph_perturb operations (wrapper, indirection,
dead_node, phantom_edge) at n_ops = 1, 10, 50 random sites in the host or the SDK,
and for the labeled, fuzzy and structure (degree) colourings measure

  L          number of unit token events (multiset symmetric difference of the
             emitted tokens over rounds 0..2) = 2 L_changed + L_new + L_removed
  l1, l2     ||phi(G') - phi(G)|| of the raw hashed vectors (d = 1,024)
  A0, Ah     directly touched nodes, and nodes whose label changed at some round
  bounds     thm:perturb 2(h+1) B Delta^h with Delta the global maximum degree and
             with the maximum degree over the touched nodes' 2-hop neighbourhood;
             thm:radius |A0| Delta^h; thm:wrapper (h+1) per transparent wrapper
Budget B counts atomic edits: wrapper 2, indirection 3, dead_node 2, phantom_edge 1
per operation, as in the threat model.

Output: results/tier4/p3/thm_bounds.json
"""
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD
import graph_perturb as GP
import provenance_localization as P
import graph_io as GI

GRAPHS = H.GRAPHS_DIR
OUT = H.out_path("tier4", "p3", "thm_bounds.json")
OPS = {"wrapper": 2, "indirection": 3, "dead_node": 2, "phantom_edge": 1}
NOPS = [1, 10, 50]
LOCALES = ["host", "sdk"]
COLOURS = ["labeled", "fuzzy", "structure"]
PER_FAM = 5
HH = 2
D = 1024


def colours(colour, nl, su, pr):
    if colour == "labeled":
        # identifier labels, library methods dropped
        lab = {}
        for n, r in nl.items():
            lab[n] = G.canonical_label(r, "fcg")
        out = {}
        for n, l in lab.items():
            if not l.startswith(G.LIB_PREFIXES):
                out[n] = l
        return out
    if colour == "structure":
        out = {}
        for n in nl:
            out[n] = f"d{len(pr.get(n, []))}_{len(su.get(n, []))}"
        return out
    # fuzzy labels
    lbl = {}
    for n, r in nl.items():
        lbl[n] = G.canonical_label(r, "fcg")
    out = {}
    for n in lbl:
        out[n] = HD.fuzzy_label(lbl[n])
    return out


def wl_rounds(cols, su, pr):
    """[labels at round 0..h] and the emitted token multiset."""
    rounds = [cols]
    cur = cols
    for i in range(1, HH + 1):
        new = {}
        for n in cur:
            inl = []
            for p in pr.get(n, []):
                if p in cur:
                    inl.append(cur[p])
            inl.sort()
            out = []
            for s in su.get(n, []):
                if s in cur:
                    out.append(cur[s])
            out.sort()
            new[n] = wl_digest(cur[n], inl, out)
        cur = new
        rounds.append(cur)
    toks = Counter()
    for i, r in enumerate(rounds):
        for c in r.values():
            toks[f"{i}|{c}"] += 1
    return rounds, toks


def hashed(toks):
    v = np.zeros(D)
    for t, k in toks.items():
        h = G.stable_hash_u32(t)
        if (h & 0x80000000) == 0:
            sign = 1.0
        else:
            sign = -1.0
        v[h % D] += k * sign
    return v


def touched(L0, S0, L1, S1):
    new = set(L1) - set(L0)
    t = set(new)
    for u in set(S0) | set(S1):
        a = Counter(S0.get(u, []))
        b = Counter(S1.get(u, []))
        if a != b:
            t.add(u)
            # endpoints of added or removed edges
            removed_targets = set((a - b).keys())
            added_targets = set((b - a).keys())
            t |= removed_targets | added_targets
    return t


def run_app(args):
    # unperturbed graph, degrees, and base WL rounds per colouring
    fam, sha = args
    nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
    su = dict(su)
    pr = dict(pr)
    deg = {}
    for n in nl:
        deg[n] = len(su.get(n, [])) + len(pr.get(n, []))
    dmax = max(deg.values())
    base = {}
    for c in COLOURS:
        base[c] = wl_rounds(colours(c, nl, su, pr), su, pr)
    rows = []
    for op, cost in OPS.items():
        for loc in LOCALES:
            for k in NOPS:
                # perturb, then the touched set and its HH-hop neighbourhood
                L1, S1, P1 = GP.apply(nl, su, pr, P.SDK_PREFIXES.get(fam, ()), op, loc, k, 0)
                t = touched(nl, su, L1, S1)
                if not t:
                    continue
                nb = set(t)
                for _ in range(HH):
                    reached = set()
                    for u in list(nb):
                        for v in S1.get(u, []) + P1.get(u, []):
                            reached.add(v)
                    nb |= reached
                local_degrees = []
                for n in nb:
                    local_degrees.append(len(S1.get(n, [])) + len(P1.get(n, [])))
                dloc = max(local_degrees)
                B = cost * k
                # measured change vs the bounds, per colouring
                for c in COLOURS:
                    r0, t0 = base[c]
                    r1, t1 = wl_rounds(colours(c, L1, S1, P1), S1, P1)
                    sym_diff = (t0 - t1) + (t1 - t0)
                    Lev = sum(sym_diff.values())
                    dv = hashed(t1) - hashed(t0)
                    ah = set()
                    for n in r1[-1]:
                        changed = False
                        for i in range(HH + 1):
                            if n not in r0[i] or r0[i][n] != r1[i].get(n):
                                changed = True
                                break
                        if changed:
                            ah.add(n)
                    rows.append({"family": fam, "op": op, "locale": loc, "n_ops": k, "B": B, "colour": c,
                                 "L": int(Lev), "l1": float(np.abs(dv).sum()), "l2": float(np.linalg.norm(dv)),
                                 "A0": len(t), "Ah": len(ah), "dmax_global": dmax, "dmax_local": dloc,
                                 "bound_global": 2 * (HH + 1) * B * dmax ** HH,
                                 "bound_local": 2 * (HH + 1) * B * dloc ** HH,
                                 "radius_bound": len(t) * dloc ** HH})
    return rows


def main():
    # smallest staged graphs per family keep the run short
    apps = []
    for fam in H.PROXY:
        dd = os.path.join(GRAPHS, fam)
        shas = sorted(H.staged(fam), key=lambda s: GI.size(os.path.join(dd, s, f"{s}.fcg")))
        for s in shas[:PER_FAM]:
            apps.append((fam, s))
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for rr in ex.map(run_app, apps):
            rows += rr
            print(f"{len(rows)} rows", flush=True)

    # per colouring and operation: how often each bound holds, and its slack
    summ = {}
    for c in COLOURS:
        for op in OPS:
            rr = []
            for r in rows:
                if r["colour"] == c and r["op"] == op:
                    rr.append(r)
            if not rr:
                continue
            L_per_op = []
            l2_per_op = []
            l1_le_L = []
            L_le_global = []
            L_le_local = []
            slack_global = []
            slack_local = []
            Ah_le_radius = []
            for r in rr:
                L_per_op.append(r["L"] / r["n_ops"])
                l2_per_op.append(r["l2"] / r["n_ops"])
                l1_le_L.append(r["l1"] <= r["L"] + 1e-9)
                L_le_global.append(r["L"] <= r["bound_global"])
                L_le_local.append(r["L"] <= r["bound_local"])
                slack_global.append(np.log10(r["bound_global"] / max(r["L"], 1)))
                slack_local.append(np.log10(r["bound_local"] / max(r["L"], 1)))
                Ah_le_radius.append(r["Ah"] <= r["radius_bound"])
            if op == "wrapper":
                L_le_wrapper = []
                l2_le_wrapper = []
                for r in rr:
                    L_le_wrapper.append(r["L"] <= (HH + 1) * r["n_ops"])
                    l2_le_wrapper.append(r["l2"] <= (HH + 1) * r["n_ops"])
                wrapper_transparent_frac = float(np.mean(L_le_wrapper))
                l2_le_wrapper_bound_frac = float(np.mean(l2_le_wrapper))
            else:
                wrapper_transparent_frac = None
                l2_le_wrapper_bound_frac = None
            summ[f"{c}|{op}"] = {
                "n": len(rr),
                "L_per_op_median": float(np.median(L_per_op)),
                "l2_per_op_median": float(np.median(l2_per_op)),
                "l1_le_L_frac": float(np.mean(l1_le_L)),
                "L_le_bound_global_frac": float(np.mean(L_le_global)),
                "L_le_bound_local_frac": float(np.mean(L_le_local)),
                "log10_slack_global_median": float(np.median(slack_global)),
                "log10_slack_local_median": float(np.median(slack_local)),
                "Ah_le_radius_frac": float(np.mean(Ah_le_radius)),
                "wrapper_transparent_frac": wrapper_transparent_frac,
                "l2_le_wrapper_bound_frac": l2_le_wrapper_bound_frac,
            }
    dmax_values = []
    for r in rows:
        dmax_values.append(r["dmax_global"])
    res = {"apps": len(apps), "rows": rows, "summary": summ,
           "dmax_global_median": float(np.median(dmax_values))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
