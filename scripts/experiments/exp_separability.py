#!/usr/bin/env python3
"""Structural separability of the payload from its host, per staged payload application, for any
dataset (DATASET env; STAGED_SET is forced to payload).

Payload methods: FCG nodes matching the dataset's payload prefixes (proxy SDK namespaces, MalNet
ad-SDK namespaces, piggyback rider packages). Host methods: all other non-library nodes.
Measures:
  rho          payload methods / non-library methods (occupancy)
  boundary     call edges between host and payload, in each direction, and per payload method
  min_cut      directed minimum edge cut from host roots (host methods without callers in the
               non-library graph) to the payload, unit capacities (scipy maximum_flow); the
               manifest entry points of the earlier analysis are not available for every dataset,
               so roots stand in for them on all datasets alike
Output: tier4/v4/separability.json (dataset-aware), one row per app plus per-family medians.
"""
import json
import os
import sys

os.environ["STAGED_SET"] = "payload"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import gen_wl_vectors as G
import provenance_localization as P

OUT = H.out_path("tier4", "v4", "separability.json")


def measure(fam, sha):
    g = H.load_graph(fam, sha, "fcg")
    if g is None:
        return None
    nl, su, pr = g
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    # non-library methods
    keep = []
    for n, l in lab.items():
        if not l.startswith(G.LIB_PREFIXES):
            keep.append(n)
    pref = tuple(P.SDK_PREFIXES.get(fam, ()))
    if not pref:
        return None
    pay = set()
    for n in keep:
        if lab[n].startswith(pref):
            pay.add(n)
    if not pay:
        return {"family": fam, "sha": sha, "n": len(keep), "payload": 0}
    ks = set(keep)
    idx = {}
    for i, n in enumerate(keep):
        idx[n] = i
    # call edges among non-library methods, and boundary counts
    e_hp = 0
    e_ph = 0
    rows = []
    cols = []
    has_caller = set()
    for u in keep:
        if u in su:
            callees = su[u]
        else:
            callees = []
        for v in callees:
            if v not in ks or v == u:
                continue
            has_caller.add(v)
            rows.append(idx[u])
            cols.append(idx[v])
            if u in pay and v not in pay:
                e_ph += 1
            elif u not in pay and v in pay:
                e_hp += 1
    roots = []
    for n in keep:
        if n not in pay and n not in has_caller:
            roots.append(n)
    # min cut: super source -> roots (inf), payload -> super sink (inf), call edges unit capacity
    import scipy.sparse as sp
    from scipy.sparse.csgraph import maximum_flow
    N = len(keep)
    s = N
    t = N + 1
    BIG = 10 ** 7
    # edge lists: call edges (capacity 1), then source -> roots, then payload -> sink
    r = list(rows)
    c = list(cols)
    w = []
    for _ in rows:
        w.append(1)
    for n in roots:
        r.append(s)
        c.append(idx[n])
        w.append(BIG)
    for n in pay:
        r.append(idx[n])
        c.append(t)
        w.append(BIG)
    A = sp.csr_matrix((np.array(w, dtype=np.int32), (np.array(r), np.array(c))), shape=(N + 2, N + 2))
    A.sum_duplicates()
    flow = int(maximum_flow(A, s, t).flow_value)
    if flow < BIG:
        min_cut = flow
    else:
        min_cut = None
    return {"family": fam, "sha": sha, "n": N, "payload": len(pay), "rho": len(pay) / max(N, 1),
            "edges_host_to_payload": e_hp, "edges_payload_to_host": e_ph,
            "boundary_per_payload_method": (e_hp + e_ph) / len(pay), "host_roots": len(roots),
            "min_cut": min_cut}


def main():
    ids, fam = H.sample_ids("four")
    pos = set(ids.tolist())
    apps = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos:
                apps.append((f, s))
    rows = []
    for i, (f, s) in enumerate(apps):
        m = measure(f, s)
        if m:
            rows.append(m)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(apps)}", flush=True)
    summ = {}
    families = set()
    for r in rows:
        families.add(r["family"])
    for f in sorted(families):
        rr = []
        for r in rows:
            if r["family"] == f and r["payload"] > 0:
                rr.append(r)

        def med(k):
            # median of field k over this family's payload apps (None if there are none)
            if not rr:
                return None
            vals = []
            for row in rr:
                if row.get(k) is not None:
                    vals.append(row[k])
            return float(np.median(vals))
        summ[f] = {"n": len(rr), "payload_methods": med("payload"), "rho": med("rho"), "min_cut": med("min_cut"),
                   "boundary_per_payload_method": med("boundary_per_payload_method"), "nodes": med("n")}
        print(f, summ[f], flush=True)
    json.dump({"dataset": H.DATASET, "summary": summ, "rows": rows}, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
