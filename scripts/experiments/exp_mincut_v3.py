#!/usr/bin/env python3
"""Host-to-SDK minimum edge cut and reflection scan on corpus v3 (mincut_sdk_analysis.py and
reflection_analysis.py on the eight classes, from one load of each FCG).

Each FCG is a flow network: unit-capacity call edges, a super-source joined to the host's lifecycle entry
methods (onCreate / attachBaseContext / onStart / onResume of classes outside the SDK), and a super-sink
joined from every SDK method (exp_label_audit.SDK, the namespaces established by the reverse engineering).
The minimum directed edge cut between them is the number of call edges that must be removed to separate
the host's entry points from the SDK. An application's boundary is unambiguous when it has at least one host
entry method and one SDK method; the others are counted but not cut. Up to PER_FAM applications per class
(seed 42), graphs from exp_harness.load_graph. Reported per class: number analysed and unambiguous, median and
quartiles of the cut, median SDK partition size (methods), and the share with cut 0 (SDK unreachable from
the host's lifecycle entry points).
Reflection scan: per application, the call edges from SDK methods and from host methods to reflection
(Class.forName, getMethod, getDeclaredMethod, Method.invoke, Class.newInstance, getDeclaredConstructor,
Constructor.newInstance, ClassLoader.loadClass), dynamic class loading (constructing a DexClassLoader,
InMemoryDexClassLoader, PathClassLoader or BaseDexClassLoader) and native loading (System.load, System.loadLibrary,
Runtime.load/loadLibrary) in the extracted call graph; reported as the share of applications whose SDK
methods call each group, and the median number of such call edges.
Topology: per application, weakly connected components of the SDK partition, its largest strongly connected
component, and the share of FCG nodes with exactly one in-edge and one out-edge.
Output: results/datasets/proxy_v3/mincut_v3.json (resumable)
"""
import json
import os
import sys
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G

OUT = os.environ.get("MC_OUT") or os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "mincut_v3.json")
PER_FAM = int(os.environ.get("MC_PER_FAM", "60"))
SEED = 42
ENTRY = (";->onCreate(", ";->attachBaseContext(", ";->onStart(", ";->onResume(")
API = {"reflection": ("Ljava/lang/Class;->forName(", "Ljava/lang/Class;->getMethod(", "Ljava/lang/Class;->getDeclaredMethod(",
                     "Ljava/lang/reflect/Method;->invoke(", "Ljava/lang/Class;->newInstance(",
                     "Ljava/lang/Class;->getDeclaredConstructor(", "Ljava/lang/reflect/Constructor;->newInstance(",
                     "Ljava/lang/ClassLoader;->loadClass("),  # loads a named class from an existing loader: reflection
       # dynamic loading = constructing a DEX class loader (new code from a file or buffer)
       "dex_loading": ("Ldalvik/system/DexClassLoader;", "Ldalvik/system/InMemoryDexClassLoader;",
                       "Ldalvik/system/PathClassLoader;", "Ldalvik/system/BaseDexClassLoader;"),
       "native_loading": ("Ljava/lang/System;->load(", "Ljava/lang/System;->loadLibrary(", "Ljava/lang/Runtime;->load(",
                          "Ljava/lang/Runtime;->loadLibrary(")}


def cut(nl, su, sdk):
    # SDK methods are the sinks, host lifecycle entry methods the sources
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    sinks = []
    for n, l in lab.items():
        if l.startswith(sdk):
            sinks.append(n)
    srcs = []
    for n, l in lab.items():
        if l.startswith(sdk):
            continue
        is_entry = False
        for e in ENTRY:
            if e in l:
                is_entry = True
                break
        if is_entry:
            srcs.append(n)
    out = {"nodes": len(lab), "sdk_methods": len(sinks), "host_entries": len(srcs)}
    isdk = set(sinks)
    # API group of each callee, computed once per node; then one pass over the call edges
    grp = {}
    for n, l in lab.items():
        for k, pats in API.items():
            matched = False
            for p in pats:
                if l.startswith(p) or p in l:
                    matched = True
                    break
            if matched:
                grp[n] = k
                break
    for side in ("sdk", "host"):
        for k in API:
            out[f"{side}_{k}"] = 0
    for u, vs in su.items():
        if u not in lab:
            continue
        if u in isdk:
            side = "sdk"
        else:
            side = "host"
        for v in vs:
            k = grp.get(v)
            if k is not None:
                out[f"{side}_{k}"] += 1
    # SDK subgraph topology: weakly connected components of the SDK partition, its largest strongly connected
    # component, and the share of all FCG nodes that are pass-through (exactly one in-edge and one out-edge)
    if sinks:
        sidx = {}
        for k, n in enumerate(sinks):
            sidx[n] = k
        se = set()
        for u in sinks:
            for v in su.get(u, []):
                if v in sidx and v != u:
                    se.add((sidx[u], sidx[v]))
        se_rows = []
        se_cols = []
        for a, b in se:
            se_rows.append(a)
            se_cols.append(b)
        Ms = csr_matrix((np.ones(len(se), dtype=np.int8), (se_rows, se_cols)),
                        shape=(len(sinks), len(sinks)))
        del se_rows, se_cols
        out["sdk_wcc"] = int(connected_components(Ms, directed=True, connection="weak")[0])
        _, scc = connected_components(Ms, directed=True, connection="strong")
        out["sdk_largest_scc"] = int(np.bincount(scc).max())
    indeg = defaultdict(int)
    for u, vs in su.items():
        if u in lab:
            for v in set(vs):
                if v in lab and v != u:
                    indeg[v] += 1
    outdeg = {}
    for u, vs in su.items():
        if u not in lab:
            continue
        targets = set()
        for v in vs:
            if v in lab and v != u:
                targets.add(v)
        outdeg[u] = len(targets)
    n_pass = 0
    for n in lab:
        if indeg.get(n, 0) == 1 and outdeg.get(n, 0) == 1:
            n_pass += 1
    out["pass_through_share"] = n_pass / max(1, len(lab))
    if not sinks or not srcs:
        return out
    # unit-capacity call edges (duplicates collapsed, self-loops dropped) plus a super-source S and a
    # super-sink T with capacity above any possible cut; scipy's CSR max-flow instead of networkx, whose
    # dict-per-edge graph needs several GB for the largest FCGs
    idx = {}
    for k, n in enumerate(lab):
        idx[n] = k
    S = len(idx)
    T = len(idx) + 1
    big = 0
    for vs in su.values():
        big += len(vs)
    big += 1
    e = set()
    for u, vs in su.items():
        if u not in idx:
            continue
        for v in vs:
            if v in idx and v != u:
                e.add((idx[u], idx[v]))
    # edge list: call edges, then S -> each host entry, then each SDK method -> T
    rows = []
    cols = []
    for a, b in e:
        rows.append(a)
        cols.append(b)
    for x in srcs:
        rows.append(S)
        cols.append(idx[x])
    for t in sinks:
        rows.append(idx[t])
        cols.append(T)
    cap = np.concatenate([np.ones(len(e), dtype=np.int32), np.full(len(srcs) + len(sinks), min(big, 2**31 - 1), dtype=np.int32)])
    del e
    M = csr_matrix((cap, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))), shape=(T + 1, T + 1), dtype=np.int32)
    M.sum_duplicates()
    out["cut"] = int(maximum_flow(M, S, T).flow_value)
    return out


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    rng = np.random.default_rng(SEED)
    man = []
    for l in open(os.path.join(H.ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    by = defaultdict(list)
    for m in sorted(man, key=lambda m: m["sha256"]):
        if m["status"] == "kept" and m.get("split") == "class":
            by[m["family"]].append(m["sha256"])
    # resume from the previous output if present
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {"apps": {}}
    for f in A.FAMS_V3:
        lst = by[f][:]
        rng.shuffle(lst)
        for sha in lst[:PER_FAM]:
            if sha in res["apps"]:
                continue
            g = H.load_graph(f, sha, "fcg")
            if g is None or not g[0]:
                continue
            r = cut(g[0], g[1], A.SDK[f])
            r["family"] = f
            res["apps"][sha] = r
            del g
            json.dump(res, open(OUT, "w"))
        print(f"[{f}] done", flush=True)
    # per-class summary (keys in the same order as before)
    summ = {}
    for f in A.FAMS_V3:
        rows = []
        for r in res["apps"].values():
            if r["family"] == f:
                rows.append(r)
        un = []
        for r in rows:
            if "cut" in r:
                un.append(r)
        if un:
            cut_list = []
            for r in un:
                cut_list.append(r["cut"])
            c = np.array(cut_list)
        else:
            c = np.array([])
        sf = {"analysed": len(rows), "unambiguous": len(un)}
        if len(c):
            sf["cut_median"] = float(np.median(c))
            sf["cut_q1_q3"] = [float(np.percentile(c, 25)), float(np.percentile(c, 75))]
            sf["cut_zero_share"] = float(np.mean(c == 0))
        else:
            sf["cut_median"] = None
            sf["cut_q1_q3"] = None
            sf["cut_zero_share"] = None
        if un:
            sdk_sizes = []
            for r in un:
                sdk_sizes.append(r["sdk_methods"])
            sf["sdk_methods_median"] = float(np.median(sdk_sizes))
        else:
            sf["sdk_methods_median"] = None
        for side in ("sdk", "host"):
            for k in API:
                if rows:
                    has_edge = []
                    for r in rows:
                        has_edge.append(r[f"{side}_{k}"] > 0)
                    sf[f"{side}_{k}_share"] = float(np.mean(has_edge))
                else:
                    sf[f"{side}_{k}_share"] = None
        for k in API:
            if rows:
                n_edges = []
                for r in rows:
                    n_edges.append(r[f"sdk_{k}"])
                sf[f"sdk_{k}_median_edges"] = float(np.median(n_edges))
            else:
                sf[f"sdk_{k}_median_edges"] = None
        if rows:
            wccs = []
            for r in rows:
                if "sdk_wcc" in r:
                    wccs.append(r["sdk_wcc"])
            sf["sdk_wcc_median"] = float(np.median(wccs))
        else:
            sf["sdk_wcc_median"] = None
        sccs = []
        for r in rows:
            if "sdk_largest_scc" in r:
                sccs.append(r["sdk_largest_scc"])
        if sccs:
            sf["sdk_largest_scc_max"] = int(max(sccs))
        else:
            sf["sdk_largest_scc_max"] = int(0)
        if rows:
            shares = []
            for r in rows:
                shares.append(r["pass_through_share"])
            sf["pass_through_median"] = float(np.median(shares))
        else:
            sf["pass_through_median"] = None
        summ[f] = sf
        print(f, summ[f], flush=True)
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
