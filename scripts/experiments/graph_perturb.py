#!/usr/bin/env python3
"""Semantics-preserving structural perturbations on lightweight FCGs, with locality
control, for robustness measurement (companion to the WL certificate paper).

A graph is (labels: id->sig, succ: id->list, pred: id->list). All operations here
preserve program behaviour in the problem space: they only add forwarding nodes,
never-executed methods, opaque-predicate edges, or duplicate leaves, and never
remove functional call relationships. This mirrors the add-only structural attack
of Zhao et al. (CCS 2021) and the wrapper/indirection operations of our threat
model. `flatten` is the extreme that collapses a chosen subgraph to one node; it
is NOT semantics-preserving and is included only as a structural-destruction bound.

Locality: nodes are partitioned into 'sdk' (label under the family SDK prefix),
'host' (developer, non-library, non-SDK) and 'lib' (framework/library, which the
WL vectorizer drops anyway). Each op is confined to a locale so we can ask whether
a perturbation inside the SDK subgraph moves the prediction more than one in the host.
"""
import copy
import hashlib
import random

LIB_PREFIXES = (
    "Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/", "Lkotlinx/",
    "Lcom/google/", "Lorg/apache/", "Lorg/json/", "Lorg/xml/", "Lorg/w3c/",
    "Lsun/", "Ldalvik/", "Ljunit/", "Lorg/junit/",
)


def _owner(sig):
    if sig:
        s = str(sig)
    else:
        s = ""
    if "->" in s:
        return s.split("->", 1)[0]
    return s


def is_lib(sig):
    return _owner(sig).startswith(LIB_PREFIXES)


def locale_of(sig, sdk_prefixes):
    o = _owner(sig)
    if o.startswith(tuple(sdk_prefixes)):
        return "sdk"
    if is_lib(sig):
        return "lib"
    return "host"


def partition(labels, sdk_prefixes):
    p = {"sdk": [], "host": [], "lib": []}
    for n, sig in labels.items():
        p[locale_of(sig, sdk_prefixes)].append(n)
    return p


def _fresh_label(rng, kind="stub"):
    # a fresh developer-owned method: obfuscator-style unique name, non-library so
    # the WL vectorizer keeps it; distinct per insertion so it never recurs.
    h = hashlib.sha1((kind + str(rng.getrandbits(64))).encode()).hexdigest()[:10]
    return f"Lp/{h};->{kind}()V"


def edges_in_locale(succ, labels, locale, sdk_prefixes, rng, want, cross=False):
    """Up to `want` shuffled existing edges (u,v): sdk<->host boundary edges if cross, else both ends in `locale`."""
    out = []
    items = list(succ.items())
    rng.shuffle(items)
    for u, vs in items:
        lu = locale_of(labels.get(u, ""), sdk_prefixes)
        for v in vs:
            lv = locale_of(labels.get(v, ""), sdk_prefixes)
            if cross:
                if (lu == "sdk" and lv == "host") or (lu == "host" and lv == "sdk"):
                    out.append((u, v))
            elif lu == locale and lv == locale:
                out.append((u, v))
            if len(out) >= want:
                return out
    return out


def apply(labels, succ, pred, sdk_prefixes, op, locale, budget, seed):
    """Perturbed (labels, succ, pred) copies; budget = inserted nodes (edges for phantom_edge); deterministic in seed."""
    # copy the graph
    L = dict(labels)
    S = {}
    for k, v in succ.items():
        S[k] = list(v)
    P = {}
    for k, v in pred.items():
        P[k] = list(v)
    seed_key = f"{op}|{locale}|{budget}|{seed}"
    rng = random.Random(hashlib.sha1(seed_key.encode()).hexdigest())
    existing = set(L)

    def add_edge(u, v):
        if u not in S:
            S[u] = []
        S[u].append(v)
        if v not in P:
            P[v] = []
        P[v].append(u)

    def add_node(lbl):
        # negative ids never collide with GML ids
        while True:
            i = -(rng.getrandbits(40) + 1)
            if i not in existing:
                break
        existing.add(i)
        L[i] = lbl
        return i

    if op == "flatten":
        # Collapse the locale's nodes (sdk, host, or all) into one node; external edges are redirected to it.
        # NOT semantics-preserving: a structural-destruction bound.
        if locale == "all":
            # one node carrying the multiset of labels is not representable; use a single
            # generic node. This is the total-destruction bound.
            return {0: "Lp/flat;->all()V"}, {}, {}
        part = partition(L, sdk_prefixes)
        if locale == "sdk":
            group = set(part["sdk"])
        else:
            group = set(part["host"])
        if not group:
            return L, S, P
        rep = min(group)
        L2 = {}
        for n, sig in L.items():
            if n not in group or n == rep:
                L2[n] = sig
        L2[rep] = f"Lp/flat;->{locale}()V"
        S2 = {}
        P2 = {}
        for u, vs in S.items():
            if u in group:
                ru = rep
            else:
                ru = u
            for v in vs:
                if v in group:
                    rv = rep
                else:
                    rv = v
                if ru == rv and ru == rep:
                    continue                       # drop edges internal to the group
                if ru not in S2:
                    S2[ru] = []
                S2[ru].append(rv)
                if rv not in P2:
                    P2[rv] = []
                P2[rv].append(ru)
        return L2, S2, P2

    if op in ("wrapper", "indirection"):
        cross = locale == "boundary"
        if cross:
            loc = "sdk"
        else:
            loc = locale
        edges = edges_in_locale(S, L, loc, sdk_prefixes, rng, budget, cross=cross)
        for (u, v) in edges:
            # remove the original edge
            if u in S and v in S[u]:
                S[u].remove(v)
            if v in P and u in P[v]:
                P[v].remove(u)
            if op == "wrapper":
                w = add_node(_fresh_label(rng, "fwd"))
                add_edge(u, w)
                add_edge(w, v)
            else:
                w1 = add_node(_fresh_label(rng, "ind"))
                w2 = add_node(_fresh_label(rng, "ind"))
                add_edge(u, w1)
                add_edge(w1, w2)
                add_edge(w2, v)
        return L, S, P

    if op == "dead_node":
        if locale != "boundary":
            pool = partition(L, sdk_prefixes)[locale]
        else:
            pool = partition(L, sdk_prefixes)["sdk"]
        for _ in range(budget):
            if not pool:
                break
            u = rng.choice(pool)
            w = add_node(_fresh_label(rng, "dead"))
            add_edge(u, w)
        return L, S, P

    if op == "leaf_dup":
        if locale != "boundary":
            locale_nodes = partition(L, sdk_prefixes)[locale]
        else:
            locale_nodes = partition(L, sdk_prefixes)["sdk"]
        pool = []
        for n in locale_nodes:
            pool.append(n)
        for _ in range(budget):
            if not pool:
                break
            u = rng.choice(pool)
            w = add_node(L[u])                 # same label: a duplicated method
            # snapshot neighbour lists before add_edge grows them
            if u in S:
                callees = list(S[u])
            else:
                callees = []
            for v in callees:
                add_edge(w, v)
            if u in P:
                callers = list(P[u])
            else:
                callers = []
            for p in callers:
                add_edge(p, w)
        return L, S, P

    if op == "phantom_edge":
        pool = partition(L, sdk_prefixes)
        if locale == "boundary":
            a = pool["sdk"]
            b = pool["host"]
        else:
            a = pool[locale]
            b = a
        added = 0
        tries = 0
        while added < budget and tries < budget * 20 and a and b:
            u = rng.choice(a)
            v = rng.choice(b)
            tries += 1
            if u != v and (u not in S or v not in S[u]):
                add_edge(u, v)
                added += 1
        return L, S, P

    raise ValueError(op)
