#!/usr/bin/env python3
"""Payload ground truth for the piggybacked-pairs dataset.

For each (original, piggybacked) pair with both FCGs, the payload ("rider", Li et al.) is the
set of non-library methods of the piggybacked app whose canonical signature does not occur in
the original. Per app: results/datasets/piggyback/payload_methods/{sha}.json (methods, their
classes and packages, counts).

For the family-keyed localization scripts, payload_prefixes.json maps "piggybacked" to the
rider packages that never occur in any original application (and are not a prefix of any
original package), as package prefixes; by construction no original (host) method matches, so
the prefix truth can only miss rider code that reuses a host package, never flag host code.
The share of rider methods covered by these prefixes is reported (recall of the prefix truth).

Usage: piggyback_payload.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_wl_vectors as G
import graph_io as GI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "datasets", "piggyback")
SRC = os.path.join(ROOT, "datasets", "piggyback")


def methods(fam, sha):
    p = os.path.join(SRC, fam, sha, f"{sha}.fcg")
    if not GI.exists(p):
        return None
    nl, _, _ = G.parse_gml_lightweight(p)
    out = set()
    for r in nl.values():
        l = G.canonical_label(r, "fcg")
        if not l.startswith(G.LIB_PREFIXES):
            out.add(l)
    return out


def cls_of(m):
    return m.split("->")[0]


def pkg_of(m):
    c = cls_of(m)
    if "/" in c:
        return c[:c.rfind("/") + 1]
    else:
        return c


def main():
    pairs = json.load(open(os.path.join(D, "pairs.json")))
    out_dir = os.path.join(D, "payload_methods")
    os.makedirs(out_dir, exist_ok=True)
    orig_pk = set()
    rider_pk = Counter()
    stats = []
    cache = {}

    # per pair: rider methods = piggybacked non-library methods absent from the original
    for o, p in pairs:
        if o in cache:
            mo = cache.get(o)
        else:
            mo = methods("original", o)
        cache[o] = mo
        mp = methods("piggybacked", p)
        if mo is None or mp is None:
            continue
        for m in mo:
            orig_pk.add(pkg_of(m))
        new = sorted(mp - mo)
        new_classes = set()
        for m in new:
            new_classes.add(cls_of(m))
        json.dump({"original": o, "methods": new, "classes": sorted(new_classes),
                   "n_new": len(new), "n_total": len(mp)}, open(os.path.join(out_dir, f"{p}.json"), "w"))
        new_pkgs = set()
        for m in new:
            new_pkgs.add(pkg_of(m))
        for k in new_pkgs:
            rider_pk[k] += 1
        stats.append((p, len(new), len(mp)))

    # rider packages that are never an original package nor a prefix of one
    clean_unsorted = []
    for k in rider_pk:
        if k in orig_pk:
            continue
        is_prefix_of_original = False
        for op in orig_pk:
            if op != k and op.startswith(k):
                is_prefix_of_original = True
                break
        if not is_prefix_of_original:
            clean_unsorted.append(k)
    clean = sorted(clean_unsorted)
    json.dump({"piggybacked": clean}, open(os.path.join(D, "payload_prefixes.json"), "w"))

    # prefix recall per piggybacked app
    cov = []
    cs = tuple(clean)
    for p, n, _ in stats:
        if n:
            m = json.load(open(os.path.join(out_dir, f"{p}.json")))["methods"]
            n_covered = 0
            for x in m:
                if x.startswith(cs):
                    n_covered += 1
            cov.append(n_covered / n)

    # summary
    occ = []
    with_new = 0
    rider_counts = []
    for _, n, t in stats:
        if t:
            occ.append(n / t)
        if n > 0:
            with_new += 1
        rider_counts.append(n)
    if stats:
        rider_median = float(sorted(rider_counts)[len(stats) // 2])
    else:
        rider_median = None
    if occ:
        occ_median = float(sorted(occ)[len(occ) // 2])
    else:
        occ_median = None
    if cov:
        cov_median = float(sorted(cov)[len(cov) // 2])
    else:
        cov_median = None
    summ = {"pairs": len(stats), "with_new_methods": with_new,
            "rider_methods_median": rider_median,
            "occupancy_median": occ_median,
            "clean_prefixes": len(clean), "prefix_recall_median": cov_median}
    json.dump(summ, open(os.path.join(D, "payload_summary.json"), "w"), indent=1)
    print(summ, flush=True)


if __name__ == "__main__":
    main()
