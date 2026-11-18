#!/usr/bin/env python3
"""Third ablation variant: degree init + library-node dropping.

Same node set as the LABELED model (library nodes removed by name, CFG line
numbers canonicalized), but each surviving node's initial color is its
(in,out)-degree *within the surviving subgraph* rather than its method name.

This isolates the value of name identity given a clean app-subgraph node set,
separating it from the value of the (name-based) library filtering itself.

Output: {gid}.cfg_wlsd_{dim} / {gid}.fcg_wlsd_{dim}  (sd = structure+drop).
"""
import gc
import os
import sys
import numpy as np
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import gen_wl_vectors as G

DIMS = [1024]
KINDS = ["cfg", "fcg"]
FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]


def wl_structdrop_vector(node_labels, successors, predecessors, kind, dim, iters=2):
    if not node_labels:
        return np.zeros(dim, dtype=np.float32)
    # canonicalize + drop library nodes exactly like the labeled pipeline
    labels = {}
    for n, r in node_labels.items():
        labels[n] = G.canonical_label(r, kind)
    drop = []
    for n, l in labels.items():
        if l.startswith(G.LIB_PREFIXES):
            drop.append(n)
    if drop and len(drop) < len(labels):
        for n in drop:
            del labels[n]
    if not labels:
        return np.zeros(dim, dtype=np.float32)
    S = set(labels)
    # initial color = degree within the surviving subgraph (no name info)
    cur = {}
    for n in S:
        n_in = 0
        for p in predecessors.get(n, []):
            if p in S:
                n_in += 1
        n_out = 0
        for s in successors.get(n, []):
            if s in S:
                n_out += 1
        cur[n] = f"d{n_in}_{n_out}"

    vec = np.zeros(dim, dtype=np.float32)

    def add(tok):
        h = G.stable_hash_u32(tok)
        if (h & 0x80000000) == 0:
            vec[h % dim] += 1.0
        else:
            vec[h % dim] += -1.0

    for l in cur.values():
        add(f"0|{l}")
    for i in range(1, iters + 1):
        new = {}
        for n in cur:
            inl = []
            for p in predecessors.get(n, []):
                if p in cur:
                    inl.append(cur[p])
            inl.sort()
            out = []
            for s in successors.get(n, []):
                if s in cur:
                    out.append(cur[s])
            out.sort()
            new[n] = wl_digest(cur[n], inl, out)
        cur = new
        for l in cur.values():
            add(f"{i}|{l}")
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec /= nrm
    return vec


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    samples = []
    for fam in FAMILIES:
        fd = os.path.join(root, fam)
        if not os.path.isdir(fd):
            continue
        for gid in sorted(os.listdir(fd)):
            d = os.path.join(fd, gid)
            if os.path.isdir(d):
                samples.append((d, gid))
    print(f"Found {len(samples)} sample dirs", flush=True)

    for kind in KINDS:
        # smallest graphs first (missing graph counts as size 0)
        sizes = {}
        for s in samples:
            p = os.path.join(s[0], f"{s[1]}.{kind}")
            if os.path.exists(p):
                sizes[s] = os.path.getsize(p)
            else:
                sizes[s] = 0
        order = sorted(samples, key=lambda s: sizes[s])
        done = 0
        skip = 0
        fail = 0
        t0 = time()
        print(f"\n=== {kind.upper()} ({len(order)}) ===", flush=True)
        for i, (d, gid) in enumerate(order):
            out = os.path.join(d, f"{gid}.{kind}_wlsd_{DIMS[0]}")
            if os.path.exists(out):
                skip += 1
                continue
            gp = os.path.join(d, f"{gid}.{kind}")
            if not os.path.exists(gp):
                fail += 1
                continue
            try:
                nl, su, pr = G.parse_gml_lightweight(gp)
            except Exception:
                fail += 1
                continue
            for dim in DIMS:
                v = wl_structdrop_vector(nl, su, pr, kind, dim)
                with open(os.path.join(d, f"{gid}.{kind}_wlsd_{dim}"), "wb") as f:
                    np.save(f, v)
            done += 1
            del nl, su, pr
            gc.collect()
            if done % 200 == 0:
                print(f"  [{i+1}/{len(order)}] {done} gen {skip} cached {fail} miss "
                      f"({time()-t0:.0f}s)", flush=True)
        print(f"  {kind}: {done} generated, {skip} cached, {fail} missing "
              f"({time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
