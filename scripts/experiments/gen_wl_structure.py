#!/usr/bin/env python3
"""Structure-only WL vectors: initial node color = directed degree, not label.

Ablation twin of gen_wl_vectors.py. Everything (2 WL iters, directed neighbor
signatures, md5 feature-hashing, L2 norm) is identical EXCEPT the initial
coloring uses each node's (in-degree, out-degree) instead of its method-name
label. No library-node dropping (that is a label-based operation), so this is
the purely topological feature set -- what an analyst with graph structure but
no meaningful symbol names (e.g. iOS) would have.

Output: {gid}.cfg_wls_{dim} / {gid}.fcg_wls_{dim}  (note the 's' = structure).
"""
import gc, hashlib, os, sys
import numpy as np
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import gen_wl_vectors as G   # reuse parser + stable_hash_u32

DIMS = [1024]
KINDS = ["cfg", "fcg"]
FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]


def wl_structure_vector(node_labels, successors, predecessors, dim, iters=2):
    if not node_labels:
        return np.zeros(dim, dtype=np.float32)
    # initial color = directed degree signature (NO name information)
    labels = {}
    for n in node_labels:
        labels[n] = f"d{len(predecessors.get(n, []))}_{len(successors.get(n, []))}"
    vec = np.zeros(dim, dtype=np.float32)

    def add(tok):
        h = G.stable_hash_u32(tok)
        if (h & 0x80000000) == 0:
            vec[h % dim] += 1.0
        else:
            vec[h % dim] += -1.0

    for l in labels.values():
        add(f"0|{l}")

    # WL refinement over directed neighbor signatures
    cur = labels
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
    # collect sample directories
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = "datasets"
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
        # smallest graphs first
        def gsize(s):
            p = os.path.join(s[0], f"{s[1]}.{kind}")
            if os.path.exists(p):
                return os.path.getsize(p)
            else:
                return 0
        order = sorted(samples, key=gsize)
        done = 0
        skip = 0
        fail = 0
        t0 = time()
        print(f"\n=== {kind.upper()} ({len(order)}) ===", flush=True)
        for i, (d, gid) in enumerate(order):
            need = []
            for dim in DIMS:
                if not os.path.exists(os.path.join(d, f"{gid}.{kind}_wls_{dim}")):
                    need.append(dim)
            if not need:
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
            for dim in need:
                v = wl_structure_vector(nl, su, pr, dim)
                with open(os.path.join(d, f"{gid}.{kind}_wls_{dim}"), "wb") as f:
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
