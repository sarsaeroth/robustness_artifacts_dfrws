#!/usr/bin/env python3
"""Stress test: simulate per-app re-obfuscation (independent identifier renaming
per sample) and regenerate the LABELED WL vectors on the renamed graphs.

Within one app, a developer identifier is renamed consistently; ACROSS apps the
mapping differs (seeded by the app id). Framework names (Landroid/, Ljava/, ...)
are never renamed. This destroys the cross-app name-sharing that the labeled
classifier relies on, while preserving intra-app structure and framework calls -
exactly what per-app ProGuard/R8 re-obfuscation would do.

The hardened variants (fuzzy/api/opcode/combined) are invariant to this by
construction, so only the labeled variant needs regenerating.

Output: {gid}.{kind}_wlren_{dim}
"""
import gc
import hashlib
import os
import re
import sys
import numpy as np
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_wl_vectors as G
from gen_wl_hardened import is_framework, wl_vector

DIM = 1024
KINDS = ["cfg", "fcg"]
FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]
TYPE_RE = re.compile(r"\[*L[^;]+;")


def ren_type(tok, seed):
    """Rename a non-framework object type; keep arrays' [ and framework/primitives."""
    arr = ""
    t = tok
    while t.startswith("["):
        arr += "["
        t = t[1:]
    if t.startswith("L") and not is_framework(t):
        h = hashlib.md5((seed + t).encode()).hexdigest()[:10]
        return arr + f"Lr/{h};"
    return tok  # framework object or primitive: unchanged


def rename_label(raw, seed):
    """Per-app consistent renaming of a method signature's developer identifiers."""
    if "->" not in raw or "(" not in raw or ")" not in raw:
        return TYPE_RE.sub(lambda m: ren_type(m.group(0), seed), raw)
    owner, rest = raw.split("->", 1)
    if is_framework(owner):
        return raw  # framework method: fully preserved
    name, sig = rest.split("(", 1)
    params, ret = sig.split(")", 1)
    owner_r = ren_type(owner, seed)
    name_r = "m" + hashlib.md5((seed + owner + name).encode()).hexdigest()[:8]
    params_r = TYPE_RE.sub(lambda m: ren_type(m.group(0), seed), params)
    if ret.startswith(("[", "L")):
        ret_r = ren_type(ret, seed)
    else:
        ret_r = ret
    return f"{owner_r}->{name_r}({params_r}){ret_r}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    samples = []
    for fam in FAMILIES:
        fd = os.path.join(root, fam)
        if os.path.isdir(fd):
            for gid in sorted(os.listdir(fd)):
                d = os.path.join(fd, gid)
                if os.path.isdir(d):
                    samples.append((d, gid))
    print(f"{len(samples)} dirs", flush=True)

    for kind in KINDS:
        # smallest graphs first (missing graph counts as size 0)
        sizes = {}
        for s in samples:
            gpath = os.path.join(s[0], f"{s[1]}.{kind}")
            if os.path.exists(gpath):
                sizes[s] = os.path.getsize(gpath)
            else:
                sizes[s] = 0
        order = sorted(samples, key=lambda s: sizes[s])
        done = 0
        skip = 0
        fail = 0
        t0 = time()
        print(f"\n=== {kind.upper()} (renamed) ===", flush=True)
        for i, (d, gid) in enumerate(order):
            out = os.path.join(d, f"{gid}.{kind}_wlren_{DIM}")
            if os.path.exists(out):
                skip += 1
                continue
            gp = os.path.join(d, f"{gid}.{kind}")
            if not os.path.exists(gp):
                fail += 1
                continue
            try:
                labels, su, pr = G.parse_gml_lightweight(gp)
            except Exception:
                fail += 1
                continue
            # rename every node label with the app's seed, then hash
            seed = gid[:16]
            colors = {}
            for n, raw in labels.items():
                colors[n] = rename_label(G.canonical_label(raw, kind), seed)
            v = wl_vector(colors, su, pr)
            with open(out, "wb") as f:
                np.save(f, v)
            done += 1
            del labels, su, pr, colors
            gc.collect()
            if done % 300 == 0:
                print(f"  [{i+1}/{len(order)}] {done} gen {skip} cached {fail} miss ({time()-t0:.0f}s)", flush=True)
        print(f"  {kind}: {done} gen, {skip} cached, {fail} miss ({time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
