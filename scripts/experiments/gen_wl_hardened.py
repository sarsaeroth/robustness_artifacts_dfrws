#!/usr/bin/env python3
"""Obfuscation-hardened WL variants: replace the raw method-name node color
with an obfuscation-invariant descriptor drawn from the TPL-detection literature.

Modes (initial node coloring):
  fuzzy   - LibScout-style fuzzy signature: developer method NAME dropped, owner
            and non-framework types -> X, framework types + framework methods kept.
  api     - LibID/LibSleuth-style: a method's color is the sorted SET of framework
            APIs it calls (its framework-prefixed successors). Framework nodes keep
            their own label. Purely structural -> renaming-invariant.
  opcode  - LibD/LibScan-style: color = hash of the method's opcode multiset
            (operands stripped). Requires a per-sample {method: opcode_hash} map
            (built by extract_opcodes.py). Method-granular (documented limitation).
  combined- hash(fuzzy | api | opcode).

Everything else (2 directed WL iterations, md5 feature-hashing, L2 norm, cfg+fcg)
is identical to gen_wl_vectors.py so results are directly comparable.

Output: {gid}.{kind}_wl{TAG}_{dim}  where TAG in {fz, api, op, all}.
"""
import gc
import json
import os
import sys
import numpy as np
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import gen_wl_vectors as G

DIM = 1024
KINDS = ["cfg", "fcg"]
FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]
TAG = {"fuzzy": "fz", "api": "api", "opcode": "op", "combined": "all"}

# Types/methods that obfuscators cannot rename (framework namespaces).
FRAMEWORK = ("Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/",
             "Lcom/google/", "Lorg/apache/", "Lorg/json/", "Lorg/w3c/",
             "Lorg/xml/", "Ljunit/", "Ldalvik/", "Lokhttp", "Lokio")


def is_framework(type_tok):
    return type_tok.startswith(FRAMEWORK)


def fuzzy_type(t):
    arr = ""
    while t.startswith("["):
        arr += "["
        t = t[1:]
    if t.startswith("L"):
        if is_framework(t):
            return arr + t
        return arr + "X"
    return arr + t  # primitive kept verbatim


def fuzzy_label(raw):
    """LibScout-style fuzzy descriptor of a method signature."""
    if "->" not in raw or "(" not in raw or ")" not in raw:
        return "X"
    owner, rest = raw.split("->", 1)
    if is_framework(owner):
        return raw  # framework method: keep fully (name is safe & meaningful)
    name, sig = rest.split("(", 1)
    params, ret = sig.split(")", 1)
    # split the Dalvik parameter descriptor into type tokens
    out = []
    i = 0
    n = len(params)
    while i < n:
        c = params[i]
        if c == "[":
            j = i
            while j < n and params[j] == "[":
                j += 1
            if j < n and params[j] == "L":
                k = params.find(";", j)
                out.append(params[i:k + 1])
                i = k + 1
            else:
                out.append(params[i:j + 1])
                i = j + 1
        elif c == "L":
            k = params.find(";", i)
            out.append(params[i:k + 1])
            i = k + 1
        else:
            out.append(c)
            i += 1
    fuzzy_parts = []
    for t in out:
        fuzzy_parts.append(fuzzy_type(t))
    pf = "".join(fuzzy_parts)
    return f"X->({pf}){fuzzy_type(ret)}"


def api_label(n, labels, successors):
    """Color = sorted set of framework APIs this node calls."""
    if is_framework(labels.get(n, "")):
        return labels[n]  # framework node keeps its identity
    api_set = set()
    for s in successors.get(n, []):
        if s in labels and is_framework(labels[s]):
            api_set.add(labels[s])
    apis = sorted(api_set)
    if apis:
        return "API[" + ",".join(apis) + "]"
    return "APInone"


def build_colors(mode, labels, successors, kind, opmap):
    colors = {}
    for n, raw in labels.items():
        lbl = G.canonical_label(raw, kind)
        if mode == "fuzzy":
            colors[n] = fuzzy_label(lbl)
        elif mode == "api":
            colors[n] = api_label(n, labels, successors)  # uses raw labels for succ
        elif mode == "opcode":
            if opmap:
                colors[n] = "OP" + opmap.get(_method_of(lbl), "?")
            else:
                colors[n] = "OP?"
        elif mode == "combined":
            fz = fuzzy_label(lbl)
            ap = api_label(n, labels, successors)
            if opmap:
                op = opmap.get(_method_of(lbl), "?")
            else:
                op = "?"
            colors[n] = f"{fz}|{ap}|{op}"
    return colors


def _method_of(lbl):
    """Opcode-map lookup key: whitespace-normalized method signature."""
    # canonical_label already stripped any CFG :offset before this is called
    return "".join(lbl.split())


def wl_vector(colors, successors, predecessors, dim=DIM, iters=2):
    if not colors:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)

    def add(tok):
        h = G.stable_hash_u32(tok)
        if (h & 0x80000000) == 0:
            vec[h % dim] += 1.0
        else:
            vec[h % dim] += -1.0

    for c in colors.values():
        add(f"0|{c}")
    cur = colors
    for i in range(1, iters + 1):
        new = {}
        for nd in cur:
            inl = []
            for p in predecessors.get(nd, []):
                if p in cur:
                    inl.append(cur[p])
            inl.sort()
            out = []
            for s in successors.get(nd, []):
                if s in cur:
                    out.append(cur[s])
            out.sort()
            new[nd] = wl_digest(cur[nd], inl, out)
        cur = new
        for c in cur.values():
            add(f"{i}|{c}")
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        return vec / nrm
    return vec


def main():
    if len(sys.argv) > 1:
        modes = sys.argv[1].split(",")
    else:
        modes = "fuzzy".split(",")
    root = sys.argv[2] if len(sys.argv) > 2 else "datasets"
    for m in modes:
        assert m in TAG, f"mode must be one of {list(TAG)}"

    samples = []
    for fam in FAMILIES:
        fd = os.path.join(root, fam)
        if os.path.isdir(fd):
            for gid in sorted(os.listdir(fd)):
                d = os.path.join(fd, gid)
                if os.path.isdir(d):
                    samples.append((d, gid))
    print(f"modes={modes}  {len(samples)} dirs", flush=True)

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
        print(f"\n=== {kind.upper()} modes={modes} ===", flush=True)
        for i, (d, gid) in enumerate(order):
            # modes whose output for this app is still missing
            todo = []
            for m in modes:
                if not os.path.exists(os.path.join(d, f"{gid}.{kind}_wl{TAG[m]}_{DIM}")):
                    todo.append(m)
            if not todo:
                skip += 1
                continue
            gp = os.path.join(d, f"{gid}.{kind}")
            if not os.path.exists(gp):
                fail += 1
                continue
            opmap = None
            needs_opcodes = False
            for m in todo:
                if m in ("opcode", "combined"):
                    needs_opcodes = True
                    break
            if needs_opcodes:
                opf = os.path.join(d, f"{gid}.method_opcodes.json")
                if os.path.exists(opf):
                    with open(opf) as f:
                        opmap = json.load(f)
            try:
                labels, su, pr = G.parse_gml_lightweight(gp)
            except Exception:
                fail += 1
                continue
            for m in todo:
                if m in ("opcode", "combined") and opmap is None:
                    continue  # skip opcode modes until the opcode map exists
                colors = build_colors(m, labels, su, kind, opmap)
                v = wl_vector(colors, su, pr)
                with open(os.path.join(d, f"{gid}.{kind}_wl{TAG[m]}_{DIM}"), "wb") as f:
                    np.save(f, v)
            done += 1
            del labels, su, pr
            gc.collect()
            if done % 300 == 0:
                print(f"  [{i+1}/{len(order)}] {done} gen {skip} cached {fail} miss ({time()-t0:.0f}s)", flush=True)
        print(f"  {kind}: {done} gen, {skip} cached, {fail} miss ({time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
