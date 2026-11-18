#!/usr/bin/env python3
"""CFG token cache with method provenance (robustness paper, CFG extension).

For every staged proxy app, parse the control-flow graph (basic blocks; the
extractor labels each block with its method's signature) and emit the WL tokens
of every coloring, clean and under per-application identifier renaming, each
token tagged with the method its block belongs to. Colorings mirror the packed
cfg_* vectors exactly: identifier drops library methods (gen_wl_vectors), the
invariant colorings use all nodes (gen_wl_hardened / gen_wl_structure).
The FCG combined coloring is also cached (FCG identifier/fuzzy/api/structure
tokens already exist in p3/colour_tokens), so the deployed CFG+FCG classifier can
be analyzed for every coloring.

Under renaming, the opcode map is re-keyed to the renamed signatures, because
renaming changes method names, not method bodies.

Output: results/tier4/p4/cfg_tokens/<sha>.npz with, per key '<graph>|<coloring>|<cond>':
  '|h' uint32 token hashes, '|m' int32 method index (-1 = library/unscored);
  'methods_truth' bool per scored method, and 'cfg_blocks_per_method'.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import gen_wl_vectors as G
import gen_wl_hardened as HD
import gen_wl_renamed as REN
import provenance_localization as P
import graph_io as GI

GRAPHS = H.GRAPHS_DIR
P4 = H.out_dir("tier4", "p4")
TOKDIR = os.path.join(H.data_dir("tier4", "p4"), "cfg_tokens")
COLOURS = ["identifier", "fuzzy", "api", "structure", "combined"]


def colours(colour, lab, su, pr, kind, opmap):
    """lab: node -> canonical label (possibly renamed), all nodes."""
    if colour == "identifier":
        keep = {}
        for n, l in lab.items():
            if not l.startswith(G.LIB_PREFIXES):
                keep[n] = l
        if keep:
            return keep
        return dict(lab)
    if colour == "structure":
        cols = {}
        for n in lab:
            cols[n] = f"d{len(pr.get(n, []))}_{len(su.get(n, []))}"
        return cols
    return HD.build_colors(colour, lab, su, kind, opmap)


def wl_tokens(cols, su, pr):
    hs = []
    nodes = []
    for n, c in cols.items():
        hs.append(G.stable_hash_u32(f"0|{c}"))
        nodes.append(n)
    cur = cols
    for it in (1, 2):
        nxt = {}
        for n in cur:
            i_l = []
            for p in pr.get(n, []):
                if p in cur:
                    i_l.append(cur[p])
            i_l.sort()
            o_l = []
            for s in su.get(n, []):
                if s in cur:
                    o_l.append(cur[s])
            o_l.sort()
            nxt[n] = wl_digest(cur[n], i_l, o_l)
        cur = nxt
        for n, c in cur.items():
            hs.append(G.stable_hash_u32(f"{it}|{c}"))
            nodes.append(n)
    return hs, nodes


def renamed_opmap(opmap, lab, ren):
    if opmap is None:
        return None
    out = {}
    for n in lab:
        ok = HD._method_of(lab[n])
        if ok in opmap:
            out[HD._method_of(ren[n])] = opmap[ok]
    return out


def extract(args):
    fam, sha = args
    path = os.path.join(TOKDIR, f"{sha}.npz")
    if os.path.exists(path):
        return sha, "cached"
    d = os.path.join(GRAPHS, fam, sha)
    opf = os.path.join(d, f"{sha}.method_opcodes.json")
    if os.path.exists(opf):
        opmap = json.load(open(opf))
    else:
        opmap = None
    out = {}
    # scored methods: non-library method signatures (shared by CFG and FCG)
    nl, su, pr = G.parse_gml_lightweight(os.path.join(d, f"{sha}.cfg"))
    su = dict(su)
    pr = dict(pr)
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "cfg")
    method_set = set()
    for l in lab.values():
        if not l.startswith(G.LIB_PREFIXES):
            method_set.add(l)
    methods = sorted(method_set)
    mid = {}
    for i, m in enumerate(methods):
        mid[m] = i
    truth = []
    for m in methods:
        truth.append(m.startswith(P.SDK_PREFIXES.get(fam, ())))
    out["methods_truth"] = np.array(truth)
    out["methods"] = np.array(methods, dtype=np.str_)
    block_mids = []
    for l in lab.values():
        if l in mid:
            block_mids.append(mid[l])
    bpm = np.bincount(block_mids, minlength=len(methods))
    out["cfg_blocks_per_method"] = bpm.astype(np.int32)
    ren = {}
    for n, l in lab.items():
        ren[n] = REN.rename_label(l, sha[:16])
    ren_opmap = renamed_opmap(opmap, lab, ren)
    for cond, L, om in (("clean", lab, opmap), ("renamed", ren, ren_opmap)):
        for c in COLOURS:
            if c == "combined" and opmap is None:
                continue
            cols = colours(c, L, su, pr, "cfg", om)
            hs, nodes = wl_tokens(cols, su, pr)
            out[f"cfg|{c}|{cond}|h"] = np.array(hs, dtype=np.uint32)
            node_mids = []
            for n in nodes:
                if lab[n] in mid:
                    node_mids.append(mid[lab[n]])
                else:
                    node_mids.append(-1)
            out[f"cfg|{c}|{cond}|m"] = np.array(node_mids, dtype=np.int32)
    del nl, su, pr
    # FCG tokens for every coloring, with the same method index, so the deployed
    # CFG+FCG classifier's attributions can be summed per method across graphs.
    nl, su, pr = G.parse_gml_lightweight(os.path.join(d, f"{sha}.fcg"))
    su = dict(su)
    pr = dict(pr)
    flab = {}
    for n, r in nl.items():
        flab[n] = G.canonical_label(r, "fcg")
    fren = {}
    for n, l in flab.items():
        fren[n] = REN.rename_label(l, sha[:16])
    fren_opmap = renamed_opmap(opmap, flab, fren)
    for cond, L, om in (("clean", flab, opmap), ("renamed", fren, fren_opmap)):
        for c in COLOURS:
            if c == "combined" and opmap is None:
                continue
            cols = colours(c, L, su, pr, "fcg", om)
            hs, nodes = wl_tokens(cols, su, pr)
            out[f"fcg|{c}|{cond}|h"] = np.array(hs, dtype=np.uint32)
            node_mids = []
            for n in nodes:
                if flab[n] in mid:
                    node_mids.append(mid[flab[n]])
                else:
                    node_mids.append(-1)
            out[f"fcg|{c}|{cond}|m"] = np.array(node_mids, dtype=np.int32)
    del nl, su, pr
    os.makedirs(TOKDIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return sha, "ok"


def main():
    ids, _ = H.sample_ids("four")
    pos = set(ids)
    staged = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos and GI.exists(os.path.join(GRAPHS, f, s, f"{s}.cfg")):
                staged.append((f, s))
    # largest graphs last so memory peaks are not concurrent with each other early on
    staged.sort(key=lambda t: GI.size(os.path.join(GRAPHS, t[0], t[1], f"{t[1]}.cfg")))
    print(f"{len(staged)} staged apps with CFGs", flush=True)
    with ProcessPoolExecutor(3) as ex:
        for k, (sha, st) in enumerate(ex.map(extract, staged), 1):
            print(f"[{k}/{len(staged)}] {sha[:10]} {st}", flush=True)


if __name__ == "__main__":
    main()
