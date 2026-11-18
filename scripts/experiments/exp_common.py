#!/usr/bin/env python3
"""Shared corpus access and WL token instrumentation for the Tier 1/Tier 3 experiments.

The experiments all need the same thing that the production vectoriser throws
away: not the hashed vector, but the individual tokens each node emits and
whether that node belongs to the target subgraph. Isolation, identifiability and
inflation are all statements about those token sets.

Corpus root is CORPUS_ROOT (default datasets/), so the same scripts run against
an unpacked AndMal1_data bundle without editing anything.
"""
import glob
import hashlib
import os
import resource
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from wl_hash import wl_digest
import gen_wl_vectors as G
import gen_wl_hardened as HD
import graph_io as GI

ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", os.path.join(ROOT, "datasets"))
# Other datasets without CORPUS_ROOT: the staged subset (results/datasets/<DATASET>/staged.json,
# make_staged.py) of the dataset directory datasets/<DATASET>.
STAGED = None
_ds = os.environ.get("DATASET", "proxy")
_v3 = _ds.startswith("proxy_v3")  # corpus v3 views share proxy_v3's staged set in work/staged_v3
if _ds != "proxy" and "CORPUS_ROOT" not in os.environ:
    if _v3:
        _st_ds = "proxy_v3"
    else:
        _st_ds = _ds
    _st = os.path.join(ROOT, "results", "datasets", _st_ds, "staged.json")
    if os.path.exists(_st):
        import json as _json
        STAGED = {}
        for _f, _v in _json.load(open(_st))["apps"].items():
            STAGED[_f] = set(_v)
        if _v3:
            CORPUS_ROOT = os.path.join(ROOT, "work", "staged_v3")
        else:
            CORPUS_ROOT = os.path.join(ROOT, "datasets", _ds)

# Family list is whatever CORPUS_ROOT contains (or the staged families), so the same scripts
# run against the proxy corpus or against an arbitrary labelled corpus.
_default_families = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]
if STAGED is not None:
    FAMILIES = sorted(STAGED)
elif not os.path.isdir(CORPUS_ROOT):
    FAMILIES = _default_families
else:
    _found = []
    for _d in sorted(os.listdir(CORPUS_ROOT)):
        if os.path.isdir(os.path.join(CORPUS_ROOT, _d)) and not _d.startswith("_"):
            _found.append(_d)
    if _found:
        FAMILIES = _found
    else:
        FAMILIES = _default_families
PROXY_FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
# other datasets: the target families are the staged families under CORPUS_ROOT, minus the benign class
if os.environ.get("DATASET", "proxy") != "proxy":
    PROXY_FAMILIES = []
    for _f in FAMILIES:
        if _f not in ("benign", "non_proxy"):
            PROXY_FAMILIES.append(_f)

# strict, calibrated namespace markers; see scripts/re_flagged_apps.py
SDK_PREFIXES = {
    "ipninja":     ("Lcom/ipninja/", "Lnet/ipninja/"),
    "luminati":    ("Lio/lum/", "Lio/topvpn/", "Lorg/hola/", "Lio/luminati/"),
    "monkeysocks": ("Lmonkeynode/",),
    "oxylabs":     ("Lframett/",),
}

# Other datasets: curated payload namespaces from payload_namespaces.py (see CROSS_DATASET_PLAN).
if os.environ.get("DATASET", "proxy") != "proxy":
    import json as _json
    if _v3:
        _pf_ds = "proxy_v3"
    else:
        _pf_ds = os.environ["DATASET"]
    _pf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "datasets",
                       _pf_ds, "payload_prefixes.json")
    SDK_PREFIXES = {}
    if os.path.exists(_pf):
        for _k, _v in _json.load(open(_pf)).items():
            SDK_PREFIXES[_k] = tuple(_v)

def out_path(*parts):
    """results/<parts> for the proxy corpus; results/datasets/<DATASET>/<parts> otherwise."""
    ds = os.environ.get("DATASET", "proxy")
    # SMOKE_DIR when set and non-empty, else results/ (see exp_harness.RESULTS)
    res = os.environ.get("SMOKE_DIR")
    if not res:
        res = os.path.join(ROOT, "results")
    if ds == "proxy":
        p = os.path.join(res, *parts)
    else:
        p = os.path.join(res, "datasets", ds, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def family_apps(fam):
    """Apps of a family under CORPUS_ROOT (the staged subset where STAGED is set)."""
    if STAGED is not None:
        if fam in STAGED:
            return sorted(STAGED[fam])
        return []
    d = os.path.join(CORPUS_ROOT, fam)
    if os.path.isdir(d):
        return os.listdir(d)
    return []


# an FCG above this size costs more to parse than it contributes
MAX_FCG_BYTES = int(os.environ.get("MAX_FCG_BYTES", 25_000_000))


def cap_memory(gb):
    """Hard address-space cap. A runaway pass should die, not take the box down.
    Skipped in the ROCm environment (.venv-rocm) and with GPU=1: the GPU runtime reserves far more
    virtual address space than it uses, so an address-space cap stops it from starting. Resident
    memory is still limited by run_guarded.sh there."""
    if os.environ.get("GPU") == "1" or ".venv-rocm" in sys.prefix:
        return
    limit = int(gb * 2 ** 30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def list_samples(per_family=None, seed=42, families=None, require_opcodes=False):
    """[(family, sha, fcg_path)], deterministically subsampled per family."""
    import random
    rng = random.Random(seed)
    out = []
    if families:
        fam_list = families
    else:
        fam_list = FAMILIES
    for fam in fam_list:
        famdir = os.path.join(CORPUS_ROOT, fam)
        if not os.path.isdir(famdir):
            continue
        # staged subset when STAGED is set, else every app directory
        if STAGED is not None:
            if fam in STAGED:
                sha_list = sorted(STAGED[fam])
            else:
                sha_list = []
        else:
            sha_list = os.listdir(famdir)
        cand = []
        for sha in sha_list:
            sdir = os.path.join(famdir, sha)
            if not os.path.isdir(sdir):
                continue
            g = GI.glob(os.path.join(sdir, "*.fcg"))
            if not g:
                continue
            if GI.size(g[0]) > MAX_FCG_BYTES:
                continue
            if require_opcodes:
                if not os.path.isfile(os.path.join(sdir, sha + ".method_opcodes.json")):
                    continue
            cand.append((fam, sha, g[0]))
        cand.sort()
        rng.shuffle(cand)
        if per_family:
            cand = cand[:per_family]
        out.extend(cand)
    return out


def kept_labels(node_labels, kind="fcg"):
    """Canonical labels with framework/library nodes dropped, as the vectoriser does."""
    labels = {}
    for nid, raw in node_labels.items():
        labels[nid] = G.canonical_label(raw, kind)
    drop = []
    for nid in labels:
        if labels[nid].startswith(G.LIB_PREFIXES):
            drop.append(nid)
    if drop and len(drop) < len(labels):
        for nid in drop:
            del labels[nid]
    return labels


def target_nodes(labels, prefixes):
    """Node ids whose canonical label sits under any of the target namespaces."""
    hits = set()
    for nid in labels:
        for p in prefixes:
            if labels[nid].startswith(p):
                hits.add(nid)
                break
    return hits


def recolour(labels, mode, successors, opmap, kind="fcg"):
    """Apply an alternative initial colouring to an already-filtered label map."""
    if mode == "labeled":
        return dict(labels)
    return HD.build_colors(mode, labels, successors, kind, opmap)


def token_provenance(colors, successors, predecessors, tgt_nodes, iters=2):
    """Distinct token hashes emitted by target and by non-target nodes.

    One token per node per round, exactly as wl_hashed_vector emits them, so the
    hashes here are the same integers that land in the production vector.
    Returns (target_hashes, nontarget_hashes, n_token_instances).
    """
    tgt = set()
    non = set()
    n_inst = 0
    cur = colors
    for i in range(0, iters + 1):
        if i > 0:
            new = {}
            for nd in cur:
                if nd in predecessors:
                    preds = predecessors[nd]
                else:
                    preds = []
                inl = []
                for p in preds:
                    if p in cur:
                        inl.append(cur[p])
                inl.sort()
                if nd in successors:
                    succs = successors[nd]
                else:
                    succs = []
                out = []
                for s in succs:
                    if s in cur:
                        out.append(cur[s])
                out.sort()
                new[nd] = wl_digest(cur[nd], inl, out)
            cur = new
        for nd in cur:
            h = G.stable_hash_u32(f"{i}|{cur[nd]}")
            if nd in tgt_nodes:
                tgt.add(h)
            else:
                non.add(h)
            n_inst += 1
    return tgt, non, n_inst


def isolation_curve(tgt, non, dims):
    """Measured isolated fraction of target tokens at each dimension.

    A target token is isolated at d when no distinct non-target token shares its
    bucket. Tokens emitted by both target and non-target nodes are genuinely
    ambiguous rather than collided, and count against isolation at every d.
    """
    if not tgt:
        return {}
    tgt_arr = np.fromiter(tgt, dtype=np.int64, count=len(tgt))
    non_arr = np.fromiter(non, dtype=np.int64, count=len(non))
    shared = len(tgt & non)
    tgt_only = np.fromiter(tgt - non, dtype=np.int64, count=len(tgt) - shared)
    out = {}
    for d in dims:
        occupied = np.zeros(d, dtype=bool)
        occupied[non_arr % d] = True
        if len(tgt_only) == 0:
            out[d] = 0.0
            continue
        free = np.count_nonzero(~occupied[tgt_only % d])
        out[d] = float(free) / len(tgt_arr)
        del occupied
    return out
