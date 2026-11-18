#!/usr/bin/env python3
"""Vectorize any dataset laid out as {root}/{family}/{sha}/{sha}.{fcg,cfg} with every WL
coloring the robustness paper uses, by calling the same functions that produced the proxy
vectors, so results are directly comparable across datasets.

  labeled    gen_wl_vectors.wl_hashed_vector          -> {kind}_wl_{d}   (d in DIMS)
  fuzzy      gen_wl_hardened fuzzy colors + wl_vector -> {kind}_wlfz_1024
  api        gen_wl_hardened api colors               -> {kind}_wlapi_1024
  combined   gen_wl_hardened combined (needs {sha}.method_opcodes.json) -> {kind}_wlall_1024
  structure  gen_wl_structure.wl_structure_vector     -> {kind}_wls_1024
  degree_ld  gen_wl_structure_libdrop                 -> {kind}_wlsd_1024
  renamed    gen_wl_renamed per-app renaming          -> {kind}_wlren_1024

Each graph is parsed once. Per-app results go to results/datasets/{ds}/pervec/{kind}/{sha}.npz
(resume-safe); `pack` then writes results/datasets/{ds}/vectors/{ext}.npz with ids, family, X,
the format exp_harness reads.

Usage: vectorize_dataset.py run  DS ROOT KIND [WORKERS]
       vectorize_dataset.py pack DS
"""
import gc
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_wl_vectors as G
import gen_wl_hardened as HD
import gen_wl_structure as GS
import gen_wl_structure_libdrop as GSD
import gen_wl_renamed as GR
import graph_io as GI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = [128, 256, 512, 1024]


def ds_dir(ds):
    return os.path.join(ROOT, "results", "datasets", ds)


def vectors_for(path, kind, sha):
    nl, su, pr = G.parse_gml_lightweight(path)
    out = {}
    for d in DIMS:
        out[f"{kind}_wl_{d}"] = G.wl_hashed_vector(nl, su, pr, kind, d)
    for mode, tag in (("fuzzy", "wlfz"), ("api", "wlapi")):
        out[f"{kind}_{tag}_1024"] = HD.wl_vector(HD.build_colors(mode, nl, su, kind, None), su, pr)
    opf = os.path.join(os.path.dirname(path), f"{sha}.method_opcodes.json")
    if os.path.exists(opf):
        opmap = json.load(open(opf))
        out[f"{kind}_wlall_1024"] = HD.wl_vector(HD.build_colors("combined", nl, su, kind, opmap), su, pr)
    out[f"{kind}_wls_1024"] = GS.wl_structure_vector(nl, su, pr, 1024)
    out[f"{kind}_wlsd_1024"] = GSD.wl_structdrop_vector(nl, su, pr, kind, 1024)
    seed = sha[:16]
    ren = {}
    for n, raw in nl.items():
        ren[n] = GR.rename_label(G.canonical_label(raw, kind), seed)
    out[f"{kind}_wlren_1024"] = HD.wl_vector(ren, su, pr)
    out32 = {}
    for k, v in out.items():
        out32[k] = np.asarray(v, dtype=np.float32)
    return out32


def _job(t):
    ds, fam, sha, path, kind = t
    dst = os.path.join(ds_dir(ds), "pervec", kind, f"{sha}.npz")
    if os.path.exists(dst):
        has_op = os.path.exists(os.path.join(os.path.dirname(path), f"{sha}.method_opcodes.json"))
        if not has_op or f"{kind}_wlall_1024" in np.load(dst).files:
            return "cached"
    # crash marker: a graph whose marker survives from an earlier attempt killed that attempt's worker
    # (memory guard / OOM); skip it rather than retrying it forever
    mark = dst + ".inprogress"
    if os.path.exists(mark):
        return f"skipcrash {sha[:10]}"
    open(mark, "w").close()
    try:
        v = vectors_for(path, kind, sha)
    except Exception as e:
        os.remove(mark)
        return f"fail {sha[:10]} {type(e).__name__}"
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, family=np.array(fam), **v)
    os.replace(tmp, dst)
    os.remove(mark)
    gc.collect()
    return "ok"


def run(ds, root, kind, workers=3):
    os.makedirs(os.path.join(ds_dir(ds), "pervec", kind), exist_ok=True)
    jobs = []
    for fam in sorted(os.listdir(root)):
        fd = os.path.join(root, fam)
        if not os.path.isdir(fd):
            continue
        for sha in sorted(os.listdir(fd)):
            p = os.path.join(fd, sha, f"{sha}.{kind}")
            if GI.exists(p):  # plain or .zst
                jobs.append((ds, fam, sha, p, kind))
    jobs.sort(key=lambda j: GI.size(j[3]))
    print(f"{ds}/{kind}: {len(jobs)} graphs, {workers} workers", flush=True)
    counts = {}
    # a worker killed by the kernel (OOM) hangs a Pool forever; a per-result timeout turns that into an exit,
    # so a retry loop can restart the (resumable) run
    import multiprocessing
    stall = int(os.environ.get("VEC_STALL_SECONDS", "1800"))
    with get_context("fork").Pool(workers, maxtasksperchild=20) as pool:
        it = pool.imap_unordered(_job, jobs, chunksize=1)
        for i in range(len(jobs)):
            try:
                r = it.next(timeout=stall)
            except multiprocessing.TimeoutError:
                print(f"stalled: no graph finished in {stall}s (worker likely killed); exiting for a retry", flush=True)
                pool.terminate()
                sys.exit(3)
            key = r.split()[0]
            counts[key] = counts.get(key, 0) + 1
            if r.startswith(("fail", "skipcrash")):
                print("  " + r, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  [{i + 1}/{len(jobs)}] {counts}", flush=True)
    print(f"done {counts}", flush=True)


def pack(ds):
    out_dir = os.path.join(ds_dir(ds), "vectors")
    os.makedirs(out_dir, exist_ok=True)
    rows = {}
    for kind in ("fcg", "cfg"):
        d = os.path.join(ds_dir(ds), "pervec", kind)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz") or ".tmp" in f:
                continue
            z = np.load(os.path.join(d, f))
            sha = f[:-4]
            for k in z.files:
                if k != "family":
                    if k not in rows:
                        rows[k] = []
                    rows[k].append((sha, str(z["family"]), z[k]))
    for ext, items in sorted(rows.items()):
        items.sort(key=lambda t: (t[1], t[0]))
        ids = []
        fams = []
        vecs = []
        for t in items:
            ids.append(t[0])
            fams.append(t[1])
            vecs.append(t[2])
        np.savez(os.path.join(out_dir, ext + ".npz"), ids=np.array(ids),
                 family=np.array(fams), X=np.stack(vecs))
        print(f"  {ext}: {len(items)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        if len(sys.argv) > 5:
            n_workers = int(sys.argv[5])
        else:
            n_workers = 3
        run(sys.argv[2], sys.argv[3], sys.argv[4], n_workers)
    else:
        pack(sys.argv[2])
