#!/usr/bin/env python3
"""Compact per-app token parts for the v3 exact-token experiments.

Each app is stored once as four arrays: unique SDK-origin token ids and counts, unique
host-origin token ids and counts (all WL rounds). This is all A2 and D1 need, and loading
it costs a fraction of the full per-position token pickles.

  results/tier4/v3/parts/{graph}_{coloring}/{sha}.npz   (graph = fcg | cfg)

FCG parts are derived from the existing token pickle (one pass, then freed); CFG parts are
built per app in worker processes that write to disk and return nothing.

Usage: python3 exp_parts_cache.py fcg identifier | fcg degree | cfg identifier | cfg degree
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_tok_analyses as T

ROOT = H.out_dir("tier4", "v3", "parts")


def _dir(graph, col):
    return os.path.join(ROOT, f"{graph}_{col}")


def split(tid, morg, msdk):
    # ts marks tokens whose origin method is in the SDK
    ts = msdk[morg]
    su, sc = np.unique(tid[ts], return_counts=True)
    hu, hc = np.unique(tid[~ts], return_counts=True)
    return su, sc.astype(np.float32), hu, hc.astype(np.float32)


def _save(path, fam, p):
    tmp = path + ".tmp.npz"
    np.savez(tmp, fam=np.array(fam), su=p[0], sc=p[1], hu=p[2], hc=p[3])
    os.replace(tmp, path)


def build_fcg(col):
    import pickle
    d = _dir("fcg", col)
    os.makedirs(d, exist_ok=True)
    # the token pickle may not exist yet (exp_tok_analyses keeps per-app parts); build_cached loads it or
    # assembles it from the per-app parts, building any missing apps
    data = T.build_cached(col, T.staged(), 1)
    for sha in list(data):
        fam, tokens = data.pop(sha)
        tid, rnd, morg, msdk = tokens
        _save(os.path.join(d, f"{sha}.npz"), fam, split(tid, morg, msdk))
    print(f"fcg {col}: {len(os.listdir(d))} apps", flush=True)


def _cfg_job(t):
    fam, sha, col = t
    path = os.path.join(_dir("cfg", col), f"{sha}.npz")
    if os.path.exists(path):
        return None
    try:
        import exp_cfg_exact as D
        tid, rnd, morg, msdk = D.cfg_tokens(fam, sha, col)
        _save(path, fam, split(tid, morg, msdk))
        return None
    except Exception as e:
        return f"{sha[:10]} {repr(e)[:100]}"


def build_cfg(col, workers=2):
    import multiprocessing as mp
    os.makedirs(_dir("cfg", col), exist_ok=True)
    apps = T.staged()
    jobs = []
    for f, s in apps:
        jobs.append((f, s, col))
    with mp.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for i, err in enumerate(pool.imap_unordered(_cfg_job, jobs)):
            if err:
                print("  skip", err, flush=True)
            if (i + 1) % 25 == 0:
                print(f"  cfg {col}: {i + 1}/{len(apps)}", flush=True)
    print(f"cfg {col}: {len(os.listdir(_dir('cfg', col)))} apps", flush=True)


def load(graph, col, fams=None):
    """{sha: (fam, su, sc, hu, hc)} in staged order, optionally restricted to families."""
    d = _dir(graph, col)
    out = {}
    for fam, sha in T.staged():
        if fams and fam not in fams:
            continue
        p = os.path.join(d, f"{sha}.npz")
        if os.path.exists(p):
            z = np.load(p)
            out[sha] = (str(z["fam"]), z["su"], z["sc"], z["hu"], z["hc"])
    return out


if __name__ == "__main__":
    g = sys.argv[1]
    c = sys.argv[2]
    if g == "fcg":
        build_fcg(c)
    else:
        build_cfg(c)
