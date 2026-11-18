#!/usr/bin/env python3
"""Add combined-coloring FCG vectors (fcg_wlall_1024) for the proxy corpus's non_proxy class.

The 996 benign applications were packed without opcode maps, so the combined coloring had no vectors for them and
open-world analyses of that coloring could not train a benign class. fetch_dataset_apks.py re-fetches them by hash
into ROOT/non_proxy/<sha>/ (FCG + method opcodes); this computes the combined vector exactly as vectorize_dataset.py
does (gen_wl_hardened combined colors) and merges them into results/datasets/proxy_v3/vectors/fcg_wlall_1024.npz
(previous file kept as .pre_nonproxy.npz).
Usage: add_nonproxy_combined.py ROOT
"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_wl_vectors as G
import gen_wl_hardened as HD
import graph_io as GI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "results", "datasets", "proxy_v3", "vectors", "fcg_wlall_1024.npz")


def main():
    # load the existing pack
    base = os.path.join(sys.argv[1], "non_proxy")
    z = np.load(PACK, allow_pickle=True)
    ids = list(z["ids"])
    fam = list(z["family"])
    X = list(z["X"])
    have = set(ids)
    added = 0

    # vectorize each fetched non_proxy app not already in the pack
    for sha in sorted(os.listdir(base)):
        d = os.path.join(base, sha)
        fcg = os.path.join(d, f"{sha}.fcg")
        opf = os.path.join(d, f"{sha}.method_opcodes.json")
        if sha in have or not GI.exists(fcg) or not os.path.exists(opf):
            continue
        fcg_path = GI.resolve(fcg)
        nl, su, pr = G.parse_gml_lightweight(fcg_path)
        opcodes = json.load(open(opf))
        colors = HD.build_colors("combined", nl, su, "fcg", opcodes)
        v = HD.wl_vector(colors, su, pr)
        ids.append(sha)
        fam.append("non_proxy")
        X.append(np.asarray(v, dtype=np.float32))
        added += 1

    # merge, keeping the previous pack
    if added:
        shutil.copy(PACK, PACK.replace(".npz", ".pre_nonproxy.npz"))
        np.savez_compressed(PACK, ids=np.array(ids), family=np.array(fam), X=np.stack(X))
    print(f"added {added} non_proxy combined vectors -> {PACK}", flush=True)


if __name__ == "__main__":
    main()
