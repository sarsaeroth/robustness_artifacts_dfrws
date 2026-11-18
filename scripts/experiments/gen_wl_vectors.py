#!/usr/bin/env python3
"""Pre-generate WL hashed vectors at multiple dimensions for all samples.

Uses a lightweight streaming GML parser instead of NetworkX.
No dependency on ml_compare (avoids loading NetworkX/sklearn at import time).
"""

import gc
import hashlib
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import graph_io as GI

DIMS = [128, 256, 512, 1024]
KINDS = ["cfg", "fcg"]
FAMILIES = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]

LIB_PREFIXES = (
    "Ljava/", "Ljavax/", "Landroid/", "Landroidx/", "Lkotlin/",
    "Lcom/google/", "Lorg/apache/", "Lorg/json/", "Lokhttp", "Lokio",
)


def stable_hash_u32(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def save_vec(path: str, vec: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, vec)


def load_vec(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return np.load(f, allow_pickle=False)
    except Exception:
        return None


def parse_gml_lightweight(path):
    """Parse GML file into lightweight adjacency structure."""
    node_labels = {}
    successors = defaultdict(list)
    predecessors = defaultdict(list)

    current_node_id = None
    current_node_method = None
    current_node_label = None
    current_edge_source = None
    current_edge_target = None
    in_node = False
    in_edge = False

    with GI.open_text(path) as f:
        for line in f:
            stripped = line.strip()

            if stripped == "node [":
                in_node = True
                current_node_id = None
                current_node_method = None
                current_node_label = None
            elif stripped == "edge [":
                in_edge = True
                current_edge_source = None
                current_edge_target = None
            elif stripped == "]":
                if in_node:
                    if current_node_id is not None:
                        lbl = current_node_method or current_node_label or str(current_node_id)
                        node_labels[current_node_id] = lbl
                    in_node = False
                elif in_edge:
                    if current_edge_source is not None and current_edge_target is not None:
                        successors[current_edge_source].append(current_edge_target)
                        predecessors[current_edge_target].append(current_edge_source)
                    in_edge = False
            elif in_node:
                if stripped.startswith("id "):
                    current_node_id = int(stripped[3:])
                elif stripped.startswith("label "):
                    current_node_label = stripped[7:-1]
                elif stripped.startswith("method "):
                    current_node_method = stripped[8:-1]
            elif in_edge:
                if stripped.startswith("source "):
                    current_edge_source = int(stripped[7:])
                elif stripped.startswith("target "):
                    current_edge_target = int(stripped[7:])

    return node_labels, successors, predecessors


def canonical_label(label, kind):
    if kind == "cfg" and ":" in label:
        left, right = label.rsplit(":", 1)
        if right.isdigit():
            return left
    return label


def wl_hashed_vector(node_labels, successors, predecessors, kind, dim,
                     iters=2, drop_lib=True, use_direction=True):
    """WL hashed vector using lightweight data structures (no LCC)."""
    if not node_labels:
        return np.zeros(dim, dtype=np.float32)

    # Canonical labels
    labels = {}
    for nid, raw in node_labels.items():
        labels[nid] = canonical_label(raw, kind)

    # Drop library nodes
    if drop_lib:
        to_drop = []
        for n, lbl in labels.items():
            if lbl.startswith(LIB_PREFIXES):
                to_drop.append(n)
        if to_drop and len(to_drop) < len(labels):
            for n in to_drop:
                del labels[n]

    if not labels:
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)

    # Iteration 0
    for n, lbl in labels.items():
        h = stable_hash_u32(f"0|{lbl}")
        idx = h % dim
        if (h & 0x80000000) == 0:
            sign = 1.0
        else:
            sign = -1.0
        vec[idx] += sign

    # WL iterations
    for i in range(1, iters + 1):
        new_labels = {}
        for n in labels:
            lbl = labels[n]
            if use_direction:
                in_l = []
                for p in predecessors.get(n, []):
                    if p in labels:
                        in_l.append(labels[p])
                in_l.sort()
                out_l = []
                for s in successors.get(n, []):
                    if s in labels:
                        out_l.append(labels[s])
                out_l.sort()
                new_labels[n] = wl_digest(lbl, in_l, out_l)
            else:
                nb_set = set(successors.get(n, []))
                for x in predecessors.get(n, []):
                    nb_set.add(x)
                nb = []
                for x in nb_set:
                    if x in labels:
                        nb.append(labels[x])
                nb.sort()
                raw = f"{lbl}|[" + ",".join(nb) + "]"
                new_labels[n] = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]

        labels = new_labels
        for n, lbl in labels.items():
            h = stable_hash_u32(f"{i}|{lbl}")
            idx = h % dim
            if (h & 0x80000000) == 0:
                sign = 1.0
            else:
                sign = -1.0
            vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def main():
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets")
    global FAMILIES
    # families are whatever the dataset root contains (falls back to the default list)
    default = ["ipninja", "luminati", "monkeysocks", "oxylabs", "non_proxy"]
    try:
        found = []
        for d in sorted(os.listdir(str(dataset_root))):
            if os.path.isdir(os.path.join(str(dataset_root), d)) and not d.startswith("_"):
                found.append(d)
        if found:
            FAMILIES = found
        else:
            FAMILIES = default
    except OSError:
        FAMILIES = default

    samples = []
    for family in FAMILIES:
        fam_dir = dataset_root / family
        if not fam_dir.exists():
            continue
        for d in sorted(fam_dir.iterdir()):
            if d.is_dir():
                samples.append(d)

    print(f"Found {len(samples)} sample directories", flush=True)

    for kind in KINDS:
        # Sort by graph file size (small first)
        size_of = {}
        for s in samples:
            p = s / f"{s.name}.{kind}"
            if GI.exists(str(p)):
                size_of[s] = GI.size(str(p))
            else:
                size_of[s] = 0
        samples_sorted = sorted(samples, key=lambda s: size_of[s])

        print(f"\n{'='*50}", flush=True)
        print(f"Processing {kind.upper()} graphs ({len(samples_sorted)} samples)", flush=True)
        print(f"{'='*50}", flush=True)
        done = 0
        skipped = 0
        failed = 0
        t0 = time()

        for i, sdir in enumerate(samples_sorted):
            gid = sdir.name
            graph_path = sdir / f"{gid}.{kind}"

            needed_dims = []
            for dim in DIMS:
                out_path = sdir / f"{gid}.{kind}_wl_{dim}"
                if out_path.exists():
                    v = load_vec(str(out_path))
                    if v is not None and v.shape[0] == dim:
                        continue
                needed_dims.append(dim)

            if not needed_dims:
                skipped += 1
                continue

            if not GI.exists(str(graph_path)):
                failed += 1
                continue

            size_mb = GI.size(str(graph_path)) / (1024 * 1024)
            try:
                node_labels, succs, preds = parse_gml_lightweight(str(graph_path))
            except Exception as e:
                print(f"  [!] {gid[:12]}... parse failed: {e}", flush=True)
                failed += 1
                continue

            for dim in needed_dims:
                out_path = sdir / f"{gid}.{kind}_wl_{dim}"
                v = wl_hashed_vector(node_labels, succs, preds, kind=kind, dim=dim)
                save_vec(str(out_path), v)

            done += 1
            del node_labels, succs, preds
            gc.collect()

            if done % 100 == 0 or size_mb > 50:
                elapsed = time() - t0
                if size_mb > 10:
                    last = f" [last: {size_mb:.0f}MB]"
                else:
                    last = ""
                print(f"  [{i+1}/{len(samples_sorted)}] {done} gen, {skipped} cached, "
                      f"{failed} fail ({elapsed:.0f}s)" + last, flush=True)

        elapsed = time() - t0
        print(f"  Done: {done} generated, {skipped} cached, {failed} failed "
              f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
