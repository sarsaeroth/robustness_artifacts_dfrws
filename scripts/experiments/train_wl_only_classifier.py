#!/usr/bin/env python3
"""
train_wl_only_classifier.py - Open-world classifier using only WL vectors.

No W2V/n2v models needed since WL is deterministic hashing. This means we can
generate vectors for new APKs without retraining embedding models.

Config: cfg_wl + fcg_wl, 1024-dim, SGD, upsample, 6 classes.

Usage:
    python3 scripts/train_wl_only_classifier.py
"""
import json
import os
import random
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DIM = 1024
VTYPES = ["cfg_wl", "fcg_wl"]  # WL only - no embedding model needed
VTYPE_EXT = {"cfg_wl": ".cfg_wl_{dim}", "fcg_wl": ".fcg_wl_{dim}"}

PROXY_FAMILIES = ["luminati", "monkeysocks", "oxylabs", "ipninja"]
LABEL_MAP = {
    "luminati.io": "luminati",
    "monkeysocks.net": "monkeysocks",
    "oxylabs.io": "oxylabs",
    "ipninja.io": "ipninja",
}

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
PROXY_JSON = os.path.join(DATASET_ROOT, "final_proxy_apks_updated.json")
BENIGN_JSON = os.path.join(DATASET_ROOT, "open_world_benign.json")
MALWARE_JSON = os.path.join(DATASET_ROOT, "open_world_malware.json")

SEED = 42
N_FOLDS = 5


def upsample(idx_labels, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for idx, lbl in idx_labels:
        if lbl not in by_label:
            by_label[lbl] = []
        by_label[lbl].append(idx)
    counts = []
    for v in by_label.values():
        counts.append(len(v))
    max_count = max(counts)
    # every original index, then random repeats up to the largest class
    out = []
    for lbl, items in by_label.items():
        for i in items:
            out.append((i, lbl))
        deficit = max_count - len(items)
        if deficit > 0:
            for _ in range(deficit):
                out.append((rng.choice(items), lbl))
    return out


def build_samples():
    samples = []
    with open(PROXY_JSON) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            gid = r["id"]
            # first listed provider, None when the list is missing or empty
            providers = r.get("proxy_providers")
            if providers:
                raw = providers[0]
            else:
                raw = None
            if raw in LABEL_MAP:
                label = LABEL_MAP[raw]
            else:
                label = raw
            if label not in PROXY_FAMILIES:
                continue
            hdir = os.path.join(DATASET_ROOT, label, gid)
            if os.path.isdir(hdir):
                samples.append((hdir, gid, label))

    # Combine benign and malware into a single "non_proxy" class
    for json_path in [BENIGN_JSON, MALWARE_JSON]:
        with open(json_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gid = r["sha256"].lower()
                hdir = os.path.join(DATASET_ROOT, "non_proxy", gid)
                if os.path.isdir(hdir):
                    samples.append((hdir, gid, "non_proxy"))
    return samples


def new_clf():
    return SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-4,
        random_state=SEED,
        n_jobs=-1,
    )


def main():
    print("[*] Building sample list...", flush=True)
    samples = build_samples()
    print(f"    {len(samples)} samples", flush=True)

    print(f"\n[*] Loading WL vectors (dim={DIM})...", flush=True)
    X_rows = []
    y_rows = []
    skipped = 0
    for hdir, gid, label in samples:
        # cfg_wl + fcg_wl vectors concatenated; skip the sample if either is missing or malformed
        parts = []
        complete = True
        for vt in VTYPES:
            path = os.path.join(hdir, f"{gid}{VTYPE_EXT[vt].format(dim=DIM)}")
            v = None
            if os.path.isfile(path):
                try:
                    loaded = np.load(path, allow_pickle=False)
                    if loaded.ndim == 1:
                        v = loaded.astype(np.float32, copy=False)
                except Exception:
                    pass
            if v is None or v.shape[0] != DIM:
                complete = False
                break
            parts.append(v)
        if not complete:
            skipped += 1
            continue
        v = np.concatenate(parts)
        X_rows.append(v)
        y_rows.append(label)

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"    X shape: {X.shape} (skipped {skipped})", flush=True)
    print(f"    Labels: {dict(Counter(y))}", flush=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n[*] {N_FOLDS}-fold stratified CV...", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fm = {"acc": [], "mf1": [], "wf1": []}
    ys_true = []
    ys_pred = []

    for i, (tr, te) in enumerate(skf.split(X, y_enc)):
        pairs = []
        for j in tr:
            pairs.append((j, y_enc[j]))
        up = upsample(pairs, seed=SEED + i)
        up_list = []
        for j, _ in up:
            up_list.append(j)
        up_idx = np.array(up_list)
        clf = new_clf()
        clf.fit(X[up_idx], y_enc[up_idx])
        y_pred = clf.predict(X[te])
        y_true = y_enc[te]
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        fm["acc"].append(acc)
        fm["mf1"].append(mf1)
        fm["wf1"].append(wf1)
        ys_true.extend(y_true)
        ys_pred.extend(y_pred)
        print(f"    Fold {i+1}: acc={acc:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}", flush=True)

    def stat(a):
        a = np.array(a)
        return float(a.mean()), float(a.std())
    am, as_ = stat(fm["acc"])
    mm, ms = stat(fm["mf1"])
    wm, ws = stat(fm["wf1"])
    print(f"\n    === CV Summary ===", flush=True)
    print(f"    Accuracy:    {am:.4f} ± {as_:.4f}", flush=True)
    print(f"    Macro F1:    {mm:.4f} ± {ms:.4f}", flush=True)
    print(f"    Weighted F1: {wm:.4f} ± {ws:.4f}", flush=True)

    print(f"\n[*] Per-class report:", flush=True)
    print(classification_report(ys_true, ys_pred, target_names=list(le.classes_), zero_division=0), flush=True)

    cm = confusion_matrix(ys_true, ys_pred)
    print(f"[*] Confusion matrix (rows=true, cols=pred):", flush=True)
    header_cells = []
    for c in le.classes_:
        header_cells.append(f"{c[:8]:>8}")
    print("    " + "  ".join(header_cells), flush=True)
    for i, row in enumerate(cm):
        row_cells = []
        for v in row:
            row_cells.append(f"{v:>8d}")
        print(f"    {le.classes_[i][:8]:>8}  " + "  ".join(row_cells), flush=True)

    print(f"\n[*] Training final model...", flush=True)
    pairs = []
    for j in range(len(X)):
        pairs.append((j, y_enc[j]))
    up_all = upsample(pairs, seed=SEED)
    up_list = []
    for j, _ in up_all:
        up_list.append(j)
    up_idx = np.array(up_list)
    clf = new_clf()
    clf.fit(X[up_idx], y_enc[up_idx])

    model_path = os.path.join(PROJECT_ROOT, "models", "wl_only_classifier.joblib")
    meta_path = os.path.join(PROJECT_ROOT, "models", "wl_only_classifier_meta.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"clf": clf, "label_encoder": le}, model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "vtypes": VTYPES, "dim": DIM, "feature_dim": int(X.shape[1]),
            "labels": list(le.classes_), "n_unique_samples": int(X.shape[0]),
            "n_train_samples": int(up_idx.shape[0]),
            "balance": "upsample", "classifier": "sgd",
            "cv": {
                "n_folds": N_FOLDS,
                "acc_mean": am, "acc_std": as_,
                "macro_f1_mean": mm, "macro_f1_std": ms,
                "weighted_f1_mean": wm, "weighted_f1_std": ws,
            },
            "classes_in_cv": list(le.classes_),
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)
    print(f"[+] Saved: {model_path}", flush=True)
    print(f"[+] Saved: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
