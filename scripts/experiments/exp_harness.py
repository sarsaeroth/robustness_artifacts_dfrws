#!/usr/bin/env python3
"""Shared data, folds and statistics for the Tier 4 robustness experiments.

Every experiment draws samples, labels, DEX groups and folds from here, so
results across representations and models are paired on identical splits.
Vectors come from results/robust2/vectors (packed from the bundle by
pack_vectors_from_bundle.py). The one dual-provider app is excluded, as in
regen_tables.py.

Datasets: env DATASET selects the corpus (default "proxy", which reproduces every
earlier result unchanged). Other datasets are vectorized by vectorize_dataset.py into
results/datasets/{ds}/vectors. "four" and "five" mean closed world (target families
only) and open world (targets plus the dataset's benign class). Env LEAK_GROUPS selects the
leakage grouping: "dex" (proxy default), or "neardup" (same package name, or labeled
FCG cosine >= 0.99, joined transitively), the scheme available for every dataset.
Env GRAPHS=fcg restricts every representation to its FCG block.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_zoo as mz
import graph_io as GI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.environ.get("DATASET", "proxy")
SPECS = {
    "proxy":  {"vec": os.path.join(ROOT, "results", "robust2", "vectors"),
               "targets": ["ipninja", "luminati", "monkeysocks", "oxylabs"],
               "benign": "non_proxy", "groups": "dex", "graphs": ("cfg", "fcg"),
               "graphs_dir": os.path.join(ROOT, "work", "graphs40"),
               "full_graphs": ("zip", os.path.join(ROOT, "bundles", "AndMal1_data.zip"), "AndMal1_data/corpus")},
    "malnet": {"vec": os.path.join(ROOT, "results", "datasets", "malnet", "vectors"),
               "targets": None, "benign": "benign", "groups": "neardup", "graphs": ("fcg",),
               "graphs_dir": os.path.join(ROOT, "datasets", "malnet"),
               "full_graphs": ("dir", os.path.join(ROOT, "datasets", "malnet"), None)},
    # Piggybacked pairs (Li et al., lilicoding/Piggybacking): binary original vs piggybacked;
    # each pair shares a package name, so package grouping keeps pairs in one fold.
    "piggyback": {"vec": os.path.join(ROOT, "results", "datasets", "piggyback", "vectors"),
                  "targets": ["original", "piggybacked"], "benign": "original", "groups": "neardup",
                  "graphs": ("fcg",), "graphs_dir": os.path.join(ROOT, "datasets", "piggyback"),
                  "full_graphs": ("dir", os.path.join(ROOT, "datasets", "piggyback"), None)},
    # #1751 corpus v3 (datasets/corpus_v3/manifest.jsonl): validated DEX pivots + version builds +
    # C2-domain-pivot providers; v2 corpus apps keep their bundle graphs/vectors, new apps are staged
    # in work/staged_v3 (stage_local_apks.py). Packed by pack_proxy_v3.py; DEX groups dex_groups_v3.py.
    "proxy_v3": {"vec": os.path.join(ROOT, "results", "datasets", "proxy_v3", "vectors"),
                 "targets": ["cashraven", "ipidea", "ipninja", "luminati", "monkeysocks", "oxylabs", "popa",
                             "traffmonetizer"],
                 "benign": "non_proxy", "groups": "dex", "graphs": ("cfg", "fcg"),
                 "graphs_dir": os.path.join(ROOT, "work", "staged_v3"),
                 "full_graphs": ("zip", os.path.join(ROOT, "bundles", "AndMal1_data.zip"), "AndMal1_data/corpus")},
    # CCCS-CIC-AndMal2020 category sample (adware, riskware, pua, trojan) + benign, via AndroZoo
    "cic_andmal": {"vec": os.path.join(ROOT, "results", "datasets", "cic_andmal", "vectors"),
                   "targets": None, "benign": "benign", "groups": "neardup", "graphs": ("fcg",),
                   "graphs_dir": os.path.join(ROOT, "datasets", "cic_andmal"),
                   "full_graphs": ("dir", os.path.join(ROOT, "datasets", "cic_andmal"), None)},
}
# v2-comparison view of corpus v3: the same vectors, folds over the four original providers only
# (clean-label counterpart of the v2 tables; the corrected labels and new builds are the only change).
SPECS["proxy_v3_four"] = dict(SPECS["proxy_v3"], targets=["ipninja", "luminati", "monkeysocks", "oxylabs"])
V3 = DATASET.startswith("proxy_v3")
SPEC = SPECS[DATASET]
VEC = SPEC["vec"]
GROUPS = os.environ.get("LEAK_GROUPS", SPEC["groups"])
GRAPHS = tuple(os.environ.get("GRAPHS", ",".join(SPEC["graphs"])).split(","))
BENIGN = SPEC["benign"]
# Staged graphs ({GRAPHS_DIR}/{family}/{sha}/{sha}.{fcg,cfg}) for experiments that re-extract,
# perturb or localize; staged(fam) lists the staged apps (for datasets other than proxy, the
# staged.json subset of the dataset directory). PROXY is the list of staged target families (the four proxy families
# for the proxy corpus; for other datasets, read from results/datasets/{ds}/staged.json).
GRAPHS_DIR = SPEC["graphs_dir"]
PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
# both proxy_v3 views share proxy_v3's staged.json
if V3:
    _st = os.path.join(ROOT, "results", "datasets", "proxy_v3", "staged.json")
else:
    _st = os.path.join(ROOT, "results", "datasets", DATASET, "staged.json")
if DATASET != "proxy" and os.path.exists(_st):
    _sj = json.load(open(_st))
else:
    _sj = None
if V3:
    PROXY = SPEC["targets"]
elif DATASET != "proxy":
    if not _sj:
        _sj = {"families": [], "payload_families": [], "apps": {}}
    # STAGED_SET=payload restricts staged experiments to families with payload ground truth
    # (localization, explainers, identifiability); attacks and certificates use all staged families.
    if os.environ.get("STAGED_SET") == "payload":
        PROXY = _sj["payload_families"]
    else:
        PROXY = _sj["families"]
    if len(PROXY) < 2:  # e.g. piggyback: one payload family; classifiers need both classes, and localization
        PROXY = _sj["families"]  # metrics still use ground truth only where it exists (originals have none)


def staged(fam):
    """Sorted staged apps of a family (staged.json if loaded, else subdirectories of GRAPHS_DIR); [] if none."""
    if _sj is not None:
        if "apps" in _sj:
            apps = _sj["apps"]
        else:
            apps = {}
        if fam in apps:
            return sorted(apps[fam])
        return sorted([])
    d = os.path.join(GRAPHS_DIR, fam)
    if os.path.isdir(d):
        return sorted(os.listdir(d))
    return []


# SMOKE_DIR redirects experiment outputs and caches (out_path, data_dir) to a scratch tree, so that a
# smoke test of a queue never writes to results/ (inputs such as vectors are still read from results/)
RESULTS = os.environ.get("SMOKE_DIR") or os.path.join(ROOT, "results")


def out_path(*parts):
    """Results path: results/<parts> for the proxy default, else results/datasets/<tag>/<parts>."""
    tag = DATASET
    if os.environ.get("STAGED_SET") == "payload" and DATASET != "proxy":
        tag += "_payload"
    if GROUPS != SPEC["groups"]:
        tag += f"_g{GROUPS}"
    if GRAPHS != SPEC["graphs"]:
        tag += "_" + "+".join(GRAPHS)
    if tag == "proxy":
        p = os.path.join(RESULTS, *parts)
    else:
        p = os.path.join(RESULTS, "datasets", tag, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def data_dir(*parts):
    """Per-dataset cache directory, independent of grouping and graph settings."""
    if DATASET == "proxy":
        p = os.path.join(RESULTS, *parts)
    else:
        p = os.path.join(RESULTS, "datasets", DATASET, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def wild_flagged():
    """Open-world negatives flagged by the marker rescan (results/ledger/rescan_wild.jsonl); dropped from negative sets."""
    p = os.path.join(ROOT, "results", "ledger", "rescan_wild.jsonl")
    if not os.path.exists(p):
        return set()
    flagged = set()
    for line in open(p):
        r = json.loads(line)
        if r.get("found"):
            flagged.add(r["sha256"])
    return flagged


def out_dir(*parts):
    p = out_path(*parts)
    os.makedirs(p, exist_ok=True)
    return p

# representation name -> packed vector files concatenated in order
REPS = {
    "labeled":   ["cfg_wl_1024", "fcg_wl_1024"],
    "combined":  ["cfg_wlall_1024", "fcg_wlall_1024"],
    "fuzzy":     ["cfg_wlfz_1024", "fcg_wlfz_1024"],
    "api":       ["cfg_wlapi_1024", "fcg_wlapi_1024"],
    "structure": ["cfg_wls_1024", "fcg_wls_1024"],
    "renamed":   ["cfg_wlren_1024", "fcg_wlren_1024"],
    "node2vec":  ["cfg_n2v_1024", "fcg_n2v_1024"],
    "dex":       ["dex_bytecode_1024"],
    "labeled_fcg": ["fcg_wl_1024"],
    "mama_family": ["mama_family"],
    "mama_package": ["mama_package"],
    "drebin": ["drebin"],
    "drebin_code": ["drebin_code"],
}

_cache = {}


def _load(ext):
    if ext not in _cache:
        f = os.path.join(VEC, ext + ".npz")
        if not os.path.exists(f):
            raise KeyError(f"{ext}: no packed vectors for dataset {DATASET}")
        z = np.load(f)
        # id -> row index
        id_to_row = {}
        for k, i in enumerate(z["ids"]):
            id_to_row[i] = k
        _cache[ext] = id_to_row, z["family"], z["X"]
    return _cache[ext]


def excluded_ids():
    if DATASET != "proxy":
        return set()
    multi = set()
    with open(os.path.join(ROOT, "datasets", "final_proxy_apks_updated.json")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if len(r.get("proxy_providers") or []) > 1:
                    multi.add(r["id"])
    return multi


def sample_ids(setting="four"):
    """Sorted (ids, families) of the targets ("four") or targets + benign ("five") in the labeled-WL set."""
    idx, fam, _ = _load(f"{GRAPHS[0]}_wl_1024")
    bad = excluded_ids()
    if SPEC["targets"]:
        targets = SPEC["targets"]
    else:
        targets = sorted(set(fam.tolist()) - {BENIGN})
    # copy so SPEC["targets"] is never mutated
    keep = list(targets)
    if setting == "five":
        keep.append(BENIGN)
    ids = []
    for i, k in idx.items():
        if fam[k] in keep and i not in bad:
            ids.append(i)
    ids = sorted(ids)
    id_fams = []
    for i in ids:
        id_fams.append(fam[idx[i]])
    return np.array(ids), np.array(id_fams)


def available(rep, ids, fam):
    """(ids, fam) restricted to apps present in every vector block of rep; logs how many were dropped."""
    exts = []
    for e in REPS[rep]:
        if e.split("_")[0] in GRAPHS or not e.startswith(("cfg_", "fcg_")):
            exts.append(e)
    keep = np.ones(len(ids), bool)
    for ext in exts:
        idx = _load(ext)[0]
        present = []
        for i in ids:
            present.append(i in idx)
        keep &= np.array(present)
    if not keep.all():
        print(f"  {rep}: {int((~keep).sum())} of {len(ids)} apps lack a {rep} vector; dropped for this representation",
              flush=True)
    return ids[keep], fam[keep]


def matrix(rep, ids):
    """Feature matrix of rep aligned to ids; raises KeyError if any id is missing."""
    parts = []
    exts = []
    for e in REPS[rep]:
        if e.split("_")[0] in GRAPHS or not e.startswith(("cfg_", "fcg_")):
            exts.append(e)
    if not exts:
        raise KeyError(f"{rep}: no block for graphs {GRAPHS}")
    for ext in exts:
        idx, _, X = _load(ext)
        missing = []
        rows = []
        for i in ids:
            if i not in idx:
                missing.append(i)
        if missing:
            raise KeyError(f"{rep}/{ext}: {len(missing)} ids missing")
        for i in ids:
            rows.append(idx[i])
        parts.append(X[rows])
    return np.hstack(parts).astype(np.float32)


def groups_for(ids):
    if GROUPS == "dex":
        if V3:
            gf = "results/corpus_v3/dex_groups.json"
        else:
            gf = "results/full_sweep/dex_groups.json"
        g = json.load(open(os.path.join(ROOT, gf)))
        return mz.dex_group_ids(list(ids), g)
    return neardup_groups(ids)


NEARDUP_COS = 0.99


def _packages():
    if DATASET == "proxy" or V3:
        pk = {}
        if V3:
            for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
                r = json.loads(l)
                if r.get("package"):
                    pk[r["sha256"]] = r["package"]
        for f in ("final_proxy_apks_updated.json", "open_world_dataset.json"):
            p = os.path.join(ROOT, "datasets", f)
            if not os.path.exists(p):
                continue
            txt = open(p).read().strip()
            if txt.startswith("["):
                recs = json.loads(txt)
            else:
                recs = []
                for l in txt.splitlines():
                    if l.strip():
                        recs.append(json.loads(l))
            if isinstance(recs, dict):
                recs = list(recs.values())
            for r in recs:
                if r.get("id"):
                    sid = r.get("id").lower()
                elif r.get("sha256"):
                    sid = r.get("sha256").lower()
                else:
                    sid = ""
                if sid and r.get("pkg_id"):
                    pk[sid] = r["pkg_id"]
        return pk
    meta = json.load(open(os.path.join(ROOT, "results", "datasets", DATASET, "meta.json")))
    pk = {}
    for k, v in meta.items():
        pk[k] = v["pkg"]
    return pk


def neardup_groups(ids):
    """Connected components of (same package name) OR (labeled FCG cosine >= NEARDUP_COS)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    idx, _, X = _load("fcg_wl_1024")
    rows_of_ids = []
    for i in ids:
        rows_of_ids.append(idx[i])
    V = X[rows_of_ids].astype(np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    rows, cols = [], []
    for a in range(0, len(V), 1024):
        S = V[a:a + 1024] @ V.T
        r, c = np.nonzero(S >= NEARDUP_COS)
        rows.append(r + a)
        cols.append(c)
    pk = _packages()
    by = defaultdict(list)
    for k, i in enumerate(ids):
        if i.lower() in pk:
            by[pk[i.lower()]].append(k)
    for ks in by.values():
        rows.append(np.array(ks[1:], dtype=np.int64))
        cols.append(np.full(len(ks) - 1, ks[0]))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(ids), len(ids)))
    _, component = connected_components(A, directed=False)
    return component.astype(np.int64)


def labels(fam):
    le = LabelEncoder().fit(fam)
    return le.transform(fam), le


def folds(y, groups, repeats, n_splits=5):
    """[(rep, fold, train_idx, test_idx)], shuffled StratifiedGroupKFold per repeat."""
    out = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            out.append((r, f, tr, te))
    return out


def upsample(tr, y, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for j in tr:
        by[y[j]].append(j)
    sizes = []
    for v in by.values():
        sizes.append(len(v))
    top = max(sizes)
    out = []
    for v in by.values():
        out.extend(v)
        if top > len(v):
            out.extend(rng.choices(v, k=top - len(v)))
    return np.array(out)


def corrected_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on paired per-fold diffs; returns (mean diff, t, two-sided p)."""
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    if J < 2 or d.var(ddof=1) == 0:
        return float(d.mean()), float("nan"), float("nan")
    var = (1.0 / J + n_test / n_train) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(d.mean()), float(t), float(p)

# full-corpus graphs
_zip = None


def _member(fam, sha, name):
    """Bytes of {fam}/{sha}/{name} from the dataset's full graph store, or None."""
    global _zip
    kind, where, prefix = SPEC["full_graphs"]
    if kind == "dir":
        p = os.path.join(where, fam, sha, name)
        if not GI.exists(p):
            return None
        with GI.open_binary(p) as fh:
            return fh.read()
    import gzip
    import zipfile
    if _zip is None:
        _zip = zipfile.ZipFile(where)
    try:
        return gzip.decompress(_zip.read(f"{prefix}/{fam}/{sha}/{name}.gz"))
    except KeyError:
        return None


def load_graph(fam, sha, kind="fcg"):
    """(node_labels, successors, predecessors) from the staged copy, else the full store; None if absent."""
    import tempfile
    import gen_wl_vectors as G
    p = os.path.join(GRAPHS_DIR, fam, sha, f"{sha}.{kind}")
    if GI.exists(p):
        return G.parse_gml_lightweight(p)
    if SPEC["full_graphs"][0] == "dir":  # parse straight from the store; no need to hold the bytes
        q = os.path.join(SPEC["full_graphs"][1], fam, sha, f"{sha}.{kind}")
        if GI.exists(q):
            return G.parse_gml_lightweight(q)
        return None
    raw = _member(fam, sha, f"{sha}.{kind}")
    if raw is None:
        return None
    tmpdir = os.path.join(ROOT, "results", "tmp_graphs")
    os.makedirs(tmpdir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=f".{kind}", delete=True) as f:
        f.write(raw)
        f.flush()
        return G.parse_gml_lightweight(f.name)


def load_opcodes(fam, sha):
    p = os.path.join(GRAPHS_DIR, fam, sha, f"{sha}.method_opcodes.json")
    if os.path.exists(p):
        return json.load(open(p))
    raw = _member(fam, sha, f"{sha}.method_opcodes.json")
    if raw is not None:
        return json.loads(raw)
    return None


# binary tasks
def _expand_binary_coef():
    # Two-class linear models get one weight row per class: coef_ = [-w/2, w/2], intercept_ = [-b/2, b/2].
    # The row difference is w, so margins, argmax predictions and probabilities (sigmoid) are unchanged.
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.svm import LinearSVC
    for cls in (SGDClassifier, LogisticRegression, LinearSVC):
        if getattr(cls, "_binary_expanded", False):
            continue
        orig = cls.fit

        def fit(self, *a, _orig=orig, **k):
            r = _orig(self, *a, **k)
            if getattr(self, "coef_", None) is not None and self.coef_.shape[0] == 1 and len(self.classes_) == 2:
                w, b = self.coef_[0], self.intercept_[0]
                self.coef_ = np.vstack([-w / 2, w / 2])
                self.intercept_ = np.array([-b / 2, b / 2])
            return r
        cls.fit = fit
        if hasattr(cls, "predict_proba"):
            orig_pp = cls.predict_proba

            def predict_proba(self, X, _orig=orig_pp):
                if getattr(self, "coef_", None) is not None and self.coef_.shape[0] == 2 and len(self.classes_) == 2:
                    z = X @ (self.coef_[1] - self.coef_[0]) + (self.intercept_[1] - self.intercept_[0])
                    z = np.asarray(z, dtype=float).ravel()
                    p1 = 1.0 / (1.0 + np.exp(-z))
                    return np.column_stack([1 - p1, p1])
                return _orig(self, X)
            cls.predict_proba = predict_proba
        cls._binary_expanded = True


if SPEC["targets"] is not None and len(SPEC["targets"]) == 2:
    _expand_binary_coef()


def with_cfg_fcg(ids, fam):
    """(ids, fam) restricted to apps with both packed CFG and FCG WL vectors (missing blocks are skipped)."""
    keep = np.ones(len(ids), dtype=bool)
    for name in ("cfg_wl_1024", "fcg_wl_1024"):
        try:
            idx, _, _ = _load(name)
        except KeyError:
            continue
        present = []
        for i in ids:
            present.append(i in idx)
        keep &= np.array(present)
    if not keep.all():
        print(f"  with_cfg_fcg: dropping {int((~keep).sum())} app(s) without both CFG and FCG vectors", flush=True)
    return ids[keep], fam[keep]
