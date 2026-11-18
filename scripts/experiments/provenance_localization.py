#!/usr/bin/env python3
"""Localization by the method the paper actually describes: provenance-tracked,
fractionally-distributed, cross-sample-aggregated attribution.

The paper states that we retain provenance for each WL contribution (graph type,
iteration, bucket, sign, originating method), distribute each bucket's
attribution fractionally among its recorded contributors WITHIN each sample, and
aggregate across samples. The published precision@k figures were in fact produced
by a document-frequency ranking (scripts/discriminative_methods.py), which is a
property of the corpus rather than of the model. This implements the described
pipeline so the reported numbers match the stated method.

Attribution is exact for a linear model: for SGD the SHAP value of feature j is
    phi_j = coef[family][j] * (x_j - E[x_j])
which is what LinearExplainer computes. No sampling, so the result is
deterministic given the seed.

Within a sample, bucket j received contributions from tokens t (each +/-1 before
L2 normalization). We split phi_j across those tokens in proportion to their
signed contribution, then credit each token's originating method.

Comparison baseline is the document-frequency ranking over the SAME samples, so
the two are directly comparable (both FCG, same subsample, same SDK ground truth).

Outputs: results/provenance_localization.json
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import gen_wl_vectors as G
import train_wl_only_classifier as T
import graph_io as GI

PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
MODEL = os.path.join(T.PROJECT_ROOT, "models", "proxy_4class_fcg.joblib")
OUT = os.path.join(T.PROJECT_ROOT, "results", "provenance_localization.json")
if os.environ.get("PROVENANCE_SPLIT", "exact") != "legacy":
    OUT = OUT.replace(".json", "_exactsplit.json")
DIM = 1024
SEED = 42
PER_FAM = 25
ITERS = 2
MAX_FCG = 50_000_000
TOP_N = [25, 50, 100]

# authoritative SDK namespaces (scripts/decompile_targets.py, first-hand decompilation)
SDK_PREFIXES = {
    # Lcom/ninjadroid/ and Lcom/holly/marge/ removed: calibration against the
    # labelled corpus (scripts/marker_recall.py) shows com/holly/marge/ firing on
    # 12.7% of oxylabs and 4.7% of luminati samples, so it is a shared component,
    # not an ipninja namespace. The strict pair below gives recall 1.000 on
    # ipninja with zero hits in any other family.
    "ipninja":     ("Lcom/ipninja/", "Lnet/ipninja/"),
    "luminati":    ("Lio/lum/", "Lio/topvpn/", "Lorg/hola/", "Lio/luminati/"),
    "monkeysocks": ("Lmonkeynode/",),
    "oxylabs":     ("Lframett/",),
}

# Other datasets: curated payload namespaces from payload_namespaces.py (see CROSS_DATASET_PLAN).
if os.environ.get("DATASET", "proxy") != "proxy":
    _pf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "datasets",
                       os.environ["DATASET"], "payload_prefixes.json")
    SDK_PREFIXES = {}
    if os.path.exists(_pf):
        for _k, _v in json.load(open(_pf)).items():
            SDK_PREFIXES[_k] = tuple(_v)


def provenance_vector(nl, su, pr):
    """WL vector for an FCG plus, per emitted token, its (bucket, sign) and originating method."""
    # mirrors gen_wl_vectors.wl_hashed_vector exactly (same tokens, same hashing, same
    # library dropping, same L2 normalization); the origin is the iteration-0 method
    lab = {}
    for n, raw in nl.items():
        lab[n] = G.canonical_label(raw, "fcg")
    drop = []
    for n, l in lab.items():
        if l.startswith(G.LIB_PREFIXES):
            drop.append(n)
    if drop and len(drop) < len(lab):
        for n in drop:
            del lab[n]
    if not lab:
        return None, None, None
    origin = dict(lab)
    vec = np.zeros(DIM, dtype=np.float64)
    prov = []                       # (bucket, sign, origin_method)
    prov_iter = []                  # WL iteration that emitted each token

    cur_iter = [0]

    def add(tok, node):
        h = G.stable_hash_u32(tok)
        b = h % DIM
        if (h & 0x80000000) == 0:
            s = 1.0
        else:
            s = -1.0
        vec[b] += s
        prov.append((b, s, origin[node]))
        prov_iter.append(cur_iter[0])

    for n, l in lab.items():
        add(f"0|{l}", n)
    cur = lab
    for it in range(1, ITERS + 1):
        cur_iter[0] = it
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
        for n in cur:
            add(f"{it}|{cur[n]}", n)
    nrm = np.linalg.norm(vec)
    provenance_vector.last_iters = prov_iter
    if nrm == 0:
        return None, None, None
    return vec / nrm, prov, nrm


SPLIT = os.environ.get("PROVENANCE_SPLIT", "exact")   # "exact" or "legacy"


def split_bucket_credit(prov, w, Ex, nrm, phi, rule=None):
    """Distribute linear SHAP phi_b = w_b (x_b - E[x_b]) over the tokens in each bucket."""
    # exact  (default): token t in bucket b gets  w_b * s_t / ||phi||  (its own term of
    #        w_b x_b, Proposition 1 of the robustness paper) minus the background
    #        term w_b E[x_b] split by occupancy |s_t| / sum|s|. Credits in a bucket
    #        sum to phi_b exactly, whatever the signs.
    # legacy: phi_b * s_t / sum|s|, the rule behind the published numbers. When a
    #        bucket holds tokens of both signs the credits do not sum to phi_b
    #        (a +1 and a -1 token each get 0), and at d=1024 with ~100 tokens per
    #        bucket mixed signs are the normal case.
    if not rule:
        rule = SPLIT
    tot = defaultdict(float)
    for b, s, _ in prov:
        tot[b] += abs(s)
    credit = defaultdict(float)
    if rule == "legacy":
        for b, s, m in prov:
            if tot[b] > 0:
                credit[m] += phi[b] * (s / tot[b])
        return credit
    for b, s, m in prov:
        if tot[b] > 0:
            credit[m] += w[b] * (s / nrm) - w[b] * Ex[b] * (abs(s) / tot[b])
    return credit


def main():
    rng = np.random.default_rng(SEED)
    d = joblib.load(MODEL)
    clf = d["clf"]
    le = d["label_encoder"]
    classes = []
    for c in le.classes_:
        classes.append(str(c))
    coef = clf.coef_

    byfam = defaultdict(list)
    for hdir, gid, lab in T.build_samples():
        if lab in PROXY:
            byfam[lab].append((hdir, gid))

    # E[x] over the corpus vectors, as LinearExplainer uses the training mean
    Xall = []
    for fam in PROXY:
        for hdir, gid in byfam[fam]:
            p = os.path.join(hdir, f"{gid}.fcg_wl_{DIM}")
            if os.path.isfile(p):
                try:
                    Xall.append(np.load(p, allow_pickle=False))
                except Exception:
                    pass
    Ex = np.vstack(Xall).mean(0).astype(np.float64)
    print(f"E[x] from {len(Xall)} corpus vectors\n", flush=True)

    result = {}
    for fam in PROXY:
        ci = classes.index(fam)
        w = coef[ci].astype(np.float64)
        lst = byfam[fam][:]
        rng.shuffle(lst)
        agg = defaultdict(float)          # method -> aggregated attribution
        docfreq = defaultdict(int)        # for the comparison baseline
        used = 0
        for hdir, gid in lst:
            fcg = os.path.join(hdir, f"{gid}.fcg")
            if not GI.exists(fcg) or GI.size(fcg) > MAX_FCG:
                continue
            try:
                nl, su, pr = G.parse_gml_lightweight(fcg)
            except Exception:
                continue
            x, prov, nrm = provenance_vector(nl, su, pr)
            if x is None:
                continue
            phi = w * (x - Ex)                     # exact linear SHAP
            for m, c in split_bucket_credit(prov, w, Ex, nrm, phi).items():
                agg[m] += c
            prov_methods = set()
            for p in prov:
                prov_methods.add(p[2])
            for m in prov_methods:
                docfreq[m] += 1
            used += 1
            if used >= PER_FAM:
                break

        # rank methods by aggregated attribution; score against SDK membership
        labels = list(agg.keys())
        scores_list = []
        truth_list = []
        for m in labels:
            scores_list.append(agg[m])
            if m.startswith(SDK_PREFIXES[fam]):
                truth_list.append(1)
            else:
                truth_list.append(0)
        scores = np.array(scores_list)
        truth = np.array(truth_list)
        n_sdk = int(truth.sum())
        if labels:
            base = n_sdk / len(labels)
        else:
            base = 0
        order = np.argsort(-scores)
        prec = {}
        for k in TOP_N:
            n_top = min(k, len(labels))
            prec[k] = float(truth[order[:n_top]].sum() / n_top)
        if 0 < n_sdk and n_sdk < len(truth):
            auc = float(roc_auc_score(truth, scores))
        else:
            auc = float("nan")
        enrichment = {}
        for k in TOP_N:
            if base:
                enrichment[str(k)] = prec[k] / base
            else:
                enrichment[str(k)] = 0.0
        top_methods = []
        for i in order[:12]:
            top_methods.append({"method": labels[i], "score": float(scores[i]), "is_sdk": bool(truth[i])})
        result[fam] = {
            "n_samples": used, "n_methods": len(labels), "n_sdk": n_sdk,
            "base_rate": base, "auc": auc, "precision_at_n": prec,
            "enrichment_at_n": enrichment,
            "top_methods": top_methods,
        }
        print(f"=== {fam} === {used} samples, {len(labels)} methods, "
              f"sdk={n_sdk} (base {base*100:.2f}%), AUC={auc:.3f}", flush=True)
        for k in TOP_N:
            if base:
                enr = prec[k] / base
            else:
                enr = 0
            print(f"   provenance precision@{k:<4}={prec[k]*100:5.1f}%  "
                  f"enrichment={enr:7.1f}x", flush=True)

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
