#!/usr/bin/env python3
"""Attribution as a label audit: provenance localization before and after the corpus-v3 label correction.

Runs the provenance-tracked attribution of provenance_localization.py (exact linear SHAP on FCG WL
features, split over each bucket's recorded contributors, aggregated over 25 apps per family) twice,
identically except for the labels:
  old   the four-family labels of the earlier corpus (results/robust2/vectors), which include the
        675 DEX-pivot apps later dropped because they carry no marker of their assigned provider
  v3    the validated labels (datasets/corpus_v3/manifest.jsonl, status kept), same four families
Each run trains its own four-class SGD (log loss, alpha 1e-4, upsampled to the largest family) on its
labels' FCG vectors, samples 25 apps per family with the same seed, and ranks methods. Ground truth is
the reference SDK namespaces (SDK below). Graphs come from
exp_harness.load_graph (staged copies or the bundle).

Output: results/datasets/proxy_v3/label_audit.json: per run and family, precision@k, AUC, the share of
the top methods in the SDK namespace and in com/facebook/ads, the number of sampled apps that the v3
label validation dropped, and the top methods.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import provenance_localization as P

ROOT = H.ROOT
FAMS = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
# the eight corpus-v3 classes, for the v3-only experiments (exp_analyst_heuristics v3 runs, highdim HD_V3, MKLDroid)
FAMS_V3 = FAMS + ["cashraven", "ipidea", "popa", "traffmonetizer"]
# the reference ground truth (provenance_localization.py with DATASET=proxy); importing it
# under DATASET=proxy_v3 would substitute the v3 payload prefixes, so it is pinned here. Luminati also
# gets the later Bright Data namespaces (lumnet, brdat, eapx; c2_reversing.py), which the reference list
# predates; the run with the reference list alone is kept in label_audit.published_ns.json
SDK = {"ipninja": ("Lcom/ipninja/", "Lnet/ipninja/"),
       "luminati": ("Lio/lum/", "Lio/topvpn/", "Lorg/hola/", "Lio/luminati/",
                    "Llumnet/sdk/", "Lbrdat/sdk/", "Lcom/android/eapx/"),
       # all three generations' namespaces (results/c2_analysis/MONKEYSOCKS_NOTES.md: 1,124 of 1,125 builds)
       "monkeysocks": ("Lmonkeynode/", "Lcom/monkeyttf/", "Lmonkeysocks/"),
       # the engine's namespace is chosen per publisher; the 20 namespaces found in 76 of the 91 builds
       # (results/c2_analysis/OXYLABS_NOTES.md; the other 15 builds carry no Oxylabs code)
       "oxylabs": ("Lframett/", "Lbackmybraces/", "Lcom/backmybraces/re/",
                   "Lcom/vinockal01/bw/", "Lcom/vinockal02/bw/", "Lcom/vinockal03/bw/", "Lcom/vinockal04/bw/",
                   "Lcom/vinockal05/bw/", "Lcom/vinockal06/bw/", "Lcom/vinockal07/bw/", "Lcom/vinockal08/bw/",
                   "Lcom/vinockal09/bw/", "Lcom/vinockal10/bw/", "Lcom/vinockal11/bw/",
                   "Lcom/vinobalk/wb/",
                   "Lcom/gummlist/ink/", "Lrep/click/", "Ldigison/eye/", "Lgo/locy/arm/", "Lrefraglione/",
                   "Lcom/microchev/fo/", "Lvolumea/", "Linstantcoffee/"),
       # the four newer classes: facade and engine namespaces from results/c2_analysis/*_NOTES.md. Code that R8
       # relocated outside them (IPIDEA's com.sun.* engine, Traffmonetizer's single-letter packages) and native
       # engines are not counted, so for these families the truth is a lower bound. Java-namespace coverage of
       # the class builds: IPIDEA 200/205, Popa 179/182, CashRaven 39/39, Traffmonetizer 16/17. IPIDEA's
       # com/through also holds the ninevpn client of 88 builds (datasets/corpus_v3/label_validity.json).
       "ipidea": ("Lcom/packet/sdk/", "Lcom/castarsdk/", "Lcom/caster/", "Lcom/through/", "Lcom/debby/"),
       "popa": ("Lio/popanet/", "Lio/netas/", "Lio/hopanet/", "Lio/nn/neunative/", "Lio/nn/lp/", "Lio/nn/alpha/",
                "Lcom/amz/appsuite/", "Lio/moneytise/", "Lio/ninjamon/", "Lio/hopmon", "Lcom/mon/app_bandwidth",
                "Lcom/amazon/streammassive/"),
       "cashraven": ("Lio/cashraven/", "Lcom/sprious/"),
       "traffmonetizer": ("Lcom/traffmonetizer/", "Lcom/tm/", "Lcom/base/")}
PER_FAM = 25
SEED = 42
TOP_N = (25, 50, 100)
OUT = os.path.join(ROOT, "results", "datasets", "proxy_v3", "label_audit.json")


def labels(run, fams=None):
    if not fams:
        fams = FAMS
    if run == "old":
        z = np.load(os.path.join(ROOT, "results", "robust2", "vectors", "fcg_wl_1024.npz"))
    else:
        z = np.load(os.path.join(ROOT, "results", "datasets", "proxy_v3", "vectors", "fcg_wl_1024.npz"))
    keep_list = []
    for f in z["family"]:
        keep_list.append(f in fams)
    keep = np.array(keep_list)
    ids = z["ids"][keep]
    fam = z["family"][keep]
    X = z["X"][keep].astype(np.float64)
    if run == "v3":
        # validated class apps of the corpus-v3 manifest
        kept = set()
        for line in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
            r = json.loads(line)
            if r["status"] == "kept" and r.get("split") == "class":
                kept.add(r["sha256"])
        ok_list = []
        for i in ids:
            ok_list.append(i in kept)
        ok = np.array(ok_list)
        ids = ids[ok]
        fam = fam[ok]
        X = X[ok]
    return ids, fam, X


def train(fam, X):
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for j, f in enumerate(fam):
        by[f].append(j)
    sizes = []
    for v in by.values():
        sizes.append(len(v))
    top = max(sizes)
    # upsample every family to the largest: its own indices, then random repeats
    parts = []
    for v in by.values():
        parts.append(np.r_[v, rng.choice(v, top - len(v))])
    idx = np.concatenate(parts)
    m = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-5, random_state=SEED)
    m.fit(X[idx], fam[idx])
    return m


def run_one(run, dropped):
    ids, fam, X = labels(run)
    m = train(fam, X)
    Ex = X.mean(0)
    classes = list(m.classes_)
    res = {"n_apps": int(len(ids))}
    rng = np.random.default_rng(SEED)
    for f in FAMS:
        w = m.coef_[classes.index(f)].astype(np.float64)
        lst = sorted(ids[fam == f].tolist())
        rng.shuffle(lst)
        agg = defaultdict(float)
        used = 0
        n_dropped = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or len(g[0]) == 0:
                continue
            x, prov, nrm = P.provenance_vector(g[0], g[1], g[2])
            if x is None:
                continue
            phi = w * (x - Ex)
            for meth, c in P.split_bucket_credit(prov, w, Ex, nrm, phi).items():
                agg[meth] += c
            used += 1
            if sha in dropped:
                n_dropped += 1
            del g
            if used >= PER_FAM:
                break
        meths = list(agg)
        sc_list = []
        truth_list = []
        for k in meths:
            sc_list.append(agg[k])
            truth_list.append(k.startswith(SDK[f]))
        sc = np.array(sc_list)
        truth = np.array(truth_list)
        order = np.argsort(-sc)
        prec = {}
        for k in TOP_N:
            prec[str(k)] = float(truth[order[:k]].mean())
        top25 = []
        for i in order[:25]:
            top25.append(meths[i])
        n_pos = truth.sum()
        if 0 < n_pos and n_pos < len(truth):
            auc = float(roc_auc_score(truth, sc))
        else:
            auc = None
        n_facebook_ads = 0
        for t in top25:
            if t.startswith("Lcom/facebook/ads/"):
                n_facebook_ads += 1
        top_methods = []
        for i in order[:12]:
            top_methods.append({"method": meths[i], "score": float(sc[i]), "is_sdk": bool(truth[i])})
        res[f] = {"n_samples": used, "n_samples_dropped_in_v3": n_dropped, "n_methods": len(meths),
                  "n_sdk": int(truth.sum()), "precision_at_n": prec,
                  "auc": auc,
                  "top25_share_facebook_ads": n_facebook_ads / 25,
                  "top_methods": top_methods}
        print(f"  {run} {f}: {used} apps ({n_dropped} dropped in v3), P@25={prec['25']:.2f}, "
              f"facebook/ads share of top 25 = {res[f]['top25_share_facebook_ads']:.2f}", flush=True)
    return res


def main():
    # apps the v3 label validation dropped as unvalidated DEX pivots
    dropped = set()
    for line in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        r = json.loads(line)
        if r["status"] == "dropped_unvalidated_pivot":
            dropped.add(r["sha256"])
    out = {"per_family": PER_FAM, "seed": SEED, "runs": {}}
    for run in ("old", "v3"):
        out["runs"][run] = run_one(run, dropped)
        json.dump(out, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
