#!/usr/bin/env python3
"""Adebayo-style sanity check on GNN node attribution over program graphs.

An attribution that survives randomising the model is not explaining the model:
whatever it shows comes from the input alone. Adebayo et al. (NeurIPS 2018)
demonstrated this failure for several image saliency methods. We are not aware of
it being run on node attribution over function-call graphs, and a method that
fails it cannot support a forensic claim about which methods constitute an SDK.

Two randomisations, both scored on the same graphs against the same ground truth:

  cascading   every learned weight resampled at the trained weights' scale
  readout     only the final linear layer resampled, message passing untouched

Readout-only is the sharper test here. Our attribution is an exact decomposition
of a mean-pooled readout, so if it tracks the classifier it should collapse when
the classifier is replaced, even though the representation is unchanged.

Reports, per randomisation, Spearman rank correlation against the trained
attribution and precision@|target| against the SDK namespace ground truth. A
faithful method shows low correlation and precision falling to the base rate.

Env: CORPUS_ROOT, MEM_GB (default 12)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E
import exp_tier3_gnn as M

OUT = E.out_path("exp_xai_sanity.json")
CKPT = E.out_path("exp_tier3_gnn_params.npz")
RNG = np.random.default_rng(11)

z = np.load(CKPT)
trained = {}
for k in z.files:
    trained[k] = z[k]

cascade = {}
for k, v in trained.items():
    if v.size == 1:
        cascade[k] = v.copy()
    else:
        noise = RNG.standard_normal(v.shape) * (v.std() + 1e-8)
        cascade[k] = noise.astype(v.dtype)

readout = {}
for k, v in trained.items():
    readout[k] = v.copy()
for k in ("Wo", "bo"):
    noise = RNG.standard_normal(trained[k].shape) * (trained[k].std() + 1e-8)
    readout[k] = noise.astype(trained[k].dtype)

graphs = M.load_cached(use_renamed=False)
families = set()
for g in graphs:
    families.add(g["family"])
classes = sorted(families)
cls_idx = {}
for i, c in enumerate(classes):
    cls_idx[c] = i

rho = {"cascading": [], "readout": []}
prec = {"trained": [], "cascading": [], "readout": []}
base_rates = []
n = 0
for g in graphs:
    if g["family"] == "non_proxy" or g["tgt"].sum() < 5:
        continue
    c = cls_idx[g["family"]]
    k = int(g["tgt"].sum())
    base_rates.append(k / g["n"])

    base, _ = M.node_contributions(trained, g, c)
    top = np.argsort(-base)[:k]
    prec["trained"].append(float(g["tgt"][top].sum()) / k)

    for name, params in (("cascading", cascade), ("readout", readout)):
        alt, _ = M.node_contributions(params, g, c)
        r = spearmanr(base, alt).statistic
        if not np.isnan(r):
            rho[name].append(float(r))
        top = np.argsort(-alt)[:k]
        prec[name].append(float(g["tgt"][top].sum()) / k)

    n += 1
    if n % 25 == 0:
        print(f"    {n} graphs", flush=True)

if n == 0:  # e.g. CIC-AndMal: no graph has >= 5 SDK/payload ground-truth nodes, so there is nothing to score
    json.dump({"graphs": 0, "skipped": "no graphs with >= 5 SDK/payload ground-truth nodes"}, open(OUT, "w"), indent=2)
    print("skipped: no graphs with >= 5 SDK/payload ground-truth nodes", flush=True)
    sys.exit(0)

result = {"graphs": n, "base_rate_mean": round(float(np.mean(base_rates)), 5),
          "spearman": {}, "precision": {}}
for name in rho:
    v = np.asarray(rho[name])
    result["spearman"][name] = {
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "frac_above_0.5": round(float((v > 0.5).mean()), 4),
    }
for name in prec:
    result["precision"][name] = round(float(np.mean(prec[name])), 5)

json.dump(result, open(OUT, "w"), indent=2)

print(f"\n[sanity] {n} graphs, base rate {result['base_rate_mean']:.4f}")
print(f"  {'model':14}{'spearman vs trained':>22}{'precision@|target|':>21}")
print(f"  {'trained':14}{'--':>22}{result['precision']['trained']:>21.4f}")
for name in ("cascading", "readout"):
    s = result["spearman"][name]
    print(f"  {name:14}{s['mean']:>22.4f}{result['precision'][name]:>21.4f}")
print(f"\n  faithful: correlation near 0, precision near the base rate")
print(f"  failing:  correlation high, precision unchanged under randomisation")
print(f"\nsaved {OUT}", flush=True)
