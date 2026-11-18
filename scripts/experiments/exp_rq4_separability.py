#!/usr/bin/env python3
"""RQ4: does structural separability of the payload predict certifiability and localizability?

Per dataset (DATASET env), for every staged payload application: the separability measures of
exp_separability.py (rho, min cut, boundary edges per payload method), the additive certificate
of the identifier-coloring linear model (exp_p3_frontier.k_cert on the staged raw vector, model
from the harness folds), normalized by the raw vector's l1 size (tokens), and exact-token
localization precision at the payload size (rank methods by their summed linear contribution).
`pool` merges the per-dataset files and reports Spearman correlations across all payload
applications and across family medians; localizability is taken per family from x2_method_level
(exact-token precision over its base rate) and related to the family-median measures.

Output: tier4/v4/rq4.json (dataset-aware); pooled: results/cross_dataset/rq4_pooled.json
Usage: DATASET=... exp_rq4_separability.py | exp_rq4_separability.py pool
"""
import json
import os
import sys

os.environ["STAGED_SET"] = "payload"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "v4", "rq4.json")


def run():
    import exp_p3_frontier as FR
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    F = H.folds(y, g, 1)
    # app id -> test fold
    fold_of = {}
    for _, k, _, te in F:
        for j in te:
            fold_of[ids[j]] = k
    apps, mats = FR.staged_raw(ids)
    app_tuples = []
    for a in apps:
        app_tuples.append(tuple(a))
    apps = app_tuples
    idx, _, XP = H._load(FR.PACK["labeled"])
    rows_of_ids = []
    for i in ids:
        rows_of_ids.append(idx[i])
    X = XP[rows_of_ids].astype(np.float64)
    # one linear model per fold
    models = {}
    for _, k, tr, _ in F:
        models[k] = FR.fit_sgd(X, y, tr, k)
    sep = json.load(open(H.out_path("tier4", "v4", "separability.json")))
    srow = {}
    for r in sep["rows"]:
        srow[r["sha"]] = r
    rows = []
    for i, (f, sha) in enumerate(apps):
        if sha not in srow or sha not in fold_of or srow[sha].get("payload", 0) == 0:
            continue
        W, b = models[fold_of[sha]]
        phi = mats["labeled"][i]
        yy = int(le.transform([f])[0])
        if np.linalg.norm(phi) == 0 or FR.pred(W, b, phi) != yy:
            continue
        kc = FR.k_cert(phi, W, b, yy)
        s = srow[sha]
        tokens = float(np.abs(phi).sum())
        rows.append({"dataset": H.DATASET, "family": f, "sha": sha, "k_cert": float(kc),
                     "tokens": tokens, "k_norm": float(kc) / max(tokens, 1.0),
                     "rho": s["rho"], "min_cut": s["min_cut"], "boundary": s["boundary_per_payload_method"],
                     "payload": s["payload"], "n": s["n"]})
    json.dump({"dataset": H.DATASET, "rows": rows}, open(OUT, "w"), indent=1)
    print(f"{H.DATASET}: {len(rows)} payload apps", flush=True)


def pool():
    from scipy.stats import spearmanr
    rows = []
    root = os.path.join(H.ROOT, "results")
    # proxy: corpus v3 under the common protocol (RQ4_PROXY=proxy restores the earlier corpus)
    ptag = os.environ.get("RQ4_PROXY", "proxy_v3_payload_gneardup_fcg")
    tags = [ptag]
    for d in ("malnet", "piggyback", "cic_andmal"):
        tags.append(f"{d}_payload")
    for tag in tags:
        if tag == "proxy":
            p = os.path.join(root, "tier4", "v4", "rq4.json")
        else:
            p = os.path.join(root, "datasets", tag, "tier4", "v4", "rq4.json")
        if os.path.exists(p):
            rows += json.load(open(p))["rows"]
    datasets = set()
    for r in rows:
        datasets.add(r["dataset"])
    out = {"n": len(rows), "datasets": sorted(datasets), "apps": {}, "family_medians": {}}
    for x in ("rho", "min_cut", "boundary"):
        measure_vals = []
        k_norms = []
        for r in rows:
            if r[x] is not None:
                measure_vals.append(r[x])
                k_norms.append(r["k_norm"])
        if len(measure_vals) > 5:
            rho, p = spearmanr(measure_vals, k_norms)
            out["apps"][x] = {"spearman_with_k_norm": float(rho), "p": float(p), "n": len(measure_vals)}
    fam_set = set()
    for r in rows:
        fam_set.add((r["dataset"], r["family"]))
    fams = sorted(fam_set)
    # within-family association: the pooled one can be driven by family differences alone
    out["within_family"] = {}
    for x in ("rho", "min_cut", "boundary"):
        cs = []
        for d, f in fams:
            measure_vals = []
            k_norms = []
            for r in rows:
                if r["dataset"] == d and r["family"] == f and r[x] is not None:
                    measure_vals.append(r[x])
                    k_norms.append(r["k_norm"])
            if len(measure_vals) >= 8 and len(set(measure_vals)) > 1:
                cs.append(float(spearmanr(measure_vals, k_norms)[0]))
        if cs:
            n_positive = 0
            for c in cs:
                if c > 0:
                    n_positive += 1
            out["within_family"][x] = {"median_spearman": float(np.median(cs)), "n_families": len(cs),
                                       "n_positive": int(n_positive)}
    # family medians, aligned with fams
    med = []
    for d, f in fams:
        rr = []
        for r in rows:
            if r["dataset"] == d and r["family"] == f:
                rr.append(r)
        fam_med = {}
        for k in ("rho", "min_cut", "boundary", "k_norm"):
            vals = []
            for r in rr:
                if r[k] is not None:
                    vals.append(r[k])
            fam_med[k] = float(np.median(vals))
        med.append(fam_med)
    for x in ("rho", "min_cut", "boundary"):
        if len(med) > 3:
            xs = []
            ks = []
            for m in med:
                xs.append(m[x])
            for m in med:
                ks.append(m["k_norm"])
            rho, p = spearmanr(xs, ks)
            out["family_medians"][x] = {"spearman_with_k_norm": float(rho), "p": float(p), "n": len(med)}
    # localizability: exact-token (lin_exact) precision at the payload size over its base rate, per family
    # (x2_method_level summaries), against the family-median separability measures
    lift = {}
    datasets = set()
    for r in rows:
        datasets.add(r["dataset"])
    for d in datasets:
        if d == "proxy":
            p = os.path.join(root, "tier4", "x2_method_level.json")
        elif d == "proxy_v3":
            p = os.path.join(root, "datasets", "proxy_v3_gneardup_fcg", "tier4", "x2_method_level.json")
        else:
            p = os.path.join(root, "datasets", f"{d}_payload", "tier4", "x2_method_level.json")
        if os.path.exists(p):
            x2 = json.load(open(p))
            if "summary" in x2:
                summary = x2["summary"]
            else:
                summary = {}
            for f, v in summary.items():
                if v.get("lin_exact"):
                    le_ = v.get("lin_exact")
                else:
                    le_ = {}
                if le_.get("p_at_target") is not None and le_.get("base_rate"):
                    lift[(d, f)] = (le_["p_at_target"], le_["base_rate"], le_["p_at_target"] / le_["base_rate"])
    for i in range(min(len(med), len(fams))):
        m = med[i]
        df = fams[i]
        if df in lift:
            m["loc_precision"] = lift[df][0]
            m["loc_base_rate"] = lift[df][1]
            m["loc_lift"] = lift[df][2]
    out["localization_family"] = {}
    lm = []
    for m in med:
        if "loc_lift" in m:
            lm.append(m)
    for x in ("rho", "min_cut", "boundary", "k_norm"):
        if len(lm) > 3:
            xs = []
            lifts = []
            for m in lm:
                xs.append(m[x])
            for m in lm:
                lifts.append(m["loc_lift"])
            rho, p = spearmanr(xs, lifts)
            out["localization_family"][x] = {"spearman_with_lift": float(rho), "p": float(p), "n": len(lm)}
    families = []
    for i in range(min(len(fams), len(med))):
        entry = {"dataset": fams[i][0], "family": fams[i][1]}
        for k, v in med[i].items():
            entry[k] = v
        families.append(entry)
    out["families"] = families
    os.makedirs(os.path.join(root, "cross_dataset"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "cross_dataset", "rq4_pooled.json"), "w"), indent=1)
    summary_out = {}
    for k, v in out.items():
        if k != "families":
            summary_out[k] = v
    print(json.dumps(summary_out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pool":
        pool()
    else:
        run()
