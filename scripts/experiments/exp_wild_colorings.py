#!/usr/bin/env python3
"""Open world per WL coloring, for any corpus (DATASET; common protocol: LEAK_GROUPS=neardup GRAPHS=fcg).

The wild Play set (results/datasets/wild_play/vectors: 4,853 random Google Play apps from AndroZoo with every
coloring vectorized; apps flagged by the proxy-marker rescan, H.wild_flagged(), and apps in the corpus removed)
is scored by models trained on the corpus, separately for each initial coloring:
  wild_fp      linear model (exp_l1_ladder.fit_linear) trained per grouped fold on the corpus with its benign
               class; share of wild apps assigned a non-benign class, mean over folds (exp_open_world O1)
  abstain      model without the benign class; AUROC of known non-benign apps (out-of-fold maximum class
               probability) against wild apps, and the share of wild apps above the threshold that keeps 95%
               of known apps (exp_open_world O1_abstain); skipped with fewer than two non-benign classes
Output: results/cross_dataset/wild_colorings_<tag>.json (tag = dataset, grouping, graphs)
"""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_l1_ladder as L
from sklearn.metrics import roc_auc_score

REPS = ["labeled", "combined", "fuzzy", "api", "structure"]
WILD = os.path.join(H.ROOT, "results", "datasets", "wild_play", "vectors")
tag = f"{H.DATASET}_g{H.GROUPS}_{'+'.join(H.GRAPHS)}"
OUT = os.path.join(H.ROOT, "results", "cross_dataset", f"wild_colorings_{tag}.json")


def main():
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    ids, fam = H.sample_ids("five")
    names = sorted(set(fam.tolist()))
    benign = names.index(H.BENIGN)
    y_list = []
    for f in fam:
        y_list.append(names.index(f))
    y = np.array(y_list)
    g = H.groups_for(ids)
    for rep in REPS:
        if rep in res and "skipped" not in res[rep]:
            continue
        try:
            rids, rfam = H.available(rep, ids, fam)  # drop the few apps without this coloring's vector
            X = H.matrix(rep, rids).astype(np.float64)
            # wild apps: this coloring's blocks for the selected graphs, minus corpus and marker-flagged apps
            exts = []
            for ext in H.REPS[rep]:
                if ext.split("_")[0] in H.GRAPHS:
                    exts.append(ext)
            parts = []
            wids = None
            for ext in exts:
                z = np.load(os.path.join(WILD, ext + ".npz"), allow_pickle=True)
                idx = {}
                for i, s in enumerate(z["ids"]):
                    idx[s] = i
                if wids is None:
                    drop = set(ids) | H.wild_flagged()
                    keep_ids = []
                    for s in idx:
                        if s not in drop:
                            keep_ids.append(s)
                    wids = sorted(keep_ids)
                wild_rows = []
                for s in wids:
                    wild_rows.append(idx[s])
                parts.append(z["X"][wild_rows])
            Xw = np.hstack(parts).astype(np.float64)
            del parts, z, idx  # only the stacked wild matrix is kept
        except (KeyError, FileNotFoundError) as e:
            res[rep] = {"skipped": str(e)}
            print(rep, res[rep], flush=True)
            continue
        yr_list = []
        for f in rfam:
            yr_list.append(names.index(f))
        yr = np.array(yr_list)
        gr = H.groups_for(rids)
        r = {"n_corpus": int(len(rids)), "n_dropped": int(len(ids) - len(rids)), "n_wild": int(len(wids))}
        y_ = yr
        g_ = gr
        fp = []
        for rr, k, tr, te in H.folds(y_, g_, 1):
            m, _ = L.fit_linear(X, y_, g_, tr, k)
            fp.append(float((m.predict(Xw) != benign).mean()))
        r["wild_fp"] = float(np.mean(fp))
        r["wild_fp_sd"] = float(np.std(fp))
        known = np.where(y_ != benign)[0]
        if len(np.unique(y_[known])) >= 2:
            kn = []
            wd = []
            for rr, k, tr, te in H.folds(y_[known], g_[known], 1):
                # fold indices are positions within `known`
                tr = known[tr]
                te = known[te]
                m, _ = L.fit_linear(X, y_, g_, tr, k)
                kn.append(m.predict_proba(X[te]).max(1))
                wd.append(m.predict_proba(Xw).max(1))
            kn = np.concatenate(kn)
            wd = np.mean(wd, axis=0)
            thr = float(np.quantile(kn, 0.05))
            # known apps labelled 1, wild apps 0
            auc_labels = np.r_[np.ones(len(kn)), np.zeros(len(wd))]
            auc_scores = np.r_[kn, wd]
            r["abstain"] = {"auroc": float(roc_auc_score(auc_labels, auc_scores)),
                            "wild_fp_at_tau95": float((wd >= thr).mean())}
        else:
            r["abstain"] = {"skipped": "fewer than two non-benign classes"}
        res[rep] = r
        print(rep, json.dumps(r), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
