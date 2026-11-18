#!/usr/bin/env python3
"""Tier 1 / E5: the M-inflation attack, measured rather than derived.

Padding an app with k inert methods adds k(H+1) fresh tokens to the non-target
side. The theory predicts two different decay rates from the same perturbation:

  attribution   p'(k) = p * exp(-k(H+1)/d)          exponential in k
  prediction    margin falls as O(sqrt(k))          the padding is L2 noise of
                                                    norm sqrt(k(H+1)) against a
                                                    normalised vector

If both hold, an adversary can destroy localization while the classifier stays
confidently correct, and a prediction-only certificate does not cover the claim
the pipeline actually makes. Both halves are measured here on real graphs: the
same parsed app yields the token sets for isolation and the raw WL vector for
the margin, so the two curves describe one perturbation, not two experiments.

Inert methods are disconnected and freshly named, so they emit tokens that
collide with existing ones only by chance, which is exactly the attack model.

Env: PER_FAM (default 40), DIM (default 1024), CORPUS_ROOT, MEM_GB (default 12)
"""
import glob
import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_common as E
import gen_wl_vectors as G

E.cap_memory(float(os.environ.get("MEM_GB", "12")))

OUT = E.out_path("exp_tier1_inflation.json")
DIM = int(os.environ.get("DIM", "1024"))
PER_FAM = int(os.environ.get("PER_FAM", "40"))
ITERS = 2
K_SWEEP = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]
# The margin is measured at the classifier's own dimension, but attribution is
# swept over d: the deployed d=1024 is the classification dimension, and the
# identifiability bound says localization needs Theta(M/ln(1/p)). Reporting only
# d=1024 would attribute to the method what is really a provisioning choice.
ISO_DIMS = [1024, 16384, 262144, 1048576]
RNG = np.random.default_rng(42)


def train_reference_classifier():
    """Linear model on the shipped fcg_wl vectors: the margin we attack."""
    X, y = [], []
    for fam in E.FAMILIES:
        famdir = os.path.join(E.CORPUS_ROOT, fam)
        for sha in E.family_apps(fam):
            p = os.path.join(famdir, sha, f"{sha}.fcg_wl_{DIM}")
            if not os.path.isfile(p):
                continue
            v = G.load_vec(p)
            if v is None:
                continue
            X.append(v)
            y.append(fam)
    if not X and os.environ.get("DATASET", "proxy") != "proxy":
        # comparison datasets keep their vectors packed: same WL FCG features, read through the harness
        import exp_harness as H
        ids = []
        for fam in E.FAMILIES:
            for sha in E.family_apps(fam):
                ids.append(sha)
        fam_of = {}
        for fam in E.FAMILIES:
            for sha in E.family_apps(fam):
                fam_of[sha] = fam
        M = H.matrix("labeled", np.array(ids))[:, -DIM:]  # the FCG block
        X = list(M)
        y = []
        for s in ids:
            y.append(fam_of[s])
    X = np.asarray(X, dtype=np.float64)
    le = LabelEncoder()
    yy = le.fit_transform(y)
    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=42)
    clf.fit(X, yy)
    print(f"[*] reference classifier on {len(X)} vectors, "
          f"train acc {clf.score(X, yy):.4f}", flush=True)
    return clf, le


def main():
    clf, le = train_reference_classifier()
    samples = E.list_samples(per_family=PER_FAM, families=E.PROXY_FAMILIES)
    print(f"[*] {len(samples)} apps", flush=True)

    iso_rows = {}
    margin_rows = {}
    pred_rows = {}
    for k in K_SWEEP:
        margin_rows[k] = []
        pred_rows[k] = []
        for d in ISO_DIMS:
            iso_rows[(k, d)] = []

    n_done = 0
    for fam, sha, path in samples:
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception:
            continue
        labels = E.kept_labels(nl, "fcg")
        del nl
        tgt_nodes = E.target_nodes(labels, E.SDK_PREFIXES.get(fam, ()))
        if len(tgt_nodes) < 5 or not labels:
            del labels, su, pr
            continue

        tgt_h, non_h, _ = E.token_provenance(labels, su, pr, tgt_nodes, ITERS)
        # unnormalised WL vector, so the inflation noise can be added at true scale
        vec = np.zeros(DIM, dtype=np.float64)
        cur = labels
        for i in range(0, ITERS + 1):
            if i > 0:
                new = {}
                for nd in cur:
                    inl = []
                    for p in pr.get(nd, []):
                        if p in cur:
                            inl.append(cur[p])
                    inl.sort()
                    out = []
                    for s in su.get(nd, []):
                        if s in cur:
                            out.append(cur[s])
                    out.sort()
                    new[nd] = wl_digest(cur[nd], inl, out)
                cur = new
            for nd in cur:
                h = G.stable_hash_u32(f"{i}|{cur[nd]}")
                if (h & 0x80000000) == 0:
                    vec[h % DIM] += 1.0
                else:
                    vec[h % DIM] += -1.0
        del cur, new  # graph-sized colour maps are freed with the graph, as before
        base_norm = float(np.linalg.norm(vec))
        del labels, su, pr
        if base_norm == 0 or not tgt_h:
            continue

        tgt_only = np.fromiter(tgt_h - non_h, dtype=np.int64,
                               count=len(tgt_h - non_h))
        non_arr = np.fromiter(non_h, dtype=np.int64, count=len(non_h))
        n_tgt = len(tgt_h)

        for k in K_SWEEP:
            n_new = k * (ITERS + 1)
            # inert methods emit fresh 32-bit tokens
            if n_new:
                fresh = RNG.integers(0, 2 ** 32, size=n_new, dtype=np.int64)
            else:
                fresh = np.empty(0, dtype=np.int64)

            for d in ISO_DIMS:
                occupied = np.zeros(d, dtype=bool)
                occupied[non_arr % d] = True
                if n_new:
                    occupied[fresh % d] = True
                if len(tgt_only):
                    free = int(np.count_nonzero(~occupied[tgt_only % d]))
                else:
                    free = 0
                iso_rows[(k, d)].append(free / n_tgt)
                del occupied

            # the same padding as L2 noise on the vector the classifier sees
            v2 = vec.copy()
            if n_new:
                signs = np.where((fresh & 0x80000000) == 0, 1.0, -1.0)
                np.add.at(v2, (fresh % DIM).astype(np.int64), signs)
            nrm = np.linalg.norm(v2)
            if nrm > 0:
                v2 /= nrm
            # margin of the true class over the best other class
            scores = clf.decision_function(v2.reshape(1, -1))[0]
            if np.ndim(scores) == 0:
                scores = np.array([-scores, scores])
            ti = int(le.transform([fam])[0])
            true_score = scores[ti]
            others = np.delete(scores, ti)
            margin_rows[k].append(float(true_score - others.max()))
            del v2

        # theory curve at the deployed dimension, anchored at this app's own k=0
        p0 = iso_rows[(0, DIM)][-1]
        for k in K_SWEEP:
            pred_rows[k].append(p0 * math.exp(-k * (ITERS + 1) / DIM))

        del tgt_h, non_h, tgt_only, non_arr, vec
        n_done += 1
        if n_done % 10 == 0:
            print(f"    {n_done} apps", flush=True)

    result = {"dim": DIM, "iters": ITERS, "apps": n_done, "k_sweep": K_SWEEP,
              "iso_dims": ISO_DIMS, "rows": {}}
    print(f"\n[E5] inflation on {n_done} apps at d={DIM}, H={ITERS}")
    print(f"  {'k':>6}{'iso meas':>11}{'iso pred':>11}{'margin':>11}"
          f"{'margin/m0':>11}{'still ok':>11}")
    if margin_rows[0]:
        m0 = float(np.mean(margin_rows[0]))
    else:
        m0 = 0.0
    n_correct0 = 0
    for v in margin_rows[0]:
        if v > 0:
            n_correct0 += 1
    for k in K_SWEEP:
        if not iso_rows[(k, DIM)]:
            continue
        im = float(np.mean(iso_rows[(k, DIM)]))
        ip = float(np.mean(pred_rows[k]))
        mm = float(np.mean(margin_rows[k]))
        still = 0
        for v in margin_rows[k]:
            if v > 0:
                still += 1
        if m0:
            margin_ratio = round(mm / m0, 5)
            ratio = mm / m0
        else:
            margin_ratio = None
            ratio = float("nan")
        by_dim = {}
        for d in ISO_DIMS:
            by_dim[str(d)] = round(float(np.mean(iso_rows[(k, d)])), 5)
        result["rows"][str(k)] = {
            "isolation_measured": round(im, 5),
            "isolation_predicted": round(ip, 5),
            "margin_mean": round(mm, 5),
            "margin_ratio": margin_ratio,
            "still_correct_frac": round(still / max(len(margin_rows[k]), 1), 4),
            "isolation_by_dim": by_dim,
            "n": len(iso_rows[(k, DIM)]),
        }
        print(f"  {k:>6}{im:>11.4f}{ip:>11.4f}{mm:>11.4f}{ratio:>11.4f}"
              f"{still/max(len(margin_rows[k]),1):>11.4f}")

    print(f"\n[E5] attribution by dimension (isolated fraction)")
    header = []
    for d in ISO_DIMS:
        header.append(f"{('d='+str(d)):>13}")
    print("  " + f"{'k':>6}" + "".join(header))
    for k in K_SWEEP:
        if not iso_rows[(k, DIM)]:
            continue
        cells = []
        for d in ISO_DIMS:
            cells.append(f"{float(np.mean(iso_rows[(k, d)])):>13.4f}")
        row = "".join(cells)
        print(f"  {k:>6}{row}")

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
