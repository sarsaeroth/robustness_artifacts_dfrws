#!/usr/bin/env python3
"""Tier 3 / E14: what the two certificate styles actually buy.

The kernel side admits a deterministic certificate. For a linear classifier w on
the L2-normalised WL vector, padding with k inert methods adds k(H+1) signed
unit spikes, so the raw perturbation has norm sqrt(k(H+1)) exactly, and

    ||v' - v|| <= 2 sqrt(k(H+1)) / ||v_raw||

Requiring that times max_j ||w_y - w_j|| to stay under the margin gives a closed
form for the largest certified k. It costs one forward pass and it is exact.

The smoothing side is probabilistic. We smooth over Bernoulli node deletion,
estimate the top-class probability with a Clopper-Pearson lower bound, and
report what that costs: N recomputations per sample, an abstention region, and
a guarantee that holds with probability 1-alpha rather than always.

We deliberately do NOT quote a matched certified radius for smoothing. Turning
p_A into a radius under node deletion needs the sparse-smoothing LP of
Bojchevski et al.; implementing it is out of scope here, and inventing a
Gaussian-style radius for a discrete edit model would be wrong. What is
comparable, and what is reported, is: the deterministic radius, p_A with its
sample cost, and both methods' empirical survival under the same attack.

Neither certificate covers attribution. That gap is the point of E5.

Env: PER_FAM (default 40), DIM (default 1024), N_SMOOTH (default 200),
     BETA (default 0.1), ALPHA (default 0.01), CORPUS_ROOT, MEM_GB (default 12)
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_common as E
import gen_wl_vectors as G

E.cap_memory(float(os.environ.get("MEM_GB", "12")))

OUT = E.out_path("exp_tier3_certificates.json")
DIM = int(os.environ.get("DIM", "1024"))
PER_FAM = int(os.environ.get("PER_FAM", "40"))
N_SMOOTH = int(os.environ.get("N_SMOOTH", "200"))
BETA = float(os.environ.get("BETA", "0.1"))
ALPHA = float(os.environ.get("ALPHA", "0.01"))
ITERS = 2
RNG = np.random.default_rng(7)


def raw_wl_from_labels(labels, successors, predecessors, dim, iters):
    import hashlib
    vec = np.zeros(dim, dtype=np.float64)
    cur = labels
    for i in range(0, iters + 1):
        if i > 0:
            new = {}
            for nd in cur:
                # sorted colours of in- and out-neighbours still present
                inl = []
                if nd in predecessors:
                    for p in predecessors[nd]:
                        if p in cur:
                            inl.append(cur[p])
                inl = sorted(inl)
                out = []
                if nd in successors:
                    for s in successors[nd]:
                        if s in cur:
                            out.append(cur[s])
                out = sorted(out)
                new[nd] = wl_digest(cur[nd], inl, out)
            cur = new
        for nd in cur:
            h = G.stable_hash_u32(f"{i}|{cur[nd]}")
            # signed hashing: the top bit picks the sign
            if (h & 0x80000000) == 0:
                vec[h % dim] += 1.0
            else:
                vec[h % dim] += -1.0
    return vec


def clopper_pearson_lower(k, n, alpha):
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def main():
    # reference linear classifier, the object being certified
    X = []
    y = []
    for fam in E.FAMILIES:
        famdir = os.path.join(E.CORPUS_ROOT, fam)
        if not os.path.isdir(famdir):
            continue
        for sha in E.family_apps(fam):
            p = os.path.join(famdir, sha, f"{sha}.fcg_wl_{DIM}")
            if not os.path.isfile(p):
                continue
            v = G.load_vec(p)
            if v is not None:
                X.append(v)
                y.append(fam)
    if not X:  # corpus v3 keeps no per-app vector files: train on the packed corpus vectors (L2-normalized)
        import exp_harness as H
        ids, famv = H.sample_ids("five")
        Xp = H.matrix("labeled", ids).astype(np.float64)
        Xp /= np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12)
        X = list(Xp)
        y = list(famv)
        print(f"[*] packed vectors: {len(X)} apps ({H.DATASET}, {H.GRAPHS})", flush=True)
    X = np.asarray(X, dtype=np.float64)
    le = LabelEncoder()
    yy = le.fit_transform(y)
    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=42)
    clf.fit(X, yy)
    W = clf.coef_
    print(f"[*] classifier on {len(X)} vectors, train acc {clf.score(X, yy):.4f}",
          flush=True)

    samples = E.list_samples(per_family=PER_FAM, families=E.PROXY_FAMILIES)
    det_k = []
    p_a_vals = []
    agree = []
    n_done = 0

    for fam, sha, path in samples:
        try:
            nl, su, pr = G.parse_gml_lightweight(path)
        except Exception:
            continue
        labels = E.kept_labels(nl, "fcg")
        del nl
        if len(labels) < 50:
            del labels, su, pr
            continue

        raw = raw_wl_from_labels(labels, su, pr, DIM, ITERS)
        nrm = float(np.linalg.norm(raw))
        if nrm == 0:
            del labels, su, pr
            continue
        v = raw / nrm
        ti = int(le.transform([fam])[0])
        scores = clf.decision_function(v.reshape(1, -1))[0]
        if int(np.argmax(scores)) != ti:
            # certificates only mean anything where the prediction is right
            del labels, su, pr, raw
            continue

        # deterministic radius: largest k with 2 sqrt(k(H+1))/||raw|| * ||dw|| < margin
        worst_k = None
        for j in range(len(scores)):
            if j == ti:
                continue
            margin = float(scores[ti] - scores[j])
            dw = float(np.linalg.norm(W[ti] - W[j]))
            if dw <= 0:
                continue
            # 2 sqrt(k(H+1)) / nrm * dw < margin
            bound = (margin * nrm / (2.0 * dw)) ** 2 / (ITERS + 1)
            k_j = math.floor(bound)
            if worst_k is None or k_j < worst_k:
                worst_k = k_j
        if worst_k is None:
            worst_k = 0
        det_k.append(worst_k)

        # smoothing: Bernoulli node deletion, N recomputations
        node_ids = list(labels)
        votes = 0
        for _ in range(N_SMOOTH):
            keep = {}
            for nd in node_ids:
                if RNG.random() >= BETA:
                    keep[nd] = labels[nd]
            if not keep:
                continue
            rv = raw_wl_from_labels(keep, su, pr, DIM, ITERS)
            n2 = np.linalg.norm(rv)
            if n2 == 0:
                continue
            s = clf.decision_function((rv / n2).reshape(1, -1))[0]
            if int(np.argmax(s)) == ti:
                votes += 1
        p_a = clopper_pearson_lower(votes, N_SMOOTH, ALPHA)
        p_a_vals.append(p_a)
        if votes > N_SMOOTH / 2:
            agree.append(1)
        else:
            agree.append(0)

        del labels, su, pr, raw
        n_done += 1
        if n_done % 5 == 0:
            print(f"    {n_done} apps  median det k={int(np.median(det_k))}  "
                  f"mean pA={np.mean(p_a_vals):.3f}", flush=True)

    # summary statistics (None when there is nothing to summarize)
    if det_k:
        det_median = int(np.median(det_k))
        det_p25 = int(np.percentile(det_k, 25))
        det_p75 = int(np.percentile(det_k, 75))
        det_zero = round(float(np.mean(np.asarray(det_k) == 0)), 4)
    else:
        det_median = None
        det_p25 = None
        det_p75 = None
        det_zero = None
    if p_a_vals:
        pa_mean = round(float(np.mean(p_a_vals)), 4)
        pa_median = round(float(np.median(p_a_vals)), 4)
    else:
        pa_mean = None
        pa_median = None
    if agree:
        abstain = round(1.0 - float(np.mean(agree)), 4)
    else:
        abstain = None
    result = {
        "dim": DIM, "iters": ITERS, "apps": n_done,
        "smoothing": {"n_samples": N_SMOOTH, "deletion_prob": BETA, "alpha": ALPHA},
        "deterministic_certified_k": {
            "median": det_median,
            "p25": det_p25,
            "p75": det_p75,
            "zero_fraction": det_zero,
            "cost_forward_passes": 1,
        },
        "smoothed_pA": {
            "mean": pa_mean,
            "median": pa_median,
            "abstain_fraction": abstain,
            "cost_forward_passes": N_SMOOTH,
            "radius": "not computed; needs the sparse-smoothing LP for discrete edits",
        },
        "attribution_covered_by_either": False,
    }
    json.dump(result, open(OUT, "w"), indent=2)

    print(f"\n[E14] certificates on {n_done} correctly classified apps")
    d = result["deterministic_certified_k"]
    s = result["smoothed_pA"]
    print(f"  deterministic certified k: median {d['median']}  "
          f"IQR [{d['p25']}, {d['p75']}]  zero for {d['zero_fraction']} of apps")
    print(f"    cost: 1 forward pass, exact, always valid")
    print(f"  smoothed pA (beta={BETA}, N={N_SMOOTH}, alpha={ALPHA}): "
          f"mean {s['mean']}  median {s['median']}")
    print(f"    abstains on {s['abstain_fraction']} of apps; "
          f"cost {N_SMOOTH} forward passes; radius not computed")
    print(f"  neither certificate covers the attribution claim (see E5)")
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
