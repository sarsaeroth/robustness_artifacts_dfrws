#!/usr/bin/env python3
"""R2: how tight is the additive certificate? (certified lower bound vs realizable
feature-space perturbation).

The certificate (theory.tex, eq:cert_add) gives a per-sample lower bound k_cert on
the number of tokens an adversary must add before the prediction can change. The
paper argues this bound is TIGHT for identifier colourings, because the hash is
public, so an added stub method's tokens can be steered into a chosen bucket; and
LOOSE-to-unreachable for invariant colourings, where a stub's tokens are fixed by
its type/degree and cannot be aimed. This experiment measures that gap. It is the
empirical half of a certified-robustness result and stays entirely in the WL
feature space (adding abstract tokens to the count vector); it does not synthesise
method names or produce an APK.

Per staged proxy app, linear model on L2-normalized labeled WL (staged apps held
out of training):
  k_cert   conservative certified lower bound (no flip is possible below it)
  k_steer  minimal AIMED additions to flip: greedily place each added token in the
           bucket/sign that most reduces the nearest score gap (worst case; upper
           bound on any realizable problem-space attack)
  k_rand   minimal UNAIMED additions to flip: random bucket/sign, median over seeds
           (the benign-padding baseline of R4 / the inflation attack)
The interval [k_cert, k_steer] is the certificate's tightness; k_rand / k_steer is
how much aiming (i.e. white-box knowledge of the hash) buys the adversary.

Output: results/tier4/r2_certificate.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H

OUT = H.out_path("tier4", "r2_certificate.json")
D = 1024
MAX_K = 20000
RAND_SEEDS = 3


def k_cert(phi, W, b, y):
    """Largest T guaranteed safe: for each c, worst case reduces the gap by at most
    T ||u_c||_inf in the numerator and |beta_c| T in the norm term, so
    s_c - T(||u_c||_inf + |beta_c|) > 0  =>  T < s_c / (||u_c||_inf + |beta_c|)."""
    # s_c = u_c . phi + beta_c ||phi||  (>0 for all c != y  <=> prediction stays y)
    nrm = np.linalg.norm(phi)
    u = W[y] - W          # (C, d)
    beta = b[y] - b
    g = u @ phi + beta * nrm
    g[y] = np.inf
    C = W.shape[0]
    ks = []
    for c in range(C):
        if c == y:
            continue
        denom = np.abs(u[c]).max() + abs(beta[c])
        if denom > 0:
            ks.append(g[c] / denom)
        else:
            ks.append(np.inf)
    return int(np.floor(max(0.0, min(ks)))), float(min(ks))


def k_steer(phi, W, b, y, max_k=MAX_K):
    """Greedy aimed feature-space attack. Each step adds one token at the (bucket,
    sign) that most decreases the current minimum score gap. Returns k to flip."""
    phi = phi.astype(np.float64).copy()
    for k in range(1, max_k + 1):
        nrm = np.linalg.norm(phi)
        u = W[y] - W
        beta = b[y] - b
        g = u @ phi + beta * nrm
        g[y] = np.inf
        c = int(np.argmin(g))            # nearest wrong class
        # marginal change to g_c of adding sign s at bucket j: s*u[c,j] + beta_c*(phi_j/nrm)
        grad = u[c] + beta[c] * (phi / nrm)
        j = int(np.argmax(np.abs(grad)))
        s = -np.sign(grad[j])
        if not s:  # zero gradient: default to a positive token
            s = 1.0
        phi[j] += s
        # recompute only the flip check
        nrm2 = np.linalg.norm(phi)
        if int(np.argmax(W @ (phi / nrm2) + b)) != y:
            return k
    return None


def k_random(phi, W, b, y, seed, max_k=MAX_K):
    rng = np.random.default_rng(seed)
    phi = phi.astype(np.float64).copy()
    d = W.shape[1]
    for k in range(1, max_k + 1):
        j = int(rng.integers(d))
        if rng.random() < 0.5:
            s = 1.0
        else:
            s = -1.0
        phi[j] += s
        nrm = np.linalg.norm(phi)
        if int(np.argmax(W @ (phi / nrm) + b)) != y:
            return k
    return None


def main():
    import gen_wl_vectors as G
    import provenance_localization as P
    ids, fam = H.sample_ids("four")
    y_all, le = H.labels(fam)
    Xn = H.matrix("labeled_fcg", ids)           # FCG-only, matches provenance_vector
    g = H.groups_for(ids)
    GRAPHS = H.GRAPHS_DIR
    staged = set()
    for f in H.PROXY:
        for sha in H.staged(f):
            staged.add(sha)
    # Train the deployed model on the full proxy corpus. (Excluding staged apps
    # would drop most of the 35-sample ipninja class; the minimal-flip geometry
    # measured here is barely affected by the clean sample being in training, and
    # the full-corpus model is the one actually deployed.)
    up = H.upsample(np.arange(len(ids)), y_all, 0)
    m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
    m.fit(Xn[up], y_all[up])
    W = m.coef_.astype(np.float64)
    b = m.intercept_.astype(np.float64)

    # raw (unnormalized) count vectors for staged apps, so one added token = one unit
    staged_apps = []
    for f in H.PROXY:
        for sha in H.staged(f):
            staged_apps.append((f, sha))
    print(f"training on {len(up)} (corpus minus {len(staged)} staged); {len(staged_apps)} staged proxy apps", flush=True)

    rows = defaultdict(list)
    for f, sha in staged_apps:
        try:
            nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, f, sha, f"{sha}.fcg"))
            xn, prov, nrm = P.provenance_vector(nl, su, pr)
        except Exception:
            continue
        if xn is None:
            continue
        phi = (xn * nrm).astype(np.float64)          # raw count vector
        y = int(le.transform([f])[0])
        if int(np.argmax(W @ (phi / np.linalg.norm(phi)) + b)) != y:
            continue                                  # certificates only where correct
        kc, kc_f = k_cert(phi, W, b, y)
        ks = k_steer(phi, W, b, y)
        krs = []
        for s in range(RAND_SEEDS):
            krs.append(k_random(phi, W, b, y, s))
        kr_ok = []
        for x in krs:
            if x is not None:
                kr_ok.append(x)
        if kr_ok:
            kr = float(np.median(kr_ok))
        else:
            kr = None    # None = random padding never flipped within MAX_K
        rows[f].append({"k_cert": kc, "k_steer": ks, "k_rand": kr,
                        "rand_no_flip": int(len(kr_ok) < len(krs)), "norm": float(nrm)})
        n_done = 0
        for v in rows.values():
            n_done += len(v)
        if n_done % 10 == 0:
            print(f"  {n_done} apps done", flush=True)

    res = {"dim": D, "n_train": int(len(up)), "per_family": {}, "overall": {}}
    allr = []
    # per-family summary
    for f, rs in rows.items():
        allr += rs
        kc_l = []
        ks_l = []
        kr_l = []
        steer_cert = []
        rand_steer = []
        rand_never = []
        steer_never = []
        for r in rs:
            kc_l.append(r["k_cert"])
            if r["k_steer"] is not None:
                ks_l.append(r["k_steer"])
            if r["k_rand"] is not None:
                kr_l.append(r["k_rand"])
            if r["k_steer"]:
                steer_cert.append(r["k_steer"] / max(r["k_cert"], 1))
            if r["k_steer"] and r["k_rand"]:
                rand_steer.append(r["k_rand"] / r["k_steer"])
            rand_never.append(r["k_rand"] is None)
            steer_never.append(r["k_steer"] is None)
        kc = np.array(kc_l)
        ks = np.array(ks_l)
        kr = np.array(kr_l)
        if len(ks):
            ks_med = float(np.median(ks))
        else:
            ks_med = None
        if len(kr):
            kr_med = float(np.median(kr))
        else:
            kr_med = None
        if rand_steer:
            rs_med = float(np.median(rand_steer))
        else:
            rs_med = None
        res["per_family"][f] = {
            "n": len(rs),
            "k_cert_median": float(np.median(kc)),
            "k_steer_median": ks_med,
            "k_rand_median": kr_med,
            "steer_over_cert_median": float(np.median(steer_cert)),
            "rand_over_steer_median": rs_med,
            "rand_never_flips_frac": float(np.mean(rand_never)),
            "steer_never_flips_frac": float(np.mean(steer_never)),
        }

    # overall summary
    kc_l = []
    ks_l = []
    kr_l = []
    steer_cert = []
    rand_steer = []
    rand_never = []
    for r in allr:
        kc_l.append(r["k_cert"])
        if r["k_steer"] is not None:
            ks_l.append(r["k_steer"])
        if r["k_rand"] is not None:
            kr_l.append(r["k_rand"])
        if r["k_steer"]:
            steer_cert.append(r["k_steer"] / max(r["k_cert"], 1))
        if r["k_steer"] and r["k_rand"]:
            rand_steer.append(r["k_rand"] / r["k_steer"])
        rand_never.append(r["k_rand"] is None)
    kc = np.array(kc_l)
    ks = np.array(ks_l)
    kr = np.array(kr_l)
    if rand_steer:
        rs_med = float(np.median(rand_steer))
    else:
        rs_med = None
    res["overall"] = {"n": len(allr),
                      "k_cert_median": float(np.median(kc)),
                      "k_steer_median": float(np.median(ks)),
                      "k_rand_median": float(np.median(kr)),
                      "steer_over_cert_median": float(np.median(steer_cert)),
                      "rand_over_steer_median": rs_med,
                      "rand_never_flips_frac": float(np.mean(rand_never))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
