#!/usr/bin/env python3
"""evasion restricted to a semantics-preserving transformation.

The aimed attack of the paper adds arbitrary tokens. Here the adversary may only bundle real
benign code: whole packages taken from the 40 staged non-proxy applications, added unused, so
their internal call structure and hence their WL tokens are unchanged. Five-class linear model
on FCG identifier vectors at d = 1,024 (the setting of the frontier analysis), benign class =
non-proxy, DEX-grouped folds; staged proxy apps held out of each fold are attacked.

  token     unconstrained aimed attack toward benign (control over every token)
  greedy    adds whole benign packages, each step the one that most raises benign's score
            relative to the true class
  random    adds benign packages in random order
  whole     appends one entire benign application's code (each donor in turn)

Budget 20,000 added tokens. Output: results/tier4/v3/bundle_attack.json
"""
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_p4_cfg_cert as C
import exp_p4_cfg_tokens as P4T
import exp_tok_analyses as T
import exp_open_world as O
import gen_wl_vectors as G

OUT = H.out_path("tier4", "v3", "bundle_attack.json")
BENIGN_DIR = os.path.join(H.GRAPHS_DIR, H.BENIGN)
BUDGET = 20000


def scores(V, W, b):
    """Class scores of raw vectors (rows of V) under the normalized-input linear model."""
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1
    return (V / n) @ W.T + b


def greedy_bundle(phi, W, b, y, c, R, sizes):
    cur = phi.copy()
    used = np.zeros(len(R), dtype=bool)
    spent = 0
    while spent <= BUDGET:
        S = scores(cur[None, :] + R, W, b)
        gain = S[:, c] - S[:, y]
        # packages already used or over budget are not candidates
        gain[used | (spent + sizes > BUDGET)] = -np.inf
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            return None, int(used.sum())
        cur += R[j]
        used[j] = True
        spent += sizes[j]
        if int(np.argmax(scores(cur[None, :], W, b)[0])) == c:
            return spent, int(used.sum())
    return None, int(used.sum())


def main():
    ids, fam = H.sample_ids("five")
    names = sorted(set(fam.tolist()))
    y_list = []
    for f in fam:
        y_list.append(names.index(f))
    y = np.array(y_list)
    benign = names.index(H.BENIGN)
    X = H.matrix("labeled_fcg", ids)
    g = H.groups_for(ids)
    pos = {}
    for i, s in enumerate(ids):
        pos[s] = i

    # benign donor pool: per donor app, raw FCG vector and {package: raw vector of its tokens}
    donors = {}
    for sha in H.staged(H.BENIGN):
        nl, su, pr = G.parse_gml_lightweight(os.path.join(BENIGN_DIR, sha, f"{sha}.fcg"))
        lab = {}
        for n, raw_label in nl.items():
            lab[n] = G.canonical_label(raw_label, "fcg")
        cols = P4T.colours("identifier", lab, su, pr, "fcg", None)
        hs, nodes = P4T.wl_tokens(cols, su, pr)
        hs = np.array(hs, dtype=np.uint64)
        by = defaultdict(list)
        for h, n in zip(hs, nodes):
            # package = first three components of the class path, or the class itself when it has no package
            cls = lab[n].split(";->")[0].lstrip("L")
            parts = cls.split("/")[:-1]
            if parts:
                pkg = "/".join(parts[:3])
            else:
                pkg = cls
            by[pkg].append(h)
        pkg_vectors = {}
        for p, v in by.items():
            pkg_vectors[p] = C.raw(np.array(v, dtype=np.uint64))
        donors[sha] = (C.raw(hs), pkg_vectors)
        del nl, su, pr, lab, cols, hs, nodes, by  # one donor graph at a time

    check = []
    for s, (v, _) in donors.items():
        if s in pos:
            check.append(float(np.dot(v / np.linalg.norm(v), H.matrix("labeled_fcg", [s])[0])))
    print(f"donors {len(donors)}; raw-vs-packed cosine min {min(check):.4f} over {len(check)}", flush=True)
    pkg_rows = []
    for _, (_, pk) in donors.items():
        for v in pk.values():
            pkg_rows.append(v)
    R = np.stack(pkg_rows)
    sizes = np.abs(R).sum(1)
    keep = sizes > 0
    R = R[keep]
    sizes = sizes[keep]
    whole_rows = []
    for v, _ in donors.values():
        whole_rows.append(v)
    whole = np.stack(whole_rows)
    print(f"package pool {len(R)}, median {np.median(sizes):.0f} tokens", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    staged = []
    for _, s in T.staged():
        if s in pos:
            staged.append(s)
    for r, k, tr, te in H.folds(y, g, 1):
        W, b = C.fit(X, y, tr, k)
        te_set = set(te.tolist())
        for s in staged:
            i = pos[s]
            if i not in te_set:
                continue
            _, phi = C.raw_vectors(s, "identifier")
            if phi is None:
                continue
            yt = int(y[i])
            if int(np.argmax(scores(phi[None, :], W, b)[0])) != yt:
                continue
            t_tok = O.attack_to([phi], W, b, yt, benign, BUDGET)
            t_gr, n_pk = greedy_bundle(phi, W, b, yt, benign, R, sizes)
            # random: benign packages in random order until the prediction turns benign
            t_rn = None
            cur = phi.copy()
            spent = 0
            for j in rng.permutation(len(R)):
                if spent + sizes[j] > BUDGET:
                    continue
                cur += R[j]
                spent += sizes[j]
                if int(np.argmax(scores(cur[None, :], W, b)[0])) == benign:
                    t_rn = spent
                    break
            wp = scores(phi[None, :] + whole, W, b).argmax(1)
            rows.append({"family": names[yt], "token": t_tok, "greedy": t_gr, "greedy_packages": n_pk,
                         "random": t_rn, "whole_app_flip_rate": float((wp == benign).mean())})
        print(f"  fold {k}: {len(rows)} apps", flush=True)

    def summ(key):
        v = []
        for r_ in rows:
            v.append(r_[key])
        ok = []
        for x in v:
            if x is not None:
                ok.append(x)
        if ok:
            median_ok = float(np.median(ok))
        else:
            median_ok = None
        return {"success": len(ok) / len(v), "median_tokens_when_successful": median_ok}
    # summary values
    n_rows = len(rows)
    token_summ = summ("token")
    greedy_summ = summ("greedy")
    random_summ = summ("random")
    greedy_pk = []
    for r_ in rows:
        if r_["greedy"] is not None:
            greedy_pk.append(r_["greedy_packages"])
    if greedy_pk:
        greedy_packages_median = float(np.median(greedy_pk))
    else:
        greedy_packages_median = None
    flip_rates = []
    for r_ in rows:
        flip_rates.append(r_["whole_app_flip_rate"])
    out = {"n": n_rows, "n_donors": len(donors), "n_packages": int(len(R)),
           "token": token_summ, "greedy": greedy_summ, "random": random_summ,
           "greedy_packages_median": greedy_packages_median,
           "whole_app_flip_rate_mean": float(np.mean(flip_rates)),
           "per_family": {}, "rows": rows}
    families = set()
    for r_ in rows:
        families.add(r_["family"])
    for f in sorted(families):
        greedy_ok = []
        random_ok = []
        fam_flip = []
        n_fam = 0
        for r_ in rows:
            if r_["family"] == f:
                n_fam += 1
                greedy_ok.append(r_["greedy"] is not None)
                random_ok.append(r_["random"] is not None)
                fam_flip.append(r_["whole_app_flip_rate"])
        out["per_family"][f] = {"n": n_fam,
                                "greedy_success": float(np.mean(greedy_ok)),
                                "random_success": float(np.mean(random_ok)),
                                "whole_app_flip_rate": float(np.mean(fam_flip))}
    print(f"  token {out['token']}\n  greedy {out['greedy']} packages {out['greedy_packages_median']}\n"
          f"  random {out['random']}\n  whole-app flip rate {out['whole_app_flip_rate_mean']:.3f}", flush=True)
    for f, v in out["per_family"].items():
        print(f"    {f}: {v}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)

if __name__ == "__main__":
    main()
