#!/usr/bin/env python3
"""Plan v3, D2 + open world: the deployed representation against real Google Play apps.

Corpus: four proxy families + 996 non-proxy apps, deployed identifier WL vectors (CFG and
FCG at d = 1,024 each, separately normalized). In the wild: 4,888 random Google Play apps
from the AndroZoo pilot (results/androzoo_pilot/vectors), built by the same vectorizer and
never used for training.

  O1  In-the-wild false positives. Every model is trained per DEX-grouped fold on the
      corpus and scores all wild apps; FP rate = share called a proxy family. Plus the
      four-class linear model with max-probability abstention: AUROC of held-out proxy
      apps against wild apps, and FP rate at the threshold that keeps 95% of proxy apps.
  O2  Linearity with a benign class: five-class macro-F1 of the tuned linear model against
      additive and pairwise boosting, a random forest and an MLP on identical folds, with
      corrected resampled t-tests.
  O3  Evading to benign. Five-class linear model; for staged proxy apps held out of each
      fold, the block certificate (Theorem certblock) against every class and against the
      benign class alone, the aimed attack toward benign, and the untargeted attack with
      the class it lands in.

Output: results/tier4/v3/open_world.json
"""
import csv
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_l1_ladder as L
import exp_p4_cfg_cert as C
import exp_tok_analyses as T

OUT = H.out_path("tier4", "v3", "open_world.json")
PILOT = os.path.join(H.ROOT, "results", "androzoo_pilot")
REPEATS = int(os.environ.get("OW_REPEATS", "2"))  # MalNet runs 1 (40 classes: ~1.5 h per fold on this machine)
FLEX = ["additive", "pairwise", "rf", "mlp"]
ATTACK_MAX = 20000


def o1_o2(X, y, g, names, Xw):
    benign = names.index(H.BENIGN)
    F = H.folds(y, g, REPEATS)
    f1 = {}
    fp = {}
    for m in ["linear"] + FLEX:
        f1[m] = []
        fp[m] = []
    ntr = 0
    nte = 0
    ck = OUT + ".o1o2_partial.json"  # per-fold checkpoint (f1 and wild FP per model)
    if os.path.exists(ck):
        done = json.load(open(ck))
    else:
        done = {}
    for r, k, tr, te in F:
        t0 = time.time()
        ntr = len(tr)
        nte = len(te)
        key = f"r{r}f{k}"
        if key in done:
            for m in f1:
                f1[m].append(done[key]["f1"][m])
                fp[m].append(done[key]["fp"][m])
            print(f"  O1/O2 {key}: resumed from checkpoint", flush=True)
            continue
        lin, _ = L.fit_linear(X, y, g, tr, k)
        models = {"linear": lin}
        up = H.upsample(tr, y, k)
        for m in FLEX:
            models[m] = L.make(m, 4).fit(X[up], y[up])
        for m, mod in models.items():
            f1[m].append(f1_score(y[te], mod.predict(X[te]), average="macro"))
            fp[m].append(float((mod.predict(Xw) != benign).mean()))
        shown = []
        for m in f1:
            shown.append(f"{m}={f1[m][-1]:.3f}/{fp[m][-1]:.3f}")
        print(f"  O1/O2 r{r} f{k}: " + " ".join(shown)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        # checkpoint this fold
        f1_last = {}
        for m in f1:
            f1_last[m] = f1[m][-1]
        fp_last = {}
        for m in fp:
            fp_last[m] = fp[m][-1]
        done[key] = {"f1": f1_last, "fp": fp_last}
        json.dump(done, open(ck, "w"))
    # O2: macro-F1 per model, corrected t-test of each flexible model against linear
    o2 = {}
    for m, v in f1.items():
        o2[m] = {"macro_f1": float(np.mean(v)), "sd": float(np.std(v))}
    for m in FLEX:
        md, t, p = H.corrected_t(np.array(f1[m]) - np.array(f1["linear"]), ntr, nte)
        o2[m]["diff_vs_linear"] = md
        o2[m]["p"] = p
    # O1: wild false-positive rate per model
    o1 = {}
    for m, v in fp.items():
        o1[m] = {"wild_fp_rate": float(np.mean(v)), "sd": float(np.std(v))}
    return o1, o2


def o1_abstain(X, y, g, names, Xw):
    """Four-class linear model on proxy apps only; reject by max probability."""
    prox = np.where(y != names.index(H.BENIGN))[0]
    Xp = X[prox]
    yp = y[prox]
    gp = g[prox]
    if len(np.unique(yp)) < 2:  # e.g. piggyback: one non-benign class, so a per-family abstention model is undefined
        print("  O1 abstain: skipped (fewer than two non-benign classes)", flush=True)
        return {"skipped": "fewer than two non-benign classes"}
    msp_known = []
    msp_wild = []
    for r, k, tr, te in H.folds(yp, gp, 1):
        m, _ = L.fit_linear(Xp, yp, gp, tr, k)
        msp_known.append(m.predict_proba(Xp[te]).max(1))
        msp_wild.append(m.predict_proba(Xw).max(1))
    known = np.concatenate(msp_known)
    wild = np.mean(msp_wild, axis=0)
    tau = float(np.quantile(known, 0.05))
    is_known = np.concatenate([np.ones(len(known)), np.zeros(len(wild))])
    msp_all = np.concatenate([known, wild])
    out = {"auroc_proxy_vs_wild": float(roc_auc_score(is_known, msp_all)),
           "tau_95_coverage": tau, "wild_fp_at_tau": float((wild >= tau).mean())}
    print(f"  O1 abstain: AUROC={out['auroc_proxy_vs_wild']:.3f} tau={tau:.3f} "
          f"wild FP at tau={out['wild_fp_at_tau']:.3f}", flush=True)
    return out


def cert_vs(phis, W, b, y, classes, t_max=C.MAX_T):
    """C.block_cert restricted to the listed rival classes."""
    dims = []
    for p in phis:
        dims.append(len(p))
    offs = np.cumsum([0] + dims)
    # per rival: (beta, block constants of each block)
    cons = []
    for c in classes:
        u = W[y] - W[c]
        block_consts = []
        for k in range(len(phis)):
            block_consts.append(C.block_consts(phis[k], u[offs[k]:offs[k + 1]]))
        cons.append((b[y] - b[c], block_consts))

    def worst(T):
        margins = []
        if len(phis) == 1:  # single block (FCG-only setting): no split of the budget
            for beta, bc in cons:
                margins.append(beta + float(C.block_L_vec(bc[0], np.array([T]))[0]))
            return min(margins)
        t = np.arange(T + 1)
        for beta, bc in cons:
            margins.append(beta + float(np.min(C.block_L_vec(bc[0], t) + C.block_L_vec(bc[1], T - t))))
        return min(margins)

    if worst(0) <= 0:
        return 0
    # exponential then binary search for the largest certified budget
    lo = 0
    hi = 1
    while hi < t_max and worst(hi) > 0:
        lo = hi
        hi = hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if worst(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def attack_to(phis, W, b, y, c, max_t=ATTACK_MAX):
    """Greedy aimed addition toward class c; tokens until the prediction is c."""
    phis_copy = []
    for p in phis:
        phis_copy.append(p.astype(float).copy())
    phis = phis_copy
    dims = []
    for p in phis:
        dims.append(len(p))
    offs = np.cumsum([0] + dims)
    u = W[y] - W[c]
    for t in range(1, max_t + 1):
        # the single token (block, index, sign) with the steepest normalized gradient
        best = (0.0, 0, 0, 1.0)
        for k, p in enumerate(phis):
            a = np.linalg.norm(p)
            uk = u[offs[k]:offs[k + 1]]
            gr = (uk - (uk @ p / a) * p / a) / a
            j = int(np.argmax(np.abs(gr)))
            if abs(gr[j]) > best[0]:
                sg_j = -np.sign(gr[j])
                if not sg_j:
                    sg_j = 1.0
                best = (abs(gr[j]), k, j, sg_j)
        _, k, j, sg = best
        phis[k][j] += sg
        if C.pred_blocks(phis, W, b) == c:
            return t
    return None


def o3(X, y, g, names, ids):
    benign = names.index(H.BENIGN)
    pos = {}
    for i, s in enumerate(ids):
        pos[s] = i
    staged = []
    for _, s in T.staged():
        if s in pos:
            staged.append(s)
    rows = []
    for r, k, tr, te in H.folds(y, g, 1):
        W, b = C.fit(X, y, tr, k)
        if W.shape[0] == 1:  # binary: one row; expand to [-w/2, w/2] (same decision and per-class margins)
            W = np.vstack([-W[0] / 2, W[0] / 2])
            b = np.array([-b[0] / 2, b[0] / 2])
        te_set = set(te.tolist())
        import gpu_models as GM
        if GM.ENABLED:  # batched GPU attacks over the fold's applications (gpu_attacks.py)
            import gpu_attacks as GA
            # correctly classified non-benign staged apps of this fold
            apps = []
            for s in staged:
                i = pos[s]
                if i not in te_set:
                    continue
                cfg, fcg = C.raw_vectors(s, "identifier")
                blocks = {"cfg": cfg, "fcg": fcg}
                missing = False
                for k_ in H.GRAPHS:
                    if blocks[k_] is None:
                        missing = True
                        break
                if missing:
                    continue
                phis = []
                for k_ in H.GRAPHS:
                    phis.append(blocks[k_].astype(float))
                yt = int(y[i])
                if yt == benign or C.pred_blocks(phis, W, b) != yt:  # evading to benign is undefined for benign apps
                    continue
                apps.append((phis, yt))
            if apps:
                P = []
                ys = []
                for a in apps:
                    P.append(a[0])
                    ys.append(a[1])
                t_any, land = GA.attack_blocks(P, W, b, ys, ATTACK_MAX)
                t_ben = GA.attack_to(P, W, b, ys, benign, ATTACK_MAX)
                for (phis, yt), ta, la, tb in zip(apps, t_any, land, t_ben):
                    rivals = []
                    for c in range(W.shape[0]):
                        if c != yt:
                            rivals.append(c)
                    if la is None:
                        lands_name = None
                    else:
                        lands_name = names[la]
                    rows.append({"family": names[yt], "k_any": cert_vs(phis, W, b, yt, rivals),
                                 "k_benign": cert_vs(phis, W, b, yt, [benign]),
                                 "attack_benign": tb, "attack_any": ta, "lands": lands_name})
            print(f"  O3 fold {k}: {len(rows)} apps certified so far", flush=True)
            continue
        for s in staged:
            i = pos[s]
            if i not in te_set:
                continue
            cfg, fcg = C.raw_vectors(s, "identifier")
            blocks = {"cfg": cfg, "fcg": fcg}
            missing = False
            for k_ in H.GRAPHS:
                if blocks[k_] is None:
                    missing = True
                    break
            if missing:
                continue
            # the blocks of the active graphs, in vector order
            phis = []
            for k_ in H.GRAPHS:
                phis.append(blocks[k_])
            yt = int(y[i])
            if yt == benign or C.pred_blocks(phis, W, b) != yt:  # evading to benign is undefined for benign apps
                continue
            rivals = []
            for c in range(W.shape[0]):
                if c != yt:
                    rivals.append(c)
            # untargeted attack (C.attack_blocks), replayed on a copy to find the class it lands in
            t_any = C.attack_blocks(phis, W, b, yt, ATTACK_MAX)
            if t_any is None:
                land = None
            else:
                phis_a = []
                for p in phis:
                    phis_a.append(p.astype(float).copy())
                dims_a = []
                for p in phis_a:
                    dims_a.append(len(p))
                offs = np.cumsum([0] + dims_a)
                for _ in range(t_any):
                    normed = []
                    for p in phis_a:
                        normed.append(p / np.linalg.norm(p))
                    x = np.concatenate(normed)
                    gap = (W[yt] - W) @ x + (b[yt] - b)
                    gap[yt] = np.inf
                    u = W[yt] - W[int(np.argmin(gap))]
                    best = (0.0, 0, 0, 1.0)
                    for kb, p in enumerate(phis_a):
                        a = np.linalg.norm(p)
                        uk = u[offs[kb]:offs[kb + 1]]
                        gr = (uk - (uk @ p / a) * p / a) / a
                        j = int(np.argmax(np.abs(gr)))
                        if abs(gr[j]) > best[0]:
                            sg_j = -np.sign(gr[j])
                            if not sg_j:
                                sg_j = 1.0
                            best = (abs(gr[j]), kb, j, sg_j)
                    _, kb, j, sg = best
                    phis_a[kb][j] += sg
                land = C.pred_blocks(phis_a, W, b)
            if land is None:
                lands_name = None
            else:
                lands_name = names[land]
            rows.append({"family": names[yt], "k_any": cert_vs(phis, W, b, yt, rivals),
                         "k_benign": cert_vs(phis, W, b, yt, [benign]),
                         "attack_benign": attack_to(phis, W, b, yt, benign),
                         "attack_any": t_any, "lands": lands_name})
        print(f"  O3 fold {k}: {len(rows)} apps certified so far", flush=True)

    def med(key):
        v = []
        for r_ in rows:
            if r_[key] is not None:
                v.append(r_[key])
        if v:
            return float(np.median(v))
        return None
    # summary (keys in the same order as before)
    lands = []
    for r_ in rows:
        if r_["lands"]:
            lands.append(r_["lands"])
    censored = []
    for r_ in rows:
        censored.append(r_["attack_benign"] is None)
    out = {"n": len(rows), "k_any_median": med("k_any"), "k_benign_median": med("k_benign"),
           "attack_any_median": med("attack_any"), "attack_benign_median": med("attack_benign"),
           "attack_benign_censored": float(np.mean(censored))}
    if lands:
        to_benign = []
        for l in lands:
            to_benign.append(l == H.BENIGN)
        out["untargeted_lands_benign"] = float(np.mean(to_benign))
    else:
        out["untargeted_lands_benign"] = None
    out["per_family"] = {}
    out["rows"] = rows
    fam_set = set()
    for r_ in rows:
        fam_set.add(r_["family"])
    for f in sorted(fam_set):
        rr = []
        for r_ in rows:
            if r_["family"] == f:
                rr.append(r_)
        k_any = []
        k_benign = []
        att_benign = []
        for r_ in rr:
            k_any.append(r_["k_any"])
            k_benign.append(r_["k_benign"])
            if r_["attack_benign"] is not None:
                att_benign.append(r_["attack_benign"])
        fam_out = {"n": len(rr), "k_any_median": float(np.median(k_any)),
                   "k_benign_median": float(np.median(k_benign))}
        if att_benign:
            fam_out["attack_benign_median"] = float(np.median(att_benign))
        else:
            fam_out["attack_benign_median"] = None
        out["per_family"][f] = fam_out
    print(f"  O3: n={out['n']} cert any={out['k_any_median']} benign={out['k_benign_median']} "
          f"attack any={out['attack_any_median']} benign={out['attack_benign_median']} "
          f"lands benign={out['untargeted_lands_benign']}", flush=True)
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    save = lambda: json.dump(res, open(OUT, "w"), indent=1)
    ids, fam = H.sample_ids("five")
    names = sorted(set(fam.tolist()))
    y_list = []
    for f in fam:
        y_list.append(names.index(f))
    y = np.array(y_list)
    X = H.matrix("labeled", ids)
    g = H.groups_for(ids)
    # wild Play apps: pilot rows with status ok, minus corpus apps and flagged apps
    ok = set()
    for f in glob.glob(os.path.join(PILOT, "results_5class_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["status"] == "ok":
                ok.add(r["sha256"])
    wild_ids = sorted(ok - set(ids) - H.wild_flagged())
    wild_vecs = []
    for s in wild_ids:
        wild_vecs.append(np.load(os.path.join(PILOT, "vectors", f"{s}.npy")))
    W = np.stack(wild_vecs).astype(np.float32)
    del wild_vecs
    # pilot vectors are [cfg_wl_1024 | fcg_wl_1024]; keep the blocks of the active graphs
    blocks = {"cfg": W[:, :1024], "fcg": W[:, 1024:]}
    active = []
    for k in H.GRAPHS:
        active.append(blocks[k])
    Xw = np.hstack(active)
    del W, blocks, active
    res["n_corpus"] = len(ids)
    res["n_wild"] = len(wild_ids)
    res["classes"] = names
    res["repeats"] = REPEATS
    print(f"corpus {len(ids)}, wild {len(wild_ids)}", flush=True)
    if "O1_abstain" not in res:
        res["O1_abstain"] = o1_abstain(X, y, g, names, Xw)
        save()
    if "O2" not in res:
        o1, o2 = o1_o2(X, y, g, names, Xw)
        res["O1"] = o1
        res["O2"] = o2
        save()
    if "O3" not in res:
        res["O3"] = o3(X, y, g, names, ids)
        save()
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
