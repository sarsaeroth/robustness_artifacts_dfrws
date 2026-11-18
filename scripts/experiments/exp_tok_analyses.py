#!/usr/bin/env python3
"""exact-WL-token analyses on the 155 staged apps (no hashing).

  L5  naive Bayes vs logistic on exact tokens, per coloring.
  L6  classification from each WL round (0, 1, 2) alone.
  L7  exact tokens vs hashed at d = 2^8..2^16, same apps and folds.
  X1  split the linear model's per-app evidence into SDK-token vs host-token mass.
  X2  classify from SDK tokens only vs host tokens only.
  X3  collision-free localization ceiling: exact-token method decomposition precision.
  X4  family-private token share counted over distinct DEX groups, not apps.
  X5  TF-IDF down-weighting of common tokens before the linear model.

X1-X5 are identifier-only (SDK membership and localization are identifier-coloring
notions). L5-L7 run for identifier, fuzzy, api, degree.
Output: results/tier4/v2/tok_analyses.json
"""
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_hash import wl_digest
import exp_harness as H
import exp_p3_frontier as FR
import gen_wl_vectors as G
import gen_wl_hardened as HD
import provenance_localization as PL

GRAPHS = FR.GRAPHS
OUT = H.out_path("tier4", "v2", "tok_analyses.json")
ITERS = 2
COLORINGS = ["identifier", "fuzzy", "api", "degree"]


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def _job(t):
    fam, sha, col = t
    try:
        # exact WL tokens of one app: per emitted token its id, round and origin method index,
        # plus the per-method SDK mask
        nl, su, pr = G.parse_gml_lightweight(os.path.join(GRAPHS, fam, sha, f"{sha}.fcg"))
        lab = {}
        for n, r in nl.items():
            lab[n] = G.canonical_label(r, "fcg")
        drop = []
        for n, l in lab.items():
            if l.startswith(G.LIB_PREFIXES):
                drop.append(n)
        if drop and len(drop) < len(lab):
            for n in drop:
                del lab[n]
        keep = set(lab)
        pref = tuple(PL.SDK_PREFIXES.get(fam, ()))
        sdk_node = {}
        for n in lab:
            sdk_node[n] = lab[n].startswith(pref)
        # initial colors for the requested coloring
        if col == "identifier":
            c0 = dict(lab)
        elif col == "degree":
            c0 = {}
            for n in lab:
                n_in = 0
                for p in pr.get(n, []):
                    if p in keep:
                        n_in += 1
                n_out = 0
                for s in su.get(n, []):
                    if s in keep:
                        n_out += 1
                c0[n] = f"deg|{n_in}|{n_out}"
        elif col == "fuzzy":
            c0 = {}
            for n in lab:
                c0[n] = HD.fuzzy_label(nl[n])
        else:
            c0 = {}
            for n in lab:
                c0[n] = HD.api_label(n, lab, su)
        nodes = list(lab)
        midx = {}
        for i, n in enumerate(nodes):
            midx[n] = i
        sdk_list = []
        for n in nodes:
            sdk_list.append(sdk_node[n])
        meth_sdk = np.array(sdk_list, dtype=bool)
        # round-0 tokens, then ITERS directed WL rounds
        tid = []
        rnd = []
        morg = []
        for n in nodes:
            tid.append(_h("0|" + c0[n]))
            rnd.append(0)
            morg.append(midx[n])
        cur = c0
        for it in range(1, ITERS + 1):
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
            for n in nodes:
                tid.append(_h(f"{it}|{cur[n]}"))
                rnd.append(it)
                morg.append(midx[n])
        del nl, su, pr
        return fam, sha, col, (np.array(tid, dtype=np.uint64), np.array(rnd, dtype=np.int8),
                               np.array(morg, dtype=np.int32), meth_sdk)
    except Exception as e:
        return fam, sha, col, repr(e)[:100]


def staged():
    ids, _ = H.sample_ids("four")
    pos = set(ids.tolist())
    apps = []
    for f in H.PROXY:
        for s in H.staged(f):
            if s in pos:
                apps.append((f, s))
    return apps


def build(col, apps, workers, part_dir=None):
    """Token arrays for every app."""
    # with part_dir, each app is also saved on its own as it is built,
    # so a run killed by the memory guard resumes where it stopped
    data = {}
    todo = []
    for f, s in apps:
        if part_dir:
            pp = os.path.join(part_dir, f"{s}.npz")
        else:
            pp = None
        if pp and os.path.exists(pp):
            z = np.load(pp)
            data[s] = (f, (z["tid"], z["rnd"], z["morg"], z["sdk"]))
        else:
            todo.append((f, s, col))
    with ProcessPoolExecutor(workers) as ex:
        for fam, sha, c, r in ex.map(_job, todo):
            if isinstance(r, tuple):
                data[sha] = (fam, r)
                if part_dir:
                    np.savez(os.path.join(part_dir, f"{sha}.npz"), tid=r[0], rnd=r[1], morg=r[2], sdk=r[3])
            else:
                print("  skip", sha[:10], r, flush=True)
    return data


_CACHE_DIR = os.environ.get("TOK_CACHE_DIR", H.out_dir("tier4", "v3", "tok_cache"))


def build_cached(col, apps, workers):
    """build() with an on-disk pickle cache, so repeated v3 scripts do not re-parse FCGs."""
    import pickle
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{col}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    part_dir = os.path.join(_CACHE_DIR, f"{col}_parts")
    os.makedirs(part_dir, exist_ok=True)
    data = build(col, apps, workers, part_dir)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    import shutil
    shutil.rmtree(part_dir, ignore_errors=True)
    return data


def cv(X, y, g, models):
    out = {}
    for k in models:
        out[k] = []
    for r, k, tr, te in H.folds(y, g, 1):
        up = H.upsample(tr, y, k)
        for name, mk in models.items():
            m = mk().fit(X[up], y[up])
            out[name].append(f1_score(y[te], m.predict(X[te]), average="macro"))
    means = {}
    for k, v in out.items():
        means[k] = float(np.mean(v))
    return means


def count_matrix(shalist, data, token_filter=None, round_filter=None, min_apps=2):
    """Sparse token-count matrix over the given apps; vocabulary = tokens in >= min_apps."""
    per = {}
    for sha in shalist:
        tid, rnd, morg, msdk = data[sha][1]
        mask = np.ones(len(tid), dtype=bool)
        if round_filter is not None:
            mask &= (rnd == round_filter)
        if token_filter == "sdk":
            mask &= msdk[morg]
        elif token_filter == "host":
            mask &= ~msdk[morg]
        u, c = np.unique(tid[mask], return_counts=True)
        per[sha] = (u, c.astype(np.float32))
    if shalist:
        tok_parts = []
        for s in shalist:
            tok_parts.append(per[s][0])
        allt = np.concatenate(tok_parts)
    else:
        allt = np.array([], dtype=np.uint64)
    # app_of is unused, but np.concatenate([]) here is where an empty shalist raises ValueError
    app_parts = []
    for j, s in enumerate(shalist):
        app_parts.append(np.full(len(per[s][0]), j))
    app_of = np.concatenate(app_parts)
    del app_parts
    vocab, inv = np.unique(allt, return_inverse=True)
    napp = np.bincount(inv, minlength=len(vocab))
    sel = napp >= min_apps
    col_of = -np.ones(len(vocab), dtype=np.int64)
    col_of[sel] = np.arange(sel.sum())
    rows = []
    cols = []
    vals = []
    for j, s in enumerate(shalist):
        u, c = per[s]
        ci = col_of[np.searchsorted(vocab, u)]
        m = ci >= 0
        rows.append(np.full(m.sum(), j))
        cols.append(ci[m])
        vals.append(c[m])
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(shalist), int(sel.sum())))
    return X, vocab[sel]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    apps = staged()
    if a.limit:
        byf = defaultdict(list)
        for f, s in apps:
            byf[f].append((f, s))
        limited = []
        for f in byf:
            for t in byf[f][:a.limit]:
                limited.append(t)
        apps = limited
    fam_set = set()
    for f, _ in apps:
        fam_set.add(f)
    fam_names = sorted(fam_set)
    # resume: finished colorings are kept
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    logreg = lambda: LogisticRegression(C=10.0, max_iter=3000)
    if os.environ.get("GPU") == "1":  # same objective on the GPU: keeps the wide exact-token model off the host
        import gpu_models as GM
        if GM.AVAILABLE:
            logreg = lambda: GM.TorchLogReg(C=10.0, max_iter=3000)
    nb = lambda: MultinomialNB(alpha=0.1, fit_prior=False)

    for col in COLORINGS:
        if col in res:
            print(f"\n=== {col} === (done, kept)", flush=True)
            continue
        print(f"\n=== {col} ===", flush=True)
        part_dir = os.path.join(_CACHE_DIR, f"{col}_parts")  # per-app token cache (resume after memory aborts)
        os.makedirs(part_dir, exist_ok=True)
        data = build(col, apps, a.workers, part_dir)
        shalist = []
        for _, s in apps:
            if s in data:
                shalist.append(s)
        fam_list = []
        for s in shalist:
            fam_list.append(data[s][0])
        fam = np.array(fam_list)
        y_list = []
        for f in fam:
            y_list.append(fam_names.index(f))
        y = np.array(y_list)
        g = H.groups_for(np.array(shalist))
        rc = {}
        # L5 + L7 (exact) + L6
        X, vocab = count_matrix(shalist, data)
        Xl = X.copy()
        Xl.data = np.log1p(Xl.data)
        rc["L5"] = cv(Xl, y, g, {"naive_bayes": nb, "logistic": logreg})
        rc["L6_round"] = {}
        for rd in (0, 1, 2):
            Xr, _ = count_matrix(shalist, data, round_filter=rd)
            Xr.data = np.log1p(Xr.data)
            rc["L6_round"][rd] = cv(Xr, y, g, {"logistic": logreg})["logistic"]
        rc["L7_exact"] = {"logistic": rc["L5"]["logistic"], "n_vocab": int(len(vocab))}
        # L7 hashed on the same apps/folds
        rc["L7_hashed"] = {}
        for d in (2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16):
            Xh = np.zeros((len(shalist), d), dtype=np.float32)
            for j, s in enumerate(shalist):
                tid = data[s][1][0]
                idx = (tid % d).astype(np.int64)
                sgn = np.where((tid >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
                np.add.at(Xh[j], idx, sgn)
            nrm = np.linalg.norm(Xh, axis=1, keepdims=True)
            nrm[nrm == 0] = 1
            Xh /= nrm
            from sklearn.linear_model import SGDClassifier
            f1 = []
            for r, k, tr, te in H.folds(y, g, 1):
                up = H.upsample(tr, y, k)
                m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
                m.fit(Xh[up], y[up])
                f1.append(f1_score(y[te], m.predict(Xh[te]), average="macro"))
            rc["L7_hashed"][str(d)] = float(np.mean(f1))
        l6_rounds = []
        for r in (0, 1, 2):
            l6_rounds.append(round(rc['L6_round'][r], 3))
        print(f"  L5 nb={rc['L5']['naive_bayes']:.3f} logistic={rc['L5']['logistic']:.3f}  "
              f"L6 rounds={l6_rounds}", flush=True)
        hashed_parts = []
        for d, v in rc['L7_hashed'].items():
            hashed_parts.append(f'{d}:{v:.3f}')
        hashed_str = ', '.join(hashed_parts)
        print(f"  L7 exact={rc['L7_exact']['logistic']:.3f} hashed={{{hashed_str}}}", flush=True)

        if col == "identifier":
            rc.update(identity_analyses(shalist, data, y, g, fam, fam_names, apps, logreg, nb))
        res[col] = rc
        json.dump(res, open(OUT, "w"), indent=1)
        del data
    print("done", flush=True)


def identity_analyses(shalist, data, y, g, fam, fam_names, apps, logreg, nb):
    out = {}
    # X2: SDK-only vs host-only classification
    for scope in ("sdk", "host"):
        X, _ = count_matrix(shalist, data, token_filter=scope)
        X.data = np.log1p(X.data)
        if "X2" not in out:
            out["X2"] = {}
        out["X2"][scope] = cv(X, y, g, {"logistic": logreg})["logistic"]
    # X1 + X3: train logistic on full exact tokens, split evidence, localize
    X, vocab = count_matrix(shalist, data)
    Xl = X.copy()
    Xl.data = np.log1p(Xl.data)
    tok_pos = {}
    for i, t in enumerate(vocab):
        tok_pos[int(t)] = i
    sdk_frac_ev = defaultdict(list)
    p_at = defaultdict(list)
    for r, k, tr, te in H.folds(y, g, 1):
        up = H.upsample(tr, y, k)
        m = logreg().fit(Xl[up], y[up])  # the passed-in model (GPU with GPU=1; same objective as LogisticRegression(C=10))
        W = m.coef_
        for j in te:
            sha = shalist[j]
            f = fam[j]
            cls = fam_names.index(f)
            # closest rival class by score (own class excluded)
            class_scores = []
            for c in range(len(fam_names)):
                if c != cls:
                    class_scores.append(W[c, :] @ Xl[j].toarray().ravel())
                else:
                    class_scores.append(-1e9)
            rival = int(np.argmax(class_scores))
            u = W[cls] - W[rival]
            tid, rnd, morg, msdk = data[sha][1]
            # per emitted token contribution = u[token] (log-count weight shared); split by sdk origin
            col_list = []
            for t in tid:
                col_list.append(tok_pos.get(int(t), -1))
            cols = np.array(col_list)
            del col_list
            keep = cols >= 0
            contrib = np.zeros(len(tid))
            contrib[keep] = u[cols[keep]]
            tok_sdk = msdk[morg]
            pos_mass = np.clip(contrib, 0, None)
            tot = pos_mass.sum()
            if tot > 0:
                sdk_frac_ev[f].append(float(pos_mass[tok_sdk].sum() / tot))
            # X3 localization: method contribution, precision@|sdk methods|
            nm = len(msdk)
            mc = np.zeros(nm)
            np.add.at(mc, morg[keep], contrib[keep])
            nt = int(msdk.sum())
            if nt > 0:
                order = np.argsort(-mc)
                p_at[f].append(float(msdk[order[:nt]].mean()))
    x1 = {}
    for f, v in sdk_frac_ev.items():
        if v:
            x1[f] = float(np.median(v))
    out["X1_sdk_evidence_frac"] = x1
    x3 = {}
    for f, v in p_at.items():
        if v:
            x3[f] = float(np.median(v))
    out["X3_localization_ceiling"] = x3
    # X4: family-private token share over DEX groups
    gid = H.groups_for(np.array(shalist))
    # number of distinct DEX groups and families per token, from unique (token, group) / (token, family)
    # pairs in numpy (a Python set per token does not fit in memory for 40-class MalNet)
    ut = []
    for sha in shalist:
        ut.append(np.unique(data[sha][1][0]))
    T = np.concatenate(ut)
    # parts lists are freed right after each concatenation, as the comprehensions were
    g_parts = []
    for j, u in enumerate(ut):
        g_parts.append(np.full(len(u), int(gid[j]), dtype=np.int64))
    Gp = np.concatenate(g_parts)
    del g_parts
    f_parts = []
    for j, u in enumerate(ut):
        f_parts.append(np.full(len(u), int(y[j]), dtype=np.int64))
    Fp = np.concatenate(f_parts)
    del f_parts

    def n_distinct(other):
        pairs = np.unique(np.stack([T.astype(np.int64, copy=False).view(np.int64), other], 1), axis=0)
        toks, cnt = np.unique(pairs[:, 0], return_counts=True)
        return toks, cnt

    tg, cg = n_distinct(Gp)
    tf_, cf = n_distinct(Fp)
    perfam = defaultdict(list)
    for j, sha in enumerate(shalist):
        tv = ut[j].astype(np.int64, copy=False).view(np.int64)
        ng = cg[np.searchsorted(tg, tv)]
        nf = cf[np.searchsorted(tf_, tv)]
        perfam[fam[j]].append(float(((ng >= 2) & (nf == 1)).mean()))
    del ut, T, Gp, Fp
    x4 = {}
    for f, v in perfam.items():
        x4[f] = float(np.median(v))
    out["X4_private_recurring_by_group"] = x4
    # X5: tf-idf weighting before logistic
    from sklearn.feature_extraction.text import TfidfTransformer
    Xt = TfidfTransformer().fit_transform(X)
    out["X5_tfidf_logistic"] = cv(Xt, y, g, {"logistic": logreg})["logistic"]
    out["X5_plain_logistic"] = cv(Xl, y, g, {"logistic": logreg})["logistic"]
    print(f"  X1 sdk-evidence={out['X1_sdk_evidence_frac']}", flush=True)
    print(f"  X2 sdk-only/host-only={out['X2']}  X3 loc-ceiling={out['X3_localization_ceiling']}", flush=True)
    print(f"  X5 tfidf={out['X5_tfidf_logistic']:.3f} plain={out['X5_plain_logistic']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    main()
