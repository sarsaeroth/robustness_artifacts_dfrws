#!/usr/bin/env python3
"""Temporal drift and date-label confounding within one corpus (DATASET env), on vectors only.

Dates come from results/datasets/<ds>/dates.json (dataset_dates.py: AndroZoo index; for corpus v3 also
vt_first_submission, VirusTotal's first_submission_date from vt_first_seen.py). DATE_FIELDS selects the fields.
Each analysis runs once per date field with at least half of the apps dated:
  first_seen  AndroZoo's vt_scan_date (first VirusTotal scan; available for every corpus)
  dex_date    build timestamp (1980-01-01 placeholder treated as missing)
Classes: the evaluation targets plus the benign class (sample_ids("five")); identifier WL
(labeled, GRAPHS env), groups from LEAK_GROUPS.

  coverage       per class: n, fraction dated, 10th/50th/90th percentile year
  date_only      macro-F1 of a depth-limited tree predicting the class from the date alone (grouped
                 5-fold), against a stratified-prior baseline: how much of the label the date encodes
  era            per class with spread: AUROC of a linear model predicting late (after the class
                 median) vs early from the features (grouped 5-fold): does the code change with time
                 inside a class
  global_split   train before the corpus median date, test after (exp_openset_lofo temporal), with
                 the class composition of each side
  within_class   every class split at its own median date: train on all early halves, test on all
                 late halves (test apps sharing a group with training removed), against a grouped
                 random split of the same dated apps: the drift gap with every class on both sides
  rolling        train on apps before year c, test on year c, for each c with at least two classes
                 tested; AUT = trapezoid area under F1(c) over the windows (Pendlebury et al.)
  feature_drift  per class and year (>= 10 apps): cosine distance of the class centroid to its first
                 year and to the previous year, and the share of active features (in >= 10% of the
                 year's apps) never active in earlier years
  oof_by_year    grouped 5-fold out-of-fold accuracy per class and year, and Spearman of correctness
                 with date per class
  versions       packages with >= 3 dated builds: Spearman of feature distance with version and time
                 gap between builds, and out-of-fold accuracy of the first vs the last build
  provenance     (corpus v3) label source and marker confirmation by year per class
Output: tier4/v4/temporal_drift.json (dataset-aware).
"""
import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_harness as H
import exp_l1_ladder as L

# TEMPORAL_SET=four: evaluation targets only (the dataset paper's protocol, no benign class)
SET = os.environ.get("TEMPORAL_SET", "five")
if SET == "five":
    OUT = H.out_path("tier4", "v4", "temporal_drift.json")
else:
    OUT = H.out_path("tier4", "v4", f"temporal_drift_{SET}.json")
REP = "labeled"
MIN_CLASS = 20  # dated apps a class needs to enter the split analyses


def years(ids, field):
    if H.V3:
        ds = "proxy_v3"
    else:
        ds = H.DATASET
    p = os.path.join(H.ROOT, "results", "datasets", ds, "dates.json")
    meta = json.load(open(p))
    # fractional year per app; NaN when missing or the 1980 placeholder
    t = np.full(len(ids), np.nan)
    for k, s in enumerate(ids):
        key = s.lower()
        if key in meta:
            d = meta[key].get(field, "")[:10]
        else:
            d = ""
        if len(d) == 10 and not d.startswith("1980"):
            y = int(d[:4])
            m = int(d[5:7])
            dd = int(d[8:10])
            t[k] = y + (m - 1) / 12 + (dd - 1) / 365
    return t, meta


def macro(yt, yp, labels):
    return float(f1_score(yt, yp, average="macro", labels=labels))


def fit_eval(X, y, g, tr, te, seed=0):
    present = sorted(set(y[tr].tolist()) & set(y[te].tolist()))
    m, _ = L.fit_linear(X, y, g, tr, seed)
    keep = te[np.isin(y[te], present)]
    return m, keep, macro(y[keep], m.predict(X[keep]), present)


def coverage(t, y, names):
    out = {}
    for c, n in enumerate(names):
        k = y == c
        v = t[k & ~np.isnan(t)]
        if len(v):
            percentiles = []
            for q in np.percentile(v, [10, 50, 90]):
                percentiles.append(round(float(q), 2))
        else:
            percentiles = None
        out[n] = {"n": int(k.sum()), "dated": round(float(len(v) / max(k.sum(), 1)), 3),
                  "p10_p50_p90": percentiles}
    return out


def date_only(t, y, g, ok):
    T = t[ok, None]
    Y = y[ok]
    G = g[ok]
    labs = sorted(set(Y.tolist()))
    f_tree = []
    f_prior = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(T, Y, G):
        m = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0).fit(T[tr], Y[tr])
        f_tree.append(macro(Y[te], m.predict(T[te]), labels=labs))
        d = DummyClassifier(strategy="stratified", random_state=0).fit(T[tr], Y[tr])
        f_prior.append(macro(Y[te], d.predict(T[te]), labels=labs))
    return {"n": int(ok.sum()), "f1_date_only": float(np.mean(f_tree)), "f1_prior": float(np.mean(f_prior))}


def era(X, t, y, g, ok, names):
    out = {}
    Xn = normalize(X)
    for c, n in enumerate(names):
        # classes need >= 40 dated apps and an interquartile range of >= 0.5 years
        k = np.where(ok & (y == c))[0]
        if len(k) < 40:
            out[n] = {"skipped": f"{len(k)} dated apps or interquartile range < 0.5 years"}
            continue
        q75, q25 = np.percentile(t[k], [75, 25])
        if q75 - q25 < 0.5:
            out[n] = {"skipped": f"{len(k)} dated apps or interquartile range < 0.5 years"}
            continue
        late = (t[k] >= np.median(t[k])).astype(int)
        gk = g[k]
        if len(set(gk.tolist())) < 5:
            out[n] = {"skipped": "fewer than 5 groups"}
            continue
        # out-of-fold late-vs-early scores
        sc = np.full(len(k), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(k, late, gk):
            if len(set(late[tr].tolist())) < 2:
                continue
            m = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-5, random_state=0,
                              class_weight="balanced").fit(Xn[k[tr]], late[tr])
            d = m.decision_function(Xn[k[te]])
            if d.ndim == 2:  # the GPU solver returns one column per class
                sc[te] = d[:, -1] - d[:, 0]
            else:
                sc[te] = d
        v = ~np.isnan(sc)
        if len(set(late[v].tolist())) == 2:
            auroc = float(roc_auc_score(late[v], sc[v]))
        else:
            auroc = None
        out[n] = {"n": int(len(k)), "split_year": round(float(np.median(t[k])), 2),
                  "auroc_late_vs_early": auroc}
    return out


def global_split(X, t, y, g, ok, names):
    med = np.median(t[ok])
    tr = np.where(ok & (t < med))[0]
    te = np.where(ok & (t >= med))[0]
    _, keep, f1 = fit_eval(X, y, g, tr, te)
    # class composition of each side
    train_classes = {}
    for c, n in sorted(Counter(y[tr].tolist()).items()):
        train_classes[names[c]] = int(n)
    test_classes = {}
    for c, n in sorted(Counter(y[te].tolist()).items()):
        test_classes[names[c]] = int(n)
    return {"split_year": round(float(med), 2), "f1": f1, "n_test_scored": int(len(keep)),
            "train_classes": train_classes, "test_classes": test_classes}


def within_class(X, t, y, g, ok, names):
    cls = []
    for c in range(len(names)):
        if (ok & (y == c)).sum() >= MIN_CLASS:
            cls.append(c)
    use = np.where(ok & np.isin(y, cls))[0]
    # each class split at its own median date
    early = np.zeros(len(y), bool)
    for c in cls:
        k = use[y[use] == c]
        early[k[t[k] < np.median(t[k])]] = True
    tr = use[early[use]]
    te = use[~early[use]]
    te = te[~np.isin(g[te], g[tr])]  # no group on both sides
    _, keep, f1_t = fit_eval(X, y, g, tr, te)
    # grouped random split of the same apps
    f1_r = []
    for r in range(3):
        splits = StratifiedGroupKFold(2, shuffle=True, random_state=r).split(use, y[use], g[use])
        a, b = next(splits)
        _, _, f1_random = fit_eval(X, y, g, use[a], use[b], r)
        f1_r.append(f1_random)
    class_names = []
    for c in cls:
        class_names.append(names[c])
    return {"classes": class_names, "n_train": int(len(tr)), "n_test": int(len(keep)),
            "f1_temporal": f1_t, "f1_random": float(np.mean(f1_r)), "gap": float(np.mean(f1_r) - f1_t)}


def rolling(X, t, y, g, ok, names):
    yrs = np.floor(t)
    wins = []
    for c in range(int(np.nanmin(yrs[ok])) + 1, int(np.nanmax(yrs[ok])) + 1):
        # train before year c, test on year c (no shared groups)
        tr = np.where(ok & (yrs < c))[0]
        te = np.where(ok & (yrs == c))[0]
        te = te[~np.isin(g[te], g[tr])]
        trc = set()
        for k, n in Counter(y[tr].tolist()).items():
            if n >= 10:
                trc.add(k)
        tec = set()
        for k, n in Counter(y[te].tolist()).items():
            if n >= 5:
                tec.add(k)
        both = sorted(trc & tec)
        if len(both) < 2:
            continue
        _, keep, f1 = fit_eval(X, y, g, tr, te[np.isin(y[te], both)])
        unseen = 0
        for k, n in Counter(y[te].tolist()).items():
            if k not in trc:
                unseen += n
        both_names = []
        for k in both:
            both_names.append(names[k])
        wins.append({"year": c, "f1": f1, "classes": both_names, "n_test": int(len(keep)),
                     "test_apps_of_untrained_classes": int(unseen)})
        print("    rolling", c, round(f1, 3), both_names, flush=True)
    # area under F1 over the windows
    f = []
    for w in wins:
        f.append(w["f1"])
    if len(f) > 1:
        aut = float(np.trapezoid(f) / (len(f) - 1))
    elif f:
        aut = f[0]
    else:
        aut = None
    return {"windows": wins, "aut": aut}


def feature_drift(X, t, y, ok, names):
    Xn = normalize(X)
    act = X > 0
    out = {}
    for c, n in enumerate(names):
        # years with >= 10 apps of this class
        k = np.where(ok & (y == c))[0]
        by = defaultdict(list)
        for i in k:
            by[int(t[i])].append(i)
        ys_unsorted = []
        for yr, v in by.items():
            if len(v) >= 10:
                ys_unsorted.append(yr)
        ys = sorted(ys_unsorted)
        if len(ys) < 2:
            continue
        # centroid distance to the first and previous year, and newly active features
        rows = []
        first = None
        prev = None
        seen = np.zeros(X.shape[1], bool)
        for yr in ys:
            cen = Xn[by[yr]].mean(0)
            cen /= max(np.linalg.norm(cen), 1e-12)
            a = act[by[yr]].mean(0) >= 0.1
            if first is None:
                dist_first = None
                new_active_share = None
            else:
                dist_first = float(1 - cen @ first)
                new_active_share = float((a & ~seen).sum() / max(a.sum(), 1))
            if prev is None:
                dist_prev = None
            else:
                dist_prev = float(1 - cen @ prev)
            rows.append({"year": yr, "n": len(by[yr]),
                         "dist_first": dist_first,
                         "dist_prev": dist_prev,
                         "new_active_share": new_active_share})
            if first is None:
                first = cen
            prev = cen
            seen = seen | a
        out[n] = rows
    return out


def oof(X, y, g):
    pred = np.full(len(y), -1)
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g)):
        m, _ = L.fit_linear(X, y, g, tr, f)
        pred[te] = m.predict(X[te])
    return pred


def oof_by_year(pred, t, y, ok, names):
    out = {}
    corr = (pred == y).astype(float)
    for c, n in enumerate(names):
        k = np.where(ok & (y == c))[0]
        if len(k) < MIN_CLASS:
            continue
        # correctness per year, and its rank correlation with date
        by = defaultdict(list)
        for i in k:
            by[int(t[i])].append(corr[i])
        if corr[k].std() > 0:
            rho = spearmanr(t[k], corr[k]).statistic
        else:
            rho = None
        by_year = {}
        for yr, v in sorted(by.items()):
            by_year[str(yr)] = [len(v), round(float(np.mean(v)), 3)]
        if rho is None or np.isnan(rho):
            spearman_correct_date = None
        else:
            spearman_correct_date = float(rho)
        out[n] = {"by_year": by_year,
                  "spearman_correct_date": spearman_correct_date}
    return out


def versions(X, t, y, ok, ids, meta, pred, names):
    # dated builds per package, by version code
    pk = defaultdict(list)
    for k, s in enumerate(ids):
        key = s.lower()
        if key in meta:
            m = meta[key]
        else:
            m = {}
        if ok[k] and m.get("pkg") and m.get("vercode", "").isdigit():
            pk[m["pkg"]].append((int(m["vercode"]), t[k], k))
    # consecutive-version pairs and first/last build correctness
    Xn = normalize(X)
    pairs = []
    firstlast = []
    per_class = Counter()
    for p, v in pk.items():
        if len(v) < 3:
            continue
        v.sort()
        per_class[names[y[v[0][2]]]] += 1
        for idx in range(len(v) - 1):
            va, ta, a = v[idx]
            vb, tb, b = v[idx + 1]
            pairs.append((abs(tb - ta), 1 - float(Xn[a] @ Xn[b])))
        firstlast.append((pred[v[0][2]] == y[v[0][2]], pred[v[-1][2]] == y[v[-1][2]]))
    if len(pairs) < 10:
        return {"skipped": f"{len(pairs)} consecutive-version pairs"}
    pair_columns = np.array(pairs).T
    dt = pair_columns[0]
    dd = pair_columns[1]
    fl = np.array(firstlast, float)
    return {"packages": len(firstlast), "packages_by_class": dict(per_class), "pairs": len(pairs),
            "spearman_dist_timegap": float(spearmanr(dt, dd).statistic),
            "median_dist_consecutive": float(np.median(dd)),
            "acc_first_build": float(fl[:, 0].mean()), "acc_last_build": float(fl[:, 1].mean())}


def provenance(ids, t, y, ok, names):
    man = {}
    for l in open(os.path.join(H.ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        r = json.loads(l)
        man[r["sha256"]] = r
    # label source and marker confirmation per class and year
    out = {}
    for c, n in enumerate(names):
        tab = defaultdict(Counter)
        for k in np.where(ok & (y == c))[0]:
            if ids[k] in man:
                r = man[ids[k]]
            else:
                r = {}
            tab[int(t[k])][r.get("label_source", "background")] += 1
            tab[int(t[k])]["marker_confirmed"] += int(bool(r.get("code_confirmed")))
        per_year = {}
        for yr, v in sorted(tab.items()):
            per_year[str(yr)] = dict(v)
        out[n] = per_year
    return out


def main():
    ids, fam = H.sample_ids(SET)
    y, le = H.labels(fam)
    names = le.classes_.tolist()
    X = H.matrix(REP, ids).astype(np.float64)
    g = H.groups_for(ids)
    res = {"dataset": H.DATASET, "set": SET, "groups": H.GROUPS, "graphs": list(H.GRAPHS), "n": int(len(ids)), "fields": {}}
    pred = None
    for field in os.environ.get("DATE_FIELDS", "first_seen,dex_date").split(","):
        # dates for this field; skip fields with under half the apps dated
        t, meta = years(ids, field)
        ok = ~np.isnan(t)
        r = {"dated": round(float(ok.mean()), 3), "coverage": coverage(t, y, names)}
        res["fields"][field] = r
        print(field, "dated", r["dated"], flush=True)
        if ok.mean() < 0.5:
            r["skipped"] = "fewer than half of the apps dated"
            continue
        # analyses, saving after the slow ones
        r["date_only"] = date_only(t, y, g, ok)
        print("  date_only", r["date_only"], flush=True)
        r["era"] = era(X, t, y, g, ok, names)
        r["feature_drift"] = feature_drift(X, t, y, ok, names)
        json.dump(res, open(OUT, "w"), indent=1)
        r["global_split"] = global_split(X, t, y, g, ok, names)
        print("  global", r["global_split"]["f1"], flush=True)
        r["within_class"] = within_class(X, t, y, g, ok, names)
        within_shown = {}
        for k in ("f1_temporal", "f1_random", "gap"):
            within_shown[k] = r["within_class"][k]
        print("  within_class", within_shown, flush=True)
        r["rolling"] = rolling(X, t, y, g, ok, names)
        json.dump(res, open(OUT, "w"), indent=1)
        if pred is None:
            pred = oof(X, y, g)
            res["oof_f1"] = macro(y, pred, list(range(len(names))))
        r["oof_by_year"] = oof_by_year(pred, t, y, ok, names)
        r["versions"] = versions(X, t, y, ok, ids, meta, pred, names)
        print("  versions", r["versions"], flush=True)
        if H.V3:
            r["provenance"] = provenance(ids, t, y, ok, names)
        json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
