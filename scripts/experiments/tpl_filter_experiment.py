#!/usr/bin/env python3
"""Does filtering common third-party libraries fix the in-the-wild false positives?

Motivation (measured, not assumed): applying the paper's provenance attribution to
the 99 flagged in-the-wild apps showed the decisions are driven by ordinary
third-party libraries that the vectorizer does NOT filter, above all org/joda/time
(up to 84% of the luminati decision in four apps), plus io/fabric, com/facebook/ads
and com/koushikdutta/async. gen_wl_vectors.LIB_PREFIXES filters only ten prefixes
(java, javax, android, androidx, kotlin, com/google, org/apache, org/json, okhttp,
okio), so every other library enters the feature vector as if it were app code.

This is the fix the TPL literature implies: filter libraries by a principled list
instead of ten hardcoded strings. Two questions, and both must be answered:

  Q1 does accuracy survive?  Filtering more code could remove real signal.
  Q2 do the false positives drop?

The model is retrained under each filter, since a vector built with a different
filter is not comparable to a model trained under the old one.

Conditions:
  narrow  : gen_wl_vectors.LIB_PREFIXES (status quo)
  tpl     : narrow + a broad common-Android-library list (below)

Outputs: results/tpl_filter_experiment.json
"""
import json, os, sys
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import gen_wl_vectors as G
import train_wl_only_classifier as T
import graph_io as GI

PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
FLAGDIR = os.path.join(ROOT, "results", "flagged_fcg")
INVEST = os.path.join(ROOT, "results", "flagged_investigation.json")
DEX_GROUPS = os.path.join(ROOT, "results", "full_sweep", "dex_groups.json")
OUT = os.path.join(ROOT, "results", "tpl_filter_experiment.json")
CACHE = os.path.join(ROOT, "results", "tpl_filter_vecs.npz")
DIM = 1024
SEED = 42
N_FOLDS = 5
PER_FAM = 150
MAX_FCG = 50_000_000

# Common Android third-party libraries. Ad/analytics/mediation, crash reporting,
# networking, utility and UI libs: code that ships INSIDE apps but is not the app,
# and is not proxy SDK code either. Deliberately broad; over-filtering is the
# conservative direction for Q1.
TPL_EXTRA = (
    # the ones the attribution actually surfaced
    "Lorg/joda/", "Lio/fabric/", "Lcom/crashlytics/", "Lcom/koushikdutta/",
    "Lcom/facebook/", "Lcom/appodeal/", "Lcom/vungle/", "Lcom/applovin/",
    "Lcom/adcolony/", "Lcom/moat/", "Lcom/my/target/", "Lcom/yandex/",
    "Lcom/startapp/", "Lcom/mopub/",
    # other widespread ad / mediation / attribution / analytics SDKs
    "Lcom/unity3d/", "Lcom/ironsource/", "Lcom/chartboost/", "Lcom/inmobi/",
    "Lcom/tapjoy/", "Lcom/fyber/", "Lcom/smaato/", "Lcom/mintegral/",
    "Lcom/mbridge/", "Lcom/bytedance/", "Lcom/qq/e/", "Lcom/baidu/mobads/",
    "Lcom/flurry/", "Lcom/appsflyer/", "Lcom/adjust/", "Lcom/amplitude/",
    "Lcom/mixpanel/", "Lcom/onesignal/", "Lcom/localytics/", "Lcom/kochava/",
    "Lcom/umeng/", "Lcom/tappx/", "Lcom/appnext/", "Lcom/integralads/",
    "Lcom/oneaudience/", "Lcom/tutelatechnologies/", "Lcom/amazon/device/ads/",
    # networking / serialization / image / utility
    "Lretrofit2/", "Lcom/squareup/", "Lcom/bumptech/", "Lcom/nostra13/",
    "Lorg/greenrobot/", "Lio/reactivex/", "Lrx/", "Lcom/fasterxml/",
    "Lorg/slf4j/", "Lorg/jsoup/", "Lorg/xmlpull/", "Lorg/w3c/", "Lorg/jdom2/",
    "Lcz/msebera/", "Lorg/altbeacon/", "Lnet/i2p/", "Lcom/airbnb/",
    "Lde/greenrobot/", "Lorg/apmem/", "Lorg/reactivestreams/", "Lcom/j256/",
    "Lorg/acra/", "Lcom/loopj/", "Lorg/codehaus/", "Lorg/bouncycastle/",
    "Lorg/spongycastle/", "Lcom/android/installreferrer/",
)


def wl_filtered(nl, su, pr, extra):
    """gen_wl_vectors.wl_hashed_vector with an optionally widened library filter."""
    if extra:
        sub = {}
        for n, raw in nl.items():
            if not G.canonical_label(raw, "fcg").startswith(extra):
                sub[n] = raw
    else:
        sub = nl
    if not sub:
        return np.zeros(DIM, dtype=np.float32)
    return G.wl_hashed_vector(sub, su, pr, kind="fcg", dim=DIM)


def cv_macro_f1(X, y, groups):
    scores = []
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        pairs = []
        for j in tr:
            pairs.append((j, y[j]))
        up = T.upsample(pairs, seed=SEED)
        up_idx = []
        for j, _ in up:
            up_idx.append(j)
        idx = np.array(up_idx)
        clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4,
                            random_state=SEED, n_jobs=-1)
        clf.fit(X[idx], y[idx])
        scores.append(f1_score(y[te], clf.predict(X[te]), average="macro",
                               zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def build():
    rng = np.random.default_rng(SEED)
    dex_group = json.load(open(DEX_GROUPS))
    byfam = defaultdict(list)
    for hdir, gid, label in T.build_samples():
        byfam[label].append((hdir, gid))

    # narrow and tpl vectors for up to PER_FAM apps per class
    rows_n = []
    rows_t = []
    labels = []
    groups = []
    for fam in PROXY + ["non_proxy"]:
        lst = byfam.get(fam, [])[:]
        rng.shuffle(lst)
        used = 0
        for hdir, gid in lst:
            fcg = os.path.join(hdir, f"{gid}.fcg")
            if not GI.exists(fcg) or GI.size(fcg) > MAX_FCG:
                continue
            try:
                nl, su, pr = G.parse_gml_lightweight(fcg)
            except Exception:
                continue
            rows_n.append(wl_filtered(nl, su, pr, None))
            rows_t.append(wl_filtered(nl, su, pr, TPL_EXTRA))
            labels.append(fam)
            groups.append(dex_group.get(gid, f"solo_{gid}"))
            used += 1
            if used % 50 == 0:
                print(f"  [{fam}] {used}", flush=True)
            if used >= PER_FAM:
                break
        print(f"[{fam}] {used}", flush=True)
    np.savez_compressed(CACHE, narrow=np.vstack(rows_n), tpl=np.vstack(rows_t),
                        labels=np.array(labels), groups=np.array(groups))
    return np.vstack(rows_n), np.vstack(rows_t), labels, np.array(groups)


def main():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        Xn = z["narrow"]
        Xt = z["tpl"]
        labels = list(z["labels"])
        groups = z["groups"]
        print("[*] using cached corpus vectors", flush=True)
    else:
        Xn, Xt, labels, groups = build()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"\ncorpus: {len(y)} {dict(Counter(labels))}\n", flush=True)

    # Q1: grouped cross-validated accuracy under each filter
    res = {}
    for name, X in (("narrow", Xn), ("tpl", Xt)):
        m, s = cv_macro_f1(np.asarray(X, np.float32), y, groups)
        res[name] = {"macro_f1": m, "std": s}
        print(f"Q1 accuracy  {name:7} macro-F1 = {m:.4f} +/- {s:.4f}", flush=True)

    # Q2: in-the-wild flag rate on the exact classified builds
    inv = {}
    for a in json.load(open(INVEST))["apps"]:
        inv[a["sha256"]] = a
    fcgs = []
    if os.path.isdir(FLAGDIR):
        for f in os.listdir(FLAGDIR):
            if f.endswith(".fcg"):
                fcgs.append(f)
    print(f"\nQ2 flagged apps with cached FCG: {len(fcgs)}", flush=True)
    if fcgs:
        for name, X in (("narrow", Xn), ("tpl", Xt)):
            pairs = []
            for i in range(len(y)):
                pairs.append((i, y[i]))
            up = T.upsample(pairs, seed=SEED)
            up_idx = []
            for i, _ in up:
                up_idx.append(i)
            idx = np.array(up_idx)
            clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4,
                                random_state=SEED, n_jobs=-1)
            clf.fit(np.asarray(X, np.float32)[idx], y[idx])
            still = []
            tot = 0
            for f in fcgs:
                sha = f[:-4]
                try:
                    nl, su, pr = G.parse_gml_lightweight(os.path.join(FLAGDIR, f))
                except Exception:
                    continue
                if name == "tpl":
                    v = wl_filtered(nl, su, pr, TPL_EXTRA)
                else:
                    v = wl_filtered(nl, su, pr, None)
                p = str(le.classes_[int(clf.predict(v.reshape(1, -1))[0])])
                tot += 1
                if p in PROXY:
                    if sha in inv:
                        pkg = inv[sha].get("pkg", "?")
                    else:
                        pkg = "?"
                    still.append({"sha": sha, "pkg": pkg, "pred": p})
            res[name]["flagged_still_proxy"] = len(still)
            res[name]["flagged_total"] = tot
            res[name]["examples"] = still[:10]
            print(f"Q2 flag rate {name:7} {len(still)}/{tot} still predicted proxy",
                  flush=True)

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
