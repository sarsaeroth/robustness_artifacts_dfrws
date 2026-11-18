#!/usr/bin/env python3
"""Canonical classifiers, training and DEX grouping for the attribution experiments.

Every experiment takes these settings from here rather than defining its own.

Canonical settings (as written by run_full_sweep.py, results/open_world_sweep/summary.csv):

  sgd  SGDClassifier(loss="log_loss"), ONE epoch via partial_fit
  rf   RandomForestClassifier(n_estimators=200)
  xgb  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)

SGD here is a single pass. `sgd_mode="converged"` gives a fully fitted model and
should be used wherever the model itself is analysed (explanations,
certificates), since a one-epoch model is not at an optimum.

LEGACY_CONFIG reproduces the hinge-loss configuration used by
eval_original_open_world.py / eval_open_world_wlcapa.py (original-dataset five-class
and WL+capa evaluations). It exists only to regenerate those numbers.
"""
import hashlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

SEED = 42

LEGACY_CONFIG = {
    "sgd": dict(loss="hinge"),
    "rf": dict(n_estimators=100),
    "xgb": dict(n_estimators=100, max_depth=6, learning_rate=0.3),
}


def make_classifier(name, n_classes, legacy_config=False, n_jobs=-1, seed=SEED):
    if name == "sgd":
        if legacy_config:
            loss = LEGACY_CONFIG["sgd"]["loss"]
        else:
            loss = "log_loss"
        return SGDClassifier(loss=loss, random_state=seed, n_jobs=n_jobs)
    if name == "rf":
        if legacy_config:
            n = LEGACY_CONFIG["rf"]["n_estimators"]
        else:
            n = 200
        return RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=n_jobs)
    if name == "xgb":
        from xgboost import XGBClassifier
        if legacy_config:
            p = LEGACY_CONFIG["xgb"]
        else:
            p = dict(n_estimators=200, max_depth=6, learning_rate=0.1)
        return XGBClassifier(**p, random_state=seed, n_jobs=n_jobs,
                             verbosity=0)   # objective and class count inferred from y
    raise ValueError(name)


def fit_classifier(model, X, y, n_classes, sgd_mode="one_epoch"):
    """sgd_mode="one_epoch": one partial_fit pass (published sweep); "converged": full fit()."""
    if isinstance(model, SGDClassifier) and sgd_mode == "one_epoch":
        model.partial_fit(X, y, classes=np.arange(n_classes))
    else:
        model.fit(X, y)
    return model


def dex_group_ids(gids, apk_to_group):
    """Group id per sample; APKs absent from the DEX-group map become singletons above every real id."""
    # sha256-derived ids: stable under PYTHONHASHSEED (unlike hash()) and never collide with real group ids
    group_values = list(apk_to_group.values())
    if group_values:
        base = max(group_values) + 1
    else:
        base = -1 + 1
    out = []
    for gid in gids:
        g = apk_to_group.get(gid)
        if g is None:
            g = base + int(hashlib.sha256(gid.encode()).hexdigest()[:12], 16)
        out.append(g)
    return np.array(out, dtype=np.int64)


def group_of(apk_to_group, gid):
    """Scalar form of dex_group_ids, for drop-in replacement inside list comps."""
    return int(dex_group_ids([gid], apk_to_group)[0])
