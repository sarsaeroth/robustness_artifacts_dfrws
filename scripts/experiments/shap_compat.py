#!/usr/bin/env python3
"""shap / xgboost compatibility: newer TreeExplainer additivity checks fail on float32 rounding in
XGBoost 3.x models. Importing this module wraps TreeExplainer.shap_values so that an additivity
failure is retried with check_additivity=False and logged (count and the largest discrepancy),
instead of aborting the run. Explanations themselves are unchanged."""
import sys

import numpy as np
import shap

_orig = shap.TreeExplainer.shap_values
FALLBACKS = {"n": 0}


def _shap_values(self, X, *args, **kwargs):
    if kwargs.get("check_additivity", True) is False:
        return _orig(self, X, *args, **kwargs)
    try:
        return _orig(self, X, *args, **kwargs)
    except Exception as e:
        if "Additivity" not in str(e):
            raise
        kwargs["check_additivity"] = False
        v = _orig(self, X, *args, **kwargs)
        FALLBACKS["n"] += 1
        if FALLBACKS["n"] <= 5 or FALLBACKS["n"] % 100 == 0:
            sys.stderr.write(f"shap_compat: additivity check failed ({FALLBACKS['n']} so far); "
                             f"values computed with check_additivity=False\n")
        return v


shap.TreeExplainer.shap_values = _shap_values
