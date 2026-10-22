"""Versioned round4 CPU regression candidate with fold-local feature blocks.

The host discovers and freezes any history configuration once, from its allowed
INNER-fit features. This candidate never rediscovers on outer/full refit and
never consumes host-held answers. Importing it does not activate any router.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from agent.boosting_candidate import _check_deadline, _deadline
from agent.history_candidate import HistoryConfig
from agent.round4_regression_base import fit_base
from agent.round4_regression_features import normalize_feature_recipe
from agent.regression_candidate import make_regression_recipe as incumbent_recipe, _recipe as validate_incumbent
from agent.regression_history import HistorySignals, apply_history, select_history


FAMILY = "round4_regression_v1"
MIN_CALIBRATION_HISTORY_ROWS = 64


def make_round4_recipe(period_column, *, engine="catboost", history_config=None,
                       history_policy="calibrated", parameters=None,
                       max_iterations=2000, stopping_rounds=100, fallback_iterations=400,
                       features=None, calibration_alignment="legacy"):
    """Freeze the complete bounded algorithm; old rescue-v1 remains separate."""
    if not isinstance(calibration_alignment, str) or calibration_alignment not in {"legacy", "refit_count"}:
        raise ValueError("Unknown calibration alignment")
    recipe = incumbent_recipe(period_column, engine=engine, history_config=history_config,
        history_policy=history_policy, parameters=parameters, max_iterations=max_iterations,
        stopping_rounds=stopping_rounds, fallback_iterations=fallback_iterations)
    recipe.update(family=FAMILY, features=normalize_feature_recipe(features),
                  calibration_alignment=calibration_alignment)
    return recipe


def _recipe(value):
    if not isinstance(value, dict) or value.get("family") != FAMILY:
        raise ValueError("A normalized round4 regression recipe is required")
    old = deepcopy(value)
    features = old.pop("features", None)
    alignment = old.pop("calibration_alignment", None)
    old["family"] = "rescue_regression_v1"
    normalized = validate_incumbent(old)
    checked = make_round4_recipe(**normalized["base"], history_config=normalized["history_config"],
        history_policy=normalized["history_policy"], features=features,
        calibration_alignment=alignment)
    if checked != value:
        raise ValueError("Use a normalized frozen round4 regression recipe")
    return checked


def execute_round4_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    if not isinstance(context, dict):
        raise ValueError("A declared host context is required")
    recipe = _recipe(context.get("recipe"))
    deadline = _deadline(context)
    config = recipe["history_config"]
    if config is not None:
        names = set(config["key_columns"]) | {config["period_column"]}
        if not names <= set(context.get("features", [])) or not set(config["numeric_columns"]) <= set(context.get("numeric", [])):
            raise ValueError("History configuration must use declared feature roles")
    # Reserve a small portion of the ORIGINAL deadline for bounded history and
    # output preparation, without extending the host's deadline.
    started = time.monotonic()
    reserve = min(8.0, max(0.0, (deadline - started) * .05))
    base_context = dict(context, deadline_monotonic=deadline - reserve)
    base, base_evidence, calibration = fit_base(X_fit, y_fit, X_predict, base_context, recipe["base"],
        feature_config=recipe["features"], calibration_alignment=recipe["calibration_alignment"])
    _check_deadline(deadline)
    history_started = time.monotonic()
    selected = {"mode": "mean", "weight": .5, "shrinkage": 0}
    reason = "fixed_mean_policy"
    calibration_evidence = None
    mean, calibrated = base.copy(), base.copy()
    coverage = {"eligible_rows": 0, "ambiguous_rows": 0, "no_history_rows": len(base)}
    if config is None:
        reason = "no_frozen_history_schema"
    else:
        y = np.asarray(y_fit, dtype=float)
        if recipe["history_policy"] == "calibrated":
            if calibration is None:
                reason = "no_internal_period_mean_fallback"
            else:
                fitting, held = calibration["fit_positions"], calibration["validation_positions"]
                # Crucially fit this encoder on the SAME earlier positions as
                # the calibration base. Full fit labels cannot enter it.
                signals = HistorySignals(config).fit(X_fit.iloc[fitting], y[fitting]).transform(X_fit.iloc[held])
                if int(signals["eligible"].sum()) < MIN_CALIBRATION_HISTORY_ROWS:
                    reason = "insufficient_calibration_history_mean_fallback"
                else:
                    calibration_evidence = select_history(signals, y[held], calibration["predictions"])
                    selected = {key: calibration_evidence[key] for key in ("mode", "weight", "shrinkage")}
                    reason = "fit_only_calibrated"
        signals = HistorySignals(config).fit(X_fit, y).transform(X_predict)
        mean = apply_history(base, signals, mode="mean", weight=.5, shrinkage=0)
        calibrated = apply_history(base, signals, **selected)
        coverage = {"eligible_rows": int(signals["eligible"].sum()),
                    "ambiguous_rows": int(signals["ambiguous"].sum()),
                    "no_history_rows": int((signals["past_rows"] == 0).sum())}
    alternatives = {"base": base, "mean": mean, "calibrated": calibrated}
    policy = recipe["history_policy"]
    values = alternatives["base" if policy == "off" else policy]
    _check_deadline(deadline)
    result = {"predictions": values}
    evidence = {"version": 1, "family": FAMILY, "status": "completed", "base": base_evidence,
        "features": deepcopy(recipe["features"]),
        "calibration_alignment": recipe["calibration_alignment"],
        "history": {"config": deepcopy(config), "policy": policy, "reason": reason,
                    "selected": selected, "calibration": calibration_evidence, **coverage,
                    "elapsed_seconds": time.monotonic() - history_started},
        "full_fit_rows": len(X_fit), "predict_rows": len(X_predict), "fresh_full_refit": True,
        "elapsed_seconds": time.monotonic() - started, "api_requests": 0}
    return (result, evidence, alternatives) if return_evidence else result


def fit_predict(X_fit, y_fit, X_predict, context):
    return execute_round4_candidate(X_fit, y_fit, X_predict, context)


def candidate_source():
    """Standalone candidate entry point, plus auditable non-scoring sidecars."""
    return '''import hashlib,json
from pathlib import Path
import pandas as pd
from agent.round4_regression_candidate import execute_round4_candidate

def fit_predict(X_fit,y_fit,X_predict,context):
    result,evidence,alternatives=execute_round4_candidate(
        X_fit,y_fit,X_predict,context,return_evidence=True)
    path=Path("regression_ablation.csv")
    with path.open("x",encoding="utf-8",newline="") as stream:
        pd.DataFrame(alternatives).to_csv(stream,index=False)
    evidence["ablation_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    with Path("regression_evidence.json").open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,sort_keys=True,allow_nan=False)
    return result
'''
