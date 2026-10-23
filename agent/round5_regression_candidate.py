"""Inactive fixed-weight regression mixtures with unchanged strict-past history.

The caller freezes one of seven declared arms and a host-discovered history
schema. All internal calibration predictions are explicitly aligned by original
supplied-row positions before history selection. Components then freshly refit
on all supplied fit rows. There is no discovery, data loading or batch routing.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from agent.boosting_candidate import _check_deadline, _deadline, _inputs
from agent.regression_candidate import make_regression_recipe
from agent.regression_history import HistorySignals, apply_history, select_history
from agent.round5_regression_base import active_component, stage_component, refit_component


FAMILY = "round5_regression_ensemble_v1"
MIN_CALIBRATION_HISTORY_ROWS = 64
_ARMS = {
    "LGB": (("lightgbm", "all_declared", 1.0),),
    "E25": (("catboost", "all_declared", .75), ("lightgbm", "all_declared", .25)),
    "E50": (("catboost", "all_declared", .50), ("lightgbm", "all_declared", .50)),
    "E75": (("catboost", "all_declared", .25), ("lightgbm", "all_declared", .75)),
    "N": (("catboost", "declared_numeric", 1.0),),
    "N25": (("catboost", "all_declared", .75), ("catboost", "declared_numeric", .25)),
    "N50": (("catboost", "all_declared", .50), ("catboost", "declared_numeric", .50)),
}
ARMS = tuple(_ARMS)


def make_round5_recipe(period_column, arm, *, history_config=None):
    """Describe the full fixed algorithm; I is the unchanged external control."""
    if not isinstance(arm, str) or arm not in _ARMS:
        raise ValueError("Choose a preregistered round5 regression arm")
    components = []
    canonical = make_regression_recipe(period_column, history_config=history_config)
    for engine, subset, weight in _ARMS[arm]:
        base = make_regression_recipe(period_column, engine=engine)["base"]
        components.append({"id": engine + ("_numeric" if subset == "declared_numeric" else "_all"),
                           "weight": weight, "feature_subset": subset, "base": base})
    return {"version": 1, "family": FAMILY, "arm": arm, "components": components,
        "history_config": canonical["history_config"], "history_policy": "calibrated",
        "minimum_calibration_history_rows": MIN_CALIBRATION_HISTORY_ROWS,
        "calibration": {"split": "last_valid_period_inside_supplied_fit",
            "position_alignment": "original_iloc_sorted_exact_sets",
            "tree_alignment": "legacy_best_vs_min50_refit", "minimum_final_iterations": 50,
            "history_order": "select_on_aligned_mixture_before_all_component_refits",
            "target_scale": "original", "seed": 42},
        "resources": {"threads": 8, "device": "CPU", "execution": "sequential",
            "history_reserve_fraction": .05, "history_reserve_seconds_max": 8.0,
            "component_time_shares": [1.0 / len(components)] * len(components),
            "component_quota": "cumulative_staging_and_refit_wall_seconds",
            "unused_component_time": "not_transferred", "deadline": "one_supplied_absolute_deadline"},
        "fallback": {"no_internal_split": "fixed_400_trees_per_component",
            "no_history_schema": "base_ensemble", "history_calibration_unavailable": "fixed_mean_weight_0.5",
            "component_failure": "raise_no_partial_ensemble"}}


def _recipe(value):
    if not isinstance(value, dict) or value.get("family") != FAMILY:
        raise ValueError("A normalized round5 regression recipe is required")
    components = value.get("components")
    if not isinstance(components, list) or not components or not isinstance(components[0], dict):
        raise ValueError("A frozen round5 component list is required")
    try:
        period = components[0]["base"]["period_column"]
        checked = make_round5_recipe(period, value.get("arm"), history_config=value.get("history_config"))
    except (KeyError, TypeError):
        raise ValueError("A normalized round5 regression recipe is required") from None
    try:
        supplied_json = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        checked_json = json.dumps(checked, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Round5 frozen recipe must contain finite JSON values") from None
    # Python equality treats True == 1 == 1.0. A frozen identity must retain
    # its canonical JSON types, not merely compare numerically equivalent.
    if checked_json != supplied_json:
        raise ValueError("Round5 recipe differs from its complete frozen configuration")
    return checked


def _positions(value, label):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or not len(array)
            or (array < 0).any() or (array > np.iinfo(np.int64).max).any()
            or len(np.unique(array)) != len(array)):
        raise ValueError("Calibration " + label + " must contain unique nonnegative positions")
    return array.astype(np.int64, copy=True)


def align_calibrations(calibrations, weights):
    """Align component predictions using original iloc positions, never order.

Every component must hold exactly the same original rows and fit the same
strictly earlier positions. A partially absent/mismatched calibration fails
the entire mixture instead of changing its frozen weights.
"""
    if (not isinstance(calibrations, (list, tuple)) or not calibrations
            or not isinstance(weights, (list, tuple)) or len(weights) != len(calibrations)
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Calibration requires a finite normalized fixed mixture")
    if all(item is None for item in calibrations):
        return None
    fitting_common = held_common = mixed = None
    for calibration, weight in zip(calibrations, weights):
        if not isinstance(calibration, dict) or set(calibration) != {"fit_positions", "validation_positions", "predictions"}:
            raise ValueError("Every ensemble component requires the same calibration contract")
        fitting = _positions(calibration["fit_positions"], "fit")
        held = _positions(calibration["validation_positions"], "held")
        if np.intersect1d(fitting, held).size:
            raise ValueError("Calibration fitting and held positions overlap")
        values = np.asarray(calibration["predictions"], dtype=float)
        if values.shape != held.shape or not np.isfinite(values).all():
            raise ValueError("Calibration predictions must match held positions")
        order = np.argsort(held)
        fitting, held, values = np.sort(fitting), held[order], values[order]
        if fitting_common is None:
            fitting_common, held_common = fitting, held
            mixed = np.zeros(len(held), dtype=float)
        elif not np.array_equal(fitting_common, fitting) or not np.array_equal(held_common, held):
            raise ValueError("Ensemble components selected different calibration positions")
        mixed += weight * values
    if not np.isfinite(mixed).all():
        raise ValueError("Ensemble calibration mixture is nonfinite")
    return {"fit_positions": fitting_common.copy(), "validation_positions": held_common.copy(),
            "predictions": mixed}


def _position_hash(values):
    return hashlib.sha256(json.dumps([int(value) for value in values], separators=(",", ":")).encode()).hexdigest()


def _component_inputs(X_fit, X_predict, context, component):
    derived = deepcopy(context)
    if component["feature_subset"] == "declared_numeric":
        numeric_set = set(context["numeric"])
        names = [name for name in context["features"] if name in numeric_set]
        if not names:
            raise ValueError("Numeric companion requires declared numeric features")
        derived.update(features=names, numeric=names.copy(), categorical=[])
        return X_fit.loc[:, names].copy(deep=True), X_predict.loc[:, names].copy(deep=True), derived
    return X_fit.copy(deep=True), X_predict.copy(deep=True), derived


def execute_round5_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    if not isinstance(context, dict):
        raise ValueError("A declared host context is required")
    started = time.monotonic()
    recipe = _recipe(context.get("recipe"))
    deadline = _deadline(context)
    # Validate the original full feature contract even for the numeric-only arm.
    from agent.model_config import make_model_config
    first = recipe["components"][0]["base"]
    _inputs(X_fit, y_fit, X_predict, context, {"base_recipe": {"model_config":
        make_model_config(first["engine"], "regression", first["parameters"])}})
    config = recipe["history_config"]
    if config is not None:
        if (not set(config["key_columns"] + [config["period_column"]]) <= set(context["features"])
                or not set(config["numeric_columns"]) <= set(context["numeric"])):
            raise ValueError("History configuration must use original declared feature roles")
    reserve = min(recipe["resources"]["history_reserve_seconds_max"],
                  max(0.0, (deadline - started) * recipe["resources"]["history_reserve_fraction"]))
    base_deadline = deadline - reserve
    base_seconds = max(0.0, base_deadline - started)
    stages = []
    weights = [component["weight"] for component in recipe["components"]]
    for component, share in zip(recipe["components"], recipe["resources"]["component_time_shares"]):
        component_started = time.monotonic()
        fit, predict, derived = _component_inputs(X_fit, X_predict, context, component)
        copying_seconds = time.monotonic() - component_started
        quota = base_seconds * share
        staged = stage_component(fit, y_fit, predict, derived, component,
            quota_seconds=quota - copying_seconds, base_deadline=base_deadline, original_deadline=deadline)
        staged.quota_seconds = quota
        staged.consumed_seconds += copying_seconds
        staged.evidence.update(quota_seconds=quota, input_copy_elapsed_seconds=copying_seconds)
        stages.append(staged)
    calibration = align_calibrations([stage.calibration for stage in stages], weights)
    _check_deadline(deadline)
    calibration_started = time.monotonic()
    selected = {"mode": "mean", "weight": .5, "shrinkage": 0}
    reason = "no_frozen_history_schema" if config is None else "no_internal_period_mean_fallback"
    calibration_evidence = None
    alignment = {"available": calibration is not None, "fit_positions_sha256": None,
                 "validation_positions_sha256": None, "fit_rows": 0, "validation_rows": 0}
    if calibration is not None:
        fitting, held = calibration["fit_positions"], calibration["validation_positions"]
        alignment.update(fit_positions_sha256=_position_hash(fitting),
            validation_positions_sha256=_position_hash(held), fit_rows=len(fitting), validation_rows=len(held))
        if config is not None:
            y = np.asarray(y_fit, dtype=float)
            signals = HistorySignals(config).fit(X_fit.iloc[fitting], y[fitting]).transform(X_fit.iloc[held])
            reason = "insufficient_calibration_history_mean_fallback"
            if int(signals["eligible"].sum()) >= MIN_CALIBRATION_HISTORY_ROWS:
                calibration_evidence = select_history(signals, y[held], calibration["predictions"])
                selected = {key: calibration_evidence[key] for key in ("mode", "weight", "shrinkage")}
                reason = "fit_only_calibrated"
    calibration_seconds = time.monotonic() - calibration_started
    _check_deadline(deadline)
    # No component full fit occurs until ensemble history selection is frozen.
    predictions, component_evidence = [], []
    for stage in stages:
        values, measured = refit_component(stage, base_deadline=base_deadline, original_deadline=deadline)
        predictions.append(values)
        component_evidence.append(measured)
    base = sum(weight * values for weight, values in zip(weights, predictions))
    if base.shape != (len(X_predict),) or not np.isfinite(base).all():
        raise ValueError("Round5 mixture returned invalid predictions")
    history_started = time.monotonic()
    mean, calibrated = base.copy(), base.copy()
    coverage = {"eligible_rows": None, "ambiguous_rows": None, "no_history_rows": None,
                "schema_available": config is not None}
    if config is not None:
        signals = HistorySignals(config).fit(X_fit, np.asarray(y_fit, dtype=float)).transform(X_predict)
        mean = apply_history(base, signals, mode="mean", weight=.5, shrinkage=0)
        calibrated = apply_history(base, signals, **selected)
        coverage.update(eligible_rows=int(signals["eligible"].sum()),
                        ambiguous_rows=int(signals["ambiguous"].sum()),
                        no_history_rows=int((signals["past_rows"] == 0).sum()))
    alternatives = {"base": base, "mean": mean, "calibrated": calibrated}
    for component, values in zip(recipe["components"], predictions):
        alternatives["component_" + component["id"]] = values.copy()
    _check_deadline(deadline)
    evidence = {"version": 1, "family": FAMILY, "arm": recipe["arm"], "status": "completed",
        "components": component_evidence, "calibration_alignment": alignment,
        "history": {"config": deepcopy(config), "policy": "calibrated", "reason": reason,
            "selected": selected, "calibration": calibration_evidence, **coverage,
            "calibration_elapsed_seconds": calibration_seconds,
            "prediction_elapsed_seconds": time.monotonic() - history_started},
        "original_deadline_monotonic": deadline, "base_deadline_monotonic": base_deadline,
        "history_reserve_seconds": reserve, "fallback": "none",
        "full_fit_rows": len(X_fit), "predict_rows": len(X_predict), "fresh_full_refit": True,
        "elapsed_seconds": time.monotonic() - started, "api_requests": 0,
        "physical_fits_expected_completed": sum(row["physical_fits_expected_completed"] for row in component_evidence),
        "physical_fit_count_source": "expected_successful_stages; actual starts/completions require observer"}
    result = {"predictions": calibrated}
    return (result, evidence, alternatives) if return_evidence else result


def fit_predict(X_fit, y_fit, X_predict, context):
    return execute_round5_candidate(X_fit, y_fit, X_predict, context)


def candidate_source():
    """Ordinary candidate entry point with data-free diagnostics sidecars."""
    return '''import hashlib,json
from pathlib import Path
import pandas as pd
from agent.round5_regression_candidate import execute_round5_candidate

def fit_predict(X_fit,y_fit,X_predict,context):
    result,evidence,alternatives=execute_round5_candidate(
        X_fit,y_fit,X_predict,context,return_evidence=True)
    path=Path("regression_ablation.csv")
    with path.open("x",encoding="utf-8",newline="") as stream:
        pd.DataFrame(alternatives).to_csv(stream,index=False)
    evidence["ablation_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    with Path("regression_evidence.json").open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,sort_keys=True,allow_nan=False)
    return result
'''
