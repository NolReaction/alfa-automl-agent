"""A-only tree selection and paid aligned prefix predictions for round 6.

The shared legacy primitives are imported unchanged. The selection helper sees
only the strictly earlier prefix and calibration A; B is absent even from its
argument frames. Its fitted model is released, so one explicit fixed-count
prefix recheck supplies aligned A/B predictions. No model survives that stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from agent import regression_base as legacy


_ACTIVE = ContextVar("round6_regression_active_component", default=None)


def active_component():
    """Measurement metadata only; it cannot affect model features or routing."""
    return deepcopy(_ACTIVE.get())


@contextmanager
def observed(phase, deadline, original_deadline, *, path="correction", engine="catboost"):
    token = _ACTIVE.set({"id": "catboost_all" if engine == "catboost" else "residual_correction",
        "engine": engine, "feature_subset": "all_declared" if engine == "catboost" else "base_predictions_only",
        "phase": phase, "path": path,
        "deadline_monotonic": deadline, "original_deadline_monotonic": original_deadline})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@dataclass
class PreparedInputs:
    X: object
    y: object
    predict: object
    features: list
    numeric: list
    categorical: list
    metric: object
    order: np.ndarray
    fitting: np.ndarray
    held: np.ndarray
    base_recipe: dict
    evidence: dict


@dataclass
class AlignedCalibration:
    selected_iterations: int
    predictions: np.ndarray
    evidence: dict


def prepare_inputs(X_fit, y_fit, X_predict, context, recipe):
    """Validate/copy raw inputs and identify the unchanged temporal boundary."""
    started = time.monotonic()
    recipe = legacy.normalize_base_recipe(recipe)
    if recipe["engine"] != "catboost":
        raise ValueError("Round6 has only the unchanged CatBoost base recipe")
    base = {"base_recipe": {"model_config": legacy.make_model_config(
        "catboost", "regression", recipe["parameters"])}}
    X, y, predict, features, numeric, categorical, _, metric = legacy._inputs(
        X_fit, y_fit, X_predict, context, base)
    if metric.name != "RMSE" or context.get("task_type") != "regression":
        raise ValueError("Round6 requires original-scale regression RMSE")
    period = recipe["period_column"]
    if period not in features or period not in numeric:
        raise ValueError("Round6 period must be a declared numeric feature")
    if (sum(X[name].nunique(dropna=False) for name in categorical) > legacy.MAX_NATIVE_CATEGORIES
            or len(X) * len(categorical) > legacy.MAX_NATIVE_CATEGORY_CELLS):
        raise legacy.BoostingSkipError("native_category_budget_exceeded")
    order = legacy._canonical_order(X, y)
    X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    periods = [legacy.numeric_value(value) for value in X[period]]
    unique = sorted({value for value in periods if value is not None})
    fitting = np.asarray([i for i, value in enumerate(periods)
        if value is not None and unique and value < unique[-1]], dtype=int)
    held = np.asarray([i for i, value in enumerate(periods)
        if value is not None and unique and value == unique[-1]], dtype=int)
    available = len(unique) >= 2 and len(fitting) >= 2 and len(held) > 0
    reason = "available" if available else (
        "single_period" if len(unique) == 1 else "no_valid_period" if not unique else "temporal_support_too_small")
    evidence = {"calibration_available": bool(available), "reason": reason,
        "calibration_period": legacy.period_json(unique[-1]) if available else None,
        "prefix_rows": len(fitting), "calibration_rows": len(held),
        "full_fit_rows": len(X), "predict_rows": len(predict),
        "invalid_period_rows": sum(value is None for value in periods),
        "preparation_elapsed_seconds": time.monotonic() - started,
        "row_order": "unchanged_declared_feature_and_target_content",
        "threads": 8, "device": "CPU", "target_scale": "original"}
    return PreparedInputs(X, y, predict, features, numeric, categorical, metric,
        order, fitting, held, recipe, evidence)


def _fixed_pipeline(prepared, selected_iterations):
    if type(selected_iterations) is not int or not 50 <= selected_iterations <= 2000:
        raise ValueError("Selected refit count is outside the frozen range")
    parameters = deepcopy(prepared.base_recipe["parameters"])
    parameters["iterations"] = selected_iterations
    config = legacy.make_model_config("catboost", "regression", parameters)
    return legacy.build_pipeline("catboost", "regression", prepared.numeric,
        prepared.categorical, config, legacy.normalize_feature_config())


def _fit_predict_fixed(prepared, selected_iterations, fitting, predict, *, phase,
                       deadline, original_deadline):
    legacy._check_deadline(deadline)
    with observed(phase, deadline, original_deadline):
        model = _fixed_pipeline(prepared, selected_iterations)
        callback = legacy._CatBoostDeadline(deadline)
        fit_started = time.monotonic()
        model.fit(prepared.X.iloc[fitting], prepared.y.iloc[fitting], model__callbacks=[callback])
        legacy._check_deadline(deadline)
        if callback.reached:
            raise TimeoutError("Round6 fixed-count fit reached its original phase deadline")
        fit_seconds = time.monotonic() - fit_started
        prediction_started = time.monotonic()
        values = legacy._predictions(model.predict(predict), len(predict))
        legacy._check_deadline(deadline)
        prediction_seconds = time.monotonic() - prediction_started
    return values, {"phase": phase, "engine": "catboost", "fit_rows": len(fitting),
        "predict_rows": len(predict), "selected_iterations": selected_iterations,
        "fit_elapsed_seconds": fit_seconds, "prediction_elapsed_seconds": prediction_seconds,
        "phase_deadline_monotonic": deadline, "original_deadline_monotonic": original_deadline,
        "fresh_model": True}


def stage_calibration(prepared, a_positions, *, deadline, original_deadline):
    """Select on A only; pay one fresh, no-ES recheck at exact refit count."""
    started = time.monotonic()
    legacy._check_deadline(deadline)
    a = np.asarray(a_positions)
    if (a.ndim != 1 or a.dtype.kind not in "iu" or len(a) < 100
            or len(np.unique(a)) != len(a) or (a < 0).any() or (a >= len(prepared.held)).any()
            or not prepared.evidence["calibration_available"]):
        raise ValueError("A-only tree selection requires the frozen supported calibration split")
    # Do not merely pass the complete y and promise not to index B: construct
    # the selector's entire argument data without B or other last-period rows.
    selector_positions = np.concatenate((prepared.fitting, prepared.held[a]))
    selector_X = prepared.X.iloc[selector_positions].reset_index(drop=True)
    selector_y = prepared.y.iloc[selector_positions].reset_index(drop=True)
    fitting = np.arange(len(prepared.fitting), dtype=int)
    validation = np.arange(len(fitting), len(selector_positions), dtype=int)
    with observed("correction_selection_A", deadline, original_deadline):
        selected, _, internal = legacy._internal_fit(selector_X, selector_y,
            fitting, validation, prepared.numeric, prepared.categorical,
            prepared.metric, prepared.base_recipe, deadline)
    legacy._check_deadline(deadline)
    # _internal_fit deliberately releases its model. This recheck is therefore
    # always paid, including best>=50, and has no early stopping/eval labels.
    predictions, recheck = _fit_predict_fixed(prepared, int(selected), prepared.fitting,
        prepared.X.iloc[prepared.held], phase="prefix_recheck", deadline=deadline,
        original_deadline=original_deadline)
    evidence = {**internal, "selection_scope": "strict_prefix_eval_A_only",
        "selection_fit_rows": len(fitting), "selection_validation_rows": len(validation),
        "B_labels_available_to_selector": False, "selected_iterations": int(selected),
        "calibration_iterations": int(selected), "calibration_alignment": "paid_fixed_count_prefix_recheck",
        "prefix_recheck_fits": 1, "prefix_recheck": recheck,
        "elapsed_seconds": time.monotonic() - started,
        "physical_fits_expected_completed": 2,
        "phase_deadline_monotonic": deadline, "original_deadline_monotonic": original_deadline}
    return AlignedCalibration(int(selected), predictions, evidence)


def refit_full(prepared, selected_iterations, *, deadline, original_deadline):
    """Fresh all-supplied-fit model after correction/history policy freeze."""
    started = time.monotonic()
    values, evidence = _fit_predict_fixed(prepared, selected_iterations,
        np.arange(len(prepared.X), dtype=int), prepared.predict, phase="full_refit",
        deadline=deadline, original_deadline=original_deadline)
    evidence.update(elapsed_seconds=time.monotonic() - started, fresh_full_refit=True,
        physical_fits_expected_completed=1)
    return values, evidence
