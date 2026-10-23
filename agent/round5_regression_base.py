"""Staged use of the unchanged rescue base primitives for fixed ensembles.

Internal selection returns only copied raw frames, tree counts and calibration
predictions. No fitted model is retained: every final component is built anew
after the ensemble's fit-only history calibration. The legacy numerical/tree
policy remains explicit; this module adds scheduling, not another parameter grid.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from agent import regression_base as legacy


_ACTIVE = ContextVar("round5_regression_active_component", default=None)


def active_component():
    """Optional child-process measurement metadata; never a model input."""
    return deepcopy(_ACTIVE.get())


@contextmanager
def _observed(component, phase, deadline):
    token = _ACTIVE.set({"id": component["id"], "engine": component["base"]["engine"],
                         "feature_subset": component["feature_subset"], "phase": phase,
                         "deadline_monotonic": deadline})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@dataclass
class StagedComponent:
    component: dict
    X: object
    y: object
    predict: object
    numeric: list
    categorical: list
    selected_iterations: int
    calibration: dict | None
    quota_seconds: float
    consumed_seconds: float
    evidence: dict


def _phase_deadline(original_deadline, base_deadline, remaining):
    legacy._check_deadline(original_deadline)
    if remaining <= 0:
        raise TimeoutError("Round5 component's fixed time quota is exhausted")
    expires = min(original_deadline, base_deadline, time.monotonic() + remaining)
    legacy._check_deadline(expires)
    return expires


def stage_component(X_fit, y_fit, X_predict, context, component, *, quota_seconds,
                    base_deadline, original_deadline):
    """Fit only the earlier calibration model, without performing full refit.

The unchanged legacy _internal_fit owns preprocessing on earlier positions,
early stopping, selected counts and calibration predictions. Its temporary
model is released on return. The quota includes copying, schema preparation
and that complete internal operation, even if preparation raises.
"""
    started = time.monotonic()
    deadline = _phase_deadline(original_deadline, base_deadline, quota_seconds)
    recipe = legacy.normalize_base_recipe(component["base"])
    base = {"base_recipe": {"model_config": legacy.make_model_config(
        recipe["engine"], "regression", recipe["parameters"])}}
    with _observed(component, "internal_selection", deadline):
        X, y, predict, features, numeric, categorical, _, metric = legacy._inputs(
            X_fit, y_fit, X_predict, context, base)
        if metric.name != "RMSE" or context.get("task_type") != "regression":
            raise ValueError("Round5 requires original-scale regression RMSE")
        period = recipe["period_column"]
        if period not in features or period not in numeric:
            raise ValueError("Round5 period must be a declared numeric feature")
        if recipe["engine"] == "catboost" and (
                sum(X[name].nunique(dropna=False) for name in categorical) > legacy.MAX_NATIVE_CATEGORIES
                or len(X) * len(categorical) > legacy.MAX_NATIVE_CATEGORY_CELLS):
            raise legacy.BoostingSkipError("native_category_budget_exceeded")
        order = legacy._canonical_order(X, y)
        X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
        legacy._check_deadline(deadline)
        periods = [legacy.numeric_value(value) for value in X[period]]
        unique = sorted({value for value in periods if value is not None})
        fitting = np.asarray([i for i, value in enumerate(periods)
                              if value is not None and unique and value < unique[-1]], dtype=int)
        held = np.asarray([i for i, value in enumerate(periods)
                           if value is not None and unique and value == unique[-1]], dtype=int)
        preparation_seconds = time.monotonic() - started
        evidence = {"id": component["id"], "engine": recipe["engine"],
            "feature_subset": component["feature_subset"], "weight": component["weight"],
            "features": list(features), "numeric": list(numeric), "categorical": list(categorical),
            "recipe": deepcopy(recipe), "quota_seconds": quota_seconds,
            "original_deadline_monotonic": original_deadline, "base_deadline_monotonic": base_deadline,
            "internal_phase_deadline_monotonic": deadline,
            "full_fit_rows": len(X), "predict_rows": len(predict),
            "internal_fit_rows": 0, "internal_validation_rows": 0,
            "invalid_period_rows": sum(value is None for value in periods),
            "preparation_elapsed_seconds": preparation_seconds,
            "internal_elapsed_seconds": 0.0, "best_iteration": None,
            "internal_iterations": 0, "calibration_period": None,
            "calibration_alignment": "legacy_best_vs_min50_refit",
            "calibration_iterations": None, "calibration_recheck_fits": 0,
            "calibration_recheck_fit_seconds": 0.0, "calibration_recheck_predict_seconds": 0.0,
            "threads": 8, "device": "CPU", "fresh_full_refit": False,
            "target_scale": "original", "row_order": "declared_feature_and_target_content",
            "physical_fits_expected_completed": 0}
        calibration = None
        selected = recipe["fallback_iterations"]
        if len(unique) >= 2 and len(fitting) >= 2 and len(held):
            selected, values, measured = legacy._internal_fit(
                X, y, fitting, held, numeric, categorical, metric, recipe, deadline)
            evidence.update(measured, reason="fit_only_early_stopping", internal_fit_rows=len(fitting),
                            internal_validation_rows=len(held), calibration_period=legacy.period_json(unique[-1]),
                            calibration_iterations=measured["best_iteration"], physical_fits_expected_completed=1)
            calibration = {"fit_positions": order[fitting].copy(),
                           "validation_positions": order[held].copy(), "predictions": values.copy()}
        else:
            evidence["reason"] = ("single_period_fixed_iterations" if len(unique) == 1 else
                "no_valid_period_fixed_iterations" if not unique else "internal_split_too_small_fixed_iterations")
        legacy._check_deadline(deadline)
    consumed = time.monotonic() - started
    evidence.update(selected_iterations=selected, staging_elapsed_seconds=consumed)
    return StagedComponent(deepcopy(component), X, y, predict, numeric, categorical, selected,
                           calibration, float(quota_seconds), consumed, evidence)


def refit_component(staged, *, base_deadline, original_deadline):
    """Fresh full-fit pipeline under its remaining fixed component quota."""
    started = time.monotonic()
    deadline = _phase_deadline(original_deadline, base_deadline,
                               staged.quota_seconds - staged.consumed_seconds)
    recipe = staged.component["base"]
    engine = recipe["engine"]
    with _observed(staged.component, "full_refit", deadline):
        parameters = deepcopy(recipe["parameters"])
        parameters["iterations" if engine == "catboost" else "n_estimators"] = staged.selected_iterations
        config = legacy.make_model_config(engine, "regression", parameters)
        model = legacy.build_pipeline(engine, "regression", staged.numeric, staged.categorical,
                                      config, legacy.normalize_feature_config())
        callback = legacy._CatBoostDeadline(deadline) if engine == "catboost" else legacy._lightgbm_refit_deadline(deadline)
        legacy._check_deadline(deadline)
        fit_started = time.monotonic()
        model.fit(staged.X, staged.y, model__callbacks=[callback])
        legacy._check_deadline(deadline)
        if engine == "catboost" and callback.reached:
            raise TimeoutError("Round5 full refit stopped at its component deadline")
        refit_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predictions = legacy._predictions(model.predict(staged.predict), len(staged.predict))
        legacy._check_deadline(deadline)
        predict_seconds = time.monotonic() - predict_started
    elapsed = time.monotonic() - started
    staged.consumed_seconds += elapsed
    staged.evidence.update(status="completed", fresh_full_refit=True,
        refit_phase_deadline_monotonic=deadline, refit_elapsed_seconds=refit_seconds,
        predict_elapsed_seconds=predict_seconds, refit_stage_elapsed_seconds=elapsed,
        total_elapsed_seconds=staged.consumed_seconds,
        physical_fits_expected_completed=staged.evidence["physical_fits_expected_completed"] + 1)
    return predictions, deepcopy(staged.evidence)
