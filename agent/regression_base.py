"""Isolated, fit-only regression boosting with reusable temporal calibration.

This module never discovers a schema or reads a task, file, ID, or held-out
answer. One last-period fit-internal model chooses the boosting length; a fresh
model then fits all supplied fit rows. The calibration labels stay with callers.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from agent.boosting_candidate import (
    BoostingSkipError, _CatBoostDeadline, _check_deadline, _deadline, _inputs,
    _lightgbm_callbacks, _lightgbm_metric, _make_estimator,
)
from agent.building_history import numeric_value, period_json
from agent.features import FeatureTransformer, normalize_feature_config
from agent.model_config import make_model_config
from agent.models import MAX_NATIVE_CATEGORIES, MAX_NATIVE_CATEGORY_CELLS, _LightGBMFrame, build_pipeline


FAMILY = "rescue_regression_base_v1"


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def normalize_base_recipe(recipe):
    """Return a detached, bounded recipe without runtime or file controls."""
    allowed = {"engine", "period_column", "max_iterations", "stopping_rounds", "fallback_iterations", "parameters"}
    if not isinstance(recipe, dict) or set(recipe) - allowed:
        raise ValueError("Invalid regression base recipe fields")
    engine, period = recipe.get("engine"), recipe.get("period_column")
    if not isinstance(engine, str) or engine not in {"catboost", "lightgbm"}:
        raise ValueError("Regression base engine must be catboost or lightgbm")
    if not isinstance(period, str) or not period or "\x00" in period:
        raise ValueError("An explicit period_column is required")
    maximum = _integer(recipe.get("max_iterations", 2000), "max_iterations", 50, 2000)
    rounds = _integer(recipe.get("stopping_rounds", 100), "stopping_rounds", 1, min(200, maximum))
    fallback = _integer(recipe.get("fallback_iterations", 400), "fallback_iterations", 50, maximum)
    parameters = recipe.get("parameters", {})
    resource = "iterations" if engine == "catboost" else "n_estimators"
    if not isinstance(parameters, dict) or resource in parameters:
        raise ValueError("Tree count belongs to the recipe, not parameters")
    normalized = make_model_config(engine, "regression", parameters)["parameters"]
    normalized.pop(resource)
    return {"engine": engine, "period_column": period, "max_iterations": maximum,
            "stopping_rounds": rounds, "fallback_iterations": fallback, "parameters": normalized}


def _token(value):
    """Content-only ordering, including missing values, with no index or ID."""
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError("Regression feature values must be scalars")
    if missing:
        return ("missing", "")
    if isinstance(value, str):
        return ("text", value)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", str(bool(value)))
    if isinstance(value, Integral):
        return ("integer", str(int(value)))
    if isinstance(value, Real):
        return ("real", float(value).hex())
    raise ValueError("Regression features must be raw text, real numbers, or missing values")


def _canonical_order(X, y):
    keys = [tuple(_token(value) for value in row) + (("target", float(answer).hex()),)
            for row, answer in zip(X.itertuples(index=False, name=None), y)]
    return np.asarray(sorted(range(len(X)), key=keys.__getitem__), dtype=int)


def _predictions(values, count):
    values = np.asarray(values, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Regression base returned invalid predictions")
    return values


def _internal_fit(X, y, fitting, held, numeric, categorical, metric, recipe, deadline):
    """Fit once and release model/transformer before the separate full refit."""
    engine = recipe["engine"]
    resource = "iterations" if engine == "catboost" else "n_estimators"
    started = time.monotonic()
    row_scale = (len(X) / len(fitting)) ** 1.25 * 1.3
    fraction = min(.30, .75 / (1 + row_scale))
    internal_deadline = started + (deadline - started) * fraction
    transformer = FeatureTransformer(numeric, categorical, normalize_feature_config())
    train_features = transformer.fit_transform(X.iloc[fitting], y.iloc[fitting])
    held_features = transformer.transform(X.iloc[held])
    parameters = deepcopy(recipe["parameters"])
    parameters[resource] = recipe["max_iterations"]
    if engine == "lightgbm":
        frame = _LightGBMFrame(categorical)
        train_features = frame.fit_transform(train_features)
        held_features = frame.transform(held_features)
        early_metric = _lightgbm_metric(metric, [], None)
        model = _make_estimator(engine, "regression", parameters, categorical, early_metric)
        callbacks = _lightgbm_callbacks(recipe["stopping_rounds"], internal_deadline)
        _check_deadline(internal_deadline)
        model.fit(train_features, y.iloc[fitting], eval_set=[(held_features, y.iloc[held])],
                  eval_metric=early_metric, callbacks=callbacks)
        best = getattr(model, "best_iteration_", None)
        seen = getattr(callbacks[-1], "seen_iterations", 0) if callbacks else 0
    else:
        model = _make_estimator(engine, "regression", parameters, categorical, "RMSE")
        callback = _CatBoostDeadline(internal_deadline)
        _check_deadline(internal_deadline)
        model.fit(train_features, y.iloc[fitting], eval_set=(held_features, y.iloc[held]),
                  early_stopping_rounds=recipe["stopping_rounds"], use_best_model=True, callbacks=[callback])
        index = model.get_best_iteration()
        best = int(index) + 1 if isinstance(index, Integral) and not isinstance(index, bool) and index >= 0 else None
        seen = callback.seen_iterations
    if not isinstance(best, Integral) or isinstance(best, bool) or not 1 <= best <= recipe["max_iterations"]:
        raise BoostingSkipError("best_iteration_unavailable")
    # Explicit best-tree prediction also protects LightGBM version differences.
    values = (model.predict(held_features, num_iteration=int(best)) if engine == "lightgbm"
              else model.predict(held_features, ntree_end=int(best)))
    calibration = _predictions(values, len(held))
    elapsed = max(0.0, time.monotonic() - started)
    best, selected = int(best), max(50, int(best))
    trained = min(recipe["max_iterations"], max(best, seen))
    estimate = elapsed * row_scale * selected / trained + .2
    _check_deadline(deadline)
    if deadline - time.monotonic() <= estimate:
        raise TimeoutError("Insufficient measured time for fresh full regression refit")
    return selected, calibration, {"best_iteration": best, "internal_iterations": trained,
        "internal_elapsed_seconds": elapsed, "internal_time_fraction": fraction,
        "estimated_refit_seconds": estimate, "refit_estimate_guaranteed": False}


def _lightgbm_refit_deadline(deadline):
    def check(_environment):
        _check_deadline(deadline)
    check.order = 0
    check.before_iteration = True
    return check


def fit_base(X_fit, y_fit, X_predict, context, recipe):
    """Return (predictions, evidence, calibration_or_None), without file I/O.

    Calibration keys are ``fit_positions``, ``validation_positions`` and
    ``predictions``. Positions index the ORIGINAL supplied X_fit/y_fit by iloc.
    Held labels never leave the caller through this return value. Equal-content
    fitting rows may exchange positions, but models see identical sorted data.
    """
    invocation_started = time.monotonic()
    recipe = normalize_base_recipe(recipe)
    if not isinstance(context, dict):
        raise ValueError("Regression base requires a host context")
    deadline = _deadline(context)
    engine = recipe["engine"]
    base = {"base_recipe": {"model_config": make_model_config(engine, "regression", recipe["parameters"])}}
    X, y, predict, features, numeric, categorical, _, metric = _inputs(X_fit, y_fit, X_predict, context, base)
    if metric.name != "RMSE" or context.get("task_type") != "regression":
        raise ValueError("Regression base requires original-scale RMSE")
    period = recipe["period_column"]
    if period not in features or period not in numeric:
        raise ValueError("period_column must be a declared numeric feature")
    # The native category decision uses fit only, never prediction statistics.
    if engine == "catboost" and (sum(X[name].nunique(dropna=False) for name in categorical) > MAX_NATIVE_CATEGORIES
            or len(X) * len(categorical) > MAX_NATIVE_CATEGORY_CELLS):
        raise BoostingSkipError("native_category_budget_exceeded")
    order = _canonical_order(X, y)
    X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    _check_deadline(deadline)
    periods = [numeric_value(value) for value in X[period]]
    unique = sorted({value for value in periods if value is not None})
    fitting = np.asarray([i for i, value in enumerate(periods) if value is not None and unique and value < unique[-1]], dtype=int)
    held = np.asarray([i for i, value in enumerate(periods) if value is not None and unique and value == unique[-1]], dtype=int)
    evidence = {"version": 1, "family": FAMILY, "engine": engine, "recipe": deepcopy(recipe),
        "selection_scope": "fit_only_last_period", "early_stopping_metric": "RMSE", "target_scale": "original",
        "full_fit_rows": len(X), "predict_rows": len(predict), "invalid_period_rows": sum(value is None for value in periods),
        "internal_fit_rows": 0, "internal_validation_rows": 0, "calibration_period": None,
        "internal_elapsed_seconds": 0.0, "estimated_refit_seconds": None, "refit_estimate_guaranteed": False,
        "preparation_elapsed_seconds": max(0.0, time.monotonic() - invocation_started),
        "best_iteration": None, "internal_iterations": 0, "internal_time_fraction": 0.0,
        "fresh_full_refit": True, "threads": 8, "device": "CPU", "row_order": "declared_feature_and_target_content",
        "api_requests": 0, "peak_rss_bytes": None, "rss_status": "not_observed"}
    calibration = None
    selected = recipe["fallback_iterations"]
    if len(unique) >= 2 and len(fitting) >= 2 and len(held):
        selected, values, measured = _internal_fit(X, y, fitting, held, numeric, categorical, metric, recipe, deadline)
        evidence.update(measured, reason="fit_only_early_stopping", internal_fit_rows=len(fitting),
                        internal_validation_rows=len(held), calibration_period=period_json(unique[-1]))
        calibration = {"fit_positions": order[fitting].copy(), "validation_positions": order[held].copy(),
                       "predictions": values.copy()}
    else:
        evidence["reason"] = "single_period_fixed_iterations" if len(unique) == 1 else (
            "no_valid_period_fixed_iterations" if not unique else "internal_split_too_small_fixed_iterations")
    parameters = deepcopy(recipe["parameters"])
    parameters["iterations" if engine == "catboost" else "n_estimators"] = selected
    config = make_model_config(engine, "regression", parameters)
    model = build_pipeline(engine, "regression", numeric, categorical, config, normalize_feature_config())
    callback = _CatBoostDeadline(deadline) if engine == "catboost" else _lightgbm_refit_deadline(deadline)
    _check_deadline(deadline)
    started = time.monotonic()
    model.fit(X, y, model__callbacks=[callback])
    _check_deadline(deadline)
    if engine == "catboost" and callback.reached:
        raise TimeoutError("Regression refit stopped at its deadline")
    refit_elapsed = max(0.0, time.monotonic() - started)
    started = time.monotonic()
    predictions = _predictions(model.predict(predict), len(predict))
    _check_deadline(deadline)
    evidence.update(status="completed", selected_iterations=selected, refit_elapsed_seconds=refit_elapsed,
                    predict_elapsed_seconds=max(0.0, time.monotonic() - started),
                    total_elapsed_seconds=max(0.0, time.monotonic() - invocation_started))
    return predictions, evidence, calibration
