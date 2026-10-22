"""Versioned regression experiments with fold-local features and calibration.

The unchanged rescue-v1 base remains the exact path when both experimental
controls are disabled. A calibration-count recheck is an additional physical
fit on the earlier internal rows, never a fit on their held labels.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
import time

import numpy as np

from agent import regression_base as legacy
from agent.round4_regression_features import RegressionFeatureAdapter, normalize_feature_recipe


FAMILY = "round4_regression_base_v1"


def _fixed_fit(X, y, predict, numeric, categorical, recipe, iterations, deadline):
    """One fresh fixed-length pipeline, preserving the caller's deadline."""
    engine = recipe["engine"]
    parameters = deepcopy(recipe["parameters"])
    parameters["iterations" if engine == "catboost" else "n_estimators"] = iterations
    config = legacy.make_model_config(engine, "regression", parameters)
    model = legacy.build_pipeline(engine, "regression", numeric, categorical, config,
                                  legacy.normalize_feature_config())
    callback = (legacy._CatBoostDeadline(deadline) if engine == "catboost"
                else legacy._lightgbm_refit_deadline(deadline))
    legacy._check_deadline(deadline)
    started = time.monotonic()
    model.fit(X, y, model__callbacks=[callback])
    legacy._check_deadline(deadline)
    if engine == "catboost" and callback.reached:
        raise TimeoutError("Round4 regression fixed fit stopped at its deadline")
    fitting_elapsed = time.monotonic() - started
    started = time.monotonic()
    values = legacy._predictions(model.predict(predict), len(predict))
    legacy._check_deadline(deadline)
    return values, fitting_elapsed, time.monotonic() - started


def _internal_fit(X, y, fitting, held, numeric, categorical, metric, recipe,
                  feature_config, calibration_alignment, deadline):
    started = time.monotonic()
    engine = recipe["engine"]
    row_scale = (len(X) / len(fitting)) ** 1.25 * 1.3
    fraction = min(.30, .75 / (1 + row_scale))
    internal_deadline = started + (deadline - started) * fraction
    adapter = RegressionFeatureAdapter(numeric, categorical, feature_config)
    train_raw = adapter.fit_transform(X.iloc[fitting], y.iloc[fitting])
    held_raw = adapter.transform(X.iloc[held])
    extended_numeric, extended_categorical = adapter.output_schema()
    transformer = legacy.FeatureTransformer(extended_numeric, extended_categorical,
                                            legacy.normalize_feature_config())
    train_features = transformer.fit_transform(train_raw, y.iloc[fitting])
    held_features = transformer.transform(held_raw)
    parameters = deepcopy(recipe["parameters"])
    parameters["iterations" if engine == "catboost" else "n_estimators"] = recipe["max_iterations"]
    if engine == "lightgbm":
        frame = legacy._LightGBMFrame(extended_categorical)
        train_features = frame.fit_transform(train_features)
        held_features = frame.transform(held_features)
        early_metric = legacy._lightgbm_metric(metric, [], None)
        model = legacy._make_estimator(engine, "regression", parameters, extended_categorical, early_metric)
        callbacks = legacy._lightgbm_callbacks(recipe["stopping_rounds"], internal_deadline)
        legacy._check_deadline(internal_deadline)
        model.fit(train_features, y.iloc[fitting], eval_set=[(held_features, y.iloc[held])],
                  eval_metric=early_metric, callbacks=callbacks)
        best = getattr(model, "best_iteration_", None)
        seen = getattr(callbacks[-1], "seen_iterations", 0) if callbacks else 0
    else:
        model = legacy._make_estimator(engine, "regression", parameters, extended_categorical, "RMSE")
        callback = legacy._CatBoostDeadline(internal_deadline)
        legacy._check_deadline(internal_deadline)
        model.fit(train_features, y.iloc[fitting], eval_set=(held_features, y.iloc[held]),
                  early_stopping_rounds=recipe["stopping_rounds"], use_best_model=True,
                  callbacks=[callback])
        index = model.get_best_iteration()
        best = int(index) + 1 if isinstance(index, Integral) and not isinstance(index, bool) and index >= 0 else None
        seen = callback.seen_iterations
    if not isinstance(best, Integral) or isinstance(best, bool) or not 1 <= best <= recipe["max_iterations"]:
        raise legacy.BoostingSkipError("best_iteration_unavailable")
    values = (model.predict(held_features, num_iteration=int(best)) if engine == "lightgbm"
              else model.predict(held_features, ntree_end=int(best)))
    calibration = legacy._predictions(values, len(held))
    elapsed = time.monotonic() - started
    best, selected = int(best), max(50, int(best))
    trained = min(recipe["max_iterations"], max(best, seen))
    estimate = elapsed * row_scale * selected / trained + .2
    del model, transformer, train_features, held_features
    legacy._check_deadline(deadline)
    recheck_fit_seconds = recheck_predict_seconds = 0.0
    recheck_fits = 0
    calibration_iterations = best
    if calibration_alignment == "refit_count" and best < selected:
        # Allocate only remaining original time, retaining the measured (not
        # guaranteed) full-fit estimate. A failed recheck aborts this candidate.
        recheck_estimate = elapsed * selected / trained + .2
        if deadline - time.monotonic() <= estimate + recheck_estimate:
            raise TimeoutError("Insufficient measured time for calibration-count recheck and full refit")
        calibration, recheck_fit_seconds, recheck_predict_seconds = _fixed_fit(
            train_raw, y.iloc[fitting], held_raw, extended_numeric, extended_categorical,
            recipe, selected, deadline - estimate)
        calibration_iterations, recheck_fits = selected, 1
    if deadline - time.monotonic() <= estimate:
        raise TimeoutError("Insufficient measured time for fresh full regression refit")
    return selected, calibration, {
        "best_iteration": best, "internal_iterations": trained,
        "internal_elapsed_seconds": elapsed, "internal_time_fraction": fraction,
        "estimated_refit_seconds": estimate, "refit_estimate_guaranteed": False,
        "internal_feature_evidence": deepcopy(adapter.evidence_),
        "calibration_iterations": calibration_iterations,
        "calibration_recheck_fits": recheck_fits,
        "calibration_recheck_fit_seconds": recheck_fit_seconds,
        "calibration_recheck_predict_seconds": recheck_predict_seconds,
        "physical_fits": 1 + recheck_fits,
    }


def fit_base(X_fit, y_fit, X_predict, context, recipe, *, feature_config=None,
             calibration_alignment="legacy"):
    """Return predictions, evidence and original-position internal calibration.

Feature adapters are independently fitted on internal-earlier and full-fit
rows. ``refit_count`` reruns only internal-earlier model fitting when the
selected full-refit count differs from the early-stopped prediction count.
"""
    if not isinstance(calibration_alignment, str) or calibration_alignment not in {"legacy", "refit_count"}:
        raise ValueError("calibration_alignment must be legacy or refit_count")
    feature_config = normalize_feature_recipe(feature_config)
    enabled = feature_config["missing_indicators"] or feature_config["category_frequencies"]
    if not enabled and calibration_alignment == "legacy":
        values, evidence, calibration = legacy.fit_base(X_fit, y_fit, X_predict, context, recipe)
        evidence = deepcopy(evidence)
        evidence.update(physical_fits=2 if calibration is not None else 1,
                        calibration_alignment="legacy", calibration_recheck_fits=0,
                        calibration_iterations=evidence.get("best_iteration"),
                        feature_config=deepcopy(feature_config),
                        round4_execution="unchanged_rescue_v1_delegate")
        return values, evidence, calibration
    started = time.monotonic()
    recipe = legacy.normalize_base_recipe(recipe)
    if not isinstance(context, dict):
        raise ValueError("Regression base requires a host context")
    deadline = legacy._deadline(context)
    engine = recipe["engine"]
    base = {"base_recipe": {"model_config": legacy.make_model_config(engine, "regression", recipe["parameters"])}}
    X, y, predict, features, numeric, categorical, _, metric = legacy._inputs(
        X_fit, y_fit, X_predict, context, base)
    if metric.name != "RMSE" or context.get("task_type") != "regression":
        raise ValueError("Regression base requires original-scale RMSE")
    period = recipe["period_column"]
    if period not in features or period not in numeric:
        raise ValueError("period_column must be a declared numeric feature")
    if engine == "catboost" and (sum(X[name].nunique(dropna=False) for name in categorical) > legacy.MAX_NATIVE_CATEGORIES
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
    evidence = {"version": 1, "family": FAMILY, "engine": engine, "recipe": deepcopy(recipe),
        "feature_config": deepcopy(feature_config), "calibration_alignment": calibration_alignment,
        "selection_scope": "fit_only_last_period", "early_stopping_metric": "RMSE", "target_scale": "original",
        "full_fit_rows": len(X), "predict_rows": len(predict),
        "invalid_period_rows": sum(value is None for value in periods),
        "internal_fit_rows": 0, "internal_validation_rows": 0, "calibration_period": None,
        "internal_elapsed_seconds": 0.0, "estimated_refit_seconds": None, "refit_estimate_guaranteed": False,
        "preparation_elapsed_seconds": time.monotonic() - started,
        "best_iteration": None, "internal_iterations": 0, "internal_time_fraction": 0.0,
        "calibration_iterations": None, "calibration_recheck_fits": 0,
        "calibration_recheck_fit_seconds": 0.0, "calibration_recheck_predict_seconds": 0.0,
        "physical_fits": 0, "fresh_full_refit": True, "threads": 8, "device": "CPU",
        "row_order": "declared_feature_and_target_content", "api_requests": 0,
        "peak_rss_bytes": None, "rss_status": "not_observed"}
    calibration = None
    selected = recipe["fallback_iterations"]
    if len(unique) >= 2 and len(fitting) >= 2 and len(held):
        selected, values, measured = _internal_fit(X, y, fitting, held, numeric, categorical, metric,
            recipe, feature_config, calibration_alignment, deadline)
        evidence.update(measured, reason="fit_only_early_stopping", internal_fit_rows=len(fitting),
                        internal_validation_rows=len(held), calibration_period=legacy.period_json(unique[-1]))
        calibration = {"fit_positions": order[fitting].copy(), "validation_positions": order[held].copy(),
                       "predictions": values.copy()}
    else:
        evidence["reason"] = "single_period_fixed_iterations" if len(unique) == 1 else (
            "no_valid_period_fixed_iterations" if not unique else "internal_split_too_small_fixed_iterations")
    prepared = time.monotonic()
    adapter = RegressionFeatureAdapter(numeric, categorical, feature_config)
    full_features = adapter.fit_transform(X, y)
    predict_features = adapter.transform(predict)
    extended_numeric, extended_categorical = adapter.output_schema()
    evidence["full_preparation_elapsed_seconds"] = time.monotonic() - prepared
    predictions, refit_elapsed, predict_elapsed = _fixed_fit(full_features, y, predict_features,
        extended_numeric, extended_categorical, recipe, selected, deadline)
    evidence.update(status="completed", selected_iterations=selected,
        physical_fits=evidence["physical_fits"] + 1, full_feature_evidence=deepcopy(adapter.evidence_),
        refit_elapsed_seconds=refit_elapsed, predict_elapsed_seconds=predict_elapsed,
        total_elapsed_seconds=time.monotonic() - started)
    return predictions, evidence, calibration
