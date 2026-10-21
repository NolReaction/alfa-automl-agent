"""Fit-only early stopping followed by a fresh full-fit boosting model.

The internal holdout is made only from this invocation's X_fit/y_fit using the
host's explicit split policy. X_predict and its outcomes never select the tree
count. Host INNER/outer scoring and the process deadline remain authoritative.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
import time

import numpy as np
import pandas as pd

from agent.features import FeatureTransformer, feature_schema, normalize_feature_config
from agent.metrics import get_metric, label_values, probability_matrix, score_predictions
from agent.model_config import make_model_config, model_task_type, validate_model_config
from agent.models import (
    MAX_NATIVE_CATEGORIES, MAX_NATIVE_CATEGORY_CELLS, THREADS, _LightGBMFrame,
    build_pipeline,
)
from agent.splits import normalize_validation, split_rows
from offline_baseline import SEED


FAMILY = "early_stopping_boosting_v1"
EVIDENCE_NAME = "boosting_evidence.json"
INTERNAL_TIME_FRACTION = .30


class BoostingSkipError(ValueError):
    """An expected eligibility stop which leaves the host baseline intact."""

    def __init__(self, reason):
        allowed = {"internal_split_unavailable", "internal_split_missing_class",
                   "native_category_budget_exceeded", "best_iteration_unavailable"}
        if reason not in allowed:
            raise ValueError("Unknown bounded boosting skip reason")
        self.reason = reason
        super().__init__(reason)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " is outside its supported integer bounds")
    return value


def make_boosting_recipe(base_recipe, validation, *, max_iterations=2000, stopping_rounds=50):
    """Freeze one ordinary boosting family and an explicit internal policy."""
    allowed = {"id", "label", "action", "model_config", "feature_config"}
    if (not isinstance(base_recipe, dict) or set(base_recipe) - allowed
            or not isinstance(base_recipe.get("action"), str)
            or base_recipe.get("action") not in {"catboost", "lightgbm"}):
        raise ValueError("Early stopping needs one ordinary CatBoost or LightGBM recipe")
    action = base_recipe["action"]
    config = validate_model_config(base_recipe.get("model_config"), action=action)
    if "target_transform" in config:
        raise ValueError("Early stopping does not support transformed targets")
    features = normalize_feature_config(base_recipe.get("feature_config"))
    if validation is None:
        raise ValueError("An explicit internal validation policy is required")
    choice = normalize_validation(validation)
    task_type = model_task_type(config)
    if (choice["strategy"] == "time" and task_type != "regression"
            or choice["strategy"] == "grouped" and task_type != "classification"):
        raise ValueError("Internal validation policy does not support this task type")
    maximum = _integer(max_iterations, "max_iterations", 50, 2000)
    rounds = _integer(stopping_rounds, "stopping_rounds", 1, min(200, maximum))
    return {"version": 1, "family": FAMILY,
            "base_recipe": {"action": action, "model_config": config, "feature_config": features},
            "validation": choice, "max_iterations": maximum, "stopping_rounds": rounds}


def _checked_recipe(recipe):
    expected = {"version", "family", "base_recipe", "validation", "max_iterations", "stopping_rounds"}
    if (not isinstance(recipe, dict) or set(recipe) != expected
            or type(recipe.get("version")) is not int or recipe["version"] != 1
            or recipe.get("family") != FAMILY):
        raise ValueError("Invalid early-stopping recipe")
    checked = make_boosting_recipe(recipe["base_recipe"], recipe["validation"],
        max_iterations=recipe["max_iterations"], stopping_rounds=recipe["stopping_rounds"])
    if checked != recipe:
        raise ValueError("Early-stopping recipe must use its normalized configuration")
    return checked


def _deadline(context):
    value = context.get("deadline_monotonic")
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Early stopping requires the host's finite monotonic deadline")
    if value <= time.monotonic():
        raise TimeoutError("Early-stopping invocation deadline exhausted")
    return float(value)


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Early-stopping invocation deadline exhausted")


def _names(value, label, *, empty=False):
    if (not isinstance(value, (list, tuple)) or not value and not empty
            or any(not isinstance(name, str) or not name or "\x00" in name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Invalid " + label + " schema")
    return list(value)


def _inputs(X_fit, y_fit, X_predict, context, recipe):
    if not isinstance(context, dict):
        raise ValueError("Early stopping requires a host context")
    metric = get_metric(context.get("metric"))
    if (context.get("task_type") != metric.task_type
            or context.get("prediction_kind") != metric.prediction_kind
            or model_task_type(recipe["base_recipe"]["model_config"]) != metric.task_type):
        raise ValueError("Early-stopping task/metric/model identity mismatch")
    features = _names(context.get("features"), "feature")
    numeric = _names(context.get("numeric"), "numeric", empty=True)
    categorical = _names(context.get("categorical"), "categorical", empty=True)
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise ValueError("Early-stopping feature roles must partition the declared features")
    for frame in (X_fit, X_predict):
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
                or list(frame.columns) != features or frame.empty):
            raise ValueError("Early stopping requires nonempty exact feature DataFrames")
    X = X_fit.copy(deep=True).reset_index(drop=True)
    predict = X_predict.copy(deep=True)
    if metric.classification:
        classes = _names(context.get("classes"), "class")
        y = pd.Series(label_values(y_fit, name="Fit labels"))
        if len(classes) < 2 or set(y) != set(classes):
            raise ValueError("Early stopping requires every declared class in fit rows")
    else:
        classes = []
        raw = np.asarray(y_fit)
        if raw.ndim != 1 or raw.dtype.kind == "b" or np.iscomplexobj(raw):
            raise ValueError("Regression fit targets must be a real numeric vector")
        y = pd.Series(pd.to_numeric(raw, errors="raise"), dtype=float)
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError("Regression fit targets must be finite")
    if len(y) != len(X):
        raise ValueError("Fit features and targets have different row counts")
    return X, y, predict, features, numeric, categorical, classes, metric


def _catboost_metric(metric, class_count):
    if metric.name == "Macro F1":
        return "TotalF1:average=Macro;use_weights=false"
    if metric.name == "Weighted F1":
        return "TotalF1:average=Weighted;use_weights=false"
    if metric.name == "Accuracy":
        return "Accuracy:use_weights=false"
    if metric.classification:
        # CatBoost's multiclass AUC variants differ from the host's OVR AUC.
        # Use an explicit proper-loss proxy rather than mislabel it exact AUC.
        return "MultiClass" if class_count > 2 else "Logloss"
    return "MAE" if metric.name == "MAE" else "R2" if metric.name == "R2" else "RMSE"


def _lightgbm_metric(metric, classes, positive_class):
    def evaluate(actual, predictions):
        values = np.asarray(predictions)
        if metric.classification:
            encoded = np.asarray(actual)
            if (not np.isfinite(encoded).all() or not np.equal(encoded, np.floor(encoded)).all()
                    or (encoded < 0).any() or (encoded >= len(classes)).any()):
                raise ValueError("LightGBM evaluation labels lost their explicit encoding")
            labels = np.asarray(classes)[encoded.astype(int)]
            if values.ndim == 1 and len(classes) == 2:
                values = np.column_stack([1 - values, values])
            if metric.prediction_kind == "labels":
                if values.shape != (len(labels), len(classes)):
                    raise ValueError("LightGBM evaluation probabilities have invalid shape")
                values = np.asarray(classes)[values.argmax(axis=1)]
            score = score_predictions(metric, labels, values, classes=classes,
                                      prediction_classes=classes, positive_class=positive_class)
        else:
            score = score_predictions(metric, actual, values)
        return "host_" + metric.name.replace(" ", "_"), score, metric.higher_is_better
    return evaluate


class _CatBoostDeadline:
    def __init__(self, deadline):
        self.deadline = deadline
        self.reached = False
        self.seen_iterations = 0

    def after_iteration(self, info):
        iteration = getattr(info, "iteration", None)
        if isinstance(iteration, Integral) and not isinstance(iteration, bool):
            self.seen_iterations = max(self.seen_iterations, int(iteration))
        self.reached = time.monotonic() >= self.deadline
        return not self.reached


def _lightgbm_callbacks(stopping_rounds, deadline):
    from lightgbm import early_stopping
    from lightgbm.callback import EarlyStopException

    early = early_stopping(stopping_rounds, first_metric_only=True, verbose=False)

    def stop_for_budget(environment):
        stop_for_budget.seen_iterations = max(stop_for_budget.seen_iterations, int(environment.iteration) + 1)
        if time.monotonic() >= deadline:
            best_iteration = early.best_iter[0] if early.best_iter else environment.iteration
            best_scores = early.best_score_list[0] if early.best_score_list else environment.evaluation_result_list
            raise EarlyStopException(best_iteration, best_scores)
    stop_for_budget.order = 40
    stop_for_budget.before_iteration = False
    stop_for_budget.seen_iterations = 0
    return [early, stop_for_budget]


def _make_estimator(action, task_type, parameters, categorical, early_metric):
    if action == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        constructor = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
        return constructor(**parameters, cat_features=list(categorical), eval_metric=early_metric,
                           random_seed=SEED, thread_count=THREADS, task_type="CPU",
                           allow_writing_files=False, verbose=False)
    from lightgbm import LGBMClassifier, LGBMRegressor

    constructor = LGBMClassifier if task_type == "classification" else LGBMRegressor
    return constructor(**parameters, metric="None", random_state=SEED, n_jobs=THREADS,
                       verbosity=-1, deterministic=True, force_col_wise=True)


def execute_boosting_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    """Measure an internal best iteration, then fit a fresh ordinary pipeline."""
    if not isinstance(context, dict):
        raise ValueError("Early stopping requires a host context")
    recipe = _checked_recipe(context.get("recipe"))
    deadline = _deadline(context)
    X, y, predict, features, numeric, categorical, classes, metric = _inputs(
        X_fit, y_fit, X_predict, context, recipe)
    try:
        fitting, held, split = split_rows(X, y, features, metric.classification, recipe["validation"])
    except ValueError:
        raise BoostingSkipError("internal_split_unavailable") from None
    if metric.classification and (set(y.iloc[fitting]) != set(classes) or set(y.iloc[held]) != set(classes)):
        raise BoostingSkipError("internal_split_missing_class")
    base = recipe["base_recipe"]
    action, config = base["action"], base["model_config"]
    feature_config = base["feature_config"]
    _, output_categorical = feature_schema(numeric, categorical, feature_config)
    if action == "catboost":
        cardinality = sum(X[name].nunique(dropna=False) for name in output_categorical)
        if (cardinality > MAX_NATIVE_CATEGORIES
                or len(X) * len(output_categorical) > MAX_NATIVE_CATEGORY_CELLS):
            raise BoostingSkipError("native_category_budget_exceeded")
    _check_deadline(deadline)
    started = time.monotonic()
    refit_row_scale = (len(X) / len(fitting)) ** 1.25 * 1.3
    internal_fraction = min(INTERNAL_TIME_FRACTION, .75 / (1 + refit_row_scale))
    internal_deadline = started + (deadline - started) * internal_fraction
    transformer = FeatureTransformer(numeric, categorical, feature_config)
    train_features = transformer.fit_transform(X.iloc[fitting], y.iloc[fitting])
    held_features = transformer.transform(X.iloc[held])
    parameters = deepcopy(config["parameters"])
    resource = "iterations" if action == "catboost" else "n_estimators"
    parameters[resource] = recipe["max_iterations"]
    if config.get("objective") == "mae":
        parameters["loss_function" if action == "catboost" else "objective"] = "MAE" if action == "catboost" else "regression_l1"
    if action == "lightgbm":
        frame = _LightGBMFrame(output_categorical)
        train_features = frame.fit_transform(train_features)
        held_features = frame.transform(held_features)
        if metric.classification:
            class_positions = {label: index for index, label in enumerate(classes)}
            train_y = y.iloc[fitting].map(class_positions).to_numpy(dtype=int)
            held_y = y.iloc[held].map(class_positions).to_numpy(dtype=int)
        else:
            train_y, held_y = y.iloc[fitting], y.iloc[held]
        early_metric = _lightgbm_metric(metric, classes, context.get("positive_class"))
        model = _make_estimator(action, metric.task_type, parameters, output_categorical, early_metric)
        callbacks = _lightgbm_callbacks(recipe["stopping_rounds"], internal_deadline)
        _check_deadline(internal_deadline)
        model.fit(train_features, train_y, eval_set=[(held_features, held_y)], eval_metric=early_metric,
                  callbacks=callbacks)
        best_count = getattr(model, "best_iteration_", None)
        seen_iterations = getattr(callbacks[-1], "seen_iterations", 0) if callbacks else 0
        early_metric_name = "host_" + metric.name.replace(" ", "_")
    else:
        early_metric = _catboost_metric(metric, len(classes))
        model = _make_estimator(action, metric.task_type, parameters, output_categorical, early_metric)
        callback = _CatBoostDeadline(internal_deadline)
        _check_deadline(internal_deadline)
        model.fit(train_features, y.iloc[fitting], eval_set=(held_features, y.iloc[held]),
                  early_stopping_rounds=recipe["stopping_rounds"], use_best_model=True,
                  callbacks=[callback])
        best_index = model.get_best_iteration()
        best_count = int(best_index) + 1 if isinstance(best_index, Integral) and not isinstance(best_index, bool) and best_index >= 0 else None
        seen_iterations = callback.seen_iterations
        early_metric_name = early_metric
    if (not isinstance(best_count, Integral) or isinstance(best_count, bool)
            or not 1 <= best_count <= recipe["max_iterations"]):
        raise BoostingSkipError("best_iteration_unavailable")
    best_count = int(best_count)
    trained_count = min(recipe["max_iterations"], max(best_count, seen_iterations))
    selected_count = max(50, best_count)
    internal_elapsed = max(0.0, time.monotonic() - started)
    _check_deadline(deadline)
    # This estimate is a bounded admission heuristic, not a promise of linear
    # training cost. The host still kills a worker at its original deadline.
    estimated_refit = internal_elapsed * refit_row_scale * (selected_count / trained_count) + .2
    if deadline - time.monotonic() <= estimated_refit:
        raise TimeoutError("Insufficient measured time estimate for full-fit boosting refit")
    selected_parameters = deepcopy(config["parameters"])
    selected_parameters[resource] = selected_count
    selected_config = make_model_config(action, metric.task_type, selected_parameters,
                                       objective=config.get("objective"))
    final_model = build_pipeline(action, metric.task_type, numeric, categorical,
                                 selected_config, feature_config)
    _check_deadline(deadline)
    final_model.fit(X, y)
    _check_deadline(deadline)
    if metric.classification:
        probabilities = probability_matrix(final_model.predict_proba(predict), classes=classes,
            prediction_classes=list(map(str, final_model.classes_)), positive_class=context.get("positive_class"), rows=len(predict))
        values = np.asarray(classes)[probabilities.argmax(axis=1)] if metric.prediction_kind == "labels" else probabilities
        result = {"predictions": values, "classes": classes}
    else:
        values = np.asarray(final_model.predict(predict), dtype=float)
        if values.shape != (len(predict),) or not np.isfinite(values).all():
            raise ValueError("Full-fit boosting returned invalid regression predictions")
        result = {"predictions": values}
    _check_deadline(deadline)
    evidence = {"version": 1, "family": FAMILY, "status": "completed", "action": action,
                "validation": split, "internal_fit_rows": len(fitting), "internal_validation_rows": len(held),
                "full_fit_rows": len(X), "predict_rows": len(predict), "best_iteration": best_count,
                "internal_iterations": trained_count,
                "selected_iterations": selected_count, "maximum_iterations": recipe["max_iterations"],
                "stopping_rounds": recipe["stopping_rounds"], "early_stopping_metric": early_metric_name,
                "internal_time_fraction": internal_fraction,
                "internal_elapsed_seconds": internal_elapsed, "estimated_refit_seconds": estimated_refit,
                "refit_estimate_guaranteed": False, "fresh_full_refit": True,
                "native_categories": action == "catboost", "classes": classes}
    return (result, evidence) if return_evidence else result


def fit_predict(X_fit, y_fit, X_predict, context):
    return execute_boosting_candidate(X_fit, y_fit, X_predict, context)


def candidate_source():
    """Generate the ordinary candidate protocol plus a non-scoring receipt."""
    return '''import json
from pathlib import Path
from agent.boosting_candidate import BoostingSkipError, execute_boosting_candidate

def fit_predict(X_fit, y_fit, X_predict, context):
    try:
        result, evidence = execute_boosting_candidate(
            X_fit, y_fit, X_predict, context, return_evidence=True)
    except BoostingSkipError as error:
        with Path("boosting_evidence.json").open("x", encoding="utf-8") as stream:
            json.dump({"version": 1, "family": "early_stopping_boosting_v1",
                       "status": "skipped", "reason": error.reason}, stream,
                      sort_keys=True, ensure_ascii=False, allow_nan=False)
        raise
    with Path("boosting_evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return result
'''
