"""Isolated CPU XGBoost candidate: fit-only early stopping, then a fresh refit.

The internal holdout is made only from this invocation's X_fit/y_fit using the
host's explicit split policy. Preprocessing is learned on the internal fitting
rows and the tree count on the internal held rows. A new preprocessor and a new
booster are then fitted on all X_fit/y_fit with that tree count. X_predict and
its outcomes never select anything, and the host phase is never a switch.

Host scoring, the process deadline and the worker supervisor remain
authoritative. The generated source writes ``xgboost_evidence.json``; it is a
diagnostic receipt, never a trusted quality score. Nothing here downloads,
calls an API or reads ``.env``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from agent.features import FeatureTransformer, feature_schema, normalize_feature_config
from agent.metrics import get_metric, label_values, probability_matrix
from agent.models import THREADS, _encoded_preparation
from agent.splits import normalize_validation, split_rows
from offline_baseline import SEED


VERSION = 1
FAMILY = "xgboost_native_v1"
EVIDENCE_NAME = "xgboost_evidence.json"
XGBOOST_VERSION = "2.1.4"
INTERNAL_TIME_FRACTION = .30
PREPROCESSING = "feature_transformer_then_ordinal_median_v1"
RUNTIME = {"package": "xgboost", "package_version": XGBOOST_VERSION, "tree_method": "hist",
           "device": "cpu", "max_threads": THREADS, "seed": SEED}

# Configured limits are admission checks made before large allocations. They
# are not measured peak resources and not an operating-system sandbox.
MAX_FIT_ROWS = 1_000_000
MAX_PREDICT_ROWS = 1_000_000
MAX_OUTPUT_FEATURES = 2048
MAX_MATRIX_CELLS = 50_000_000
MAX_CATEGORY_LEVELS = 500_000
MAX_CLASSES = 50
MAX_TOTAL_TREES = 40_000

_FLOAT_PARAMETERS = {
    "learning_rate": (0.01, 0.3), "min_child_weight": (0.0, 100.0),
    "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (0.0, 100.0), "reg_alpha": (0.0, 100.0), "gamma": (0.0, 10.0),
}
_INTEGER_PARAMETERS = {"max_depth": (1, 10), "max_bin": (16, 256)}
_NATIVE_NAMES = {"learning_rate": "eta", "reg_lambda": "lambda", "reg_alpha": "alpha"}
# Two fixed configurations; there is no hyperparameter search in v1.
FIXED_CONFIGURATIONS = {
    "depth6_v1": {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0,
                  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                  "reg_alpha": 0.0, "gamma": 0.0, "max_bin": 256},
    "depth4_regularized_v1": {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5.0,
                              "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0,
                              "reg_alpha": 0.1, "gamma": 0.0, "max_bin": 256},
}
DEFAULT_CONFIGURATION = "depth6_v1"
_LOSSES = {"squared_error": ("reg:squarederror", "rmse"),
           "absolute_error": ("reg:absoluteerror", "mae")}


class XGBoostSkipError(ValueError):
    """An expected eligibility stop which leaves the host baseline intact."""

    def __init__(self, reason, *, detail=None):
        allowed = {"xgboost_unavailable", "xgboost_version_mismatch", "fit_missing_class",
                   "internal_split_unavailable", "internal_split_missing_class",
                   "resource_cap_exceeded", "best_iteration_unavailable"}
        if reason not in allowed or detail is not None and not isinstance(detail, str):
            raise ValueError("Unknown bounded XGBoost skip reason")
        self.reason = reason
        self.detail = detail
        self.trace = {}
        super().__init__(reason if detail is None else reason + ":" + detail)


class XGBoostDeadlineError(TimeoutError):
    """The host's original deadline ended a stage; no new time is created."""

    def __init__(self, stage):
        self.stage = stage
        self.trace = {}
        super().__init__("XGBoost candidate deadline exhausted before or during " + stage)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " is outside its supported integer bounds")
    return value


def _parameters(value):
    result = dict(FIXED_CONFIGURATIONS[DEFAULT_CONFIGURATION])
    if value is None:
        return result
    known = set(_FLOAT_PARAMETERS) | set(_INTEGER_PARAMETERS)
    if (not isinstance(value, dict) or any(not isinstance(name, str) for name in value)
            or set(value) - known):
        raise ValueError("XGBoost parameters must use the fixed whitelist")
    for name, raw in value.items():
        if name in _INTEGER_PARAMETERS:
            result[name] = _integer(raw, name, *_INTEGER_PARAMETERS[name])
            continue
        low, high = _FLOAT_PARAMETERS[name]
        if (type(raw) not in {int, float} or not math.isfinite(raw)
                or not low <= raw <= high):
            raise ValueError(name + " is outside its supported numeric bounds")
        result[name] = float(raw)
    return result


def _loss(metric):
    if metric.classification:
        return "log_loss"
    return "absolute_error" if metric.name == "MAE" else "squared_error"


def make_xgboost_recipe(task_type, metric_name, *, validation, parameters=None,
                        feature_config=None, max_iterations=2000, stopping_rounds=50):
    """Freeze one normalized XGBoost configuration and an explicit split policy."""
    if not isinstance(task_type, str) or task_type not in {"regression", "classification"}:
        raise ValueError("XGBoost task_type must be regression or classification")
    metric = get_metric(metric_name)
    if metric.task_type != task_type:
        raise ValueError("XGBoost metric does not match the task type")
    if validation is None:
        raise ValueError("An explicit internal validation policy is required")
    choice = normalize_validation(validation)
    if (choice["strategy"] == "time" and task_type != "regression"
            or choice["strategy"] == "grouped" and task_type != "classification"):
        raise ValueError("Internal validation policy does not support this task type")
    maximum = _integer(max_iterations, "max_iterations", 50, 2000)
    rounds = _integer(stopping_rounds, "stopping_rounds", 1, min(200, maximum))
    return {"version": VERSION, "family": FAMILY, "task_type": task_type, "metric": metric.name,
            "loss": _loss(metric), "parameters": _parameters(deepcopy(parameters)),
            "feature_config": normalize_feature_config(deepcopy(feature_config)),
            "preprocessing": PREPROCESSING, "validation": choice,
            "max_iterations": maximum, "stopping_rounds": rounds, "runtime": dict(RUNTIME)}


def _checked_recipe(recipe):
    expected = {"version", "family", "task_type", "metric", "loss", "parameters", "feature_config",
                "preprocessing", "validation", "max_iterations", "stopping_rounds", "runtime"}
    if (not isinstance(recipe, dict) or set(recipe) != expected
            or type(recipe.get("version")) is not int or recipe["version"] != VERSION
            or recipe.get("family") != FAMILY):
        raise ValueError("Invalid XGBoost recipe")
    checked = make_xgboost_recipe(
        recipe["task_type"], recipe["metric"], validation=recipe["validation"],
        parameters=recipe["parameters"], feature_config=recipe["feature_config"],
        max_iterations=recipe["max_iterations"], stopping_rounds=recipe["stopping_rounds"])
    if checked != recipe:
        raise ValueError("XGBoost recipe must use its normalized configuration")
    return checked


def _deadline(context):
    value = context.get("deadline_monotonic")
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("XGBoost requires the host's finite monotonic deadline")
    if value <= time.monotonic():
        raise XGBoostDeadlineError("start")
    return float(value)


def _check_deadline(deadline, stage):
    if time.monotonic() >= deadline:
        raise XGBoostDeadlineError(stage)


def _names(value, label, *, empty=False):
    if (not isinstance(value, (list, tuple)) or not value and not empty
            or any(not isinstance(name, str) or not name or "\x00" in name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Invalid " + label + " schema")
    return list(value)


def _inputs(X_fit, y_fit, X_predict, context, recipe):
    metric = get_metric(context.get("metric"))
    if (context.get("task_type") != metric.task_type
            or context.get("prediction_kind") != metric.prediction_kind
            or recipe["metric"] != metric.name or recipe["task_type"] != metric.task_type):
        raise ValueError("XGBoost task/metric/recipe identity mismatch")
    features = _names(context.get("features"), "feature")
    numeric = _names(context.get("numeric"), "numeric", empty=True)
    categorical = _names(context.get("categorical"), "categorical", empty=True)
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise ValueError("XGBoost feature roles must partition the declared features")
    for frame in (X_fit, X_predict):
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
                or list(frame.columns) != features or frame.empty):
            raise ValueError("XGBoost requires nonempty exact feature DataFrames")
    # Keep frame validation allocation-light until _check_caps admits the
    # original shapes. Defensive copies are made only after that admission.
    X, predict = X_fit, X_predict
    if metric.classification:
        classes = _names(context.get("classes"), "class")
        y = pd.Series(label_values(y_fit, name="Fit labels"))
        if len(classes) < 2 or not set(y) <= set(classes):
            raise ValueError("XGBoost fit labels must belong to the declared classes")
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
    if metric.classification and set(y) != set(classes):
        raise XGBoostSkipError("fit_missing_class")
    return X, y, predict, features, numeric, categorical, classes, metric


def _xgboost():
    try:
        import xgboost
    except ImportError:
        raise XGBoostSkipError("xgboost_unavailable") from None
    if getattr(xgboost, "__version__", None) != XGBOOST_VERSION:
        raise XGBoostSkipError("xgboost_version_mismatch")
    return xgboost


def _threads():
    return max(1, min(THREADS, os.cpu_count() or 1))


def _check_caps(X, predict, output_features, output_categorical, classes, recipe):
    multiclass = len(classes) > 2
    checks = (("fit_rows", len(X), MAX_FIT_ROWS), ("predict_rows", len(predict), MAX_PREDICT_ROWS),
              ("output_features", len(output_features), MAX_OUTPUT_FEATURES),
              ("fit_matrix_cells", len(X) * len(output_features), MAX_MATRIX_CELLS),
              ("predict_matrix_cells", len(predict) * len(output_features), MAX_MATRIX_CELLS),
              ("classes", len(classes), MAX_CLASSES),
              ("total_trees", recipe["max_iterations"] * (len(classes) if multiclass else 1),
               MAX_TOTAL_TREES))
    for name, value, limit in checks:
        if value > limit:
            raise XGBoostSkipError("resource_cap_exceeded", detail=name)
    levels = sum(int(X[name].nunique(dropna=False)) for name in output_categorical)
    if levels > MAX_CATEGORY_LEVELS:
        raise XGBoostSkipError("resource_cap_exceeded", detail="category_levels")
    return {"fit_rows": MAX_FIT_ROWS, "predict_rows": MAX_PREDICT_ROWS,
            "output_features": MAX_OUTPUT_FEATURES, "matrix_cells": MAX_MATRIX_CELLS,
            "category_levels": MAX_CATEGORY_LEVELS, "classes": MAX_CLASSES,
            "total_trees": MAX_TOTAL_TREES, "observed_category_levels": levels,
            "configured_limits_are_measured_peaks": False}


def _preprocessor(numeric, categorical, feature_config):
    # Existing fit-only ordinal preparation: median numeric imputation and
    # literal-category ordinal codes; unseen categories become -1.
    output_numeric, output_categorical = feature_schema(numeric, categorical, feature_config)
    return Pipeline([("features", FeatureTransformer(numeric, categorical, feature_config)),
                     ("prepare", _encoded_preparation(output_numeric, output_categorical, linear=False))])


def _matrix(values, rows, columns):
    result = np.asarray(values, dtype=float)
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise ValueError("XGBoost preprocessing returned an invalid finite matrix")
    return result


def _fingerprint(pipeline):
    """Hash fitted medians and category vocabularies; no row values are kept."""
    prepare = pipeline.named_steps["prepare"]
    learned = {}
    numeric = prepare.named_transformers_.get("numeric")
    if numeric is not None:
        learned["medians"] = [float(value).hex()
                              for value in numeric.named_steps["missing"].statistics_]
    category = prepare.named_transformers_.get("category")
    if category is not None:
        learned["categories"] = [[str(value) for value in values]
                                 for values in category.named_steps["encode"].categories_]
    raw = json.dumps(learned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _native(recipe, class_count, threads):
    parameters = recipe["parameters"]
    if recipe["loss"] == "log_loss":
        objective, metric = (("binary:logistic", "logloss") if class_count == 2
                             else ("multi:softprob", "mlogloss"))
    else:
        objective, metric = _LOSSES[recipe["loss"]]
    result = {"objective": objective, "eval_metric": metric, "tree_method": "hist",
              "device": "cpu", "nthread": threads, "seed": SEED, "verbosity": 0,
              "validate_parameters": True}
    result.update({_NATIVE_NAMES.get(name, name): value for name, value in parameters.items()})
    if objective == "multi:softprob":
        result["num_class"] = class_count
    return result


def _relation(metric):
    if metric.name in {"RMSE", "MAE", "LogLoss"}:
        return "same_quantity"
    if metric.name in {"MSE", "R2"}:
        return "monotone_equivalent_on_fixed_held"
    return "proxy_not_host_metric"


def _train(xgb, parameters, matrix, rounds, deadline, *, held=None, stopping_rounds=None):
    class StopAtDeadline(xgb.callback.TrainingCallback):
        def __init__(self):
            super().__init__()
            self.reached = False

        def after_iteration(self, model, epoch, evals_log):
            self.reached = time.monotonic() >= deadline
            return self.reached

    clock = StopAtDeadline()
    callbacks, evals = [clock], []
    if held is not None:
        callbacks.insert(0, xgb.callback.EarlyStopping(
            rounds=stopping_rounds, metric_name=parameters["eval_metric"],
            data_name="held", save_best=False))
        evals = [(held, "held")]
    with xgb.config_context(verbosity=0):
        booster = xgb.train(parameters, matrix, num_boost_round=rounds, evals=evals,
                            callbacks=callbacks, verbose_eval=False)
    return booster, clock.reached


def _best_count(booster, maximum):
    try:
        best = booster.best_iteration
    except AttributeError:
        raise XGBoostSkipError("best_iteration_unavailable") from None
    if type(best) is not int or not 0 <= best < maximum:
        raise XGBoostSkipError("best_iteration_unavailable")
    return best + 1


def _probabilities(raw, classes, rows):
    values = np.asarray(raw, dtype=float)
    if len(classes) == 2:
        if values.shape != (rows,):
            raise ValueError("XGBoost binary probabilities have an invalid shape")
        values = np.column_stack([1.0 - values, values])
    elif values.shape != (rows, len(classes)):
        raise ValueError("XGBoost multiclass probabilities have an invalid shape")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("XGBoost returned invalid probabilities")
    sums = values.sum(axis=1)
    if np.abs(sums - 1.0).max() > 1e-4:
        raise ValueError("XGBoost probability rows do not sum to one")
    # Only float32 rounding is removed; class columns keep the host order.
    return probability_matrix(values / sums[:, None], classes=classes,
                              prediction_classes=classes, rows=rows)


def _execute(X_fit, y_fit, X_predict, context, trace):
    started = time.monotonic()
    recipe = _checked_recipe(context.get("recipe"))
    deadline = _deadline(context)
    X, y, predict, features, numeric, categorical, classes, metric = _inputs(
        X_fit, y_fit, X_predict, context, recipe)
    threads = _threads()
    trace.update({"version": VERSION, "family": FAMILY, "phase": context.get("phase"),
                  "recipe": deepcopy(recipe), "classes": list(classes),
                  "cpu": {"device": "cpu", "tree_method": "hist", "nthread": threads,
                          "max_threads": THREADS, "os_cpu_count": os.cpu_count()}})
    feature_config = recipe["feature_config"]
    output_numeric, output_categorical = feature_schema(numeric, categorical, feature_config)
    output_features = output_numeric + output_categorical
    trace["limits"] = _check_caps(X, predict, output_features, output_categorical, classes, recipe)
    X = X.copy(deep=True).reset_index(drop=True)
    predict = predict.copy(deep=True)
    try:
        fitting, held, split = split_rows(X, y, features, metric.classification, recipe["validation"])
    except ValueError:
        raise XGBoostSkipError("internal_split_unavailable") from None
    trace.update({"validation": split, "internal_fit_rows": len(fitting),
                  "internal_validation_rows": len(held), "full_fit_rows": len(X),
                  "predict_rows": len(predict), "output_features": len(output_features)})
    if metric.classification and (set(y.iloc[fitting]) != set(classes) or set(y.iloc[held]) != set(classes)):
        raise XGBoostSkipError("internal_split_missing_class")
    xgb = _xgboost()
    trace["package"] = {"name": "xgboost", "version": xgb.__version__}
    if metric.classification:
        positions = {label: index for index, label in enumerate(classes)}
        target = y.map(positions).to_numpy(dtype=float)
    else:
        target = y.to_numpy(dtype=float)
    parameters = _native(recipe, len(classes), threads)
    times = trace["times"] = {}
    _check_deadline(deadline, "internal_preprocessing")
    internal_started = time.monotonic()
    refit_row_scale = (len(X) / len(fitting)) ** 1.25 * 1.3
    internal_fraction = min(INTERNAL_TIME_FRACTION, .75 / (1 + refit_row_scale))
    internal_deadline = internal_started + (deadline - internal_started) * internal_fraction
    internal = _preprocessor(numeric, categorical, feature_config)
    train_matrix = _matrix(internal.fit_transform(X.iloc[fitting]), len(fitting), len(output_features))
    held_matrix = _matrix(internal.transform(X.iloc[held]), len(held), len(output_features))
    trace["internal_preprocessing_sha256"] = _fingerprint(internal)
    times["internal_preprocessing_seconds"] = time.monotonic() - internal_started
    _check_deadline(internal_deadline, "internal_training")
    stage = time.monotonic()
    train_data = xgb.DMatrix(train_matrix, label=target[fitting], nthread=threads)
    held_data = xgb.DMatrix(held_matrix, label=target[held], nthread=threads)
    booster, internal_stopped = _train(xgb, parameters, train_data, recipe["max_iterations"],
                                       internal_deadline, held=held_data,
                                       stopping_rounds=recipe["stopping_rounds"])
    trained = int(booster.num_boosted_rounds())
    times["internal_training_seconds"] = time.monotonic() - stage
    trace["trees"] = {"maximum": recipe["max_iterations"], "internal_trained": trained}
    selected = _best_count(booster, recipe["max_iterations"])
    best_score = float(booster.best_score)
    trace["trees"].update(best=selected, selected=selected)
    trace["stopping"] = {
        "stopped_by": ("internal_deadline" if internal_stopped else "early_stopping"
                       if trained < recipe["max_iterations"] else "max_iterations"),
        "objective": parameters["objective"], "metric": parameters["eval_metric"],
        "stopping_rounds": recipe["stopping_rounds"], "host_metric": metric.name,
        "relation": _relation(metric),
        "internal_best_score": best_score if math.isfinite(best_score) else None}
    internal_elapsed = time.monotonic() - internal_started
    _check_deadline(deadline, "refit_admission")
    # A bounded admission heuristic, not a promise of linear training cost.
    estimated_refit = internal_elapsed * refit_row_scale * (selected / trained) + .2
    trace["deadline"] = {"internal_time_fraction": internal_fraction,
                         "estimated_refit_seconds": estimated_refit, "estimate_guaranteed": False}
    if deadline - time.monotonic() <= estimated_refit:
        raise XGBoostDeadlineError("refit_admission")
    stage = time.monotonic()
    final_preprocessor = _preprocessor(numeric, categorical, feature_config)
    full_matrix = _matrix(final_preprocessor.fit_transform(X), len(X), len(output_features))
    trace["refit_preprocessing_sha256"] = _fingerprint(final_preprocessor)
    times["refit_preprocessing_seconds"] = time.monotonic() - stage
    _check_deadline(deadline, "refit_training")
    stage = time.monotonic()
    full_data = xgb.DMatrix(full_matrix, label=target, nthread=threads)
    final, refit_stopped = _train(xgb, parameters, full_data, selected, deadline)
    refit_trained = int(final.num_boosted_rounds())
    times["refit_training_seconds"] = time.monotonic() - stage
    trace["trees"]["refit_trained"] = refit_trained
    if refit_stopped or refit_trained != selected:
        raise XGBoostDeadlineError("refit_training")
    stage = time.monotonic()
    predict_matrix = _matrix(final_preprocessor.transform(predict), len(predict), len(output_features))
    _check_deadline(deadline, "prediction")
    with xgb.config_context(verbosity=0):
        raw = final.predict(xgb.DMatrix(predict_matrix, nthread=threads),
                            iteration_range=(0, selected))
    if metric.classification:
        probabilities = _probabilities(raw, classes, len(predict))
        values = (np.asarray(classes)[probabilities.argmax(axis=1)]
                  if metric.prediction_kind == "labels" else probabilities)
        result = {"predictions": values, "classes": list(classes)}
    else:
        values = np.asarray(raw, dtype=float)
        if values.shape != (len(predict),) or not np.isfinite(values).all():
            raise ValueError("XGBoost returned invalid regression predictions")
        result = {"predictions": values}
    times["prediction_seconds"] = time.monotonic() - stage
    _check_deadline(deadline, "prediction")
    times["total_seconds"] = time.monotonic() - started
    trace["deadline"].update(remaining_seconds_at_finish=deadline - time.monotonic(),
                             outcome="completed_before_deadline")
    trace["fresh_full_refit"] = True
    return result


def execute_xgboost_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    """Measure the internal best tree count, then refit all fit rows; no file I/O."""
    if not isinstance(context, dict):
        raise ValueError("XGBoost requires a host context")
    trace = {}
    try:
        result = _execute(X_fit, y_fit, X_predict, context, trace)
    except (XGBoostSkipError, XGBoostDeadlineError) as error:
        error.trace = deepcopy(trace)
        raise
    evidence = {"status": "completed", **trace}
    return (result, evidence) if return_evidence else result


def failure_evidence(error):
    """Receipt for a typed skip or deadline stop; it never carries predictions."""
    if isinstance(error, XGBoostSkipError):
        status, reason, detail, stage = "skipped", error.reason, error.detail, None
    elif isinstance(error, XGBoostDeadlineError):
        status, reason, detail, stage = "failed", "deadline_exhausted", None, error.stage
    else:
        raise TypeError("Only typed XGBoost stops have failure evidence")
    return {"version": VERSION, "family": FAMILY, "status": status, "reason": reason,
            "detail": detail, "stage": stage, "partial": deepcopy(error.trace)}


def fit_predict(X_fit, y_fit, X_predict, context):
    return execute_xgboost_candidate(X_fit, y_fit, X_predict, context)


def candidate_source():
    """Generate the ordinary candidate protocol plus a non-scoring receipt."""
    return '''import json
from pathlib import Path
from agent.xgboost_candidate import (
    XGBoostDeadlineError, XGBoostSkipError, execute_xgboost_candidate, failure_evidence)


def _write_evidence(evidence):
    with Path("xgboost_evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fit_predict(X_fit, y_fit, X_predict, context):
    try:
        result, evidence = execute_xgboost_candidate(
            X_fit, y_fit, X_predict, context, return_evidence=True)
    except (XGBoostSkipError, XGBoostDeadlineError) as error:
        _write_evidence(failure_evidence(error))
        raise
    _write_evidence(evidence)
    return result
'''
