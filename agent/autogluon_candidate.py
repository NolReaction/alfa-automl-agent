"""Fit-only, phase-invariant AutoGluon 1.3.1 candidate for research v3.

The supervisor owns the hard process deadline. AutoGluon's ``time_limit`` and
``memory_limit`` are soft controls, including during ``refit_full``. A recipe
never sees host validation labels: its additional tuning split is made only
inside the supplied fit rows, following the host's explicit split policy.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from agent.metrics import get_metric, probability_matrix
from agent.splits import normalize_validation, split_rows


VERSION = 1
BACKEND_VERSION = "1.3.1"
FAMILY = "autogluon_tabular_v1"
EVIDENCE_NAME = "autogluon_evidence.json"
MODEL_FAMILIES = ("GBM", "CAT", "RF", "XT", "LR")
MAX_ROWS = 250_000
MAX_FEATURES = 256
MAX_CELLS = 16_000_000
_METRICS = {
    "RMSE": "root_mean_squared_error", "MAE": "mean_absolute_error",
    "MSE": "mean_squared_error", "R2": "r2", "Macro F1": "f1_macro",
    "Weighted F1": "f1_weighted", "Accuracy": "accuracy", "LogLoss": "log_loss",
    "ROC AUC": "roc_auc",
}


class AutoGluonCandidateError(ValueError):
    """The recipe, fit-only policy or returned model violates its contract."""


class AutoGluonUnavailableError(AutoGluonCandidateError):
    """The separately installed, pinned backend is unavailable."""


def _process_probe():
    """AutoGluon's soft memory limiter requires metadata for this real PID."""
    try:
        import psutil
        from agent.resources import _namespace_matches
    except ImportError:
        return False
    if not _namespace_matches():
        return False
    try:
        psutil.Process(os.getpid()).memory_info()
    except (psutil.Error, OSError):
        return False
    return True


def availability():
    """Inspect pinned installation and real process metadata, never secrets."""
    try:
        found = importlib.util.find_spec("autogluon.tabular") is not None
        version = importlib.metadata.version("autogluon.tabular") if found else None
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        found, version = False, None
    installed = found and version == BACKEND_VERSION
    process_ready = _process_probe() if installed else False
    return {"available": installed and process_ready,
            "reason": ("ready" if process_ready else "process_inventory_unavailable") if installed else
                      "version_mismatch" if found else "package_unavailable",
            "version": version, "required_version": BACKEND_VERSION,
            "process_inventory_available": process_ready}


def _number(value, name, low, high):
    if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
        raise AutoGluonCandidateError(f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def make_autogluon_recipe(task_type, metric_name, *, time_limit=180.0,
                         validation=None, num_cpus=8, memory_limit=24.0,
                         model_families=MODEL_FAMILIES):
    """Build a bounded recipe; the caller must pass its resolved split policy."""
    metric = get_metric(metric_name)
    if task_type not in {"classification", "regression"} or metric.task_type != task_type:
        raise AutoGluonCandidateError("Task type and metric must agree")
    if validation is None:
        raise AutoGluonCandidateError("An explicit host validation policy is required")
    try:
        policy = normalize_validation(validation)
    except (ValueError, TypeError) as error:
        raise AutoGluonCandidateError("Invalid resolved host validation policy") from error
    if ((policy["strategy"] == "time" and metric.classification)
            or (policy["strategy"] == "grouped" and not metric.classification)):
        raise AutoGluonCandidateError("Validation policy is unsupported for this task type")
    if type(num_cpus) is not int or not 1 <= num_cpus <= 8:
        raise AutoGluonCandidateError("num_cpus must be an integer from 1 to 8")
    if (not isinstance(model_families, (list, tuple)) or not model_families
            or any(type(name) is not str or name not in MODEL_FAMILIES for name in model_families)
            or len(set(model_families)) != len(model_families)):
        raise AutoGluonCandidateError("Only unique CPU model families GBM/CAT/RF/XT/LR are allowed")
    return {"version": VERSION, "family": FAMILY, "task_type": task_type,
            "metric": metric.name, "validation": policy,
            "time_limit": _number(time_limit, "time_limit", 15, 1800),
            "num_cpus": num_cpus, "memory_limit": _number(memory_limit, "memory_limit", 1, 24),
            "model_families": list(model_families)}


make_recipe = make_autogluon_recipe


def _recipe(value):
    required = {"version", "family", "task_type", "metric", "validation", "time_limit",
                "num_cpus", "memory_limit", "model_families"}
    if (not isinstance(value, dict) or set(value) != required or value["version"] != VERSION
            or type(value["version"]) is not int or value["family"] != FAMILY):
        raise AutoGluonCandidateError("Invalid AutoGluon recipe identity or fields")
    return make_autogluon_recipe(value["task_type"], value["metric"],
        time_limit=value["time_limit"], validation=value["validation"],
        num_cpus=value["num_cpus"], memory_limit=value["memory_limit"],
        model_families=value["model_families"])


def candidate_source():
    """The normal worker imports the same module from source or the flat ZIP."""
    return ("from agent.autogluon_candidate import fit_predict as _autogluon_fit_predict\n\n"
            "def fit_predict(X_fit, y_fit, X_predict, context):\n"
            "    return _autogluon_fit_predict(X_fit, y_fit, X_predict, context)\n")


def _load_backend():
    readiness = availability()
    if not readiness["available"]:
        raise AutoGluonUnavailableError("AutoGluon Tabular 1.3.1 unavailable: " + readiness["reason"])
    try:
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise AutoGluonUnavailableError("AutoGluon backend dependencies cannot be imported") from error
    return TabularPredictor, AutoMLPipelineFeatureGenerator


def _features(X_fit, X_predict, context):
    names = context.get("features")
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise AutoGluonCandidateError("Context must declare unique feature columns")
    for frame in (X_fit, X_predict):
        if (not isinstance(frame, pd.DataFrame) or not len(frame)
                or list(frame.columns) != list(names)):
            raise AutoGluonCandidateError("Feature frames must preserve the declared ordered columns")
    if (len(X_fit) > MAX_ROWS or len(X_predict) > MAX_ROWS or len(names) > MAX_FEATURES
            or (len(X_fit) + len(X_predict)) * len(names) > MAX_CELLS):
        raise AutoGluonCandidateError("AutoGluon candidate feature size limit exceeded")
    numeric, categorical = context.get("numeric"), context.get("categorical")
    if (not isinstance(numeric, (list, tuple)) or not isinstance(categorical, (list, tuple))
            or any(not isinstance(name, str) for name in [*numeric, *categorical])
            or len([*numeric, *categorical]) != len(names)
            or len(set([*numeric, *categorical])) != len(names)
            or set([*numeric, *categorical]) != set(names)):
        raise AutoGluonCandidateError("Numeric and categorical columns must partition the feature schema")
    fitting, prediction = X_fit.copy(deep=True), X_predict.copy(deep=True)
    for frame in (fitting, prediction):
        for name in numeric:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for name in categorical:
            frame[name] = frame[name].astype(object)
    return fitting.reset_index(drop=True), prediction.reset_index(drop=True)


def _deadline(context, limit, started):
    deadline = started + limit
    if "time_budget_seconds" in context:
        deadline = min(deadline, started + _number(context["time_budget_seconds"], "host time budget", 0.001, 10500))
    if "deadline_monotonic" in context:
        deadline = min(deadline, _number(context["deadline_monotonic"], "host deadline", 0.001, 1e15))
    return deadline


def _remaining(deadline, label):
    value = deadline - time.monotonic()
    if value <= 0.5:
        raise TimeoutError("AutoGluon candidate budget exhausted before " + label)
    return value


def _hyperparameters(families):
    # No neural networks, foundation models, external weights, HPO or Ray.
    definitions = {
        "GBM": {"num_boost_round": 400, "learning_rate": 0.05, "num_leaves": 31},
        "CAT": {"iterations": 400, "depth": 6, "learning_rate": 0.05},
        "RF": {"n_estimators": 160, "max_depth": 18},
        "XT": {"n_estimators": 160, "max_depth": 18},
        "LR": {"max_iter": 500},
    }
    return {name: deepcopy(definitions[name]) for name in families}


def fit_predict(X_fit, y_fit, X_predict, context):
    """Fit internal selection then the chosen family on all supplied fit rows.

    Identical code runs for inner, outer and full refit. Only the invocation's
    fit rows and the frozen recipe influence training; ``phase`` is not read.
    """
    started = time.monotonic()
    if not isinstance(context, dict) or context.get("protocol_version") != 3:
        raise AutoGluonCandidateError("AutoGluon requires the research v3 protocol")
    recipe = _recipe(context.get("recipe"))
    metric = get_metric(context.get("metric"))
    if (context.get("task_type") != recipe["task_type"] or metric.task_type != recipe["task_type"]
            or metric.name != recipe["metric"] or context.get("prediction_kind") != metric.prediction_kind):
        raise AutoGluonCandidateError("Recipe and host task/metric must match")
    deadline = _deadline(context, recipe["time_limit"], started)
    fitting, prediction = _features(X_fit, X_predict, context)
    labels = pd.Series(y_fit).reset_index(drop=True)
    if len(labels) != len(fitting) or labels.ndim != 1:
        raise AutoGluonCandidateError("Fit target must match feature rows")
    classes = context.get("classes", [])
    if metric.classification:
        if (not isinstance(classes, (list, tuple)) or len(classes) < 2
                or any(not isinstance(label, str) or not label.strip() for label in classes)
                or len(set(classes)) != len(classes)
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or set(labels) != set(classes)):
            raise AutoGluonCandidateError("Fit labels must contain every declared literal class")
    else:
        if any(isinstance(value, (bool, np.bool_)) for value in labels):
            raise AutoGluonCandidateError("Regression target must contain real finite numbers")
        labels = pd.to_numeric(labels, errors="raise")
        if not np.isfinite(labels.to_numpy(dtype=float)).all():
            raise AutoGluonCandidateError("Regression target must contain real finite numbers")
    # Split ORIGINAL raw values, before numeric/category preprocessing, exactly
    # as the host groups duplicate features or orders complete time periods.
    try:
        train_rows, tuning_rows, split = split_rows(X_fit.reset_index(drop=True), labels,
            list(context["features"]), metric.classification, recipe["validation"])
    except (ValueError, TypeError) as error:
        raise AutoGluonCandidateError("Fit-only tuning split unavailable for declared policy") from error
    if metric.classification and any(set(labels.iloc[rows]) != set(classes) for rows in (train_rows, tuning_rows)):
        raise AutoGluonCandidateError("Internal fit and tuning must each contain every declared class")
    target = "__alfa_autogluon_target__"
    while target in fitting.columns:
        target += "_"
    training = fitting.copy(deep=True)
    training[target] = labels.to_numpy()
    _remaining(deadline, "backend loading")
    predictor_type, generator_type = _load_backend()
    problem_type = ("binary" if len(classes) == 2 else "multiclass") if metric.classification else "regression"
    ag_metric = "roc_auc_ovr" if metric.name == "ROC AUC" and len(classes) > 2 else _METRICS[metric.name]
    elapsed = {}
    with tempfile.TemporaryDirectory(prefix=".autogluon_", dir=".") as temporary:
        predictor = predictor_type(label=target, problem_type=problem_type, eval_metric=ag_metric,
            path=str(Path(temporary) / "models"), verbosity=0, log_to_file=False,
            learner_kwargs={"label_count_threshold": 1},
            positive_class=context.get("positive_class") if problem_type == "binary" else None)
        generator = generator_type(enable_text_special_features=False,
                                   enable_text_ngram_features=False, enable_raw_text_features=False,
                                   enable_vision_features=False)
        initial_started = time.monotonic()
        predictor.fit(train_data=training.iloc[train_rows].copy(),
            tuning_data=training.iloc[tuning_rows].copy(),
            time_limit=_remaining(deadline, "initial fit") * 0.45,
            hyperparameters=_hyperparameters(recipe["model_families"]), feature_generator=generator,
            num_cpus=recipe["num_cpus"], num_gpus=0, memory_limit=recipe["memory_limit"],
            fit_strategy="sequential", auto_stack=False, dynamic_stacking=False,
            num_bag_folds=0, num_stack_levels=0, num_bag_sets=1,
            fit_weighted_ensemble=False, calibrate_decision_threshold=False,
            raise_on_no_models_fitted=True)
        elapsed["initial_fit"] = time.monotonic() - initial_started
        best = predictor.model_best
        if not isinstance(best, str) or not best:
            raise AutoGluonCandidateError("AutoGluon did not select a valid model")
        _remaining(deadline, "full fit")
        refit_started = time.monotonic()
        refit_map = predictor.refit_full(model=best, set_best_to_refit_full=False,
            num_cpus=recipe["num_cpus"], num_gpus=0, fit_strategy="sequential")
        elapsed["full_fit"] = time.monotonic() - refit_started
        if (not isinstance(refit_map, dict) or not isinstance(refit_map.get(best), str)
                or not refit_map[best] or refit_map[best] == best):
            raise AutoGluonCandidateError("AutoGluon did not produce the required full-data refit")
        refitted = refit_map[best]
        full_info = predictor.model_info(refitted)
        backend_fit_rows = full_info.get("num_samples") if isinstance(full_info, dict) else None
        if type(backend_fit_rows) is not int or backend_fit_rows != len(training):
            raise AutoGluonCandidateError("AutoGluon full-data refit row count is missing or incorrect")
        _remaining(deadline, "prediction")
        prediction_started = time.monotonic()
        if metric.classification:
            probabilities = predictor.predict_proba(prediction, model=refitted, as_multiclass=True, as_pandas=True)
            if (not isinstance(probabilities, pd.DataFrame) or len(probabilities) != len(prediction)
                    or not probabilities.index.equals(prediction.index)):
                raise AutoGluonCandidateError("AutoGluon probabilities must name their literal class columns")
            matrix = probability_matrix(probabilities.to_numpy(), classes=tuple(classes),
                prediction_classes=tuple(probabilities.columns), positive_class=context.get("positive_class"))
            values = np.asarray(classes)[matrix.argmax(axis=1)] if metric.prediction_kind == "labels" else matrix
        else:
            raw_predictions = predictor.predict(prediction, model=refitted, as_pandas=True)
            if not isinstance(raw_predictions, pd.Series) or not raw_predictions.index.equals(prediction.index):
                raise AutoGluonCandidateError("AutoGluon regression predictions must preserve row order")
            values = np.asarray(raw_predictions, dtype=float)
            if values.ndim != 1 or len(values) != len(prediction) or not np.isfinite(values).all():
                raise AutoGluonCandidateError("AutoGluon regression predictions must be a finite row-aligned vector")
        elapsed["prediction"] = time.monotonic() - prediction_started
        _remaining(deadline, "publication")
    evidence = {"version": VERSION, "family": FAMILY, "backend_version": BACKEND_VERSION,
        "fit_rows": len(fitting), "predict_rows": len(prediction),
        "internal_split": split, "internal_train_rows": len(train_rows),
        "internal_tuning_rows": len(tuning_rows), "full_fit_rows": backend_fit_rows,
        "best_model": best, "refit_model": refitted, "full_refit": True,
        "requested_time_limit": recipe["time_limit"],
        "effective_time_limit": deadline - started,
        "elapsed_seconds": elapsed, "num_cpus": recipe["num_cpus"],
        "memory_limit_gb": recipe["memory_limit"], "memory_limit_enforced": False,
        "model_families": recipe["model_families"], "api_requests": 0,
        "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()}
    Path(EVIDENCE_NAME).write_text(json.dumps(evidence, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False), encoding="utf-8")
    return {"predictions": values, **({"classes": list(classes)} if metric.classification else {})}
