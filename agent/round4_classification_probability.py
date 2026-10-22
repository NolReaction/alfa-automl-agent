"""Real, class-aligned probabilities for one fixed classification recipe.

This internal adapter preserves the host's Macro F1/labels task. It constructs
fresh public ``build_pipeline`` models, trains each on only the supplied fit,
and returns real predict_proba values to OOF callers. A two-component recipe
uses exactly the equal probability average of the current BASELINE_SOURCE.
The outer v3 wrapper owns label decisions, evidence files and the hard timeout.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agent.features import feature_schema, normalize_feature_config
from agent.metrics import get_metric, label_values, probability_matrix
from agent.model_config import ACTIONS, make_model_config, model_task_type, validate_model_config
from agent.models import build_pipeline


VERSION = 1
FAMILY = "round4_classification_probability_v1"
CAPS = {"max_fit_rows": 100_000, "max_predict_rows": 500_000, "max_output_features": 512,
        "max_matrix_cells": 10_000_000, "max_classes": 50, "max_components": 2, "max_threads": 8}


class ProbabilitySkipError(ValueError):
    """A checked coverage/resource condition prevents this invocation."""

    def __init__(self, reason):
        self.reason = reason
        self.evidence = {}
        super().__init__(reason)


class ProbabilityDeadlineError(TimeoutError):
    """No successful result is returned after the caller's original deadline."""

    def __init__(self, stage):
        self.stage = stage
        self.evidence = {}
        super().__init__("Probability adapter deadline exhausted at " + stage)


class ProbabilityModelError(RuntimeError):
    """Model failure with attempted/completed physical-fit accounting."""

    def __init__(self, stage, error_type):
        self.stage, self.error_type = stage, error_type
        self.evidence = {}
        super().__init__("Probability adapter " + stage + " failed: " + error_type)


def _names(values, name, *, minimum=0):
    if (not isinstance(values, (list, tuple)) or len(values) < minimum
            or any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in values)
            or len(values) != len(set(values))):
        raise ValueError(name + " must be an explicit unique string list")
    return list(values)


def _base(base_recipe):
    if not isinstance(base_recipe, dict):
        raise ValueError("A fixed ordinary base recipe is required")
    if "components" in base_recipe:
        if set(base_recipe) - {"components", "id", "label"}:
            raise ValueError("Unknown fixed-blend recipe fields")
        parts = base_recipe["components"]
        if not isinstance(parts, list) or not 1 <= len(parts) <= CAPS["max_components"]:
            raise ValueError("Use one classifier or an equal two-classifier blend")
    else:
        parts = [base_recipe]
    result = []
    for item in parts:
        if (not isinstance(item, dict) or set(item) - {"action", "model_config", "feature_config", "id", "label"}
                or item.get("action") not in ACTIONS):
            raise ValueError("Unknown ordinary classification component")
        config = validate_model_config(item.get("model_config"), action=item["action"])
        if model_task_type(config) != "classification":
            raise ValueError("Probability adapter requires classification components")
        result.append({"action": item["action"], "model_config": config,
                       "feature_config": normalize_feature_config(item.get("feature_config"))})
    return {"components": result}


def fixed_incumbent_recipe():
    """The previously published LGB400/RF120 control, supported by current main.

    This is a fixed recipe, not a claim that the current scheduler selects it
    on every task. No dataset, feature or class names are embedded.
    """
    return {"components": [
        {"action": "lightgbm", "model_config": make_model_config("lightgbm", "classification",
            {"n_estimators": 400, "num_leaves": 15, "reg_lambda": 3., "class_weight": "balanced"}),
         "feature_config": normalize_feature_config()},
        {"action": "baseline", "model_config": make_model_config("baseline", "classification"),
         "feature_config": normalize_feature_config()},
    ]}


def make_probability_recipe(base_recipe, classes):
    """Freeze complete model/feature settings and explicit probability order."""
    order = _names(classes, "classes", minimum=2)
    if len(order) > CAPS["max_classes"]:
        raise ValueError("Too many declared classes")
    recipe = {"version": VERSION, "family": FAMILY, "base_recipe": _base(deepcopy(base_recipe)),
              "classes": order, "aggregation": "arithmetic_mean", "caps": dict(CAPS),
              "metric": "Macro F1", "prediction_kind": "labels"}
    # Reject nonfinite or non-JSON model values before any fit occurs.
    return json.loads(json.dumps(recipe, allow_nan=False))


def _checked_recipe(value):
    expected = {"version", "family", "base_recipe", "classes", "aggregation", "caps", "metric", "prediction_kind"}
    if not isinstance(value, dict) or set(value) != expected or type(value.get("version")) is not int:
        raise ValueError("Invalid probability adapter recipe")
    normalized = make_probability_recipe(value["base_recipe"], value["classes"])
    if value != normalized:
        raise ValueError("Probability adapter requires its normalized fixed recipe")
    return normalized


def _deadline(context):
    value = context.get("deadline_monotonic")
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("A finite original host deadline is required")
    return float(value)


def _check_deadline(deadline, stage):
    if time.monotonic() >= deadline:
        raise ProbabilityDeadlineError(stage)


def _inputs(X_fit, y_fit, X_predict, context, recipe):
    metric = get_metric(context.get("metric"))
    if (metric.name != "Macro F1" or context.get("task_type") != "classification"
            or context.get("prediction_kind") != "labels"):
        raise ValueError("Keep the external task as classification Macro F1 labels")
    classes = _names(context.get("classes"), "context.classes", minimum=2)
    if classes != recipe["classes"]:
        raise ValueError("Host and frozen probability class orders differ")
    features = _names(context.get("features"), "features", minimum=1)
    numeric = _names(context.get("numeric"), "numeric")
    categorical = _names(context.get("categorical"), "categorical")
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise ValueError("Numeric and categorical roles must partition exact input features")
    for frame in (X_fit, X_predict):
        if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique
                or list(frame.columns) != features):
            raise ValueError("Nonempty DataFrames with the exact declared feature order are required")
    if len(X_fit) > CAPS["max_fit_rows"] or len(X_predict) > CAPS["max_predict_rows"]:
        raise ProbabilitySkipError("row_cap_exceeded")
    for component in recipe["base_recipe"]["components"]:
        outputs = feature_schema(numeric, categorical, component["feature_config"])
        columns = len(outputs[0]) + len(outputs[1])
        if (columns > CAPS["max_output_features"]
                or max(len(X_fit), len(X_predict)) * columns > CAPS["max_matrix_cells"]):
            raise ProbabilitySkipError("feature_matrix_cap_exceeded")
    labels = label_values(y_fit, name="Fit labels")
    if len(labels) != len(X_fit) or not set(labels) <= set(classes):
        raise ValueError("Fit labels must match the input rows and declared classes")
    if set(labels) != set(classes):
        raise ProbabilitySkipError("fit_missing_class")
    return labels.copy(), numeric, categorical, classes


def fit_predict_probabilities(X_fit, y_fit, X_predict, context, *, recipe=None,
                              sample_weight=None, fit_observer=None):
    """Return genuine probabilities for an internal caller; never return v3 labels.

    Fits are sequential, with no early-stopping/refit duplication. A hard limit
    requires the existing isolated worker: checking a deadline between calls
    cannot interrupt a running native fit. Exception.evidence records physical
    fit attempts/completions when a controlled skip, model error or timeout occurs.
    Optional fit_observer receives a fresh event dict immediately before each
    physical fit and after its completion; it owns any persistent event log.
    If observer work exhausts the deadline, cancelled_before_fit retracts the
    preceding started event before any model.fit call is made.
    Optional sample weights are computed by the caller from this fit only and
    passed unchanged through the public Pipeline model__sample_weight route.
    """
    started = time.monotonic()
    evidence = {"version": VERSION, "family": FAMILY, "physical_fits_started": 0,
                "physical_fits_completed": 0, "components": [], "threads": CAPS["max_threads"],
                "deadline_enforcement": "checks_between_stages; hard_limit_requires_host_worker"}
    try:
        if not isinstance(context, dict):
            raise ValueError("An explicit host context is required")
        checked = _checked_recipe(context.get("recipe") if recipe is None else recipe)
        if fit_observer is not None and not callable(fit_observer):
            raise ValueError("fit_observer must be callable or None")
        deadline = _deadline(context)
        _check_deadline(deadline, "start")
        labels, numeric, categorical, classes = _inputs(X_fit, y_fit, X_predict, context, checked)
        weights = None
        if sample_weight is not None:
            raw = np.asarray(sample_weight)
            originals = np.asarray(sample_weight, dtype=object)
            if (raw.dtype.kind not in "iuf" or raw.ndim != 1 or len(raw) != len(X_fit)
                    or any(isinstance(value, (bool, np.bool_)) for value in originals.flat)):
                raise ValueError("Sample weights must be one real numeric value per fit row")
            weights = np.asarray(raw, dtype=float).copy()
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Sample weights must be finite and strictly positive")
        evidence.update(classes=list(classes), fit_rows=len(X_fit), predict_rows=len(X_predict),
                        planned_physical_fits=len(checked["base_recipe"]["components"]),
                        sample_weight_provided=weights is not None)
        results = []
        for index, component in enumerate(checked["base_recipe"]["components"]):
            _check_deadline(deadline, "component_preparation")
            # Each model receives independent copies; preprocessing and the
            # estimator cannot mutate caller inputs or another component's fit.
            fitting, predicting = X_fit.copy(deep=True), X_predict.copy(deep=True)
            target = labels.copy()
            model = build_pipeline(component["action"], "classification", list(numeric), list(categorical),
                                   deepcopy(component["model_config"]), deepcopy(component["feature_config"]))
            record = {"index": index, "action": component["action"], "status": "prepared"}
            evidence["components"].append(record)
            with threadpool_limits(limits=CAPS["max_threads"]):
                _check_deadline(deadline, "model_fit")
                stage = time.monotonic()
                if fit_observer is not None:
                    fit_observer({"component_index": index, "action": component["action"], "status": "started",
                                  "fit_rows": len(X_fit), "predict_rows": len(X_predict)})
                    try:
                        _check_deadline(deadline, "after_fit_observer")
                    except ProbabilityDeadlineError:
                        record["status"] = "cancelled_before_fit"
                        fit_observer({"component_index": index, "action": component["action"],
                                      "status": "cancelled_before_fit", "fit_rows": len(X_fit),
                                      "predict_rows": len(X_predict)})
                        raise
                record["status"] = "fit_started"
                evidence["physical_fits_started"] += 1
                try:
                    kwargs = {} if weights is None else {"model__sample_weight": weights.copy()}
                    model.fit(fitting, target, **kwargs)
                except Exception as error:
                    raise ProbabilityModelError("fit", type(error).__name__) from error
                evidence["physical_fits_completed"] += 1
                record.update(status="fit_completed", fit_seconds=time.monotonic() - stage)
                if fit_observer is not None:
                    fit_observer({"component_index": index, "action": component["action"], "status": "completed",
                                  "fit_rows": len(X_fit), "predict_rows": len(X_predict)})
                _check_deadline(deadline, "after_model_fit")
                stage = time.monotonic()
                try:
                    raw = model.predict_proba(predicting)
                except Exception as error:
                    raise ProbabilityModelError("predict_proba", type(error).__name__) from error
                _check_deadline(deadline, "after_predict_proba")
                supplied = _names(list(map(str, model.classes_)), "model.classes_", minimum=2)
                # Full native matrix only; no one-column inference, absent
                # class filling, labels-to-probability conversion or renormalization.
                if np.asarray(raw).shape != (len(X_predict), len(supplied)):
                    raise ValueError("predict_proba must return every explicit model class column")
                values = probability_matrix(raw, classes=classes, prediction_classes=supplied, rows=len(X_predict))
                record.update(status="completed", predict_seconds=time.monotonic() - stage,
                              prediction_classes=supplied)
                results.append(values)
            del model, fitting, predicting, target
        _check_deadline(deadline, "aggregation")
        # Same operation/order as BASELINE_SOURCE; class ties follow the frozen
        # host class order when the outer wrapper later applies np.argmax.
        combined = sum(results) / len(results)
        combined = probability_matrix(combined, classes=classes, prediction_classes=classes, rows=len(X_predict))
        _check_deadline(deadline, "return")
        evidence.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        fresh_fit_on_all_supplied_rows=True)
        return {"probabilities": combined, "classes": list(classes), "evidence": evidence}
    except Exception as error:
        evidence.update(status="skipped" if isinstance(error, ProbabilitySkipError) else "failed",
                        elapsed_seconds=time.monotonic() - started)
        error.evidence = deepcopy(evidence)
        raise
