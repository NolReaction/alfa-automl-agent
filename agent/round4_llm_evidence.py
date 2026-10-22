"""Bounded INNER evidence for an optional Qwen policy, not a host/report reader.

The caller is the trusted pre-freeze producer. A scope string is not proof of
provenance; schema checks cannot identify a secret disguised as a class label.
No helper here reads files, environment variables, targets or outer results.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
import re
import time

from agent.metrics import get_metric


MAX_SNAPSHOT_BYTES = 16 * 1024
MAX_RECIPE_BYTES = 8 * 1024
MAX_FEATURES = 64
MAX_CLASSES = 64
ALLOWED_INPUTS = frozenset({"task.md", "research_validation.json", "task_description.md",
    "data/train.csv", "data/test.csv", "data/data_dictionary.csv", "data/sample_submission.csv"})
INSTALLED_TOOLS = frozenset({"numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm", "xgboost"})
SECTIONS = frozenset({"version", "scope", "task", "allowed_inputs", "features", "shape", "split",
                     "baseline", "current_metric", "diagnostics", "timing", "installed_tools", "profile"})
_FORBIDDEN_KEY = re.compile(r"api.?key|secret|password|authorization|credential|request.?id|prompt|actuals|targets|row.?ids|outer|holdout", re.I)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _object(value, keys, *, required=()):
    if not isinstance(value, dict) or set(value) - set(keys) or set(required) - set(value):
        raise ValueError("Evidence object has missing or unsupported fields")
    return value


def _number(value, *, integer=False, nullable=True, lower=0, upper=1e15):
    if value is None and nullable:
        return None
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Evidence numeric aggregate is invalid")
    return value


def _literal(value, *, nullable=False, maximum=200):
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value or "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise ValueError("Evidence literal is invalid or too long")
    return value


def _hash(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Evidence requires a SHA-256 identity")
    return value


def _metric(value, *, measured=False):
    if value is None:
        return None
    fields = {"name", "higher_is_better"} | ({"value"} if measured else set())
    _object(value, fields, required=fields)
    try:
        metric = get_metric(value["name"])
    except (ValueError, TypeError):
        raise ValueError("Evidence metric must be a registered metric") from None
    if type(value["higher_is_better"]) is not bool or value["higher_is_better"] != metric.higher_is_better:
        raise ValueError("Evidence metric direction conflicts with the registry")
    result = {"name": metric.name, "higher_is_better": metric.higher_is_better}
    if measured:
        result["value"] = _number(value["value"], nullable=False, lower=-1e100, upper=1e100)
    return result


def validate_recipe_identity(value, *, require_recipe=False):
    """Check code identity syntax and canonical JSON recipe identity.

    Code bytes are intentionally unavailable; the host must supply their pinned
    hash. A recipe hash is checked against the exact normalized settings when
    those settings are supplied. No fuzzy model-name equivalence is used.
    """
    _object(value, {"code_hash", "recipe_hash", "recipe"}, required={"code_hash", "recipe_hash"})
    result = {name: _hash(value[name]) for name in ("code_hash", "recipe_hash")}
    recipe = value.get("recipe")
    if recipe is None:
        if require_recipe:
            raise ValueError("Frozen baseline recipe settings are required")
        return result
    if not isinstance(recipe, dict):
        raise ValueError("Recipe settings must be an object")
    nodes = [0]
    def check(item, depth=0):
        nodes[0] += 1
        if depth > 8 or nodes[0] > 512:
            raise ValueError("Recipe settings exceed structural limits")
        if isinstance(item, dict):
            for key, child in item.items():
                _literal(key, maximum=100)
                if _FORBIDDEN_KEY.search(key):
                    raise ValueError("Recipe contains a forbidden field")
                check(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            if len(item) > 64:
                raise ValueError("Recipe settings contain an oversized sequence")
            for child in item:
                check(child, depth + 1)
        elif isinstance(item, str):
            _literal(item, maximum=200)
            if ("\n" in item or "\r" in item or "://" in item or "\\" in item
                    or item.startswith(("/", "~", "sk-"))):
                raise ValueError("Recipe strings must be settings, not source or private paths")
        elif item is not None and type(item) not in (bool, int, float):
            raise ValueError("Recipe settings must be finite JSON values")
    check(recipe)
    try:
        encoded = _encoded(recipe)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Recipe settings must be finite JSON values") from None
    if len(encoded) > MAX_RECIPE_BYTES or hashlib.sha256(encoded).hexdigest() != result["recipe_hash"]:
        raise ValueError("Recipe size or normalized identity is invalid")
    result["recipe"] = json.loads(encoded)
    return result


def _diagnostics(value, task):
    if value is None:
        return None
    _object(value, {"version", "scope", "rows", "metric", "regression", "classification"},
            required={"scope", "rows", "metric"})
    if value.get("scope") != "inner_selection" or value.get("version", 1) != 1:
        raise ValueError("Diagnostics require verified INNER aggregate provenance")
    result = {"version": 1, "scope": "inner_selection", "rows": _number(value["rows"], integer=True),
              "metric": _metric(value["metric"], measured=True)}
    if result["metric"] is not None and task["metric"] is not None and result["metric"]["name"] != task["metric"]["name"]:
        raise ValueError("INNER diagnostic uses a different metric")
    if "regression" in value:
        if task["task_type"] != "regression" or "classification" in value:
            raise ValueError("Diagnostic task type does not match")
        keys = {"mean_bias", "mae", "absolute_error_p50", "absolute_error_p90", "absolute_error_p99"}
        result["regression"] = {key: _number(item, lower=-1e100 if key == "mean_bias" else 0, upper=1e100)
                                for key, item in _object(value["regression"], keys).items()}
    if "classification" in value:
        if task["task_type"] != "classification" or task["classes"] is None:
            raise ValueError("Classification diagnostics require declared classes")
        keys = {"classes_total", "observed_classes", "accuracy", "per_class", "worst_confusions",
                "omitted_classes", "omitted_confusions", "label_rule"}
        data = _object(value["classification"], keys)
        summary = {}
        for key in keys - {"per_class", "worst_confusions", "label_rule"}:
            if key in data:
                summary[key] = _number(data[key], integer=key != "accuracy", upper=1 if key == "accuracy" else 1e15)
        for key, limit in (("per_class", 12), ("worst_confusions", 8)):
            records = data.get(key, [])
            if not isinstance(records, list) or len(records) > limit:
                raise ValueError("Classification diagnostics exceed record limits")
            summary[key] = []
            for record in records:
                fields = {"label", "support", "recall", "f1"} if key == "per_class" else {"actual_class", "predicted_class", "count"}
                _object(record, fields, required=fields)
                item = {}
                for field, entry in record.items():
                    if field in {"label", "actual_class", "predicted_class"}:
                        if entry not in task["classes"]:
                            raise ValueError("INNER diagnostic label is undeclared")
                        item[field] = entry
                    else:
                        item[field] = _number(entry, integer=field in {"support", "count"},
                                               upper=1 if field in {"recall", "f1"} else 1e15)
                summary[key].append(item)
        if "label_rule" in data:
            if data["label_rule"] != "argmax_in_declared_class_order":
                raise ValueError("Unsupported diagnostic label rule")
            summary["label_rule"] = data["label_rule"]
        result["classification"] = summary
    return result


def validate_inner_snapshot(snapshot):
    """Validate/copy an explicit pre-freeze projection, never a full report.

    Missing optional information becomes null. Oversized fields are rejected,
    not silently truncated into an incomplete task or probability contract.
    The producer must already have removed IDs/targets from feature roles.
    """
    _object(snapshot, SECTIONS, required={"version", "scope", "task"})
    if type(snapshot["version"]) is not int or snapshot["version"] != 1 or snapshot["scope"] != "inner_selection":
        raise ValueError("Only version-one INNER evidence is supported")
    task = _object(snapshot["task"], {"task_type", "metric", "prediction_kind", "classes", "probability_classes", "positive_class"},
                   required={"task_type", "metric", "prediction_kind"})
    kind, prediction = task["task_type"], task["prediction_kind"]
    if not isinstance(kind, str) or not isinstance(prediction, str) or kind not in {"regression", "classification"} or prediction not in {"regression", "labels", "probabilities"}:
        raise ValueError("Unsupported task or prediction contract")
    metric = _metric(task["metric"])
    if metric is None:
        raise ValueError("The required task metric cannot be unknown")
    if metric is not None:
        definition = get_metric(metric["name"])
        if (kind, prediction) != (definition.task_type, definition.prediction_kind):
            raise ValueError("Task or prediction kind conflicts with its metric")
    classes = task.get("classes", [] if kind == "regression" else None)
    probability_classes = task.get("probability_classes", [] if prediction != "probabilities" else None)
    for values in (classes, probability_classes):
        if values is not None:
            if not isinstance(values, (list, tuple)) or len(values) > MAX_CLASSES:
                raise ValueError("Class contract exceeds size limits")
            for label in values:
                _literal(label)
            if len(set(values)) != len(values):
                raise ValueError("Class contract contains duplicate labels")
    classes = list(classes) if classes is not None else None
    probability_classes = list(probability_classes) if probability_classes is not None else None
    positive = _literal(task.get("positive_class"), nullable=True)
    if kind == "classification" and not classes:
        raise ValueError("Classification requires complete nonempty literal labels")
    if prediction == "probabilities" and not probability_classes:
        raise ValueError("Probability prediction requires complete column order")
    if kind == "regression" and (classes or positive is not None or prediction != "regression"):
        raise ValueError("Regression cannot carry a classification contract")
    if classes is not None and positive is not None and positive not in classes:
        raise ValueError("Positive class is undeclared")
    if prediction != "probabilities" and probability_classes:
        raise ValueError("Probability order supplied for non-probability predictions")
    if prediction == "probabilities" and classes is not None and probability_classes is not None:
        if not probability_classes or (set(probability_classes) != set(classes)
            and not (len(classes) == 2 and probability_classes == [positive])):
            raise ValueError("Probability columns do not match declared classes")
    task = {"task_type": kind, "metric": metric, "prediction_kind": prediction,
            "classes": classes, "probability_classes": probability_classes, "positive_class": positive}
    result = {key: None for key in SECTIONS}
    result.update(version=1, scope="inner_selection", task=task)
    if snapshot.get("allowed_inputs") is not None:
        inputs = snapshot["allowed_inputs"]
        if not isinstance(inputs, (list, tuple)) or any(not isinstance(value, str) or value not in ALLOWED_INPUTS for value in inputs) or len(set(inputs)) != len(inputs):
            raise ValueError("Only declared host relative inputs are allowed")
        result["allowed_inputs"] = list(inputs)
    if snapshot.get("features") is not None:
        features = snapshot["features"]
        if not isinstance(features, (list, tuple)) or len(features) > MAX_FEATURES:
            raise ValueError("Feature schema exceeds its size limit")
        result["features"] = []
        for feature in features:
            _object(feature, {"name", "role"}, required={"name", "role"})
            name = _literal(feature["name"])
            if not isinstance(feature["role"], str) or feature["role"] not in {"numeric", "categorical"}:
                raise ValueError("Feature roles exclude target and ID")
            result["features"].append({"name": name, "role": feature["role"]})
        if len({item["name"] for item in result["features"]}) != len(features):
            raise ValueError("Feature names must be unique")
    if snapshot.get("shape") is not None:
        keys = {"fit_rows", "validation_rows", "feature_count"}
        shape = _object(snapshot["shape"], keys)
        result["shape"] = {key: _number(shape.get(key), integer=True) for key in keys}
        if result["features"] is not None and result["shape"].get("feature_count") not in {None, len(result["features"])}:
            raise ValueError("Feature count conflicts with schema")
    if snapshot.get("split") is not None:
        split = _object(snapshot["split"], {"strategy", "seed"})
        if split.get("strategy") is not None and (not isinstance(split["strategy"], str) or split["strategy"] not in {"random", "stratified", "group", "time", "temporal", "grouped"}):
            raise ValueError("Split strategy is unsupported")
        result["split"] = {"strategy": split.get("strategy"), "seed": _number(split.get("seed"), integer=True, upper=2**32 - 1)}
    if snapshot.get("baseline") is not None:
        result["baseline"] = validate_recipe_identity(snapshot["baseline"], require_recipe=True)
    result["current_metric"] = _metric(snapshot.get("current_metric"), measured=True)
    if result["current_metric"] and metric and result["current_metric"]["name"] != metric["name"]:
        raise ValueError("Current INNER metric differs from task metric")
    result["diagnostics"] = _diagnostics(snapshot.get("diagnostics"), task)
    if snapshot.get("timing") is not None:
        keys = {"remaining_seconds", "final_reserve_seconds", "maximum_experiment_seconds", "last_fit_seconds", "memory_limit_bytes", "cpu_threads"}
        timing = _object(snapshot["timing"], keys)
        result["timing"] = {key: _number(timing.get(key), integer=key in {"memory_limit_bytes", "cpu_threads"},
                                       upper=8 if key == "cpu_threads" else 1e15)
                            for key in keys}
    if snapshot.get("installed_tools") is not None:
        tools = snapshot["installed_tools"]
        if not isinstance(tools, (list, tuple)) or len(tools) > len(INSTALLED_TOOLS) or any(not isinstance(tool, str) or tool not in INSTALLED_TOOLS for tool in tools):
            raise ValueError("Installed tools must be verified names from the permitted set")
        result["installed_tools"] = sorted(set(tools))
    if snapshot.get("profile") is not None:
        result["profile"] = _validate_profile(snapshot["profile"], result["features"])
    if len(_encoded(result)) > MAX_SNAPSHOT_BYTES:
        raise ValueError("INNER evidence exceeds 16 KiB")
    return deepcopy(result)


def _validate_profile(profile, features):
    _object(profile, {"scope", "sample_rows", "rows_available", "columns", "omitted_columns", "limited", "elapsed_seconds"},
            required={"scope", "sample_rows", "rows_available", "columns", "omitted_columns", "limited", "elapsed_seconds"})
    if profile["scope"] != "inner_fit_features" or features is None or type(profile["limited"]) is not bool:
        raise ValueError("Profile requires INNER-fit feature provenance and schema")
    rows = _number(profile["sample_rows"], integer=True, nullable=False, upper=512)
    available = _number(profile["rows_available"], integer=True, nullable=False)
    if rows > available or not isinstance(profile["columns"], list) or len(profile["columns"]) > 32:
        raise ValueError("Invalid bounded feature profile")
    names = {feature["name"] for feature in features}
    columns = []
    for column in profile["columns"]:
        _object(column, {"name", "missing_count", "distinct_count", "numeric_finite_count"}, required={"name", "missing_count", "distinct_count", "numeric_finite_count"})
        if column["name"] not in names:
            raise ValueError("Profile references undeclared feature")
        columns.append({"name": column["name"], **{key: _number(column[key], integer=True, upper=rows)
                        for key in ("missing_count", "distinct_count", "numeric_finite_count")}})
    return {"scope": "inner_fit_features", "sample_rows": rows, "rows_available": available,
            "columns": columns, "omitted_columns": _number(profile["omitted_columns"], integer=True),
            "limited": profile["limited"], "elapsed_seconds": _number(profile["elapsed_seconds"])}


def profile_inner_fit(frame, feature_roles, *, max_rows=512, max_cells=16384, max_seconds=.05):
    """Optional cheap counts from an explicitly feature-only INNER-fit frame.

    Its producer must select only gate.research_train.iloc[gate.fit_rows] and
    declared feature columns. Passing a larger/other scope is a caller error
    that this pure argument boundary cannot discover from row values.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 512 or type(max_cells) is not int or not 1 <= max_cells <= 16384:
        raise ValueError("Profile row/cell limit is invalid")
    _number(max_seconds, nullable=False, lower=.001, upper=.25)
    if not isinstance(feature_roles, list) or not feature_roles or len(feature_roles) > MAX_FEATURES:
        raise ValueError("Profile requires a bounded feature schema")
    for feature in feature_roles:
        _object(feature, {"name", "role"}, required={"name", "role"})
        _literal(feature["name"])
        if not isinstance(feature["role"], str) or feature["role"] not in {"numeric", "categorical"}:
            raise ValueError("Profile feature role cannot be ID or target")
    names = [feature["name"] for feature in feature_roles]
    if list(frame.columns) != names or len(set(names)) != len(names):
        raise ValueError("Profile frame must contain exactly the declared features")
    start = time.monotonic()
    columns = []
    selected = feature_roles[:min(32, max_cells)]
    count = min(len(frame), max_rows, max_cells // max(1, len(selected)))
    for feature in selected:
        if time.monotonic() - start >= max_seconds:
            break
        values = frame[feature["name"]].iloc[:count]
        missing = finite = 0
        distinct = set()
        incomplete = expired = False
        for value in values:
            if time.monotonic() - start >= max_seconds:
                incomplete = expired = True
                break
            if ((value is not None and not isinstance(value, (str, Real)))
                    or (isinstance(value, str) and len(value) > 4096)):
                incomplete = True
                break
            absent = value is None or (isinstance(value, str) and not value.strip())
            if not absent and isinstance(value, float):
                absent = math.isnan(value)
            if absent:
                missing += 1
                continue
            # Distinct values remain local; no examples or raw values are emitted.
            distinct.add(str(value))
            if feature["role"] == "numeric":
                try:
                    finite += int(math.isfinite(float(value)))
                except (ValueError, TypeError, OverflowError):
                    pass
        if expired:
            break
        if incomplete:
            continue
        columns.append({"name": feature["name"], "missing_count": missing,
                        "distinct_count": len(distinct),
                        "numeric_finite_count": finite if feature["role"] == "numeric" else None})
    profile = {"scope": "inner_fit_features", "sample_rows": count if columns else 0, "rows_available": len(frame),
               "columns": columns, "omitted_columns": len(names) - len(columns),
               "limited": count < len(frame) or len(columns) < len(names),
               "elapsed_seconds": time.monotonic() - start}
    return _validate_profile(profile, feature_roles)
