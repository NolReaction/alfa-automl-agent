"""An isolated, fit-only frequency-feature candidate for a frozen v3 baseline.

Each explicitly selected categorical feature receives one numeric value,
``log1p(count_fit(category))``. Exact strings remain exact: ``"NA"``, ``"0007"``,
``"7"`` and ``""`` are distinct. Genuine missing values share their own typed
category, separate from every string. Unknown categories have value zero.

There is no target argument to fitting the transform, no vocabulary learned
from prediction rows, and no global fitted dictionary. Every candidate call
constructs a fresh transform. Limits reject a trial before its baseline model
is fitted; no target-based selection or truncation takes place.

The transform and execution helper perform no I/O. The generated candidate
source writes ``frequency_evidence.json`` after the baseline returns, leaving
the host's strict predictions/classes return protocol unchanged. That artifact
is diagnostic evidence, never an admission score or a replacement receipt.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd


VERSION = 1
MAX_COLUMNS = 16
MAX_CATEGORIES = 4096
MAX_RECIPE_BYTES = 64 * 1024
MAX_CODE_BYTES = 256 * 1024
FEATURE_PREFIX = "__frequency_log1p_"
MISSING_POLICY = "genuine_missing_own_category"
FAMILY = "frequency_candidate_v1"
EVIDENCE_NAME = "frequency_evidence.json"


class FrequencyContractError(ValueError):
    """The caller's schema, recipe or categorical representation is invalid."""


class FrequencyLimitError(FrequencyContractError):
    """A fixed resource limit prevents the frequency candidate from running."""


class FrequencySkipError(FrequencyContractError):
    """No categorical feature was requested, so no additional model is needed."""


def _json_copy(value, *, object_only=False, limit=None):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise FrequencyContractError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise FrequencyContractError("Recipe/evidence must contain finite JSON values")

    if object_only and not isinstance(value, dict):
        raise FrequencyContractError("base_recipe must be a JSON object")
    try:
        check(value)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise FrequencyContractError("Recipe/evidence must contain finite JSON values") from error
    if limit is not None and len(raw) > limit:
        raise FrequencyLimitError("Frequency wrapper recipe exceeds 64 KiB")
    return json.loads(raw)


def _json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _names(values, label, *, allow_empty=False):
    if not isinstance(values, (list, tuple)):
        raise FrequencyContractError(f"{label} must be an explicit list/tuple of column names")
    if any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in values):
        raise FrequencyContractError(f"{label} must contain exact nonempty string column names")
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise FrequencyContractError(f"{label} must not contain duplicate column names")
    if not result and not allow_empty:
        raise FrequencyContractError(f"{label} must not be empty")
    return result


def _categorical_names(values):
    result = _names(values, "categorical_columns", allow_empty=True)
    if not result:
        raise FrequencySkipError("no_categorical_features")
    if len(result) > MAX_COLUMNS:
        raise FrequencyLimitError("categorical_column_limit_exceeded")
    return result


def _frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise FrequencyContractError(f"{label} must be a DataFrame with unique feature columns")
    _names(list(frame.columns), f"{label} columns")
    return frame


def _copy_frame(frame):
    result = frame.copy(deep=True)
    # pandas deep copies arrays, not nested Python objects stored in object cells.
    for column, dtype in enumerate(result.dtypes):
        if getattr(dtype, "kind", None) == "O":
            for row in range(len(result)):
                result.iat[row, column] = deepcopy(result.iat[row, column])
    return result


def _copy_labels(values):
    if isinstance(values, pd.Series):
        result = values.copy(deep=True)
        if getattr(result.dtype, "kind", None) == "O":
            for row in range(len(result)):
                result.iat[row] = deepcopy(result.iat[row])
        return result
    return deepcopy(values)


def _category_key(value):
    if isinstance(value, str):
        return ("string", str(value))
    # Explicitly handle genuine missing scalars. pd.isna on arbitrary values
    # may produce arrays; such values are not a categorical string contract.
    if value is None or value is pd.NA or value is pd.NaT:
        return ("missing", None)
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ("missing", None)
    if isinstance(value, (np.datetime64, np.timedelta64)) and bool(np.isnat(value)):
        return ("missing", None)
    raise FrequencyContractError("Categorical values must be exact strings or genuine missing scalars")


def frequency_feature_name(column):
    """Stable names depend only on the declared column, never on fit/test data."""
    name = _names([column], "column")[0]
    return FEATURE_PREFIX + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _fingerprint_value(value):
    if isinstance(value, str):
        return ["str", str(value)]
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pd.NA"]
    if value is pd.NaT:
        return ["pd.NaT"]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", str(int(value))]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        # float.hex also gives explicit text for nan and signed infinities.
        return ["float", number.hex()]
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return ["numpy_time", str(value.dtype), str(value)]
    raise FrequencyContractError("Frame fingerprint requires scalar string/numeric/missing feature values")


def frame_fingerprint(frame):
    """Typed SHA-256 of ordered columns/rows, excluding the pandas index.

    The host separately pins source IDs in fit/predict CSV. Excluding the index
    lets an auditor reconstruct this hash after CSV round-tripping RangeIndex.
    Raw string values and IEEE float hex values are not stripped or rounded.
    """
    frame = _frame(frame, "frame")
    # Stream the exact canonical {columns, rows} JSON rather than materializing
    # another complete Python/JSON copy of both potentially large frames.
    digest = hashlib.sha256()

    def write(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8"))

    digest.update(b'{"columns":')
    write(list(frame.columns))
    digest.update(b',"rows":[')
    first = True
    for row in frame.itertuples(index=False, name=None):
        if not first:
            digest.update(b",")
        write([_fingerprint_value(value) for value in row])
        first = False
    digest.update(b"]}")
    return digest.hexdigest()


@dataclass(frozen=True)
class FittedFrequencyTransform:
    """Immutable fitted counts; transform/to_json never extend or refit them."""

    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    added_columns: tuple[str, ...]
    fit_rows: int
    _counts: tuple[tuple[tuple[tuple[str, str | None], int], ...], ...]

    def transform(self, X_predict):
        frame = _frame(X_predict, "X_predict")
        if tuple(frame.columns) != self.feature_columns:
            raise FrequencyContractError("Prediction feature schema/order differs from the fitted schema")
        result = _copy_frame(frame)
        for source, output, frozen in zip(self.categorical_columns, self.added_columns, self._counts):
            counts = dict(frozen)
            result[output] = np.asarray([
                math.log1p(counts.get(_category_key(value), 0)) for value in frame[source]
            ], dtype=np.float64)
        return result

    def to_json(self):
        return {
            "version": VERSION, "kind": "fit_only_frequency_transform",
            "formula": "log1p(count_fit(category))", "missing_policy": MISSING_POLICY,
            "unknown_value": 0.0, "feature_columns": list(self.feature_columns),
            "categorical_columns": list(self.categorical_columns),
            "added_columns": list(self.added_columns), "fit_rows": self.fit_rows,
            "limits": {"max_columns": MAX_COLUMNS, "max_categories": MAX_CATEGORIES},
            "columns": [{"source_column": source, "output_column": output,
                         "counts": [{"kind": key[0], "value": key[1], "count": count}
                                    for key, count in frozen]}
                        for source, output, frozen in zip(
                            self.categorical_columns, self.added_columns, self._counts)],
        }

    @property
    def diagnostics(self):
        full = self.to_json()
        return {"version": VERSION, "active": True, "fit_rows": self.fit_rows,
                "column_count": len(self.categorical_columns), "missing_policy": MISSING_POLICY,
                "columns": [{"source_column": column["source_column"],
                             "output_column": column["output_column"],
                             "observed_categories": len(column["counts"]),
                             "missing_count": sum(item["count"] for item in column["counts"]
                                                  if item["kind"] == "missing"),
                             "counts_sha256": _json_hash(column["counts"])}
                            for column in full["columns"]],
                "transform_sha256": _json_hash(full)}


def fit_frequency_transform(X_fit, *, categorical_columns, feature_columns=None):
    """Count only X_fit, rejecting fixed limits rather than truncating columns."""
    frame = _frame(X_fit, "X_fit")
    if not len(frame):
        raise FrequencyContractError("X_fit must contain fitting rows")
    features = (tuple(frame.columns) if feature_columns is None
                else _names(feature_columns, "feature_columns"))
    if tuple(frame.columns) != features:
        raise FrequencyContractError("Fitting feature schema/order differs from declared features")
    categorical = _categorical_names(categorical_columns)
    if not set(categorical) <= set(features):
        raise FrequencyContractError("Categorical columns must be declared input features")
    additions = tuple(frequency_feature_name(column) for column in categorical)
    if len(set(additions)) != len(additions) or set(additions) & set(features):
        raise FrequencyContractError("Frequency feature names collide with the input schema")
    fitted = []
    for column in categorical:
        counts = {}
        for value in frame[column]:
            key = _category_key(value)
            if key not in counts and len(counts) >= MAX_CATEGORIES:
                raise FrequencyLimitError("observed_category_limit_exceeded")
            counts[key] = counts.get(key, 0) + 1
        fitted.append(tuple(sorted(counts.items(), key=lambda item: (item[0][0], item[0][1] or ""))))
    return FittedFrequencyTransform(features, categorical, additions, len(frame), tuple(fitted))


def make_frequency_recipe(base_recipe, categorical_columns):
    """Detach the complete baseline recipe, including all component recipes."""
    categories = _categorical_names(categorical_columns)
    base = _json_copy(base_recipe, object_only=True, limit=MAX_RECIPE_BYTES)
    return _json_copy({"version": VERSION, "family": FAMILY, "base_recipe": base,
                       "frequency": {"categorical_columns": list(categories),
                                     "max_columns": MAX_COLUMNS, "max_categories": MAX_CATEGORIES,
                                     "missing_policy": MISSING_POLICY, "feature_name_prefix": FEATURE_PREFIX}},
                      object_only=True, limit=MAX_RECIPE_BYTES)


def _checked_recipe(recipe):
    recipe = _json_copy(recipe, object_only=True, limit=MAX_RECIPE_BYTES)
    if set(recipe) != {"version", "family", "base_recipe", "frequency"}:
        raise FrequencyContractError("Invalid frequency wrapper recipe fields")
    if type(recipe["version"]) is not int or recipe["version"] != VERSION or recipe["family"] != FAMILY:
        raise FrequencyContractError("Unsupported frequency wrapper recipe version/family")
    options = recipe["frequency"]
    if not isinstance(options, dict) or set(options) != {
            "categorical_columns", "max_columns", "max_categories", "missing_policy", "feature_name_prefix"}:
        raise FrequencyContractError("Invalid frequency options")
    if (type(options["max_columns"]) is not int or options["max_columns"] != MAX_COLUMNS
            or type(options["max_categories"]) is not int or options["max_categories"] != MAX_CATEGORIES
            or options["missing_policy"] != MISSING_POLICY or options["feature_name_prefix"] != FEATURE_PREFIX):
        raise FrequencyContractError("Frequency v1 has fixed limits, naming and missing policy")
    categories = _categorical_names(options["categorical_columns"])
    base = _json_copy(recipe["base_recipe"], object_only=True, limit=MAX_RECIPE_BYTES)
    return base, categories


def execute_frequency_candidate(base_fit_predict, X_fit, y_fit, X_predict, context, *, return_evidence=False):
    """Pure wrapper execution; only the generated source writes its trace file."""
    if not callable(base_fit_predict) or not isinstance(context, dict):
        raise FrequencyContractError("A callable baseline and explicit context are required")
    base_recipe, categories = _checked_recipe(context.get("recipe"))
    features = _names(context.get("features"), "context.features")
    numeric = _names(context.get("numeric"), "context.numeric", allow_empty=True)
    categorical = _names(context.get("categorical"), "context.categorical", allow_empty=True)
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise FrequencyContractError("Context numeric/categorical columns must partition input features")
    if not set(categories) <= set(categorical):
        raise FrequencyContractError("Selected categories must be declared categorical features, never ID/target/numeric")
    fitted = fit_frequency_transform(X_fit, categorical_columns=categories, feature_columns=features)
    augmented_fit = fitted.transform(X_fit)
    augmented_predict = fitted.transform(X_predict)
    copied_context = deepcopy(context)
    copied_context["recipe"] = deepcopy(base_recipe)
    copied_context["features"] = list(features) + list(fitted.added_columns)
    copied_context["numeric"] = list(numeric) + list(fitted.added_columns)
    copied_context["categorical"] = list(categorical)
    evidence = {
        "version": VERSION, "phase": context.get("phase"), "nonce": context.get("nonce"),
        "code_hash": context.get("code_hash"), "recipe_hash": context.get("recipe_hash"),
        "base_recipe_hash": _json_hash(base_recipe), "transform": fitted.to_json(),
        "features": list(copied_context["features"]), "numeric": list(copied_context["numeric"]),
        "categorical": list(copied_context["categorical"]),
        "fit_rows": len(augmented_fit), "predict_rows": len(augmented_predict),
        "augmented_fit_sha256": frame_fingerprint(augmented_fit),
        "augmented_predict_sha256": frame_fingerprint(augmented_predict),
    }
    evidence = _json_copy(evidence)
    result = base_fit_predict(augmented_fit, _copy_labels(y_fit), augmented_predict, copied_context)
    if not isinstance(result, dict) or "predictions" not in result or set(result) - {"predictions", "classes"}:
        raise FrequencyContractError("Baseline must return predictions and optional classes")
    return (result, evidence) if return_evidence else result


def frequency_candidate_source(base_source):
    """Freeze the actual baseline implementation, not a substitute weak model.

    The baseline has its own globals; its components recipe and probability
    output stay intact. A trace is written after a successful base call inside
    the native invocation CWD, without changing the strict host return object.
    """
    if not isinstance(base_source, str) or not base_source.strip():
        raise FrequencyContractError("base_source must be a nonempty Python source string")
    try:
        compile(base_source, "frequency_base_candidate.py", "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise FrequencyContractError("base_source must be valid Python source") from error
    base_hash = hashlib.sha256(base_source.encode("utf-8")).hexdigest()
    source = f'''import json as _frequency_json
from pathlib import Path as _FrequencyPath
from agent.frequency_candidate import execute_frequency_candidate as _frequency_execute

_frequency_baseline_namespace = {{"__name__": "_frozen_frequency_baseline_"}}
exec(compile({base_source!r}, "frequency_base_candidate.py", "exec"), _frequency_baseline_namespace)
_frequency_baseline_fit_predict = _frequency_baseline_namespace.get("fit_predict")
if not callable(_frequency_baseline_fit_predict):
    raise ValueError("Frozen baseline must define callable fit_predict")

def fit_predict(X_fit, y_fit, X_predict, context):
    _result, _evidence = _frequency_execute(
        _frequency_baseline_fit_predict, X_fit, y_fit, X_predict, context,
        return_evidence=True)
    _evidence["base_source_hash"] = {base_hash!r}
    _FrequencyPath({EVIDENCE_NAME!r}).write_text(
        _frequency_json.dumps(_evidence, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return _result
'''
    if len(source.encode("utf-8")) > MAX_CODE_BYTES:
        raise FrequencyLimitError("Frequency wrapper source exceeds 256 KiB")
    return source
