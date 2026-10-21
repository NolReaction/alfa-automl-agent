"""Explicit, fixed-weight history recipes for the research v3 candidate host.

These helpers do not discover keys, choose a weight, execute a candidate or
enable history. The host owns task eligibility and compares the frozen recipe
on its ordinary inner and outer splits.
"""

from __future__ import annotations

import json
import math


MAX_RECIPE_BYTES = 64 * 1024


def _detached_json(value, label: str):
    """Validate finite JSON values and detach every nested mutable container."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{label} object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError(f"{label} must contain only JSON values")

    try:
        check(value)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ValueError(f"{label} must contain finite, serializable JSON values") from error
    if len(encoded) > MAX_RECIPE_BYTES:
        raise ValueError(f"{label} exceeds 64 KiB")
    return json.loads(encoded)


def _weight(value) -> float:
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and 0 <= value <= 1)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("blend_weight must be a finite JSON number in [0, 1]")
    return float(value)


def normalize_history_options(config, blend_weight=0.5) -> dict | None:
    """Return detached, explicit options, or None when no config was supplied.

    Even disabled options must have a valid weight. Configuration validation is
    delegated to the existing history adapter; no columns are inferred here.
    Tuples are serialized as JSON lists, as in the frozen-candidate contract.
    """
    weight = _weight(blend_weight)
    if config is None:
        return None
    from agent.history_candidate import HistoryConfig

    parsed = HistoryConfig.from_json(_detached_json(config, "History config"))
    return _detached_json({"history_config": parsed.to_json(), "blend_weight": weight},
                          "History options")


def make_history_recipe(base_recipe, config, blend_weight=0.5) -> dict:
    """Wrap an unchanged base recipe in an explicit, versioned history recipe."""
    options = normalize_history_options(config, blend_weight)
    if options is None:
        raise ValueError("A history recipe requires an explicit history config")
    if not isinstance(base_recipe, dict):
        raise ValueError("base_recipe must be a JSON object")
    return _detached_json({"version": 1, "family": "history_blend",
                           "base_recipe": base_recipe, **options}, "History recipe")


def history_candidate_source(base_source: str) -> str:
    """Append a pure fit/predict wrapper to the original executable base code.

    The base receives its own recipe and copies of the arguments, preserving
    the raw feature strings and fitted answers used by the history adapter.
    The same function and fixed recipe run on every host stage.
    """
    if not isinstance(base_source, str) or not base_source.strip():
        raise ValueError("base_source must be a non-empty Python source string")
    try:
        compile(base_source, "history_base_candidate.py", "exec")
    except (SyntaxError, ValueError) as error:
        raise ValueError("base_source must be valid Python source") from error
    return base_source + '''

_history_runtime_base_fit_predict = fit_predict
import copy as _history_runtime_copy
from agent.history_candidate import fit_history_candidate as _history_runtime_fit_history_candidate


def fit_predict(X_fit, y_fit, X_predict, context):
    if context["task_type"] != "regression":
        raise ValueError("History runtime supports regression only")
    _history_runtime_recipe = context["recipe"]
    _history_runtime_base_context = _history_runtime_copy.deepcopy(context)
    _history_runtime_base_context["recipe"] = _history_runtime_copy.deepcopy(
        _history_runtime_recipe["base_recipe"])
    _history_runtime_base_result = _history_runtime_base_fit_predict(
        _history_runtime_copy.deepcopy(X_fit), _history_runtime_copy.deepcopy(y_fit),
        _history_runtime_copy.deepcopy(X_predict), _history_runtime_base_context)
    if (not isinstance(_history_runtime_base_result, dict)
            or "predictions" not in _history_runtime_base_result):
        raise ValueError("Base fit_predict must return a predictions dictionary")
    _history_runtime_fitted = _history_runtime_fit_history_candidate(
        task_type=context["task_type"], metric_name=context["metric"]["name"],
        feature_columns=tuple(context["features"]), X_raw=X_fit, y=y_fit,
        config=_history_runtime_recipe["history_config"],
        blend_weight=_history_runtime_recipe["blend_weight"])
    _history_runtime_applied = _history_runtime_fitted.apply(
        X_predict, _history_runtime_base_result["predictions"])
    return {"predictions": _history_runtime_applied.prediction}
'''
