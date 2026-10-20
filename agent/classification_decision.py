"""Calibration-only, bounded class decisions for Macro F1.

The caller supplies probabilities on a calibration set excluded from model and
preprocessing fitting. It must fix group/time-safe splits before training. This
module neither infers a split policy nor reads files, environment, or audit
answers. An invalid or unavailable split must not be replaced with a random
split: the caller can pass empty calibration arrays to obtain identity.

Only label decisions change. Do not use this rule to alter probabilities or to
score LogLoss, ROC AUC, or metrics other than Macro F1. JSON rules retain the
canonical class order, including the tie order, independently of prediction
column order. No model refit or automatic runtime integration happens here.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


DEFAULT_OPTIONS = {
    "multiplier_grid": [0.5, 0.75, 1.0, 4.0 / 3.0, 2.0],
    "multiplier_min": 0.5,
    "multiplier_max": 2.0,
    "max_rounds": 3,
    "max_candidates": 128,
    "min_samples": 100,
    "min_class_count": 10,
}
_SCORE_TOLERANCE = 1e-12
_DIAGNOSTIC_KEYS = {
    "n_samples", "class_counts", "candidates_evaluated", "rounds_completed",
    "stop_reason", "fallback_reason", "control_score", "selected_score",
}


def _strings(values, name, *, minimum=0):
    try:
        result = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional string sequence") from error
    if result.ndim != 1 or len(result) < minimum:
        raise ValueError(f"{name} must be a one-dimensional string sequence")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain exact nonempty string labels")
    return [str(value) for value in result]


def _classes(values):
    result = _strings(values, "classes", minimum=2)
    if len(set(result)) != len(result):
        raise ValueError("classes must be unique")
    return result


def _probabilities(values, n_classes):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind in "bcUS":
            raise ValueError("Probabilities must be real numeric values")
        if raw.dtype.kind == "O" and any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in raw.flat):
            raise ValueError("Probabilities must be real numeric values")
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Probabilities must be real numeric values") from error
    if result.ndim != 2 or result.shape[1] != n_classes:
        raise ValueError("Probabilities must have shape (rows, number of classes)")
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    return result


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _options(options):
    if options is not None and not isinstance(options, dict):
        raise ValueError("options must be a dictionary")
    options = {} if options is None else options
    if not set(options) <= set(DEFAULT_OPTIONS):
        raise ValueError("Unknown decision-rule options")
    result = {**DEFAULT_OPTIONS, **options}
    for name, maximum in (("max_rounds", 10), ("max_candidates", 4096),
                          ("min_samples", 1_000_000_000),
                          ("min_class_count", 1_000_000_000)):
        value = result[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    low = _number(result["multiplier_min"], "multiplier_min")
    high = _number(result["multiplier_max"], "multiplier_max")
    if not 0 < low <= 1 <= high:
        raise ValueError("Multiplier bounds must be positive and include identity")
    grid = result["multiplier_grid"]
    if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 33:
        raise ValueError("multiplier_grid must contain between 1 and 33 values")
    grid = [_number(value, "multiplier_grid") for value in grid]
    if any(not low <= value <= high for value in grid) or 1.0 not in grid:
        raise ValueError("multiplier_grid must lie within bounds and include 1")
    if len(set(grid)) != len(grid):
        raise ValueError("multiplier_grid must have unique values")
    result.update(multiplier_grid=sorted(grid), multiplier_min=low, multiplier_max=high)
    return result


def _macro_f1(actual, predicted, n_classes):
    counts = np.bincount(actual * n_classes + predicted,
                         minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    numerator = 2.0 * counts.diagonal()
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    values = np.divide(numerator, denominator, out=np.zeros(n_classes), where=denominator != 0)
    return float(values.mean())


def _simplicity(multipliers):
    # Identity has zero changed coordinates and wins every equal-score tie.
    return (sum(value != 1.0 for value in multipliers),
            sum(abs(math.log(value)) for value in multipliers), tuple(multipliers))


def _better(score, multipliers, best_score, best_multipliers):
    if score > best_score + _SCORE_TOLERANCE:
        return True
    return (abs(score - best_score) <= _SCORE_TOLERANCE
            and _simplicity(multipliers) < _simplicity(best_multipliers))


def fit_decision_rule(probabilities, calibration_labels, classes, *, options=None):
    """Fit bounded multipliers using only supplied calibration probabilities.

    ``classes`` describes probability columns in their exact order. Defaults
    need 100 calibration rows and 10 examples of every declared class. Missing
    classes or insufficient support return identity, with a diagnostic reason.
    An empty ``(0, K)`` matrix represents unavailable valid calibration and also
    returns identity. Malformed inputs always raise ``ValueError``.

    Coordinate search uses a predeclared grid, at most three rounds and 128
    unique candidate evaluations including identity by default. Options are
    explicit experiment configuration, never inferred from external scores.
    Even with overrides, hard caps are 10 rounds, 4096 candidates and 33 grid
    values. Equal scores prefer identity, then fewer changed coordinates,
    smaller absolute log distance to identity, then lexicographic multipliers.
    """
    classes = _classes(classes)
    matrix = _probabilities(probabilities, len(classes))
    labels = _strings(calibration_labels, "calibration_labels")
    if len(labels) != len(matrix):
        raise ValueError("Calibration label and probability row counts differ")
    positions = {label: index for index, label in enumerate(classes)}
    if not set(labels) <= set(classes):
        raise ValueError("Calibration labels contain undeclared classes")
    config = _options(options)
    actual = np.asarray([positions[label] for label in labels], dtype=np.int64)
    counts = np.bincount(actual, minlength=len(classes)).tolist()
    identity = tuple(1.0 for _ in classes)
    control = (_macro_f1(actual, matrix.argmax(axis=1), len(classes))
               if len(matrix) else None)
    diagnostics = {
        "n_samples": len(matrix), "class_counts": counts,
        "candidates_evaluated": int(bool(len(matrix))), "rounds_completed": 0,
        "stop_reason": "identity_fallback", "fallback_reason": None,
        "control_score": control, "selected_score": control,
    }
    rule = {"version": 1, "metric": "macro_f1", "classes": classes,
            "multipliers": list(identity), "options": config, "diagnostics": diagnostics}
    reason = None
    if not len(matrix):
        reason = "empty_calibration"
    elif min(counts) == 0:
        reason = "missing_class"
    elif len(matrix) < config["min_samples"]:
        reason = "insufficient_samples"
    elif min(counts) < config["min_class_count"]:
        reason = "insufficient_class_support"
    if reason is not None:
        diagnostics["fallback_reason"] = reason
        return rule

    cache = {identity: control}
    current, current_score = identity, control
    stop = "max_rounds"
    exhausted = False
    for round_index in range(config["max_rounds"]):
        before = current
        for column in range(len(classes)):
            best, best_score = current, current_score
            anchor = current
            for value in config["multiplier_grid"]:
                candidate = anchor[:column] + (value,) + anchor[column + 1:]
                if candidate not in cache:
                    if len(cache) >= config["max_candidates"]:
                        exhausted = True
                        break
                    predicted = (matrix * np.asarray(candidate)).argmax(axis=1)
                    cache[candidate] = _macro_f1(actual, predicted, len(classes))
                score = cache[candidate]
                if _better(score, candidate, best_score, best):
                    best, best_score = candidate, score
            current, current_score = best, best_score
            if exhausted:
                break
        if exhausted:
            stop = "candidate_budget"
            break
        diagnostics["rounds_completed"] = round_index + 1
        if current == before:
            stop = "no_improvement"
            break
    diagnostics.update(candidates_evaluated=len(cache), stop_reason=stop,
                       selected_score=current_score)
    rule["multipliers"] = list(current)
    return rule


def _finite_json(value, depth=0):
    if depth > 32:
        raise ValueError("rule JSON nesting is too deep or cyclic")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            _finite_json(child, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _finite_json(child, depth + 1)
        return
    raise ValueError("rule must contain only finite JSON-compatible data")


def _validate_rule(rule):
    if type(rule) is not dict or set(rule) != {
            "version", "metric", "classes", "multipliers", "options", "diagnostics"}:
        raise ValueError("Invalid decision-rule schema")
    _finite_json(rule)
    if type(rule["version"]) is not int or rule["version"] != 1 or rule["metric"] != "macro_f1":
        raise ValueError("Unsupported decision-rule version or metric")
    classes = _classes(rule["classes"])
    config = _options(rule["options"])
    if set(rule["options"]) != set(DEFAULT_OPTIONS):
        raise ValueError("Stored rule must include all options")
    multipliers = rule["multipliers"]
    if type(multipliers) is not list or len(multipliers) != len(classes):
        raise ValueError("Stored multipliers do not match classes")
    multipliers = [_number(value, "multipliers") for value in multipliers]
    if any(not config["multiplier_min"] <= value <= config["multiplier_max"]
           for value in multipliers):
        raise ValueError("Stored multipliers are outside positive bounds")
    diagnostics = rule["diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise ValueError("Invalid decision-rule diagnostics")
    for name in ("n_samples", "candidates_evaluated", "rounds_completed"):
        if type(diagnostics[name]) is not int or diagnostics[name] < 0:
            raise ValueError("Invalid decision-rule diagnostic counts")
    counts = diagnostics["class_counts"]
    if (type(counts) is not list or len(counts) != len(classes)
            or any(type(value) is not int or value < 0 for value in counts)
            or sum(counts) != diagnostics["n_samples"]):
        raise ValueError("Invalid stored class counts")
    if (diagnostics["candidates_evaluated"] > config["max_candidates"]
            or diagnostics["rounds_completed"] > config["max_rounds"]):
        raise ValueError("Stored diagnostic counts exceed configured limits")
    if type(diagnostics["stop_reason"]) is not str or diagnostics["stop_reason"] not in {
            "identity_fallback", "candidate_budget", "no_improvement", "max_rounds"}:
        raise ValueError("Invalid stored stopping reason")
    reason = diagnostics["fallback_reason"]
    if reason is not None and (type(reason) is not str or reason not in {
            "empty_calibration", "missing_class", "insufficient_samples",
            "insufficient_class_support"}):
        raise ValueError("Invalid stored fallback reason")
    fallback = diagnostics["fallback_reason"] is not None
    if (fallback != (diagnostics["stop_reason"] == "identity_fallback")
            or (fallback and any(value != 1.0 for value in multipliers))):
        raise ValueError("Fallback rules must use identity and an explicit reason")
    if diagnostics["n_samples"] == 0:
        if (diagnostics["fallback_reason"] != "empty_calibration"
                or diagnostics["candidates_evaluated"] != 0
                or diagnostics["control_score"] is not None
                or diagnostics["selected_score"] is not None):
            raise ValueError("Invalid empty-calibration diagnostics")
    elif diagnostics["candidates_evaluated"] < 1:
        raise ValueError("Nonempty calibration must evaluate identity")
    for name in ("control_score", "selected_score"):
        value = diagnostics[name]
        if value is None:
            if diagnostics["n_samples"] != 0:
                raise ValueError("Only empty calibration may have null scores")
        elif not 0 <= _number(value, name) <= 1:
            raise ValueError("Stored Macro F1 must lie in [0, 1]")
    return classes, np.asarray(multipliers)


def predict_with_rule(probabilities, prediction_classes, rule):
    """Return exact string labels after explicit column alignment.

    ``prediction_classes`` must describe every probability column exactly once.
    Columns are reordered to the stored rule's canonical order *before* argmax,
    so ties do not depend on the incoming column permutation. Inputs and rule
    are not mutated; the original probabilities remain available for scoring.
    """
    canonical, multipliers = _validate_rule(rule)
    supplied = _classes(prediction_classes)
    if set(supplied) != set(canonical):
        raise ValueError("Prediction classes do not match stored rule classes")
    matrix = _probabilities(probabilities, len(supplied))
    aligned = matrix[:, [supplied.index(label) for label in canonical]]
    predicted = (aligned * multipliers).argmax(axis=1)
    return np.asarray(canonical, dtype=str)[predicted]
