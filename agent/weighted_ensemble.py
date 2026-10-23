"""Small, deterministic ensembles learned exclusively on inner predictions.

Callers must pass predictions from models fitted without these inner targets,
then freeze the returned component IDs and weights *before* outer verification.
This module never fits a model and never receives outer validation outcomes.
"""

from __future__ import annotations

import math
import time

import numpy as np

from agent.metrics import score_predictions
from agent.search import align_predictions, metric_predictions


def _weights(values, size):
    if (not isinstance(values, (list, tuple)) or len(values) != size
            or not 1 <= size <= 3
            or any(type(value) not in {int, float} or not math.isfinite(value)
                   or value <= 0 or value > 1 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Ensemble requires one to three positive weights summing to one")
    return np.asarray(values, dtype=float) / sum(values)


def apply_weighted_ensemble(spec, predictions, weights, *, prediction_classes=None):
    """Apply already frozen weights to class-aligned probabilities or values."""
    if not isinstance(predictions, (list, tuple)):
        raise ValueError("Ensemble predictions must be a list of component arrays")
    weights = _weights(weights, len(predictions))
    classes = ([spec.classes] * len(predictions) if prediction_classes is None
               else prediction_classes)
    if len(classes) != len(predictions):
        raise ValueError("Class orders must match the number of ensemble components")
    arrays = [align_predictions(spec, value, order) for value, order in zip(predictions, classes)]
    if not len(arrays[0]) or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Ensemble components must have the same nonempty prediction shape")
    result = np.zeros_like(arrays[0], dtype=float)
    for value, weight in zip(arrays, weights):
        result += value * weight
    # Remove floating-point drift in probability rows, without accepting
    # malformed probabilities from an individual component.
    if spec.classification:
        result /= result.sum(axis=1, keepdims=True)
    return result


def _grid(size):
    # All simplex points in tenths: 9 candidates for two components, or
    # 63 for three (including their pairs). Vertices are scored separately.
    if size == 2:
        for first in range(1, 10):
            yield (first / 10, (10 - first) / 10)
    else:
        for first in range(11):
            for second in range(11 - first):
                third = 10 - first - second
                if sum(value > 0 for value in (first, second, third)) >= 2:
                    yield (first / 10, second / 10, third / 10)


def select_weighted_ensemble(spec, actual, candidates, *, deadline=None):
    """Choose at most one blend on a fixed INNER holdout, or the best single.

    ``candidates`` is an inner-frozen shortlist of two or three dictionaries:
    ``id``, ``predictions`` and optional ``classes``. All prediction rows must
    follow exactly the same inner row order. The host checks IDs/targets while
    loading each raw prediction bundle. A strict improvement is required;
    ties keep the cheaper single model, then the first deterministic blend.
    The optional monotonic deadline stops further weight evaluations. At most
    three single scores are always computed to retain a valid fallback.
    """
    if (not isinstance(candidates, (list, tuple)) or not 2 <= len(candidates) <= 3
            or any(not isinstance(item, dict) or not {"id", "predictions"} <= set(item)
                   or not isinstance(item["id"], str) or not item["id"] for item in candidates)):
        raise ValueError("Weighted selection requires two or three identified inner candidates")
    if len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("Weighted selection candidate IDs must be unique")
    if deadline is not None and (type(deadline) not in {int, float} or not math.isfinite(deadline)):
        raise ValueError("Weighted selection deadline must be finite")
    arrays = [align_predictions(spec, item["predictions"], item.get("classes", spec.classes))
              for item in candidates]
    if not len(arrays[0]) or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Inner prediction arrays must have the same nonempty shape")

    def score(values):
        return score_predictions(spec.metric, actual, metric_predictions(spec, values),
                                 classes=spec.classes, prediction_classes=spec.classes,
                                 positive_class=spec.positive_class)

    def better(left, right):
        margin = 1e-12 * max(1, abs(left), abs(right))
        return left > right + margin if spec.metric.higher_is_better else left < right - margin

    scores = [score(value) for value in arrays]
    best_index = min(range(len(scores)), key=lambda index: -scores[index]
                     if spec.metric.higher_is_better else scores[index])
    best_value = scores[best_index]
    best_weights = [float(index == best_index) for index in range(len(arrays))]
    attempted = 0
    reason = "grid_exhausted"
    # Prefer a two-model blend on equal scores before paying for a third fit.
    grid = sorted(_grid(len(arrays)), key=lambda weights: sum(weight > 0 for weight in weights))
    for weights in grid:
        if deadline is not None and time.monotonic() >= deadline:
            reason = "time_limit"
            break
        values = np.zeros_like(arrays[0], dtype=float)
        for array, weight in zip(arrays, weights):
            if weight:
                values += array * weight
        if spec.classification:
            values /= values.sum(axis=1, keepdims=True)
        value = score(values)
        attempted += 1
        if better(value, best_value):
            best_value, best_weights = value, list(weights)
    selected = [index for index, weight in enumerate(best_weights) if weight > 0]
    return {
        "fitted_on": "inner_selection", "strategy": "simplex_grid_10",
        "candidate_ids": [item["id"] for item in candidates],
        "selected_ids": [candidates[index]["id"] for index in selected],
        "weights": [best_weights[index] for index in selected],
        "score": best_value, "best_single_id": candidates[best_index]["id"],
        "best_single_score": scores[best_index], "improved": len(selected) > 1,
        "evaluated_weights": attempted, "stop_reason": reason,
    }
