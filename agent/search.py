"""Deterministic search spaces, nested splits and class-aligned fixed blending.

Adaptive choices use the inner holdout only. The outer holdout is reserved for
the short list and a single, predetermined 50/50 blend; its weights are not fit.
"""

from __future__ import annotations

import json

import numpy as np

from agent.features import normalize_feature_config
from agent.metrics import probability_matrix
from agent.model_config import make_model_config
from agent.splits import split_rows


def inner_split_rows(train, y, features, classification, choice, outer_fit):
    """Return global row positions, using only the frozen outer training rows.

    The same time/group policy is applied one level deeper. An impossible inner
    split raises ValueError: callers must keep the untuned seed, not tune on the
    final holdout as a silent fallback.
    """
    outer_fit = np.asarray(outer_fit, dtype=int)
    fitting, held, metadata = split_rows(
        train.iloc[outer_fit].reset_index(drop=True), y.iloc[outer_fit].reset_index(drop=True),
        features, classification, choice,
    )
    if classification:
        expected = set(map(str, y))
        if (set(map(str, y.iloc[outer_fit[fitting]])) != expected
                or set(map(str, y.iloc[outer_fit[held]])) != expected):
            raise ValueError("Inner validation must contain every class in both partitions")
    return outer_fit[fitting], outer_fit[held], metadata


def _candidate(identifier, action, task_type, parameters, label):
    return {"id": identifier, "action": action, "label": label,
            "model_config": make_model_config(action, task_type, parameters),
            "feature_config": normalize_feature_config()}


def seed_candidate(classification):
    task_type = "classification" if classification else "regression"
    parameters = {"max_iter": 100, "class_weight": "balanced"} if classification else {"alpha": 1.0}
    return _candidate("seed", "linear", task_type, parameters, "Быстрый старт · линейная модель")


def exploration_plan(classification):
    task_type = "classification" if classification else "regression"
    variants = [
        ("baseline", None, "Стартовая модель"),
        ("extra_trees", None, "ExtraTrees"),
        ("extra_trees", {"n_estimators": 200, "min_samples_leaf": 1}, "ExtraTrees · малые листья"),
        ("catboost", {"iterations": 100}, "CatBoost · быстрый поиск"),
        ("catboost", {"iterations": 100, "depth": 8, **({"auto_class_weights": "SqrtBalanced"} if classification else {})},
         "CatBoost · глубина 8"),
        ("lightgbm", {"n_estimators": 100}, "LightGBM · быстрый поиск"),
        ("lightgbm", {"n_estimators": 100, "num_leaves": 63, **({"class_weight": "balanced"} if classification else {})},
         "LightGBM · 63 листа"),
        ("linear", None, "Линейная модель"),
    ]
    result = [_candidate(f"candidate_{index}", action, task_type, parameters, label)
              for index, (action, parameters, label) in enumerate(variants, 1)]
    result[4]["feature_config"] = normalize_feature_config({"missing_indicators": True, "rare_min_count": 5})
    result[7]["feature_config"] = normalize_feature_config({"missing_indicators": True, "signed_log": True})
    return result


def refinement_plan(classification, ranked_candidates, existing=()):
    """Explore three new settings for each of the two best distinct families.

    ``ranked_candidates`` is sorted by the independently recomputed INNER score.
    This bounded adaptive step never consumes outer scores or adjusts blend
    weights. Canonical configuration deduplication also covers custom plans.
    """
    task_type = "classification" if classification else "regression"
    spaces = {
        "baseline": [
            {"n_estimators": 160, "max_depth": 24, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 4},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        ] if classification else [
            {"max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 1.0},
            {"max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 1.0},
            {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.05},
        ],
        "extra_trees": [
            {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.5},
            {"n_estimators": 250, "min_samples_leaf": 4},
            {"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 2},
        ],
        "catboost": [
            {"iterations": 400, "depth": 4, "learning_rate": 0.08},
            {"iterations": 500, "depth": 8, "l2_leaf_reg": 10.0},
            {"iterations": 400, "depth": 10, "learning_rate": 0.03},
        ],
        "lightgbm": [
            {"n_estimators": 400, "num_leaves": 15, "reg_lambda": 3.0},
            {"n_estimators": 500, "num_leaves": 63, "min_child_samples": 40},
            {"n_estimators": 400, "num_leaves": 127, "learning_rate": 0.03},
        ],
        "linear": ([{"C": value, "max_iter": 500} for value in (0.1, 10.0, 100.0)]
                   if classification else [{"alpha": value} for value in (0.1, 10.0, 100.0)]),
    }
    def identity(item):
        return json.dumps([item["model_config"], normalize_feature_config(item.get("feature_config"))], sort_keys=True)
    seen = {identity(item) for item in existing}
    families, result, features, bases = [], [], {}, {}
    for item in ranked_candidates:
        action = item["action"]
        if action not in families:
            families.append(action)
            features[action] = normalize_feature_config(item.get("feature_config"))
            bases[action] = item["model_config"]["parameters"]
        if len(families) == 2:
            break
    for action in families:
        for parameters in spaces[action]:
            model_config = make_model_config(action, task_type, {**bases[action], **parameters})
            key = identity({"model_config": model_config, "feature_config": features[action]})
            if key in seen:
                continue
            seen.add(key)
            result.append({"id": f"refine_{len(result) + 1}", "action": action,
                           "label": f"Уточнение {model_config['model']} · {len(result) + 1}",
                           "model_config": model_config, "feature_config": features[action]})
    return result


def select_shortlist(ranked_candidates, limit=3):
    """Give distinct model families the bounded outer verification slots first.

    Input order is determined exclusively by inner scores. Select the best
    member of each distinct action family, then fill spare slots from that same
    ranking only when fewer families succeeded. No family receives a hardcoded
    preference and no outer scores enter this decision.
    """
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("Outer verification shortlist must contain at most three candidates")
    ranked_candidates = list(ranked_candidates)
    selected, families, identifiers = [], set(), set()
    for candidate in ranked_candidates:
        if candidate["action"] in families or candidate["id"] in identifiers:
            continue
        selected.append(candidate)
        families.add(candidate["action"])
        identifiers.add(candidate["id"])
        if len(selected) == limit:
            return selected
    for candidate in ranked_candidates:
        if candidate["id"] not in identifiers:
            selected.append(candidate)
            identifiers.add(candidate["id"])
        if len(selected) == limit:
            break
    return selected


def align_predictions(spec, predictions, prediction_classes=()):
    """Canonical raw values for blending: regression vector or full probabilities."""
    if spec.classification:
        values = np.asarray(predictions)
        if (values.ndim != 2 or values.shape[1] != len(spec.classes)
                or len(prediction_classes) != len(spec.classes)
                or set(map(str, prediction_classes)) != set(spec.classes)):
            raise ValueError("Blend components must provide every class probability")
        return probability_matrix(predictions, classes=spec.classes,
                                  prediction_classes=prediction_classes,
                                  positive_class=spec.positive_class)
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Regression blend predictions must be a finite vector")
    return values


def metric_predictions(spec, aligned):
    values = np.asarray(aligned)
    if spec.classification and spec.prediction_kind == "labels":
        return np.asarray(spec.classes)[values.argmax(axis=1)]
    return values


def fixed_blend(spec, left, right, *, left_classes=(), right_classes=()):
    left = align_predictions(spec, left, left_classes)
    right = align_predictions(spec, right, right_classes)
    if left.shape != right.shape:
        raise ValueError("Blend components have different prediction shapes")
    return (left + right) * 0.5
