"""Standalone, frozen v1 recipes; inactive until explicitly integrated into AutoML."""

from __future__ import annotations

from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import ACTIONS, make_model_config


def portfolio_candidates(
    profile: dict,
    *,
    available_actions: tuple[str, ...],
    limit: int = 8,
) -> list[dict]:
    """Return deterministic, detached recipes without data access or model fitting."""
    if not isinstance(profile, dict) or set(profile) != {"task_type", "metric"}:
        raise ValueError("profile must contain exactly task_type and metric")
    task_type, metric = profile["task_type"], profile["metric"]
    if not isinstance(task_type, str) or task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    if not isinstance(metric, str):
        raise ValueError("metric must be a supported metric name")
    if get_metric(metric).task_type != task_type:
        raise ValueError("metric does not match task_type")
    if (not isinstance(available_actions, tuple)
            or any(not isinstance(action, str) or action not in ACTIONS
                   for action in available_actions)
            or len(set(available_actions)) != len(available_actions)):
        raise ValueError("available_actions must be a tuple of unique known actions")
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("limit must be an integer from 1 to 8")
    if not available_actions:
        return []

    classification = task_type == "classification"
    linear = ({"C": 0.1, "max_iter": 1000, "class_weight": None}
              if classification else {"alpha": 10})
    baseline = ({"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 4,
                 "max_features": 0.5, "class_weight": "balanced_subsample"}
                if classification else
                {"max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                 "learning_rate": 0.05, "l2_regularization": 5})
    extra_trees = {"n_estimators": 300, "max_depth": None,
                   "min_samples_leaf": 4, "max_features": 0.5}
    if classification:
        extra_trees["class_weight"] = "balanced"
    conservative_lgbm = {
        "n_estimators": 400, "num_leaves": 15, "max_depth": 5,
        "min_child_samples": 40, "learning_rate": 0.05,
        "reg_lambda": 10, "colsample_bytree": 0.8,
    }
    recipes = (
        ("linear_regularized", "linear", "Линейная модель · регуляризация", linear, None),
        ("baseline_conservative", "baseline", "Стартовая модель · консервативная", baseline, None),
        ("et_regularized", "extra_trees", "ExtraTrees · регуляризация", extra_trees, None),
        ("cb_shallow", "catboost", "CatBoost · небольшая глубина",
         {"iterations": 500, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 10}, None),
        ("cb_medium", "catboost", "CatBoost · средняя глубина",
         {"iterations": 500, "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 20}, None),
        ("lgbm_conservative", "lightgbm", "LightGBM · консервативный", conservative_lgbm, None),
        ("lgbm_medium", "lightgbm", "LightGBM · средняя сложность",
         {"n_estimators": 500, "num_leaves": 63, "max_depth": 8,
          "min_child_samples": 60, "learning_rate": 0.03,
          "reg_lambda": 10, "colsample_bytree": 0.8}, None),
        ("lgbm_representation", "lightgbm", "LightGBM · пропуски и редкие категории",
         conservative_lgbm, {"missing_indicators": True, "rare_min_count": 10}),
    )
    return [
        {"id": "portfolio_v1_" + name, "action": action, "label": label,
         "model_config": make_model_config(action, task_type, parameters),
         "feature_config": normalize_feature_config(features)}
        for name, action, label, parameters, features in recipes
        if action in available_actions
    ][:limit]
