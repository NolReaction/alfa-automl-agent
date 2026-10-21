"""Opt-in longer CPU training recipes for the existing isolated AutoML worker.

This is a search catalogue, not a second execution or publication engine. The
caller must retain the existing nested validation, process deadlines and full
refit/checkpoint rules. Models are fitted from scratch; no pretrained weights,
network calls or dataset-specific column names occur here.
"""

from __future__ import annotations

from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import make_model_config


PROFILE = "strong_v1"


def strong_candidate_plan(classification: bool, metric_name: str, y_fit=None) -> list[dict]:
    """Return detached ordinary recipes, ordered before any measured outcomes.

    ``y_fit`` is an optional integration seam for future fit-only diagnostics;
    version 1 deliberately does not inspect targets. In particular it does not
    switch to log targets based on positive skew when the metric is RMSE.
    Balanced classification is a separate candidate only for Macro F1, never a
    forced change to probability objectives or to the evaluation metric.
    """
    if type(classification) is not bool:
        raise ValueError("classification must be boolean")
    metric = get_metric(metric_name)
    if metric.classification != classification:
        raise ValueError("Task type differs from the declared metric")
    task_type = metric.task_type

    def candidate(identifier, action, parameters=None, *, balanced=False):
        options = dict(parameters or {})
        if balanced:
            options["class_weight"] = "balanced"
        quality = {"objective": "mae"} if metric.name == "MAE" and action in {"baseline", "catboost", "lightgbm"} else {}
        return {"id": "strong_" + identifier, "action": action,
                "label": "Strong CPU · " + identifier,
                "model_config": make_model_config(action, task_type, options, **quality),
                "feature_config": normalize_feature_config()}

    # Keep a fast, independently competitive reference. Eight-core workers run
    # sequentially; missing optional packages are rejected by the existing host.
    result = [
        candidate("reference", "baseline"),
        candidate("lgbm_800", "lightgbm", {
            "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 20, "reg_lambda": 3.0}),
        candidate("cat_1200", "catboost", {
            "iterations": 1200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0}),
        candidate("lgbm_1600_regularized", "lightgbm", {
            "n_estimators": 1600, "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 60, "reg_lambda": 10.0, "colsample_bytree": 0.8}),
        candidate("trees_300", "extra_trees", {
            "n_estimators": 300, "min_samples_leaf": 2, "max_features": 1.0}),
        candidate("lgbm_1200_leaves63", "lightgbm", {
            "n_estimators": 1200, "learning_rate": 0.04, "num_leaves": 63,
            "min_child_samples": 30, "reg_lambda": 5.0}),
        candidate("cat_1600_depth8", "catboost", {
            "iterations": 1600, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 10.0}),
    ]
    if metric.name == "Macro F1":
        result.append(candidate("lgbm_1000_balanced", "lightgbm", {
            "n_estimators": 1000, "learning_rate": 0.04, "num_leaves": 31,
            "min_child_samples": 20, "reg_lambda": 5.0}, balanced=True))
    else:
        result.append(candidate("lgbm_1000_shallow", "lightgbm", {
            "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 15,
            "min_child_samples": 40, "reg_lambda": 10.0}))
    return result
