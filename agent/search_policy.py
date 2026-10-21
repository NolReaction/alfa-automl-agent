"""Finite, metric-aware recipes for the opt-in adaptive search policy.

This module does not inspect paths, environment, package availability, outer
answers or timings. The host supplies the available actions and admits each
recipe against its measured remaining budget. Ordering is a hypothesis tested
by inner selection, never an assertion that a family or objective is better.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import islice

from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import ACTIONS, make_model_config, model_task_type, validate_model_config
from agent.portfolio_recipes import portfolio_candidates
from agent.search import exploration_plan, metric_candidates, recipe_identity, refinement_plan


SEARCH_POLICIES = ("legacy", "adaptive_v1")
MAX_EXPLORATION_CANDIDATES = 12
MAX_REFINEMENT_CANDIDATES = 4
MAX_INPUT_CANDIDATES = 256


def normalize_search_policy(value="legacy"):
    if not isinstance(value, str) or value not in SEARCH_POLICIES:
        raise ValueError("search_policy must be legacy or adaptive_v1")
    return value


def _profile(classification, metric_name, available_actions):
    if type(classification) is not bool:
        raise ValueError("classification must be a boolean")
    if not isinstance(metric_name, str):
        raise ValueError("metric_name must be a supported metric name")
    metric = get_metric(metric_name)
    if metric.classification != classification:
        raise ValueError("Metric and task type disagree")
    if (not isinstance(available_actions, tuple)
            or any(not isinstance(action, str) or action not in ACTIONS for action in available_actions)
            or len(set(available_actions)) != len(available_actions)):
        raise ValueError("available_actions must be a tuple of unique known actions")
    return metric


def _checked(recipe, task_type):
    if not isinstance(recipe, dict) or recipe.get("action") not in ACTIONS:
        raise ValueError("A candidate must declare a known action")
    config = validate_model_config(recipe.get("model_config"), action=recipe["action"])
    if model_task_type(config) != task_type:
        raise ValueError("Candidate and task type disagree")
    label = recipe.get("label", config["model"])
    if not isinstance(label, str):
        raise ValueError("Candidate label must be a string")
    # Ranking metadata is host-owned and not part of the executable recipe.
    return {"action": recipe["action"], "model_config": config,
            "feature_config": normalize_feature_config(recipe.get("feature_config")),
            "label": label}


def _bounded(recipes, task_type):
    if isinstance(recipes, (str, bytes, dict)):
        raise ValueError("Candidates must be an iterable of recipe objects")
    try:
        values = list(islice(iter(recipes), MAX_INPUT_CANDIDATES + 1))
    except TypeError as error:
        raise ValueError("Candidates must be an iterable of recipe objects") from error
    if len(values) > MAX_INPUT_CANDIDATES:
        raise ValueError("Too many candidate recipes")
    return [_checked(item, task_type) for item in values]


def _unique(recipes, task_type, *, available_actions, prefix, limit, existing=()):
    seen = {recipe_identity(item) for item in existing}
    result = []
    for recipe in recipes:
        item = _checked(recipe, task_type)
        if item["action"] not in available_actions:
            continue
        key = recipe_identity(item)
        if key in seen:
            continue
        seen.add(key)
        item["id"] = f"{prefix}_{len(result) + 1}"
        result.append(deepcopy(item))
        if len(result) == limit:
            break
    return result


def exploration_candidates(classification, metric_name, y_fit, *, available_actions=ACTIONS):
    """Cover distinct families before expensive variants, using fit targets only.

    Proper MAE objectives are tested early, alongside an ordinary squared-loss
    control later. Nonnegative skew may propose the existing log1p alternatives;
    scoring and inverse transformation remain the normal host's responsibility.
    Classification never consumes ``y_fit`` or infers class weights from it.
    """
    metric = _profile(classification, metric_name, available_actions)
    if not available_actions:
        return []
    task_type = metric.task_type
    original = exploration_plan(classification)
    first = {}
    for item in original:
        first.setdefault(item["action"], item)
    first["linear"] = {
        "action": "linear", "label": "Plain linear control",
        "model_config": make_model_config("linear", task_type),
        "feature_config": normalize_feature_config(),
    }
    metric_options = metric_candidates(metric.name, y_fit) if not classification else []
    if not classification and metric.name == "MAE":
        for item in metric_options:
            if item["model_config"].get("objective") == "mae":
                first[item["action"]] = item
        order = ("baseline", "linear", "lightgbm", "extra_trees", "catboost")
    elif not classification:
        order = ("linear", "baseline", "extra_trees", "lightgbm", "catboost")
    elif metric.prediction_kind == "probabilities":
        order = ("linear", "baseline", "lightgbm", "extra_trees", "catboost")
    else:
        order = ("baseline", "linear", "lightgbm", "extra_trees", "catboost")
    queue = [first[action] for action in order]
    if not classification and metric.name == "MAE":
        queue.append(original[0])
    queue.extend(item for action in ("linear", "lightgbm", "catboost")
                 for item in metric_options
                 if item["action"] == action
                 and item["model_config"].get("target_transform") == "log1p")
    # Only the first conservative recipe of each family is eligible. In
    # particular, the 500-iteration medium CatBoost does not follow the shallow
    # variant. The host may skip any late variant using measured cost evidence.
    conservative = {}
    for item in portfolio_candidates({"task_type": task_type, "metric": metric.name},
                                     available_actions=available_actions):
        conservative.setdefault(item["action"], item)
    queue.extend(conservative[action] for action in order if action in conservative)
    return _unique(queue, task_type, available_actions=available_actions, prefix="aq",
                   limit=MAX_EXPLORATION_CANDIDATES)


def refinement_candidates(classification, metric_name, ranked_candidates, *, existing=(),
                          available_actions=ACTIONS):
    """At most two refinements for each of the top two available inner families.

    The host owns the supplied ranking. Existing objective, target transform
    and feature settings are inherited from each family's best inner recipe.
    Families alternate, so a second setting cannot precede the other family's
    first opportunity. No outer score or audit result is accepted here.
    """
    metric = _profile(classification, metric_name, available_actions)
    if not available_actions:
        return []
    ranked = _bounded(ranked_candidates, metric.task_type)
    previous = _bounded(existing, metric.task_type)
    families = {}
    for item in ranked:
        if item["action"] in available_actions:
            families.setdefault(item["action"], item)
        if len(families) == 2:
            break
    alternatives = [refinement_plan(classification, [item], existing=previous)
                    for item in families.values()]
    queue = [items[index] for index in range(2) for items in alternatives if len(items) > index]
    return _unique(queue, metric.task_type, available_actions=available_actions, prefix="ar",
                   limit=MAX_REFINEMENT_CANDIDATES, existing=previous)
