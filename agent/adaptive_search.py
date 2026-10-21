"""Finite, INNER-only adaptive proposals for the existing tabular evaluator.

This module receives recipes and host feedback, never data frames or answers.
The host owns training, deadlines, validation splits and admission. Resource
promotion means a fresh fit with more estimators on the SAME permitted rows;
neither warm starts nor subsampling nor an independent quality gain is claimed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
import random

from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import ACTIONS, make_model_config, validate_model_config


MAX_TRIALS = 96
MAX_DRAW_ATTEMPTS = 32
FAILURE_LIMIT = 3
PROMOTION_FACTOR = 3
OPTIMIZER_ERRORS = (ImportError, AttributeError, RuntimeError, ValueError, TypeError)
SKIP_REASONS = frozenset({
    "duplicate", "insufficient_budget", "finalization_reserve", "family_blocked",
})
FEEDBACK_REASONS = frozenset({
    "improved", "not_better", "refit_budget", "candidate_execution_failed",
    "execution_failed", "host_rejected", *SKIP_REASONS,
})


def _load_optuna():
    """Import the optional optimizer only when adaptive sampling is needed."""
    try:
        return importlib.import_module("optuna")
    except ImportError:
        return None


def _identity(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Adaptive recipe must be an ordinary model recipe")
    action = recipe.get("action")
    if action not in ACTIONS:
        raise ValueError("Adaptive recipe requires a known action")
    value = {
        "action": action,
        "model_config": validate_model_config(recipe.get("model_config"), action=action),
        "feature_config": normalize_feature_config(recipe.get("feature_config")),
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value, name, *, minimum=None):
    if (type(value) not in {int, float} or not math.isfinite(value)
            or minimum is not None and value < minimum):
        raise ValueError(name + " must be a finite number")
    return float(value)


class AdaptiveSearch:
    """Sequential, bounded TPE proposals with same-fidelity resource promotion.

    ``suggest`` is idempotent while one proposal is outstanding. Every issued
    proposal must be observed, including neutral duplicate/budget skips. Host
    warmup recipes not issued by this instance are ignored by ``observe``;
    their quality is not silently mixed with this search's fidelity levels.
    """

    def __init__(self, classification: bool, metric_name: str, *, seed=42, max_trials=64):
        if type(classification) is not bool:
            raise ValueError("classification must be a boolean")
        if not isinstance(metric_name, str):
            raise ValueError("metric_name must be a registered metric name")
        metric = get_metric(metric_name)
        if metric.classification != classification:
            raise ValueError("Adaptive task type differs from the declared metric")
        if type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32-1]")
        if type(max_trials) is not int or not 1 <= max_trials <= MAX_TRIALS:
            raise ValueError("max_trials must be an integer in [1, 96]")
        self.classification = classification
        self.task_type = metric.task_type
        self.metric = metric
        self.seed = seed
        self.max_trials = max_trials
        self._random = random.Random(seed)
        self._records = []
        self._by_identity = {}
        self._by_id = {}
        self._pending = None
        self._used = set()
        self._warmup_index = 0
        self._family_index = 0
        self._promotion_index = 0
        self._failures = {action: 0 for action in ACTIONS}
        self._disabled = set()
        self._ignored = 0
        self._duplicates = 0
        self._stop_reason = None
        self._optuna_checked = False
        self._optuna = None
        self._optimizer_error = None
        self._studies = {}
        self._handles = {}
        self._ladders = {
            "baseline": {"parameter": "n_estimators" if classification else "max_iter", "levels": [80, 240, 600]},
            "extra_trees": {"parameter": "n_estimators", "levels": [80, 240, 600]},
            "catboost": {"parameter": "iterations", "levels": [100, 400, 1200]},
            "lightgbm": {"parameter": "n_estimators", "levels": [100, 400, 1200]},
            "linear": {"parameter": None, "levels": []},
        }

    def _space(self, action):
        categorical = lambda values: ("categorical", list(values))
        integer = lambda low, high: ("int", low, high)
        number = lambda low, high, log=False: ("float", low, high, log)
        if action in {"extra_trees", "baseline"} and (action != "baseline" or self.classification):
            space = {
                "max_depth": categorical([8, 12, 18, 24, None]),
                "min_samples_leaf": integer(1, 16),
                "max_features": categorical(["sqrt", "log2", 0.5, 1.0]),
            }
            if self.classification:
                space["class_weight"] = categorical([None, "balanced", "balanced_subsample"]
                    if action == "baseline" else [None, "balanced"])
        elif action == "baseline":
            space = {
                "max_leaf_nodes": categorical([15, 31, 63, 95]),
                "min_samples_leaf": integer(10, 80),
                "learning_rate": number(.02, .2, True),
                "l2_regularization": number(0, 10),
            }
        elif action == "catboost":
            space = {"depth": integer(3, 9), "learning_rate": number(.02, .2, True),
                     "l2_leaf_reg": number(.5, 30, True)}
            if self.classification:
                space["auto_class_weights"] = categorical([None, "Balanced", "SqrtBalanced"])
        elif action == "lightgbm":
            space = {
                "num_leaves": categorical([15, 31, 63, 95]),
                "max_depth": categorical([-1, 5, 8, 12]),
                "min_child_samples": integer(10, 100),
                "learning_rate": number(.02, .2, True),
                "reg_lambda": number(.05, 30, True),
                "colsample_bytree": categorical([.7, .85, 1.0]),
            }
            if self.classification:
                space["class_weight"] = categorical([None, "balanced"])
        else:
            space = ({"C": number(.001, 100, True), "class_weight": categorical([None, "balanced"])}
                     if self.classification else {"alpha": number(.001, 1000, True)})
        space.update({"feature_missing_indicators": categorical([False, True]),
                      "feature_rare_min_count": categorical([0, 5, 20]),
                      "feature_signed_log": categorical([False, True])})
        return space

    def _values(self, recipe):
        parameters = recipe["model_config"]["parameters"]
        features = recipe["feature_config"]
        return {name: features[name[len("feature_"):]] if name.startswith("feature_") else parameters[name]
                for name in self._space(recipe["action"])}

    def _recipe(self, action, values, *, rung=0):
        parameters = {name: value for name, value in values.items() if not name.startswith("feature_")}
        ladder = self._ladders[action]
        if ladder["parameter"] is not None:
            parameters[ladder["parameter"]] = ladder["levels"][rung]
        elif self.classification:
            parameters["max_iter"] = 800
        objective = "mae" if (self.metric.name == "MAE" and action in {"baseline", "catboost", "lightgbm"}) else None
        return {"action": action, "model_config": make_model_config(action, self.task_type, parameters, objective=objective),
                "feature_config": normalize_feature_config({name[len("feature_"):]: value
                    for name, value in values.items() if name.startswith("feature_")})}

    def _warmup(self, action):
        values = {
            "baseline": ({"max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None}
                         if self.classification else {"max_leaf_nodes": 15, "min_samples_leaf": 20,
                                                     "learning_rate": .08, "l2_regularization": 1.0}),
            "extra_trees": {"max_depth": 18, "min_samples_leaf": 2, "max_features": .5},
            "catboost": {"depth": 5, "learning_rate": .08, "l2_leaf_reg": 5.0},
            "lightgbm": {"num_leaves": 31, "max_depth": -1, "min_child_samples": 30,
                         "learning_rate": .08, "reg_lambda": 3.0, "colsample_bytree": 1.0},
            "linear": {"C": .5, "class_weight": None} if self.classification else {"alpha": 3.0},
        }[action]
        if self.classification and action == "extra_trees":
            values["class_weight"] = None
        if self.classification and action == "catboost":
            values["auto_class_weights"] = None
        if self.classification and action == "lightgbm":
            values["class_weight"] = None
        values.update(feature_missing_indicators=False, feature_rare_min_count=0, feature_signed_log=False)
        return self._recipe(action, values)

    def _optimizer(self):
        if not self._optuna_checked:
            self._optuna_checked = True
            try:
                self._optuna = _load_optuna()
            except OPTIMIZER_ERRORS as error:
                self._degrade_optimizer(error)
        return self._optuna

    def _degrade_optimizer(self, error):
        # An optional in-memory sampler failure cannot undo a host-measured
        # trial. Never record arbitrary exception text or retry the same fit.
        self._optimizer_error = next(kind.__name__ for kind in OPTIMIZER_ERRORS if isinstance(error, kind))
        self._optuna_checked = True
        self._optuna = None
        self._handles.clear()

    def _distribution(self, specification):
        kind, *settings = specification
        distributions = self._optuna.distributions
        if kind == "categorical":
            return distributions.CategoricalDistribution(settings[0])
        if kind == "int":
            return distributions.IntDistribution(*settings)
        return distributions.FloatDistribution(settings[0], settings[1], log=settings[2])

    def _study(self, action):
        optuna = self._optimizer()
        if optuna is None:
            return None
        if action in self._studies:
            return self._studies[action]
        sampler = optuna.samplers.TPESampler(seed=(self.seed + ACTIONS.index(action)) % (2 ** 32),
                                            n_startup_trials=3)
        study = optuna.create_study(direction="maximize" if self.metric.higher_is_better else "minimize", sampler=sampler)
        distributions = {name: self._distribution(specification) for name, specification in self._space(action).items()}
        for record in self._records:
            if record["family"] == action and record["rung"] == 0 and record["status"] == "verified":
                study.add_trial(optuna.trial.create_trial(params=self._values(record["recipe"]),
                    distributions=distributions, value=record["metric"]["value"]))
        self._studies[action] = study
        return study

    def _sample(self, action):
        try:
            study = self._study(action)
            if study is not None:
                trial = study.ask()
                values = {}
                for name, specification in self._space(action).items():
                    kind, *settings = specification
                    if kind == "categorical":
                        values[name] = trial.suggest_categorical(name, settings[0])
                    elif kind == "int":
                        values[name] = trial.suggest_int(name, *settings)
                    else:
                        values[name] = trial.suggest_float(name, settings[0], settings[1], log=settings[2])
                return self._recipe(action, values), "tpe", trial
        except OPTIMIZER_ERRORS as error:
            self._degrade_optimizer(error)
        values = {}
        for name, specification in self._space(action).items():
            kind, *settings = specification
            if kind == "categorical":
                values[name] = self._random.choice(settings[0])
            elif kind == "int":
                values[name] = self._random.randint(*settings)
            elif settings[2]:
                values[name] = math.exp(self._random.uniform(math.log(settings[0]), math.log(settings[1])))
            else:
                values[name] = self._random.uniform(settings[0], settings[1])
        return self._recipe(action, values), "fallback", None

    def _promotion(self):
        # Each rung is ranked separately. Higher-resource scores are never fed
        # to the base-resource TPE study or mixed with low-resource rankings.
        actions = ACTIONS[self._promotion_index:] + ACTIONS[:self._promotion_index]
        for rung in (1, 0):
            for action in actions:
                if action in self._disabled or len(self._ladders[action]["levels"]) <= rung + 1:
                    continue
                records = [record for record in self._records if record["family"] == action
                           and record["rung"] == rung and record["status"] == "verified"]
                allowed = len(records) // PROMOTION_FACTOR
                if allowed <= sum(record.get("promoted", False) for record in records):
                    continue
                ranked = sorted(records, key=lambda record: ((-1 if self.metric.higher_is_better else 1)
                                * record["metric"]["value"], record["id"]))
                for record in ranked[:allowed]:
                    if record.get("promoted"):
                        continue
                    promoted = deepcopy(record["recipe"])
                    promoted["model_config"]["parameters"][self._ladders[action]["parameter"]] = self._ladders[action]["levels"][rung + 1]
                    record["promoted"] = True
                    if _identity(promoted) in self._used:
                        self._duplicates += 1
                        continue
                    self._promotion_index = (ACTIONS.index(action) + 1) % len(ACTIONS)
                    return self._issue(promoted, "promotion", rung + 1, parent=record)
        return None

    def _issue(self, recipe, origin, rung, *, parent=None, trial=None):
        identifier = "adaptive_" + str(len(self._records) + 1).zfill(4)
        recipe = deepcopy(recipe)
        recipe.update(id=identifier, label="Adaptive " + recipe["action"] + " / rung " + str(rung))
        key = _identity(recipe)
        if key in self._used:
            raise ValueError("Adaptive search attempted a duplicate executable recipe")
        ladder = self._ladders[recipe["action"]]
        record = {"id": identifier, "identity": key, "family": recipe["action"], "origin": origin,
                  "rung": rung, "resource": ladder["levels"][rung] if ladder["levels"] else None,
                  "parent_id": parent["id"] if parent else None, "recipe": recipe,
                  "status": "pending", "metric": None, "accepted": False, "reason": None,
                  "elapsed_seconds": None}
        self._records.append(record)
        self._by_identity[key] = record
        self._by_id[identifier] = record
        self._used.add(key)
        if trial is not None:
            self._handles[key] = trial
        self._pending = key
        return deepcopy(recipe)

    def suggest(self):
        if self._pending is not None:
            return deepcopy(self._by_identity[self._pending]["recipe"])
        if self._stop_reason is not None:
            return None
        if len(self._records) >= self.max_trials:
            self._stop_reason = "trial_limit"
            return None
        if len(self._disabled) == len(ACTIONS):
            self._stop_reason = "families_exhausted"
            return None
        while self._warmup_index < len(ACTIONS):
            action = ACTIONS[self._warmup_index]
            self._warmup_index += 1
            if action not in self._disabled:
                return self._issue(self._warmup(action), "warmup", 0)
        promotion = self._promotion()
        if promotion is not None:
            return promotion
        for _ in range(MAX_DRAW_ATTEMPTS):
            action = ACTIONS[self._family_index % len(ACTIONS)]
            self._family_index += 1
            if action in self._disabled:
                continue
            recipe, origin, trial = self._sample(action)
            if _identity(recipe) in self._used:
                self._duplicates += 1
                if trial is not None:
                    try:
                        self._studies[action].tell(trial, state=self._optuna.trial.TrialState.PRUNED)
                    except OPTIMIZER_ERRORS as error:
                        self._degrade_optimizer(error)
                continue
            return self._issue(recipe, origin, 0, trial=trial)
        self._stop_reason = "search_exhausted"
        return None

    def _feedback(self, feedback):
        if not isinstance(feedback, dict):
            raise ValueError("Adaptive feedback must be a host object")
        verified, accepted = feedback.get("verified", False), feedback.get("accepted", False)
        if type(verified) is not bool or type(accepted) is not bool or accepted and not verified:
            raise ValueError("Adaptive feedback requires boolean verified/accepted flags")
        if feedback.get("scope", "inner_selection") != "inner_selection":
            raise ValueError("Adaptive search accepts INNER selection feedback only")
        supplied = feedback.get("metric")
        metric = None
        if supplied is not None:
            if (not isinstance(supplied, dict) or supplied.get("name") != self.metric.name
                    or type(supplied.get("higher_is_better")) is not bool
                    or supplied["higher_is_better"] != self.metric.higher_is_better):
                raise ValueError("Adaptive metric identity/direction differs from the declared task")
            metric = {"name": self.metric.name, "higher_is_better": self.metric.higher_is_better,
                      "value": _finite(supplied.get("value"), "Metric value")}
        if verified and metric is None:
            raise ValueError("A verified adaptive observation requires a host metric")
        elapsed = feedback.get("elapsed_seconds")
        if elapsed is not None:
            elapsed = _finite(elapsed, "Elapsed seconds", minimum=0)
        reason = feedback.get("reason")
        reason = reason if isinstance(reason, str) and reason in FEEDBACK_REASONS else "host_rejected"
        return verified, accepted, metric, elapsed, reason

    def observe(self, recipe, feedback):
        identifier = recipe.get("id") if isinstance(recipe, dict) else None
        known_id = isinstance(identifier, str) and identifier in self._by_id
        try:
            key = _identity(recipe)
        except (ValueError, TypeError):
            if known_id:
                raise ValueError("An issued adaptive recipe changed its executable configuration") from None
            self._ignored += 1
            return False
        if known_id and self._by_id[identifier]["identity"] != key:
            raise ValueError("An issued adaptive recipe changed its executable configuration")
        record = self._by_identity.get(key)
        if record is None:
            self._ignored += 1
            return False
        if key != self._pending or record["status"] != "pending":
            raise ValueError("Adaptive observation is stale or already recorded")
        verified, accepted, metric, elapsed, reason = self._feedback(feedback)
        action = record["family"]
        record.update(status="verified" if verified else "skipped" if reason in SKIP_REASONS else "failed",
                      metric=metric if verified else None, accepted=accepted, reason=reason, elapsed_seconds=elapsed)
        if verified:
            self._failures[action] = 0
        elif reason == "family_blocked":
            self._disabled.add(action)
        elif reason not in SKIP_REASONS:
            self._failures[action] += 1
            if self._failures[action] >= FAILURE_LIMIT:
                self._disabled.add(action)
        handle = self._handles.pop(key, None)
        if handle is not None:
            try:
                if verified:
                    self._studies[action].tell(handle, metric["value"])
                else:
                    state = self._optuna.trial.TrialState.PRUNED if reason in SKIP_REASONS else self._optuna.trial.TrialState.FAIL
                    self._studies[action].tell(handle, state=state)
            except OPTIMIZER_ERRORS as error:
                self._degrade_optimizer(error)
        self._pending = None
        if len(self._records) >= self.max_trials:
            self._stop_reason = "trial_limit"
        elif len(self._disabled) == len(ACTIONS):
            self._stop_reason = "families_exhausted"
        return True

    def report(self):
        statuses = [record["status"] for record in self._records]
        return deepcopy({"version": 1, "policy": "adaptive_multifidelity_v1", "task_type": self.task_type,
            "metric": self.metric.to_dict(), "seed": self.seed, "max_trials": self.max_trials,
            "proposed": len(self._records), "observed": sum(status != "pending" for status in statuses),
            "verified": statuses.count("verified"), "failed": statuses.count("failed"), "skipped": statuses.count("skipped"),
            "ignored_external_observations": self._ignored, "duplicates": self._duplicates,
            "pending": self._by_identity[self._pending]["id"] if self._pending else None,
            "stop_reason": self._stop_reason, "family_failures": self._failures,
            "disabled_families": sorted(self._disabled), "resource_ladders": self._ladders,
            "optimizer": {"backend": "optuna_tpe" if self._optuna is not None else
                          "deterministic_random" if self._optuna_checked else "pending",
                          "available": self._optuna is not None if self._optuna_checked else None,
                          "error_type": self._optimizer_error, "studies": sorted(self._studies),
                          "learning_scope": "verified_same_family_base_rung_only"},
            "trials": self._records})
