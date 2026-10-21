"""Pure, bounded scheduling heuristics for host-authored INNER trials.

The caller owns the clock, split provenance and training. Family timing may
guide ordering, but is not a replacement for exact-recipe BudgetPolicy evidence
or an absolute child-process deadline. No score is accepted by ``observe``.
"""

from __future__ import annotations

from copy import deepcopy
import math

from agent.model_config import validate_model_config
from agent.search import recipe_identity


MAX_TRIALS = 12
MAX_RECIPES = 256
MAX_DECISIONS = 64
MAX_ESTIMATE_SECONDS = 1_000_000_000_000.0
MAX_ROW_SCALE = 1_000_000.0
UNKNOWN_TRIAL_SECONDS = 5.0
MIN_TRIAL_SECONDS = 2.0


def _count(value, name, *, minimum=1, maximum=1_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _seconds(value, name, *, positive=False):
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must have a finite numeric representation") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _bounded_product(*values):
    result = 1.0
    for value in values:
        if value >= MAX_ESTIMATE_SECONDS / result:
            return MAX_ESTIMATE_SECONDS
        result *= value
    return min(MAX_ESTIMATE_SECONDS, result)


def _recipe(value):
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("A scheduled recipe must declare its action and model_config")
    checked = validate_model_config(value.get("model_config"), action=value["action"])
    # Validation and normalization happen without changing the caller's object.
    canonical = {**value, "model_config": checked}
    key = recipe_identity(canonical)
    parameters = checked["parameters"]
    iterations = next((parameters[name] for name in ("n_estimators", "max_iter", "iterations")
                       if name in parameters), 1)
    depth = parameters.get("depth", parameters.get("max_depth"))
    if depth is None or depth == -1:
        leaves = parameters.get("max_leaf_nodes", parameters.get("num_leaves"))
        depth = max(1, math.ceil(math.log2(leaves))) if leaves else 10
    # Depth is a coarse relative work indicator, never an exponential guarantee.
    depth = min(16, max(1, depth)) if iterations != 1 else 1
    return {"key": key, "family": value["action"], "work_units": float(iterations * depth)}


def _sequence(values, name):
    if not isinstance(values, (list, tuple)) or len(values) > MAX_RECIPES:
        raise ValueError(f"{name} must be a bounded list or tuple of at most {MAX_RECIPES} recipes")
    return values


class AdaptiveSearchSchedule:
    """Choose finite candidate indices using caller-supplied remaining time.

    ``choose`` records a dispatch immediately. ``observe`` can also preload
    previous INNER host measurements; these suppress duplicate trials but do
    not consume this scheduler's dispatch limit. Observations contain no metric.
    ``ranked`` is an ordered list produced exclusively from INNER scores by the
    host; the scheduler never receives the scores or final holdout answers.
    """

    def __init__(self, *, fit_rows, predict_rows, outer_fit_rows, outer_predict_rows,
                 full_fit_rows, full_predict_rows, max_trials=MAX_TRIALS):
        self.fit_rows = _count(fit_rows, "fit_rows")
        self.predict_rows = _count(predict_rows, "predict_rows", minimum=0)
        self.outer_fit_rows = _count(outer_fit_rows, "outer_fit_rows")
        self.outer_predict_rows = _count(outer_predict_rows, "outer_predict_rows", minimum=0)
        self.full_fit_rows = _count(full_fit_rows, "full_fit_rows")
        self.full_predict_rows = _count(full_predict_rows, "full_predict_rows", minimum=0)
        if not self.fit_rows <= self.outer_fit_rows <= self.full_fit_rows:
            raise ValueError("fit_rows must not exceed outer_fit_rows or full_fit_rows")
        self.max_trials = _count(max_trials, "max_trials", maximum=MAX_TRIALS)
        self._observations = {}
        self._attempted = set()
        self._families = set()
        self._dispatched = 0
        self._decisions = []

    def observe(self, recipe, elapsed_seconds, completed=True):
        item = _recipe(recipe)
        elapsed = _seconds(elapsed_seconds, "elapsed_seconds", positive=True)
        if type(completed) is not bool:
            raise ValueError("completed must be a boolean")
        if item["key"] in self._observations:
            raise ValueError("A recipe already has an INNER timing observation")
        if len(self._observations) >= MAX_RECIPES:
            raise ValueError("Too many scheduler timing observations")
        self._observations[item["key"]] = {**item, "elapsed_seconds": elapsed, "completed": completed}
        self._attempted.add(item["key"])
        self._families.add(item["family"])

    def _trial_estimate(self, item):
        same = [entry for entry in self._observations.values() if entry["family"] == item["family"]]
        if not same:
            return UNKNOWN_TRIAL_SECONDS, "unmeasured_probe"
        estimates = []
        for entry in same:
            relative_work = min(16.0, max(1 / 16, item["work_units"] / entry["work_units"]))
            elapsed = entry["elapsed_seconds"]
            if not entry["completed"]:
                # A fast failed import/fit is not evidence that this family
                # trains quickly. A timeout is only a lower bound on its cost.
                elapsed = max(UNKNOWN_TRIAL_SECONDS, _bounded_product(elapsed, 2.0))
            estimates.append(_bounded_product(elapsed, relative_work))
        return max(MIN_TRIAL_SECONDS, max(estimates)), "family_host_wall_heuristic"

    def _row_scale(self, fitting, predicting):
        ratio = max(1.0, fitting / self.fit_rows, predicting / max(1, self.predict_rows))
        return min(MAX_ROW_SCALE, ratio ** 1.25)

    def _finalization_estimate(self, trial_seconds):
        outer = _bounded_product(trial_seconds, 1.5,
                                self._row_scale(self.outer_fit_rows, self.outer_predict_rows))
        refit = _bounded_product(trial_seconds, 1.5,
                                self._row_scale(self.full_fit_rows, self.full_predict_rows))
        return min(MAX_ESTIMATE_SECONDS, outer + refit + 2.0)

    def _record_decision(self, value):
        result = {"kind": "heuristic", "guaranteed": False, **value}
        self._decisions.append(deepcopy(result))
        self._decisions = self._decisions[-MAX_DECISIONS:]
        return result

    def choose(self, candidates, *, selection_remaining, total_remaining, ranked=()):
        selection = _seconds(selection_remaining, "selection_remaining")
        total = _seconds(total_remaining, "total_remaining")
        values = _sequence(candidates, "candidates")
        rankings = _sequence(ranked, "ranked")
        family_ranks = {}
        for value in rankings:
            family = _recipe(value)["family"]
            family_ranks.setdefault(family, len(family_ranks))
        pending, seen = [], set()
        for index, value in enumerate(values):
            item = _recipe(value)
            if item["key"] not in seen and item["key"] not in self._attempted:
                pending.append({**item, "index": index})
            seen.add(item["key"])
        base = {"index": None, "timeout": 0.0, "estimated_trial_seconds": 0.0,
                "estimated_finalization_seconds": 0.0, "selection_remaining": selection,
                "total_remaining": total, "pending": len(pending)}
        if self._dispatched >= self.max_trials:
            return self._record_decision({**base, "reason": "trial_limit"})
        if not pending:
            return self._record_decision({**base, "reason": "queue_exhausted"})
        untried_families = list(dict.fromkeys(item["family"] for item in pending
                                            if item["family"] not in self._families))
        shares = max(1, min(3, len(untried_families) or len(pending)))
        viable, rejected = [], []
        for item in pending:
            estimate, basis = self._trial_estimate(item)
            finalization = self._finalization_estimate(estimate)
            timeout = min(selection, max(UNKNOWN_TRIAL_SECONDS, _bounded_product(estimate, 2.0)),
                          selection / shares, max(0.0, total - finalization))
            reason = ("finalization_budget" if total < estimate + finalization else
                      "trial_budget" if timeout < estimate else None)
            details = {**item, "estimated_trial_seconds": estimate,
                       "estimated_finalization_seconds": finalization, "timing_basis": basis,
                       "timeout": max(0.0, timeout)}
            if reason:
                rejected.append({"index": item["index"], "family": item["family"], "reason": reason,
                                 "estimated_trial_seconds": estimate,
                                 "estimated_finalization_seconds": finalization})
            else:
                viable.append(details)
        if not viable:
            reason = "finalization_budget" if all(item["reason"] == "finalization_budget"
                                                   for item in rejected) else "trial_budget"
            return self._record_decision({**base, "reason": reason, "rejected": rejected})

        def order(item):
            if item["family"] not in self._families:
                return (0, untried_families.index(item["family"]), item["index"], 0)
            rank = family_ranks.get(item["family"], len(family_ranks))
            return (1, (rank + 1) * item["estimated_trial_seconds"], rank, item["index"])

        chosen = min(viable, key=order)
        self._attempted.add(chosen["key"])
        self._families.add(chosen["family"])
        self._dispatched += 1
        return self._record_decision({**base, **chosen, "reason": "dispatch",
                                     "shares": shares, "rejected": rejected})

    def report(self):
        return deepcopy({"version": 1, "kind": "heuristic", "guaranteed": False,
            "max_trials": self.max_trials, "dispatched": self._dispatched,
            "finalization_scope": "candidate_outer_and_full_refit",
            "row_scale_cap": MAX_ROW_SCALE, "work_ratio_cap": 16.0,
            "observations": list(self._observations.values()), "decisions": self._decisions})
