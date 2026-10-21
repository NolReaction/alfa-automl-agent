"""Host preset scheduling within the existing bounded research evaluator."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
import time

from agent.offline_research import canonical_recipe_key, run_local_continuation
from agent.model_config import validate_model_config
from agent.search_policy import exploration_candidates, refinement_candidates
from agent.search_schedule import AdaptiveSearchSchedule


def run_adaptive_continuation(state, *, evaluate, seen, code_hash, ranked=()):
    """Reorder finite host presets; original integrity and admission guards remain.

    Startup ranks are a prior ordering only. New scores are compared solely on
    the current research inner split. Family timing heuristics never populate
    the exact frozen-recipe BudgetPolicy measurements.
    """
    gate = state.gate
    if (state.dialogue_closed or not len(gate.held_rows) or state.baseline is None
            or gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False):
        return run_local_continuation(state, (), evaluate=evaluate, seen=seen, code_hash=code_hash, patience=6)
    available = tuple(action for action in ("linear", "baseline", "extra_trees", "lightgbm", "catboost")
                      if action not in {"lightgbm", "catboost"} or importlib.util.find_spec(action) is not None)
    pending = exploration_candidates(gate.spec.classification, gate.spec.metric.name,
        gate.research_train.iloc[gate.fit_rows][gate.spec.target_column], available_actions=available)
    schedule = AdaptiveSearchSchedule(fit_rows=len(gate.fit_rows), predict_rows=len(gate.held_rows),
        outer_fit_rows=len(gate.outer_fit_frame), outer_predict_rows=len(gate.outer_holdout_frame),
        full_fit_rows=len(gate.train), full_predict_rows=len(gate.test))
    known, covered, previous_presets = set(), set(), []
    for record in seen:
        try:
            key = canonical_recipe_key(record["recipe"], record["code_hash"])
            known.add(key)
            # Only presets implemented by this exact source count as coverage.
            if record["code_hash"] == code_hash and "components" not in record["recipe"]:
                # startup_seen intentionally omits presentation/action fields.
                # Infer a preset family only from its validated model config.
                for action in available:
                    try:
                        validate_model_config(record["recipe"].get("model_config"), action=action)
                    except ValueError:
                        continue
                    covered.add(action)
                    previous_presets.append({**record["recipe"], "action": action,
                                             "label": "Previously measured host preset"})
                    break
        except (KeyError, TypeError, ValueError):
            continue
    pending = [item for item in pending if canonical_recipe_key(item, code_hash) not in known]
    original = deepcopy([*pending, *previous_presets])
    families = {item["action"] for item in pending}
    measured = {}
    current_limits = {}
    refined = False

    def ranking():
        # Never numerically mix startup scores from another nested split with
        # the present inner measurements. Retain startup order only as a prior.
        ordered = sorted(measured.values(), key=lambda item: item["value"],
                         reverse=gate.spec.metric.higher_is_better)
        return [item["recipe"] for item in ordered] + list(ranked)

    def recipes():
        nonlocal refined
        while pending or not refined:
            if not refined and (covered >= families or not pending):
                proposals = refinement_candidates(gate.spec.classification, gate.spec.metric.name,
                    ranking(), existing=original, available_actions=available)
                pending.extend(item for item in proposals if canonical_recipe_key(item, code_hash) not in known)
                refined = True
            if not pending:
                return
            decision = schedule.choose(pending, selection_remaining=max(0, state.research_timeout()),
                                       total_remaining=max(0, state.remaining()), ranked=ranking())
            state.report["local_search_policy"] = schedule.report()
            if decision.get("index") is None:
                return
            recipe = pending.pop(decision["index"])
            current_limits[recipe["id"]] = decision["timeout"]
            covered.add(recipe["action"])
            yield recipe

    def timed_evaluate(recipe):
        original_recipe = deepcopy(recipe)
        started = time.monotonic()
        completed = False
        try:
            feedback = evaluate(recipe, maximum=current_limits[recipe["id"]])
            completed = isinstance(feedback, dict) and feedback.get("verified") is True
            metric = feedback.get("metric") if isinstance(feedback, dict) else None
            if (completed and isinstance(metric, dict) and type(metric.get("value")) in {int, float}
                    and math.isfinite(metric["value"])):
                measured[canonical_recipe_key(original_recipe, code_hash)] = {"recipe": original_recipe,
                                                                    "value": metric["value"]}
            return feedback
        finally:
            schedule.observe(original_recipe, elapsed_seconds=max(.001, time.monotonic() - started), completed=completed)
            state.report["local_search_policy"] = schedule.report()

    return run_local_continuation(state, recipes(), evaluate=timed_evaluate, seen=seen,
                                  code_hash=code_hash, patience=6)
