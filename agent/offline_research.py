"""Bounded host-local continuation on the existing INNER research view.

The caller supplies a finite recipe queue and the normal host evaluator. This
module never trains on its own, reads outer answers, requests an LLM, or changes
the frozen recovery baseline. Actual scoring and admission remain host-owned.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import islice
import json
import math
import re

from agent.features import normalize_feature_config
from agent.model_config import validate_model_config


MAX_LOCAL_TRIALS = 12
MAX_DEADLINE_TRIALS = 96
MAX_DEADLINE_INSPECTIONS = 512
MAX_INSPECTED_RECIPES = 256
MAX_RECIPE_BYTES = 64 * 1024


def _json_value(value):
    def keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Recipe keys must be strings")
            for child in item.values():
                keys(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                keys(child)

    try:
        keys(value)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Local recipe must contain finite JSON values") from error
    if len(raw) > MAX_RECIPE_BYTES:
        raise ValueError("Local recipe exceeds 64 KiB")
    return json.loads(raw)


def _canonical_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Local recipe must be an object")
    value = _json_value(recipe)
    if value.get("components"):
        components = value["components"]
        if not isinstance(components, list) or not 1 <= len(components) <= 8:
            raise ValueError("Local blend requires one to eight components")
        # BASELINE_SOURCE uses components in preference to the enclosing model
        # fields and equally averages them. Presentation IDs do not affect fit.
        result = {key: item for key, item in value.items() if key not in {
            "id", "label", "action", "model_config", "feature_config", "components"}}
        parts = [_canonical_recipe(item) for item in components]
        result["components"] = sorted(parts, key=lambda item: json.dumps(item, sort_keys=True))
        return result
    if "model_config" in value:
        checked = validate_model_config(value["model_config"], action=value.get("action"))
        if checked["model"] == "FixedBlend":
            raise ValueError("A local blend must declare its component feature settings")
        result = {key: item for key, item in value.items() if key not in {"id", "label", "action"}}
        result["model_config"] = checked
        result["feature_config"] = normalize_feature_config(value.get("feature_config"))
        return result
    # Custom LLM recipes may attach semantics to any field. Keep them intact;
    # the code hash prevents a different implementation suppressing a host fit.
    return value


def canonical_recipe_key(recipe, code_hash):
    """Identify implementation plus effective recipe, excluding host labels."""
    if not isinstance(code_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", code_hash):
        raise ValueError("Local recipe implementation requires a SHA-256 code hash")
    value = json.dumps(_canonical_recipe(recipe), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(code_hash.encode("ascii") + b"\0" + value).hexdigest()


def startup_seen(report_or_search, *, code_hash):
    """Recover measured startup fits, even if their later verification failed."""
    if not isinstance(report_or_search, dict):
        return []
    search = report_or_search.get("search", report_or_search)
    if not isinstance(search, dict) or not isinstance(search.get("candidates"), list):
        return []
    result = []
    for row in search["candidates"][:MAX_INSPECTED_RECIPES]:
        if not isinstance(row, dict) or not any(
                type(row.get(name)) in {int, float} and math.isfinite(row[name])
                for name in ("selection_metric", "metric")):
            continue
        try:
            config = validate_model_config(row["model_config"])
            if config["model"] == "FixedBlend":
                configs = config["parameters"]["components"]
                features = row.get("component_feature_configs")
                if not isinstance(features, list) or len(features) != len(configs):
                    continue
                recipe = {"components": [{"model_config": item,
                            "feature_config": normalize_feature_config(options)}
                            for item, options in zip(configs, features)]}
            else:
                recipe = {"model_config": config,
                          "feature_config": normalize_feature_config(row.get("feature_config"))}
            canonical_recipe_key(recipe, code_hash)
            result.append({"recipe": recipe, "code_hash": code_hash})
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _count(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _baseline_identity(state):
    baseline = state.baseline
    baseline.verify()
    return baseline, baseline.code_hash, baseline.recipe_hash


def _guard(state, baseline_identity):
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        raise RuntimeError("Local research is terminal after candidate freeze")
    state.check_source()
    state.gate.check_inputs()
    baseline, code_hash, recipe_hash = baseline_identity
    if (state.baseline is not baseline or state.baseline.code_hash != code_hash
            or state.baseline.recipe_hash != recipe_hash):
        raise RuntimeError("Local research changed its frozen recovery baseline")
    baseline.verify()


def run_local_continuation(state, recipes, *, evaluate, seen=(), code_hash,
                           max_trials=MAX_LOCAL_TRIALS, patience=4,
                           min_timeout=2.0, max_inspected=MAX_INSPECTED_RECIPES):
    """Evaluate new host presets without spending any additional API requests.

    ``evaluate(recipe)`` must use the existing host inner-scoring/admission
    protocol and return feedback with ``verified``/``accepted`` flags. Training
    failures are bounded, but source/input integrity failures are rechecked
    outside the exception handler and always propagate to the host.
    """
    max_trials = _count(max_trials, "max_trials", MAX_LOCAL_TRIALS)
    patience = _count(patience, "patience", MAX_LOCAL_TRIALS)
    max_inspected = _count(max_inspected, "max_inspected", MAX_INSPECTED_RECIPES)
    if (type(min_timeout) not in {int, float} or not math.isfinite(min_timeout)
            or not 0 < min_timeout <= 1800 or not callable(evaluate)):
        raise ValueError("Local continuation requires a finite timeout and evaluator")
    # Validate implementation identity before consuming a caller's generator.
    canonical_recipe_key({}, code_hash)
    report = {"origin": "host_local", "scope": "inner_selection", "api_requests": 0,
              "status": "running", "reason": None, "max_trials": max_trials,
              "patience": patience, "max_inspected": max_inspected,
              "attempted": 0, "verified": 0, "improvements": 0, "failed": 0,
              "duplicates": 0, "invalid_recipes": 0, "inspected": 0, "trials": []}
    state.report["local_research"] = report

    def finish(status, reason):
        report.update(status=status, reason=reason)
        state.save()
        return report

    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return finish("skipped", "research_closed")
    if state.gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return finish("skipped", "task_semantics_unavailable")
    if not len(state.gate.held_rows):
        return finish("skipped", "inner_validation_unavailable")
    if state.baseline is None:
        return finish("skipped", "baseline_unavailable")
    baseline_identity = _baseline_identity(state)
    _guard(state, baseline_identity)
    known = set()
    for record in islice(seen, MAX_INSPECTED_RECIPES):
        try:
            known.add(canonical_recipe_key(record["recipe"], record["code_hash"]))
        except (KeyError, TypeError, ValueError):
            continue
    no_improvement = 0
    iterator = iter(recipes)
    while report["attempted"] < max_trials and report["inspected"] < max_inspected:
        _guard(state, baseline_identity)
        if state.research_timeout() <= min_timeout:
            return finish("completed", "finalization_reserve")
        try:
            recipe = next(iterator)
        except StopIteration:
            return finish("completed", "queue_exhausted")
        report["inspected"] += 1
        try:
            key = canonical_recipe_key(recipe, code_hash)
        except (TypeError, ValueError):
            report["invalid_recipes"] += 1
            continue
        if key in known:
            report["duplicates"] += 1
            continue
        known.add(key)
        report["attempted"] += 1
        state.report["local_experiments"] = state.report.get("local_experiments", 0) + 1
        state.save()
        try:
            feedback = evaluate(deepcopy(recipe))
            if not isinstance(feedback, dict):
                raise ValueError("Local evaluator returned invalid host feedback")
            verified = feedback.get("verified") is True
            accepted = verified and feedback.get("accepted") is True
            reason = feedback.get("reason")
            reason = reason if reason in {"improved", "not_better", "refit_budget",
                "candidate_execution_failed", "execution_failed"} else "host_rejected"
            error_type = None
        except (OSError, ValueError, TimeoutError) as error:
            verified, accepted, reason = False, False, "candidate_execution_failed"
            # Arbitrary exception messages may contain data or credentials.
            error_type = ("TimeoutError" if isinstance(error, TimeoutError) else
                          "OSError" if isinstance(error, OSError) else "ValueError")
        # An evaluator exception cannot swallow mutation of source or inputs.
        _guard(state, baseline_identity)
        report["verified"] += int(verified)
        report["improvements"] += int(accepted)
        report["failed"] += int(not verified)
        no_improvement = 0 if accepted else no_improvement + 1
        entry = {"index": report["attempted"], "identity": key,
                 "verified": verified, "accepted": accepted, "reason": reason}
        if error_type:
            entry["error_type"] = error_type
        report["trials"].append(entry)
        state.save()
        if no_improvement >= patience:
            return finish("completed", "patience")
    return finish("completed", "trial_limit" if report["attempted"] >= max_trials else "inspection_limit")


def run_deadline_continuation(state, recipes, *, adaptive, evaluate, seen=(), code_hash,
                              after_warmup=None, max_trials=64, min_timeout=2.0,
                              max_inspected=MAX_DEADLINE_INSPECTIONS, family_failure_limit=3):
    """Use a long run's spare budget without a global no-improvement stop.

    Legacy presets form the warmup; a bounded proposer supplies new ordinary
    recipes afterwards. Failed families are isolated, while a verified but
    non-improving trial never closes other families. Only the caller's normal
    INNER evaluator can accept a recipe. Optional custom-source trials have
    their own small cap and run between warmup and adaptive proposals.
    """
    max_trials = _count(max_trials, "max_trials", MAX_DEADLINE_TRIALS)
    max_inspected = _count(max_inspected, "max_inspected", MAX_DEADLINE_INSPECTIONS)
    family_failure_limit = _count(family_failure_limit, "family_failure_limit", 10)
    if (type(min_timeout) not in {int, float} or not math.isfinite(min_timeout)
            or not 0 < min_timeout <= 1800 or not callable(evaluate)
            or (after_warmup is not None and not callable(after_warmup))):
        raise ValueError("Deadline continuation requires a finite timeout and evaluators")
    if adaptive is not None and not all(callable(getattr(adaptive, name, None))
                                        for name in ("suggest", "observe", "report")):
        raise ValueError("Adaptive proposer must implement suggest, observe and report")
    canonical_recipe_key({}, code_hash)

    def remaining():
        value = state.remaining() if callable(getattr(state, "remaining", None)) else None
        return max(0.0, float(value)) if type(value) in {int, float} and math.isfinite(value) else None

    report = {"version": 2, "origin": "host_local", "scope": "inner_selection", "api_requests": 0,
              "policy": "deadline_adaptive", "status": "running", "reason": None,
              "max_trials": max_trials, "max_inspected": max_inspected,
              "family_failure_limit": family_failure_limit, "attempted": 0,
              "verified": 0, "improvements": 0, "failed": 0, "duplicates": 0,
              "invalid_recipes": 0, "inspected": 0, "warmup_attempted": 0,
              "adaptive_attempted": 0, "blocked_families": [], "trials": [],
              "started_remaining_seconds": remaining(), "extension_hook_called": False}
    state.report["local_research"] = report

    def finish(status, reason, *, error=None):
        report.update(status=status, reason=reason, ended_remaining_seconds=remaining())
        if error is not None:
            report["coordinator_error_type"] = type(error).__name__
        if adaptive is not None:
            try:
                report["adaptive"] = adaptive.report()
            except (OSError, ValueError, RuntimeError, TypeError):
                report["adaptive"] = {"status": "report_unavailable"}
        state.save()
        return report

    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return finish("skipped", "research_closed")
    if state.gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return finish("skipped", "task_semantics_unavailable")
    if not len(state.gate.held_rows):
        return finish("skipped", "inner_validation_unavailable")
    if state.baseline is None:
        return finish("skipped", "baseline_unavailable")
    baseline_identity = _baseline_identity(state)
    _guard(state, baseline_identity)
    known = set()
    for record in islice(seen, MAX_DEADLINE_INSPECTIONS):
        try:
            known.add(canonical_recipe_key(record["recipe"], record["code_hash"]))
        except (KeyError, TypeError, ValueError):
            continue
    iterator, phase, failures, blocked = iter(recipes), "warmup", {}, set()

    def observe(item, feedback):
        if phase == "adaptive":
            try:
                adaptive.observe(deepcopy(item), deepcopy(feedback))
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                _guard(state, baseline_identity)
                return error
        _guard(state, baseline_identity)
        return None

    while report["attempted"] < max_trials and report["inspected"] < max_inspected:
        _guard(state, baseline_identity)
        if state.research_timeout() <= min_timeout:
            return finish("completed", "finalization_reserve")
        if phase == "warmup":
            try:
                item = next(iterator)
            except StopIteration:
                if after_warmup is not None:
                    report["extension_hook_called"] = True
                    try:
                        after_warmup()
                    except (OSError, ValueError, RuntimeError, ImportError) as error:
                        _guard(state, baseline_identity)
                        report["extension_hook_error_type"] = type(error).__name__
                    _guard(state, baseline_identity)
                phase = "adaptive"
                continue
        else:
            if adaptive is None:
                return finish("completed", "adaptive_unavailable")
            try:
                item = adaptive.suggest()
            except (OSError, ValueError, RuntimeError, TypeError, ImportError) as error:
                _guard(state, baseline_identity)
                return finish("completed", "adaptive_unavailable", error=error)
            _guard(state, baseline_identity)
            if item is None:
                return finish("completed", "queue_exhausted")
        report["inspected"] += 1
        try:
            key = canonical_recipe_key(item, code_hash)
        except (TypeError, ValueError) as error:
            report["invalid_recipes"] += 1
            observed = observe(item, {"verified": False, "accepted": False, "reason": "invalid_recipe"})
            if observed:
                return finish("completed", "adaptive_feedback_unavailable", error=observed)
            continue
        family = str(item.get("action", "custom"))[:120]
        skip = "duplicate" if key in known else "family_blocked" if family in blocked else None
        if skip:
            report["duplicates"] += int(skip == "duplicate")
            observed = observe(item, {"verified": False, "accepted": False, "reason": skip})
            if observed:
                return finish("completed", "adaptive_feedback_unavailable", error=observed)
            continue
        if state.research_timeout() <= min_timeout:
            observed = observe(item, {"verified": False, "accepted": False, "reason": "insufficient_budget"})
            if observed:
                return finish("completed", "adaptive_feedback_unavailable", error=observed)
            return finish("completed", "finalization_reserve")
        known.add(key)
        report["attempted"] += 1
        report[phase + "_attempted"] += 1
        state.report["local_experiments"] = state.report.get("local_experiments", 0) + 1
        state.save()
        error_type = None
        try:
            feedback = evaluate(deepcopy(item))
            if not isinstance(feedback, dict):
                raise ValueError("Local evaluator returned invalid host feedback")
            verified = feedback.get("verified") is True
            accepted = verified and feedback.get("accepted") is True
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            verified, accepted = False, False
            error_type = type(error).__name__
            feedback = {"verified": False, "accepted": False, "reason": "candidate_execution_failed"}
        _guard(state, baseline_identity)
        observed = observe(item, feedback)
        report["verified"] += int(verified)
        report["improvements"] += int(accepted)
        report["failed"] += int(not verified)
        failures[family] = 0 if verified else failures.get(family, 0) + 1
        if failures[family] >= family_failure_limit:
            blocked.add(family)
            report["blocked_families"] = sorted(blocked)
        reason = feedback.get("reason")
        reason = reason if reason in {"improved", "not_better", "refit_budget",
            "candidate_execution_failed", "execution_failed"} else "host_rejected"
        entry = {"index": report["attempted"], "phase": phase, "identity": key, "family": family,
                 "verified": verified, "accepted": accepted, "reason": reason}
        if error_type:
            entry["error_type"] = error_type
        report["trials"].append(entry)
        state.save()
        if observed:
            return finish("completed", "adaptive_feedback_unavailable", error=observed)
    return finish("completed", "trial_limit" if report["attempted"] >= max_trials else "inspection_limit")
