"""Bounded host candidates on the existing INNER selection protocol.

The coordinator calls this hook only in the explicit adaptive mode.  Factories
do not receive outer targets.  Every actual fit still passes through the host's
ordinary frozen-candidate assessment, final validation and full-data refit.
Optional modelling failures can be skipped; source/input/baseline mutations
are checked outside those exception handlers and always propagate.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

from agent.features import normalize_feature_config
from agent.model_config import make_model_config, validate_model_config
from agent.offline_research import canonical_recipe_key
from agent.research_candidate import freeze_candidate


MAX_EXTENSION_TRIALS = 6
MAX_FREQUENCY_FIT_CELLS = 2_000_000
KINDS = ("frequency", "history", "boost_catboost", "boost_lightgbm", "ensemble", "autogluon")


class _Skip(ValueError):
    """An expected, bounded eligibility diagnostic, never provider text."""


def _guard(state, identity):
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        raise RuntimeError("Host extensions are terminal after candidate freeze")
    state.check_source()
    state.gate.check_inputs()
    baseline, code_hash, recipe_hash = identity
    if (state.baseline is not baseline or baseline.code_hash != code_hash
            or baseline.recipe_hash != recipe_hash):
        raise RuntimeError("Host extensions changed the frozen recovery baseline")
    baseline.verify()
    if state.selected_candidate is not None:
        state.selected_candidate.verify()


def _source(candidate):
    candidate.verify()
    return candidate.code_path.read_text(encoding="utf-8")


def _frequency_trial(state, _finalist):
    from agent.frequency_candidate import (
        MAX_COLUMNS, frequency_candidate_source, make_frequency_recipe,
    )

    gate = state.gate
    categories = tuple(gate.research_spec.categorical)
    if not categories:
        raise _Skip("no_categorical_features")
    if len(categories) > MAX_COLUMNS:
        raise _Skip("categorical_column_limit")
    if not set(categories) <= set(gate.research_spec.features):
        raise _Skip("categorical_schema_unavailable")
    if len(gate.fit_rows) * len(categories) > MAX_FREQUENCY_FIT_CELLS:
        raise _Skip("frequency_fit_cell_limit")
    # Vocabularies are deliberately not built here or from prediction rows.
    # The existing wrapper fits its bounded dictionary afresh in every phase.
    return (frequency_candidate_source(_source(state.baseline)),
            make_frequency_recipe(state.baseline.recipe, categories),
            "Add fit-only log category counts to the entire frozen baseline.")


def _history_trial(state, _finalist):
    options = state.report.get("quality_options", {})
    if (options.get("history_config") is not None or options.get("history_auto")
            or state.report.get("history_research", {}).get("requested")
            or state.report.get("history_discovery", {}).get("requested")):
        raise _Skip("history_already_requested")
    gate = state.gate
    if gate.spec.metric.task_type != "regression":
        raise _Skip("regression_required")
    if (gate.choice.get("strategy") != "time"
            or gate.inner_split.get("strategy") != "time"
            or gate.inner_split.get("time_column") != gate.choice.get("time_column")):
        raise _Skip("matching_time_policy_required")
    from agent.history_discovery_host import discover_history_options
    from agent.history_runtime import history_candidate_source, make_history_recipe

    discovered = discover_history_options(state)
    if discovered is None:
        raise _Skip("history_discovery_abstained")
    recipe = make_history_recipe(state.baseline.recipe, discovered["history_config"],
                                 discovered["blend_weight"])
    return (history_candidate_source(_source(state.baseline)), recipe,
            "Use one fit-feature-discovered history schema and a fixed past-only blend.")


def _ordinary_components(candidate):
    from agent.research_v3 import BASELINE_SOURCE

    if candidate is None or _source(candidate) != BASELINE_SOURCE:
        raise _Skip("ordinary_measured_finalist_required")
    recipe = candidate.recipe
    components = recipe.get("components") or [recipe]
    if not isinstance(components, list) or not 1 <= len(components) <= 8:
        raise _Skip("ordinary_component_limit")
    result = []
    for item in components:
        if not isinstance(item, dict) or not isinstance(item.get("action"), str):
            raise _Skip("ordinary_component_required")
        checked = validate_model_config(item.get("model_config"), action=item["action"])
        if checked["model"] == "FixedBlend":
            raise _Skip("nested_ensemble_unsupported")
        result.append({"action": item["action"], "model_config": checked,
                       "feature_config": normalize_feature_config(item.get("feature_config"))})
    return result


def _boosting_trial(state, finalist, action):
    from agent.boosting_candidate import candidate_source, make_boosting_recipe

    base = None
    for candidate in (finalist, state.baseline):
        try:
            components = _ordinary_components(candidate)
        except _Skip:
            continue
        base = next((item for item in components if item["action"] == action
                     and "target_transform" not in item["model_config"]), None)
        if base is not None:
            break
    if base is None:
        base = {"action": action,
                "model_config": make_model_config(action, state.gate.spec.metric.task_type),
                "feature_config": normalize_feature_config()}
    recipe = make_boosting_recipe(base, deepcopy(state.gate.choice),
                                  max_iterations=2000, stopping_rounds=50)
    return candidate_source(), recipe, "Choose bounded boosting length on a fit-only policy-preserving split, then refit all fit rows."


def _boost_catboost_trial(state, finalist):
    return _boosting_trial(state, finalist, "catboost")


def _boost_lightgbm_trial(state, finalist):
    return _boosting_trial(state, finalist, "lightgbm")


def _ensemble_trial(state, finalist):
    from agent.research_v3 import BASELINE_SOURCE

    if finalist is None or state.baseline_inner is None:
        raise _Skip("ordinary_measured_finalist_required")
    # This is the accepted, already-measured finalist captured BEFORE this
    # extension round, not an entry merely appended before a failed fit.
    baseline_parts = _ordinary_components(state.baseline)
    finalist_parts = _ordinary_components(finalist)
    parts, keys = [], set()
    for item in [*baseline_parts, *finalist_parts]:
        key = canonical_recipe_key(item, "0" * 64)
        if key not in keys:
            keys.add(key)
            parts.append(item)
    if not 2 <= len(parts) <= 3:
        raise _Skip("distinct_component_limit")
    recipe = {"id": "host_equal_ensemble", "label": "Host equal ensemble",
              "components": parts}
    key = canonical_recipe_key(recipe, "0" * 64)
    if key in {canonical_recipe_key(state.baseline.recipe, "0" * 64),
               canonical_recipe_key(finalist.recipe, "0" * 64)}:
        raise _Skip("ensemble_already_measured")
    return BASELINE_SOURCE, recipe, "Equally combine distinct ordinary components from measured recipes."


def _autogluon_trial(state, _finalist):
    from agent.autogluon_candidate import availability, candidate_source, make_autogluon_recipe

    available = availability()
    if not isinstance(available, dict) or available.get("available") is not True:
        raise _Skip("autogluon_unavailable")
    timeout = state.research_timeout()
    if timeout < 60:
        raise _Skip("autogluon_budget")
    recipe = make_autogluon_recipe(state.gate.spec.metric.task_type, state.gate.spec.metric.name,
        time_limit=max(30.0, min(180.0, timeout * .25)),
        validation=deepcopy(state.gate.choice), num_cpus=8, memory_limit=24.0)
    return candidate_source(), recipe, "Compare a bounded fit-only AutoGluon CPU recipe."


def _eligibility(state):
    if state.report.get("quality_options", {}).get("adaptive_search") is not True:
        return "adaptive_search_disabled"
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return "research_closed"
    if state.gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return "task_semantics_unavailable"
    if state.baseline is None or state.baseline_inner is None:
        return "measured_baseline_required"
    if not state.gate.fit_rows or not state.gate.held_rows:
        return "inner_validation_unavailable"
    return None


def run_research_extensions(state, assess, *, max_trials=MAX_EXTENSION_TRIALS):
    """Try at most six optional candidates without bypassing host admission.

    ``assess`` has the same signature as ``research_v3._assess_candidate``.
    Successful INNER evaluation is not final admission or a quality claim.
    """
    if type(max_trials) is not int or not 1 <= max_trials <= MAX_EXTENSION_TRIALS:
        raise ValueError("Host extensions require one to six trial slots")
    if not callable(assess):
        raise ValueError("Host extensions require the host candidate assessor")
    report = {"version": 1, "policy": "host_extensions_v1", "scope": "inner_selection",
              "status": "running", "reason": None, "max_trials": max_trials,
              "attempted": 0, "verified": 0, "accepted": 0, "api_requests": 0, "trials": []}
    state.report["research_extensions"] = report
    reason = _eligibility(state)
    if reason:
        report.update(status="skipped", reason=reason)
        state.save()
        return report
    baseline = state.baseline
    identity = baseline, baseline.code_hash, baseline.recipe_hash
    finalist = state.selected_candidate
    _guard(state, identity)
    factories = (_frequency_trial, _history_trial, _boost_catboost_trial,
                 _boost_lightgbm_trial, _ensemble_trial, _autogluon_trial)
    state.report.setdefault("extensions", {})["host_candidates"] = "adaptive_v1"
    for kind, factory in zip(KINDS, factories):
        _guard(state, identity)
        entry = {"kind": kind, "status": "skipped", "reason": None, "attempted": False}
        report["trials"].append(entry)
        if report["attempted"] >= max_trials:
            entry["reason"] = "trial_limit"
            continue
        if state.research_timeout() <= 2:
            entry["reason"] = "finalization_reserve"
            continue
        candidate = None
        try:
            source, recipe, hypothesis = factory(state, finalist)
            # Factories may spend some bounded preparation time (history's
            # discovery worker has its own allowance within the same deadline).
            _guard(state, identity)
            if state.research_timeout() <= 2:
                raise _Skip("finalization_reserve")
            directory = Path(state.run_dir) / "host_extensions" / kind / uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            source_path = directory / "source.py"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            candidate = freeze_candidate(source_path, recipe, directory / "frozen")
            entry.update(status="running", reason=None, attempted=True,
                         code_hash=candidate.code_hash, recipe_hash=candidate.recipe_hash)
            report["attempted"] += 1
            if kind == "history":
                state.report["history_research"] = {
                    "version": 1, "policy": "adaptive_auto_v1", "origin": "host_history",
                    "scope": "inner_selection", "requested": True, "active": True,
                    "status": "running", "reason": None, "attempted": 1,
                    "history_config": recipe["history_config"], "blend_weight": recipe["blend_weight"],
                    "candidate": {"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash},
                    "feedback": None}
            state.save()
            origin = "history" if kind == "history" else "local"
            feedback = assess(state, candidate, {"hypothesis": hypothesis}, origin=origin)
            if not isinstance(feedback, dict) or type(feedback.get("verified")) is not bool:
                raise ValueError("Invalid host extension assessment")
            json.dumps(feedback, allow_nan=False)
            verified = feedback["verified"] is True
            accepted = verified and feedback.get("accepted") is True
            reason = feedback.get("reason")
            reason = reason if reason in {"improved", "not_better", "refit_budget",
                "candidate_execution_failed", "execution_failed"} else "host_rejected"
            entry.update(status="completed" if verified else "failed",
                         reason=reason,
                         feedback=feedback)
            report["verified"] += int(verified)
            report["accepted"] += int(accepted)
            if kind == "history":
                state.report["history_research"].update(status=entry["status"], reason=entry["reason"], feedback=feedback)
        except _Skip as error:
            entry.update(status="skipped", reason=str(error))
        except ImportError:
            entry.update(status="skipped", reason="optional_dependency_unavailable")
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            error_type = ("TimeoutError" if isinstance(error, TimeoutError) else
                          "OSError" if isinstance(error, OSError) else
                          "ValueError" if isinstance(error, ValueError) else "RuntimeError")
            entry.update(status="failed", reason="candidate_execution_failed",
                         error_type=error_type)
        # These checks intentionally do not sit in the optional-failure catch.
        _guard(state, identity)
        if candidate is not None:
            candidate.verify()
        if kind == "history" and entry["attempted"]:
            state.report["history_research"].update(status=entry["status"], reason=entry["reason"])
        state.event("host_extension", result=entry)
        state.save()
    report.update(status="completed", reason="trial_limit" if report["attempted"] >= max_trials else "queue_exhausted")
    state.save()
    return report
