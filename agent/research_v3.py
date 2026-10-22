"""Terminal outer acceptance of recipes selected on an outer-fit-only view.

The dialogue never receives an outer score. Separate input views and immutable
recipes control the normal protocol; they are not an OS sandbox.
"""

from __future__ import annotations

import json
from io import BytesIO
import math
from pathlib import Path
import shutil
import time
from uuid import uuid4

import pandas as pd

from agent.automl import _atomic_copy, _file_hash
from agent.features import normalize_feature_config
from agent.llm import ROOT, TOOL, get_model, safe_error_details, safe_error_message
from agent.model_config import ACTIONS, make_model_config
from agent.research import RunState, checked_completion, record_usage
from agent.research_budget import ADAPTIVE_RESERVE_MIN_BUDGET, BudgetPolicy
from agent.inner_feedback import inner_feedback
from agent.research_candidate import (
    CANDIDATE_NAME, RECIPE_NAME, CandidateExecutionError, freeze_candidate, run_candidate,
)
from agent.research_validation import HYPOTHESIS_NAME, read_hypothesis
from agent.research_workspace import ResearchWorkspace
from agent.search import local_candidate_plan, seed_candidate
from validation import validate_predictions


STAGES = ("preparation", "inner_research", "candidate_frozen", "outer_validation", "full_refit", "result")
MAX_API_RETRIES = 1
MAX_COMPLETION_TOKENS = 8192
# These stops end the paid dialogue, not the original CPU/time budget. The
# local coordinator still checks input integrity, inner eligibility and the
# measured finalization reserve before admitting a bounded new trial.
LOCAL_CONTINUATION_REASONS = frozenset({
    "api_error", "truncated_response", "context_budget", "step_limit", "repeated_failures",
})
ADAPTIVE_CONTINUATION_REASONS = LOCAL_CONTINUATION_REASONS | {"model_finished", "time_limit"}
MAX_LOCAL_FROZEN_ATTEMPTS = 70  # 64 ordinary trials plus six custom-source trials.
SYSTEM_V3 = """You are an autonomous ML researcher using run_python.
Read task.md and research_validation.json in the working directory. Only the
approved outer-fit training view is available for research. All validation
feedback is INNER selection feedback. Fit preprocessing on each fit partition.
Inspect the schema and test a concrete hypothesis; do not assume column names.

To propose a measured trial write three fresh files in ONE run_python call:
research_hypothesis.json: {"hypothesis": "change and reason"};
research_recipe.json: a JSON object with the selected settings;
research_candidate.py: defines fit_predict(X_fit, y_fit, X_predict, context).
The function receives RAW STRING feature DataFrames, a numeric regression
target or exact string class labels, and a version-3 context with recipe,
features, numeric, categorical, classes, positive_class, metric, task_type,
prediction_kind and phase. Return {"predictions": values}; for probabilities
also return {"classes": [exact labels in probability column order]}.
Use function arguments for ALL training data. Do not read training files inside
the recipe, use row positions as learned answers, or branch on the phase.
The host executes this function separately on the inner fit/validation rows,
checks predictions and recomputes the selection score. Analysis-only calls are
allowed but do not count as measured trials. Do not write submission.csv or
include actual answers in prediction output. The host owns final CSV creation.

On completion the host freezes the best inner recipe, ends this dialogue, then
evaluates the frozen baseline and candidate on a withheld outer partition.
No outer result returns to you. The admitted recipe is refit on full train.
An unsuccessful trial preserves the previous valid result. Use installed
libraries only, at most 8 CPU threads; no network, .env or credentials.
Separate files are not an OS sandbox: use only the declared allowed inputs.
"""


BASELINE_SOURCE = '''import numpy as np
from agent.models import build_pipeline
from agent.metrics import probability_matrix

def fit_predict(X_fit, y_fit, X_predict, context):
    recipe = context["recipe"]
    components = recipe.get("components") or [recipe]
    results = []
    for item in components:
        model = build_pipeline(item["action"], context["task_type"],
                               context["numeric"], context["categorical"],
                               item["model_config"], item.get("feature_config"))
        model.fit(X_fit, y_fit)
        if context["task_type"] == "classification":
            value = probability_matrix(model.predict_proba(X_predict),
                classes=tuple(context["classes"]),
                prediction_classes=tuple(map(str, model.classes_)),
                positive_class=context["positive_class"])
        else:
            value = np.asarray(model.predict(X_predict), dtype=float)
        results.append(value)
    values = sum(results) / len(results)
    if context["task_type"] == "classification":
        if context["prediction_kind"] == "labels":
            values = np.asarray(context["classes"])[values.argmax(axis=1)]
        return {"predictions": values, "classes": context["classes"]}
    return {"predictions": values}
'''


def _better(left, right):
    return (left["value"] > right["value"] if left["higher_is_better"]
            else left["value"] < right["value"])


def _recipe_from_report(report, task_type):
    """Translate an existing, bounded AutoML winner without trusting its score."""
    config = report["model_config"]

    def component(value, features):
        for action in ACTIONS:
            if make_model_config(action, task_type)["model"] == value["model"]:
                from agent.model_config import validate_model_config
                return {"action": action, "model_config": validate_model_config(value, action=action),
                        "feature_config": normalize_feature_config(features)}
        raise ValueError("Unsupported AutoML winner model")

    if config["model"] == "FixedBlend":
        parts = config["parameters"]["components"]
        features = report.get("component_feature_configs", [])
        if len(parts) != 2 or len(features) != 2 or config["parameters"]["weight"] != .5:
            raise ValueError("Invalid fixed blend winner")
        return {"components": [component(value, options) for value, options in zip(parts, features)]}
    return component(config, report.get("feature_config"))


class V3State(RunState):
    def __init__(self, task_path, max_steps, time_budget, *, adaptive_search=False):
        if type(adaptive_search) is not bool:
            raise ValueError("adaptive_search must be a boolean")
        super().__init__(task_path, max_steps, time_budget)
        self._accepted_backup_hash = None
        self.gate = None
        self.baseline = None
        self.baseline_inner = None
        self.selected_candidate = None
        self.best_inner = None
        self.dialogue_closed = False
        self.tried_recipes = []
        self.adaptive_search = adaptive_search and time_budget > ADAPTIVE_RESERVE_MIN_BUDGET
        self.budget = BudgetPolicy(time_budget, started=self.started, adaptive_reserve=self.adaptive_search)
        self.report.update(research_protocol_version=3, stage="preparation",
            stages=[{"id": item, "status": "pending"} for item in STAGES],
            evaluations={"selection": None, "final_check": None},
            isolation={"kind": "allowed_input_view", "os_sandbox": False},
            extensions={"policy": "not_connected", "portfolio": "not_connected", "history": "disabled"})
        self.report["api_recovery"] = {"maximum_retries": MAX_API_RETRIES, "retries": 0}
        self.transition("preparation")

    def checkpoint(self, directory, metric=None):
        accepted = super().checkpoint(directory, metric)
        if accepted:
            self._accepted_backup_hash = _file_hash(self.backup)
        return accepted

    def recover(self, *, force=False):
        if self._accepted_backup_hash is None:
            return False
        if _file_hash(self.backup) != self._accepted_backup_hash:
            raise ValueError("Accepted research checkpoint bytes changed")
        return super().recover(force=force)

    def transition(self, stage, *, reason=None):
        if stage not in STAGES:
            raise ValueError("Unknown research stage")
        for item in self.report["stages"]:
            if item["status"] == "running":
                item["status"] = "completed"
            if item["id"] == stage:
                item["status"] = "running"
                if reason:
                    item["reason"] = reason
        self.report["stage"] = stage
        self.save()

    def check_source(self):
        if self.gate is not None:
            self.gate.check_source()

    @staticmethod
    def recipe_key(candidate):
        return candidate.code_hash + ":" + candidate.recipe_hash

    def finalization_plan(self, candidate=None):
        gate = self.gate
        candidate = candidate if candidate is not None else self.selected_candidate
        return self.budget.finalization_estimate(
            baseline_key=self.recipe_key(self.baseline) if self.baseline is not None else "baseline_pending",
            candidate_key=self.recipe_key(candidate) if candidate is not None else None,
            outer_fit_rows=len(gate.outer_fit_frame), outer_predict_rows=len(gate.outer_holdout_frame),
            full_fit_rows=len(gate.train), test_rows=len(gate.test))

    @property
    def final_reserve(self):
        return self.finalization_plan().reserve_seconds

    def research_timeout(self, maximum=1800):
        return self.budget.research_timeout(self.finalization_plan(), maximum=maximum)

    def record_timing(self, candidate, started, fit_rows, predict_rows):
        self.budget.record_measurement(self.recipe_key(candidate),
            max(.001, time.monotonic() - started), fit_rows, predict_rows)

    def estimate_research_cost(self, candidate, *, fit_rows, predict_rows):
        return self.budget.estimate_trial_cost(self.recipe_key(candidate), fit_rows, predict_rows)

    def budget_context(self):
        plan = self.finalization_plan()
        summary = self.budget.report(plan)
        self.report["time_allocation"] = summary
        return {"scope": "inner_selection", "time_allocation": summary,
            "remaining_requests": max(0, self.report["max_steps"] - self.report["api"]["attempts"]),
            "maximum_experiment_seconds": self.research_timeout(),
            "instruction": "Use the measured time estimates to choose a feasible next experiment. "
                "The host reserves time for outer checks and full-data refits. "
                "These are conservative planning estimates, not guaranteed training durations."}

    def measure_inner(self, candidate, *, rows=None, maximum=1800):
        gate = self.gate
        gate.check_inputs()
        if not gate.held_rows:
            raise ValueError("Inner validation unavailable; no adaptive trial is allowed")
        if self.dialogue_closed:
            raise RuntimeError("Inner research is terminal after candidate freeze")
        fitting, held = rows if rows is not None else (gate.fit_rows, gate.held_rows)
        timeout = self.research_timeout(maximum)
        if timeout <= 0:
            raise TimeoutError("Research time is reserved for final validation and refit")
        started = time.monotonic()
        result = run_candidate(candidate, spec=gate.research_spec,
            fit_frame=gate.research_train.iloc[fitting],
            predict_frame=gate.research_train.iloc[held].drop(columns=[gate.spec.target_column]),
            actuals=gate.research_train.iloc[held][gate.spec.target_column].tolist(), phase="inner_selection",
            output_dir=self.run_dir / "inner_evaluations",
            timeout=timeout, deadline=min(started + timeout, self.budget.deadline - self.budget.publication_buffer))
        self.record_timing(candidate, started, len(fitting), len(held))
        return result

    def request_allowed(self, messages):
        if self.dialogue_closed:
            raise RuntimeError("Research is terminal after candidate freeze")
        if self.report["api"]["attempts"] >= self.report["max_steps"]:
            return "step_limit"
        if self.research_timeout() <= 2:
            return "time_limit"
        payload = json.dumps(self.clean(messages), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # A host-side estimate, explicitly not provider usage or currency.
        estimate = math.ceil(len(payload) / 3) + MAX_COMPLETION_TOKENS
        budget = self.report.setdefault("request_guard", {
            "estimated_tokens_reserved": 0, "estimated_token_limit": 100000,
            "estimation": f"ceil(UTF-8 bytes / 3) + {MAX_COMPLETION_TOKENS} output reserve per request"})
        if len(payload) > 64 * 1024 or budget["estimated_tokens_reserved"] + estimate > budget["estimated_token_limit"]:
            return "context_budget"
        budget["estimated_tokens_reserved"] += estimate
        return None


def prepare_research(state):
    state.gate = ResearchWorkspace(state.task_path, state.workspace)
    state.validator = state.gate
    state.work_task = state.gate.work_task
    state.work_data = state.gate.work_data
    state.report["host_validation"] = {"status": "supported", "manifest": "research_validation.json",
        "metric": state.gate.spec.metric.to_dict(), "split": state.gate.split,
        "purpose": "final_check", "selection_uses_outer": True, "unbiased_estimate": False}
    state.report["task_semantics"] = state.gate.manifest.get("task_semantics", {})
    if validate_predictions(state.task_dir)["valid"]:
        state.checkpoint(state.task_dir)
        state.report["baseline"] = "existing_output"
    state.save()


def prepare_baseline(state):
    """Run the existing search ONLY on the permitted research data."""
    gate = state.gate
    recipe = seed_candidate(gate.spec.classification)
    state.report["baseline_search"] = {"scope": "outer_fit_only", "status": "seed_only"}
    profile = state.report.get("quality_options", {}).get("search_profile", "legacy")
    startup_cap = 1800.0 if profile == "strong_v1" else 600.0
    timeout = min(startup_cap, max(0, state.remaining() - state.final_reserve) * .40)
    semantics_allowed = gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed", True)
    if not semantics_allowed:
        state.report["baseline_search"]["reason"] = "task_semantics_too_large"
    if semantics_allowed and gate.held_rows and timeout > 8:
        startup = state.run_dir / "startup"
        profile_option = f", search_profile={profile!r}" if profile != "legacy" else ""
        code = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
            "from pathlib import Path\nfrom agent.automl import run_automl\n"
            f"run_automl(Path({str(gate.work_task)!r}), time_budget={timeout - 2!r}, "
            f"validation={gate.choice!r}, output_dir=Path({str(startup)!r}), "
            f"metric_recipes={state.report.get('quality_options', {}).get('metric_recipes', False)!r}"
            f"{profile_option})\n")
        from agent.execution import run_python
        result = run_python(code, state.workspace, state.run_dir / "startup_execution", 0, timeout=timeout)
        gate.check_inputs()
        report_path = startup / "automl_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            recipe = _recipe_from_report(report, gate.spec.metric.task_type)
            state.report["baseline_search"] = {"scope": "outer_fit_only", "status": "searched",
                "search": report.get("search"), "execution": result}
        except (OSError, ValueError, KeyError, TypeError) as error:
            state.report["baseline_search"].update(reason=str(error), execution=result)
    source = state.run_dir / "baseline_source.py"
    source.write_text(BASELINE_SOURCE, encoding="utf-8")
    state.baseline = freeze_candidate(source, recipe, state.run_dir / "baseline_frozen")
    state.report["baseline"] = "recipe_frozen"
    state.save()


def _context(state):
    selection = {"metric": state.best_inner, "scope": "inner_selection",
                 "split": state.gate.inner_split}
    search_status = state.report.get("baseline_search", {}).get("status", "fixture_or_unavailable")
    history = state.report.get("history_research")
    history_context = (" Explicit host history INNER trial: " + json.dumps({key: history.get(key) for key in
        ("status", "reason", "history_config", "blend_weight", "candidate", "feedback")}, ensure_ascii=False)
        if isinstance(history, dict) and history.get("requested") else "")
    return ("The manifest is research_validation.json (version 3). Helper imports are available "
            f"with sys.path.insert(0, {str(ROOT)!r}). Read only the allowed files. "
            f"Baseline preparation status: {search_status}. A prepared baseline recipe is frozen "
            "before outer evaluation. All preparation measurements use the approved outer-fit "
            "view and are selection, not independent evaluation. "
            "No original test data or outer answers are supplied. Selected baseline recipe: "
            + json.dumps(state.baseline.recipe if state.baseline is not None else None, ensure_ascii=False)
            + ". Current INNER selection: " + json.dumps(selection, ensure_ascii=False)
            + ". Host-computed INNER errors: " + json.dumps(state.report.get("baseline_inner_diagnostics"), ensure_ascii=False)
            + history_context + ". Propose a fresh executable candidate, recipe and hypothesis; analysis-only calls "
            "do not count as measured trials. Finish after completing at least one measured trial.")


def _fresh_attempt(state):
    state.gate.check_inputs()
    state.gate.write_manifest()
    for name in (CANDIDATE_NAME, RECIPE_NAME, HYPOTHESIS_NAME, "submission.csv", "inner_predictions.csv"):
        path = state.workspace / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _consider(state):
    gate = state.gate
    gate.check_inputs()
    hypothesis = read_hypothesis(state.workspace)
    recipe_path = state.workspace / RECIPE_NAME
    if recipe_path.is_symlink() or recipe_path.stat().st_size > 64 * 1024:
        raise ValueError("Recipe must be a regular JSON object under 64 KiB")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    candidate = freeze_candidate(state.workspace / CANDIDATE_NAME, recipe,
                                 state.run_dir / "inner_best" / uuid4().hex)
    return _assess_candidate(state, candidate, hypothesis, origin="llm")


def _track_local_candidate(state, candidate):
    """Retain bounded frozen identities until optional research has been guarded."""
    attempts = getattr(state, "_local_frozen_attempts", None)
    if attempts is not None and not any(item is candidate for item in attempts):
        if len(attempts) >= MAX_LOCAL_FROZEN_ATTEMPTS:
            raise RuntimeError("Local research exceeded its frozen-attempt limit")
        attempts.append(candidate)


def _assess_candidate(state, candidate, hypothesis, *, origin):
    """One host-scored inner trial, shared by LLM and bounded local search."""
    _track_local_candidate(state, candidate)
    assessment_started = time.monotonic()
    gate, recipe = state.gate, candidate.recipe
    state.tried_recipes.append({"recipe": recipe, "code_hash": candidate.code_hash})
    result = state.measure_inner(candidate)
    state.gate.check_inputs()
    counter = ("successful_experiments" if origin == "llm" else
               "history_successful_experiments" if origin == "history" else "local_successful_experiments")
    state.report[counter] = state.report.get(counter, 0) + 1
    better = state.best_inner is None or _better(result.metric, state.best_inner)
    plan = state.finalization_plan(candidate)
    within_stage_limit = all(value <= 1800 for name, value in plan.stage_seconds.items()
                             if name.startswith("candidate_"))
    feasible = plan.estimated_seconds <= state.remaining() and within_stage_limit
    if better and feasible:
        state.selected_candidate, state.best_inner = candidate, result.metric
        state.report["improvements"] += 1
    state.report["evaluations"]["selection"] = {
        "metric": state.best_inner, "scope": "inner_selection", "split": state.gate.inner_split}
    feedback = {"accepted": better and feasible, "verified": True, "metric": result.metric,
            "reason": ("improved" if feasible else "refit_budget") if better else "not_better", "hypothesis": hypothesis,
            "scope": "inner_selection", "origin": origin, "code_hash": candidate.code_hash,
            "recipe_hash": candidate.recipe_hash,
            "recipe_summary": {"family": str(recipe.get("family", recipe.get("action", "custom")))[:120]},
            "timing": {"estimated_finalization_seconds": plan.estimated_seconds,
                       "remaining_seconds": max(0, state.remaining()), "feasible_estimate": feasible,
                       "within_stage_limit": within_stage_limit,
                       "guaranteed": False}}
    try:
        feedback["diagnostics"] = inner_feedback(gate.research_spec, result, gate.expected_inner_actuals)
    except (OSError, ValueError, TypeError) as error:
        # Optional detail cannot undo a valid host-scored trial.
        feedback["diagnostics"] = {"scope": "inner_selection", "available": False,
                                   "reason": safe_error_message(error)}
    elapsed = time.monotonic() - assessment_started
    if math.isfinite(elapsed) and elapsed > 0:
        feedback["elapsed_seconds"] = elapsed
    return feedback


def research_configured_history(state, options):
    """Measure one explicit history recipe on INNER, preserving the recovery baseline."""
    from agent.history_runtime import history_candidate_source, make_history_recipe

    gate = state.gate
    report = {"version": 1, "policy": "explicit_v1", "origin": "host_history",
              "scope": "inner_selection", "requested": options is not None,
              "active": False, "status": "disabled", "reason": "not_configured",
              "attempted": 0, "history_config": options["history_config"] if options else None,
              "blend_weight": options["blend_weight"] if options else 0.5,
              "baseline": None, "candidate": None, "feedback": None}
    state.report["history_research"] = report
    if options is None:
        return report

    def skip(reason):
        report.update(status="skipped", reason=reason)
        state.report["extensions"]["history"] = "skipped"
        state.save()
        return report

    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return skip("research_closed")
    if gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return skip("task_semantics_unavailable")
    if gate.spec.metric.task_type != "regression":
        return skip("regression_required")
    config = options["history_config"]
    if not set([*config["key_columns"], config["period_column"]]) <= set(gate.spec.features):
        return skip("history_columns_not_in_features")
    if gate.choice.get("strategy") != "time":
        return skip("time_validation_required")
    if gate.choice.get("time_column") != config["period_column"]:
        return skip("period_column_mismatch")
    if not gate.fit_rows or not gate.held_rows or not gate.inner_split:
        return skip("inner_validation_unavailable")
    if (gate.inner_split.get("strategy") != "time"
            or gate.inner_split.get("time_column") != config["period_column"]):
        return skip("inner_time_validation_required")
    if options["blend_weight"] == 0:
        return skip("zero_weight")
    if state.baseline is None or state.baseline_inner is None:
        return skip("measured_baseline_required")
    if state.research_timeout() <= 2:
        return skip("finalization_reserve")
    state.check_source()
    gate.check_inputs()
    baseline = state.baseline
    baseline.verify()
    identity = (baseline.code_hash, baseline.recipe_hash)
    report["baseline"] = {"code_hash": identity[0], "recipe_hash": identity[1]}
    report.update(active=True, status="running", reason=None, attempted=1)
    state.report["extensions"]["history"] = "explicit_v1"
    state.save()
    try:
        source = state.run_dir / "history_source.py"
        source.write_text(history_candidate_source(baseline.code_path.read_text(encoding="utf-8")),
                          encoding="utf-8", newline="\n")
        recipe = make_history_recipe(baseline.recipe, config, options["blend_weight"])
        candidate = freeze_candidate(source, recipe, state.run_dir / "history_frozen")
        report["candidate"] = {"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash}
        feedback = _assess_candidate(state, candidate, {
            "hypothesis": "Blend the frozen baseline with strictly past target history using "
                          "an explicit key/period configuration and a fixed weight."}, origin="history")
        report.update(status="completed", reason=feedback["reason"], feedback=feedback)
        state.event("history_experiment", result=feedback)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        report.update(status="failed", reason="candidate_execution_failed", error=safe_error_message(error))
        if isinstance(error, CandidateExecutionError):
            report["execution"] = error.execution
        state.event("history_experiment_failed", error=safe_error_message(error))
    # Candidate failure must never swallow source mutation or replace baseline.
    state.check_source()
    gate.check_inputs()
    if state.baseline is not baseline or (baseline.code_hash, baseline.recipe_hash) != identity:
        raise RuntimeError("History research changed the frozen recovery baseline")
    baseline.verify()
    state.save()
    return report


def continue_locally(state):
    """Optional research may fail; retained recipe integrity may not fail.

    Planning, proposal and presentation errors must not bypass the normal
    outer/refit publication stages. This boundary intentionally excludes
    finalization and BaseException. Every integrity check remains outside the
    optional-operation catch, including when that operation raised an error.
    """
    baseline = state.baseline
    identity = (baseline.code_hash, baseline.recipe_hash) if baseline is not None else None

    def integrity():
        try:
            if getattr(state, "_local_integrity_failed", False):
                raise RuntimeError("Local research integrity previously failed")
            if state.dialogue_closed or state.report.get("stage") != "inner_research":
                raise RuntimeError("Local research is terminal after candidate freeze")
            state.check_source()
            state.gate.check_inputs()
            if state.baseline is not baseline or (baseline is not None
                    and (baseline.code_hash, baseline.recipe_hash) != identity):
                raise RuntimeError("Local research changed its frozen recovery baseline")
            if baseline is not None:
                baseline.verify()
            if state.selected_candidate is not None:
                state.selected_candidate.verify()
            for candidate in getattr(state, "_local_frozen_attempts", ()):
                candidate.verify()
        except Exception:
            # A later passing verifier cannot erase a known protocol breach.
            # This marker is host memory only and is not supplied to workers.
            state._local_integrity_failed = True
            raise

    integrity()
    if getattr(state, "_local_frozen_attempts", None) is not None:
        state._local_integrity_failed = True
        raise RuntimeError("Local continuation cannot be re-entered")
    state._local_frozen_attempts = []
    error_type = None
    try:
        result = _continue_locally(state)
    except Exception as error:
        error_type = type(error).__name__
    integrity()
    del state._local_frozen_attempts
    if error_type is None:
        return result
    report = state.report.get("local_research")
    report = report if isinstance(report, dict) else {
        "origin": "host_local", "scope": "inner_selection", "api_requests": 0, "attempted": 0}
    report.update(status="interrupted", reason="optional_search_error", error_type=error_type,
                  finalization_requested=True)
    state.report["local_research"] = report
    state.event("local_research_interrupted", error_type=error_type, finalization_requested=True)
    return report


def _early_extensions_allowed(gate):
    """Prioritize temporal regression; other tasks retain the measured incumbent route."""
    inner = gate.inner_split or {}
    period = gate.choice.get("time_column")
    return (gate.spec.metric.task_type == "regression"
            and gate.choice.get("strategy") == "time"
            and isinstance(period, str) and bool(period)
            and period in gate.spec.features
            and inner.get("strategy") == "time"
            and inner.get("time_column") == period)


def _continue_locally(state):
    """Use spare CPU after dialogue stops without reopening its frozen baseline."""
    from agent.offline_research import run_deadline_continuation, run_local_continuation, startup_seen
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        raise RuntimeError("Local research is terminal after candidate freeze")
    gate = state.gate
    search = state.report.get("baseline_search", {}).get("search") or {}
    ranked = []
    completed = [row for row in search.get("candidates", []) if row.get("selection_metric") is not None]
    completed.sort(key=lambda row: row["selection_metric"], reverse=gate.spec.metric.higher_is_better)
    for row in completed:
        try:
            item = _recipe_from_report(row, gate.spec.metric.task_type)
            if "components" not in item:
                ranked.append(item)
        except (KeyError, ValueError, TypeError):
            continue
    recipes = local_candidate_plan(gate.spec.classification, gate.spec.metric.name,
                                  gate.research_train.iloc[gate.fit_rows][gate.spec.target_column], ranked,
                                  metric_recipes=state.report.get("quality_options", {}).get("metric_recipes", False))
    if state.report.get("quality_options", {}).get("search_profile", "legacy") == "strong_v1":
        from agent.tabular_backend import strong_candidate_plan
        recipes = [*strong_candidate_plan(gate.spec.classification, gate.spec.metric.name,
                    gate.research_train.iloc[gate.fit_rows][gate.spec.target_column]), *recipes]
    source = state.run_dir / "local_source.py"
    source.write_text(BASELINE_SOURCE, encoding="utf-8")
    import hashlib
    code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    seen = [*startup_seen(search, code_hash=code_hash), *state.tried_recipes,
            {"recipe": state.baseline.recipe, "code_hash": state.baseline.code_hash}]

    def evaluate(recipe):
        frozen = freeze_candidate(source, recipe, state.run_dir / "local_candidates" / uuid4().hex)
        _track_local_candidate(state, frozen)
        label = recipe.get("label")
        if not isinstance(label, str) or not label.strip():
            label = recipe.get("action")
        if not isinstance(label, str) or not label.strip():
            label = "fixed blend" if recipe.get("components") else "local model"
        feedback = _assess_candidate(state, frozen,
            {"hypothesis": "Bounded host-local alternative after research dialogue stopped: "
             + label.strip()[:160]}, origin="local")
        state.event("local_experiment", result=feedback)
        return feedback

    if getattr(state, "adaptive_search", False):
        early_extensions = _early_extensions_allowed(gate)
        try:
            from agent.adaptive_search import AdaptiveSearch
            adaptive = AdaptiveSearch(gate.spec.classification, gate.spec.metric.name, max_trials=64)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            adaptive = None
            state.event("adaptive_search_unavailable", error_type=type(error).__name__)

        def extensions():
            from agent.research_extensions import run_research_extensions
            return run_research_extensions(state, _assess_candidate, max_trials=6,
                                           prioritize=early_extensions)

        return run_deadline_continuation(state, recipes, adaptive=adaptive, evaluate=evaluate,
            seen=seen, code_hash=code_hash, after_warmup=extensions, max_trials=64,
            extensions_first=early_extensions)
    return run_local_continuation(state, recipes, evaluate=evaluate, seen=seen, code_hash=code_hash)


def _outer_evidence(state, result, actuals):
    """Legacy-compatible evidence is host-authored only AFTER dialogue closes."""
    spec = state.gate.spec
    frame = pd.read_csv(BytesIO(result.read_output()), dtype=str, keep_default_na=False)
    values = frame.loc[:, list(spec.prediction_columns)].to_numpy()
    classes = [label for _, label in spec.probability_columns] if spec.prediction_kind == "probabilities" else None
    if spec.prediction_kind != "probabilities" or len(spec.prediction_columns) == 1:
        values = values[:, 0]
        if spec.prediction_kind == "probabilities":
            classes = None
    return spec.holdout_frame(state.gate.expected_ids, actuals, values, prediction_classes=classes)


FINAL_FAILURE_MESSAGE_LIMIT = 1024
FINAL_FAILURE_STDERR_LIMIT = 4096


def _failure_text(value, limit, *, tail=False):
    """Bound and redact host-only failure details before saving or printing."""
    error = value if isinstance(value, Exception) else RuntimeError(str(value))
    text = safe_error_message(error)
    if len(text) <= limit:
        return text
    marker = "[truncated]"
    return (marker + "\n" + text[-(limit - len(marker) - 1):] if tail
            else text[:limit - len(marker) - 1] + "\n" + marker)


def _finalization_cause(error, *, phase, recipe):
    """Keep terminal diagnostics out of the already-closed research dialogue."""
    cause = {"phase": phase, "recipe": recipe, "error_type": type(error).__name__,
             "message": _failure_text(error, FINAL_FAILURE_MESSAGE_LIMIT)}
    if isinstance(error, CandidateExecutionError) and isinstance(error.execution, dict):
        execution = error.execution
        detail = {"exit_code": execution.get("exit_code")
                  if type(execution.get("exit_code")) is int else None,
                  "timed_out": execution.get("timed_out") is True,
                  "memory_limit_exceeded": execution.get("memory_limit_exceeded") is True}
        for source, destination, limit, tail in (
                ("error", "error", FINAL_FAILURE_MESSAGE_LIMIT, False),
                ("stderr", "stderr_tail", FINAL_FAILURE_STDERR_LIMIT, True)):
            if isinstance(execution.get(source), str) and execution[source]:
                detail[destination] = _failure_text(execution[source], limit, tail=tail)
        cause["execution"] = detail
    return cause


def finalize(state, reason):
    """Once entered, this function cannot request the model or select a new recipe."""
    state.dialogue_closed = True
    state.check_source()
    state.gate.check_inputs()
    state.transition("candidate_frozen")
    candidate = state.selected_candidate
    for item in (state.baseline, candidate):
        if item is not None:
            item.verify()
    state.report["frozen_candidate"] = ({"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash}
        if candidate is not None else None)
    state.report["frozen_baseline"] = ({"code_hash": state.baseline.code_hash, "recipe_hash": state.baseline.recipe_hash}
        if state.baseline is not None else None)
    state.transition("outer_validation")
    gate = state.gate
    results, errors = {}, {}
    terminal_causes = []
    completed_stages = []

    def stage_timeout(stage):
        plan = state.finalization_plan()
        state.report["time_allocation"] = state.budget.report(plan)
        allocations = state.budget.finalization_timeouts(plan, completed=completed_stages)
        timeout = allocations.get(stage, 0)
        state.report.setdefault("final_stage_budgets", {})[stage] = {
            "timeout_seconds": timeout, "remaining_seconds": state.budget.remaining(),
            "pending_allocations": allocations, "completed_stages": list(completed_stages)}
        if timeout <= 0:
            raise TimeoutError("No time remains for " + stage + " after reserving the fallback")
        return timeout

    # Both recipes were fixed before seeing either outer score. Outer acceptance
    # chooses between them, so the winner's score is NOT an unbiased estimate.
    for name, frozen in (("baseline", state.baseline), ("candidate", candidate)):
        if frozen is None:
            continue
        try:
            timeout = stage_timeout(name + "_outer")
            started = time.monotonic()
            result = run_candidate(frozen, spec=gate.spec, fit_frame=gate.outer_fit_frame,
                predict_frame=gate.outer_holdout_frame.drop(columns=[gate.spec.target_column]),
                actuals=gate.expected_actuals, phase="outer_validation", timeout=timeout,
                deadline=min(started + timeout, state.budget.deadline - state.budget.publication_buffer),
                output_dir=state.run_dir / "outer_evaluations" / name)
            gate.check_inputs()
            state.record_timing(frozen, started, len(gate.outer_fit_frame), len(gate.outer_holdout_frame))
            results[name] = result
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            errors[name] = safe_error_message(error)
            terminal_causes.append(_finalization_cause(error, phase="outer_validation", recipe=name))
            if isinstance(error, CandidateExecutionError):
                state.event("outer_execution_failed", recipe=name, execution=error.execution)
        finally:
            completed_stages.append(name + "_outer")
            if name not in results:
                completed_stages.append(name + "_refit")
    selected = "baseline"
    if "candidate" in results and ("baseline" not in results
            or _better(results["candidate"].metric, results["baseline"].metric)):
        selected = "candidate"
    final_check = {"baseline": results["baseline"].metric if "baseline" in results else None,
        "candidate": results["candidate"].metric if "candidate" in results else None,
        "selected": selected, "selection_uses_outer": True, "unbiased_estimate": False,
        "scope": "final_check", "split": gate.split, "errors": errors,
        "execution": {name: value.execution for name, value in results.items()},
        "candidate_acceptance": "improved" if selected == "candidate" else (
            "not_better" if "candidate" in results else "unavailable")}
    state.report["evaluations"]["final_check"] = final_check
    if not results:
        state.report["stages"][3].update(status="failed", reason="No valid outer predictions")
    state.save()
    state.transition("full_refit")
    frozen_by_name = {"baseline": state.baseline, "candidate": candidate}
    # If the admitted candidate cannot refit, the already fixed baseline is a
    # recovery recipe, not a new search informed by outer feedback.
    order = (["baseline", "candidate"] if selected == "candidate" and "baseline" in results else [selected])
    for name in ("baseline", "candidate"):
        stage = name + "_refit"
        if name not in order and stage in state.finalization_plan().stage_seconds and stage not in completed_stages:
            completed_stages.append(stage)
    published = False
    for name in order:
        if name not in results:
            continue
        try:
            timeout = stage_timeout(name + "_refit")
            started = time.monotonic()
            result = run_candidate(frozen_by_name[name], spec=gate.spec, fit_frame=gate.train,
                predict_frame=gate.test, phase="full_refit", timeout=timeout,
                deadline=min(started + timeout, state.budget.deadline - state.budget.publication_buffer),
                output_dir=state.run_dir / "full_refit" / name)
            gate.check_inputs()
            state.record_timing(frozen_by_name[name], started, len(gate.train), len(gate.test))
            temporary = result.output_path.parent / "publication"
            temporary.mkdir()
            result.copy_output(temporary / "submission.csv")
            evidence = temporary / "holdout_predictions.csv"
            _outer_evidence(state, results[name], gate.expected_actuals).to_csv(evidence, index=False)
            if not state.checkpoint(temporary, results[name].metric):
                raise ValueError("Refit output could not be published")
            state.report["published_recipe"] = name
            state.report["baseline"] = "trained"
            state.report["candidate_receipts"] = {
                "outer": str(results[name].receipt_path), "refit": str(result.receipt_path),
                "code_hash": result.code_hash, "recipe_hash": result.recipe_hash}
            state.report.setdefault("refit_execution", {})[name] = result.execution
            published = True
            try:
                _atomic_copy(evidence, state.run_dir / "best_holdout_predictions.csv")
            except OSError as error:
                # The checked evidence remains in this invocation. Failure of
                # the convenience export cannot undo an already committed CSV.
                state.event("evidence_export_unavailable", error=safe_error_message(error))
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            terminal_causes.append(_finalization_cause(error, phase="full_refit", recipe=name))
            state.event("refit_failed", recipe=name, error=safe_error_message(error),
                        execution=error.execution if isinstance(error, CandidateExecutionError) else None)
        finally:
            completed_stages.append(name + "_refit")
    state.check_source()
    if not published and not state.recover():
        report_path = _failure_text(state.run_dir / "run.json", FINAL_FAILURE_MESSAGE_LIMIT)
        state.report["finalization_failure"] = {
            "reason": "no_verified_output_or_recovery", "causes": terminal_causes,
            "report_path": report_path,
        }
        detail = ""
        if terminal_causes:
            last = terminal_causes[-1]
            message = last["message"]
            execution = last.get("execution", {})
            if execution.get("stderr_tail", "").strip():
                message += "; child stderr: " + execution["stderr_tail"].strip().splitlines()[-1]
            elif execution.get("error"):
                message += "; execution error: " + execution["error"]
            detail = (f" Last {last['phase']} failure ({last['recipe']}): "
                      + _failure_text(message, FINAL_FAILURE_MESSAGE_LIMIT))
        raise RuntimeError("No verified full-data predictions; no successful output can be reported."
                           + detail + " Diagnostics: " + report_path) from None
    if not published:
        reason = "refit_failed"
    elif state.report.get("published_recipe") != selected:
        reason = "refit_failed"
    if not published:
        state.report["stages"][4].update(
            status="failed" if results else "skipped", reason="refit_failed" if results else "No admissible outer result")
    state.transition("result")
    state.report["stages"][-1]["status"] = "completed"
    state.report["best_metric_provenance"] = ({
        "scope": "final_check", "selection_uses_outer": True, "unbiased_estimate": False}
        if published else {"scope": "existing_output", "quality_verified": False})
    state.report["publication"] = {"new_result_published": published,
                                  "recipe": state.report.get("published_recipe"), "reason": reason}
    state.report["time_allocation"] = state.budget.report(state.finalization_plan())
    status = "completed" if ((reason == "model_finished" and state.report["successful_experiments"])
                              or (reason == "local_only" and published)
                              or (getattr(state, "adaptive_search", False) and published)) else "fallback"
    return state.finish(status, reason)


def _retryable_api_error(error):
    from agent.gateways import retryable_api_error

    return retryable_api_error(error)


def _request_completion(state, client, messages):
    """Charge every actual request to the original guards, including one retry."""
    from agent.gateways import client_gateways, request_options

    pending_retry = False
    recovery = state.report["api_recovery"]
    routes = client_gateways(client)
    route_index = getattr(state, "_gateway_route_index", 0)
    if not 0 <= route_index < max(1, len(routes)):
        route_index = 0
    gateway_report = state.report.setdefault("api_gateway", {
        "policy": "official_alternative_first" if len(routes) == 2 else "configured_only",
        "active": routes[route_index][0] if routes else "embedded_client",
        "attempts": {}, "switches": 0,
    })
    while True:
        request_messages = [*messages, {"role": "user", "content":
            "Current host budget: " + json.dumps(state.budget_context(), ensure_ascii=False, allow_nan=False)}]
        blocked = state.request_allowed(request_messages)
        timeout = state.research_timeout(120) if not blocked else 0
        if not blocked and timeout <= 0:
            blocked = "time_limit"
        if blocked:
            if pending_retry:
                state.event("api_retry_skipped", reason=blocked)
            return None, "api_error" if pending_retry else blocked
        state.report["api"]["attempts"] += 1
        route = routes[route_index] if routes else None
        route_name = route[0] if route is not None else "embedded_client"
        if gateway_report["active"] != route_name:
            gateway_report["switches"] += 1
            state.event("api_gateway_failover", source=gateway_report["active"], destination=route_name)
        gateway_report["active"] = route_name
        gateway_report["attempts"][route_name] = gateway_report["attempts"].get(route_name, 0) + 1
        if pending_retry:
            recovery["retries"] += 1
            state.event("api_retry", attempt=state.report["api"]["attempts"],
                        retry_number=recovery["retries"])
        state.save()
        try:
            response = client.with_options(**request_options(timeout, route)).chat.completions.create(
                model=get_model(), messages=state.clean(request_messages), tools=[TOOL],
                parallel_tool_calls=False, max_tokens=MAX_COMPLETION_TOKENS)
        except Exception as error:
            state.event("api_error", error=safe_error_message(error), details=safe_error_details(error))
            if not pending_retry and recovery["retries"] < MAX_API_RETRIES and _retryable_api_error(error):
                if len(routes) == 2:
                    route_index = 1 - route_index
                    state._gateway_route_index = route_index
                pending_retry = True
                continue
            return None, "api_error"
        state.report["api"]["responses"] += 1
        try:
            record_usage(state, response)
            completion = checked_completion(response)
        except Exception as error:
            # The provider already returned a response; malformed paid output
            # does not justify repeating the request as a transport failure.
            state.event("api_error", error=safe_error_message(error), details=safe_error_details(error))
            return None, "api_error"
        if pending_retry:
            state.event("api_retry_recovered", attempt=state.report["api"]["attempts"])
        return completion, None


def _recover_after_unexpected_error(state):
    """Restore only an accepted checkpoint after rechecking every integrity gate."""
    if getattr(state, "_local_integrity_failed", False):
        raise RuntimeError("Local research integrity failed; checkpoint recovery cancelled")
    state.check_source()
    if state.gate is None:
        return None
    state.gate.check_inputs()
    for candidate in (state.baseline, state.selected_candidate):
        if candidate is not None:
            candidate.verify()
    for candidate in getattr(state, "_local_frozen_attempts", ()):
        candidate.verify()
    if state._accepted_backup_hash is None or not state.recover():
        return None
    state.report["recovery"] = {"status": "restored", "kind": "accepted_checkpoint",
                                "submission_sha256": state._accepted_backup_hash}
    state.report["publication"] = {"new_result_published": False,
                                   "reason": "unexpected_error", "recovered_checkpoint": True}
    state.report["best_metric_provenance"] = {
        "scope": "accepted_checkpoint", "quality_verified": state.report["best_metric"] is not None}
    state.report["stage"] = "result"
    state.report["stages"][-1]["status"] = "completed"
    state.report["events"].append({"kind": "unexpected_error_recovered"})
    return state.finish("fallback", "unexpected_error")


def run_agent_v3(task_path, client, *, max_steps=12, time_budget=10500,
                 stability_splits=1, metric_recipes=False, search_profile="legacy",
                 history_config=None, history_weight=0.5, local_only=False, history_auto=False,
                 adaptive_search=False, rescue_regression=False):
    if (type(max_steps) is not int or max_steps < 1 or type(time_budget) not in {int, float}
            or not math.isfinite(time_budget) or not 0 < time_budget <= 10500):
        raise ValueError("Invalid research request/time budget")
    if type(stability_splits) is not int or stability_splits not in {1, 2, 3} or type(metric_recipes) is not bool:
        raise ValueError("Invalid quality experiment options")
    if not isinstance(search_profile, str) or search_profile not in {"legacy", "strong_v1"}:
        raise ValueError("Unknown research search profile")
    if type(local_only) is not bool:
        raise ValueError("local_only must be a boolean")
    if type(history_auto) is not bool:
        raise ValueError("history_auto must be a boolean")
    if type(adaptive_search) is not bool:
        raise ValueError("adaptive_search must be a boolean")
    if type(rescue_regression) is not bool:
        raise ValueError("rescue_regression must be a boolean")
    if rescue_regression and (not adaptive_search or time_budget <= ADAPTIVE_RESERVE_MIN_BUDGET):
        raise ValueError("rescue_regression requires v3 adaptive search and a budget greater than 300 seconds")
    if history_auto and (history_config is not None or history_weight != 0.5):
        raise ValueError("Automatic history uses its own config and fixed weight 0.5")
    from agent.history_runtime import normalize_history_options
    history_options = normalize_history_options(history_config, history_weight)
    state = (V3State(Path(task_path).resolve(), max_steps, time_budget, adaptive_search=True)
             if adaptive_search else V3State(Path(task_path).resolve(), max_steps, time_budget))
    state.report["quality_options"] = {"stability_splits": stability_splits,
        "metric_recipes": metric_recipes, "search_profile": search_profile}
    if history_options is not None:
        state.report["quality_options"].update(history_options)
    if history_auto:
        state.report["quality_options"]["history_auto"] = True
    if rescue_regression:
        state.report["quality_options"]["rescue_regression"] = True
    if adaptive_search:
        state.report["quality_options"]["adaptive_search"] = True
        state.report["coordinator"] = {"version": 2, "requested": True, "active": state.adaptive_search,
            "reason": "enabled" if state.adaptive_search else "short_budget",
            "minimum_long_budget_seconds": ADAPTIVE_RESERVE_MIN_BUDGET,
            "local_trial_limit": 64, "extension_trial_limit": 6,
            "extension_policy": "early_temporal_regression_only",
            "api_request_limit": max_steps}
    if local_only:
        state.report["mode"] = "local_only"
    try:
        prepare_research(state)
        if rescue_regression:
            from agent.research_extensions import (
                RESCUE_REGRESSION_MIN_SECONDS, _rescue_regression_eligibility,
            )
            reason = _rescue_regression_eligibility(state)
            state.report["rescue_regression"] = {
                "requested": True, "replacement_slot": "boost_catboost",
                "status": "skipped" if reason else "pending", "reason": reason,
                "attempted": False, "history_schema": "not_checked",
                "minimum_research_seconds": RESCUE_REGRESSION_MIN_SECONDS,
                "time_guard_kind": "heuristic_admission_floor", "runtime_guaranteed": False,
                "refit_cost_status": "unknown_until_host_measurement",
            }
        prepare_baseline(state)
        semantics_allowed = state.gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed", True)
        if semantics_allowed and state.gate.held_rows and state.baseline is not None:
            try:
                measured = state.measure_inner(state.baseline)
                state.baseline_inner = measured.metric
                state.best_inner = state.baseline_inner
                try:
                    state.report["baseline_inner_diagnostics"] = inner_feedback(
                        state.gate.research_spec, measured, state.gate.expected_inner_actuals)
                except (OSError, ValueError, TypeError) as error:
                    state.event("inner_diagnostics_unavailable", error=safe_error_message(error))
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                state.event("inner_baseline_unavailable", error=safe_error_message(error))
        state.report["evaluations"]["selection"] = {
            "metric": state.best_inner, "scope": "inner_selection", "split": state.gate.inner_split}
        state.transition("inner_research")
        if history_auto:
            from agent.history_discovery_host import discover_history_options
            history_options = discover_history_options(state)
        research_configured_history(state, history_options)
        if not state.gate.held_rows:
            state.report["stages"][1].update(status="skipped", reason=state.gate.inner_unavailable_reason)
            return finalize(state, "inner_validation_unavailable")
        if state.gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
            reason = state.gate.manifest["task_semantics"].get("fallback_reason", "task_semantics_unavailable")
            state.report["stages"][1].update(status="skipped", reason=reason)
            return finalize(state, reason)
        if local_only:
            state.report["research_stop"] = {"reason": "local_only", "local_continuation_requested": True}
            continue_locally(state)
            if stability_splits > 1:
                from agent.research_stability import verify_research_finalists
                verify_research_finalists(state, splits=stability_splits)
            return finalize(state, "local_only")
        messages = [{"role": "system", "content": SYSTEM_V3},
                    {"role": "user", "content": state.gate.task_text},
                    {"role": "user", "content": state.redact(_context(state))}]
        reason = "step_limit"
        for step in range(1, max_steps + 1):
            # This host snapshot is refreshed, not appended to accumulated
            # history. It contains only inner information and measured costs.
            completion, stopped = _request_completion(state, client, messages)
            if stopped:
                reason = stopped
                break
            if completion["finish_reason"] == "length":
                reason = "truncated_response"
                break
            calls = completion["tool_calls"]
            if not calls:
                if state.report["successful_experiments"]:
                    reason = "model_finished"
                    break
                messages.extend([{"role": "assistant", "content": completion["content"] or "Finished."},
                    {"role": "user", "content": "Research is not complete: test a hypothesis with inner validation. "
                     "Write fresh research_candidate.py, research_recipe.json and research_hypothesis.json in one tool call."}])
                continue
            messages.append({"role": "assistant", "content": completion["content"],
                             "tool_calls": [call["history"] for call in calls]})
            for index, call in enumerate(calls, 1):
                code = ""
                try:
                    if call["name"] != "run_python" or not isinstance(call["arguments"], str):
                        raise ValueError("run_python requires JSON string arguments")
                    arguments = json.loads(call["arguments"])
                    if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str) or not arguments["code"].strip():
                        raise ValueError("run_python requires exactly one non-empty code string")
                    code = arguments["code"]
                    if state.ledger.is_repeated_failure(code):
                        raise ValueError("This code already failed; change the approach")
                    _fresh_attempt(state)
                    from agent.execution import run_python
                    timeout = state.research_timeout()
                    if timeout <= 0:
                        raise TimeoutError("Research time is reserved for final validation and refit")
                    result = run_python(code, state.workspace, state.run_dir,
                        step * 100 + index, timeout=timeout)
                    state.report["experiments"] += 1
                except (ValueError, OSError, TimeoutError) as error:
                    result = {"error": safe_error_message(error)}
                failed = result.get("exit_code") != 0 or result.get("error") or result.get("timed_out") or result.get("memory_limit_exceeded")
                if not failed:
                    try:
                        result["selection"] = _consider(state)
                    except CandidateExecutionError as error:
                        result["candidate_execution"] = error.execution
                        result["error"] = safe_error_message(error)
                        result["selection"] = {"accepted": False, "verified": False,
                            "scope": "inner_selection", "reason": "candidate_execution_failed"}
                    except (OSError, ValueError, RuntimeError, UnicodeError) as error:
                        result["selection"] = {"accepted": False, "verified": False,
                            "scope": "inner_selection", "reason": safe_error_message(error)}
                else:
                    result["selection"] = {"accepted": False, "verified": False, "reason": "execution_failed", "scope": "inner_selection"}
                state.check_source()
                if state.backup.is_file():
                    state.recover()
                state.ledger.record(code, result, remaining=state.remaining())
                state.event("experiment", step=step, tool_index=index, result=result)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(state.clean(result), ensure_ascii=False, allow_nan=False)})
            if state.ledger.failure_streak >= 3:
                reason = "repeated_failures"
                break
        continuation_reasons = (ADAPTIVE_CONTINUATION_REASONS if state.adaptive_search
                                else LOCAL_CONTINUATION_REASONS)
        state.report["research_stop"] = {
            "reason": reason, "local_continuation_requested": reason in continuation_reasons,
        }
        state.save()
        if reason in continuation_reasons:
            continue_locally(state)
        if stability_splits > 1:
            from agent.research_stability import verify_research_finalists
            verify_research_finalists(state, splits=stability_splits)
        return finalize(state, reason)
    except Exception as error:
        state.report.update(status="failed", error=safe_error_message(error))
        for stage in state.report.get("stages", []):
            if stage["status"] == "running":
                stage.update(status="failed", reason=safe_error_message(error))
        try:
            state.save()
        except (OSError, ValueError, TypeError):
            pass
        try:
            recovered = _recover_after_unexpected_error(state)
            if recovered is not None:
                return recovered
        except Exception as recovery_error:
            state.report["recovery"] = {"status": "rejected", "error": safe_error_message(recovery_error)}
            try:
                state.save()
            except (OSError, ValueError, TypeError):
                pass
        raise
