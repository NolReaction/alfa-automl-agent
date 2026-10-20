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
from uuid import uuid4

import pandas as pd

from agent.automl import _atomic_copy
from agent.features import normalize_feature_config
from agent.llm import ROOT, TOOL, get_model, safe_error_message
from agent.model_config import ACTIONS, make_model_config
from agent.research import RunState, checked_completion, record_usage
from agent.research_candidate import (
    CANDIDATE_NAME, RECIPE_NAME, CandidateExecutionError, freeze_candidate, run_candidate,
)
from agent.research_validation import HYPOTHESIS_NAME, read_hypothesis
from agent.research_workspace import ResearchWorkspace
from agent.search import seed_candidate
from validation import validate_predictions


STAGES = ("preparation", "inner_research", "candidate_frozen", "outer_validation", "full_refit", "result")
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
    def __init__(self, task_path, max_steps, time_budget):
        super().__init__(task_path, max_steps, time_budget)
        self.gate = None
        self.baseline = None
        self.baseline_inner = None
        self.selected_candidate = None
        self.best_inner = None
        self.dialogue_closed = False
        # Fixed before research: expensive outer fits/refit cannot consume the
        # time granted to the next model request. Minimum reserve fits toy jobs.
        self.final_reserve = min(time_budget * .5, max(15.0, time_budget * .30))
        self.report.update(research_protocol_version=3, stage="preparation",
            stages=[{"id": item, "status": "pending"} for item in STAGES],
            evaluations={"selection": None, "final_check": None},
            isolation={"kind": "allowed_input_view", "os_sandbox": False},
            extensions={"policy": "not_connected", "portfolio": "not_connected", "history": "disabled"})
        self.transition("preparation")

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

    def measure_inner(self, candidate):
        gate = self.gate
        gate.check_inputs()
        if not gate.held_rows:
            raise ValueError("Inner validation unavailable; no adaptive trial is allowed")
        return run_candidate(candidate, spec=gate.research_spec,
            fit_frame=gate.research_train.iloc[gate.fit_rows],
            predict_frame=gate.research_train.iloc[gate.held_rows].drop(columns=[gate.spec.target_column]),
            actuals=gate.expected_inner_actuals, phase="inner_selection",
            output_dir=self.run_dir / "inner_evaluations",
            timeout=min(1800, self.remaining() - self.final_reserve))

    def request_allowed(self, messages):
        if self.dialogue_closed:
            raise RuntimeError("Research is terminal after candidate freeze")
        if self.report["api"]["attempts"] >= self.report["max_steps"]:
            return "step_limit"
        if self.remaining() <= self.final_reserve + 2:
            return "time_limit"
        payload = json.dumps(self.clean(messages), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # A host-side estimate, explicitly not provider usage or currency.
        estimate = math.ceil(len(payload) / 3) + 4096
        budget = self.report.setdefault("request_guard", {
            "estimated_tokens_reserved": 0, "estimated_token_limit": 100000,
            "estimation": "ceil(UTF-8 bytes / 3) + 4096 output reserve per request"})
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
    if validate_predictions(state.task_dir)["valid"]:
        state.checkpoint(state.task_dir)
        state.report["baseline"] = "existing_output"
    state.save()


def prepare_baseline(state):
    """Run the existing search ONLY on the permitted research data."""
    gate = state.gate
    recipe = seed_candidate(gate.spec.classification)
    state.report["baseline_search"] = {"scope": "outer_fit_only", "status": "seed_only"}
    timeout = min(600.0, max(0, state.remaining() - state.final_reserve) * .40)
    if gate.held_rows and timeout > 8:
        startup = state.run_dir / "startup"
        code = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
            "from pathlib import Path\nfrom agent.automl import run_automl\n"
            f"run_automl(Path({str(gate.work_task)!r}), time_budget={timeout - 2!r}, "
            f"validation={gate.choice!r}, output_dir=Path({str(startup)!r}))\n")
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
    return ("The manifest is research_validation.json (version 3). Helper imports are available "
            f"with sys.path.insert(0, {str(ROOT)!r}). Read only the allowed files. "
            f"Baseline preparation status: {search_status}. A prepared baseline recipe is frozen "
            "before outer evaluation. All preparation measurements use the approved outer-fit "
            "view and are selection, not independent evaluation. "
            "No original test data or outer answers are supplied. Selected baseline recipe: "
            + json.dumps(state.baseline.recipe if state.baseline is not None else None, ensure_ascii=False)
            + ". Current INNER selection: " + json.dumps(selection, ensure_ascii=False)
            + ". Propose a fresh executable candidate, recipe and hypothesis; analysis-only calls "
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
    state.gate.check_inputs()
    hypothesis = read_hypothesis(state.workspace)
    recipe_path = state.workspace / RECIPE_NAME
    if recipe_path.is_symlink() or recipe_path.stat().st_size > 64 * 1024:
        raise ValueError("Recipe must be a regular JSON object under 64 KiB")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    candidate = freeze_candidate(state.workspace / CANDIDATE_NAME, recipe,
                                 state.run_dir / "inner_best" / uuid4().hex)
    result = state.measure_inner(candidate)
    state.gate.check_inputs()
    state.report["successful_experiments"] += 1
    better = state.best_inner is None or _better(result.metric, state.best_inner)
    if better:
        state.selected_candidate, state.best_inner = candidate, result.metric
        state.report["improvements"] += 1
    state.report["evaluations"]["selection"] = {
        "metric": state.best_inner, "scope": "inner_selection", "split": state.gate.inner_split}
    return {"accepted": better, "verified": True, "metric": result.metric,
            "reason": "improved" if better else "not_better", "hypothesis": hypothesis,
            "scope": "inner_selection", "code_hash": candidate.code_hash,
            "recipe_hash": candidate.recipe_hash,
            "recipe_summary": {"family": str(recipe.get("family", recipe.get("action", "custom")))[:120]}}


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
    # Both recipes were fixed before seeing either outer score. Outer acceptance
    # chooses between them, so the winner's score is NOT an unbiased estimate.
    for name, frozen in (("baseline", state.baseline), ("candidate", candidate)):
        if frozen is None:
            continue
        try:
            timeout = min(1800.0, state.remaining() * (.30 if name == "baseline" and candidate else .45))
            result = run_candidate(frozen, spec=gate.spec, fit_frame=gate.outer_fit_frame,
                predict_frame=gate.outer_holdout_frame.drop(columns=[gate.spec.target_column]),
                actuals=gate.expected_actuals, phase="outer_validation", timeout=timeout,
                output_dir=state.run_dir / "outer_evaluations" / name)
            gate.check_inputs()
            results[name] = result
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            errors[name] = safe_error_message(error)
            if isinstance(error, CandidateExecutionError):
                state.event("outer_execution_failed", recipe=name, execution=error.execution)
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
    published = False
    for name in order:
        if name not in results:
            continue
        try:
            remaining = state.remaining() - 1
            timeout = remaining * .40 if name == "baseline" and selected == "candidate" else remaining
            result = run_candidate(frozen_by_name[name], spec=gate.spec, fit_frame=gate.train,
                predict_frame=gate.test, phase="full_refit", timeout=min(1800, timeout),
                output_dir=state.run_dir / "full_refit" / name)
            gate.check_inputs()
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
            state.event("refit_failed", recipe=name, error=safe_error_message(error),
                        execution=error.execution if isinstance(error, CandidateExecutionError) else None)
    state.check_source()
    if not published and not state.recover():
        raise RuntimeError("No verified full-data predictions; no successful output can be reported")
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
    status = "completed" if reason == "model_finished" and state.report["successful_experiments"] else "fallback"
    return state.finish(status, reason)


def run_agent_v3(task_path, client, *, max_steps=12, time_budget=10500):
    if (type(max_steps) is not int or max_steps < 1 or type(time_budget) not in {int, float}
            or not math.isfinite(time_budget) or not 0 < time_budget <= 10500):
        raise ValueError("Invalid research request/time budget")
    state = V3State(Path(task_path).resolve(), max_steps, time_budget)
    try:
        prepare_research(state)
        prepare_baseline(state)
        if state.gate.held_rows and state.baseline is not None:
            try:
                state.baseline_inner = state.measure_inner(state.baseline).metric
                state.best_inner = state.baseline_inner
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                state.event("inner_baseline_unavailable", error=safe_error_message(error))
        state.report["evaluations"]["selection"] = {
            "metric": state.best_inner, "scope": "inner_selection", "split": state.gate.inner_split}
        state.transition("inner_research")
        if not state.gate.held_rows:
            state.report["stages"][1].update(status="skipped", reason=state.gate.inner_unavailable_reason)
            return finalize(state, "inner_validation_unavailable")
        messages = [{"role": "system", "content": SYSTEM_V3},
                    {"role": "user", "content": state.gate.task_text},
                    {"role": "user", "content": state.redact(_context(state))}]
        reason = "step_limit"
        for step in range(1, max_steps + 1):
            blocked = state.request_allowed(messages)
            if blocked:
                reason = blocked
                break
            state.report["api"]["attempts"] += 1
            state.save()
            try:
                response = client.with_options(timeout=min(120, state.remaining() - state.final_reserve)).chat.completions.create(
                    model=get_model(), messages=state.clean(messages), tools=[TOOL],
                    parallel_tool_calls=False, max_tokens=4096)
                state.report["api"]["responses"] += 1
                record_usage(state, response)
                completion = checked_completion(response)
            except Exception as error:
                state.event("api_error", error=safe_error_message(error))
                reason = "api_error"
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
                    result = run_python(code, state.workspace, state.run_dir,
                        step * 100 + index, timeout=min(1800, state.remaining() - state.final_reserve))
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
        return finalize(state, reason)
    except Exception as error:
        state.report.update(status="failed", error=safe_error_message(error))
        for stage in state.report.get("stages", []):
            if stage["status"] == "running":
                stage.update(status="failed", reason=safe_error_message(error))
        state.save()
        state.recover()
        raise
