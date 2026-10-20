"""LLM research with an isolated workspace and independently checked best output."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from agent.automl import INPUT_NAMES, _atomic_copy, _atomic_json, data_fingerprint
from agent.execution import run_python
from agent.llm import ROOT, SYSTEM_PROMPT, TOOL
from agent.research_validation import (
    HOLDOUT_NAME, HYPOTHESIS_NAME, MANIFEST_NAME, ResearchValidation, read_hypothesis,
)
from offline_baseline import find_data
from validation import validate_predictions


def checked_completion(response) -> dict:
    """Validate the SDK envelope before executing any returned tool call."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("Invalid API response: choices must be a non-empty list")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if (message is None or not hasattr(message, "content") or not hasattr(message, "tool_calls")
            or getattr(message, "role", "assistant") != "assistant"):
        raise ValueError("Invalid API response: missing assistant message")
    content = message.content
    if content is not None and not isinstance(content, str):
        raise ValueError("Invalid API response: assistant content must be text or null")
    finish_reason = getattr(choice, "finish_reason", None)
    if not isinstance(finish_reason, str):
        raise ValueError("Invalid API response: missing finish reason")
    calls = message.tool_calls
    if calls is not None and not isinstance(calls, list):
        raise ValueError("Invalid API response: tool_calls must be a list or null")
    checked_calls, seen_ids = [], set()
    for call in calls or []:
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        if (not isinstance(call_id, str) or not call_id.strip() or call_id in seen_ids
                or getattr(call, "type", "function") != "function"
                or not isinstance(name, str) or not name.strip()):
            raise ValueError("Invalid API response: tool calls require unique IDs and named functions")
        seen_ids.add(call_id)
        arguments = getattr(function, "arguments", None)
        # Invalid argument types receive a tool error below. Keep the assistant
        # history schema valid so the next request can carry that feedback.
        history_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments, allow_nan=False)
        checked_calls.append({"id": call_id, "name": name, "arguments": arguments,
                              "history": {"id": call_id, "type": "function", "function": {
                                  "name": name, "arguments": history_arguments}}})
    return {"content": content, "finish_reason": finish_reason, "tool_calls": checked_calls}


def record_usage(state, response) -> None:
    """Optional provider accounting must not invalidate a usable response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    counts = {field: getattr(usage, field, None) for field in fields}
    invalid = [field for field, value in counts.items() if type(value) is not int or value < 0]
    if invalid:
        state.event("invalid_api_usage", fields=invalid)
        return
    state.report["api"]["usage_reported_responses"] += 1
    for field, value in counts.items():
        state.report["api"][field] += value


class RunState:
    def __init__(self, task_path: Path, max_steps: int, time_budget: float):
        self.task_path = task_path
        self.task_dir = task_path.parent
        self.task_text = task_path.read_text(encoding="utf-8-sig")
        self.started = time.monotonic()
        self.deadline = self.started + time_budget
        self.run_dir = self.task_dir / ".agent_runs" / uuid4().hex
        self.run_dir.mkdir(parents=True)
        self.backup = self.run_dir / "last_valid_submission.csv"
        self.workspace = self.run_dir / "workspace"
        self.work_task = self.workspace / task_path.name
        self.validator = None
        self.source_dir = None
        self.source_fingerprint = None
        self.work_data = None
        self.report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "task": str(task_path), "model": os.environ.get("OPENAI_MODEL", ""),
            "max_steps": max_steps, "time_budget_seconds": time_budget,
            "status": "running", "baseline": "not_started", "experiments": 0,
            "successful_experiments": 0, "improvements": 0,
            "unverified_experiments": 0, "best_metric": None,
            "host_validation": {"status": "not_started"},
            "api": {"attempts": 0, "responses": 0, "usage_reported_responses": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "events": [],
        }
        self.save()

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def redact(self, value: str) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        return value.replace(key, "[REDACTED]") if key else value

    def clean(self, value):
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {self.redact(key): self.clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value]
        return value

    def save(self) -> None:
        self.report["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        _atomic_json(self.run_dir / "run.json", self.clean(self.report))

    def event(self, kind: str, **details) -> None:
        self.report["events"].append({"kind": kind, **details})
        self.save()

    def check_source(self) -> None:
        if self.source_dir is not None and (
                data_fingerprint(self.source_dir) != self.source_fingerprint
                or self.task_path.read_text(encoding="utf-8-sig") != self.task_text):
            raise ValueError("Original task inputs changed during research; publication cancelled")

    def checkpoint(self, directory: Path, metric: dict | None = None) -> bool:
        """Publish only after the caller has checked quality; keep a private backup."""
        self.check_source()
        if not validate_predictions(directory)["valid"]:
            return False
        pending = self.run_dir / "pending_submission.csv"
        try:
            _atomic_copy(directory / "submission.csv", pending)
            _atomic_copy(pending, self.task_dir / "submission.csv")
            _atomic_copy(pending, self.backup)
        finally:
            pending.unlink(missing_ok=True)
        self.report["best_metric"] = metric
        return True

    def check_workspace(self) -> None:
        if (self.work_data is None or data_fingerprint(self.work_data) != self.source_fingerprint
                or self.work_task.read_text(encoding="utf-8-sig") != self.task_text):
            raise ValueError("Research task or input data changed; candidate cannot be accepted")

    def recover(self, *, force: bool = False) -> bool:
        # Always restore the accepted bytes. Merely valid CSV may be a worse
        # result written by generated code through an absolute path.
        if not self.backup.is_file():
            return False
        destination = self.task_dir / "submission.csv"
        if force or not destination.is_file() or destination.read_bytes() != self.backup.read_bytes():
            _atomic_copy(self.backup, destination)
            self.event("output_restored", valid=True)
        return validate_predictions(self.task_dir)["valid"]

    def finish(self, status: str, reason: str) -> Path:
        self.check_source()
        self.recover()
        self.report.update(status=status, reason=reason)
        self.report["output_check"] = validate_predictions(self.task_dir)
        self.save()
        print(f"Run report: {self.run_dir / 'run.json'}", flush=True)
        return self.run_dir

    def begin_attempt(self) -> None:
        self.check_source()
        self.check_workspace()
        if self.validator is not None:
            self.validator.check_inputs()
            self.validator.write_manifest()
        # Evidence belongs to one tool invocation. Models/intermediates persist,
        # but old predictions cannot validate a print-only or failed experiment.
        for name in ("submission.csv", HOLDOUT_NAME, HYPOTHESIS_NAME):
            path = self.workspace / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def consider(self) -> dict:
        self.check_source()
        self.check_workspace()
        hypothesis = read_hypothesis(self.workspace)
        if self.validator is None:
            if not validate_predictions(self.workspace)["valid"]:
                raise ValueError("The generated submission did not pass the structural check")
            if not self.checkpoint(self.workspace):
                raise ValueError("Could not retain the generated submission")
            self.report["unverified_experiments"] += 1
            return {"accepted": True, "verified": False, "metric": None,
                    "reason": "host_metric_unsupported", "hypothesis": hypothesis}
        metric = self.validator.evaluate(self.workspace / HOLDOUT_NAME, self.workspace)
        self.report["successful_experiments"] += 1
        best = self.report["best_metric"]
        better = best is None or (metric["value"] > best["value"] if metric["higher_is_better"]
                                 else metric["value"] < best["value"])
        if better:
            if not self.checkpoint(self.workspace, metric):
                raise ValueError("Could not retain the verified submission")
            _atomic_copy(self.workspace / HOLDOUT_NAME, self.run_dir / "best_holdout_predictions.csv")
            self.report["improvements"] += 1
        return {"accepted": better, "verified": True, "metric": metric,
                "reason": "improved" if better else "not_better", "hypothesis": hypothesis}


def prepare_workspace(state: RunState) -> None:
    # Copy only the task and its four input files. No hardlinks, secrets, app state
    # or recursive copies of earlier research runs enter the generated workspace.
    state.source_dir = find_data(state.task_dir)
    state.source_fingerprint = data_fingerprint(state.source_dir)
    state.workspace.mkdir()
    state.work_task.write_text(state.task_text, encoding="utf-8")
    destination = state.workspace / state.source_dir.relative_to(state.task_dir)
    state.work_data = destination
    destination.mkdir(parents=True, exist_ok=True)
    for name in INPUT_NAMES:
        shutil.copyfile(state.source_dir / name, destination / name)
    state.check_source()
    if data_fingerprint(destination) != state.source_fingerprint:
        raise ValueError("Input copy changed during preparation")
    if validate_predictions(state.task_dir)["valid"]:
        state.checkpoint(state.task_dir)
        state.report["baseline"] = "existing_output"
    try:
        state.validator = ResearchValidation(state.work_task)
        state.report["host_validation"] = {
            "status": "supported", "metric": state.validator.manifest["metric"],
            "split": state.validator.split, "manifest": MANIFEST_NAME,
        }
    except ValueError as error:
        # Private metrics are unknown. Preserve the general LLM route for the
        # supplied CSV contract, but never label its quality independently checked.
        state.report["host_validation"] = {"status": "unsupported", "reason": str(error)}
        state.event("host_validation_unavailable", reason=str(error))
    state.save()


def prepare_starting_output(state: RunState) -> None:
    if state.validator is None:
        if state.report["baseline"] != "existing_output":
            state.report["baseline"] = "unavailable"
        state.save()
        return
    timeout = min(600.0, state.remaining() / 4)
    if timeout <= 5:
        state.report["baseline"] = "skipped_for_time"
        state.save()
        return
    startup_dir = state.run_dir / "startup"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\nfrom agent.automl import run_automl\n"
        f"run_automl(Path({str(state.work_task)!r}), time_budget={timeout - 3!r}, "
        f"output_dir=Path({str(startup_dir)!r}))\n"
    )
    print(f"Preparing checked AutoML starting predictions (up to {timeout:.0f}s)...", flush=True)
    try:
        result = run_python(code, state.workspace, state.run_dir, 0, timeout=timeout)
        if result.get("exit_code") != 0 or result.get("timed_out") or "error" in result:
            raise ValueError(result.get("stderr") or result.get("error") or "Startup timed out")
        state.check_source()
        metric = state.validator.evaluate(startup_dir / HOLDOUT_NAME, state.workspace)
        if not state.checkpoint(state.workspace, metric):
            raise ValueError("Starting predictions failed their final check")
        _atomic_copy(startup_dir / HOLDOUT_NAME, state.run_dir / "best_holdout_predictions.csv")
        state.report["baseline"] = "trained"
        state.event("baseline", valid=True, metric=metric, result=result)
        print(f"Starting {metric['name']}: {metric['value']:.6f}; predictions saved.", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        state.report["baseline"] = "existing_output" if state.backup.is_file() else "unavailable"
        state.event("baseline", valid=False, error=str(error))
        state.recover()
        print("Startup search unavailable; continuing with the task and LLM research.", flush=True)


def starting_context(state: RunState) -> str:
    context = (
        f"Startup result: {state.report['baseline']}. "
        "A starting CSV is not proof that research is complete. "
        f"run_python runs in a persistent isolated workspace: {state.workspace}. "
        "The task and its input files have been copied there with the same relative paths. "
        "Use only these copied inputs; never alter them or access the original task directory. "
        "Intermediate model files persist between tool calls. "
        "Before EVERY run_python call the host deletes submission.csv, holdout_predictions.csv "
        "and research_hypothesis.json from the workspace. To submit an experiment, write all "
        "required evidence within ONE tool call. Analysis-only calls are permitted but do not "
        "count as evaluated experiments. Write research_hypothesis.json with "
        "{\"hypothesis\": \"what was changed and why it may improve validation\"}. "
        "Also refit your selected pipeline on all training rows and write submission.csv. "
        "The accepted result is stored separately and survives failed experiments. "
        f"Local helper modules can be imported after sys.path.insert(0, {str(ROOT)!r}). "
    )
    if state.validator is not None:
        context += (
            f"Read {MANIFEST_NAME}: it gives exact fit_rows and validation_rows as zero-based "
            "train.csv positions, data_directory, feature lists, target, ID, metric and split. "
            "Fit preprocessing/model only on fit_rows, then predict validation_rows in the "
            "specified order. Save holdout_predictions.csv with exactly id,actual,prediction. "
            "id and actual must retain the original CSV strings: read source columns with "
            "dtype=str, keep_default_na=False (including literal NA and leading zeroes). "
            "Do not train on validation answers before recording holdout predictions. "
            "The host independently recomputes the metric, rejects changed IDs/answers/inputs "
            "and only publishes strict improvements over the current best. A valid trial "
            "that does not improve still provides feedback for the next hypothesis. "
            f"Current checked best: {json.dumps(state.report['best_metric'], ensure_ascii=False)}. "
        )
    else:
        context += (
            "The host cannot independently score this task's metric: "
            f"{state.report['host_validation'].get('reason', '')}. "
            "Infer the metric from the task, design validation, compare experiments and preserve "
            "your best model yourself. Report the measurements and uncertainty. The host only "
            "checks the supplied CSV template, so it will label the result quality-unverified. "
        )
    startup = state.run_dir / "startup" / "automl_report.json"
    if startup.is_file():
        report = json.loads(startup.read_text(encoding="utf-8"))
        summary = {key: report[key] for key in ("model", "model_config", "metric", "classification_diagnostics")
                   if key in report}
        context += "\nStarting model and error analysis:\n" + json.dumps(summary, ensure_ascii=False)
    return state.redact(context)


def run_agent(task_path: Path, client, *, max_steps: int = 12,
              time_budget: float = 10500) -> Path:
    task_path = task_path.resolve()
    state = RunState(task_path, max_steps, time_budget)
    try:
        prepare_workspace(state)
        prepare_starting_output(state)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.task_text},
            {"role": "user", "content": starting_context(state)},
        ]
        for step in range(1, max_steps + 1):
            remaining = state.remaining()
            if remaining <= 5:
                if state.recover():
                    print("Time budget reached; the retained submission.csv was preserved.")
                    return state.finish("fallback", "time_limit")
                raise TimeoutError(f"Agent time budget reached. Experiments: {state.run_dir}")
            print(f"Step {step}/{max_steps}; remaining {int(remaining)}s", file=sys.stderr)
            state.report["api"]["attempts"] += 1
            state.save()
            try:
                response = client.with_options(timeout=min(120, remaining)).chat.completions.create(
                    model=os.environ["OPENAI_MODEL"], messages=messages,
                    tools=[TOOL], parallel_tool_calls=False, max_tokens=4096,
                )
                state.report["api"]["responses"] += 1
                completion = checked_completion(response)
            except Exception as error:
                state.event("api_error", error=str(error))
                if state.recover():
                    print("API request failed; the retained submission.csv was preserved.")
                    return state.finish("fallback", "api_error")
                raise
            record_usage(state, response)
            state.save()
            if completion["finish_reason"] == "length":
                if state.recover():
                    print("Model response was truncated; retained predictions were preserved.")
                    return state.finish("fallback", "truncated_response")
                raise RuntimeError("Model response was truncated; no code from it was executed.")
            if completion["tool_calls"]:
                messages.append({
                    "role": "assistant", "content": completion["content"],
                    "tool_calls": [call["history"] for call in completion["tool_calls"]],
                })
                for index, call in enumerate(completion["tool_calls"], start=1):
                    try:
                        if call["name"] != "run_python":
                            raise ValueError("Unknown tool: " + call["name"])
                        if not isinstance(call["arguments"], str):
                            raise ValueError("run_python arguments must be a JSON string")
                        arguments = json.loads(call["arguments"])
                        if not isinstance(arguments, dict) or set(arguments) != {"code"}:
                            raise ValueError("run_python requires exactly one argument: code")
                        if not isinstance(arguments["code"], str) or not arguments["code"].strip():
                            raise ValueError("run_python code must be a non-empty string")
                        state.begin_attempt()
                        remaining = state.remaining()
                        if remaining <= 5:
                            raise TimeoutError("No time remains for another experiment")
                        result = run_python(
                            arguments["code"], state.workspace, state.run_dir,
                            step * 100 + index, timeout=min(1800, remaining - 2),
                        )
                        state.report["experiments"] += 1
                    except (ValueError, OSError, TimeoutError) as error:
                        result = {"error": str(error)}
                    result["output_check"] = validate_predictions(state.workspace)
                    failed = ("error" in result or result.get("exit_code") != 0
                              or result.get("timed_out", False))
                    if not failed:
                        try:
                            result["selection"] = state.consider()
                        except (ValueError, OSError, UnicodeError) as error:
                            result["selection"] = {"accepted": False, "verified": False,
                                                   "reason": str(error)}
                    else:
                        result["selection"] = {"accepted": False, "verified": False,
                                               "reason": "execution_failed"}
                    # The task-level file always contains accepted bytes, even if
                    # generated code attempted to write through an absolute path.
                    if state.backup.is_file():
                        result["previous_output_restored"] = state.recover()
                    state.event("experiment", step=step, tool_index=index, result=result)
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": json.dumps(state.clean(result), ensure_ascii=False),
                    })
            else:
                valid = state.recover()
                if valid and (state.report["successful_experiments"] or state.report["unverified_experiments"]):
                    print(state.redact(completion["content"] or "Agent finished."))
                    if state.validator is None:
                        return state.finish("fallback", "host_metric_unsupported")
                    return state.finish("completed", "model_finished")
                instruction = (
                    "Research is not complete. Use run_python to test a hypothesis and write fresh "
                    "research_hypothesis.json, submission.csv and, when host validation is supported, "
                    "holdout_predictions.csv in the SAME call. "
                    "Inspect the returned selection/error and measure validation quality."
                )
                if state.validator is not None:
                    instruction += " Follow research_validation.json."
                messages.extend([
                    {"role": "assistant", "content": completion["content"] or "Finished."},
                    {"role": "user", "content": instruction},
                ])
        if state.recover():
            print("Step limit reached; the retained submission.csv was preserved.")
            return state.finish("fallback", "step_limit")
        raise RuntimeError(f"Step limit reached without an accepted output. Logs: {state.run_dir}")
    except Exception as error:
        state.report.update(status="failed", error=state.redact(str(error)))
        state.save()
        state.recover()
        raise
