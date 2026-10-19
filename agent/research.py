"""Research loop, startup predictions and recovery after failed experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from agent.execution import run_python
from agent.llm import ROOT, SYSTEM_PROMPT, TOOL
from validation import validate_predictions


class RunState:
    def __init__(self, task_path: Path, max_steps: int, time_budget: float):
        self.task_path = task_path
        self.task_dir = task_path.parent
        self.started = time.monotonic()
        self.deadline = self.started + time_budget
        self.run_dir = self.task_dir / ".agent_runs" / uuid4().hex
        self.run_dir.mkdir(parents=True)
        self.backup = self.run_dir / "last_valid_submission.csv"
        self.report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "task": str(task_path), "model": os.environ.get("OPENAI_MODEL", ""),
            "max_steps": max_steps, "time_budget_seconds": time_budget,
            "status": "running", "baseline": "not_started", "experiments": 0,
            "successful_experiments": 0,
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
        payload = json.dumps(self.clean(self.report), ensure_ascii=False, indent=2)
        temporary = self.run_dir / "run.json.tmp"
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(self.run_dir / "run.json")

    def event(self, kind: str, **details) -> None:
        self.report["events"].append({"kind": kind, **details})
        self.save()

    def checkpoint(self, validation: dict | None = None) -> bool:
        validation = validation if validation is not None else validate_predictions(self.task_dir)
        if not validation["valid"]:
            return False
        temporary = self.run_dir / "last_valid_submission.tmp.csv"
        shutil.copyfile(self.task_dir / "submission.csv", temporary)
        temporary.replace(self.backup)
        return True

    def recover(self, *, force: bool = False) -> bool:
        if not force and validate_predictions(self.task_dir)["valid"]:
            return True
        if not self.backup.is_file():
            return False
        temporary = self.run_dir / "restored_submission.tmp.csv"
        shutil.copyfile(self.backup, temporary)
        temporary.replace(self.task_dir / "submission.csv")
        valid = validate_predictions(self.task_dir)["valid"]
        self.event("output_restored", valid=valid)
        return valid

    def finish(self, status: str, reason: str) -> Path:
        self.report.update(status=status, reason=reason)
        self.report["output_check"] = validate_predictions(self.task_dir)
        self.save()
        print(f"Run report: {self.run_dir / 'run.json'}", flush=True)
        return self.run_dir


def prepare_starting_output(state: RunState) -> None:
    if state.checkpoint():
        state.report["baseline"] = "existing_output"
        state.event("existing_output_preserved")
        return
    # Include startup training in the same total deadline as the research loop.
    timeout = min(120.0, state.remaining() / 4)
    if timeout <= 2:
        state.report["baseline"] = "skipped_for_time"
        state.save()
        return
    script = ROOT / "offline_baseline.py"
    code = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.argv = [{str(script)!r}, {str(state.task_path)!r}, '--overwrite']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    print(f"Preparing starting predictions (up to {timeout:.0f}s)...", flush=True)
    try:
        result = run_python(code, state.task_dir, state.run_dir, 0, timeout=timeout)
    except (OSError, ValueError) as error:
        result = {"error": str(error)}
    valid = state.checkpoint()
    state.report["baseline"] = "trained" if valid else "unavailable"
    state.event("baseline", valid=valid, result=result)
    if valid:
        print("Starting submission.csv passed the structural check.", flush=True)
    else:
        print("Local baseline unavailable; the agent will solve the task from its description.", flush=True)


def starting_context(state: RunState) -> str:
    context = (
        f"Startup result: {state.report['baseline']}. "
        "Any existing submission.csv is a starting result, not proof the research is complete. "
        "Inspect the task and perform your own hypothesis/experiment/validation cycle. "
        "Use the same validation split when comparing model quality. "
        f"The local baseline implementation, if useful, is {ROOT / 'offline_baseline.py'}."
    )
    metrics = state.task_dir / "baseline_metrics.json"
    if metrics.is_file():
        with metrics.open(encoding="utf-8") as stream:
            context += "\nStartup metrics (local holdout, not platform feedback):\n" + stream.read(6000)
    return state.redact(context)


def run_agent(task_path: Path, client, *, max_steps: int = 12,
              time_budget: float = 10500) -> Path:
    task_path = task_path.resolve()
    task = task_path.read_text(encoding="utf-8")
    state = RunState(task_path, max_steps, time_budget)
    try:
        prepare_starting_output(state)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "user", "content": starting_context(state)},
        ]
        for step in range(1, max_steps + 1):
            remaining = state.remaining()
            if remaining <= 5:
                if state.recover():
                    print("Time budget reached; a structurally valid submission.csv was preserved.")
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
            except Exception as error:
                state.event("api_error", error=str(error))
                if state.recover():
                    print("API request failed; a structurally valid submission.csv was preserved.")
                    return state.finish("fallback", "api_error")
                raise
            state.report["api"]["responses"] += 1
            usage = getattr(response, "usage", None)
            if usage is not None:
                state.report["api"]["usage_reported_responses"] += 1
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    state.report["api"][field] += getattr(usage, field, None) or 0
            state.save()
            message = response.choices[0].message
            if response.choices[0].finish_reason == "length":
                if state.recover():
                    print("Model response was truncated; valid predictions were preserved.")
                    return state.finish("fallback", "truncated_response")
                raise RuntimeError("Model response was truncated; no code from it was executed.")
            if message.tool_calls:
                messages.append({
                    "role": "assistant", "content": message.content,
                    "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls],
                })
                for index, call in enumerate(message.tool_calls, start=1):
                    try:
                        if call.function.name != "run_python":
                            raise ValueError("Unknown tool: " + call.function.name)
                        arguments = json.loads(call.function.arguments)
                        if not isinstance(arguments, dict) or set(arguments) != {"code"}:
                            raise ValueError("run_python requires exactly one argument: code")
                        remaining = state.remaining()
                        if remaining <= 5:
                            raise TimeoutError("No time remains for another experiment")
                        result = run_python(
                            arguments["code"], state.task_dir, state.run_dir,
                            step * 100 + index, timeout=min(1800, remaining - 2),
                        )
                        state.report["experiments"] += 1
                    except (ValueError, OSError, TimeoutError) as error:
                        result = {"error": str(error)}
                    validation = validate_predictions(state.task_dir)
                    result["output_check"] = validation
                    failed = ("error" in result or result.get("exit_code") != 0
                              or result.get("timed_out", False))
                    if failed and state.backup.is_file():
                        result["previous_output_restored"] = state.recover(force=True)
                    elif validation["valid"]:
                        state.checkpoint(validation)
                    elif state.backup.is_file():
                        result["previous_output_restored"] = state.recover(force=True)
                    if not failed and validation["valid"]:
                        state.report["successful_experiments"] += 1
                    state.event("experiment", step=step, tool_index=index, result=result)
                    messages.append({
                        "role": "tool", "tool_call_id": call.id,
                        "content": json.dumps(state.clean(result), ensure_ascii=False),
                    })
            else:
                validation = validate_predictions(state.task_dir)
                if validation["valid"] and state.report["successful_experiments"]:
                    print(state.redact(message.content or "Agent finished."))
                    print("submission.csv passed the structural check.")
                    return state.finish("completed", "model_finished")
                if validation["valid"]:
                    instruction = (
                        "Only starting predictions exist so far. Use run_python to inspect the data, "
                        "test a hypothesis and measure validation quality before finishing."
                    )
                else:
                    instruction = (
                        "The output check failed. Use run_python to fix the output before finishing: "
                        + json.dumps(validation, ensure_ascii=False)
                    )
                messages.extend([
                    {"role": "assistant", "content": message.content or "Finished."},
                    {"role": "user", "content": instruction},
                ])
        if state.recover():
            print("Step limit reached; a structurally valid submission.csv was preserved.")
            return state.finish("fallback", "step_limit")
        raise RuntimeError(f"Step limit reached. Existing output files were preserved. Logs: {state.run_dir}")
    except Exception as error:
        state.report.update(status="failed", error=state.redact(str(error)))
        state.save()
        state.recover()
        raise
