"""Opt-in staged Qwen probe: text, tools, research shape, then real v3.

Without --execute this writes a preflight report and never creates an API
client. Native v3 artifacts stay local. probe_report.json contains only safe
metadata, never prompts, responses, generated code or provider bodies.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from agent.api_probe import ProbeBudget, ProbeClient, ProbeLimitReached, timestamps
from agent.gateways import client_gateways, request_options, retryable_api_error
from agent.llm import (API_FIELDS, REQUIRED_API_FIELDS, ROOT, TOOL, get_model,
                       load_local_env, make_client, safe_error_details, public_error_message)
from agent.research import checked_completion
from agent.research_v3 import SYSTEM_V3, run_agent_v3
from experiment_research_smoke import create_tasks, source_provenance, summarize_run


TOOL_CODE = 'print("rescue-api-tool-ok")'
STAGES = ("text", "tool_schema", "research_shape", "v3_smoke")


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _arguments(stage):
    common = {"model": get_model()}
    if stage == "text":
        return {**common, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 64}
    if stage == "tool_schema":
        messages = [{"role": "user", "content":
                     "Call run_python exactly once with this exact code, without additions: " + TOOL_CODE}]
        tokens = 256
    else:
        messages = [{"role": "system", "content": SYSTEM_V3}, {"role": "user", "content":
            "Synthetic regression; metric RMSE. Allowed inputs: task.md, research_validation.json, "
            "data/train.csv, data/test.csv, data/data_dictionary.csv, data/sample_submission.csv. "
            "Features measurement (numeric), segment (categorical); target outcome; row_key is an ID. "
            "All research inputs contain outer-fit only; use INNER selection. Baseline metric is "
            "not available in this schema-only diagnostic. Use the manifest and emit one small "
            "run_python call creating fresh research_candidate.py, research_recipe.json and "
            "research_hypothesis.json with fit_predict(X_fit, y_fit, X_predict, context). "
            "Do not install or download packages. This checks request shape only; the returned code "
            "will not execute in this stage."}]
        tokens = 4096
    return {**common, "messages": messages, "tools": [TOOL],
            "parallel_tool_calls": False, "max_tokens": tokens}


def _validate(response, stage):
    try:
        completion = checked_completion(response)
    except (ValueError, TypeError, AttributeError):
        return "malformed_response", False
    if completion["finish_reason"] == "length":
        return "truncated_response", False
    if stage == "text":
        valid = bool(completion["content"] and completion["content"].strip()) and not completion["tool_calls"]
        return ("validated" if valid else "empty_or_unexpected_response"), valid
    calls = completion["tool_calls"]
    if len(calls) != 1 or calls[0]["name"] != "run_python":
        return "invalid_tool_call", False
    try:
        arguments = json.loads(calls[0]["arguments"])
    except (ValueError, TypeError):
        return "invalid_tool_json", False
    if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str) or not arguments["code"].strip():
        return "invalid_tool_arguments", False
    if stage == "tool_schema" and arguments["code"] != TOOL_CODE:
        return "unexpected_probe_code", False
    return "validated", True


def _diagnostic_request(client, budget, stage, recovery):
    """At most one recovery across all three diagnostics; no paid-output retry."""
    routes = client_gateways(client) or (None,)
    index = min(recovery["route_index"], len(routes) - 1)
    arguments = _arguments(stage)
    while True:
        budget.stage = stage
        remaining = budget.remaining()
        if remaining <= 0:
            budget.blocked = "deadline"
            budget.checkpoint()
            raise ProbeLimitReached("Shared probe deadline reached")
        try:
            response = client.with_options(**request_options(min(120., remaining), routes[index])).chat.completions.create(**arguments)
        except Exception as error:
            if recovery["retries"] < 1 and retryable_api_error(error):
                recovery["retries"] += 1
                if len(routes) == 2:
                    index = 1 - index
                    recovery["route_index"] = index
                continue
            raise
        result, valid = _validate(response, stage)
        return {"variant": stage, "status": result, "api_response_received": True,
                "tool_call_valid": valid and stage != "text", "candidate_executed": False}, valid


def execute_probe(directory, report, *, deadline, max_requests, max_tokens):
    """Worker entry; injectable make_client/run_agent_v3 support offline tests."""
    path = directory / "probe_report.json"
    budget = ProbeBudget(deadline=deadline, max_requests=max_requests, max_tokens=max_tokens)

    def checkpoint():
        report.update(budget=budget.summary(), attempts=budget.attempts, api_called=bool(budget.attempts))
        _write(path, report)

    budget.checkpoint = checkpoint
    report.update(status="running", stages=[], smoke=None)
    checkpoint()
    try:
        with make_client() as sdk:
            client = ProbeClient(sdk, budget)
            recovery = {"retries": 0, "route_index": 0}
            for stage in STAGES[:3]:
                item, valid = _diagnostic_request(client, budget, stage, recovery)
                report["stages"].append(item)
                if not valid:
                    report["status"] = "diagnostic_failed"
                    return 1
                if stage == "tool_schema":
                    # Run only our exact constant after strict equality. Never
                    # evaluate arbitrary provider code in the diagnostic stages.
                    from agent.execution import run_python
                    result = run_python(TOOL_CODE, directory, directory / "tool_check", 1,
                                        timeout=min(5., budget.remaining()))
                    ok = (result.get("exit_code") == 0 and not result.get("timed_out")
                          and not result.get("error") and not result.get("memory_limit_exceeded")
                          and result.get("stdout", "").strip() == "rescue-api-tool-ok")
                    item["deterministic_tool_executed"] = ok
                    if not ok:
                        report["status"] = "tool_execution_failed"
                        return 1
                checkpoint()
            # A single regression task is sufficient for the minimum probe;
            # the standalone smoke retains its separate two-task coverage.
            if budget.remaining() < 45 or len(budget.attempts) >= max_requests:
                report["status"] = "smoke_budget_unavailable"
                return 1
            task = create_tasks(directory / "smoke")["regression"]
            budget.stage = "v3_smoke"
            output, errors = io.StringIO(), io.StringIO()
            native = None
            try:
                with redirect_stdout(output), redirect_stderr(errors):
                    native = run_agent_v3(task, client, max_steps=min(3, max_requests - len(budget.attempts)),
                                          time_budget=min(300., budget.remaining()))
            finally:
                # Native logs are private, never copied into the public report.
                (directory / "native_smoke.log").write_text(output.getvalue() + errors.getvalue(), encoding="utf-8")
            run = json.loads((native / "run.json").read_text(encoding="utf-8"))
            smoke = summarize_run(task, run, run_dir=native)
            report["smoke"] = smoke
            coverage = smoke["coverage"]
            integration = (coverage["inner_trial_verified"] and coverage["candidate_frozen"]
                           and coverage["finalization_verified"] and smoke["status"] in {"completed", "fallback"}
                           and smoke["output_check"].get("valid") is True
                           and (smoke.get("publication") or {}).get("new_result_published") is True)
            report["integration_passed"] = bool(integration)
            report["full_candidate_path_passed"] = bool(integration and coverage.get("candidate_published"))
            report["status"] = "integration_passed" if integration else "integration_unverified"
            return 0 if integration else 1
    except Exception as error:
        report.update(status="blocked" if isinstance(error, ProbeLimitReached) else "failed",
                      error=public_error_message(error), error_details=safe_error_details(error))
        return 1
    finally:
        report["finished_at"] = timestamps()
        checkpoint()


def _bounded_int(maximum):
    def parse(value):
        integer = int(value)
        if not 1 <= integer <= maximum:
            raise argparse.ArgumentTypeError(f"Must be in 1..{maximum}")
        return integer
    return parse


def _seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise argparse.ArgumentTypeError("Deadline must be positive and at most 600 seconds")
    return seconds


def _supervise(command, deadline, log):
    """Bound the entire worker, including slow HTTP bodies and native children."""
    from agent.resources import ProcessTreeMonitor, POLL_SECONDS
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                   start_new_session=os.name != "nt",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        tracker = ProcessTreeMonitor(process)
        reason = None
        try:
            while True:
                exceeded = tracker.sample()
                if process.poll() is not None:
                    break
                if exceeded or time.monotonic() >= deadline:
                    reason = "memory_limit" if exceeded else "deadline"
                    break
                try:
                    process.wait(timeout=min(POLL_SECONDS, max(.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stopped = tracker.terminate()
        return process.returncode, reason or (None if stopped else "cleanup_failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments" / "rescue_api")
    parser.add_argument("--max-requests", type=_bounded_int(6), default=6)
    parser.add_argument("--max-estimated-tokens", type=_bounded_int(30000), default=30000)
    parser.add_argument("--deadline-seconds", type=_seconds, default=600.)
    parser.add_argument("--_worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    load_local_env()
    if args._worker_dir is not None:
        if not args.execute or args._deadline is None or not math.isfinite(args._deadline):
            parser.error("A worker requires explicit --execute and a finite inherited deadline")
        directory = args._worker_dir
        report = json.loads((directory / "probe_report.json").read_text(encoding="utf-8"))
        return execute_probe(directory, report, deadline=min(args._deadline, time.monotonic() + args.deadline_seconds),
                             max_requests=args.max_requests, max_tokens=args.max_estimated_tokens)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()) and not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("Use gitignored data/ or an external directory for reports")
    directory = output / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    present = {name: bool(os.environ.get(name, "").strip()) for name in API_FIELDS}
    configured = all(present[name] for name in REQUIRED_API_FIELDS)
    provenance = source_provenance()
    report = {"version": 1, "kind": "rescue_api_probe", "synthetic_only": True,
              "base_commit": "d05572846369f3b1e9cfa33f4008dda028db4fe0", **provenance,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_at": timestamps(), "mode": "execute" if args.execute else "preflight",
              "api_settings_present": present, "quota_remaining": None,
              "quota_status": "unknown_no_documented_balance_endpoint",
              "limits": {"max_requests": args.max_requests, "max_estimated_tokens": args.max_estimated_tokens,
                         "deadline_seconds": args.deadline_seconds},
              "planned_stages": list(STAGES), "attempts": [], "api_called": False,
              "integration_passed": False, "full_candidate_path_passed": False,
              "context_helper_active": False,
              "status": "ready" if configured else "missing_api_settings"}
    path = directory / "probe_report.json"
    _write(path, report)
    if not args.execute or not configured:
        report["finished_at"] = timestamps()
        _write(path, report)
        print(f"Probe {report['status']}; API attempts: 0. Report: {path}")
        return 0 if not args.execute else 2
    deadline = time.monotonic() + args.deadline_seconds
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--_worker-dir", str(directory),
               "--_deadline", str(deadline), "--deadline-seconds", str(args.deadline_seconds),
               "--max-requests", str(args.max_requests), "--max-estimated-tokens", str(args.max_estimated_tokens)]
    code, reason = _supervise(command, deadline, directory / "private_worker.log")
    report = json.loads(path.read_text(encoding="utf-8"))
    if reason or report["status"] in {"running", "ready", "missing_api_settings"}:
        report.update(status="supervisor_stopped", supervisor_reason=reason or "worker_exited",
                      worker_exit_code=code, integration_passed=False,
                      full_candidate_path_passed=False, finished_at=timestamps())
        _write(path, report)
    print(f"Probe {report['status']}; API attempts: {len(report['attempts'])}. Report: {path}")
    return 0 if code == 0 and reason is None and report["integration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
