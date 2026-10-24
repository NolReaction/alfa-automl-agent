"""Offline replay: uncompressed history versus the bounded research policy. Owner: Nika.

No API request, no code execution, no ML fit, no real dataset. Two series:

* ``legacy`` - the old replay (kept unchanged). It reconstructs what the pre-v3
  ``agent/research.py`` sent: every tool result stays in ``messages`` and a fresh
  ``ResearchLedger`` summary is appended before each request. It is NOT a measurement
  of the current Research v3.
* ``v3`` - the current Research v3: the real ``SYSTEM_V3``, the real ``_context`` text,
  a real v3 manifest from ``ResearchWorkspace`` (tiny synthetic task), the real
  ``V3State.budget_context`` budget message and v3-shaped feedback. The control keeps
  the history exactly as ``run_agent_v3`` does and appends the budget message to a copy
  of each request. The policy run uses the same transcript, its projection, and the same
  budget message. The mock host projection (``v3_snapshot``/``v3_contract``) is
  documented below and receives only INNER facts.

Sizes are UTF-8 bytes of compact JSON, not tokens and not money. Token numbers are the
host's own estimate ``ceil(bytes / 3)``; no provider usage exists here. This is a replay
of transcripts, not evidence about ML quality.

    python experiment_research_policy.py [--series both|legacy|v3] [--label NAME] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

from agent.llm import SYSTEM_PROMPT
from agent.research_feedback import ResearchLedger
from agent.research_policy import (
    PolicyBudgetExceeded, PolicyInputError, PolicyLimits, build_model_messages,
    decide_next_action, json_size,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("data/experiments/nika-research-policy")
FAKE_KEY = "placeholder-secret-ABCDEF0123456789"
TASK_TEXT = ("# Task\nPredict the target for every row of test.csv using train.csv. Metric: RMSE.\n"
             "Follow data_dictionary.csv and sample_submission.csv.\n" * 6)
HOST_INSTRUCTIONS = (
    "run_python runs in a persistent isolated workspace. Before EVERY run_python call the host deletes the "
    "evidence files. To submit an experiment write all required evidence within ONE tool call and record a "
    "hypothesis file. Fit only on fit_rows, predict validation_rows in the manifest order. Tune parameters "
    "only on the inner split. Refit the selected pipeline on all training rows for the final output. "
    "The host recomputes the metric, rejects changed ids/answers/inputs and publishes only strict "
    "improvements. A valid trial that does not improve still provides feedback. ") * 3


def redact(text: str) -> str:
    return text.replace(FAKE_KEY, "[REDACTED]")


def contract(*, supported=True, manifest_version=2, outputs=None, required=2, allow_unverified=False,
             scope="inner_validation"):
    return {
        "version": 1, "manifest_version": manifest_version, "manifest_path": "research_validation.json",
        "host_instructions": HOST_INSTRUCTIONS,
        "allowed_inputs": [{"path": "train.csv", "purpose": "fit data"},
                           {"path": "data_dictionary.csv", "purpose": "column meanings"}],
        "required_outputs": outputs or [
            {"path": "submission.csv", "columns": ["id", "target"]},
            {"path": "holdout_predictions.csv", "columns": ["id", "actual", "prediction"]}],
        "validation": {"supported": supported, "scope": scope, "metric": (
            {"name": "RMSE", "value": 10.0, "higher_is_better": False} if supported else None)},
        "completion": {"required_verified_trials": required, "allow_unverified": allow_unverified},
    }


# ---------------------------------------------------------------- text generators

def table_text(rng: random.Random, rows: int, columns: int) -> str:
    header = " ".join(f"col_{i:02d}" for i in range(columns))
    lines = [header]
    for index in range(rows):
        lines.append(f"{index:5d} " + " ".join(f"{rng.uniform(-3, 3):8.4f}" for _ in range(columns)))
    return "\n".join(lines)


def inspection_stdout(rng: random.Random) -> str:
    return ("train shape (120000, 48)\n" + table_text(rng, 400, 12) + "\n\ndescribe():\n"
            + table_text(rng, 150, 12) + "\n\nvalue_counts:\n" + table_text(rng, 80, 4))


def fold_log(rng: random.Random, lines: int = 40) -> str:
    return "\n".join(f"fold {i % 5 + 1} params={{'n_estimators': {100 + i * 20}, 'depth': {3 + i % 4}}} "
                     f"rmse={rng.uniform(8, 12):.5f} fit_seconds={rng.uniform(1, 9):.2f}"
                     for i in range(lines))


def traceback_text(frames: int, final: str) -> str:
    lines = ["Traceback (most recent call last):"]
    for index in range(frames):
        lines += [f'  File "/work/runs/ws/experiment_107.py", line {40 + index % 7}, in step_{index % 9}',
                  f"    value = transform(frame[column_{index % 13}], mapping)"]
    return "\n".join(lines + [final])


def selection(metric, *, accepted, verified=True, reason=None, hypothesis=None, diagnostics=None):
    value = {"accepted": accepted, "verified": verified, "metric": metric,
             "reason": reason or ("improved" if accepted else "not_better"), "hypothesis": hypothesis}
    if diagnostics is not None:
        value["diagnostics"] = diagnostics
    return value


def rmse(value: float) -> dict:
    return {"name": "RMSE", "value": value, "higher_is_better": False}


def regression_diagnostics(rng: random.Random) -> dict:
    return {"rows": 24000, "scope": "inner_validation_diagnostics", "regression": {
        "mean_error": rng.uniform(-1, 1), "absolute_error_p50": rng.uniform(3, 5),
        "absolute_error_p90": rng.uniform(9, 14), "absolute_error_p99": rng.uniform(25, 40)},
        "groups": [{"feature": f"cat_{i}", "group": f"level_{j}", "rows": 500 + j, "metric": "RMSE",
                    "value": rng.uniform(8, 13)} for i in range(3) for j in range(15)]}


def ok_result(rng, metric, *, accepted, hypothesis, stdout=None, diagnostics=True, reason=None):
    return {"stdout": stdout if stdout is not None else fold_log(rng), "stderr": "", "exit_code": 0,
            "timed_out": False, "memory_limit_exceeded": False, "elapsed_seconds": round(rng.uniform(20, 300), 3),
            "resources": {"available": True, "peak_rss_bytes": 900_000_000, "cpu_seconds": 120.5,
                          "memory_limit_bytes": 8_000_000_000, "sample_count": 40, "tracked_processes": 3,
                          "cleanup_complete": True, "unavailable_reason": None},
            "output_check": {"valid": True, "errors": []},
            "selection": selection(metric, accepted=accepted, hypothesis=hypothesis, reason=reason,
                                   diagnostics=regression_diagnostics(rng) if diagnostics else None),
            "previous_output_restored": True}


def failed_result(*, stderr="", stdout="", exit_code=1, timed_out=False, oom=False, error=None):
    result = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "timed_out": timed_out,
              "memory_limit_exceeded": oom, "elapsed_seconds": 12.5,
              "resources": {"available": True, "peak_rss_bytes": 7_900_000_000 if oom else 400_000_000,
                            "cpu_seconds": 30.0, "memory_limit_bytes": 8_000_000_000},
              "output_check": {"valid": True, "errors": []},
              "selection": {"accepted": False, "verified": False, "reason": "execution_failed"},
              "previous_output_restored": True}
    if error:
        result["error"] = error
    return result


def analysis_result(stdout: str) -> dict:
    return {"stdout": stdout, "stderr": "", "exit_code": 0, "timed_out": False,
            "memory_limit_exceeded": False, "elapsed_seconds": 4.2, "output_check": {"valid": True, "errors": []},
            "selection": {"accepted": False, "verified": False,
                          "reason": "[Errno 2] No such file or directory: '/work/runs/ws/research_hypothesis.json'"},
            "previous_output_restored": True}


def call(code: str, result: dict, kind=None) -> dict:
    return {"code": code, "result": result, "kind": kind}


def code_for(index: int, note: str = "") -> str:
    return (f"import pandas as pd\n# experiment {index} {note}\n"
            + "\n".join(f"feature_{i} = frame['c{i}'].fillna(0) * {index}" for i in range(30)))


# ------------------------------------------------------------------- scenarios

def build_scenarios() -> list[dict]:
    rng = random.Random(20261020)
    hyp = lambda n: f"Experiment {n}: change regularization and add fold-fitted target encoding for column {n}"
    scenarios = []

    scenarios.append({"name": "big_inspection", "contract": contract(), "start": {
        "stage": "inspect", "expect": ("inspect", "continue", "inspect_with_baseline")}, "turns": [
        {"calls": [call("print(df.describe())", analysis_result(inspection_stdout(rng)))],
         "stage": "research", "expect": ("experiment", "continue", "first_hypothesis")},
        {"calls": [call(code_for(2), ok_result(rng, rmse(9.2), accepted=True, hypothesis=hyp(2)))],
         "expect": ("experiment", "continue", "improved_continue")},
        {"calls": [call(code_for(3), ok_result(rng, rmse(9.6), accepted=False, hypothesis=hyp(3)))],
         "expect": ("experiment", "continue", "not_better_continue")}]})

    series, best = [], 10.0
    for n in range(1, 9):
        better = n in (1, 4, 6, 8)
        value = round(best - 0.4, 3) if better else round(best + 0.3, 3)
        best = value if better else best
        series.append({"calls": [call(code_for(n), ok_result(rng, rmse(value), accepted=better, hypothesis=hyp(n)))],
                       "expect": ("experiment", "continue", "improved_continue" if better else "not_better_continue")})
    series.append({"finish": True, "expect": ("finished", "complete", "contract_fulfilled")})
    scenarios.append({"name": "many_experiments", "contract": contract(), "turns": series})

    scenarios.append({"name": "long_traceback", "contract": contract(),
                      "must_contain": {2: ["unique-final-error-line"]}, "turns": [
        {"calls": [call(code_for(1), failed_result(
            stdout=fold_log(rng, 150), stderr=traceback_text(
                450, "ValueError: could not convert string to float: 'x12' (unique-final-error-line)")), "execution")],
         "expect": ("repair", "continue", "repair_execution_error")},
        {"calls": [call(code_for(2), ok_result(rng, rmse(9.1), accepted=True, hypothesis=hyp(2)))],
         "expect": ("experiment", "continue", "improved_continue")}]})

    scenarios.append({"name": "error_then_fix", "contract": contract(), "turns": [
        {"calls": [call(code_for(1), failed_result(stderr=traceback_text(6, "KeyError: 'region'")), "execution")],
         "expect": ("repair", "continue", "repair_execution_error")},
        {"calls": [call(code_for(2, "fixed"), ok_result(rng, rmse(9.5), accepted=True, hypothesis=hyp(2)))],
         "expect": ("experiment", "continue", "improved_continue")}]})

    scenarios.append({"name": "multi_call_group", "contract": contract(), "turns": [
        {"calls": [call("print(df.shape)", analysis_result(table_text(rng, 60, 8))),
                   call(code_for(2), ok_result(rng, rmse(9.4), accepted=True, hypothesis=hyp(2))),
                   call(code_for(3), ok_result(rng, rmse(9.9), accepted=False, hypothesis=hyp(3)))],
         "expect": ("experiment", "continue", "not_better_continue")}]})

    scenarios.append({"name": "incomplete_group", "contract": contract(), "expect_input_error": True, "turns": [
        {"calls": [call(code_for(1), ok_result(rng, rmse(9.4), accepted=True, hypothesis=hyp(1)))],
         "expect": ("experiment", "continue", "improved_continue"), "pending_tail": True}]})

    scenarios.append({"name": "early_finish", "contract": contract(), "turns": [
        {"calls": [call(code_for(1), ok_result(rng, rmse(9.4), accepted=True, hypothesis=hyp(1)))],
         "finish": True, "expect": ("experiment", "continue", "finish_too_early")},
        {"calls": [call(code_for(2), ok_result(rng, rmse(9.0), accepted=True, hypothesis=hyp(2)))],
         "finish": True, "expect": ("finished", "complete", "contract_fulfilled")}]})

    scenarios.append({"name": "low_time", "contract": contract(), "turns": [
        {"calls": [call(code_for(1), ok_result(rng, rmse(9.4), accepted=True, hypothesis=hyp(1)))],
         "seconds": 120.0, "expect": ("finalize", "continue", "finalize_low_resources")},
        {"calls": [call(code_for(2), ok_result(rng, rmse(9.2), accepted=True, hypothesis=hyp(2)))],
         "seconds": 0.0, "can_request": False, "expect": ("finished", "fallback", "resources_exhausted_keep_result")}]})

    scenarios.append({"name": "unsupported_metric", "contract": contract(supported=False, allow_unverified=True),
                      "baseline_metric": None, "turns": [
        {"calls": [call(code_for(1), {**analysis_result(fold_log(rng)), "selection": {
            "accepted": True, "verified": False, "metric": None, "reason": "host_metric_unsupported",
            "hypothesis": hyp(1)}}, None)],
         "unverified": True, "finish": True, "expect": ("finished", "complete", "unverified_completion")}]})

    v3 = contract(manifest_version=3, scope="inner_validation", outputs=[
        {"path": "submission.csv", "columns": ["id", "target"]},
        {"path": "inner_predictions.csv", "columns": ["id", "prediction"]}])
    v3["allowed_inputs"] = [{"path": "fit_view.csv", "purpose": "fit rows only"}]
    scenarios.append({"name": "holdout_without_actual", "contract": v3, "forbidden": ["actual", "train.csv"],
                      "start": {"stage": "inspect", "expect": ("inspect", "continue", "inspect_with_baseline")},
                      "turns": [
        {"calls": [call(code_for(1), ok_result(rng, rmse(9.3), accepted=True, hypothesis=hyp(1),
                                               diagnostics=False))],
         "stage": "research", "expect": ("experiment", "continue", "improved_continue")}]})

    scenarios.append({"name": "repeated_failures", "contract": contract(), "turns": [
        {"calls": [call(code_for(1), failed_result(stderr=traceback_text(8, "TypeError: bad operand")), "execution")],
         "expect": ("repair", "continue", "repair_execution_error")},
        {"calls": [call(code_for(2), failed_result(stderr=traceback_text(8, "TypeError: bad operand again")), "execution")],
         "expect": ("repair", "continue", "repair_consecutive_failures")},
        {"calls": [call(code_for(3), failed_result(stderr=traceback_text(8, "TypeError: still bad")), "execution")],
         "expect": ("finished", "fallback", "repeated_failures_keep_result")}]})
    return scenarios


# ------------------------------------------------------------------- replay

def trial_from(row: dict, kind) -> dict:
    return {"number": row["number"], "code_hash": row["code_hash"], "hypothesis": row.get("hypothesis"),
            "accepted": bool(row["accepted"]), "verified": bool(row["verified"]), "metric": row.get("metric"),
            "reason": row.get("reason"), "error_kind": kind, "error": row.get("error"),
            **({"diagnostics": row["diagnostics"]} if row.get("diagnostics") else {}),
            **({"resources": row["resources"]} if row.get("resources") else {})}


def snapshot_for(ledger, kinds, *, stage, seconds, requests, can_request, finish, best, unverified, baseline=True):
    trials = [trial_from(row, kind) for row, kind in zip(ledger.trials, kinds)]
    return {"version": 1, "stage": stage,
            "budget": {"remaining_seconds": seconds, "remaining_requests": requests, "can_request": can_request},
            "baseline": {"status": "trained" if baseline else "unavailable", "retained_output_valid": baseline,
                         "best_verified_metric": best},
            "last_trial": trials[-1] if trials else None, "recent_trials": trials[-6:],
            "counts": {"verified_trials": sum(1 for t in trials if t["verified"]),
                       "unverified_trials": 1 if unverified else 0,
                       "consecutive_execution_failures": ledger.failure_streak},
            "finish_requested": finish}


def run_scenario(scenario: dict, limits: PolicyLimits) -> dict:
    ledger, kinds, history = ResearchLedger(), [], []
    legacy_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": TASK_TEXT},
                       {"role": "user", "content": HOST_INSTRUCTIONS}]
    host_contract = scenario["contract"]
    baseline_metric = scenario.get("baseline_metric", rmse(10.0))
    best = baseline_metric
    legacy_sizes, policy_sizes, transitions, diagnostics, facts, prefixes = [], [], [], [], [], []
    texts = []
    stage, unverified = "research", False

    def request(index, expect, *, stage, seconds, can_request, finish):
        snap = snapshot_for(ledger, kinds, stage=stage, seconds=seconds, requests=max(0, 30 - index),
                            can_request=can_request, finish=finish, best=best, unverified=unverified,
                            baseline=True)
        decision = decide_next_action(snap, contract=host_contract)
        got = (decision.stage, decision.action, decision.reason)
        transitions.append({"request": index, "expected": expect, "got": got, "ok": got == tuple(expect)})
        if ledger.trials:
            legacy_messages.append({"role": "user", "content": "Host research history:\n" + ledger.context(
                best=best, remaining=seconds, api={})})
        legacy_sizes.append(json_size(legacy_messages))
        try:
            context = build_model_messages(
                system_prompt=SYSTEM_PROMPT, task_text=TASK_TEXT, contract=host_contract, history=history,
                snapshot=snap, limits=limits, redact=redact)
        except PolicyInputError as error:
            return {"input_error": str(error)}
        policy_sizes.append(context.diagnostics["after_bytes"])
        diagnostics.append(context.diagnostics)
        prefixes.append(json.dumps(context.messages[:3], ensure_ascii=False))
        text = json.dumps(context.messages, ensure_ascii=False)
        texts.append(text)
        record = {"request": index}
        last = snap["last_trial"]
        record["within_limit"] = context.diagnostics["after_bytes"] <= limits.max_messages_bytes
        record["no_secret"] = FAKE_KEY not in text
        record["contract_verbatim"] = host_contract["host_instructions"] in context.messages[2]["content"]
        if best is not None:
            record["best_metric_present"] = str(best["value"]) in context.messages[-1]["content"]
        if last is not None and last["hypothesis"]:
            record["last_hypothesis_present"] = last["hypothesis"][:40] in context.messages[-1]["content"]
        if last is not None and last["error_kind"]:
            record["last_failure_present"] = last["error_kind"] in context.messages[-1]["content"]
        # Only the parts written by the policy and the host are checked; the task text may name files.
        policy_text = context.messages[2]["content"] + context.messages[-1]["content"]
        for marker in scenario.get("must_contain", {}).get(index, []):
            record[f"contains_{marker[:18]}"] = marker in text
        for word in scenario.get("forbidden", []):
            record[f"absent_{word}"] = re.search(rf"(?<![\w.]){re.escape(word)}(?![\w])", policy_text) is None
        facts.append(record)
        return {"context": context}

    start = scenario.get("start", {"stage": "research", "expect": ("experiment", "continue", "first_hypothesis")})
    outcome = request(1, start["expect"], stage=start["stage"], seconds=9000.0, can_request=True, finish=False)
    for number, turn in enumerate(scenario["turns"], start=1):
        stage = turn.get("stage", stage)
        unverified = unverified or turn.get("unverified", False)
        if turn.get("calls"):
            identifiers = [f"call-{number}-{i}" for i in range(len(turn["calls"]))]
            history.append({"role": "assistant", "content": "Next I test the following idea.", "tool_calls": [
                {"id": identifier, "type": "function", "function": {
                    "name": "run_python", "arguments": json.dumps({"code": item["code"]})}}
                for identifier, item in zip(identifiers, turn["calls"])]})
            legacy_messages.append(history[-1])
            for identifier, item in zip(identifiers, turn["calls"]):
                result = dict(item["result"])
                ledger.record(item["code"], result, remaining=turn.get("seconds", 9000.0))
                kinds.append(item["kind"])
                selection_value = result.get("selection", {})
                if selection_value.get("accepted") and selection_value.get("verified") and selection_value.get("metric"):
                    best = selection_value["metric"]
                content = json.dumps(result, ensure_ascii=False)
                if scenario.get("leak_key"):
                    content = content.replace("fold", FAKE_KEY + " fold", 1)
                message = {"role": "tool", "tool_call_id": identifier, "content": content}
                history.append(message)
                legacy_messages.append(message)
            if turn.get("pending_tail"):
                history.pop()  # the host has not finished the group: results are missing
        outcome = request(number + 1, turn["expect"], stage=turn.get("stage", stage),
                          seconds=turn.get("seconds", 9000.0 - 400 * number),
                          can_request=turn.get("can_request", True), finish=turn.get("finish", False))
        if "input_error" in outcome:
            transitions[-1]["ok"] = scenario.get("expect_input_error", False)
            transitions[-1]["got"] = ("input_error", outcome["input_error"])
    total_legacy, total_policy = sum(legacy_sizes), sum(policy_sizes)
    return {
        "name": scenario["name"], "requests": len(legacy_sizes), "policy_requests_built": len(policy_sizes),
        "legacy_total_bytes": total_legacy, "policy_total_bytes": total_policy,
        "legacy_max_request": max(legacy_sizes, default=0), "policy_max_request": max(policy_sizes, default=0),
        "legacy_sizes": legacy_sizes, "policy_sizes": policy_sizes,
        "reduction_percent": round(100 * (1 - total_policy / total_legacy), 1) if total_legacy else None,
        "transitions": transitions, "facts": facts,
        "static_prefix_stable": len(set(prefixes)) <= 1, "diagnostics": diagnostics,
    }


def verify_unsupported_and_mixed() -> dict:
    """Spot checks outside the replay: incompatible input and an impossible limit."""
    results = {}
    try:
        build_model_messages(system_prompt="s", task_text="t", contract=contract(), history=[],
                             snapshot={"version": 2}, limits=PolicyLimits(), redact=redact)
        results["bad_snapshot_rejected"] = False
    except PolicyInputError:
        results["bad_snapshot_rejected"] = True
    try:
        build_model_messages(system_prompt="s", task_text="t", contract=contract(), history=[],
                             snapshot=snapshot_for(ResearchLedger(), [], stage="research", seconds=100.0,
                                                   requests=3, can_request=True, finish=False,
                                                   best=rmse(10.0), unverified=False),
                             limits=PolicyLimits(max_messages_bytes=500), redact=redact)
        results["impossible_limit_rejected"] = False
    except PolicyBudgetExceeded:
        results["impossible_limit_rejected"] = True
    return results


# ===================================================================== v3 series
# Mock host projection. The real host builds the same two documents from its own state; the policy
# receives only INNER facts. ``OUTER_SENTINEL`` stands for outer/test information that exists in the
# mock host but is never put into the snapshot, so it must never influence a message or a decision.

OUTER_SENTINEL = "OUTER-SENTINEL-731"
NUDGE = ("Research is not complete: test a hypothesis with inner validation. Write fresh "
         "research_candidate.py, research_recipe.json and research_hypothesis.json in one tool call.")
PAYLOAD_LIMIT, ESTIMATE_LIMIT, OUTPUT_RESERVE = 64 * 1024, 100000, 8192  # the v3 request guard


def v3_fixture() -> dict:
    """Real SYSTEM_V3, real _context text and a real v3 manifest from a tiny synthetic task."""
    import csv
    from types import SimpleNamespace
    from agent.research_v3 import SYSTEM_V3, _context
    from agent.research_workspace import ResearchWorkspace

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    with tempfile.TemporaryDirectory(prefix="nika-v3-fixture-") as directory:
        root = Path(directory)
        data, task = root / "source" / "data", root / "source" / "original.md"
        task.parent.mkdir(parents=True)
        task.write_text("Primary metric: RMSE\nPredict the answer from the measurement.\n", encoding="utf-8")
        write(data / "data_dictionary.csv", [["feature", "role", "dtype"], ["key", "id", "category"],
                                              ["measurement", "feature", "numeric"], ["answer", "target", "numeric"]])
        write(data / "train.csv", [["key", "measurement", "answer"],
                                   *[[f"train-{i:04d}", i, 2 * i + 1] for i in range(120)]])
        write(data / "test.csv", [["key", "measurement"], ["0007", 7], ["NA", 11]])
        write(data / "sample_submission.csv", [["key", "answer"], ["0007", ""], ["NA", ""]])
        view = ResearchWorkspace(task, root / "view")
        baseline = {"name": "RMSE", "value": 4.2, "higher_is_better": False}
        state = SimpleNamespace(
            best_inner={**baseline, "purpose": "selection", "split": "inner"},
            gate=SimpleNamespace(inner_split=view.inner_split),
            report={"baseline_search": {"status": "fixture_or_unavailable"}, "history_research": None,
                    "baseline_inner_diagnostics": None},
            baseline=SimpleNamespace(recipe={"action": "ridge", "model_config": {"alpha": 1.0}}))
        return {"system": SYSTEM_V3, "task_text": view.task_text, "baseline_metric": baseline,
                "manifest": json.loads(json.dumps(view.manifest)), "context": _context(state)}


def v3_budget_message(attempts: int, seconds_left: float) -> dict:
    """The real ``V3State.budget_context`` text, appended to a copy of every request like v3 does."""
    from types import SimpleNamespace
    from agent.research_budget import BudgetPolicy
    from agent.research_v3 import V3State
    fake = SimpleNamespace(
        budget=BudgetPolicy(10500, started=0.0, clock=lambda: 10500.0 - seconds_left),
        report={"max_steps": 12, "api": {"attempts": attempts}},
        finalization_plan=lambda: None, research_timeout=lambda *args: 1800)
    return {"role": "user", "content": "Current host budget: "
            + json.dumps(V3State.budget_context(fake), ensure_ascii=False)}


def v3_contract(fixture: dict, *, supported=True, required=1, allow_unverified=False, pacing=None,
                close_when_met=False) -> dict:
    """Mock host projection of the manifest into the policy contract (research-dialogue mode)."""
    manifest = fixture["manifest"]
    completion = {"required_verified_trials": required, "allow_unverified": allow_unverified,
                  "mode": "research_dialogue"}
    if close_when_met:
        completion["close_when_met"] = True
    contract = {
        "version": 1, "manifest_version": manifest["version"], "manifest_path": "research_validation.json",
        "host_instructions": redact(fixture["context"]),
        "allowed_inputs": list(manifest["allowed_inputs"]), "required_outputs": list(manifest["required_outputs"]),
        "validation": {"supported": supported, "metric": fixture["baseline_metric"] if supported else None,
                       "scope": "inner_selection"},
        "completion": completion}
    if pacing:
        contract["pacing"] = pacing
    return contract


def v3_inner_diagnostics(rng: random.Random, value: float, *, classification=False) -> dict:
    """Shaped like ``inner_feedback`` output (bounded by the host at 8 KiB)."""
    if classification:
        classes = [f"class_{i:02d}" for i in range(30)]
        rows = sorted(({"label": label, "support": rng.randint(5, 90), "recall": round(rng.random(), 3),
                        "f1": round(rng.random(), 3)} for label in classes[:12]), key=lambda row: row["f1"])
        pairs = [{"actual_class": rng.choice(classes), "predicted_class": rng.choice(classes),
                  "count": rng.randint(2, 40)} for _ in range(8)]
        details = {"classes_total": 30, "observed_classes": 30, "accuracy": 0.61, "per_class": rows,
                   "worst_confusions": pairs, "omitted_classes": 18, "omitted_confusions": 40,
                   "label_rule": "argmax_in_declared_class_order"}
        key = "classification"
    else:
        details = {"mean_bias": rng.uniform(-1, 1), "mae": rng.uniform(2, 4), "absolute_error_p50": rng.uniform(1, 3),
                   "absolute_error_p90": rng.uniform(5, 8), "absolute_error_p99": rng.uniform(9, 14)}
        key = "regression"
    return {"version": 1, "scope": "inner_selection", "rows": 3000,
            "metric": {"name": "RMSE", "value": value, "higher_is_better": False}, key: details}


def v3_selection(accepted, value, reason, *, hypothesis, recipe_hash, origin="llm", verified=True,
                 diagnostics=None, code_hash="0" * 12) -> dict:
    """The dictionary ``_assess_candidate`` returns (a failure returns only the first four keys)."""
    selection = {"accepted": accepted, "verified": verified, "scope": "inner_selection", "reason": reason}
    if value is None and diagnostics is None and hypothesis is None:
        return selection
    selection.update(
        metric=None if value is None else {"name": "RMSE", "value": value, "higher_is_better": False,
                                          "purpose": "selection", "split": "inner"},
        hypothesis=hypothesis, origin=origin, code_hash=code_hash, recipe_hash=recipe_hash,
        recipe_summary={"family": "gradient_boosting"},
        timing={"estimated_finalization_seconds": 410.5, "remaining_seconds": 8000, "feasible_estimate": True,
                "within_stage_limit": True, "guaranteed": False})
    if diagnostics is not None:
        selection["diagnostics"] = diagnostics
    return selection


def v3_result(rng: random.Random, selection: dict, *, stdout=None, extra=None) -> dict:
    result = {"stdout": stdout if stdout is not None else fold_log(rng), "stderr": "", "exit_code": 0,
              "timed_out": False, "memory_limit_exceeded": False, "elapsed_seconds": round(rng.uniform(15, 200), 3),
              "resources": {"available": True, "peak_rss_bytes": 900_000_000, "cpu_seconds": 120.5,
                            "memory_limit_bytes": 8_000_000_000, "sample_count": 40},
              "selection": selection}
    result.update(extra or {})
    return result


def v3_failed(stderr: str, *, exit_code=1, timed_out=False, error=None, extra=None) -> dict:
    result = {"stdout": "", "stderr": stderr, "exit_code": exit_code, "timed_out": timed_out,
              "memory_limit_exceeded": False, "elapsed_seconds": 12.5,
              "selection": {"accepted": False, "verified": False, "reason": "execution_failed",
                            "scope": "inner_selection"}}
    if error:
        result["error"] = error
    result.update(extra or {})
    return result


def v3_analysis(stdout: str) -> dict:
    return {"stdout": stdout, "stderr": "", "exit_code": 0, "timed_out": False, "memory_limit_exceeded": False,
            "elapsed_seconds": 4.2, "selection": {
                "accepted": False, "verified": False, "scope": "inner_selection",
                "reason": "[Errno 2] No such file or directory: '/work/runs/ws/research_hypothesis.json'"}}


def v3_row(number: int, code: str, result: dict, kind, *, origin="llm", recipe_hash=None) -> dict:
    chosen = result.get("selection") or {}
    text = result.get("stderr") or result.get("error")
    row = {"number": number, "code_hash": hashlib.sha1(code.encode()).hexdigest()[:12],
           "recipe_hash": recipe_hash, "origin": origin, "hypothesis": chosen.get("hypothesis"),
           "accepted": bool(chosen.get("accepted")), "verified": bool(chosen.get("verified")),
           "metric": ({key: chosen["metric"][key] for key in ("name", "value", "higher_is_better")}
                      if chosen.get("metric") else None),
           "reason": chosen.get("reason"), "error_kind": kind, "error": text[-1500:] if text else None}
    if chosen.get("diagnostics"):
        row["diagnostics"] = chosen["diagnostics"]
    return row


def v3_snapshot(rows, *, stage, seconds, research_seconds, requests, best, finish=False, can_request=True,
                closed=False, host_can_finalize=True) -> dict:
    llm = [row for row in rows if row["origin"] == "llm"]
    streak = 0
    for row in reversed(rows):
        if row["error_kind"] in (None, "api"):
            break
        streak += 1
    return {
        "version": 1, "stage": stage,
        "budget": {"remaining_seconds": seconds, "remaining_requests": requests, "can_request": can_request,
                   "research_seconds_available": research_seconds},
        "baseline": {"status": "measured", "retained_output_valid": False, "best_verified_metric": best},
        "last_trial": rows[-1] if rows else None, "recent_trials": rows[-6:],
        "counts": {"verified_trials": sum(row["verified"] for row in rows),
                   "unverified_trials": sum(1 for row in rows if row["accepted"] and not row["verified"]),
                   "consecutive_execution_failures": streak,
                   "verified_llm_trials": sum(row["verified"] for row in llm),
                   "unverified_llm_trials": sum(1 for row in llm if row["accepted"] and not row["verified"]),
                   "host_local_trials": sum(1 for row in rows if row["origin"] == "local" and row["verified"]),
                   "host_history_trials": sum(1 for row in rows if row["origin"] == "history" and row["verified"])},
        "finish_requested": finish,
        "finalization": {"verified_candidate": any(row["accepted"] and row["verified"] for row in rows),
                         "host_can_finalize": host_can_finalize, "research_closed": closed}}


def guard_block(sizes: list[int]) -> int | None:
    """First request (1-based) that the v3 host guard would refuse, or None."""
    reserved = 0
    for index, size in enumerate(sizes, start=1):
        estimate = math.ceil(size / 3) + OUTPUT_RESERVE
        if size > PAYLOAD_LIMIT or reserved + estimate > ESTIMATE_LIMIT:
            return index
        reserved += estimate
    return None


def build_v3_scenarios() -> list[dict]:
    rng = random.Random(20261021)
    hyp = lambda n: f"Trial {n}: change the boosting depth and add target-aware smoothing for feature {n}"

    def trial(n, accepted, value, reason, *, recipe, classification=False, verified=True, origin="llm",
              object_hypothesis=False):
        text = hyp(n)
        selection = v3_selection(accepted, value, reason, hypothesis={"hypothesis": text} if object_hypothesis else text,
                                 recipe_hash=recipe, origin=origin, verified=verified,
                                 diagnostics=v3_inner_diagnostics(rng, value or 0.0, classification=classification))
        return {"code": code_for(n), "result": v3_result(rng, selection), "recipe": recipe, "kind": None}

    first = ("experiment", "continue", "first_hypothesis")
    improved = ("experiment", "continue", "improved_continue")
    worse = ("experiment", "continue", "not_better_continue")
    done = ("finished", "complete", "dialogue_complete")
    scenarios = [
        {"name": "big_inspection", "steps": [
            {"expect": ("inspect", "continue", "inspect_dialogue"), "stage": "inspect"},
            {"calls": [{"code": "print(df.describe())", "result": v3_analysis(inspection_stdout(rng)),
                        "recipe": None, "kind": None}], "expect": first},
            {"calls": [trial(2, True, 3.9, "improved", recipe="d3")], "expect": improved},
            {"calls": [trial(3, False, 4.1, "not_better", recipe="d6")], "expect": worse},
            {"finish": True, "decide_only": True, "expect": done}]},
        {"name": "several_trials", "steps": [
            {"expect": first},
            *[{"calls": [trial(n, accepted, value, reason, recipe=f"d{n}", classification=n % 2 == 0)],
               "expect": expected} for n, accepted, value, reason, expected in (
                (1, True, 4.0, "improved", improved), (2, False, 4.3, "not_better", worse),
                (3, False, 3.8, "refit_budget", ("experiment", "continue", "refit_budget_continue")),
                (4, False, 4.4, "not_better", worse), (5, True, 3.7, "improved", improved),
                (6, False, 4.0, "not_better", worse))],
            {"finish": True, "decide_only": True, "expect": done}]},
        {"name": "traceback_and_fix", "steps": [
            {"expect": first},
            {"calls": [{"code": code_for(1), "result": v3_failed(
                traceback_text(450, "ValueError: could not convert string to float: 'x12' (unique-final-error-line)"),
                extra={"candidate_execution": {"phase": "inner_selection", "exit_code": 1}}),
                "recipe": "d3", "kind": "execution"}],
             "expect": ("repair", "continue", "repair_execution_error")},
            {"calls": [trial(2, True, 3.9, "improved", recipe="d3")], "expect": improved}]},
        {"name": "multi_tool_group", "steps": [
            {"expect": first},
            {"calls": [{"code": "print(df.shape)", "result": v3_analysis(table_text(rng, 60, 8)),
                        "recipe": None, "kind": None},
                       trial(2, True, 3.95, "improved", recipe="d3"),
                       trial(3, False, 4.2, "not_better", recipe="d6")], "expect": worse}]},
        {"name": "early_finish", "contract": {"required": 2}, "steps": [
            {"expect": first},
            {"plain": "I think the research is done.", "finish": True,
             "expect": ("experiment", "continue", "finish_too_early")},
            {"calls": [trial(2, True, 3.9, "improved", recipe="d3")], "finish": True,
             "expect": ("experiment", "continue", "finish_too_early")},
            {"calls": [trial(3, False, 4.1, "not_better", recipe="d6")], "finish": True, "decide_only": True,
             "expect": done}]},
        {"name": "low_time", "contract": {"pacing": {"low_remaining_seconds": 300, "low_remaining_requests": 1}},
         "steps": [
            {"seconds": 250.0, "research_seconds": 120.0, "expect": first, "tight": True},
            {"calls": [trial(1, True, 3.9, "improved", recipe="d3")], "seconds": 190.0,
             "research_seconds": 90.0, "expect": improved, "tight": True},
            {"calls": [trial(2, False, 4.1, "not_better", recipe="d6")], "seconds": 40.0,
             "research_seconds": 0.0, "decide_only": True,
             "expect": ("finished", "fallback", "dialogue_handover_resources")}]},
        {"name": "api_error", "steps": [
            {"expect": first},
            {"calls": [trial(1, True, 3.9, "improved", recipe="d3")], "expect": improved},
            {"api_error": True, "expect": ("experiment", "continue", "retry_after_api_error")},
            {"calls": [trial(3, False, 4.1, "not_better", recipe="d6")], "expect": worse}]},
        {"name": "unsupported_metric", "contract": {"supported": False, "allow_unverified": True}, "steps": [
            {"expect": first},
            {"calls": [{"code": code_for(1), "result": v3_result(rng, {
                "accepted": True, "verified": False, "reason": "host_metric_unsupported",
                "scope": "inner_selection", "hypothesis": hyp(1)}), "recipe": "d3", "kind": None}],
             "expect": ("experiment", "continue", "unverified_continue")},
            {"finish": True, "decide_only": True,
             "expect": ("finished", "complete", "dialogue_unverified_complete")}]},
        {"name": "incomplete_group", "expect_input_error": True, "steps": [
            {"expect": first},
            {"calls": [trial(1, True, 3.9, "improved", recipe="d3")], "pending_tail": True, "expect": improved}]},
        {"name": "history_local_feedback", "initial_rows": [
            {"number": 1, "code_hash": "h" * 12, "recipe_hash": "hist-1", "origin": "history",
             "hypothesis": {"hypothesis": "Blend the frozen baseline with strictly past target history."},
             "accepted": True, "verified": True, "metric": rmse(3.95), "reason": "improved",
             "error_kind": None, "error": None},
            {"number": 2, "code_hash": "l" * 12, "recipe_hash": "loc-1", "origin": "local",
             "hypothesis": {"hypothesis": "Bounded host-local alternative: depth 6"},
             "accepted": False, "verified": True, "metric": rmse(4.05), "reason": "refit_budget",
             "error_kind": None, "error": None}], "steps": [
            {"expect": ("experiment", "continue", "refit_budget_continue")},
            {"calls": [trial(3, True, 3.85, "improved", recipe="d3", object_hypothesis=True)], "finish": True,
             "decide_only": True, "expect": done}]},
    ]
    return scenarios


def run_v3_scenario(fixture: dict, scenario: dict, limits: PolicyLimits) -> dict:
    host_contract = v3_contract(fixture, **scenario.get("contract", {}))
    context_text = redact(fixture["context"])
    system, task = fixture["system"], fixture["task_text"]
    control = [{"role": "system", "content": system}, {"role": "user", "content": task},
               {"role": "user", "content": context_text}]
    history, rows = [], list(scenario.get("initial_rows", []))
    best = dict(fixture["baseline_metric"])
    for row in rows:
        if row["accepted"] and row["verified"]:
            best = row["metric"]
    sizes = {"control": [], "ablation": [], "policy": []}
    transitions, facts, diagnostics, prefixes = [], [], [], []
    attempts = 0
    for position, step in enumerate(scenario["steps"]):
        if step.get("plain") is not None:
            history.append({"role": "assistant", "content": step["plain"]})
            control += [history[-1], {"role": "user", "content": NUDGE}]
        if step.get("calls"):
            identifiers = [f"call_{position}_{index}" for index in range(len(step["calls"]))]
            assistant = {"role": "assistant", "content": "Next I test the following idea.", "tool_calls": [
                {"id": identifier, "type": "function", "function": {
                    "name": "run_python", "arguments": json.dumps({"code": item["code"]})}}
                for identifier, item in zip(identifiers, step["calls"])]}
            history.append(assistant)
            control.append(assistant)
            for identifier, item in zip(identifiers, step["calls"]):
                row = v3_row(len(rows) + 1, item["code"], item["result"], item["kind"], recipe_hash=item["recipe"])
                rows.append(row)
                if row["accepted"] and row["verified"]:
                    best = row["metric"]
                message = {"role": "tool", "tool_call_id": identifier,
                           "content": json.dumps(item["result"], ensure_ascii=False, allow_nan=False)}
                history.append(message)
                control.append(message)
            if step.get("pending_tail"):
                history.pop()
        if step.get("api_error"):
            rows.append({"number": len(rows) + 1, "code_hash": None, "recipe_hash": None, "origin": "llm",
                         "hypothesis": None, "accepted": False, "verified": False, "metric": None,
                         "reason": "api_error", "error_kind": "api", "error": "HTTP 429 (rate_limit)"})
        seconds = step.get("seconds", 9000.0 - 500.0 * position)
        research = step.get("research_seconds", max(0.0, seconds - 3200.0))
        snapshot = v3_snapshot(rows, stage=step.get("stage", "research"), seconds=seconds, research_seconds=research, requests=max(0, 12 - attempts),
                               best=best, finish=step.get("finish", False))
        decision = decide_next_action(snapshot, contract=host_contract)
        got = (decision.stage, decision.action, decision.reason)
        expected = tuple(step["expect"])
        transitions.append({"step": position, "expected": expected, "got": got, "ok": got == expected})
        if step.get("tight") is not None:
            transitions[-1]["ok"] = transitions[-1]["ok"] and ("tight" in decision.instruction) == step["tight"]
        if step.get("decide_only") or decision.action != "continue":
            continue
        attempts += 1
        budget = v3_budget_message(attempts, seconds)
        try:
            context = build_model_messages(system_prompt=system, task_text=task, contract=host_contract,
                                           history=history, snapshot=snapshot, limits=limits, redact=redact)
        except PolicyInputError as error:
            transitions[-1].update(ok=scenario.get("expect_input_error", False),
                                   got=("input_error", str(error)))
            continue
        policy_messages = context.messages + [budget]
        contract_message = context.messages[2]
        control_size = json_size(control + [budget])
        ablation_size = json_size([context.messages[0], context.messages[1], contract_message, *control[3:], budget])
        sizes["control"].append(control_size)
        sizes["ablation"].append(ablation_size)
        sizes["policy"].append(json_size(policy_messages))
        diagnostics.append(context.diagnostics)
        prefixes.append(json.dumps(context.messages[:3], ensure_ascii=False))
        text = json.dumps(policy_messages, ensure_ascii=False)
        summary = context.messages[-1]["content"]
        state = json.loads(summary.split("\n")[1])
        last = snapshot["last_trial"]
        record = {"step": position, "within_limit": context.diagnostics["after_bytes"] <= limits.max_messages_bytes,
                  "contract_verbatim": context_text in contract_message["content"],
                  "no_secret": FAKE_KEY not in text, "no_outer_sentinel": OUTER_SENTINEL not in text,
                  "best_metric_present": str(best["value"]) in summary,
                  "separate_counters": all(name in state["counts"] for name in (
                      "verified_llm_trials", "host_local_trials", "host_history_trials"))}
        if last is not None and state["recent_trials"]:
            entry = state["recent_trials"][-1]
            record["typed_facts_kept"] = (entry["accepted"] is last["accepted"] and entry["verified"] is last["verified"]
                                          and entry["metric"] == (None if last["metric"] is None else last["metric"]["value"]))
        facts.append(record)
    return {
        "name": scenario["name"], "requests": len(sizes["policy"]),
        "control_sizes": sizes["control"], "ablation_sizes": sizes["ablation"], "policy_sizes": sizes["policy"],
        "control_total_bytes": sum(sizes["control"]), "ablation_total_bytes": sum(sizes["ablation"]),
        "policy_total_bytes": sum(sizes["policy"]),
        "control_max_request": max(sizes["control"], default=0), "policy_max_request": max(sizes["policy"], default=0),
        "policy_larger_steps": [index + 1 for index, (c, p) in enumerate(zip(sizes["control"], sizes["policy"])) if p > c],
        "estimated_tokens": {"control": sum(math.ceil(size / 3) for size in sizes["control"]),
                             "policy": sum(math.ceil(size / 3) for size in sizes["policy"]),
                             "basis": "ceil(UTF-8 bytes / 3) per request; not provider usage"},
        "guard_blocked_at_request": {"control": guard_block(sizes["control"]), "policy": guard_block(sizes["policy"])},
        "transitions": transitions, "facts": facts, "static_prefix_stable": len(set(prefixes)) <= 1,
        "diagnostics": diagnostics}


def v3_markdown(results: list[dict]) -> str:
    lines = ["| Scenario | Requests | Control, B | Ablation, B | Policy, B | vs control | vs ablation | "
             "Policy larger at | Guard blocks (control / policy) | Transitions | Facts |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | :---: | :---: | :---: |"]
    for item in results:
        change = lambda base: ("n/a" if not item[base] else f"{round(100 * (item['policy_total_bytes'] / item[base] - 1), 1):+}%")
        transitions = f"{sum(t['ok'] for t in item['transitions'])}/{len(item['transitions'])}"
        checks = [value for record in item["facts"] for key, value in record.items() if key != "step"]
        guard = item["guard_blocked_at_request"]
        lines.append(f"| {item['name']} | {item['requests']} | {item['control_total_bytes']} | "
                     f"{item['ablation_total_bytes']} | {item['policy_total_bytes']} | {change('control_total_bytes')} | "
                     f"{change('ablation_total_bytes')} | {', '.join(map(str, item['policy_larger_steps'])) or '-'} | "
                     f"{guard['control'] or '-'} / {guard['policy'] or '-'} | {transitions} | "
                     f"{sum(checks)}/{len(checks)} |")
    return "\n".join(lines)


def resolve_output_dir(requested: Path) -> tuple[Path, str]:
    """Heavy artefacts go only to an ignored directory; otherwise to a temporary one."""
    target = requested if requested.is_absolute() else ROOT / requested
    try:
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(target)], cwd=ROOT,
                                 capture_output=True, check=False).returncode == 0
    except OSError:
        ignored = False
    if ignored:
        target.mkdir(parents=True, exist_ok=True)
        return target, "ignored repository directory"
    fallback = Path(tempfile.mkdtemp(prefix="nika-research-policy-"))
    return fallback, "temporary directory (the requested path is not ignored by git)"


def markdown_table(results: list[dict]) -> str:
    lines = ["| Scenario | Requests | Legacy total, B | Policy total, B | Legacy max, B | Policy max, B | Reduction | Transitions | Facts |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |"]
    for item in results:
        transitions = f"{sum(t['ok'] for t in item['transitions'])}/{len(item['transitions'])}"
        checks = [value for record in item["facts"] for key, value in record.items() if key != "request"]
        facts = f"{sum(checks)}/{len(checks)}"
        reduction = "n/a" if item["reduction_percent"] is None else f"{item['reduction_percent']}%"
        lines.append(f"| {item['name']} | {item['requests']} | {item['legacy_total_bytes']} | "
                     f"{item['policy_total_bytes']} | {item['legacy_max_request']} | {item['policy_max_request']} | "
                     f"{reduction} | {transitions} | {facts} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="run")
    parser.add_argument("--series", choices=("both", "legacy", "v3"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-messages-bytes", type=int, default=PolicyLimits().max_messages_bytes)
    parser.add_argument("--max-feedback-bytes", type=int, default=PolicyLimits().max_tool_feedback_bytes)
    args = parser.parse_args(argv)
    limits = PolicyLimits(max_messages_bytes=args.max_messages_bytes,
                          max_tool_feedback_bytes=args.max_feedback_bytes)
    payload = {"label": args.label, "limits": vars(limits), "api_attempts": 0, "provider_reported_tokens": None}
    problems = []
    if args.series in ("both", "legacy"):
        results = [run_scenario(scenario, limits) for scenario in build_scenarios()]
        extra = verify_unsupported_and_mixed()
        payload.update(results=results, extra_checks=extra)
        print("LEGACY REPLAY (reconstruction of the pre-v3 runner; not a v3 measurement)")
        print(markdown_table(results))
        total_legacy = sum(item["legacy_total_bytes"] for item in results)
        total_policy = sum(item["policy_total_bytes"] for item in results)
        print(f"\nLegacy series total: control {total_legacy} B, policy {total_policy} B "
              f"({round(100 * (1 - total_policy / total_legacy), 1)}% less); bytes, not tokens or roubles.")
        failed = [(item["name"], t) for item in results for t in item["transitions"] if not t["ok"]]
        bad = [(item["name"], record) for item in results for record in item["facts"]
               if not all(value for key, value in record.items() if key != "request")]
        print("Transition mismatches:", failed or "none")
        print("Fact-check failures:", bad or "none")
        print("Extra checks:", extra)
        problems += failed + bad + ([] if all(extra.values()) else [("extra", extra)])
    if args.series in ("both", "v3"):
        fixture = v3_fixture()
        v3_results = [run_v3_scenario(fixture, scenario, limits) for scenario in build_v3_scenarios()]
        payload.update(v3_results=v3_results, v3_meta={
            "system_v3_bytes": len(fixture["system"].encode("utf-8")),
            "task_text_bytes": len(fixture["task_text"].encode("utf-8")),
            "context_bytes": len(fixture["context"].encode("utf-8")),
            "manifest_version": fixture["manifest"]["version"],
            "request_guard": {"payload_limit": PAYLOAD_LIMIT, "estimated_token_limit": ESTIMATE_LIMIT,
                              "output_reserve": OUTPUT_RESERVE}})
        print("\nRESEARCH v3 (real SYSTEM_V3, _context, manifest and budget message; control = history as run_agent_v3 keeps it)")
        print(v3_markdown(v3_results))
        control, policy = (sum(item["control_total_bytes"] for item in v3_results),
                           sum(item["policy_total_bytes"] for item in v3_results))
        ablation = sum(item["ablation_total_bytes"] for item in v3_results)
        print(f"\nv3 series total: control {control} B, ablation {ablation} B, policy {policy} B "
              f"({round(100 * (policy / control - 1), 1):+}% vs control, {round(100 * (policy / ablation - 1), 1):+}% "
              "vs ablation); bytes, not tokens or roubles; API attempts 0, provider-reported tokens none.")
        larger = {item["name"]: item["policy_larger_steps"] for item in v3_results if item["policy_larger_steps"]}
        print("Requests where the policy is larger than the control:", larger or "none")
        failed = [(item["name"], t) for item in v3_results for t in item["transitions"] if not t["ok"]]
        bad = [(item["name"], record) for item in v3_results for record in item["facts"]
               if not all(value for key, value in record.items() if key != "step")]
        print("v3 transition mismatches:", failed or "none")
        print("v3 fact-check failures:", bad or "none")
        problems += failed + bad
    directory, where = resolve_output_dir(args.output_dir)
    target = directory / f"results-{args.label}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {target} ({where})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
