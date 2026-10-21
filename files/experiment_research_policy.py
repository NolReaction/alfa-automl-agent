"""Offline replay: uncompressed history versus the bounded research policy. Owner: Nika.

No API request, no code execution, no dataset. Scenarios are synthetic but shaped
like real runner results. The "legacy" series reproduces what ``agent/research.py``
sends today: every tool result stays in ``messages`` and a fresh ledger summary is
appended before each request. Sizes are UTF-8 bytes of compact JSON, not tokens and
not money. This is a replay of transcripts, not evidence about ML quality.

    python experiment_research_policy.py [--label NAME] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-messages-bytes", type=int, default=PolicyLimits().max_messages_bytes)
    parser.add_argument("--max-feedback-bytes", type=int, default=PolicyLimits().max_tool_feedback_bytes)
    args = parser.parse_args(argv)
    limits = PolicyLimits(max_messages_bytes=args.max_messages_bytes,
                          max_tool_feedback_bytes=args.max_feedback_bytes)
    results = [run_scenario(scenario, limits) for scenario in build_scenarios()]
    extra = verify_unsupported_and_mixed()
    directory, where = resolve_output_dir(args.output_dir)
    target = directory / f"results-{args.label}.json"
    target.write_text(json.dumps({"label": args.label, "limits": vars(limits), "results": results,
                                  "extra_checks": extra}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(markdown_table(results))
    total_legacy = sum(item["legacy_total_bytes"] for item in results)
    total_policy = sum(item["policy_total_bytes"] for item in results)
    print(f"\nSeries total: legacy {total_legacy} B, policy {total_policy} B "
          f"({round(100 * (1 - total_policy / total_legacy), 1)}% less); sizes are bytes, not tokens or roubles.")
    failed = [(item["name"], t) for item in results for t in item["transitions"] if not t["ok"]]
    bad_facts = [(item["name"], record) for item in results for record in item["facts"]
                 if not all(value for key, value in record.items() if key != "request")]
    print("Transition mismatches:", failed or "none")
    print("Fact-check failures:", bad_facts or "none")
    print("Extra checks:", extra)
    print(f"Saved: {target} ({where})")
    return 1 if (failed or bad_facts or not all(extra.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
