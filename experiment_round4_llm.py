"""Paired, no-network protocol measurement of stock and proposed Qwen context.

The unchanged v3 host executes/scorers/freezes/refits fixture candidates. A
test-only adapter prepares messages before the original request guards. This
does not activate policy in the competition host, model live Qwen behavior, or
compare production AutoML quality: baseline portfolio search and local recipe
selection are explicitly bounded fixtures in both arms.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch
from uuid import uuid4

import httpx
from openai import OpenAI

from agent import research_v3 as host
from agent.api_probe import request_size, timestamps
from agent.gateways import ALTERNATE_BASE_URL
from agent.llm import DEFAULT_MODEL, REQUIRED_API_FIELDS, ROOT, load_local_env, public_error_message, safe_error_details
from agent.research_candidate import freeze_candidate
from experiment_research_smoke import source_provenance, summarize_run
from tests.test_research import make_regression
from tests.test_team_integration import candidate_tool


BASE_SHA = "e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb"
ARMS = ("main_context", "round4_context")
CASES = ("useful", "analysis_only", "no_training", "exact_duplicate", "late_useful")
FIT_MARKER = "ROUND4_COMPLETED_FIXTURE_FIT"
FIXTURE_KEY = "round4-mock-only-no-real-key"

BASELINE_CODE = f'''import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    mean = np.asarray(y_fit, dtype=float).mean()
    print({FIT_MARKER!r})
    return {{'predictions': np.full(len(X_predict), mean)}}
'''
LINEAR_CODE = f'''import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    slope, intercept = np.polyfit(np.asarray(X_fit['measurement'], dtype=float),
                                 np.asarray(y_fit, dtype=float), 1)
    print({FIT_MARKER!r})
    return {{'predictions': slope * np.asarray(X_predict['measurement'], dtype=float) + intercept}}
'''
NO_TRAINING_CODE = '''import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    return {'predictions': np.zeros(len(X_predict))}
'''


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _script(case):
    useful = candidate_tool(LINEAR_CODE, {"family": "linear", "degree": 1})
    if case == "useful":
        return [("tool", useful), ("text", "Measured trial complete.")]
    if case == "analysis_only":
        return [("text", "I will inspect and plan an experiment.")] * 3
    if case == "no_training":
        return [("tool", candidate_tool(NO_TRAINING_CODE, {"family": "zero_without_training"})),
                ("text", "Candidate process completed.")]
    if case == "exact_duplicate":
        return [("tool", useful)] * 3 + [("text", "Measured trial complete.")]
    if case == "late_useful":
        return [("text", "I will inspect and plan an experiment.")] * 2 + [
            ("tool", useful), ("text", "Measured trial complete.")]
    raise ValueError("Unknown pre-registered case")


def _reply(request, entry, ordinal):
    kind, content = entry
    message = {"role": "assistant", "content": content if kind == "text" else None}
    if kind == "tool":
        message["tool_calls"] = [{"id": f"fixture-call-{ordinal}", "type": "function", "function": {
            "name": "run_python", "arguments": json.dumps({"code": content})}}]
    return httpx.Response(200, request=request, json={
        "id": "mock-response", "object": "chat.completion", "created": 1, "model": DEFAULT_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if kind == "tool" else "stop"}],
    })


def _baseline(state):
    path = state.run_dir / "fixture_baseline.py"
    path.write_text(BASELINE_CODE, encoding="utf-8")
    state.baseline = freeze_candidate(path, {"family": "fixture_mean"}, state.run_dir / "fixture_baseline_frozen")
    state.report["baseline"] = "recipe_frozen"


def _history(messages, measured):
    """Translate only pre-freeze host tool feedback, retaining complete history."""
    rows = []
    for message in messages:
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            rows.append({"scope": "inner_selection", "outcome": "analysis_only"})
        if message.get("role") != "tool":
            continue
        result = json.loads(message["content"])
        selection = result.get("selection", {})
        if selection.get("verified") is True and selection.get("origin") == "llm":
            key = (selection["code_hash"], selection["recipe_hash"])
            identity = {"code_hash": key[0], "recipe_hash": key[1]}
            if key in measured:
                identity["recipe"] = measured[key]
            row = {"scope": "inner_selection", "outcome": "verified_inner", "identity": identity,
                   "metric": selection["metric"], "accepted": selection["accepted"]}
            if isinstance(selection.get("elapsed_seconds"), (float, int)):
                row["elapsed_seconds"] = selection["elapsed_seconds"]
            rows.append(row)
        else:
            rows.append({"scope": "inner_selection", "outcome":
                         "no_candidate" if result.get("exit_code") == 0 else "execution_failed"})
    return rows


def _snapshot(state):
    """Harness-owned trusted producer; no complete report crosses helper API."""
    gate, spec = state.gate, state.gate.spec
    from agent.round4_llm_evidence import profile_inner_fit
    features = [{"name": name, "role": "numeric" if name in gate.numeric else "categorical"}
                for name in gate.features]
    snapshot = {"version": 1, "scope": "inner_selection",
            "task": {"task_type": spec.metric.task_type, "metric": {
                "name": spec.metric.name, "higher_is_better": spec.metric.higher_is_better},
                "prediction_kind": spec.prediction_kind, "classes": list(spec.classes),
                "probability_classes": [label for _, label in spec.probability_columns],
                "positive_class": spec.positive_class},
            "allowed_inputs": list(gate.manifest["allowed_inputs"]),
            "features": features,
            "shape": {"fit_rows": len(gate.fit_rows), "validation_rows": len(gate.held_rows),
                      "feature_count": len(gate.features)},
            "split": {"strategy": gate.inner_split["strategy"], "seed": gate.inner_split.get("seed")},
            "baseline": ({"code_hash": state.baseline.code_hash, "recipe_hash": state.baseline.recipe_hash,
                          "recipe": state.baseline.recipe} if state.baseline else None),
            "current_metric": state.best_inner,
            "diagnostics": state.report.get("baseline_inner_diagnostics"),
            "timing": {"remaining_seconds": state.remaining(), "final_reserve_seconds": state.final_reserve,
                       "maximum_experiment_seconds": state.research_timeout(), "last_fit_seconds": None,
                       "memory_limit_bytes": None, "cpu_threads": 8},
            "installed_tools": [name for name in ("numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm", "xgboost")
                                if importlib.util.find_spec(name) is not None]}
    snapshot["profile"] = profile_inner_fit(gate.research_train.iloc[gate.fit_rows][list(gate.features)],
                                           features, max_rows=512, max_cells=16384, max_seconds=.05)
    return snapshot


def run_arm(directory: Path, case: str, arm: str, *, time_budget=90):
    """Execute one bounded protocol arm; all native artifacts stay under directory."""
    if case not in CASES or arm not in ARMS:
        raise ValueError("Unknown case or arm")
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=False)
    task = make_regression(directory)
    inputs = sorted((directory / "regression" / "participants").glob("*.csv"))
    if {path.name for path in inputs} != {"train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"}:
        raise ValueError("Fixture inputs are missing or unexpected")
    def input_digest():
        return _hash(task.read_bytes() + b"\0" + b"".join(
            path.name.encode() + b"\0" + path.read_bytes() for path in inputs))
    input_hash = input_digest()
    script = _script(case)
    requests, preparations, executions, inner_trials, guarded_payloads, context_errors = [], [], [], [], [], []
    measured_recipes, source_state = {}, {}
    original_request, original_run, original_assess = host._request_completion, host.run_candidate, host._assess_candidate
    original_guard = host.V3State.request_allowed
    first_verified = None
    original_started = None

    def execute(candidate, **kwargs):
        started = time.monotonic()
        result = original_run(candidate, **kwargs)
        stdout = result.execution.get("stdout", "")
        resources = result.execution.get("resources", {})
        executions.append({"phase": kwargs["phase"], "code_hash": candidate.code_hash,
                           "recipe_hash": candidate.recipe_hash,
                           "completed_fixture_fits": stdout.splitlines().count(FIT_MARKER),
                           "elapsed_seconds": time.monotonic() - started,
                           "peak_rss_bytes": resources.get("peak_rss_bytes"),
                           "cpu_seconds": resources.get("cpu_seconds"),
                           "resource_measurement_reason": (None if resources.get("available") is True
                                                           else "native_process_resource_sampling_unavailable")})
        measured_recipes[(candidate.code_hash, candidate.recipe_hash)] = candidate.recipe
        return result

    def assess(state, candidate, hypothesis, *, origin):
        nonlocal first_verified
        result = original_assess(state, candidate, hypothesis, origin=origin)
        if origin == "llm" and result.get("verified") is True:
            when = time.monotonic() - state.started
            if first_verified is None:
                first_verified = when
            inner_trials.append({"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash,
                                 "metric": result["metric"], "accepted": result["accepted"],
                                 "elapsed_from_host_start": when})
        return result

    def prepare(state, client, messages):
        nonlocal original_started
        assert not state.dialogue_closed and state.report["stage"] == "inner_research"
        original_started = state.started
        gate = state.gate
        source_state.setdefault("split_sha256", _hash(_canonical({"inner_fit": gate.fit_rows,
            "inner_validation": gate.held_rows, "inner_split": gate.inner_split,
            "outer_fit": gate.outer_fit_rows.tolist(), "outer_validation": gate.outer_holdout_rows.tolist(),
            "outer_split": gate.split})))
        # Public report metadata only. Outer rows/answers never reach the policy.
        source_state.setdefault("split_summary", {
            "inner": {"strategy": gate.inner_split["strategy"], "seed": gate.inner_split.get("seed"),
                      "fit_rows": len(gate.fit_rows), "validation_rows": len(gate.held_rows)},
            "outer": {"strategy": gate.split["strategy"], "seed": gate.split.get("seed"),
                      "fit_rows": len(gate.outer_fit_rows), "validation_rows": len(gate.outer_holdout_rows)}})
        source_state.setdefault("allowed_input_sha256", _hash(b"".join(
            name.encode() + b"\0" + (gate.workspace / name).read_bytes()
            for name in sorted(gate.manifest["allowed_inputs"]))))
        if arm == "main_context":
            return original_request(state, client, messages)
        from agent.round4_llm_policy import prepare_round4_messages
        try:
            prepared = prepare_round4_messages(state.clean(messages),
                remaining_requests=max(0, state.report["max_steps"] - state.report["api"]["attempts"]),
                inner_snapshot=_snapshot(state),
                experiment_history=_history(messages, measured_recipes))
        except (ValueError, TypeError) as error:
            context_errors.append(safe_error_details(error))
            return None, "context_budget"
        preparations.append({"policy_version": prepared.policy_version, "action": prepared.action,
                             "stop_reason": prepared.stop_reason, "summary": prepared.summary})
        if prepared.action == "stop_local":
            # Existing reason retains the native local-continuation/finalization
            # path and original deadline. No helper rejection is sent unchanged.
            reason = {"attempts_exhausted": "step_limit", "research_time_exhausted": "time_limit"}.get(
                prepared.stop_reason, "repeated_failures")
            return None, reason
        return original_request(state, client, prepared.messages)

    def guard(state, messages):
        result = original_guard(state, messages)
        guarded_payloads.append({"sha256": _hash(_canonical(state.clean(messages))), "allowed": result is None})
        return result

    def transport(request):
        arguments = json.loads(request.content)
        ordinal = len(requests) + 1
        requests.append({"attempt": ordinal, **request_size(arguments),
                         "messages_sha256": _hash(_canonical(arguments["messages"])),
                         "host_elapsed_seconds": time.monotonic() - original_started,
                         "usage": None})
        if ordinal > len(script):
            raise AssertionError("Unplanned extra mock request")
        return _reply(request, script[ordinal - 1], ordinal)

    output, errors = io.StringIO(), io.StringIO()
    failure = None
    native = None
    with OpenAI(api_key=FIXTURE_KEY, base_url=ALTERNATE_BASE_URL, max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport))) as sdk, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"OPENAI_API_KEY": FIXTURE_KEY,
            "OPENAI_BASE_URL": ALTERNATE_BASE_URL, "OPENAI_MODEL": DEFAULT_MODEL,
            "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}))
        stack.enter_context(patch("httpx.HTTPTransport.handle_request", side_effect=AssertionError("Network forbidden")))
        stack.enter_context(patch.object(host, "prepare_baseline", side_effect=_baseline))
        stack.enter_context(patch.object(host, "local_candidate_plan", return_value=[]))
        stack.enter_context(patch.object(host, "run_candidate", side_effect=execute))
        stack.enter_context(patch.object(host, "_assess_candidate", side_effect=assess))
        stack.enter_context(patch.object(host, "_request_completion", side_effect=prepare))
        stack.enter_context(patch.object(host.V3State, "request_allowed", new=guard))
        try:
            with redirect_stdout(output), redirect_stderr(errors):
                native = host.run_agent_v3(task, sdk, max_steps=len(script), time_budget=time_budget)
        except Exception as error:
            failure = {"message": public_error_message(error), "details": safe_error_details(error)}
    (directory / "native_execution.log").write_text(output.getvalue() + errors.getvalue(), encoding="utf-8")
    if native is None:
        candidates = sorted((directory / ".agent_runs").glob("*/run.json"))
        native = candidates[-1].parent if candidates else None
    report = json.loads((native / "run.json").read_text(encoding="utf-8")) if native else {}
    summary = summarize_run(task, report, run_dir=native) if native else None
    hashes = [(item["code_hash"], item["recipe_hash"]) for item in inner_trials]
    candidate_hashes = {_hash(LINEAR_CODE.encode()), _hash(NO_TRAINING_CODE.encode())}
    candidate_executions = [item for item in executions if item["code_hash"] in candidate_hashes]
    unique = len(set(hashes))
    if context_errors:
        failure = {"stage": "pre_guard_context", "details": context_errors}
    unchanged_inputs = input_digest() == input_hash
    matching_guards = ([row["sha256"] for row in guarded_payloads if row["allowed"]]
                       == [row["messages_sha256"] for row in requests])
    integrity_errors = []
    if not unchanged_inputs:
        integrity_errors.append("input_integrity_changed")
    if not matching_guards:
        integrity_errors.append("guard_payload_mismatch")
    if not summary or not summary["coverage"]["finalization_verified"]:
        integrity_errors.append("native_finalization_not_verified")
    if any(not source_state.get(field) for field in ("split_sha256", "allowed_input_sha256")):
        integrity_errors.append("required_measurement_missing")
    if integrity_errors and failure is None:
        failure = {"stage": "measurement_integrity", "errors": integrity_errors}
    elapsed = time.monotonic() - started
    guard_budget = report.get("request_guard", {})
    row = {"case": case, "arm": arm, "measurement_kind": "scripted_mock_protocol",
           "helper_active_in_harness": arm == "round4_context", "helper_active_in_production": False,
           "network_requests": 0, "baseline_search": "fixed_mean_fixture_no_portfolio_search",
           "local_plan": "empty_protocol_fixture", "input_sha256": input_hash,
           "inputs_unchanged": unchanged_inputs, **source_state,
           "script_sha256": _hash(_canonical(script)), "max_requests": len(script),
           "time_budget_seconds": time_budget, "elapsed_seconds": elapsed,
           "elapsed_measurement": "whole_arm_including_task_setup_and_native_summary_validation",
           "host_elapsed_seconds": report.get("elapsed_seconds"),
           "requests": requests, "request_count": len(requests),
           "guards_match_transmitted_payloads": matching_guards,
           "payload_bytes": sum(item["request_bytes"] for item in requests),
           "estimated_tokens_reserved": guard_budget.get("estimated_tokens_reserved"),
           "estimated_token_limit": guard_budget.get("estimated_token_limit"),
           "estimated_tokens_reason": None if guard_budget else "no_native_request_guard_reservation",
           "transport_estimated_tokens": sum(item["estimated_tokens"] for item in requests),
           "transport_estimation_method": "ceil(whole_request_UTF8_bytes/3)+max_tokens",
           "provider_usage": None, "usage_reason": "mock_responses_without_usage",
           "time_to_first_verified_inner_seconds": first_verified,
           "time_to_first_verified_inner_reason": None if first_verified is not None else "no_verified_inner_trial",
           "fit_only_seconds": None, "fit_only_seconds_reason": "execution_wall_includes_process_setup_and_prediction",
           "verified_inner_trials": len(inner_trials), "unique_verified_inner_trials": unique,
           "duplicate_verified_trials": len(hashes) - unique,
           "valid_recipe_fraction": len(inner_trials) / report["experiments"] if report.get("experiments") else None,
           "valid_recipe_fraction_definition": "host_verified_inner_trials_per_executed_tool_experiment",
           "valid_recipe_fraction_reason": None if report.get("experiments") else "no_tool_experiments",
           "useful_change_fraction": sum(item["accepted"] for item in inner_trials) / len(inner_trials) if inner_trials else None,
           "useful_change_fraction_reason": None if inner_trials else "no_verified_inner_trial",
           "physical_fits": sum(item["completed_fixture_fits"] for item in executions),
           "physical_fit_method": "instrumented_known_fixture_mean_or_polyfit_completed_markers",
           "physical_fits_applicability": "only_the_fixed_fixture_sources_not_arbitrary_generated_code",
           "resource_measurement_reason": (None if any(item["peak_rss_bytes"] is not None for item in executions)
                                           else "native_process_resource_sampling_unavailable"),
           "candidate_invocations": len(candidate_executions),
           "candidate_physical_fits": sum(item["completed_fixture_fits"] for item in candidate_executions),
           "executions": executions, "trials": inner_trials, "policy_preparations": preparations,
           "failure": failure, "smoke": summary}
    _write(directory / "public_summary.json", row)
    return row


def run_pairs(output: Path, *, cases=CASES, time_budget=90):
    output.mkdir(parents=True, exist_ok=False)
    # Presence only, before fixture credentials temporarily enter os.environ.
    load_local_env()
    settings_present = {name: bool(os.environ.get(name, "").strip()) for name in REQUIRED_API_FIELDS}
    provenance = source_provenance()
    harness_hash = _hash(Path(__file__).read_bytes())
    started = timestamps()
    rows = []
    for case in cases:
        for arm in ARMS:
            rows.append(run_arm(output / case / arm, case, arm, time_budget=time_budget))
    pairs = []
    for case in cases:
        before, after = [row for row in rows if row["case"] == case]
        same = all(before.get(field) is not None and before.get(field) == after.get(field) for field in
                   ("input_sha256", "split_sha256", "allowed_input_sha256", "script_sha256", "max_requests", "time_budget_seconds"))
        same = same and before["inputs_unchanged"] and after["inputs_unchanged"]
        pairs.append({"case": case, "identical_inputs_splits_script_limits": same,
                      "request_delta": after["request_count"] - before["request_count"],
                      "payload_byte_delta": after["payload_bytes"] - before["payload_bytes"],
                      "physical_fit_delta": after["physical_fits"] - before["physical_fits"],
                      "unique_verified_inner_delta": after["unique_verified_inner_trials"] - before["unique_verified_inner_trials"],
                      "elapsed_delta_seconds": after["elapsed_seconds"] - before["elapsed_seconds"],
                      "model_behavior_effect": "not_measured_scripted_responses_ignore_context"})
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "openai", "httpx")}
    provenance_after = source_provenance()
    provenance_fields = ("source_commit", "source_worktree_dirty", "source_files_sha256", "harness_sha256")
    source_unchanged = (all(provenance[field] == provenance_after[field] for field in provenance_fields)
                        and harness_hash == _hash(Path(__file__).read_bytes()))
    report = {"version": 1, "base_commit": BASE_SHA, **provenance,
              "smoke_harness_sha256": provenance["harness_sha256"], "harness_sha256": harness_hash,
              "source_unchanged_during_measurement": source_unchanged,
              "source_provenance_after": {field: provenance_after[field] for field in provenance_fields},
              "started_at": started, "finished_at": timestamps(), "mode": "mock",
              "live_status": ("not_executed_mock_mode" if all(settings_present.values())
                              else "not_executed_no_credentials"),
              "api_settings_present": settings_present, "network_requests": 0,
              "policy_activation": "test_harness_only_before_native_guards",
              "quality_effect": "not_measured_fixed_baseline_protocol_fixture",
              "environment": {"python": platform.python_version(), "packages": versions, "cpu_thread_limit": 8},
              "pre_registered_cases": list(cases), "rows": rows, "pairs": pairs}
    _write(output / "round4_llm_results.json", report)
    columns = ("case", "arm", "request_count", "payload_bytes", "estimated_tokens_reserved", "elapsed_seconds",
               "time_to_first_verified_inner_seconds", "verified_inner_trials", "unique_verified_inner_trials",
               "duplicate_verified_trials", "physical_fits", "candidate_physical_fits", "input_sha256", "split_sha256")
    with (output / "round4_llm_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments" / "round4_llm")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()) and not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("Use gitignored data/ or an external directory")
    directory = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    report = run_pairs(directory)
    failures = (sum(row["failure"] is not None for row in report["rows"])
                + sum(not pair["identical_inputs_splits_script_limits"] for pair in report["pairs"])
                + int(not report["source_unchanged_during_measurement"]))
    print(f"Round4 scripted pairs: {len(report['pairs'])}; network requests: 0; harness failures: {failures}. Report: {directory / 'round4_llm_results.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
