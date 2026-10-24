"""Offline tests for the bounded research-dialogue policy (agent/research_policy.py)."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import threading
import unittest

import experiment_research_policy as lab
from agent import research_policy as policy
from agent.research_v3 import SYSTEM_V3
from agent.research_policy import (
    FAILURE_STOP_STREAK, REASON_CODES, PolicyBudgetExceeded, PolicyDecision, PolicyInputError,
    PolicyLimits, build_model_messages, compact_tool_feedback, decide_next_action, json_size,
)


KEY = "placeholder-secret-ABCDEF0123456789"
HAS_SDK = importlib.util.find_spec("openai") is not None


def redact(text: str) -> str:
    return text.replace(KEY, "[REDACTED]")


def metric(value=9.0, name="RMSE", higher=False):
    return {"name": name, "value": value, "higher_is_better": higher}


def make_contract(**changes):
    value = {
        "version": 1, "manifest_version": 2, "manifest_path": "research_validation.json",
        "host_instructions": "Fit on fit rows only.\nRefit at the stage the host allows.\nКириллица: проверка.",
        "allowed_inputs": [{"path": "fit_view.csv", "purpose": "fit rows"}],
        "required_outputs": [{"path": "submission.csv", "columns": ["id", "target"]},
                             {"path": "inner_predictions.csv", "columns": ["id", "prediction"]}],
        "validation": {"supported": True, "metric": metric(10.0), "scope": "inner_validation"},
        "completion": {"required_verified_trials": 2, "allow_unverified": False},
    }
    value.update(changes)
    return value


def make_trial(number=1, **changes):
    value = {"number": number, "code_hash": f"hash{number}", "hypothesis": f"hypothesis {number}",
             "accepted": True, "verified": True, "metric": metric(9.5), "reason": "improved",
             "error_kind": None, "error": None}
    value.update(changes)
    return value


def make_snapshot(**changes):
    value = {
        "version": 1, "stage": "research",
        "budget": {"remaining_seconds": 5000.0, "remaining_requests": 12, "can_request": True},
        "baseline": {"status": "trained", "retained_output_valid": True,
                     "best_verified_metric": metric(10.0)},
        "last_trial": None, "recent_trials": [],
        "counts": {"verified_trials": 0, "unverified_trials": 0, "consecutive_execution_failures": 0},
        "finish_requested": False,
    }
    for key, item in changes.items():
        if isinstance(item, dict) and isinstance(value.get(key), dict):
            value[key] = {**value[key], **item}
        else:
            value[key] = item
    return value


def snapshot_with(last=None, *, verified=0, unverified=0, streak=0, **changes):
    trials = [] if last is None else [last]
    return make_snapshot(last_trial=last, recent_trials=trials, counts={
        "verified_trials": verified, "unverified_trials": unverified,
        "consecutive_execution_failures": streak}, **changes)


def ok_result(**changes):
    value = {"stdout": "fold 1 rmse=9.5\n", "stderr": "", "exit_code": 0, "timed_out": False,
             "memory_limit_exceeded": False, "elapsed_seconds": 12.5,
             "resources": {"peak_rss_bytes": 1000, "cpu_seconds": 3.5, "memory_limit_bytes": 9000,
                           "sample_count": 4},
             "output_check": {"valid": True, "errors": []},
             "selection": {"accepted": True, "verified": True, "metric": metric(9.5), "reason": "improved",
                           "hypothesis": "target encoding"},
             "previous_output_restored": True}
    value.update(changes)
    return value


def make_group(number, *, code="print(1)", result=None, calls=1, content="idea"):
    ids = [f"call-{number}-{index}" for index in range(calls)]
    messages = [{"role": "assistant", "content": content, "tool_calls": [
        {"id": identifier, "type": "function",
         "function": {"name": "run_python", "arguments": json.dumps({"code": code})}} for identifier in ids]}]
    messages += [{"role": "tool", "tool_call_id": identifier,
                  "content": json.dumps(result if result is not None else ok_result())} for identifier in ids]
    return messages


def build(history=(), snapshot=None, contract=None, limits=None, **changes):
    arguments = dict(system_prompt="SYSTEM", task_text="TASK", contract=contract or make_contract(),
                     history=list(history), snapshot=snapshot or make_snapshot(),
                     limits=limits or PolicyLimits(), redact=redact)
    arguments.update(changes)
    return build_model_messages(**arguments)


def has_word(word: str, text: str) -> bool:
    """Whole-word search: 'actual' must not match inside 'factual'."""
    return re.search(rf"(?<![\w.]){re.escape(word)}(?![\w])", text) is not None


def decide(snapshot=None, contract=None):
    return decide_next_action(snapshot or make_snapshot(), contract=contract or make_contract())


class LimitsAndSizeTests(unittest.TestCase):
    def test_defaults_and_zero_counts_are_allowed(self):
        limits = PolicyLimits()
        self.assertEqual((limits.max_messages_bytes, limits.max_tool_feedback_bytes,
                          limits.max_recent_trials, limits.max_recent_tool_groups), (65536, 6144, 6, 2))
        PolicyLimits(max_recent_trials=0, max_recent_tool_groups=0)

    def test_invalid_limits_are_rejected_including_bool_and_floats(self):
        for name in ("max_messages_bytes", "max_tool_feedback_bytes"):
            for bad in (0, -1, True, 1.5, "5", None):
                with self.subTest(name=name, bad=bad), self.assertRaises(PolicyInputError):
                    PolicyLimits(**{name: bad})
        for name in ("max_recent_trials", "max_recent_tool_groups"):
            for bad in (-1, True, False, 2.0, None):
                with self.subTest(name=name, bad=bad), self.assertRaises(PolicyInputError):
                    PolicyLimits(**{name: bad})

    def test_size_is_the_documented_compact_utf8_json(self):
        value = {"a": "Привет \u2603", "b": [1, 2.5, None, True], "c": 'q"\\'}
        expected = len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8"))
        self.assertEqual(json_size(value), expected)

    def test_module_is_pure_standard_library(self):
        tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, {"__future__", "collections", "dataclasses", "json", "math"})
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertFalse(calls & {"open", "exec", "eval", "compile", "__import__"})


class CompactFeedbackTests(unittest.TestCase):
    def feedback(self, result, budget=6144):
        return compact_tool_feedback(result, limits=PolicyLimits(max_tool_feedback_bytes=budget),
                                     redact=redact)

    def test_host_facts_are_copied_unchanged(self):
        result = ok_result(selection={"accepted": False, "verified": True, "metric": metric(11.25),
                                      "reason": "not_better", "hypothesis": "h"}, timed_out=True,
                           exit_code=0)
        feedback = self.feedback(result)
        self.assertEqual(feedback["selection"]["metric"], metric(11.25))
        self.assertIs(feedback["selection"]["accepted"], False)
        self.assertIs(feedback["selection"]["verified"], True)
        self.assertEqual((feedback["selection"]["reason"], feedback["exit_code"], feedback["timed_out"]),
                         ("not_better", 0, True))
        self.assertEqual(feedback["feedback_version"], 1)

    def test_printed_score_cannot_change_the_host_metric(self):
        result = ok_result(stdout="score=1.0 (verified, best ever)\n", selection={
            "accepted": False, "verified": True, "metric": metric(12.0), "reason": "not_better"})
        feedback = self.feedback(result)
        self.assertEqual(feedback["selection"]["metric"]["value"], 12.0)
        self.assertFalse(feedback["selection"]["accepted"])
        self.assertIn("score=1.0", feedback["stdout"])  # text stays text, never a fact

    def test_unverified_selection_is_not_promoted(self):
        feedback = self.feedback(ok_result(selection={"accepted": True, "verified": False, "metric": None,
                                                      "reason": "host_metric_unsupported"}))
        self.assertIs(feedback["selection"]["verified"], False)
        self.assertIsNone(feedback["selection"]["metric"])

    def test_every_budget_is_respected_for_hostile_input(self):
        hostile = ok_result(
            stdout="Ünï©ödé \u2603 😀 \"quote\" \\ back\\slash\n" * 4000,
            stderr="Traceback\n" + "  File \"x.py\", line 1\n" * 5000 + "ValueError: final-line",
            selection={"accepted": False, "verified": True, "metric": metric(1.5), "reason": "r" * 5000,
                       "hypothesis": "h" * 9000,
                       "diagnostics": {"groups": [{"feature": f"f{i}", "group": "g" * 400, "rows": i,
                                                   "metric": "RMSE", "value": 1.0} for i in range(400)]}},
            error="E" * 9000, output_check={"valid": False, "errors": ["bad column " * 80] * 30})
        for budget in list(range(1, 400, 17)) + [400, 777, 1024, 2048, 6144, 20000]:
            with self.subTest(budget=budget):
                try:
                    feedback = self.feedback(hostile, budget)
                except PolicyBudgetExceeded:
                    self.assertLess(budget, 400)
                    continue
                self.assertLessEqual(json_size(feedback), budget)
                json.dumps(feedback, ensure_ascii=False).encode("utf-8")
                if budget >= 400:
                    self.assertEqual(feedback["selection"]["metric"], metric(1.5))
                    self.assertEqual(feedback["exit_code"], 0)

    def test_minimal_feedback_that_does_not_fit_raises_budget_error(self):
        with self.assertRaises(PolicyBudgetExceeded):
            self.feedback(ok_result(), budget=40)

    def test_truncation_is_explicit_and_keeps_the_final_traceback_line(self):
        stderr = "Traceback (most recent call last):\n" + "  File \"a.py\", line 3, in f\n" * 3000 \
                 + "KeyError: 'last-line-marker'"
        feedback = self.feedback(ok_result(exit_code=1, stderr=stderr, stdout="o" * 40000), budget=3000)
        self.assertIn("KeyError: 'last-line-marker'", feedback["stderr"])
        self.assertIn("cut", feedback["stderr"])
        self.assertEqual(feedback["truncated"]["stderr"], len(stderr.encode("utf-8")))
        self.assertEqual(feedback["truncated"]["stdout"], 40000)
        self.assertLessEqual(json_size(feedback), 3000)

    def test_stderr_has_priority_over_a_huge_stdout_after_a_failure(self):
        feedback = self.feedback(ok_result(exit_code=1, stdout="x" * 60000,
                                           stderr="boom " * 400 + "RuntimeError: tail-marker"), budget=2500)
        self.assertIn("RuntimeError: tail-marker", feedback["stderr"])

    def test_timeout_and_memory_flags_and_safe_error_survive(self):
        feedback = self.feedback(ok_result(exit_code=-9, timed_out=True, memory_limit_exceeded=True,
                                           error="No time remains"), budget=900)
        self.assertTrue(feedback["timed_out"] and feedback["memory_limit_exceeded"])
        self.assertEqual(feedback["error"], "No time remains")

    def test_redaction_happens_before_truncation_even_on_the_cut_boundary(self):
        fragments = {KEY[i:i + 6] for i in range(len(KEY) - 5)}
        for offset in range(0, 4000, 37):
            stdout = "A" * (3000 + offset) + KEY + "B" * (3000 - offset)
            for budget in (600, 1200, 2500):
                feedback = self.feedback(ok_result(stdout=stdout, stderr=KEY + "\n" * 10), budget=budget)
                text = json.dumps(feedback, ensure_ascii=False)
                self.assertNotIn(KEY, text)
                self.assertFalse([item for item in fragments if item in text], (offset, budget))

    def test_secrets_are_removed_from_every_text_field(self):
        result = ok_result(stdout=KEY, stderr=KEY, error=KEY, selection={
            "accepted": False, "verified": False, "metric": None, "reason": KEY, "hypothesis": KEY,
            "diagnostics": {"note": KEY}}, output_check={"valid": False, "errors": [KEY]})
        text = json.dumps(self.feedback(result), ensure_ascii=False)
        self.assertNotIn(KEY, text)
        self.assertIn("[REDACTED]", text)

    def test_unicode_and_lone_surrogates_are_serializable(self):
        feedback = self.feedback(ok_result(stdout="Привет, мир 😀 \ud800 конец" * 800, stderr="\udfff"), 1500)
        text = json.dumps(feedback, ensure_ascii=False)
        text.encode("utf-8")  # raises for a lone surrogate
        self.assertLessEqual(json_size(feedback), 1500)

    def test_quotes_slashes_and_newlines_round_trip(self):
        tricky = 'say "hi"\\n C:\\path\\file\n\t{"k": "v"}\r\n' * 300
        feedback = self.feedback(ok_result(stdout=tricky), budget=1800)
        self.assertEqual(json.loads(json.dumps(feedback)), feedback)
        self.assertLessEqual(json_size(feedback), 1800)

    def test_unknown_large_fields_are_dropped_and_counted(self):
        result = ok_result(blob="Z" * 100000, other={"deep": ["Z" * 5000] * 50})
        feedback = self.feedback(result)
        self.assertNotIn("blob", feedback)
        self.assertNotIn("Z" * 50, json.dumps(feedback))
        self.assertEqual(feedback["dropped_fields"], 2)

    def test_diagnostics_are_summarised_when_too_big(self):
        labels = [f"class_{i}" for i in range(20)]
        matrix = [[(5 if i == j else (i + j) % 3) for j in range(20)] for i in range(20)]
        diagnostics = {"rows": 900, "classification": {
            "version": 1, "kind": "classification", "labels": labels, "rows": 900,
            "confusion_matrix": matrix, "macro_f1": 0.5, "accuracy": 0.6,
            "per_class": [{"precision": 0.5, "recall": 0.5, "f1": 0.01 * i, "support": 45, "predicted": 45}
                          for i in range(20)]}}
        feedback = self.feedback(ok_result(selection={"accepted": False, "verified": True,
                                                      "metric": metric(0.5, higher=True), "reason": "not_better",
                                                      "diagnostics": diagnostics}), budget=4000)
        summary = feedback["selection"]["diagnostics"]["classification"]
        self.assertEqual(summary["macro_f1"], 0.5)
        self.assertEqual(summary["worst_classes"][0]["label"], "class_0")
        self.assertIn("selection.diagnostics", feedback["truncated"])
        self.assertNotIn("confusion_matrix", summary)

    def test_small_diagnostics_are_kept_verbatim(self):
        diagnostics = {"rows": 10, "regression": {"mean_error": 0.5}}
        feedback = self.feedback(ok_result(selection={"accepted": True, "verified": True,
                                                      "metric": metric(), "diagnostics": diagnostics}))
        self.assertEqual(feedback["selection"]["diagnostics"], diagnostics)

    def test_inputs_are_not_mutated_and_output_is_deterministic(self):
        result = ok_result(stdout="x" * 20000)
        before = copy.deepcopy(result)
        first, second = self.feedback(result, 2000), self.feedback(result, 2000)
        self.assertEqual(result, before)
        self.assertEqual(first, second)

    def test_invalid_results_raise_safe_input_errors(self):
        bad_values = [
            ("exit_code", True), ("exit_code", "0"), ("stdout", 5), ("stderr", ["x"]), ("error", 3),
            ("elapsed_seconds", float("nan")), ("elapsed_seconds", -1), ("selection", "text"),
            ("timed_out", "yes"), ("output_check", {"valid": "yes"}), ("resources", [1]),
            ("selection", {"metric": {"name": "m", "value": float("inf"), "higher_is_better": True}}),
        ]
        for key, value in bad_values:
            with self.subTest(key=key):
                with self.assertRaises(PolicyInputError) as caught:
                    self.feedback(ok_result(**{key: value}))
                self.assertNotIn(KEY, str(caught.exception))
        with self.assertRaises(PolicyInputError):
            compact_tool_feedback("not a mapping", limits=PolicyLimits(), redact=redact)
        with self.assertRaises(PolicyInputError):
            compact_tool_feedback(ok_result(), limits=PolicyLimits(), redact="not callable")
        with self.assertRaises(PolicyInputError):
            compact_tool_feedback(ok_result(), limits=PolicyLimits(), redact=lambda text: 5)

    def test_redact_errors_never_echo_the_secret(self):
        def broken(text):
            raise RuntimeError(KEY)
        with self.assertRaises(PolicyInputError) as caught:
            compact_tool_feedback(ok_result(), limits=PolicyLimits(), redact=broken)
        self.assertNotIn(KEY, str(caught.exception))


class DecisionTests(unittest.TestCase):
    def assert_decision(self, decision, stage, action, reason):
        self.assertEqual((decision.stage, decision.action, decision.reason), (stage, action, reason),
                         decision.instruction)
        self.assertIn(decision.reason, REASON_CODES)
        self.assertIn(decision.stage, policy.DECISION_STAGES)
        self.assertIn(decision.action, policy.DECISION_ACTIONS)
        self.assertTrue(decision.instruction.strip())

    def test_initial_inspection_with_and_without_baseline(self):
        decision = decide(make_snapshot(stage="inspect"))
        self.assert_decision(decision, "inspect", "continue", "inspect_with_baseline")
        self.assertIn("fit_view.csv", decision.instruction)
        self.assertIn("never whole tables", decision.instruction)
        decision = decide(make_snapshot(stage="inspect", baseline={"retained_output_valid": False,
                                                                    "status": "unavailable"}))
        self.assert_decision(decision, "inspect", "continue", "inspect_without_baseline")

    def test_after_inspection_the_model_is_pushed_to_a_hypothesis(self):
        analysis = make_trial(1, accepted=False, verified=False, metric=None, reason="no evidence")
        self.assert_decision(decide(snapshot_with(analysis, stage="inspect")),
                             "experiment", "continue", "first_hypothesis")
        self.assert_decision(decide(snapshot_with(analysis, stage="research")),
                             "experiment", "continue", "first_hypothesis")
        self.assert_decision(decide(snapshot_with(None)), "experiment", "continue", "first_hypothesis")

    def test_improvement_and_not_better(self):
        self.assert_decision(decide(snapshot_with(make_trial(2), verified=1)),
                             "experiment", "continue", "improved_continue")
        not_better = make_trial(3, accepted=False, reason="not_better")
        decision = decide(snapshot_with(not_better, verified=1))
        self.assert_decision(decision, "experiment", "continue", "not_better_continue")
        self.assertIn("preserved", decision.instruction)

    def test_error_kinds_choose_specific_repairs(self):
        cases = {"execution": "repair_execution_error", "timeout": "repair_resource_limit",
                 "memory": "repair_resource_limit", "evidence": "repair_evidence_contract",
                 "repeated_failure": "repair_repeated_failure", "unknown": "repair_unknown_error"}
        for kind, reason in cases.items():
            with self.subTest(kind=kind):
                failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind=kind,
                                    error="boom", reason="execution_failed")
                self.assert_decision(decide(snapshot_with(failed, streak=1)), "repair", "continue", reason)

    def test_api_error_is_not_a_code_repair(self):
        failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind="api", error="x")
        self.assert_decision(decide(snapshot_with(failed)), "experiment", "continue", "retry_after_api_error")
        self.assert_decision(decide(snapshot_with(failed, stage="finalize")),
                             "finalize", "continue", "retry_after_api_error")

    def test_evidence_instruction_comes_from_the_contract_not_from_constants(self):
        failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind="evidence", error="x")
        contract = make_contract(required_outputs=[{"path": "alpha.csv", "columns": ["k", "p"]}],
                                 manifest_path="v3_manifest.json", manifest_version=3)
        text = decide(snapshot_with(failed), contract).instruction
        self.assertIn("alpha.csv", text)
        self.assertIn("v3_manifest.json", text)
        for forbidden in ("train.csv", "actual", "submission.csv", "holdout_predictions.csv"):
            self.assertFalse(has_word(forbidden, text), forbidden)

    def test_resource_repair_mentions_the_remaining_time(self):
        failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind="timeout", error="t")
        text = decide(snapshot_with(failed, budget={"remaining_seconds": 1234.9})).instruction
        self.assertIn("1234s", text)

    def test_repeated_failures_change_the_instruction(self):
        first = make_trial(1, accepted=False, verified=False, metric=None, error_kind="execution", error="e")
        second = make_trial(2, accepted=False, verified=False, metric=None, error_kind="execution", error="e")
        one = decide(snapshot_with(first, streak=1))
        two = decide(snapshot_with(second, streak=2))
        self.assert_decision(two, "repair", "continue", "repair_consecutive_failures")
        self.assertNotEqual(one.instruction, two.instruction)
        self.assertIn("simplest pipeline", two.instruction)
        limit = decide(snapshot_with(second, streak=FAILURE_STOP_STREAK))
        self.assert_decision(limit, "finished", "fallback", "repeated_failures_keep_result")
        none = decide(snapshot_with(second, streak=FAILURE_STOP_STREAK,
                                    baseline={"retained_output_valid": False}))
        self.assert_decision(none, "finished", "stop", "repeated_failures_no_result")

    def test_unchanged_code_or_hypothesis_is_not_a_new_idea(self):
        earlier = make_trial(1, accepted=False, reason="not_better", code_hash="same", hypothesis="Try ridge")
        same_code = make_trial(2, accepted=False, reason="not_better", code_hash="same", hypothesis="other")
        snapshot = make_snapshot(last_trial=same_code, recent_trials=[earlier, same_code],
                                 counts={"verified_trials": 2})
        self.assert_decision(decide(snapshot), "experiment", "continue", "repeated_code")
        same_idea = make_trial(2, accepted=False, reason="not_better", code_hash="new", hypothesis="  try   RIDGE ")
        snapshot = make_snapshot(last_trial=same_idea, recent_trials=[earlier, same_idea])
        self.assert_decision(decide(snapshot), "experiment", "continue", "repeated_code")

    def test_low_resources_ask_to_finalize(self):
        for budget in ({"remaining_seconds": 120.0}, {"remaining_requests": 1}):
            with self.subTest(budget=budget):
                decision = decide(snapshot_with(make_trial(2), verified=1, budget=budget))
                self.assert_decision(decision, "finalize", "continue", "finalize_low_resources")

    def test_host_finalize_stage(self):
        decision = decide(snapshot_with(make_trial(2), verified=1, stage="finalize"))
        self.assert_decision(decision, "finalize", "continue", "finalize_requested")
        self.assertIn("host instructions allow", decision.instruction)
        decision = decide(snapshot_with(make_trial(3), verified=2, stage="finalize"))
        self.assert_decision(decision, "finalize", "continue", "ready_to_finish")

    def test_completion_requires_the_host_contract_not_words(self):
        done = decide(snapshot_with(make_trial(3), verified=2, finish_requested=True))
        self.assert_decision(done, "finished", "complete", "contract_fulfilled")
        early = decide(snapshot_with(make_trial(2), verified=1, finish_requested=True))
        self.assert_decision(early, "experiment", "continue", "finish_too_early")
        self.assertIn("1 more host-verified experiment", early.instruction)
        baseline_only = decide(make_snapshot(finish_requested=True))
        self.assert_decision(baseline_only, "experiment", "continue", "finish_too_early")
        self.assertIn("2 more", baseline_only.instruction)
        not_requested = decide(snapshot_with(make_trial(3), verified=2))
        self.assertNotEqual(not_requested.action, "complete")

    def test_not_better_trials_count_towards_completion(self):
        not_better = make_trial(3, accepted=False, reason="not_better")
        decision = decide(snapshot_with(not_better, verified=2, finish_requested=True))
        self.assert_decision(decision, "finished", "complete", "contract_fulfilled")

    def test_finish_without_a_retained_output_is_never_complete(self):
        snapshot = snapshot_with(make_trial(3), verified=2, finish_requested=True,
                                 baseline={"retained_output_valid": False})
        self.assert_decision(decide(snapshot), "experiment", "continue", "finish_too_early")
        self.assertIn("no valid output", decide(snapshot).instruction)
        snapshot["budget"]["can_request"] = False
        self.assert_decision(decide(snapshot), "finished", "stop", "resources_exhausted_no_result")

    def test_unsupported_metric_completion_is_labelled_unverified(self):
        contract = make_contract(validation={"supported": False, "metric": None, "scope": "none"},
                                 completion={"required_verified_trials": 2, "allow_unverified": True})
        trial = make_trial(1, verified=False, metric=None, reason="host_metric_unsupported")
        done = decide(snapshot_with(trial, unverified=1, finish_requested=True), contract)
        self.assert_decision(done, "finished", "complete", "unverified_completion")
        self.assertIn("NOT independently verified", done.instruction)
        early = decide(make_snapshot(finish_requested=True), contract)
        self.assert_decision(early, "experiment", "continue", "finish_too_early")
        self.assertIn("at least one experiment", early.instruction)
        forbidden = make_contract(validation={"supported": False, "metric": None, "scope": "none"})
        stopped = decide(snapshot_with(trial, unverified=1, finish_requested=True), forbidden)
        self.assert_decision(stopped, "finished", "fallback", "contract_unreachable")

    def test_resource_exhaustion_prefers_fallback_and_never_hides_a_met_contract(self):
        for budget in ({"remaining_seconds": 0.0}, {"remaining_requests": 0}, {"can_request": False}):
            with self.subTest(budget=budget):
                snapshot = snapshot_with(make_trial(2), verified=1, budget=budget)
                self.assert_decision(decide(snapshot), "finished", "fallback", "resources_exhausted_keep_result")
                snapshot["baseline"]["retained_output_valid"] = False
                self.assert_decision(decide(snapshot), "finished", "stop", "resources_exhausted_no_result")
                met = snapshot_with(make_trial(3), verified=2, finish_requested=True, budget=budget)
                self.assert_decision(decide(met), "finished", "complete", "contract_fulfilled")

    def test_v3_style_contract_does_not_hardcode_names(self):
        contract = make_contract(manifest_version=3, allowed_inputs=[{"path": "fit_only.parquet", "purpose": "fit"}],
                                 required_outputs=[{"path": "inner.csv", "columns": ["row", "score"]}])
        text = " ".join(decide(make_snapshot(stage=stage), contract).instruction
                        for stage in ("inspect", "research", "finalize"))
        for forbidden in ("train.csv", "actual", "holdout", "submission.csv"):
            self.assertFalse(has_word(forbidden, text), forbidden)

    def test_unknown_keys_are_ignored_and_not_reflected(self):
        snapshot = make_snapshot(secret_extra="LEAK-SNAPSHOT")
        contract = make_contract(secret_extra="LEAK-CONTRACT")
        decision = decide(snapshot, contract)
        self.assertNotIn("LEAK", decision.instruction)
        context = build(snapshot=snapshot, contract=contract)
        self.assertNotIn("LEAK", json.dumps(context.messages))

    def test_invalid_snapshot_and_contract_are_rejected_safely(self):
        bad_snapshots = [
            {**make_snapshot(), "version": 2}, {**make_snapshot(), "version": True},
            {**make_snapshot(), "stage": "other"}, {**make_snapshot(), "finish_requested": "no"},
            make_snapshot(budget={"remaining_seconds": float("nan")}),
            make_snapshot(budget={"remaining_seconds": float("inf")}),
            make_snapshot(budget={"remaining_seconds": -1}),
            make_snapshot(budget={"remaining_requests": True}),
            make_snapshot(budget={"remaining_requests": -1}),
            make_snapshot(budget={"can_request": 1}),
            make_snapshot(counts={"verified_trials": -1}),
            make_snapshot(counts={"unverified_trials": 1.5}),
            make_snapshot(baseline={"best_verified_metric": {"name": "m", "value": float("nan"),
                                                              "higher_is_better": True}}),
            make_snapshot(baseline={"best_verified_metric": {"name": "m", "value": True,
                                                              "higher_is_better": True}}),
            make_snapshot(recent_trials="no"), make_snapshot(last_trial={"number": 0}),
            make_snapshot(recent_trials=[make_trial(1, error_kind="mystery")]),
            make_snapshot(recent_trials=[make_trial(1, diagnostics=[1])]),
            make_snapshot(recent_trials=[make_trial(1, resources={"x": float("nan")})]),
            "not an object",
        ]
        for snapshot in bad_snapshots:
            with self.subTest(snapshot=str(snapshot)[:60]):
                with self.assertRaises(PolicyInputError):
                    decide(snapshot)
        missing = make_snapshot()
        del missing["counts"]
        with self.assertRaises(PolicyInputError):
            decide(missing)
        bad_contracts = [
            {**make_contract(), "version": 2}, make_contract(manifest_version=0), make_contract(manifest_path=" "),
            make_contract(host_instructions=5), make_contract(allowed_inputs=[{"path": "", "purpose": ""}]),
            make_contract(required_outputs=[{"path": "a", "columns": "id"}]),
            make_contract(validation={"supported": "yes", "metric": None, "scope": "s"}),
            make_contract(completion={"required_verified_trials": 0, "allow_unverified": False}),
            make_contract(completion={"required_verified_trials": 1, "allow_unverified": 1}),
        ]
        for contract in bad_contracts:
            with self.subTest(contract=str(contract)[:60]):
                with self.assertRaises(PolicyInputError):
                    decide(contract=contract)

    def test_inputs_are_not_mutated_and_decisions_are_deterministic(self):
        snapshot, contract = snapshot_with(make_trial(2), verified=1), make_contract()
        before = copy.deepcopy((snapshot, contract))
        first, second = decide(snapshot, contract), decide(snapshot, contract)
        self.assertEqual(first, second)
        self.assertEqual((snapshot, contract), before)
        self.assertIsInstance(first, PolicyDecision)


class BuildMessagesTests(unittest.TestCase):
    def test_static_prefix_history_and_exactly_one_host_summary(self):
        history = make_group(1) + make_group(2)
        context = build(history, snapshot_with(make_trial(2), verified=1))
        roles = [message["role"] for message in context.messages]
        self.assertEqual(roles, ["system", "user", "user", "assistant", "tool", "assistant", "tool", "user"])
        self.assertEqual(context.messages[0]["content"], "SYSTEM")
        self.assertEqual(context.messages[1]["content"], "TASK")
        summaries = [m for m in context.messages if m["role"] == "user" and m["content"].startswith("HOST STATE")]
        self.assertEqual(len(summaries), 1)
        self.assertIs(context.messages[-1], summaries[0])
        self.assertEqual(set(context.diagnostics), {"version", "before_bytes", "after_bytes",
                                                    "removed_tool_groups", "omitted_plain_messages",
                                                    "truncated_fields", "recent_trials_included"})
        self.assertEqual(context.diagnostics["version"], 1)

    def test_contract_is_complete_and_verbatim(self):
        contract = make_contract(host_instructions="Строка 1\n\"кавычки\" \\ слеш\n" + "x" * 5000)
        context = build(contract=contract)
        text = context.messages[2]["content"]
        self.assertIn(contract["host_instructions"], text)
        for needle in ("manifest_version: 2", "research_validation.json", "fit_view.csv",
                       "inner_predictions.csv", '"supported":true', "inner_validation",
                       '"required_verified_trials":2'):
            self.assertIn(needle, text)

    def test_after_bytes_is_exact_and_limits_hold_for_many_sizes(self):
        history = []
        for number in range(1, 7):
            history += make_group(number, code="x = 1\n" * 400, result=ok_result(
                stdout="line\n" * 6000, stderr="warn\n" * 600), calls=2 if number % 2 else 1)
        snapshot = snapshot_with(make_trial(6, error="E" * 3000, diagnostics={"rows": 1}), verified=3)
        full = None
        for limit in (3000, 4000, 6000, 9000, 16000, 30000, 64 * 1024, 200000):
            with self.subTest(limit=limit):
                try:
                    context = build(history, snapshot, limits=PolicyLimits(max_messages_bytes=limit))
                except PolicyBudgetExceeded:
                    self.assertLess(limit, 4000)
                    continue
                size = json_size(context.messages)
                self.assertEqual(context.diagnostics["after_bytes"], size)
                self.assertLessEqual(size, limit)
                self.assertLessEqual(context.diagnostics["after_bytes"], context.diagnostics["before_bytes"])
                roles = [m["role"] for m in context.messages]
                self.assertEqual(roles[:3], ["system", "user", "user"])
                self.assertEqual(roles[-1], "user")
                full = context
        self.assertIsNotNone(full)

    def test_bulky_output_yields_before_the_trial_history(self):
        history = []
        for number in range(1, 7):
            history += make_group(number, result=ok_result(stdout="line\n" * 8000))
        trials = [make_trial(n, hypothesis=f"hypothesis-{n}") for n in range(1, 7)]
        snapshot = make_snapshot(last_trial=trials[-1], recent_trials=trials, counts={"verified_trials": 6})
        for limit in (8000, 10000, 12000, 20000):
            with self.subTest(limit=limit):
                context = build(history, snapshot, limits=PolicyLimits(max_messages_bytes=limit))
                self.assertEqual(context.diagnostics["recent_trials_included"], 6)
                summary = context.messages[-1]["content"]
                for number in range(1, 7):
                    self.assertIn(f"hypothesis-{number}", summary)
                self.assertLessEqual(json_size(context.messages), limit)

    def test_only_the_newest_groups_are_kept_and_counted(self):
        history = []
        for number in range(1, 6):
            history += make_group(number, code=f"print({number})")
        context = build(history, snapshot_with(make_trial(5), verified=2),
                        limits=PolicyLimits(max_recent_tool_groups=2))
        calls = [m for m in context.messages if m["role"] == "assistant"]
        self.assertEqual([json.loads(m["tool_calls"][0]["function"]["arguments"])["code"] for m in calls],
                         ["print(4)", "print(5)"])
        self.assertEqual(context.diagnostics["removed_tool_groups"], 3)
        none = build(history, snapshot_with(make_trial(5), verified=2), limits=PolicyLimits(max_recent_tool_groups=0))
        self.assertEqual([m["role"] for m in none.messages], ["system", "user", "user", "user"])
        self.assertEqual(none.diagnostics["removed_tool_groups"], 5)

    def test_tool_messages_always_answer_a_kept_assistant_call(self):
        history = make_group(1, calls=3) + make_group(2, calls=2) + make_group(3, calls=3)
        for limit in (4000, 6000, 12000, 30000, 64 * 1024):
            for groups in (0, 1, 2, 3):
                try:
                    context = build(history, snapshot_with(make_trial(3), verified=1), limits=PolicyLimits(
                        max_messages_bytes=limit, max_recent_tool_groups=groups))
                except PolicyBudgetExceeded:
                    continue
                pending = set()
                for message in context.messages:
                    if message["role"] == "assistant" and message.get("tool_calls"):
                        self.assertFalse(pending)
                        pending = {call["id"] for call in message["tool_calls"]}
                    elif message["role"] == "tool":
                        self.assertIn(message["tool_call_id"], pending)
                        pending.discard(message["tool_call_id"])
                    elif pending:
                        self.fail("assistant calls left without all results")
                self.assertFalse(pending)

    def test_a_group_with_several_calls_is_never_split(self):
        history = make_group(1, calls=3, result=ok_result(stdout="s" * 9000))
        context = build(history, snapshot_with(make_trial(1), verified=1))
        calls = next(m for m in context.messages if m.get("tool_calls"))["tool_calls"]
        answers = [m["tool_call_id"] for m in context.messages if m["role"] == "tool"]
        self.assertEqual([c["id"] for c in calls], answers)
        self.assertEqual(len(answers), 3)

    def test_malformed_history_raises_input_errors(self):
        good = make_group(1)
        incomplete_tail = good[:1]
        incomplete_middle = make_group(1, calls=2)[:2] + make_group(2)
        unknown = [good[0], {**good[1], "tool_call_id": "nope"}]
        duplicate = [good[0], good[1], good[1]]
        orphan = [good[1]]
        wrong_role = [{"role": "user", "content": "x"}]
        system_role = [{"role": "system", "content": "x"}]
        arguments = copy.deepcopy(good)
        arguments[0]["tool_calls"][0]["function"]["arguments"] = {"code": "x"}
        no_id = copy.deepcopy(good)
        no_id[0]["tool_calls"][0]["id"] = ""
        duplicate_ids = copy.deepcopy(make_group(1, calls=2))
        duplicate_ids[0]["tool_calls"][1]["id"] = duplicate_ids[0]["tool_calls"][0]["id"]
        bad_content = copy.deepcopy(good)
        bad_content[1]["content"] = {"a": 1}
        cases = {"incomplete_tail": incomplete_tail, "incomplete_middle": incomplete_middle,
                 "unknown_id": unknown, "duplicate_result": duplicate, "orphan_tool": orphan,
                 "user_role": wrong_role, "system_role": system_role, "dict_arguments": arguments,
                 "empty_id": no_id, "duplicate_call_ids": duplicate_ids, "tool_content": bad_content,
                 "not_a_message": ["text"]}
        for name, history in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(PolicyInputError):
                    build(history)
        with self.assertRaises(PolicyInputError):
            build("not a list")

    def test_pending_calls_never_get_fake_results(self):
        with self.assertRaises(PolicyInputError) as caught:
            build(make_group(1)[:1])
        self.assertIn("unfinished", str(caught.exception))

    def test_tool_call_arguments_are_never_shortened(self):
        code = "value = 1\n" * 4000
        history = make_group(1, code=code)
        context = build(history, snapshot_with(make_trial(1), verified=1),
                        limits=PolicyLimits(max_messages_bytes=200000))
        call = next(m for m in context.messages if m.get("tool_calls"))["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], history[0]["tool_calls"][0]["function"]["arguments"])
        small = build(history, snapshot_with(make_trial(1), verified=1),
                      limits=PolicyLimits(max_messages_bytes=12000))
        for message in small.messages:  # the whole group is dropped, never half of it
            if message.get("tool_calls"):
                self.assertEqual(message["tool_calls"][0]["function"]["arguments"],
                                 history[0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(small.diagnostics["removed_tool_groups"], 1)

    def test_unchanged_arguments_keep_their_exact_text(self):
        history = make_group(1)
        history[0]["tool_calls"][0]["function"]["arguments"] = '{"code":   "print(1)" }'
        context = build(history, snapshot_with(make_trial(1), verified=1))
        call = next(m for m in context.messages if m.get("tool_calls"))["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], '{"code":   "print(1)" }')

    def test_secret_in_arguments_is_removed_and_json_stays_valid(self):
        secrets = [KEY, 'k"e\\y\nsecret', "ключ 😀 \u2028 secret"]
        for secret in secrets:
            def special(text, secret=secret):
                return text.replace(secret, "[REDACTED]").replace(KEY, "[REDACTED]")
            code = f"token = '''{secret}'''\nprint(token)"
            history = make_group(1, code=code)
            context = build(history, snapshot_with(make_trial(1), verified=1), redact=special)
            call = next(m for m in context.messages if m.get("tool_calls"))["tool_calls"][0]
            decoded = json.loads(call["function"]["arguments"])
            self.assertEqual(decoded["code"], code.replace(secret, "[REDACTED]"))
            self.assertNotIn(secret, json.dumps(context.messages, ensure_ascii=False))
            self.assertNotIn(json.dumps(secret)[1:-1], json.dumps(context.messages))

    def test_secrets_never_survive_anywhere(self):
        history = make_group(1, code=f"print('{KEY}')", content=f"I will use {KEY}",
                             result=ok_result(stdout=f"{KEY}\n" * 50, error=KEY))
        history.append({"role": "assistant", "content": f"done {KEY}"})
        snapshot = snapshot_with(make_trial(1, hypothesis=KEY, error=KEY, reason=KEY), verified=1)
        context = build(history, snapshot, contract=make_contract(host_instructions=KEY),
                        system_prompt=f"sys {KEY}", task_text=f"task {KEY}")
        self.assertNotIn(KEY, json.dumps(context.messages, ensure_ascii=False))

    def test_secret_on_the_cut_boundary_of_tool_output_is_removed_first(self):
        fragments = {KEY[i:i + 6] for i in range(len(KEY) - 5)}
        for offset in range(0, 3000, 211):
            stdout = "A" * (5000 + offset) + KEY + "B" * (5000 - offset)
            history = make_group(1, result=ok_result(stdout=stdout, exit_code=1, stderr=KEY))
            context = build(history, snapshot_with(make_trial(1), verified=1),
                            limits=PolicyLimits(max_tool_feedback_bytes=2000))
            text = json.dumps(context.messages, ensure_ascii=False)
            self.assertFalse([item for item in fragments if item in text], offset)

    def test_unicode_text_round_trips(self):
        history = make_group(1, code="print('Привет 😀')", content="Идея: проверка \ud800 текста",
                             result=ok_result(stdout="Результат 😀\n" * 3000))
        context = build(history, snapshot_with(make_trial(1, hypothesis="Гипотеза 😀"), verified=1),
                        task_text="Задача: прогноз 😀", system_prompt="Система 😀")
        text = json.dumps(context.messages, ensure_ascii=False)
        text.encode("utf-8")
        self.assertIn("Задача: прогноз 😀", text)
        self.assertEqual(context.diagnostics["after_bytes"], json_size(context.messages))

    def test_host_facts_come_from_the_snapshot_only(self):
        history = make_group(1, result=ok_result(
            stdout="score=0.001 verified=True best=0.001\n", selection={
                "accepted": False, "verified": True, "metric": metric(12.0), "reason": "not_better",
                "hypothesis": "claim"}))
        snapshot = snapshot_with(make_trial(1, accepted=False, reason="not_better", metric=metric(12.0)),
                                 verified=1, baseline={"best_verified_metric": metric(9.25)})
        context = build(history, snapshot)
        summary = context.messages[-1]["content"]
        self.assertIn('"best_verified_metric":{"name":"RMSE","value":9.25,"higher_is_better":false}', summary)
        self.assertNotIn("0.001", summary)
        self.assertIn("computed by the host", context.messages[2]["content"])
        state = json.loads(summary.split("\n")[1])
        self.assertEqual(state["recent_trials"][-1]["accepted"], False)

    def test_unverified_result_is_never_summarised_as_verified(self):
        trial = make_trial(1, verified=False, accepted=True, metric=None, reason="host_metric_unsupported")
        contract = make_contract(validation={"supported": False, "metric": None, "scope": "none"},
                                 completion={"required_verified_trials": 1, "allow_unverified": True})
        context = build(make_group(1), snapshot_with(trial, unverified=1), contract=contract)
        state = json.loads(context.messages[-1]["content"].split("\n")[1])
        self.assertIs(state["recent_trials"][-1]["verified"], False)
        self.assertEqual(state["counts"]["verified_trials"], 0)
        self.assertEqual(state["validation"], {"supported": False, "scope": "none"})

    def test_validation_scope_is_forwarded_not_guessed(self):
        context = build(contract=make_contract(validation={"supported": True, "metric": metric(),
                                                           "scope": "inner_only_v3"}))
        self.assertIn('"scope":"inner_only_v3"', context.messages[-1]["content"])

    def test_oversized_last_group_is_replaced_by_the_host_summary(self):
        giant = "print(1)\n" * 30000
        history = make_group(1, code=giant)
        trial = make_trial(1, accepted=False, verified=False, metric=None, error_kind="execution",
                           error="NameError: marker-name", reason="execution_failed")
        context = build(history, snapshot_with(trial, streak=1),
                        limits=PolicyLimits(max_messages_bytes=16000))
        self.assertEqual(context.diagnostics["removed_tool_groups"], 1)
        self.assertEqual([m["role"] for m in context.messages], ["system", "user", "user", "user"])
        summary = context.messages[-1]["content"]
        self.assertIn("NameError: marker-name", summary)
        self.assertIn('"earlier_tool_groups_omitted":1', summary)
        self.assertIn('"verified":false', summary)

    def test_traceback_is_not_sent_twice_while_the_tool_result_is_kept(self):
        stderr = "Traceback\n" + "frame\n" * 50 + "ValueError: duplicated-marker"
        history = make_group(1, result=ok_result(exit_code=1, stderr=stderr, selection={
            "accepted": False, "verified": False, "reason": "execution_failed"}))
        trial = make_trial(1, accepted=False, verified=False, metric=None, error_kind="execution",
                           error=stderr, reason="execution_failed")
        context = build(history, snapshot_with(trial, streak=1))
        text = json.dumps(context.messages)
        self.assertEqual(text.count("duplicated-marker"), 1)
        self.assertIn("error_shown_above", context.messages[-1]["content"])
        without_history = build([], snapshot_with(trial, streak=1))
        self.assertEqual(json.dumps(without_history.messages).count("duplicated-marker"), 1)

    def test_diagnostic_counters_describe_the_compaction(self):
        history = []
        for number in range(1, 5):
            history += make_group(number, result=ok_result(stdout="s" * 20000))
        history.append({"role": "assistant", "content": "all done"})
        history.insert(0, {"role": "assistant", "content": "early chatter"})
        context = build(history, snapshot_with(make_trial(4), verified=2))
        diagnostics = context.diagnostics
        self.assertGreater(diagnostics["before_bytes"], 4 * 20000)
        self.assertLess(diagnostics["after_bytes"], diagnostics["before_bytes"] // 3)
        self.assertEqual(diagnostics["removed_tool_groups"], 2)
        self.assertEqual(diagnostics["omitted_plain_messages"], 1)  # the trailing one stays
        self.assertGreaterEqual(diagnostics["truncated_fields"], 2)
        self.assertEqual(diagnostics["recent_trials_included"], 1)
        self.assertEqual(context.messages[-2]["content"], "all done")

    def test_plain_messages_are_dropped_without_groups(self):
        context = build([{"role": "assistant", "content": "chat"}], limits=PolicyLimits(max_recent_tool_groups=0))
        self.assertEqual(context.diagnostics["omitted_plain_messages"], 1)
        self.assertNotIn("chat", json.dumps(context.messages))

    def test_empty_plain_messages_are_counted_as_omitted(self):
        context = build([{"role": "assistant", "content": None}, {"role": "assistant", "content": ""}])
        self.assertEqual(context.diagnostics["omitted_plain_messages"], 2)
        self.assertEqual([m["role"] for m in context.messages], ["system", "user", "user", "user"])

    def test_recent_trials_limit_and_zero(self):
        trials = [make_trial(n, hypothesis=f"hyp-{n}") for n in range(1, 10)]
        snapshot = make_snapshot(last_trial=trials[-1], recent_trials=trials, counts={"verified_trials": 9})
        context = build(snapshot=snapshot, limits=PolicyLimits(max_recent_trials=3))
        state = json.loads(context.messages[-1]["content"].split("\n")[1])
        self.assertEqual([t["n"] for t in state["recent_trials"]], [7, 8, 9])
        self.assertEqual(context.diagnostics["recent_trials_included"], 3)
        zero = build(snapshot=snapshot, limits=PolicyLimits(max_recent_trials=0))
        self.assertEqual(json.loads(zero.messages[-1]["content"].split("\n")[1])["recent_trials"], [])
        self.assertEqual(zero.diagnostics["recent_trials_included"], 0)

    def test_last_trial_missing_from_recent_list_is_still_described(self):
        snapshot = make_snapshot(last_trial=make_trial(5, hypothesis="the newest"),
                                 recent_trials=[make_trial(4)], counts={"verified_trials": 2})
        state = json.loads(build(snapshot=snapshot).messages[-1]["content"].split("\n")[1])
        self.assertEqual([t["n"] for t in state["recent_trials"]], [4, 5])

    def test_impossible_mandatory_parts_raise_budget_error(self):
        with self.assertRaises(PolicyBudgetExceeded):
            build(limits=PolicyLimits(max_messages_bytes=300))
        huge_contract = make_contract(host_instructions="Z" * 5000)
        with self.assertRaises(PolicyBudgetExceeded):
            build(contract=huge_contract, limits=PolicyLimits(max_messages_bytes=4000))
        # The contract is never shortened to fit.
        context = build(contract=huge_contract, limits=PolicyLimits(max_messages_bytes=20000))
        self.assertIn("Z" * 5000, context.messages[2]["content"])

    def test_inputs_are_not_mutated_and_messages_are_deterministic(self):
        history = make_group(1, calls=2) + make_group(2, result=ok_result(stdout="y" * 30000))
        snapshot, contract = snapshot_with(make_trial(2, diagnostics={"rows": 3}), verified=1), make_contract()
        before = copy.deepcopy((history, snapshot, contract))
        first = build(history, snapshot, contract)
        second = build(history, snapshot, contract)
        self.assertEqual((history, snapshot, contract), before)
        self.assertEqual(first, second)
        self.assertIsNot(first.messages, second.messages)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(PolicyInputError):
            build(system_prompt=5)
        with self.assertRaises(PolicyInputError):
            build(task_text=None)
        with self.assertRaises(PolicyInputError):
            build(limits=object())
        with self.assertRaises(PolicyInputError):
            build(redact=None)
        with self.assertRaises(PolicyInputError):
            build(snapshot={**make_snapshot(), "version": 2})
        with self.assertRaises(PolicyInputError):
            build(contract={**make_contract(), "version": 2})

    def test_manifest_without_actual_column_works(self):
        contract = make_contract(manifest_version=3, required_outputs=[
            {"path": "inner_predictions.csv", "columns": ["row", "prediction"]}],
            allowed_inputs=[{"path": "fit_only.csv", "purpose": "fit rows"}],
            validation={"supported": True, "metric": metric(), "scope": "inner_only"})
        context = build(snapshot=make_snapshot(stage="inspect"), contract=contract)
        policy_text = context.messages[2]["content"] + context.messages[-1]["content"]
        self.assertFalse(has_word("actual", policy_text))
        self.assertFalse(has_word("train.csv", policy_text))
        self.assertIn("fit_only.csv", policy_text)

    def test_unparseable_tool_output_falls_back_to_bounded_text(self):
        history = make_group(1)
        history[1]["content"] = "plain text output " * 2000
        context = build(history, snapshot_with(make_trial(1), verified=1))
        feedback = json.loads(next(m for m in context.messages if m["role"] == "tool")["content"])
        self.assertIn("unstructured", feedback)
        self.assertLessEqual(json_size(feedback), PolicyLimits().max_tool_feedback_bytes)
        history[1]["content"] = json.dumps({"exit_code": "weird", "stdout": 5})
        context = build(history, snapshot_with(make_trial(1), verified=1))
        self.assertTrue(any(m["role"] == "tool" for m in context.messages))



def dlg_contract(**changes):
    value = make_contract(
        manifest_version=3, allowed_inputs=["task.md", "data/train.csv"],
        required_outputs=["research_candidate.py", "research_recipe.json", "research_hypothesis.json"],
        validation={"supported": True, "metric": metric(4.2), "scope": "inner_selection"},
        completion={"required_verified_trials": 1, "allow_unverified": False, "mode": "research_dialogue"})
    value.update(changes)
    return value


def llm_trial(number=1, **changes):
    changes.setdefault("recipe_hash", f"recipe{number}")
    return make_trial(number, origin="llm", **changes)


def dlg_snapshot(rows=(), *, verified_llm=None, unverified_llm=0, host_can_finalize=True, closed=False,
                 finish=False, **changes):
    rows = list(rows)
    llm = verified_llm if verified_llm is not None else sum(
        1 for row in rows if row["verified"] and row.get("origin") == "llm")
    merged = {
        "baseline": {"retained_output_valid": False, "status": "measured"},
        "counts": {"verified_trials": sum(1 for row in rows if row["verified"]), "unverified_trials": unverified_llm,
                   "consecutive_execution_failures": 0, "verified_llm_trials": llm,
                   "unverified_llm_trials": unverified_llm, "host_local_trials": 0, "host_history_trials": 0},
        "finalization": {"verified_candidate": llm > 0, "host_can_finalize": host_can_finalize,
                         "research_closed": closed}}
    for key in merged:  # nested overrides merge into the defaults
        if key in changes:
            merged[key] = {**merged[key], **changes.pop(key)}
    return make_snapshot(last_trial=rows[-1] if rows else None, recent_trials=rows, finish_requested=finish,
                         **merged, **changes)


def dlg_decide(snapshot=None, contract=None):
    return decide_next_action(snapshot or dlg_snapshot(), contract=contract or dlg_contract())


class ResearchDialogueCompletionTests(unittest.TestCase):
    """A research dialogue closes on host facts; a final CSV is a separate fact."""

    def test_frozen_inner_receipt_completes_without_a_final_csv(self):
        rows = [llm_trial(1)]
        snapshot = dlg_snapshot(rows, finish=True)
        self.assertFalse(snapshot["baseline"]["retained_output_valid"])
        decision = dlg_decide(snapshot)
        self.assertEqual((decision.stage, decision.action, decision.reason), ("finished", "complete", "dialogue_complete"))
        self.assertIn("does not mean that a final submission already exists", decision.instruction)
        closed = dlg_decide(dlg_snapshot(rows, closed=True))
        self.assertEqual((closed.action, closed.reason), ("complete", "dialogue_closed_by_host"))

    def test_the_legacy_contract_still_needs_a_saved_csv(self):
        legacy = snapshot_with(make_trial(2), verified=2, finish_requested=True,
                               baseline={"retained_output_valid": False})
        decision = decide(legacy)
        self.assertEqual(decision.reason, "finish_too_early")
        self.assertIn("no valid output", decision.instruction)
        self.assertNotIn("finalization", json.dumps(build(snapshot=legacy).messages))

    def test_no_trial_and_analysis_never_complete(self):
        analysis = llm_trial(1, accepted=False, verified=False, metric=None, reason="no evidence")
        for rows in ([], [analysis]):
            with self.subTest(rows=len(rows)):
                decision = dlg_decide(dlg_snapshot(rows, finish=True))
                self.assertEqual((decision.action, decision.reason), ("continue", "finish_too_early"))
                self.assertIn("1 more host-verified LLM experiment", decision.instruction)
        closed = dlg_decide(dlg_snapshot([analysis], closed=True))
        self.assertEqual((closed.action, closed.reason), ("fallback", "dialogue_closed_by_host"))

    def test_other_origins_and_baseline_never_raise_the_llm_counter(self):
        history = make_trial(1, origin="history", recipe_hash="h", hypothesis={"hypothesis": "blend"})
        local = make_trial(2, origin="local", recipe_hash="l", hypothesis={"hypothesis": "depth 6"})
        snapshot = dlg_snapshot([history, local], finish=True)
        self.assertEqual(snapshot["counts"]["verified_llm_trials"], 0)
        self.assertEqual(snapshot["counts"]["verified_trials"], 2)
        decision = dlg_decide(snapshot)
        self.assertEqual((decision.action, decision.reason), ("continue", "finish_too_early"))
        context = build(snapshot=snapshot, contract=dlg_contract())
        state = json.loads(context.messages[-1]["content"].split("\n")[1])
        self.assertEqual((state["counts"]["verified_llm_trials"], state["counts"]["verified_trials"]), (0, 2))
        self.assertEqual([entry["origin"] for entry in state["recent_trials"]], ["history", "local"])
        self.assertEqual(state["recent_trials"][0]["hypothesis"], "blend")

    def test_host_must_be_able_to_take_over(self):
        decision = dlg_decide(dlg_snapshot([llm_trial(1)], finish=True, host_can_finalize=False))
        self.assertEqual(decision.reason, "finish_too_early")
        self.assertIn("host cannot take over", decision.instruction)

    def test_unsupported_metric_dialogue_completion_is_unverified(self):
        contract = dlg_contract(validation={"supported": False, "metric": None, "scope": "inner_selection"},
                                completion={"required_verified_trials": 1, "allow_unverified": True,
                                            "mode": "research_dialogue"})
        trial = llm_trial(1, accepted=True, verified=False, metric=None, reason="host_metric_unsupported")
        done = dlg_decide(dlg_snapshot([trial], unverified_llm=1, finish=True), contract)
        self.assertEqual((done.action, done.reason), ("complete", "dialogue_unverified_complete"))
        self.assertIn("NOT independently verified", done.instruction)
        blocked = make_contract(validation={"supported": False, "metric": None, "scope": "s"},
                                completion={"required_verified_trials": 1, "allow_unverified": False,
                                            "mode": "research_dialogue"})
        stopped = dlg_decide(dlg_snapshot([trial], finish=True), blocked)
        self.assertEqual((stopped.action, stopped.reason), ("fallback", "dialogue_handover_unreachable"))

    def test_close_when_met_is_explicit_and_off_by_default(self):
        rows = [llm_trial(1)]
        self.assertEqual(dlg_decide(dlg_snapshot(rows)).action, "continue")
        contract = dlg_contract(completion={"required_verified_trials": 1, "allow_unverified": False,
                                            "mode": "research_dialogue", "close_when_met": True})
        decision = dlg_decide(dlg_snapshot(rows), contract)
        self.assertEqual((decision.action, decision.reason), ("complete", "dialogue_complete"))

    def test_after_a_freeze_research_is_never_resumed(self):
        trials = [llm_trial(1), llm_trial(2, accepted=False, reason="not_better"),
                  llm_trial(3, accepted=False, verified=False, metric=None, error_kind="execution", error="e")]
        for rows in ([], trials[:1], trials[:2], trials):
            for finish in (False, True):
                for stage in ("inspect", "research", "finalize"):
                    for can_request in (True, False):
                        snapshot = dlg_snapshot(rows, closed=True, finish=finish, stage=stage,
                                                budget={"can_request": can_request})
                        self.assertNotEqual(dlg_decide(snapshot).action, "continue", (len(rows), finish, stage))

    def test_hand_over_never_takes_the_place_of_host_guards(self):
        decision = dlg_decide(dlg_snapshot([llm_trial(1, accepted=False, reason="not_better")],
                                           budget={"can_request": False}))
        self.assertEqual((decision.action, decision.reason), ("fallback", "dialogue_handover_resources"))
        for text in (decision.instruction,):
            self.assertIn("its own deadline", text)
            self.assertIn("this policy changes none of them", text)
        for retained in (True, False):  # a dialogue never reports "stop" because a CSV is missing
            snapshot = dlg_snapshot([], budget={"remaining_requests": 0}, baseline={"retained_output_valid": retained})
            self.assertEqual(dlg_decide(snapshot).action, "fallback")

    def test_repeated_failures_hand_over(self):
        failing = llm_trial(1, accepted=False, verified=False, metric=None, error_kind="execution", error="e")
        snapshot = dlg_snapshot([failing], counts={"consecutive_execution_failures": FAILURE_STOP_STREAK})
        snapshot["counts"].update(verified_llm_trials=0, unverified_llm_trials=0, host_local_trials=0,
                                  host_history_trials=0)
        self.assertEqual(dlg_decide(snapshot).reason, "dialogue_handover_failures")

    def test_dialogue_contract_requires_its_facts_and_consistent_counters(self):
        snapshot = dlg_snapshot()
        del snapshot["finalization"]
        with self.assertRaises(PolicyInputError):
            dlg_decide(snapshot)
        snapshot = dlg_snapshot()
        del snapshot["counts"]["verified_llm_trials"]
        with self.assertRaises(PolicyInputError):
            dlg_decide(snapshot)
        with self.assertRaises(PolicyInputError):
            dlg_decide(dlg_snapshot(verified_llm=3))  # more LLM-verified trials than verified trials
        for bad in ({"verified_candidate": "yes", "host_can_finalize": True},
                    {"verified_candidate": True, "host_can_finalize": 1},
                    {"verified_candidate": True}):
            snapshot = dlg_snapshot()
            snapshot["finalization"] = bad
            with self.subTest(bad=bad), self.assertRaises(PolicyInputError):
                dlg_decide(snapshot)
        with self.assertRaises(PolicyInputError):
            dlg_decide(dlg_snapshot(), dlg_contract(completion={
                "required_verified_trials": 1, "allow_unverified": False, "mode": "other"}))
        with self.assertRaises(PolicyInputError):
            dlg_decide(dlg_snapshot(), dlg_contract(completion={
                "required_verified_trials": 1, "allow_unverified": False, "mode": "research_dialogue",
                "close_when_met": "yes"}))

    def test_instructions_never_delegate_host_finalization(self):
        failures = [llm_trial(2, accepted=False, verified=False, metric=None, error_kind=kind, error="e")
                    for kind in policy.ERROR_KINDS]
        trials = [None, llm_trial(1), llm_trial(1, accepted=False, reason="not_better"),
                  llm_trial(1, accepted=False, reason="refit_budget"),
                  llm_trial(1, accepted=False, verified=False, metric=None, reason="no evidence"), *failures]
        forbidden = ("refit the", "write the final", "final required outputs", "fast baseline", "save the best valid",
                     "output is saved", "submission.csv", "re-check its columns", "outer")
        seen = set()
        for last in trials:
            for stage in ("inspect", "research", "finalize"):
                for low in (False, True):
                    rows = [] if last is None else [last]
                    snapshot = dlg_snapshot(rows, stage=stage, budget={"remaining_seconds": 40.0 if low else 9000.0})
                    contract = dlg_contract(pacing={"low_remaining_seconds": 300})
                    decision = dlg_decide(snapshot, contract)
                    seen.add(decision.reason)
                    if decision.action != "continue":
                        continue
                    text = decision.instruction.lower()
                    for phrase in forbidden:
                        self.assertNotIn(phrase, text, (decision.reason, phrase))
        self.assertNotIn("finalize_low_resources", seen)
        self.assertIn("inspect_dialogue", seen)
        self.assertIn("refit_budget_continue", seen)
        self.assertIn("repair_execution_error", seen)

    def test_finalize_stage_asks_only_for_a_short_account(self):
        decision = dlg_decide(dlg_snapshot([llm_trial(1)], stage="finalize"))
        self.assertEqual((decision.action, decision.reason), ("continue", "ready_to_finish"))
        self.assertIn("no tool call", decision.instruction)
        self.assertIn("do not write or train final outputs yourself", decision.instruction)
        early = dlg_decide(dlg_snapshot([], stage="finalize"))
        self.assertEqual(early.reason, "finalize_requested")

    def test_contract_text_states_the_new_semantics(self):
        context = build(contract=dlg_contract(pacing={"low_remaining_seconds": 60, "low_remaining_requests": 0}),
                        snapshot=dlg_snapshot())
        text = context.messages[2]["content"]
        self.assertIn('"mode":"research_dialogue"', text)
        self.assertIn("completion_mode: research_dialogue", text)
        self.assertIn("retained_output_valid reports only a CSV that is already saved", text)
        self.assertIn('pacing: {"low_remaining_seconds":60,"low_remaining_requests":0}', text)
        state = json.loads(context.messages[-1]["content"].split("\n")[1])
        self.assertEqual(state["finalization"], {"verified_candidate": False, "host_can_finalize": True,
                                                 "research_closed": False})
        self.assertFalse(state["baseline"]["retained_output_valid"])


class ResearchBudgetPacingTests(unittest.TestCase):
    def test_a_short_smoke_budget_is_not_forced_to_finish_before_the_first_trial(self):
        for seconds, research in ((250.0, 120.0), (90.0, 90.0), (300.0, 200.0)):
            snapshot = dlg_snapshot([], budget={"remaining_seconds": seconds, "research_seconds_available": research})
            decision = dlg_decide(snapshot)
            self.assertEqual((decision.action, decision.reason), ("continue", "first_hypothesis"), seconds)
            self.assertNotIn("tight", decision.instruction)

    def test_explicit_thresholds_only_add_a_note_for_the_dialogue_contract(self):
        contract = dlg_contract(pacing={"low_remaining_seconds": 300, "low_remaining_requests": 1})
        snapshot = dlg_snapshot([], budget={"remaining_seconds": 250.0})
        decision = dlg_decide(snapshot, contract)
        self.assertEqual(decision.reason, "first_hypothesis")
        self.assertIn("time budget is tight", decision.instruction)
        requests = dlg_decide(dlg_snapshot([], budget={"remaining_requests": 1}), contract)
        self.assertEqual(requests.reason, "first_hypothesis")
        self.assertIn("tight", requests.instruction)

    def test_the_host_research_time_replaces_the_total_remaining_time(self):
        snapshot = dlg_snapshot([llm_trial(1)], budget={"remaining_seconds": 9000.0, "research_seconds_available": 0.0})
        self.assertEqual(dlg_decide(snapshot).reason, "dialogue_handover_resources")
        plenty = dlg_snapshot([llm_trial(1)], budget={"remaining_seconds": 5.0, "research_seconds_available": 900.0})
        self.assertEqual(dlg_decide(plenty).action, "continue")
        with self.assertRaises(PolicyInputError):
            dlg_decide(dlg_snapshot(budget={"research_seconds_available": -1.0}))
        with self.assertRaises(PolicyInputError):
            dlg_decide(dlg_snapshot(budget={"research_seconds_available": float("nan")}))

    def test_legacy_threshold_is_kept_by_default_and_can_be_made_explicit(self):
        short = snapshot_with(make_trial(1), verified=1, budget={"remaining_seconds": 120.0})
        self.assertEqual(decide(short).reason, "finalize_low_resources")
        explicit = make_contract(pacing={"low_remaining_seconds": 60, "low_remaining_requests": 0})
        self.assertNotEqual(decide(short, explicit).reason, "finalize_low_resources")
        shorter = snapshot_with(make_trial(1), verified=1, budget={"remaining_seconds": 30.0})
        self.assertEqual(decide(shorter, explicit).reason, "finalize_low_resources")
        for bad in ({"low_remaining_seconds": -1}, {"low_remaining_seconds": True},
                    {"low_remaining_requests": -1}, {"low_remaining_requests": 1.5}, "no"):
            with self.subTest(bad=bad), self.assertRaises(PolicyInputError):
                decide(short, make_contract(pacing=bad))

    def test_policy_does_not_recompute_a_reserve(self):
        snapshot = dlg_snapshot([llm_trial(1)], budget={"remaining_seconds": 700.0, "research_seconds_available": 100.0})
        again = dlg_snapshot([llm_trial(1)], budget={"remaining_seconds": 7000.0, "research_seconds_available": 100.0})
        self.assertEqual(dlg_decide(snapshot), dlg_decide(again))


class ResearchRepeatIdentityTests(unittest.TestCase):
    def snapshot_for(self, first, second, *, dialogue=True):
        builder = dlg_snapshot if dialogue else (lambda rows, **kw: make_snapshot(
            last_trial=rows[-1], recent_trials=rows, counts={"verified_trials": 2}))
        return builder([first, second])

    def test_same_code_with_a_different_recipe_is_a_new_attempt(self):
        first = llm_trial(1, code_hash="same", hypothesis="Boost trees, depth 3", accepted=False, reason="not_better")
        second = llm_trial(2, code_hash="same", hypothesis="Boost trees, depth 6", accepted=False,
                           reason="not_better", recipe_hash="recipe-depth-6")
        decision = dlg_decide(self.snapshot_for(first, second))
        self.assertNotEqual(decision.reason, "repeated_code")
        similar = llm_trial(2, code_hash="same", hypothesis="  boost TREES, depth 3 ", recipe_hash="other",
                            accepted=False, reason="not_better")
        self.assertNotEqual(dlg_decide(self.snapshot_for(first, similar)).reason, "repeated_code")

    def test_identical_code_and_recipe_is_a_repeat(self):
        first = llm_trial(1, code_hash="same", recipe_hash="r1", accepted=False, reason="not_better")
        second = llm_trial(2, code_hash="same", recipe_hash="r1", hypothesis="completely different words",
                           accepted=False, reason="not_better")
        self.assertEqual(dlg_decide(self.snapshot_for(first, second)).reason, "repeated_code")

    def test_a_missing_recipe_hash_in_a_dialogue_contract_claims_no_repeat(self):
        first = make_trial(1, code_hash="same", hypothesis="Try ridge", accepted=False, reason="not_better")
        second = make_trial(2, code_hash="same", hypothesis="Try ridge", accepted=False, reason="not_better")
        for trial in (first, second):
            trial["origin"] = "llm"
        snapshot = dlg_snapshot([first, second], verified_llm=2)
        self.assertNotEqual(dlg_decide(snapshot).reason, "repeated_code")

    def test_legacy_contract_without_recipe_hash_is_explicit(self):
        first = make_trial(1, code_hash="same", accepted=False, reason="not_better")
        second = make_trial(2, code_hash="same", accepted=False, reason="not_better", hypothesis="new words")
        self.assertEqual(decide(self.snapshot_for(first, second, dialogue=False)).reason, "repeated_code")
        by_text = make_trial(2, code_hash="other", accepted=False, reason="not_better", hypothesis="Hypothesis 1")
        self.assertEqual(decide(self.snapshot_for(first, by_text, dialogue=False)).reason, "repeated_code")
        # Recipe hashes present on both sides win even under a legacy contract.
        with_recipes_a = make_trial(1, code_hash="same", recipe_hash="a", accepted=False, reason="not_better")
        with_recipes_b = make_trial(2, code_hash="same", recipe_hash="b", accepted=False, reason="not_better")
        self.assertNotEqual(decide(self.snapshot_for(with_recipes_a, with_recipes_b, dialogue=False)).reason,
                            "repeated_code")

    def test_recipe_hash_type_is_checked_but_never_invented(self):
        with self.assertRaises(PolicyInputError):
            decide(make_snapshot(last_trial=make_trial(1, recipe_hash=5)))
        trial = decide_next_action(make_snapshot(last_trial=make_trial(1)), contract=make_contract())
        self.assertIsInstance(trial, PolicyDecision)


class HypothesisShapeTests(unittest.TestCase):
    def feedback(self, selection, budget=6144):
        return compact_tool_feedback(ok_result(selection=selection), limits=PolicyLimits(
            max_tool_feedback_bytes=budget), redact=redact)

    def v3_selection(self, **changes):
        value = lab.v3_selection(True, 3.9, "improved", hypothesis="Use shallower trees", recipe_hash="r1",
                                 diagnostics=lab.v3_inner_diagnostics(__import__("random").Random(1), 3.9))
        value.update(changes)
        return value

    def test_a_string_llm_hypothesis_and_an_object_host_hypothesis_agree(self):
        text = self.feedback(self.v3_selection())
        host = self.feedback(self.v3_selection(origin="history", hypothesis={"hypothesis": "Use shallower trees"}))
        self.assertEqual(text["selection"]["hypothesis"], host["selection"]["hypothesis"])
        for feedback in (text, host):
            self.assertEqual((feedback["selection"]["accepted"], feedback["selection"]["verified"]), (True, True))
            self.assertEqual(feedback["selection"]["metric"], {"name": "RMSE", "value": 3.9, "higher_is_better": False})
        self.assertEqual((text["selection"]["origin"], host["selection"]["origin"]), ("llm", "history"))
        self.assertEqual(host["selection"]["scope"], "inner_selection")
        self.assertEqual(host["selection"]["recipe_family"], "gradient_boosting")

    def test_only_the_hypothesis_text_of_an_object_is_forwarded(self):
        selection = self.v3_selection(origin="local", hypothesis={"hypothesis": "Host local depth 6",
                                                                  "nested": {"x": "LEAK-NESTED"}, "extra": "LEAK-EXTRA"})
        feedback = self.feedback(selection)
        self.assertEqual(feedback["selection"]["hypothesis"], "Host local depth 6")
        self.assertNotIn("LEAK", json.dumps(feedback))

    def test_unusable_hypothesis_shapes_never_hide_typed_facts(self):
        for bad in (5, [1, 2], {"nothing": 1}, {"hypothesis": 7}, True):
            with self.subTest(bad=repr(bad)):
                feedback = self.feedback(self.v3_selection(hypothesis=bad))
                selection = feedback["selection"]
                self.assertEqual((selection["accepted"], selection["verified"], selection["metric"]["value"]),
                                 (True, True, 3.9))
                self.assertNotIn("hypothesis", selection)
                self.assertGreaterEqual(feedback["dropped_fields"], 1)

    def test_history_tool_content_with_an_object_hypothesis_keeps_structure(self):
        selection = self.v3_selection(origin="history", hypothesis={"hypothesis": "Blend with past history"})
        history = make_group(1, result=ok_result(selection=selection))
        context = build(history, snapshot_with(make_trial(1), verified=1))
        feedback = json.loads(next(m for m in context.messages if m["role"] == "tool")["content"])
        self.assertNotIn("unstructured", feedback)
        self.assertEqual(feedback["selection"]["hypothesis"], "Blend with past history")
        self.assertEqual(feedback["selection"]["metric"]["value"], 3.9)
        weird = ok_result(selection=self.v3_selection(hypothesis=[1, 2]))
        context = build(make_group(1, result=weird), snapshot_with(make_trial(1), verified=1))
        self.assertNotIn("unstructured", next(m for m in context.messages if m["role"] == "tool")["content"])

    def test_snapshot_trials_accept_the_real_host_forms_and_reject_the_rest(self):
        for hypothesis in ("text", {"hypothesis": "text"}, None):
            with self.subTest(hypothesis=hypothesis):
                decide(make_snapshot(last_trial=make_trial(1, hypothesis=hypothesis)))
        for bad in (5, ["text"], {"hypothesis": 3}, {"text": "t"}, True):
            with self.subTest(bad=repr(bad)), self.assertRaises(PolicyInputError):
                decide(make_snapshot(last_trial=make_trial(1, hypothesis=bad)))
        with self.assertRaises(PolicyInputError):
            decide(make_snapshot(last_trial=make_trial(1, origin="outer")))

    def test_object_hypothesis_text_is_redacted_before_it_is_shortened(self):
        long_secret = {"hypothesis": "x" * 200 + KEY + "y" * 200}
        context = build(snapshot=snapshot_with(make_trial(1, hypothesis=long_secret), verified=1))
        self.assertNotIn(KEY, json.dumps(context.messages))
        self.assertIn("[REDACTED]", context.messages[-1]["content"])

    def test_real_v3_result_shapes_keep_their_typed_facts(self):
        rng = __import__("random").Random(3)
        cases = {
            "improved": lab.v3_result(rng, lab.v3_selection(True, 3.9, "improved", hypothesis="h", recipe_hash="r")),
            "not_better": lab.v3_result(rng, lab.v3_selection(False, 4.4, "not_better", hypothesis="h", recipe_hash="r")),
            "refit_budget": lab.v3_result(rng, lab.v3_selection(False, 3.5, "refit_budget", hypothesis="h", recipe_hash="r")),
            "no_diagnostics": lab.v3_result(rng, lab.v3_selection(True, 3.9, "improved", hypothesis="h", recipe_hash="r",
                                                                  diagnostics=None)),
            "diagnostics_unavailable": lab.v3_result(rng, lab.v3_selection(
                True, 3.9, "improved", hypothesis="h", recipe_hash="r",
                diagnostics={"scope": "inner_selection", "available": False, "reason": "labels too long"})),
            "unsupported": lab.v3_result(rng, {"accepted": True, "verified": False, "reason": "host_metric_unsupported",
                                               "scope": "inner_selection", "hypothesis": "h"}),
            "execution_failed": lab.v3_failed("Traceback\nValueError: x", extra={
                "candidate_execution": {"phase": "inner_selection"}}),
            "analysis": lab.v3_analysis("shape (10, 3)"),
        }
        for name, result in cases.items():
            with self.subTest(name=name):
                feedback = compact_tool_feedback(result, limits=PolicyLimits(), redact=redact)
                chosen, original = feedback["selection"], result["selection"]
                self.assertIs(chosen["accepted"], original["accepted"])
                self.assertIs(chosen["verified"], original["verified"])
                expected = original.get("metric")
                self.assertEqual(chosen["metric"], None if expected is None else
                                 {key: expected[key] for key in ("name", "value", "higher_is_better")})
                self.assertEqual(chosen["reason"], original["reason"])
        self.assertEqual(compact_tool_feedback(cases["improved"], limits=PolicyLimits(), redact=redact)["dropped_fields"], 3)
        self.assertGreaterEqual(compact_tool_feedback(cases["execution_failed"], limits=PolicyLimits(),
                                                      redact=redact)["dropped_fields"], 1)

    def test_refit_budget_is_a_resource_problem_not_a_weaker_idea(self):
        trial = llm_trial(1, accepted=False, verified=True, reason="refit_budget")
        decision = dlg_decide(dlg_snapshot([trial]))
        self.assertEqual(decision.reason, "refit_budget_continue")
        self.assertIn("cheaper to train", decision.instruction)
        self.assertNotIn("not better", decision.instruction)
        legacy = decide(snapshot_with(make_trial(1, accepted=False, reason="refit_budget"), verified=1))
        self.assertEqual(legacy.reason, "refit_budget_continue")

    def test_inner_classification_diagnostics_are_summarised_in_the_v3_shape(self):
        rng = __import__("random").Random(5)
        diagnostics = lab.v3_inner_diagnostics(rng, 0.7, classification=True)
        selection = lab.v3_selection(False, 0.7, "not_better", hypothesis="h", recipe_hash="r", diagnostics=diagnostics)
        feedback = compact_tool_feedback(lab.v3_result(rng, selection, stdout="ok"),
                                         limits=PolicyLimits(max_tool_feedback_bytes=2500), redact=redact)
        summary = feedback["selection"]["diagnostics"]["classification"]
        self.assertLessEqual(len(summary["per_class"]), 5)
        self.assertEqual(summary["per_class"][0]["label"], diagnostics["classification"]["per_class"][0]["label"])
        self.assertEqual(summary["accuracy"], 0.61)
        self.assertLessEqual(json_size(feedback), 2500)
        self.assertEqual(feedback["selection"]["metric"]["value"], 0.7)


class MessageMetadataTests(unittest.TestCase):
    def test_normal_identifiers_and_run_python_are_unchanged(self):
        history = make_group(1, calls=2)
        history[0]["tool_calls"][0]["id"] = "call_AbC123xyz"
        history[0]["tool_calls"][1]["id"] = "chatcmpl-tool-7f31"
        history[1]["tool_call_id"], history[2]["tool_call_id"] = "call_AbC123xyz", "chatcmpl-tool-7f31"
        context = build(history, snapshot_with(make_trial(1), verified=1))
        calls = next(m for m in context.messages if m.get("tool_calls"))["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_AbC123xyz", "chatcmpl-tool-7f31"])
        self.assertEqual({call["function"]["name"] for call in calls}, {"run_python"})
        self.assertEqual([m["tool_call_id"] for m in context.messages if m["role"] == "tool"],
                         ["call_AbC123xyz", "chatcmpl-tool-7f31"])

    def test_a_secret_in_an_id_or_a_tool_name_is_a_safe_input_error(self):
        def mutate(path):
            history = make_group(1)
            if path == "id":
                history[0]["tool_calls"][0]["id"] = KEY
                history[1]["tool_call_id"] = KEY
            elif path == "name":
                history[0]["tool_calls"][0]["function"]["name"] = KEY
            else:
                history[1]["tool_call_id"] = KEY  # also breaks the link, still never echoed
            return history
        for path in ("id", "name", "tool_call_id"):
            with self.subTest(path=path):
                with self.assertRaises(PolicyInputError) as caught:
                    build(mutate(path))
                self.assertNotIn(KEY, str(caught.exception))
        embedded = make_group(1)
        embedded[0]["tool_calls"][0]["id"] = f"call_{KEY}_tail"
        embedded[1]["tool_call_id"] = f"call_{KEY}_tail"
        with self.assertRaises(PolicyInputError) as caught:
            build(embedded)
        self.assertNotIn(KEY, str(caught.exception))

    def test_metadata_is_never_shortened_for_size(self):
        long_id = "call_" + "x" * 300
        history = make_group(1)
        history[0]["tool_calls"][0]["id"] = long_id
        history[1]["tool_call_id"] = long_id
        context = build(history, snapshot_with(make_trial(1), verified=1))
        self.assertEqual(next(m for m in context.messages if m.get("tool_calls"))["tool_calls"][0]["id"], long_id)

    def test_pending_v3_group_is_never_completed_with_a_fake_result(self):
        pending = make_group(1, calls=2)[:2]  # two calls, one result
        with self.assertRaises(PolicyInputError) as caught:
            build(pending)
        self.assertIn("unfinished", str(caught.exception))


class ResearchV3MockHostTests(unittest.TestCase):
    """Real SYSTEM_V3, a real v3 manifest and the real budget message through the policy."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = lab.v3_fixture()

    def contract(self, **changes):
        return lab.v3_contract(self.fixture, **changes)

    def test_real_system_prompt_manifest_and_context(self):
        self.assertIs(self.fixture["system"], SYSTEM_V3)
        self.assertEqual(self.fixture["manifest"]["version"], 3)
        self.assertEqual(self.fixture["manifest"]["required_outputs"],
                         ["research_candidate.py", "research_recipe.json", "research_hypothesis.json"])
        contract = self.contract()
        snapshot = lab.v3_snapshot([], stage="inspect", seconds=9000.0, research_seconds=5800.0, requests=12,
                                   best=self.fixture["baseline_metric"])
        decision = decide_next_action(snapshot, contract=contract)
        self.assertEqual((decision.stage, decision.reason), ("inspect", "inspect_dialogue"))
        context = build_model_messages(
            system_prompt=SYSTEM_V3, task_text=self.fixture["task_text"], contract=contract, history=[],
            snapshot=snapshot, limits=PolicyLimits(), redact=lab.redact)
        self.assertEqual(context.messages[0]["content"], SYSTEM_V3)
        self.assertEqual(context.messages[1]["content"], self.fixture["task_text"])
        self.assertIn(self.fixture["context"], context.messages[2]["content"])
        for path in self.fixture["manifest"]["allowed_inputs"]:
            self.assertIn(path, context.messages[2]["content"])
        self.assertNotIn("columns", context.messages[2]["content"].split("required_outputs:")[1].split("\n")[0])

    def test_the_v3_scenarios_hold_their_transitions_facts_and_limits(self):
        limits = PolicyLimits()
        for scenario in lab.build_v3_scenarios():
            with self.subTest(scenario=scenario["name"]):
                result = lab.run_v3_scenario(self.fixture, scenario, limits)
                self.assertEqual([t for t in result["transitions"] if not t["ok"]], [])
                for record in result["facts"]:
                    self.assertTrue(all(value for key, value in record.items() if key != "step"), record)
                self.assertTrue(result["static_prefix_stable"])
                for size in result["policy_sizes"]:
                    self.assertGreater(size, 0)
                for diagnostics in result["diagnostics"]:
                    self.assertLessEqual(diagnostics["after_bytes"], limits.max_messages_bytes)

    def test_outer_and_test_sentinels_cannot_change_messages_or_decisions(self):
        def project(outer):
            host = {"rows": [lab.v3_row(1, "code", lab.v3_result(__import__("random").Random(2), lab.v3_selection(
                True, 3.9, "improved", hypothesis="h", recipe_hash="r")), None, recipe_hash="r")],
                    "outer_score": outer, "test_labels": [outer], "outer_sentinel": outer}
            snapshot = lab.v3_snapshot(host["rows"], stage="research", seconds=9000.0, research_seconds=5800.0,
                                       requests=10, best=self.fixture["baseline_metric"])
            decision = decide_next_action(snapshot, contract=self.contract())
            context = build_model_messages(system_prompt=SYSTEM_V3, task_text=self.fixture["task_text"],
                                           contract=self.contract(), history=[], snapshot=snapshot,
                                           limits=PolicyLimits(), redact=lab.redact)
            return decision, context.messages
        first = project(0.91)
        second = project(lab.OUTER_SENTINEL)
        self.assertEqual(first, second)
        self.assertNotIn(lab.OUTER_SENTINEL, json.dumps(second[1]))

    def test_unknown_outer_like_fields_are_ignored_and_never_reflected(self):
        clean = dlg_snapshot([llm_trial(1)])
        noisy = copy.deepcopy(clean)
        noisy["outer_score"] = {"value": 0.99}
        noisy["last_trial"]["outer_diagnostics"] = {"secret": "OUTER-LEAK"}
        noisy["recent_trials"][0]["outer_diagnostics"] = {"secret": "OUTER-LEAK"}
        self.assertEqual(build(snapshot=clean, contract=dlg_contract()).messages,
                         build(snapshot=noisy, contract=dlg_contract()).messages)
        self.assertEqual(dlg_decide(clean), dlg_decide(noisy))

    def test_system_v3_and_contract_that_cannot_fit_give_a_controlled_refusal(self):
        contract = self.contract()
        snapshot = lab.v3_snapshot([], stage="inspect", seconds=9000.0, research_seconds=5800.0, requests=12,
                                   best=self.fixture["baseline_metric"])
        decision = decide_next_action(snapshot, contract=contract)  # the host can still ask for a decision
        for limit in (500, 2000, 4500):
            with self.subTest(limit=limit), self.assertRaises(PolicyBudgetExceeded) as caught:
                build_model_messages(system_prompt=SYSTEM_V3, task_text=self.fixture["task_text"],
                                     contract=contract, history=[], snapshot=snapshot,
                                     limits=PolicyLimits(max_messages_bytes=limit), redact=lab.redact)
            self.assertNotIsInstance(caught.exception, PolicyInputError)
        self.assertEqual(decision.stage, "inspect")
        fits = build_model_messages(system_prompt=SYSTEM_V3, task_text=self.fixture["task_text"], contract=contract,
                                    history=[], snapshot=snapshot, limits=PolicyLimits(max_messages_bytes=12000),
                                    redact=lab.redact)
        self.assertLessEqual(fits.diagnostics["after_bytes"], 12000)

    def test_budget_message_is_part_of_the_measured_request_and_counted(self):
        message = lab.v3_budget_message(2, 8000.0)
        self.assertTrue(message["content"].startswith("Current host budget: "))
        payload = json.loads(message["content"].split(": ", 1)[1])
        self.assertEqual(payload["remaining_requests"], 10)
        self.assertEqual(payload["scope"], "inner_selection")
        result = lab.run_v3_scenario(self.fixture, lab.build_v3_scenarios()[0], PolicyLimits())
        self.assertGreater(result["control_total_bytes"], result["policy_total_bytes"])
        self.assertEqual(result["guard_blocked_at_request"]["policy"], None)

    def test_host_history_and_local_trials_keep_separate_counters(self):
        scenario = next(item for item in lab.build_v3_scenarios() if item["name"] == "history_local_feedback")
        result = lab.run_v3_scenario(self.fixture, scenario, PolicyLimits())
        self.assertTrue(all(record["separate_counters"] for record in result["facts"]))
        self.assertEqual([t["got"][2] for t in result["transitions"]], ["refit_budget_continue", "dialogue_complete"])


@contextmanager
def loopback(handler_factory):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def sdk_client(server):
    import httpx
    from openai import OpenAI
    return OpenAI(api_key="local-policy-placeholder", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                  timeout=10, max_retries=0, http_client=httpx.Client(trust_env=False, timeout=10))


@unittest.skipUnless(HAS_SDK, "Install requirements.txt for the SDK compatibility check")
class SdkCompatibilityTests(unittest.TestCase):
    """The compact request must be accepted by the real SDK; the server is local and the key fake."""

    def client(self, server):
        return sdk_client(server)

    def test_compact_request_reaches_the_server_and_error_snapshot_is_carried_forward(self):
        from agent.llm import TOOL, safe_error_message
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(payload)
                if len(received) == 1:
                    body = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": "m",
                                       "choices": [{"index": 0, "finish_reason": "stop",
                                                    "message": {"role": "assistant", "content": "ok"}}]})
                    status = 200
                else:
                    body = json.dumps({"error": {"message": "private-body-marker", "code": "rate_limit_exceeded"}})
                    status = 429
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-Id", "req-policy-1")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        history = make_group(1, code="print('é')", result=ok_result(stdout="Привет\n" * 5000))
        snapshot = snapshot_with(make_trial(1), verified=1)
        with loopback(Handler) as server, self.client(server) as client:
            first = build(history, snapshot)
            response = client.chat.completions.create(model="m", messages=first.messages, tools=[TOOL],
                                                      parallel_tool_calls=False, max_tokens=64)
            self.assertEqual(response.choices[0].message.content, "ok")
            self.assertEqual(received[0]["messages"], first.messages)
            self.assertLess(json_size(received[0]["messages"]), json_size(first.messages) + 1)
            self.assertLess(first.diagnostics["after_bytes"], first.diagnostics["before_bytes"])

            from openai import APIStatusError
            with self.assertRaises(APIStatusError) as caught:
                client.chat.completions.create(model="m", messages=first.messages, tools=[TOOL], max_tokens=64)
            safe = safe_error_message(caught.exception)
            failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind="api", error=safe,
                                reason="api_error")
            second = build(history, snapshot_with(failed, verified=1))
            self.assertIn("HTTP 429", second.messages[-1]["content"])
            self.assertIn("rate_limit_exceeded", second.messages[-1]["content"])
            self.assertNotIn("private-body-marker", json.dumps(second.messages))
            self.assertIn("retry_after_api_error", second.messages[-1]["content"])
            with self.assertRaises(APIStatusError):
                client.chat.completions.create(model="m", messages=second.messages, tools=[TOOL], max_tokens=64)
            self.assertEqual(received[2]["messages"], second.messages)
        self.assertEqual(len(received), 3)


@unittest.skipUnless(HAS_SDK, "Install requirements.txt for the SDK compatibility check")
class SdkV3CompatibilityTests(unittest.TestCase):
    """One localhost POST: the v3 request (policy messages plus the real budget message) through the SDK."""

    def test_v3_request_with_the_budget_message_reaches_the_loopback_server(self):
        from agent.llm import TOOL
        import random
        fixture = lab.v3_fixture()
        selection = lab.v3_selection(True, 3.9, "improved", hypothesis="Use shallower trees", recipe_hash="r1")
        result = lab.v3_result(random.Random(4), selection)
        history = make_group(1, code="print('v3')", result=result)
        rows = [lab.v3_row(1, "print('v3')", result, None, recipe_hash="r1")]
        snapshot = lab.v3_snapshot(rows, stage="research", seconds=8000.0, research_seconds=4800.0, requests=10,
                                   best=rows[0]["metric"])
        context = build_model_messages(
            system_prompt=SYSTEM_V3, task_text=fixture["task_text"], contract=lab.v3_contract(fixture),
            history=history, snapshot=snapshot, limits=PolicyLimits(), redact=lab.redact)
        messages = context.messages + [lab.v3_budget_message(2, 8000.0)]
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": "m",
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with loopback(Handler) as server, sdk_client(server) as client:
            response = client.chat.completions.create(model="m", messages=messages, tools=[TOOL], max_tokens=64)
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["messages"], messages)
        self.assertEqual(received[0]["messages"][0]["content"], SYSTEM_V3)
        self.assertTrue(received[0]["messages"][-1]["content"].startswith("Current host budget: "))


class LegacyBytesUnchangedTests(unittest.TestCase):
    """Digests were computed with the version-1 module (PR head 3e10edd) on the same legacy inputs."""

    EXPECTED = {
        "inspect": "6d512fa1dd58664ecd58e36935bdb7741588e0c34c7d63649f9c9fd22c38d766",
        "repair": "c4071e1b5e6bd11f96bcfd99b35440c44610958555711ede04823ae372785dd7",
        "improved": "293dc915f69e82304889b8d5db0bdc8f48b99dac15528e430b99536cead5f7db",
        "finalize": "093de6f8166d8af5d45cdf15f31bacc431f23342230e84b1048b89f32471fa11",
        "complete": "586c2e4ecb36b41b97aa15e5de864ee2e590a4b3c6fb704f1a9891c8c4c844b0",
    }

    def cases(self):
        failed = make_trial(2, accepted=False, verified=False, metric=None, error_kind="execution",
                            error="NameError: x", reason="execution_failed")
        stderr_result = ok_result(exit_code=1, stderr="Traceback\nNameError: x", selection={
            "accepted": False, "verified": False, "reason": "execution_failed"})
        return {
            "inspect": (make_snapshot(stage="inspect"), []),
            "repair": (snapshot_with(failed, streak=1), make_group(1) + make_group(2, result=stderr_result)),
            "improved": (snapshot_with(make_trial(2), verified=1), make_group(1) + make_group(2, calls=2)),
            "finalize": (snapshot_with(make_trial(2), verified=1, stage="finalize"), make_group(1)),
            "complete": (snapshot_with(make_trial(3), verified=2, finish_requested=True), make_group(1)),
        }

    def test_legacy_decisions_and_messages_are_byte_identical_to_version_1(self):
        for name, (snapshot, history) in self.cases().items():
            with self.subTest(case=name):
                contract = make_contract()
                decision = decide_next_action(snapshot, contract=contract)
                context = build_model_messages(
                    system_prompt="SYSTEM", task_text="TASK", contract=contract, history=history,
                    snapshot=snapshot, limits=PolicyLimits(), redact=redact)
                payload = json.dumps([decision.stage, decision.action, decision.reason, decision.instruction,
                                      context.messages, context.diagnostics], ensure_ascii=False)
                self.assertEqual(hashlib.sha256(payload.encode("utf-8")).hexdigest(), self.EXPECTED[name])


class InputsStayUntouchedTests(unittest.TestCase):
    def test_dialogue_inputs_are_not_mutated_and_everything_is_deterministic(self):
        rng = __import__("random").Random(8)
        history = make_group(1, calls=2, result=lab.v3_result(rng, lab.v3_selection(
            False, 4.0, "refit_budget", origin="local", hypothesis={"hypothesis": "local depth 6", "x": {"y": 1}},
            recipe_hash="r")))
        rows = [llm_trial(1, hypothesis={"hypothesis": "object text"}), llm_trial(2, accepted=False)]
        snapshot = dlg_snapshot(rows, finish=True)
        snapshot["budget"]["research_seconds_available"] = 500.0
        contract = dlg_contract(pacing={"low_remaining_seconds": 60})
        before = copy.deepcopy((history, snapshot, contract))
        first = build(history, snapshot, contract)
        second = build(history, snapshot, contract)
        decision = dlg_decide(snapshot, contract)
        feedback = compact_tool_feedback(json.loads(history[1]["content"]), limits=PolicyLimits(), redact=redact)
        self.assertEqual((history, snapshot, contract), before)
        self.assertEqual(first, second)
        self.assertEqual(decision, dlg_decide(snapshot, contract))
        self.assertEqual(feedback, compact_tool_feedback(json.loads(history[1]["content"]),
                                                         limits=PolicyLimits(), redact=redact))


if __name__ == "__main__":
    unittest.main()
