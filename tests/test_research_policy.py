"""Offline tests for the bounded research-dialogue policy (agent/research_policy.py)."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import threading
import unittest

from agent import research_policy as policy
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


@unittest.skipUnless(HAS_SDK, "Install requirements.txt for the SDK compatibility check")
class SdkCompatibilityTests(unittest.TestCase):
    """The compact request must be accepted by the real SDK; the server is local and the key fake."""

    def client(self, server):
        import httpx
        from openai import OpenAI
        return OpenAI(api_key="local-policy-placeholder", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                      timeout=10, max_retries=0, http_client=httpx.Client(trust_env=False, timeout=10))

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


if __name__ == "__main__":
    unittest.main()
