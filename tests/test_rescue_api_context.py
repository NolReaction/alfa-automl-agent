"""Conservative v3 context preparation, without paid calls or model fitting."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from agent.llm_context import FIRST_TRIAL_HINT, LAST_TRIAL_HINT, compact_research_messages
from build_submission import build_archive


ROOT = Path(__file__).resolve().parents[1]


def system_v3():
    tree = ast.parse((ROOT / "agent/research_v3.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SYSTEM_V3"
                        for target in node.targets))


def dialogue(result=None):
    messages = [
        {"role": "system", "content": system_v3()},
        {"role": "user", "content": "Allowed task with raw feature schema and INNER-only inputs."},
        {"role": "user", "content": "Current recipe: {\"family\":\"linear\",\"degree\":1}"},
    ]
    if result is not None:
        messages += [{"role": "assistant", "content": "Measure one candidate.", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "run_python", "arguments": json.dumps({"code": "write_three_fresh_files()"})}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": json.dumps(result, ensure_ascii=False)}]
    return messages


def successful_result():
    return {
        "stdout": "All deterministic model setup diagnostics are unchanged.\n" * 150
                  + "Unique final diagnostic: coefficient=2.5\n",
        "stderr": "Keep this significant warning verbatim.\n" * 20,
        "exit_code": 0, "timed_out": False, "memory_limit_exceeded": False,
        "selection": {"scope": "inner_selection", "verified": True, "accepted": False,
                      "reason": "not_better", "metric": {"name": "RMSE", "value": 0.7,
                                                          "higher_is_better": False},
                      "code_hash": "a" * 64, "recipe_hash": "b" * 64,
                      "diagnostics": {"scope": "inner_selection", "bias": 0.1}},
        "recipe": {"family": "linear", "degree": 1},
    }


class RescueContextTests(unittest.TestCase):
    def test_first_trial_hint_preserves_actual_system_and_contract(self):
        original = dialogue()
        result = compact_research_messages(original, remaining_requests=2)
        self.assertEqual(result.messages[:-1], original)
        self.assertEqual(result.messages[0]["content"], system_v3())
        self.assertIn("fit_predict(X_fit, y_fit, X_predict, context)", result.messages[-1]["content"])
        for filename in ("research_candidate.py", "research_recipe.json", "research_hypothesis.json"):
            self.assertIn(filename, result.messages[-1]["content"])
        self.assertIn(LAST_TRIAL_HINT, result.messages[-1]["content"])
        self.assertTrue(result.summary["first_trial_hint_added"])
        self.assertFalse(result.summary["verified_inner_seen"])

    def test_fixture_saves_bytes_and_preserves_candidate_recipe_metric_and_pairs(self):
        result = successful_result()
        original = dialogue(result)
        before = deepcopy(original)
        prepared = compact_research_messages(original, remaining_requests=2)
        self.assertEqual(original, before)
        self.assertEqual(prepared.messages[:-1], original[:-1])
        self.assertEqual(prepared.messages[-1]["tool_call_id"], "call-1")
        payload = json.loads(prepared.messages[-1]["content"])
        self.assertEqual({key: value for key, value in payload.items() if key != "stdout"},
                         {key: value for key, value in result.items() if key != "stdout"})
        self.assertIn("149 additional times", payload["stdout"])
        self.assertIn("Unique final diagnostic: coefficient=2.5", payload["stdout"])
        self.assertLess(prepared.summary["prepared_bytes"], prepared.summary["original_bytes"] / 2)
        self.assertEqual(prepared.summary["stdout_runs_compacted"], 1)
        self.assertTrue(prepared.summary["verified_inner_seen"])
        self.assertFalse(prepared.summary["first_trial_hint_added"])
        prepared.messages[0]["content"] = "Changed copy"
        self.assertEqual(original, before)

    def test_all_failure_outputs_and_error_details_remain_bytewise_intact(self):
        variants = [{"exit_code": 1}, {"timed_out": True}, {"memory_limit_exceeded": True},
                    {"error": "Important failure"}, {"candidate_execution": {"error": "nested failure"}}]
        for change in variants:
            with self.subTest(change=change):
                result = successful_result()
                result.update(change)
                original = dialogue(result)
                prepared = compact_research_messages(original, remaining_requests=3)
                self.assertEqual(prepared.messages, original)
                self.assertEqual(prepared.summary["stdout_runs_compacted"], 0)

    def test_unverified_process_success_is_not_an_inner_trial(self):
        for selection in (None, {"scope": "inner_selection", "verified": False},
                          {"scope": "inner_selection", "verified": True}):
            result = successful_result()
            result["selection"] = selection
            original = dialogue(result)
            prepared = compact_research_messages(original, remaining_requests=4)
            self.assertEqual(prepared.messages[:-1], original)
            self.assertFalse(prepared.summary["verified_inner_seen"])
            self.assertNotIn(LAST_TRIAL_HINT, prepared.messages[-1]["content"])

    def test_unique_stdout_and_repeated_stderr_are_never_truncated(self):
        result = successful_result()
        result["stdout"] = "".join(f"Distinct diagnostic {number}\n" for number in range(500))
        original = dialogue(result)
        prepared = compact_research_messages(original, remaining_requests=3)
        self.assertEqual(prepared.messages, original)
        self.assertEqual(prepared.summary["original_bytes"], prepared.summary["prepared_bytes"])

    def test_short_repetitions_and_unicode_remain_valid(self):
        result = successful_result()
        result["stdout"] = "x\nx\n" + "Числовая диагностика без изменений.\n" * 30 + "край\n"
        prepared = compact_research_messages(dialogue(result), remaining_requests=3)
        stdout = json.loads(prepared.messages[-1]["content"])["stdout"]
        self.assertTrue(stdout.startswith("x\nx\n"))
        self.assertIn("29 additional times", stdout)
        self.assertTrue(stdout.endswith("край\n"))

    def test_no_hint_when_disabled_or_no_requests_and_no_duplicate_hint(self):
        original = dialogue()
        self.assertEqual(compact_research_messages(original, remaining_requests=0).messages, original)
        self.assertEqual(compact_research_messages(original, remaining_requests=2,
                                                 encourage_first_trial=False).messages, original)
        once = compact_research_messages(original, remaining_requests=2)
        twice = compact_research_messages(once.messages, remaining_requests=2)
        self.assertEqual(once.messages, twice.messages)
        self.assertFalse(twice.summary["first_trial_hint_added"])
        self.assertEqual(twice.messages[-1]["content"], FIRST_TRIAL_HINT + LAST_TRIAL_HINT)

    def test_outer_or_unscoped_selection_is_rejected(self):
        for scope in ("outer_validation", "final_check", "full_refit", None):
            with self.subTest(scope=scope):
                result = successful_result()
                result["selection"]["scope"] = scope
                with self.assertRaisesRegex(ValueError, "INNER"):
                    compact_research_messages(dialogue(result), remaining_requests=2)

    def test_unknown_tool_content_is_preserved_without_claiming_verified_trial(self):
        for content in ("not JSON", "[]", "null", '{"opaque":"keep all"}'):
            with self.subTest(content=content):
                original = dialogue(successful_result())
                original[-1]["content"] = content
                prepared = compact_research_messages(original, remaining_requests=2)
                self.assertEqual(prepared.messages[:-1], original)
                self.assertFalse(prepared.summary["verified_inner_seen"])

    def test_tool_pairs_reject_missing_unknown_and_duplicate_ids(self):
        original = dialogue(successful_result())
        missing = original[:-1]
        unknown = deepcopy(original)
        unknown[-1]["tool_call_id"] = "missing"
        duplicate = original + original[-2:]
        interrupted = original[:-1] + [{"role": "user", "content": "interrupt"}] + original[-1:]
        for messages in (missing, unknown, duplicate, interrupted):
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    compact_research_messages(messages, remaining_requests=2)

    def test_multiple_call_results_keep_order_and_all_pairs(self):
        original = dialogue(successful_result())
        second_call = deepcopy(original[-2]["tool_calls"][0])
        second_call["id"] = "call-2"
        original[-2]["tool_calls"].append(second_call)
        original.append({"role": "tool", "tool_call_id": "call-2", "content": "unknown output"})
        prepared = compact_research_messages(original, remaining_requests=2)
        self.assertEqual([value.get("tool_call_id") for value in prepared.messages],
                         [value.get("tool_call_id") for value in original])
        self.assertEqual(prepared.messages[-1], original[-1])

    def test_invalid_limits_are_rejected(self):
        for value in (-1, True, 2.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compact_research_messages(dialogue(), remaining_requests=value)

    def test_helper_imports_from_flat_archive_without_source_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "solution.zip"
            build_archive(archive)
            with zipfile.ZipFile(archive) as stream:
                self.assertIn("_alfa_agent__llm_context.py", stream.namelist())
                stream.extractall(root)
            env = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME"}}
            process = subprocess.run([sys.executable, "-c",
                "from agent.llm_context import compact_research_messages; "
                "from agent.api_probe import ProbeBudget, ProbeClient; "
                "from agent.llm import public_error_message; "
                "result = compact_research_messages([], remaining_requests=0); "
                "assert result.messages == []; print('flat import OK')"],
                cwd=root, env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.strip(), "flat import OK")


if __name__ == "__main__":
    unittest.main()
