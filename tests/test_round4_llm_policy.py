"""Risk-focused checks for an inactive, append-only INNER action policy."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from agent.round4_llm_policy import prepare_round4_messages, validate_experiment_history
from agent.research_v3 import SYSTEM_V3
from build_submission import build_archive


def snapshot():
    return {"version": 1, "scope": "inner_selection",
            "task": {"task_type": "regression", "metric": {"name": "RMSE", "higher_is_better": False},
                     "prediction_kind": "regression"},
            "features": [{"name": "measurement", "role": "numeric"}],
            "allowed_inputs": ["task.md", "research_validation.json", "data/train.csv"],
            "shape": {"fit_rows": 24, "validation_rows": 8, "feature_count": 1},
            "split": {"strategy": "random", "seed": 42}, "installed_tools": ["numpy", "sklearn"],
            "timing": {"remaining_seconds": 90, "final_reserve_seconds": 30,
                       "maximum_experiment_seconds": 60, "last_fit_seconds": None}}


def identity(alpha=1, code="a" * 64):
    recipe = {"family": "linear", "alpha": alpha}
    canonical = json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"code_hash": code, "recipe_hash": hashlib.sha256(canonical.encode()).hexdigest(), "recipe": recipe}


def verified(alpha=1, code="a" * 64):
    return {"scope": "inner_selection", "outcome": "verified_inner", "identity": identity(alpha, code),
            "metric": {"name": "RMSE", "value": .7, "higher_is_better": False}, "accepted": False,
            "elapsed_seconds": .2}


def dialogue():
    return [{"role": "system", "content": SYSTEM_V3},
            {"role": "user", "content": "Preserve task and initial recipe exactly."},
            {"role": "assistant", "content": "Inspect one fact.", "tool_calls": [{"id": "one",
                "type": "function", "function": {"name": "run_python", "arguments": '{"code":"print(1)"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": json.dumps({"exit_code": 0,
                "stdout": "A unique fact that must remain.\n", "stderr": "A substantial warning.\n",
                "selection": {"scope": "inner_selection", "verified": False, "reason": "missing_files"}})}]


class Round4PolicyTests(unittest.TestCase):
    def prepare(self, *, messages=None, evidence=None, history=None, remaining=3):
        return prepare_round4_messages(dialogue() if messages is None else messages, remaining_requests=remaining,
                                      inner_snapshot=snapshot() if evidence is None else evidence,
                                      experiment_history=[] if history is None else history)

    def test_real_system_pairs_errors_and_input_objects_stay_unchanged(self):
        messages, evidence, history = dialogue(), snapshot(), [verified()]
        before = deepcopy((messages, evidence, history))
        prepared = self.prepare(messages=messages, evidence=evidence, history=history)
        self.assertEqual(prepared.action, "create_candidate")
        self.assertEqual(prepared.messages[:-1], messages)
        self.assertEqual((messages, evidence, history), before)
        self.assertIn("fit_predict(X_fit, y_fit, X_predict, context)", prepared.messages[-1]["content"])
        for name in ("research_hypothesis.json", "research_recipe.json", "research_candidate.py"):
            self.assertIn(name, prepared.messages[-1]["content"])
        self.assertIn("Do not read original train", prepared.messages[-1]["content"])
        self.assertIn("branch on phase", prepared.messages[-1]["content"])
        self.assertGreater(prepared.summary["prepared_bytes"], prepared.summary["original_bytes"])
        prepared.messages[0]["content"] = "mutated output copy"
        self.assertEqual(messages, before[0])

    def test_unknown_essential_allows_one_inspection_and_optional_missing_never_invents_values(self):
        evidence = snapshot()
        evidence["features"] = None
        first = self.prepare(evidence=evidence)
        self.assertEqual(first.action, "inspect_one_field")
        self.assertEqual(first.summary["selected_inspection_field"], "features")
        second = self.prepare(evidence=evidence, history=[{"scope": "inner_selection", "outcome": "narrow_inspection",
                                                         "inspected_field": "features"}])
        self.assertEqual(second.action, "stop_local")
        self.assertEqual(second.stop_reason, "essential_context_still_missing")
        optional = self.prepare()
        self.assertEqual(optional.action, "create_candidate")
        self.assertIn('"current_metric":null', optional.messages[-1]["content"])
        self.assertIn('"last_fit_seconds":null', optional.messages[-1]["content"])

    def test_unknown_literal_class_order_does_not_get_guessed(self):
        evidence = snapshot()
        evidence["task"] = {"task_type": "classification", "metric": {"name": "Macro F1", "higher_is_better": True},
                            "prediction_kind": "labels", "classes": None}
        with self.assertRaises(ValueError):
            self.prepare(evidence=evidence)

    def test_exact_pair_identity_not_family_or_tool_script_controls_duplicates(self):
        for history in ([verified(1), verified(2)], [verified(1), verified(1, "b" * 64)]):
            with self.subTest(history=history):
                prepared = self.prepare(history=history)
                self.assertEqual(prepared.action, "create_candidate")
                self.assertEqual(prepared.summary["duplicate_verified_observations"], 0)
        duplicate = self.prepare(history=[verified(), verified()])
        self.assertEqual(duplicate.action, "stop_local")
        self.assertEqual(duplicate.stop_reason, "verified_recipe_repeated")
        self.assertEqual(duplicate.summary["unique_verified_identities"], 1)
        self.assertEqual(duplicate.messages, dialogue())
        dishonest = verified()
        dishonest["identity"]["recipe"]["alpha"] = 999
        with self.assertRaises(ValueError):
            self.prepare(history=[dishonest])

    def test_two_nontrials_stop_but_single_failure_does_not_equal_fit(self):
        row = {"scope": "inner_selection", "outcome": "no_candidate", "elapsed_seconds": .01}
        first = self.prepare(history=[row])
        self.assertEqual(first.action, "create_candidate")
        self.assertEqual(first.summary["verified_inner_observations"], 0)
        stopped = self.prepare(history=[row, row])
        self.assertEqual(stopped.stop_reason, "unproductive_attempts")
        recovered = self.prepare(history=[row, row, verified()])
        self.assertEqual(recovered.action, "create_candidate")
        fake = {**row, "metric": verified()["metric"], "accepted": True}
        with self.assertRaises(ValueError):
            self.prepare(history=[fake])

    def test_no_new_budget_or_unknown_time_invention(self):
        self.assertEqual(self.prepare(remaining=0).stop_reason, "attempts_exhausted")
        evidence = snapshot()
        evidence["timing"] = {"remaining_seconds": 30, "final_reserve_seconds": 30}
        self.assertEqual(self.prepare(evidence=evidence).stop_reason, "research_time_exhausted")
        evidence["timing"] = None
        self.assertEqual(self.prepare(evidence=evidence).action, "create_candidate")
        for remaining in (-1, True, 1.5):
            with self.subTest(remaining=remaining), self.assertRaises(ValueError):
                self.prepare(remaining=remaining)

    def test_rejected_pairs_scope_and_raw_report_fail_before_preparation(self):
        with self.assertRaises(ValueError):
            self.prepare(messages=dialogue()[:-1])
        messages = dialogue()
        messages[-1]["content"] = '{"selection":{"scope":"outer_validation","verified":true}}'
        with self.assertRaises(ValueError):
            self.prepare(messages=messages)
        for row in ({**verified(), "scope": "outer_validation"}, {**verified(), "error": "secret-provider-body"},
                    {"scope": "inner_selection", "outcome": "verified_inner", "identity": identity()}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.prepare(history=[row])
        with self.assertRaises(ValueError):
            self.prepare(history=[verified()] * 13)
        with self.assertRaises(ValueError):
            self.prepare(evidence={**snapshot(), "evaluations": {"final_check": "private"}})

    def test_policy_import_from_flat_zip_has_no_source_checkout_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_archive(root / "solution.zip")
            with zipfile.ZipFile(root / "solution.zip") as archive:
                archive.extractall(root)
            env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
            result = subprocess.run([sys.executable, "-c",
                "from agent.round4_llm_policy import POLICY_VERSION; "
                "from agent.round4_llm_evidence import validate_inner_snapshot; "
                "assert POLICY_VERSION == 'round4_inner_action_v1'; print('flat policy OK')"],
                cwd=root, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "flat policy OK")


if __name__ == "__main__":
    unittest.main()
