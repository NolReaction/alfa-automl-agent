"""Lightweight native-harness evidence checks; no solver/model is launched."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_rescue_classification_native as native


class SourceSnapshotTests(unittest.TestCase):
    def test_dirty_tracked_checkout_is_rejected_before_hashing(self):
        with patch.object(native, "git", return_value=" M solution.py") as git:
            with self.assertRaisesRegex(ValueError, "tracked changes"):
                native.source_snapshot(Path("unused-checkout"))
        self.assertEqual(git.call_count, 1)

    def test_wrong_source_sha_is_rejected_before_hashing(self):
        with patch.object(native, "git", side_effect=["", "0" * 40]) as git:
            with self.assertRaisesRegex(ValueError, "preregistered main SHA"):
                native.source_snapshot(Path("unused-checkout"))
        self.assertEqual(git.call_count, 2)


class _NativeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()
        self.task = self.output / "task"
        self.run = self.task / ".agent_runs" / "one-run"
        self.run.mkdir(parents=True)
        (self.output / "host_only").mkdir()
        self.submission = self.task / "submission.csv"
        self.submission.write_text("id,attack_cat\n10,Normal\n20,Worms\n", encoding="utf-8")
        (self.output / "host_only/answers.csv").write_text(self.submission.read_text(), encoding="utf-8")
        self.plan = {"scope": "public_train_grouped_development_incumbent_only",
                     "source": {"sha": native.BASE_SHA}, "classes": ["Normal", "Worms"],
                     "id_column": "id", "target_column": "attack_cat",
                     "public_task": str(self.output / "original/task.md"),
                     "source_inputs": {"train.csv": "sealed-original-input"}}
        native.write_json(self.output / "plan.json", self.plan)
        native.write_json(self.output / "execution.json", {
            "exit_code": 0, "timed_out": False, "memory_limit_exceeded": False})
        self.code_hash, self.recipe_hash = "a" * 64, "b" * 64
        self.outer_path, self.refit_path = self.run / "outer/receipt.json", self.run / "refit/receipt.json"
        for path, phase in ((self.outer_path, "outer_validation"), (self.refit_path, "full_refit")):
            native.write_json(path, {"phase": phase, "code_hash": self.code_hash,
                                    "recipe_hash": self.recipe_hash,
                                    "output_hash": native.file_hash(self.submission)})
        self.report = {"api": {"attempts": 0, "responses": 0},
                       "candidate_receipts": {"outer": str(self.outer_path), "refit": str(self.refit_path),
                                              "code_hash": self.code_hash, "recipe_hash": self.recipe_hash},
                       "publication": {"new_result_published": True, "recipe": "baseline"},
                       "published_recipe": "baseline"}
        self.save_native()
        self.addCleanup(patch.stopall)
        patch.object(native, "load_plan", return_value=self.plan).start()
        patch.object(native, "task_hashes", return_value=self.plan["source_inputs"]).start()

    def save_native(self):
        native.write_json(self.run / "run.json", self.report)


class NativeAuditTests(_NativeFixture):
    def test_valid_native_output_recomputes_exact_metrics(self):
        result = native.audit(self.output)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scores"]["macro_f1"], 1.0)
        self.assertEqual(result["scores"]["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertTrue(result["api_zero_verified"])
        self.assertTrue(result["refit_output_hash_matches"])

    def test_publication_with_wrong_id_order_is_rejected(self):
        self.submission.write_text("id,attack_cat\n20,Worms\n10,Normal\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            native.audit(self.output)

    def test_publication_with_unknown_label_is_rejected(self):
        self.submission.write_text("id,attack_cat\n10,UNKNOWN\n20,Worms\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown predicted labels"):
            native.audit(self.output)

    def test_missing_actual_api_counter_is_never_inferred_as_zero(self):
        self.report["api"] = {"attempts": 0}
        self.save_native()
        result = native.audit(self.output)
        self.assertEqual(result["api"], {"attempts": 0, "responses": None})
        self.assertFalse(result["api_zero_verified"])
        self.assertNotEqual(result["status"], "completed")

    def test_nonzero_actual_api_counter_prevents_completed_status(self):
        self.report["api"] = {"attempts": 1, "responses": 0}
        self.save_native()
        result = native.audit(self.output)
        self.assertFalse(result["api_zero_verified"])
        self.assertNotEqual(result["status"], "completed")

    def test_changed_refit_output_hash_is_not_called_completed(self):
        receipt = native.read_json(self.refit_path)
        receipt["output_hash"] = "0" * 64
        native.write_json(self.refit_path, receipt)
        result = native.audit(self.output)
        self.assertFalse(result["refit_output_hash_matches"])
        self.assertNotEqual(result["status"], "completed")

    def test_multiple_native_runs_are_not_silently_cherry_picked(self):
        native.write_json(self.task / ".agent_runs/second-run/run.json", self.report)
        with self.assertRaisesRegex(ValueError, "Multiple native runs"):
            native.audit(self.output)

    def test_unavailable_monitor_does_not_claim_sampled_rss_or_enforcement(self):
        self.plan["memory_limit_bytes"] = 6 * 1024 ** 3
        self.plan["environment"] = {"cgroup": {"memory.max": str(8 * 1024 ** 3)}}
        execution = native.read_json(self.output / "execution.json")
        execution["resources"] = {"available": False, "peak_rss_bytes": 12345,
                                  "unavailable_reason": "Process namespace mismatch"}
        native.write_json(self.output / "execution.json", execution)
        result = native.audit(self.output)["resource_limit_evidence"]
        self.assertEqual(result["requested_limit_enforcement"], "unavailable")
        self.assertIsNone(result["peak_rss_bytes"])
        self.assertEqual(result["requested_tree_rss_limit_bytes"], 6 * 1024 ** 3)
        self.assertEqual(result["host_cgroup_hard_limit_bytes_at_prepare"], 8 * 1024 ** 3)
        self.assertFalse(result["official_32_gib_resource_confirmed"])


class NativeReceiptIdentityTests(_NativeFixture):
    """Regression guards for the audit tightening after the sealed run ends."""

    def test_refit_receipt_identity_tamper_is_rejected(self):
        for field, value in (("phase", "inner_selection"), ("code_hash", "0" * 64),
                             ("recipe_hash", "0" * 64)):
            with self.subTest(field=field):
                receipt = {"phase": "full_refit", "code_hash": self.code_hash,
                           "recipe_hash": self.recipe_hash, "output_hash": native.file_hash(self.submission)}
                receipt[field] = value
                native.write_json(self.refit_path, receipt)
                with self.assertRaisesRegex(ValueError, "receipt.*identity|identity.*receipt"):
                    native.audit(self.output)

    def test_outer_receipt_identity_tamper_is_rejected(self):
        for field, value in (("phase", "full_refit"), ("code_hash", "0" * 64),
                             ("recipe_hash", "0" * 64)):
            with self.subTest(field=field):
                receipt = {"phase": "outer_validation", "code_hash": self.code_hash,
                           "recipe_hash": self.recipe_hash, "output_hash": native.file_hash(self.submission)}
                receipt[field] = value
                native.write_json(self.outer_path, receipt)
                with self.assertRaisesRegex(ValueError, "receipt.*identity|identity.*receipt"):
                    native.audit(self.output)

    def test_changed_original_csv_is_detected(self):
        with patch.object(native, "task_hashes", return_value={"train.csv": "changed"}):
            with self.assertRaisesRegex(ValueError, "Original.*changed|original.*drift"):
                native.audit(self.output)

    def test_missing_outer_receipt_cannot_claim_complete_native_validation(self):
        self.report["candidate_receipts"].pop("outer")
        self.save_native()
        result = native.audit(self.output)
        self.assertFalse(result["outer_and_refit_receipts_verified"])
        self.assertNotEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
