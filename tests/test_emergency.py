"""A genuine early target prior survives failures without trusting arbitrary CSVs."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.emergency import EmergencyCheckpoint, _fit_prior
from agent.execution import run_python as real_run_python
from offline_baseline import read_csv


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def task_fixture(root, targets=(1, 3, 101), *, metric="MAE", classes=None,
                 columns=None, positive=None):
    root = Path(root)
    data = root / "original inputs"
    data.mkdir()
    classification = classes is not None
    write_csv(data / "data_dictionary.csv", [
        ["feature", "role", "dtype"], ["row_key", "id", "category"],
        ["x", "feature", "numeric"],
        ["outcome", "target", "category" if classification else "numeric"],
    ])
    write_csv(data / "train.csv", [["row_key", "x", "outcome"], *[
        [f"fit-{index}", index, value] for index, value in enumerate(targets)
    ]])
    write_csv(data / "test.csv", [["row_key", "x"], ["0003", 999], ["NA", -999], ["01", 0]])
    columns = columns or ["row_key", "outcome"]
    write_csv(data / "sample_submission.csv", [columns, *[
        [key if name == "row_key" else 0.123 for name in columns] for key in ("0003", "NA", "01")
    ]])
    text = f"Metric: {metric}\nPredict outcome using original inputs/*.csv. Save submission.csv here.\n"
    if classes is not None:
        text += "Class order: " + json.dumps(classes) + "\n"
    if positive is not None:
        text += "Positive class: " + json.dumps(positive) + "\n"
    path = root / "task.md"
    path.write_text(text, encoding="utf-8")
    return path, data


class EmergencyCheckpointTests(unittest.TestCase):
    def prepare(self, task, **kwargs):
        checkpoint = EmergencyCheckpoint(task)
        self.assertTrue(checkpoint.prepare(timeout=20, deadline=time.monotonic() + 600, **kwargs),
                        checkpoint.metadata)
        return checkpoint

    def test_mae_median_is_trained_on_full_target_and_predicts_original_test_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = task_fixture(directory)
            originals = {path: path.read_bytes() for path in [task, *data.iterdir()]}
            checkpoint = self.prepare(task)
            predicted = read_csv(task.parent / "submission.csv")
            self.assertEqual(predicted["row_key"].tolist(), ["0003", "NA", "01"])
            self.assertEqual(predicted["outcome"].astype(float).tolist(), [3.0] * 3)
            self.assertEqual(checkpoint.metadata["model"], {"strategy": "median", "fit_rows": 3, "value": 3.0})
            self.assertFalse(checkpoint.metadata["quality_verified"])
            self.assertEqual(checkpoint.metadata["kind"], "target_prior")
            self.assertTrue(checkpoint.metadata["published"])
            self.assertEqual(checkpoint.report_path.name, "emergency.json")
            self.assertEqual(list((task.parent / ".agent_runs").glob("*/run.json")), [])
            self.assertTrue(all(path.read_bytes() == original for path, original in originals.items()))
            self.assertNotEqual((task.parent / "submission.csv").read_bytes(),
                                (data / "sample_submission.csv").read_bytes())

    def test_rmse_mean_is_a_target_fit_not_a_template_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory, (1, 3, 101), metric="RMSE")
            checkpoint = self.prepare(task)
            self.assertEqual(checkpoint.metadata["model"]["strategy"], "mean")
            values = read_csv(task.parent / "submission.csv")["outcome"].astype(float)
            self.assertTrue(all(math.isclose(value, 35.0) for value in values))

    def test_label_prior_preserves_literal_classes_and_declared_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory, ("01", "NA", "01", "NA"),
                                   metric="Macro F1", classes=["NA", "01"],
                                   columns=["outcome", "row_key"])
            checkpoint = self.prepare(task)
            predicted = read_csv(task.parent / "submission.csv")
            self.assertEqual(list(predicted.columns), ["outcome", "row_key"])
            self.assertEqual(predicted["outcome"].tolist(), ["NA"] * 3)
            self.assertEqual(checkpoint.metadata["model"]["classes"], ["NA", "01"])

    def test_binary_probability_prior_uses_the_declared_positive_class(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory, ("NA", "NA", "01", "NA"),
                                   metric="LogLoss", classes=["NA", "01"], positive="01")
            self.prepare(task)
            predicted = read_csv(task.parent / "submission.csv")
            self.assertEqual(predicted["row_key"].tolist(), ["0003", "NA", "01"])
            self.assertEqual(predicted["outcome"].astype(float).tolist(), [0.25] * 3)

    def test_multiclass_probability_columns_follow_template_not_model_order(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory, ("NA", "NA", "01", "worm"),
                                   metric="LogLoss", classes=["worm", "NA", "01"],
                                   columns=["01", "row_key", "worm", "NA"])
            self.prepare(task)
            predicted = read_csv(task.parent / "submission.csv")
            self.assertEqual(list(predicted.columns), ["01", "row_key", "worm", "NA"])
            for label, expected in (("01", .25), ("worm", .25), ("NA", .5)):
                self.assertEqual(predicted[label].astype(float).tolist(), [expected] * 3)

    def test_prepare_preserves_valid_existing_output_but_restore_uses_its_own_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            output = task.parent / "submission.csv"
            original = b"row_key,outcome\n0003,42.0000\nNA,43.0000\n01,44.0000\n"
            output.write_bytes(original)
            checkpoint = self.prepare(task)
            self.assertEqual(output.read_bytes(), original)
            self.assertTrue(checkpoint.metadata["preserved_existing"])
            self.assertFalse(checkpoint.metadata["published"])
            output.write_text("broken\n", encoding="utf-8")
            self.assertTrue(checkpoint.restore(), checkpoint.metadata)
            self.assertEqual(read_csv(output)["outcome"].astype(float).tolist(), [3.0] * 3)
            self.assertNotEqual(output.read_bytes(), original)

    def test_invalid_existing_output_is_replaced_by_verified_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            output = task.parent / "submission.csv"
            output.write_text("wrong_id,outcome\nmissing,0\n", encoding="utf-8")
            checkpoint = self.prepare(task)
            self.assertTrue(checkpoint.metadata["published"])
            self.assertEqual(read_csv(output)["row_key"].tolist(), ["0003", "NA", "01"])

    def test_mutated_original_inputs_prevent_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = task_fixture(directory)
            checkpoint = self.prepare(task)
            output = task.parent / "submission.csv"
            output.write_text("broken\n", encoding="utf-8")
            with (data / "train.csv").open("a", encoding="utf-8") as stream:
                stream.write("new-fit,0,999\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                checkpoint.check_inputs()
            self.assertFalse(checkpoint.restore())
            self.assertEqual(output.read_text(encoding="utf-8"), "broken\n")

    def test_mutation_during_worker_prevents_initial_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            checkpoint = EmergencyCheckpoint(task)

            def run_then_mutate(*args, **kwargs):
                result = real_run_python(*args, **kwargs)
                task.write_text(task.read_text(encoding="utf-8") + "Changed task.\n", encoding="utf-8")
                return result

            with patch("agent.emergency.run_python", side_effect=run_then_mutate):
                self.assertFalse(checkpoint.prepare(timeout=20, deadline=time.monotonic() + 600))
            self.assertFalse((task.parent / "submission.csv").exists())
            self.assertTrue(checkpoint.inputs_frozen)
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                checkpoint.check_inputs()
            self.assertFalse(checkpoint.restore())

    def test_rewriting_backup_and_matching_receipt_does_not_replace_host_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            checkpoint = self.prepare(task)
            prior = checkpoint.directory / "prior.csv"
            changed = b"row_key,outcome\n0003,888\nNA,888\n01,888\n"
            prior.write_bytes(changed)
            receipt = json.loads(checkpoint.report_path.read_text(encoding="utf-8"))
            receipt["submission_sha256"] = hashlib.sha256(changed).hexdigest()
            checkpoint.report_path.write_text(json.dumps(receipt), encoding="utf-8")
            output = task.parent / "submission.csv"
            before = output.read_bytes()
            self.assertFalse(checkpoint.restore())
            self.assertEqual(output.read_bytes(), before)

    def test_expired_budget_does_not_launch_or_clobber_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            output = task.parent / "submission.csv"
            original = b"existing bytes stay untouched"
            output.write_bytes(original)
            checkpoint = EmergencyCheckpoint(task)
            with patch("agent.emergency.run_python") as worker:
                self.assertFalse(checkpoint.prepare(timeout=20, deadline=time.monotonic() - 1))
            worker.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_worker_timeout_is_bounded_by_five_percent_and_never_clobbers_output(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            output = task.parent / "submission.csv"
            original = b"existing bytes stay untouched"
            output.write_bytes(original)
            checkpoint = EmergencyCheckpoint(task)
            with patch("agent.emergency.run_python", return_value={"exit_code": -1, "timed_out": True}) as worker:
                self.assertFalse(checkpoint.prepare(timeout=30, deadline=time.monotonic() + 100))
            self.assertLessEqual(worker.call_args.kwargs["timeout"], 5)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse(checkpoint.restore())

    def test_escape_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            with self.assertRaisesRegex(ValueError, "inside task"):
                EmergencyCheckpoint(task, task.parent / "outside emergency")

    def test_symlink_backup_is_rejected_without_touching_its_target(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory)
            checkpoint = self.prepare(task)
            outside = task.parent / "outside.csv"
            outside.write_text("outside remains", encoding="utf-8")
            prior = checkpoint.directory / "prior.csv"
            prior.unlink()
            try:
                prior.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"Symlinks unavailable: {error}")
            self.assertFalse(checkpoint.restore())
            self.assertEqual(outside.read_text(encoding="utf-8"), "outside remains")

    def test_invalid_train_ids_cannot_publish_an_emergency_checkpoint(self):
        for ids in (("same", "same", "different"), ("first", "   ", "third")):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as directory:
                task, data = task_fixture(directory)
                write_csv(data / "train.csv", [["row_key", "x", "outcome"], *[
                    [row_id, index, index + 1] for index, row_id in enumerate(ids)
                ]])
                checkpoint = EmergencyCheckpoint(task)
                self.assertFalse(checkpoint.prepare(timeout=20, deadline=time.monotonic() + 600))
                self.assertFalse((task.parent / "submission.csv").exists())
                self.assertFalse(checkpoint.restore())

    def test_duplicate_train_ids_across_streaming_chunks_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = task_fixture(directory)
            write_csv(data / "train.csv", [["row_key", "x", "outcome"],
                ["first", 1, 1], ["second", 2, 2], ["third", 3, 3], ["first", 4, 4]])
            output = task.parent / ".agent_runs" / "streaming-test"
            output.mkdir(parents=True)
            with patch("agent.emergency.CHUNK_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "training IDs are empty or duplicated"):
                    _fit_prior(task, output, "streaming-fixture")
            self.assertFalse((output / "prior.csv").exists())
            self.assertFalse((task.parent / "submission.csv").exists())

    def test_nonfinite_target_does_not_publish_or_claim_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _ = task_fixture(directory, (1, "inf", 3))
            checkpoint = EmergencyCheckpoint(task)
            self.assertFalse(checkpoint.prepare(timeout=20, deadline=time.monotonic() + 600))
            self.assertFalse((task.parent / "submission.csv").exists())
            self.assertFalse(checkpoint.restore())
            self.assertFalse(checkpoint.metadata["available"])


if __name__ == "__main__":
    unittest.main()
