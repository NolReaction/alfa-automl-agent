"""A reporting failure cannot downgrade the pinned, verified AutoML winner."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from agent import automl
from tests import test_automl as automl_fixture


class AutoMLRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing protocol fixture: genuine TaskSpec scoring and
        # checkpoint publication, with deterministic worker-produced CSVs.
        self.fixture = automl_fixture.AutoMLTests("test_all_failures_preserve_previous_csv_and_report_failure")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.task = self.fixture.root, self.fixture.task
        self.output = self.root / ".agent_runs" / "recovery_output"
        self.accepted = None
        self.accepted_bytes = None
        self.accepted_metadata = None

    def run_failure(self, *, mutate=None, exception=None, persistent=False, outcomes=None, clock=None):
        original = automl._atomic_json
        triggered = False

        def save(path, value):
            nonlocal triggered
            if Path(path).name == "automl_report.json" and value.get("output_valid") and (persistent or not triggered):
                if not triggered:
                    self.accepted = automl.read_accepted_checkpoint(self.output)
                    self.accepted_bytes = (self.accepted / "submission.csv").read_bytes()
                    self.accepted_metadata = deepcopy({key: value[key] for key in (
                        "model", "model_config", "feature_config", "metric")})
                    triggered = True
                    if mutate is not None:
                        mutate(value)
                raise exception if exception is not None else OSError("Injected report write failure")
            original(path, value)

        with redirect_stdout(io.StringIO()), \
                patch.object(automl, "candidate_plan", return_value=[]), \
                patch.object(automl, "run_python", side_effect=self.fixture.fake_execution(outcomes)), \
                patch.object(automl, "_atomic_json", side_effect=save):
            if clock is None:
                return automl.run_automl(self.task, time_budget=120, output_dir=self.output)
            with patch.object(automl.time, "monotonic", side_effect=clock):
                return automl.run_automl(self.task, time_budget=120, output_dir=self.output)

    def test_report_failure_returns_the_pinned_model_bytes_and_metadata(self):
        def corrupt_loose_output(report):
            (self.root / "submission.csv").write_text("id,y\n0001,999\nNA,999\n", encoding="utf-8")
            report["model_config"]["parameters"]["alpha"] = 999
            report["metric"]["value"] = 999

        report = self.run_failure(mutate=corrupt_loose_output)
        self.assertEqual((self.root / "submission.csv").read_bytes(), self.accepted_bytes)
        self.assertEqual(report["status"], "fallback")
        self.assertEqual(report["search"]["status"], "fallback")
        self.assertTrue(report["output_valid"])
        self.assertEqual(report["recovery"]["kind"], "accepted_automl_checkpoint")
        self.assertEqual(report["recovery"]["status"], "restored")
        for key, value in self.accepted_metadata.items():
            self.assertEqual(report[key], value)
        self.assertEqual([call["phase"] for call in self.fixture.calls], ["seed"])
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "fallback")
        self.assertEqual(saved["metric"], report["metric"])

    def test_persistent_report_failure_keeps_valid_model_output_without_training_again(self):
        report = self.run_failure(persistent=True)
        self.assertTrue(report["output_valid"])
        self.assertTrue(report["recovery"]["report_write_failed"])
        self.assertEqual((self.root / "submission.csv").read_bytes(), self.accepted_bytes)
        self.assertEqual(len(self.fixture.calls), 1)

    def test_a_rewritten_checkpoint_pointer_cannot_select_a_different_csv(self):
        def replace_pointer(_):
            relative = "checkpoints/" + "f" * 32
            fake = self.output / relative
            fake.mkdir(parents=True)
            (fake / "submission.csv").write_text("id,y\n0001,999\nNA,999\n", encoding="utf-8")
            shutil.copyfile(self.accepted / "holdout_predictions.csv", fake / "holdout_predictions.csv")
            (self.output / "checkpoint.json").write_text(json.dumps({
                "version": 1, "directory": relative,
                "submission_sha256": automl._file_hash(fake / "submission.csv"),
                "holdout_sha256": automl._file_hash(fake / "holdout_predictions.csv"),
            }), encoding="utf-8")

        report = self.run_failure(mutate=replace_pointer)
        self.assertTrue(report["output_valid"])
        self.assertEqual((self.root / "submission.csv").read_bytes(), self.accepted_bytes)
        self.assertEqual(report["recovery"]["submission_sha256"], automl._file_hash(self.accepted / "submission.csv"))

    def test_tampered_accepted_csv_is_rejected_even_if_the_pointer_hash_is_rewritten(self):
        def change_checkpoint(_):
            (self.accepted / "submission.csv").write_text("id,y\n0001,999\nNA,999\n", encoding="utf-8")
            pointer = self.output / "checkpoint.json"
            value = json.loads(pointer.read_text(encoding="utf-8"))
            value["submission_sha256"] = automl._file_hash(self.accepted / "submission.csv")
            pointer.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
            self.run_failure(mutate=change_checkpoint)
        report = json.loads((self.output / "automl_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["output_valid"])
        self.assertEqual(report["recovery"]["status"], "rejected")

    def test_tampered_or_missing_accepted_holdout_is_fatal(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                # Each run needs a fresh output path while sharing source data.
                self.output = self.root / ".agent_runs" / ("holdout_missing" if missing else "holdout_changed")

                def mutate(_):
                    path = self.accepted / "holdout_predictions.csv"
                    if missing:
                        path.unlink()
                    else:
                        path.write_text("id,actual,prediction\nforged,0,0\n", encoding="utf-8")

                with self.assertRaises(ValueError):
                    self.run_failure(mutate=mutate)

    def test_source_or_snapshot_changes_cannot_be_hidden_by_recovery(self):
        for snapshot in (False, True):
            with self.subTest(snapshot=snapshot):
                self.output = self.root / ".agent_runs" / ("snapshot_changed" if snapshot else "source_changed")
                original = (self.fixture.inputs / "train.csv").read_bytes()

                def mutate(_):
                    if snapshot:
                        candidates = list(self.root.glob(".agent_runs/*/input_snapshot/train.csv"))
                        # Select this run's snapshot, rather than an older run.
                        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
                    else:
                        path = self.fixture.inputs / "train.csv"
                    path.write_bytes(path.read_bytes() + b"\n")

                try:
                    with self.assertRaisesRegex(ValueError, "изменились"):
                        self.run_failure(mutate=mutate)
                finally:
                    (self.fixture.inputs / "train.csv").write_bytes(original)

    def test_value_error_after_a_checkpoint_is_not_relabelled_as_an_operational_failure(self):
        with self.assertRaisesRegex(ValueError, "Injected invariant failure"):
            self.run_failure(exception=ValueError("Injected invariant failure"))
        report = json.loads((self.output / "automl_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["output_valid"])
        self.assertEqual(report["search"]["status"], "failed")

    def test_no_accepted_seed_preserves_the_previous_csv_and_still_raises(self):
        previous = b"id,y\n0001,90\nNA,90\n"
        (self.root / "submission.csv").write_bytes(previous)
        with self.assertRaisesRegex(RuntimeError, "ни одного"):
            self.run_failure(outcomes={"seed": {"exit_code": 1}})
        self.assertEqual((self.root / "submission.csv").read_bytes(), previous)

    def test_recovery_cannot_start_after_the_original_deadline(self):
        now = [None]
        real_clock = automl.time.monotonic

        def clock():
            return real_clock() if now[0] is None else now[0]

        def exhaust_budget(_):
            now[0] = real_clock() + 121

        with patch("agent.emergency._bounded_copy") as copy:
            with self.assertRaises(TimeoutError):
                self.run_failure(mutate=exhaust_budget, clock=clock)
        copy.assert_not_called()
        self.assertEqual(len(self.fixture.calls), 1)


if __name__ == "__main__":
    unittest.main()
