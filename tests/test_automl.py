"""Search must select by independently verified holdout evidence, not recency."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent import automl, file_io
from agent.splits import split_rows
from validation import validate_predictions


class AutoMLTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task.md"
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.make_data()

    def make_data(self, classification=False):
        self.classification = classification
        self.task.write_text("Primary metric: " + ("Macro F1" if classification else "RMSE"), encoding="utf-8")
        pd.DataFrame({"id": [f"row-{i:03d}" for i in range(80)], "x": list(range(80)),
                      "cat": ["NA" if i % 2 else "tcp" for i in range(80)],
                      "y": ["NA" if i % 2 else "classA" for i in range(80)] if classification
                      else [i * 0.5 for i in range(80)]}).to_csv(self.inputs / "train.csv", index=False)
        pd.DataFrame({"id": ["0001", "NA"], "x": [5, 9], "cat": ["new", ""]}).to_csv(
            self.inputs / "test.csv", index=False)
        pd.DataFrame({"id": ["0001", "NA"], "y": ["placeholder", "placeholder"]}).to_csv(
            self.inputs / "sample_submission.csv", index=False)
        pd.DataFrame({"feature": ["id", "x", "cat", "y"], "role": ["id", "feature", "feature", "target"],
                      "dtype": ["numeric", "numeric", "category", "category" if classification else "numeric"]}).to_csv(
            self.inputs / "data_dictionary.csv", index=False)

    def fake_execution(self, outcomes):
        outcomes = iter(outcomes)

        def execute(code, task_dir, run_dir, step, timeout):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
            train = pd.read_csv(task_dir / "inputs" / "train.csv", dtype=str, keep_default_na=False)
            held = train.iloc[config["validation_rows"]]
            if self.classification:
                predictions = held["y"].tolist()
                for i in range(int(len(predictions) * outcome.get("wrong", 0))):
                    predictions[i] = "NA" if predictions[i] == "classA" else "classA"
            else:
                predictions = pd.to_numeric(held["y"]).to_numpy() + outcome.get("error", 0)
            evidence = pd.DataFrame({"id": held["id"].to_numpy(), "actual": held["y"].to_numpy(),
                                     "prediction": predictions})
            if outcome.get("wrong_id"):
                evidence.loc[0, "id"] = "forged"
            evidence.to_csv(task_dir / "holdout_predictions.csv", index=False)
            pd.DataFrame({"id": ["0001", "NA"], "y": ["NA", "classA"] if self.classification else [step, step]}).to_csv(
                task_dir / "submission.csv", index=False)
            return {"exit_code": outcome.get("exit_code", 0), "timed_out": outcome.get("timeout", False),
                    "stdout": "metric claimed: 999999", "stderr": "failed child"}
        return execute

    def run_search(self, outcomes, **kwargs):
        with patch.object(automl, "run_python", side_effect=self.fake_execution(outcomes)), \
                contextlib.redirect_stdout(io.StringIO()):
            return automl.run_automl(self.task, time_budget=120, **kwargs)

    def test_metric_recomputed_worse_and_tied_candidates_do_not_replace_first_best(self):
        result = self.run_search([{"error": 1}, {"error": 5}, {"error": 1}, {"error": 3}])
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual(result["metric"]["value"], 1)
        self.assertEqual(result["search"]["completed"], 4)
        self.assertEqual(sum(item["selected"] for item in result["search"]["candidates"]), 1)
        self.assertEqual(pd.read_csv(self.root / "submission.csv")["y"].tolist(), [1, 1])
        self.assertEqual(result["api"], {"used": False, "calls": 0})
        self.assertTrue(validate_predictions(self.root)["valid"])

    def test_later_strict_improvement_wins_with_identical_split_comparison_identity(self):
        fingerprint = "a" * 64
        result = self.run_search([{"error": 4}, {"error": 2}, {"error": 3}, {"error": 1}],
                                 dataset_fingerprint=fingerprint, validation={"strategy": "random"})
        self.assertEqual(result["search"]["best_candidate"], "candidate_4")
        self.assertEqual(result["metric"]["value"], 1)
        expected = automl._json_hash({"dataset_fingerprint": fingerprint, "metric": "RMSE",
                                    "higher_is_better": False, "split": result["split"]})
        self.assertEqual(result["comparison_key"], expected)
        evidence = pd.read_csv(Path(result["report_path"]).parent / "holdout_predictions.csv",
                               dtype=str, keep_default_na=False)
        self.assertEqual(result["split"]["validation_hash"], automl._json_hash({
            "id_column": "id", "target_column": "y", "ids": evidence["id"].tolist(),
            "actuals": evidence["actual"].tolist(),
        }))

    def test_classification_maximizes_macro_f1_and_preserves_literal_na_label_and_ids(self):
        self.make_data(classification=True)
        result = self.run_search([{"wrong": 1}, {"wrong": 0}, {"wrong": .5}, {"wrong": 0}])
        self.assertEqual(result["metric"], {"name": "Macro F1", "value": 1.0, "higher_is_better": True})
        self.assertEqual(result["search"]["best_candidate"], "candidate_2")
        self.assertEqual(automl.read_csv(self.root / "submission.csv")["id"].tolist(), ["0001", "NA"])
        self.assertEqual(automl.read_csv(self.root / "submission.csv")["y"].tolist(), ["NA", "classA"])

    def test_failed_timeout_and_forged_holdout_cannot_replace_previous_winner(self):
        result = self.run_search([{"error": 2}, {"error": 0, "exit_code": 1},
                                 {"error": 0, "timeout": True}, {"error": 0, "wrong_id": True}])
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual(result["search"]["completed"], 1)
        self.assertEqual(result["search"]["status"], "time_limit")
        self.assertEqual([row["status"] for row in result["search"]["candidates"]],
                         ["completed", "failed", "timed_out", "failed"])
        self.assertEqual(pd.read_csv(self.root / "submission.csv")["y"].tolist(), [1, 1])

    def test_all_failures_preserve_preexisting_csv_and_are_not_reported_as_success(self):
        previous = b"id,y\n0001,100\nNA,200\n"
        (self.root / "submission.csv").write_bytes(previous)
        with self.assertRaisesRegex(RuntimeError, "ни одного"):
            self.run_search([{"exit_code": 1}] * 4)
        self.assertEqual((self.root / "submission.csv").read_bytes(), previous)
        report = json.loads(next(self.root.glob(".agent_runs/*/automl_report.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["search"]["status"], "failed")
        self.assertFalse(report["output_valid"])

    def test_staging_failure_is_an_attempt_failure_and_does_not_discard_winner(self):
        stage = automl._stage_inputs
        count = 0
        def stage_with_failure(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("copy unavailable")
            return stage(source, destination)
        with patch.object(automl, "_stage_inputs", side_effect=stage_with_failure):
            result = self.run_search([{"error": 1}, {"error": 3}, {"error": 4}])
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual(result["search"]["candidates"][1]["status"], "failed")
        self.assertTrue(result["output_valid"])

    def test_auto_detects_future_period_and_duplicate_groups_without_target_leakage(self):
        train = pd.DataFrame({"period": [1, 1, 2, 2, 3, 3], "x": [1, 1, 2, 2, 3, 3]})
        test = pd.DataFrame({"period": [4, 4], "x": [9, 10]})
        choice, reason = automl.choose_validation(train, test, ["period", "x"], ["period", "x"], False)
        self.assertEqual(choice, {"strategy": "time", "time_column": "period"})
        _, held, split = split_rows(train, pd.Series(range(6)), ["period", "x"], False, choice)
        self.assertEqual(held.tolist(), [4, 5])
        self.assertEqual(split["validation_period"], "3")
        choice, reason = automl.choose_validation(train, test, ["period", "x"], ["period", "x"], True)
        self.assertEqual(choice, {"strategy": "grouped"})
        manual, _ = automl.choose_validation(train, test, ["period", "x"], ["period", "x"], False,
                                             {"strategy": "random"})
        self.assertEqual(manual, {"strategy": "random"})

    def test_unknown_metric_and_tiny_budget_preserve_previous_output_without_child(self):
        (self.root / "submission.csv").write_text("previous", encoding="utf-8")
        self.task.write_text("Primary metric: MAE", encoding="utf-8")
        with patch.object(automl, "run_python") as child:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                automl.run_automl(self.task)
            self.task.write_text("Metric: RMSE", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "ни одного"):
                automl.run_automl(self.task, time_budget=.001)
            child.assert_not_called()
        self.assertEqual((self.root / "submission.csv").read_text(), "previous")

    def test_real_subprocess_recipe_refits_and_outputs_recomputed_evidence_without_api(self):
        plan = automl.candidate_plan(False)[:1]
        plan[0]["model_config"]["parameters"].update(max_iter=50, min_samples_leaf=5)
        with patch.object(automl, "candidate_plan", return_value=plan), contextlib.redirect_stdout(io.StringIO()):
            result = automl.run_automl(self.task, time_budget=60)
        self.assertTrue(result["output_valid"])
        self.assertEqual(result["search"]["completed"], 1)
        self.assertTrue(validate_predictions(self.root)["valid"])
        evidence = automl.read_csv(Path(result["report_path"]).parent / "holdout_predictions.csv")
        expected = float(((pd.to_numeric(evidence["actual"]) - pd.to_numeric(evidence["prediction"])) ** 2).mean() ** .5)
        self.assertAlmostEqual(result["metric"]["value"], expected, places=12)

    def test_original_feature_mutation_cannot_change_candidate_inputs_or_publish_stale_result(self):
        previous = b"id,y\n0001,100\nNA,200\n"
        (self.root / "submission.csv").write_bytes(previous)
        execute = self.fake_execution([{"error": 1}] * 4)
        seen = []
        def changing_original(code, task_dir, run_dir, step, timeout):
            source = pd.read_csv(self.inputs / "train.csv", dtype=str, keep_default_na=False)
            source.loc[0, "x"] = "99999"
            source.to_csv(self.inputs / "train.csv", index=False)
            seen.append(pd.read_csv(task_dir / "inputs" / "train.csv").iloc[0]["x"])
            return execute(code, task_dir, run_dir, step, timeout)
        with patch.object(automl, "run_python", side_effect=changing_original), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "данные изменились"):
                automl.run_automl(self.task, time_budget=120)
        self.assertEqual(seen, [0, 0, 0, 0])
        self.assertEqual((self.root / "submission.csv").read_bytes(), previous)

    def test_candidate_timeout_is_recomputed_after_input_staging(self):
        clock = [0.0]
        stage = automl._stage_inputs
        calls = []
        execute = self.fake_execution([{"error": 1}] * 4)
        def slow_first_stage(source, destination):
            result = stage(source, destination)
            clock[0] = 70.0
            return result
        def capture_timeout(code, task_dir, run_dir, step, timeout):
            calls.append(timeout)
            return execute(code, task_dir, run_dir, step, timeout)
        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "_stage_inputs", side_effect=slow_first_stage), \
                patch.object(automl, "run_python", side_effect=capture_timeout), \
                contextlib.redirect_stdout(io.StringIO()):
            result = automl.run_automl(self.task, time_budget=120)
        self.assertEqual(calls[0], 12.0)
        self.assertTrue(result["output_valid"])

    def test_budget_exhausted_during_first_staging_reports_original_no_result_error(self):
        clock = [0.0]
        stage = automl._stage_inputs
        def exhausted_stage(source, destination):
            stage(source, destination)
            clock[0] = 120.0
        def strict_progress(event):
            search = event["search"]
            if search["status"] in {"completed", "time_limit"}:
                self.assertIsNotNone(search["best_candidate"])
        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "_stage_inputs", side_effect=exhausted_stage), \
                patch.object(automl, "run_python") as child, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "ни одного"):
                automl.run_automl(self.task, time_budget=120, progress=strict_progress)
            child.assert_not_called()

    def test_core_atomic_copy_retries_windows_lock_and_preserves_previous_file_on_exhaustion(self):
        source, destination = self.root / "source.csv", self.root / "destination.csv"
        source.write_bytes(b"new")
        destination.write_bytes(b"old")
        error = PermissionError("temporarily locked")
        error.winerror = 32
        replace = file_io.os.replace
        calls = 0
        def temporarily_locked(src, dst):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return replace(src, dst)
        with patch.object(file_io, "WINDOWS", True), patch.object(file_io.os, "replace", side_effect=temporarily_locked), \
                patch.object(file_io.time, "sleep"):
            automl._atomic_copy(source, destination)
        self.assertEqual(destination.read_bytes(), b"new")
        destination.write_bytes(b"old")
        with patch.object(file_io, "WINDOWS", True), patch.object(file_io.os, "replace", side_effect=error), \
                patch.object(file_io.time, "sleep"):
            with self.assertRaises(PermissionError):
                automl._atomic_copy(source, destination)
        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob(".destination.csv.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
