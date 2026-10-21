"""Nested search, independent evidence and surviving published checkpoints."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import automl, file_io
from agent.search import metric_predictions
from agent.splits import split_rows
from agent.task_spec import TaskSpec
from app.search_summary import validate_search_summary
from validation import validate_predictions


class AutoMLTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task.md"
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.calls = []
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

    def fake_execution(self, outcomes=None, callback=None):
        outcomes = outcomes or {}

        def execute(code, task_dir, run_dir, step, timeout):
            config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
            self.calls.append({**config, "timeout": timeout})
            if callback:
                callback(config)
            outcome = outcomes.get((config["id"], config["phase"]), outcomes.get(config["phase"], {}))
            if isinstance(outcome, Exception):
                raise outcome
            spec = TaskSpec.from_dict(config["task_spec"])
            train = automl.read_csv(task_dir / "inputs" / "train.csv")
            held = train.iloc[config["validation_rows"]]
            if config["phase"] != "refit" and not outcome.get("no_evidence"):
                if self.classification:
                    labels = held["y"].tolist()
                    for index in range(int(len(labels) * outcome.get("wrong", 0))):
                        labels[index] = "NA" if labels[index] == "classA" else "classA"
                    raw = np.asarray([[float(label == cls) for cls in spec.classes] for label in labels])
                else:
                    raw = pd.to_numeric(held["y"]).to_numpy() + outcome.get("error", 5)
                ids, actual = held["id"].tolist(), held["y"].tolist()
                frame = spec.holdout_frame(ids, actual, metric_predictions(spec, raw), prediction_classes=spec.classes)
                if outcome.get("wrong_id"):
                    frame.loc[0, "id"] = "forged"
                frame.to_csv(task_dir / "holdout_predictions.csv", index=False)
                bundle_ids = ["forged", *ids[1:]] if outcome.get("wrong_raw_id") else ids
                if outcome.get("wrong_raw"):
                    raw = raw + 1
                np.savez_compressed(task_dir / "holdout_raw.npz", ids=np.asarray(bundle_ids, dtype=str),
                                    actual=np.asarray(actual, dtype=str), predictions=raw,
                                    classes=np.asarray(spec.classes, dtype=str))
            if config["phase"] in {"seed", "refit"} and not outcome.get("no_submission"):
                predictions = ["NA", "classA"] if self.classification else [step, step]
                spec.prediction_frame(["0001", "NA"], predictions).to_csv(task_dir / "submission.csv", index=False)
            result = {"exit_code": outcome.get("exit_code", 0), "timed_out": outcome.get("timeout", False),
                      "memory_limit_exceeded": outcome.get("oom", False),
                      "stdout": "metric claimed: -999999", "stderr": "failed child",
                      "resources": {"available": True, "unavailable_reason": None,
                                    "peak_rss_bytes": 1000, "cpu_seconds": 0.1, "memory_limit_bytes": 2000}}
            if outcome.get("cleanup_error"):
                result["error"] = "Owned descendants could not be terminated"
            if "resources" in outcome:
                result["resources"] = outcome["resources"]
            return result
        return execute

    def run_search(self, outcomes=None, *, full_plan=False, count=2, callback=None, **kwargs):
        plan = automl.candidate_plan(self.classification)[:count]
        patches = contextlib.ExitStack()
        with patches, contextlib.redirect_stdout(io.StringIO()):
            patches.enter_context(patch.object(automl, "run_python", side_effect=self.fake_execution(outcomes, callback)))
            if not full_plan:
                patches.enter_context(patch.object(automl, "candidate_plan", return_value=plan))
            result = automl.run_automl(self.task, time_budget=120, **kwargs)
        validate_search_summary(result["search"], metric=result["metric"], model=result["model"])
        return result

    def test_exploration_adapts_families_using_inner_only_and_publishes_seed_first(self):
        def observe(config):
            if config["phase"] == "selection":
                self.assertTrue(validate_predictions(self.root)["valid"], "seed must already be published")
        outcomes = {"selection": {"error": 10}, ("candidate_4", "selection"): {"error": 1},
                    ("candidate_6", "selection"): {"error": 2}}
        result = self.run_search(outcomes, full_plan=True, callback=observe)
        selections = [call for call in self.calls if call["phase"] == "selection"]
        self.assertEqual(len(selections), 14)
        self.assertEqual({call["action"] for call in selections}, {"baseline", "extra_trees", "catboost", "lightgbm", "linear"})
        refined = [call for call in selections if call["id"].startswith("refine_")]
        self.assertEqual({call["action"] for call in refined}, {"catboost", "lightgbm"})
        outer = next(call for call in self.calls if call["phase"] == "seed")
        for call in selections:
            self.assertFalse(set(call["fit_rows"]) & set(call["validation_rows"]))
            self.assertTrue(set(call["fit_rows"]) | set(call["validation_rows"]) <= set(outer["fit_rows"]))
            self.assertFalse(set(call["validation_rows"]) & set(outer["validation_rows"]))
            self.assertFalse((Path(call["candidate_dir"]) / "submission.csv").exists())
        self.assertLessEqual(sum(call["phase"] == "verification" for call in self.calls), 3)
        self.assertFalse(any(call["phase"] == "refit" for call in self.calls), "ties retain the seed")
        self.assertEqual(result["search"]["selection_completed"], 14)
        self.assertEqual(result["search"]["best_candidate"], "seed")

    def test_metric_is_recomputed_outer_selection_is_not_reused_and_only_winner_refits(self):
        result = self.run_search({("candidate_1", "selection"): {"error": 0},
                                  ("candidate_2", "selection"): {"error": 1},
                                  ("candidate_1", "verification"): {"error": 2},
                                  ("candidate_2", "verification"): {"error": 4}})
        self.assertEqual(result["metric"]["value"], 2)
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual([(call["id"], call["phase"]) for call in self.calls if call["phase"] == "refit"],
                         [("candidate_1", "refit")])
        rows = {row["id"]: row for row in result["search"]["candidates"]}
        self.assertEqual(rows["candidate_1"]["selection_metric"], 0)
        self.assertEqual(rows["candidate_1"]["metric"], rows["candidate_1"]["verification_metric"])
        self.assertTrue(validate_predictions(self.root)["valid"])
        self.assertEqual(result["api"], {"used": False, "calls": 0})

    def test_outer_verification_uses_three_distinct_inner_ranked_families(self):
        result = self.run_search({"selection": {"error": 10},
                                  ("candidate_6", "selection"): {"error": 0.1},
                                  ("candidate_7", "selection"): {"error": 0.2},
                                  ("refine_1", "selection"): {"error": 0.05},
                                  ("candidate_2", "selection"): {"error": 1},
                                  ("candidate_4", "selection"): {"error": 2}}, full_plan=True)
        verified = [call for call in self.calls if call["phase"] == "verification"]
        self.assertCountEqual([call["action"] for call in verified], ["lightgbm", "extra_trees", "catboost"])
        self.assertCountEqual([call["id"] for call in verified], ["refine_1", "candidate_2", "candidate_4"])
        self.assertEqual(result["search_policy"]["outer_shortlist_limit"], 3)

    def test_paired_stability_can_reject_one_split_winner_without_using_outer_rows(self):
        outcomes = {"selection": {"error": 2}}

        def vary_by_repeat(config):
            if config["phase"] == "selection":
                repeat = Path(config["candidate_dir"]).name.startswith("stability_")
                outcomes[(config["id"], "selection")] = {"error": (
                    10 if repeat and config["id"] == "candidate_1" else
                    0 if config["id"] == "candidate_1" else 2 if config["id"] == "candidate_2" else 3)}

        result = self.run_search(outcomes, count=3, callback=vary_by_repeat, stability_splits=3)
        stable = result["stability"]
        self.assertTrue(stable["applied"])
        self.assertEqual(stable["summary"]["used_rounds"], 3)
        self.assertEqual(stable["summary"]["ranking"][0], "candidate_2")
        self.assertNotIn("candidate_1", [call["id"] for call in self.calls if call["phase"] == "verification"])
        outer = next(call for call in self.calls if call["phase"] == "seed")
        for call in self.calls:
            if Path(call["candidate_dir"]).name.startswith("stability_"):
                self.assertTrue(set(call["fit_rows"]) | set(call["validation_rows"]) <= set(outer["fit_rows"]))
                self.assertFalse(set(call["validation_rows"]) & set(outer["validation_rows"]))

    def test_incomplete_paired_round_keeps_original_outer_shortlist(self):
        outcomes = {"selection": {"error": 2}}

        def fail_repeat(config):
            if Path(config["candidate_dir"]).name.startswith("stability_"):
                outcomes[(config["id"], "selection")] = {"exit_code": 1}

        result = self.run_search(outcomes, count=3, callback=fail_repeat, stability_splits=3)
        self.assertFalse(result["stability"]["applied"])
        self.assertEqual(result["stability"]["summary"]["used_rounds"], 1)
        self.assertEqual(sum(call["phase"] == "verification" for call in self.calls), 3)

    def test_verification_equal_inner_durations_preserve_score_rank(self):
        with patch.object(automl.time, "monotonic", return_value=0):
            self.run_search({("candidate_1", "selection"): {"error": 2},
                             ("candidate_2", "selection"): {"error": 1}})
        self.assertEqual([call["id"] for call in self.calls if call["phase"] == "verification"],
                         ["candidate_2", "candidate_1"])

    def test_fast_verification_releases_budget_to_stronger_slow_finalist(self):
        self.make_data(classification=True)
        plan = automl.candidate_plan(True)
        # Inner rank prefers LGBM then CatBoost then RF; measured costs differ.
        plan = [plan[5], plan[3], plan[0]]
        clock, verification_started, schedule = [0.0], [False], []
        costs = {"selection": {"lightgbm": 30, "catboost": 55, "baseline": 5},
                 "verification": {"lightgbm": 38, "catboost": 68, "baseline": 7}}
        execute = self.fake_execution({
            "seed": {"wrong": 1},
            ("candidate_6", "selection"): {"wrong": 0},
            ("candidate_4", "selection"): {"wrong": 0.25},
            ("candidate_1", "selection"): {"wrong": 0.5},
            ("candidate_6", "verification"): {"wrong": 0},
            ("candidate_4", "verification"): {"wrong": 0},
            ("candidate_1", "verification"): {"wrong": 0.5},
        })

        def progress(event):
            # Charge earlier preparation to the shared budget while preserving
            # measured 5/30/55-second INNER worker costs. Exactly 95s remain in
            # the verification phase; the final 150s are still reserved.
            if event["search"]["phase"] == "verification" and not verification_started[0]:
                clock[0] = 355
                verification_started[0] = True

        def measured_execution(code, task_dir, run_dir, step, timeout):
            config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
            phase, action = config["phase"], config["action"]
            cost = costs.get(phase, {}).get(action, 1 if phase == "seed" else 20)
            result = execute(code, task_dir, run_dir, step, timeout)
            schedule.append((phase, action, clock[0], timeout))
            clock[0] += min(cost, timeout)
            if cost > timeout:
                result.update(exit_code=-9, timed_out=True)
            return result

        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "candidate_plan", return_value=plan), \
                patch.object(automl, "run_python", side_effect=measured_execution), \
                contextlib.redirect_stdout(io.StringIO()):
            result = automl.run_automl(self.task, time_budget=600, progress=progress)
        verification = [value for value in schedule if value[0] == "verification"]
        self.assertEqual([value[1] for value in verification], ["baseline", "lightgbm", "catboost"])
        self.assertEqual([value[2] for value in verification], [355, 362, 400])
        self.assertAlmostEqual(verification[0][3], 95 / 3)
        self.assertEqual(verification[1][3], 44)
        self.assertEqual(verification[2][3], 50)
        rows = {row["id"]: row for row in result["search"]["candidates"]}
        self.assertEqual(rows["candidate_1"]["status"], "completed")
        self.assertEqual(rows["candidate_6"]["status"], "completed")
        self.assertEqual(rows["candidate_4"]["status"], "timed_out")
        self.assertIsNone(rows["candidate_4"]["metric"], "timed-out evidence cannot compete")
        self.assertEqual(result["search"]["best_candidate"], "candidate_6")
        self.assertEqual(result["metric"]["value"], 1)
        refit = next(value for value in schedule if value[0] == "refit")
        self.assertEqual(refit[1:3], ("lightgbm", 450))
        self.assertEqual(refit[3], 148)  # 25% reserve minus the two-second I/O margin.
        self.assertLessEqual(clock[0], 600)
        validate_search_summary(result["search"], metric=result["metric"], model=result["model"])

    def test_single_fixed_blend_requires_strict_outer_improvement(self):
        result = self.run_search({("candidate_1", "verification"): {"error": 2},
                                  ("candidate_2", "verification"): {"error": -2}})
        self.assertEqual(result["model"], "FixedBlend")
        self.assertEqual(result["metric"]["value"], 0)
        self.assertEqual(result["model_config"]["parameters"]["weight"], 0.5)
        self.assertEqual(len(result["model_config"]["parameters"]["components"]), 2)
        self.assertEqual(len(result["component_feature_configs"]), 2)
        refits = [call for call in self.calls if call["phase"] == "refit"]
        self.assertEqual([call["id"] for call in refits], ["blend"])
        self.assertEqual(len(refits[0]["components"]), 2)

    def test_failed_refit_preserves_seed_bytes_and_can_try_next_verified_candidate(self):
        seed_bytes = []
        def observe(config):
            if config["phase"] == "selection" and not seed_bytes:
                seed_bytes.append((self.root / "submission.csv").read_bytes())
        result = self.run_search({("candidate_1", "verification"): {"error": 1},
                                  ("candidate_2", "verification"): {"error": 2},
                                  "refit": {"exit_code": 1}}, callback=observe)
        self.assertEqual(result["search"]["best_candidate"], "seed")
        self.assertEqual((self.root / "submission.csv").read_bytes(), seed_bytes[0])
        self.assertEqual([call["id"] for call in self.calls if call["phase"] == "refit"], ["candidate_1", "candidate_2"])
        self.assertEqual({row["rejection_reason"] for row in result["search"]["candidates"] if row["phase"] == "refit"}, {"refit_failed"})

    def test_failed_best_refit_can_publish_second_candidate(self):
        result = self.run_search({("candidate_1", "verification"): {"error": 1},
                                  ("candidate_2", "verification"): {"error": 2},
                                  ("candidate_1", "refit"): {"exit_code": 1}})
        self.assertEqual(result["search"]["best_candidate"], "candidate_2")
        self.assertEqual(result["metric"]["value"], 2)

    def test_failed_timeout_forged_and_stale_evidence_cannot_replace_seed(self):
        for fault in ({"exit_code": 1}, {"timeout": True}, {"wrong_id": True}, {"wrong_raw_id": True},
                      {"wrong_raw": True}, {"no_evidence": True}, {"oom": True}, {"cleanup_error": True}):
            with self.subTest(fault=fault):
                result = self.run_search({"selection": {"error": 0, **fault}})
                self.assertEqual(result["search"]["best_candidate"], "seed")
                self.assertEqual(result["search"]["selection_completed"], 0)
                self.assertTrue(result["output_valid"])

    def test_classification_maximizes_macro_f1_preserving_literal_na_labels_and_ids(self):
        self.make_data(True)
        result = self.run_search({"seed": {"wrong": .5}, ("candidate_1", "verification"): {"wrong": 0},
                                  ("candidate_2", "verification"): {"wrong": .5}})
        self.assertEqual(result["metric"], {"name": "Macro F1", "value": 1.0, "higher_is_better": True})
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual(automl.read_csv(self.root / "submission.csv")["id"].tolist(), ["0001", "NA"])
        diagnostics = result["classification_diagnostics"]
        self.assertEqual(diagnostics["labels"], ["NA", "classA"])
        self.assertEqual(diagnostics["macro_f1"], result["metric"]["value"])
        self.assertEqual(sum(map(sum, diagnostics["confusion_matrix"])), result["split"]["validation_rows"])

    def test_resource_gap_in_earlier_phase_is_not_relabelled_as_complete_telemetry(self):
        result = self.run_search({("candidate_1", "selection"): {"resources": {
            "available": False, "unavailable_reason": "Permission denied"}},
            ("candidate_1", "verification"): {"error": 1}})
        row = next(row for row in result["search"]["candidates"] if row["selected"])
        self.assertFalse(row["resources"]["available"])
        self.assertEqual(row["resources"]["unavailable_reason"], "Permission denied")
        self.assertIsNone(row["resources"]["cpu_seconds"])

    def test_metric_direction_r2_is_not_inferred_from_classification_flag(self):
        self.task.write_text("Metric: R2", encoding="utf-8")
        result = self.run_search({("candidate_1", "verification"): {"error": 0}})
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")
        self.assertEqual(result["metric"], {"name": "R2", "value": 1.0, "higher_is_better": True})

    def test_outer_comparison_identity_stays_stable_despite_inner_split(self):
        fingerprint = "a" * 64
        result = self.run_search(dataset_fingerprint=fingerprint, validation={"strategy": "random"})
        self.assertEqual(result["comparison_key"], automl._json_hash({
            "dataset_fingerprint": fingerprint, "metric": "RMSE", "higher_is_better": False, "split": result["split"]}))
        held = automl.read_csv(Path(result["report_path"]).parent / "holdout_predictions.csv")
        self.assertEqual(result["split"]["validation_hash"], automl._json_hash({
            "id_column": "id", "target_column": "y", "ids": held["id"].tolist(), "actuals": held["actual"].tolist()}))
        self.assertNotEqual(result["inner_split"]["validation_hash"], result["split"]["validation_hash"])

    def test_unavailable_inner_split_runs_only_untuned_seed(self):
        train = automl.read_csv(self.inputs / "train.csv")
        train["x"] = [1] * 40 + [2] * 40
        train.to_csv(self.inputs / "train.csv", index=False)
        result = self.run_search(validation={"strategy": "time", "time_column": "x"})
        self.assertEqual([call["phase"] for call in self.calls], ["seed"])
        self.assertIsNotNone(result["inner_unavailable"])
        self.assertIsNone(result["inner_split"])
        self.assertEqual(result["search"]["planned"], 1)

    def test_all_failures_preserve_previous_csv_and_report_failure(self):
        previous = b"id,y\n0001,100\nNA,200\n"
        (self.root / "submission.csv").write_bytes(previous)
        with self.assertRaisesRegex(RuntimeError, "ни одного"):
            self.run_search({"seed": {"exit_code": 1}, "selection": {"exit_code": 1}})
        self.assertEqual((self.root / "submission.csv").read_bytes(), previous)
        report = json.loads(next(self.root.glob(".agent_runs/*/automl_report.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["search"]["status"], "failed")
        self.assertFalse(report["output_valid"])

    def test_source_mutation_before_seed_publication_is_fatal_and_snapshot_stays_frozen(self):
        previous = b"id,y\n0001,100\nNA,200\n"
        (self.root / "submission.csv").write_bytes(previous)
        def changing_original(config):
            source = automl.read_csv(self.inputs / "train.csv")
            source.loc[0, "x"] = "99999"
            source.to_csv(self.inputs / "train.csv", index=False)
            self.assertEqual(automl.read_csv(Path(config["candidate_dir"]) / "inputs" / "train.csv").iloc[0]["x"], "0")
        with self.assertRaisesRegex(ValueError, "данные изменились"):
            self.run_search(callback=changing_original)
        self.assertEqual((self.root / "submission.csv").read_bytes(), previous)

    def test_source_mutation_after_seed_is_fatal_even_when_all_following_trials_fail(self):
        def change_after_seed(config):
            if config["phase"] == "selection":
                (self.inputs / "train.csv").write_bytes((self.inputs / "train.csv").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "данные изменились"):
            self.run_search({"selection": {"exit_code": 1}}, callback=change_after_seed)

    def test_accidental_worker_input_mutation_is_detected_even_if_original_is_unchanged(self):
        original = (self.inputs / "train.csv").read_bytes()
        def mutate_snapshot(config):
            candidate_input = Path(config["candidate_dir"]) / "inputs" / "train.csv"
            frame = automl.read_csv(candidate_input)
            frame.loc[0, "x"] = "99999"
            candidate_input.write_text(frame.to_csv(index=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "данные изменились"):
            self.run_search(callback=mutate_snapshot)
        self.assertEqual((self.inputs / "train.csv").read_bytes(), original)

    def test_budget_is_recomputed_after_staging_and_reserves_final_quarter(self):
        clock, stage = [0.0], automl._stage_inputs
        def slow_stage(source, destination):
            stage(source, destination)
            clock[0] += 5
        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "_stage_inputs", side_effect=slow_stage):
            result = self.run_search({("candidate_1", "verification"): {"error": 1}})
        self.assertEqual(self.calls[0]["timeout"], 19)
        for call in self.calls:
            if call["phase"] == "selection":
                self.assertLessEqual(call["timeout"], 72)
            if call["phase"] == "verification":
                self.assertLessEqual(call["timeout"], 90)
        self.assertTrue(result["output_valid"])

    def test_budget_exhausted_during_staging_does_not_launch_child(self):
        clock, stage = [0.0], automl._stage_inputs
        def exhausted_stage(source, destination):
            stage(source, destination)
            clock[0] = 120.0
        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "_stage_inputs", side_effect=exhausted_stage):
            with self.assertRaisesRegex(RuntimeError, "ни одного"):
                self.run_search()
        self.assertEqual(self.calls, [])

    def test_inner_and_outer_trials_cannot_consume_final_refit_quarter(self):
        clock = [0.0]
        execute = self.fake_execution({("candidate_1", "verification"): {"error": 1}})
        seen = []
        def consume_share(code, task_dir, run_dir, step, timeout):
            config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
            seen.append((config["phase"], clock[0], timeout))
            result = execute(code, task_dir, run_dir, step, timeout)
            clock[0] += 1 if config["phase"] == "seed" else timeout
            return result
        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(automl, "candidate_plan", return_value=automl.candidate_plan(False)[:2]), \
                patch.object(automl, "run_python", side_effect=consume_share), contextlib.redirect_stdout(io.StringIO()):
            result = automl.run_automl(self.task, time_budget=120)
        refit = next(value for value in seen if value[0] == "refit")
        self.assertEqual(refit[1], 90)
        self.assertEqual(refit[2], 28)
        self.assertLessEqual(clock[0], 120)
        self.assertEqual(result["search"]["best_candidate"], "candidate_1")

    def test_unknown_metric_and_tiny_budget_do_not_launch_child(self):
        self.task.write_text("Metric: SMAPE", encoding="utf-8")
        with patch.object(automl, "run_python") as child:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                automl.run_automl(self.task)
            self.task.write_text("Metric: RMSE", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "ни одного"):
                automl.run_automl(self.task, time_budget=.001)
            child.assert_not_called()

    def test_progress_summary_stays_app_compatible_through_all_phases(self):
        def progress(event):
            validate_search_summary(event["search"], allow_running=True)
        self.run_search({("candidate_1", "verification"): {"error": -2},
                         ("candidate_2", "verification"): {"error": 2}}, progress=progress)

    def test_auto_detects_period_and_duplicate_groups_without_target_leakage(self):
        train = pd.DataFrame({"period": [1, 1, 2, 2, 3, 3], "x": [1, 1, 2, 2, 3, 3]})
        test = pd.DataFrame({"period": [4, 4], "x": [9, 10]})
        choice, _ = automl.choose_validation(train, test, ["period", "x"], ["period", "x"], False)
        self.assertEqual(choice, {"strategy": "time", "time_column": "period"})
        _, held, _ = split_rows(train, pd.Series(range(6)), ["period", "x"], False, choice)
        self.assertEqual(held.tolist(), [4, 5])
        choice, _ = automl.choose_validation(train, test, ["period", "x"], ["period", "x"], True)
        self.assertEqual(choice, {"strategy": "grouped"})

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

    def test_checkpoint_pair_commits_before_loose_copy_and_detects_tampering(self):
        result = self.run_search({("candidate_1", "verification"): {"error": 1}})
        directory = Path(result["report_path"]).parent
        accepted = automl.read_accepted_checkpoint(directory)
        self.assertEqual((accepted / "submission.csv").read_bytes(), (self.root / "submission.csv").read_bytes())
        self.assertEqual((accepted / "holdout_predictions.csv").read_bytes(), (directory / "holdout_predictions.csv").read_bytes())
        (accepted / "submission.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            automl.read_accepted_checkpoint(directory)

    def test_interruption_before_new_pointer_keeps_previous_complete_pair(self):
        output = self.root / "checkpoints-test"
        output.mkdir()
        submission, held = self.root / "new.csv", self.root / "held.csv"
        submission.write_bytes(b"old predictions")
        held.write_bytes(b"old evidence")
        previous = automl._commit_checkpoint(submission, held, output)
        submission.write_bytes(b"new predictions")
        held.write_bytes(b"new evidence")
        with patch.object(automl, "_atomic_json", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                automl._commit_checkpoint(submission, held, output)
        self.assertEqual(automl.read_accepted_checkpoint(output), previous)
        self.assertEqual((previous / "submission.csv").read_bytes(), b"old predictions")
        self.assertEqual((previous / "holdout_predictions.csv").read_bytes(), b"old evidence")


if __name__ == "__main__":
    unittest.main()
