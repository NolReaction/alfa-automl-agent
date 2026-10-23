"""Real offline worker checks on small isolated regression/classification jobs."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from app import worker
from app.model_config import make_model_config
from app.model_artifacts import artifact_download_path


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_source(self, classification=False):
        source = self.root / ("classification" if classification else "regression")
        source.mkdir()
        (source / "task.md").write_text("Primary metric: " + ("Macro F1" if classification else "RMSE"), encoding="utf-8")
        inputs = source / "nested" / "participants"
        write_csv(inputs / "data_dictionary.csv", [
            ["feature", "role", "dtype"],
            ["row_key", "id", "category"],
            ["x", "feature", "numeric"],
            ["flag", "feature", "binary"],
            ["protocol", "feature", "category"],
            ["answer", "target", "category" if classification else "numeric"],
        ])
        rows = [["row_key", "x", "flag", "protocol", "answer"]]
        for index in range(80):
            row_id = "NA" if index == 0 else f"{index:04d}"
            target = ("NA" if index % 2 == 0 else "Positive") if classification else index * 1.5
            rows.append([row_id, index, index % 2, "tcp", target])
        write_csv(inputs / "train.csv", rows)
        write_csv(inputs / "test.csv", [
            ["row_key", "x", "flag", "protocol"],
            ["NA", 2, 0, "unseen"], ["0001", 3, 1, ""],
        ])
        write_csv(inputs / "sample_submission.csv", [
            ["row_key", "answer"], ["NA", "0"], ["0001", "0"],
        ])
        return source

    def make_config(self, source, action, name=None):
        job_dir = self.root / (name or (source.name + "-" + action))
        job_dir.mkdir()
        staged = job_dir / "staged"
        shutil.copytree(source, staged)
        config = {
            "job_dir": str(job_dir), "task_path": str(staged / "task.md"),
            "action": action, "dataset_fingerprint": "a" * 64,
        }
        path = job_dir / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path, job_dir, staged

    def test_recipes_use_same_holdout_and_publish_real_metrics_without_touching_source(self):
        for classification in (False, True):
            with self.subTest(classification=classification):
                source = self.make_source(classification)
                original = {str(path.relative_to(source)): path.read_bytes()
                            for path in source.rglob("*") if path.is_file()}
                results, full_metrics = [], []
                for action in ("baseline", "extra_trees"):
                    config_path, job_dir, staged = self.make_config(source, action)
                    transitions = []
                    save_json = worker._atomic_json

                    def record_json(path, value):
                        if path.name == "stage.json":
                            transitions.append(value["stage"])
                        return save_json(path, value)

                    with patch.object(worker, "_atomic_json", side_effect=record_json):
                        with contextlib.redirect_stdout(io.StringIO()):
                            exit_code = worker.main(["--config", str(config_path)])
                    self.assertEqual(exit_code, 0)
                    self.assertEqual(transitions, ["training", "validating", "refitting", "saving_model", "finished"])
                    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
                    metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
                    results.append(result)
                    full_metrics.append(metrics)
                    holdout = pd.read_csv(job_dir / "holdout_predictions.csv", dtype=str,
                                          keep_default_na=False)
                    self.assertEqual(list(holdout.columns), ["id", "actual", "prediction"])
                    self.assertEqual(len(holdout), 16)
                    source_train = pd.read_csv(source / "nested/participants/train.csv",
                                               dtype=str, keep_default_na=False)
                    _, expected_rows = train_test_split(
                        np.arange(len(source_train)), test_size=0.2, random_state=42,
                        shuffle=True, stratify=source_train["answer"] if classification else None,
                    )
                    self.assertEqual(holdout["id"].tolist(),
                                     source_train["row_key"].iloc[expected_rows].tolist())
                    self.assertEqual(holdout["actual"].tolist(),
                                     source_train["answer"].iloc[expected_rows].tolist())
                    if classification:
                        measured = f1_score(holdout["actual"], holdout["prediction"], average="macro",
                                            zero_division=0)
                        self.assertIn("NA", set(holdout["actual"]))
                        self.assertEqual(result["metric"]["name"], "Macro F1")
                        diagnostics = result["classification_diagnostics"]
                        self.assertAlmostEqual(diagnostics["macro_f1"], measured)
                        self.assertEqual(diagnostics["rows"], len(holdout))
                        self.assertIn("NA", diagnostics["labels"])
                        self.assertEqual(metrics["classification_diagnostics"], diagnostics)
                    else:
                        measured = np.sqrt(mean_squared_error(holdout["actual"].astype(float),
                                                              holdout["prediction"].astype(float)))
                        self.assertEqual(result["metric"]["name"], "RMSE")
                        self.assertNotIn("classification_diagnostics", result)
                        self.assertNotIn("classification_diagnostics", metrics)
                    self.assertAlmostEqual(result["metric"]["value"], measured)
                    self.assertTrue(result["output_valid"])
                    self.assertEqual(result["split"]["seed"], 42)
                    self.assertEqual(metrics["evaluation_source"], "app_local_holdout")
                    self.assertEqual(metrics["api"], {"used": False, "calls": 0})
                    expected_config = make_model_config(action, "classification" if classification else "regression")
                    self.assertEqual(result["model_config"], expected_config)
                    self.assertEqual(metrics["model_config"], expected_config)
                    self.assertIn("flag", metrics["features"]["numeric"])
                    self.assertGreater(metrics["elapsed_seconds"], 0)
                    submission = pd.read_csv(staged / "submission.csv", dtype=str,
                                             keep_default_na=False)
                    self.assertEqual(submission["row_key"].tolist(), ["NA", "0001"])
                    if classification:
                        self.assertEqual(submission["answer"].tolist(), ["NA", "Positive"])
                    if action == "extra_trees":
                        self.assertEqual(metrics["model"]["parameters"]["n_estimators"], 160)
                        self.assertEqual(metrics["model"]["parameters"]["n_jobs"], 8)
                self.assertEqual(results[0]["split"]["validation_hash"],
                                 results[1]["split"]["validation_hash"])
                self.assertEqual(results[0]["comparison_key"], results[1]["comparison_key"])
                self.assertEqual(full_metrics[0]["validation_signature"],
                                 full_metrics[1]["validation_signature"])
                current = {str(path.relative_to(source)): path.read_bytes()
                           for path in source.rglob("*") if path.is_file()}
                self.assertEqual(current, original)

    def test_registry_metrics_use_correct_predictions_and_direction(self):
        from agent.task_spec import load_task, score_holdout
        for metric_name, classification in (("MAE", False), ("MSE", False), ("R2", False),
                                            ("Weighted F1", True), ("Accuracy", True),
                                            ("LogLoss", True), ("ROC AUC", True)):
            with self.subTest(metric=metric_name):
                source = self.make_source(classification)
                (source / "task.md").write_text("Primary metric: " + metric_name + ("\nPositive class: Positive\n" if classification else "\n"), encoding="utf-8")
                path, job_dir, staged = self.make_config(source, "linear", metric_name.replace(" ", "-"))
                config = worker.load_config(path)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = worker.run_job(config)
                spec = load_task(staged / "task.md")
                held = pd.read_csv(job_dir / "holdout_predictions.csv", dtype=str, keep_default_na=False)
                independently_scored = score_holdout(job_dir / "holdout_predictions.csv", spec, held["id"].tolist(), held["actual"].tolist())
                self.assertAlmostEqual(result["metric"]["value"], independently_scored)
                self.assertEqual(result["metric"]["name"], metric_name)
                self.assertEqual(result["metric"]["higher_is_better"], metric_name in {"R2", "Weighted F1", "Accuracy", "ROC AUC"})
                if metric_name in {"LogLoss", "ROC AUC"}:
                    output = pd.read_csv(staged / "submission.csv")
                    self.assertTrue(output["answer"].between(0, 1).all())
                shutil.rmtree(source)

    def test_saved_full_pipeline_reproduces_submission_from_raw_csv(self):
        from agent.features import normalize_feature_config
        for feature_options in ({}, {"signed_log": True, "rare_min_count": 2, "missing_indicators": True}):
            with self.subTest(feature_options=feature_options):
                source = self.make_source(classification=True)
                inputs = source / "nested/participants"
                frame = pd.read_csv(inputs / "train.csv", dtype=str, keep_default_na=False)
                frame["x"], frame["flag"] = "0", "0"
                frame["protocol"] = ["__MISSING__", "", "__ALFA_LEGACY_VALUE__x", "plain"] * 20
                frame["answer"] = ["NA", "Positive", "NA", "Positive"] * 20
                frame.to_csv(inputs / "train.csv", index=False)
                write_csv(inputs / "test.csv", [
                    ["row_key", "x", "flag", "protocol"],
                    ["NA", "0", "0", "__MISSING__"], ["0001", "0", "0", ""],
                ])
                path, job_dir, staged = self.make_config(source, "extra_trees")
                config = json.loads(path.read_text())
                config["feature_config"] = normalize_feature_config(feature_options)
                path.write_text(json.dumps(config))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(worker.main(["--config", str(path)]), 0)
                result = json.loads((job_dir / "result.json").read_text())
                self.assertTrue(result["model_artifact"]["available"])
                self.assertEqual(result["feature_config"], normalize_feature_config(feature_options))
                self.assertEqual(result["training_device"], "cpu")
                self.assertTrue(result["training_device_evidence"]["fit_completed"])
                restored = joblib.load(artifact_download_path(job_dir, "model"))
                test = pd.read_csv(inputs / "test.csv", dtype=str, keep_default_na=False)
                submission = pd.read_csv(staged / "submission.csv", dtype=str, keep_default_na=False)
                self.assertEqual(submission["answer"].tolist(), ["NA", "Positive"])
                self.assertEqual(restored.predict(test[["x", "flag", "protocol"]]).tolist(),
                                 submission["answer"].tolist())
                shutil.rmtree(source)
                shutil.rmtree(job_dir)

    def test_model_export_error_preserves_verified_training_result(self):
        source = self.make_source()
        path, job_dir, staged = self.make_config(source, "linear")
        with patch.object(worker, "save_model_artifacts", side_effect=OSError("disk unavailable")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(["--config", str(path)]), 0)
        result = json.loads((job_dir / "result.json").read_text())
        self.assertTrue(result["output_valid"])
        self.assertFalse(result["model_artifact"]["available"])
        self.assertTrue((staged / "submission.csv").is_file())

    def test_config_refuses_unrecognized_recipe_and_paths_outside_job(self):
        source = self.make_source()
        path, _, _ = self.make_config(source, "baseline")
        config = json.loads(path.read_text(encoding="utf-8"))
        config["action"] = "execute_anything"
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            worker.load_config(path)
        config["action"] = "baseline"
        config["task_path"] = str(source / "task.md")
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            worker.load_config(path)

    def test_time_and_grouped_recipes_keep_their_evidence_and_comparison_identity(self):
        for classification, validation in (
            (False, {"strategy": "time", "time_column": "x"}),
            (True, {"strategy": "grouped"}),
        ):
            with self.subTest(validation=validation):
                source = self.make_source(classification)
                original_train_path = source / "nested/participants/train.csv"
                original = pd.read_csv(original_train_path, dtype=str, keep_default_na=False)
                original["x"] = [str(i // 4 if classification else i // 16) for i in range(80)]
                if classification:
                    original["flag"] = [str((i // 4) % 2) for i in range(80)]
                original.to_csv(original_train_path, index=False)
                results = []
                for action in ("baseline", "extra_trees"):
                    path, job_dir, _ = self.make_config(source, action)
                    config = json.loads(path.read_text(encoding="utf-8"))
                    config["validation"] = validation
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = worker.main(["--config", str(path)])
                    self.assertEqual(code, 0)
                    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
                    results.append(result)
                    self.assertEqual(result["split"]["strategy"], validation["strategy"])
                    holdout = pd.read_csv(job_dir / "holdout_predictions.csv", dtype=str, keep_default_na=False)
                    mask = original["row_key"].isin(holdout["id"])
                    if classification:
                        features = ["x", "flag", "protocol"]
                        fitted = set(map(tuple, original.loc[~mask, features].to_numpy()))
                        held = set(map(tuple, original.loc[mask, features].to_numpy()))
                        self.assertFalse(fitted & held)
                        self.assertEqual(result["split"]["group_overlap"], 0)
                    else:
                        self.assertEqual(set(original.loc[mask, "x"]), {"4"})
                        self.assertEqual(result["split"]["validation_period"], "4")
                    self.assertAlmostEqual(result["split"]["validation_fraction"], len(holdout) / len(original))
                    self.assertTrue(result["output_valid"])
                self.assertEqual(results[0]["comparison_key"], results[1]["comparison_key"])
                self.assertEqual(results[0]["split"]["validation_hash"], results[1]["split"]["validation_hash"])

                random_path, random_job_dir, _ = self.make_config(source, "baseline", source.name + "-random-reference")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(worker.main(["--config", str(random_path)]), 0)
                random_result = json.loads((random_job_dir / "result.json").read_text(encoding="utf-8"))
                self.assertNotEqual(results[0]["comparison_key"], random_result["comparison_key"])

    def test_automl_preserves_verified_winner_diagnostics_in_result_and_metrics(self):
        from agent.diagnostics import classification_diagnostics

        source = self.make_source(classification=True)
        path, job_dir, staged = self.make_config(source, "automl")
        diagnostics = classification_diagnostics(["NA", "Positive"], ["NA", "Positive"])
        model_config = make_model_config("baseline", "classification")
        evidence = {
            "model": model_config["model"], "model_config": model_config,
            "metric": {"name": "Macro F1", "value": 1.0, "higher_is_better": True},
            "split": {"validation_rows": 2}, "comparison_key": "b" * 64,
            "output_valid": True, "task_type": "classification", "elapsed_seconds": 1.0,
            "classification_diagnostics": diagnostics,
            "search": {"status": "completed", "planned": 1, "completed": 1,
                       "best_candidate": "candidate_1", "validation_reason": "Проверка",
                       "candidates": [{"id": "candidate_1", "label": "Победитель",
                                       "model": model_config["model"], "status": "completed",
                                       "metric": 1.0, "selected": True,
                                       "elapsed_seconds": 1.0, "error": None}]},
        }

        def completed_search(task_path, **kwargs):
            write_csv(task_path.parent / "submission.csv", [
                ["row_key", "answer"], ["NA", "NA"], ["0001", "Positive"],
            ])
            return evidence

        with patch("agent.automl.run_automl", side_effect=completed_search), contextlib.redirect_stdout(io.StringIO()):
            result = worker.run_job(worker.load_config(path))
        self.assertEqual(result["classification_diagnostics"], diagnostics)
        for name in ("result.json", "metrics.json"):
            saved = json.loads((job_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(saved["classification_diagnostics"], diagnostics)

    def test_automl_config_accepts_auto_and_rejects_parameters_and_bad_budgets(self):
        source = self.make_source()
        path, _, _ = self.make_config(source, "automl")
        config = json.loads(path.read_text(encoding="utf-8"))
        loaded = worker.load_config(path)
        self.assertEqual(loaded["validation"], {"strategy": "auto"})
        self.assertEqual(loaded["time_budget"], 600)
        for updates in (
            {"model_params": None}, {"model_params": {"n_estimators": 50}},
            {"time_budget": True}, {"time_budget": "60"}, {"time_budget": 0},
            {"time_budget": float("nan")}, {"time_budget": 10501},
            {"validation": {"strategy": "auto", "time_column": "x"}},
        ):
            with self.subTest(updates=updates):
                path.write_text(json.dumps({**config, **updates}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    worker.load_config(path)
        path.write_text(json.dumps({**config, "action": "baseline", "validation": {"strategy": "auto"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            worker.load_config(path)

    def test_config_accepts_legacy_and_rejects_unknown_validation_options(self):
        source = self.make_source()
        path, _, _ = self.make_config(source, "baseline")
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(worker.load_config(path)["validation"], {"strategy": "random"})
        for value in ({"strategy": "time"}, {"strategy": "shell"}, {"strategy": "random", "code": "x"}):
            with self.subTest(validation=value):
                path.write_text(json.dumps({**config, "validation": value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    worker.load_config(path)
        path.write_text(json.dumps({**config, "unexpected": True}), encoding="utf-8")
        with self.assertRaises(ValueError):
            worker.load_config(path)

    def test_custom_model_parameters_change_training_and_refit_without_changing_holdout(self):
        for classification in (False, True):
            source = self.make_source(classification)
            task_type = "classification" if classification else "regression"
            for action in ("baseline", "extra_trees"):
                with self.subTest(task_type=task_type, action=action):
                    outputs = []
                    custom = ({"max_iter": 50, "max_leaf_nodes": 4, "min_samples_leaf": 100,
                               "learning_rate": 0.03, "l2_regularization": 0.5}
                              if action == "baseline" and not classification else
                              {"n_estimators": 50, "max_depth": 2, "min_samples_leaf": 100})
                    for suffix, params in (("default", None), ("custom", custom)):
                        path, job_dir, staged = self.make_config(source, action, f"{task_type}-{action}-{suffix}")
                        config = json.loads(path.read_text(encoding="utf-8"))
                        if params is not None:
                            config["model_params"] = params
                        path.write_text(json.dumps(config), encoding="utf-8")
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(worker.main(["--config", str(path)]), 0)
                        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
                        metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
                        holdout = pd.read_csv(job_dir / "holdout_predictions.csv", dtype=str, keep_default_na=False)
                        outputs.append((result, holdout))
                        self.assertEqual(result["model_config"], make_model_config(action, task_type, params))
                        self.assertEqual(result["model_config"], metrics["model_config"])
                        for key, value in result["model_config"]["parameters"].items():
                            self.assertEqual(metrics["model"]["parameters"][key], value)
                        self.assertEqual(metrics["model"]["parameters"]["random_state"], 42)
                        if "n_jobs" in metrics["model"]["parameters"]:
                            self.assertEqual(metrics["model"]["parameters"]["n_jobs"], 8)
                        if suffix == "custom":
                            # Both the holdout model (64 rows) and final clone
                            # (80 rows) are too small to split with leaf=100.
                            submission = pd.read_csv(staged / "submission.csv", dtype=str, keep_default_na=False)
                            self.assertEqual(holdout["prediction"].nunique(), 1)
                            self.assertEqual(submission["answer"].nunique(), 1)
                    self.assertEqual(outputs[0][0]["comparison_key"], outputs[1][0]["comparison_key"])
                    self.assertEqual(outputs[0][1]["id"].tolist(), outputs[1][1]["id"].tolist())
                    self.assertNotEqual(outputs[0][1]["prediction"].tolist(), outputs[1][1]["prediction"].tolist())

    def test_model_params_reject_unknown_controls_under_actual_task_type(self):
        source = self.make_source()
        path, job_dir, _ = self.make_config(source, "baseline")
        base = json.loads(path.read_text(encoding="utf-8"))
        for params in ({"n_jobs": 1000}, {"n_estimators": 120}, {"max_iter": True}, {"learning_rate": "0.1"}):
            with self.subTest(params=params):
                path.write_text(json.dumps({**base, "model_params": params}), encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(worker.main(["--config", str(path)]), 1)
                self.assertFalse((job_dir / "holdout_predictions.csv").exists())
                self.assertFalse(json.loads((job_dir / "result.json").read_text(encoding="utf-8"))["output_valid"])

    def test_full_recipe_preserves_objective_and_target_transform_through_refit_and_json(self):
        source = self.make_source()
        path, job_dir, staged = self.make_config(source, "baseline")
        config = json.loads(path.read_text(encoding="utf-8"))
        recipe = make_model_config("baseline", "regression", {"max_iter": 50},
                                   objective="mae", target_transform="log1p")
        path.write_text(json.dumps({**config, "model_config": recipe}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(["--config", str(path)]), 0)
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(result["model_config"], recipe)
        self.assertEqual(metrics["model_config"], recipe)
        self.assertEqual(metrics["model"]["parameters"]["loss"], "absolute_error")
        self.assertNotIn("estimator", metrics["model"]["parameters"])
        held = pd.read_csv(job_dir / "holdout_predictions.csv")
        expected = np.sqrt(mean_squared_error(held["actual"], held["prediction"]))
        self.assertAlmostEqual(result["metric"]["value"], expected)
        self.assertTrue(worker.validate_predictions(staged)["valid"])

    def test_full_recipe_rejects_conflicting_legacy_params_action_and_automl(self):
        source = self.make_source()
        path, _, _ = self.make_config(source, "baseline")
        base = json.loads(path.read_text(encoding="utf-8"))
        recipe = make_model_config("baseline", "regression", objective="mae")
        invalid = [
            {"model_config": recipe, "model_params": None},
            {"model_config": recipe, "model_params": {}},
            {"model_config": recipe, "action": "automl"},
            {"model_config": make_model_config("extra_trees", "regression")},
            {"model_config": None},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**base, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    worker.load_config(path)

    def test_full_recipe_rejects_wrong_actual_task_type_before_building_model(self):
        for classification in (False, True):
            with self.subTest(classification=classification):
                source = self.make_source(classification)
                path, job_dir, staged = self.make_config(source, "baseline")
                config = json.loads(path.read_text(encoding="utf-8"))
                wrong_type = "regression" if classification else "classification"
                recipe = make_model_config("baseline", wrong_type)
                path.write_text(json.dumps({**config, "model_config": recipe}), encoding="utf-8")
                with patch.object(worker, "build_pipeline") as build, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(worker.main(["--config", str(path)]), 1)
                build.assert_not_called()
                self.assertFalse((job_dir / "holdout_predictions.csv").exists())
                self.assertFalse((staged / "submission.csv").exists())

    def test_existing_symlink_output_is_refused_before_training(self):
        source = self.make_source()
        path, job_dir, _ = self.make_config(source, "baseline")
        outside = self.root / "private.json"
        outside.write_text("unchanged", encoding="utf-8")
        try:
            (job_dir / "metrics.json").symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            worker.load_config(path)
        self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged")

    def test_training_failure_returns_nonzero_and_a_readable_error(self):
        source = self.make_source(classification=True)
        path, job_dir, staged = self.make_config(source, "baseline")
        train_path = staged / "nested/participants/train.csv"
        train = pd.read_csv(train_path, dtype=str, keep_default_na=False)
        train["answer"] = "only-one-class"
        train.to_csv(train_path, index=False)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(worker.main(["--config", str(path)]), 1)
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(result["output_valid"])
        self.assertTrue(result["error"])
        self.assertFalse((staged / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
