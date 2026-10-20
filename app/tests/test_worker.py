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
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from app import worker
from app.model_config import make_model_config


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
        (source / "task.md").write_text("Synthetic test task", encoding="utf-8")
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
                    self.assertEqual(transitions, ["training", "validating", "refitting", "finished"])
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
                    else:
                        measured = np.sqrt(mean_squared_error(holdout["actual"].astype(float),
                                                              holdout["prediction"].astype(float)))
                        self.assertEqual(result["metric"]["name"], "RMSE")
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
