"""Points formula, derived holdout tasks and scoring of saved submissions; no training."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.task_spec import load_task
import score_meter as meter


@contextmanager
def workspace():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(meter, "RAW_ROOT", root / "score_meter"):
            yield root


def write_task(root, name, metric, frame):
    task = root / "sources" / name / "task.md"
    data = task.parent / name / "participants"
    data.mkdir(parents=True)
    task.write_text("Задача.\nPrimary metric: " + metric + "\n", encoding="utf-8")
    frame.to_csv(data / "train.csv", index=False)
    frame.drop(columns="target").head(3).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame["id"].head(3), "target": frame["target"].head(3)}).to_csv(
        data / "sample_submission.csv", index=False)
    roles = {"id": ("id", "numeric"), "target": ("target", "numeric" if metric == "RMSE" else "category")}
    dictionary = [(name, *roles.get(name, ("feature", "category" if name in {"kind", "proto"} else "numeric")))
                  for name in frame.columns]
    pd.DataFrame(dictionary, columns=["feature", "role", "dtype"]).to_csv(data / "data_dictionary.csv", index=False)
    return task


def sources(root):
    random = np.random.default_rng(5)
    rows = 300
    regression = pd.DataFrame({
        "id": [str(1000 + index) for index in range(rows)],
        "Year_Factor": [str(1 + index % 3) for index in range(rows)],
        "kind": random.choice(["NA", "0007", ""], size=rows),
        "x": [f"{value:.3f}" for value in random.normal(size=rows)],
        "target": [f"{value:.3f}" for value in random.normal(50, 10, size=rows)]})
    labels = np.asarray(["Normal", "DoS", "Worms"])[np.arange(rows) % 3]
    classification = pd.DataFrame({
        "id": [str(index) for index in range(rows)],
        "proto": random.choice(["tcp", "udp"], size=rows),
        "x": [f"{value:.3f}" for value in random.normal(size=rows)],
        "target": labels})
    return {"regression": write_task(root, "regression", "RMSE", regression),
            "classification": write_task(root, "classification", "Macro F1", classification)}


class PointsTests(unittest.TestCase):
    def test_regression_formula_and_clipping(self):
        self.assertAlmostEqual(meter.points("RMSE", 40.0, 50.0), 20.0)
        self.assertEqual(meter.points("rmse", 50.0, 50.0), 0.0)
        self.assertEqual(meter.points("RMSE", 60.0, 50.0), 0.0)
        self.assertAlmostEqual(meter.points("RMSE", 0.0, 50.0), 100.0)

    def test_classification_formula_and_clipping(self):
        self.assertAlmostEqual(meter.points("Macro F1", 0.7, 0.6), 25.0)
        self.assertEqual(meter.points("Macro F1", 0.6, 0.6), 0.0)
        self.assertEqual(meter.points("Macro F1", 0.5, 0.6), 0.0)
        self.assertAlmostEqual(meter.points("Macro F1", 1.0, 0.6), 100.0)

    def test_unknown_metrics_and_invalid_values_are_rejected(self):
        for arguments in (("MAE", 1.0, 2.0), ("Accuracy", 0.9, 0.8), ("RMSE", 1.0, 0.0),
                          ("RMSE", float("nan"), 1.0), ("Macro F1", 0.5, 1.0),
                          ("Macro F1", 1.2, 0.5), ("RMSE", True, 2.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                meter.points(*arguments)


class PrepareTests(unittest.TestCase):
    def test_prepare_writes_derived_tasks_and_keeps_answers_host_only(self):
        with workspace() as root:
            output = root / "score_meter" / "run"
            with patch.object(meter, "split_rows", wraps=meter.split_rows) as splitter:
                plan = meter.prepare(output, sources(root))
            self.assertEqual(splitter.call_count, 1)
            regression = plan["cases"]["regression_last_period"]
            self.assertEqual((regression["split"]["fit_period_max"], regression["split"]["holdout_period"]),
                             ("2", "3"))
            self.assertEqual(regression["rows"], {"fit": 200, "holdout": 100})
            for name in meter.CASES:
                with self.subTest(name=name):
                    inputs = output / "inputs" / name
                    spec = load_task(inputs / "task.md")
                    test = pd.read_csv(spec.data_dir / "test.csv", **meter.CSV)
                    train = pd.read_csv(spec.data_dir / "train.csv", **meter.CSV)
                    sample = pd.read_csv(spec.data_dir / "sample_submission.csv", **meter.CSV)
                    answers = pd.read_csv(output / "host_only" / name / "answers.csv", **meter.CSV)
                    self.assertNotIn("target", test.columns)
                    self.assertEqual(test["id"].tolist(), sample["id"].tolist())
                    self.assertEqual(test["id"].tolist(), answers["id"].tolist())
                    self.assertFalse(set(train["id"]) & set(test["id"]))
                    self.assertFalse(list(inputs.rglob("answers.csv")))
            raw = load_task(output / "inputs" / "regression_last_period" / "task.md")
            kinds = pd.read_csv(raw.data_dir / "train.csv", **meter.CSV)["kind"]
            self.assertTrue({"NA", "0007", ""} <= set(kinds))
            classification = plan["cases"]["classification_grouped"]["split"]
            self.assertEqual(classification["group_overlap"], 0)
            with self.assertRaises(ValueError):
                meter.prepare(output, sources(root / "again"))

    def test_output_must_be_below_raw_root_and_inputs_are_sealed(self):
        with workspace() as root:
            with self.assertRaises(ValueError):
                meter.prepare(root / "elsewhere", sources(root))
            output = root / "score_meter" / "sealed"
            meter.prepare(output, sources(root / "second"))
            train = next((output / "inputs" / "regression_last_period").rglob("train.csv"))
            train.write_bytes(train.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                meter.stage(output, "arm")


class ScoreTests(unittest.TestCase):
    def submit(self, output, arm, name, predictions):
        task = output / "arms" / arm / name / "task.md"
        spec = load_task(task)
        test = pd.read_csv(spec.data_dir / "test.csv", **meter.CSV)
        pd.DataFrame({"id": test["id"], "target": predictions(test)}).to_csv(
            task.parent / "submission.csv", index=False)

    def test_points_against_a_baseline_arm_and_assumed_values(self):
        with workspace() as root:
            output = root / "score_meter" / "scores"
            meter.prepare(output, sources(root))
            answers = {name: pd.read_csv(output / "host_only" / name / "answers.csv", **meter.CSV)
                       for name in meter.CASES}
            for arm in ("organizer", "agent", "broken"):
                meter.stage(output, arm)
            truth = answers["regression_last_period"]["target"].astype(float).to_numpy()
            self.submit(output, "organizer", "regression_last_period", lambda test: truth + 2.0)
            self.submit(output, "agent", "regression_last_period", lambda test: truth + 1.0)
            labels = answers["classification_grouped"]["target"].to_numpy()
            self.submit(output, "organizer", "classification_grouped",
                        lambda test: np.where(np.arange(len(test)) % 2 == 0, labels, "Normal"))
            self.submit(output, "agent", "classification_grouped", lambda test: labels)
            self.submit(output, "broken", "regression_last_period", lambda test: ["x"] * len(test))
            result = meter.score(output, baseline_arm="organizer")
            agent = result["arms"]["agent"]
            self.assertAlmostEqual(agent["regression_last_period"]["metric"], 1.0)
            self.assertAlmostEqual(agent["regression_last_period"]["points"], 50.0)
            self.assertAlmostEqual(agent["classification_grouped"]["points"], 100.0)
            self.assertAlmostEqual(agent["public_mean_points"], 75.0)
            self.assertEqual(result["arms"]["organizer"]["regression_last_period"]["points"], 0.0)
            self.assertEqual(result["arms"]["broken"]["regression_last_period"]["status"], "invalid")
            self.assertEqual(result["arms"]["broken"]["classification_grouped"]["status"], "missing")
            self.assertIsNone(result["arms"]["broken"]["public_mean_points"])
            saved = json.loads((output / "scores.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["estimate_not_platform_score"])
            assumed = meter.score(output, assumed={"regression_last_period": 4.0})
            self.assertAlmostEqual(assumed["arms"]["agent"]["regression_last_period"]["points"], 75.0)
            self.assertNotIn("points", assumed["arms"]["agent"]["classification_grouped"])
            with self.assertRaises(ValueError):
                meter.score(output, baseline_arm="broken")

    def test_run_uses_a_short_copy_and_brings_back_only_submission_and_log(self):
        def fake_solution(command, cwd=None, stdout=None, **kwargs):
            if command[0] == "git":
                return type("Completed", (), {"returncode": 1, "stdout": ""})()
            spec = load_task(Path(command[2]))
            test = pd.read_csv(spec.data_dir / "test.csv", **meter.CSV)
            pd.DataFrame({"id": test["id"], "target": "7.0"}).to_csv(Path(cwd) / "submission.csv", index=False)
            (Path(cwd) / ".agent_runs").mkdir()
            stdout.write(b"fake run\n")
            return type("Completed", (), {"returncode": 0})()

        with workspace() as root:
            output = root / "score_meter" / "run"
            meter.prepare(output, sources(root))
            with patch.object(meter.subprocess, "run", side_effect=fake_solution) as runner:
                records = meter.run(output, "arm", 60, ["--history-auto"], ["regression_last_period"],
                                    work_root=root)
            solutions = [call.args[0] for call in runner.call_args_list if call.args[0][0] != "git"]
            self.assertEqual(len(solutions), 1)
            command = solutions[0]
            self.assertEqual(command[3:], ["--mode", "offline", "--time-budget", "60", "--history-auto"])
            self.assertTrue(Path(command[2]).is_relative_to(root) and "score_meter" not in command[2])
            staged = output / "arms" / "arm" / "regression_last_period"
            self.assertEqual((staged / "run.log").read_bytes(), b"fake run\n")
            self.assertFalse((staged / ".agent_runs").exists())
            self.assertEqual(records["regression_last_period"]["exit_code"], 0)
            self.assertEqual(json.loads((staged / "run.json").read_text(encoding="utf-8"))["exit_code"], 0)
            self.assertEqual(meter.score(output)["arms"]["arm"]["regression_last_period"]["status"], "scored")
            with self.assertRaises(ValueError):
                meter.run(output, "other", 10)

    def test_stage_refuses_to_reuse_an_arm(self):
        with workspace() as root:
            output = root / "score_meter" / "stage"
            meter.prepare(output, sources(root))
            meter.stage(output, "arm")
            with self.assertRaises(ValueError):
                meter.stage(output, "arm")
            with self.assertRaises(ValueError):
                meter.stage(output, "../escape")


if __name__ == "__main__":
    unittest.main()
