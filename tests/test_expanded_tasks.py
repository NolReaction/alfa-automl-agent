"""Real worker/search checks on new task contracts, without external APIs."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score, roc_auc_score

from agent import automl
from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.research_validation import ResearchValidation
from agent.task_spec import load_task
from offline_baseline import read_csv
from validation import validate_predictions


def task_fixture(root, metric, *, multiclass=False, rare_groups=False):
    inputs = root / "inputs"
    inputs.mkdir()
    classification = metric in {"Accuracy", "LogLoss", "ROC AUC"}
    if rare_groups:
        groups = np.repeat(np.arange(10), 2)
        x = groups.astype(object)
        categories = [f"group-{group}" for group in groups]
        target = ["1" if group >= 8 else "0" for group in groups]
    else:
        x = np.linspace(-3, 3, 150).astype(object)
        categories = ["NA" if index % 3 else "blue" for index in range(len(x))]
        if multiclass:
            target = ["A" if value < -1 else "B" if value < 1 else "C" for value in x]
        elif classification:
            target = ["1" if value > 0 else "0" for value in x]
        else:
            target = [float(value) * 1.7 + (index % 3) * .15 for index, value in enumerate(x)]
        x[::13] = ""  # missing numeric values are prepared inside each fitted fold
    train = pd.DataFrame({"key": [f"train-{index:04d}" for index in range(len(x))],
                          "measurement": x, "category": categories, "answer": target})
    test = pd.DataFrame({"key": ["0007", "NA", "002"],
                         "measurement": [-2, "", 2], "category": ["never-seen", "", "NA"]})
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    columns = ["key", "C", "A", "B"] if multiclass else ["key", "answer"]
    pd.DataFrame([[key, *[0 for _ in columns[1:]]] for key in test["key"]], columns=columns).to_csv(
        inputs / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["key", "measurement", "category", "answer"],
                  "role": ["id", "feature", "feature", "target"],
                  "dtype": ["category", "numeric", "category", "category" if multiclass else "numeric"]}).to_csv(
        inputs / "data_dictionary.csv", index=False)
    task = root / "task.md"
    task.write_text(f"Predict answer from inputs/*.csv.\nPrimary metric: {metric}\n"
                    + ('Class order: ["C", "B", "A"]\n' if multiclass else "")
                    + "Write submission.csv here in template column order.\n", encoding="utf-8")
    return task


def small_plan(classification):
    task_type = "classification" if classification else "regression"
    parameters = ([{"C": .25, "max_iter": 100}, {"C": 4.0, "max_iter": 100}]
                  if classification else [{"alpha": .1}, {"alpha": 10.0}])
    return [{"id": f"tiny_{index}", "action": "linear", "label": f"Small linear {index}",
             "model_config": make_model_config("linear", task_type, values),
             "feature_config": normalize_feature_config({"missing_indicators": True})}
            for index, values in enumerate(parameters, 1)]


class ExpandedTaskEndToEndTests(unittest.TestCase):
    def run_checked_search(self, task):
        originals = {path: path.read_bytes() for path in [task, *sorted((task.parent / "inputs").iterdir())]}
        spec = load_task(task)
        publications = []

        def progress(event):
            if event["search"]["best_candidate"] is not None:
                # A seed remains downloadable while later selection/refits run.
                self.assertTrue(validate_predictions(task.parent, spec=spec)["valid"])
                publications.append(event["search"]["best_candidate"])

        with patch.object(automl, "candidate_plan", return_value=small_plan(spec.classification)), \
                redirect_stdout(io.StringIO()):
            report = automl.run_automl(task, time_budget=90, progress=progress)
        self.assertTrue(publications)
        self.assertEqual(publications[0], "seed")
        self.assertTrue(report["output_valid"])
        self.assertEqual(report["api"], {"used": False, "calls": 0})
        self.assertEqual(report["task_type"], spec.metric.task_type)
        self.assertEqual(report["metric"]["name"], spec.metric.name)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        submission = read_csv(task.parent / "submission.csv")
        self.assertEqual(submission["key"].tolist(), ["0007", "NA", "002"])
        self.assertEqual(submission.columns.tolist(), list(spec.output_columns))
        held_path = Path(report["report_path"]).parent / "holdout_predictions.csv"
        held = read_csv(held_path)
        self.assertEqual(held.columns.tolist(), list(spec.holdout_columns))
        gate = ResearchValidation(task)
        self.assertEqual(gate.evaluate(held_path, task.parent), report["metric"])
        self.assertEqual(gate.split["validation_hash"], report["split"]["validation_hash"])
        altered = held.copy()
        altered.loc[0, "actual"] = "forged-answer"
        bad = task.parent / "tampered_holdout.csv"
        altered.to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, "actual answers"):
            gate.evaluate(bad, task.parent)
        return report, submission, held

    def test_numeric_class_labels_accuracy_and_probability_metrics_use_real_workers(self):
        for metric in ("Accuracy", "LogLoss", "ROC AUC"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                task = task_fixture(Path(directory), metric)
                report, submission, held = self.run_checked_search(task)
                self.assertEqual(report["task_type"], "classification")
                if metric == "Accuracy":
                    expected = accuracy_score(held["actual"], held["prediction"])
                    self.assertTrue(set(submission["answer"]) <= {"0", "1"})
                else:
                    probability = pd.to_numeric(held["prediction"]).to_numpy()
                    self.assertTrue(((probability > 0) & (probability < 1)).any())
                    expected = (log_loss(held["actual"], probability, labels=["0", "1"])
                                if metric == "LogLoss" else roc_auc_score(held["actual"] == "1", probability))
                self.assertAlmostEqual(report["metric"]["value"], expected, places=12)

    def test_multiclass_probability_columns_are_realigned_to_reversed_task_and_template_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            task = task_fixture(Path(directory), "LogLoss", multiclass=True)
            report, submission, held = self.run_checked_search(task)
            self.assertEqual(submission.columns.tolist(), ["key", "C", "A", "B"])
            self.assertEqual(held.columns.tolist(), ["id", "actual", "probability:C", "probability:B", "probability:A"])
            probabilities = held.loc[:, ["probability:A", "probability:B", "probability:C"]].astype(float)
            self.assertAlmostEqual(report["metric"]["value"],
                                   log_loss(held["actual"], probabilities, labels=["A", "B", "C"]), places=12)
            np.testing.assert_allclose(submission.loc[:, ["A", "B", "C"]].astype(float).sum(axis=1), 1)

    def test_regression_mae_minimizes_and_r2_maximizes_independently_recomputed_scores(self):
        for metric in ("MAE", "R2"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                task = task_fixture(Path(directory), metric)
                report, _, held = self.run_checked_search(task)
                actual, predicted = held["actual"].astype(float), held["prediction"].astype(float)
                expected = mean_absolute_error(actual, predicted) if metric == "MAE" else r2_score(actual, predicted)
                self.assertAlmostEqual(report["metric"]["value"], expected, places=12)
                self.assertEqual(report["metric"]["higher_is_better"], metric == "R2")

    def test_rare_grouped_class_without_safe_inner_split_keeps_only_the_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            task = task_fixture(Path(directory), "LogLoss", rare_groups=True)
            report, _, _ = self.run_checked_search(task)
            self.assertEqual(report["split"]["strategy"], "grouped")
            self.assertEqual(report["split"]["group_overlap"], 0)
            self.assertIsNone(report["inner_split"])
            self.assertTrue(report["inner_unavailable"])
            self.assertEqual(report["search"]["best_candidate"], "seed")
            self.assertEqual([candidate["id"] for candidate in report["search"]["candidates"]], ["seed"])


if __name__ == "__main__":
    unittest.main()
