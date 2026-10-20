import csv
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.task_spec import TaskSpec, load_task, load_task_data, score_holdout
from agent.metrics import score_predictions
from offline_baseline import load_data
from validation import validate_predictions


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def make_task(root, *, metric="Macro F1", labels=("0", "1"), dtype="numeric",
              outputs=("answer",), extra=""):
    inputs = root / "nested" / "participants"
    write_csv(inputs / "data_dictionary.csv", [
        ["feature", "role", "dtype"], ["row_key", "id", "category"],
        ["x", "feature", "numeric"], ["kind", "feature", "category"],
        ["answer", "target", dtype],
    ])
    write_csv(inputs / "train.csv", [["row_key", "x", "kind", "answer"]] + [
        [f"train-{i:03d}", i, "a" if i % 2 else "b", labels[i % len(labels)]] for i in range(60)
    ])
    write_csv(inputs / "test.csv", [["row_key", "x", "kind"], ["0005", 5, "a"], ["NA", 6, "new"]])
    write_csv(inputs / "sample_submission.csv", [["row_key", *outputs],
                                                 ["0005", *[0 for _ in outputs]],
                                                 ["NA", *[0 for _ in outputs]]])
    task = root / "task.md"
    task.write_text(f"Predict answer.\nPrimary metric: {metric}\n{extra}", encoding="utf-8")
    return task


class TaskSpecTests(unittest.TestCase):
    def test_numeric_coded_classes_use_declared_metric_and_remain_literal_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory))
            spec = load_task(task)
            self.assertTrue(spec.classification)
            self.assertEqual(spec.classes, ("0", "1"))
            self.assertEqual(load_task_data(spec)[5].tolist()[:4], ["0", "1", "0", "1"])
            self.assertTrue(load_task_data(spec)[-1])
            self.assertFalse(load_data(spec.data_dir)[-1])  # legacy signature stays compatible
            frame = spec.prediction_frame(["0005", "NA"], ["0", "1"])
            frame.to_csv(task.parent / "submission.csv", index=False)
            self.assertTrue(validate_predictions(task.parent)["valid"])
            frame.loc[0, "answer"] = "2"
            frame.to_csv(task.parent / "submission.csv", index=False)
            self.assertFalse(validate_predictions(task.parent)["valid"])

    def test_spec_roundtrips_and_cannot_be_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_task(make_task(Path(directory)))
            self.assertEqual(TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))), spec)
            with self.assertRaises(FrozenInstanceError):
                spec.target_column = "other"

    def test_literal_nan_class_is_valid_but_actual_missing_predictions_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_task(make_task(Path(directory), metric="Accuracy",
                                       labels=("nan", "yes"), dtype="category"))
            ids, actual = ["0005", "NA"], ["nan", "yes"]
            self.assertEqual(spec.prediction_frame(ids, actual)["answer"].tolist(), actual)
            self.assertEqual(spec.holdout_frame(ids, actual, actual)["prediction"].tolist(), actual)
            for missing in (None, float("nan"), np.float32("nan")):
                with self.subTest(missing=repr(missing)):
                    with self.assertRaisesRegex(ValueError, "empty labels"):
                        spec.prediction_frame(ids, [missing, "yes"])
                    with self.assertRaisesRegex(ValueError, "empty labels"):
                        spec.holdout_frame(ids, actual, [missing, "yes"])
                    with self.assertRaisesRegex(ValueError, "empty labels"):
                        spec.holdout_frame(ids, [missing, "yes"], actual)

    def test_binary_probability_contract_preserves_ids_and_checks_range(self):
        for metric in ("ROC AUC", "LogLoss"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                task = make_task(Path(directory), metric=metric)
                spec = load_task(task)
                self.assertEqual(spec.positive_class, "1")
                frame = spec.prediction_frame(["0005", "NA"], [[.8, .2], [.1, .9]],
                                              prediction_classes=("0", "1"))
                self.assertEqual(frame["answer"].tolist(), [.2, .9])
                frame.to_csv(task.parent / "submission.csv", index=False)
                self.assertTrue(validate_predictions(task.parent)["valid"])
                frame.loc[0, "answer"] = 1.1
                frame.to_csv(task.parent / "submission.csv", index=False)
                self.assertFalse(validate_predictions(task.parent)["valid"])

    def test_arbitrary_binary_labels_need_explicit_positive_class(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="ROC AUC", labels=("no", "yes"), dtype="category")
            with self.assertRaisesRegex(ValueError, "Positive class"):
                load_task(task)
            task.write_text(task.read_text() + "Positive class: yes\n", encoding="utf-8")
            self.assertEqual(load_task(task).positive_class, "yes")

    def test_multiclass_probabilities_follow_template_and_model_class_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="LogLoss", labels=("A", "B", "C"),
                             dtype="category", outputs=("C", "A", "B"))
            spec = load_task(task)
            frame = spec.prediction_frame(["0005", "NA"], [[.7, .2, .1], [.1, .3, .6]],
                                          prediction_classes=("B", "C", "A"))
            self.assertEqual(frame.columns.tolist(), ["row_key", "C", "A", "B"])
            self.assertEqual(frame.iloc[0].tolist(), ["0005", .2, .1, .7])
            frame.to_csv(task.parent / "submission.csv", index=False)
            self.assertTrue(validate_predictions(task.parent)["valid"])
            frame.loc[0, "A"] = .8
            frame.to_csv(task.parent / "submission.csv", index=False)
            self.assertFalse(validate_predictions(task.parent)["valid"])

    def test_host_score_uses_frozen_actuals_and_fresh_normalized_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="LogLoss", labels=("A", "B", "C"),
                             dtype="category", outputs=("C", "A", "B"))
            spec = load_task(task)
            ids, actual = ["001", "NA", "03"], ["A", "B", "C"]
            frame = spec.holdout_frame(ids, actual, [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]],
                                      prediction_classes=("A", "B", "C"))
            path = task.parent / "holdout.csv"
            frame.to_csv(path, index=False)
            self.assertAlmostEqual(score_holdout(path, spec, ids, actual), -math.log(.8))
            frame.loc[0, "actual"] = "C"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "actual answers"):
                score_holdout(path, spec, ids, actual)

    def test_regression_holdout_accepts_equivalent_numbers_but_scores_frozen_actuals(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="RMSE")
            spec = load_task(task)
            ids = ["001", "NA", "03", "04", "05"]
            frozen = ["251", "1.25", "-0", "9007199254740993", "0.000000000000000000123"]
            written = ["251.0", "125e-2", "0.0", "9.007199254740993e15", "123e-21"]
            path = task.parent / "holdout.csv"
            write_csv(path, [["id", "actual", "prediction"], *zip(ids, written, frozen)])
            with patch("agent.task_spec.score_predictions", wraps=score_predictions) as score:
                self.assertEqual(score_holdout(path, spec, ids, iter(frozen)), 0)
            self.assertEqual(score.call_args.args[1], frozen)
            self.assertNotEqual(score.call_args.args[1], written)

    def test_regression_holdout_rejects_changed_values_without_float_rounding_or_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="RMSE")
            spec = load_task(task)
            path = task.parent / "holdout.csv"
            cases = [
                ("251", "251.00000000000000000000000000001"),
                ("9007199254740993", "9007199254740992"),
                ("1e-100", "0"),
            ]
            for frozen, written in cases:
                with self.subTest(frozen=frozen, written=written):
                    write_csv(path, [["id", "actual", "prediction"], ["001", written, frozen]])
                    with patch("agent.task_spec.score_predictions") as score:
                        with self.assertRaisesRegex(ValueError, "actual answers"):
                            score_holdout(path, spec, ["001"], [frozen])
                    score.assert_not_called()

    def test_regression_holdout_rejects_missing_nonfinite_or_nonnumeric_actuals(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="RMSE")
            spec = load_task(task)
            path = task.parent / "holdout.csv"
            for invalid in ("", " ", "NaN", "sNaN", "Infinity", "-inf", "missing"):
                # Invalid frozen values must fail even if the CSV repeats them.
                for frozen in ("251", invalid):
                    with self.subTest(frozen=frozen, written=invalid):
                        write_csv(path, [["id", "actual", "prediction"], ["001", invalid, "251"]])
                        with self.assertRaisesRegex(ValueError, "actual answers"):
                            score_holdout(path, spec, ["001"], [frozen])

    def test_regression_numeric_equivalence_never_normalizes_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="RMSE")
            spec = load_task(task)
            path = task.parent / "holdout.csv"
            write_csv(path, [["id", "actual", "prediction"], ["1", "251.0", "251"]])
            with self.assertRaisesRegex(ValueError, "Holdout IDs"):
                score_holdout(path, spec, ["001"], ["251"])

    def test_classification_numeric_looking_actuals_remain_distinct_literal_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="Accuracy", labels=("1", "1.0"), dtype="category")
            spec = load_task(task)
            path = task.parent / "holdout.csv"
            ids, frozen = ["001", "NA"], ["1", "1.0"]
            write_csv(path, [["id", "actual", "prediction"], *zip(ids, frozen, frozen)])
            self.assertEqual(score_holdout(path, spec, ids, frozen), 1)
            write_csv(path, [["id", "actual", "prediction"], *zip(ids, ["1.0", "1.0"], frozen)])
            with self.assertRaisesRegex(ValueError, "actual answers"):
                score_holdout(path, spec, ids, frozen)

    def test_ambiguous_probability_columns_and_task_type_are_rejected(self):
        for kwargs in (
            {"metric": "LogLoss", "labels": ("A", "B", "C"), "dtype": "category", "outputs": ("answer",)},
            {"metric": "LogLoss", "labels": ("A", "B", "C"), "dtype": "category", "outputs": ("a", "b", "c")},
            {"metric": "Accuracy", "extra": "Task type: regression"},
            {"metric": "Macro F1", "extra": 'Class order: ["0", "2"]'},
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    load_task(make_task(Path(directory), **kwargs))

    def test_multiclass_auc_requires_explicit_averaging(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="ROC AUC", labels=("A", "B", "C"),
                             dtype="category", outputs=("A", "B", "C"))
            with self.assertRaisesRegex(ValueError, "OVR"):
                load_task(task)
            task.write_text(task.read_text() + "Additional metric: Macro F1 (OVR diagnostic).\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OVR"):
                load_task(task)
            task.write_text(task.read_text().replace("ROC AUC", "ROC AUC (OVR, macro)"), encoding="utf-8")
            self.assertEqual(load_task(task).metric.name, "ROC AUC")


if __name__ == "__main__":
    unittest.main()
