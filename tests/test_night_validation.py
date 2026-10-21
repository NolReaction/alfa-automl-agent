"""Minimal regressions for task-local CSV validation without false success."""

import csv
from pathlib import Path
import tempfile
import unittest

from agent.task_spec import load_task
from validation import validate_predictions


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def make_task(root, *, description=None, classification=False):
    data = root / "data"
    dictionary = [["feature", "role", "dtype"], ["ид", "id", "category"],
                  ["x", "feature", "numeric"],
                  ["ответ", "target", "category" if classification else "numeric"]]
    write_csv(data / "data_dictionary.csv", dictionary)
    labels = ["01", "NA", "class c", "класс"] if classification else ["-1", "2", "3", "4"]
    write_csv(data / "train.csv", [["ид", "x", "ответ"],
                                   *[[f"train-{i}", str(i), label] for i, label in enumerate(labels)]])
    ids = ["NA", "001", " β id "]
    write_csv(data / "test.csv", [["ид", "x"], *[[value, str(i)] for i, value in enumerate(ids)]])
    write_csv(data / "sample_submission.csv", [["ответ", "ид"],
                                               *[[labels[0] if classification else "0", value] for value in ids]])
    write_csv(root / "submission.csv", [["ответ", "ид"],
                                        *[[labels[i] if classification else str(i - 1), value]
                                          for i, value in enumerate(ids)]])
    if description is not None:
        (root / "task.md").write_text(description, encoding="utf-8")
    return data, dictionary, ids


class NightValidationTests(unittest.TestCase):
    def test_legacy_rejects_empty_test_and_empty_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _, _ = make_task(root)
            write_csv(data / "test.csv", [["ид", "x"]])
            write_csv(root / "submission.csv", [["ответ", "ид"]])
            self.assertFalse(validate_predictions(root)["valid"])

    def test_legacy_rejects_template_prediction_without_declared_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _, ids = make_task(root)
            rows = [["prediction", "ид"], *[["not-a-number", value] for value in ids]]
            write_csv(data / "sample_submission.csv", rows)
            write_csv(root / "submission.csv", rows)
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("target" in item.lower() for item in result["errors"]))

    def test_legacy_rejects_duplicate_dictionary_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, dictionary, _ = make_task(root)
            write_csv(data / "data_dictionary.csv", [*dictionary, ["ответ", "target", "numeric"]])
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("duplicate" in item.lower() for item in result["errors"]))

    def test_legacy_requires_dictionary_schema_and_supported_target_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, dictionary, ids = make_task(root)
            write_csv(data / "data_dictionary.csv", [["feature", "role"], ["ид", "id"],
                                                     ["ответ", "target"]])
            self.assertFalse(validate_predictions(root)["valid"])
            dictionary[-1][-1] = "unknown"
            write_csv(data / "data_dictionary.csv", dictionary)
            write_csv(root / "submission.csv", [["ответ", "ид"], *[["anything", value] for value in ids]])
            self.assertFalse(validate_predictions(root)["valid"])

    def test_legacy_requires_explicit_dictionary_id_and_target(self):
        for role in ("id", "target"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, dictionary, ids = make_task(root)
                rows = [["ид", "ответ"], *[[value, "1"] for value in ids]]
                write_csv(data / "sample_submission.csv", rows)
                write_csv(root / "submission.csv", rows)
                write_csv(data / "data_dictionary.csv", [dictionary[0],
                          *[row for row in dictionary[1:] if row[1] != role]])
                self.assertFalse(validate_predictions(root)["valid"])

    def test_known_metric_rejects_conflicting_task_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root, description="Metric: RMSE\nTask type: classification\n")
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("conflict" in item.lower() for item in result["errors"]))

    def test_conflicting_and_ambiguous_metrics_cannot_downgrade_to_structural_success(self):
        for description in ("Metric: RMSE\nPrimary metric: MAE\n", "Metric: RMSE or MAE\n"):
            with self.subTest(description=description), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_task(root, description=description)
                self.assertFalse(validate_predictions(root)["valid"])

    def test_unknown_primary_cannot_hide_a_second_primary_declaration(self):
        descriptions = (
            "Metric: RMSLE\nPrimary metric: RMSE\n",
            "Metric: RMSE\nPrimary metric: RMSLE\n",
            "Metric:\nRMSLE\nPrimary metric:\nRMSE\n",
            "Metric: RMSLE\nPrimary metric: MAPE\n",
        )
        for description in descriptions:
            with self.subTest(description=description), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_task(root, description=description)
                self.assertFalse(validate_predictions(root)["valid"])

    def test_spec_path_reports_missing_test_id_column_instead_of_raising_keyerror(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _, _ = make_task(root, description="Metric: RMSE\n")
            spec = load_task(root / "task.md")
            write_csv(data / "test.csv", [["x"], ["1"], ["2"], ["3"]])
            result = validate_predictions(root, spec=spec)
            self.assertFalse(result["valid"])
            self.assertTrue(any("ID" in item for item in result["errors"]))

    def test_spec_path_rejects_changed_empty_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _, _ = make_task(root, description="Metric: RMSE\n")
            spec = load_task(root / "task.md")
            write_csv(data / "test.csv", [["ид", "x"]])
            write_csv(root / "submission.csv", [["ответ", "ид"]])
            self.assertFalse(validate_predictions(root, spec=spec)["valid"])

    def test_literal_labels_ids_and_template_column_order_remain_supported(self):
        for description in (None, "Primary metric: Macro F1\n"):
            with self.subTest(description=description), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_task(root, description=description, classification=True)
                self.assertEqual(validate_predictions(root), {"valid": True, "errors": []})

    def test_unknown_or_absent_metric_keeps_legacy_structural_validation(self):
        for description in (None, "Predict the requested continuous value.\n", "Metric: RMSLE\n",
                            "Metric:\nRMSLE\n",
                            "Metric: RMSLE\n```text\nPrimary metric: RMSE\n```\n",
                            "Metric: RMSLE\n<!-- Primary metric: RMSE -->\n"):
            with self.subTest(description=description), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_task(root, description=description)
                self.assertEqual(validate_predictions(root), {"valid": True, "errors": []})


if __name__ == "__main__":
    unittest.main()
