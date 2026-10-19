import csv
from pathlib import Path
import tempfile
import unittest

from validation import validate_predictions


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def make_task(root: Path, dtype: str = "numeric") -> None:
    folder = root / "nested" / "participants"
    write_csv(folder / "sample_submission.csv", [
        ["row_key", "answer"], ["a", "0"], ["b", "0"],
    ])
    write_csv(folder / "data_dictionary.csv", [
        ["feature", "role", "dtype", "description_ru"],
        ["row_key", "id", "category", "Row ID"],
        ["answer", "target", dtype, "Target\nwith a multiline description"],
    ])
    write_csv(folder / "test.csv", [
        ["row_key", "feature"], ["a", "quoted\nmultiline value"], ["b", "value"],
    ])
    write_csv(folder / "train.csv", [
        ["row_key", "feature", "answer"],
        ["train-a", "value", "Rare\nclass" if dtype == "category" else "1.0"],
        ["train-b", "value", "Common" if dtype == "category" else "2.0"],
    ])


class ValidationTests(unittest.TestCase):
    def test_valid_numeric_output_and_ignored_experiment_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root)
            write_csv(root / ".agent_runs" / "old" / "sample_submission.csv", [["ignored"]])
            write_csv(root / ".offline-baseline-interrupted" / "inputs" / "sample_submission.csv",
                      [["unfinished temporary baseline"]])
            write_csv(root / "submission.csv", [["row_key", "answer"], ["a", "1.5"], ["b", "-2e3"]])
            self.assertEqual(validate_predictions(root), {"valid": True, "errors": []})

    def test_wrong_id_order_and_nonfinite_numeric_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root)
            write_csv(root / "submission.csv", [["row_key", "answer"], ["b", "NaN"], ["a", "1.0"]])
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("finite number" in error for error in result["errors"]))
            self.assertTrue(any("order" in error for error in result["errors"]))

    def test_known_multiline_class_is_valid_but_unknown_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root, "category")
            write_csv(root / "submission.csv", [
                ["row_key", "answer"], ["a", "Rare\nclass"], ["b", "Common"],
            ])
            self.assertTrue(validate_predictions(root)["valid"])
            write_csv(root / "submission.csv", [
                ["row_key", "answer"], ["a", "Unknown"], ["b", "Common"],
            ])
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("unknown class" in error for error in result["errors"]))

    def test_extra_fields_are_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root)
            write_csv(root / "submission.csv", [
                ["row_key", "answer"], ["a", "1.0", "extra"], ["b", "2.0"],
            ])
            result = validate_predictions(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("fields" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
