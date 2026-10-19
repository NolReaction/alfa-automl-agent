import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from offline_baseline import run_baseline
from validation import validate_predictions


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class OfflineBaselineTests(unittest.TestCase):
    def test_classification_uses_binary_features_and_preserves_submission_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            inputs = task_dir / "multiclass" / "participants"
            write_csv(inputs / "data_dictionary.csv", [
                ["feature", "role", "dtype"],
                ["row_key", "id", "numeric"],
                ["constant_measurement", "feature", "numeric"],
                ["protocol", "feature", "category"],
                ["is_ftp_login", "feature", "binary"],
                ["is_sm_ips_ports", "feature", "binary"],
                ["verdict", "target", "category"],
            ])
            # All other features are constant: predicting these four classes
            # requires using both binary columns, rather than merely accepting them.
            header = ["row_key", "constant_measurement", "protocol",
                      "is_ftp_login", "is_sm_ips_ports"]
            write_csv(inputs / "train.csv", [header + ["verdict"]] + [
                [f"train-{index:03d}", 7, "tcp", (index // 2) % 2, index % 2,
                 f"class_{(index // 2) % 2}{index % 2}"]
                for index in range(160)
            ])
            ids = ["0009", "0010", "NA", "0001"]
            write_csv(inputs / "test.csv", [header] + [
                [row_id, 7, "unseen" if index % 2 else "", index // 2, index % 2]
                for index, row_id in enumerate(ids)
            ])
            write_csv(inputs / "sample_submission.csv", [
                ["row_key", "verdict"], *[[row_id, "placeholder"] for row_id in ids],
            ])

            with contextlib.redirect_stdout(io.StringIO()):
                run_baseline(task_dir)

            self.assertEqual(validate_predictions(task_dir), {"valid": True, "errors": []})
            with (task_dir / "submission.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual([row["row_key"] for row in predictions], ids)
            self.assertEqual([row["verdict"] for row in predictions],
                             ["class_00", "class_01", "class_10", "class_11"])
            report = json.loads((task_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["features"]["numeric"]),
                             {"constant_measurement", "is_ftp_login", "is_sm_ips_ports"})
            self.assertEqual(report["features"]["categorical"], ["protocol"])
            self.assertEqual(report["metric"]["name"], "Macro F1")
            self.assertGreater(report["metric"]["value"], 0.95)
            self.assertTrue(report["split"]["stratified"])
            self.assertEqual(report["api"], {"used": False, "calls": 0})


if __name__ == "__main__":
    unittest.main()
