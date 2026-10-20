"""The delivered ZIP must run without the team's app or API access."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from build_submission import build_archive
from validation import validate_predictions


class OfflineArchiveTests(unittest.TestCase):
    def test_extracted_archive_trains_without_app_or_api_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "solution.zip"
            names = build_archive(archive)
            self.assertFalse(any(name.startswith("app/") for name in names))
            self.assertIn("agent/automl.py", names)
            package = root / "package"
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(package)
            task = root / "task"
            inputs = task / "inputs"
            inputs.mkdir(parents=True)
            (task / "task.md").write_text("Predict outcome from inputs/*.csv.\nMetric: RMSE\n", encoding="utf-8")
            (inputs / "data_dictionary.csv").write_text(
                "feature,role,dtype\nrow_id,id,category\nx,feature,numeric\nkind,feature,category\n"
                "outcome,target,numeric\n", encoding="utf-8",
            )
            with (inputs / "train.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["row_id", "x", "kind", "outcome"])
                writer.writerows((f"t{i:03}", i, "a" if i % 2 else "b", i * 1.5 + i % 2)
                                 for i in range(80))
            (inputs / "test.csv").write_text("row_id,x,kind\n001,81,new\n000,82,a\n", encoding="utf-8")
            (inputs / "sample_submission.csv").write_text("row_id,outcome\n001,0\n000,0\n", encoding="utf-8")
            originals = {p.name: p.read_bytes() for p in inputs.iterdir()}
            env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            env.update(OPENAI_API_KEY="this-key-must-not-be-used", OPENAI_BASE_URL="http://127.0.0.1:1/v1",
                       OPENAI_MODEL="must-not-call", PYTHONIOENCODING="utf-8")
            result = subprocess.run(
                [sys.executable, str(package / "solution.py"), str(task / "task.md"), "--time-budget", "60"],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=85,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("API calls: 0", result.stdout)
            self.assertTrue(validate_predictions(task)["valid"])
            with (task / "submission.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([row["row_id"] for row in csv.DictReader(stream)], ["001", "000"])
            self.assertEqual(originals, {p.name: p.read_bytes() for p in inputs.iterdir()})
            reports = list((task / ".agent_runs").glob("*/automl_report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertIn("search", report)
            self.assertGreater(report["search"]["completed"], 0)
            candidates = report["search"]["candidates"]
            winner = next(candidate for candidate in candidates if candidate["selected"])
            values = [candidate["metric"] for candidate in candidates if candidate["status"] == "completed"]
            self.assertEqual(winner["metric"], min(values))


if __name__ == "__main__":
    unittest.main()
