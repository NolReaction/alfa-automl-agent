"""The explicit offline history path runs from the platform's root-only files."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_submission import archive_files
from tests.test_history_research import CONFIG, TEST_IDS, make_panel
from validation import validate_predictions


class HistoryFlatTests(unittest.TestCase):
    def test_flat_cli_runs_real_startup_search_history_and_full_refit_without_api(self):
        with tempfile.TemporaryDirectory(prefix="history-flat-") as directory:
            root = Path(directory)
            sources = archive_files()
            for name, content in sources.items():
                (root / name).write_bytes(content)
            self.assertIn("_alfa_agent__history_runtime.py", sources)
            self.assertIn("_alfa_agent__history_candidate.py", sources)
            self.assertTrue(all("/" not in name for name in sources))
            task = make_panel(root)
            config_file = root / "history.json"
            config_file.write_text(json.dumps(CONFIG), encoding="utf-8")
            inputs = root / "regression" / "participants"
            before = {path.name: path.read_bytes() for path in inputs.iterdir()}
            env = {name: value for name, value in os.environ.items()
                   if name not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"}}
            env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       OPENAI_API_KEY="offline-history-must-not-be-used",
                       OPENAI_BASE_URL="http://127.0.0.1:1/v1")
            result = subprocess.run([sys.executable, "solution.py", "task.md", "--mode", "offline",
                "--history-config", "@history.json", "--time-budget", "90"], cwd=root, env=env,
                capture_output=True, text=True, encoding="utf-8", timeout=115)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("API calls: 0", result.stdout)
            self.assertTrue(validate_predictions(task.parent)["valid"])
            reports = list(root.glob(".agent_runs/*/run.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "local_only")
            self.assertEqual(report["api"]["attempts"], 0)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertEqual(report["history_research"]["attempted"], 1)
            self.assertEqual(report["history_research"]["status"], "completed")
            self.assertEqual(report["baseline_search"]["status"], "searched")
            self.assertTrue(report["publication"]["new_result_published"])
            self.assertEqual(report["reason"], "local_only")
            receipts = report["candidate_receipts"]
            outer = json.loads(Path(receipts["outer"]).read_text(encoding="utf-8"))
            refit = json.loads(Path(receipts["refit"]).read_text(encoding="utf-8"))
            for key in ("code_hash", "recipe_hash"):
                self.assertEqual(outer[key], refit[key])
                self.assertEqual(refit[key], receipts[key])
            with (root / "submission.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([row["row_key"] for row in csv.DictReader(stream)], TEST_IDS)
            self.assertEqual(before, {path.name: path.read_bytes() for path in inputs.iterdir()})
            self.assertEqual(sources, {name: (root / name).read_bytes() for name in sources})
            self.assertFalse((root / "agent").exists())
            self.assertFalse((root / "app").exists())


if __name__ == "__main__":
    unittest.main()
