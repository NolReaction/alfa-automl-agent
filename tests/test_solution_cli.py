"""Mode selection must never spend API budget implicitly."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solution


class ModeTests(unittest.TestCase):
    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["solution.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
            status = solution.main()
        return status, output.getvalue(), errors.getvalue()

    def test_default_mode_ignores_configured_api_and_uses_auto_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            result = {"model": "ExtraTreesRegressor", "metric": {"name": "RMSE", "value": 2.0},
                      "report_path": str(Path(directory) / "report.json")}
            with patch("agent.automl.run_automl", return_value=result) as offline, \
                    patch.object(solution, "make_client", side_effect=AssertionError("Unexpected API")), \
                    patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected credentials")):
                status, output, errors = self.invoke([str(task)])
            self.assertEqual(status, 0, errors)
            offline.assert_called_once_with(task, time_budget=600, validation={"strategy": "auto"})
            self.assertIn("API calls: 0", output)

    def test_explicit_llm_preserves_research_and_default_time_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Research this task", encoding="utf-8")
            with patch.object(solution, "load_local_env"), patch.object(solution, "make_client") as client, \
                    patch.object(solution, "run_agent") as research, patch("agent.automl.run_automl") as offline:
                status, _, errors = self.invoke([str(task), "--mode", "llm"])
            self.assertEqual(status, 0, errors)
            research.assert_called_once_with(task, client.return_value.__enter__.return_value,
                                             max_steps=12, time_budget=10500)
            offline.assert_not_called()

    def test_unsupported_or_incomplete_validation_does_not_start_training(self):
        invalid = [
            ["--mode", "llm", "--validation", "random"],
            ["--validation", "time"],
            ["--validation", "random", "--time-column", "period"],
            ["--time-budget", "nan"],
            ["--time-budget", "inf"],
        ]
        with patch.object(solution, "make_client") as client, patch("agent.automl.run_automl") as offline:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    status, _, errors = self.invoke(arguments)
                    self.assertEqual(status, 1)
                    self.assertIn("Error:", errors)
            client.assert_not_called()
            offline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
