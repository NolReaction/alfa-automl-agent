"""Platform auto mode follows only its environment; explicit offline stays offline."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solution


PLATFORM_ENV = {
    "OPENAI_API_KEY": "test-placeholder",
    "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
    "OPENAI_MODEL": "provided-test-model",
}


class ModeTests(unittest.TestCase):
    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["solution.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
            status = solution.main()
        return status, output.getvalue(), errors.getvalue()

    def test_explicit_offline_ignores_configured_api_and_uses_auto_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            result = {"model": "ExtraTreesRegressor", "metric": {"name": "RMSE", "value": 2.0},
                      "report_path": str(Path(directory) / "report.json")}
            with patch.dict(os.environ, PLATFORM_ENV, clear=True), \
                    patch("agent.automl.run_automl", return_value=result) as offline, \
                    patch.object(solution, "make_client", side_effect=AssertionError("Unexpected API")), \
                    patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected credentials")):
                status, output, errors = self.invoke([str(task), "--mode", "offline"])
            self.assertEqual(status, 0, errors)
            offline.assert_called_once_with(task, time_budget=600, validation={"strategy": "auto"})
            self.assertIn("API calls: 0", output)

    def test_auto_without_api_uses_offline_without_reading_local_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            result = {"model": "ExtraTreesRegressor", "metric": {"name": "RMSE", "value": 2.0},
                      "report_path": str(Path(directory) / "report.json")}
            for environment in ({}, {name: "  " for name in PLATFORM_ENV}):
                with self.subTest(environment=environment), \
                        patch.dict(os.environ, environment, clear=True), \
                        patch("agent.automl.run_automl", return_value=result) as offline, \
                        patch.object(solution, "make_client", side_effect=AssertionError("Unexpected API")), \
                        patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected dotenv")):
                    status, output, errors = self.invoke([str(task)])
                self.assertEqual(status, 0, errors)
                offline.assert_called_once_with(task, time_budget=600, validation={"strategy": "auto"})
                self.assertIn("API calls: 0", output)

    def test_auto_with_complete_platform_environment_starts_research_without_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Research this task", encoding="utf-8")
            with patch.dict(os.environ, PLATFORM_ENV, clear=True), \
                    patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected dotenv")), \
                    patch.object(solution, "make_client") as client, \
                    patch.object(solution, "run_agent") as research, patch("agent.automl.run_automl") as offline:
                status, _, errors = self.invoke([str(task)])
            self.assertEqual(status, 0, errors)
            research.assert_called_once_with(task, client.return_value.__enter__.return_value,
                                             max_steps=12, time_budget=10500)
            offline.assert_not_called()

    def test_auto_with_required_platform_environment_and_optional_model_starts_research(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Research this task", encoding="utf-8")
            for optional_model in (None, "   "):
                environment = {name: value for name, value in PLATFORM_ENV.items()
                               if name != "OPENAI_MODEL"}
                if optional_model is not None:
                    environment["OPENAI_MODEL"] = optional_model
                with self.subTest(model=optional_model), \
                        patch.dict(os.environ, environment, clear=True), \
                        patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected dotenv")), \
                        patch.object(solution, "make_client") as client, \
                        patch.object(solution, "run_agent") as research, patch("agent.automl.run_automl") as offline:
                    status, _, errors = self.invoke([str(task)])
                self.assertEqual(status, 0, errors)
                research.assert_called_once_with(task, client.return_value.__enter__.return_value,
                                                 max_steps=12, time_budget=10500)
                offline.assert_not_called()

    def test_auto_rejects_incomplete_platform_environment_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            for missing in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
                for absent_value in (None, "   "):
                    environment = dict(PLATFORM_ENV)
                    if absent_value is None:
                        del environment[missing]
                    else:
                        environment[missing] = absent_value
                    with self.subTest(missing=missing, absent_value=absent_value), \
                            patch.dict(os.environ, environment, clear=True), \
                            patch.object(solution, "load_local_env", side_effect=AssertionError("Unexpected dotenv")), \
                            patch.object(solution, "make_client") as client, \
                            patch("agent.automl.run_automl") as offline:
                        status, _, errors = self.invoke([str(task)])
                    self.assertEqual(status, 1)
                    self.assertIn(missing, errors)
                    self.assertNotIn(PLATFORM_ENV["OPENAI_API_KEY"], errors)
                    client.assert_not_called()
                    offline.assert_not_called()

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
        with patch.dict(os.environ, {}, clear=True), patch.object(solution, "make_client") as client, \
                patch("agent.automl.run_automl") as offline:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    status, _, errors = self.invoke(arguments)
                    self.assertEqual(status, 1)
                    self.assertIn("Error:", errors)
            client.assert_not_called()
            offline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
