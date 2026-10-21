"""Opt-in automatic history routing, without clients or model training."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_submission
import solution
from agent.research import run_agent


CONFIG = {"key_columns": ["entity"], "period_column": "period"}


class AutomaticHistoryCliTests(unittest.TestCase):
    def invoke(self, task, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["solution.py", str(task), *arguments]), \
                redirect_stdout(output), redirect_stderr(errors):
            status = solution.main()
        return status, output.getvalue(), errors.getvalue()

    def task(self, directory, metric="RMSE"):
        task = Path(directory) / "task.md"
        task.write_text(f"Task type: regression\nMetric: {metric}\n", encoding="utf-8")
        return task

    def test_offline_auto_uses_remaining_original_deadline_without_credentials_or_old_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(solution.time, "monotonic", return_value=100.0) as clock, \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client") as client, \
                    patch.object(solution, "run_agent", return_value=Path(directory) / "run") as research, \
                    patch("agent.automl.run_automl") as automl:
                def emergency(task_path, deadline):
                    self.assertEqual((task_path, deadline), (task, 190.0))
                    clock.return_value = 107.0  # Emergency preparation consumes the original allowance.
                    return None

                with patch.object(solution, "_prepare_emergency", side_effect=emergency) as prepare:
                    status, output, errors = self.invoke(task, ["--mode", "offline", "--time-budget", "90",
                                                              "--history-auto"])
            self.assertEqual(status, 0, errors)
            prepare.assert_called_once_with(task, 190.0)
            dotenv.assert_not_called()
            client.assert_not_called()
            automl.assert_not_called()
            research.assert_called_once()
            self.assertEqual(research.call_args.args, (task, None))
            self.assertEqual(set(research.call_args.kwargs), {"time_budget", "local_only", "history_auto"})
            self.assertTrue(research.call_args.kwargs["local_only"])
            self.assertTrue(research.call_args.kwargs["history_auto"])
            self.assertAlmostEqual(research.call_args.kwargs["time_budget"], 81.2)
            self.assertIn("API calls: 0", output)

    def test_implicit_auto_mode_without_environment_uses_local_v3_and_does_not_load_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client") as client, \
                    patch.object(solution, "run_agent", return_value=Path(directory) / "run") as research, \
                    patch("agent.automl.run_automl") as automl:
                status, output, errors = self.invoke(task, ["--history-auto", "--time-budget", "90"])
            self.assertEqual(status, 0, errors)
            dotenv.assert_not_called()
            client.assert_not_called()
            automl.assert_not_called()
            self.assertEqual(research.call_args.args, (task, None))
            self.assertTrue(research.call_args.kwargs["local_only"])
            self.assertTrue(research.call_args.kwargs["history_auto"])
            self.assertIn("API calls: 0", output)

    def test_explicit_llm_forwards_auto_without_enabling_local_only_or_fixed_manual_options(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            with patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client") as construct, \
                    patch.object(solution, "run_agent") as research, \
                    patch("agent.automl.run_automl") as automl:
                fake_client = construct.return_value.__enter__.return_value
                status, _, errors = self.invoke(task, ["--mode", "llm", "--history-auto", "--max-steps", "3",
                                                       "--time-budget", "90"])
            self.assertEqual(status, 0, errors)
            dotenv.assert_called_once_with()
            construct.assert_called_once_with()
            automl.assert_not_called()
            research.assert_called_once()
            self.assertEqual(research.call_args.args, (task, fake_client))
            self.assertEqual(research.call_args.kwargs["max_steps"], 3)
            self.assertTrue(research.call_args.kwargs["history_auto"])
            for name in ("local_only", "history_config", "history_weight"):
                self.assertNotIn(name, research.call_args.kwargs)

    def test_auto_mode_client_creation_failure_preserves_history_auto_on_offline_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder",
                                        "OPENAI_BASE_URL": "https://unused.invalid/v1"}, clear=True), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client", side_effect=OSError("Fixture client unavailable")), \
                    patch.object(solution, "run_agent", return_value=Path(directory) / "run") as research, \
                    patch("agent.automl.run_automl") as automl:
                status, output, errors = self.invoke(task, ["--history-auto", "--time-budget", "90"])
            self.assertEqual(status, 0, errors)
            dotenv.assert_not_called()
            automl.assert_not_called()
            research.assert_called_once()
            self.assertEqual(research.call_args.args, (task, None))
            self.assertTrue(research.call_args.kwargs["local_only"])
            self.assertTrue(research.call_args.kwargs["history_auto"])
            self.assertIn("API calls: 0", output)

    def test_incompatible_flags_fail_before_emergency_training_client_or_config_file_access(self):
        variants = [
            ["--history-config", json.dumps(CONFIG)],
            ["--history-config", "@does-not-exist.json"],
            ["--history-weight", "0.5"],  # Even the automatic default weight must not be supplied manually.
            ["--history-weight", "0"],
            ["--history-weight", "nan"],
            ["--validation", "auto"],
            ["--validation", "random"],
            ["--validation", "time", "--time-column", "period"],
            ["--time-column", "period"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            for flags in variants:
                with self.subTest(flags=flags), patch.dict(os.environ, {}, clear=True), \
                        patch.object(solution, "_prepare_emergency") as emergency, \
                        patch.object(solution, "load_local_env") as dotenv, \
                        patch.object(solution, "make_client") as client, \
                        patch.object(solution, "run_agent") as research, \
                        patch("agent.automl.run_automl") as automl:
                    status, _, errors = self.invoke(task, ["--mode", "offline", "--history-auto", *flags])
                self.assertEqual(status, 1)
                self.assertTrue(errors.strip())
                emergency.assert_not_called()
                dotenv.assert_not_called()
                client.assert_not_called()
                research.assert_not_called()
                automl.assert_not_called()

    def test_flat_archive_contains_discovery_dependencies_only_as_root_python_files(self):
        files = build_submission.archive_files()
        required = {"agent.py", "solution.py", "requirements.txt", "_alfa_agent__history_discovery.py",
                    "_alfa_agent__history_discovery_host.py", "_alfa_agent__history_schema_proposer.py"}
        self.assertLessEqual(required, set(files))
        self.assertTrue(all("/" not in name and "\\" not in name for name in files))
        self.assertTrue(all(name == "requirements.txt" or name.endswith(".py") for name in files))


class AutomaticHistoryRouterTests(unittest.TestCase):
    def task(self, directory, metric="RMSE"):
        task = Path(directory) / "task.md"
        task.write_text(f"Task type: regression\nMetric: {metric}\n", encoding="utf-8")
        return task

    def test_router_forwards_automatic_history_and_local_mode_without_manual_options(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            sentinel = Path(directory) / "run"
            with patch("agent.research_v3.run_agent_v3", return_value=sentinel) as engine:
                result = run_agent(task, None, history_auto=True, local_only=True, max_steps=2, time_budget=90)
            self.assertIs(result, sentinel)
            engine.assert_called_once_with(task, None, max_steps=2, time_budget=90,
                                           stability_splits=1, metric_recipes=False,
                                           local_only=True, history_auto=True)

    def test_router_default_keywords_are_unchanged_including_explicit_false_auto(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            client = object()
            for arguments in ({}, {"history_auto": False}):
                with self.subTest(arguments=arguments), patch("agent.research_v3.run_agent_v3") as engine:
                    run_agent(task, client, max_steps=3, time_budget=50, **arguments)
                engine.assert_called_once_with(task, client, max_steps=3, time_budget=50,
                                               stability_splits=1, metric_recipes=False)

    def test_router_rejects_nonboolean_auto_before_any_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            for value in (0, 1, "yes", None):
                with self.subTest(value=value), patch("agent.research_v3.run_agent_v3") as engine, \
                        patch("agent.research.run_agent_v2") as legacy, \
                        self.assertRaisesRegex(ValueError, "history_auto"):
                    run_agent(task, None, history_auto=value)
                engine.assert_not_called()
                legacy.assert_not_called()

    def test_router_rejects_auto_on_unsupported_metric_and_preserves_default_legacy_route(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory, "SMAPE")
            with patch("agent.research_v3.run_agent_v3") as engine, \
                    patch("agent.research.run_agent_v2") as legacy, \
                    self.assertRaisesRegex(ValueError, "supported task metric"):
                run_agent(task, None, history_auto=True)
            engine.assert_not_called()
            legacy.assert_not_called()
            client = object()
            with patch("agent.research.run_agent_v2") as legacy:
                run_agent(task, client, max_steps=4, time_budget=70)
            legacy.assert_called_once_with(task, client, max_steps=4, time_budget=70)


if __name__ == "__main__":
    unittest.main()
