"""Explicit history routes through v3 without changing default offline execution."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solution
from agent.research import run_agent


CONFIG = {"key_columns": ["entity"], "period_column": "period"}


class HistoryCliTests(unittest.TestCase):
    def invoke(self, task, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["solution.py", str(task), *arguments]), \
                redirect_stdout(output), redirect_stderr(errors):
            status = solution.main()
        return status, output.getvalue(), errors.getvalue()

    def test_offline_history_uses_one_deadline_and_never_loads_credentials_or_client(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "load_local_env", side_effect=AssertionError("dotenv")), \
                    patch.object(solution, "make_client", side_effect=AssertionError("client")), \
                    patch.object(solution, "run_agent", return_value=Path(directory)/"run") as research, \
                    patch("agent.automl.run_automl", side_effect=AssertionError("old solver")):
                status, output, errors = self.invoke(task, ["--mode", "offline", "--time-budget", "90",
                    "--history-config", json.dumps(CONFIG), "--history-weight", "0.25"])
            self.assertEqual(status, 0, errors)
            self.assertIn("API calls: 0", output)
            research.assert_called_once()
            positional, options = research.call_args
            self.assertEqual(positional, (task, None))
            self.assertTrue(options["local_only"])
            self.assertEqual(options["history_weight"], 0.25)
            self.assertEqual(options["history_config"]["key_columns"], ["entity"])
            self.assertGreater(options["time_budget"], 80)
            self.assertLess(options["time_budget"], 90)

    def test_history_options_reach_explicit_llm_without_enabling_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            with patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "load_local_env"), patch.object(solution, "make_client"), \
                    patch.object(solution, "run_agent") as research:
                status, _, errors = self.invoke(task, ["--mode", "llm", "--history-config", json.dumps(CONFIG)])
            self.assertEqual(status, 0, errors)
            self.assertNotIn("local_only", research.call_args.kwargs)
            self.assertEqual(research.call_args.kwargs["history_weight"], 0.5)

    def test_config_file_avoids_platform_shell_json_quoting(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            config_path = Path(directory) / "history.json"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8-sig")
            with patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "run_agent", return_value=Path(directory)/"run") as research:
                status, _, errors = self.invoke(task, ["--mode", "offline", "--history-config", "@" + str(config_path)])
            self.assertEqual(status, 0, errors)
            self.assertEqual(research.call_args.kwargs["history_config"]["period_column"], "period")

    def test_invalid_history_options_stop_before_emergency_training_or_api(self):
        variants = [
            ["--history-weight", "0.5"],
            ["--history-config", "null"],
            ["--history-config", "{}"],
            ["--history-config", "[1,2]"],
            ["--history-config", json.dumps(CONFIG), "--history-weight", "nan"],
            ["--history-config", json.dumps(CONFIG), "--history-weight", "1.1"],
            ["--history-config", json.dumps(CONFIG), "--validation", "random"],
            ["--history-config", json.dumps(CONFIG), "--validation", "time", "--time-column", "period"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            for flags in variants:
                with self.subTest(flags=flags), patch.dict(os.environ, {}, clear=True), \
                        patch.object(solution, "_prepare_emergency") as emergency, \
                        patch.object(solution, "make_client") as client, \
                        patch.object(solution, "run_agent") as research:
                    status, _, _ = self.invoke(task, ["--mode", "offline", *flags])
                self.assertEqual(status, 1)
                emergency.assert_not_called()
                client.assert_not_called()
                research.assert_not_called()

    def test_router_forwards_only_explicit_history_and_local_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            with patch("agent.research_v3.run_agent_v3") as engine:
                run_agent(task, None, history_config=CONFIG, history_weight=0.75, local_only=True)
            self.assertEqual(engine.call_args.kwargs["history_config"], CONFIG)
            self.assertEqual(engine.call_args.kwargs["history_weight"], 0.75)
            self.assertTrue(engine.call_args.kwargs["local_only"])
            with patch("agent.research_v3.run_agent_v3") as engine:
                run_agent(task, object())
            self.assertNotIn("history_config", engine.call_args.kwargs)
            self.assertNotIn("history_weight", engine.call_args.kwargs)
            self.assertNotIn("local_only", engine.call_args.kwargs)

    def test_router_rejects_nonboolean_local_mode_and_unsupported_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: SMAPE", encoding="utf-8")
            for value in (0, 1, "yes", None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "local_only"):
                    run_agent(task, None, local_only=value)
            with patch("agent.research.run_agent_v2") as legacy, self.assertRaises(ValueError):
                run_agent(task, None, local_only=True)
            legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
