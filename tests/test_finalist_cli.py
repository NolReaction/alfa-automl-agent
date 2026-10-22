"""Experimental finalist selection stays explicit and on the local v3 route."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solution
from agent.research import run_agent
from tests.test_research import make_regression


class FinalistCliTests(unittest.TestCase):
    def invoke(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), \
                    patch("sys.argv", ["solution.py", str(task), "--mode", "offline",
                                       "--time-budget", "120", *options]), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "run_agent", return_value=Path(directory)) as research, \
                    patch.object(solution, "_offline_solution", wraps=solution._offline_solution) as dispatch, \
                    patch("agent.automl.run_automl", return_value={
                        "model": "baseline", "metric": {"name": "RMSE", "value": 1},
                        "report_path": "report.json"}) as legacy, \
                    patch.object(solution, "make_client") as client, \
                    patch.object(solution, "load_local_env") as dotenv, \
                    redirect_stdout(output), redirect_stderr(output):
                status = solution.main()
            return status, output.getvalue(), research, legacy, dispatch, client, dotenv

    def test_default_does_not_enable_new_selection_or_change_short_offline_route(self):
        status, error, research, legacy, dispatch, client, dotenv = self.invoke()
        self.assertEqual(status, 0, error)
        self.assertNotIn("finalist_selection", dispatch.call_args.kwargs["quality_options"])
        research.assert_not_called()
        legacy.assert_called_once()
        client.assert_not_called()
        dotenv.assert_not_called()

    def test_explicit_selection_uses_local_v3_even_for_short_or_legacy_search(self):
        for splits in (2, 3):
            with self.subTest(splits=splits):
                status, error, research, legacy, _, client, dotenv = self.invoke(
                    "--finalist-selection", "--stability-splits", str(splits),
                    "--research-search", "legacy")
                self.assertEqual(status, 0, error)
                options = research.call_args.kwargs
                self.assertTrue(options["local_only"])
                self.assertTrue(options["finalist_selection"])
                self.assertEqual(options["stability_splits"], splits)
                self.assertGreater(options["time_budget"], 0)
                self.assertLessEqual(options["time_budget"], 120)
                self.assertIsNone(research.call_args.args[1])
                legacy.assert_not_called()
                client.assert_not_called()
                dotenv.assert_not_called()

    def test_missing_repeats_or_manual_validation_refuses_before_training(self):
        for options in (("--finalist-selection",),
                        ("--finalist-selection", "--stability-splits", "2", "--validation", "random"),
                        ("--finalist-selection", "--stability-splits", "2", "--validation", "time",
                         "--time-column", "year")):
            with self.subTest(options=options):
                status, _, research, legacy, _, client, _ = self.invoke(*options)
                self.assertEqual(status, 1)
                research.assert_not_called()
                legacy.assert_not_called()
                client.assert_not_called()

    def test_router_requires_strict_boolean_and_paired_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as execute:
                run_agent(task, None, local_only=True, finalist_selection=True, stability_splits=3)
                self.assertTrue(execute.call_args.kwargs["finalist_selection"])
                self.assertEqual(execute.call_args.kwargs["stability_splits"], 3)
                for value in (1, "true", None):
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                        run_agent(task, None, finalist_selection=value, stability_splits=3)
                with self.assertRaisesRegex(ValueError, "two or three"):
                    run_agent(task, None, finalist_selection=True)
                task.write_text("Metric: AUCPR", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "supported task metric"):
                    run_agent(task, None, finalist_selection=True, stability_splits=2)


if __name__ == "__main__":
    unittest.main()
