"""Contest routing changes only long supported automatic-validation runs."""

from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import solution
from agent.research import run_agent
from tests.test_research import make_regression


class AdaptiveCliTests(unittest.TestCase):
    def run_offline(self, *arguments, unsupported=False, task_text=None):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            if unsupported:
                task.write_text("Metric: AUCPR\nUse train.csv and test.csv; write submission.csv.", encoding="utf-8")
            if task_text is not None:
                task.write_text(task_text, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), \
                    patch("sys.argv", ["solution.py", str(task), "--mode", "offline", *arguments]), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "_offline_solution") as offline, \
                    patch.object(solution, "make_client") as client, \
                    patch.object(solution, "load_local_env") as dotenv, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(solution.main(), 0)
                client.assert_not_called()
                dotenv.assert_not_called()
                return offline.call_args.kwargs

    def test_default_contest_budget_selects_adaptive_without_reading_credentials(self):
        started = time.monotonic()
        options = self.run_offline()
        self.assertTrue(options["quality_options"]["adaptive_search"])
        self.assertLessEqual(options["deadline"] - started, 10500)

    def test_explicit_auto_validation_has_the_same_long_run_behavior(self):
        options = self.run_offline("--time-budget", "360", "--validation", "auto")
        self.assertEqual(options["validation"], {"strategy": "auto"})
        self.assertTrue(options["quality_options"]["adaptive_search"])

    def test_short_budgets_legacy_request_and_manual_split_keep_original_route(self):
        for arguments in (("--time-budget", "300"),
                          ("--time-budget", "360", "--research-search", "legacy"),
                          ("--time-budget", "360", "--validation", "random"),
                          ("--time-budget", "360", "--validation", "grouped")):
            with self.subTest(arguments=arguments):
                self.assertNotIn("adaptive_search", self.run_offline(*arguments)["quality_options"])

    def test_unknown_metric_is_not_sent_to_the_v3_adaptive_protocol(self):
        self.assertNotIn("adaptive_search", self.run_offline("--time-budget", "360", unsupported=True)["quality_options"])

    def test_missing_or_conflicting_metric_does_not_block_the_existing_solver_route(self):
        # Selection of an optional coordinator is not task validation. Legacy
        # parsing and output acceptance remain responsible for the task itself.
        for task_text in ("Research this task", "Primary metric: RMSE\nPrimary metric: MAE"):
            with self.subTest(task_text=task_text):
                options = self.run_offline("--time-budget", "360", task_text=task_text)
                self.assertNotIn("adaptive_search", options["quality_options"])

    def test_adaptive_offline_dispatches_to_local_v3_without_api_client(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch.object(solution, "run_agent", return_value=Path(directory)) as execute, \
                    patch("agent.automl.run_automl") as legacy, redirect_stdout(io.StringIO()):
                solution._offline_solution(task, deadline=time.monotonic() + 350,
                    validation={"strategy": "auto"}, quality_options={"adaptive_search": True})
            self.assertEqual(execute.call_args.args, (task, None))
            self.assertTrue(execute.call_args.kwargs["local_only"])
            self.assertTrue(execute.call_args.kwargs["adaptive_search"])
            self.assertLessEqual(execute.call_args.kwargs["time_budget"], 350)
            legacy.assert_not_called()

    def test_router_forwards_adaptive_flag_and_rejects_non_boolean_values(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as execute:
                run_agent(task, None, local_only=True, adaptive_search=True, time_budget=360)
            self.assertTrue(execute.call_args.kwargs["adaptive_search"])
            for value in (1, "true", None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    run_agent(task, None, adaptive_search=value)


if __name__ == "__main__":
    unittest.main()
