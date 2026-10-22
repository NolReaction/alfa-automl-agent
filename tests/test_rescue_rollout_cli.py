"""Conditional CLI rollout policy: real dispatch, mocked solver, no fits/API."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import solution
from agent.research_v3 import run_agent_v3


class RescueRolloutCliTests(unittest.TestCase):
    def invoke(self, *arguments, mode="offline", metric="RMSE",
               emergency_clock=None, client_clock=None, client_error=False):
        clock = SimpleNamespace(value=100.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.md"
            # Deliberately no CSV files: CLI routing must use the declared
            # metric and leave temporal-policy validation to the host.
            task.write_text("Metric: " + metric + "\n", encoding="utf-8")
            environment = ({"OPENAI_API_KEY": "test-placeholder",
                            "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"} if mode == "auto" else {})
            client = object()

            def prepare(*args):
                if emergency_clock is not None:
                    clock.value = emergency_clock
                return None

            def enter():
                if client_clock is not None:
                    clock.value = client_clock
                if client_error:
                    raise RuntimeError("fixture client setup unavailable")
                return client

            def solve(task_path, actual_client, **kwargs):
                # Exercise the real strict API rejection before its state
                # construction, while successful solver calls remain mocked.
                if kwargs.get("rescue_regression") is True and kwargs["time_budget"] <= 300:
                    return run_agent_v3(task_path, actual_client, **kwargs)
                return root / "run"

            legacy_result = {"model": "fixture", "metric": {"name": "RMSE", "value": 1.0},
                             "report_path": str(root / "legacy.json")}
            output, errors = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, environment, clear=True), \
                    patch("sys.argv", ["solution.py", str(task), "--mode", mode, *arguments]), \
                    patch.object(solution.time, "monotonic", side_effect=lambda: clock.value) as monotonic, \
                    patch.object(solution, "_prepare_emergency", side_effect=prepare), \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client") as construct, \
                    patch.object(solution, "run_agent", side_effect=solve) as research, \
                    patch("agent.automl.run_automl", return_value=legacy_result) as legacy, \
                    redirect_stdout(output), redirect_stderr(errors):
                construct.return_value.__enter__.side_effect = enter
                status = solution.main()
            return {"status": status, "errors": errors.getvalue(), "output": output.getvalue(),
                    "research": research, "legacy": legacy, "client": construct, "dotenv": dotenv,
                    "clock_calls": monotonic.call_count}

    def test_default_rmse_auto_and_explicit_enable_reach_same_solver_options(self):
        for mode in ("offline", "llm"):
            with self.subTest(mode=mode):
                automatic = self.invoke("--time-budget", "600", mode=mode)
                explicit = self.invoke("--time-budget", "600", "--rescue-regression", mode=mode)
                self.assertEqual(automatic["status"], 0, automatic["errors"])
                self.assertEqual(explicit["status"], 0, explicit["errors"])
                self.assertEqual(automatic["research"].call_args.kwargs, explicit["research"].call_args.kwargs)
                self.assertIs(automatic["research"].call_args.kwargs["rescue_regression"], True)
                self.assertEqual(automatic["research"].call_args.kwargs["time_budget"], 595.0)
                self.assertEqual(automatic["clock_calls"], 2)
                self.assertEqual(explicit["clock_calls"], 2)
                automatic["legacy"].assert_not_called()

    def test_explicit_disable_retains_previous_options_in_both_modes(self):
        for mode in ("offline", "llm"):
            with self.subTest(mode=mode):
                result = self.invoke("--time-budget", "600", "--no-rescue-regression", mode=mode)
                self.assertEqual(result["status"], 0, result["errors"])
                kwargs = result["research"].call_args.kwargs
                self.assertNotIn("rescue_regression", kwargs)
                self.assertIs(kwargs["adaptive_search"], True)
                self.assertEqual(kwargs["time_budget"], 595.0)
                self.assertEqual(set(kwargs), {"time_budget", "adaptive_search",
                                              "local_only" if mode == "offline" else "max_steps"})

    def test_classification_and_other_supported_metrics_do_not_request_auto_regression(self):
        for mode in ("offline", "llm"):
            for metric in ("Macro F1", "MAE"):
                with self.subTest(mode=mode, metric=metric):
                    result = self.invoke("--time-budget", "600", mode=mode, metric=metric)
                    self.assertEqual(result["status"], 0, result["errors"])
                    kwargs = result["research"].call_args.kwargs
                    self.assertIs(kwargs["adaptive_search"], True)
                    self.assertNotIn("rescue_regression", kwargs)

    def test_short_legacy_manual_and_unknown_metric_routes_remain_unmodified(self):
        for arguments, metric in (
                (("--time-budget", "300"), "RMSE"),
                (("--time-budget", "600", "--research-search", "legacy"), "RMSE"),
                (("--time-budget", "600", "--validation", "random"), "RMSE"),
                (("--time-budget", "600"), "SMAPE")):
            with self.subTest(arguments=arguments, metric=metric):
                result = self.invoke(*arguments, metric=metric)
                self.assertEqual(result["status"], 0, result["errors"])
                result["research"].assert_not_called()
                result["legacy"].assert_called_once()
                self.assertNotIn("rescue_regression", result["legacy"].call_args.kwargs)
                self.assertNotIn("automatic_rescue", result["legacy"].call_args.kwargs)
                result["client"].assert_not_called()

    def test_auto_uses_actual_remaining_after_recovery_buffer_and_emergency(self):
        cases = ((301, None, 296.0), (350, 170.0, 275.0))
        for budget, advanced, expected in cases:
            with self.subTest(budget=budget, emergency_clock=advanced):
                result = self.invoke("--time-budget", str(budget), emergency_clock=advanced)
                self.assertEqual(result["status"], 0, result["errors"])
                kwargs = result["research"].call_args.kwargs
                self.assertEqual(kwargs["time_budget"], expected)
                self.assertIs(kwargs["adaptive_search"], True)
                self.assertNotIn("rescue_regression", kwargs)
                self.assertEqual(result["clock_calls"], 2)

    def test_auto_rechecks_remaining_after_client_creation_from_one_snapshot(self):
        result = self.invoke("--time-budget", "600", mode="llm", client_clock=450.0)
        self.assertEqual(result["status"], 0, result["errors"])
        result["client"].assert_called_once_with()
        self.assertEqual(result["research"].call_args.kwargs["time_budget"], 245.0)
        self.assertNotIn("rescue_regression", result["research"].call_args.kwargs)
        self.assertEqual(result["clock_calls"], 2)

    def test_auto_client_failure_fallback_reuses_original_deadline_and_rollback(self):
        for advanced, enabled in ((250.0, True), (450.0, False)):
            for disable in (False, True):
                with self.subTest(client_clock=advanced, disable=disable):
                    extra = ("--no-rescue-regression",) if disable else ()
                    result = self.invoke("--time-budget", "600", *extra, mode="auto",
                                         client_clock=advanced, client_error=True)
                    self.assertEqual(result["status"], 0, result["errors"])
                    kwargs = result["research"].call_args.kwargs
                    self.assertIs(kwargs["local_only"], True)
                    self.assertEqual(kwargs["time_budget"], 695.0 - advanced)
                    self.assertEqual(kwargs.get("rescue_regression", False), enabled and not disable)
                    if disable or not enabled:
                        self.assertNotIn("rescue_regression", kwargs)
                    self.assertEqual(result["clock_calls"], 2)
                    result["dotenv"].assert_not_called()
                    self.assertIn("starting offline fallback", result["output"] + result["errors"])

    def test_explicit_enable_still_rejects_insufficient_actual_budget(self):
        for mode, budget, advanced in (("offline", "301", None), ("llm", "600", 450.0)):
            with self.subTest(mode=mode):
                result = self.invoke("--time-budget", budget, "--rescue-regression",
                                     mode=mode, client_clock=advanced)
                self.assertEqual(result["status"], 1)
                self.assertIn("rescue_regression requires", result["errors"])
                self.assertIs(result["research"].call_args.kwargs["rescue_regression"], True)
                self.assertLessEqual(result["research"].call_args.kwargs["time_budget"], 300)

    def test_explicit_enable_and_rollback_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke("--rescue-regression", "--no-rescue-regression")
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

