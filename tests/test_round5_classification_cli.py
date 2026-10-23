"""Real CLI/dispatcher routing with mocked solvers: no models or API calls."""
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import solution
from agent.research import run_agent


class Round5ClassificationCliTests(unittest.TestCase):
    def invoke(self, *arguments, mode="offline", metric="Macro F1", client_error=False,
               emergency_clock=None, client_clock=None, v3_impl=None):
        clock = SimpleNamespace(value=100.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.md"
            task.write_text("Metric: " + metric + "\n", encoding="utf-8")
            environment = ({"OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}
                           if mode == "auto" else {})
            client = object()
            def prepare(*_):
                if emergency_clock is not None:
                    clock.value = emergency_clock
                return None
            def enter():
                if client_clock is not None:
                    clock.value = client_clock
                if client_error:
                    raise RuntimeError("fixture client setup unavailable")
                return client
            legacy_result = {"model": "fixture", "metric": {"name": metric, "value": .5},
                             "report_path": str(root / "legacy.json")}
            output, errors = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, environment, clear=True), \
                    patch("sys.argv", ["solution.py", str(task), "--mode", mode, *arguments]), \
                    patch.object(solution.time, "monotonic", side_effect=lambda: clock.value) as monotonic, \
                    patch.object(solution, "_prepare_emergency", side_effect=prepare) as emergency, \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "make_client") as construct, \
                    patch("agent.research_v3.run_agent_v3", return_value=root / "run", side_effect=v3_impl) as v3, \
                    patch("agent.research.run_agent_v2", return_value=root / "run") as v2, \
                    patch("agent.automl.run_automl", return_value=legacy_result) as automl, \
                    redirect_stdout(output), redirect_stderr(errors):
                construct.return_value.__enter__.side_effect = enter
                status = solution.main()
            return {"status": status, "errors": errors.getvalue(), "output": output.getvalue(), "v3": v3, "v2": v2,
                    "automl": automl, "client": construct, "dotenv": dotenv, "emergency": emergency,
                    "clock_calls": monotonic.call_count}

    def test_default_and_explicit_off_keep_exact_supported_v3_calls(self):
        for mode in ("offline", "llm", "auto"):
            with self.subTest(mode=mode):
                default = self.invoke("--time-budget", "600", mode=mode)
                explicit = self.invoke("--time-budget", "600", "--classification-round4-mode", "off", mode=mode)
                for result in (default, explicit):
                    self.assertEqual(result["status"], 0, result["errors"])
                    result["v3"].assert_called_once()
                    result["automl"].assert_not_called()
                    result["v2"].assert_not_called()
                    self.assertNotIn("classification_round4_mode", result["v3"].call_args.kwargs)
                    self.assertEqual(result["clock_calls"], 2)
                expected = {"max_steps": 12, "time_budget": 595., "stability_splits": 1,
                            "metric_recipes": False, "adaptive_search": True}
                if mode == "offline":
                    expected["local_only"] = True
                self.assertEqual(default["v3"].call_args.kwargs, expected)
                self.assertEqual(explicit["v3"].call_args.kwargs, expected)

    def test_off_keeps_short_manual_legacy_and_unknown_metric_routes(self):
        for arguments, metric, mode, solver in (
                (("--time-budget", "300"), "Macro F1", "offline", "automl"),
                (("--time-budget", "600", "--research-search", "legacy"), "Macro F1", "offline", "automl"),
                (("--time-budget", "600", "--validation", "grouped"), "Macro F1", "offline", "automl"),
                (("--time-budget", "600"), "SMAPE", "offline", "automl"),
                (("--time-budget", "600"), "SMAPE", "llm", "v2")):
            with self.subTest(arguments=arguments, metric=metric, mode=mode):
                default = self.invoke(*arguments, metric=metric, mode=mode)
                explicit = self.invoke(*arguments, "--classification-round4-mode", "off", metric=metric, mode=mode)
                for result in (default, explicit):
                    self.assertEqual(result["status"], 0, result["errors"])
                    result[solver].assert_called_once()
                    self.assertNotIn("classification_round4_mode", result[solver].call_args.kwargs)
                self.assertEqual(default[solver].call_args.kwargs, explicit[solver].call_args.kwargs)

    def test_on_reaches_real_dispatcher_and_v3_without_offline_api_setup(self):
        result = self.invoke("--time-budget", "3600", "--classification-round4-mode", "on")
        self.assertEqual(result["status"], 0, result["errors"])
        result["v3"].assert_called_once()
        self.assertEqual(result["v3"].call_args.kwargs, {
            "max_steps": 12, "time_budget": 3595., "stability_splits": 1, "metric_recipes": False,
            "local_only": True, "adaptive_search": True, "classification_round4_mode": "on"})
        result["client"].assert_not_called()
        result["dotenv"].assert_not_called()
        result["automl"].assert_not_called()

    def test_real_cli_on_binds_and_validates_actual_v3_before_state_construction(self):
        from agent.research_v3 import run_agent_v3
        self.assertEqual(inspect.signature(run_agent_v3).parameters["classification_round4_mode"].default, "off")
        # A sentinel stops at state construction, after the actual v3 signature
        # and option validation have run. No task CSV, supervisor or fit exists.
        with patch("agent.research_v3.V3State", side_effect=RuntimeError("validated_actual_v3_boundary")) as state:
            result = self.invoke("--time-budget", "600", "--classification-round4-mode", "on", v3_impl=run_agent_v3)
        self.assertEqual(result["status"], 1)
        self.assertIn("validated_actual_v3_boundary", result["errors"])
        state.assert_called_once()
        self.assertIs(state.call_args.kwargs["adaptive_search"], True)
        result["client"].assert_not_called()
        result["automl"].assert_not_called()

    def test_explicit_llm_on_uses_same_dispatcher_without_local_only(self):
        result = self.invoke("--time-budget", "600", "--classification-round4-mode", "on", mode="llm")
        self.assertEqual(result["status"], 0, result["errors"])
        result["v3"].assert_called_once()
        self.assertEqual(result["v3"].call_args.kwargs["classification_round4_mode"], "on")
        self.assertNotIn("local_only", result["v3"].call_args.kwargs)
        result["client"].assert_called_once_with()
        result["dotenv"].assert_called_once_with()
        result["automl"].assert_not_called()

    def test_on_rejects_incompatible_nominal_cli_routes_before_any_preparation(self):
        for arguments, metric in (
                (("--time-budget", "300"), "Macro F1"),
                (("--time-budget", "600", "--research-search", "legacy"), "Macro F1"),
                (("--time-budget", "600", "--validation", "random"), "Macro F1"),
                (("--time-budget", "600", "--validation", "grouped"), "Macro F1"),
                (("--time-budget", "600", "--validation", "time", "--time-column", "period"), "Macro F1"),
                (("--time-budget", "600"), "SMAPE")):
            with self.subTest(arguments=arguments, metric=metric):
                result = self.invoke(*arguments, "--classification-round4-mode", "on", metric=metric)
                self.assertEqual(result["status"], 1)
                self.assertIn("--classification-round4-mode on requires", result["errors"])
                result["emergency"].assert_not_called()
                result["v3"].assert_not_called()
                result["automl"].assert_not_called()
                result["v2"].assert_not_called()
                result["client"].assert_not_called()

    def test_on_survives_auto_client_failure_without_budget_reset(self):
        result = self.invoke("--time-budget", "600", "--classification-round4-mode", "on", mode="auto",
                             client_error=True, client_clock=450.)
        self.assertEqual(result["status"], 0, result["errors"])
        options = result["v3"].call_args.kwargs
        self.assertEqual(options["classification_round4_mode"], "on")
        self.assertEqual(options["time_budget"], 245.)
        self.assertIs(options["local_only"], True)
        self.assertEqual(result["clock_calls"], 2)
        result["dotenv"].assert_not_called()
        result["automl"].assert_not_called()

    def test_preparation_cost_remains_inside_original_on_budget(self):
        result = self.invoke("--time-budget", "600", "--classification-round4-mode", "on", emergency_clock=450.)
        self.assertEqual(result["status"], 0, result["errors"])
        # Nominal adaptive eligibility is preserved; the host decides whether
        # the depleted 245 seconds permit a probe/OOF, without a new budget.
        self.assertEqual(result["v3"].call_args.kwargs["time_budget"], 245.)
        self.assertEqual(result["v3"].call_args.kwargs["classification_round4_mode"], "on")
        self.assertEqual(result["clock_calls"], 2)

    def test_classification_option_does_not_disable_existing_regression_rescue(self):
        for mode in ("off", "on"):
            for rescue in ((), ("--rescue-regression",), ("--no-rescue-regression",)):
                with self.subTest(mode=mode, rescue=rescue):
                    result = self.invoke("--time-budget", "600", "--classification-round4-mode", mode,
                                         *rescue, metric="RMSE")
                    self.assertEqual(result["status"], 0, result["errors"])
                    options = result["v3"].call_args.kwargs
                    self.assertEqual(options.get("rescue_regression", False), rescue != ("--no-rescue-regression",))
                    self.assertEqual(options.get("classification_round4_mode", "off"), mode)
                    self.assertEqual(result["clock_calls"], 2)

    def test_invalid_cli_mode_fails_before_emergency_or_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: Macro F1\n")
            with patch("sys.argv", ["solution.py", str(task), "--classification-round4-mode", "true"]), \
                    patch.object(solution, "_prepare_emergency") as emergency, \
                    patch.object(solution, "run_agent") as agent, redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                solution.main()
            self.assertEqual(error.exception.code, 2)
            emergency.assert_not_called()
            agent.assert_not_called()

    def test_help_exposes_the_real_flag_without_starting_work(self):
        output = io.StringIO()
        with patch("sys.argv", ["solution.py", "--help"]), patch.object(solution, "_prepare_emergency") as emergency, \
                redirect_stdout(output), self.assertRaises(SystemExit) as error:
            solution.main()
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--classification-round4-mode {off,on}", output.getvalue())
        emergency.assert_not_called()


class Round5ClassificationDispatcherTests(unittest.TestCase):
    def test_actual_v3_rejects_invalid_mode_before_constructing_state(self):
        from agent.research_v3 import run_agent_v3
        with patch("agent.research_v3.V3State") as state:
            for value in (True, False, None, 0, 1, "", "ON", [], {}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "classification_round4_mode"):
                    run_agent_v3(Path("not-a-task.md"), None, classification_round4_mode=value)
            state.assert_not_called()

    def test_python_default_and_explicit_off_preserve_the_v3_keyword_contract(self):
        self.assertEqual(inspect.signature(run_agent).parameters["classification_round4_mode"].default, "off")
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: Macro F1\n")
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as v3:
                run_agent(task, None, time_budget=600, adaptive_search=True)
                original = v3.call_args
                run_agent(task, None, time_budget=600, adaptive_search=True, classification_round4_mode="off")
                self.assertEqual(v3.call_args, original)
                self.assertNotIn("classification_round4_mode", original.kwargs)

    def test_enum_validation_precedes_task_access_and_does_not_accept_truthiness(self):
        for value in (True, False, None, 0, 1, "", "ON", " off", [], {}):
            with self.subTest(value=value), patch("agent.research_v3.run_agent_v3") as v3, \
                    self.assertRaisesRegex(ValueError, "classification_round4_mode"):
                run_agent(Path("not-a-task.md"), None, classification_round4_mode=value)
            v3.assert_not_called()

    def test_unknown_metric_preserves_off_v2_route_but_rejects_on(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: SMAPE\n")
            with patch("agent.research.run_agent_v2", return_value=Path(directory)) as v2, \
                    patch("agent.research_v3.run_agent_v3") as v3:
                run_agent(task, None, time_budget=90)
                previous = v2.call_args
                run_agent(task, None, time_budget=90, classification_round4_mode="off")
                self.assertEqual(v2.call_args, previous)
                with self.assertRaisesRegex(ValueError, "supported task metric"):
                    run_agent(task, None, time_budget=90, classification_round4_mode="on")
                self.assertEqual(v2.call_count, 2)
                v3.assert_not_called()

    def test_on_forwards_supported_task_eligibility_to_host_without_changing_other_options(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE\n")
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as v3:
                run_agent(task, None, time_budget=600, adaptive_search=True, rescue_regression=True,
                          classification_round4_mode="on", local_only=True)
                self.assertEqual(v3.call_args.kwargs, {"max_steps": 12, "time_budget": 600, "stability_splits": 1,
                    "metric_recipes": False, "local_only": True, "adaptive_search": True,
                    "rescue_regression": True, "classification_round4_mode": "on"})

    def test_direct_python_on_leaves_adaptive_eligibility_to_host(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: Macro F1\n")
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as v3:
                run_agent(task, None, time_budget=90, classification_round4_mode="on")
                self.assertEqual(v3.call_args.kwargs["classification_round4_mode"], "on")
                self.assertNotIn("adaptive_search", v3.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
