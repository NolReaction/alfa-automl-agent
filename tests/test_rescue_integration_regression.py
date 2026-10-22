"""Opt-in regression host routing; no API requests or model training."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import solution
from agent import research_extensions as extensions
from agent.research import run_agent
from agent.research_v3 import BASELINE_SOURCE, run_agent_v3
from tests import test_research_extensions as extension_fixture
from tests.test_history_research import make_panel
from tests.test_research import make_regression
from tests.test_team_integration import fixture_baseline


class RescueRegressionCliTests(unittest.TestCase):
    def invoke(self, *arguments, llm=False):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch.dict(os.environ, {}, clear=True), \
                    patch("sys.argv", ["solution.py", str(task), "--mode", "llm" if llm else "offline", *arguments]), \
                    patch.object(solution, "_prepare_emergency", return_value=None) as emergency, \
                    patch.object(solution, "_offline_solution") as offline, \
                    patch.object(solution, "make_client") as construct, \
                    patch.object(solution, "load_local_env") as dotenv, \
                    patch.object(solution, "run_agent") as research, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                status = solution.main()
            return status, errors.getvalue(), offline, research, construct, dotenv, emergency

    def test_cli_default_omits_new_option_and_explicit_flag_reaches_both_modes(self):
        status, errors, offline, research, construct, dotenv, _ = self.invoke("--time-budget", "600")
        self.assertEqual(status, 0, errors)
        self.assertNotIn("rescue_regression", offline.call_args.kwargs["quality_options"])
        research.assert_not_called()
        construct.assert_not_called()
        dotenv.assert_not_called()
        for llm in (False, True):
            with self.subTest(llm=llm):
                status, errors, offline, research, construct, dotenv, _ = self.invoke(
                    "--time-budget", "600", "--rescue-regression", llm=llm)
                self.assertEqual(status, 0, errors)
                if llm:
                    self.assertTrue(research.call_args.kwargs["rescue_regression"])
                    self.assertTrue(research.call_args.kwargs["adaptive_search"])
                    self.assertNotIn("local_only", research.call_args.kwargs)
                    construct.assert_called_once_with()
                    dotenv.assert_called_once_with()
                    offline.assert_not_called()
                else:
                    self.assertEqual(offline.call_args.kwargs["quality_options"],
                                     {"adaptive_search": True, "rescue_regression": True})
                    construct.assert_not_called()
                    dotenv.assert_not_called()

    def test_cli_rejects_short_legacy_and_manual_validation_before_startup(self):
        for arguments in (("--time-budget", "300"), ("--research-search", "legacy"),
                          ("--validation", "random"), ("--validation", "grouped")):
            with self.subTest(arguments=arguments):
                status, errors, offline, research, construct, _, emergency = self.invoke(
                    "--rescue-regression", *arguments)
                self.assertEqual(status, 1)
                self.assertIn("--rescue-regression requires", errors)
                emergency.assert_not_called()
                offline.assert_not_called()
                research.assert_not_called()
                construct.assert_not_called()

    def test_router_default_preserves_kwargs_and_opt_in_is_strict_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.run_agent_v3") as execute:
                run_agent(task, None, adaptive_search=True, time_budget=600)
                self.assertNotIn("rescue_regression", execute.call_args.kwargs)
                execute.reset_mock()
                run_agent(task, None, adaptive_search=True, rescue_regression=True, time_budget=600)
                self.assertIs(execute.call_args.kwargs["rescue_regression"], True)
                self.assertIs(execute.call_args.kwargs["adaptive_search"], True)
            for value in (None, 0, 1, "true", []):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "rescue_regression must be a boolean"):
                        run_agent(task, None, rescue_regression=value)
                    with self.assertRaisesRegex(ValueError, "rescue_regression must be a boolean"):
                        run_agent_v3(task, None, rescue_regression=value)
            for adaptive, seconds in ((False, 600), (True, 300)):
                with self.subTest(adaptive=adaptive, seconds=seconds):
                    with self.assertRaisesRegex(ValueError, "rescue_regression requires"):
                        run_agent_v3(task, None, adaptive_search=adaptive,
                                     rescue_regression=True, time_budget=seconds)

    def test_v3_reports_requested_scope_and_unsupported_task_skip_without_fitting(self):
        for task_factory, requested, expected in (
                (make_panel, False, None), (make_panel, True, "pending"),
                (make_regression, True, "skipped")):
            with self.subTest(factory=task_factory.__name__, requested=requested), \
                    tempfile.TemporaryDirectory() as directory:
                task = task_factory(Path(directory))
                with patch("agent.research_v3.prepare_baseline", side_effect=fixture_baseline), \
                        patch("agent.research_v3.V3State.measure_inner", return_value=Mock(
                            metric={"name": "RMSE", "value": 1.0, "higher_is_better": False})), \
                        patch("agent.research_v3.inner_feedback", return_value={}), \
                        patch("agent.research_v3.continue_locally") as local, \
                        patch("agent.research_v3.finalize", side_effect=lambda state, reason: state), \
                        patch("agent.research_v3._request_completion", side_effect=AssertionError("No API")):
                    state = run_agent_v3(task, None, local_only=True, adaptive_search=True,
                                         rescue_regression=requested, time_budget=600)
                local.assert_called_once_with(state)
                if not requested:
                    self.assertNotIn("rescue_regression", state.report["quality_options"])
                    self.assertNotIn("rescue_regression", state.report)
                else:
                    self.assertIs(state.report["quality_options"]["rescue_regression"], True)
                    self.assertEqual(state.report["rescue_regression"]["status"], expected)
                    self.assertEqual(state.report["rescue_regression"]["time_guard_kind"], "heuristic_admission_floor")
                    self.assertFalse(state.report["rescue_regression"]["runtime_guaranteed"])
                    if expected == "skipped":
                        self.assertEqual(state.report["rescue_regression"]["reason"],
                                         "matching_numeric_time_policy_required")


class RescueRegressionFactoryTests(unittest.TestCase):
    def fixture(self, *, requested=True):
        fixture = extension_fixture.ExtensionTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        state = fixture.state
        state.adaptive_search = True
        state.gate.choice = {"strategy": "time", "time_column": "period"}
        state.gate.inner_split = {"strategy": "time", "time_column": "period"}
        if requested is not None:
            state.report["quality_options"]["rescue_regression"] = requested
        return fixture

    @staticmethod
    def valid_discovery(state):
        return {"status": "selected", "scope": "inner_fit_features_only", "identity_verified": False,
                "fit_rows": len(state.gate.fit_rows), "feature_columns": list(state.gate.research_spec.features),
                "period_column": "period", "selected_config": {
                    "key_columns": ["cat"], "period_column": "period", "numeric_columns": [],
                    "positive_columns": [], "ambiguity_policy": "past_unique_only"}}

    def isolated(self, fixture, **kwargs):
        others = {"_" + kind + "_trial": Mock(side_effect=extensions._Skip("fixture_skip"))
                  for kind in extensions.KINDS if kind != "boost_catboost"}
        with patch.multiple(extensions, **others):
            return extensions.run_research_extensions(fixture.state, fixture.assess, prioritize=True, **kwargs)

    def test_disabled_flag_keeps_identical_ordinary_factory_and_no_rescue_metadata(self):
        evidence = []
        for requested in (None, False):
            fixture = self.fixture(requested=requested)
            with patch.object(extensions, "_rescue_regression_trial", side_effect=AssertionError("Not enabled")), \
                    patch.object(extensions, "_boost_catboost_trial", return_value=(
                        BASELINE_SOURCE, extension_fixture.ordinary("catboost"), "unchanged ordinary")) as ordinary:
                report = self.isolated(fixture)
            ordinary.assert_called_once()
            frozen = fixture.assess.call_args.args[1]
            evidence.append((frozen.code_path.read_bytes(), frozen.recipe))
            self.assertEqual(report["attempted"], 1)
            self.assertEqual(report["order"], ["history", "boost_lightgbm", "boost_catboost",
                                             "frequency", "ensemble", "autogluon"])
            self.assertEqual(sum(item["kind"] == "boost_catboost" for item in report["trials"]), 1)
            self.assertNotIn("rescue_regression", fixture.state.report)
        self.assertEqual(evidence[0], evidence[1])

    def test_supported_opt_in_freezes_one_recipe_reusing_current_inner_fit_schema(self):
        fixture = self.fixture()
        discovery = self.valid_discovery(fixture.state)
        fixture.state.report["history_discovery"] = deepcopy(discovery)
        baseline_bytes = (fixture.state.baseline.code_path.read_bytes(), fixture.state.baseline.recipe_path.read_bytes())
        with patch.object(extensions, "_boost_catboost_trial", side_effect=AssertionError("Slot replaced")), \
                patch("agent.history_discovery_host.discover_history_options", side_effect=AssertionError("No rediscovery")), \
                patch("agent.regression_discovery.discover_regression_schema_bounded", side_effect=AssertionError("No new discovery")):
            report = self.isolated(fixture)
        self.assertEqual((report["attempted"], report["verified"]), (1, 1))
        self.assertEqual(len(report["trials"]), 6)
        self.assertEqual(report["order"].count("rescue_regression"), 1)
        self.assertNotIn("boost_catboost", report["order"])
        entry = next(item for item in report["trials"] if item["kind"] == "rescue_regression")
        self.assertEqual(entry["slot"], "boost_catboost")
        self.assertEqual(entry["family"], "rescue_regression_v1")
        frozen = fixture.assess.call_args.args[1]
        frozen.verify()
        self.assertEqual(frozen.recipe["family"], "rescue_regression_v1")
        self.assertEqual(frozen.recipe["base"]["engine"], "catboost")
        self.assertEqual(frozen.recipe["history_config"], discovery["selected_config"])
        self.assertEqual(frozen.recipe["history_policy"], "calibrated")
        self.assertEqual(json.loads(json.dumps(frozen.recipe, allow_nan=False)), frozen.recipe)
        self.assertEqual(fixture.assess.call_args.kwargs, {"origin": "local"})
        self.assertEqual(fixture.state.report["history_discovery"], discovery)
        self.assertEqual(baseline_bytes, (fixture.state.baseline.code_path.read_bytes(),
                                         fixture.state.baseline.recipe_path.read_bytes()))
        self.assertEqual(fixture.state.report["rescue_regression"]["history_schema"], "current_inner_fit_discovery")
        self.assertEqual(fixture.state.report["rescue_regression"]["status"], "completed")

    def test_absent_schema_uses_base_only_without_reading_prediction_rows_or_discovering(self):
        for discovery in (None, {"status": "limited", "selected_config": None}):
            with self.subTest(discovery=discovery):
                fixture = self.fixture()
                if discovery is not None:
                    fixture.state.report["history_discovery"] = discovery
                fixture.state.gate.research_train = Mock(side_effect=AssertionError("Factory must not read frame values"))
                with patch("agent.history_discovery_host.discover_history_options", side_effect=AssertionError("No rediscovery")):
                    report = self.isolated(fixture)
                self.assertEqual(report["attempted"], 1)
                self.assertIsNone(fixture.assess.call_args.args[1].recipe["history_config"])
                self.assertEqual(fixture.state.report["rescue_regression"]["history_schema"], "none_base_only")

    def test_malformed_or_wrong_provenance_schema_is_an_explicit_skip(self):
        changes = (
            {"status": "proposed"}, {"scope": "outer_fit_features"}, {"fit_rows": 999},
            {"feature_columns": ["cat", "period"]}, {"period_column": "other"},
            {"identity_verified": True},
        )
        for change in (*changes, {"selected_config": {"key_columns": ["target"], "period_column": "period"}}):
            with self.subTest(change=change):
                fixture = self.fixture()
                fixture.state.report["history_discovery"] = {**self.valid_discovery(fixture.state), **change}
                report = self.isolated(fixture)
                entry = next(item for item in report["trials"] if item["kind"] == "rescue_regression")
                self.assertEqual(entry["reason"], "rescue_regression_history_schema_invalid")
                self.assertFalse(entry["attempted"])
                fixture.assess.assert_not_called()

    def test_unsupported_tasks_keep_ordinary_catboost_and_explain_rescue_skip(self):
        cases = ("classification", "mae", "random", "mismatched", "missing_period", "nonnumeric", "not_adaptive")
        for case in cases:
            with self.subTest(case=case):
                fixture = self.fixture()
                gate = fixture.state.gate
                if case == "classification":
                    gate.spec.metric.task_type, gate.spec.metric.name = "classification", "Macro F1"
                elif case == "mae":
                    gate.spec.metric.name = "MAE"
                elif case == "random":
                    gate.choice = gate.inner_split = {"strategy": "random"}
                elif case == "mismatched":
                    gate.inner_split["time_column"] = "other"
                elif case == "missing_period":
                    gate.choice.pop("time_column")
                    gate.inner_split.pop("time_column")
                elif case == "nonnumeric":
                    gate.spec.numeric = ()
                else:
                    fixture.state.adaptive_search = False
                with patch.object(extensions, "_rescue_regression_trial", side_effect=AssertionError("Unsupported")), \
                        patch.object(extensions, "_boost_catboost_trial", return_value=(
                            BASELINE_SOURCE, extension_fixture.ordinary("catboost"), "old candidate")) as ordinary:
                    report = self.isolated(fixture)
                ordinary.assert_called_once()
                self.assertEqual(report["attempted"], 1)
                self.assertNotIn("rescue_regression", report["order"])
                self.assertEqual(fixture.state.report["rescue_regression"]["status"], "skipped")
                self.assertIsInstance(fixture.state.report["rescue_regression"]["reason"], str)

    def test_heuristic_time_floor_is_reserve_aware_and_unknown_is_not_zero(self):
        for spare, expected in ((119.999, False), (120.0, True)):
            with self.subTest(spare=spare):
                fixture = self.fixture()
                fixture.state.research_timeout.return_value = spare
                report = self.isolated(fixture)
                entry = next(item for item in report["trials"] if item["kind"] == "rescue_regression")
                self.assertIs(entry["attempted"], expected)
                self.assertEqual(fixture.state.report["rescue_regression"]["available_research_seconds"], spare)
                if not expected:
                    self.assertEqual(entry["reason"], "rescue_regression_budget")
                    fixture.assess.assert_not_called()
        for spare in (None, float("nan")):
            with self.subTest(spare=spare):
                fixture = self.fixture()
                fixture.state.report["rescue_regression"] = {}
                fixture.state.research_timeout.return_value = spare
                with self.assertRaisesRegex(extensions._Skip, "rescue_regression_budget_unknown"):
                    extensions._rescue_regression_trial(fixture.state, None)
                self.assertIsNone(fixture.state.report["rescue_regression"]["available_research_seconds"])

    def test_six_slot_cap_includes_replacement_and_stops_later_preparation(self):
        fixture = self.fixture()
        calls = []

        def ordinary(kind):
            def factory(*args):
                calls.append(kind)
                value = extension_fixture.ordinary()
                if kind == "history":
                    value.update(history_config=self.valid_discovery(fixture.state)["selected_config"], blend_weight=.5)
                return BASELINE_SOURCE, value, "bounded ordinary fixture"
            return Mock(side_effect=factory)

        factories = {"_" + kind + "_trial": ordinary(kind) for kind in extensions.KINDS}
        with patch.multiple(extensions, **factories):
            report = extensions.run_research_extensions(fixture.state, fixture.assess, prioritize=True, max_trials=6)
        self.assertEqual((report["attempted"], report["verified"]), (6, 6))
        self.assertEqual(fixture.assess.call_count, 6)
        self.assertNotIn("boost_catboost", calls)
        self.assertEqual(sum(item["kind"] == "rescue_regression" for item in report["trials"]), 1)
        self.assertEqual(len(report["trials"]), 6)

        fixture = self.fixture()
        factories = {"_" + kind + "_trial": ordinary(kind) for kind in extensions.KINDS}
        with patch.multiple(extensions, **factories), \
                patch.object(extensions, "_rescue_regression_trial", side_effect=AssertionError("No slot left")):
            report = extensions.run_research_extensions(
                fixture.state, fixture.assess, prioritize=True, max_trials=1)
        entry = next(item for item in report["trials"] if item["kind"] == "rescue_regression")
        self.assertEqual(entry["reason"], "trial_limit")
        self.assertFalse(entry["attempted"])

    def test_attempted_rescue_source_mutation_remains_fatal_after_assessment_failure(self):
        fixture = self.fixture()

        def corrupt(current, frozen, hypothesis, *, origin):
            frozen.code_path.write_text("changed frozen rescue source", encoding="utf-8")
            raise RuntimeError("training failed after corruption")

        fixture.assess.side_effect = corrupt
        with self.assertRaisesRegex(ValueError, "changed"):
            self.isolated(fixture)


if __name__ == "__main__":
    unittest.main()

