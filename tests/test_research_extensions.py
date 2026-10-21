"""Host extension admission and failure isolation, with no model fitting."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from agent import research_extensions as extensions
from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.research_candidate import freeze_candidate
from agent.research_v3 import BASELINE_SOURCE


def ordinary(action="linear", **features):
    return {"action": action, "model_config": make_model_config(action, "regression"),
            "feature_config": normalize_feature_config(features)}


def feedback(*, accepted=False, reason=None):
    return {"verified": True, "accepted": accepted, "reason": reason or ("improved" if accepted else "not_better"),
            "metric": {"name": "RMSE", "value": 1.0, "higher_is_better": False}, "scope": "inner_selection"}


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = SimpleNamespace(
            report={"stage": "inner_research", "quality_options": {"adaptive_search": True}, "extensions": {}},
            dialogue_closed=False, run_dir=self.root, baseline_inner={"value": 2.0}, selected_candidate=None,
            check_source=Mock(), save=Mock(), event=Mock(), research_timeout=Mock(return_value=1000.0))
        metric = SimpleNamespace(task_type="regression", name="RMSE")
        spec = SimpleNamespace(metric=metric, features=("x", "cat", "period"),
                               categorical=("cat",), numeric=("x", "period"))
        self.state.gate = SimpleNamespace(
            spec=spec, research_spec=spec,
            manifest={"task_semantics": {"adaptive_research_allowed": True}},
            fit_rows=[0, 1, 2], held_rows=[3], choice={"strategy": "random"},
            inner_split={"strategy": "random"}, check_inputs=Mock(),
            research_train=pd.DataFrame({"x": [1, 2, 3, 99], "cat": ["a", "a", "b", "HELD_ONLY"],
                                         "period": [1, 1, 2, 3], "target": [1, 2, 3, 99999]}))
        self.state.baseline = self.freeze("baseline", ordinary())
        self.assess = Mock(return_value=feedback())

    def freeze(self, name, recipe, source=BASELINE_SOURCE):
        code = self.root / (name + ".py")
        code.write_text(source, encoding="utf-8", newline="\n")
        return freeze_candidate(code, recipe, self.root / (name + "_frozen"))

    def isolated(self, kind="frequency", **kwargs):
        names = {name: Mock(side_effect=extensions._Skip("fixture_skip"))
                 for name in ("_" + value + "_trial" for value in extensions.KINDS)
                 if name != "_" + kind + "_trial"}
        with patch.multiple(extensions, **names):
            return extensions.run_research_extensions(self.state, self.assess, **kwargs)

    def test_adaptive_mode_is_required(self):
        self.state.report["quality_options"]["adaptive_search"] = False
        report = self.isolated()
        self.assertEqual(report["reason"], "adaptive_search_disabled")
        self.assess.assert_not_called()

    def test_closed_research_is_skipped_without_fitting(self):
        self.state.dialogue_closed = True
        self.assertEqual(self.isolated()["reason"], "research_closed")
        self.assess.assert_not_called()

    def test_measured_baseline_is_required(self):
        self.state.baseline_inner = None
        self.assertEqual(self.isolated()["reason"], "measured_baseline_required")

    def test_inner_validation_and_supported_semantics_are_required(self):
        self.state.gate.held_rows = []
        self.assertEqual(self.isolated()["reason"], "inner_validation_unavailable")
        self.state.gate.held_rows = [3]
        self.state.gate.manifest["task_semantics"]["adaptive_research_allowed"] = False
        self.assertEqual(self.isolated()["reason"], "task_semantics_unavailable")

    def test_trial_cap_rejects_booleans_and_out_of_range_values(self):
        for count in (True, 0, 7, 1.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.isolated(max_trials=count)

    def test_frequency_preserves_actual_entire_baseline_and_uses_no_frame_values(self):
        recipe = {"components": [ordinary(), ordinary("catboost", missing_indicators=True)]}
        self.state.baseline = self.freeze("blend", recipe)
        # The host may use declared feature metadata and fit row count only.
        # Neither fitting labels nor held-only category vocabulary is inspected.
        self.state.gate.research_train = Mock(side_effect=AssertionError("No data access"))
        before = self.state.baseline.code_path.read_bytes(), self.state.baseline.recipe_path.read_bytes()
        report = self.isolated()
        self.assertEqual((report["attempted"], report["verified"]), (1, 1))
        candidate = self.assess.call_args.args[1]
        self.assertEqual(candidate.recipe["base_recipe"], recipe)
        self.assertEqual(candidate.recipe["frequency"]["categorical_columns"], ["cat"])
        self.assertEqual(self.assess.call_args.kwargs, {"origin": "local"})
        self.assertEqual(before, (self.state.baseline.code_path.read_bytes(), self.state.baseline.recipe_path.read_bytes()))
        self.assertIn("fit-only", self.assess.call_args.args[2]["hypothesis"])

    def test_frequency_no_categories_and_fixed_limits_skip_before_assessment(self):
        for categories, reason in (((), "no_categorical_features"),
                                   (tuple("c" + str(i) for i in range(17)), "categorical_column_limit")):
            with self.subTest(reason=reason):
                self.state.gate.research_spec.categorical = categories
                self.assertEqual(self.isolated()["trials"][0]["reason"], reason)
        self.assess.assert_not_called()

    def test_frequency_cannot_use_an_undeclared_target_as_a_category(self):
        self.state.gate.research_spec.categorical = ("target",)
        self.assertEqual(self.isolated()["trials"][0]["reason"], "categorical_schema_unavailable")
        self.assess.assert_not_called()

    def test_frequency_host_preparation_has_a_fixed_fit_cell_cap(self):
        self.state.gate.fit_rows = range(extensions.MAX_FREQUENCY_FIT_CELLS + 1)
        self.assertEqual(self.isolated()["trials"][0]["reason"], "frequency_fit_cell_limit")

    def test_host_refit_budget_reason_is_not_mislabeled_as_poor_quality(self):
        self.assess.return_value = feedback(reason="refit_budget")
        report = self.isolated()
        self.assertEqual(report["trials"][0]["reason"], "refit_budget")
        self.assertEqual(report["accepted"], 0)

    def test_optional_failed_fit_does_not_stop_the_other_candidates(self):
        recipe = {**ordinary(), "history_config": {"key_columns": ["cat"], "period_column": "period",
                  "numeric_columns": [], "positive_columns": [], "ambiguity_policy": "past_unique_only"},
                  "blend_weight": .5}
        trial = (BASELINE_SOURCE, recipe, "Fixture candidate")
        with patch.multiple(extensions, **{"_" + value + "_trial": Mock(return_value=trial)
                                        for value in extensions.KINDS}):
            self.assess.side_effect = [TimeoutError("secret provider text")] + [feedback() for _ in range(5)]
            report = extensions.run_research_extensions(self.state, self.assess)
        self.assertEqual((report["attempted"], report["verified"]), (6, 5))
        self.assertEqual(report["trials"][0]["status"], "failed")
        self.assertNotIn("secret", json.dumps(report))

    def test_actual_trial_cap_prevents_later_factories_including_discovery(self):
        later = {"_" + value + "_trial": Mock(side_effect=AssertionError("No late factory"))
                 for value in extensions.KINDS[1:]}
        with patch.multiple(extensions, **later):
            report = extensions.run_research_extensions(self.state, self.assess, max_trials=1)
        self.assertEqual(report["attempted"], 1)
        self.assertTrue(all(row["reason"] == "trial_limit" for row in report["trials"][1:]))
        for factory in later.values():
            factory.assert_not_called()

    def test_finalization_reserve_prevents_new_work(self):
        self.state.research_timeout.return_value = 1
        report = self.isolated()
        self.assertEqual(report["attempted"], 0)
        self.assess.assert_not_called()

    def test_preparation_cannot_consume_the_fit_budget_and_then_start_a_fit(self):
        self.state.research_timeout.side_effect = [1000] + [1] * 6
        report = self.isolated()
        self.assertEqual(report["trials"][0]["reason"], "finalization_reserve")
        self.assertEqual(report["attempted"], 0)

    def test_source_input_and_baseline_integrity_errors_escape_optional_failure_handler(self):
        def mutate(*_args, **_kwargs):
            self.state.gate.check_inputs.side_effect = ValueError("Inputs changed")
            raise TimeoutError("Optional trial failure")
        self.assess.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "Inputs changed"):
            self.isolated()

    def test_replacing_frozen_baseline_is_fatal(self):
        def mutate(*_args, **_kwargs):
            self.state.baseline = self.freeze("replacement", ordinary("catboost"))
            return feedback()
        self.assess.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "frozen recovery baseline"):
            self.isolated()

    def test_mutating_the_attempted_candidate_is_fatal_even_when_fit_failed(self):
        def mutate(_state, candidate, *_args, **_kwargs):
            candidate.code_path.write_text("# changed", encoding="utf-8")
            raise TimeoutError("Fit failed")
        self.assess.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "Frozen candidate"):
            self.isolated()

    def test_nonfinite_assessment_cannot_be_reported_as_verified(self):
        self.assess.return_value = feedback()
        self.assess.return_value["metric"]["value"] = float("nan")
        report = self.isolated()
        self.assertEqual((report["verified"], report["accepted"]), (0, 0))
        self.assertEqual(report["trials"][0]["status"], "failed")
        json.dumps(report, allow_nan=False)

    def history_policy(self):
        self.state.gate.choice = {"strategy": "time", "time_column": "period"}
        self.state.gate.inner_split = dict(self.state.gate.choice)

    def test_explicit_history_request_is_preserved_without_duplicate_discovery(self):
        self.history_policy()
        original = {"requested": True, "status": "completed", "attempted": 1}
        self.state.report["history_research"] = original
        with patch("agent.history_discovery_host.discover_history_options") as discover:
            report = self.isolated("history")
        discover.assert_not_called()
        self.assertIs(self.state.report["history_research"], original)
        self.assertEqual(report["trials"][1]["reason"], "history_already_requested")

    def test_history_requires_matching_inner_and_outer_time_policies(self):
        report = self.isolated("history")
        self.assertEqual(report["trials"][1]["reason"], "matching_time_policy_required")

    def test_history_discovery_abstention_does_not_count_as_model_trial(self):
        self.history_policy()
        with patch("agent.history_discovery_host.discover_history_options", return_value=None):
            report = self.isolated("history")
        self.assertEqual(report["attempted"], 0)
        self.assertEqual(report["trials"][1]["reason"], "history_discovery_abstained")

    def test_discovered_history_uses_normal_assessment_and_fixed_recipe(self):
        self.history_policy()
        config = {"key_columns": ["cat"], "period_column": "period", "numeric_columns": [],
                  "positive_columns": [], "ambiguity_policy": "past_unique_only"}
        options = {"history_config": config, "blend_weight": .5}
        with patch("agent.history_discovery_host.discover_history_options", return_value=options):
            report = self.isolated("history")
        self.assertEqual(report["attempted"], 1)
        self.assertEqual(self.assess.call_args.kwargs, {"origin": "history"})
        self.assertEqual(self.assess.call_args.args[1].recipe["history_config"], config)
        self.assertEqual(self.state.report["history_research"]["attempted"], 1)

    def test_failed_history_still_records_an_actual_attempt(self):
        self.history_policy()
        config = {"key_columns": ["cat"], "period_column": "period", "numeric_columns": [],
                  "positive_columns": [], "ambiguity_policy": "past_unique_only"}
        self.assess.side_effect = TimeoutError("Fit failed")
        with patch("agent.history_discovery_host.discover_history_options", return_value={
                "history_config": config, "blend_weight": .5}):
            self.isolated("history")
        self.assertEqual(self.state.report["history_research"]["attempted"], 1)
        self.assertEqual(self.state.report["history_research"]["status"], "failed")

    def test_ensemble_combines_distinct_measured_ordinary_components(self):
        selected = ordinary("catboost", missing_indicators=True)
        self.state.selected_candidate = self.freeze("selected", selected)
        report = self.isolated("ensemble")
        self.assertEqual(report["attempted"], 1)
        parts = self.assess.call_args.args[1].recipe["components"]
        self.assertEqual(parts, [ordinary(), selected])

    def test_ensemble_cannot_embed_a_custom_llm_implementation(self):
        self.state.selected_candidate = self.freeze("custom", {"anything": True},
            "def fit_predict(X_fit,y_fit,X_predict,context):\n return {'predictions': [0]*len(X_predict)}\n")
        report = self.isolated("ensemble")
        self.assertEqual(report["trials"][4]["reason"], "ordinary_measured_finalist_required")
        self.assess.assert_not_called()

    def test_duplicate_or_already_measured_ensemble_is_not_refitted(self):
        for name, selected, reason in (
                ("same", ordinary(), "distinct_component_limit"),
                ("existing", {"components": [ordinary(), ordinary("catboost")]}, "ensemble_already_measured")):
            with self.subTest(name=name):
                self.state.selected_candidate = self.freeze(name, selected)
                self.assertEqual(self.isolated("ensemble")["trials"][4]["reason"], reason)
        self.assess.assert_not_called()

    def fake_ag(self, *, available=True):
        return SimpleNamespace(availability=Mock(return_value={"available": available}),
            candidate_source=Mock(return_value=BASELINE_SOURCE),
            make_autogluon_recipe=Mock(return_value={"family": "autogluon_fixture"}))

    def fake_boosting(self):
        return SimpleNamespace(candidate_source=Mock(return_value=BASELINE_SOURCE),
            make_boosting_recipe=Mock(return_value={"family": "boosting_fixture"}))

    def test_boosting_families_have_a_bounded_default_and_preserve_split_policy(self):
        for action in ("catboost", "lightgbm"):
            with self.subTest(action=action):
                backend = self.fake_boosting()
                with patch.dict("sys.modules", {"agent.boosting_candidate": backend}):
                    report = self.isolated("boost_" + action)
                self.assertEqual(report["attempted"], 1)
                base, policy = backend.make_boosting_recipe.call_args.args
                self.assertEqual(base["action"], action)
                self.assertEqual(base["model_config"], make_model_config(action, "regression"))
                self.assertEqual(policy, self.state.gate.choice)
                self.assertIsNot(policy, self.state.gate.choice)
                self.assertEqual(backend.make_boosting_recipe.call_args.kwargs,
                                 {"max_iterations": 2000, "stopping_rounds": 50})

    def test_boosting_reuses_accepted_ordinary_family_settings(self):
        recipe = {"action": "catboost", "model_config": make_model_config(
            "catboost", "regression", {"depth": 4, "learning_rate": .1}),
            "feature_config": normalize_feature_config({"missing_indicators": True})}
        self.state.selected_candidate = self.freeze("boost_winner", recipe)
        backend = self.fake_boosting()
        with patch.dict("sys.modules", {"agent.boosting_candidate": backend}):
            self.isolated("boost_catboost")
        self.assertEqual(backend.make_boosting_recipe.call_args.args[0], recipe)

    def test_boosting_does_not_copy_arbitrary_llm_recipe_fields(self):
        self.state.selected_candidate = self.freeze("custom_boost", {"secret_custom_parameter": 9},
            "def fit_predict(X_fit,y_fit,X_predict,context):\n return {'predictions': [0]*len(X_predict)}\n")
        backend = self.fake_boosting()
        with patch.dict("sys.modules", {"agent.boosting_candidate": backend}):
            self.isolated("boost_catboost")
        self.assertNotIn("secret_custom_parameter", backend.make_boosting_recipe.call_args.args[0])

    def test_autogluon_unavailable_is_an_explicit_skip(self):
        with patch.dict("sys.modules", {"agent.autogluon_candidate": self.fake_ag(available=False)}):
            report = self.isolated("autogluon")
        self.assertEqual(report["trials"][5]["reason"], "autogluon_unavailable")
        self.assess.assert_not_called()

    def test_autogluon_has_fixed_cpu_memory_and_fit_only_validation_policy(self):
        ag = self.fake_ag()
        self.history_policy()
        with patch.dict("sys.modules", {"agent.autogluon_candidate": ag}):
            report = self.isolated("autogluon")
        self.assertEqual(report["attempted"], 1)
        self.assertEqual(ag.make_autogluon_recipe.call_args.args, ("regression", "RMSE"))
        options = ag.make_autogluon_recipe.call_args.kwargs
        self.assertEqual(options, {"time_limit": 180.0, "validation": self.state.gate.choice,
                                  "num_cpus": 8, "memory_limit": 24.0})
        self.assertIsNot(options["validation"], self.state.gate.choice)

    def test_autogluon_small_budget_is_skipped_before_factory_fit_settings(self):
        ag = self.fake_ag()
        self.state.research_timeout.return_value = 30
        with patch.dict("sys.modules", {"agent.autogluon_candidate": ag}):
            report = self.isolated("autogluon")
        self.assertEqual(report["trials"][5]["reason"], "autogluon_budget")
        ag.make_autogluon_recipe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
