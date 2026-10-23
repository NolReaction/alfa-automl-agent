"""Early host-extension scheduling contracts without API calls or model fits."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from agent import research_extensions
from agent.offline_research import run_deadline_continuation
from agent.research_v3 import BASELINE_SOURCE
from tests import test_research_extensions as extension_fixture
from tests.test_coordinator_v2 import Proposer, cat_recipe, feedback
from tests.test_offline_research import CODE_HASH, State, recipe


class RescueIntegrationScheduleTests(unittest.TestCase):
    def run_policy(self, state, warmup, proposer=None, **options):
        return run_deadline_continuation(
            state, warmup, adaptive=proposer,
            evaluate=options.pop("evaluate", lambda item: feedback()),
            code_hash=CODE_HASH, **options)

    def extension_fixture(self):
        fixture = extension_fixture.ExtensionTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_early_hook_precedes_queue_consumption_then_warmup_and_adaptive_continue(self):
        state, order = State(), []
        state.remaining = lambda: state.timeout
        api_before = deepcopy(state.report["api"])

        def warmup():
            order.append("consume_warmup")
            yield recipe(1)

        def hook():
            order.append("extensions")
            state.timeout -= 10

        def evaluate(item):
            order.append(item["model_config"]["parameters"]["alpha"])
            state.timeout -= 1
            return feedback()

        proposer = Proposer([recipe(2)])
        result = self.run_policy(
            state, warmup(), proposer, evaluate=evaluate,
            after_warmup=hook, extensions_first=True)
        self.assertEqual(order, ["extensions", "consume_warmup", 1, 2])
        self.assertEqual(result["extension_schedule"], "before_warmup")
        self.assertTrue(result["extension_hook_called"])
        self.assertEqual(result["extension_hook_started_remaining_seconds"], 120)
        self.assertEqual(result["extension_hook_ended_remaining_seconds"], 110)
        self.assertEqual((result["warmup_attempted"], result["adaptive_attempted"]), (1, 1))
        self.assertEqual(len(proposer.observations), 1)
        self.assertEqual(proposer.observations[0][0]["model_config"]["parameters"]["alpha"], 2)
        self.assertEqual(state.report["api"], api_before)

    def test_default_order_remains_after_warmup_and_hook_is_not_repeated(self):
        state, order = State(), []
        hook = Mock(side_effect=lambda: order.append("extensions"))

        def evaluate(item):
            order.append(item["model_config"]["parameters"]["alpha"])
            return feedback()

        result = self.run_policy(
            state, [recipe(1)], Proposer([recipe(2)]),
            evaluate=evaluate, after_warmup=hook)
        self.assertEqual(order, [1, "extensions", 2])
        self.assertEqual(result["extension_schedule"], "after_warmup")
        hook.assert_called_once_with()

    def test_ineligible_runs_do_not_call_hook_consume_recipes_or_change_baseline(self):
        mutations = {
            "closed": lambda state: setattr(state, "dialogue_closed", True),
            "stage": lambda state: state.report.update(stage="outer_validation"),
            "no_inner": lambda state: setattr(state.gate, "held_rows", []),
            "no_baseline": lambda state: setattr(state, "baseline", None),
            "semantics": lambda state: state.gate.manifest["task_semantics"].update(
                adaptive_research_allowed=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                state, consumed = State(), []
                mutate(state)
                baseline = state.baseline

                def warmup():
                    consumed.append(True)
                    yield recipe(1)

                hook, evaluate, proposer = Mock(), Mock(), Proposer([recipe(2)])
                result = self.run_policy(
                    state, warmup(), proposer, evaluate=evaluate,
                    after_warmup=hook, extensions_first=True)
                self.assertEqual(result["status"], "skipped")
                self.assertFalse(result["extension_hook_called"])
                self.assertIsNone(result["extension_hook_started_remaining_seconds"])
                self.assertIsNone(result["extension_hook_ended_remaining_seconds"])
                self.assertEqual(consumed, [])
                self.assertEqual(proposer.suggestions, 0)
                self.assertIs(state.baseline, baseline)
                hook.assert_not_called()
                evaluate.assert_not_called()

    def test_initial_finalization_reserve_prevents_hook_and_queue_consumption(self):
        state, consumed = State(), []
        state.timeout = 2

        def warmup():
            consumed.append(True)
            yield recipe(1)

        hook, evaluate, proposer = Mock(), Mock(), Proposer([recipe(2)])
        result = self.run_policy(
            state, warmup(), proposer, evaluate=evaluate,
            after_warmup=hook, extensions_first=True)
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertFalse(result["extension_hook_called"])
        self.assertEqual(consumed, [])
        self.assertEqual(proposer.suggestions, 0)
        hook.assert_not_called()
        evaluate.assert_not_called()

    def test_hook_acceptance_rechecks_dynamic_reserve_before_any_ordinary_fit(self):
        state, consumed, accepted = State(), [], object()
        state.selected_candidate = None
        baseline = state.baseline

        def warmup():
            consumed.append(True)
            yield recipe(1)

        def hook():
            # Represents the assessor accepting a measured, expensive candidate:
            # its new finalization plan leaves no spare research allowance.
            state.selected_candidate = accepted
            state.timeout = 0

        evaluate, proposer = Mock(), Proposer([recipe(2)])
        result = self.run_policy(
            state, warmup(), proposer, evaluate=evaluate,
            after_warmup=hook, extensions_first=True)
        self.assertTrue(result["extension_hook_called"])
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(consumed, [])
        self.assertEqual(proposer.suggestions, 0)
        self.assertIs(state.baseline, baseline)
        self.assertIs(state.selected_candidate, accepted)
        evaluate.assert_not_called()

    def test_hook_does_not_reset_dedup_failed_family_or_attempt_caps(self):
        state, hook = State(), Mock()
        proposer = Proposer([recipe(1), cat_recipe(8), recipe(2)])
        calls = []

        def evaluate(item):
            calls.append(item["action"])
            return feedback(verified=item["action"] != "catboost")

        result = self.run_policy(
            state, [cat_recipe(depth) for depth in (3, 4, 5, 6)] + [recipe(1)],
            proposer, evaluate=evaluate, after_warmup=hook,
            extensions_first=True, max_trials=5)
        self.assertEqual(calls, ["catboost", "catboost", "catboost", "linear", "linear"])
        self.assertEqual(result["blocked_families"], ["catboost"])
        self.assertEqual((result["attempted"], result["failed"], result["duplicates"]), (5, 3, 1))
        self.assertEqual(result["reason"], "trial_limit")
        self.assertEqual([item[1]["reason"] for item in proposer.observations],
                         ["duplicate", "family_blocked", "not_better"])
        hook.assert_called_once_with()

    def test_early_hook_integrity_mutations_escape_its_optional_exception_handler(self):
        for mutation in ("source", "inputs", "baseline", "closed"):
            with self.subTest(mutation=mutation):
                state, evaluate = State(), Mock()

                def hook():
                    if mutation == "source":
                        state.source_changed = True
                    elif mutation == "inputs":
                        state.gate.changed = True
                    elif mutation == "baseline":
                        state.baseline = type(state.baseline)()
                    else:
                        state.dialogue_closed = True
                    raise RuntimeError("optional failure after integrity mutation")

                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_policy(
                        state, [recipe(1)], Proposer(), evaluate=evaluate,
                        after_warmup=hook, extensions_first=True)
                evaluate.assert_not_called()

    def test_optional_hook_failure_is_reported_once_without_stopping_other_families(self):
        state = State()
        hook = Mock(side_effect=RuntimeError("secret provider failure text"))
        result = self.run_policy(
            state, [recipe(1)], Proposer([recipe(2)]),
            after_warmup=hook, extensions_first=True)
        hook.assert_called_once_with()
        self.assertEqual(result["verified"], 2)
        self.assertEqual(result["extension_hook_error_type"], "RuntimeError")
        self.assertNotIn("secret provider", str(result))

    def test_new_scheduling_options_require_explicit_booleans(self):
        fixture = self.extension_fixture()
        for value in (None, 0, 1, "true", []):
            with self.subTest(option="extensions_first", value=value):
                with self.assertRaises(ValueError):
                    self.run_policy(
                        State(), [], Proposer(), after_warmup=Mock(),
                        extensions_first=value)
            with self.subTest(option="prioritize", value=value):
                with self.assertRaises(ValueError):
                    research_extensions.run_research_extensions(
                        fixture.state, fixture.assess, prioritize=value)

    def test_prioritized_factory_order_uses_matching_time_policy_and_preserves_legacy(self):
        cases = (
            ("legacy", False, "regression", "time", "period", "period",
             list(research_extensions.KINDS)),
            ("time", True, "regression", "time", "period", "period",
             ["history", "boost_lightgbm", "boost_catboost", "frequency", "ensemble", "autogluon"]),
            ("random", True, "regression", "random", None, None,
             ["boost_lightgbm", "boost_catboost", "frequency", "history", "ensemble", "autogluon"]),
            ("different_period", True, "regression", "time", "period", "different",
             ["boost_lightgbm", "boost_catboost", "frequency", "history", "ensemble", "autogluon"]),
            ("classification", True, "classification", "time", "period", "period",
             ["boost_lightgbm", "boost_catboost", "frequency", "history", "ensemble", "autogluon"]),
        )
        for name, prioritize, task_type, strategy, outer_time, inner_time, expected in cases:
            with self.subTest(name=name):
                fixture = self.extension_fixture()
                fixture.state.gate.spec.metric.task_type = task_type
                fixture.state.gate.choice = {"strategy": strategy, "time_column": outer_time}
                fixture.state.gate.inner_split = {"strategy": strategy, "time_column": inner_time}
                called = []

                def factory(kind):
                    def skip(*args):
                        called.append(kind)
                        raise research_extensions._Skip("fixture_skip")
                    return Mock(side_effect=skip)

                factories = {"_" + kind + "_trial": factory(kind)
                             for kind in research_extensions.KINDS}
                before = (fixture.state.baseline.code_path.read_bytes(),
                          fixture.state.baseline.recipe_path.read_bytes())
                with patch.multiple(research_extensions, **factories):
                    result = research_extensions.run_research_extensions(
                        fixture.state, fixture.assess, prioritize=prioritize)
                self.assertEqual(called, expected)
                self.assertEqual([entry["kind"] for entry in result["trials"]], expected)
                self.assertEqual(result["attempted"], 0)
                self.assertEqual((fixture.state.baseline.code_path.read_bytes(),
                                  fixture.state.baseline.recipe_path.read_bytes()), before)
                fixture.assess.assert_not_called()

    def test_prioritized_actual_trial_cap_stops_later_factory_preparation(self):
        fixture = self.extension_fixture()
        fixture.state.gate.choice = {"strategy": "time", "time_column": "period"}
        fixture.state.gate.inner_split = {"strategy": "time", "time_column": "period"}
        called = []

        def factory(kind):
            def propose(*args):
                called.append(kind)
                candidate_recipe = extension_fixture.ordinary()
                candidate_recipe.update(
                    history_config={"key_columns": ["x"], "period_column": "period"},
                    blend_weight=0.5)
                return BASELINE_SOURCE, candidate_recipe, "bounded fixture"
            return Mock(side_effect=propose)

        factories = {"_" + kind + "_trial": factory(kind)
                     for kind in research_extensions.KINDS}
        with patch.multiple(research_extensions, **factories):
            result = research_extensions.run_research_extensions(
                fixture.state, fixture.assess, prioritize=True, max_trials=1)
        self.assertEqual(called, ["history"])
        self.assertEqual((result["attempted"], result["verified"]), (1, 1))
        self.assertEqual(result["reason"], "trial_limit")
        self.assertEqual(fixture.assess.call_count, 1)
        self.assertTrue(all(entry["reason"] == "trial_limit"
                            for entry in result["trials"][1:]))


if __name__ == "__main__":
    unittest.main()
