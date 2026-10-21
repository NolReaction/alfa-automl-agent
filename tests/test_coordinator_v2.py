"""Deadline-driven coordination without model fitting, sleeps or API requests."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.offline_research import run_deadline_continuation
from agent.research_budget import BudgetPolicy
from agent.research_v3 import V3State, continue_locally, prepare_research, run_agent_v3
from tests.test_offline_research import CODE_HASH, State, recipe
from tests.test_research_budget import Clock, plan
from tests.test_research import make_regression
from tests.test_team_integration import fixture_baseline


class Proposer:
    def __init__(self, recipes=()):
        self.recipes = iter(recipes)
        self.suggestions = 0
        self.observations = []

    def suggest(self):
        self.suggestions += 1
        return next(self.recipes, None)

    def observe(self, item, feedback):
        self.observations.append((deepcopy(item), deepcopy(feedback)))

    def report(self):
        return {"suggestions": self.suggestions, "observations": len(self.observations)}


def cat_recipe(depth):
    return {"id": "cat-" + str(depth), "label": "CatBoost", "action": "catboost",
            "model_config": make_model_config("catboost", "regression", {"depth": depth}),
            "feature_config": normalize_feature_config()}


def feedback(*, verified=True, accepted=False):
    return {"verified": verified, "accepted": accepted,
            "reason": "improved" if accepted else "not_better" if verified else "execution_failed",
            "metric": {"name": "RMSE", "value": 1.0, "higher_is_better": False}}


class AdaptiveReserveTests(unittest.TestCase):
    def test_only_all_measured_long_run_recipes_release_fractional_floor(self):
        clock = Clock()
        policy = BudgetPolicy(10500, clock=clock, adaptive_reserve=True)
        before = plan(policy)
        self.assertEqual(before.reserve_seconds, 3150)
        clock.value = 5
        policy.record_measurement("baseline-hash", 1, 64, 16)
        partly_measured = plan(policy)
        self.assertEqual(partly_measured.reserve_seconds, 3150)
        self.assertEqual(partly_measured.diagnostics["reserve_policy"], "fixed_fraction")
        clock.value = 7
        policy.record_measurement("candidate-hash", 1, 64, 16)
        measured = plan(policy)
        self.assertEqual(measured.reserve_seconds, 120)
        self.assertGreaterEqual(measured.reserve_seconds, measured.estimated_seconds)
        self.assertEqual(measured.diagnostics["reserve_policy"], "measured_pending_recipes")
        self.assertEqual(policy.deadline, 10500)
        self.assertFalse(measured.diagnostics["guaranteed"])
        self.assertEqual(policy.report(measured)["elapsed_seconds"], 7)

    def test_unknown_new_candidate_restores_full_discovery_floor(self):
        clock = Clock()
        policy = BudgetPolicy(10500, clock=clock, adaptive_reserve=True)
        clock.value = 5
        policy.record_measurement("baseline-hash", 1, 64, 16)
        self.assertEqual(plan(policy, candidate=None).reserve_seconds, 120)
        self.assertEqual(plan(policy, candidate="new-unmeasured").reserve_seconds, 3150)

    def test_measured_expensive_recipes_still_reserve_all_stages(self):
        clock = Clock()
        policy = BudgetPolicy(10500, clock=clock, adaptive_reserve=True)
        clock.value = 800
        policy.record_measurement("baseline-hash", 600, 64, 16)
        clock.value = 1600
        policy.record_measurement("candidate-hash", 700, 64, 16)
        estimate = plan(policy)
        self.assertGreater(estimate.reserve_seconds, 120)
        self.assertEqual(estimate.reserve_seconds, estimate.estimated_seconds)
        allocations = policy.finalization_timeouts(estimate)
        self.assertLessEqual(sum(allocations.values()), policy.remaining() - policy.publication_buffer + 1e-9)
        self.assertTrue(all(value <= 1800 for value in allocations.values()))
        clock.value = policy.deadline - estimate.reserve_seconds
        self.assertEqual(policy.research_timeout(estimate), 0)

    def test_short_budgets_and_legacy_mode_keep_historical_reserve(self):
        for seconds, enabled, expected in ((90, True, 27), (300, True, 90), (10500, False, 3150)):
            with self.subTest(seconds=seconds, enabled=enabled):
                clock = Clock()
                policy = BudgetPolicy(seconds, clock=clock, adaptive_reserve=enabled)
                clock.value = 2
                policy.record_measurement("baseline-hash", .5, 64, 16)
                clock.value = 3
                policy.record_measurement("candidate-hash", .5, 64, 16)
                self.assertEqual(plan(policy).reserve_seconds, expected)

    def test_mode_is_explicit_boolean(self):
        for value in (1, None, "true", []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "adaptive_reserve"):
                BudgetPolicy(600, adaptive_reserve=value)


class DeadlineContinuationTests(unittest.TestCase):
    def run_policy(self, state, warmup, proposer=None, **options):
        return run_deadline_continuation(state, warmup, adaptive=proposer,
            evaluate=options.pop("evaluate", lambda item: feedback()), code_hash=CODE_HASH, **options)

    def test_four_nonimprovements_do_not_close_other_useful_trials(self):
        state = State()
        proposer = Proposer([recipe(8), recipe(9)])
        original_api = deepcopy(state.report["api"])
        result = self.run_policy(state, [recipe(alpha) for alpha in range(1, 8)], proposer)
        self.assertEqual(result["reason"], "queue_exhausted")
        self.assertEqual((result["attempted"], result["warmup_attempted"], result["adaptive_attempted"]), (9, 7, 2))
        self.assertEqual(result["verified"], 9)
        self.assertEqual(result["improvements"], 0)
        self.assertEqual(state.report["api"], original_api)
        self.assertEqual(state.report["successful_experiments"], 7)

    def test_failed_family_isolated_while_another_family_is_measured(self):
        state = State()
        called = []

        def evaluate(item):
            called.append(item["action"])
            return feedback(verified=item["action"] != "catboost")

        result = self.run_policy(state, [cat_recipe(depth) for depth in (3, 4, 5, 6)] + [recipe(1)],
                                 Proposer(), evaluate=evaluate)
        self.assertEqual(called, ["catboost", "catboost", "catboost", "linear"])
        self.assertEqual(result["blocked_families"], ["catboost"])
        self.assertEqual((result["failed"], result["verified"]), (3, 1))

    def test_adaptive_duplicate_is_observed_without_charging_a_fit(self):
        state = State()
        proposer = Proposer([recipe(1, label="Different presentation"), recipe(2)])
        result = self.run_policy(state, [recipe(1)], proposer)
        self.assertEqual((result["attempted"], result["duplicates"]), (2, 1))
        self.assertEqual(proposer.observations[0][1]["reason"], "duplicate")
        self.assertEqual(len(proposer.observations), 2)

    def test_budget_guard_runs_before_consuming_warmup_or_adaptive_queue(self):
        state, consumed, proposer = State(), [], Proposer([recipe(2)])
        state.timeout = 1

        def warmup():
            consumed.append(True)
            yield recipe(1)

        result = self.run_policy(state, warmup(), proposer)
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertEqual(consumed, [])
        self.assertEqual(proposer.suggestions, 0)

    def test_post_warmup_hook_precedes_adaptive_and_rechecks_remaining_budget(self):
        state, order, proposer = State(), [], Proposer([recipe(2)])

        def evaluate(item):
            order.append("warmup")
            return feedback()

        def extensions():
            order.append("extensions")
            state.timeout = 1

        result = self.run_policy(state, [recipe(1)], proposer,
                                 evaluate=evaluate, after_warmup=extensions)
        self.assertEqual(order, ["warmup", "extensions"])
        self.assertTrue(result["extension_hook_called"])
        self.assertEqual(proposer.suggestions, 0)
        self.assertEqual(result["reason"], "finalization_reserve")

    def test_actual_attempt_and_inspection_caps_bound_generators(self):
        state = State()
        result = self.run_policy(state, (recipe(alpha) for alpha in range(1, 100)), Proposer(), max_trials=5)
        self.assertEqual((result["reason"], result["attempted"]), ("trial_limit", 5))
        state = State()

        def forever():
            while True:
                yield recipe(1)

        result = self.run_policy(state, forever(), Proposer(), max_inspected=6)
        self.assertEqual((result["reason"], result["attempted"], result["inspected"]), ("inspection_limit", 1, 6))

    def test_training_errors_are_bounded_without_free_form_error_text(self):
        state = State()

        def evaluate(item):
            raise RuntimeError("secret provider text and credentials")

        result = self.run_policy(state, [recipe(1)], Proposer(), evaluate=evaluate)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["trials"][0]["error_type"], "RuntimeError")
        self.assertNotIn("secret provider", str(result))

    def test_source_view_baseline_and_closed_mutation_cannot_be_swallowed(self):
        for mutation in ("source", "view", "baseline", "closed"):
            with self.subTest(mutation=mutation):
                state = State()

                def evaluate(item):
                    if mutation == "source":
                        state.source_changed = True
                    elif mutation == "view":
                        state.gate.changed = True
                    elif mutation == "baseline":
                        state.baseline = type(state.baseline)()
                    else:
                        state.dialogue_closed = True
                    raise ValueError("Injected failure after mutation")

                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_policy(state, [recipe(1)], Proposer(), evaluate=evaluate)

    def test_hook_mutation_cannot_be_swallowed(self):
        state = State()

        def hook():
            state.gate.changed = True
            raise RuntimeError("optional hook error")

        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.run_policy(state, [], Proposer(), after_warmup=hook)

    def test_adaptive_library_failure_stops_cleanly_for_finalization(self):
        class Broken(Proposer):
            def suggest(self):
                raise RuntimeError("private diagnostic")

        state = State()
        original_baseline = state.baseline
        result = self.run_policy(state, [recipe(1)], Broken())
        self.assertEqual(result["reason"], "adaptive_unavailable")
        self.assertEqual(result["verified"], 1)
        self.assertIs(state.baseline, original_baseline)
        self.assertNotIn("private diagnostic", str(result))

    def test_callback_mutations_do_not_change_recipe_queue(self):
        state = State()
        warmup = [recipe(1)]
        original = deepcopy(warmup)

        def evaluate(item):
            item["model_config"]["parameters"]["alpha"] = 999
            return feedback()

        self.run_policy(state, warmup, Proposer(), evaluate=evaluate)
        self.assertEqual(warmup, original)

    def test_remaining_budget_is_reported_without_extending_deadline(self):
        state = State()
        clock = Clock(100)
        state.remaining = lambda: 600 - clock.value

        def evaluate(item):
            clock.value += 10
            state.timeout = 1
            return feedback()

        result = self.run_policy(state, [recipe(1), recipe(2)], Proposer(), evaluate=evaluate)
        self.assertEqual(result["started_remaining_seconds"], 500)
        self.assertEqual(result["ended_remaining_seconds"], 490)
        self.assertEqual(result["attempted"], 1)

    def test_ineligible_research_never_consumes_warmup_or_calls_hook(self):
        for condition in ("closed", "outer", "no_inner", "semantics", "no_baseline"):
            with self.subTest(condition=condition):
                state, called = State(), []
                if condition == "closed":
                    state.dialogue_closed = True
                elif condition == "outer":
                    state.report["stage"] = "outer_validation"
                elif condition == "no_inner":
                    state.gate.held_rows = []
                elif condition == "semantics":
                    state.gate.manifest["task_semantics"]["adaptive_research_allowed"] = False
                else:
                    state.baseline = None

                def forbidden():
                    raise AssertionError("Ineligible state consumed queue")
                    yield

                result = self.run_policy(state, forbidden(), Proposer(), after_warmup=lambda: called.append(True))
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(called, [])

    def test_invalid_mode_limits_are_rejected(self):
        for arguments in ({"max_trials": True}, {"max_trials": 97}, {"max_inspected": 513},
                          {"family_failure_limit": 0}, {"min_timeout": float("nan")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_policy(State(), [], Proposer(), **arguments)


class OrchestrationState:
    def __init__(self, task, max_steps, time_budget, *, adaptive_search=False):
        self.adaptive_search = adaptive_search and time_budget > 300
        self.report = {"successful_experiments": 1, "api": {"attempts": 0, "responses": 0},
                       "evaluations": {"selection": None, "final_check": None}}
        self.selected_candidate = None
        self.baseline = object()
        self.gate = SimpleNamespace(manifest={}, held_rows=[1], inner_split={},
            research_spec=object(), expected_inner_actuals=[], task_text="Regression task")

    def measure_inner(self, candidate):
        return SimpleNamespace(metric={"name": "RMSE", "value": 1.0, "higher_is_better": False})

    def transition(self, stage):
        self.report["stage"] = stage

    def redact(self, value):
        return value

    def save(self):
        pass


class DialogueExitTests(unittest.TestCase):
    def test_real_inner_view_warmup_extensions_and_adaptive_use_same_host_assessor(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            state = V3State(task, 12, 600, adaptive_search=True)
            state.report["quality_options"] = {"adaptive_search": True}
            prepare_research(state)
            fixture_baseline(state)
            state.transition("inner_research")
            proposer, order = Proposer([recipe(2)]), []

            def assess(current, frozen, hypothesis, *, origin):
                self.assertIs(current, state)
                frozen.verify()
                order.append(("local", frozen.recipe["model_config"]["parameters"]["alpha"]))
                self.assertEqual(origin, "local")
                return feedback()

            def extensions(current, assessor, *, max_trials):
                self.assertIs(current, state)
                self.assertIs(assessor, assess)
                self.assertEqual(max_trials, 6)
                order.append(("extensions", max_trials))
                return {"attempted": 0}

            with patch("agent.research_v3.local_candidate_plan", return_value=[recipe(1)]), \
                    patch("agent.adaptive_search.AdaptiveSearch", return_value=proposer), \
                    patch("agent.research_extensions.run_research_extensions", side_effect=extensions), \
                    patch("agent.research_v3._assess_candidate", new=assess):
                result = continue_locally(state)
            self.assertEqual(order, [("local", 1.0), ("extensions", 6), ("local", 2.0)])
            self.assertEqual(result["reason"], "queue_exhausted")
            self.assertEqual(state.report["api"]["attempts"], 0)
            self.assertEqual(state.report["successful_experiments"], 0)

    def test_model_finished_continues_only_for_explicit_long_budget_mode(self):
        for requested, seconds, active in ((False, 10500, False), (True, 10500, True), (True, 300, False)):
            with self.subTest(requested=requested, seconds=seconds), \
                    patch("agent.research_v3.V3State", OrchestrationState), \
                    patch("agent.research_v3.prepare_research"), \
                    patch("agent.research_v3.prepare_baseline"), \
                    patch("agent.research_v3.inner_feedback", return_value={}), \
                    patch("agent.research_v3._context", return_value="Allowed inner context"), \
                    patch("agent.research_v3._request_completion", return_value=(
                        {"finish_reason": "stop", "tool_calls": [], "content": "Finished"}, None)) as request, \
                    patch("agent.research_v3.continue_locally") as local, \
                    patch("agent.research_v3.finalize", side_effect=lambda state, reason: state):
                state = run_agent_v3(Path("unused-task.md"), object(), time_budget=seconds,
                                     adaptive_search=requested)
                self.assertEqual(local.call_count, int(active))
                self.assertEqual(request.call_count, 1)
                self.assertEqual(state.report["research_stop"], {
                    "reason": "model_finished", "local_continuation_requested": active})
                self.assertEqual(state.report["api"], {"attempts": 0, "responses": 0})
                if requested:
                    self.assertEqual(state.report["coordinator"]["active"], active)
                    self.assertEqual(state.report["coordinator"]["reason"], "enabled" if active else "short_budget")

    def test_invalid_adaptive_mode_rejected_before_task_read(self):
        for value in (1, None, "true", []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "adaptive_search"):
                run_agent_v3(Path("does-not-exist.md"), object(), adaptive_search=value)


if __name__ == "__main__":
    unittest.main()
