"""Host scheduling checks without training, files, clocks, or network access."""

from copy import deepcopy
import json
import math
import unittest

from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.search_schedule import AdaptiveSearchSchedule, MAX_DECISIONS


def recipe(action="linear", parameters=None, *, identifier="a", objective=None):
    return {"id": identifier, "label": "Presentation only", "action": action,
            "model_config": make_model_config(action, "regression", parameters, objective=objective),
            "feature_config": normalize_feature_config()}


def schedule(**kwargs):
    values = {"fit_rows": 64, "predict_rows": 16, "outer_fit_rows": 80,
              "outer_predict_rows": 20, "full_fit_rows": 100, "full_predict_rows": 10}
    return AdaptiveSearchSchedule(**{**values, **kwargs})


def choose(policy, candidates, **kwargs):
    return policy.choose(candidates, **{"selection_remaining": 90, "total_remaining": 180, **kwargs})


class AdaptiveSearchScheduleTests(unittest.TestCase):
    def test_unmeasured_probe_has_useful_timeout_and_original_index(self):
        policy = schedule()
        candidates = [recipe(), recipe("baseline"), recipe("catboost")]
        decision = choose(policy, candidates)
        self.assertEqual(decision["index"], 0)
        self.assertEqual(decision["estimated_trial_seconds"], 5)
        self.assertEqual(decision["timeout"], 10)
        self.assertEqual(decision["shares"], 3)
        self.assertFalse(decision["guaranteed"])
        self.assertEqual(policy.report()["dispatched"], 1)

    def test_dispatch_suppresses_same_configuration_with_different_labels(self):
        policy = schedule()
        first = recipe()
        alias = {**first, "id": "b", "label": "Another description"}
        candidates = [first, alias, recipe("baseline")]
        self.assertEqual(choose(policy, candidates)["index"], 0)
        self.assertEqual(choose(policy, candidates)["index"], 2)
        self.assertEqual(choose(policy, candidates)["reason"], "queue_exhausted")

    def test_objectives_and_feature_settings_remain_distinct(self):
        policy = schedule()
        ordinary = recipe("baseline")
        objective = recipe("baseline", objective="mae")
        representation = deepcopy(ordinary)
        representation["feature_config"] = normalize_feature_config({"missing_indicators": True})
        candidates = [ordinary, objective, representation]
        self.assertEqual([choose(policy, candidates)["index"] for _ in candidates], [0, 1, 2])

    def test_cheap_ranked_family_cannot_starve_untried_families(self):
        policy = schedule()
        known = recipe()
        policy.observe(known, .01)
        variant = recipe(parameters={"alpha": 10})
        candidates = [variant, recipe("baseline"), recipe("catboost")]
        self.assertEqual(choose(policy, candidates, ranked=[known])["index"], 1)
        self.assertEqual(choose(policy, candidates, ranked=[known])["index"], 2)
        self.assertEqual(choose(policy, candidates, ranked=[known])["index"], 0)

    def test_inner_family_order_and_cost_both_guide_refinement(self):
        policy = schedule()
        linear, baseline = recipe(), recipe("baseline")
        policy.observe(linear, 2)
        policy.observe(baseline, 2)
        candidates = [recipe(parameters={"alpha": 10}), recipe("baseline", {"max_iter": 200})]
        # Baseline ranks first but costs ~2.67s versus the second family's 2s.
        self.assertEqual(choose(policy, candidates, ranked=[baseline, linear])["index"], 1)
        expensive = schedule()
        expensive.observe(linear, 2)
        expensive.observe(baseline, 20)
        self.assertEqual(choose(expensive, candidates, ranked=[baseline, linear])["index"], 0)

    def test_work_units_scale_cost_of_a_measured_family(self):
        policy = schedule()
        reference = recipe("catboost", {"iterations": 100, "depth": 4})
        policy.observe(reference, 10)
        large = recipe("catboost", {"iterations": 200, "depth": 8})
        result = choose(policy, [large], selection_remaining=100, total_remaining=500)
        self.assertEqual(result["estimated_trial_seconds"], 40)
        self.assertEqual(result["timeout"], 80)
        self.assertEqual(result["timing_basis"], "family_host_wall_heuristic")

    def test_row_growth_increases_finalization_reserve(self):
        ordinary, large = schedule(), schedule(outer_fit_rows=1000, full_fit_rows=2000)
        a = choose(ordinary, [recipe()], total_remaining=100000)
        b = choose(large, [recipe()], total_remaining=100000)
        self.assertGreater(b["estimated_finalization_seconds"], a["estimated_finalization_seconds"])

    def test_trial_cannot_consume_finalization_time(self):
        policy = schedule()
        trial = choose(policy, [recipe()], total_remaining=1000)
        reserve = trial["estimated_finalization_seconds"]
        constrained = choose(schedule(), [recipe()], total_remaining=reserve + 6)
        self.assertLessEqual(constrained["timeout"] + reserve, reserve + 6)
        self.assertEqual(constrained["timeout"], 6)
        rejected = choose(schedule(), [recipe()], total_remaining=reserve + 4)
        self.assertIsNone(rejected["index"])
        self.assertEqual(rejected["reason"], "finalization_budget")

    def test_selection_share_does_not_grant_less_than_estimated_cost(self):
        policy = schedule()
        result = choose(policy, [recipe(), recipe("baseline"), recipe("catboost")], selection_remaining=12)
        self.assertIsNone(result["index"])
        self.assertEqual(result["reason"], "trial_budget")
        self.assertEqual(policy.report()["dispatched"], 0)

    def test_failed_fast_fit_does_not_make_its_family_look_cheap(self):
        policy = schedule()
        policy.observe(recipe(), .01, completed=False)
        result = choose(policy, [recipe(parameters={"alpha": 10})])
        self.assertEqual(result["estimated_trial_seconds"], 5)
        self.assertFalse(policy.report()["observations"][0]["completed"])

    def test_observation_preload_suppresses_duplicates_without_spending_dispatches(self):
        policy = schedule(max_trials=1)
        measured = recipe()
        policy.observe(measured, 1)
        self.assertEqual(policy.report()["dispatched"], 0)
        self.assertEqual(choose(policy, [measured])["reason"], "queue_exhausted")
        self.assertEqual(choose(policy, [recipe("baseline")])["index"], 0)
        self.assertEqual(choose(policy, [recipe("catboost")])["reason"], "trial_limit")

    def test_observation_after_dispatch_is_allowed_but_duplicate_observation_is_not(self):
        policy = schedule()
        measured = recipe()
        choose(policy, [measured])
        policy.observe(measured, 1)
        with self.assertRaisesRegex(ValueError, "already"):
            policy.observe(measured, 2)

    def test_report_is_detached_finite_json_and_decisions_are_bounded(self):
        policy = schedule()
        policy.observe(recipe(), 1)
        for _ in range(MAX_DECISIONS + 5):
            choose(policy, [])
        report = policy.report()
        self.assertEqual(len(report["decisions"]), MAX_DECISIONS)
        json.dumps(report, allow_nan=False)
        report["observations"][0]["elapsed_seconds"] = 999
        self.assertEqual(policy.report()["observations"][0]["elapsed_seconds"], 1)

    def test_extreme_finite_observation_and_row_ratios_stay_finite(self):
        policy = schedule(fit_rows=1, predict_rows=0, outer_fit_rows=1_000_000,
                          full_fit_rows=1_000_000_000, full_predict_rows=1_000_000_000)
        policy.observe(recipe(), 1e308, completed=False)
        result = choose(policy, [recipe(parameters={"alpha": 10})])
        self.assertIsNone(result["index"])
        self.assertEqual(result["reason"], "finalization_budget")
        self.assertTrue(math.isfinite(result["rejected"][0]["estimated_finalization_seconds"]))
        json.dumps(policy.report(), allow_nan=False)

    def test_inputs_and_results_do_not_mutate_caller_recipes(self):
        policy = schedule()
        values = [recipe(), recipe("baseline")]
        before = deepcopy(values)
        result = choose(policy, values, ranked=values)
        result["key"] = "changed"
        self.assertEqual(values, before)
        self.assertNotEqual(policy.report()["decisions"][-1]["key"], "changed")

    def test_invalid_constructor_inputs_are_rejected(self):
        for kwargs in ({"fit_rows": True}, {"fit_rows": 0}, {"predict_rows": -1},
                       {"outer_fit_rows": 50}, {"full_fit_rows": 70}, {"max_trials": True},
                       {"max_trials": 0}, {"max_trials": 13}, {"full_predict_rows": 1e4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                schedule(**kwargs)

    def test_invalid_observations_and_budget_values_are_rejected(self):
        for value in (True, -1, 0, math.nan, math.inf, "2", 10 ** 1000):
            with self.subTest(observation=value), self.assertRaises(ValueError):
                schedule().observe(recipe(), value)
        for value in (True, -1, math.nan, math.inf, "2", 10 ** 1000):
            for name in ("selection_remaining", "total_remaining"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    choose(schedule(), [recipe()], **{name: value})
        with self.assertRaisesRegex(ValueError, "boolean"):
            schedule().observe(recipe(), 1, completed=1)

    def test_malformed_recipes_and_unbounded_iterables_are_rejected(self):
        for values in ([{}], [recipe(), {"action": "linear"}], [recipe()] * 257,
                       iter([recipe()]), "recipes"):
            with self.subTest(values=type(values)), self.assertRaises(ValueError):
                choose(schedule(), values)
        wrong_action = {**recipe(), "action": "catboost"}
        with self.assertRaises(ValueError):
            choose(schedule(), [wrong_action])
        with self.assertRaises(ValueError):
            choose(schedule(), [recipe()], ranked=[{}])


if __name__ == "__main__":
    unittest.main()
