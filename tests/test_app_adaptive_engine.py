"""Local adaptive control keeps inner feedback and the original time budget."""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import test_automl as fixtures
from agent import adaptive_search, automl
from agent.app_adaptive_search import AppAdaptiveSearch


def feedback(value=1.0, elapsed=1.0):
    return {"verified": True, "accepted": False, "scope": "inner_selection",
            "metric": {"name": "RMSE", "value": value, "higher_is_better": False},
            "elapsed_seconds": elapsed, "reason": "not_better"}


class AppAdaptiveControllerTests(unittest.TestCase):
    def test_local_cap_exceeds_native_without_changing_native_contract(self):
        with self.assertRaises(ValueError):
            adaptive_search.AdaptiveSearch(False, "RMSE", max_trials=97)
        with patch.object(adaptive_search, "_load_optuna", return_value=None):
            engine = AppAdaptiveSearch(False, "RMSE", max_trials=128)
            recipes = []
            while (recipe := engine.suggest()) is not None:
                recipes.append(json.dumps(recipe["model_config"], sort_keys=True)
                               + json.dumps(recipe["feature_config"], sort_keys=True))
                engine.observe(recipe, feedback())
        self.assertEqual(len(recipes), 128)
        self.assertEqual(len(set(recipes)), 128)
        self.assertEqual(engine.summary()["stop_reason"], "trial_limit")
        self.assertFalse(engine.summary()["resumable"])

    def test_measured_trial_allowance_never_consumes_reserved_time(self):
        engine = AppAdaptiveSearch(False, "RMSE")
        recipe = engine.suggest()
        self.assertEqual(engine.trial_timeout(recipe, remaining=10, selection_window=360), 10)
        engine.observe(recipe, feedback(elapsed=20))
        self.assertEqual(engine.trial_timeout(recipe, remaining=300, selection_window=360), 45)
        self.assertEqual(engine.trial_timeout(recipe, remaining=.5, selection_window=360), 0)
        for invalid in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                engine.trial_timeout(recipe, remaining=invalid, selection_window=360)

    def test_local_guard_rejects_unbounded_or_boolean_trials(self):
        for cap in (0, True, 513, 1.5):
            with self.assertRaises(ValueError):
                AppAdaptiveSearch(False, "RMSE", max_trials=cap)


class AppAdaptiveIntegrationTests(unittest.TestCase):
    setUp = fixtures.AutoMLTests.setUp
    make_data = fixtures.AutoMLTests.make_data
    fake_execution = fixtures.AutoMLTests.fake_execution

    def run_adaptive(self, *, cap=24, outcomes=None, clock=None, cost=1):
        execute = self.fake_execution(outcomes)

        def measured(code, task_dir, run_dir, step, timeout):
            result = execute(code, task_dir, run_dir, step, timeout)
            if clock is not None:
                clock[0] += min(cost, timeout)
                if timeout < cost:
                    result.update(timed_out=True, exit_code=-9)
            return result

        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(automl, "run_python", side_effect=measured))
            stack.enter_context(patch.object(adaptive_search, "_load_optuna", return_value=None))
            if clock is not None:
                stack.enter_context(patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]))
            return automl.run_automl(self.task, time_budget=120, search_strategy="adaptive",
                                    adaptive_max_trials=cap)

    def test_more_than_finite_portfolio_trials_use_inner_rows_only(self):
        result = self.run_adaptive(cap=24)
        candidates = [call for call in self.calls if call["id"].startswith("adaptive_")
                      and call["phase"] == "selection"]
        self.assertEqual(len(candidates), 24)
        outer = next(call for call in self.calls if call["phase"] == "seed")
        for call in candidates:
            self.assertTrue(set(call["fit_rows"]) | set(call["validation_rows"]) <= set(outer["fit_rows"]))
            self.assertFalse(set(call["validation_rows"]) & set(outer["validation_rows"]))
        self.assertLessEqual(sum(call["phase"] == "verification" for call in self.calls), 3)
        self.assertEqual(result["search"]["controller"]["stop_reason"], "trial_limit")
        self.assertEqual(result["search"]["controller"]["observed"], 24)
        self.assertTrue(all(row["metric"]["value"] == 5 for row in result["adaptive_search"]["trials"]))

    def test_budget_stops_search_and_keeps_verification_and_refit_reserve(self):
        clock = [0.0]
        result = self.run_adaptive(cap=256, clock=clock, cost=4)
        self.assertEqual(result["search"]["controller"]["stop_reason"], "selection_budget")
        self.assertEqual(result["search"]["status"], "time_limit")
        self.assertLess(result["search"]["controller"]["proposed"], 256)
        self.assertLessEqual(clock[0], 120)
        self.assertTrue(result["output_valid"])
        self.assertTrue(any(call["phase"] == "verification" for call in self.calls))

    def test_invalid_strategy_fails_before_any_training(self):
        for strategy in ("something", None, True, []):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                automl.run_automl(self.task, search_strategy=strategy)
        self.assertEqual(self.calls, [])

    def test_weights_freeze_on_inner_scores_and_outer_reversal_cannot_retune_them(self):
        from app.search_summary import validate_search_summary

        outcomes = {("candidate_1", "selection"): {"error": 1},
                    ("candidate_2", "selection"): {"error": -3},
                    ("adaptive_0001", "selection"): {"exit_code": 1},
                    ("candidate_1", "verification"): {"error": 1},
                    ("candidate_2", "verification"): {"error": -3}}
        with patch.object(automl, "candidate_plan", return_value=automl.candidate_plan(False)[:2]):
            first = self.run_adaptive(cap=1, outcomes=outcomes)
            self.assertEqual(first["model"], "WeightedBlend")
            self.assertLess(first["metric"]["value"], 1)
            self.assertEqual(first["weighted_ensemble"]["fitted_on"], "inner_selection")
            validate_search_summary(first["search"], metric=first["metric"], model=first["model"])
            outcomes.update({("candidate_1", "verification"): {"error": 0},
                             ("candidate_2", "verification"): {"error": 100}})
            second = self.run_adaptive(cap=1, outcomes=outcomes)
        self.assertEqual(first["weighted_ensemble"], second["weighted_ensemble"])
        self.assertEqual(second["search"]["best_candidate"], "candidate_1")
        weighted = next(row for row in second["search"]["candidates"] if row["id"] == "weighted_blend")
        self.assertEqual(weighted["rejection_reason"], "not_improved")
        validate_search_summary(second["search"], metric=second["metric"], model=second["model"])


if __name__ == "__main__":
    unittest.main()
