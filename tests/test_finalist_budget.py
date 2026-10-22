"""Measured finalist rechecks and final refits share the original deadline."""

from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from agent.research_budget import BudgetPolicy
from agent.research_finalists import FinalistPool
from agent.research_v3 import V3State
from agent.splits import split_rows


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class ShapeOnly:
    """Planning may know row counts, never outer/full feature or target values."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def __getitem__(self, key):
        raise AssertionError("Finalist budget planning accessed outer/full data")


class Frozen:
    def __init__(self, label, family):
        self.code_hash = sha256((label + ":code").encode()).hexdigest()
        self.recipe_hash = sha256((label + ":recipe").encode()).hexdigest()
        self.recipe = {"action": family}
        self.changed = False

    def verify(self):
        if self.changed:
            raise ValueError("Frozen candidate code or recipe changed")


def score(value):
    return {"name": "RMSE", "value": value, "higher_is_better": False}


def host(*, budget=360, baseline_seconds=.5, selected_seconds=1.0,
         alternative_seconds=8.0, splits=3):
    """Build a real host budget/pool around a fixed, permitted INNER view."""
    clock = Clock()
    state = V3State.__new__(V3State)
    state.budget = BudgetPolicy(budget, clock=clock, adaptive_reserve=True)
    state.remaining = state.budget.remaining
    state.report = {}
    state.finalist_selection = True
    state.finalist_splits = splits
    state._checking_finalists = False
    state._finalist_split_cache = None
    state.finalist_pool = FinalistPool()
    state.baseline = Frozen("baseline", "baseline")
    state.selected_candidate = Frozen("selected", "catboost")
    alternative = Frozen("alternative", "lightgbm")
    state.baseline_inner = score(1.0)
    state.best_inner = score(.8)
    frame = pd.DataFrame({"x": range(80), "target": [value / 10 for value in range(80)]})
    fitting, held, metadata = split_rows(frame, frame["target"], ("x",), False)
    state.gate = SimpleNamespace(
        research_train=frame, spec=SimpleNamespace(target_column="target"),
        classification=False, features=("x",), choice={"strategy": "random"},
        fit_rows=fitting.tolist(), held_rows=held.tolist(), inner_split=metadata,
        outer_fit_frame=ShapeOnly(80), outer_holdout_frame=ShapeOnly(20),
        train=ShapeOnly(100), test=ShapeOnly(10),
    )
    for candidate, elapsed in ((state.baseline, baseline_seconds),
                                (state.selected_candidate, selected_seconds),
                                (alternative, alternative_seconds)):
        clock.value += elapsed + 1
        state.budget.record_measurement(state.recipe_key(candidate), elapsed,
                                       len(fitting), len(held))
    state.finalist_pool.record(state.selected_candidate, state.best_inner)
    state.finalist_pool.record(alternative, score(.9))
    return state, clock, alternative


class FinalistBudgetTests(unittest.TestCase):
    def test_another_search_trial_leaves_both_shared_rechecks_and_any_winners_refit(self):
        state, _, _ = host()
        finalists = state.finalist_shortlist()
        self.assertEqual(len(finalists), 3)
        extra = state.finalist_inner_splits()
        self.assertEqual(len(extra), 2)
        repeat_cost = sum(state.estimate_research_cost(entry["candidate"],
            fit_rows=len(fitting), predict_rows=len(held))
            for fitting, held, _ in extra for entry in finalists)
        final_reserve = max(state._finalization_plan_for(entry["candidate"]).reserve_seconds
                            for entry in finalists)
        timeout = state.research_timeout(maximum=10500)
        self.assertGreater(timeout, 0)
        self.assertGreater(repeat_cost, 0)
        self.assertLessEqual(timeout + repeat_cost + final_reserve,
                             state.remaining() + 1e-9)
        # Plenty of spare time remains: the promised reserve is not an
        # accidental consequence of stopping all further work.
        self.assertAlmostEqual(timeout + repeat_cost + final_reserve, state.remaining())

    def test_short_worker_cap_does_not_get_reduced_by_an_unrelated_future_round(self):
        state, _, _ = host(budget=10500)
        self.assertEqual(state.research_timeout(maximum=5), 5)
        self.assertGreater(state.research_timeout(maximum=10500), 5)

    def test_shared_rechecks_release_their_own_reserve_but_keep_the_expensive_refit(self):
        state, _, _ = host()
        frozen = tuple(state.finalist_shortlist())
        search_timeout = state.research_timeout(maximum=10500)
        state._active_finalist_shortlist = frozen
        state._checking_finalists = True
        check_timeout = state.research_timeout(maximum=10500)
        self.assertGreater(check_timeout, search_timeout)
        prospective_reserve = max(state._finalization_plan_for(entry["candidate"]).reserve_seconds
                                  for entry in frozen)
        self.assertAlmostEqual(check_timeout + prospective_reserve, state.remaining())

    def test_cheap_incumbent_cannot_lend_its_refit_budget_to_an_expensive_runner_up(self):
        state, _, alternative = host(alternative_seconds=30)
        finalists = state.finalist_shortlist()
        self.assertIn(alternative, [entry["candidate"] for entry in finalists])
        incumbent_spare = state.budget.research_timeout(state.finalization_plan(), maximum=10500)
        all_winners_spare = state.finalist_research_timeout(finalists, maximum=10500)
        self.assertGreater(incumbent_spare, all_winners_spare)
        alternative_plan = state._finalization_plan_for(alternative)
        self.assertAlmostEqual(all_winners_spare + alternative_plan.reserve_seconds,
                               state.remaining())

    def test_new_host_timing_shrinks_spare_without_refreshing_the_original_deadline(self):
        state, clock, alternative = host(budget=10500, alternative_seconds=2)
        deadline = state.budget.deadline
        before = state.research_timeout(maximum=10500)
        clock.value += 201
        state.budget.record_measurement(state.recipe_key(alternative), 200,
                                       len(state.gate.fit_rows), len(state.gate.held_rows))
        after = state.research_timeout(maximum=10500)
        self.assertLess(after, before - 201)
        self.assertEqual(state.budget.deadline, deadline)
        self.assertIn(alternative, [entry["candidate"] for entry in state.finalist_shortlist()])

    def test_runner_up_with_an_overlong_own_final_stage_is_excluded(self):
        state, _, alternative = host(budget=10500, alternative_seconds=1000)
        plan = state._finalization_plan_for(alternative)
        self.assertGreater(plan.stage_seconds["candidate_refit"], 1800)
        self.assertFalse(state.can_finalize_candidate(alternative))
        self.assertNotIn(alternative, [entry["candidate"] for entry in state.finalist_shortlist()])
        self.assertEqual([entry["id"] for entry in state.finalist_shortlist()],
                         ["baseline", "candidate"])

    def test_runner_up_that_no_longer_fits_the_remaining_budget_is_excluded(self):
        state, clock, alternative = host(alternative_seconds=10)
        self.assertTrue(state.can_finalize_candidate(alternative))
        clock.value = state.budget.deadline - 15
        self.assertFalse(state.can_finalize_candidate(alternative))
        self.assertNotIn(alternative, [entry["candidate"] for entry in state.finalist_shortlist()])

    def test_baseline_participation_does_not_add_a_second_copy_of_the_same_final_refit(self):
        state, _, _ = host()
        plan = state._finalization_plan_for(state.baseline)
        self.assertEqual(set(plan.stage_seconds), {"baseline_outer", "baseline_refit"})

    def test_disabled_option_keeps_the_existing_single_winner_budget(self):
        state, _, _ = host()
        state.finalist_selection = False
        expected = state.budget.research_timeout(state.finalization_plan(), maximum=10500)
        with patch.object(state, "finalist_shortlist", side_effect=AssertionError("Disabled shortlist read")):
            self.assertEqual(state.research_timeout(maximum=10500), expected)

    def test_unmeasured_baseline_keeps_the_existing_preparation_path(self):
        state, _, _ = host()
        state.baseline = state.selected_candidate = None
        state.baseline_inner = state.best_inner = None
        expected = state.budget.research_timeout(state.finalization_plan(), maximum=10500)
        self.assertEqual(state.finalist_shortlist(), [])
        self.assertEqual(state.research_timeout(maximum=10500), expected)

    def test_no_admissible_extra_split_reserves_no_nonexistent_training(self):
        state, _, _ = host()
        with patch("agent.stability.additional_inner_splits", return_value=[]):
            expected = state.finalist_research_timeout(state.finalist_shortlist(), maximum=10500)
            self.assertEqual(state.research_timeout(maximum=10500), expected)
        self.assertEqual(state.finalist_inner_splits(), [])

    def test_unavailable_optional_split_preparation_skips_without_bypassing_refit_reserve(self):
        state, _, _ = host()
        with patch("agent.stability.additional_inner_splits", side_effect=ValueError("No complete grouped split")):
            expected = state.finalist_research_timeout(state.finalist_shortlist(), maximum=10500)
            self.assertEqual(state.research_timeout(maximum=10500), expected)
        self.assertEqual(state.finalist_inner_splits(), [])

    def test_retained_recipe_corruption_propagates_even_if_the_recipe_is_now_unaffordable(self):
        state, clock, alternative = host(alternative_seconds=10)
        clock.value = state.budget.deadline - 1
        alternative.changed = True
        with self.assertRaisesRegex(ValueError, "changed"):
            state.research_timeout(maximum=10500)

    def test_finalist_budget_cannot_open_new_time_after_the_original_deadline(self):
        state, clock, _ = host()
        deadline = state.budget.deadline
        self.assertGreater(state.research_timeout(maximum=10500), 0)
        clock.value = deadline + 1
        self.assertEqual(state.research_timeout(maximum=10500), 0)
        state._checking_finalists = True
        self.assertEqual(state.research_timeout(maximum=10500), 0)
        self.assertEqual(state.budget.deadline, deadline)


if __name__ == "__main__":
    unittest.main()
