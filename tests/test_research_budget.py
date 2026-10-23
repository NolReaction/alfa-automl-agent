"""Time planning behavior without training, sleeping, network or API calls."""

import json
import math
import unittest

from agent.research_budget import BudgetPolicy


class Clock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value


def plan(policy, *, baseline="baseline-hash", candidate="candidate-hash"):
    return policy.finalization_estimate(
        baseline_key=baseline, candidate_key=candidate, outer_fit_rows=80,
        outer_predict_rows=20, full_fit_rows=100, test_rows=2)


class ResearchBudgetTests(unittest.TestCase):
    def test_cheap_measured_recipes_keep_original_reserve_and_allow_research(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        clock.value = 2
        policy.record_measurement("baseline-hash", .7, 64, 16)
        clock.value = 3
        policy.record_measurement("candidate-hash", .2, 64, 16)
        estimate = plan(policy)
        self.assertEqual(estimate.reserve_seconds, 27)
        self.assertEqual(policy.research_timeout(estimate, maximum=30), 30)
        self.assertFalse(estimate.diagnostics["guaranteed"])
        self.assertEqual(set(estimate.stage_seconds), {
            "baseline_outer", "candidate_outer", "baseline_refit", "candidate_refit"})
        json.dumps(policy.report(estimate), allow_nan=False)

    def test_expensive_candidate_grows_reserve_and_stops_another_experiment(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        initial = plan(policy)
        clock.value = 5
        policy.record_measurement("baseline-hash", 1, 64, 16)
        clock.value = 25
        policy.record_measurement("candidate-hash", 20, 64, 16)
        estimate = plan(policy)
        self.assertGreater(estimate.reserve_seconds, initial.reserve_seconds)
        self.assertEqual(policy.research_timeout(estimate), 0)
        self.assertGreater(estimate.stage_seconds["candidate_refit"],
                           estimate.stage_seconds["baseline_refit"])

    def test_expected_next_trial_must_fit_alongside_finalization(self):
        clock = Clock(10)
        policy = BudgetPolicy(90, started=0, clock=clock)
        estimate = plan(policy)
        self.assertEqual(policy.research_timeout(estimate, expected_seconds=54), 0)
        self.assertGreater(policy.research_timeout(estimate, expected_seconds=10), 0)

    def test_fixed_deadline_is_never_extended_by_measurements_or_replanning(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        deadline = policy.deadline
        clock.value = 10
        policy.record_measurement("baseline-hash", 1, 64, 16)
        estimate = plan(policy)
        self.assertEqual(policy.remaining(), 80)
        clock.value = 80
        self.assertEqual(policy.research_timeout(estimate), 0)
        clock.value = 92
        self.assertEqual(policy.remaining(), 0)
        self.assertTrue(all(value == 0 for value in policy.finalization_timeouts(estimate).values()))
        policy.record_measurement("candidate-hash", .5, 64, 16)
        self.assertEqual(policy.deadline, deadline)
        self.assertEqual(plan(policy).deadline, deadline)

    def test_new_measurement_invalidates_old_plan_and_older_samples_are_rejected(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        before = plan(policy)
        clock.value = 10
        policy.record_measurement("baseline-hash", 2, 64, 16)
        with self.assertRaisesRegex(ValueError, "Stale"):
            policy.research_timeout(before)
        with self.assertRaisesRegex(ValueError, "Stale"):
            policy.finalization_timeouts(before)
        current = plan(policy)
        self.assertGreater(policy.research_timeout(current), 0)
        clock.value = 11
        for timestamp in (9, 10):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "Stale"):
                policy.record_measurement("baseline-hash", 1, 64, 16, measured_at=timestamp)
        with self.assertRaisesRegex(ValueError, "outside"):
            policy.record_measurement("candidate-hash", 1, 64, 16, measured_at=12)
        with self.assertRaisesRegex(ValueError, "duration"):
            policy.record_measurement("candidate-hash", 20, 64, 16)

    def test_different_recipe_does_not_inherit_expensive_timing(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        clock.value = 20
        policy.record_measurement("old-hash", 19, 64, 16)
        estimate = plan(policy, baseline="new-hash", candidate=None)
        self.assertEqual(estimate.reserve_seconds, 27)
        self.assertEqual(estimate.diagnostics["stage_estimates"]["baseline_outer"]["samples"], 0)

    def test_stage_timeouts_account_for_both_refits_and_protect_baseline(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        clock.value = 5
        policy.record_measurement("baseline-hash", 1, 64, 16)
        clock.value = 25
        policy.record_measurement("candidate-hash", 20, 64, 16)
        estimate = plan(policy)
        caps = policy.finalization_timeouts(estimate)
        self.assertLessEqual(sum(caps.values()), policy.remaining() - policy.publication_buffer + 1e-9)
        self.assertGreaterEqual(caps["baseline_refit"], estimate.stage_seconds["baseline_refit"])
        self.assertLess(caps["baseline_outer"], policy.remaining() - caps["baseline_refit"])
        self.assertGreater(caps["candidate_refit"], 0)
        clock.value += caps["baseline_outer"]
        following = policy.finalization_timeouts(estimate, completed=("baseline_outer",))
        self.assertNotIn("baseline_outer", following)
        self.assertLessEqual(sum(following.values()), policy.remaining() - policy.publication_buffer + 1e-9)
        self.assertGreaterEqual(following["baseline_refit"], estimate.stage_seconds["baseline_refit"])

    def test_all_stage_caps_spent_sequentially_stay_inside_original_deadline(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        estimate = plan(policy)
        completed = []
        for stage in ("baseline_outer", "candidate_outer", "baseline_refit", "candidate_refit"):
            caps = policy.finalization_timeouts(estimate, completed=completed)
            clock.value += caps[stage]
            completed.append(stage)
            self.assertLessEqual(clock.value, policy.deadline - policy.publication_buffer + 1e-9)
        self.assertEqual(policy.finalization_timeouts(estimate, completed=completed), {})
        # Narrowing a later plan to baseline-only must allow already completed
        # candidate stages to remain in the host's lifecycle history.
        baseline_only = plan(policy, candidate=None)
        self.assertEqual(policy.finalization_timeouts(baseline_only, completed=completed), {})

    def test_feasible_long_finalization_funds_each_pending_stage_at_the_reserve_boundary(self):
        for seconds in (3600, 10500):
            with self.subTest(seconds=seconds):
                clock = Clock()
                policy = BudgetPolicy(seconds, clock=clock, adaptive_reserve=True)
                clock.value = 20
                policy.record_measurement("baseline-hash", 10, 64, 16)
                clock.value = 200
                policy.record_measurement("candidate-hash", 100, 64, 16)
                # Equal row counts make this exact boundary representable:
                # each baseline stage needs 15.25 s, each candidate 150.25 s.
                estimate = policy.finalization_estimate(
                    baseline_key="baseline-hash", candidate_key="candidate-hash",
                    outer_fit_rows=64, outer_predict_rows=16, full_fit_rows=64, test_rows=16)
                self.assertEqual(estimate.reserve_seconds, 333)
                clock.value = policy.deadline - estimate.reserve_seconds
                completed = []
                for stage in estimate.stage_seconds:
                    caps = policy.finalization_timeouts(estimate, completed=completed)
                    self.assertEqual(set(caps), set(estimate.stage_seconds) - set(completed))
                    for pending, value in caps.items():
                        self.assertGreaterEqual(value, estimate.stage_seconds[pending])
                        self.assertLessEqual(value, 1800)
                    self.assertLessEqual(sum(caps.values()), policy.remaining() - policy.publication_buffer)
                    clock.value += caps[stage]
                    completed.append(stage)
                self.assertEqual(policy.finalization_timeouts(estimate, completed=completed), {})
                self.assertEqual(clock.value, policy.deadline - policy.publication_buffer)

    def test_feasible_stage_floors_respect_the_per_stage_cap(self):
        clock = Clock()
        policy = BudgetPolicy(10500, clock=clock, adaptive_reserve=True)
        clock.value = 20
        policy.record_measurement("baseline-hash", 10, 64, 16)
        clock.value = 200
        policy.record_measurement("candidate-hash", 100, 64, 16)
        estimate = policy.finalization_estimate(
            baseline_key="baseline-hash", candidate_key="candidate-hash",
            outer_fit_rows=64, outer_predict_rows=16, full_fit_rows=64, test_rows=16)
        clock.value = policy.deadline - policy.publication_buffer - 230.5
        caps = policy.finalization_timeouts(estimate, maximum=100)
        self.assertEqual(caps, {"baseline_outer": 15.25, "candidate_outer": 100,
                                "baseline_refit": 15.25, "candidate_refit": 100})
        self.assertEqual(sum(caps.values()), policy.remaining() - policy.publication_buffer)

    def test_absolute_clock_rounding_at_reserve_boundary_does_not_shrink_candidate_shares(self):
        clock = Clock(30000)
        policy = BudgetPolicy(3600, clock=clock, adaptive_reserve=True)
        clock.value += 20
        policy.record_measurement("baseline-hash", 10, 64, 16)
        clock.value += 180
        policy.record_measurement("candidate-hash", 100, 64, 16)
        estimate = plan(policy)
        clock.value = policy.deadline - estimate.reserve_seconds
        available = policy.remaining() - policy.publication_buffer
        self.assertLess(available, sum(estimate.stage_seconds.values()))
        caps = policy.finalization_timeouts(estimate)
        for stage, value in caps.items():
            self.assertAlmostEqual(value, estimate.stage_seconds[stage], places=9)
            self.assertLessEqual(value, 1800)
        self.assertLessEqual(sum(caps.values()), available)
        # A real deficit must still use the existing fallback-priority path.
        clock.value += .001
        deficit = policy.finalization_timeouts(estimate)
        self.assertLess(deficit["candidate_outer"], estimate.stage_seconds["candidate_outer"] - 1)
        self.assertGreaterEqual(deficit["baseline_refit"], estimate.stage_seconds["baseline_refit"])
        self.assertLessEqual(sum(deficit.values()), policy.remaining() - policy.publication_buffer)

    def test_infeasible_finalization_preserves_existing_fallback_priority(self):
        clock = Clock()
        policy = BudgetPolicy(10500, clock=clock, adaptive_reserve=True)
        clock.value = 20
        policy.record_measurement("baseline-hash", 10, 64, 16)
        clock.value = 200
        policy.record_measurement("candidate-hash", 100, 64, 16)
        estimate = policy.finalization_estimate(
            baseline_key="baseline-hash", candidate_key="candidate-hash",
            outer_fit_rows=64, outer_predict_rows=16, full_fit_rows=64, test_rows=16)
        # At 100 s the fallback keeps both floors; at 20 s it alone receives
        # the insufficient budget. These are the pre-fix deficit allocations.
        for available, baseline, candidate in ((100, 18.429003021148038, 31.570996978851962),
                                                (20, 10, 0)):
            with self.subTest(available=available):
                clock.value = policy.deadline - policy.publication_buffer - available
                caps = policy.finalization_timeouts(estimate)
                for stage in ("baseline_outer", "baseline_refit"):
                    self.assertAlmostEqual(caps[stage], baseline)
                for stage in ("candidate_outer", "candidate_refit"):
                    self.assertAlmostEqual(caps[stage], candidate)
                self.assertLessEqual(sum(caps.values()), available)

    def test_extreme_row_growth_is_bounded_and_explicitly_a_heuristic(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        clock.value = 2
        policy.record_measurement("baseline-hash", 1, 1, 1)
        estimate = policy.finalization_estimate(
            baseline_key="baseline-hash", outer_fit_rows=1_000_000,
            outer_predict_rows=1_000_000, full_fit_rows=1_000_000_000, test_rows=1_000_000)
        self.assertTrue(estimate.diagnostics["stage_estimates"]["baseline_refit"]["bounded"])
        self.assertTrue(all(math.isfinite(value) and value <= 90
                            for value in estimate.stage_seconds.values()))
        self.assertLessEqual(estimate.reserve_seconds, 90)
        self.assertEqual(estimate.diagnostics["kind"], "heuristic")

    def test_maximum_cap_and_tiny_budget_do_not_create_extra_time(self):
        clock = Clock()
        policy = BudgetPolicy(90, clock=clock)
        caps = policy.finalization_timeouts(plan(policy), maximum=1)
        self.assertTrue(all(value <= 1 for value in caps.values()))
        tiny = BudgetPolicy(1, clock=clock)
        tiny_plan = plan(tiny)
        self.assertEqual(tiny.research_timeout(tiny_plan), 0)
        caps = tiny.finalization_timeouts(tiny_plan)
        self.assertEqual(caps["candidate_outer"], 0)
        self.assertEqual(caps["candidate_refit"], 0)
        self.assertLessEqual(sum(caps.values()), .95 + 1e-9)

    def test_invalid_measurements_inputs_and_clock_are_rejected(self):
        clock = Clock(10)
        policy = BudgetPolicy(90, started=0, clock=clock)
        cases = [("", 1, 64, 16), ("k", True, 64, 16), ("k", float("nan"), 64, 16),
                 ("k", 0, 64, 16), ("k", -1, 64, 16), ("k", 1, True, 16),
                 ("k", 1, 0, 16), ("k", 1, 64, -1)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                policy.record_measurement(*args)
        with self.assertRaises(ValueError):
            BudgetPolicy(0)
        with self.assertRaises(ValueError):
            BudgetPolicy(10501)
        with self.assertRaises(ValueError):
            BudgetPolicy(90, started=11, clock=clock)
        estimate = plan(policy)
        with self.assertRaises(ValueError):
            policy.finalization_timeouts(estimate, completed=("missing",))
        with self.assertRaises(ValueError):
            policy.research_timeout(estimate, maximum=1, minimum=2)
        clock.value = 9
        with self.assertRaisesRegex(ValueError, "backwards"):
            policy.remaining()


if __name__ == "__main__":
    unittest.main()
