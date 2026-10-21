"""Only paired inner checks may change a frozen v3 finalist selection."""

from types import SimpleNamespace
import unittest

import pandas as pd

from agent.research_stability import verify_research_finalists
from agent.splits import split_rows


def score(value):
    return {"name": "rmse", "value": value, "higher_is_better": False}


class InnerOnlyGate:
    def __init__(self):
        self.research_train = pd.DataFrame({"x": [str(i) for i in range(100)],
                                            "target": [str(i / 10) for i in range(100)]})
        self.spec = SimpleNamespace(target_column="target", metric=SimpleNamespace(
            name="rmse", higher_is_better=False))
        self.classification = False
        self.features = ("x",)
        self.choice = {"strategy": "random"}
        self.manifest = {"task_semantics": {"adaptive_research_allowed": True}}
        fitting, held, self.inner_split = split_rows(
            self.research_train, pd.to_numeric(self.research_train["target"]), self.features, False)
        self.fit_rows, self.held_rows = fitting.tolist(), held.tolist()
        self.input_checks = 0

    def check_inputs(self):
        self.input_checks += 1

    def __getattr__(self, name):
        if name in {"expected_actuals", "outer_actuals", "outer_holdout_frame", "outer_fit_frame",
                    "outer_fit_rows", "outer_holdout_rows", "train", "y", "test"}:
            raise AssertionError("The stability controller must not read outer/full data: " + name)
        raise AttributeError(name)


class FakeState:
    def __init__(self, candidate_scores=(), baseline_scores=()):
        self.gate = InnerOnlyGate()
        self.report = {"stage": "inner_research", "evaluations": {"selection": {"metric": score(1.0)}}}
        self.dialogue_closed = False
        self.baseline = SimpleNamespace(code_hash="a" * 64, recipe_hash="b" * 64)
        self.selected_candidate = SimpleNamespace(code_hash="c" * 64, recipe_hash="d" * 64)
        self.original_candidate = self.selected_candidate
        self.baseline_inner, self.best_inner = score(2.0), score(1.0)
        self.available = 100.0
        self.cost = 5.0
        self.calls, self.estimates = [], []
        self.values = {"candidate": list(candidate_scores), "baseline": list(baseline_scores)}
        self.saves = 0
        self.on_measure = None

    def save(self):
        self.saves += 1

    def research_timeout(self, maximum=1800):
        return min(maximum, self.available)

    def estimate_research_cost(self, candidate, *, fit_rows, predict_rows):
        self.estimates.append((candidate, fit_rows, predict_rows))
        return self.cost

    def measure_inner(self, candidate, *, rows, maximum):
        name = "baseline" if candidate is self.baseline else "candidate"
        self.calls.append({"name": name, "rows": rows, "maximum": maximum})
        if self.on_measure is not None:
            self.on_measure(self)
        value = self.values[name].pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(metric=score(value))


class ResearchStabilityTests(unittest.TestCase):
    def test_baseline_can_win_shared_repeats_without_overwriting_its_frozen_recipe(self):
        state = FakeState(candidate_scores=[4.0, 5.0], baseline_scores=[1.0, 1.0])
        baseline = state.baseline
        canonical = dict(state.baseline_inner)
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["used_rounds"], 3)
        self.assertEqual(report["summary"]["ranking"], ["baseline", "candidate"])
        self.assertTrue(report["changed_selection"])
        self.assertIsNone(state.selected_candidate)
        self.assertIs(state.baseline, baseline)
        self.assertEqual(state.best_inner, canonical)
        self.assertEqual(state.report["evaluations"]["selection"]["metric"], canonical)
        self.assertNotEqual(state.best_inner["value"], report["summary"]["candidates"]["baseline"]["mean"])
        self.assertFalse(report["outer_answers_used"])
        self.assertEqual(len(state.calls), 4)
        for left, right in zip(state.calls[::2], state.calls[1::2]):
            self.assertEqual(left["rows"][0].tolist(), right["rows"][0].tolist())
            self.assertEqual(left["rows"][1].tolist(), right["rows"][1].tolist())
            self.assertLessEqual(left["maximum"], 50.0)
        self.assertIs(state.report["inner_stability"], report)

    def test_candidate_stays_selected_and_canonical_score_is_not_replaced_with_mean(self):
        state = FakeState(candidate_scores=[0.5], baseline_scores=[1.0])
        canonical = dict(state.best_inner)
        report = verify_research_finalists(state, splits=2)
        self.assertIs(state.selected_candidate, state.original_candidate)
        self.assertEqual(state.best_inner, canonical)
        self.assertFalse(report["changed_selection"])
        self.assertEqual(report["summary"]["candidates"]["candidate"]["mean"], 0.75)

    def test_partial_round_failure_does_not_reward_the_only_successful_finalist(self):
        # The candidate looks disastrous on its completed half of a round,
        # but the baseline fails. Both extra scores must therefore be ignored.
        state = FakeState(candidate_scores=[100.0], baseline_scores=[RuntimeError("worker failed")])
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(report["summary"]["ignored_rounds"], 1)
        self.assertEqual(report["rounds"][1]["status"], "partial")
        self.assertEqual(report["summary"]["candidates"]["candidate"]["scores"], [1.0])
        self.assertIs(state.selected_candidate, state.original_candidate)
        self.assertEqual(len(state.calls), 2)

    def test_failed_first_finalist_does_not_start_second_one(self):
        state = FakeState(candidate_scores=[TimeoutError("worker timed out")])
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertEqual(len(state.calls), 1)
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_short_budget_never_starts_an_unfinishable_pair(self):
        state = FakeState()
        state.available = 9.0
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "finalization_reserve")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(state.calls, [])

    def test_growing_reserve_aborts_partial_round_and_preserves_previous_candidate(self):
        state = FakeState(candidate_scores=[100.0])
        state.on_measure = lambda value: setattr(value, "available", 0.5)
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "finalization_reserve")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(len(state.calls), 1)
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_complete_first_repeat_remains_valid_if_second_round_becomes_unaffordable(self):
        state = FakeState(candidate_scores=[0.5], baseline_scores=[3.0])

        def grow_after_pair(value):
            if len(value.calls) == 2:
                value.available = 9.0

        state.on_measure = grow_after_pair
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["used_rounds"], 2)
        self.assertEqual(len(state.calls), 2)

    def test_terminal_or_disabled_checks_cannot_inspect_gate_or_run_models(self):
        for stage in ("candidate_frozen", "outer_validation", "full_refit", "result"):
            state = FakeState()
            state.report["stage"] = stage
            state.gate = None
            report = verify_research_finalists(state, splits=3)
            self.assertEqual(report["reason"], "research_terminal")
            self.assertEqual(state.calls, [])
        state = FakeState()
        state.dialogue_closed = True
        self.assertEqual(verify_research_finalists(state, splits=2)["reason"], "research_terminal")
        state = FakeState()
        self.assertEqual(verify_research_finalists(state)["reason"], "disabled")
        self.assertEqual(state.calls, [])

    def test_unavailable_semantics_or_missing_scores_preserve_baseline(self):
        state = FakeState()
        state.gate.manifest["task_semantics"]["adaptive_research_allowed"] = False
        self.assertEqual(verify_research_finalists(state, splits=2)["reason"], "task_semantics_unavailable")
        self.assertEqual(state.calls, [])
        state = FakeState()
        state.baseline_inner = None
        self.assertEqual(verify_research_finalists(state, splits=2)["reason"], "inner_stability_unavailable")
        self.assertEqual(state.calls, [])

    def test_mismatched_metric_or_invalid_repeat_is_ignored_as_a_pair(self):
        state = FakeState(candidate_scores=[float("nan")])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertIs(state.selected_candidate, state.original_candidate)
        state = FakeState()
        state.best_inner["higher_is_better"] = True
        self.assertEqual(verify_research_finalists(state, splits=2)["reason"], "inner_stability_unavailable")
        self.assertEqual(state.calls, [])

    def test_exact_mean_tie_keeps_preexisting_candidate(self):
        state = FakeState(candidate_scores=[2.0], baseline_scores=[1.0])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["summary"]["candidates"]["baseline"]["mean"], 1.5)
        self.assertEqual(report["summary"]["candidates"]["candidate"]["mean"], 1.5)
        self.assertIs(state.selected_candidate, state.original_candidate)


if __name__ == "__main__":
    unittest.main()
