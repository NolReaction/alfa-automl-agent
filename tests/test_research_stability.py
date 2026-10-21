"""Only paired inner checks may change a frozen v3 finalist selection."""

from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import patch

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


class CheckedCandidate:
    def __init__(self, name, code, recipe):
        self.name, self.code_hash, self.recipe_hash = name, code * 64, recipe * 64
        self.corrupted = False
        self.checks = 0

    def verify(self):
        self.checks += 1
        if self.corrupted:
            raise ValueError("Frozen candidate code or recipe changed")


class FinalistState(FakeState):
    def __init__(self, candidate_scores=(), baseline_scores=(), alternative_scores=()):
        super().__init__(candidate_scores=candidate_scores, baseline_scores=baseline_scores)
        self.finalist_selection = True
        self._checking_finalists = False
        self.baseline = CheckedCandidate("baseline", "a", "b")
        self.selected_candidate = CheckedCandidate("candidate", "c", "d")
        self.original_candidate = self.selected_candidate
        self.alternative = CheckedCandidate("alternative", "e", "f")
        self.alternative_metric = score(1.5)
        self.entries = [
            {"id": "baseline", "candidate": self.baseline,
             "metric": self.baseline_inner, "family": "linear"},
            {"id": "candidate", "candidate": self.original_candidate,
             "metric": self.best_inner, "family": "catboost"},
            {"id": "alternative", "candidate": self.alternative,
             "metric": self.alternative_metric, "family": "extra_trees"},
        ]
        self.values["alternative"] = list(alternative_scores)
        self.feasible = {"candidate": True, "alternative": True}
        self.timeout_shortlists = []
        self.source_checks = 0

    def check_source(self):
        self.source_checks += 1

    def finalist_shortlist(self):
        return self.entries

    def can_finalize_candidate(self, candidate):
        return self.feasible[candidate.name]

    def finalist_research_timeout(self, finalists, maximum=3600):
        if not self._checking_finalists:
            raise AssertionError("Future-repeat reserve must not be counted during rechecks")
        self.timeout_shortlists.append(tuple(item["id"] for item in finalists))
        return min(maximum, self.available)

    def measure_inner(self, candidate, *, rows, maximum):
        if tuple(item["id"] for item in self._active_finalist_shortlist) != self.timeout_shortlists[-1]:
            raise AssertionError("Every fit must retain the same frozen prospective reserve")
        self.calls.append({"name": candidate.name, "rows": rows, "maximum": maximum})
        if self.on_measure is not None:
            self.on_measure(self)
        value = self.values[candidate.name].pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(metric=score(value))


class MeasuredFinalistStabilityTests(unittest.TestCase):
    def test_runner_up_can_win_shared_repeats_with_its_original_canonical_metric(self):
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])
        original_metric = deepcopy(state.alternative_metric)
        report = verify_research_finalists(state, splits=2)
        self.assertIs(state.selected_candidate, state.alternative)
        self.assertEqual(state.best_inner, original_metric)
        self.assertEqual(state.report["evaluations"]["selection"]["metric"], original_metric)
        self.assertNotEqual(state.best_inner["value"], report["summary"]["candidates"]["alternative"]["mean"])
        self.assertTrue(report["changed_selection"])
        self.assertEqual(report["selected_recipe"], "alternative")
        self.assertEqual(report["selected_frozen"], {"code_hash": "e" * 64,
                         "recipe_hash": "f" * 64, "metric": original_metric})
        self.assertEqual(report["status"], "completed")
        self.assertFalse(state._checking_finalists)
        self.assertFalse(hasattr(state, "_active_finalist_shortlist"))
        self.assertEqual([call["name"] for call in state.calls], ["candidate", "baseline", "alternative"])
        for call in state.calls:
            self.assertEqual(call["rows"][0].tolist(), state.calls[0]["rows"][0].tolist())
            self.assertEqual(call["rows"][1].tolist(), state.calls[0]["rows"][1].tolist())
            self.assertLessEqual(call["maximum"], 100.0 / 3 if call is state.calls[0] else 100.0)
        self.assertTrue(all(names == ("candidate", "baseline", "alternative")
                            for names in state.timeout_shortlists))
        self.assertGreater(state.alternative.checks, len(state.calls))
        self.assertGreater(state.source_checks, len(state.calls))

    def test_mean_ties_preserve_the_incumbent_candidate(self):
        state = FinalistState(candidate_scores=[2.0], baseline_scores=[1.0], alternative_scores=[1.5])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["summary"]["ranking"][0], "candidate")
        self.assertIs(state.selected_candidate, state.original_candidate)
        self.assertEqual(state.best_inner, score(1.0))
        self.assertFalse(report["changed_selection"])

    def test_baseline_incumbent_wins_exact_ties_without_a_selected_candidate(self):
        state = FinalistState(baseline_scores=[2.0], alternative_scores=[1.0])
        state.selected_candidate = None
        state.best_inner = state.baseline_inner
        state.alternative_metric["value"] = 3.0
        state.entries = [item for item in state.entries if item["id"] != "candidate"]
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["summary"]["ranking"], ["baseline", "alternative"])
        self.assertIsNone(state.selected_candidate)
        self.assertFalse(report["changed_selection"])
        self.assertEqual(report["selected_recipe"], "baseline")

    def test_failure_of_third_finalist_discards_the_entire_extra_round(self):
        state = FinalistState(candidate_scores=[100.0], baseline_scores=[0.1],
                              alternative_scores=[RuntimeError("optional worker failed")])
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(report["summary"]["ignored_rounds"], 1)
        self.assertEqual(report["summary"]["candidates"]["baseline"]["scores"], [2.0])
        self.assertIs(state.selected_candidate, state.original_candidate)
        self.assertEqual(len(state.calls), 3)
        self.assertFalse(state._checking_finalists)

    def test_complete_prefix_can_choose_runner_up_despite_second_partial_round(self):
        state = FinalistState(candidate_scores=[6.0, 0.01], baseline_scores=[5.0, 0.1],
                              alternative_scores=[0.2, TimeoutError("second repeat expired")])
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["used_rounds"], 2)
        self.assertEqual(report["summary"]["ignored_rounds"], 1)
        self.assertIs(state.selected_candidate, state.alternative)
        self.assertEqual(state.best_inner, score(1.5))

    def test_infeasible_alternative_is_excluded_before_shared_rounds(self):
        state = FinalistState(candidate_scores=[0.5], baseline_scores=[4.0])
        state.feasible["alternative"] = False
        report = verify_research_finalists(state, splits=2)
        self.assertEqual([item["id"] for item in report["finalists"]], ["candidate", "baseline"])
        self.assertEqual(report["excluded_finalists"], [{"id": "alternative", "reason": "finalization_reserve"}])
        self.assertEqual(len(state.calls), 2)
        self.assertTrue(all("alternative" not in names for names in state.timeout_shortlists))

    def test_late_infeasible_winner_cannot_replace_existing_choice(self):
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])

        def exhaust_alternative_budget(current):
            if len(current.calls) == 3:
                current.feasible["alternative"] = False

        state.on_measure = exhaust_alternative_budget
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["summary"]["ranking"][0], "alternative")
        self.assertEqual(report["selection_reason"], "finalization_reserve")
        self.assertIs(state.selected_candidate, state.original_candidate)
        self.assertEqual(state.best_inner, score(1.0))
        self.assertFalse(report["changed_selection"])

    def test_no_complete_extra_round_never_selects_a_runner_up(self):
        state = FinalistState()
        state.available = 14.0
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "finalization_reserve")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(state.calls, [])
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_growing_reserve_aborts_shared_round_and_keeps_canonical_winner(self):
        state = FinalistState(candidate_scores=[100.0])
        state.on_measure = lambda current: setattr(current, "available", 0.5)
        report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "finalization_reserve")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(len(state.calls), 1)
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_mutated_unselected_finalist_is_not_hidden_by_an_optional_fit_failure(self):
        state = FinalistState(candidate_scores=[RuntimeError("worker failed")])

        def corrupt_unselected_candidate(current):
            current.alternative.corrupted = True

        state.on_measure = corrupt_unselected_candidate
        with self.assertRaisesRegex(ValueError, "Frozen candidate"):
            verify_research_finalists(state, splits=2)
        self.assertFalse(state._checking_finalists)
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_input_integrity_failure_after_worker_error_propagates(self):
        state = FinalistState(candidate_scores=[RuntimeError("worker failed")])
        original_check = state.gate.check_inputs

        def check():
            if state.calls:
                raise ValueError("Research input changed")
            original_check()

        state.gate.check_inputs = check
        with self.assertRaisesRegex(ValueError, "Research input changed"):
            verify_research_finalists(state, splits=2)
        self.assertFalse(state._checking_finalists)

    def test_shortlist_metadata_is_frozen_before_repeats(self):
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])

        def mutate_live_pool_metadata(current):
            current.alternative_metric["value"] = 999.0
            current.entries.reverse()

        state.on_measure = mutate_live_pool_metadata
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["rounds"][0]["scores"]["alternative"], 1.5)
        self.assertEqual(state.best_inner, score(1.5))
        self.assertEqual([item["id"] for item in report["finalists"]], ["candidate", "baseline", "alternative"])

    def test_nonfinite_metric_invalidates_an_entire_repeat(self):
        state = FinalistState(candidate_scores=[0.1], baseline_scores=[0.1], alternative_scores=[float("nan")])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_terminal_or_disabled_mode_never_queries_the_pool(self):
        for stage in ("candidate_frozen", "outer_validation", "full_refit", "result"):
            state = FinalistState()
            state.report["stage"] = stage
            state.gate = None
            with patch.object(state, "finalist_shortlist", side_effect=AssertionError("Must not access pool")):
                self.assertEqual(verify_research_finalists(state, splits=2)["reason"], "research_terminal")
            self.assertFalse(state._checking_finalists)
        state = FinalistState()
        with patch.object(state, "finalist_shortlist", side_effect=AssertionError("Must not access pool")):
            self.assertEqual(verify_research_finalists(state, splits=1)["reason"], "disabled")
        self.assertFalse(state._checking_finalists)

    def test_impossible_additional_split_keeps_original_choice_without_random_fallback(self):
        state = FinalistState()
        with patch("agent.research_stability.additional_inner_splits", return_value=[]):
            report = verify_research_finalists(state, splits=3)
        self.assertEqual(report["reason"], "fewer_admissible_splits")
        self.assertEqual(report["summary"]["used_rounds"], 1)
        self.assertEqual(state.calls, [])
        self.assertIs(state.selected_candidate, state.original_candidate)

    def test_rechecks_never_read_outer_values_or_change_api_counters(self):
        # InnerOnlyGate raises if any outer/full-data property is accessed.
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])
        state.report["api"] = {"attempts": 7}
        before = deepcopy(state.report["api"])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(state.report["api"], before)
        self.assertFalse(report["outer_answers_used"])

    def test_shared_repeats_preserve_group_and_time_boundaries(self):
        for strategy in ("grouped", "time"):
            with self.subTest(strategy=strategy):
                state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])
                gate = state.gate
                gate.classification = strategy == "grouped"
                if gate.classification:
                    groups = [index // 3 for index in range(180)]
                    gate.research_train = pd.DataFrame({"x": list(map(str, groups)),
                                                       "target": ["class " + str(group % 3) for group in groups]})
                    gate.choice = {"strategy": "grouped"}
                else:
                    gate.research_train = pd.DataFrame({"x": [str(index // 20) for index in range(100)],
                                                       "target": [float(index) for index in range(100)]})
                    gate.choice = {"strategy": "time", "time_column": "x"}
                fitting, held, gate.inner_split = split_rows(gate.research_train,
                    gate.research_train["target"], gate.features, gate.classification, gate.choice)
                gate.fit_rows, gate.held_rows = fitting.tolist(), held.tolist()
                report = verify_research_finalists(state, splits=2)
                self.assertEqual(report["rounds"][1]["split"]["strategy"], strategy)
                for call in state.calls:
                    fitting, held = call["rows"]
                    fit_values = set(gate.research_train.iloc[fitting]["x"])
                    held_values = set(gate.research_train.iloc[held]["x"])
                    self.assertFalse(fit_values & held_values)
                    if not gate.classification:
                        self.assertLess(max(map(float, fit_values)), min(map(float, held_values)))

    def test_baseline_can_win_and_preserves_the_original_baseline_recipe(self):
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[0.0], alternative_scores=[5.0])
        baseline = state.baseline
        metric = deepcopy(state.baseline_inner)
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["selected_recipe"], "baseline")
        self.assertTrue(report["changed_selection"])
        self.assertIsNone(state.selected_candidate)
        self.assertIs(state.baseline, baseline)
        self.assertEqual(state.best_inner, metric)

    def test_direction_of_higher_is_better_metric_is_used(self):
        state = FinalistState(candidate_scores=[0.5], baseline_scores=[0.3], alternative_scores=[0.95])
        state.gate.spec.metric = SimpleNamespace(name="accuracy", higher_is_better=True)
        for item, value in zip(state.entries, (0.6, 0.7, 0.65)):
            item["metric"].update(name="accuracy", higher_is_better=True, value=value)

        def measure(candidate, *, rows, maximum):
            state.calls.append({"name": candidate.name, "rows": rows, "maximum": maximum})
            return SimpleNamespace(metric={"name": "accuracy", "higher_is_better": True,
                                          "value": state.values[candidate.name].pop(0)})

        with patch.object(state, "measure_inner", side_effect=measure):
            report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["selected_recipe"], "alternative")
        self.assertEqual(state.best_inner, {"name": "accuracy", "higher_is_better": True, "value": 0.65})

    def test_active_shortlist_is_restored_even_when_integrity_check_raises(self):
        state = FinalistState(candidate_scores=[RuntimeError("worker failed")])
        original = ({"old": "parent context"},)
        state._active_finalist_shortlist = original
        state.on_measure = lambda current: setattr(current.baseline, "corrupted", True)
        with self.assertRaisesRegex(ValueError, "Frozen candidate"):
            verify_research_finalists(state, splits=2)
        self.assertIs(state._active_finalist_shortlist, original)
        self.assertFalse(state._checking_finalists)

    def test_missing_canonical_measurement_skips_rechecks_without_querying_empty_pool(self):
        for missing in ("baseline_inner", "best_inner"):
            with self.subTest(missing=missing):
                state = FinalistState()
                setattr(state, missing, None)
                with patch.object(state, "finalist_shortlist", side_effect=AssertionError("No measured shortlist")):
                    report = verify_research_finalists(state, splits=2)
                self.assertEqual(report["reason"], "inner_baseline_unavailable")
                self.assertIs(state.selected_candidate, state.original_candidate)
                self.assertEqual(state.calls, [])
                self.assertFalse(getattr(state, "_local_integrity_failed", False))
                if missing == "best_inner":
                    self.assertNotIn("metric", report["selected_frozen"])

    def test_reserved_cached_splits_are_the_exact_executed_plan(self):
        import numpy as np
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])
        frozen_fit, frozen_held = np.arange(80), np.arange(80, 100)
        frozen = [(frozen_fit, frozen_held, {"strategy": "random", "split_hash": "reserved-plan"})]
        with patch.object(state, "finalist_inner_splits", return_value=frozen, create=True) as cached, \
                patch("agent.research_stability.additional_inner_splits", side_effect=AssertionError("Do not replan")):
            report = verify_research_finalists(state, splits=2)
        cached.assert_called_once_with()
        self.assertEqual(report["rounds"][1]["split"]["split_hash"], "reserved-plan")
        for call in state.calls:
            self.assertIs(call["rows"][0], frozen_fit)
            self.assertIs(call["rows"][1], frozen_held)

    def test_baseline_object_replacement_latches_integrity_even_if_replacement_verifies(self):
        from agent.research_v3 import _recover_after_unexpected_error
        state = FinalistState(candidate_scores=[RuntimeError("worker failed")])
        replacement = CheckedCandidate("replacement", "7", "8")
        state.on_measure = lambda current: setattr(current, "baseline", replacement)
        with self.assertRaisesRegex(ValueError, "recovery baseline"):
            verify_research_finalists(state, splits=2)
        replacement.verify()
        self.assertTrue(state._local_integrity_failed)
        with self.assertRaisesRegex(RuntimeError, "integrity failed"):
            _recover_after_unexpected_error(state)

    def test_transient_integrity_failure_stays_latched_after_checker_recovers(self):
        from agent.research_v3 import _recover_after_unexpected_error
        state = FinalistState(candidate_scores=[RuntimeError("worker failed")])
        failed = False

        def source_check():
            nonlocal failed
            if state.calls and not failed:
                failed = True
                raise ValueError("Source snapshot changed")

        state.check_source = source_check
        with self.assertRaisesRegex(ValueError, "Source snapshot"):
            verify_research_finalists(state, splits=2)
        state.check_source()
        self.assertTrue(state._local_integrity_failed)
        with self.assertRaisesRegex(RuntimeError, "integrity failed"):
            _recover_after_unexpected_error(state)

    def test_ordinary_model_failure_does_not_latch_an_integrity_failure(self):
        state = FinalistState(candidate_scores=[RuntimeError("ordinary model failed")])
        report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["reason"], "repeat_failed")
        self.assertFalse(getattr(state, "_local_integrity_failed", False))

    def test_three_long_models_share_a_5400_second_round_with_individual_1800_caps(self):
        state = FinalistState(candidate_scores=[6.0], baseline_scores=[5.0], alternative_scores=[0.2])
        state.cost = 1500.0
        state.available = 5200.0
        with patch.object(state, "finalist_research_timeout", wraps=state.finalist_research_timeout) as budget:
            report = verify_research_finalists(state, splits=2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(state.calls), 3)
        self.assertTrue(all(call.kwargs["maximum"] == 5400 for call in budget.call_args_list))
        self.assertTrue(all(1500 <= call["maximum"] <= 1800 for call in state.calls))


if __name__ == "__main__":
    unittest.main()
