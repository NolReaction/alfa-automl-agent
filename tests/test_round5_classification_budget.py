"""OOF host admission uses complete invocation cost and protects recovery.

No estimator, subprocess, sleep, API, or public data is used here. The real
V3State timing/allocation methods run against a deterministic clock, while the
existing run_candidate boundary supplies controlled completed/failed results.
"""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from agent.research_budget import BudgetPolicy
from agent.research_candidate import CandidateExecutionError, CandidateResult, freeze_candidate
from agent.research_v3 import V3State, _assess_candidate, finalize
from agent.metrics import get_metric
from agent.task_spec import TaskSpec
from agent import round5_classification_host as host
from validation import validate_predictions


def metric(value):
    return {"name": "Macro F1", "value": value, "higher_is_better": True}


def frozen(marker, family):
    return SimpleNamespace(code_hash=marker * 64, recipe_hash=("b" if marker == "a" else "d") * 64,
                           recipe={"family": family}, verify=Mock())


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


class TimingState:
    """Use production timing methods without filesystem-heavy RunState setup."""
    recipe_key = staticmethod(V3State.recipe_key)
    finalization_plan = V3State.finalization_plan
    research_timeout = V3State.research_timeout
    record_timing = V3State.record_timing
    measure_inner = V3State.measure_inner

    def __init__(self):
        self.clock = Clock()
        self.budget = BudgetPolicy(3600, clock=self.clock, adaptive_reserve=True)
        self.baseline = frozen("a", "existing_fixed_baseline")
        self.selected_candidate = None
        self.baseline_inner = metric(.60)
        self.best_inner = deepcopy(self.baseline_inner)
        self.dialogue_closed = False
        self.tried_recipes = []
        self._local_frozen_attempts = []
        self.run_dir = Path("unused-no-worker-launched")
        self.report = {"improvements": 0, "evaluations": {}, "stage": "inner_research"}
        frame = pd.DataFrame({"row_key": [str(i) for i in range(100)],
                              "x": [str(i % 7) for i in range(100)],
                              "answer": ["a", "b"] * 50})
        spec = SimpleNamespace(target_column="answer")
        self.gate = SimpleNamespace(
            check_inputs=Mock(), spec=spec, research_spec=spec, research_train=frame.iloc[:80],
            fit_rows=list(range(64)), held_rows=list(range(64, 80)),
            outer_fit_frame=frame.iloc[:80], outer_holdout_frame=frame.iloc[80:],
            train=frame, test=frame.iloc[:2].drop(columns="answer"),
            inner_split={"strategy": "grouped", "seed": 42}, expected_inner_actuals=["a", "b"] * 8,
        )
        self.clock.value = 10.
        self.budget.record_measurement(self.recipe_key(self.baseline), 1., 64, 16)

    def remaining(self):
        return self.budget.remaining()


def assessment_state(*, outer=100., refit=120., remaining=3000.):
    state = TimingState()
    state.measure_inner = Mock(return_value=SimpleNamespace(metric=metric(.70)))
    stages = {"baseline_outer": 20., "candidate_outer": outer,
              "baseline_refit": 25., "candidate_refit": refit}
    state.finalization_plan = Mock(return_value=SimpleNamespace(
        stage_seconds=stages, estimated_seconds=sum(stages.values()) + 2.))
    state.remaining = Mock(return_value=remaining)
    return state


class Round5ClassificationBudgetTests(unittest.TestCase):
    def assess(self, state, candidate, **kwargs):
        with patch("agent.research_v3.inner_feedback", return_value={}), \
                patch("agent.research_v3.time.monotonic", state.clock):
            return _assess_candidate(state, candidate, {"hypothesis": "Frozen OOF"}, origin="local", **kwargs)

    def test_custom_cap_is_forwarded_and_ordinary_call_shape_stays_unchanged(self):
        for maximum in (None, 1800, 600, 350):
            with self.subTest(maximum=maximum):
                state = assessment_state()
                candidate = frozen("c", "round4_classification_v1")
                options = {} if maximum is None else {"maximum": maximum}
                result = self.assess(state, candidate, **options)
                if maximum in (None, 1800):
                    state.measure_inner.assert_called_once_with(candidate)
                else:
                    state.measure_inner.assert_called_once_with(candidate, maximum=maximum)
                self.assertTrue(result["accepted"])
                self.assertIs(state.selected_candidate, candidate)

    def test_either_future_oof_stage_exceeding_600_rejects_a_better_inner_score(self):
        for outer, refit in ((600.001, 100.), (100., 600.001), (600., 600.)):
            with self.subTest(outer=outer, refit=refit):
                state = assessment_state(outer=outer, refit=refit)
                candidate = frozen("c", "round4_classification_v1")
                baseline, original_best = state.baseline, deepcopy(state.best_inner)
                result = self.assess(state, candidate, maximum=600)
                expected = outer <= 600 and refit <= 600
                self.assertEqual(result["accepted"], expected)
                self.assertEqual(result["timing"]["within_stage_limit"], expected)
                self.assertIs(state.baseline, baseline)
                if not expected:
                    self.assertTrue(result["verified"])
                    self.assertEqual(result["reason"], "refit_budget")
                    self.assertIsNone(state.selected_candidate)
                    self.assertEqual(state.best_inner, original_best)
                    self.assertEqual(state.report["improvements"], 0)

    def test_two_future_stages_and_baseline_must_fit_remaining_budget_together(self):
        state = assessment_state(outer=310., refit=350., remaining=600.)
        candidate = frozen("c", "round4_classification_v1")
        baseline = state.baseline
        result = self.assess(state, candidate, maximum=600)
        self.assertTrue(result["timing"]["within_stage_limit"])
        self.assertFalse(result["timing"]["feasible_estimate"])
        self.assertEqual(result["timing"]["estimated_finalization_seconds"], 707.)
        self.assertEqual(result["reason"], "refit_budget")
        self.assertFalse(result["accepted"])
        self.assertIs(state.baseline, baseline)
        self.assertIsNone(state.selected_candidate)
        self.assertEqual(state.best_inner, metric(.60))

    def test_full_host_invocation_not_last_fit_drives_both_future_stage_estimates(self):
        state = TimingState()
        candidate = frozen("c", "round4_classification_v1")
        original_deadline = state.budget.deadline
        phases = {"host_staging": 20., "oof_subset_copies": 112., "calibration_fits": 45.,
                  "fresh_final_fit": 20., "prediction_and_validation": 8.}
        def complete(_candidate, **kwargs):
            self.assertEqual(kwargs["timeout"], 600.)
            self.assertEqual(kwargs["deadline"], 610.)
            self.assertNotIn("answer", kwargs["predict_frame"])
            self.assertEqual(len(kwargs["actuals"]), 16)
            for elapsed in phases.values():
                state.clock.value += elapsed
            return SimpleNamespace(metric=metric(.70))
        with patch("agent.research_v3.run_candidate", side_effect=complete) as invocation:
            result = self.assess(state, candidate, maximum=600)
        invocation.assert_called_once()
        self.assertTrue(result["accepted"])
        report = state.budget.report(state.finalization_plan())
        measurements = report["measurements"][state.recipe_key(candidate)]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]["elapsed_seconds"], sum(phases.values()))
        self.assertEqual((measurements[0]["fit_rows"], measurements[0]["predict_rows"]), (64, 16))
        stages = report["stage_seconds"]
        self.assertGreater(stages["candidate_outer"], 1.5 * sum(phases.values()))
        self.assertGreater(stages["candidate_refit"], stages["candidate_outer"])
        self.assertGreater(report["final_reserve_seconds"], 2 * phases["fresh_final_fit"])
        self.assertFalse(report["diagnostics"]["guaranteed"])
        self.assertEqual(state.budget.deadline, original_deadline)

    def test_no_research_invocation_can_spend_existing_finalization_reserve(self):
        state = TimingState()
        candidate = frozen("c", "round4_classification_v1")
        baseline = state.baseline
        deadline = state.budget.deadline
        state.clock.value = deadline - state.finalization_plan().reserve_seconds
        with patch("agent.research_v3.run_candidate") as invocation, self.assertRaises(TimeoutError):
            self.assess(state, candidate, maximum=600)
        invocation.assert_not_called()
        self.assertEqual(state.budget.deadline, deadline)
        self.assertIs(state.baseline, baseline)
        self.assertIsNone(state.selected_candidate)
        self.assertEqual(state.best_inner, metric(.60))
        self.assertNotIn(state.recipe_key(candidate), state.budget.report()["measurements"])

    def test_failed_invocation_consumes_time_but_cannot_replace_baseline_or_invent_completed_measurement(self):
        state = TimingState()
        candidate = frozen("c", "round4_classification_v1")
        baseline = state.baseline
        deadline, before = state.budget.deadline, state.remaining()
        baseline_samples = deepcopy(state.budget.report()["measurements"][state.recipe_key(baseline)])
        def timeout(_candidate, **kwargs):
            state.clock.value += 120.
            raise CandidateExecutionError("controlled timeout", {"timed_out": True, "exit_code": -9})
        with patch("agent.research_v3.run_candidate", side_effect=timeout), self.assertRaises(CandidateExecutionError):
            self.assess(state, candidate, maximum=600)
        self.assertEqual(state.remaining(), before - 120.)
        self.assertEqual(state.budget.deadline, deadline)
        self.assertIs(state.baseline, baseline)
        self.assertIsNone(state.selected_candidate)
        self.assertEqual(state.best_inner, metric(.60))
        measurements = state.budget.report()["measurements"]
        self.assertNotIn(state.recipe_key(candidate), measurements)
        self.assertEqual(measurements[state.recipe_key(baseline)], baseline_samples)
        plan = state.finalization_plan()
        self.assertEqual(set(plan.stage_seconds), {"baseline_outer", "baseline_refit"})
        allocations = state.budget.finalization_timeouts(plan)
        self.assertTrue(all(allocations[name] >= cost for name, cost in plan.stage_seconds.items()))
        self.assertLessEqual(sum(allocations.values()), state.remaining() - state.budget.publication_buffer)

    def lifecycle(self, *, fail_custom_refit):
        """Real freeze, allocator, finalization and CSV publication; only execution is mocked."""
        with TemporaryDirectory() as temporary:
            root, clock = Path(temporary), Clock()
            task = root / "task.md"
            task.write_text("Metric: Macro F1\n", encoding="utf-8")
            with patch("agent.research_v3.time.monotonic", clock), redirect_stdout(io.StringIO()):
                state = V3State(task, 12, 3600, adaptive_search=True)
                state.budget = BudgetPolicy(3600, started=0, clock=clock, adaptive_reserve=True)
                frame = TimingState().gate.train
                test = pd.DataFrame({"row_key": ["0007", "NA"], "x": ["1", "2"]})
                data = root / "inputs"
                data.mkdir()
                test.to_csv(data / "test.csv", index=False)
                spec = TaskSpec(task, data, "row_key", "answer", ("x",), ("x",), (),
                                ("a", "b"), ("row_key", "answer"), (), None, get_metric("Macro F1"))
                state.gate = SimpleNamespace(
                    check_source=Mock(), check_inputs=Mock(), spec=spec, research_spec=spec,
                    train=frame, test=test, research_train=frame.iloc[:80],
                    fit_rows=list(range(64)), held_rows=list(range(64, 80)),
                    outer_fit_frame=frame.iloc[:80], outer_holdout_frame=frame.iloc[80:],
                    expected_ids=frame.iloc[80:].row_key.tolist(),
                    expected_actuals=frame.iloc[80:].answer.tolist(),
                    expected_inner_actuals=frame.iloc[64:80].answer.tolist(),
                    split={"strategy": "grouped", "seed": 42},
                    inner_split={"strategy": "grouped", "seed": 42})
                state.validator = SimpleNamespace(spec=spec, expected_test_ids=test.row_key.tolist())
                source = root / "baseline.py"
                source.write_text("def fit_predict(*args):\n    raise AssertionError('mock boundary only')\n")
                baseline = freeze_candidate(source, {"family": "baseline"}, root / "baseline")
                source.write_text(host.candidate_source(), encoding="utf-8")
                candidate = freeze_candidate(source,
                    host.make_recipe(host.fixed_incumbent_recipe(), spec.classes, variant="oof"), root / "oof")
                state.baseline, state.baseline_inner, state.best_inner = baseline, metric(.60), metric(.60)
                state._round5_snapshot = {"source_hash": candidate.code_hash, "oof_recipe_hash": candidate.recipe_hash}
                state._local_frozen_attempts = []
                state.transition("inner_research")
                clock.value = 10.
                state.budget.record_measurement(state.recipe_key(baseline), 1., 64, 16)
                calls, results = [], {}

                def execute(frozen_candidate, **kwargs):
                    frozen_candidate.verify()
                    custom = frozen_candidate is candidate
                    phase = kwargs["phase"]
                    self.assertNotIn("answer", kwargs["predict_frame"])
                    self.assertEqual(state.dialogue_closed, phase != "inner_selection")
                    self.assertLessEqual(kwargs["deadline"], state.budget.deadline - state.budget.publication_buffer)
                    calls.append((frozen_candidate, dict(kwargs)))
                    clock.value += 25. if custom else 1.
                    if custom and phase == "full_refit" and fail_custom_refit:
                        raise CandidateExecutionError("controlled custom refit timeout", {"timed_out": True})
                    directory = kwargs["output_dir"] / str(len(calls))
                    directory.mkdir(parents=True)
                    output = directory / "predictions.csv"
                    pd.DataFrame({"row_key": kwargs["predict_frame"].row_key.tolist(),
                                  "answer": ["b" if custom else "a"] * len(kwargs["predict_frame"])}).to_csv(output, index=False)
                    receipt = directory / "receipt.json"
                    receipt.write_text(json.dumps({"phase": phase, "code_hash": frozen_candidate.code_hash,
                                                   "recipe_hash": frozen_candidate.recipe_hash}))
                    result = CandidateResult(output, None if phase == "full_refit" else metric(.70 if custom else .60),
                        {"exit_code": 0}, frozen_candidate.code_hash, frozen_candidate.recipe_hash, phase, receipt,
                        hashlib.sha256(output.read_bytes()).hexdigest(), output.stat().st_size,
                        hashlib.sha256(receipt.read_bytes()).hexdigest())
                    results[(custom, phase)] = result
                    return result

                def verify(_state, frozen_candidate, result):
                    self.assertEqual((result.code_hash, result.recipe_hash),
                                     (frozen_candidate.code_hash, frozen_candidate.recipe_hash))
                    result.verify()

                with patch("agent.research_v3.run_candidate", side_effect=execute), \
                        patch("agent.research_v3.inner_feedback", return_value={}), \
                        patch.object(host, "verify_dependencies") as dependencies, \
                        patch.object(host, "verify_result", side_effect=verify) as verification, \
                        patch.object(host, "stage_limit", wraps=host.stage_limit) as caps, \
                        patch.object(state, "checkpoint", wraps=state.checkpoint) as checkpoints, \
                        patch("agent.research.validate_predictions", side_effect=lambda directory: validate_predictions(directory, spec=spec)):
                    feedback = _assess_candidate(state, candidate, {"hypothesis": "Frozen OOF"}, origin="local", maximum=600)
                    self.assertTrue(feedback["accepted"])
                    self.assertIs(state.selected_candidate, candidate)
                    self.assertEqual(finalize(state, "local_only"), state.run_dir)
                custom_calls = [kwargs for item, kwargs in calls if item is candidate]
                self.assertEqual([item["phase"] for item in custom_calls], ["inner_selection", "outer_validation", "full_refit"])
                self.assertEqual([(len(item["fit_frame"]), len(item["predict_frame"])) for item in custom_calls],
                                 [(64, 16), (80, 20), (100, 2)])
                self.assertNotIn("actuals", custom_calls[-1])
                self.assertEqual(len({item["output_dir"] for item in custom_calls}), 3)
                self.assertTrue(all(0 < item["timeout"] <= 600 for item in custom_calls))
                self.assertEqual([call.args[3] for call in caps.call_args_list],
                                 ["baseline_outer", "candidate_outer", "baseline_refit", "candidate_refit"])
                self.assertTrue(any(state.report["final_stage_budgets"][stage]["allocator_timeout_seconds"] > 600
                                    for stage in ("candidate_outer", "candidate_refit")))
                self.assertEqual(verification.call_count, 4 if fail_custom_refit else 5)
                self.assertGreater(dependencies.call_count, verification.call_count)
                self.assertEqual(checkpoints.call_count, 1 if fail_custom_refit else 2)
                winner = baseline if fail_custom_refit else candidate
                expected = results[(not fail_custom_refit, "full_refit")].read_output()
                self.assertEqual((root / "submission.csv").read_bytes(), expected)
                self.assertEqual(state.backup.read_bytes(), expected)
                self.assertTrue(state.report["output_check"]["valid"])
                self.assertEqual(state.report["candidate_receipts"]["code_hash"], winner.code_hash)
                self.assertEqual(state.report["published_recipe"], "baseline" if fail_custom_refit else "candidate")
                self.assertEqual(state.report["reason"], "refit_failed" if fail_custom_refit else "local_only")
                self.assertEqual(state.report["evaluations"]["final_check"]["selected"], "candidate")
                if fail_custom_refit:
                    self.assertTrue(any(event["kind"] == "refit_failed" and event["recipe"] == "candidate"
                                        for event in state.report["events"]))

    def test_frozen_oof_inner_outer_and_fresh_full_refit_keep_identity_caps_and_publish(self):
        self.lifecycle(fail_custom_refit=False)

    def test_failed_custom_full_refit_retains_real_verified_baseline_checkpoint(self):
        self.lifecycle(fail_custom_refit=True)


if __name__ == "__main__":
    unittest.main()
