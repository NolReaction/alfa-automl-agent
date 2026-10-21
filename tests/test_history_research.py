"""Explicit history trials through the real v3 host and bounded unit guards.

The constant baseline is a test fixture, not a quality benchmark. Child
processes still execute and score the frozen baseline/history implementations.
No AutoML portfolio or API is used by these tests.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agent.history_runtime import make_history_recipe, normalize_history_options
from agent.research_candidate import freeze_candidate, run_candidate
from agent.research_v3 import research_configured_history, run_agent_v3
from agent.research_workspace import ResearchWorkspace
from tests.test_research import write_csv
from tests.test_team_integration import CONSTANT_CODE, fixture_baseline
from validation import validate_predictions


CONFIG = {"key_columns": ["entity"], "period_column": "period"}
ENTITIES = ("9007199254740992", "9007199254740993", "61242.123456789012345",
            "61242.123456789012346", "NA", "0007")
TEST_IDS = ["0005", "NA", "tail,id"]


def make_panel(directory, *, intervene_outer=False):
    """Six stable entity levels, four train periods and feature-only period 5."""
    directory = Path(directory)
    inputs = directory / "regression" / "participants"
    write_csv(inputs / "data_dictionary.csv", [
        ["feature", "role", "dtype"], ["row_key", "id", "category"],
        ["entity", "feature", "category"], ["period", "feature", "numeric"],
        ["measurement", "feature", "numeric"], ["outcome", "target", "numeric"]])
    rows = []
    for period in range(1, 5):
        for index, entity in enumerate(ENTITIES):
            outcome = 10 * (index + 1) + (900000 if intervene_outer and period == 4 else 0)
            row_id = ("0001", "NA", "comma,id", 'quote"id', "normal-4", "normal-5")[index]
            rows.append([f"{period}:{row_id}", entity, period, index % 2, outcome])
    write_csv(inputs / "train.csv", [["row_key", "entity", "period", "measurement", "outcome"]] + rows)
    write_csv(inputs / "test.csv", [["row_key", "entity", "period", "measurement"],
              [TEST_IDS[0], ENTITIES[0], 5, 0], [TEST_IDS[1], ENTITIES[1], 5, 1],
              [TEST_IDS[2], "unseen", 5, 0]])
    write_csv(inputs / "sample_submission.csv", [["row_key", "outcome"]] + [[value, 0] for value in TEST_IDS])
    task = directory / "task.md"
    task.write_text("Task type: regression\nUse regression/participants/*.csv to predict outcome.\n"
                    "Metric: RMSE\nPredict a future period from strictly earlier periods.\n"
                    "Save submission.csv here.", encoding="utf-8")
    return task


class HistoryV3IntegrationTests(unittest.TestCase):
    def run_panel(self, task, *, fail_history_refit=False):
        executions, views, baselines, failed_refits = [], [], [], []

        def baseline(state):
            fixture_baseline(state)
            baselines.append({"object": state.baseline, "code_hash": state.baseline.code_hash,
                              "recipe_hash": state.baseline.recipe_hash,
                              "code_bytes": state.baseline.code_path.read_bytes(),
                              "recipe_bytes": state.baseline.recipe_path.read_bytes()})

        def workspace(*args, **kwargs):
            gate = ResearchWorkspace(*args, **kwargs)
            views.append({name: (gate.workspace / name).read_bytes() for name in gate.manifest["allowed_inputs"]})
            return gate

        def execute(candidate, **kwargs):
            recipe = candidate.recipe
            entry = {"phase": kwargs["phase"], "code_hash": candidate.code_hash,
                     "recipe_hash": candidate.recipe_hash, "recipe": copy.deepcopy(recipe),
                     "fit_periods": sorted(set(kwargs["fit_frame"]["period"]), key=int),
                     "query_periods": sorted(set(kwargs["predict_frame"]["period"]), key=int),
                     "fit_rows": len(kwargs["fit_frame"]), "predict_rows": len(kwargs["predict_frame"])}
            if fail_history_refit and kwargs["phase"] == "full_refit" and recipe.get("family") == "history_blend":
                failed_refits.append(entry)
                raise RuntimeError("Fixture history full refit failed")
            result = run_candidate(candidate, **kwargs)
            entry.update(receipt=str(result.receipt_path), output=str(result.output_path))
            executions.append(entry)
            return result

        with patch("agent.research_v3.prepare_baseline", side_effect=baseline), \
                patch("agent.research_v3.ResearchWorkspace", side_effect=workspace), \
                patch("agent.research_v3.local_candidate_plan", return_value=[]), \
                patch("agent.research_v3.run_candidate", side_effect=execute), \
                patch("agent.research_v3._request_completion", side_effect=AssertionError("No API in local-only mode")) as api, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_dir = run_agent_v3(task, None, local_only=True, history_config=CONFIG,
                                   history_weight=0.5, time_budget=90)
        api.assert_not_called()
        self.assertEqual(len(baselines), 1)
        baseline_identity = baselines[0]
        baseline_identity["object"].verify()
        self.assertEqual(baseline_identity["object"].code_path.read_bytes(), baseline_identity["code_bytes"])
        self.assertEqual(baseline_identity["object"].recipe_path.read_bytes(), baseline_identity["recipe_bytes"])
        report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["frozen_baseline"], {key: baseline_identity[key] for key in ("code_hash", "recipe_hash")})
        self.assertEqual(report["history_research"]["baseline"], report["frozen_baseline"])
        return run_dir, report, executions, views, failed_refits

    def assert_no_api_and_one_history(self, report, executions):
        self.assertEqual(report["mode"], "local_only")
        self.assertEqual(report["api"]["attempts"], 0)
        self.assertEqual(report["api"]["responses"], 0)
        self.assertEqual(report["history_research"]["attempted"], 1)
        self.assertEqual(report["history_research"]["status"], "completed")
        self.assertEqual(report["history_successful_experiments"], 1)
        self.assertEqual(report["local_research"]["attempted"], 0)
        self.assertEqual(len([event for event in report["events"] if event["kind"] == "history_experiment"]), 1)
        self.assertEqual(len([entry for entry in executions if entry["phase"] == "inner_selection"
                              and entry["recipe"].get("family") == "history_blend"]), 1)

    def test_explicit_history_uses_one_inner_trial_and_same_frozen_recipe_through_full_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_panel(directory)
            _, report, executions, _, _ = self.run_panel(task)
            self.assert_no_api_and_one_history(report, executions)
            self.assertEqual((report["status"], report["reason"]), ("completed", "local_only"))
            self.assertEqual(report["history_research"]["feedback"]["reason"], "improved")
            self.assertEqual(report["published_recipe"], "candidate")
            frozen = report["frozen_candidate"]
            self.assertEqual(frozen, report["history_research"]["candidate"])
            candidate_calls = [entry for entry in executions if entry["code_hash"] == frozen["code_hash"]]
            self.assertEqual([entry["phase"] for entry in candidate_calls],
                             ["inner_selection", "outer_validation", "full_refit"])
            expected = [(["1", "2"], ["3"], 12, 6), (["1", "2", "3"], ["4"], 18, 6),
                        (["1", "2", "3", "4"], ["5"], 24, 3)]
            for entry, temporal in zip(candidate_calls, expected):
                self.assertEqual((entry["fit_periods"], entry["query_periods"], entry["fit_rows"],
                                  entry["predict_rows"]), temporal)
                self.assertEqual(entry["recipe_hash"], frozen["recipe_hash"])
                self.assertEqual(entry["recipe"], make_history_recipe({"family": "constant"}, CONFIG, 0.5))
                receipt = json.loads(Path(entry["receipt"]).read_text(encoding="utf-8"))
                self.assertEqual((receipt["code_hash"], receipt["recipe_hash"]),
                                 (frozen["code_hash"], frozen["recipe_hash"]))
                invocation = Path(entry["receipt"]).parent
                raw_fit = pd.read_csv(invocation / "fit.csv", dtype=str, keep_default_na=False)
                self.assertEqual(set(raw_fit["entity"]), set(ENTITIES))
            self.assertEqual(report["evaluations"]["selection"]["split"]["strategy"], "time")
            self.assertEqual(report["evaluations"]["final_check"]["split"]["strategy"], "time")
            self.assertEqual(validate_predictions(task.parent), {"valid": True, "errors": []})
            submission = pd.read_csv(task.parent / "submission.csv", dtype=str, keep_default_na=False)
            self.assertEqual(submission["row_key"].tolist(), TEST_IDS)
            np.testing.assert_allclose(submission["outcome"].astype(float), [22.5, 27.5, 35], rtol=0, atol=1e-12)
            refit_receipt = Path(report["candidate_receipts"]["refit"])
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             (refit_receipt.parent / "predictions.csv").read_bytes())

    def test_outer_target_intervention_preserves_inner_view_measurement_and_frozen_recipe(self):
        evidence = []
        outer_metrics = []
        with tempfile.TemporaryDirectory() as directory:
            for label, intervention in (("original", False), ("intervened", True)):
                task = make_panel(Path(directory) / label, intervene_outer=intervention)
                _, report, executions, views, _ = self.run_panel(task)
                self.assert_no_api_and_one_history(report, executions)
                evidence.append((views, report["frozen_baseline"], report["frozen_candidate"],
                                 report["evaluations"]["selection"],
                                 report["history_research"]["feedback"]["metric"]))
                outer_metrics.append(report["evaluations"]["final_check"]["candidate"]["value"])
        self.assertEqual(evidence[0], evidence[1])
        self.assertGreater(outer_metrics[1], outer_metrics[0] + 100000)

    def test_failed_history_refit_publishes_the_real_verified_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_panel(directory)
            _, report, executions, _, failed_refits = self.run_panel(task, fail_history_refit=True)
            self.assert_no_api_and_one_history(report, executions)
            self.assertEqual(report["evaluations"]["final_check"]["selected"], "candidate")
            self.assertEqual((report["status"], report["reason"]), ("fallback", "refit_failed"))
            self.assertEqual(report["published_recipe"], "baseline")
            self.assertEqual(len(failed_refits), 1)
            self.assertEqual(failed_refits[0]["code_hash"], report["frozen_candidate"]["code_hash"])
            baseline_refit = [entry for entry in executions if entry["phase"] == "full_refit"
                              and entry["code_hash"] == report["frozen_baseline"]["code_hash"]]
            self.assertEqual(len(baseline_refit), 1)
            self.assertEqual(validate_predictions(task.parent), {"valid": True, "errors": []})
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             Path(baseline_refit[0]["output"]).read_bytes())
            submission = pd.read_csv(task.parent / "submission.csv", dtype=str, keep_default_na=False)
            self.assertEqual(submission["row_key"].tolist(), TEST_IDS)
            np.testing.assert_array_equal(submission["outcome"].astype(float), [35, 35, 35])


class ConfiguredHistoryGuardTests(unittest.TestCase):
    def make_state(self, directory):
        run_dir = Path(directory)
        source = run_dir / "baseline.py"
        source.write_text(CONSTANT_CODE, encoding="utf-8")
        baseline = freeze_candidate(source, {"family": "constant"}, run_dir / "baseline_frozen")
        marker = run_dir / "input.csv"
        marker.write_bytes(b"original input\n")

        def check_source():
            if marker.read_bytes() != b"original input\n":
                raise ValueError("Source input changed")

        gate = SimpleNamespace(manifest={}, spec=SimpleNamespace(metric=SimpleNamespace(task_type="regression"),
            features=("entity", "period")), choice={"strategy": "time", "time_column": "period"},
            inner_split={"strategy": "time", "time_column": "period"}, fit_rows=[0, 1], held_rows=[2],
            research_spec=SimpleNamespace(), expected_inner_actuals=[0.0],
            check_inputs=Mock(side_effect=check_source))
        metric = {"name": "RMSE", "value": 1.0, "higher_is_better": False}
        state = SimpleNamespace(run_dir=run_dir, gate=gate, report={"stage": "inner_research", "extensions": {},
            "evaluations": {"selection": None}, "improvements": 0}, dialogue_closed=False, baseline=baseline,
            baseline_inner=metric, best_inner=metric, selected_candidate=None, tried_recipes=[],
            research_timeout=Mock(return_value=90), check_source=Mock(side_effect=check_source), save=Mock(),
            event=Mock(), measure_inner=Mock(return_value=SimpleNamespace(metric={**metric, "value": 2.0})),
            finalization_plan=Mock(return_value=SimpleNamespace(stage_seconds={"candidate_outer": 1,
                "candidate_refit": 1}, estimated_seconds=4)), remaining=Mock(return_value=90))
        return state, marker

    def test_disabled_history_never_assesses_a_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.make_state(directory)
            with patch("agent.research_v3._assess_candidate") as assess:
                report = research_configured_history(state, None)
            self.assertEqual((report["status"], report["reason"], report["attempted"]),
                             ("disabled", "not_configured", 0))
            assess.assert_not_called()

    def test_ineligible_cases_skip_without_fit_or_changes_to_frozen_baseline(self):
        cases = {
            "classification": "regression_required", "random_outer": "time_validation_required",
            "wrong_outer_period": "period_column_mismatch", "random_inner": "inner_time_validation_required",
            "wrong_inner_period": "inner_time_validation_required", "missing_columns": "history_columns_not_in_features",
            "no_baseline": "measured_baseline_required", "no_baseline_metric": "measured_baseline_required",
            "zero_weight": "zero_weight", "budget_two": "finalization_reserve",
            "budget_below_two": "finalization_reserve", "no_inner": "inner_validation_unavailable",
            "closed": "research_closed", "unavailable_semantics": "task_semantics_unavailable"}
        for case, reason in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                state, _ = self.make_state(directory)
                options = normalize_history_options(CONFIG)
                if case == "classification":
                    state.gate.spec.metric.task_type = "classification"
                elif case == "random_outer":
                    state.gate.choice = {"strategy": "random"}
                elif case == "wrong_outer_period":
                    state.gate.choice["time_column"] = "other"
                elif case == "random_inner":
                    state.gate.inner_split = {"strategy": "random"}
                elif case == "wrong_inner_period":
                    state.gate.inner_split["time_column"] = "other"
                elif case == "missing_columns":
                    state.gate.spec.features = ("period",)
                elif case == "no_baseline":
                    state.baseline = None
                elif case == "no_baseline_metric":
                    state.baseline_inner = None
                elif case == "zero_weight":
                    options["blend_weight"] = 0.0
                elif case in {"budget_two", "budget_below_two"}:
                    state.research_timeout.return_value = 2 if case == "budget_two" else 1
                elif case == "no_inner":
                    state.gate.held_rows = []
                elif case == "closed":
                    state.dialogue_closed = True
                elif case == "unavailable_semantics":
                    state.gate.manifest = {"task_semantics": {"adaptive_research_allowed": False}}
                baseline = state.baseline
                with patch("agent.research_v3._assess_candidate") as assess:
                    result = research_configured_history(state, options)
                self.assertEqual((result["status"], result["reason"], result["attempted"]), ("skipped", reason, 0))
                self.assertIs(state.baseline, baseline)
                if baseline is not None:
                    baseline.verify()
                assess.assert_not_called()
                state.measure_inner.assert_not_called()
                self.assertFalse((state.run_dir / "history_frozen").exists())

    def test_not_better_is_a_verified_single_trial_and_keeps_the_previous_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.make_state(directory)
            baseline = state.baseline
            previous_metric = state.best_inner
            with patch("agent.research_v3.inner_feedback", return_value={"available": False}):
                result = research_configured_history(state, normalize_history_options(CONFIG))
            self.assertEqual((result["status"], result["reason"], result["attempted"]), ("completed", "not_better", 1))
            self.assertTrue(result["feedback"]["verified"])
            self.assertFalse(result["feedback"]["accepted"])
            self.assertEqual(state.report["history_successful_experiments"], 1)
            self.assertEqual(state.measure_inner.call_count, 1)
            self.assertIsNone(state.selected_candidate)
            self.assertIs(state.best_inner, previous_metric)
            self.assertIs(state.baseline, baseline)
            baseline.verify()

    def test_failed_assessment_cannot_swallow_mutation_of_source_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            state, marker = self.make_state(directory)

            def mutate_and_fail(*args, **kwargs):
                marker.write_bytes(b"changed input\n")
                raise ValueError("Fixture candidate failed after source mutation")

            with patch("agent.research_v3._assess_candidate", side_effect=mutate_and_fail), \
                    self.assertRaisesRegex(ValueError, "Source input changed"):
                research_configured_history(state, normalize_history_options(CONFIG))

    def test_failed_assessment_cannot_swallow_mutation_of_frozen_baseline_code(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.make_state(directory)

            def mutate_and_fail(*args, **kwargs):
                state.baseline.code_path.write_text("changed baseline", encoding="utf-8")
                raise ValueError("Fixture candidate failed after baseline mutation")

            with patch("agent.research_v3._assess_candidate", side_effect=mutate_and_fail), \
                    self.assertRaisesRegex(ValueError, "Frozen candidate code or recipe changed"):
                research_configured_history(state, normalize_history_options(CONFIG))


if __name__ == "__main__":
    unittest.main()
