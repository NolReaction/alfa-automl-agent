"""Model-free spies for A-only selection, exact fallback and frozen lifecycle."""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import regression_base as legacy_base
from agent import round6_regression_base as base
from agent import round6_regression_candidate as candidate
from agent.metrics import get_metric
from agent.regression_candidate import make_regression_recipe, execute_regression_candidate
from agent.round6_regression_calibration import split_calibration
from tests.test_rescue_regression_base import spies


def fixture(*, keys=500, history=False):
    rows = [{"x": str(key), "period": str(period), "kind": "kind_" + str(key % 3)}
            for period in (1, 2, 3) for key in range(keys)]
    X = pd.DataFrame(rows, index=np.arange(len(rows)) * 11 + 7)
    y = pd.Series([10. * int(row["period"]) + 2. * int(row["x"]) for row in rows], index=X.index)
    query = pd.DataFrame({"x": ["0", "1", "2"], "period": ["2", "4", "4"],
                          "kind": ["query_only", "kind_1", "kind_2"]}, index=[75, 9, 123])
    schema = {"key_columns": ["x"], "period_column": "period", "numeric_columns": ["x"],
              "positive_columns": [], "ambiguity_policy": "past_unique_only"} if history else None
    metric = get_metric("RMSE")
    context = {"features": list(X), "numeric": ["x", "period"], "categorical": ["kind"],
        "task_type": "regression", "prediction_kind": metric.prediction_kind,
        "metric": metric.to_dict(), "deadline_monotonic": time.monotonic() + 180,
        "recipe": candidate.make_round6_recipe("period", "S20", history_config=schema)}
    return X, y, query, context


def aligned_without_models(prepared, a_positions, **_budgets):
    return base.AlignedCalibration(50, prepared.y.iloc[prepared.held].to_numpy() - 5.,
        {"selected_iterations": 50, "calibration_iterations": 50,
         "physical_fits_expected_completed": 2, "prefix_recheck_fits": 1})


def refit_without_model(prepared, selected_iterations, **_budgets):
    return pd.to_numeric(prepared.predict["x"]).to_numpy() + 10., {
        "selected_iterations": selected_iterations, "fresh_full_refit": True}


class Round6CandidateTests(unittest.TestCase):
    def test_complete_frozen_recipe_and_declared_roles_reject_changes_before_fit(self):
        X, y, query, context = fixture()
        recipe = context["recipe"]
        self.assertEqual(candidate._recipe(json.loads(json.dumps(recipe))), recipe)
        detached = candidate._recipe(recipe)
        detached["base"]["parameters"]["depth"] = 99
        self.assertEqual(recipe["base"]["parameters"]["depth"], 6)
        variants = []
        for key, value in (("version", True), ("arm", "UNKNOWN"), ("path", "private")):
            other = deepcopy(recipe); other[key] = value; variants.append(other)
        other = deepcopy(recipe); other["base"]["parameters"]["depth"] = 7; variants.append(other)
        other = deepcopy(recipe); other["calibration"]["rules"]["numerics"]["ridge_penalty"] = 3.; variants.append(other)
        other = deepcopy(recipe); other["resources"]["trial_fraction_of_initial_model_window"] = .9; variants.append(other)
        for other in variants:
            with self.subTest(other=other), self.assertRaises(ValueError):
                candidate._recipe(other)
        with self.assertRaises(ValueError):
            candidate.make_round6_recipe("period", "I")
        with patch.object(candidate, "stage_calibration") as stage, \
                patch.object(candidate, "execute_regression_candidate") as fallback:
            for badX, badY, badQuery in ((X.assign(undeclared_id="x"), y, query),
                                       (X, y.iloc[:-1], query),
                                       (X, y, query.assign(external_held_target=123.))):
                with self.assertRaises(ValueError):
                    candidate.execute_round6_candidate(badX, badY, badQuery, context)
            stage.assert_not_called(); fallback.assert_not_called()

    def test_selection_argument_excludes_B_and_recheck_refit_use_aligned_count(self):
        X, y, query, context = fixture()
        prepared = base.prepare_inputs(X, y, query, context, context["recipe"]["base"])
        a, b, split = split_calibration(prepared.X.iloc[prepared.held], prepared.features)
        self.assertTrue(split["supported"])
        expected_A = prepared.y.iloc[prepared.held[a]].to_numpy()
        expected_prefix = prepared.y.iloc[prepared.fitting].to_numpy()
        seen_arguments = []
        original_internal = legacy_base._internal_fit

        def selection(frame, target, fitting, held, *args):
            seen_arguments.append((frame.copy(), target.copy(), fitting.copy(), held.copy()))
            self.assertEqual(len(frame), len(expected_prefix) + len(expected_A))
            np.testing.assert_array_equal(target.iloc[held], expected_A)
            np.testing.assert_array_equal(target.iloc[fitting], expected_prefix)
            self.assertTrue((pd.to_numeric(frame.iloc[fitting]["period"]) < 3).all())
            return original_internal(frame, target, fitting, held, *args)

        with spies(best=10) as calls, patch.object(legacy_base, "_internal_fit", side_effect=selection):
            aligned = base.stage_calibration(prepared, a, deadline=context["deadline_monotonic"] - 20,
                original_deadline=context["deadline_monotonic"])
            _, final = base.refit_full(prepared, aligned.selected_iterations,
                deadline=context["deadline_monotonic"] - 8, original_deadline=context["deadline_monotonic"])
        self.assertEqual(aligned.selected_iterations, 50)
        self.assertEqual(aligned.evidence["calibration_iterations"], 50)
        self.assertFalse(aligned.evidence["B_labels_available_to_selector"])
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(len(calls["refit"]), 2)
        self.assertTrue(all((pd.to_numeric(frame["period"]) < 3).all() for frame in calls["feature_fit"]))
        for model in calls["refit"]:
            self.assertEqual(model["args"][4]["parameters"]["iterations"], 50)
            self.assertNotIn("eval_set", model["model"].kwargs)
        self.assertEqual(len(calls["refit"][0]["model"].train), len(prepared.fitting))
        self.assertEqual(len(calls["refit"][0]["model"].predict_frame), len(prepared.held))
        self.assertEqual(len(calls["refit"][1]["model"].train), len(X))
        self.assertIsNot(calls["refit"][0]["model"], calls["refit"][1]["model"])
        self.assertTrue(final["fresh_full_refit"])
        changed = deepcopy(prepared)
        changed.y.iloc[changed.held[b]] = 999999.
        with spies(best=10), patch.object(legacy_base, "_internal_fit", side_effect=selection):
            aligned2 = base.stage_calibration(changed, a, deadline=context["deadline_monotonic"] - 20,
                original_deadline=context["deadline_monotonic"])
        np.testing.assert_array_equal(aligned.predictions, aligned2.predictions)
        pd.testing.assert_frame_equal(seen_arguments[0][0], seen_arguments[1][0])
        pd.testing.assert_series_equal(seen_arguments[0][1], seen_arguments[1][1])

    def test_A_policy_and_B_gate_freeze_before_fresh_refit_without_input_mutation(self):
        X, y, query, context = fixture()
        before = deepcopy((X, y, query, context))
        prepared = base.prepare_inputs(X, y, query, context, context["recipe"]["base"])
        a, b, _ = split_calibration(prepared.X.iloc[prepared.held], prepared.features)
        events = []
        original_fit, original_gate, original_history = candidate.fit_correction, candidate.gate_correction, candidate._history_selection

        def stage(*args, **kwargs):
            events.append("selection_and_recheck")
            return aligned_without_models(*args, **kwargs)
        def fit(arm, target, predictions):
            events.append("fit_A")
            np.testing.assert_array_equal(target, prepared.y.iloc[prepared.held[a]])
            np.testing.assert_array_equal(predictions, prepared.y.iloc[prepared.held[a]] - 5.)
            observed = base.active_component()
            self.assertEqual(observed["phase"], "correction_fit_A")
            self.assertEqual(observed["engine"], "analytic_residual")
            self.assertEqual(observed["id"], "residual_correction")
            self.assertEqual(observed["original_deadline_monotonic"], context["deadline_monotonic"])
            return original_fit(arm, target, predictions)
        def gate(*args, **kwargs):
            events.append("gate_B")
            pd.testing.assert_frame_equal(args[0], prepared.X.iloc[prepared.fitting])
            pd.testing.assert_frame_equal(args[2], prepared.X.iloc[prepared.held[b]])
            np.testing.assert_array_equal(args[3], prepared.y.iloc[prepared.held[b]])
            np.testing.assert_array_equal(args[4], prepared.y.iloc[prepared.held[b]] - 5.)
            return original_gate(*args, **kwargs)
        def history(*args, **kwargs):
            events.append("freeze_history")
            return original_history(*args, **kwargs)
        def refit(*args, **kwargs):
            events.append("fresh_full_refit")
            return refit_without_model(*args, **kwargs)

        with patch.object(candidate, "stage_calibration", side_effect=stage), \
                patch.object(candidate, "fit_correction", side_effect=fit) as fitter, \
                patch.object(candidate, "gate_correction", side_effect=gate) as gate_call, \
                patch.object(candidate, "_history_selection", side_effect=history), \
                patch.object(candidate, "refit_full", side_effect=refit), \
                patch.object(candidate, "execute_regression_candidate") as fallback:
            result, evidence, alternatives = candidate.execute_round6_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(events, ["selection_and_recheck", "fit_A", "gate_B", "freeze_history", "fresh_full_refit"])
        fitter.assert_called_once(); gate_call.assert_called_once(); fallback.assert_not_called()
        self.assertEqual(evidence["correction"]["status"], "applied")
        self.assertFalse(evidence["correction"]["parameter_refit_after_B"])
        self.assertAlmostEqual(evidence["correction"]["policy"]["offset"], 1.)
        self.assertEqual(evidence["physical_fits_expected_completed"], 3)
        self.assertEqual(evidence["correction"]["analytic_parameter_fit_calls"], 1)
        self.assertIn("not_incumbent_I", evidence["identity_ablation"])
        np.testing.assert_array_equal(alternatives["base"], alternatives["identity"] + 1.)
        np.testing.assert_array_equal(result["predictions"], alternatives["calibrated"])
        for key in ("eligible_rows", "ambiguous_rows", "no_history_rows"):
            self.assertIsNone(evidence["history"][key])
        pd.testing.assert_frame_equal(X, before[0]); pd.testing.assert_series_equal(y, before[1])
        pd.testing.assert_frame_equal(query, before[2]); self.assertEqual(context, before[3])

    def test_single_period_and_insufficient_support_delegate_exact_incumbent_without_trial(self):
        for single_period in (True, False):
            X, y, query, context = fixture(keys=12)
            if single_period:
                X["period"] = "1"
            legacy_context = deepcopy(context)
            legacy_context["recipe"] = make_regression_recipe("period")
            with spies() as baseline_calls:
                expected, old_evidence, old_alternatives = execute_regression_candidate(X, y, query, legacy_context, return_evidence=True)
            with spies() as calls, patch.object(candidate, "stage_calibration") as stage, \
                    patch.object(candidate, "fit_correction") as correction_fit, \
                    patch.object(candidate, "refit_full") as altered_refit:
                actual, evidence, alternatives = candidate.execute_round6_candidate(X, y, query, context, return_evidence=True)
            np.testing.assert_array_equal(actual["predictions"], expected["predictions"])
            np.testing.assert_array_equal(alternatives["base"], old_alternatives["base"])
            self.assertEqual(evidence["correction"]["status"], "no_calibration" if single_period else "insufficient_support")
            self.assertTrue(evidence["incumbent_fallback"])
            self.assertEqual(len(calls["early"]), len(baseline_calls["early"]))
            self.assertEqual(len(calls["refit"]), 1)
            self.assertEqual(calls["refit"][0]["args"][4], baseline_calls["refit"][0]["args"][4])
            stage.assert_not_called(); correction_fit.assert_not_called(); altered_refit.assert_not_called()

    def test_B_rejection_returns_actual_incumbent_with_original_deadline(self):
        X, y, query, context = fixture()
        legacy_context = deepcopy(context); legacy_context["recipe"] = make_regression_recipe("period")
        with spies():
            expected = execute_regression_candidate(X, y, query, legacy_context)
        with spies(), patch.object(candidate, "stage_calibration", side_effect=aligned_without_models), \
                patch.object(candidate, "gate_correction", return_value={"accepted": False, "reason": "B_SSE_worsened"}), \
                patch.object(candidate, "refit_full") as altered_refit, \
                patch.object(candidate, "execute_regression_candidate", wraps=execute_regression_candidate) as incumbent:
            result, evidence, _ = candidate.execute_round6_candidate(X, y, query, context, return_evidence=True)
        np.testing.assert_array_equal(result["predictions"], expected["predictions"])
        self.assertEqual(evidence["correction"]["status"], "B_rejected")
        self.assertEqual(evidence["physical_fits_expected_completed"], 4)
        altered_refit.assert_not_called()
        supplied_context = incumbent.call_args.args[3]
        self.assertEqual(supplied_context["recipe"], legacy_context["recipe"])
        self.assertEqual(supplied_context["deadline_monotonic"], context["deadline_monotonic"])

    def test_strict_past_history_sees_corrected_calibration_before_full_refit(self):
        X, y, query, context = fixture(history=True)
        seen = []
        def select(signals, labels, corrected):
            seen.append((signals.copy(), np.asarray(labels).copy(), np.asarray(corrected).copy()))
            return {"mode": "last", "weight": 1., "shrinkage": 0}
        with patch.object(candidate, "stage_calibration", side_effect=aligned_without_models), \
                patch.object(candidate, "refit_full", side_effect=refit_without_model), \
                patch.object(candidate, "select_history", side_effect=select):
            result, evidence, _ = candidate.execute_round6_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(evidence["correction"]["status"], "applied")
        self.assertEqual(len(seen), 1)
        np.testing.assert_allclose(seen[0][2], seen[0][1] - 4.)
        # Query period2 must not observe supplied-fit labels from periods2/3.
        np.testing.assert_allclose(result["predictions"], [10., 32., 34.])
        self.assertTrue(evidence["history"]["schema_available"])

    def test_deadline_exhaustion_cannot_reset_allowance_or_publish_partial_fallback(self):
        X, y, query, context = fixture()
        clock = [100.]
        context["deadline_monotonic"] = 200.
        received = []
        def slow_stage(prepared, a, **budgets):
            received.append(budgets)
            clock[0] = budgets["deadline"] + 1.
            return aligned_without_models(prepared, a)
        with patch.object(candidate.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(candidate, "stage_calibration", side_effect=slow_stage), \
                patch.object(candidate, "refit_full") as refit, \
                patch.object(candidate, "execute_regression_candidate") as fallback:
            with self.assertRaises(TimeoutError):
                candidate.execute_round6_candidate(X, y, query, context)
        self.assertEqual(received[0]["original_deadline"], 200.)
        self.assertEqual(received[0]["deadline"], 138.)
        refit.assert_not_called(); fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
