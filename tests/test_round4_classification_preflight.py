"""OOF eligibility must not spend model fits on impossible admission support."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agent.classification_decision import fit_guarded_decision_rule, predict_with_rule
from agent.oof_classification import OOFContractError, fit_oof_guarded_rule
from tests.test_oof_classification import Callback, Clock, CLASSES, fixture, known_probabilities


class Round4PreflightTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.tuning, self.selection, self.folds = fixture()

    def run_rule(self, callback, **overrides):
        arguments = dict(groups=self.groups, split_policy="grouped", tuning_rows=self.tuning,
                         selection_rows=self.selection, folds=self.folds, fit_predict=callback,
                         fold_timeout_seconds=10., deadline=100., clock=Clock())
        arguments.update(overrides)
        return fit_oof_guarded_rule(self.X, self.y, CLASSES, **arguments)

    def assert_preflight(self, expected_reason, **overrides):
        callback = Mock(side_effect=AssertionError("Ineligible admission must not train"))
        before = (self.X.copy(deep=True), self.y.copy(), self.groups.copy(),
                  self.tuning.copy(), self.selection.copy(), deepcopy(self.folds))
        with patch("agent.oof_classification._subset", side_effect=AssertionError("No staging needed")):
            result = self.run_rule(callback, **overrides)
        callback.assert_not_called()
        self.assertEqual(result["reason"], expected_reason)
        self.assertEqual(result["status"], "identity")
        self.assertEqual(result["completed_fits"], 0)
        self.assertEqual(result["fits"], [])
        self.assertIsNone(result["guarded"])
        self.assertEqual(result["timing"]["calibration_fit_seconds"], 0.)
        self.assertEqual(result["rule"]["multipliers"], [1.] * len(CLASSES))
        self.assertEqual(result["candidate_rule"]["multipliers"], [1.] * len(CLASSES))
        probabilities = known_probabilities(self.X)
        saved = probabilities.copy()
        predicted = predict_with_rule(probabilities, CLASSES, result["rule"])
        np.testing.assert_array_equal(predicted, np.asarray(CLASSES)[probabilities.argmax(axis=1)])
        np.testing.assert_array_equal(probabilities, saved)
        pd.testing.assert_frame_equal(self.X, before[0])
        for actual, original in zip((self.y, self.groups, self.tuning, self.selection), before[1:5]):
            np.testing.assert_array_equal(actual, original)
        for actual, original in zip(self.folds, before[5]):
            for name in ("fit_rows", "held_rows"):
                np.testing.assert_array_equal(actual[name], original[name])
        json.dumps(result, allow_nan=False)
        return result

    def test_missing_selection_class_skips_all_four_fits_with_existing_reason(self):
        self.y[self.selection] = "01"
        result = self.assert_preflight("selection_missing_class")
        self.assertEqual(result["support"]["selection"], [90, 0, 0])

    def test_selection_support_matches_guard_and_keeps_reason_precedence(self):
        # Conflict labels inside duplicate groups remain one feature group.
        self.y[self.selection] = "01"
        self.y[self.selection[:4]] = "class c"
        self.y[self.selection[4:10]] = "NA"
        options = {"min_selection_class_count": 5}
        joined = np.r_[self.tuning, self.selection]
        original_guard = fit_guarded_decision_rule(
            known_probabilities(self.X)[joined], self.y[joined], CLASSES,
            tuning_rows=np.arange(len(self.tuning)),
            selection_rows=np.arange(len(self.tuning), len(self.X)), options=options)
        result = self.assert_preflight(original_guard["diagnostics"]["reason"], guard_options=options)
        self.assertEqual(result["reason"], "selection_insufficient_class_support")
        self.assertEqual(result["support"]["selection"], [80, 6, 4])
        # When sample count and per-class count both fail, sample count is first.
        self.assert_preflight("selection_insufficient_samples",
                              guard_options={"min_selection_samples": 91, "min_selection_class_count": 31})
        self.y[self.selection] = "01"
        self.assert_preflight("selection_missing_class", guard_options={"min_selection_samples": 91})

    def test_custom_support_threshold_is_respected_without_relaxing_guard(self):
        self.assert_preflight("selection_insufficient_class_support",
                              guard_options={"min_selection_class_count": 31})
        self.assert_preflight("selection_insufficient_samples", guard_options={"min_selection_samples": 91})
        callback = Callback()
        result = self.run_rule(callback, guard_options={"min_selection_samples": 90, "min_selection_class_count": 30})
        self.assertEqual(len(callback.calls), 4)
        self.assertEqual(result["reason"], "accepted")

    def test_insufficient_B_cannot_hide_group_leakage_or_incomplete_oof(self):
        self.y[self.selection] = "01"
        callback = Mock()
        bad_groups = self.groups.copy()
        bad_groups[self.selection[0]] = bad_groups[self.tuning[0]]
        with self.assertRaisesRegex(OOFContractError, "A/B groups"):
            self.run_rule(callback, groups=bad_groups)
        bad_folds = deepcopy(self.folds)
        bad_folds[0]["held_rows"] = bad_folds[0]["held_rows"][:-1]
        with self.assertRaisesRegex(OOFContractError, "cover every A row"):
            self.run_rule(callback, folds=bad_folds)
        bad_folds = deepcopy(self.folds)
        bad_folds[0]["fit_rows"][0] = self.selection[0]
        with self.assertRaisesRegex(OOFContractError, "B cannot enter any fit"):
            self.run_rule(callback, folds=bad_folds)
        callback.assert_not_called()

    def test_rare_class_in_one_feature_group_cannot_be_rescued_by_duplicate_rows(self):
        # Many rare rows from one indivisible group still cannot cover A and B.
        self.y[:] = np.asarray(CLASSES[:2], dtype=object)[self.groups % 2]
        rare = self.tuning[:20]
        self.y[rare] = "class c"
        self.groups[rare] = 999
        self.X.iloc[rare] = [999, "same-rare-feature-group"]
        result = self.assert_preflight("fit_missing_class")
        self.assertEqual(result["support"]["allowed"][2], 20)
        self.assertEqual(len(set(self.groups[self.y == "class c"])), 1)
        self.assertEqual(result["support"]["folds"][0]["fit"][2], 0)

    def test_rare_class_only_in_B_keeps_earlier_A_missing_class_reason(self):
        self.y[self.tuning] = np.asarray(CLASSES[:2], dtype=object)[self.groups[self.tuning] % 2]
        self.assert_preflight("tuning_missing_class")

    def test_eligible_B_labels_can_only_admit_the_same_A_selected_candidate(self):
        first_callback = Callback()
        first = self.run_rule(first_callback)
        # Preserve every B class count, while changing all B answers.
        self.y[self.selection] = np.roll(self.y[self.selection], 2)
        second_callback = Callback()
        second = self.run_rule(second_callback)
        self.assertEqual(first["candidate_rule"], second["candidate_rule"])
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(second["status"], "identity")
        self.assertEqual(len(first_callback.calls), 4)
        self.assertEqual(len(second_callback.calls), 4)
        for first_fit, second_fit in zip(first_callback.calls, second_callback.calls):
            np.testing.assert_array_equal(first_fit["y"], second_fit["y"])
            self.assertFalse(set(first_fit["fit"]) & set(self.selection))


if __name__ == "__main__":
    unittest.main()
