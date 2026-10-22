"""Check new fold-local feature/calibration boundaries without model training."""

from copy import deepcopy
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import regression_base as legacy
from agent import round4_regression_base as base

# Reuse the existing fit-spy contract fixture; its tests stay in their file.
from test_rescue_regression_base import fixture, spies


class Round4RegressionBaseTests(unittest.TestCase):
    def test_disabled_features_legacy_alignment_delegate_unchanged_algorithm(self):
        X, y, predict, context, recipe = fixture()
        with spies():
            expected, original, original_calibration = legacy.fit_base(X, y, predict, context, recipe)
        with spies():
            actual, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        np.testing.assert_array_equal(expected, actual)
        for name in original_calibration:
            np.testing.assert_array_equal(original_calibration[name], calibration[name])
        self.assertEqual(evidence["family"], original["family"])
        self.assertEqual(evidence["selected_iterations"], original["selected_iterations"])
        self.assertEqual(evidence["physical_fits"], 2)
        self.assertEqual(evidence["round4_execution"], "unchanged_rescue_v1_delegate")

    def test_feature_frequencies_fit_independently_on_earlier_and_full_rows(self):
        X, y, predict, context, recipe = fixture()
        before = (X.copy(deep=True), y.copy(deep=True), predict.copy(deep=True), deepcopy(context), deepcopy(recipe))
        with spies() as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe,
                feature_config={"category_frequencies": True, "missing_indicators": True})
        added = "__round4_frequency__kind"
        early = calls["early"][0]["model"]
        np.testing.assert_array_equal(early.predict_frame[added], np.zeros(2))
        final = calls["refit"][0]["model"]
        held_only = final.train["kind"] == "held_only"
        np.testing.assert_array_equal(final.train.loc[held_only, added], np.full(2, 2 / 6))
        self.assertEqual(len(final.train), len(X))
        self.assertEqual(evidence["physical_fits"], 2)
        self.assertEqual(set(calibration["fit_positions"]), {1, 2, 4, 5})
        pd.testing.assert_frame_equal(X, before[0])
        pd.testing.assert_series_equal(y, before[1])
        pd.testing.assert_frame_equal(predict, before[2])
        self.assertEqual(context, before[3])
        self.assertEqual(recipe, before[4])

    def test_clamped_catboost_recheck_uses_only_earlier_labels_and_same_tree_count(self):
        X, y, predict, context, recipe = fixture()
        with spies(best=10) as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe,
                calibration_alignment="refit_count")
        self.assertEqual(evidence["best_iteration"], 11)
        self.assertEqual(evidence["selected_iterations"], 50)
        self.assertEqual(evidence["calibration_iterations"], 50)
        self.assertEqual(evidence["calibration_recheck_fits"], 1)
        self.assertEqual(evidence["physical_fits"], 3)
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(len(calls["refit"]), 2)
        earlier, final = (item["model"] for item in calls["refit"])
        self.assertEqual(set(earlier.target), {10., 20., 30., 40.})
        self.assertEqual(set(final.target), set(y))
        for item in calls["refit"]:
            self.assertEqual(item["args"][4]["parameters"]["iterations"], 50)
            self.assertLessEqual(item["model"].kwargs["model__callbacks"][0].deadline,
                                 context["deadline_monotonic"])
        expected = pd.to_numeric(X.iloc[calibration["validation_positions"]]["x"]).to_numpy() + earlier.target.mean()
        np.testing.assert_array_equal(calibration["predictions"], expected)

    def test_clamped_lightgbm_recheck_uses_selected_50_not_best_31(self):
        X, y, predict, context, recipe = fixture()
        recipe.update(engine="lightgbm", parameters={"num_leaves": 15})
        with spies(best=31) as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe,
                calibration_alignment="refit_count", feature_config={"category_frequencies": True})
        self.assertEqual(evidence["best_iteration"], 31)
        self.assertEqual(evidence["calibration_iterations"], 50)
        self.assertEqual(evidence["physical_fits"], 3)
        self.assertEqual(calls["early"][0]["model"].predict_kwargs, {"num_iteration": 31})
        self.assertEqual(calls["refit"][0]["args"][4]["parameters"]["n_estimators"], 50)
        self.assertEqual(len(calls["refit"][0]["model"].train), len(calibration["fit_positions"]))

    def test_alignment_adds_no_recheck_when_best_already_at_least_50(self):
        X, y, predict, context, recipe = fixture()
        with spies(best=77) as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe,
                calibration_alignment="refit_count")
        self.assertEqual(evidence["calibration_iterations"], 78)
        self.assertEqual(evidence["selected_iterations"], 78)
        self.assertEqual(evidence["physical_fits"], 2)
        self.assertEqual(len(calls["refit"]), 1)
        np.testing.assert_array_equal(calibration["predictions"], [25., 25.])

    def test_one_period_remains_one_full_fit_with_no_calibration(self):
        X, y, predict, context, recipe = fixture()
        X["period"] = "1"
        with spies() as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe,
                calibration_alignment="refit_count", feature_config={"missing_indicators": True})
        self.assertEqual(len(calls["early"]), 0)
        self.assertEqual(len(calls["refit"]), 1)
        self.assertIsNone(calibration)
        self.assertEqual(evidence["physical_fits"], 1)
        self.assertEqual(evidence["selected_iterations"], 100)

    def test_failed_alignment_aborts_without_running_full_refit(self):
        X, y, predict, context, recipe = fixture()
        with spies(best=10) as calls, patch.object(base, "_fixed_fit", side_effect=TimeoutError("recheck timeout")) as fixed:
            with self.assertRaisesRegex(TimeoutError, "recheck timeout"):
                base.fit_base(X, y, predict, context, recipe, calibration_alignment="refit_count")
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(fixed.call_count, 1)
        self.assertEqual(len(calls["refit"]), 0)

    def test_expired_deadline_runs_no_model(self):
        X, y, predict, context, recipe = fixture()
        context["deadline_monotonic"] = time.monotonic() - 1
        with spies() as calls:
            with self.assertRaises((TimeoutError, ValueError)):
                base.fit_base(X, y, predict, context, recipe,
                    feature_config={"category_frequencies": True})
        self.assertEqual(calls["early"], [])
        self.assertEqual(calls["refit"], [])


if __name__ == "__main__":
    unittest.main()
