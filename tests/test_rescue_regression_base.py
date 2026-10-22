"""Fit spies verify the rescue base contract; no boosting model is trained."""

from contextlib import contextmanager
from copy import deepcopy
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import regression_base as base
from agent.features import FeatureTransformer
from agent.metrics import get_metric
from agent.models import _LightGBMFrame


def fixture():
    X = pd.DataFrame({"x": ["6", "2", None, "5", "1", "4"],
                      "period": ["3", "1", "2", "3", "1", "2"],
                      "kind": ["held_only", "NA", "0007", "held_only", "NA", "0007"]},
                     index=[81, 16, 83, 54, 99, 12])
    y = pd.Series([60., 20., 30., 50., 10., 40.], index=X.index)
    predict = pd.DataFrame({"x": ["12", None], "period": ["4", "4"], "kind": ["query_only", "NA"]})
    metric = get_metric("RMSE")
    context = {"features": list(X.columns), "numeric": ["x", "period"], "categorical": ["kind"],
               "task_type": "regression", "metric": metric.to_dict(), "prediction_kind": metric.prediction_kind,
               "deadline_monotonic": time.monotonic() + 120}
    recipe = {"engine": "catboost", "period_column": "period", "max_iterations": 500,
              "stopping_rounds": 30, "fallback_iterations": 100, "parameters": {"depth": 5}}
    return X, y, predict, context, recipe


@contextmanager
def spies(*, best=77, early_fit=None, bad_predictions=False):
    calls = {"early": [], "refit": [], "feature_fit": [], "category_fit": []}

    class Features(FeatureTransformer):
        def fit(self, frame, target=None):
            calls["feature_fit"].append(frame.copy(deep=True))
            return super().fit(frame, target)

    class Categories(_LightGBMFrame):
        def fit(self, frame, target=None):
            result = super().fit(frame, target)
            calls["category_fit"].append(deepcopy(self.categories_))
            return result

    class Early:
        best_iteration_ = best

        def fit(self, frame, target, **kwargs):
            self.train = frame.copy(deep=True)
            self.target = np.asarray(target).copy()
            self.kwargs = kwargs
            if early_fit is not None:
                early_fit()
            return self

        def get_best_iteration(self):
            return best

        def predict(self, frame, **kwargs):
            self.predict_frame = frame.copy(deep=True)
            self.predict_kwargs = kwargs
            return np.full(len(frame), self.target.mean())

    class Refit:
        def fit(self, frame, target, **kwargs):
            self.train, self.target = frame.copy(deep=True), np.asarray(target).copy()
            self.kwargs = kwargs
            return self

        def predict(self, frame):
            self.predict_frame = frame.copy(deep=True)
            return (np.full(len(frame), np.nan) if bad_predictions else
                    pd.to_numeric(frame["x"], errors="coerce").fillna(0).to_numpy() + self.target.mean())

    def estimator(*args):
        model = Early()
        calls["early"].append({"args": deepcopy(args), "model": model})
        return model

    def pipeline(*args):
        model = Refit()
        calls["refit"].append({"args": deepcopy(args), "model": model})
        return model

    with patch.object(base, "_make_estimator", side_effect=estimator), \
            patch.object(base, "_lightgbm_callbacks", return_value=[]), \
            patch.object(base, "build_pipeline", side_effect=pipeline), \
            patch.object(base, "FeatureTransformer", Features), \
            patch.object(base, "_LightGBMFrame", Categories):
        yield calls


class RescueRegressionBaseTests(unittest.TestCase):
    def test_normalized_recipe_has_detached_bounded_quality_controls(self):
        *_, recipe = fixture()
        before = deepcopy(recipe)
        normalized = base.normalize_base_recipe(recipe)
        self.assertEqual(set(normalized), {"engine", "period_column", "max_iterations", "stopping_rounds", "fallback_iterations", "parameters"})
        self.assertNotIn("iterations", normalized["parameters"])
        self.assertEqual(normalized["parameters"]["depth"], 5)
        normalized["parameters"]["depth"] = 8
        self.assertEqual(recipe, before)

    def test_recipe_rejects_unbounded_or_runtime_settings(self):
        *_, recipe = fixture()
        variants = [{"max_iterations": 2001}, {"max_iterations": True}, {"stopping_rounds": 0},
                    {"fallback_iterations": 501}, {"engine": "gpu"}, {"period_column": ""},
                    {"parameters": {"thread_count": 48}}, {"parameters": {"iterations": 100}},
                    {"parameters": {"depth": 30}}, {"parameters": {"learning_rate": float("nan")}},
                    {"feature_config": {}}, {"path": "anything"}]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValueError):
                base.normalize_base_recipe({**recipe, **change})

    def test_calibration_positions_and_predictions_use_one_past_only_fit(self):
        X, y, predict, context, recipe = fixture()
        with spies() as calls:
            result, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(len(calls["refit"]), 1)
        self.assertEqual(set(calibration), {"fit_positions", "validation_positions", "predictions"})
        self.assertEqual(set(calibration["fit_positions"]), {1, 2, 4, 5})
        self.assertEqual(set(calibration["validation_positions"]), {0, 3})
        early = calls["early"][0]["model"]
        np.testing.assert_array_equal(early.target, y.iloc[calibration["fit_positions"]].to_numpy())
        np.testing.assert_array_equal(calibration["predictions"], [25., 25.])
        self.assertEqual(early.predict_kwargs, {"ntree_end": 78})
        self.assertEqual(evidence["selected_iterations"], 78)
        self.assertEqual(evidence["calibration_period"], 3)
        self.assertEqual(evidence["internal_fit_rows"], 4)
        self.assertEqual(evidence["internal_validation_rows"], 2)
        self.assertEqual(evidence["full_fit_rows"], 6)
        self.assertEqual(evidence["rss_status"], "not_observed")
        self.assertIsNone(evidence["peak_rss_bytes"])
        self.assertEqual(result.shape, (2,))

    def test_fresh_full_refit_uses_all_fit_rows_with_deadline_callback(self):
        X, y, predict, context, recipe = fixture()
        with spies() as calls:
            _, evidence, _ = base.fit_base(X, y, predict, context, recipe)
        final = calls["refit"][0]
        self.assertEqual(final["args"][4]["parameters"]["iterations"], 78)
        self.assertEqual(set(final["model"].target), set(y))
        self.assertEqual(len(final["model"].train), len(X))
        self.assertIn("model__callbacks", final["model"].kwargs)
        self.assertIsInstance(final["model"].kwargs["model__callbacks"][0], base._CatBoostDeadline)
        self.assertTrue(evidence["fresh_full_refit"])
        for key in ("internal_elapsed_seconds", "refit_elapsed_seconds", "predict_elapsed_seconds", "estimated_refit_seconds"):
            self.assertGreaterEqual(evidence[key], 0)

    def test_lightgbm_categories_fit_only_and_unknown_held_category_is_missing(self):
        X, y, predict, context, recipe = fixture()
        recipe.update(engine="lightgbm", parameters={"num_leaves": 15})
        with spies() as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        self.assertEqual(evidence["selected_iterations"], 77)
        self.assertNotIn("held_only", calls["category_fit"][0]["kind"])
        self.assertNotIn("query_only", calls["category_fit"][0]["kind"])
        early = calls["early"][0]["model"]
        self.assertTrue(early.predict_frame["feature_2"].isna().all())
        self.assertEqual(early.predict_kwargs, {"num_iteration": 77})
        np.testing.assert_array_equal(calibration["predictions"], [25., 25.])

    def test_single_period_fits_fixed_model_once_and_returns_no_calibration(self):
        X, y, predict, context, recipe = fixture()
        X["period"] = "1"
        with spies() as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        self.assertEqual(len(calls["early"]), 0)
        self.assertEqual(len(calls["refit"]), 1)
        self.assertIsNone(calibration)
        self.assertEqual(evidence["selected_iterations"], 100)
        self.assertEqual(evidence["reason"], "single_period_fixed_iterations")

    def test_missing_period_excluded_from_internal_but_kept_full_refit(self):
        X, y, predict, context, recipe = fixture()
        X.iloc[2, X.columns.get_loc("period")] = np.nan
        with spies() as calls:
            _, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        self.assertNotIn(2, calibration["fit_positions"])
        self.assertNotIn(2, calibration["validation_positions"])
        self.assertEqual(evidence["invalid_period_rows"], 1)
        self.assertEqual(len(calls["refit"][0]["model"].train), len(X))

    def test_row_permutation_and_changed_indices_preserve_training_content(self):
        X, y, predict, context, recipe = fixture()
        permutation = [4, 0, 5, 1, 3, 2]
        changed = X.iloc[permutation].copy()
        changed.index = ["unrelated"] * len(X)
        with spies() as first:
            result, _, calibration = base.fit_base(X, y, predict, context, recipe)
        with spies() as second:
            result2, _, calibration2 = base.fit_base(changed, y.iloc[permutation], predict, context, recipe)
        np.testing.assert_array_equal(result, result2)
        for kind in ("early", "refit"):
            pd.testing.assert_frame_equal(first[kind][0]["model"].train, second[kind][0]["model"].train)
            np.testing.assert_array_equal(first[kind][0]["model"].target, second[kind][0]["model"].target)
        np.testing.assert_array_equal(y.iloc[calibration["validation_positions"]],
                                      y.iloc[permutation].iloc[calibration2["validation_positions"]])

    def test_query_values_and_order_never_change_fitted_data_or_selected_trees(self):
        X, y, predict, context, recipe = fixture()
        changed = pd.concat([predict.iloc[::-1], pd.DataFrame({"x": ["1e100"], "period": ["99999"], "kind": ["never_seen"]})])
        with spies() as first:
            result, evidence, calibration = base.fit_base(X, y, predict, context, recipe)
        with spies() as second:
            result2, evidence2, calibration2 = base.fit_base(X, y, changed, context, recipe)
        for kind in ("early", "refit"):
            pd.testing.assert_frame_equal(first[kind][0]["model"].train, second[kind][0]["model"].train)
        np.testing.assert_array_equal(result, result2[:2][::-1])
        np.testing.assert_array_equal(calibration["predictions"], calibration2["predictions"])
        self.assertEqual(evidence["selected_iterations"], evidence2["selected_iterations"])

    def test_inputs_remain_unmodified(self):
        X, y, predict, context, recipe = fixture()
        originals = deepcopy((X, y, predict, context, recipe))
        with spies():
            base.fit_base(X, y, predict, context, recipe)
        pd.testing.assert_frame_equal(X, originals[0])
        pd.testing.assert_series_equal(y, originals[1])
        pd.testing.assert_frame_equal(predict, originals[2])
        self.assertEqual(context, originals[3])
        self.assertEqual(recipe, originals[4])

    def test_extra_undeclared_feature_and_invalid_target_are_rejected_before_fit(self):
        X, y, predict, context, recipe = fixture()
        cases = [(X.assign(technical_id="x"), y, predict), (X, y, predict.assign(target=999)),
                 (X, [1, 2, 3, 4, 5, float("nan")], predict)]
        for fit, target, query in cases:
            with spies() as calls, self.assertRaises(ValueError):
                base.fit_base(fit, target, query, context, recipe)
            self.assertFalse(calls["early"])
            self.assertFalse(calls["refit"])

    def test_non_rmse_and_undeclared_period_rejected(self):
        X, y, predict, context, recipe = fixture()
        for change in ({"period_column": "id"}, {"period_column": "kind"}):
            with spies(), self.assertRaises(ValueError):
                base.fit_base(X, y, predict, context, {**recipe, **change})
        context["metric"] = get_metric("MAE").to_dict()
        with spies(), self.assertRaises(ValueError):
            base.fit_base(X, y, predict, context, recipe)

    def test_invalid_best_iteration_or_predictions_rejected(self):
        X, y, predict, context, recipe = fixture()
        with spies(best=9000), self.assertRaises(base.BoostingSkipError):
            base.fit_base(X, y, predict, context, recipe)
        with spies(bad_predictions=True), self.assertRaises(ValueError):
            base.fit_base(X, y, predict, context, recipe)

    def test_expired_deadline_prevents_all_fitting(self):
        X, y, predict, context, recipe = fixture()
        context["deadline_monotonic"] = time.monotonic() - 1
        with spies() as calls, self.assertRaises(TimeoutError):
            base.fit_base(X, y, predict, context, recipe)
        self.assertFalse(calls["early"])
        self.assertFalse(calls["refit"])

    def test_measured_refit_budget_failure_prevents_full_fit(self):
        X, y, predict, context, recipe = fixture()
        clock = [0.0]
        context["deadline_monotonic"] = 100.0
        def advance():
            clock[0] = 95.0
        with patch.object(base.time, "monotonic", side_effect=lambda: clock[0]), spies(early_fit=advance) as calls:
            with self.assertRaisesRegex(TimeoutError, "measured time"):
                base.fit_base(X, y, predict, context, recipe)
        self.assertEqual(len(calls["early"]), 1)
        self.assertFalse(calls["refit"])


if __name__ == "__main__":
    unittest.main()
