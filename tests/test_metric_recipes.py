"""Opt-in losses and log targets preserve legacy recipes and output scale."""

from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from threadpoolctl import threadpool_limits

from agent.model_config import ACTIONS, make_model_config, model_task_type, validate_model_config
from agent.models import build_pipeline, predict_for_metric
from agent.target_transform import Log1pTargetRegressor


class MalformedPredictionRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, kind="nan"):
        self.kind = kind

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        count = len(X)
        if self.kind == "matrix":
            return np.ones((count, 1))
        if self.kind == "wrong_length":
            return np.ones(count + 1)
        if self.kind == "complex":
            return np.full(count, 1 + 1j)
        return np.full(count, np.nan if self.kind == "nan" else np.inf)


class MetricRecipeTests(unittest.TestCase):
    @staticmethod
    def example():
        values = np.linspace(0, 4, 72)
        return pd.DataFrame({"x": values.astype(str), "category": ["a", "b", "NA"] * 24}), np.expm1(values + 1)

    def test_old_config_defaults_and_roundtrip_remain_exact_without_quality_options(self):
        baseline = make_model_config("baseline", "regression")
        self.assertEqual(baseline, {
            "model": "HistGradientBoostingRegressor",
            "parameters": {"max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                           "learning_rate": .1, "l2_regularization": 0.0},
        })
        for task_type in ("classification", "regression"):
            for action in ACTIONS:
                with self.subTest(task_type=task_type, action=action):
                    config = make_model_config(action, task_type)
                    self.assertEqual(set(config), {"model", "parameters"})
                    self.assertEqual(validate_model_config(json.loads(json.dumps(config))), config)
                    self.assertEqual(make_model_config(action, task_type, objective=None, target_transform=None), config)

    def test_quality_controls_roundtrip_as_explicit_recipe_identity(self):
        for action in ("baseline", "catboost", "lightgbm"):
            with self.subTest(action=action):
                original = make_model_config(action, "regression")
                config = make_model_config(action, "regression", objective="mae", target_transform="log1p")
                self.assertEqual(config["parameters"], original["parameters"])
                self.assertEqual(config["objective"], "mae")
                self.assertEqual(config["target_transform"], "log1p")
                self.assertEqual(validate_model_config(json.loads(json.dumps(config)), action), config)
                self.assertEqual(model_task_type(config), "regression")
                self.assertNotEqual(config, original)
        for action in ACTIONS:
            config = make_model_config(action, "regression", target_transform="log1p")
            self.assertEqual(validate_model_config(config), config)

    def test_unsupported_tasks_objectives_and_transform_values_are_rejected(self):
        for action in ACTIONS:
            for options in ({"objective": "mae"}, {"target_transform": "log1p"}):
                with self.subTest(action=action, options=options), self.assertRaises(ValueError):
                    make_model_config(action, "classification", **options)
        for action in ("linear", "extra_trees"):
            with self.assertRaises(ValueError):
                make_model_config(action, "regression", objective="mae")
        for field in ("objective", "target_transform"):
            for value in (True, False, 1, "default", "none", "RMSE", {}, [], lambda: None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_model_config("baseline", "regression", **{field: value})
            saved = make_model_config("baseline", "regression")
            saved[field] = None
            with self.assertRaises(ValueError):
                validate_model_config(saved)
        for parameter, value in (("loss", "absolute_error"), ("objective", "regression_l1"),
                                 ("target_transform", "log1p"), ("callbacks", [])):
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                make_model_config("baseline", "regression", {parameter: value})

    def test_fixed_blend_keeps_component_options_but_rejects_blend_level_controls(self):
        config = {"model": "FixedBlend", "parameters": {"weight": .5, "components": [
            make_model_config("catboost", "regression", objective="mae", target_transform="log1p"),
            make_model_config("linear", "regression"),
        ]}}
        self.assertEqual(validate_model_config(config), config)
        self.assertEqual(model_task_type(config), "regression")
        for field, value in (("objective", "mae"), ("target_transform", "log1p")):
            with self.assertRaises(ValueError):
                validate_model_config({**config, field: value})
        broken = deepcopy(config)
        broken["parameters"]["components"][0]["target_transform"] = "arbitrary"
        with self.assertRaises(ValueError):
            validate_model_config(broken)

    def test_mae_objectives_fit_real_models_with_the_selected_native_loss(self):
        frame, target = self.example()
        for action in ("baseline", "catboost", "lightgbm"):
            with self.subTest(action=action):
                parameters = ({"max_iter": 50, "min_samples_leaf": 5} if action == "baseline" else
                              {"iterations": 50, "depth": 3} if action == "catboost" else
                              {"n_estimators": 50, "min_child_samples": 5})
                config = make_model_config(action, "regression", parameters, objective="mae")
                pipeline = build_pipeline(action, "regression", ["x"], ["category"], config)
                self.assertEqual(list(pipeline.named_steps), ["features", "prepare", "model"]
                                 if action != "catboost" else ["features", "model"])
                with threadpool_limits(limits=2):
                    pipeline.fit(frame, target)
                    predictions = predict_for_metric(pipeline, frame.iloc[:6], "MAE")
                self.assertEqual(predictions.shape, (6,))
                self.assertTrue(np.isfinite(predictions).all())
                estimator = pipeline.named_steps["model"]
                if action == "baseline":
                    self.assertEqual(estimator.loss, "absolute_error")
                elif action == "catboost":
                    self.assertEqual(estimator.estimator_.get_params()["loss_function"], "MAE")
                else:
                    self.assertEqual(estimator.get_params()["objective"], "regression_l1")

    def test_mae_histogram_model_predicts_median_instead_of_mean_on_constant_features(self):
        frame = pd.DataFrame({"x": np.zeros(100)})
        target = np.asarray([0.] * 80 + [100.] * 20)
        mae = build_pipeline("baseline", "regression", ["x"], [],
                             make_model_config("baseline", "regression", {"max_iter": 50}, objective="mae"))
        original = build_pipeline("baseline", "regression", ["x"], [])
        with threadpool_limits(limits=2):
            mae.fit(frame, target)
            original.fit(frame, target)
        np.testing.assert_allclose(mae.predict(frame.iloc[:3]), np.median(target))
        np.testing.assert_allclose(original.predict(frame.iloc[:3]), np.mean(target))

    def test_log_target_predictions_match_manual_training_and_inverse_in_original_scale(self):
        frame, target = self.example()
        original_target = target.copy()
        parameters = {"alpha": .1}
        transformed = build_pipeline("linear", "regression", ["x"], ["category"],
            make_model_config("linear", "regression", parameters, target_transform="log1p"))
        manual = build_pipeline("linear", "regression", ["x"], ["category"],
                                make_model_config("linear", "regression", parameters))
        with threadpool_limits(limits=2):
            transformed.fit(frame, target)
            manual.fit(frame, np.log1p(target))
        self.assertEqual(list(transformed.named_steps), ["features", "prepare", "model"])
        self.assertIsInstance(transformed.named_steps["model"], Log1pTargetRegressor)
        np.testing.assert_array_equal(target, original_target)
        np.testing.assert_allclose(transformed.predict(frame.iloc[-6:]), np.expm1(manual.predict(frame.iloc[-6:])))
        self.assertGreater(transformed.predict(frame.iloc[[-1]])[0], 50)
        self.assertFalse(hasattr(clone(transformed).named_steps["model"], "regressor_"))

    def test_log_target_works_with_all_regression_families_and_mae_composition(self):
        frame, target = self.example()
        overrides = {"baseline": {"max_iter": 50}, "extra_trees": {"n_estimators": 50},
                     "catboost": {"iterations": 50, "depth": 3},
                     "lightgbm": {"n_estimators": 50, "min_child_samples": 5}, "linear": {}}
        for action in ACTIONS:
            with self.subTest(action=action):
                config = make_model_config(action, "regression", overrides[action],
                    objective="mae" if action in {"baseline", "catboost", "lightgbm"} else None,
                    target_transform="log1p")
                pipeline = build_pipeline(action, "regression", ["x"], ["category"], config)
                with threadpool_limits(limits=2):
                    pipeline.fit(frame, target)
                    values = predict_for_metric(pipeline, frame.iloc[:4], "RMSE")
                self.assertEqual(values.shape, (4,))
                self.assertTrue(np.isfinite(values).all())
                clone(pipeline)

    def test_history_uses_original_targets_before_the_final_log_target_model(self):
        raw = pd.DataFrame({"entity": ["001", "001", "001", "001"],
                            "period": ["1", "2", "3", "4"], "x": ["1", "2", "3", "4"]})
        pipeline = build_pipeline("linear", "regression", ["entity", "period", "x"], [],
            make_model_config("linear", "regression", target_transform="log1p"),
            history_config={"entity_columns": ["entity"], "time_column": "period"})
        with threadpool_limits(limits=2):
            pipeline.fit(raw, [10., 20., 30., 40.])
        self.assertEqual(list(pipeline.named_steps), ["history", "features", "prepare", "model"])
        self.assertEqual(pipeline.named_steps["history"].transform(raw)["history_mean"].iloc[2], 15.)
        self.assertTrue(np.isfinite(pipeline.predict(raw)).all())

    def test_direct_wrapper_rejects_classifiers_and_bad_fit_targets(self):
        frame = np.arange(4).reshape(-1, 1)
        invalid = ([-1., 0., 1., 2.], [0., np.nan, 1., 2.], [0., np.inf, 1., 2.],
                   [0., -np.inf, 1., 2.], [0., "unknown", 1., 2.],
                   [0., 1 + 2j, 1., 2.], [], [[1.], [2.], [3.], [4.]], [10**1000] * 4)
        for values in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "log1p targets"):
                Log1pTargetRegressor(LinearRegression()).fit(frame, values)
        with self.assertRaises(ValueError):
            Log1pTargetRegressor(LinearRegression()).fit(frame, [0., 1.])
        with self.assertRaisesRegex(ValueError, "regression only"):
            Log1pTargetRegressor(LogisticRegression()).fit(frame, [0., 1., 0., 1.])

    def test_wrapper_checks_inverse_overflow_and_malformed_estimator_predictions(self):
        frame = np.arange(4).reshape(-1, 1)
        overflow = Log1pTargetRegressor(DummyRegressor(strategy="constant", constant=1000.)).fit(frame, [0., 1., 2., 3.])
        with self.assertRaisesRegex(ValueError, "inverse overflow"):
            overflow.predict(frame)
        for kind in ("nan", "infinite", "matrix", "wrong_length", "complex"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                Log1pTargetRegressor(MalformedPredictionRegressor(kind)).fit(frame, [0., 1., 2., 3.]).predict(frame)

    def test_wrapper_is_cloneable_accepts_zero_and_refuses_stale_predictions_after_failed_refit(self):
        frame = [[0.], [1.], [2.], [3.]]
        wrapper = Log1pTargetRegressor(LinearRegression()).fit(frame, [0.] * 4)
        np.testing.assert_array_equal(wrapper.predict(frame), [0.] * 4)
        self.assertFalse(hasattr(clone(wrapper), "regressor_"))
        wrapper.fit(frame, np.expm1(np.arange(4)))
        self.assertAlmostEqual(wrapper.predict([[4.]])[0], np.expm1(4))
        with self.assertRaises(ValueError):
            wrapper.fit(frame, [-1., 0., 1., 2.])
        with self.assertRaises(NotFittedError):
            wrapper.predict(frame)

    def test_fit_weights_are_forwarded_but_external_validation_targets_are_rejected(self):
        frame = [[0.], [1.]]
        weighted = Log1pTargetRegressor(DummyRegressor()).fit(frame, [0., 1000.], sample_weight=[1., 0.])
        np.testing.assert_array_equal(weighted.predict(frame), [0., 0.])
        for option in ("eval_set", "validation_data"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "fit-fold targets only"):
                Log1pTargetRegressor(DummyRegressor()).fit(frame, [0., 1.], **{option: [(frame, [2., 3.])]})

    def test_large_finite_nonnegative_targets_remain_representable(self):
        frame = [[0.], [1.]]
        target = np.full(2, np.finfo(float).max)
        model = Log1pTargetRegressor(DummyRegressor()).fit(frame, target)
        predicted = model.predict(frame)
        self.assertTrue(np.isfinite(predicted).all())
        np.testing.assert_allclose(predicted / target, 1., rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
