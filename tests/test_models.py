"""Real small portfolio fits, literal labels and fold-local categorical state."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from agent.metrics import get_metric, score_predictions
from agent.model_config import make_model_config, model_options, model_task_type, validate_model_config
from agent.models import build_pipeline, normalize_history_config, predict_for_metric
from offline_baseline import make_pipeline, prepare_features


class ModelPortfolioTests(unittest.TestCase):
    @staticmethod
    def example():
        rows = 72
        x = pd.DataFrame({
            "x": np.arange(rows, dtype=float),
            'category:["raw"]': ["NA", "red", "blue"] * (rows // 3),
            "empty_numeric": np.full(rows, np.nan),
        })
        x.loc[3, "x"] = np.nan
        x.loc[4, 'category:["raw"]'] = ""
        return x

    def test_portfolio_config_roundtrips_and_runtime_settings_cannot_be_injected(self):
        for task_type in ("classification", "regression"):
            options = model_options(task_type)
            self.assertEqual(set(options), {"baseline", "extra_trees", "catboost", "lightgbm", "linear"})
            for action, definition in options.items():
                with self.subTest(task_type=task_type, action=action):
                    self.assertEqual(set(definition["defaults"]), {field["name"] for field in definition["fields"]})
                    config = make_model_config(action, task_type)
                    self.assertEqual(validate_model_config(json.loads(json.dumps(config)), action), config)
                    for runtime in ("n_jobs", "thread_count", "random_seed", "train_dir", "device", "callbacks"):
                        with self.assertRaises(ValueError):
                            make_model_config(action, task_type, {runtime: 1})
        for action, task_type, bad in (
            ("catboost", "classification", {"auto_class_weights": {"NA": 2}}),
            ("catboost", "regression", {"iterations": True}),
            ("catboost", "regression", {"depth": 11}),
            ("lightgbm", "classification", {"max_depth": 0}),
            ("lightgbm", "regression", {"learning_rate": float("nan")}),
            ("linear", "classification", {"C": float("inf")}),
            ("linear", "regression", {"alpha": 0}),
        ):
            with self.subTest(action=action, bad=bad), self.assertRaises(ValueError):
                make_model_config(action, task_type, bad)

    def test_legacy_default_pipeline_and_model_parameters_stay_exact(self):
        for task_type in ("classification", "regression"):
            pipeline = build_pipeline("baseline", task_type, ["x"], ["category"])
            old = make_pipeline(["x"], ["category"], task_type == "classification")
            self.assertEqual(list(pipeline.named_steps), ["features", "prepare", "model"])
            self.assertEqual(pipeline.named_steps["model"].get_params(), old.named_steps["model"].get_params())
            self.assertEqual(repr(pipeline.named_steps["prepare"]), repr(old.named_steps["prepare"]))

    def test_saved_fixed_blends_are_bounded_same_task_and_never_nested_or_manual(self):
        config = {"model": "FixedBlend", "parameters": {
            "weight": 0.5, "components": [make_model_config("linear", "regression"),
                                           make_model_config("lightgbm", "regression")]}}
        self.assertEqual(validate_model_config(json.loads(json.dumps(config))), config)
        self.assertEqual(model_task_type(config), "regression")
        with self.assertRaises(ValueError):
            validate_model_config(config, action="baseline")
        variants = []
        for weight in (True, 0, 1, 0.25, float("nan")):
            wrong = deepcopy(config)
            wrong["parameters"]["weight"] = weight
            variants.append(wrong)
        wrong = deepcopy(config)
        wrong["parameters"]["components"][1] = make_model_config("linear", "classification")
        variants.append(wrong)
        wrong = deepcopy(config)
        wrong["parameters"]["components"][1] = deepcopy(config)
        variants.append(wrong)
        wrong = deepcopy(config)
        wrong["parameters"]["components"].append(make_model_config("catboost", "regression"))
        variants.append(wrong)
        for wrong in variants:
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                validate_model_config(wrong)

    def test_legacy_raw_numeric_missing_coercion_matches_original_prepared_pipeline(self):
        raw = pd.DataFrame({"x": ["1", "", "3", "inf", "5"], "category": ["NA", "", "a", "b", "a"]})
        pipeline = build_pipeline("baseline", "regression", ["x"], ["category"])
        expected = prepare_features(raw, ["x", "category"], ["x"], ["category"])
        old = make_pipeline(["x"], ["category"], False)
        transformed = pipeline[:-1].fit_transform(raw, np.arange(len(raw)))
        np.testing.assert_array_equal(transformed, old.named_steps["prepare"].fit_transform(expected))

    def test_catboost_and_lightgbm_fit_both_tasks_with_missing_and_unseen_categories(self):
        numeric, categorical = ["x", "empty_numeric"], ['category:["raw"]']
        x = self.example()
        original = x.copy(deep=True)
        validation = x.iloc[:6].copy()
        validation.loc[:, categorical[0]] = ["unseen", "NA", "blue", "", "red", "unseen"]
        for action in ("catboost", "lightgbm"):
            for task_type in ("classification", "regression"):
                with self.subTest(action=action, task_type=task_type):
                    overrides = ({"iterations": 50, "depth": 3} if action == "catboost"
                                 else {"n_estimators": 50, "min_child_samples": 5})
                    config = make_model_config(action, task_type, overrides)
                    pipeline = build_pipeline(action, task_type, numeric, categorical, config)
                    y = np.asarray(["NA", "Attack", "Rare"] * 24) if task_type == "classification" else np.arange(len(x)) * 1.7
                    with threadpool_limits(limits=2):
                        pipeline.fit(x, y)
                        metric = get_metric("Macro F1" if task_type == "classification" else "RMSE")
                        values = predict_for_metric(pipeline, validation, SimpleNamespace(metric=metric))
                    self.assertEqual(values.shape, (6,))
                    self.assertTrue(np.isfinite(values.astype(float)).all() if task_type == "regression"
                                    else set(values) <= {"NA", "Attack", "Rare"})
                    if task_type == "classification":
                        proba = predict_for_metric(pipeline, validation, get_metric("LogLoss"))
                        self.assertEqual(proba.shape, (6, 3))
                        self.assertEqual(set(pipeline.classes_), {"NA", "Attack", "Rare"})
                        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
                    fitted = pipeline.named_steps["model"]
                    if action == "catboost":
                        self.assertEqual(fitted.category_encoding_, "native")
                        self.assertEqual(fitted.estimator_.get_params()["thread_count"], 8)
                        self.assertFalse(fitted.estimator_.get_params()["allow_writing_files"])
                    else:
                        self.assertEqual(fitted.get_params()["n_jobs"], 8)
                        self.assertNotIn("unseen", pipeline.named_steps["prepare"].categories_[categorical[0]])
                    fresh = clone(pipeline)
                    self.assertFalse(hasattr(fresh.named_steps["model"], "classes_"))
        pd.testing.assert_frame_equal(x, original)

    def test_catboost_high_cardinality_fallback_is_fit_local_and_cloneable(self):
        x = self.example()
        x.loc[:, 'category:["raw"]'] = [f"category-{index}" for index in range(len(x))]
        config = make_model_config("catboost", "classification", {"iterations": 50, "depth": 3})
        pipeline = build_pipeline("catboost", "classification", ["x", "empty_numeric"], ['category:["raw"]'], config)
        with patch("agent.models.MAX_NATIVE_CATEGORIES", 2), threadpool_limits(limits=2):
            pipeline.fit(x, ["NA", "Attack", "Rare"] * 24)
        model = pipeline.named_steps["model"]
        self.assertEqual(model.category_encoding_, "ordinal")
        original_categories = deepcopy(model.category_encoder_.categories_)
        validation = x.iloc[:3].copy()
        validation.loc[:, 'category:["raw"]'] = "never-seen"
        self.assertTrue(np.isfinite(predict_for_metric(pipeline, validation, "LogLoss")).all())
        for before, after in zip(original_categories, model.category_encoder_.categories_):
            np.testing.assert_array_equal(before, after)
        clone(pipeline)

    def test_linear_models_use_sparse_bounded_onehot_and_fit_only_vocabulary(self):
        x = self.example()
        for task_type in ("classification", "regression"):
            with self.subTest(task_type=task_type):
                pipeline = build_pipeline("linear", task_type, ["x", "empty_numeric"], ['category:["raw"]'])
                y = ["NA", "Attack", "Rare"] * 24 if task_type == "classification" else np.arange(len(x)) * 2
                with threadpool_limits(limits=2):
                    pipeline.fit(x, y)
                features = pipeline.named_steps["features"].transform(x)
                encoded = pipeline.named_steps["prepare"].transform(features)
                self.assertTrue(issparse(encoded))
                unknown = x.iloc[:4].copy()
                unknown.loc[:, 'category:["raw"]'] = "future-category"
                output = predict_for_metric(pipeline, unknown, "Macro F1" if task_type == "classification" else "RMSE")
                self.assertEqual(output.shape, (4,))
                encoder = pipeline.named_steps["prepare"].named_transformers_["category"].named_steps["encode"]
                self.assertNotIn("future-category", encoder.categories_[0])
                self.assertEqual(encoder.max_categories, 256)

    def test_binary_probabilities_keep_native_class_order_without_positive_guess(self):
        x = self.example()
        pipeline = build_pipeline("linear", "classification", ["x", "empty_numeric"], ['category:["raw"]'])
        with threadpool_limits(limits=2):
            pipeline.fit(x, ["no", "yes"] * 36)
        proba = predict_for_metric(pipeline, x.iloc[:8], "ROC AUC")
        self.assertEqual(proba.shape, (8, 2))
        value = score_predictions("ROC AUC", ["no", "yes"] * 4, proba,
                                  classes=("yes", "no"), prediction_classes=pipeline.classes_, positive_class="yes")
        self.assertTrue(0 <= value <= 1)

    def test_feature_options_propagate_through_new_and_legacy_factories(self):
        x = pd.DataFrame({"x": [1., np.nan, 3., 4., 5., 6.],
                          "date": ["2024-01-01", "invalid", "2024-02-03", "2025-01-01", "2025-02-01", "2025-03-01"]})
        options = {"date_columns": ["date"], "missing_indicators": True, "signed_log": True}
        for action in ("baseline", "extra_trees", "linear"):
            pipeline = build_pipeline(action, "regression", ["x"], ["date"], feature_config=options)
            transformed = pipeline.named_steps["features"].fit_transform(x, np.arange(len(x)))
            self.assertIn("date__year", transformed)
            self.assertIn("x__missing", transformed)
            self.assertNotIn("date", transformed)
            clone(pipeline)

    def test_explicit_history_precedes_numeric_cleaning_and_uses_only_past_targets(self):
        raw = pd.DataFrame({"entity": ["001", "001", "002", "001", "002", "001"],
                            "period": ["1", "2", "2", "3", "3", "4"],
                            "x": ["1", "2", "3", "4", "5", "6"]})
        target = np.asarray([10., 20., 200., 30., 300., 40.])
        options = {"entity_columns": ["entity"], "time_column": "period"}
        pipeline = build_pipeline("linear", "regression", ["entity", "period", "x"], [],
                                  history_config=options)
        self.assertEqual(list(pipeline.named_steps)[:2], ["history", "features"])
        with threadpool_limits(limits=2):
            pipeline.fit(raw, target)
        state = pipeline.named_steps["history"]
        self.assertIn(("001",), state.entity_history_)
        self.assertNotIn(("1",), state.entity_history_)
        actual = state.transform(raw)
        self.assertEqual(actual["history_count"].tolist(), [0, 1, 0, 2, 1, 3])
        self.assertEqual(actual["history_mean"].iloc[3], 15.)
        # The transformer has never seen these future targets: a query uses
        # only fitted earlier rows and preserves the literal entity key.
        future = raw.iloc[[0]].copy()
        future.loc[:, "period"] = "5"
        self.assertEqual(state.transform(future)["history_mean"].iloc[0], 25.)
        clone(pipeline)

    def test_history_configuration_is_explicit_bounded_and_regression_only(self):
        config = {"entity_columns": ["entity"], "time_column": "period"}
        self.assertEqual(normalize_history_config(config)["time_kind"], "numeric")
        for value in (False, {}, {**config, "entity_columns": []},
                      {**config, "time_kind": []}, {**config, "prefix": "../../target"},
                      {**config, "path": "anything"}, {**config, "time_column": "entity"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_history_config(value)
        with self.assertRaises(ValueError):
            build_pipeline("linear", "classification", ["period"], ["entity"], history_config=config)
        with self.assertRaises(ValueError):
            build_pipeline("linear", "regression", ["period"], ["entity"],
                           history_config={**config, "entity_columns": ["target"]})


if __name__ == "__main__":
    unittest.main()
