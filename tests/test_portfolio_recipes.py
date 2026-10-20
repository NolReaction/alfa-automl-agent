"""Contracts and execution probes for the opt-in experimental recipes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agent.features import normalize_feature_config
from agent.metrics import get_metric, registered_metrics, score_predictions
from agent.model_config import ACTIONS, validate_model_config
from agent.models import build_pipeline, predict_for_metric
from agent.portfolio_recipes import portfolio_candidates
from experiment_portfolio_recipes import run_smoke, score_saved, smoke_data


class RecipeContractTests(unittest.TestCase):
    def test_all_metrics_return_eight_unique_valid_json_recipes(self):
        for metric in registered_metrics():
            with self.subTest(metric=metric.name):
                result = portfolio_candidates({"task_type": metric.task_type, "metric": metric.name},
                                              available_actions=ACTIONS)
                self.assertEqual(len(result), 8)
                self.assertEqual(len({item["id"] for item in result}), 8)
                identities = set()
                for item in result:
                    self.assertEqual(set(item), {"id", "action", "label", "model_config", "feature_config"})
                    self.assertTrue(item["id"].startswith("portfolio_v1_"))
                    self.assertEqual(validate_model_config(item["model_config"], item["action"]), item["model_config"])
                    self.assertEqual(normalize_feature_config(item["feature_config"]), item["feature_config"])
                    identities.add(json.dumps([item["model_config"], item["feature_config"]], sort_keys=True))
                self.assertEqual(len(identities), 8)
                self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_filtering_precedes_limit_and_preserves_stable_order(self):
        profile = {"task_type": "regression", "metric": "RMSE"}
        all_items = portfolio_candidates(profile, available_actions=ACTIONS)
        filtered = [item for item in all_items if item["action"] == "lightgbm"]
        self.assertEqual(portfolio_candidates(profile, available_actions=("lightgbm",), limit=2), filtered[:2])
        self.assertEqual(portfolio_candidates(profile, available_actions=()), [])
        with patch.dict("sys.modules", {"catboost": None, "lightgbm": None}):
            self.assertEqual(len(portfolio_candidates(profile, available_actions=("baseline", "extra_trees", "linear"))), 3)

    def test_nested_results_and_inputs_are_independent(self):
        profile = {"task_type": "classification", "metric": "Macro F1"}
        original = deepcopy(profile)
        expected = portfolio_candidates(profile, available_actions=ACTIONS)
        result = portfolio_candidates(profile, available_actions=ACTIONS)
        result[0]["model_config"]["parameters"]["C"] = 999
        result[0]["feature_config"]["date_columns"].append("injected")
        self.assertEqual(portfolio_candidates(profile, available_actions=ACTIONS), expected)
        self.assertEqual(profile, original)
        self.assertEqual(result[1]["feature_config"]["date_columns"], [])

    def test_invalid_profiles_are_rejected_even_without_available_models(self):
        valid = {"task_type": "regression", "metric": "RMSE"}
        for profile in (None, [], {}, {**valid, "answers": [1]}, {"task_type": True, "metric": "RMSE"},
                        {"task_type": "other", "metric": "RMSE"}, {"task_type": "classification", "metric": "RMSE"},
                        {"task_type": "regression", "metric": "unknown"},
                        {"task_type": "regression", "metric": {"name": "RMSE"}}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=())

    def test_runtime_controls_and_malformed_arguments_are_rejected(self):
        profile = {"task_type": "regression", "metric": "RMSE"}
        for limit in (True, False, 0, 9, 1.0, "1", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=(), limit=limit)
        for actions in (None, [], "linear", ("linear", "linear"), ("unknown",), (True,), ([],)):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=actions)


class RecipeExecutionTests(unittest.TestCase):
    def test_real_recipes_fit_regression_binary_and_multiclass_mixed_inputs(self):
        for task_type, binary in (("regression", False), ("classification", True), ("classification", False)):
            metric = "RMSE" if task_type == "regression" else "LogLoss"
            fit, y_fit, held, actual = smoke_data(task_type, binary)
            before = fit.copy(deep=True)
            # Actual recipes, including the representation variant; small fixed fixture.
            for candidate in portfolio_candidates({"task_type": task_type, "metric": metric}, available_actions=ACTIONS):
                with self.subTest(task_type=task_type, binary=binary, candidate=candidate["id"]):
                    model = build_pipeline(candidate["action"], task_type, ["number", "empty"], ["category"],
                                           candidate["model_config"], candidate["feature_config"])
                    with threadpool_limits(limits=8):
                        model.fit(fit, y_fit)
                        prediction = predict_for_metric(model, held, metric)
                    classes = tuple(map(str, model.classes_)) if task_type == "classification" else ()
                    value = score_predictions(metric, actual, prediction, classes=classes, prediction_classes=classes)
                    self.assertTrue(np.isfinite(value))
                    self.assertEqual(len(prediction), len(held))
                    pd.testing.assert_frame_equal(fit, before)

    def test_linear_preprocessing_learns_category_vocabulary_from_fit_only(self):
        fit, y, held, _ = smoke_data("classification", True)
        candidate = portfolio_candidates({"task_type": "classification", "metric": "LogLoss"},
                                         available_actions=("linear",))[0]
        model = build_pipeline("linear", "classification", ["number", "empty"], ["category"],
                               candidate["model_config"], candidate["feature_config"])
        with threadpool_limits(limits=8):
            model.fit(fit, y)
        categories = model.named_steps["prepare"].named_transformers_["category"].named_steps["encode"].categories_[0]
        self.assertNotIn("holdout_only", categories)
        self.assertEqual(predict_for_metric(model, held, "LogLoss").shape, (32, 2))

    def test_subprocess_smoke_rescores_saved_predictions_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            profile = {"task_type": "classification", "metric": "ROC AUC"}
            report = run_smoke(profile, output, available_actions=("linear",), binary=True,
                               attempt_seconds=30, total_seconds=40)
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(report["results"][0]["status"], "completed", report)
            item = report["results"][0]
            self.assertEqual(score_saved(output / item["id"], "classification", "ROC AUC", True), item["score"])
            with self.assertRaises(FileExistsError):
                run_smoke(profile, output, available_actions=("linear",), binary=True)
            prediction_path = output / item["id"] / "predictions.csv"
            table = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
            table.loc[0, "id"] = "wrong"
            table.to_csv(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "IDs"):
                score_saved(prediction_path.parent, "classification", "ROC AUC", True)

    def test_invalid_smoke_limits_are_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for bad in (True, 0, -1, float("nan"), float("inf"), 10**1000):
                with self.subTest(limit=bad), self.assertRaises(ValueError):
                    run_smoke({"task_type": "regression", "metric": "RMSE"}, output, attempt_seconds=bad)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
