"""Inner-only weight selection and frozen deployment of the same ensemble."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.automl_worker import run_candidate
from agent.metrics import get_metric
from agent.model_config import make_model_config, model_task_type, validate_model_config
from agent.task_spec import load_task
from agent.weighted_ensemble import apply_weighted_ensemble, select_weighted_ensemble


def spec(name, classes=()):
    metric = get_metric(name)
    return SimpleNamespace(metric=metric, classification=metric.classification,
                           prediction_kind=metric.prediction_kind, classes=classes,
                           positive_class="1" if classes == ("0", "1") else None)


class WeightedSelectionTests(unittest.TestCase):
    def test_regression_uses_metric_direction_and_can_improve_both_singles(self):
        actual = np.arange(12, dtype=float)
        candidates = [{"id": "a", "predictions": actual + 1},
                      {"id": "b", "predictions": actual - 2}]
        for metric in ("MAE", "MSE", "RMSE", "R2"):
            with self.subTest(metric=metric):
                result = select_weighted_ensemble(spec(metric), actual, candidates)
                self.assertTrue(result["improved"])
                self.assertEqual(result["selected_ids"], ["a", "b"])
                self.assertEqual(result["weights"], [.7, .3])
                self.assertEqual(result["best_single_id"], "a")
                self.assertEqual(result["evaluated_weights"], 9)
                self.assertEqual(result["fitted_on"], "inner_selection")
                json.dumps(result, allow_nan=False)
                if metric == "R2":
                    self.assertGreater(result["score"], result["best_single_score"])
                else:
                    self.assertLess(result["score"], result["best_single_score"])

    def test_classification_learns_on_aligned_probabilities_and_freezes_for_new_rows(self):
        task = spec("Macro F1", ("a", "b"))
        left = np.array([[.8, .2], [.4, .6], [.45, .55], [.2, .8]])
        right = np.array([[.4, .6], [.1, .9], [.9, .1], [.1, .9]])
        result = select_weighted_ensemble(task, ["a", "b", "a", "b"], [
            {"id": "a", "predictions": left, "classes": ("a", "b")},
            {"id": "b", "predictions": right[:, ::-1], "classes": ("b", "a")},
        ])
        self.assertTrue(result["improved"])
        self.assertEqual(result["score"], 1)
        # Deployment consumes only predictions and frozen weights, never new
        # target labels. Column order can differ from fitting and between models.
        output = apply_weighted_ensemble(task, [left[:2], right[:2, ::-1]], result["weights"],
                                         prediction_classes=[("a", "b"), ("b", "a")])
        expected = left[:2] * result["weights"][0] + right[:2] * result["weights"][1]
        np.testing.assert_allclose(output, expected)
        np.testing.assert_allclose(output.sum(axis=1), 1)

    def test_probabilistic_metrics_choose_finite_weights_without_argmax(self):
        for metric in ("LogLoss", "ROC AUC"):
            task = spec(metric, ("0", "1"))
            result = select_weighted_ensemble(task, ["0", "1", "0", "1"], [
                {"id": "a", "predictions": [[.9, .1], [.2, .8], [.3, .7], [.2, .8]]},
                {"id": "b", "predictions": [[.4, .6], [.2, .8], [.9, .1], [.1, .9]]},
            ])
            self.assertTrue(np.isfinite(result["score"]))
            self.assertAlmostEqual(sum(result["weights"]), 1)

    def test_three_model_grid_is_bounded_and_best_single_wins_ties(self):
        actual = np.arange(8, dtype=float)
        candidates = [{"id": name, "predictions": actual.copy()} for name in ("a", "b", "c")]
        result = select_weighted_ensemble(spec("RMSE"), actual, candidates)
        self.assertEqual(result["evaluated_weights"], 63)
        self.assertEqual(result["selected_ids"], ["a"])
        self.assertEqual(result["weights"], [1.0])
        self.assertFalse(result["improved"])
        self.assertEqual(result["stop_reason"], "grid_exhausted")

    def test_deadline_retains_best_single_without_beginning_grid(self):
        candidates = [{"id": "worse", "predictions": [4., 5.]},
                      {"id": "best", "predictions": [1., 2.]}]
        with patch("agent.weighted_ensemble.time.monotonic", return_value=20):
            result = select_weighted_ensemble(spec("MAE"), [1., 2.], candidates, deadline=20)
        self.assertEqual(result["selected_ids"], ["best"])
        self.assertEqual(result["evaluated_weights"], 0)
        self.assertEqual(result["stop_reason"], "time_limit")

    def test_invalid_ids_probabilities_shapes_targets_or_weights_fail_closed(self):
        task = spec("RMSE")
        valid = [{"id": "a", "predictions": [1., 2.]}, {"id": "b", "predictions": [2., 3.]}]
        for candidates in ([valid[0]], [valid[0], valid[0]],
                           [valid[0], {"id": "b", "predictions": [1.]}],
                           [valid[0], {"id": "b", "predictions": [1., float("nan")]}]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                select_weighted_ensemble(task, [1., 2.], candidates)
        with self.assertRaises(ValueError):
            select_weighted_ensemble(task, [1.], valid)
        for weights in ([.5, .4], [1, 0], [True, False], [.5, float("inf")], [.5]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                apply_weighted_ensemble(task, [[1., 2.], [3., 4.]], weights)
        with self.assertRaises(ValueError):
            select_weighted_ensemble(spec("Accuracy", ("a", "b")), ["a"], [
                {"id": "a", "predictions": [[1.1, -.1]]},
                {"id": "b", "predictions": [[.5, .5]]}])


class FrozenRecipeTests(unittest.TestCase):
    @staticmethod
    def config():
        return {"model": "WeightedBlend", "parameters": {"weights": [.7, .3], "components": [
            make_model_config("linear", "regression", {"alpha": 1}),
            make_model_config("linear", "regression", {"alpha": 2})]}}

    def test_schema_rejects_nested_or_mixed_tasks_and_invalid_weights(self):
        config = self.config()
        self.assertEqual(validate_model_config(config), config)
        self.assertEqual(model_task_type(config), "regression")
        with self.assertRaises(ValueError):
            validate_model_config(config, action="linear")
        for weights in ([.5, .4], [1., 0.], [.5, float("nan")], [True, False], [.2, .3, .5]):
            wrong = deepcopy(config)
            wrong["parameters"]["weights"] = weights
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_model_config(wrong)
        for component in (config, {"model": "FixedBlend", "parameters": {}},
                          make_model_config("linear", "classification")):
            wrong = deepcopy(config)
            wrong["parameters"]["components"][0] = component
            with self.subTest(component=component), self.assertRaises(ValueError):
                validate_model_config(wrong)
        three = deepcopy(config)
        three["parameters"]["components"].append(make_model_config("baseline", "regression"))
        three["parameters"]["weights"] = [.2, .3, .5]
        self.assertEqual(validate_model_config(three), three)

    def test_worker_refits_each_component_once_and_uses_the_frozen_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "inputs"
            data.mkdir()
            (root / "task.md").write_text("Metric: MAE", encoding="utf-8")
            pd.DataFrame({"id": [f"t{i}" for i in range(20)], "x": range(20), "y": range(20)}).to_csv(
                data / "train.csv", index=False)
            pd.DataFrame({"id": ["NA", "001"], "x": [20, 21]}).to_csv(data / "test.csv", index=False)
            pd.DataFrame({"id": ["NA", "001"], "y": [0, 0]}).to_csv(data / "sample_submission.csv", index=False)
            pd.DataFrame({"feature": ["id", "x", "y"], "role": ["id", "feature", "target"],
                          "dtype": ["category", "numeric", "numeric"]}).to_csv(data / "data_dictionary.csv", index=False)
            config = {"candidate_dir": str(root), "phase": "refit", "action": "blend",
                      "task_spec": load_task(root / "task.md").to_dict(), "model_config": self.config(),
                      "fit_rows": list(range(10)), "validation_rows": list(range(10, 20))}
            config["components"] = [{"action": "linear", "model_config": model}
                                    for model in config["model_config"]["parameters"]["components"]]
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            fitted = []

            class Model:
                def __init__(self, value):
                    self.value = value
                def fit(self, x, y):
                    fitted.append((self.value, len(x)))
                    return self
                def predict(self, x):
                    return np.full(len(x), self.value)

            def factory(action, task, numeric, categorical, model_config, features):
                return Model(model_config["parameters"]["alpha"])

            with patch("agent.automl_worker.build_pipeline", side_effect=factory):
                run_candidate(path)
            self.assertEqual(fitted, [(1, 20), (2, 20)])
            output = pd.read_csv(root / "submission.csv", keep_default_na=False)
            self.assertEqual(output["id"].tolist(), ["NA", "001"])
            np.testing.assert_allclose(output["y"], 1.3)
            self.assertFalse((root / "holdout_predictions.csv").exists())
            config["components"][0]["model_config"] = make_model_config("linear", "regression", {"alpha": 3})
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen ensemble recipe"):
                run_candidate(path)


if __name__ == "__main__":
    unittest.main()
