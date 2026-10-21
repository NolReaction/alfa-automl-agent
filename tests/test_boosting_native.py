"""Tiny CatBoost metric compatibility regressions, not quality benchmarks."""

import importlib.util
import unittest

import numpy as np

from agent.boosting_candidate import _catboost_metric, _make_estimator
from agent.metrics import get_metric, score_predictions


@unittest.skipUnless(importlib.util.find_spec("catboost"), "Optional CatBoost is not installed")
class CatBoostWeightMetricCompatibilityTests(unittest.TestCase):
    def fit_macro_metric(self, weight_options):
        random = np.random.default_rng(732)
        classes = np.asarray(["01", "NA", "class c"])
        X_fit, X_valid = random.normal(size=(48, 3)), random.normal(size=(24, 3))
        y_fit = np.repeat(classes, [30, 12, 6])
        y_valid = np.repeat(classes, [8, 8, 8])
        random.shuffle(y_fit)
        random.shuffle(y_valid)
        parameters = {"iterations": 6, "depth": 2, "learning_rate": .1,
                      "l2_leaf_reg": 3.0, **weight_options}
        metric = get_metric("Macro F1")
        native_metric = _catboost_metric(metric, len(classes), parameters)
        model = _make_estimator("catboost", "classification", parameters, [], native_metric)
        model.fit(X_fit, y_fit, eval_set=(X_valid, y_valid), use_best_model=False)
        predicted = np.asarray(model.predict(X_valid)).reshape(-1).astype(str)
        expected = score_predictions(metric, y_valid, predicted, classes=classes)
        metric_values = model.get_evals_result()["validation"]
        actual = next(values[-1] for name, values in metric_values.items() if name.startswith("TotalF1"))
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(set(map(str, model.classes_)), set(classes))
        self.assertTrue(set(predicted) <= set(classes))
        return native_metric

    def test_unweighted_macro_f1_omits_unsupported_use_weights_parameter(self):
        metric = self.fit_macro_metric({"auto_class_weights": None})
        self.assertEqual(metric, "TotalF1:average=Macro")

    def test_automatic_class_weights_keep_validation_macro_f1_unweighted(self):
        metric = self.fit_macro_metric({"auto_class_weights": "Balanced"})
        self.assertEqual(metric, "TotalF1:average=Macro;use_weights=false")

    def test_explicit_class_weights_keep_validation_macro_f1_unweighted(self):
        # The ordinary recipe currently exposes auto weights only. The native
        # adapter must still distinguish explicitly configured object weights.
        metric = self.fit_macro_metric({"class_weights": [1.0, 2.0, 4.0]})
        self.assertEqual(metric, "TotalF1:average=Macro;use_weights=false")


if __name__ == "__main__":
    unittest.main()
