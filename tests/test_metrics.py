import math
import unittest

from agent.metrics import MetricSpec, get_metric, probability_matrix, score_predictions
from agent.task_contract import parse_task_metric


class MetricRegistryTests(unittest.TestCase):
    def test_regression_scores_and_directions(self):
        cases = {"RMSE": math.sqrt(2.5), "MSE": 2.5, "MAE": 1.5, "R2": -1.5}
        for name, expected in cases.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(score_predictions(name, [1, 3], [2, 5]), expected)
                self.assertEqual(get_metric(name).higher_is_better, name == "R2")

    def test_labels_are_not_inferred_as_continuous_numbers(self):
        actual, predicted = ["0", "0", "1", "1"], ["0", "0", "0", "1"]
        self.assertAlmostEqual(score_predictions("Accuracy", actual, predicted, classes=("0", "1")), .75)
        self.assertAlmostEqual(score_predictions("Macro F1", actual, predicted, classes=("0", "1")), 11 / 15)
        self.assertAlmostEqual(score_predictions("Weighted F1", actual, predicted, classes=("0", "1")), 11 / 15)
        with self.assertRaisesRegex(ValueError, "labels"):
            score_predictions("Accuracy", actual, ["0", "0", "0", "2"], classes=("0", "1"))

    def test_binary_auc_uses_identified_probability_not_thresholded_labels(self):
        self.assertEqual(score_predictions("ROC-AUC", ["0", "1", "0", "1"], [.1, .9, .2, .8],
                                           classes=("0", "1")), 1)
        self.assertEqual(score_predictions("ROC AUC", ["no", "yes"], [[.9, .1], [.1, .9]],
                                           classes=("yes", "no"), prediction_classes=("no", "yes"),
                                           positive_class="yes"), 1)
        with self.assertRaisesRegex(ValueError, "positive_class"):
            score_predictions("ROC AUC", ["no", "yes"], [.1, .9], classes=("no", "yes"))

    def test_logloss_reorders_model_columns_to_literal_class_order(self):
        value = score_predictions("LogLoss", ["A", "B", "C"],
                                  [[.1, .8, .1], [.1, .1, .8], [.8, .1, .1]],
                                  classes=("C", "A", "B"), prediction_classes=("C", "A", "B"))
        self.assertAlmostEqual(value, -math.log(.8))
        self.assertAlmostEqual(score_predictions("log_loss", ["0", "1"], [.2, .9],
                                                 classes=("0", "1")), -(math.log(.8) + math.log(.9)) / 2)

    def test_probability_evidence_rejects_invalid_shapes_ranges_and_sums(self):
        for values in ([[.2, .2]], [[float("nan"), .5]], [[-.1, 1.1]], [[.2, .3, .5]], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                probability_matrix(values, classes=("0", "1"), rows=1)
        with self.assertRaisesRegex(ValueError, "class order"):
            probability_matrix([[.5, .5]], classes=("0", "1"), prediction_classes=("1", "1"))

    def test_regression_nonfinite_and_auc_missing_validation_class_are_errors(self):
        for name in ("RMSE", "MAE", "MSE", "R2"):
            with self.subTest(metric=name), self.assertRaises(ValueError):
                score_predictions(name, [1, 2], [1, float("inf")])
        with self.assertRaisesRegex(ValueError, "both classes"):
            score_predictions("ROC AUC", ["0", "0"], [.1, .2], classes=("0", "1"))
        with self.assertRaisesRegex(ValueError, "empty labels"):
            score_predictions("Accuracy", ["nan", "yes"], [float("nan"), "yes"], classes=("nan", "yes"))

    def test_metric_definitions_and_primary_declarations_are_allowlisted(self):
        for name in ("RMSE", "MAE", "MSE", "R2", "Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss"):
            with self.subTest(metric=name):
                self.assertEqual(parse_task_metric("Primary metric: " + name), get_metric(name))
        for text in ("Metric: F1", "Metric: AUC", "Metric: RMSLE", "Metric: RMSE or MAE",
                     "Metric: MSE\nPrimary metric: MAE", "AUC is useful for classification"):
            with self.subTest(task=text), self.assertRaises(ValueError):
                parse_task_metric(text)
        with self.assertRaisesRegex(ValueError, "registry"):
            get_metric(MetricSpec("RMSE", True, "classification", "labels"))


if __name__ == "__main__":
    unittest.main()
