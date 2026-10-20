"""Classification detail must agree with the measured holdout, including rare classes."""

from copy import deepcopy
import json
import math
import unittest

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from agent.diagnostics import (
    MAX_CLASSES, MAX_LABEL_LENGTH, MAX_ROWS, classification_diagnostics,
    validate_classification_diagnostics,
)


class ClassificationDiagnosticsTests(unittest.TestCase):
    def example(self):
        return classification_diagnostics(["NA", "NA", "Attack", "Attack", "Rare"],
                                          ["NA", "Attack", "Attack", "NA", "Attack"])

    def test_confusion_rows_are_actual_columns_predicted_and_labels_keep_literal_na(self):
        result = self.example()
        self.assertEqual(result["labels"], ["Attack", "NA", "Rare"])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [1, 1, 0], [1, 0, 0]])
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["accuracy"], .4)
        self.assertEqual(result["per_class"][2],
                         {"precision": 0., "recall": 0., "f1": 0., "support": 1, "predicted": 0})
        self.assertEqual(result["per_class"][0]["predicted"], 3)

    def test_scores_match_sklearn_for_missing_predictions_and_prediction_only_class(self):
        actual = ["Rare", "Common", "Common", "Common", "AbsentPrediction"]
        predicted = ["Common", "Common", "Rare", "PredictionOnly", "Common"]
        result = classification_diagnostics(actual, predicted)
        expected = precision_recall_fscore_support(actual, predicted, labels=result["labels"], zero_division=0)
        for index, row in enumerate(result["per_class"]):
            self.assertAlmostEqual(row["precision"], expected[0][index])
            self.assertAlmostEqual(row["recall"], expected[1][index])
            self.assertAlmostEqual(row["f1"], expected[2][index])
            self.assertEqual(row["support"], expected[3][index])
        self.assertAlmostEqual(result["macro_f1"], f1_score(actual, predicted, average="macro", zero_division=0))
        self.assertEqual(result["accuracy"], accuracy_score(actual, predicted))
        index = result["labels"].index("PredictionOnly")
        self.assertEqual(result["per_class"][index]["support"], 0)
        self.assertEqual(result["per_class"][index]["predicted"], 1)

    def test_perfect_single_class_is_valid_and_json_contains_no_nan(self):
        result = classification_diagnostics(iter(["NA", "NA"]), iter(["NA", "NA"]))
        self.assertEqual(result["macro_f1"], 1.)
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(validate_classification_diagnostics(json.loads(json.dumps(result, allow_nan=False))), result)

    def test_malformed_input_never_silently_converts_labels(self):
        for actual, predicted in (([], []), (["a"], []), (None, ["a"]), ("a", ["a"]),
                                  ([1], ["a"]), (["a"], [None]), ([" "], ["a"]), (["a\0"], ["a"])):
            with self.subTest(actual=actual, predicted=predicted), self.assertRaises(ValueError):
                classification_diagnostics(actual, predicted)

    def test_large_label_space_or_long_names_omit_optional_detail(self):
        classes = [str(index) for index in range(MAX_CLASSES + 1)]
        self.assertIsNone(classification_diagnostics(classes, classes))
        self.assertIsNone(classification_diagnostics(["x" * (MAX_LABEL_LENGTH + 1)], ["a"]))
        # The exact documented boundaries remain supported.
        self.assertEqual(classification_diagnostics(classes[:-1], classes[:-1])["rows"], MAX_CLASSES)
        self.assertIsNotNone(classification_diagnostics(["x" * MAX_LABEL_LENGTH], ["a"]))

    def test_validator_cross_checks_enclosing_metric_and_holdout_count(self):
        result = self.example()
        metric = {"name": "Macro F1", "value": result["macro_f1"], "higher_is_better": True}
        self.assertEqual(validate_classification_diagnostics(result, metric=metric, validation_rows=5), result)
        for kwargs in ({"validation_rows": 6}, {"validation_rows": True},
                       {"metric": {**metric, "value": 1}},
                       {"metric": {**metric, "name": "Accuracy"}},
                       {"metric": {**metric, "higher_is_better": 1}},
                       {"metric": {**metric, "value": math.nan}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_classification_diagnostics(result, **kwargs)

    def test_forged_summaries_and_count_types_are_rejected(self):
        changes = [
            lambda x: x.update(rows=6), lambda x: x.update(rows=True),
            lambda x: x.update(accuracy=1), lambda x: x.update(macro_f1=1),
            lambda x: x.update(accuracy=math.nan), lambda x: x.update(macro_f1=math.inf),
            lambda x: x.update(accuracy=10 ** 1000),
            lambda x: x["confusion_matrix"][0].__setitem__(0, True),
            lambda x: x["confusion_matrix"][0].__setitem__(0, -1),
            lambda x: x["confusion_matrix"][0].__setitem__(0, MAX_ROWS + 1),
            lambda x: x["per_class"][0].update(support=3),
            lambda x: x["per_class"][0].update(predicted=4),
            lambda x: x["per_class"][0].update(f1=.9),
            lambda x: x["per_class"][0].update(recall=True),
        ]
        for change in changes:
            result = self.example()
            change(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_classification_diagnostics(result)

    def test_malformed_schema_or_label_matrix_shapes_are_rejected(self):
        changes = [
            lambda x: x.update(unknown="hidden"), lambda x: x.pop("accuracy"),
            lambda x: x.update(version=True), lambda x: x.update(kind="regression"),
            lambda x: x["labels"].__setitem__(1, "Attack"),
            lambda x: x["labels"].__setitem__(1, ""),
            lambda x: x["labels"].__setitem__(1, "x" * (MAX_LABEL_LENGTH + 1)),
            lambda x: x["confusion_matrix"].pop(), lambda x: x["confusion_matrix"][0].pop(),
            lambda x: x["per_class"].pop(), lambda x: x["per_class"][0].update(hidden=1),
        ]
        for change in changes:
            result = self.example()
            change(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_classification_diagnostics(result)
        with self.assertRaises(ValueError):
            validate_classification_diagnostics(None)

    def test_unused_phantom_label_cannot_change_macro_average(self):
        result = classification_diagnostics(["a"], ["a"])
        result.update(labels=["a", "never-observed"], confusion_matrix=[[1, 0], [0, 0]], macro_f1=.5)
        result["per_class"].append({"precision": 0., "recall": 0., "f1": 0., "support": 0, "predicted": 0})
        with self.assertRaises(ValueError):
            validate_classification_diagnostics(result)

    def test_validation_returns_detached_canonical_data(self):
        result = self.example()
        original = deepcopy(result)
        canonical = validate_classification_diagnostics(result)
        canonical["labels"][0] = "changed"
        canonical["confusion_matrix"][0][0] = 999
        canonical["per_class"][0]["support"] = 999
        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
