"""Regression cases for literal task declarations and unambiguous semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from agent.task_contract import parse_task_metric
from agent.task_spec import _class_list, _positive_class, load_task
from tests.test_task_spec import make_task


class NightTaskContractTests(unittest.TestCase):
    def test_expanded_secondary_metric_cannot_hide_in_primary_declaration(self):
        for value in (
            "RMSE and mean absolute error",
            "MAE + mean squared error",
            "Accuracy and cross entropy",
            "R2 with logarithmic loss",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Ambiguous"):
                parse_task_metric("Primary metric: " + value)

    def test_same_metric_parenthetical_expansions_remain_unambiguous(self):
        for value, expected in (
            ("root mean squared error (RMSE)", "RMSE"),
            ("RMSE (root mean squared error)", "RMSE"),
            ("mean absolute error (MAE)", "MAE"),
            ("MAE (mean absolute error)", "MAE"),
            ("weighted F1 (F1 weighted)", "Weighted F1"),
        ):
            with self.subTest(value=value):
                self.assertEqual(parse_task_metric("Primary metric: " + value).name, expected)

    def test_literal_class_declarations_preserve_markdown_characters(self):
        expected = ("01", "NA", "код__класса", "class ** c", "class ` d")
        text = 'Class order: ["01", "NA", "код__класса", "class ** c", "class ` d"]'
        self.assertEqual(_class_list(text), expected)
        self.assertEqual(_class_list('Class order: 01,код__класса'), ("01", "код__класса"))
        self.assertEqual(_positive_class('Positive class: "class ` d"'), "class ` d")

    def test_markdown_around_declaration_does_not_change_quoted_literal(self):
        for text in (
            '**Class order:** ["a__b", "NA"]',
            '**Class order**: ["a__b", "NA"]',
            'Class order: **["a__b", "NA"]**',
            '`Class order: ["a__b", "NA"]`',
        ):
            with self.subTest(text=text):
                self.assertEqual(_class_list(text), ("a__b", "NA"))
        self.assertEqual(_positive_class('**Positive class:** "a__b"'), "a__b")

    def test_literal_markdown_label_loads_and_predicts_without_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="LogLoss", labels=("a__b", "NA"),
                             dtype="category", extra='Class order: ["NA", "a__b"]\nPositive class: "a__b"')
            spec = load_task(task)
            self.assertEqual(spec.classes, ("NA", "a__b"))
            self.assertEqual(spec.positive_class, "a__b")
            frame = spec.prediction_frame(["0005", "NA"], [[.1, .9], [.7, .3]],
                                          prediction_classes=("NA", "a__b"))
            self.assertEqual(frame["answer"].tolist(), [.9, .3])

    def test_target_named_probability_column_is_not_a_class_named_column(self):
        for metric in ("ROC AUC", "LogLoss"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                task = make_task(Path(directory), metric=metric, labels=("answer", "yes"),
                                 dtype="category", extra="Positive class: yes")
                spec = load_task(task)
                self.assertEqual(spec.probability_columns, (("answer", "yes"),))
                frame = spec.prediction_frame(["0005", "NA"], [[.8, .2], [.1, .9]],
                                              prediction_classes=("answer", "yes"))
                self.assertEqual(frame["answer"].tolist(), [.2, .9])

    def test_real_class_named_probability_column_still_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="LogLoss", labels=("no", "yes"),
                             dtype="category", outputs=("no",), extra="Positive class: yes")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_task(task)

    def test_multiclass_auc_averaging_cannot_be_taken_from_secondary_sentence(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="ROC AUC. Example only: OVR macro",
                             labels=("A", "B", "C"), dtype="category", outputs=("C", "A", "B"))
            with self.assertRaisesRegex(ValueError, "OVR"):
                load_task(task)

    def test_multiclass_auc_ignores_secondary_sentence_when_primary_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(Path(directory), metric="ROC AUC (OVR, macro). Weighted F1 is diagnostic",
                             labels=("A", "B", "C"), dtype="category", outputs=("C", "A", "B"))
            self.assertEqual(load_task(task).metric.name, "ROC AUC")

    def test_regression_frame_rejects_nonreal_numeric_representations(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_task(make_task(Path(directory), metric="RMSE"))
            for values in (
                np.array([1 + 2j, 3 + 4j]), np.array([True, False]),
                np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
                np.array([1, 2], dtype="timedelta64[D]"),
                [True, 0.5], [1, False], [[True], [0.5]],
            ):
                with self.subTest(values=repr(values)):
                    with self.assertRaises(ValueError):
                        spec.prediction_frame(["0005", "NA"], values)
                    with self.assertRaises(ValueError):
                        spec.holdout_frame(["0005", "NA"], [1., 2.], values)
            self.assertEqual(spec.prediction_frame(["0005", "NA"], ["1.25", "-2"])["answer"].tolist(),
                             ["1.25", "-2"])


if __name__ == "__main__":
    unittest.main()
