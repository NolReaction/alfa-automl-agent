"""Inner feedback uses pinned outputs and no outer/source data or row details."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.inner_feedback import MAX_FEEDBACK_BYTES, inner_feedback
from agent.metrics import get_metric
from agent.research_candidate import CandidateResult
from agent.task_spec import TaskSpec


def spec(root, metric="RMSE", *, classes=("NA", "001"), probability_columns=None,
         positive_class=None):
    metric_spec = get_metric(metric)
    probability_columns = tuple(probability_columns or ())
    output_columns = (("row_key", *(column for column, _ in probability_columns))
                      if probability_columns else ("row_key", "answer"))
    return TaskSpec(root / "must-not-open" / "task.md", root / "must-not-open" / "source",
                    "row_key", "answer", ("x",), ("x",), (),
                    tuple(classes) if metric_spec.classification else (), output_columns,
                    probability_columns, positive_class, metric_spec)


def pinned(root, frame, *, phase="inner_selection"):
    output = root / "pinned_predictions.csv"
    output.write_bytes(frame.to_csv(index=False).encode("utf-8"))
    content = output.read_bytes()
    receipt = root / "pinned_receipt.json"
    receipt.write_bytes(json.dumps({"version": 3, "phase": phase}).encode("utf-8"))
    return CandidateResult(output, None, {"exit_code": 0}, "a" * 64, "b" * 64,
                           phase, receipt, hashlib.sha256(content).hexdigest(), len(content),
                           hashlib.sha256(receipt.read_bytes()).hexdigest())


class InnerFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_regression_reports_bias_mae_and_tail_without_source_or_row_details(self):
        task = spec(self.root)
        frame = pd.DataFrame({"row_key": [f"private-id-{index}" for index in range(4)],
                              "answer": [1, 4, 2, 8]})
        result = pinned(self.root, frame)
        with patch.object(Path, "read_text", side_effect=AssertionError("No source/task reads")):
            feedback = inner_feedback(task, result, [1, 2, 3, 4])
        self.assertEqual(feedback["scope"], "inner_selection")
        self.assertAlmostEqual(feedback["metric"]["value"], math.sqrt(21 / 4))
        self.assertEqual(feedback["regression"]["mean_bias"], 1.25)
        self.assertEqual(feedback["regression"]["mae"], 1.75)
        self.assertEqual(feedback["regression"]["absolute_error_p50"], 1.5)
        self.assertAlmostEqual(feedback["regression"]["absolute_error_p90"], 3.4)
        self.assertAlmostEqual(feedback["regression"]["absolute_error_p99"], 3.94)
        encoded = json.dumps(feedback)
        self.assertNotIn("private-id", encoded)
        self.assertNotIn("must-not-open", encoded)
        self.assertNotIn("row_key", encoded)
        self.assertNotIn("answer", encoded)

    def test_literal_binary_labels_keep_leading_zeroes_and_na_and_report_confusions(self):
        frame = pd.DataFrame({"row_key": ["0001", "0002", "0003", "0004"],
                              "answer": ["NA", "NA", "001", "001"]})
        feedback = inner_feedback(spec(self.root, "Macro F1"), pinned(self.root, frame),
                                  ["NA", "001", "NA", "001"])
        self.assertEqual(feedback["metric"]["value"], .5)
        classes = {row["label"]: row for row in feedback["classification"]["per_class"]}
        self.assertEqual(set(classes), {"NA", "001"})
        for row in classes.values():
            self.assertEqual(row["support"], 2)
            self.assertEqual(row["recall"], .5)
            self.assertEqual(row["f1"], .5)
        self.assertEqual(feedback["classification"]["worst_confusions"], [
            {"actual_class": "NA", "predicted_class": "001", "count": 1},
            {"actual_class": "001", "predicted_class": "NA", "count": 1},
        ])
        self.assertNotIn("confusion_matrix", feedback["classification"])

    def test_binary_probabilities_use_declared_positive_class_in_first_position(self):
        task = spec(self.root, "ROC AUC", probability_columns=(("score", "NA"),), positive_class="NA")
        frame = pd.DataFrame({"row_key": ["0001", "0002", "0003", "0004"],
                              "score": [.9, .1, .7, .4]})
        feedback = inner_feedback(task, pinned(self.root, frame), ["NA", "001", "NA", "001"])
        self.assertEqual(feedback["metric"]["value"], 1)
        self.assertEqual(feedback["classification"]["accuracy"], 1)
        self.assertEqual(feedback["classification"]["worst_confusions"], [])
        self.assertEqual(feedback["classification"]["label_rule"], "argmax_in_declared_class_order")

    def test_multiclass_probability_columns_are_aligned_by_explicit_class_mapping(self):
        task = spec(self.root, "LogLoss", classes=("NA", "001", "B"),
                    probability_columns=(("p_b", "B"), ("p_001", "001"), ("p_na", "NA")))
        frame = pd.DataFrame({"row_key": ["0001", "0002", "0003"],
                              "p_b": [.1, .2, .8], "p_001": [.1, .7, .1], "p_na": [.8, .1, .1]})
        feedback = inner_feedback(task, pinned(self.root, frame), ["NA", "001", "B"])
        self.assertEqual(feedback["classification"]["accuracy"], 1)
        self.assertTrue(all(row["f1"] == 1 for row in feedback["classification"]["per_class"]))
        self.assertAlmostEqual(feedback["metric"]["value"], -sum(map(math.log, [.8, .7, .8])) / 3)

    def test_outer_or_refit_phase_is_refused_before_reading_any_output(self):
        result = pinned(self.root, pd.DataFrame({"row_key": ["001"], "answer": [1]}))
        for phase in ("outer_validation", "final_check", "full_refit", "selection"):
            with self.subTest(phase=phase), patch.object(CandidateResult, "read_output") as read:
                with self.assertRaisesRegex(ValueError, "inner_selection"):
                    inner_feedback(spec(self.root), replace(result, phase=phase), [1])
                read.assert_not_called()

    def test_changed_output_nonfinite_values_and_misaligned_schema_are_refused(self):
        task = spec(self.root)
        frame = pd.DataFrame({"row_key": ["001", "002"], "answer": [1, 2]})
        result = pinned(self.root, frame)
        result.output_path.write_bytes(result.output_path.read_bytes().replace(b"001,1", b"001,9"))
        with self.assertRaisesRegex(ValueError, "changed"):
            inner_feedback(task, result, [1, 2])
        for invalid, actuals in (
            (frame, [float("inf"), 2]),
            (frame, [1]),
            (frame.assign(answer=[1, float("inf")]), [1, 2]),
            (frame.rename(columns={"answer": "other"}), [1, 2]),
        ):
            with self.subTest(actuals=actuals, columns=list(invalid)):
                with self.assertRaises(ValueError):
                    inner_feedback(task, pinned(self.root, invalid), actuals)

    def test_unicode_labels_stay_exact_while_feedback_obeys_utf8_byte_limit(self):
        classes = tuple(f"{index:03}-" + "😀" * 196 for index in range(50))
        task = spec(self.root, "Macro F1", classes=classes)
        frame = pd.DataFrame({"row_key": [f"private-{index}" for index in range(50)],
                              "answer": [*classes[1:], classes[0]]})
        feedback = inner_feedback(task, pinned(self.root, frame), classes, max_classes=50)
        encoded = json.dumps(feedback, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(encoded), MAX_FEEDBACK_BYTES)
        details = feedback["classification"]
        self.assertGreater(details["omitted_classes"], 0)
        self.assertLessEqual(len(details["worst_confusions"]), 8)
        self.assertEqual(len(details["per_class"]) + details["omitted_classes"], 50)
        self.assertTrue(all(row["label"] in classes for row in details["per_class"]))
        self.assertTrue(all(row["actual_class"] in classes and row["predicted_class"] in classes
                            for row in details["worst_confusions"]))

    def test_invalid_probability_ranges_or_class_budget_do_not_produce_feedback(self):
        task = spec(self.root, "ROC AUC", probability_columns=(("score", "001"),), positive_class="001")
        for probabilities in ([1.1, .2], [-.1, .2], [float("nan"), .2], [float("inf"), .2]):
            with self.subTest(probabilities=probabilities):
                frame = pd.DataFrame({"row_key": ["001", "002"], "score": probabilities})
                with self.assertRaises(ValueError):
                    inner_feedback(task, pinned(self.root, frame), ["NA", "001"])
        result = pinned(self.root, pd.DataFrame({"row_key": ["001", "002"], "score": [.1, .9]}))
        for invalid in (True, 0, 51, 1.5):
            with self.subTest(max_classes=invalid), self.assertRaises(ValueError):
                inner_feedback(task, result, ["NA", "001"], max_classes=invalid)


if __name__ == "__main__":
    unittest.main()
