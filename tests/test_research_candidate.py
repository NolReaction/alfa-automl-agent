"""Behavioral checks for the fixed-recipe subprocess boundary."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.metrics import get_metric, score_predictions
from agent.research_candidate import (
    CandidateExecutionError, freeze_candidate, run_candidate,
)
from agent.task_spec import TaskSpec
from offline_baseline import read_csv


LINEAR = '''def fit_predict(X_fit, y_fit, X_predict, context):
    import numpy as np
    import pandas as pd
    assert context["protocol_version"] == 3
    assert "answer" not in X_fit and "answer" not in X_predict
    assert "task_spec" not in context
    slope, intercept = np.polyfit(pd.to_numeric(X_fit["x"]), y_fit, 1)
    return {"predictions": pd.to_numeric(X_predict["x"]) * slope + intercept}
'''


def make_spec(root, metric="RMSE", classes=("01", "NA"), single=False):
    metric_spec = get_metric(metric)
    if metric_spec.prediction_kind == "probabilities":
        outputs = ("row_key", "answer") if single else ("row_key", *classes)
        probability_columns = (("answer", classes[-1]),) if single else tuple((v, v) for v in classes)
    else:
        outputs, probability_columns = ("row_key", "answer"), ()
    return TaskSpec(root / "original_task.md", root / "original_private_data",
                    "row_key", "answer", ("x", "kind"), ("x",), ("kind",),
                    classes if metric_spec.classification else (), outputs,
                    probability_columns, classes[-1] if metric_spec.classification else None,
                    metric_spec)


def data(classification=False, classes=("01", "NA")):
    fit = pd.DataFrame({"row_key": [f"train-{i:02d}" for i in range(8)],
                        "x": [str(i) for i in range(8)],
                        "kind": ["NA", "nan", "", "a"] * 2,
                        "answer": [classes[i % len(classes)] if classification else str(2 * i + 1)
                                   for i in range(8)]})
    predict = pd.DataFrame({"row_key": ["0005", "NA"], "x": ["9", "10"],
                            "kind": ["nan", ""]})
    return fit, predict


def frozen(root, code=LINEAR, recipe=None):
    source = root / "proposal.py"
    source.write_text(code, encoding="utf-8")
    return freeze_candidate(source, recipe or {"model": "linear"}, root / "frozen")


def fake_success(work):
    context = json.loads((work / "context.json").read_text(encoding="utf-8"))
    output = b"row_key,answer\n0005,19\nNA,21\n"
    (work / "predictions.csv").write_bytes(output)
    (work / "receipt.json").write_text(json.dumps({
        "version": 3, "nonce": context["nonce"], "phase": context["phase"],
        "code_hash": context["code_hash"], "recipe_hash": context["recipe_hash"],
        "output_hash": hashlib.sha256(output).hexdigest(),
    }), encoding="utf-8")
    return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}


class ResearchCandidateTests(unittest.TestCase):
    def run_recipe(self, root, candidate, spec, fit, predict, **kwargs):
        return run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                             output_dir=root / "executions", phase="inner_selection",
                             timeout=20, **kwargs)

    def test_frozen_recipe_is_detached_and_never_imported_in_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "imported.txt"
            code = f'from pathlib import Path\nPath({str(marker)!r}).write_text("imported")\n' + LINEAR
            recipe = {"model": {"depth": 2}}
            candidate = frozen(root, code, recipe)
            recipe["model"]["depth"] = 9
            self.assertFalse(marker.exists())
            self.assertEqual(candidate.recipe["model"]["depth"], 2)
            candidate.recipe["model"]["depth"] = 7
            self.assertEqual(candidate.recipe["model"]["depth"], 2)
            candidate.code_path.write_text(LINEAR + "\n# altered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                candidate.verify()

    def test_all_regression_metrics_use_host_only_targets_and_same_recipe_for_refit(self):
        for metric in ("RMSE", "MAE", "MSE", "R2"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate, spec = frozen(root), make_spec(root, metric)
                fit, predict = data()
                result = self.run_recipe(root, candidate, spec, fit, predict, actuals=[19, 21])
                self.assertEqual(set(result.metric), {"name", "value", "higher_is_better"})
                expected = 1.0 if metric == "R2" else 0.0
                self.assertAlmostEqual(result.metric["value"], expected, places=10)
                self.assertEqual(read_csv(result.output_path)["row_key"].tolist(), ["0005", "NA"])
                changed = self.run_recipe(root, candidate, spec, fit, predict,
                                          actuals=[913847.125, 913848.125])
                self.assertEqual(result.output_path.read_bytes(), changed.output_path.read_bytes())
                self.assertNotEqual(result.metric["value"], changed.metric["value"])
                for path in changed.output_path.parent.rglob("*"):
                    if path.is_file():
                        value = path.read_bytes()
                        self.assertNotIn(b"913847.125", value)
                        self.assertNotIn(b"original_private_data", value)
                        self.assertNotIn(b"original_task.md", value)
                refit = run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                      output_dir=root / "executions", phase="full_refit", timeout=20)
                self.assertIsNone(refit.metric)
                self.assertEqual(result.code_hash, refit.code_hash)
                self.assertEqual(result.recipe_hash, refit.recipe_hash)

    def test_verified_result_rejects_later_output_and_receipt_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit, predict = data()
            result = self.run_recipe(root, frozen(root), make_spec(root), fit, predict,
                                     actuals=[19, 21])
            result.verify()
            destination = root / "published" / "submission.csv"
            result.copy_output(destination)
            self.assertEqual(destination.read_bytes(), result.read_output())
            original = result.output_path.read_bytes()
            result.output_path.write_bytes(original.replace(b"0005", b"0006"))
            with self.assertRaisesRegex(ValueError, "changed"):
                result.verify()
            result.output_path.write_bytes(original)
            result.receipt_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                result.copy_output(destination)
            self.assertEqual(destination.read_bytes(), original)

    def test_literal_class_labels_and_three_label_metrics(self):
        code = '''def fit_predict(X_fit, y_fit, X_predict, context):
    assert set(y_fit) == {"01", "NA"}
    return {"predictions": ["01", "NA"]}
'''
        for metric in ("Macro F1", "Weighted F1", "Accuracy"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fit, predict = data(True)
                result = self.run_recipe(root, frozen(root, code), make_spec(root, metric),
                                         fit, predict, actuals=["01", "NA"])
                self.assertEqual(result.metric["value"], 1.0)
                self.assertEqual(read_csv(result.output_path)["answer"].tolist(), ["01", "NA"])

    def test_binary_and_multiclass_probabilities_align_explicit_reversed_class_order(self):
        for metric, classes, single in (("ROC AUC", ("01", "NA"), True),
                                       ("LogLoss", ("01", "NA"), False),
                                       ("LogLoss", ("01", "NA", "nan"), False)):
            with self.subTest(metric=metric, classes=classes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                probabilities = ([[0.1, 0.9], [0.9, 0.1]] if len(classes) == 2 else
                                 [[0.05, 0.05, 0.9], [0.05, 0.9, 0.05]])
                code = ("def fit_predict(X_fit, y_fit, X_predict, context):\n"
                        f"    return {{'predictions': {probabilities!r}, 'classes': {list(reversed(classes))!r}}}\n")
                fit, predict = data(True, classes)
                spec = make_spec(root, metric, classes, single)
                result = self.run_recipe(root, frozen(root, code), spec, fit, predict,
                                         actuals=[classes[0], classes[1]])
                if metric == "ROC AUC":
                    self.assertEqual(result.metric["value"], 1.0)
                else:
                    self.assertLess(result.metric["value"], 0.2)

    def test_probabilities_require_class_order_and_reject_nonfinite_values(self):
        cases = ["{'predictions': [[0.5, 0.5], [0.5, 0.5]]}",
                 "{'predictions': [[float('nan'), 0.5], [0.5, 0.5]], 'classes': ['01', 'NA']}",
                 "{'predictions': [[0.9, 0.9], [0.5, 0.5]], 'classes': ['01', 'NA']}"]
        for expression in cases:
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fit, predict = data(True)
                code = f"def fit_predict(X_fit, y_fit, X_predict, context):\n    return {expression}\n"
                with self.assertRaises(CandidateExecutionError):
                    self.run_recipe(root, frozen(root, code), make_spec(root, "LogLoss"), fit, predict)

    def test_multiclass_auc_needs_every_frozen_class_in_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = ("01", "NA", "nan")
            fit, predict = data(True, classes)
            predict.loc[len(predict)] = ["0007", "11", "NA"]
            code = '''def fit_predict(X_fit, y_fit, X_predict, context):
    return {"predictions": [[0.05, 0.05, 0.9], [0.05, 0.9, 0.05], [0.9, 0.05, 0.05]],
            "classes": ["nan", "NA", "01"]}
'''
            candidate, spec = frozen(root, code), make_spec(root, "ROC AUC", classes)
            result = self.run_recipe(root, candidate, spec, fit, predict, actuals=classes)
            self.assertEqual(result.metric["value"], 1.0)
            with self.assertRaisesRegex(CandidateExecutionError, "every class"):
                self.run_recipe(root, candidate, spec, fit, predict,
                                 actuals=["01", "NA", "NA"])

    def test_changed_execution_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = LINEAR.replace('    import numpy as np',
                                  '    from pathlib import Path\n    Path("fit.csv").write_text("changed")\n    import numpy as np')
            fit, predict = data()
            with self.assertRaisesRegex(CandidateExecutionError, "execution inputs"):
                self.run_recipe(root, frozen(root, code), make_spec(root), fit, predict)

    def test_wrong_output_order_and_stale_receipt_are_independently_rejected(self):
        for wrong_order in (True, False):
            with self.subTest(wrong_order=wrong_order), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fit, predict = data()
                def fake_driver(code, work, logs, step, **kwargs):
                    context = json.loads((work / "context.json").read_text(encoding="utf-8"))
                    rows = predict.iloc[::-1] if wrong_order else predict
                    frame = pd.DataFrame({"row_key": rows["row_key"].tolist(), "answer": [19, 21]})
                    output = work / "predictions.csv"
                    frame.to_csv(output, index=False)
                    receipt = {"version": 3, "nonce": context["nonce"] if wrong_order else "stale",
                               "phase": context["phase"], "code_hash": context["code_hash"],
                               "recipe_hash": context["recipe_hash"],
                               "output_hash": hashlib.sha256(output.read_bytes()).hexdigest()}
                    (work / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                    return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}
                with patch("agent.research_candidate.run_python", side_effect=fake_driver):
                    with self.assertRaisesRegex(CandidateExecutionError,
                                                "IDs|receipt"):
                        self.run_recipe(root, frozen(root), make_spec(root), fit, predict)

    def test_subprocess_timeout_cannot_produce_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit, predict = data()
            candidate = frozen(root, "import time\ntime.sleep(10)\n" + LINEAR)
            with self.assertRaises(CandidateExecutionError) as raised:
                run_candidate(candidate, spec=make_spec(root), fit_frame=fit, predict_frame=predict,
                              output_dir=root / "executions", phase="outer_validation", timeout=0.2)
            self.assertTrue(raised.exception.execution["timed_out"])

    def test_host_csv_preparation_reduces_child_timeout_and_absolute_deadline_can_only_shorten(self):
        for deadline, expected in ((None, 4), (108, 2), (1000, 4)):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate, spec = frozen(root), make_spec(root)
                fit, predict = data()
                clock, captured = {"now": 100.0}, []
                original_to_csv = pd.DataFrame.to_csv

                def prepare(frame, path, *args, **kwargs):
                    value = original_to_csv(frame, path, *args, **kwargs)
                    if Path(path).name in {"fit.csv", "predict.csv"}:
                        clock["now"] += 3
                    return value

                def execute(code, work, logs, step, **kwargs):
                    captured.append(kwargs["timeout"])
                    clock["now"] += 1
                    return fake_success(work)

                with patch("agent.research_candidate.time.monotonic", side_effect=lambda: clock["now"]), \
                        patch.object(pd.DataFrame, "to_csv", prepare), \
                        patch("agent.research_candidate.run_python", side_effect=execute):
                    result = run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                           output_dir=root / "executions", phase="inner_selection",
                                           timeout=10, deadline=deadline, actuals=[19, 21])
                self.assertEqual(captured, [expected])
                self.assertEqual(result.metric["value"], 0)
                self.assertEqual(result.execution["host_elapsed_seconds"], 7)

    def test_preparation_exhaustion_and_past_deadline_never_launch_child(self):
        for deadline in (None, 99):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate, spec = frozen(root), make_spec(root)
                fit, predict = data()
                clock = {"now": 100.0}
                original_to_csv = pd.DataFrame.to_csv

                def prepare(frame, path, *args, **kwargs):
                    value = original_to_csv(frame, path, *args, **kwargs)
                    clock["now"] += 5
                    return value

                with patch("agent.research_candidate.time.monotonic", side_effect=lambda: clock["now"]), \
                        patch.object(pd.DataFrame, "to_csv", prepare), \
                        patch("agent.research_candidate.run_python") as child:
                    with self.assertRaisesRegex(TimeoutError, "input preparation"):
                        run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                      output_dir=root / "executions", phase="outer_validation",
                                      timeout=10, deadline=deadline)
                    child.assert_not_called()
                self.assertFalse(list((root / "executions").rglob("predictions.csv")))

    def test_required_host_scoring_over_deadline_cannot_return_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, spec = frozen(root), make_spec(root)
            fit, predict = data()
            clock = {"now": 100.0}

            def execute(code, work, logs, step, **kwargs):
                clock["now"] += 1
                return fake_success(work)

            def score(*args, **kwargs):
                value = score_predictions(*args, **kwargs)
                clock["now"] += 10
                return value

            with patch("agent.research_candidate.time.monotonic", side_effect=lambda: clock["now"]), \
                    patch("agent.research_candidate.run_python", side_effect=execute), \
                    patch("agent.research_candidate.score_predictions", side_effect=score):
                with self.assertRaisesRegex(CandidateExecutionError, "output validation") as raised:
                    run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                  output_dir=root / "executions", phase="outer_validation",
                                  timeout=10, actuals=[19, 21])
            self.assertTrue(raised.exception.execution["timed_out"])
            self.assertTrue(raised.exception.execution["host_deadline_exceeded"])
            self.assertEqual(raised.exception.execution["host_elapsed_seconds"], 11)

    def test_absolute_deadline_requires_a_finite_numeric_timestamp_without_bool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, spec = frozen(root), make_spec(root)
            fit, predict = data()
            for deadline in (True, False, float("nan"), float("inf"), "100"):
                with self.subTest(deadline=deadline), patch("agent.research_candidate.run_python") as child:
                    with self.assertRaisesRegex(ValueError, "deadline"):
                        run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                      output_dir=root / "executions", phase="inner_selection",
                                      timeout=10, deadline=deadline)
                    child.assert_not_called()

    def test_invalid_phase_refit_answers_target_in_predict_and_bad_recipe_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, spec = frozen(root), make_spec(root)
            fit, predict = data()
            for phase, actuals in (("unknown", None), ("full_refit", [19, 21])):
                with self.assertRaises(ValueError):
                    run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=predict,
                                  output_dir=root / "executions", phase=phase, timeout=20,
                                  actuals=actuals)
            with self.assertRaisesRegex(ValueError, "target answers"):
                self.run_recipe(root, candidate, spec, fit, predict.assign(answer="hidden"))
            with self.assertRaises(ValueError):
                freeze_candidate(root / "proposal.py", {"bad": float("nan")}, root / "other")


if __name__ == "__main__":
    unittest.main()
