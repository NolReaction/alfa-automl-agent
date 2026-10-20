"""Generated experiments need unchanged inputs and independently scored evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from agent.research_validation import ResearchValidation, read_hypothesis


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class ResearchValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "data"
        self.task = self.root / "task.md"
        self.make_case()

    def make_case(self, classification=False):
        self.task.write_text(
            "Primary metric: " + ("Macro F1" if classification else "RMSE")
            + "\nUse data/*.csv and write submission.csv next to this task.\n",
            encoding="utf-8",
        )
        write_csv(self.inputs / "data_dictionary.csv", [
            ["feature", "role", "dtype"],
            ["key", "id", "category"],
            ["measurement", "feature", "numeric"],
            ["target", "target", "category" if classification else "numeric"],
        ])
        write_csv(self.inputs / "train.csv", [
            ["key", "measurement", "target"],
            *[[f"train-{index:03d}", index,
               ("NA" if index % 2 else "classA") if classification else 2 * index + 1]
              for index in range(80)],
        ])
        write_csv(self.inputs / "test.csv", [
            ["key", "measurement"], ["0007", 7], ["NA", 11],
        ])
        write_csv(self.inputs / "sample_submission.csv", [
            ["key", "target"], ["0007", ""], ["NA", ""],
        ])
        write_csv(self.root / "submission.csv", [
            ["key", "target"],
            ["0007", "NA" if classification else 15],
            ["NA", "classA" if classification else 23],
        ])

    def evidence(self, gate, predictions=None):
        predictions = predictions if predictions is not None else gate.expected_actuals
        rows = [["id", "actual", "prediction"], *[
            [row_id, actual, prediction]
            for row_id, actual, prediction in zip(
                gate.expected_ids, gate.expected_actuals, predictions)
        ]]
        path = self.root / "research_holdout_predictions.csv"
        write_csv(path, rows)
        return path, rows

    def test_regression_score_is_recomputed_from_frozen_holdout(self):
        gate = ResearchValidation(self.task)
        path, _ = self.evidence(gate, [float(value) + 3.25 for value in gate.expected_actuals])
        # An invented report cannot override the independently calculated score.
        (self.root / "research_metrics.json").write_text(
            json.dumps({"metric": "RMSE", "value": 0}), encoding="utf-8")
        result = gate.evaluate(path, self.root)
        self.assertEqual(result["name"], "RMSE")
        self.assertEqual(result["value"], 3.25)
        self.assertFalse(result["higher_is_better"])
        self.assertEqual(len(gate.fit_rows) + len(gate.held_rows), 80)
        self.assertFalse(set(gate.fit_rows) & set(gate.held_rows))

    def test_equivalent_parent_path_alias_uses_one_canonical_root_for_manifest_and_scoring(self):
        # This lexical alias reproduces resolved-data/unresolved-parent failures
        # on every platform without requiring Windows symlink privileges.
        alias_component = self.root / "parent-alias"
        alias_component.mkdir()
        alias_task = alias_component / ".." / self.task.name
        self.assertNotEqual(alias_task.parent, self.task.parent.resolve())
        gate = ResearchValidation(alias_task)
        self.assertEqual(gate.task_path, self.task.resolve())
        self.assertEqual(gate.task_path, gate.spec.task_path)
        self.assertEqual(gate.task_dir, gate.data_dir.parent)
        self.assertEqual(gate.manifest["data_directory"], "data")
        manifest = json.loads((self.root / "research_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["data_directory"], "data")
        evidence, _ = self.evidence(gate, [float(value) + 2 for value in gate.expected_actuals])
        self.assertEqual(gate.evaluate(evidence, alias_task.parent)["value"], 2)
        # Canonicalization must preserve the frozen source guard, not bypass it.
        alias_task.write_text("Primary metric: MAE\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input data changed"):
            gate.check_inputs()

    def test_directory_symlink_alias_uses_resolved_task_and_data_paths(self):
        with tempfile.TemporaryDirectory() as alias_parent:
            alias = Path(alias_parent) / "task-alias"
            try:
                alias.symlink_to(self.root, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Directory symlinks unavailable: {error}")
            gate = ResearchValidation(alias / self.task.name)
            self.assertEqual(gate.task_path, self.task.resolve())
            self.assertEqual(gate.task_dir, self.root.resolve())
            self.assertEqual(gate.manifest["data_directory"], "data")
            evidence, _ = self.evidence(gate)
            self.assertEqual(gate.evaluate(evidence, alias)["value"], 0)

    def test_classification_score_preserves_literal_na_class(self):
        self.make_case(classification=True)
        gate = ResearchValidation(self.task)
        self.assertEqual(set(gate.expected_actuals), {"classA", "NA"})
        path, _ = self.evidence(gate, ["NA"] * len(gate.expected_ids))
        result = gate.evaluate(path, self.root)
        self.assertEqual(result["name"], "Macro F1")
        self.assertAlmostEqual(result["value"], 1 / 3)
        self.assertTrue(result["higher_is_better"])
        path, _ = self.evidence(gate)
        self.assertEqual(gate.evaluate(path, self.root)["value"], 1.0)

    def test_mismatched_ids_order_and_actuals_are_rejected(self):
        gate = ResearchValidation(self.task)
        for mutation in ("id", "actual", "order", "missing_row"):
            with self.subTest(mutation=mutation):
                path, rows = self.evidence(gate)
                if mutation == "id":
                    rows[1][0] = "forged-id"
                elif mutation == "actual":
                    rows[1][1] = "999999"
                elif mutation == "order":
                    rows[1], rows[2] = rows[2], rows[1]
                else:
                    rows.pop()
                write_csv(path, rows)
                with self.assertRaises(ValueError):
                    gate.evaluate(path, self.root)

    def test_nonfinite_regression_predictions_and_unknown_classes_are_rejected(self):
        gate = ResearchValidation(self.task)
        for value in ("NaN", "inf", "-inf", ""):
            with self.subTest(prediction=value):
                predictions = list(gate.expected_actuals)
                predictions[0] = value
                path, _ = self.evidence(gate, predictions)
                with self.assertRaises(ValueError):
                    gate.evaluate(path, self.root)
        self.make_case(classification=True)
        gate = ResearchValidation(self.task)
        predictions = list(gate.expected_actuals)
        predictions[0] = "new-unseen-class"
        path, _ = self.evidence(gate, predictions)
        with self.assertRaises(ValueError):
            gate.evaluate(path, self.root)

    def test_changed_inputs_cannot_redefine_validation_or_output_contract(self):
        gate = ResearchValidation(self.task)
        path, _ = self.evidence(gate)
        original = (self.inputs / "train.csv").read_text(encoding="utf-8")
        (self.inputs / "train.csv").write_text(
            original.replace("train-000,0,1", "train-000,999,1"), encoding="utf-8")
        with self.assertRaises(ValueError):
            gate.evaluate(path, self.root)

    def test_correct_holdout_does_not_make_invalid_submission_acceptable(self):
        gate = ResearchValidation(self.task)
        path, _ = self.evidence(gate)
        for rows in (
            [["target", "key"], [15, "0007"], [23, "NA"]],
            [["key", "target"], ["NA", 23], ["0007", 15]],
            [["key", "target"], ["0007", "NaN"], ["NA", 23]],
        ):
            with self.subTest(rows=rows):
                write_csv(self.root / "submission.csv", rows)
                with self.assertRaises(ValueError):
                    gate.evaluate(path, self.root)

    def test_missing_holdout_or_hypothesis_is_not_an_experiment(self):
        gate = ResearchValidation(self.task)
        with self.assertRaises((OSError, ValueError)):
            gate.evaluate(self.root / "absent.csv", self.root)
        with self.assertRaises((OSError, ValueError)):
            read_hypothesis(self.root)
        path = self.root / "research_hypothesis.json"
        for value in ({"hypothesis": ""}, {"hypothesis": "   "},
                      {"hypothesis": True}, {"hypothesis": "x" * 2001}, {}):
            with self.subTest(value=str(value)[:80]):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_hypothesis(self.root)
        path.write_text("not JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_hypothesis(self.root)
        path.write_text(" " * 17000 + '{"hypothesis": "small"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_hypothesis(self.root)
        path.write_text(json.dumps({"hypothesis": "A new preprocessing choice may reduce RMSE."}),
                        encoding="utf-8")
        self.assertEqual(read_hypothesis(self.root), "A new preprocessing choice may reduce RMSE.")

    def test_unsupported_metric_is_not_silently_scored_as_rmse(self):
        self.task.write_text("Primary metric: SMAPE\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ResearchValidation(self.task)

    def test_mae_and_r2_use_their_own_metric_direction(self):
        for metric, direction in (("MAE", False), ("R2", True)):
            with self.subTest(metric=metric):
                self.task.write_text(f"Primary metric: {metric}\n", encoding="utf-8")
                gate = ResearchValidation(self.task)
                path, _ = self.evidence(gate)
                result = gate.evaluate(path, self.root)
                self.assertEqual(result["name"], metric)
                self.assertEqual(result["higher_is_better"], direction)
                self.assertAlmostEqual(result["value"], 1 if metric == "R2" else 0)


if __name__ == "__main__":
    unittest.main()
