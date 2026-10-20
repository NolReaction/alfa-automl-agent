"""The ordinary research data path must not reveal external holdout answers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent.research_validation import ResearchValidation
from agent.research_workspace import (
    DESCRIPTION_LIMIT_BYTES, DESCRIPTION_NAME, DESCRIPTION_PREVIEW_BYTES, ResearchWorkspace,
)
from agent.task_spec import load_task
from offline_baseline import read_csv


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class ResearchWorkspaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = self.root / "source"
        self.data = self.case / "data"
        self.task = self.case / "original.md"
        self.make_case()

    def make_case(self, metric="RMSE", *, grouped=False, chronological=False, multiclass=False):
        classification = metric in {"Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss"}
        labels = ["001", "NA", "blue"] if multiclass else ["001", "NA"]
        self.case.mkdir(exist_ok=True)
        declaration = metric + (" OVR macro" if multiclass and metric == "ROC AUC" else "")
        text = f"Primary metric: {declaration}\nOriginal files: /unrelated/private/train.csv\n"
        if classification:
            text += "Class order: " + json.dumps(labels[::-1]) + "\n"
            text += 'Positive class: "NA"\n'
        self.task.write_text(text, encoding="utf-8")
        features = ["measurement", "Year_Factor"] if chronological else ["measurement"]
        write_csv(self.data / "data_dictionary.csv", [
            ["feature", "role", "dtype"], ["key", "id", "category"],
            *[[name, "feature", "numeric"] for name in features],
            ["answer", "target", "category" if classification else "numeric"],
        ])
        self.rows = []
        for i in range(120):
            # Repeated raw features occur only within a single class, while
            # every class still spans many independent groups.
            measurement = i // 2 if grouped else i
            row = [f"train-{i:04d}", measurement]
            if chronological:
                row += [i // 24 + 1]
            row += [labels[(i // 2 if grouped else i) % len(labels)] if classification else 2 * i + 1]
            self.rows.append(row)
        write_csv(self.data / "train.csv", [["key", *features, "answer"], *self.rows])
        future = [["0007", 7], ["NA", 11]]
        if chronological:
            future = [row + [6] for row in future]
        write_csv(self.data / "test.csv", [["key", *features], *future])
        columns = ["key", *labels[::-1]] if metric in {"ROC AUC", "LogLoss"} and multiclass else ["key", "answer"]
        write_csv(self.data / "sample_submission.csv", [columns, *[[row[0], *([""] * (len(columns) - 1))] for row in future]])

    def workspace(self, suffix="view", **kwargs):
        return ResearchWorkspace(self.task, self.root / suffix, **kwargs)

    def test_v2_fixture_reproduces_visible_outer_answers(self):
        gate = ResearchValidation(self.task)
        ordinary_input = gate.task_dir / gate.manifest["data_directory"] / "train.csv"
        visible = read_csv(ordinary_input).iloc[gate.held_rows]
        self.assertEqual(visible[gate.spec.target_column].tolist(), gate.expected_actuals)
        self.assertEqual(visible[gate.spec.id_column].tolist(), gate.expected_ids)

    def test_allowed_view_has_only_outer_fit_and_sanitized_task(self):
        view = self.workspace()
        visible = read_csv(view.work_data / "train.csv")
        self.assertEqual(visible["key"].tolist(), view.train["key"].iloc[view.outer_fit_rows].tolist())
        self.assertFalse(set(visible["key"]) & set(view.expected_ids))
        self.assertFalse(set(read_csv(view.work_data / "test.csv")["key"]) & set(view.expected_test_ids))
        self.assertNotIn("/unrelated/private", view.task_text)
        self.assertNotIn(str(self.case), json.dumps(view.manifest))
        self.assertEqual(view.manifest["version"], 3)
        self.assertEqual(view.manifest["purpose"], "selection")
        self.assertNotIn("validation_hash", json.dumps(view.manifest))
        self.assertEqual(load_task(view.work_task).classes, view.spec.classes)
        self.assertEqual(set(view.manifest["fit_rows"]) | set(view.manifest["validation_rows"]), set(range(len(visible))))

    def test_source_meaning_and_domain_constraints_are_preserved_without_parser_override(self):
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write("\nPredict yearly building energy consumption in kWh/m2.\n"
                         "Do not use future years. Missing readings mean no observation, not zero.\n"
                         "Feature A/B and heating/area are ratios; see https://example.org/energy/units.\n"
                         "```text\nMetric: Accuracy\nClass order: [\"not-a-class\"]\n```\n")
        view = self.workspace()
        for fragment in ("yearly building energy", "kWh/m2", "Do not use future years",
                         "not zero", "Feature A/B", "heating/area", "https://example.org/energy/units"):
            self.assertIn(fragment, view.task_text)
            self.assertIn(fragment, (view.workspace / DESCRIPTION_NAME).read_text(encoding="utf-8"))
        self.assertEqual(load_task(view.work_task).metric.name, "RMSE")
        self.assertTrue(view.manifest["task_semantics"]["complete"])
        self.assertTrue(view.manifest["task_semantics"]["preview_complete"])
        self.assertFalse(view.manifest["task_semantics"]["truncated"])

    def test_task_paths_are_remapped_and_known_source_roots_do_not_enter_description(self):
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\nTask: `{self.task}`; root: {self.case}; dictionary: {self.data / 'data_dictionary.csv'}\n"
                "Input `C:\\Team Folder\\private\\train.csv`, `/srv/Team Folder/test.csv`, `../../raw/test.csv`, "
                "[template](/srv/private/sample_submission.csv), `\\\\server\\private\\data_dictionary.csv`.\n"
                "All CSV: /unrelated/private/*.csv. Do not open the original competition test.\n")
        view = self.workspace()
        cleaned = (view.workspace / DESCRIPTION_NAME).read_text(encoding="utf-8")
        for forbidden in (str(self.case), "Team Folder", "../../raw", "/srv/private", "\\\\server\\private", "/unrelated/private"):
            self.assertNotIn(forbidden, cleaned)
            self.assertNotIn(forbidden, view.task_text)
        for allowed in ("data/train.csv", "data/test.csv", "data/data_dictionary.csv", "data/sample_submission.csv", "data/*.csv", "task.md"):
            self.assertIn(allowed, cleaned)
        self.assertEqual(read_csv(view.work_data / "test.csv")["key"].tolist(), view.expected_inner_ids)
        self.assertFalse(set(view.expected_test_ids) & set(read_csv(view.work_data / "test.csv")["key"]))

    def test_quoted_instructions_keep_their_constraints_when_embedded_file_paths_are_remapped(self):
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write('\n"Use only buildings from 2024 in data/train.csv"\n'
                         "'Exclude readings after 2024 from /srv/private/train.csv'\n"
                         '`Keep the annual units unchanged for ../../raw/test.csv`\n')
        view = self.workspace()
        for expected in ('"Use only buildings from 2024 in data/train.csv"',
                         "'Exclude readings after 2024 from data/train.csv'",
                         '`Keep the annual units unchanged for data/test.csv`'):
            self.assertIn(expected, view.description_text)
            self.assertIn(expected, view.task_text)

    def test_recognized_api_secrets_are_redacted_from_preview_description_and_hint(self):
        key = "known-development-secret-123456"
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write(f'\nOPENAI_API_KEY="{key}"\nAuthorization: Bearer hidden-header-token\n'
                         'api_key: "unconfigured-source-secret"\n'
                         'OPENAI_TOKEN=\nKeep the target in annual energy units.\n'
                         'A diagnostic example: sk-exampletokenwithatleast12chars\n'
                         '## Дополнительная гипотеза для исследования\n'
                         f'Try linear models; debug credential {key}.\n\nПроверь эту гипотезу, ')
        with patch.dict("os.environ", {"OPENAI_API_KEY": key}):
            view = self.workspace()
        combined = view.task_text + view.description_text + json.dumps(view.manifest)
        for secret in (key, "hidden-header-token", "unconfigured-source-secret", "sk-exampletokenwithatleast12chars"):
            self.assertNotIn(secret, combined)
        self.assertIn("[REDACTED]", combined)
        self.assertIn("Additional user hypothesis", view.task_text)
        self.assertIn("Try linear models", view.task_text)
        self.assertIn("Keep the target in annual energy units.", view.description_text)

    def test_long_description_has_bounded_preview_and_complete_allowed_file(self):
        padding = "Длинное описание предметной области. " * 500
        constraint = "Final constraint: weather inputs must precede the prediction period."
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write("\n" + padding + "\n" + constraint)
        view = self.workspace()
        self.assertNotIn(constraint, view.task_text)
        self.assertIn(constraint, view.description_text)
        self.assertIn(DESCRIPTION_NAME, view.manifest["allowed_inputs"])
        self.assertIn("Preview complete: false", view.task_text)
        self.assertIn("Full description complete: true", view.task_text)
        semantics = view.manifest["task_semantics"]
        self.assertTrue(semantics["complete"])
        self.assertFalse(semantics["preview_complete"])
        self.assertTrue(semantics["adaptive_research_allowed"])
        self.assertGreater(semantics["sanitized_bytes"], DESCRIPTION_PREVIEW_BYTES)
        self.assertLess(len(view.task_text.encode("utf-8")), DESCRIPTION_PREVIEW_BYTES + 4000)
        view.check_inputs()

    def test_oversized_description_reports_explicit_baseline_fallback_without_breaking_task(self):
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write("\n" + "domain text " * (DESCRIPTION_LIMIT_BYTES // 6))
            stream.write("Critical tail: must not be assumed absent.")
        view = self.workspace()
        semantics = view.manifest["task_semantics"]
        self.assertFalse(semantics["complete"])
        self.assertTrue(semantics["truncated"])
        self.assertFalse(semantics["adaptive_research_allowed"])
        self.assertEqual(semantics["fallback_reason"], "task_semantics_too_large")
        self.assertIn("description is incomplete", view.task_text)
        self.assertIn("Use baseline fallback", view.description_text)
        self.assertLess(len(view.description_text.encode("utf-8")), DESCRIPTION_LIMIT_BYTES + 1500)
        self.assertEqual(load_task(view.work_task).metric.name, "RMSE")
        self.assertGreater(len(read_csv(view.work_data / "train.csv")), 0)

    def test_dictionary_descriptions_and_extra_columns_are_preserved(self):
        dictionary = read_csv(self.data / "data_dictionary.csv")
        dictionary["description"] = ["Literal key", "Energy intensity in kWh/m2; missing means unobserved", "Annual target"]
        dictionary["unit"] = ["", "kWh/m2", "kWh"]
        dictionary.to_csv(self.data / "data_dictionary.csv", index=False)
        view = self.workspace()
        self.assertEqual(read_csv(view.work_data / "data_dictionary.csv").to_dict("list"), dictionary.to_dict("list"))

    def test_changed_full_description_is_rejected_and_app_hypothesis_bound_remains(self):
        view = self.workspace()
        with (view.workspace / DESCRIPTION_NAME).open("a", encoding="utf-8") as stream:
            stream.write("\nAltered domain constraint.")
        with self.assertRaisesRegex(ValueError, "Allowed research inputs changed"):
            view.check_inputs()
        with self.task.open("a", encoding="utf-8") as stream:
            stream.write("\n## Дополнительная гипотеза для исследования\n" + "x" * 501)
        with self.assertRaisesRegex(ValueError, "1..500"):
            self.workspace("invalid-hint")

    def test_outer_target_intervention_leaves_every_allowed_byte_identical(self):
        first = self.workspace("first")
        allowed = {name: (first.workspace / name).read_bytes() for name in first.manifest["allowed_inputs"]}
        updated = read_csv(self.data / "train.csv")
        updated.loc[first.outer_holdout_rows, "answer"] = "987654321.125"
        updated.to_csv(self.data / "train.csv", index=False)
        second = self.workspace("second", frozen_split=first.frozen_split)
        self.assertEqual(allowed, {name: (second.workspace / name).read_bytes() for name in second.manifest["allowed_inputs"]})
        self.assertNotEqual(first.expected_actuals, second.expected_actuals)
        self.assertEqual(first.expected_inner_actuals, second.expected_inner_actuals)
        self.assertEqual(first.fit_rows, second.fit_rows)
        with self.assertRaisesRegex(ValueError, "Original task inputs changed"):
            first.check_inputs()

    def test_classification_outer_label_intervention_does_not_change_allowed_view(self):
        self.make_case("Macro F1", grouped=True)
        first = self.workspace("first")
        allowed = {name: (first.workspace / name).read_bytes() for name in first.manifest["allowed_inputs"]}
        updated = read_csv(self.data / "train.csv")
        updated.loc[first.outer_holdout_rows, "answer"] = updated.loc[first.outer_holdout_rows, "answer"].map({"NA": "001", "001": "NA"})
        updated.to_csv(self.data / "train.csv", index=False)
        second = self.workspace("second", frozen_split=first.frozen_split)
        self.assertEqual(allowed, {name: (second.workspace / name).read_bytes() for name in second.manifest["allowed_inputs"]})
        self.assertEqual(first.spec.classes, second.spec.classes)
        self.assertNotEqual(first.expected_actuals, second.expected_actuals)

    def test_time_policy_stays_chronological_at_both_levels(self):
        self.make_case(chronological=True)
        view = self.workspace()
        self.assertEqual(view.choice, {"strategy": "time", "time_column": "Year_Factor"})
        self.assertEqual(set(view.outer_fit_frame["Year_Factor"]), {"1", "2", "3", "4"})
        self.assertEqual(set(view.outer_holdout_frame["Year_Factor"]), {"5"})
        self.assertEqual(set(view.research_train.iloc[view.held_rows]["Year_Factor"]), {"4"})
        self.assertEqual(set(view.research_train.iloc[view.fit_rows]["Year_Factor"]), {"1", "2", "3"})

    def test_grouped_policy_keeps_raw_duplicates_apart_at_both_levels(self):
        self.make_case("Macro F1", grouped=True)
        view = self.workspace()
        self.assertEqual(view.choice, {"strategy": "grouped"})
        self.assertFalse(set(view.outer_fit_frame["measurement"]) & set(view.outer_holdout_frame["measurement"]))
        self.assertFalse(set(view.research_train.iloc[view.fit_rows]["measurement"]) & set(view.research_train.iloc[view.held_rows]["measurement"]))
        self.assertEqual(view.spec.classes, ("NA", "001"))
        self.assertTrue(set(view.research_train["answer"]) == {"NA", "001"})

    def test_predictions_only_inner_score_uses_frozen_actuals_for_all_metrics(self):
        for index, metric in enumerate(("RMSE", "MAE", "MSE", "R2", "Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss")):
            with self.subTest(metric=metric):
                self.make_case(metric)
                view = self.workspace(f"view-{index}")
                actual = np.asarray(view.expected_inner_actuals)
                if view.spec.prediction_kind == "probabilities":
                    prediction = (actual == "NA").astype(float) * 0.8 + 0.1
                    frame = view.spec.prediction_frame(view.expected_inner_ids, prediction)
                else:
                    frame = view.spec.prediction_frame(view.expected_inner_ids, actual)
                path = view.workspace / "inner_predictions.csv"
                frame.to_csv(path, index=False)
                result = view.evaluate_inner(path)
                self.assertEqual(result["name"], metric)
                self.assertEqual(result["purpose"], "selection")
                self.assertEqual(result["split"], "inner")
                if metric in {"RMSE", "MAE", "MSE"}:
                    self.assertEqual(result["value"], 0)
                elif metric == "LogLoss":
                    self.assertGreater(result["value"], 0)
                else:
                    self.assertEqual(result["value"], 1)

    def test_multiclass_probability_schema_and_class_order_are_preserved(self):
        self.make_case("ROC AUC", multiclass=True)
        view = self.workspace()
        self.assertEqual(load_task(view.work_task).classes, view.spec.classes)
        predictions = np.array([[0.8 if label == actual else 0.1 for label in view.spec.classes]
                                for actual in view.expected_inner_actuals])
        frame = view.spec.prediction_frame(view.expected_inner_ids, predictions,
                                           prediction_classes=view.spec.classes)
        path = view.workspace / "inner_predictions.csv"
        frame.to_csv(path, index=False)
        self.assertEqual(view.evaluate_inner(path)["value"], 1)

    def test_changed_inputs_reversed_ids_and_nonfinite_predictions_are_rejected(self):
        view = self.workspace()
        path = view.workspace / "inner_predictions.csv"
        frame = view.spec.prediction_frame(view.expected_inner_ids, view.expected_inner_actuals)
        frame.iloc[::-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "IDs"):
            view.evaluate_inner(path)
        frame.loc[0, "answer"] = "nan"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            view.evaluate_inner(path)
        with (view.work_data / "train.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Allowed research inputs changed"):
            view.check_inputs()

    def test_impossible_inner_split_is_explicit_and_never_uses_outer(self):
        self.make_case(chronological=True)
        updated = read_csv(self.data / "train.csv")
        updated["Year_Factor"] = np.where(np.arange(len(updated)) < 96, "1", "2")
        updated.to_csv(self.data / "train.csv", index=False)
        view = self.workspace()
        self.assertEqual(view.fit_rows, [])
        self.assertEqual(view.held_rows, [])
        self.assertIn("inner_unavailable_reason", view.manifest)
        self.assertEqual(set(read_csv(view.work_data / "train.csv")["Year_Factor"]), {"1"})
        with self.assertRaisesRegex(ValueError, "Inner validation is unavailable"):
            view.evaluate_inner(view.workspace / "missing.csv")

    def test_replayed_split_rejects_overlap_and_crossed_chronology(self):
        first = self.workspace("first")
        bad = json.loads(json.dumps(first.frozen_split))
        bad["fit_rows"][0] = bad["validation_rows"][0]
        with self.assertRaisesRegex(ValueError, "partition|repeated"):
            self.workspace("bad", frozen_split=bad)
        self.make_case(chronological=True)
        timed = self.workspace("time")
        bad = json.loads(json.dumps(timed.frozen_split))
        bad["fit_rows"][0], bad["validation_rows"][0] = bad["validation_rows"][0], bad["fit_rows"][0]
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            self.workspace("bad-time", frozen_split=bad)


if __name__ == "__main__":
    unittest.main()
