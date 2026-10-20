"""Decision rules use calibration only, preserve labels and align columns."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from agent.classification_decision import fit_decision_rule, predict_with_rule
import classification_decision_cases as cases
import experiment_classification_decision as experiment


class ClassificationDecisionTests(unittest.TestCase):
    CLASSES = ["01", "NA", "class c"]

    def calibration(self, *, ambiguous=False):
        labels = np.asarray(["01"] * 60 + ["NA"] * 20 + ["class c"] * 20)
        middle = [.49, .46, .05] if ambiguous else [.1, .8, .1]
        probabilities = np.asarray([[.75, .15, .1]] * 60
                                   + [middle] * 20 + [[.3, .1, .6]] * 20)
        return probabilities, labels

    def identity_rule(self):
        probabilities, labels = self.calibration()
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        return rule

    def test_calibration_gain_changes_decisions_without_retraining(self):
        probabilities, labels = self.calibration(ambiguous=True)
        before = probabilities.copy()
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        control = np.asarray(self.CLASSES)[np.argmax(probabilities, axis=1)]
        candidate = predict_with_rule(probabilities, self.CLASSES, rule)
        control_score = f1_score(labels, control, labels=self.CLASSES,
                                 average="macro", zero_division=0)
        candidate_score = f1_score(labels, candidate, labels=self.CLASSES,
                                   average="macro", zero_division=0)
        self.assertGreater(candidate_score, control_score)
        self.assertEqual(candidate_score, 1.)
        self.assertNotEqual(rule["multipliers"], [1., 1., 1.])
        self.assertAlmostEqual(rule["diagnostics"]["control_score"], control_score)
        self.assertAlmostEqual(rule["diagnostics"]["selected_score"], candidate_score)
        np.testing.assert_array_equal(probabilities, before)

    def test_equal_quality_prefers_identity(self):
        rule = self.identity_rule()
        self.assertEqual(rule["diagnostics"]["selected_score"], 1.)
        self.assertIsNone(rule["diagnostics"]["fallback_reason"])

    def test_explicit_column_permutation_preserves_nonidentity_predictions(self):
        probabilities, labels = self.calibration(ambiguous=True)
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        order = [2, 0, 1]
        reordered_classes = [self.CLASSES[index] for index in order]
        expected = predict_with_rule(probabilities, self.CLASSES, rule)
        actual = predict_with_rule(probabilities[:, order], reordered_classes, rule)
        np.testing.assert_array_equal(actual, expected)

    def test_ties_use_stored_order_even_when_columns_are_permuted(self):
        rule = self.identity_rule()
        probabilities = np.asarray([[.5, .5, 0.], [0., .5, .5], [1 / 3] * 3])
        expected = np.asarray(["01", "NA", "01"])
        np.testing.assert_array_equal(
            predict_with_rule(probabilities, self.CLASSES, rule), expected)
        np.testing.assert_array_equal(
            predict_with_rule(probabilities[:, [2, 1, 0]],
                              self.CLASSES[::-1], rule), expected)

    def test_exact_strings_survive_json_roundtrip(self):
        probabilities, labels = self.calibration(ambiguous=True)
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        restored = json.loads(json.dumps(rule, allow_nan=False))
        self.assertEqual(restored["classes"], ["01", "NA", "class c"])
        predictions = predict_with_rule(probabilities, self.CLASSES, restored)
        self.assertEqual(set(predictions.tolist()), set(self.CLASSES))
        np.testing.assert_array_equal(predictions, labels)

    def test_empty_calibration_is_explicit_identity_fallback(self):
        rule = fit_decision_rule(np.empty((0, 3)), [], self.CLASSES)
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        diagnostics = rule["diagnostics"]
        self.assertEqual(diagnostics["fallback_reason"], "empty_calibration")
        self.assertEqual(diagnostics["stop_reason"], "identity_fallback")
        self.assertEqual(diagnostics["candidates_evaluated"], 0)
        self.assertIsNone(diagnostics["control_score"])
        self.assertIsNone(diagnostics["selected_score"])
        self.assertEqual(diagnostics["class_counts"], [0, 0, 0])
        self.assertEqual(predict_with_rule(np.empty((0, 3)), self.CLASSES, rule).shape, (0,))

    def test_absent_class_falls_back_without_dropping_schema_class(self):
        probabilities = np.asarray([[.7, .2, .1]] * 60 + [[.1, .8, .1]] * 60)
        labels = ["01"] * 60 + ["NA"] * 60
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        self.assertEqual(rule["classes"], self.CLASSES)
        self.assertEqual(rule["diagnostics"]["fallback_reason"], "missing_class")
        self.assertEqual(rule["diagnostics"]["class_counts"], [60, 60, 0])
        self.assertEqual(rule["diagnostics"]["candidates_evaluated"], 1)

    def test_too_few_samples_falls_back(self):
        labels = [label for label in self.CLASSES for _ in range(30)]
        probabilities = np.repeat(np.eye(3), 30, axis=0)
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        self.assertEqual(rule["diagnostics"]["fallback_reason"], "insufficient_samples")

    def test_rare_class_support_falls_back(self):
        probabilities = np.asarray([[.7, .2, .1]] * 60 + [[.1, .8, .1]] * 60
                                   + [[.2, .2, .6]] * 9)
        labels = ["01"] * 60 + ["NA"] * 60 + ["class c"] * 9
        rule = fit_decision_rule(probabilities, labels, self.CLASSES)
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        self.assertEqual(rule["diagnostics"]["fallback_reason"],
                         "insufficient_class_support")
        self.assertEqual(rule["diagnostics"]["class_counts"], [60, 60, 9])

    def test_optimization_is_deterministic_bounded_and_never_worse_on_calibration(self):
        rng = np.random.default_rng(31415)
        probabilities = rng.dirichlet([2., 1., .8], size=180)
        labels = np.asarray(self.CLASSES)[rng.permutation(np.repeat(np.arange(3), 60))]
        options = {"max_candidates": 7, "max_rounds": 2}
        first = fit_decision_rule(probabilities, labels, self.CLASSES, options=options)
        second = fit_decision_rule(probabilities, labels, self.CLASSES, options=options)
        self.assertEqual(first, second)
        self.assertLessEqual(first["diagnostics"]["candidates_evaluated"], 7)
        self.assertLessEqual(first["diagnostics"]["rounds_completed"], 2)
        self.assertTrue(all(.5 <= value <= 2. for value in first["multipliers"]))
        self.assertGreaterEqual(first["diagnostics"]["selected_score"],
                                first["diagnostics"]["control_score"] - 1e-12)

    def test_identity_only_budget_does_not_evaluate_candidate(self):
        probabilities, labels = self.calibration(ambiguous=True)
        rule = fit_decision_rule(probabilities, labels, self.CLASSES,
                                 options={"max_candidates": 1})
        self.assertEqual(rule["multipliers"], [1., 1., 1.])
        self.assertEqual(rule["diagnostics"]["candidates_evaluated"], 1)
        self.assertEqual(rule["diagnostics"]["stop_reason"], "candidate_budget")

    def test_invalid_probability_shapes_are_rejected(self):
        for probabilities in ([.5, .5, 0.], [[[.5, .5, 0.]]], [[.5, .5]],
                              [[.25, .25, .25, .25]], [[.2, .3, .5], [.5]]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                fit_decision_rule(probabilities, ["01"], self.CLASSES)

    def test_invalid_probability_values_are_rejected_at_fit_and_prediction(self):
        rule = self.identity_rule()
        cases = ([np.nan, .5, .5], [np.inf, 0., 0.], [-.1, .5, .6],
                 [1.01, -.01, 0.], [.2, .2, .2], [0., 0., 0.],
                 ["0.3", "0.3", "0.4"], [True, False, False],
                 [.3 + 0.j, .3 + 0.j, .4 + 0.j])
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    fit_decision_rule([row], ["01"], self.CLASSES)
                with self.assertRaises(ValueError):
                    predict_with_rule([row], self.CLASSES, rule)

    def test_invalid_class_schemas_are_rejected(self):
        for classes in (["01", "01", "class c"], ["01", "", "class c"],
                        ["01", None, "class c"], ["01", 1, "class c"],
                        [], ["01"], "01", [["01"], ["NA"], ["class c"]]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                fit_decision_rule([[.3, .3, .4]], ["01"], classes)

    def test_invalid_calibration_labels_are_rejected(self):
        for labels in ([], ["unknown"], [1], [None], [["01"]], "01"):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                fit_decision_rule([[.3, .3, .4]], labels, self.CLASSES)

    def test_prediction_requires_exact_same_class_set_and_correct_width(self):
        rule = self.identity_rule()
        for classes in (["01", "NA", "other"], ["01", "NA"],
                        ["01", "NA", "NA"], ["01", "na", "class c"]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                predict_with_rule([[.3, .3, .4]], classes, rule)

    def test_invalid_options_are_rejected_instead_of_silently_clamped(self):
        probabilities, labels = self.calibration()
        cases = ([], {"metric": "accuracy"}, {"max_candidates": 0},
                 {"max_candidates": 4097}, {"max_rounds": 11},
                 {"max_rounds": -1}, {"max_rounds": True},
                 {"min_samples": -1}, {"min_class_count": -1},
                 {"multiplier_min": 0.}, {"multiplier_max": np.inf},
                 {"multiplier_grid": [.5, 1., np.nan]},
                 {"multiplier_grid": [.5, 1., 3.]},
                 {"multiplier_min": 2., "multiplier_max": 3.})
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_decision_rule(probabilities, labels, self.CLASSES, options=options)

    def test_corrupted_saved_rules_are_rejected(self):
        original = self.identity_rule()
        cases = []
        for key, value in (("metric", "log_loss"), ("version", 999),
                           ("multipliers", [1., 1.]),
                           ("multipliers", [1., 0., 1.]),
                           ("multipliers", [1., np.inf, 1.]),
                           ("multipliers", [1., 2.1, 1.]),
                           ("classes", ["01", "NA", "NA"])):
            rule = copy.deepcopy(original)
            rule[key] = value
            cases.append(rule)
        missing = copy.deepcopy(original)
        del missing["classes"]
        cases.append(missing)
        for reason, value in (("stop_reason", []), ("fallback_reason", {})):
            rule = copy.deepcopy(original)
            rule["diagnostics"][reason] = value
            cases.append(rule)
        for index, rule in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                predict_with_rule([[.3, .3, .4]], self.CLASSES, rule)


class ClassificationDecisionProtocolTests(unittest.TestCase):
    """Protocol checks never train a model or inspect external audit scores."""

    def test_all_nine_generators_are_deterministic_and_partitions_are_disjoint(self):
        for name in cases.CASE_DEFINITIONS:
            for seed in cases.SEEDS:
                with self.subTest(name=name, seed=seed):
                    first, second = cases.make_case(name, seed), cases.make_case(name, seed)
                    pd.testing.assert_frame_equal(first["frame"], second["frame"])
                    for key in ("labels", "ids", "fit_rows", "calibration_rows", "audit_rows"):
                        np.testing.assert_array_equal(first[key], second[key])
                    hashes = [hashlib.sha256(case["frame"].to_csv(index=False).encode()).hexdigest()
                              for case in (first, second)]
                    self.assertEqual(hashes[0], hashes[1])
                    partitions = [first[key] for key in
                                  ("fit_rows", "calibration_rows", "audit_rows")]
                    combined = np.concatenate(partitions)
                    np.testing.assert_array_equal(np.sort(combined), np.arange(len(first["frame"])))
                    self.assertEqual(len(np.unique(combined)), len(combined))
                    groups = pd.util.hash_pandas_object(first["frame"], index=False).to_numpy()
                    grouped = [set(groups[part].tolist()) for part in partitions]
                    for left in range(3):
                        for right in range(left + 1, 3):
                            self.assertFalse(grouped[left] & grouped[right])
                    self.assertEqual(first["split"]["raw_feature_group_overlap"], 0)

    def test_duplicate_scenario_preserves_four_copy_groups_across_boundaries(self):
        for seed in cases.SEEDS:
            with self.subTest(seed=seed):
                case = cases.make_case("contradictory_duplicates", seed)
                frame = case["frame"]
                groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
                unique, counts = np.unique(groups, return_counts=True)
                self.assertEqual(len(unique), 300)
                np.testing.assert_array_equal(counts, np.full(300, 4))
                # Contradictory labels belong to a whole group, never to an
                # artificial second copy that can leak across a boundary.
                labels_by_group = pd.DataFrame({"group": groups, "label": case["labels"]})
                self.assertTrue((labels_by_group.groupby("group")["label"].nunique() > 1).any())
                self.assertEqual([len(case[key]) for key in
                                  ("fit_rows", "calibration_rows", "audit_rows")], [720, 240, 240])
                for key in ("fit_rows", "calibration_rows", "audit_rows"):
                    _, part_counts = np.unique(groups[case[key]], return_counts=True)
                    np.testing.assert_array_equal(part_counts, np.full(len(part_counts), 4))

    def test_shift_scenario_moves_forward_and_unseen_category_is_audit_only(self):
        for seed in cases.SEEDS:
            with self.subTest(seed=seed):
                case = cases.make_case("categorical_shift", seed)
                frame = case["frame"]
                fitting, calibration, audit = (case[key] for key in
                                               ("fit_rows", "calibration_rows", "audit_rows"))
                self.assertLess(frame.iloc[fitting]["period"].max(),
                                frame.iloc[calibration]["period"].min())
                self.assertLess(frame.iloc[calibration]["period"].max(),
                                frame.iloc[audit]["period"].min())
                self.assertNotIn("unseen", frame.iloc[fitting]["habitat"].tolist())
                self.assertNotIn("unseen", frame.iloc[calibration]["habitat"].tolist())
                self.assertIn("unseen", frame.iloc[audit]["habitat"].tolist())
                self.assertEqual([len(fitting), len(calibration), len(audit)], [675, 225, 225])

    def test_split_validator_rejects_duplicate_leak_and_backwards_time(self):
        duplicate_frame = pd.DataFrame({"x": [1, 1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "groups cross"):
            cases.validate_splits(duplicate_frame, [0], [1], [2, 3])
        unique_frame = pd.DataFrame({"period": [0, 1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "strictly forward"):
            cases.validate_splits(unique_frame, [2], [1], [0, 3], chronological=True)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            cases.validate_splits(unique_frame, [0, 1], [1], [2, 3])

    def test_mocked_worker_freezes_before_audit_and_audit_labels_cannot_change_rule(self):
        classes = ["01", "NA", "class c"]
        fit, calibration, audit = np.arange(30), np.arange(30, 150), np.arange(150, 180)
        fitting_labels = np.repeat(classes, 10)
        calibration_labels = np.asarray(["01"] * 72 + ["NA"] * 24 + ["class c"] * 24)
        calibration_probabilities = np.asarray([[.75, .15, .1]] * 72
                                               + [[.49, .46, .05]] * 24
                                               + [[.3, .1, .6]] * 24)
        audit_probabilities = np.asarray([[.49, .46, .05]] * len(audit))
        frame = pd.DataFrame({"x": np.arange(180)})
        plan = {"plan_sha256": "fixed-unit-test-plan"}
        captures = []
        original_diagnostics = experiment.diagnostics
        original_fit_rule = experiment.fit_decision_rule
        with tempfile.TemporaryDirectory() as temporary:
            for attempt, audit_label in enumerate(("NA", "01")):
                output = Path(temporary) / f"run-{attempt}"
                output.mkdir()
                directory = output / "fixture-73"
                labels = np.concatenate([fitting_labels, calibration_labels,
                                         np.repeat(audit_label, len(audit))])
                fixture = {
                    "name": "fixture", "seed": 73, "frame": frame, "labels": labels,
                    "ids": np.asarray([f"{index:04d}" for index in range(len(frame))]),
                    "classes": classes, "fit_rows": fit, "calibration_rows": calibration,
                    "audit_rows": audit, "split": {"strategy": "grouped"},
                }
                calls = {"fit": [], "probabilities": [], "rule_labels": [], "score_labels": []}

                class StubPipeline:
                    named_steps = {"model": SimpleNamespace(classes_=np.asarray(classes))}

                    def fit(self, supplied, supplied_labels):
                        calls["fit"].append((supplied.index.tolist(), supplied_labels.copy()))
                        return self

                    def predict_proba(self, supplied):
                        calls["probabilities"].append(supplied.index.tolist())
                        if supplied.index.tolist() == calibration.tolist():
                            self_test.assertFalse((directory / "frozen.json").exists())
                            return calibration_probabilities.copy()
                        self_test.assertEqual(supplied.index.tolist(), audit.tolist())
                        self_test.assertTrue((directory / "frozen.json").is_file())
                        return audit_probabilities.copy()

                def fake_dump(model, path, **unused):
                    Path(path).write_bytes(b"fixed-unit-test-model")

                def checked_fit_rule(probabilities, supplied_labels, supplied_classes, **options):
                    self.assertFalse((directory / "frozen.json").exists())
                    np.testing.assert_array_equal(supplied_labels, calibration_labels)
                    np.testing.assert_array_equal(probabilities, calibration_probabilities)
                    self.assertEqual(supplied_classes, classes)
                    calls["rule_labels"].append(supplied_labels.copy())
                    return original_fit_rule(probabilities, supplied_labels, supplied_classes, **options)

                def checked_diagnostics(actual, predictions, supplied_classes):
                    frozen_path = directory / "frozen.json"
                    self.assertTrue(frozen_path.is_file())
                    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                    self.assertEqual(frozen["plan_sha256"], plan["plan_sha256"])
                    self.assertEqual(frozen["rule_sha256"], experiment.json_hash(frozen["rule"]))
                    self.assertEqual(frozen["model_sha256"],
                                     experiment.file_hash(directory / "frozen_model.joblib"))
                    calls["score_labels"].append(actual.copy())
                    return original_diagnostics(actual, predictions, supplied_classes)

                self_test = self
                with patch.object(experiment, "load_plan", return_value=plan), \
                        patch.object(experiment, "make_case", return_value=fixture), \
                        patch.object(experiment, "make_pipeline", return_value=StubPipeline()), \
                        patch.object(experiment.joblib, "dump", side_effect=fake_dump), \
                        patch.object(experiment, "memory_state", return_value={
                            "measurement_available": False, "memory_limit_enforced": False}), \
                        patch.object(experiment, "fit_decision_rule", side_effect=checked_fit_rule), \
                        patch.object(experiment, "diagnostics", side_effect=checked_diagnostics):
                    result = experiment.run_case(output, {"name": "fixture", "seed": 73})
                self.assertEqual(len(calls["fit"]), 1)
                self.assertEqual(calls["fit"][0][0], fit.tolist())
                np.testing.assert_array_equal(calls["fit"][0][1], fitting_labels)
                self.assertEqual(calls["probabilities"], [calibration.tolist(), audit.tolist()])
                self.assertEqual(len(calls["rule_labels"]), 1)
                self.assertEqual(len(calls["score_labels"]), 2)
                for actual in calls["score_labels"]:
                    np.testing.assert_array_equal(actual, labels[audit])
                frozen = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
                self.assertEqual(frozen["source_sha256"], experiment.file_hash(directory / "source.csv"))
                self.assertEqual(frozen["split_sha256"], experiment.json_hash({
                    "fit": fit.tolist(), "calibration": calibration.tolist(), "audit": audit.tolist()}))
                captures.append(result)
        self.assertEqual(captures[0]["rule"], captures[1]["rule"])
        self.assertEqual(captures[0]["hashes"]["control_predictions"],
                         captures[1]["hashes"]["control_predictions"])
        self.assertEqual(captures[0]["hashes"]["candidate_predictions"],
                         captures[1]["hashes"]["candidate_predictions"])
        self.assertNotEqual(captures[0]["scores"], captures[1]["scores"])


if __name__ == "__main__":
    unittest.main()
