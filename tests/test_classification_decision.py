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
import agent.classification_decision as decision
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

    def test_date_and_duration_arrays_are_not_probabilities(self):
        rule = self.identity_rule()
        # These values cast to the plausible probability row [0, 1, 0]. Their
        # dtype still represents dates/durations, not real numeric model output.
        for dtype in ("datetime64[D]", "datetime64[ns]", "timedelta64[D]", "timedelta64[ns]"):
            probabilities = np.asarray([[0, 1, 0]], dtype=dtype)
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    fit_decision_rule(probabilities, ["NA"], self.CLASSES)
                with self.assertRaises(ValueError):
                    predict_with_rule(probabilities, self.CLASSES, rule)
                with self.assertRaises(ValueError):
                    decision.fit_guarded_decision_rule(
                        probabilities, ["NA"], self.CLASSES,
                        tuning_rows=[0], selection_rows=[],
                    )

    def test_numeric_object_and_integer_probabilities_retain_their_contract(self):
        rule = self.identity_rule()
        expected = np.asarray(self.CLASSES)
        for dtype in (np.int64, np.uint8, object):
            probabilities = np.eye(3, dtype=dtype)
            with self.subTest(dtype=dtype):
                np.testing.assert_array_equal(
                    predict_with_rule(probabilities, self.CLASSES, rule), expected)
                fitted = fit_decision_rule(probabilities, self.CLASSES, self.CLASSES)
                self.assertEqual(fitted["diagnostics"]["control_score"], 1.0)
                self.assertEqual(fitted["multipliers"], [1.0, 1.0, 1.0])

    def test_per_class_counts_match_sklearn_with_missing_and_empty_classes(self):
        rng = np.random.default_rng(441)
        fixtures = [(np.array([], dtype=int), np.array([], dtype=int), 3),
                    (np.asarray([0, 0, 1, 1]), np.asarray([1, 2, 1, 2]), 4)]
        fixtures.extend((rng.integers(0, count - 1, 71), rng.integers(0, count, 71), count)
                        for count in (2, 3, 7))
        for actual, predicted, count in fixtures:
            with self.subTest(classes=count, rows=len(actual)):
                expected = f1_score(actual, predicted, labels=np.arange(count),
                                    average=None, zero_division=0)
                np.testing.assert_allclose(decision._class_f1(actual, predicted, count),
                                           expected, rtol=0, atol=1e-15)
                self.assertAlmostEqual(decision._macro_f1(actual, predicted, count),
                                       float(expected.mean()), places=15)

    def test_many_declared_classes_use_linear_counters_and_fall_back(self):
        count = 20_000
        classes = [f"class {index}" for index in range(count)]
        probabilities = np.zeros((2, count))
        probabilities[0, 0] = probabilities[1, 1] = 1.0
        labels = classes[:2]
        original_bincount = np.bincount

        def bounded_bincount(values, weights=None, minlength=0):
            # Refuse a quadratic allocation before NumPy can attempt it. A
            # 20k-class confusion matrix alone would request over 3 GiB.
            self.assertLessEqual(minlength, count)
            return original_bincount(values, weights=weights, minlength=minlength)

        with patch.object(decision.np, "bincount", side_effect=bounded_bincount):
            fitted = fit_decision_rule(probabilities, labels, classes)
            guarded = decision.fit_guarded_decision_rule(
                probabilities, labels, classes, tuning_rows=[0], selection_rows=[1])
        self.assertEqual(fitted["diagnostics"]["fallback_reason"], "missing_class")
        self.assertEqual(fitted["diagnostics"]["control_score"], 2.0 / count)
        self.assertEqual(fitted["multipliers"], [1.0] * count)
        self.assertFalse(guarded["diagnostics"]["accepted"])
        self.assertEqual(guarded["diagnostics"]["reason"], "tuning_missing_class")
        self.assertEqual(guarded["rule"]["multipliers"], [1.0] * count)

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


class GuardedClassificationDecisionTests(unittest.TestCase):
    CLASSES = ["01", "NA", "class c"]

    def fixture(self, *, adverse_selection=False):
        tuning_p = np.asarray([[.75, .15, .1]] * 108
                              + [[.49, .46, .05]] * 36 + [[.3, .1, .6]] * 36)
        tuning_y = np.asarray(["01"] * 108 + ["NA"] * 36 + ["class c"] * 36)
        selection_p = np.asarray([[.75, .15, .1]] * 60
                                 + [[.49, .46, .05]] * 20
                                 + [[.1, .8, .1]] * 20 + [[.3, .1, .6]] * 20)
        selection_y = (["01"] * 80 + ["NA"] * 20 + ["class c"] * 20
                       if adverse_selection else ["01"] * 60 + ["NA"] * 40 + ["class c"] * 20)
        return (np.concatenate((tuning_p, selection_p)),
                np.concatenate((tuning_y, selection_y)), np.arange(180), np.arange(180, 300))

    def fit_fixture(self, *, adverse_selection=False, options=None):
        probabilities, labels, tuning, selection = self.fixture(adverse_selection=adverse_selection)
        return decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning,
            selection_rows=selection, options=options)

    def test_independent_positive_selection_admits_fixed_candidate(self):
        probabilities, labels, tuning, selection = self.fixture()
        result = self.fit_fixture()
        self.assertTrue(result["diagnostics"]["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "accepted")
        self.assertNotEqual(result["rule"]["multipliers"], [1., 1., 1.])
        self.assertEqual(result["rule"], result["candidate_rule"])
        predicted = predict_with_rule(probabilities[selection], self.CLASSES, result["rule"])
        np.testing.assert_array_equal(predicted, labels[selection])
        self.assertGreaterEqual(result["diagnostics"]["selection_macro_f1_gain"], .01)
        self.assertGreaterEqual(min(result["diagnostics"]["selection_class_f1_changes"]), -.02)
        self.assertTrue(set(result["candidate_rule"]["multipliers"]) <= {.8, 1., 1.25})

    def test_adverse_selection_returns_exact_identity(self):
        probabilities, labels, tuning, selection = self.fixture(adverse_selection=True)
        result = self.fit_fixture(adverse_selection=True)
        self.assertFalse(result["diagnostics"]["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "selection_gain_below_minimum")
        self.assertEqual(result["rule"]["multipliers"], [1., 1., 1.])
        self.assertNotEqual(result["candidate_rule"]["multipliers"], [1., 1., 1.])
        np.testing.assert_array_equal(
            predict_with_rule(probabilities, self.CLASSES, result["rule"]),
            np.asarray(self.CLASSES)[probabilities.argmax(axis=1)])

    def test_selection_answers_only_accept_or_reject_never_reoptimize_candidate(self):
        accepted, rejected = self.fit_fixture(), self.fit_fixture(adverse_selection=True)
        self.assertEqual(accepted["candidate_rule"], rejected["candidate_rule"])
        self.assertEqual(accepted["diagnostics"]["tuning_candidate_objective"],
                         rejected["diagnostics"]["tuning_candidate_objective"])
        self.assertTrue(accepted["diagnostics"]["accepted"])
        self.assertFalse(rejected["diagnostics"]["accepted"])
        # Even impossible selection support must not alter the tuned candidate.
        probabilities, labels, tuning, selection = self.fixture()
        labels[selection] = "01"
        missing = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning, selection_rows=selection)
        self.assertEqual(missing["candidate_rule"], accepted["candidate_rule"])
        self.assertEqual(missing["diagnostics"]["reason"], "selection_missing_class")

    def test_per_class_guard_can_reject_despite_positive_macro_gain(self):
        probabilities, labels, tuning, selection = self.fixture()
        probabilities[selection] = np.asarray([[.75, .15, .1]] * 60
                                              + [[.49, .46, .05]] * 30
                                              + [[.1, .8, .1]] * 10 + [[.3, .1, .6]] * 20)
        labels[selection] = ["01"] * 77 + ["NA"] * 23 + ["class c"] * 20
        result = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning, selection_rows=selection)
        self.assertGreater(result["diagnostics"]["selection_macro_f1_gain"], .01)
        self.assertLess(min(result["diagnostics"]["selection_class_f1_changes"]), -.02)
        self.assertFalse(result["diagnostics"]["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "selection_class_drop")
        self.assertEqual(result["rule"]["multipliers"], [1., 1., 1.])

    def test_regularization_discards_a_tiny_tuning_gain(self):
        probabilities, labels, tuning, selection = self.fixture()
        probabilities[tuning] = np.asarray([[.75, .15, .1]] * 60
                                           + [[.49, .46, .05]] * 5
                                           + [[.1, .8, .1]] * 55 + [[.3, .1, .6]] * 60)
        labels[tuning] = ["01"] * 60 + ["NA"] * 60 + ["class c"] * 60
        weak = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning,
            selection_rows=selection, options={"regularization": 0.})
        strong = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning,
            selection_rows=selection, options={"regularization": 1.})
        self.assertNotEqual(weak["candidate_rule"]["multipliers"], [1., 1., 1.])
        self.assertEqual(strong["candidate_rule"]["multipliers"], [1., 1., 1.])
        self.assertEqual(strong["diagnostics"]["reason"], "identity_on_tuning")
        self.assertEqual(strong["diagnostics"]["regularization_cost"], 0.)

    def test_guarded_json_preserves_exact_labels_and_prediction_alignment(self):
        result = self.fit_fixture()
        restored = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(restored["classes"], self.CLASSES)
        self.assertEqual(restored["kind"], "guarded_macro_f1")
        probabilities, labels, _, _ = self.fixture()
        order = [2, 0, 1]
        np.testing.assert_array_equal(
            predict_with_rule(probabilities[:, order], [self.CLASSES[i] for i in order], restored["rule"]),
            predict_with_rule(probabilities, self.CLASSES, restored["rule"]))
        self.assertEqual(restored["partitions"], {"tuning_rows": list(range(180)),
                                                  "selection_rows": list(range(180, 300))})
        # Nested rules deliberately retain the old v1 API, including tie order.
        tied = np.asarray([[.5, .5, 0.]])
        rejected = self.fit_fixture(adverse_selection=True)
        self.assertEqual(predict_with_rule(tied[:, order], [self.CLASSES[i] for i in order],
                                          rejected["rule"]).tolist(), ["01"])

    def test_support_guards_cover_tuning_and_selection_separately(self):
        base_p, base_y, tuning, selection = self.fixture()
        for partition, reason, replacement_count in (
                (tuning, "tuning_missing_class", 36),
                (tuning, "tuning_insufficient_class_support", 32),
                (selection, "selection_missing_class", 20),
                (selection, "selection_insufficient_class_support", 16)):
            with self.subTest(reason=reason):
                labels = base_y.copy()
                class_rows = partition[labels[partition] == "class c"]
                labels[class_rows[:replacement_count]] = "NA"
                result = decision.fit_guarded_decision_rule(
                    base_p, labels, self.CLASSES, tuning_rows=tuning, selection_rows=selection)
                self.assertFalse(result["diagnostics"]["accepted"])
                self.assertEqual(result["diagnostics"]["reason"], reason)
                self.assertEqual(result["rule"]["multipliers"], [1., 1., 1.])

    def test_sample_count_guards_and_unavailable_partitions_are_identity(self):
        probabilities = np.tile([[.75, .15, .1], [.49, .46, .05], [.3, .1, .6]], (60, 1))
        labels = np.tile(self.CLASSES, 60)
        for tuning, selection, reason in (
                (np.arange(90), np.arange(90, 180), "tuning_insufficient_samples"),
                (np.arange(130), np.arange(130, 180), "selection_insufficient_samples"),
                ([], [], "unavailable_partitions")):
            with self.subTest(reason=reason):
                result = decision.fit_guarded_decision_rule(
                    probabilities, labels, self.CLASSES, tuning_rows=tuning, selection_rows=selection)
                self.assertEqual(result["diagnostics"]["reason"], reason)
                self.assertEqual(result["rule"]["multipliers"], [1., 1., 1.])
        result = decision.fit_guarded_decision_rule(
            np.empty((0, 3)), [], self.CLASSES, tuning_rows=[], selection_rows=[])
        self.assertEqual(result["diagnostics"]["reason"], "unavailable_partitions")
        self.assertEqual(result["rule"]["multipliers"], [1., 1., 1.])

    def test_invalid_partitions_are_rejected_without_casting_or_splitting(self):
        probabilities, labels, tuning, selection = self.fixture()
        bad = (
            ([0, 0, *range(1, 180)], selection),
            (list(range(179)), selection),
            (list(range(180)), list(range(179, 300))),
            (np.arange(180, dtype=float), selection),
            ([False, *range(1, 180)], selection),
            ([-1, *range(1, 180)], selection),
            ([300, *range(1, 180)], selection),
            ([[0, 1]], selection), ("0", selection),
        )
        for fitting_rows, selected_rows in bad:
            with self.subTest(tuning=repr(fitting_rows)[:50]), self.assertRaises(ValueError):
                decision.fit_guarded_decision_rule(
                    probabilities, labels, self.CLASSES,
                    tuning_rows=fitting_rows, selection_rows=selected_rows)

    def test_guarded_options_are_strict_finite_and_bounded(self):
        probabilities, labels, tuning, selection = self.fixture()
        for options in ([], {"unknown": 1}, {"regularization": -1},
                        {"regularization": float("nan")}, {"regularization": True},
                        {"min_selection_gain": 1.1}, {"max_selection_class_f1_drop": -1},
                        {"min_tuning_class_count": 0}, {"min_selection_samples": True},
                        {"max_rounds": 11}, {"max_candidates": 4097}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                decision.fit_guarded_decision_rule(
                    probabilities, labels, self.CLASSES, tuning_rows=tuning,
                    selection_rows=selection, options=options)

    def test_guarded_fit_is_deterministic_bounded_and_does_not_mutate_inputs(self):
        probabilities, labels, tuning, selection = self.fixture()
        snapshots = [value.copy() for value in (probabilities, labels, tuning, selection)]
        options = {"max_candidates": 5, "max_rounds": 2}
        options_before = copy.deepcopy(options)
        first = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning,
            selection_rows=selection, options=options)
        second = decision.fit_guarded_decision_rule(
            probabilities, labels, self.CLASSES, tuning_rows=tuning,
            selection_rows=selection, options=options)
        self.assertEqual(first, second)
        self.assertLessEqual(first["diagnostics"]["candidates_evaluated"], 5)
        self.assertLessEqual(first["diagnostics"]["rounds_completed"], 2)
        self.assertEqual(options, options_before)
        for actual, before in zip((probabilities, labels, tuning, selection), snapshots):
            np.testing.assert_array_equal(actual, before)

    def test_existing_v1_recipes_keep_published_json_fingerprints(self):
        # Golden rule hashes were calculated from the published f0ba480 module,
        # before the guarded implementation, not regenerated from current code.
        classes = self.CLASSES
        probabilities = np.asarray([[.75, .15, .1]] * 60
                                   + [[.49, .46, .05]] * 20 + [[.3, .1, .6]] * 20)
        labels = np.asarray(["01"] * 60 + ["NA"] * 20 + ["class c"] * 20)
        result = fit_decision_rule(probabilities, labels, classes)
        payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         "4b71f78b7402603e6055d484d452fd3a5858d335c52a3c71e436788531220cb9")
        rng = np.random.default_rng(31415)
        probabilities = rng.dirichlet([2., 1., .8], size=180)
        labels = np.asarray(classes)[rng.permutation(np.repeat(np.arange(3), 60))]
        result = fit_decision_rule(probabilities, labels, classes,
                                   options={"max_candidates": 7, "max_rounds": 2})
        payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         "ab2529dae28cbe40883b9bbdaca5b2e5fc2af823590ebaad89e7ba0c7ea573a7")


class GuardedClassificationProtocolTests(unittest.TestCase):
    def test_fresh_suite_has_fixed_seeds_and_new_disjoint_partitions(self):
        self.assertEqual(cases.suite_seeds("guarded"), (314159, 271828, 161803))
        self.assertTrue(set(cases.suite_seeds("guarded")).isdisjoint(cases.SEEDS))
        for name in cases.case_definitions("guarded"):
            for seed in cases.suite_seeds("guarded"):
                with self.subTest(name=name, seed=seed):
                    case = cases.make_case(name, seed, suite="guarded")
                    repeated = cases.make_case(name, seed, suite="guarded")
                    pd.testing.assert_frame_equal(case["frame"], repeated["frame"])
                    for key in ("labels", "ids", "fit_rows", "calibration_rows", "audit_rows"):
                        np.testing.assert_array_equal(case[key], repeated[key])
                    self.assertEqual([len(case[key]) for key in
                                      ("fit_rows", "calibration_rows", "audit_rows")], [900, 300, 300])
                    tuning, selection, metadata = cases.calibration_partition(case)
                    self.assertEqual((len(tuning), len(selection)), (180, 120))
                    np.testing.assert_array_equal(np.sort(np.concatenate((tuning, selection))), np.arange(300))
                    self.assertEqual(metadata["raw_feature_group_overlap"], 0)
                    frame = case["frame"].iloc[case["calibration_rows"]]
                    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
                    self.assertFalse(set(groups[tuning]) & set(groups[selection]))
                    if name == "categorical_shift":
                        self.assertLess(frame.iloc[tuning]["period"].max(), frame.iloc[selection]["period"].min())
                        for key in ("fit_rows", "calibration_rows"):
                            self.assertNotIn("unseen", case["frame"].iloc[case[key]]["habitat"].tolist())
                        self.assertIn("unseen", case["frame"].iloc[case["audit_rows"]]["habitat"].tolist())

    def test_calibration_split_does_not_depend_on_any_answers(self):
        for name in cases.case_definitions("guarded"):
            case = cases.make_case(name, 314159, suite="guarded")
            before = cases.calibration_partition(case)
            changed = dict(case)
            changed["labels"] = np.roll(case["labels"], 23)
            after = cases.calibration_partition(changed)
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2], after[2])

    def test_invalid_suite_and_seed_are_not_replaced_with_pilot(self):
        for suite in ("unknown", "", None):
            with self.subTest(suite=suite), self.assertRaises(ValueError):
                cases.make_case("imbalance_overlap", 314159, suite=suite)
        with self.assertRaises(ValueError):
            cases.make_case("imbalance_overlap", 73, suite="guarded")
        with self.assertRaises(ValueError):
            cases.make_case("imbalance_overlap", 314159)

    def test_mocked_guarded_worker_freezes_both_rules_and_audit_answers_cannot_change_them(self):
        fixture = GuardedClassificationDecisionTests()
        probabilities, calibration_y, tuning, selection = fixture.fixture()
        classes = fixture.CLASSES
        fit, calibration, audit = np.arange(30), np.arange(30, 330), np.arange(330, 360)
        frame = pd.DataFrame({"x": np.arange(360)})
        audit_p = np.asarray([[.49, .46, .05]] * 30)
        plan = {"plan_sha256": "fixed-guarded-unit-plan", "suite": "guarded",
                "guarded_method": {"options": {}}}
        original_guard = decision.fit_guarded_decision_rule
        original_diagnostics = experiment.diagnostics
        snapshots = []
        with tempfile.TemporaryDirectory() as temporary:
            for attempt, audit_label in enumerate(("01", "NA")):
                output = Path(temporary) / f"run-{attempt}"
                output.mkdir()
                directory = output / "fixture-314159"
                labels = np.concatenate((np.repeat(classes, 10), calibration_y, np.repeat(audit_label, 30)))
                case = {"name": "fixture", "seed": 314159, "suite": "guarded", "frame": frame,
                        "labels": labels, "ids": np.asarray([f"{i:04d}" for i in range(360)]),
                        "classes": classes, "fit_rows": fit, "calibration_rows": calibration,
                        "audit_rows": audit, "split": {"strategy": "grouped"}}
                calls = {"fit": 0, "guard": 0, "score": 0}
                self_test = self

                class StubPipeline:
                    named_steps = {"model": SimpleNamespace(classes_=np.asarray(classes))}

                    def fit(self, supplied, supplied_labels):
                        self_test.assertEqual(supplied.index.tolist(), fit.tolist())
                        np.testing.assert_array_equal(supplied_labels, labels[fit])
                        calls["fit"] += 1
                        return self

                    def predict_proba(self, supplied):
                        if supplied.index.tolist() == calibration.tolist():
                            self_test.assertFalse((directory / "frozen.json").exists())
                            return probabilities.copy()
                        self_test.assertEqual(supplied.index.tolist(), audit.tolist())
                        frozen = json.loads((directory / "frozen.json").read_text())
                        self_test.assertIn("rule", frozen)
                        self_test.assertIn("guarded_rule", frozen)
                        self_test.assertEqual(frozen["rule_sha256"], experiment.json_hash(frozen["rule"]))
                        self_test.assertEqual(frozen["guarded_rule_sha256"],
                                              experiment.json_hash(frozen["guarded_rule"]))
                        return audit_p.copy()

                def fake_dump(model, path, **unused):
                    Path(path).write_bytes(b"fixed-guarded-unit-model")

                def checked_guard(supplied_p, supplied_y, supplied_classes, **options):
                    self.assertFalse((directory / "frozen.json").exists())
                    np.testing.assert_array_equal(supplied_p, probabilities)
                    np.testing.assert_array_equal(supplied_y, calibration_y)
                    np.testing.assert_array_equal(options["tuning_rows"], tuning)
                    np.testing.assert_array_equal(options["selection_rows"], selection)
                    calls["guard"] += 1
                    return original_guard(supplied_p, supplied_y, supplied_classes, **options)

                def checked_diagnostics(actual, predictions, supplied_classes):
                    frozen = json.loads((directory / "frozen.json").read_text())
                    self.assertEqual(frozen["plan_sha256"], plan["plan_sha256"])
                    np.testing.assert_array_equal(actual, labels[audit])
                    calls["score"] += 1
                    return original_diagnostics(actual, predictions, supplied_classes)

                with patch.object(experiment, "load_plan", return_value=plan), \
                        patch.object(experiment, "make_case", return_value=case), \
                        patch.object(experiment, "calibration_partition", return_value=(
                            tuning, selection, {"strategy": "grouped", "raw_feature_group_overlap": 0})), \
                        patch.object(experiment, "make_pipeline", return_value=StubPipeline()), \
                        patch.object(experiment.joblib, "dump", side_effect=fake_dump), \
                        patch.object(experiment, "memory_state", return_value={
                            "measurement_available": False, "memory_limit_enforced": False}), \
                        patch.object(decision, "fit_guarded_decision_rule", side_effect=checked_guard), \
                        patch.object(experiment, "diagnostics", side_effect=checked_diagnostics):
                    result = experiment.run_case(output, {"name": "fixture", "seed": 314159})
                self.assertEqual(calls, {"fit": 1, "guard": 1, "score": 3})
                frozen = json.loads((directory / "frozen.json").read_text())
                self.assertEqual(frozen["guarded_rule"]["partitions"], {
                    "tuning_rows": tuning.tolist(), "selection_rows": selection.tolist()})
                audit_csv = pd.read_csv(directory / "audit_predictions.csv", keep_default_na=False,
                                        dtype={"id": str, "actual": str, "legacy": str, "guarded": str})
                self.assertTrue({"control", "legacy", "guarded"} <= set(audit_csv.columns))
                self.assertEqual(set(audit_csv["actual"].tolist()), {audit_label})
                snapshots.append((result, frozen))
        for key in ("rule", "guarded_rule"):
            self.assertEqual(snapshots[0][1][key], snapshots[1][1][key])
        for key in ("control_predictions", "legacy_predictions", "guarded_predictions"):
            self.assertEqual(snapshots[0][0]["hashes"][key], snapshots[1][0]["hashes"][key])
        self.assertNotEqual(snapshots[0][0]["scores"], snapshots[1][0]["scores"])


if __name__ == "__main__":
    unittest.main()
