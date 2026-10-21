"""Independent OOF admission never trains on selection labels or held groups."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from agent.classification_decision import predict_with_rule
from agent.oof_classification import OOFContractError, fit_oof_guarded_rule


CLASSES = ["01", "NA", "class c"]


class Clock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now


def fixture():
    # Every pair is an indivisible duplicate group; the DataFrame index is
    # deliberately not a local row position. Folds must use iloc semantics.
    group = np.repeat(np.arange(135), 2)
    frame = pd.DataFrame({
        "group_feature": group,
        "category": [f"A-{value}" if value < 90 else "B-only" for value in group],
    }, index=np.arange(1000, 1270))
    labels = np.asarray(CLASSES, dtype=object)[group % 3]
    tuning = np.arange(180)
    selection = np.arange(180, 270)
    folds = []
    for part in range(3):
        held = tuning[part * 60:(part + 1) * 60]
        folds.append({"fit_rows": np.setdiff1d(tuning, held), "held_rows": held})
    return frame, labels, group, tuning, selection, folds


def known_probabilities(frame):
    choices = np.asarray([[.75, .15, .10], [.49, .46, .05], [.20, .10, .70]])
    return choices[frame["group_feature"].to_numpy(dtype=int) % 3].copy()


class Callback:
    def __init__(self, *, permutation=None, clock=None, seconds=0.0):
        self.permutation = list(range(3)) if permutation is None else list(permutation)
        self.clock = clock
        self.seconds = seconds
        self.calls = []
        self.matrices = []

    def __call__(self, X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
        self.calls.append({
            "fit": (X_fit.index.to_numpy() - 1000).tolist(),
            "predict": (X_predict.index.to_numpy() - 1000).tolist(),
            "y": np.asarray(y_fit).copy(), "timeout": timeout_seconds,
            "fit_index": fit_index,
        })
        probabilities = known_probabilities(X_predict)[:, self.permutation]
        self.matrices.append(probabilities)
        if self.clock is not None:
            self.clock.now += self.seconds
        return {"probabilities": probabilities,
                "classes": [CLASSES[index] for index in self.permutation]}


class OOFTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.tuning, self.selection, self.folds = fixture()

    def run_rule(self, callback=None, **overrides):
        arguments = {
            "groups": self.groups, "split_policy": "grouped",
            "tuning_rows": self.tuning, "selection_rows": self.selection,
            "folds": self.folds, "fit_predict": callback or Callback(),
            "fold_timeout_seconds": 10.0, "deadline": 100.0, "clock": Clock(),
        }
        arguments.update(overrides)
        return fit_oof_guarded_rule(self.X, self.y, CLASSES, **arguments)

    def assert_identity(self, result):
        self.assertEqual(result["status"], "identity")
        self.assertEqual(result["rule"]["multipliers"], [1.0, 1.0, 1.0])
        json.dumps(result, allow_nan=False)

    def assert_contract_without_fit(self, **changes):
        callback = Callback()
        with self.assertRaises(OOFContractError):
            self.run_rule(callback, **changes)
        self.assertEqual(callback.calls, [])

    def test_oof_coverage_and_group_separation_and_B_never_enters_fit(self):
        callback = Callback()
        result = self.run_rule(callback)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["reason"], "accepted")
        self.assertEqual(result["completed_fits"], 4)
        self.assertEqual([row["fit_index"] for row in callback.calls], [0, 1, 2, 3])
        seen = []
        for index, call in enumerate(callback.calls[:3]):
            self.assertEqual(set(call["fit"]), set(self.folds[index]["fit_rows"]))
            self.assertEqual(set(call["predict"]), set(self.folds[index]["held_rows"]))
            self.assertFalse(set(call["fit"]) & set(self.selection))
            self.assertFalse(set(self.groups[call["fit"]]) & set(self.groups[call["predict"]]))
            self.assertEqual(set(call["y"]), set(CLASSES))
            seen.extend(call["predict"])
        self.assertEqual(sorted(seen), self.tuning.tolist())
        self.assertEqual(len(seen), len(set(seen)))
        last = callback.calls[3]
        self.assertEqual(set(last["fit"]), set(self.tuning))
        self.assertEqual(set(last["predict"]), set(self.selection))
        self.assertFalse(set(self.groups[last["fit"]]) & set(self.groups[last["predict"]]))
        self.assertEqual(result["support"]["tuning"], [60, 60, 60])
        self.assertEqual(result["support"]["selection"], [30, 30, 30])

    def test_iid_policy_still_honours_explicit_duplicate_groups(self):
        callback = Callback()
        self.run_rule(callback, split_policy="iid")
        for call in callback.calls:
            self.assertFalse(set(self.groups[call["fit"]]) & set(self.groups[call["predict"]]))

    def test_predict_labels_are_not_callback_arguments_and_preprocessing_fits_only_fit(self):
        records = []

        def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(X_fit[["category"]])
            encoded = encoder.transform(X_predict[["category"]])
            records.append((encoder.categories_[0].tolist(), encoded.copy()))
            # Categories are unique to raw groups; all held groups are unseen.
            self.assertTrue(np.all(encoded == 0))
            self.assertNotIn("B-only", encoder.categories_[0].tolist())
            self.assertEqual(len(y_fit), len(X_fit))
            return {"probabilities": known_probabilities(X_predict), "classes": CLASSES}

        result = self.run_rule(callback)
        self.assertEqual(result["completed_fits"], 4)
        self.assertEqual(len(records), 4)

    def test_class_column_permutation_and_literal_labels_preserve_frozen_rule(self):
        direct = self.run_rule(Callback())
        permuted = self.run_rule(Callback(permutation=[2, 0, 1]))
        self.assertEqual(direct["classes"], CLASSES)
        self.assertEqual(direct["rule"], permuted["rule"])
        self.assertEqual(direct["candidate_rule"], permuted["candidate_rule"])
        restored = json.loads(json.dumps(permuted, allow_nan=False))
        expected = predict_with_rule(known_probabilities(self.X), CLASSES, direct["rule"])
        actual = predict_with_rule(known_probabilities(self.X)[:, [2, 0, 1]],
                                   ["class c", "01", "NA"], restored["rule"])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(set(actual), set(CLASSES))

    def test_guard_uses_local_order_when_tuning_and_selection_rows_are_permuted(self):
        original = self.run_rule()
        reordered = self.run_rule(tuning_rows=self.tuning[::-1], selection_rows=self.selection[::-1])
        self.assertEqual(original["rule"], reordered["rule"])
        self.assertEqual(original["candidate_rule"], reordered["candidate_rule"])

    def test_selection_target_intervention_cannot_change_tuned_candidate(self):
        first = self.run_rule()
        self.y[self.selection] = np.roll(self.y[self.selection], 2)
        second = self.run_rule()
        self.assertEqual(first["candidate_rule"], second["candidate_rule"])
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(second["status"], "identity")

    def test_callback_cannot_change_the_already_frozen_guard_configuration(self):
        expected = self.run_rule()
        options = {"min_selection_gain": .01}
        spy = Callback()

        def callback(*args, **kwargs):
            options["min_selection_gain"] = .99
            return spy(*args, **kwargs)

        result = self.run_rule(callback, guard_options=options)
        self.assertEqual(options["min_selection_gain"], .99)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["guarded"]["options"]["min_selection_gain"], .01)
        self.assertEqual(result["candidate_rule"], expected["candidate_rule"])

    def test_true_identity_stays_identity_with_perfect_probabilities(self):
        def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
            labels = X_predict["group_feature"].to_numpy(dtype=int) % 3
            matrix = np.eye(3)[labels]
            return {"probabilities": matrix, "classes": CLASSES}

        result = self.run_rule(callback)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "identity_on_tuning")
        self.assertEqual(result["completed_fits"], 4)

    def test_inputs_are_unchanged_even_when_callback_mutates_its_subsets(self):
        original = (self.X.copy(deep=True), self.y.copy(), self.groups.copy(),
                    self.tuning.copy(), self.selection.copy(), copy.deepcopy(self.folds))

        def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
            probabilities = known_probabilities(X_predict)
            X_fit.iloc[:, X_fit.columns.get_loc("category")] = "mutated fit"
            X_predict.iloc[:, X_predict.columns.get_loc("category")] = "mutated predict"
            y_fit[0] = "mutated label"
            return {"probabilities": probabilities, "classes": list(CLASSES)}

        self.run_rule(callback)
        pd.testing.assert_frame_equal(self.X, original[0])
        for actual, expected in zip((self.y, self.groups, self.tuning, self.selection), original[1:5]):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(self.folds, original[5]):
            for key in ("fit_rows", "held_rows"):
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_nested_object_cells_are_not_shared_with_callback_subsets(self):
        self.X["nested"] = [{"tokens": ["original"]} for _ in range(len(self.X))]
        original = copy.deepcopy(self.X.to_dict("records"))

        def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
            probabilities = known_probabilities(X_predict)
            X_fit.at[X_fit.index[0], "nested"]["tokens"].append("mutated-fit")
            X_predict.at[X_predict.index[0], "nested"]["tokens"].append("mutated-predict")
            return {"probabilities": probabilities, "classes": CLASSES}

        self.run_rule(callback)
        self.assertEqual(self.X.to_dict("records"), original)

    def test_numpy_feature_table_uses_local_rows_and_copied_subsets(self):
        values = self.X.to_numpy(dtype=object)
        original = values.copy()
        calls = []

        def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
            groups = np.asarray(X_predict[:, 0], dtype=int)
            probabilities = np.asarray([[.75, .15, .1], [.49, .46, .05], [.2, .1, .7]])[groups % 3]
            calls.append(fit_index)
            X_fit[0, 0], X_predict[0, 0] = -999, -999
            return {"probabilities": probabilities, "classes": CLASSES}

        result = fit_oof_guarded_rule(
            values, self.y, CLASSES, groups=self.groups, split_policy="grouped",
            tuning_rows=self.tuning, selection_rows=self.selection, folds=self.folds,
            fit_predict=callback, fold_timeout_seconds=10., deadline=100., clock=Clock())
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(calls, [0, 1, 2, 3])
        np.testing.assert_array_equal(values, original)

    def test_refit_and_apply_leave_frozen_rule_and_raw_probabilities_unchanged(self):
        callback = Callback()
        result = self.run_rule(callback)
        frozen = json.dumps(result["rule"], sort_keys=True, allow_nan=False)
        probabilities = known_probabilities(self.X)
        saved = probabilities.copy()
        # Later production work has no reference to anything inside the rule.
        for matrix in callback.matrices:
            matrix[:] = np.nan
        self.run_rule(Callback(permutation=[1, 2, 0]))
        predict_with_rule(probabilities, CLASSES, result["rule"])
        self.assertEqual(json.dumps(result["rule"], sort_keys=True, allow_nan=False), frozen)
        np.testing.assert_array_equal(probabilities, saved)

    def test_wrapper_is_finite_json_with_reproducible_split_and_rule_hashes(self):
        first, second = self.run_rule(), self.run_rule()
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        for name in ("split_sha256", "rule_sha256", "candidate_rule_sha256"):
            self.assertEqual(first["hashes"][name], second["hashes"][name])
            self.assertEqual(len(first["hashes"][name]), 64)
            int(first["hashes"][name], 16)
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["kind"], "oof_guarded_macro_f1")

    def test_invalid_partition_indices_are_rejected_before_training(self):
        invalid = ([-1, *self.tuning[1:]], [True, *self.tuning[1:]],
                   [0.5, *self.tuning[1:]], [[0], *self.tuning[1:]],
                   [270, *self.tuning[1:]], [0, 0, *self.tuning[2:]],
                   self.tuning[:-1], "all")
        for rows in invalid:
            with self.subTest(rows=str(rows)[:60]):
                self.assert_contract_without_fit(tuning_rows=rows)

    def test_overlapping_A_B_and_uncovered_rows_are_contract_errors(self):
        self.assert_contract_without_fit(selection_rows=np.r_[self.tuning[0], self.selection[1:]])
        self.assert_contract_without_fit(selection_rows=self.selection[:-1])

    def test_group_overlap_between_A_and_B_is_rejected_for_both_supported_policies(self):
        self.groups[self.selection[0]] = self.groups[self.tuning[0]]
        for policy in ("iid", "grouped"):
            with self.subTest(policy=policy):
                self.assert_contract_without_fit(split_policy=policy)

    def test_invalid_group_vectors_are_contract_errors(self):
        bad = [self.groups[:-1], self.groups[:, None], [None] * len(self.groups),
               [np.nan] * len(self.groups)]
        for groups in bad:
            with self.subTest(shape=str(np.shape(groups))):
                self.assert_contract_without_fit(groups=groups)

    def test_exactly_three_structured_folds_are_required(self):
        for folds in (self.folds[:2], self.folds + [self.folds[0]], {},
                      [{"fit_rows": [], "held_rows": []}] * 2,
                      [{"fit_rows": self.tuning}] * 3):
            with self.subTest(folds=str(folds)[:70]):
                self.assert_contract_without_fit(folds=folds)

    def test_empty_held_fold_is_invalid_even_when_other_two_cover_all_A(self):
        parts = [np.asarray([], dtype=int), self.tuning[:90], self.tuning[90:]]
        folds = [{"fit_rows": np.setdiff1d(self.tuning, held), "held_rows": held} for held in parts]
        self.assert_contract_without_fit(folds=folds)

    def test_no_selection_row_may_appear_in_any_fold(self):
        for key in ("fit_rows", "held_rows"):
            folds = copy.deepcopy(self.folds)
            folds[0][key][0] = self.selection[0]
            with self.subTest(key=key):
                self.assert_contract_without_fit(folds=folds)

    def test_fold_fit_held_overlap_and_held_coverage_errors_are_rejected(self):
        folds = copy.deepcopy(self.folds)
        folds[0]["fit_rows"][0] = folds[0]["held_rows"][0]
        self.assert_contract_without_fit(folds=folds)
        folds = copy.deepcopy(self.folds)
        folds[1]["held_rows"] = folds[0]["held_rows"].copy()
        self.assert_contract_without_fit(folds=folds)
        folds = copy.deepcopy(self.folds)
        folds[0]["held_rows"] = folds[0]["held_rows"][:-1]
        self.assert_contract_without_fit(folds=folds)

    def test_group_overlap_inside_one_fold_is_rejected_before_training(self):
        self.groups[self.folds[0]["fit_rows"][0]] = self.groups[self.folds[0]["held_rows"][0]]
        self.assert_contract_without_fit()

    def test_fold_index_validation_does_not_silently_cast_bool_or_float(self):
        for key in ("fit_rows", "held_rows"):
            for bad in (True, 1.5, -1, 270):
                folds = copy.deepcopy(self.folds)
                rows = folds[0][key].tolist()
                rows[0] = bad
                folds[0][key] = rows
                with self.subTest(key=key, bad=bad):
                    self.assert_contract_without_fit(folds=folds)

    def test_time_policy_returns_explicit_identity_without_fitting(self):
        callback = Callback()
        result = self.run_rule(callback, split_policy="time")
        self.assert_identity(result)
        self.assertEqual(result["reason"], "unsupported_time_policy")
        self.assertEqual(result["completed_fits"], 0)
        self.assertEqual(callback.calls, [])

    def test_unknown_split_policy_is_not_silently_treated_as_iid(self):
        self.assert_contract_without_fit(split_policy="guess")

    def test_insufficient_selection_support_returns_identity_without_new_splits(self):
        self.y[self.selection] = "01"
        self.y[self.selection[:4]] = "class c"
        self.y[self.selection[4:10]] = "NA"
        callback = Callback()
        result = self.run_rule(callback)
        self.assert_identity(result)
        self.assertIn("selection", result["reason"])
        self.assertLessEqual(result["completed_fits"], 4)
        self.assertEqual(result["support"]["selection"], [80, 6, 4])

    def test_fit_missing_a_class_returns_support_identity_without_resplitting(self):
        self.y[self.folds[0]["fit_rows"]] = "01"
        callback = Callback()
        result = self.run_rule(callback)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "fit_missing_class")
        self.assertEqual(callback.calls, [])

    def test_callback_failures_return_identity_and_never_use_partial_oof(self):
        for failure in (TimeoutError("late"), RuntimeError("model failure"),
                        ValueError("model cannot fit"), MemoryError("limit")):
            callback = Callback()

            def failing(*args, **kwargs):
                if len(callback.calls) == 1:
                    raise failure
                return callback(*args, **kwargs)

            with self.subTest(failure=type(failure).__name__):
                result = self.run_rule(failing)
                self.assert_identity(result)
                self.assertEqual(result["completed_fits"], 1)
                self.assertIsNone(result["guarded"])
                self.assertEqual(result["candidate_rule"]["multipliers"], [1.0] * 3)
                expected = "callback_timeout" if isinstance(failure, TimeoutError) else "callback_error"
                self.assertEqual(result["reason"], expected)

    def test_one_global_deadline_reduces_each_callback_allowance(self):
        clock = Clock()
        callback = Callback(clock=clock, seconds=6.0)
        result = self.run_rule(callback, fold_timeout_seconds=10.0, deadline=25.0, clock=clock)
        self.assertEqual(result["completed_fits"], 4)
        self.assertEqual([call["timeout"] for call in callback.calls], [10., 10., 10., 7.])
        self.assertAlmostEqual(result["timing"]["elapsed_seconds"], 24.)
        self.assertAlmostEqual(result["timing"]["calibration_fit_seconds"], 24.)

    def test_already_exhausted_deadline_does_not_launch_a_fit(self):
        callback = Callback()
        result = self.run_rule(callback, deadline=5.0, clock=Clock(5.0))
        self.assert_identity(result)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertEqual(result["completed_fits"], 0)
        self.assertEqual(callback.calls, [])

    def test_callback_return_after_deadline_is_not_accepted(self):
        clock = Clock()
        callback = Callback(clock=clock, seconds=11.)
        result = self.run_rule(callback, fold_timeout_seconds=20., deadline=10., clock=clock)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertEqual(len(callback.calls), 1)
        self.assertIsNone(result["guarded"])

    def test_return_after_individual_fit_allowance_cannot_hide_inside_global_budget(self):
        clock = Clock()
        callback = Callback(clock=clock, seconds=6.)
        result = self.run_rule(callback, fold_timeout_seconds=5., deadline=100., clock=clock)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "callback_timeout")
        self.assertEqual(len(callback.calls), 1)
        self.assertIsNone(result["guarded"])

    def test_result_finalization_must_not_accept_after_global_deadline(self):
        import agent.oof_classification as oof

        class FinalizationClock:
            def __init__(self):
                self.stage = 0

            def __call__(self):
                if self.stage == 1:
                    self.stage = 2
                    return .9
                return 1.1 if self.stage == 2 else 0.

        # Time also passes outside callback execution. Finalization must
        # discard an accepted rule if that last work crosses the deadline.
        clock = FinalizationClock()
        original_guard = oof.fit_guarded_decision_rule

        def guard(*args, **kwargs):
            result = original_guard(*args, **kwargs)
            if len(args[0]):
                clock.stage = 1
            return result

        with patch.object(oof, "fit_guarded_decision_rule", side_effect=guard):
            result = self.run_rule(deadline=1., clock=clock)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertEqual(result["completed_fits"], 4)

    def test_expensive_subset_staging_cannot_launch_fit_after_deadline(self):
        clock = Clock()

        class CostlyCell:
            def __deepcopy__(self, memo):
                clock.now += .1
                return CostlyCell()

        self.X["nested"] = [CostlyCell() for _ in range(len(self.X))]
        callback = Callback()
        result = self.run_rule(callback, clock=clock, deadline=1.)
        self.assert_identity(result)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertEqual(callback.calls, [])
        self.assertEqual(result["completed_fits"], 0)

    def test_invalid_budget_and_guard_options_are_contract_errors_before_fit(self):
        for value in (0, -1, True, float("nan"), float("inf"), "10"):
            with self.subTest(timeout=value):
                self.assert_contract_without_fit(fold_timeout_seconds=value)
        for value in (True, float("nan"), float("inf"), "10"):
            with self.subTest(deadline=value):
                self.assert_contract_without_fit(deadline=value)
        self.assert_contract_without_fit(guard_options={"unknown": 1})
        self.assert_contract_without_fit(guard_options={"min_selection_class_count": 0})

    def test_clock_must_be_finite_and_monotonic(self):
        for value in (float("nan"), float("inf"), True, "0"):
            with self.subTest(clock=value):
                self.assert_contract_without_fit(clock=lambda: value)
        values = iter([2.0, 1.0])
        self.assert_contract_without_fit(clock=lambda: next(values))

    def test_malformed_callback_outputs_are_contract_errors_not_identity(self):
        valid = {"probabilities": np.tile([.5, .3, .2], (60, 1)), "classes": CLASSES}
        outputs = [None, [], {"classes": CLASSES}, {"probabilities": valid["probabilities"]}]
        for classes in (["01", "NA", "NA"], ["01", "NA", "other"], [1, "NA", "class c"],
                        ["01", "NA"], "01"):
            outputs.append({**valid, "classes": classes})
        for matrix in (np.ones((60, 2)), np.ones((59, 3)) / 3,
                       np.tile([.7, .4, .2], (60, 1)), np.tile([1.1, -.1, 0], (60, 1)),
                       np.tile([np.nan, .4, .6], (60, 1)), np.tile([np.inf, 0, 0], (60, 1)),
                       np.ones((60, 3), dtype=complex) / 3,
                       np.tile([True, False, False], (60, 1)),
                       np.full((60, 3), "0.3333333333333333")):
            outputs.append({**valid, "probabilities": matrix})
        for output in outputs:
            def callback(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
                return output

            with self.subTest(output=str(output)[:85]), self.assertRaises(OOFContractError):
                self.run_rule(callback)


class OOFCaseTests(unittest.TestCase):
    def test_frozen_stages_use_disjoint_predeclared_seeds(self):
        import oof_classification_cases as cases

        development = cases.suite_cases("development")
        confirmation = cases.suite_cases("confirmation")
        self.assertEqual(len(development), 6)
        self.assertEqual(len(confirmation), 9)
        self.assertEqual({row["seed"] for row in development}, {131, 379})
        self.assertEqual({row["seed"] for row in confirmation}, {2203, 4001, 10009})
        self.assertEqual({row["name"] for row in development}, set(cases.FAMILIES))
        self.assertEqual({row["name"] for row in confirmation}, set(cases.FAMILIES))

    def test_cases_and_partitions_are_reproducible_and_label_independent(self):
        import oof_classification_cases as cases

        for name in cases.FAMILIES:
            with self.subTest(family=name):
                case = cases.make_case(name, 131)
                duplicate = cases.make_case(name, 131)
                pd.testing.assert_frame_equal(case["frame"], duplicate["frame"])
                np.testing.assert_array_equal(case["labels"], duplicate["labels"])
                np.testing.assert_array_equal(case["groups"], duplicate["groups"])
                parts = cases.make_partitions(case)
                self.assertEqual(parts["status"], "ready")
                case["labels"][:] = "NA"
                altered = cases.make_partitions(case)
                for key in ("tuning_rows", "selection_rows"):
                    np.testing.assert_array_equal(parts[key], altered[key])
                for original, changed in zip(parts["folds"], altered["folds"]):
                    for key in ("fit_rows", "held_rows"):
                        np.testing.assert_array_equal(original[key], changed[key])
                groups = case["groups"][case["allowed_rows"]]
                tune, selection = parts["tuning_rows"], parts["selection_rows"]
                self.assertEqual(sorted(np.r_[tune, selection]), list(range(1500)))
                self.assertFalse(set(groups[tune]) & set(groups[selection]))
                held_once = []
                for fold in parts["folds"]:
                    fit, held = fold["fit_rows"], fold["held_rows"]
                    self.assertFalse(set(groups[fit]) & set(groups[held]))
                    self.assertFalse(set(fit) & set(selection))
                    held_once.extend(held)
                self.assertEqual(sorted(held_once), sorted(tune))
                for left, right in (("allowed_rows", "outer_rows"),
                                    ("allowed_rows", "production_rows"),
                                    ("outer_rows", "production_rows")):
                    self.assertFalse(set(case["groups"][case[left]]) & set(case["groups"][case[right]]))

    def test_raw_duplicate_group_keys_exclude_index_and_keep_literal_categories(self):
        import oof_classification_cases as cases

        raw = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0],
                            "category": ["NA", "NA", "01", " NA "]}, index=[9, 8, 7, 6])
        groups = cases.raw_feature_groups(raw)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])
        np.testing.assert_array_equal(groups, cases.raw_feature_groups(raw.reset_index(drop=True)))

    def test_impossible_group_split_is_explicit_unavailable_without_retry(self):
        import oof_classification_cases as cases

        case = cases.make_case("impossible_split", 131)
        parts = cases.make_partitions(case)
        self.assertEqual(parts["status"], "unavailable")
        self.assertEqual(parts["reason"], "insufficient_groups_for_tuning_selection")
        self.assertEqual(parts["metadata"]["resplit_attempts"], 0)
        self.assertEqual(parts["folds"], [])
        self.assertEqual(len(set(case["groups"][case["allowed_rows"]])), 1)


class OOFHostTests(unittest.TestCase):
    def test_rule_is_frozen_before_both_refits_and_before_any_audit_scoring(self):
        import experiment_oof_classification as host

        frame, labels, groups, tuning, selection, folds = fixture()
        extension_groups = np.repeat(np.arange(135, 165), 2)
        extension = pd.DataFrame({"group_feature": extension_groups,
                                  "category": "external"}, index=np.arange(1270, 1330))
        case = {"frame": pd.concat((frame, extension)),
                "labels": np.r_[labels, np.asarray(CLASSES, dtype=object)[extension_groups % 3]],
                "classes": list(CLASSES), "groups": np.r_[groups, extension_groups],
                "ids": np.asarray([f"row-{index}" for index in range(330)]),
                "allowed_rows": np.arange(270), "outer_rows": np.arange(270, 300),
                "production_rows": np.arange(300, 330), "split_policy": "grouped"}
        splits = {"partitions": {"status": "ready", "tuning_rows": tuning,
                                  "selection_rows": selection, "folds": folds}}
        entry = {"key": "mock-17", "seed": 17, "arm_cap_seconds": 180,
                 "calibration_fit_timeout_seconds": 25, "final_reserve_seconds": 60,
                 "source_sha256": "source", "split_sha256": "split"}
        plan = {"plan_sha256": "plan", "guard_options": dict(host.DEFAULT_GUARD_OPTIONS)}
        events, fitted_rows = [], []

        class FitSpy:
            def __init__(self, directory, seed):
                self.directory = Path(directory)
                self.records = []

            def __call__(self, X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
                if fit_index >= 4:
                    frozen = self.directory.parent / "frozen.json"
                    if not frozen.is_file():
                        raise AssertionError("Final model started before rule freeze")
                events.append(f"fit-{fit_index}")
                fitted_rows.append((fit_index, X_fit.index.tolist(), np.asarray(y_fit).copy()))
                self.records.append({"metadata": {"fit_seconds": .01}, "wall_seconds": .01})
                return {"probabilities": known_probabilities(X_predict), "classes": CLASSES}

        original_writer, original_score = host.write_json, host._score

        def write(path, value):
            if Path(path).name == "frozen.json":
                events.append("freeze")
            original_writer(path, value)

        def score(*args):
            self.assertIn("fit-5", events)
            events.append("score")
            return original_score(*args)

        with tempfile.TemporaryDirectory(prefix="oof-host-freeze-") as temporary:
            directory = Path(temporary)
            (directory / "cases" / entry["key"]).mkdir(parents=True)
            with patch.object(host, "load_case", return_value=(case, splits)), \
                    patch.object(host, "BoundedFitPredict", FitSpy), \
                    patch.object(host, "write_json", side_effect=write), \
                    patch.object(host, "_score", side_effect=score):
                result = host.run_arm(directory, entry, "oof", plan)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["immutable_rule_after_refits"])
            self.assertEqual(result["calibration_fit_count"], 4)
            self.assertEqual(result["total_fit_count"], 6)
            self.assertEqual(events[:7], ["fit-0", "fit-1", "fit-2", "fit-3", "freeze", "fit-4", "fit-5"])
            self.assertEqual(events[7:], ["score"] * 4)
            for index, rows, answers in fitted_rows[:4]:
                self.assertFalse(set(rows) & set(frame.iloc[selection].index))
            self.assertEqual(fitted_rows[4][1], case["frame"].iloc[case["allowed_rows"]].index.tolist())
            self.assertEqual(fitted_rows[5][1], case["frame"].iloc[np.arange(300)].index.tolist())

    def test_reaudit_rejects_empty_missing_arm_and_duplicate_case_reports(self):
        import experiment_oof_classification as host

        entry = {"key": "planned", "stage": "development"}
        plan = {"plan_sha256": "plan", "cases": [entry]}
        failed = {"status": "failed"}
        malformed = [[], [{"key": "planned", "stage": "development", "arms": {"control": failed}}],
                     [{"key": "planned", "stage": "development", "arms": {"control": failed, "oof": failed}}] * 2]
        with tempfile.TemporaryDirectory(prefix="oof-audit-coverage-") as temporary:
            directory = Path(temporary)
            for pairs in malformed:
                report = {"status": "completed", "plan_sha256": "plan", "pairs": pairs}
                host.write_json(directory / "report.json", report)
                with self.subTest(pairs=len(pairs)), \
                        patch.object(host, "load_plan", return_value=plan), \
                        patch.object(host, "load_case", side_effect=AssertionError("Invalid report reached case audit")), \
                        self.assertRaises(ValueError):
                    host.re_audit(directory)

    def test_optional_public_task_uses_actual_TaskSpec_fields_and_literal_ids(self):
        import experiment_oof_classification as host

        train = pd.DataFrame({"id": [f"0{i}" for i in range(9)], "x": np.arange(9),
                              "target": CLASSES * 3})
        test = pd.DataFrame({"id": ["01", "NA"], "x": [99, 100]})
        y = train["target"].copy()
        with tempfile.TemporaryDirectory(prefix="oof-public-contract-") as temporary:
            task = Path(temporary) / "task.md"
            task.write_text("Metric: Macro F1\n", encoding="utf-8")
            spec = SimpleNamespace(metric=SimpleNamespace(name="Macro F1"),
                                   task_path=task, data_dir=Path(temporary), classes=tuple(CLASSES))

            def grouped(train_frame, labels, features, fold):
                held = np.arange(fold * 3, fold * 3 + 3)
                return np.setdiff1d(np.arange(9), held), held, {"fold_index": fold}

            data = (train, test, None, "id", "target", y, ["x"], ["x"], [], True)
            with patch("agent.task_spec.load_task", return_value=spec), \
                    patch("offline_baseline.load_data", return_value=data) as loader, \
                    patch("offline_baseline.prepare_features", side_effect=lambda frame, features, *args: frame[features].copy()), \
                    patch("experiment_classification_weights.grouped_fold", side_effect=grouped) as splitter:
                cases = host._public_cases(task)
            self.assertEqual(len(cases), 3)
            self.assertEqual([call.args[-1] for call in splitter.call_args_list], [0, 1, 2])
            loader.assert_called_once_with(Path(temporary), task_text="Metric: Macro F1\n", classification=True)
            for index, case in enumerate(cases):
                self.assertEqual(case["name"], f"public_fold{index}")
                self.assertEqual(case["classes"], CLASSES)
                self.assertEqual(case["production_ids"].tolist(), ["01", "NA"])
                self.assertEqual(case["production_frame"].index.tolist(), [9, 10])
                self.assertEqual(case["labels"].tolist(), CLASSES * 3)
                self.assertEqual(case["production_rows"].tolist(), [])

    def test_preregistration_rejects_plan_source_and_environment_changes_before_run(self):
        import experiment_oof_classification as host

        unsigned = {"public": {"status": "skipped_missing_data", "input_sha256": {}},
                    "source_sha256": {"frozen.py": "old"}, "environment": {"python": "fixed"},
                    "model_config": {"n_estimators": 64}}
        plan = {**unsigned, "plan_sha256": host.json_hash(unsigned)}
        with tempfile.TemporaryDirectory(prefix="oof-plan-contract-") as temporary:
            directory = Path(temporary)
            path = directory / "preregistration.json"
            host.write_json(path, plan)
            with patch.object(host, "source_hashes", return_value={"frozen.py": "old"}), \
                    patch.object(host, "environment_versions", return_value={"python": "fixed"}):
                self.assertEqual(host.load_plan(directory), plan)
                changed = copy.deepcopy(plan)
                changed["model_config"]["n_estimators"] = 999
                host.write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    host.load_plan(directory)
            host.write_json(path, plan)
            with patch.object(host, "source_hashes", return_value={"frozen.py": "changed"}):
                with self.assertRaisesRegex(ValueError, "sources changed"):
                    host.load_plan(directory)
            with patch.object(host, "source_hashes", return_value={"frozen.py": "old"}), \
                    patch.object(host, "environment_versions", return_value={"python": "changed"}):
                with self.assertRaisesRegex(ValueError, "Environment changed"):
                    host.load_plan(directory)

    def test_real_bounded_child_fits_preprocessing_only_on_fitting_rows(self):
        import experiment_oof_classification as host

        X_fit = pd.DataFrame({"value": [0., 2., 4., np.nan, 0., 4.],
                              "category": ["seen"] * 6}, index=[10, 11, 12, 13, 14, 15])
        y_fit = np.asarray(CLASSES * 2)
        X_predict = pd.DataFrame({"value": [999.], "category": ["novel"]}, index=[999])
        fit_before, predict_before = X_fit.copy(deep=True), X_predict.copy(deep=True)
        with tempfile.TemporaryDirectory(prefix="oof-child-fit-") as temporary:
            callback = host.BoundedFitPredict(Path(temporary) / "fits", seed=17)
            output = callback(X_fit, y_fit, X_predict, timeout_seconds=30., fit_index=0)
            self.assertEqual(len(callback.records), 1)
            record = callback.records[0]
            self.assertEqual(record["receipt"]["exit_code"], 0)
            self.assertFalse(record["receipt"]["timed_out"])
            metadata = record["metadata"]
            self.assertEqual(metadata["preprocessing"]["numeric_statistics"], {"value": 2.})
            self.assertEqual(metadata["preprocessing"]["categorical_categories"], {"category": ["seen"]})
            self.assertEqual(metadata["fit_source_rows"], [10, 11, 12, 13, 14, 15])
            self.assertEqual(metadata["predict_source_rows"], [999])
            request = joblib.load(Path(record["directory"]) / "request.joblib")
            self.assertEqual(set(request), {"X_fit", "y_fit", "X_predict", "seed", "model_config"})
            np.testing.assert_array_equal(request["y_fit"], y_fit)
            self.assertEqual(output["classes"], CLASSES)
            self.assertEqual(output["probabilities"].shape, (1, 3))
            self.assertAlmostEqual(output["probabilities"].sum(), 1.)
            self.assertGreater(metadata["fit_seconds"], 0.)
            for name, key in (("request.joblib", "request_sha256"),
                              ("model.joblib", "model_sha256"),
                              ("probabilities.npy", "probabilities_sha256")):
                self.assertEqual(host.file_hash(Path(record["directory"]) / name), metadata[key])
        pd.testing.assert_frame_equal(X_fit, fit_before)
        pd.testing.assert_frame_equal(X_predict, predict_before)


if __name__ == "__main__":
    unittest.main()
