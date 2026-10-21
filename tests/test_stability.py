"""Repeat evaluation cannot cross the outer boundary or reward partial rounds."""

import unittest

import numpy as np
import pandas as pd

from agent.splits import split_rows
from agent.stability import additional_inner_splits, paired_stability_summary


class InnerStabilityTests(unittest.TestCase):
    def test_random_repeats_are_fixed_local_and_do_not_reuse_canonical_split(self):
        full = pd.DataFrame({"x": np.arange(160), "unused": "keep out"})
        y = pd.Series(["a", "b"] * 80)
        outer, external, _ = split_rows(full, y, ["x"], True)
        view, labels = full.iloc[outer], y.iloc[outer]
        fit, held, _ = split_rows(view, labels, ["x"], True)
        checks = additional_inner_splits(view, labels, ["x"], True, canonical=(fit, held))
        self.assertEqual(len(checks), 2)
        self.assertEqual([metadata["seed"] for _, _, metadata in checks], [43, 44])
        self.assertNotEqual(checks[0][2]["split_hash"], checks[1][2]["split_hash"])
        for fitting, validation, metadata in checks:
            self.assertFalse(set(fitting) & set(validation))
            self.assertEqual(set(fitting) | set(validation), set(range(len(outer))))
            self.assertFalse(set(outer[validation]) & set(external))
            self.assertNotEqual(set(validation), set(held))
            self.assertEqual(set(labels.iloc[fitting]), {"a", "b"})
            self.assertEqual(set(labels.iloc[validation]), {"a", "b"})
            self.assertEqual(len(metadata["split_hash"]), 64)
        # Host labels outside the explicitly supplied view cannot alter the
        # extra rows, class coverage decisions, metadata or ranking policy.
        y.iloc[external] = "new outer-only class"
        changed = additional_inner_splits(full.iloc[outer], y.iloc[outer], ["x"], True,
                                          canonical=(fit, held))
        for before, after in zip(checks, changed):
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2], after[2])

    def test_implicit_canonical_and_explicit_canonical_give_identical_checks(self):
        frame = pd.DataFrame({"x": range(50)})
        y = pd.Series(range(50))
        fitting, held, _ = split_rows(frame, y, ["x"], False)
        implicit = additional_inner_splits(frame, y, ["x"], False)
        explicit = additional_inner_splits(frame, y, ["x"], False, canonical=(fitting, held))
        self.assertEqual([item[2] for item in implicit], [item[2] for item in explicit])
        self.assertEqual(len(additional_inner_splits(frame, y, ["x"], False, limit=1)), 1)
        self.assertEqual(additional_inner_splits(frame, y, ["x"], False, limit=0), [])

    def test_grouped_repeats_keep_conflicting_label_duplicates_together(self):
        # Each exact feature group intentionally contains both labels; IDs and
        # targets must not become grouping features that break the duplicates.
        frame = pd.DataFrame({"x": np.repeat(np.arange(40), 2), "id": range(80)})
        y = pd.Series(["a", "b"] * 40)
        choice = {"strategy": "grouped"}
        fit, held, metadata = split_rows(frame, y, ["x"], True, choice)
        checks = additional_inner_splits(frame, y, ["x"], True, choice, canonical=(fit, held))
        self.assertEqual(len(checks), 2)
        all_validation_groups = [set(frame.iloc[held]["x"])]
        for fitting, validation, repeat in checks:
            train_groups, held_groups = set(frame.iloc[fitting]["x"]), set(frame.iloc[validation]["x"])
            self.assertFalse(train_groups & held_groups)
            self.assertTrue(all(not held_groups & previous for previous in all_validation_groups))
            all_validation_groups.append(held_groups)
            self.assertEqual(repeat["n_splits"], metadata["n_splits"])
            self.assertEqual(repeat["group_overlap"], 0)
            self.assertEqual(repeat["grouping"], "raw_feature_fingerprint")
            self.assertEqual(set(y.iloc[fitting]), {"a", "b"})
            self.assertEqual(set(y.iloc[validation]), {"a", "b"})

    def test_time_repeats_roll_back_and_exclude_all_later_periods(self):
        frame = pd.DataFrame({"year": np.repeat(np.arange(1, 7), 5), "x": np.arange(30)})
        y = pd.Series(range(30))
        choice = {"strategy": "time", "time_column": "year"}
        outer, external, _ = split_rows(frame, y, ["year", "x"], False, choice)
        view, labels = frame.iloc[outer], y.iloc[outer]
        fit, held, _ = split_rows(view, labels, ["year", "x"], False, choice)
        checks = additional_inner_splits(view, labels, ["year", "x"], False, choice,
                                         canonical=(fit, held))
        self.assertEqual([float(item[2]["validation_period"]) for item in checks], [4, 3])
        for fitting, validation, repeat in checks:
            period = view.iloc[validation]["year"].unique().item()
            self.assertLess(view.iloc[fitting]["year"].max(), period)
            self.assertEqual(set(view.iloc[validation]["year"]), {period})
            self.assertFalse(set(outer[fitting]) & set(external))
            self.assertFalse(set(outer[validation]) & set(external))
            self.assertEqual(repeat["excluded_rows"], int((view["year"] > period).sum()))
            self.assertFalse(repeat["shuffle"])

    def test_time_insufficient_periods_returns_no_random_fallback(self):
        frame = pd.DataFrame({"year": [1, 1, 2, 2]})
        y = pd.Series(range(4))
        self.assertEqual(additional_inner_splits(frame, y, ["year"], False,
                                                 {"strategy": "time", "time_column": "year"}), [])

    def test_invalid_canonical_cannot_hide_missing_classes_or_boundary_crossing(self):
        frame = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3], "year": [1, 1, 2, 2, 3, 3]})
        y = pd.Series(["a", "b"] * 3)
        invalid = [([0, 2, 4], [1, 3, 5]), ([0, 1], [1, 2]), ([0, 1], [2, 6]),
                   ([0.0, 1.0], [2, 3]), ([0, 0, 1], [2, 3])]
        for canonical in invalid:
            with self.subTest(canonical=canonical), self.assertRaises(ValueError):
                additional_inner_splits(frame, y, ["x"], True, canonical=canonical)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            additional_inner_splits(frame, y, ["x"], True, {"strategy": "grouped"},
                                    canonical=([0, 3], [1, 2, 4, 5]))
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            additional_inner_splits(frame, y, ["year"], False,
                                    {"strategy": "time", "time_column": "year"},
                                    canonical=([4, 5], [0, 1]))

    def test_extra_time_classification_split_skips_period_missing_a_class(self):
        # Explicit canonical supports an already validated temporal task;
        # missing class in period 3 must skip that repeat, never alter its rows.
        frame = pd.DataFrame({"year": np.repeat(np.arange(1, 5), 2)})
        y = pd.Series(["a", "b", "a", "b", "a", "a", "a", "b"])
        checks = additional_inner_splits(frame, y, ["year"], True,
                                         {"strategy": "time", "time_column": "year"},
                                         canonical=(list(range(6)), [6, 7]))
        self.assertEqual(len(checks), 1)
        self.assertEqual(float(checks[0][2]["validation_period"]), 2)
        self.assertEqual(set(y.iloc[checks[0][0]]), {"a", "b"})
        self.assertEqual(set(y.iloc[checks[0][1]]), {"a", "b"})

    def test_split_count_is_bounded(self):
        frame = pd.DataFrame({"x": range(10)})
        for value in (-1, 3, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                additional_inner_splits(frame, pd.Series(range(10)), ["x"], False, limit=value)


class PairedStabilitySummaryTests(unittest.TestCase):
    def test_complete_rounds_change_selection_by_original_metric_mean(self):
        rounds = [{"existing": 1.0, "candidate": 0.8},
                  {"existing": 1.0, "candidate": 1.5},
                  {"existing": 1.0, "candidate": 1.6}]
        result = paired_stability_summary(["existing", "candidate"], rounds, higher_is_better=False)
        self.assertEqual(result["ranking"], ["existing", "candidate"])
        self.assertEqual(result["used_rounds"], 3)
        self.assertEqual(result["candidates"]["existing"]["std"], 0.0)
        self.assertAlmostEqual(result["candidates"]["candidate"]["mean"], 1.3)
        self.assertAlmostEqual(result["candidates"]["candidate"]["std"], np.std([0.8, 1.5, 1.6]))
        reverse = paired_stability_summary(["existing", "candidate"], rounds, higher_is_better=True)
        self.assertEqual(reverse["ranking"], ["candidate", "existing"])

    def test_partial_or_failed_round_never_rewards_fast_successful_candidate(self):
        base = {"existing": 0.6, "candidate": 0.5}
        for partial in ({"candidate": 1.0}, {"existing": None, "candidate": 1.0},
                        {"existing": float("nan"), "candidate": 1.0},
                        {"existing": float("inf"), "candidate": 1.0},
                        {"existing": True, "candidate": 1.0}, None):
            with self.subTest(partial=partial):
                result = paired_stability_summary(["existing", "candidate"],
                                                   [base, partial, {"existing": 0.0, "candidate": 1.0}],
                                                   higher_is_better=True)
                self.assertEqual(result["ranking"], ["existing", "candidate"])
                self.assertEqual(result["used_rounds"], 1)
                self.assertEqual(result["ignored_rounds"], 2)
                self.assertEqual(result["candidates"]["candidate"]["scores"], [0.5])

    def test_no_complete_rounds_preserves_frozen_order_with_no_invented_score(self):
        result = paired_stability_summary(["existing", "candidate"], [{"candidate": 1}],
                                           higher_is_better=True)
        self.assertEqual(result["ranking"], ["existing", "candidate"])
        self.assertEqual(result["used_rounds"], 0)
        self.assertIsNone(result["candidates"]["existing"]["mean"])
        self.assertIsNone(result["candidates"]["existing"]["std"])

    def test_exact_ties_preserve_existing_preference_in_both_directions(self):
        rounds = [{"existing": 0.5, "candidate": 0.75}, {"existing": 0.75, "candidate": 0.5}]
        for direction in (True, False):
            result = paired_stability_summary(["existing", "candidate"], rounds,
                                               higher_is_better=direction)
            self.assertEqual(result["ranking"], ["existing", "candidate"])

    def test_large_finite_metrics_do_not_overflow_aggregate(self):
        result = paired_stability_summary(["a", "b"],
                                           [{"a": 1e308, "b": -1e308},
                                            {"a": 1e308, "b": 1e308}],
                                           higher_is_better=False)
        self.assertEqual(result["candidates"]["a"]["mean"], 1e308)
        self.assertEqual(result["candidates"]["a"]["std"], 0.0)
        self.assertEqual(result["candidates"]["b"]["mean"], 0.0)
        self.assertEqual(result["candidates"]["b"]["std"], 1e308)

    def test_summary_rejects_unbounded_shortlist_or_rounds(self):
        for candidates in ([], ["x", "x"], ["a", "b", "c", "d"]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                paired_stability_summary(candidates, [], higher_is_better=True)
        with self.assertRaises(ValueError):
            paired_stability_summary(["a"], [{"a": 1}] * 4, higher_is_better=True)


if __name__ == "__main__":
    unittest.main()
