"""Behavioral checks for time order, duplicate groups and legacy holdouts."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.splits import normalize_validation, split_rows


class SplitTests(unittest.TestCase):
    def test_random_keeps_legacy_rows_for_both_task_types(self):
        train = pd.DataFrame({"id": range(50), "x": range(50)})
        for classification in (False, True):
            y = pd.Series(["a", "b"] * 25) if classification else pd.Series(range(50))
            fit, held, metadata = split_rows(train, y, ["x"], classification)
            old_fit, old_held = train_test_split(np.arange(50), test_size=0.2,
                                                random_state=42, shuffle=True,
                                                stratify=y if classification else None)
            np.testing.assert_array_equal(fit, old_fit)
            np.testing.assert_array_equal(held, old_held)
            self.assertEqual(metadata["strategy"], "random")
            self.assertEqual(metadata["stratified"], classification)

    def test_time_holds_out_latest_complete_period_in_unsorted_data(self):
        train = pd.DataFrame({"id": list("abcdef"), "period": [3, 1, 4, 2, 4, 1],
                              "x": range(6), "answer": range(6)})
        fit, held, metadata = split_rows(train, train["answer"], ["x", "period"], False,
                                         {"strategy": "time", "time_column": "period"})
        self.assertEqual(held.tolist(), [2, 4])
        self.assertLess(train["period"].iloc[fit].max(), train["period"].iloc[held].min())
        self.assertEqual(metadata["train_period_min"], "1")
        self.assertEqual(metadata["train_period_max"], "3")
        self.assertEqual(metadata["validation_period"], "4")
        self.assertAlmostEqual(metadata["validation_fraction"], 2 / 6)
        self.assertFalse(metadata["shuffle"])
        self.assertFalse(metadata["stratified"])

    def test_time_rejects_id_target_bad_periods_and_single_period(self):
        train = pd.DataFrame({"id": [1, 2, 3, 4], "period": [1, 1, 2, 2],
                              "answer": [1, 2, 3, 4]})
        for column in ("id", "answer", "missing"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                split_rows(train, train["answer"], ["period"], False,
                           {"strategy": "time", "time_column": column})
        for values in ([1, 1, 1, 1], [1, "bad", 2, 2], [1, None, 2, 2],
                       [1, float("inf"), 2, 2], [1, float("nan"), 2, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                altered = train.assign(period=values)
                split_rows(altered, altered["answer"], ["period"], False,
                           {"strategy": "time", "time_column": "period"})

    def test_grouped_keeps_conflicting_duplicate_inputs_together_and_is_reproducible(self):
        # Every feature group contains both targets and unique IDs. Neither the
        # labels nor IDs may split otherwise identical rows into separate groups.
        train = pd.DataFrame({"id": range(80), "x": np.repeat(range(20), 4),
                              "text": ["same"] * 80, "answer": ["a", "b"] * 40})
        options = {"strategy": "grouped"}
        fit, held, metadata = split_rows(train, train["answer"], ["x", "text"], True, options)
        fit_again, held_again, again = split_rows(train, train["answer"], ["x", "text"], True, options)
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(held, held_again)
        self.assertEqual(metadata, again)
        self.assertFalse(set(train["x"].iloc[fit]) & set(train["x"].iloc[held]))
        self.assertEqual(set(train["answer"].iloc[fit]), {"a", "b"})
        self.assertEqual(set(train["answer"].iloc[held]), {"a", "b"})
        self.assertEqual(len(fit) + len(held), len(train))
        self.assertEqual(metadata["group_count"], 20)
        self.assertEqual(metadata["group_overlap"], 0)
        self.assertEqual(metadata["validation_fraction"], len(held) / len(train))
        self.assertEqual(metadata["train_groups"] + metadata["validation_groups"], 20)

    def test_grouped_rejects_class_confined_to_one_group(self):
        train = pd.DataFrame({"x": [0, 0, 1, 1, 2, 2], "answer": ["rare", "rare", "a", "a", "a", "a"]})
        with self.assertRaisesRegex(ValueError, "одной группе"):
            split_rows(train, train["answer"], ["x"], True, {"strategy": "grouped"})

    def test_normalization_and_task_types_reject_unsupported_choices(self):
        self.assertEqual(normalize_validation(), {"strategy": "random"})
        invalid = [[], "time", {}, {"strategy": []}, {"strategy": "other"},
                   {"strategy": "random", "time_column": "x"}, {"strategy": "time"},
                   {"strategy": "time", "time_column": " "},
                   {"strategy": "time", "time_column": "x" * 501},
                   {"strategy": "grouped", "seed": 0}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_validation(value)
        train = pd.DataFrame({"x": range(20)})
        y = pd.Series([0, 1] * 10)
        with self.assertRaises(ValueError):
            split_rows(train, y, ["x"], True, {"strategy": "time", "time_column": "x"})
        with self.assertRaises(ValueError):
            split_rows(train, y, ["x"], False, {"strategy": "grouped"})


if __name__ == "__main__":
    unittest.main()
