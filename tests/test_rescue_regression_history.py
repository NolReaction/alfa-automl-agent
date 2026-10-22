import copy
from decimal import MAX_EMAX, Decimal, Inexact, Rounded, localcontext
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.history_candidate import HistoryConfig
from agent.regression_history import (
    DEFAULT_OPTIONS, HistorySignals, SIGNAL_COLUMNS, apply_history, select_history,
)


CONFIG = HistoryConfig(("object", "number"), "period", ("number",), ("number",), "past_unique_only")


def frame(periods, objects=None, numbers=None, index=None):
    size = len(periods)
    return pd.DataFrame({"object": objects if objects is not None else ["A"] * size,
                         "number": numbers if numbers is not None else ["1"] * size,
                         "period": periods}, index=index)


class HistorySignalsTests(unittest.TestCase):
    def test_training_signals_exclude_same_period_and_all_future_targets(self):
        X = frame([1, 2, 2, 3, 4])
        first = HistorySignals(CONFIG).fit_transform(X, [10, 20, 40, 80, 90])
        changed = HistorySignals(CONFIG).fit_transform(X, [10, 2000, -4000, 8e6, 9e6])
        pd.testing.assert_frame_equal(first.iloc[:3], changed.iloc[:3])
        self.assertEqual(tuple(first), SIGNAL_COLUMNS)
        self.assertEqual(first["eligible"].tolist(), [False, True, True, False, False])
        self.assertEqual(first["ambiguous"].tolist(), [False, False, False, True, True])
        np.testing.assert_array_equal(first["past_rows"], [0, 1, 1, 3, 4])
        self.assertTrue(np.isnan(first.loc[0, "mean"]))
        self.assertEqual(first.loc[3, "mean"], 70 / 3)
        self.assertEqual(first.loc[3, "last"], 30)

    def test_numeric_gap_decay_and_latest_period_age(self):
        encoder = HistorySignals(CONFIG).fit(frame([1, 3]), [10, 30])
        result = encoder.transform(frame([3, 4, 100]))
        np.testing.assert_array_equal(result["mean"], [10, 20, 20])
        np.testing.assert_array_equal(result["last"], [10, 30, 30])
        np.testing.assert_array_equal(result["decay"], [10, 26, 26])
        np.testing.assert_array_equal(result["age"], [2, 1, 97])
        np.testing.assert_array_equal(result["past_periods"], [1, 2, 2])

    def test_ambiguity_is_prior_only_and_allow_policy_uses_latest_period_mean(self):
        X, y = frame([1, 2, 2, 4, 4]), [10, 20, 40, 100, 200]
        query = frame([2, 2, 3, 4, 5])
        strict = HistorySignals(CONFIG).fit(X, y).transform(query)
        self.assertEqual(strict["eligible"].tolist(), [True, True, False, False, False])
        allow = HistorySignals({**CONFIG.to_json(), "ambiguity_policy": "allow"}).fit(X, y).transform(query)
        self.assertTrue(allow["eligible"].all())
        np.testing.assert_array_equal(allow["last"], [10, 10, 30, 30, 150])
        np.testing.assert_array_equal(apply_history(np.zeros(5), strict, mode="last", weight=1), [10, 10, 0, 0, 0])
        np.testing.assert_array_equal(apply_history(np.zeros(5), allow, mode="last", weight=1), [10, 10, 30, 30, 150])

    def test_fit_and_query_permutations_and_single_row_batches_are_exact(self):
        X = frame([1, 1, 1, 3, 2, 2], objects=["A", "A", "A", "A", "B", "C"])
        y = np.array([1e16, 1, -1e16, 7, 21, 51.5])
        config = {**CONFIG.to_json(), "ambiguity_policy": "allow"}
        query = frame([4, 2, 3, 3, 4, 2], objects=["A", "A", "B", "C", "A", "D"], index=list("abcdef"))
        expected = HistorySignals(config).fit(X, y).transform(query)
        for permutation in ([5, 1, 2, 4, 0, 3], list(reversed(range(6)))):
            model = HistorySignals(config).fit(X.iloc[permutation], y[permutation])
            pd.testing.assert_frame_equal(model.transform(query), expected, check_exact=True)
            pd.testing.assert_frame_equal(model.transform(query.iloc[permutation]), expected.iloc[permutation], check_exact=True)
            for i in range(len(query)):
                pd.testing.assert_frame_equal(model.transform(query.iloc[[i]]), expected.iloc[[i]], check_exact=True)

    def test_missing_invalid_keys_and_periods_and_new_keys_fall_back(self):
        X = frame([1, None, "bad", 1, 1, 1, 1], objects=["A", "A", "A", "", None, "B", "C"],
                  numbers=[1, 1, 1, 1, 1, 0, "inf"])
        model = HistorySignals(CONFIG).fit(X, [10, 100, 200, 300, 400, 500, 600])
        query = frame([2, None, "NaN", "inf", 2, 2, 2, 2, 2],
                      objects=["A", "A", "A", "A", "B", "C", "D", "", "A"],
                      numbers=[1, 1, 1, 1, 0, "inf", 1, 1, None])
        signals = model.transform(query)
        self.assertEqual(signals["eligible"].tolist(), [True] + [False] * 8)
        self.assertTrue(signals.iloc[1:][["mean", "last", "decay", "age"]].isna().all().all())
        self.assertTrue((signals.iloc[1:][["past_rows", "past_periods"]] == 0).all().all())
        base = np.arange(9, dtype=float)
        expected = base.copy()
        expected[0] = 5
        np.testing.assert_array_equal(apply_history(base, signals), expected)

    def test_missing_columns_safe_fallback_and_empty_frames(self):
        model = HistorySignals(CONFIG).fit(frame([1]), [10])
        for column in CONFIG.columns:
            query = frame([2]).drop(columns=column)
            self.assertFalse(model.transform(query)["eligible"].any())
            inactive = HistorySignals(CONFIG).fit(query, [100]).transform(frame([3]))
            self.assertFalse(inactive["eligible"].any())
        empty = model.transform(frame([]))
        self.assertEqual(tuple(empty), SIGNAL_COLUMNS)
        self.assertEqual(len(empty), 0)
        self.assertFalse(HistorySignals(CONFIG).fit(frame([]), []).transform(frame([2]))["eligible"].any())

    def test_exact_numeric_keys_large_and_fractional_periods(self):
        start = "1000000000000000000000000000000.00000000000000000000000000001"
        next_period = "1000000000000000000000000000000.00000000000000000000000000002"
        X = frame([start, "9007199254740992", 1, 1], objects=["A", "B", "C", "D"],
                  numbers=["9007199254740992", "1.0", "9007199254740993", "1.00"])
        query = frame([next_period, "9007199254740993", 2, 2], objects=["A", "B", "C", "D"],
                      numbers=[Decimal("9007199254740992.0"), 1, "9007199254740992", 1])
        result = HistorySignals(CONFIG).fit(X, [11, 22, 33, 44]).transform(query)
        self.assertEqual(result["eligible"].tolist(), [True, True, False, True])
        self.assertEqual(result.loc[0, "age"], 1e-29)
        self.assertEqual(result.loc[1, "age"], 1)
        self.assertEqual(result.loc[3, "mean"], 44)

    def test_literal_text_missing_tokens_are_not_missing_and_text_is_stripped(self):
        model = HistorySignals(CONFIG).fit(frame([1, 1], objects=[" NA ", "?"]), [10, 30])
        result = model.transform(frame([2, 2], objects=["NA", " ? "]))
        np.testing.assert_array_equal(result["mean"], [10, 30])

    def test_large_numeric_gap_is_bounded_and_remains_finite(self):
        model = HistorySignals(CONFIG).fit(frame([1, "1e1000"]), [10, 30])
        result = model.transform(frame(["2e1000"]))
        self.assertEqual(result.loc[0, "decay"], 30)
        self.assertTrue(np.isfinite(result.loc[0, "age"]))

    def test_decimal_gap_scaling_and_caller_traps_do_not_round_age(self):
        with localcontext() as context:
            context.traps[Inexact] = context.traps[Rounded] = True
            model = HistorySignals(CONFIG).fit(frame(["1E-5", 1]), [10, 30])
            result = model.transform(frame([1, 2]))
            self.assertEqual(result.loc[0, "age"], 0.99999)
            self.assertEqual(result.loc[1, "age"], 1)
            factor = 2 ** -0.99999
            self.assertEqual(result.loc[1, "decay"], (10 * factor + 30) / (factor + 1))
            model = HistorySignals(CONFIG).fit(frame(["1E-3"]), [10])
            self.assertEqual(model.transform(frame(["1E3"])).loc[0, "age"], 999.999)
        model = HistorySignals(CONFIG).fit(frame([f"-9E{MAX_EMAX}"]), [10])
        result = model.transform(frame([f"9E{MAX_EMAX}"]))
        self.assertTrue(np.isfinite(result.loc[0, "age"]))

    def test_no_mutation_and_duplicate_indexes_are_positional(self):
        X = frame([1, 2, 3], index=[4, 4, 4])
        y = pd.Series([10, 20, 30], index=X.index)
        query = frame([3, 2, 3], index=[8, 8, 8])
        saved = (X.copy(deep=True), y.copy(deep=True), query.copy(deep=True))
        model = HistorySignals(CONFIG).fit(X, y)
        state = copy.deepcopy(model._past)
        first = model.transform(query)
        pd.testing.assert_frame_equal(first, model.transform(query), check_exact=True)
        self.assertEqual(model._past, state)
        pd.testing.assert_frame_equal(X, saved[0])
        pd.testing.assert_series_equal(y, saved[1])
        pd.testing.assert_frame_equal(query, saved[2])
        np.testing.assert_array_equal(first["mean"], [15, 10, 15])
        np.testing.assert_array_equal(apply_history(pd.Series([0, 0, 0], index=query.index), first), [7.5, 5, 7.5])
        model.fit(frame([1], objects=["B"]), [99])
        self.assertFalse(model.transform(frame([2]))["eligible"].any())

    def test_validation_and_caps(self):
        with self.assertRaises(RuntimeError):
            HistorySignals(CONFIG).transform(frame([2]))
        for bad in ([np.nan], [np.inf], [[1]], [], ["oops"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                HistorySignals(CONFIG).fit(frame([1]), bad)
        with self.assertRaisesRegex(ValueError, "index"):
            HistorySignals(CONFIG).fit(frame([1]), pd.Series([1], index=[999]))
        with self.assertRaisesRegex(ValueError, "unique"):
            HistorySignals(CONFIG).fit(pd.DataFrame([[1, 1]], columns=["period", "period"]), [1])
        with self.assertRaises(ValueError):
            HistorySignals(HistoryConfig(("object",), "object"))
        with patch("agent.regression_history.MAX_ROWS", 1), self.assertRaisesRegex(ValueError, "cap"):
            HistorySignals(CONFIG).fit(frame([1, 2]), [1, 2])
        with patch("agent.regression_history.MAX_KEY_CELLS", 1), self.assertRaisesRegex(ValueError, "cap"):
            HistorySignals(CONFIG).fit(frame([1]), [1])
        with self.assertRaisesRegex(ValueError, "overflow"):
            HistorySignals(CONFIG).fit(frame([1, 1]), [1e308, 1e308])


class BlendAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.signals = HistorySignals(CONFIG).fit(frame([1, 2]), [10, 30]).transform(frame([3, 2, 3], objects=["A", "A", "B"]))

    def test_shrinkage_formula_zero_weight_and_unchanged_ineligible_rows(self):
        base = np.array([100, 100, -123.25])
        saved_signals, saved_base = self.signals.copy(deep=True), base.copy()
        result = apply_history(base, self.signals, mode="last", weight=0.75, shrinkage=1)
        np.testing.assert_array_equal(result, [65, 66.25, -123.25])
        np.testing.assert_array_equal(apply_history(base, self.signals, weight=0), base)
        np.testing.assert_array_equal(base, saved_base)
        pd.testing.assert_frame_equal(self.signals, saved_signals)

    def test_blend_input_validation(self):
        for options in ({"weight": np.nan}, {"weight": -1}, {"weight": True}, {"mode": "bad"},
                        {"shrinkage": -1}, {"shrinkage": True}, {"shrinkage": 0.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                apply_history(np.zeros(3), self.signals, **options)
        for bad in ([1, 2], [[1], [2], [3]], [1, 2, np.nan]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_history(bad, self.signals)
        corrupt = self.signals.copy()
        corrupt.loc[0, "mean"] = np.inf
        with self.assertRaises(ValueError):
            apply_history(np.zeros(3), corrupt)
        corrupt = self.signals.copy()
        corrupt.loc[0, "past_rows"] = 0
        with self.assertRaises(ValueError):
            apply_history(np.zeros(3), corrupt)
        corrupt = self.signals.copy()
        corrupt["eligible"] = ["true", "false", "false"]
        with self.assertRaises(ValueError):
            apply_history(np.zeros(3), corrupt)

    def test_selection_improves_on_supplied_calibration_and_keeps_grid_small(self):
        signals = HistorySignals(CONFIG).fit(frame([1, 2]), [0, 20]).transform(frame([3]))
        result = select_history(signals, [15], [0])
        self.assertEqual({key: result[key] for key in ("mode", "weight", "shrinkage")},
                         {"mode": "last", "weight": 0.75, "shrinkage": 0})
        self.assertEqual(result["rmse"], 0)
        self.assertEqual(result["baseline_rmse"], 15)
        self.assertEqual(result["evaluated_options"], 19)
        self.assertEqual(len(DEFAULT_OPTIONS), 19)

    def test_selection_can_choose_shrinkage_and_has_deterministic_ties(self):
        signals = HistorySignals(CONFIG).fit(frame([1]), [10]).transform(frame([2]))
        selected = select_history(signals, [3.75], [0])
        self.assertEqual((selected["mode"], selected["weight"], selected["shrinkage"]), ("mean", 0.75, 1))
        options = list(DEFAULT_OPTIONS)
        self.assertEqual(select_history(signals, [3.75], [0], options[::-1]), selected)
        baseline = select_history(signals, [10], [10])
        self.assertEqual((baseline["mode"], baseline["weight"], baseline["shrinkage"]), ("mean", 0, 0))
        absent = HistorySignals(CONFIG).fit(frame([1]), [10]).transform(frame([2], objects=["new"]))
        self.assertEqual(select_history(absent, [30], [0])["weight"], 0)

    def test_selection_row_permutations_and_inputs_are_unchanged(self):
        signals = self.signals.copy()
        target, base = np.array([10, 6, 17.5]), np.array([2, 4, 9])
        original = copy.deepcopy((signals, target, base, DEFAULT_OPTIONS))
        expected = select_history(signals, target, base)
        for permutation in itertools.permutations(range(3)):
            order = list(permutation)
            self.assertEqual(select_history(signals.iloc[order], target[order], base[order]), expected)
        pd.testing.assert_frame_equal(signals, original[0])
        np.testing.assert_array_equal(target, original[1])
        np.testing.assert_array_equal(base, original[2])
        self.assertEqual(DEFAULT_OPTIONS, original[3])

    def test_selection_validation_empty_and_oversized_grid(self):
        for options in ("grid", [{}], [{"mode": "mean", "weight": 0.8, "shrinkage": 0}], list(DEFAULT_OPTIONS) * 2):
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_history(self.signals, [1, 2, 3], [1, 2, 3], options)
        with self.assertRaises(ValueError):
            select_history(self.signals.iloc[:0], [], [])
        with self.assertRaises(ValueError):
            select_history(self.signals, [1, np.nan, 3], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "index"):
            select_history(self.signals, pd.Series([1, 2, 3], index=[3, 2, 1]), [1, 2, 3])
        self.assertEqual(select_history(self.signals, [1, 2, 3], [1, 2, 3], [])["evaluated_options"], 1)
        invalid = self.signals.copy()
        invalid["eligible"] = [2, 2, 0]
        with self.assertRaisesRegex(ValueError, "booleans"):
            select_history(invalid, [1, 2, 3], [1, 2, 3], [])


if __name__ == "__main__":
    unittest.main()
