import copy
from decimal import Decimal, Inexact, ROUND_FLOOR, ROUND_UP, Rounded, localcontext
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from agent.building_history import FEATURES, HistoryEncoder, period_json
import experiment_regression_history as experiment

HISTORY = list(FEATURES)


KEY = ["state", "kind", "area", "built", "elevation"]
NUMERIC = ["area", "built", "elevation"]
POSITIVE = ["built"]


def encoder() -> HistoryEncoder:
    return HistoryEncoder(KEY, "year", numeric_columns=NUMERIC, positive_columns=POSITIVE)


def rows(*items) -> pd.DataFrame:
    """items: (year, area) or (year, area, extra dict); every row has the same other key parts."""
    data = []
    for item in items:
        year, area, *rest = item
        row = {"state": "S1", "kind": "office", "area": area, "built": 1990.0, "elevation": 2.5, "year": year}
        row.update(rest[0] if rest else {})
        data.append(row)
    return pd.DataFrame(data)


def one(frame: pd.DataFrame, position: int) -> dict:
    return frame.iloc[position].to_dict()


class PeriodJsonTests(unittest.TestCase):
    def test_long_fractional_periods_keep_exact_plain_decimal_text(self):
        cases = (
            ("1.123456789012345678901234567890123000", "1.123456789012345678901234567890123"),
            ("-1.123456789012345678901234567890123000", "-1.123456789012345678901234567890123"),
            ("123456789012345678901234567890.000000000000000000000000000123000",
             "123456789012345678901234567890.000000000000000000000000000123"),
            ("-123456789012345678901234567890.000000000000000000000000000123000",
             "-123456789012345678901234567890.000000000000000000000000000123"),
            ("1E-40", "0." + "0" * 39 + "1"),
            ("-1E-40", "-0." + "0" * 39 + "1"),
        )
        for original, expected in cases:
            with self.subTest(period=original):
                encoded = period_json(Decimal(original))
                self.assertIs(type(encoded), str)
                self.assertEqual(encoded, expected)
                self.assertEqual(Decimal(json.loads(json.dumps(encoded))), Decimal(original))

    def test_integral_periods_remain_json_integers_without_size_loss(self):
        cases = ((Decimal("5.000"), 5), (Decimal("5E60"), 5 * 10 ** 60),
                 (Decimal("-5E60"), -5 * 10 ** 60), (Decimal("-0.000"), 0),
                 (10 ** 80 + 1, 10 ** 80 + 1))
        for original, expected in cases:
            with self.subTest(period=original):
                encoded = period_json(original)
                self.assertIs(type(encoded), int)
                self.assertEqual(encoded, expected)
                self.assertEqual(json.loads(json.dumps(encoded)), expected)

    def test_serialization_ignores_decimal_precision_rounding_and_exponent_limits(self):
        cases = (
            (Decimal("123456789012345678901234567890.12500"), "123456789012345678901234567890.125"),
            (Decimal("-123456789012345678901234567890.12500"), "-123456789012345678901234567890.125"),
            (Decimal("1E-40"), "0." + "0" * 39 + "1"),
            (Decimal("5E60"), 5 * 10 ** 60),
        )
        for precision in (1, 2, 5):
            for rounding in (ROUND_FLOOR, ROUND_UP):
                with self.subTest(precision=precision, rounding=rounding), localcontext() as context:
                    context.prec, context.rounding = precision, rounding
                    context.Emin, context.Emax = -9, 9
                    context.traps[Inexact] = context.traps[Rounded] = True
                    context.clear_flags()
                    for value, expected in cases:
                        self.assertEqual(period_json(value), expected)
                    self.assertFalse(any(context.flags.values()), "Serialization must not signal rounding")

    def test_nearby_fractional_periods_stay_distinct_after_json_round_trip(self):
        values = [Decimal("1.123456789012345678901234567890001"),
                  Decimal("1.123456789012345678901234567890002")]
        encoded = json.loads(json.dumps([period_json(value) for value in values]))
        self.assertNotEqual(encoded[0], encoded[1])
        self.assertEqual([Decimal(value) for value in encoded], values)


class FeatureValueTests(unittest.TestCase):
    def test_row_does_not_see_its_own_answer(self):
        X, y = rows((1, 100.0)), [10.0]

        result = encoder().fit_transform(X, y)

        self.assertTrue(math.isnan(result.loc[0, "history_mean"]))
        self.assertEqual(result.loc[0, ["history_rows", "history_periods", "history_present"]].tolist(), [0, 0, 0])

    def test_rows_of_one_period_do_not_use_each_other(self):
        X, y = rows((2, 100.0), (2, 100.0), (2, 100.0)), [10.0, 20.0, 30.0]

        result = encoder().fit_transform(X, y)

        self.assertTrue(result["history_mean"].isna().all())
        self.assertEqual(result["history_present"].tolist(), [0, 0, 0])

    def test_only_strictly_earlier_periods_are_used_also_for_fitting_rows(self):
        X, y = rows((1, 100.0), (2, 100.0), (3, 100.0)), [10.0, 20.0, 40.0]

        result = encoder().fit_transform(X, y)

        self.assertTrue(math.isnan(result.loc[0, "history_mean"]))
        self.assertEqual(result.loc[1, "history_mean"], 10.0)
        self.assertEqual(result.loc[2, "history_mean"], 15.0)  # not 23.33: the own answer 40 is excluded
        self.assertEqual(result["history_rows"].tolist(), [0, 1, 2])
        self.assertEqual(result["history_periods"].tolist(), [0, 1, 2])
        self.assertEqual(result["history_present"].tolist(), [0, 1, 1])

    def test_future_answers_do_not_change_earlier_features(self):
        X = rows((1, 100.0), (2, 100.0), (3, 100.0), (4, 100.0))
        first = encoder().fit_transform(X, [10.0, 20.0, 30.0, 40.0])

        second = encoder().fit_transform(X, [10.0, 20.0, 999.0, -999.0])

        pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])  # rows of periods 1-3 never see periods 3-4's answers
        self.assertNotEqual(first.loc[3, "history_mean"], second.loc[3, "history_mean"])

    def test_several_past_periods_and_repeated_key_inside_a_period(self):
        X = rows((1, 100.0), (1, 100.0), (2, 100.0), (3, 100.0), (4, 100.0))
        y = [10.0, 30.0, 50.0, 70.0, 0.0]

        result = encoder().fit_transform(X, y)

        self.assertEqual(result["history_rows"].tolist(), [0, 0, 2, 3, 4])
        self.assertEqual(result["history_periods"].tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(result.loc[2, "history_mean"], 20.0)
        self.assertEqual(result.loc[4, "history_mean"], (10 + 30 + 50 + 70) / 4)
        self.assertEqual(result["history_ambiguous"].tolist(), [0, 0, 1, 1, 1])

    def test_ambiguity_counts_only_available_past_history(self):
        # the key repeats in period 3, but a row of period 2 must not know it
        X = rows((1, 100.0), (2, 100.0), (3, 100.0), (3, 100.0), (4, 100.0))

        result = encoder().fit_transform(X, [1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result["history_ambiguous"].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result.loc[4, "history_rows"], 4)

    def test_no_history_gives_nan_mean_and_zero_counts(self):
        fitted = encoder().fit(rows((1, 100.0)), [10.0])

        result = fitted.transform(rows((2, 555.0)))  # unknown key

        self.assertTrue(math.isnan(result.loc[0, "history_mean"]))
        self.assertEqual(result.loc[0, [c for c in FEATURES if c != "history_mean"]].tolist(), [0, 0, 0, 0])

    def test_new_rows_use_all_fitted_periods_before_theirs(self):
        fitted = encoder().fit(rows((1, 100.0), (2, 100.0), (3, 100.0)), [10.0, 20.0, 60.0])

        result = fitted.transform(rows((2, 100.0), (4, 100.0), (1, 100.0)))

        self.assertEqual(result["history_mean"].tolist()[:2], [10.0, 30.0])
        self.assertTrue(math.isnan(result.loc[2, "history_mean"]))

    def test_missing_history_of_fitting_rows_is_not_filled_with_the_mean_target(self):
        result = encoder().fit_transform(rows((1, 1.0), (2, 2.0)), [5.0, 7.0])

        self.assertTrue(math.isnan(result.loc[0, "history_mean"]))  # not 6.0


class KeyTests(unittest.TestCase):
    def test_large_integer_and_decimal_keys_keep_their_exact_identity(self):
        for first, second in ((2**53, 2**53 + 1), (str(2**53), str(2**53 + 1)),
                              ("61242.123456789012345", "61242.123456789012346"),
                              (Decimal("1.000000000000000000001"), Decimal("1.000000000000000000002"))):
            with self.subTest(first=first, second=second):
                X = rows((1, first), (2, second), (3, first))
                result = encoder().fit_transform(X, [10.0, 20.0, 30.0])
                self.assertEqual(result["history_present"].tolist(), [0, 0, 1])
                self.assertEqual(result.loc[2, "history_mean"], 10.0)

    def test_equivalent_exact_numeric_spellings_match(self):
        X = rows((1, "9007199254740993"), (2, "9007199254740993.000"), (3, Decimal("9.007199254740993E15")))
        result = encoder().fit_transform(X, [10.0, 20.0, 30.0])
        self.assertEqual(result["history_rows"].tolist(), [0, 1, 2])
        self.assertEqual(result.loc[2, "history_mean"], 15.0)

    def test_missing_key_part_never_matches(self):
        for column, value in (("state", None), ("kind", ""), ("kind", "  "), ("area", np.nan), ("elevation", np.nan)):
            with self.subTest(column=column, value=value):
                X = rows((1, 100.0, {column: value}), (2, 100.0, {column: value}))

                result = encoder().fit_transform(X, [10.0, 20.0])

                self.assertEqual(result["history_present"].tolist(), [0, 0])

    def test_unknown_or_not_positive_built_means_no_reliable_key(self):
        for value in (np.nan, 0.0, -1.0, "", "unknown"):
            with self.subTest(value=value):
                X = rows((1, 100.0, {"built": value}), (2, 100.0, {"built": value}))

                self.assertEqual(encoder().fit_transform(X, [10.0, 20.0])["history_present"].tolist(), [0, 0])

    def test_zero_elevation_is_a_valid_key_part(self):
        X = rows((1, 100.0, {"elevation": 0.0}), (2, 100.0, {"elevation": 0.0}))

        self.assertEqual(encoder().fit_transform(X, [10.0, 20.0])["history_present"].tolist(), [0, 1])

    def test_text_and_numbers_are_compared_as_values(self):
        X = rows((1, "100", {"state": " S1 "}), (2, 100.0, {"built": "1990"}))

        self.assertEqual(encoder().fit_transform(X, [10.0, 20.0])["history_present"].tolist(), [0, 1])

    def test_numbers_that_differ_slightly_are_not_merged_by_rounding(self):
        base = 61242.123456789012
        for other in (base + 1e-9, base * (1 + 1e-12), 61242.123456789):
            with self.subTest(other=other):
                X = rows((1, base), (2, other))
                self.assertNotEqual(base, other)

                result = encoder().fit_transform(X, [10.0, 20.0])

                self.assertEqual(result["history_present"].tolist(), [0, 0])
        X = rows((1, base), (2, base))
        self.assertEqual(encoder().fit_transform(X, [10.0, 20.0])["history_present"].tolist(), [0, 1])

    def test_text_with_separator_characters_does_not_create_false_matches(self):
        X = pd.DataFrame({"state": ["a|b", "a"], "kind": ["c", "b|c"], "area": [1.0, 1.0], "built": [1990.0, 1990.0],
                          "elevation": [0.0, 0.0], "year": [1, 2]})

        self.assertEqual(encoder().fit_transform(X, [1.0, 2.0])["history_present"].tolist(), [0, 0])


class ContractTests(unittest.TestCase):
    def test_large_and_close_decimal_periods_remain_strictly_ordered(self):
        for periods in ((2**53, 2**53 + 1, 2**53 + 2),
                        ("9007199254740992", "9007199254740993", "9007199254740994"),
                        ("1.000000000000000000001", "1.000000000000000000002", "1.000000000000000000003")):
            with self.subTest(periods=periods):
                X = rows(*[(period, 100.0) for period in periods])
                result = encoder().fit_transform(X, [10.0, 20.0, 999.0])
                self.assertEqual(result["history_rows"].tolist(), [0, 1, 2])
                self.assertEqual(result.loc[1:, "history_mean"].tolist(), [10.0, 15.0])
                future_changed = encoder().fit_transform(X, [10.0, 20.0, -999.0])
                pd.testing.assert_frame_equal(result, future_changed)

    def test_transform_needs_no_answers_and_does_not_change_state(self):
        fitted = encoder().fit(rows((1, 100.0), (2, 100.0)), [10.0, 20.0])
        state = copy.deepcopy(fitted._past)
        batch = rows((3, 100.0), (3, 100.0), (5, 7.0))

        fitted.transform(batch)

        self.assertEqual(fitted._past, state)

    def test_repeated_transform_gives_the_same_result(self):
        fitted = encoder().fit(rows((1, 100.0), (2, 100.0), (2, 100.0)), [10.0, 20.0, 40.0])
        batch = rows((3, 100.0), (4, 100.0), (3, 9.0))

        pd.testing.assert_frame_equal(fitted.transform(batch), fitted.transform(batch))

    def test_features_of_a_row_do_not_depend_on_other_rows_of_the_batch(self):
        fitted = encoder().fit(rows((1, 100.0), (2, 100.0)), [10.0, 20.0])
        batch = rows((3, 100.0), (3, 100.0), (4, 100.0), (3, 5.0))

        together = fitted.transform(batch)

        for position in range(len(batch)):
            alone = fitted.transform(batch.iloc[[position]])
            pd.testing.assert_frame_equal(together.iloc[[position]], alone)

    def test_shuffled_rows_keep_their_features_by_index(self):
        X = rows((1, 100.0), (2, 100.0), (3, 100.0), (3, 7.0), (4, 100.0))
        y = pd.Series([10.0, 20.0, 40.0, 5.0, 1.0], index=X.index)
        ordered = encoder().fit_transform(X, y)

        order = [3, 0, 4, 2, 1]
        shuffled = encoder().fit_transform(X.iloc[order], y.iloc[order])

        self.assertEqual(shuffled.index.tolist(), order)
        pd.testing.assert_frame_equal(shuffled.sort_index(), ordered)

    def test_result_has_only_history_columns_the_index_and_order_of_x_and_inputs_are_unchanged(self):
        X = rows((2, 100.0), (1, 100.0), (3, 100.0)).set_index(pd.Index(["c", "a", "b"]))
        y = pd.Series([20.0, 10.0, 30.0], index=X.index)
        X_before, y_before = X.copy(deep=True), y.copy(deep=True)

        result = encoder().fit_transform(X, y)
        transformed = encoder().fit(X, y).transform(X)

        self.assertEqual(list(result.columns), list(FEATURES))
        self.assertEqual(result.index.tolist(), ["c", "a", "b"])
        self.assertEqual(result["history_mean"].tolist()[0], 10.0)  # row "c" is period 2
        pd.testing.assert_frame_equal(result, transformed)
        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)

    def test_refitting_forgets_the_old_history(self):
        model = encoder().fit(rows((1, 100.0)), [10.0])

        model.fit(rows((1, 200.0)), [99.0])

        self.assertEqual(model.transform(rows((2, 100.0)))["history_present"].tolist(), [0])
        self.assertEqual(model.transform(rows((2, 200.0)))["history_mean"].tolist(), [99.0])

    def test_errors_are_clear(self):
        X = rows((1, 100.0), (2, 100.0))
        with self.assertRaisesRegex(ValueError, "2 rows but y has 1"):
            encoder().fit(X, [1.0])
        with self.assertRaisesRegex(ValueError, "index of y differs"):
            encoder().fit(X, pd.Series([1.0, 2.0], index=[5, 6]))
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder().fit(X, [1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "numbers"):
            encoder().fit(X, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "Missing columns: \\['year'\\]"):
            encoder().fit(X.drop(columns="year"), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Period column 'year' has 1"):
            encoder().fit(rows((1, 100.0), (np.nan, 100.0)), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Period column 'year'"):
            encoder().fit(rows((1, 100.0), ("later", 100.0)), [1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            encoder().transform(X)
        with self.assertRaisesRegex(ValueError, "Period column"):
            encoder().fit(X, [1.0, 2.0]).transform(rows((np.inf, 100.0)))
        with self.assertRaisesRegex(TypeError, "DataFrame"):
            encoder().fit(X.to_numpy(), [1.0, 2.0])

    def test_constructor_rejects_inconsistent_columns(self):
        for kwargs in (dict(key_columns=[], period_column="year"), dict(key_columns=["a", "a"], period_column="year"),
                       dict(key_columns=["a", "year"], period_column="year"),
                       dict(key_columns=["a"], period_column="year", numeric_columns=["b"]),
                       dict(key_columns=["a"], period_column="year", numeric_columns=["a"], positive_columns=["b"])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HistoryEncoder(**kwargs)


class RunningSumTests(unittest.TestCase):
    @staticmethod
    def long_history(periods: int, values) -> tuple[pd.DataFrame, list]:
        return rows(*[(period, 100.0) for period in range(1, periods + 1)]), list(values)

    def test_long_history_of_one_key_needs_linear_summation_work(self):
        periods = 2000
        X, y = self.long_history(periods, np.linspace(1.0, 2.0, periods))
        summed = []
        real_fsum = math.fsum

        def counting_fsum(values):
            values = list(values)
            summed.append(len(values))
            return real_fsum(values)

        with mock.patch("agent.building_history.math.fsum", counting_fsum):
            encoder().fit(X, y)

        self.assertLess(sum(summed), 10 * periods)  # the old prefix fsum added periods * (periods + 1) / 2 numbers

    def test_running_sum_equals_fsum_of_all_earlier_period_totals_bit_for_bit(self):
        rng = np.random.default_rng(3)
        values = rng.choice([1e16, -1e16, 1.0, 1e-8, 3.3, -7.25e15], size=400) * rng.uniform(0.5, 1.5, 400)
        X, y = self.long_history(400, values)

        result = encoder().fit_transform(X, y)

        expected = [math.nan] + [math.fsum(values[:i]) / i for i in range(1, 400)]
        np.testing.assert_array_equal(result["history_mean"].to_numpy(), np.asarray(expected))
        naive = np.cumsum(values)[:-1] / np.arange(1, 400)
        self.assertGreater(np.max(np.abs(naive - np.asarray(expected[1:]))), 0)  # plain float summation drifts here

    def test_sums_that_overflow_are_a_clear_error_and_never_infinite_features(self):
        for y in ([1.5e308, 1.5e308], [-1.7e308, -1.7e308]):
            with self.subTest(y=y), self.assertRaisesRegex(ValueError, "overflow"):
                encoder().fit(rows((1, 100.0), (2, 100.0)), y)
        with self.assertRaisesRegex(ValueError, "overflow"):  # two rows of one period overflow as well
            encoder().fit(rows((1, 100.0), (1, 100.0)), [1.5e308, 1.5e308])
        result = encoder().fit_transform(rows((1, 100.0), (2, 100.0), (3, 100.0)), [1.5e308, -1.5e308, 0.0])
        self.assertTrue(np.isfinite(result["history_mean"].iloc[1:]).all())  # large finite values that cancel are fine


class SyntheticCycleTests(unittest.TestCase):
    def test_encoder_to_model_to_validation_with_known_history(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(0)
        n = 150
        buildings = pd.DataFrame({"state": rng.choice(["S1", "S2"], n), "kind": rng.choice(["office", "school"], n),
                                  "area": 1000.0 + 13.0 * np.arange(n), "built": 1950.0 + np.arange(n) % 60,
                                  "elevation": (np.arange(n) % 5) * 1.5, "level": rng.uniform(10, 200, n)})
        frames = []
        for year in range(1, 6):
            part = buildings[rng.random(n) < 0.8].copy()
            part["year"] = year
            part["noise"] = rng.normal(size=len(part))
            part["y"] = part["level"] + rng.normal(scale=2.0, size=len(part))
            frames.append(part)
        data = pd.concat(frames, ignore_index=True)
        fit, valid = data[data["year"] < 5], data[data["year"] == 5]
        X_fit, X_valid = fit.drop(columns=["y", "level"]), valid.drop(columns=["y", "level"])
        history_fit = encoder().fit_transform(X_fit, fit["y"])
        history_valid = encoder().fit(X_fit, fit["y"]).transform(X_valid)
        columns = ["noise", "area", "built", "elevation"]

        expected = fit.groupby(["area"])["y"].mean().reindex(valid["area"]).to_numpy()  # area identifies a building here
        np.testing.assert_allclose(history_valid["history_mean"].to_numpy(), expected, equal_nan=True)
        self.assertTrue((history_fit.loc[fit["year"] == 1, "history_present"] == 0).all())  # first period has no past

        with threadpool_limits(limits=experiment.THREADS):
            plain = HistGradientBoostingRegressor(random_state=0).fit(X_fit[columns], fit["y"])
            with_history = HistGradientBoostingRegressor(random_state=0).fit(
                pd.concat([X_fit[columns], history_fit], axis=1), fit["y"])
            error_plain = np.sqrt(np.mean((plain.predict(X_valid[columns]) - valid["y"]) ** 2))
            error_history = np.sqrt(np.mean((with_history.predict(pd.concat([X_valid[columns], history_valid], axis=1))
                                             - valid["y"]) ** 2))

        self.assertGreater(history_valid["history_present"].mean(), 0.9)
        self.assertLess(error_history, error_plain / 2)  # the planted building level is learned only through history


def make_task(root: Path) -> Path:
    """A small regression task in the layout offline_baseline reads: train years 1-5, test year 6, text ids with zeros."""
    rng = np.random.default_rng(1)
    n = 120
    buildings = pd.DataFrame({
        "State_Factor": rng.choice(["State_1", "State_2"], n), "building_class": rng.choice(["Commercial", "Residential"], n),
        "facility_type": rng.choice(["Office", "Retail", "School"], n), "floor_area": 5000.0 + 91.5 * np.arange(n),
        "year_built": 1940.0 + np.arange(n) % 70, "ELEVATION": (np.arange(n) % 6) * 1.5, "level": rng.uniform(20, 200, n)})
    frames = []
    for year in range(1, 7):
        part = buildings[rng.random(n) < 0.7].copy()
        part["Year_Factor"] = float(year)
        part["temp"] = rng.normal(size=len(part))
        part["site_eui"] = part["level"] + rng.normal(scale=3.0, size=len(part))
        frames.append(part.drop(columns="level"))
    everything = pd.concat(frames, ignore_index=True)
    everything.insert(0, "id", [f"{i:05d}" for i in range(len(everything))])
    train, test = everything[everything["Year_Factor"] < 6], everything[everything["Year_Factor"] == 6]
    inputs = Path(root) / "regression" / "participants"
    inputs.mkdir(parents=True)
    train.to_csv(inputs / "train.csv", index=False)
    test.drop(columns="site_eui").to_csv(inputs / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "site_eui": 0.0}).to_csv(inputs / "sample_submission.csv", index=False)
    roles = [("id", "id", "category"), ("site_eui", "target", "numeric"), ("Year_Factor", "feature", "numeric"),
             ("State_Factor", "feature", "category"), ("building_class", "feature", "category"),
             ("facility_type", "feature", "category"), ("floor_area", "feature", "numeric"),
             ("year_built", "feature", "numeric"), ("ELEVATION", "feature", "numeric"), ("temp", "feature", "numeric")]
    pd.DataFrame(roles, columns=["feature", "role", "dtype"]).to_csv(inputs / "data_dictionary.csv", index=False)
    return inputs


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.inputs = make_task(Path(self.directory.name))

    def run_experiment(self, name: str = "out", years=(3, 4, 5)) -> tuple[dict, Path]:
        out = Path(self.directory.name) / name
        return experiment.run_experiment(self.inputs, out, years=years), out

    @staticmethod
    def predictions(out: Path, year: int) -> pd.DataFrame:
        return pd.read_csv(out / f"predictions_year{year}.csv", dtype={"id": str})

    def test_fit_and_predict_limit_previously_initialized_native_pools(self):
        observed = []

        class CheckingModel:
            def fit(self, X, y):
                observed.append(("fit", threadpool_info()))
                return self

            def predict(self, X):
                observed.append(("predict", threadpool_info()))
                return np.zeros(len(X))

            def __getitem__(self, index):
                return self

            def get_params(self):
                return {}

        # A runtime setting above the cap simulates imports made before offline_baseline sets its environment.
        with threadpool_limits(limits=12), mock.patch.object(experiment, "make_model", return_value=CheckingModel()):
            experiment.run_year(5, *experiment.load(self.inputs))
        self.assertEqual([stage for stage, _ in observed], ["fit", "predict"] * len(experiment.VARIANTS))
        for stage, pools in observed:
            self.assertTrue(pools, stage)
            self.assertTrue(all(pool["num_threads"] <= experiment.THREADS for pool in pools), (stage, pools))

    def test_csv_precision_survives_load_and_history_encoding_before_model_preparation(self):
        train = pd.read_csv(self.inputs / "train.csv", dtype=str, keep_default_na=False)
        train.loc[1, list(experiment.KEY_COLUMNS)] = train.loc[0, list(experiment.KEY_COLUMNS)].to_numpy()
        cases = ((["9007199254740992.1", "9007199254740992.2"], ["1", "2"], 0),
                 (["9007199254740993", "9007199254740993.000"],
                  ["9007199254740992", "9007199254740993"], 1))
        for areas, periods, expected in cases:
            with self.subTest(areas=areas, periods=periods):
                train.loc[[0, 1], "floor_area"] = areas
                train.loc[[0, 1], experiment.PERIOD] = periods
                train.to_csv(self.inputs / "train.csv", index=False)
                x, y, ids, features, numeric, categorical = experiment.load(self.inputs)
                self.assertEqual(x.iloc[:2]["floor_area"].tolist(), areas)
                self.assertEqual(x.iloc[:2][experiment.PERIOD].tolist(), periods)
                with mock.patch.object(experiment, "VARIANTS", ()), mock.patch.object(experiment, "BLENDS", ()):
                    _, _, table = experiment.run_year(int(periods[1]), x, y, ids, features, numeric, categorical)
                row = table.loc[table["id"] == ids.iloc[1]].iloc[0]
                self.assertEqual(row["history_present"], expected)
                if expected:
                    self.assertEqual(row["history_mean"], y[0])

    def test_matrix_files_metrics_and_independent_recomputation(self):
        summary, out = self.run_experiment()

        self.assertEqual(sorted(path.name for path in out.iterdir()), [
            "input_hashes.json", "metrics.csv", "metrics.json", "predictions_year3.csv", "predictions_year4.csv",
            "predictions_year5.csv"])
        names = [name for name, *_ in experiment.VARIANTS] + [name for name, _ in experiment.BLENDS]
        self.assertEqual(len(names), 10)
        for year in (3, 4, 5):
            table = self.predictions(out, year)
            train = pd.read_csv(self.inputs / "train.csv", dtype={"id": str})
            expected_ids = train.loc[train["Year_Factor"] == year, "id"].tolist()
            self.assertEqual(table["id"].tolist(), expected_ids)  # text ids, leading zeros kept, same rows for all variants
            self.assertTrue(table["id"].str.startswith("0").all())
            records = {r["variant"]: r for r in summary["records"] if r["year"] == year}
            self.assertEqual(sorted(records), sorted(names))
            for name in names:
                error = table["y_true"] - table[name]
                self.assertAlmostEqual(math.sqrt((error ** 2).mean()), records[name]["errors"]["all"]["rmse"], places=9)
                has = table["history_present"] == 1
                self.assertAlmostEqual(math.sqrt((error[has] ** 2).mean()), records[name]["errors"]["history"]["rmse"], places=9)
            for name, source in experiment.BLENDS:
                expected = np.where(table["history_present"] == 1, 0.5 * table[source] + 0.5 * table["history_mean"].fillna(0),
                                    table[source])
                np.testing.assert_allclose(table[name], expected)
                rows_without = table["history_present"] == 0
                pd.testing.assert_series_equal(table.loc[rows_without, name], table.loc[rows_without, source], check_names=False)
            fold = next(f for f in summary["folds"] if f["year"] == year)
            self.assertEqual(fold["fit_rows"], int((train["Year_Factor"] < year).sum()))
            self.assertEqual(fold["validation_rows"], len(expected_ids))
        combined = {c["variant"]: c["all"]["rmse"] for c in summary["combined"]}
        for name in names:
            sse = sum(((self.predictions(out, y)["y_true"] - self.predictions(out, y)[name]) ** 2).sum() for y in (3, 4, 5))
            rows = sum(len(self.predictions(out, y)) for y in (3, 4, 5))
            self.assertAlmostEqual(combined[name], math.sqrt(sse / rows), places=9)  # sqrt(sum SSE / sum N)
        yearly = [next(r for r in summary["records"] if r["variant"] == "et_all" and r["year"] == y)["errors"]["all"]["rmse"]
                  for y in (3, 4, 5)]
        self.assertNotAlmostEqual(combined["et_all"], sum(yearly) / 3, places=6)  # not a mean of yearly RMSE

    def test_fixed_models_features_and_history_columns(self):
        summary, _ = self.run_experiment(years=(4,))
        records = {r["variant"]: r for r in summary["records"]}

        hgb, extra = records["hgb_all"]["parameters"], records["et_all"]["parameters"]
        self.assertEqual((hgb["max_iter"], hgb["max_leaf_nodes"], hgb["learning_rate"], hgb["random_state"]), (150, 31, 0.1, 42))
        self.assertEqual((extra["n_estimators"], extra["min_samples_leaf"], extra["max_depth"], extra["max_features"],
                          extra["random_state"], extra["n_jobs"]), (200, 1, None, 1.0, 42, 8))
        self.assertIn("Year_Factor", records["hgb_all"]["features"])
        self.assertNotIn("Year_Factor", records["hgb_no_period"]["features"])
        self.assertEqual(records["hgb_no_period_history"]["features"], [*records["hgb_no_period"]["features"], *HISTORY])
        self.assertEqual(records["et_no_period_history"]["features"], [*records["et_no_period"]["features"], *HISTORY])
        self.assertFalse(set(HISTORY) & set(records["et_all"]["features"]))

    def test_changing_validation_answers_changes_no_prediction_and_no_history(self):
        _, first = self.run_experiment("first")
        train = pd.read_csv(self.inputs / "train.csv", dtype=str, keep_default_na=False)
        late = train["Year_Factor"].astype(float) == 5
        train.loc[late, "site_eui"] = (train.loc[late, "site_eui"].astype(float) * 3 + 777).astype(str)
        train.to_csv(self.inputs / "train.csv", index=False)

        _, second = self.run_experiment("second")

        a, b = self.predictions(first, 5), self.predictions(second, 5)
        self.assertFalse(np.allclose(a["y_true"], b["y_true"]))
        pd.testing.assert_frame_equal(a.drop(columns="y_true"), b.drop(columns="y_true"))  # models and history never saw them
        for year in (3, 4):
            pd.testing.assert_frame_equal(self.predictions(first, year), self.predictions(second, year))

    def test_every_fold_is_fitted_on_strictly_earlier_years_only(self):
        summary, _ = self.run_experiment(years=(3, 5))

        self.assertEqual([fold["fit_years"] for fold in summary["folds"]], [[1, 2], [1, 2, 3, 4]])

    def test_input_files_changed_during_the_run_are_detected(self):
        original = experiment.run_year

        def tamper(*args, **kwargs):
            with open(self.inputs / "sample_submission.csv", "ab") as file:
                file.write(b"x")
            return original(*args, **kwargs)

        with mock.patch.object(experiment, "run_year", tamper), self.assertRaisesRegex(RuntimeError, "Input files changed"):
            experiment.run_experiment(self.inputs, Path(self.directory.name) / "out", years=(5,))

        self.assertFalse((Path(self.directory.name) / "out").exists())

    def test_output_folder_inside_the_input_folder_is_refused(self):
        before = {path.name: path.read_bytes() for path in self.inputs.iterdir()}

        for target in (self.inputs, self.inputs / "results"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "inside the input folder"):
                experiment.run_experiment(self.inputs, target)

        self.assertEqual({path.name: path.read_bytes() for path in self.inputs.iterdir()}, before)

    def test_input_symlink_to_a_planned_prediction_output_is_rejected_before_loading(self):
        out = Path(self.directory.name) / "out"
        out.mkdir()
        original = self.inputs / "train.csv"
        before = original.read_bytes()
        target = out / "predictions_year5.csv"
        target.write_bytes(before)
        original.unlink()
        try:
            original.symlink_to(target)
        except (OSError, NotImplementedError):
            original.write_bytes(before)
            self.skipTest("Symbolic links are unavailable")
        with mock.patch.object(experiment, "load") as load, mock.patch.object(experiment, "run_year") as fit:
            with self.assertRaisesRegex(ValueError, "alias an input"):
                experiment.run_experiment(self.inputs, out, years=(5,))
            load.assert_not_called()
            fit.assert_not_called()
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(target.read_bytes(), before)

    def test_all_planned_report_and_prediction_hardlinks_are_rejected_before_loading(self):
        out = Path(self.directory.name) / "out"
        out.mkdir()
        source = self.inputs / "sample_submission.csv"
        before = source.read_bytes()
        for name in ("metrics.json", "metrics.csv", "input_hashes.json", "predictions_year5.csv"):
            target = out / name
            with self.subTest(name=name):
                os.link(source, target)
                with mock.patch.object(experiment, "load") as load, self.assertRaisesRegex(ValueError, "alias an input"):
                    experiment.run_experiment(self.inputs, out, years=(5,))
                load.assert_not_called()
                self.assertEqual(source.read_bytes(), before)
                target.unlink()

    def test_input_mutation_during_output_writing_is_detected(self):
        errors = {name: {"rows": 1, "sse": 1.0, "rmse": 1.0} for name in experiment.SUBSETS}
        record = {"variant": "hgb_all", "kind": "model", "year": 5, "fit_rows": 1, "validation_rows": 1,
                  "errors": errors}
        real_write = experiment.write_file

        def tamper_after_output(path, write):
            real_write(path, write)
            if path.name == "input_hashes.json":
                with (self.inputs / "sample_submission.csv").open("ab") as stream:
                    stream.write(b"changed after outputs")

        with mock.patch.object(experiment, "run_year", return_value=({}, [record], pd.DataFrame({"prediction": [1.0]}))), \
                mock.patch.object(experiment, "write_file", side_effect=tamper_after_output), \
                self.assertRaisesRegex(RuntimeError, "Input files changed"):
            experiment.run_experiment(self.inputs, Path(self.directory.name) / "out", years=(5,))
        self.assertFalse((Path(self.directory.name) / "out" / "metrics.json").exists())
        self.assertFalse((Path(self.directory.name) / "out" / "input_hashes.json").exists())

    def test_hashing_ignores_linked_subdirectories(self):
        before = experiment.file_hashes(self.inputs)
        elsewhere = Path(self.directory.name) / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "notes.txt").write_text("not an input", encoding="utf-8")
        try:
            (self.inputs / "linked_directory").symlink_to(elsewhere, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable")
        self.assertEqual(experiment.file_hashes(self.inputs), before)

    def test_text_ids_with_leading_zeros_survive_writing(self):
        path = Path(self.directory.name) / "ids.csv"
        table = pd.DataFrame({"id": ["007", "010", "100"], "value": [1.0, 2.0, 3.0]})

        experiment.write_file(path, lambda file: table.to_csv(file, index=False, lineterminator="\n"))

        self.assertEqual(pd.read_csv(path, dtype={"id": str})["id"].tolist(), ["007", "010", "100"])

    def test_failed_file_write_keeps_the_previous_file(self):
        path = Path(self.directory.name) / "metrics.json"
        path.write_text("old", encoding="utf-8")

        def failing(file):
            file.write("partial")
            raise OSError("disk full")

        with self.assertRaises(OSError):
            experiment.write_file(path, failing)

        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual([item.name for item in Path(self.directory.name).iterdir() if item.name.startswith("metrics")],
                         ["metrics.json"])


class NestedTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.ids, features, numeric, categorical = experiment.synthetic_task(7, True, objects=120)
        self.features, self.numeric, self.categorical = features, numeric, categorical
        self.period = experiment.exact_periods(self.x)

    def fit_base(self, calls=None, fail_on=None):
        inner = experiment.in_process_fits(self.x, self.y, self.features, self.numeric, self.categorical,
                                           deadline=math.inf)

        def fit_base(model, fit_rows, predict_rows, label):
            if calls is not None:
                calls.append((model, fit_rows.copy(), predict_rows.copy()))
            return None if label == fail_on else inner(model, fit_rows, predict_rows, label)

        return fit_base

    def test_select_weight_direction_and_ties(self):
        self.assertEqual(experiment.select_weight({0.0: 1.0, 0.5: 1.0, 1.0: 2.0}, False), 0.0)
        self.assertEqual(experiment.select_weight({0.0: 2.0, 0.25: 1.0 + 1e-15, 0.5: 1.0}, False), 0.25)
        self.assertEqual(experiment.select_weight({0.0: 0.1, 0.5: 0.9, 1.0: 0.2}, True), 0.5)

    def test_every_model_and_adapter_fit_sees_only_its_own_past(self):
        calls, adapter_rows = [], []
        real_fit_candidate = experiment.fit_candidate

        def recording(x, y, rows, features, policy, weight):
            adapter_rows.append(rows.copy())
            return real_fit_candidate(x, y, rows, features, policy, weight)

        with mock.patch.object(experiment, "fit_candidate", recording):
            result, _ = experiment.nested_outer(5, self.x, self.y, self.period, self.ids, self.features,
                                                self.fit_base(calls))

        self.assertEqual(result["status"], "complete")
        self.assertEqual([split["period"] for split in result["inner_splits"]], [2, 3, 4])
        self.assertEqual(len(calls), 5)  # three inner fits, then the base and the reference model on years 1-4
        for _, fit_rows, predict_rows in calls:
            predicted = set(self.period[predict_rows])
            self.assertEqual(len(predicted), 1)
            self.assertLess(max(self.period[fit_rows]), min(predicted))
            self.assertFalse((fit_rows & (self.period == 5)).any())
        self.assertTrue(adapter_rows)
        self.assertTrue(all(not (rows & (self.period == 5)).any() for rows in adapter_rows))

    def test_outer_answers_change_no_weight_history_or_prediction(self):
        first, table = experiment.nested_outer(5, self.x, self.y, self.period, self.ids, self.features, self.fit_base())
        changed = self.y.copy()
        late = self.period == 5
        changed[late] = changed[late] * 3 + 777
        second, other = experiment.nested_outer(5, self.x, changed, self.period, self.ids, self.features,
                                                self.fit_base())

        # ExtraTrees with n_jobs=8 adds its trees' predictions in thread completion order: identical inputs may differ
        # in the last bits (about 1e-13), so scores and predictions are compared with a tolerance, weights exactly.
        for policy, chosen in first["selection"].items():
            self.assertEqual(chosen["weight"], second["selection"][policy]["weight"])
            for weight, score in chosen["scores"].items():
                self.assertAlmostEqual(score, second["selection"][policy]["scores"][weight], delta=1e-9)
        self.assertFalse(np.allclose(table["y_true"], other["y_true"]))
        pd.testing.assert_frame_equal(table.drop(columns="y_true"), other.drop(columns="y_true"), check_exact=False,
                                      rtol=1e-12, atol=1e-9)

    def test_a_failed_fit_leaves_the_year_incomplete_and_unscored(self):
        for fail_on in ("outer 4, inner 3", "outer 4"):
            with self.subTest(fail_on=fail_on):
                result, table = experiment.nested_outer(4, self.x, self.y, self.period, self.ids, self.features,
                                                        self.fit_base(fail_on=fail_on))
                self.assertTrue(result["status"].startswith("incomplete"))
                self.assertIsNone(table)
                self.assertNotIn("records", result)
        summary = experiment.summarize_recipes([result], {}, {}, 10)
        self.assertEqual((summary["complete_years"], summary["combined"]), ([], []))

    def test_too_few_earlier_periods_switch_history_off(self):
        result, table = experiment.nested_outer(2, self.x, self.y, self.period, self.ids, self.features, self.fit_base())
        self.assertEqual(result["inner_splits"], [])
        self.assertEqual({policy: item["weight"] for policy, item in result["selection"].items()},
                         {"allow": 0.0, "past_unique_only": 0.0})
        np.testing.assert_array_equal(table["nested_allow"], table["et_no_period"])

    def test_shuffled_control_permutes_answers_only_inside_each_period_of_the_given_rows(self):
        rows = self.period < 4
        shuffled = experiment.shuffled_within_periods(self.y, self.period, rows)
        np.testing.assert_array_equal(shuffled[~rows], self.y[~rows])
        for value in (1, 2, 3):
            part = self.period == value
            np.testing.assert_array_equal(np.sort(shuffled[part]), np.sort(self.y[part]))
        self.assertFalse(np.array_equal(shuffled[rows], self.y[rows]))


class BootstrapTests(unittest.TestCase):
    def test_cluster_ids_keep_a_key_together_and_rows_without_key_alone(self):
        ids = experiment.cluster_ids([("a",), None, ("b",), ("a",), None])
        self.assertEqual(ids[0], ids[3])
        self.assertEqual(len(set(ids.tolist())), 4)

    def test_paired_intervals(self):
        rng = np.random.default_rng(0)
        actual = rng.normal(size=300)
        base = actual + rng.normal(scale=2.0, size=300)
        result = experiment.paired_bootstrap(actual, {"base": base, "same": base.copy(), "better": actual + 0.1 * (base - actual)},
                                             "base", np.arange(300) // 3, 200)
        self.assertEqual((result["clusters"], result["rows"]), (100, 300))
        self.assertEqual(result["recipes"]["same"]["interval_95"], [0.0, 0.0])
        low, high = result["recipes"]["better"]["interval_95"]
        self.assertTrue(low < high < 0)
        self.assertEqual(result["recipes"]["better"]["share_of_resamples_better"], 1.0)
        single = experiment.paired_bootstrap(actual, {"base": base, "better": actual}, "base", np.zeros(300), 50)
        low, high = single["recipes"]["better"]["interval_95"]
        self.assertAlmostEqual(low, high)  # one cluster: every resample is the whole data


class BoundedRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.inputs = make_task(Path(self.directory.name))

    def test_nested_run_with_bounded_fits_files_and_recomputation(self):
        out = Path(self.directory.name) / "nested"
        summary = experiment.run_nested(self.inputs, out, years=(3,), resamples=50)

        self.assertEqual(sorted(path.name for path in out.iterdir() if path.is_file()),
                         ["input_hashes.json", "nested_metrics.csv", "nested_metrics.json", "nested_predictions_year3.csv"])
        self.assertEqual(len(summary["fits"]), 3)  # one inner fit (years 1 -> 2), base and reference on years 1-2
        for fit in summary["fits"]:
            self.assertEqual(fit["status"], "complete")
            self.assertGreater(fit["fit_seconds"], 0)
        table = pd.read_csv(out / "nested_predictions_year3.csv", dtype={"id": str})
        self.assertTrue(table["id"].str.startswith("0").all())
        combined = {row["variant"]: row["all"]["rmse"] for row in summary["combined"]}
        for name in experiment.RECIPES:
            self.assertAlmostEqual(combined[name], math.sqrt(((table["y_true"] - table[name]) ** 2).mean()), places=9)
        self.assertEqual(json.loads((out / "nested_metrics.json").read_text(encoding="utf-8"))["complete_years"], [3])
        self.assertTrue(summary["input_hashes"]["unchanged"])

    def test_timeouts_and_the_deadline_are_reported_not_hidden(self):
        timed_out = experiment.run_nested(self.inputs, Path(self.directory.name) / "a", years=(3,), fit_timeout=0.01,
                                          resamples=10)
        self.assertTrue(timed_out["fits"][0]["status"].startswith("failed: timeout"))
        self.assertTrue(timed_out["outer"][0]["status"].startswith("incomplete"))
        self.assertEqual(timed_out["combined"], [])
        late = experiment.run_nested(self.inputs, Path(self.directory.name) / "b", years=(3,), deadline_seconds=0,
                                     resamples=10)
        self.assertEqual(late["fits"][0]["status"], "skipped: run deadline reached")

    def test_scaling_worker_matches_the_former_sums_exactly(self):
        output = Path(self.directory.name) / "scaling.json"
        spec = Path(self.directory.name) / "spec.json"
        spec.write_text(json.dumps({"seed": 1, "seconds": 60, "start": 100, "largest": 400, "former_limit_seconds": 10,
                                    "output": str(output)}), encoding="utf-8")

        experiment.scaling_worker(str(spec))

        rows = json.loads(output.read_text(encoding="utf-8"))["synthetic_one_key"]
        self.assertEqual([row["periods_of_one_key"] for row in rows], [100, 200, 400])
        self.assertTrue(all(row["max_abs_difference_to_former"] == 0.0 for row in rows))
        self.assertTrue(all(row["traced_peak_bytes"] > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
