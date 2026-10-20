import copy
import math
import unittest

import numpy as np
import pandas as pd

from agent.building_history import FEATURES, HistoryEncoder


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

        plain = HistGradientBoostingRegressor(random_state=0).fit(X_fit[columns], fit["y"])
        with_history = HistGradientBoostingRegressor(random_state=0).fit(
            pd.concat([X_fit[columns], history_fit], axis=1), fit["y"])
        error_plain = np.sqrt(np.mean((plain.predict(X_valid[columns]) - valid["y"]) ** 2))
        error_history = np.sqrt(np.mean((with_history.predict(pd.concat([X_valid[columns], history_valid], axis=1))
                                         - valid["y"]) ** 2))

        self.assertGreater(history_valid["history_present"].mean(), 0.9)
        self.assertLess(error_history, error_plain / 2)  # the planted building level is learned only through history


if __name__ == "__main__":
    unittest.main()
