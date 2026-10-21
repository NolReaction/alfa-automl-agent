import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from agent.history import PastOnlyHistory


class PastOnlyHistoryTests(unittest.TestCase):
    def test_training_excludes_own_same_and_future_period_targets(self):
        X = pd.DataFrame({"entity": ["A", "A", "A", "A", "B"],
                          "period": ["1", "2", "2", "3", "2"], "other": [9] * 5})
        before = X.copy(deep=True)
        result = PastOnlyHistory(["entity"], "period").fit_transform(X, [10, 20, 40, 1000, 50])
        self.assertEqual(result["history_count"].tolist(), [0, 1, 1, 3, 0])
        np.testing.assert_allclose(result["history_mean"], [0, 10, 10, 70 / 3, 10])
        np.testing.assert_allclose(result["history_last"], [0, 10, 10, 30, 10])
        self.assertEqual(result["history_has_history"].tolist(), [0, 1, 1, 1, 0])
        assert_frame_equal(X, before)
        self.assertEqual(result.columns.tolist(), [*X.columns, "history_count", "history_mean",
                                                   "history_last", "history_has_history"])

    def test_mutating_future_targets_cannot_change_earlier_training_features(self):
        X = pd.DataFrame({"entity": ["A"] * 5, "period": ["1", "2", "3", "4", "5"]})
        first = PastOnlyHistory(["entity"], "period").fit_transform(X, [10, 20, 30, 40, 50])
        changed = PastOnlyHistory(["entity"], "period").fit_transform(X, [10, 20, 3000, -4000, 5000])
        assert_frame_equal(first.iloc[:3], changed.iloc[:3])
        self.assertNotEqual(first.loc[3, "history_mean"], changed.loc[3, "history_mean"])

    def test_prediction_uses_only_earlier_fitted_rows_in_any_query_order(self):
        train = pd.DataFrame({"entity": ["A", "A", "A", "B"], "period": [1, 3, 5, 2]})
        history = PastOnlyHistory(["entity"], "period").fit(train, [10, 30, 5000, 40])
        query = pd.DataFrame({"entity": ["A", "A", "A", "new"], "period": [4, 1, 3, 3]})
        before = query.copy(deep=True)
        result = history.transform(query)
        np.testing.assert_allclose(result["history_mean"], [20, 0, 10, 25])
        self.assertEqual(result["history_count"].tolist(), [2, 0, 1, 0])
        np.testing.assert_allclose(result["history_last"], [30, 0, 10, 25])
        assert_frame_equal(query, before)
        assert_frame_equal(history.transform(query.iloc[::-1]).sort_index(), result)

    def test_same_period_last_is_group_mean_and_independent_of_row_order(self):
        train = pd.DataFrame({"entity": ["A"] * 4, "period": [1, 2, 2, 3]})
        y = np.array([10, 20, 40, 900])
        query = pd.DataFrame({"entity": ["A"], "period": [3]})
        expected = PastOnlyHistory(["entity"], "period").fit(train, y).transform(query)
        order = [3, 2, 0, 1]
        actual = PastOnlyHistory(["entity"], "period").fit(train.iloc[order], y[order]).transform(query)
        assert_frame_equal(expected, actual)
        self.assertEqual(actual.loc[0, "history_last"], 30)

    def test_exact_entity_keys_do_not_round_or_collide_and_na_is_literal(self):
        train = pd.DataFrame({"first": ["a|b", "a", "NA", "1.000000000000001", "1.000000000000002"],
                              "second": ["c", "b|c", "NA", "x", "x"], "period": [1] * 5})
        history = PastOnlyHistory(["first", "second"], "period").fit(train, [10, 20, 30, 40, 50])
        query = train.copy()
        query["period"] = 2
        result = history.transform(query)
        self.assertEqual(result["history_mean"].tolist(), [10, 20, 30, 40, 50])
        self.assertEqual(result["history_count"].tolist(), [1] * 5)

    def test_missing_keys_are_not_a_shared_entity_and_prior_uses_only_past(self):
        train = pd.DataFrame({"entity": [None, "", "NA", "A"], "period": [1, 2, 1, 5]})
        history = PastOnlyHistory(["entity"], "period").fit(train, [10, 20, 30, 10000])
        query = pd.DataFrame({"entity": [None, "", " ", "NA", "new", "new"],
                              "period": [2, 3, 3, 2, 1, None]})
        result = history.transform(query)
        self.assertEqual(result["history_count"].tolist(), [0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(result["history_mean"], [20, 20, 20, 30, 0, 0])
        self.assertEqual(result["history_has_history"].tolist(), [0, 0, 0, 1, 0, 0])

    def test_numeric_period_order_does_not_round_large_integers(self):
        train = pd.DataFrame({"entity": ["A", "A"],
                              "period": ["9007199254740992", "9007199254740993"]})
        result = PastOnlyHistory(["entity"], "period").fit_transform(train, [10, 20])
        self.assertEqual(result["history_count"].tolist(), [0, 1])
        self.assertEqual(result["history_mean"].tolist(), [0, 10])

    def test_datetime_periods_compare_instants_and_do_not_guess_local_dates(self):
        train = pd.DataFrame({"entity": ["A", "A", "A"],
                              "period": ["2020-01-01T00:00:00Z", "2020-01-02T03:00:00+03:00", "01/02/2020"]})
        history = PastOnlyHistory(["entity"], "period", time_kind="datetime").fit(train, [10, 20, 10000])
        query = pd.DataFrame({"entity": ["A", "A"],
                              "period": ["2020-01-02T00:00:00Z", "2020-01-03"]})
        result = history.transform(query)
        self.assertEqual(result["history_count"].tolist(), [1, 2])
        self.assertEqual(result["history_mean"].tolist(), [10, 15])

    def test_sklearn_clone_names_and_invalid_contracts(self):
        train = pd.DataFrame({"entity": ["A", "A"], "period": [1, 2]})
        original = PastOnlyHistory(["entity"], "period", prefix="past")
        copied = clone(original)
        with self.assertRaises(NotFittedError):
            copied.transform(train)
        result = copied.fit_transform(train, [10, 20])
        self.assertEqual(copied.get_feature_names_out().tolist(), result.columns.tolist())
        for transformer, target in (
            (PastOnlyHistory([], "period"), [1, 2]),
            (PastOnlyHistory(["entity"], "entity"), [1, 2]),
            (PastOnlyHistory(["entity"], "period", time_kind="guess"), [1, 2]),
            (PastOnlyHistory(["entity"], "period"), [1, float("nan")]),
            (PastOnlyHistory(["entity"], "period"), ["not", "numeric"]),
            (PastOnlyHistory(["entity"], "period"), [1]),
        ):
            with self.subTest(transformer=transformer), self.assertRaises(ValueError):
                transformer.fit(train, target)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            copied.fit(train.assign(past_mean=0), [10, 20])


if __name__ == "__main__":
    unittest.main()
