import copy
from decimal import Decimal
import inspect
import math
import unittest

import numpy as np
import pandas as pd

from agent.history_candidate import FittedHistoryCandidate, HistoryApplication, fit_history_candidate


CONFIG = {"key_columns": ["state", "kind", "area", "built"], "period_column": "year",
          "numeric_columns": ["area", "built"], "positive_columns": ["built"], "ambiguity_policy": "allow"}
FEATURES = ("state", "kind", "area", "built", "year", "temp")


def frame(*items) -> pd.DataFrame:
    """items: (year, area) or (year, area, extra dict); the other key parts are equal, values are raw text."""
    data = []
    for item in items:
        year, area, *rest = item
        row = {"state": "S1", "kind": "office", "area": area, "built": "1990", "year": year, "temp": "0.5"}
        row.update(rest[0] if rest else {})
        data.append(row)
    return pd.DataFrame(data)


def fit(X, y, *, config=CONFIG, weight=0.5, task_type="regression", metric="RMSE", features=FEATURES):
    return fit_history_candidate(task_type=task_type, metric_name=metric, feature_columns=features, X_raw=X, y=y,
                                 config=config, blend_weight=weight)


class ContractTests(unittest.TestCase):
    def test_public_signatures_match_the_agreed_contract(self):
        parameters = inspect.signature(fit_history_candidate).parameters
        self.assertEqual(list(parameters), ["task_type", "metric_name", "feature_columns", "X_raw", "y", "config",
                                            "blend_weight"])
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
                            for p in parameters.values()))
        self.assertEqual(list(inspect.signature(FittedHistoryCandidate.apply).parameters),
                         ["self", "X_raw", "base_prediction"])
        self.assertEqual(list(HistoryApplication.__dataclass_fields__), ["prediction", "applied_mask", "diagnostics"])
        result = fit(frame((1, "100")), [1.0]).apply(frame((2, "100")), np.zeros(1))
        with self.assertRaises(Exception):
            result.prediction = None  # frozen

    def test_explicit_config_errors_are_raised(self):
        X, y = frame((1, "100")), [1.0]
        bad = [{**CONFIG, "keys": ["state"]}, {k: v for k, v in CONFIG.items() if k != "key_columns"},
               {k: v for k, v in CONFIG.items() if k != "period_column"}, {**CONFIG, "key_columns": ["state", "state"]},
               {**CONFIG, "key_columns": "state"}, {**CONFIG, "key_columns": []}, {**CONFIG, "key_columns": ["", "kind"]},
               {**CONFIG, "ambiguity_policy": "sometimes"}, {**CONFIG, "key_columns": ["state", "year"]},
               {**CONFIG, "numeric_columns": ["temp"]}, {**CONFIG, "positive_columns": ["state"]},
               {**CONFIG, "period_column": ""}, ["state", "year"]]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                fit(X, y, config=config)
        for weight in (-0.1, 1.5, math.nan, math.inf, True, "0.5", None):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "blend_weight"):
                fit(X, y, weight=weight)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit(X, y, features=("state", "state", "kind", "area", "built", "year"))
        with self.assertRaisesRegex(ValueError, "rows but y has"):
            fit(frame((1, "100"), (2, "100")), [1.0])
        with self.assertRaisesRegex(ValueError, "index of y"):
            fit(frame((1, "100")), pd.Series([1.0], index=[7]))
        with self.assertRaisesRegex(ValueError, "finite"):
            fit(frame((1, "100"), (2, "100")), [1.0, math.nan])
        with self.assertRaises(TypeError):
            fit(X.to_numpy(), y)
        # a bad explicit config is an error even for a task that would leave the candidate inactive
        with self.assertRaises(ValueError):
            fit(X, ["a"], config={**CONFIG, "ambiguity_policy": "maybe"}, task_type="classification")


class InactiveTests(unittest.TestCase):
    def assert_unchanged(self, candidate, X, base, reason):
        before = copy.deepcopy(base)
        result = candidate.apply(X, base)
        self.assertFalse(candidate.active)
        self.assertEqual((candidate.reason, result.diagnostics["status"], result.diagnostics["reason"]),
                         (reason, "inactive", reason))
        self.assertEqual((result.prediction.dtype, result.prediction.shape), (np.asarray(base).dtype, np.shape(base)))
        np.testing.assert_array_equal(result.prediction, base)
        self.assertEqual(result.applied_mask.dtype, bool)
        self.assertEqual(result.applied_mask.tolist(), [False] * len(X))
        np.testing.assert_array_equal(base, before)

    def test_reasons_and_unchanged_predictions(self):
        X, y, query = frame((1, "100"), (2, "100")), [10.0, 20.0], frame((3, "100"), (3, "100"))
        base = np.array([1.5, 2.5], dtype=np.float32)
        cases = [({"config": None}, "no_config"),
                 ({"task_type": "multiclass"}, "unknown_task_type"), ({"task_type": None}, "unknown_task_type"),
                 ({"metric": "MAPE"}, "unsupported_metric"), ({"metric": "Macro F1"}, "metric_not_for_regression"),
                 ({"features": ("state", "kind", "area", "built", "temp")}, "history_columns_not_in_features")]
        for options, reason in cases:
            with self.subTest(options=options):
                self.assert_unchanged(fit(X, y, **options), query, base, reason)
        missing = fit(X.drop(columns="built"), y, features=("state", "kind", "area", "built", "year"))
        self.assert_unchanged(missing, query, base, "missing_columns")
        nothing = fit(frame((1, "100", {"built": "0"}), (None, "100")), [1.0, 2.0])
        self.assert_unchanged(nothing, query, base, "no_usable_fit_rows")

    def test_familiar_column_names_do_not_enable_history_without_config(self):
        X = pd.DataFrame({"State_Factor": ["S"] * 2, "building_class": ["C"] * 2, "facility_type": ["O"] * 2,
                          "floor_area": ["1"] * 2, "year_built": ["1990"] * 2, "ELEVATION": ["0"] * 2,
                          "Year_Factor": ["1", "2"]})
        candidate = fit_history_candidate(task_type="regression", metric_name="RMSE", feature_columns=tuple(X),
                                          X_raw=X, y=[1.0, 2.0], config=None, blend_weight=1.0)
        self.assert_unchanged(candidate, X, np.array([5.0, 6.0]), "no_config")

    def test_classification_labels_and_probabilities_keep_dtype_shape_and_order(self):
        X, labels = frame((1, "100"), (2, "100")), ["NA", "b"]
        candidate = fit(X, labels, task_type="classification", metric="Macro F1")
        self.assert_unchanged(candidate, X, np.array(["b", "NA"], dtype=object), "classification_task")
        self.assert_unchanged(candidate, X, np.array([[0.2, 0.8], [0.7, 0.3]]), "classification_task")
        with self.assertRaisesRegex(ValueError, "one entry per row"):
            candidate.apply(X, np.zeros(3))


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.X = frame((1, "100"), (2, "100"), (1, "200"), (2, "200"), (2, "300"))
        self.y = [10.0, 30.0, 100.0, 300.0, 7.0]

    def test_formula_rows_without_history_and_metric(self):
        query = frame((3, "100"), (3, "200"), (3, "999"), (2, "100"), (1, "100"))
        base = np.array([40.0, 120.0, 55.5, 13.0, 1.0])
        candidate = fit(self.X, self.y, weight=0.25, metric="MAE")

        result = candidate.apply(query, base)

        self.assertEqual(result.diagnostics["metric"], "MAE")  # the task metric is kept, not replaced by RMSE
        expected = [0.75 * 40 + 0.25 * 20, 0.75 * 120 + 0.25 * 200, 55.5, 0.75 * 13 + 0.25 * 10, 1.0]
        np.testing.assert_allclose(result.prediction, expected, rtol=0, atol=1e-12)
        self.assertEqual(result.applied_mask.tolist(), [True, True, False, True, False])
        self.assertEqual(result.prediction[2], base[2])  # exactly the base where there is no history
        self.assertEqual(result.prediction[4], base[4])
        d = result.diagnostics
        self.assertEqual((d["rows"], d["rows_with_full_key"], d["rows_with_past_history"], d["rows_applied"],
                          d["rows_changed"]), (5, 5, 3, 3, 3))
        self.assertAlmostEqual(d["share_applied"], 0.6)

    def test_weight_zero_keeps_the_whole_prediction_and_weight_one_gives_the_past_mean(self):
        query, base = frame((3, "100"), (3, "200"), (3, "999")), np.array([1.0, 2.0, 3.0])
        zero = fit(self.X, self.y, weight=0).apply(query, base)
        np.testing.assert_array_equal(zero.prediction, base)
        self.assertFalse(zero.applied_mask.any())
        self.assertEqual(zero.diagnostics["rows_with_past_history"], 2)
        one = fit(self.X, self.y, weight=1).apply(query, base)
        self.assertEqual(one.prediction.tolist(), [20.0, 200.0, 3.0])

    def test_no_arithmetic_with_missing_base_values_outside_applied_rows(self):
        query, base = frame((3, "999"), (3, "100")), np.array([np.nan, 2.0])
        result = fit(self.X, self.y).apply(query, base)
        self.assertTrue(math.isnan(result.prediction[0]))
        self.assertEqual(result.diagnostics["rows_changed"], 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit(self.X, self.y).apply(frame((3, "100")), np.array([np.inf]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            fit(self.X, self.y).apply(query, np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "lacks the history columns"):
            fit(self.X, self.y).apply(query.drop(columns="year"), np.zeros(2))

    def test_continuous_original_scale_without_rounding_or_clipping(self):
        X, y = frame((1, "1"), (1, "2")), [-1e6 + 0.123456789, 1e9 + 0.987654321]
        result = fit(X, y, weight=0.5).apply(frame((2, "1"), (2, "2")), np.array([-5.5, 3.25]))
        np.testing.assert_array_equal(result.prediction, [0.5 * -5.5 + 0.5 * y[0], 0.5 * 3.25 + 0.5 * y[1]])


class CausalityTests(unittest.TestCase):
    def test_fit_rows_use_strictly_earlier_periods_only(self):
        X = frame((1, "100"), (2, "100"), (2, "100"), (3, "100"))
        base = np.zeros(4)
        first = fit(X, [10.0, 20.0, 40.0, 70.0], weight=1).apply(X, base).prediction
        changed = fit(X, [10.0, 999.0, -999.0, 5000.0], weight=1).apply(X, base).prediction

        self.assertEqual(first.tolist(), [0.0, 10.0, 10.0, 70.0 / 3])
        self.assertEqual(changed[:3].tolist(), first[:3].tolist())  # same-period and later answers are never seen

    def test_apply_needs_no_answers_changes_no_state_and_keeps_inputs(self):
        X, y = frame((1, "100"), (2, "100")), pd.Series([10.0, 20.0])
        query, base = frame((3, "100"), (3, "100"), (2, "5")), np.array([1.0, 2.0, 3.0])
        snapshots = [X.copy(deep=True), y.copy(deep=True), query.copy(deep=True), base.copy()]
        candidate = fit(X, y)
        state = copy.deepcopy(candidate._encoder._past)

        first, second = candidate.apply(query, base), candidate.apply(query, base)

        self.assertEqual(candidate._encoder._past, state)
        np.testing.assert_array_equal(first.prediction, second.prediction)
        self.assertEqual(first.diagnostics, second.diagnostics)
        pd.testing.assert_frame_equal(X, snapshots[0])
        pd.testing.assert_series_equal(y, snapshots[1])
        pd.testing.assert_frame_equal(query, snapshots[2])
        np.testing.assert_array_equal(base, snapshots[3])

    def test_permuted_rows_and_other_query_batches_keep_each_row_result(self):
        X = frame((1, "100"), (2, "100"), (1, "200"), (1, "200"), (2, "300"))
        candidate = fit(X, [10.0, 20.0, 1.0, 3.0, 5.0], config={**CONFIG, "ambiguity_policy": "past_unique_only"})
        query = frame((3, "100"), (3, "200"), (3, "300"), (3, "300"), (2, "100"), (3, "404")).set_index(
            pd.Index(list("abcdef")))
        base = np.arange(6, dtype=float)
        together = candidate.apply(query, base)

        order = [5, 2, 0, 4, 1, 3]
        shuffled = candidate.apply(query.iloc[order], base[order])
        np.testing.assert_array_equal(shuffled.prediction, together.prediction[order])
        np.testing.assert_array_equal(shuffled.applied_mask, together.applied_mask[order])
        for position in range(len(query)):
            alone = candidate.apply(query.iloc[[position]], base[[position]])
            self.assertEqual(alone.prediction[0], together.prediction[position])
        # key 300 repeats only inside the query batch: a diagnostic, not a reason to refuse the history
        self.assertTrue(together.applied_mask[2] and together.applied_mask[3])
        self.assertEqual(together.diagnostics["rows_repeated_key_in_query"], 2)
        self.assertFalse(together.applied_mask[1])  # key 200 repeated inside past period 1

    def test_refitting_forgets_the_previous_data(self):
        old = fit(frame((1, "100")), [10.0])
        new = fit(frame((1, "200")), [99.0])
        base = np.zeros(2)
        self.assertEqual(new.apply(frame((2, "100"), (2, "200")), base).applied_mask.tolist(), [False, True])
        self.assertEqual(old.apply(frame((2, "100"), (2, "200")), base).applied_mask.tolist(), [True, False])


class AmbiguityAndKeyTests(unittest.TestCase):
    def test_policies_use_only_the_available_past(self):
        X = frame((1, "100"), (1, "100"), (2, "100"), (2, "200"))
        y = [10.0, 30.0, 50.0, 7.0]
        query = frame((2, "100"), (3, "100"), (3, "200"))
        base = np.zeros(3)
        allow = fit(X, y, weight=1).apply(query, base)
        unique = fit(X, y, weight=1, config={**CONFIG, "ambiguity_policy": "past_unique_only"}).apply(query, base)

        self.assertEqual(allow.prediction.tolist(), [20.0, 30.0, 7.0])
        self.assertEqual(unique.prediction.tolist(), [0.0, 0.0, 7.0])
        self.assertEqual(unique.diagnostics["rows_with_ambiguous_past_history"], 2)
        self.assertEqual(unique.diagnostics["rows_allowed_by_policy"], 1)

    def test_no_false_matches_from_numbers_text_or_missing_values(self):
        cases = [("area", str(2**53), str(2**53 + 1)), ("area", "61242.123456789012345", "61242.123456789012346"),
                 ("state", "a|b", "a"), ("state", "NA", np.nan), ("state", "NA", None), ("state", "nan", ""),
                 ("area", "100", "100.0000000000000000001")]
        for column, first, second in cases:
            with self.subTest(column=column, first=first, second=second):
                extra = {"kind": "c"} if first == "a|b" else {}
                X = frame((1, "100", {column: first, **extra}))
                query = frame((2, "100", {column: second, "kind": "b|c" if first == "a|b" else "office"}))
                self.assertFalse(fit(X, [1.0]).apply(query, np.zeros(1)).applied_mask.any())
        same = frame((1, "100", {"state": "NA"}))
        self.assertTrue(fit(same, [1.0]).apply(frame((2, "100.0", {"state": " NA "})), np.zeros(1)).applied_mask[0])

    def test_invalid_periods_refuse_history_without_losing_rows(self):
        X = frame((1, "100"), ("later", "100"), (None, "100"), ("2.5", "100"))
        candidate = fit(X, [10.0, 999.0, 999.0, 30.0], weight=1)
        query = frame(("", "100"), ("inf", "100"), (3, "100"), ("2.5", "100"), (Decimal("2.50000000000000000001"), "100"))
        result = candidate.apply(query, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        self.assertEqual(candidate.fit_diagnostics["rows_remembered"], 2)
        self.assertEqual(candidate.fit_diagnostics["periods"], [1, "2.5"])
        self.assertEqual(result.prediction.tolist(), [1.0, 2.0, 20.0, 10.0, 20.0])
        self.assertEqual(result.diagnostics["rows_with_valid_period"], 3)
        self.assertEqual(len(result.prediction), len(query))


class SyntheticIntegrationTests(unittest.TestCase):
    def test_encoder_model_adapter_cycle_with_known_object_level(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(5)
        n = 200
        objects = pd.DataFrame({"state": rng.choice(["S1", "S2"], n), "kind": rng.choice(["a", "b"], n),
                                "area": [str(1000 + 7 * i) for i in range(n)], "built": "1990",
                                "level": rng.uniform(10, 200, n)})
        parts = []
        for year in range(1, 6):
            part = objects[rng.random(n) < 0.8].copy()
            part["year"], part["temp"] = str(year), rng.normal(size=len(part)).astype(str)
            part["y"] = part["level"] + rng.normal(scale=2.0, size=len(part))
            parts.append(part)
        data = pd.concat(parts, ignore_index=True)
        fit_rows, valid_rows = data[data["year"] != "5"], data[data["year"] == "5"]
        model = HistGradientBoostingRegressor(random_state=0).fit(fit_rows[["temp"]].astype(float), fit_rows["y"])
        base = model.predict(valid_rows[["temp"]].astype(float))

        candidate = fit(fit_rows[list(FEATURES)], fit_rows["y"].to_numpy(), weight=0.9)
        result = candidate.apply(valid_rows[list(FEATURES)], base)

        error = lambda prediction: math.sqrt(np.mean((valid_rows["y"].to_numpy() - prediction) ** 2))
        self.assertGreater(result.diagnostics["share_applied"], 0.9)
        self.assertLess(error(result.prediction), error(base) / 3)


if __name__ == "__main__":
    unittest.main()
