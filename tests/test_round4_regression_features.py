"""Fold-local feature risks without model fitting or public-data access."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from agent.round4_regression_features import RegressionFeatureAdapter, normalize_feature_recipe


def fixture():
    fit = pd.DataFrame({"period": [1, 1, 2, 2], "value": ["3", None, "bad", float("inf")],
                        "kind": ["A", "A", "B", None]}, index=[8, 3, 3, 1])
    query = pd.DataFrame({"period": [3, 3], "value": [None, "12"], "kind": ["new", "A"]}, index=[31, 29])
    options = {"missing_indicators": True, "category_frequencies": True}
    return fit, query, options


class Round4RegressionFeatureTests(unittest.TestCase):
    def test_internal_only_fit_does_not_learn_held_categories_and_full_refit_is_fresh(self):
        fit, query, options = fixture()
        earlier = fit.iloc[:2]
        held = fit.iloc[2:]
        internal = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit(earlier)
        np.testing.assert_array_equal(internal.transform(held)["__round4_frequency__kind"], [0., 0.])
        full = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit(fit)
        np.testing.assert_array_equal(full.transform(held)["__round4_frequency__kind"], [.25, .25])
        self.assertEqual(internal.evidence_["fit_rows"], 2)
        self.assertEqual(full.evidence_["fit_rows"], 4)
        self.assertEqual(internal.output_schema(), full.output_schema())
        self.assertEqual(internal.transform(query).iloc[1]["__round4_frequency__kind"], 1.)

    def test_predict_batch_and_order_never_change_feature_values_or_fit_maps(self):
        fit, query, options = fixture()
        adapter = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit(fit)
        maps, evidence = deepcopy(adapter.frequency_maps_), deepcopy(adapter.evidence_)
        baseline = adapter.transform(query)
        augmented = pd.concat([query.iloc[::-1], query.iloc[[0]]], ignore_index=True)
        result = adapter.transform(augmented)
        pd.testing.assert_frame_equal(baseline.reset_index(drop=True), result.iloc[:2][::-1].reset_index(drop=True))
        self.assertEqual(adapter.frequency_maps_, maps)
        self.assertEqual(adapter.evidence_, evidence)
        self.assertEqual(baseline.index.tolist(), query.index.tolist())

    def test_raw_inputs_roles_and_options_remain_unchanged_and_outputs_detached(self):
        fit, query, options = fixture()
        numeric, categorical = ["period", "value"], ["kind"]
        saved = deepcopy((fit, query, options, numeric, categorical))
        adapter = RegressionFeatureAdapter(numeric, categorical, options)
        transformed = adapter.fit_transform(fit)
        pd.testing.assert_frame_equal(transformed.loc[:, fit.columns], fit)
        transformed.iloc[0, 0] = 999
        transformed_query = adapter.transform(query)
        transformed_query.iloc[0, 1] = "changed"
        pd.testing.assert_frame_equal(fit, saved[0])
        pd.testing.assert_frame_equal(query, saved[1])
        self.assertEqual((options, numeric, categorical), saved[2:])

    def test_numeric_missing_definition_matches_finite_coercion(self):
        fit, _, options = fixture()
        result = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit_transform(fit)
        np.testing.assert_array_equal(result["__round4_missing__value"], [0., 1., 1., 1.])
        np.testing.assert_array_equal(result["__round4_missing__period"], np.zeros(4))

    def test_category_missing_literal_sentinels_and_text_are_distinct(self):
        fit = pd.DataFrame({"kind": [None, "", "__ALFA_MISSING__", "NA", " ", "0007", "7"]})
        adapter = RegressionFeatureAdapter([], ["kind"], {"category_frequencies": True}).fit(fit)
        expected = np.asarray([2., 2., 1., 1., 1., 1., 1.]) / 7
        np.testing.assert_allclose(adapter.transform(fit)["__round4_frequency__kind"], expected)
        self.assertEqual(adapter.evidence_["frequency_columns"][0]["fit_categories"], 6)

    def test_frequency_cap_disables_whole_column_without_partial_map(self):
        fit = pd.DataFrame({"kind": ["A", "B", "C", "A"]})
        adapter = RegressionFeatureAdapter([], ["kind"],
            {"category_frequencies": True, "max_categories_per_column": 2}).fit(fit)
        self.assertEqual(adapter.frequency_maps_["kind"], {})
        np.testing.assert_array_equal(adapter.transform(fit)["__round4_frequency__kind"], np.zeros(4))
        report = adapter.evidence_["frequency_columns"][0]
        self.assertEqual(report["reason"], "fit_cardinality_exceeds_cap")
        self.assertIsNone(report["fit_categories"])
        self.assertEqual(report["fit_categories_lower_bound"], 3)

    def test_column_cap_uses_declared_roles_and_preserves_other_raw_columns(self):
        fit = pd.DataFrame({"a": ["A", "B"], "b": ["x", "y"]})
        adapter = RegressionFeatureAdapter([], ["b", "a"],
            {"category_frequencies": True, "max_frequency_columns": 1})
        self.assertEqual(adapter.output_schema(), (["__round4_frequency__b"], ["b", "a"]))
        result = adapter.fit_transform(fit)
        self.assertEqual(list(result.columns), ["a", "b", "__round4_frequency__b"])
        self.assertEqual(adapter.evidence_["frequency_columns_omitted_by_cap"], ["a"])

    def test_labels_ignored_and_clone_refits_without_reusing_maps(self):
        fit, query, options = fixture()

        class ForbiddenLabels:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Feature adapter must not read targets")

        adapter = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit(fit, ForbiddenLabels())
        other = clone(adapter)
        with self.assertRaises(NotFittedError):
            other.transform(query)
        other.fit(fit.iloc[:1])
        self.assertEqual(other.transform(query).iloc[1]["__round4_frequency__kind"], 1.)
        self.assertEqual(adapter.transform(query).iloc[1]["__round4_frequency__kind"], .5)

    def test_disabled_blocks_preserve_frame_and_empty_query_is_supported(self):
        fit, query, _ = fixture()
        adapter = RegressionFeatureAdapter(["period", "value"], ["kind"]).fit(fit)
        pd.testing.assert_frame_equal(adapter.transform(query), query)
        pd.testing.assert_frame_equal(adapter.transform(query.iloc[:0]), query.iloc[:0])
        self.assertEqual(adapter.evidence_["added_numeric_features"], [])

    def test_schema_collision_and_failed_refit_cannot_reuse_stale_state(self):
        fit, query, options = fixture()
        with self.assertRaises(ValueError):
            RegressionFeatureAdapter(["__round4_frequency__kind"], ["kind"], options).output_schema()
        adapter = RegressionFeatureAdapter(["period", "value"], ["kind"], options).fit(fit)
        for broken in (query[["value", "period", "kind"]], query.drop(columns="kind")):
            with self.assertRaises(ValueError):
                adapter.transform(broken)
        with self.assertRaises(ValueError):
            adapter.fit(fit.iloc[:0])
        with self.assertRaises(NotFittedError):
            adapter.transform(query)

    def test_recipe_is_explicit_bounded_and_detached(self):
        original = {"missing_indicators": True}
        normalized = normalize_feature_recipe(original)
        self.assertEqual(normalized, {"missing_indicators": True, "category_frequencies": False,
            "max_frequency_columns": 16, "max_categories_per_column": 512})
        normalized["missing_indicators"] = False
        self.assertEqual(original, {"missing_indicators": True})
        for recipe in ({"category_frequencies": 1}, {"max_categories_per_column": 513},
                       {"max_frequency_columns": 0}, {"max_categories_per_column": True}, {"file": "anything"}):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                normalize_feature_recipe(recipe)


if __name__ == "__main__":
    unittest.main()
