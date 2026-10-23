"""Feature state comes from fitting rows, while validation/test remain untouched."""

from copy import deepcopy
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from agent.features import (
    FeatureTransformer, MISSING_CATEGORY, RARE_CATEGORY,
    feature_schema, normalize_feature_config,
)


class FeatureTests(unittest.TestCase):
    def test_wide_feature_expansion_preserves_rows_without_fragmentation_warnings(self):
        frame = pd.DataFrame({f"x{i}": [i, np.nan, -i] for i in range(120)},
                             index=[9, 3, 9])
        original = frame.copy(deep=True)
        feature = FeatureTransformer(list(frame), [],
                                     {"missing_indicators": True, "signed_log": True})
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            result = feature.fit_transform(frame)
        self.assertEqual(result.shape, (3, 240))
        self.assertEqual(result.index.tolist(), [9, 3, 9])
        self.assertEqual(result.columns.tolist(), feature.get_feature_names_out().tolist())
        self.assertEqual(result["x119__missing"].tolist(), [0., 1., 0.])
        self.assertAlmostEqual(result["x119"].iloc[2], -np.log1p(119))
        pd.testing.assert_frame_equal(frame, original)

    def test_rare_vocabulary_is_fitted_only_on_training_rows_and_keeps_missing_separate(self):
        train = pd.DataFrame({"n": [1, 2, 3, 4, 5], "cat": ["NA", "NA", "once", "", None]})
        validation = pd.DataFrame({"n": [6, 7, 8], "cat": ["new", "NA", ""]})
        original_train, original_validation = train.copy(deep=True), validation.copy(deep=True)
        feature = FeatureTransformer(["n"], ["cat"], {"rare_min_count": 2}).fit(train)
        self.assertEqual(feature.allowed_categories_["cat"], frozenset({"NA"}))
        self.assertEqual(feature.transform(train)["cat"].tolist(),
                         ["NA", "NA", RARE_CATEGORY, MISSING_CATEGORY, MISSING_CATEGORY])
        self.assertEqual(feature.transform(validation)["cat"].tolist(), [RARE_CATEGORY, "NA", MISSING_CATEGORY])
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(validation, original_validation)

    def test_real_reserved_category_cannot_collide_with_missing_or_rare(self):
        train = pd.DataFrame({"cat": [MISSING_CATEGORY, RARE_CATEGORY, "__ALFA_VALUE__x", "", "NA"]})
        transformed = FeatureTransformer([], ["cat"]).fit_transform(train)
        self.assertEqual(transformed["cat"].nunique(), 5)
        self.assertEqual(transformed["cat"].iloc[3], MISSING_CATEGORY)
        self.assertEqual(transformed["cat"].iloc[4], "NA")

    def test_numeric_coercion_missing_indicators_and_signed_log_do_not_depend_on_validation(self):
        train = pd.DataFrame({"value": [0, 1, -3]})
        feature = FeatureTransformer(["value"], [], {"missing_indicators": True, "signed_log": True}).fit(train)
        check = pd.DataFrame({"value": [np.nan, "bad", np.inf, -np.inf, -3, 1e300]})
        transformed = feature.transform(check)
        self.assertEqual(list(transformed), ["value", "value__missing"])
        self.assertEqual(transformed["value__missing"].tolist(), [1., 1., 1., 1., 0., 0.])
        self.assertTrue(transformed["value"].iloc[:4].isna().all())
        self.assertAlmostEqual(transformed["value"].iloc[4], -np.log(4))
        self.assertTrue(np.isfinite(transformed["value"].iloc[5]))

    def test_explicit_date_and_text_columns_expand_safely_and_drop_raw_values(self):
        frame = pd.DataFrame({
            "number": [1, 2, 3],
            "date": ["2024-02-29T10:30:00+02:00", "not-a-date", None],
            "text": ["hello  world", "", None], "category": ["A", "B", "NA"],
        }, index=[7, 8, 9])
        config = {"date_columns": ["date"], "text_columns": ["text"]}
        numeric, categorical = feature_schema(["number"], ["date", "text", "category"], config)
        feature = FeatureTransformer(["number"], ["date", "text", "category"], config)
        transformed = feature.fit_transform(frame, y=object())
        self.assertEqual(list(transformed), numeric + categorical)
        self.assertNotIn("date", transformed)
        self.assertNotIn("text", transformed)
        self.assertEqual(transformed.index.tolist(), [7, 8, 9])
        self.assertEqual(transformed.loc[7, ["date__year", "date__month", "date__day", "date__hour"]].tolist(),
                         [2024., 2., 29., 8.])
        self.assertTrue(transformed.loc[8, [name for name in numeric if name.startswith("date__")]].isna().all())
        self.assertEqual(transformed["text__length"].tolist(), [12., 0., 0.])
        self.assertEqual(transformed["text__word_count"].tolist(), [2., 0., 0.])
        self.assertEqual(transformed["text__missing"].tolist(), [0., 1., 1.])
        self.assertEqual(categorical, ["category"])

    def test_clone_refits_independent_state_and_never_mutates_constructor_config(self):
        config = {"rare_min_count": 2, "date_columns": []}
        original = deepcopy(config)
        fitted = FeatureTransformer([], ["c"], config).fit(pd.DataFrame({"c": ["a", "a", "b"]}))
        cloned = clone(fitted)
        with self.assertRaises(NotFittedError):
            cloned.transform(pd.DataFrame({"c": ["a"]}))
        cloned.fit(pd.DataFrame({"c": ["b", "b"]}))
        self.assertEqual(fitted.allowed_categories_["c"], frozenset({"a"}))
        self.assertEqual(cloned.allowed_categories_["c"], frozenset({"b"}))
        self.assertEqual(config, original)
        names = fitted.get_feature_names_out(["c"])
        self.assertEqual(names.tolist(), ["c"])
        with self.assertRaises(ValueError):
            fitted.get_feature_names_out(["other"])

    def test_config_is_strict_and_returns_detached_lists(self):
        invalid = [True, [], {"unknown": 1}, {"missing_indicators": 1}, {"signed_log": "true"},
                   {"rare_min_count": True}, {"rare_min_count": 1.5}, {"rare_min_count": -1},
                   {"rare_min_count": 1001}, {"date_columns": "date"}, {"text_columns": [None]},
                   {"date_columns": ["x", "x"]}, {"date_columns": [" "]},
                   {"date_columns": ["x"], "text_columns": ["x"]},
                   {"text_columns": [str(index) for index in range(101)]}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_feature_config(config)
        source = {"date_columns": ["d"]}
        normalized = normalize_feature_config(source)
        normalized["date_columns"].append("other")
        self.assertEqual(source["date_columns"], ["d"])
        self.assertEqual(normalize_feature_config()["rare_min_count"], 0)

    def test_schema_rejects_generated_name_collisions_and_unknown_or_overlapping_features(self):
        cases = [(["x", "x"], [], None), (["x"], ["x"], None), ([], [], None),
                 (["x"], [], {"date_columns": ["missing"]}),
                 (["x", "x__missing"], [], {"missing_indicators": True}),
                 (["d__year"], ["d"], {"date_columns": ["d"]}),
                 ([], ["t", "t__length"], {"text_columns": ["t"]})]
        for numeric, categorical, config in cases:
            with self.subTest(numeric=numeric, categorical=categorical, config=config), self.assertRaises(ValueError):
                feature_schema(numeric, categorical, config)

    def test_missing_or_duplicate_input_columns_and_empty_fit_fail_clearly(self):
        feature = FeatureTransformer(["x"], [])
        for frame in (pd.DataFrame({"other": [1]}), pd.DataFrame([[1, 2]], columns=["x", "x"]),
                      pd.DataFrame({"x": []}), [[1]]):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                feature.fit(frame)
        feature.fit(pd.DataFrame({"x": [1]}))
        with self.assertRaises(ValueError):
            feature.transform(pd.DataFrame({"other": [2]}))
        self.assertEqual(feature.transform(pd.DataFrame({"x": []})).shape, (0, 1))

    def test_empty_inference_batches_keep_expanded_date_text_schema(self):
        feature = FeatureTransformer([], ["date", "text"],
                                     {"date_columns": ["date"], "text_columns": ["text"]})
        feature.fit(pd.DataFrame({"date": ["2025-01-01"], "text": ["hello"]}))
        transformed = feature.transform(pd.DataFrame({"date": [], "text": []}))
        self.assertEqual(transformed.shape, (0, 8))
        self.assertEqual(transformed.columns.tolist(), feature.get_feature_names_out().tolist())


if __name__ == "__main__":
    unittest.main()
