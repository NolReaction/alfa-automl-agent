"""Leakage, schema and numeric edge cases for bounded round4 features."""
import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from agent.frequency_candidate import frequency_feature_name, FrequencySkipError
from agent.round4_classification_features import (
    FeatureContractError, augment_features, asymmetry_feature_name, normalize_options,
)


def pair(forward="out", reverse="in"):
    return {"forward": forward, "reverse": reverse, "quantity": "transfer", "unit": "bytes"}


def context(features=None, numeric=None, categorical=None):
    return {"features": features or ["out", "in", "kind"],
            "numeric": numeric if numeric is not None else ["out", "in"],
            "categorical": categorical if categorical is not None else ["kind"],
            "task_spec": {"id_column": "technical_key", "target_column": "answer"}}


class Round4FeatureTests(unittest.TestCase):
    def test_prediction_rows_cannot_change_frequency_dictionary_or_group_weights(self):
        fit = pd.DataFrame({"out": ["1", "1", "2", "3"], "in": ["0"] * 4,
                            "kind": ["NA", "NA", "0007", "7"]}, index=[9, 4, 2, 1])
        predict = pd.DataFrame({"out": ["8", "8", "8"], "in": ["0"] * 3,
                                "kind": ["NA", "unknown", "7"]})
        opt = {"frequency": True, "asymmetry_pairs": [], "group_weight": True}
        before_fit, before_predict, ctx = fit.copy(deep=True), predict.copy(deep=True), context()
        first = augment_features(fit, predict, ctx, opt)
        expanded = pd.concat([predict, pd.DataFrame({"out": ["999"] * 100, "in": ["999"] * 100,
                                                   "kind": ["unknown"] * 100})], ignore_index=True)
        second = augment_features(fit, expanded, ctx, opt)
        assert_frame_equal(first[0], second[0])
        np.testing.assert_array_equal(first[4], second[4])
        self.assertEqual(first[3]["frequency"], second[3]["frequency"])
        self.assertEqual(first[3]["group_weight"], second[3]["group_weight"])
        added = frequency_feature_name("kind")
        np.testing.assert_allclose(first[1][added], [np.log1p(2), 0., np.log1p(1)])
        self.assertEqual(first[2]["numeric"], ["out", "in", added])
        self.assertEqual(first[2]["categorical"], ["kind"])
        self.assertEqual(first[2]["features"], list(first[0].columns))
        assert_frame_equal(fit, before_fit)
        assert_frame_equal(predict, before_predict)
        self.assertEqual(ctx, context())
        json.dumps(first[3], allow_nan=False)

    def test_each_fold_rebuilds_frequency_counts_without_global_state(self):
        frame = pd.DataFrame({"out": ["1", "2", "3"], "in": ["0"] * 3,
                              "kind": ["x", "x", "y"]})
        opt = {"frequency": True, "asymmetry_pairs": [], "group_weight": False}
        first = augment_features(frame.iloc[:2], frame.iloc[2:], context(), opt)
        second = augment_features(frame.iloc[2:], frame.iloc[:2], context(), opt)
        added = frequency_feature_name("kind")
        self.assertEqual(first[1][added].tolist(), [0.])
        self.assertEqual(second[1][added].tolist(), [0., 0.])
        self.assertNotEqual(first[3]["frequency"]["transform_sha256"],
                            second[3]["frequency"]["transform_sha256"])
        self.assertNotIn("counts", first[3]["frequency"]["columns"][0])

    def test_weights_use_raw_feature_groups_before_added_columns_and_ignore_index(self):
        frame = pd.DataFrame({"out": ["1"] * 20 + ["2", "3"], "in": ["0"] * 22,
                              "kind": ["x"] * 22}, index=["same"] * 22)
        predict = frame.iloc[:2].copy()
        bare = augment_features(frame, predict, context(),
                                {"frequency": False, "asymmetry_pairs": [], "group_weight": True})
        augmented = augment_features(frame, predict, context(),
                                     {"frequency": True, "asymmetry_pairs": [pair()], "group_weight": True})
        raw = np.array([.25] * 20 + [1., 1.])
        np.testing.assert_allclose(bare[4], raw / raw.mean())
        np.testing.assert_array_equal(bare[4], augmented[4])
        self.assertEqual(bare[3]["group_weight"]["group_count"], 3)
        self.assertEqual(bare[3]["group_weight"]["largest_group"], 20)
        shuffled = frame.sample(frac=1, random_state=2)
        shuffled_output = augment_features(shuffled, predict, context(),
                                          {"frequency": False, "asymmetry_pairs": [], "group_weight": True})
        # Distinct raw rows retain their weights under a positional reordering.
        expected = np.where(shuffled["out"].eq("1"), .25, 1.) / raw.mean()
        np.testing.assert_allclose(shuffled_output[4], expected)

    def test_asymmetry_handles_zero_invalid_and_extreme_values_without_overflow(self):
        frame = pd.DataFrame({"out": [0., 1., 0., 1e308, 1e308, -1., np.nan, np.inf, 1e-320],
                              "in": [0., 0., 1., 1e308, 5e307, 2., 3., 1., 2e-320]})
        ctx = context(["out", "in"], ["out", "in"], [])
        before = frame.copy(deep=True)
        output = augment_features(frame, frame.iloc[:2], ctx,
                                  {"frequency": False, "asymmetry_pairs": [pair()], "group_weight": False})
        values = output[0][asymmetry_feature_name(pair())].to_numpy()
        np.testing.assert_allclose(values[:5], [0., 1., -1., 0., 1 / 3])
        self.assertTrue(np.isnan(values[5:8]).all())
        self.assertAlmostEqual(values[8], -1 / 3)
        self.assertTrue((np.abs(values[np.isfinite(values)]) <= 1).all())
        self.assertEqual(output[3]["asymmetry"]["pairs"][0]["fit"]["invalid_rows"], 3)
        self.assertIsNone(output[4])
        assert_frame_equal(frame, before)
        json.dumps(output[3], allow_nan=False)

    def test_schema_rejects_technical_id_target_and_undeclared_pair_columns(self):
        frame = pd.DataFrame({"out": ["1"], "in": ["0"], "kind": ["x"]})
        with self.assertRaises(FeatureContractError):
            augment_features(frame.assign(technical_key="a"), frame, context())
        for field in ("id_column", "target_column"):
            ctx = context()
            ctx["task_spec"][field] = "out"
            with self.assertRaises(FeatureContractError):
                augment_features(frame, frame, ctx)
        with self.assertRaises(FeatureContractError):
            augment_features(frame, frame[["in", "out", "kind"]], context())
        for declaration in (pair("kind", "in"), pair("answer", "in"), pair("missing", "in")):
            with self.assertRaises(FeatureContractError):
                augment_features(frame, frame, context(),
                                 {"frequency": False, "asymmetry_pairs": [declaration], "group_weight": False})
        ctx = context()
        ctx["numeric"].append("kind")
        with self.assertRaises(FeatureContractError):
            augment_features(frame, frame, ctx)

    def test_options_are_fixed_bounded_and_pair_semantics_are_explicit(self):
        valid = {"frequency": False, "asymmetry_pairs": [pair()], "group_weight": False}
        normalized = normalize_options(valid)
        normalized["asymmetry_pairs"][0]["unit"] = "changed"
        self.assertEqual(valid["asymmetry_pairs"][0]["unit"], "bytes")
        invalid = [dict(valid, frequency=1), dict(valid, group_weight="yes"), dict(valid, extra=True),
                   dict(valid, asymmetry_pairs=[pair(), pair("in", "out")]),
                   dict(valid, asymmetry_pairs=[pair("out", "out")]),
                   dict(valid, asymmetry_pairs=[dict(pair(), unit="")]),
                   dict(valid, asymmetry_pairs=[{"forward": "out", "reverse": "in"}]),
                   dict(valid, asymmetry_pairs=[pair(str(i), str(i + 1)) for i in range(4)])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(FeatureContractError):
                normalize_options(value)

    def test_generated_feature_collisions_fail_before_augmentation(self):
        name = asymmetry_feature_name(pair())
        frame = pd.DataFrame({"out": [1.], "in": [0.], name: [0.]})
        ctx = context(list(frame.columns), list(frame.columns), [])
        with self.assertRaises(FeatureContractError):
            augment_features(frame, frame, ctx,
                             {"frequency": False, "asymmetry_pairs": [pair()], "group_weight": False})

    def test_identity_returns_detached_tables_and_context(self):
        frame = pd.DataFrame({"out": [1.], "in": [0.], "kind": [["nested"]]})
        ctx = context()
        output = augment_features(frame, frame, ctx)
        output[0].at[0, "kind"].append("mutated")
        output[1].at[0, "kind"].append("also_mutated")
        output[2]["task_spec"]["id_column"] = "changed"
        self.assertEqual(frame.at[0, "kind"], ["nested"])
        self.assertEqual(ctx, context())
        self.assertIsNone(output[4])
        self.assertEqual(output[3]["added_columns"], [])

    def test_frequency_with_no_categories_explicitly_skips(self):
        frame = pd.DataFrame({"out": [1.], "in": [0.]})
        with self.assertRaises(FrequencySkipError):
            augment_features(frame, frame, context(["out", "in"], ["out", "in"], []),
                             {"frequency": True, "asymmetry_pairs": [], "group_weight": False})


if __name__ == "__main__":
    unittest.main()
