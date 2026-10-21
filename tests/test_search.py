"""Inner-only adaptation and deterministic class-aligned fixed ensembles."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from agent.features import normalize_feature_config
from agent.model_config import validate_model_config
from agent.search import (exploration_plan, fixed_blend, inner_split_rows, metric_predictions,
                          refinement_plan, seed_candidate, select_shortlist)
from agent.splits import split_rows


class SearchTests(unittest.TestCase):
    def test_outer_shortlist_cannot_be_monopolized_by_one_inner_winning_family(self):
        ranked = [{"id": str(index), "action": family} for index, family in enumerate(
            ("lightgbm", "lightgbm", "lightgbm", "extra_trees", "catboost", "baseline"))]
        selected = select_shortlist(ranked)
        self.assertEqual([item["id"] for item in selected], ["0", "3", "4"])
        self.assertEqual(len(selected), 3)
        # The policy is symmetric: reversing family names gives no preference
        # to a baseline or to a model known to work on one public dataset.
        for item in ranked:
            item["action"] = {"lightgbm": "linear", "extra_trees": "baseline",
                              "catboost": "extra_trees", "baseline": "catboost"}[item["action"]]
        self.assertEqual([item["id"] for item in select_shortlist(ranked)], ["0", "3", "4"])

    def test_shortlist_fills_spare_slots_by_inner_rank_only_when_families_are_few(self):
        ranked = [{"id": str(index), "action": family} for index, family in enumerate(
            ("catboost", "catboost", "catboost", "linear"))]
        self.assertEqual([item["id"] for item in select_shortlist(ranked)], ["0", "3", "1"])
        self.assertEqual(select_shortlist([]), [])
        self.assertEqual(select_shortlist(ranked[:1]), ranked[:1])
        self.assertEqual(select_shortlist(ranked, limit=1), ranked[:1])
        for limit in (0, 4, True):
            with self.assertRaises(ValueError):
                select_shortlist(ranked, limit)

    def test_exploration_and_adaptive_spaces_are_valid_bounded_and_family_dependent(self):
        for classification in (False, True):
            initial = exploration_plan(classification)
            self.assertEqual(len(initial), 8)
            self.assertEqual(len({item["action"] for item in initial}), 5)
            ranked = [next(item for item in initial if item["action"] == name)
                      for name in ("lightgbm", "linear", "baseline")]
            extra = refinement_plan(classification, ranked, initial)
            self.assertEqual(len(extra), 6)
            self.assertEqual({item["action"] for item in extra}, {"lightgbm", "linear"})
            for item in [seed_candidate(classification), *initial, *extra]:
                self.assertEqual(validate_model_config(item["model_config"], action=item["action"]), item["model_config"])
            reranked = [next(item for item in initial if item["action"] == name)
                        for name in ("catboost", "extra_trees")]
            self.assertEqual({item["action"] for item in refinement_plan(classification, reranked, initial)},
                             {"catboost", "extra_trees"})

    def test_feature_variants_are_real_and_refinement_preserves_winning_family_features(self):
        initial = exploration_plan(False)
        cat = initial[4]
        self.assertTrue(cat["feature_config"]["missing_indicators"])
        self.assertEqual(cat["feature_config"]["rare_min_count"], 5)
        self.assertTrue(initial[7]["feature_config"]["signed_log"])
        refined = refinement_plan(False, [cat, initial[7]], initial)
        for item in refined:
            expected = cat["feature_config"] if item["action"] == "catboost" else initial[7]["feature_config"]
            self.assertEqual(item["feature_config"], expected)

    def test_warmup_boosts_are_cheap_and_refinement_keeps_chosen_class_weights(self):
        initial = exploration_plan(True)
        cat, light = initial[4], initial[6]
        self.assertEqual(cat["model_config"]["parameters"]["iterations"], 100)
        self.assertEqual(light["model_config"]["parameters"]["n_estimators"], 100)
        refined = refinement_plan(True, [cat, light], initial)
        for item in refined:
            parameters = item["model_config"]["parameters"]
            if item["action"] == "catboost":
                self.assertEqual(parameters["auto_class_weights"], "SqrtBalanced")
                self.assertGreater(parameters["iterations"], 100)
            else:
                self.assertEqual(parameters["class_weight"], "balanced")
                self.assertGreater(parameters["n_estimators"], 100)

    def test_refinement_deduplicates_model_and_feature_configuration(self):
        initial = exploration_plan(False)
        extra = refinement_plan(False, initial, initial)
        self.assertEqual(refinement_plan(False, initial, initial + extra), [])
        self.assertEqual(refinement_plan(False, [], initial), [])

    def test_inner_time_split_never_sees_outer_period(self):
        train = pd.DataFrame({"year": [1, 1, 2, 2, 3, 3, 4, 4], "x": range(8)})
        y = pd.Series(range(8))
        choice = {"strategy": "time", "time_column": "year"}
        outer, held, _ = split_rows(train, y, ["year", "x"], False, choice)
        inner, selection, metadata = inner_split_rows(train, y, ["year", "x"], False, choice, outer)
        self.assertEqual(train.iloc[selection]["year"].unique().tolist(), [3])
        self.assertTrue((train.iloc[inner]["year"] < 3).all())
        self.assertFalse(set(held) & set(selection))
        self.assertEqual(set(inner) | set(selection), set(outer))
        self.assertEqual(metadata["validation_period"], "3")

    def test_impossible_inner_time_split_raises_without_touching_outer_holdout(self):
        train = pd.DataFrame({"year": [1, 1, 2, 2]})
        y = pd.Series(range(4))
        choice = {"strategy": "time", "time_column": "year"}
        outer, _, _ = split_rows(train, y, ["year"], False, choice)
        with self.assertRaises(ValueError):
            inner_split_rows(train, y, ["year"], False, choice, outer)

    def test_inner_duplicate_groups_are_kept_separate_and_indices_are_global(self):
        train = pd.DataFrame({"x": np.repeat(np.arange(40), 2)})
        y = pd.Series(np.repeat(["a", "b"] * 20, 2))
        choice = {"strategy": "grouped"}
        outer, held, _ = split_rows(train, y, ["x"], True, choice)
        fitting, selection, _ = inner_split_rows(train, y, ["x"], True, choice, outer)
        self.assertFalse(set(train.iloc[fitting]["x"]) & set(train.iloc[selection]["x"]))
        self.assertFalse(set(train.iloc[selection]["x"]) & set(train.iloc[held]["x"]))
        self.assertEqual(set(fitting) | set(selection), set(outer))
        self.assertEqual(set(y.iloc[fitting]), {"a", "b"})
        self.assertEqual(set(y.iloc[selection]), {"a", "b"})

    def test_regression_blend_is_fixed_mean_with_no_weight_fitting(self):
        spec = SimpleNamespace(classification=False)
        np.testing.assert_array_equal(fixed_blend(spec, [1, 4, -3], [3, 2, 1]), [2, 3, -1])
        for right in ([1, 2], [np.nan, 2, 3], [[1], [2], [3]]):
            with self.assertRaises(ValueError):
                fixed_blend(spec, [1, 2, 3], right)

    def test_class_probability_blend_aligns_literal_classes_before_averaging(self):
        spec = SimpleNamespace(classification=True, prediction_kind="labels", positive_class=None,
                               classes=("001", "NA", "tail"))
        left = [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
        # Right's order is tail, 001, NA, deliberately unrelated to output order.
        right = [[0.2, 0.6, 0.2], [0.8, 0.1, 0.1]]
        blended = fixed_blend(spec, left, right, left_classes=spec.classes, right_classes=("tail", "001", "NA"))
        np.testing.assert_allclose(blended, [[0.65, 0.2, 0.15], [0.1, 0.15, 0.75]])
        self.assertEqual(metric_predictions(spec, blended).tolist(), ["001", "tail"])
        spec.prediction_kind = "probabilities"
        np.testing.assert_array_equal(metric_predictions(spec, blended), blended)

    def test_probability_blend_rejects_missing_classes_invalid_rows_and_nonfinite_values(self):
        spec = SimpleNamespace(classification=True, prediction_kind="probabilities", positive_class="1", classes=("0", "1"))
        invalid = [([[1]], ("1",)), ([[0.8, 0.8]], ("0", "1")),
                   ([[float("nan"), 0.4]], ("0", "1")), ([[0.4, 0.6]], ("0", "unexpected"))]
        for values, classes in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                fixed_blend(spec, [[0.4, 0.6]], values, left_classes=spec.classes, right_classes=classes)


if __name__ == "__main__":
    unittest.main()
