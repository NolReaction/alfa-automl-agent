"""Pure policy invariants; quality and runtime are evaluated by the experiment runner."""

from copy import deepcopy
import json
import unittest

from agent.features import normalize_feature_config
from agent.model_config import ACTIONS, make_model_config, validate_model_config
from agent.search import exploration_plan, recipe_identity
from agent.search_policy import exploration_candidates, normalize_search_policy, refinement_candidates


class UnreadableTargets:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Classification policy consumed target values")

    def __iter__(self):
        raise AssertionError("Classification policy consumed target values")


class AdaptivePolicyTests(unittest.TestCase):
    def test_policy_names_are_explicit_and_legacy_remains_default(self):
        self.assertEqual(normalize_search_policy(), "legacy")
        self.assertEqual(normalize_search_policy("adaptive_v1"), "adaptive_v1")
        for value in (None, True, 1, "ADAPTIVE_V1", " adaptive_v1", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_search_policy(value)

    def test_all_nine_metrics_return_valid_finite_detached_recipes(self):
        for name in ("RMSE", "MAE", "MSE", "R2", "Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss"):
            classification = name in {"Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss"}
            with self.subTest(metric=name):
                recipes = exploration_candidates(classification, name, list(range(100)))
                self.assertGreaterEqual(len(recipes), 5)
                self.assertLessEqual(len(recipes), 12)
                self.assertEqual(len({recipe_identity(item) for item in recipes}), len(recipes))
                self.assertEqual([item["id"] for item in recipes], [f"aq_{i + 1}" for i in range(len(recipes))])
                json.dumps(recipes, allow_nan=False)
                for recipe in recipes:
                    validate_model_config(recipe["model_config"], action=recipe["action"])

    def test_first_pass_covers_five_families_before_repeating_one(self):
        for classification, metric in ((False, "MAE"), (False, "RMSE"), (True, "LogLoss"), (True, "Macro F1")):
            with self.subTest(metric=metric):
                recipes = exploration_candidates(classification, metric, list(range(100)))
                self.assertEqual({item["action"] for item in recipes[:5]}, set(ACTIONS))

    def test_mae_objectives_precede_controls_but_do_not_remove_squared_loss(self):
        recipes = exploration_candidates(False, "MAE", list(range(100)))
        early = {item["action"]: item for item in recipes[:5]}
        for action in ("baseline", "catboost", "lightgbm"):
            self.assertEqual(early[action]["model_config"]["objective"], "mae")
        baseline = [item for item in recipes if item["action"] == "baseline"]
        self.assertTrue(any("objective" not in item["model_config"] for item in baseline))

    def test_plain_linear_control_is_early_without_balanced_weights_or_feature_log(self):
        for classification, metric in ((True, "LogLoss"), (True, "ROC AUC"), (False, "RMSE")):
            with self.subTest(metric=metric):
                first = exploration_candidates(classification, metric, list(range(100)))[0]
                self.assertEqual(first["action"], "linear")
                self.assertEqual(first["feature_config"], normalize_feature_config())
                if classification:
                    self.assertIsNone(first["model_config"]["parameters"]["class_weight"])
                    self.assertEqual(first["model_config"]["parameters"]["max_iter"], 500)

    def test_classification_does_not_read_targets(self):
        recipes = exploration_candidates(True, "Macro F1", UnreadableTargets())
        self.assertEqual(recipes[0]["action"], "baseline")

    def test_log_transform_eligibility_uses_only_provided_fit_targets(self):
        regular = list(range(100))
        skewed = [1.0] * 98 + [100.0, 1000.0]
        plain = exploration_candidates(False, "MAE", regular)
        transformed = exploration_candidates(False, "MAE", skewed)
        negative = exploration_candidates(False, "MAE", [-1000.0, *skewed[1:]])
        has_log = lambda values: any(item["model_config"].get("target_transform") == "log1p" for item in values)
        self.assertFalse(has_log(plain))
        self.assertTrue(has_log(transformed))
        self.assertFalse(has_log(negative))

    def test_unavailable_packages_are_filtered_without_losing_sklearn_coverage(self):
        available = ("linear", "baseline", "extra_trees")
        recipes = exploration_candidates(False, "MAE", list(range(100)), available_actions=available)
        self.assertEqual({item["action"] for item in recipes[:3]}, set(available))
        self.assertTrue(all(item["action"] in available for item in recipes))

    def test_empty_available_actions_produce_no_candidates_without_consuming_targets(self):
        self.assertEqual(exploration_candidates(False, "MAE", UnreadableTargets(), available_actions=()), [])
        self.assertEqual(refinement_candidates(False, "MAE", None, existing=None, available_actions=()), [])

    def test_bad_profiles_and_action_declarations_are_rejected(self):
        for classification, metric in ((1, "RMSE"), (True, "MAE"), (False, "Macro F1"), (False, "unsupported"), (False, None)):
            with self.subTest(classification=classification, metric=metric), self.assertRaises(ValueError):
                exploration_candidates(classification, metric, list(range(100)))
        for available in (["linear"], ("linear", "linear"), ("unknown",), (True,)):
            with self.subTest(available=available), self.assertRaises(ValueError):
                exploration_candidates(False, "RMSE", list(range(100)), available_actions=available)

    def test_results_do_not_share_mutable_configuration(self):
        original = exploration_candidates(False, "MAE", list(range(100)))
        expected = deepcopy(original)
        original[0]["model_config"]["parameters"]["max_iter"] = 599
        original[0]["feature_config"]["missing_indicators"] = True
        self.assertEqual(exploration_candidates(False, "MAE", list(range(100))), expected)

    def test_expensive_medium_catboost_is_not_queued_after_shallow_portfolio(self):
        recipes = exploration_candidates(True, "Macro F1", UnreadableTargets())
        costly = [item for item in recipes if item["action"] == "catboost" and item["model_config"]["parameters"]["iterations"] >= 500]
        self.assertLessEqual(len(costly), 1)
        self.assertTrue(all(item["model_config"]["parameters"]["depth"] == 4 for item in costly))

    def test_refinements_alternate_top_two_available_families(self):
        initial = exploration_candidates(False, "RMSE", list(range(100)))
        ranked = [next(item for item in initial if item["action"] == action) for action in ("lightgbm", "linear", "catboost")]
        saved = deepcopy(ranked)
        refined = refinement_candidates(False, "RMSE", ranked, existing=initial)
        self.assertEqual([item["action"] for item in refined], ["lightgbm", "linear", "lightgbm", "linear"])
        self.assertEqual([item["id"] for item in refined], ["ar_1", "ar_2", "ar_3", "ar_4"])
        self.assertEqual(ranked, saved)
        self.assertFalse({recipe_identity(item) for item in refined} & {recipe_identity(item) for item in initial})

    def test_refinement_preserves_mae_objective_transform_and_features(self):
        recipe = {"action": "lightgbm", "model_config": make_model_config("lightgbm", "regression", {"n_estimators": 100}, objective="mae", target_transform="log1p"),
                  "feature_config": normalize_feature_config({"missing_indicators": True, "rare_min_count": 7}), "label": "test"}
        refined = refinement_candidates(False, "MAE", [recipe])
        self.assertEqual(len(refined), 2)
        for item in refined:
            self.assertEqual(item["model_config"]["objective"], "mae")
            self.assertEqual(item["model_config"]["target_transform"], "log1p")
            self.assertEqual(item["feature_config"], recipe["feature_config"])

    def test_refinement_skips_unavailable_top_rank_without_blocking_other_families(self):
        ranked = exploration_plan(False)
        ranked.sort(key=lambda item: item["action"] != "catboost")
        refined = refinement_candidates(False, "RMSE", ranked, available_actions=("linear", "baseline"))
        self.assertEqual({item["action"] for item in refined}, {"linear", "baseline"})

    def test_existing_identity_ignores_labels_and_deduplicates_refinements(self):
        ranked = exploration_candidates(False, "RMSE", list(range(100)))[:1]
        first = refinement_candidates(False, "RMSE", ranked)
        existing = [{**item, "id": "different", "label": "presentation"} for item in first]
        second = refinement_candidates(False, "RMSE", ranked, existing=existing)
        self.assertFalse({recipe_identity(item) for item in second} & {recipe_identity(item) for item in first})

    def test_refinement_rejects_malformed_foreign_or_unbounded_input(self):
        foreign = exploration_candidates(True, "LogLoss", UnreadableTargets())[0]
        invalid_label = {**exploration_candidates(False, "RMSE", list(range(100)))[0], "label": float("nan")}
        for ranked in ([foreign], [invalid_label], [{"action": "unknown"}], "not-recipes", None, [{}] * 257):
            with self.subTest(type=type(ranked).__name__), self.assertRaises(ValueError):
                refinement_candidates(False, "RMSE", ranked)


if __name__ == "__main__":
    unittest.main()
