"""Contract/selection tests for the optional stronger CPU catalogue."""

from __future__ import annotations

import unittest

from agent.features import normalize_feature_config
from agent.metrics import registered_metrics
from agent.model_config import validate_model_config
from agent.search import exploration_plan, recipe_identity
from agent.tabular_backend import strong_candidate_plan


class StrongTabularTests(unittest.TestCase):
    def test_all_registered_metrics_keep_their_task_and_validated_controls(self):
        for metric in registered_metrics():
            with self.subTest(metric=metric.name):
                plan = strong_candidate_plan(metric.classification, metric.name)
                self.assertEqual(8, len(plan))
                self.assertEqual(len(plan), len({item["id"] for item in plan}))
                self.assertEqual(len(plan), len({recipe_identity(item) for item in plan}))
                for item in plan:
                    self.assertEqual(item["model_config"], validate_model_config(item["model_config"], action=item["action"]))
                    self.assertEqual(normalize_feature_config(), item["feature_config"])
                    self.assertNotIn("target_transform", item["model_config"])

    def test_mae_is_a_training_objective_without_changing_scoring_contract(self):
        plan = strong_candidate_plan(False, "MAE")
        for item in plan:
            if item["action"] in {"baseline", "catboost", "lightgbm"}:
                self.assertEqual("mae", item["model_config"]["objective"])
            else:
                self.assertNotIn("objective", item["model_config"])
        self.assertTrue(all("objective" not in item["model_config"]
                            for item in strong_candidate_plan(False, "RMSE")))

    def test_balancing_is_a_macro_f1_candidate_not_a_probability_default(self):
        macro = strong_candidate_plan(True, "Macro F1")
        self.assertEqual(1, sum(item["model_config"]["parameters"].get("class_weight") == "balanced"
                                for item in macro if item["action"] == "lightgbm"))
        for name in ("LogLoss", "ROC AUC", "Accuracy", "Weighted F1"):
            self.assertTrue(all(item["model_config"]["parameters"].get("class_weight") != "balanced"
                                for item in strong_candidate_plan(True, name) if item["action"] == "lightgbm"))

    def test_returns_detached_recipes_and_never_reads_targets(self):
        class ForbiddenTargets:
            def __iter__(self):
                raise AssertionError("The catalogue must not inspect targets")
        original = strong_candidate_plan(False, "RMSE", ForbiddenTargets())
        original[1]["model_config"]["parameters"]["n_estimators"] = 50
        original[1]["feature_config"]["date_columns"].append("not_a_real_column")
        again = strong_candidate_plan(False, "RMSE")
        self.assertEqual(800, again[1]["model_config"]["parameters"]["n_estimators"])
        self.assertEqual([], again[1]["feature_config"]["date_columns"])

    def test_legacy_catalogue_is_unchanged(self):
        for classification in (False, True):
            before = exploration_plan(classification)
            strong_candidate_plan(classification, "Macro F1" if classification else "RMSE")
            self.assertEqual(before, exploration_plan(classification))
            self.assertEqual(100, next(item for item in before if item["action"] == "lightgbm")["model_config"]["parameters"]["n_estimators"])

    def test_rejects_mismatched_or_ambiguous_task_flags(self):
        for classification, metric in ((False, "Accuracy"), (True, "RMSE"), (1, "Accuracy"), ("false", "RMSE")):
            with self.subTest(classification=classification, metric=metric), self.assertRaises(ValueError):
                strong_candidate_plan(classification, metric)


if __name__ == "__main__":
    unittest.main()
