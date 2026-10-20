"""Fit-only metric candidates and bounded distinct local alternatives."""

import unittest

import numpy as np

from agent.model_config import validate_model_config
from agent.search import (exploration_plan, local_candidate_plan, metric_candidates,
                          recipe_identity, refinement_plan)


class SearchQualityTests(unittest.TestCase):
    def test_mae_adds_objectives_without_changing_untransformed_defaults(self):
        fit = np.linspace(-2, 3, 100)
        candidates = metric_candidates("MAE", fit)
        self.assertEqual(len(candidates), 3)
        self.assertEqual({item["action"] for item in candidates}, {"baseline", "catboost", "lightgbm"})
        for item in candidates:
            self.assertEqual(item["model_config"]["objective"], "mae")
            self.assertNotIn("target_transform", item["model_config"])
        self.assertFalse(any("objective" in item["model_config"] for item in exploration_plan(False)))
        self.assertEqual(metric_candidates("MAE", fit, existing=candidates), [])

    def test_log_is_only_proposed_for_nonnegative_finite_skewed_fit_targets(self):
        skew = np.r_[np.ones(95), [100, 200, 300, 400, 500]]
        self.assertEqual(len(metric_candidates("RMSE", skew)), 3)
        for target in (skew - 2, np.arange(100), np.r_[skew[:-1], np.nan], skew[:19]):
            self.assertEqual(metric_candidates("RMSE", target), [])
        self.assertEqual(metric_candidates("Macro F1", skew), [])

    def test_refinement_preserves_metric_options_and_local_queue_deduplicates(self):
        candidates = metric_candidates("MAE", np.arange(100))
        refined = refinement_plan(False, candidates, existing=candidates)
        self.assertTrue(refined)
        self.assertTrue(all(item["model_config"]["objective"] == "mae" for item in refined))
        for classification in (False, True):
            queue = local_candidate_plan(classification, "Macro F1" if classification else "MAE", np.arange(100))
            self.assertLessEqual(len(queue), 32)
            self.assertEqual(len(queue), len({recipe_identity(item) for item in queue}))
            self.assertGreater(len(queue), len(exploration_plan(classification)))
            for item in queue:
                self.assertEqual(validate_model_config(item["model_config"], action=item["action"]), item["model_config"])

    def test_local_metric_options_require_explicit_opt_in(self):
        default = local_candidate_plan(False, "MAE", np.arange(100))
        enabled = local_candidate_plan(False, "MAE", np.arange(100), metric_recipes=True)
        self.assertFalse(any("objective" in item["model_config"] for item in default))
        self.assertTrue(any("objective" in item["model_config"] for item in enabled))
