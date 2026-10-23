"""An inner-selected ensemble is frozen before external model verification."""

import unittest
from unittest.mock import patch

import test_automl as fixtures
from agent import adaptive_search, weighted_ensemble


class WeightedAutoMLIntegrationTests(unittest.TestCase):
    setUp = fixtures.AutoMLTests.setUp
    make_data = fixtures.AutoMLTests.make_data
    fake_execution = fixtures.AutoMLTests.fake_execution
    run_search = fixtures.AutoMLTests.run_search

    def run_weighted(self, left_outer, right_outer, *, fail_right=False):
        outcomes = {
            "selection": {"error": 10},
            ("candidate_1", "selection"): {"error": 1},
            ("candidate_2", "selection"): {"error": -2},
            ("candidate_1", "verification"): {"error": left_outer},
            ("candidate_2", "verification"): {"error": right_outer,
                                                  "exit_code": 1 if fail_right else 0},
        }
        select = weighted_ensemble.select_weighted_ensemble

        def before_outer(spec, actual, candidates, **kwargs):
            self.assertFalse(any(call["phase"] == "verification" for call in self.calls))
            first = next(call for call in self.calls if call["phase"] == "selection")
            outer = next(call for call in self.calls if call["phase"] == "seed")
            self.assertFalse(set(first["validation_rows"]) & set(outer["validation_rows"]))
            self.assertEqual(len(actual), len(first["validation_rows"]))
            return select(spec, actual, candidates, **kwargs)

        with patch.object(adaptive_search, "_load_optuna", return_value=None), \
                patch.object(weighted_ensemble, "select_weighted_ensemble", side_effect=before_outer):
            return self.run_search(outcomes, search_strategy="adaptive", adaptive_max_trials=1)

    def test_changing_outer_scores_cannot_change_component_ids_or_weights(self):
        first = self.run_weighted(3, -1)
        first_choice = first["weighted_ensemble"]
        first_blend = next(row for row in first["search"]["candidates"] if row["id"] == "weighted_blend")
        self.assertEqual(first_blend["rejection_reason"], "not_improved")
        self.assertNotEqual(first["model"], "WeightedBlend")
        self.calls.clear()
        second = self.run_weighted(.5, -2)
        self.assertEqual(second["weighted_ensemble"], first_choice)
        self.assertEqual(second["model"], "WeightedBlend")
        self.assertEqual(first_choice["selected_ids"], ["candidate_1", "candidate_2"])
        self.assertEqual(first_choice["weights"], [.7, .3])
        self.assertAlmostEqual(second["metric"]["value"], .25)
        refits = [call for call in self.calls if call["phase"] == "refit"]
        self.assertEqual(len(refits), 1)
        self.assertEqual(refits[0]["model_config"], second["model_config"])
        self.assertEqual([item["id"] for item in refits[0]["components"]], first_choice["selected_ids"])

    def test_missing_outer_component_rejects_frozen_blend_without_retuning(self):
        result = self.run_weighted(.5, -2, fail_right=True)
        choice = result["weighted_ensemble"]
        self.assertTrue(choice["improved"])
        self.assertEqual(choice["selected_ids"], ["candidate_1", "candidate_2"])
        blend = next(row for row in result["search"]["candidates"] if row["id"] == "weighted_blend")
        self.assertEqual(blend["status"], "skipped")
        self.assertEqual(blend["rejection_reason"], "invalid_evidence")
        self.assertIsNone(blend["metric"])
        self.assertNotEqual(result["model"], "WeightedBlend")
        self.assertFalse(any(call["phase"] == "refit" and call["id"] == "weighted_blend" for call in self.calls))


if __name__ == "__main__":
    unittest.main()
