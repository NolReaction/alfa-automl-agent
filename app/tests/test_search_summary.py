"""The expanded bounded plan remains valid across application report boundaries."""

import unittest

from app.model_config import make_model_config
from app.search_summary import validate_search_summary


def completed_search(count):
    rows = [{"id": f"candidate-{index}", "label": f"Recipe {index}", "model": "Ridge",
             "model_config": make_model_config("linear", "regression", {"alpha": index + 1}),
             "status": "completed", "metric": float(index), "selected": index == 0,
             "elapsed_seconds": 1.0, "error": None} for index in range(count)]
    return {"status": "completed", "planned": count, "completed": count,
            "best_candidate": "candidate-0", "validation_reason": "Fixed holdout", "candidates": rows}


class SearchSummaryBoundTests(unittest.TestCase):
    def test_full_twenty_two_recipe_plan_is_accepted_without_truncation(self):
        summary = completed_search(22)
        validated = validate_search_summary(summary, metric={"name": "MAE", "value": 0.0,
                                            "higher_is_better": False}, model="Ridge")
        self.assertEqual(validated, summary)
        self.assertEqual(len(validated["candidates"]), 22)
        self.assertEqual(validated["candidates"][-1]["id"], "candidate-21")

    def test_twenty_three_recipe_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "число попыток"):
            validate_search_summary(completed_search(23))


if __name__ == "__main__":
    unittest.main()
