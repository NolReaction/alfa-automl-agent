"""Data-boundary tests; no model fits or API calls."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmark_cases import write_case
from experiment_portfolio_recipes import comparison_dataset, freeze_splits
from experiment_strong_tabular import _full_refit_evidence, public_case, run


class StrongBenchmarkTests(unittest.TestCase):
    def test_public_classification_held_answers_are_not_written_to_solver(self):
        data = comparison_dataset(builtin="wine")
        split = freeze_splits(data, seeds=(31415,), outer_folds=(0,))[0]
        case = public_case(data, split, "wine")
        with tempfile.TemporaryDirectory() as temporary:
            task = write_case(case, Path(temporary) / "solver")
            held = pd.read_csv(task.parent / "inputs/test.csv", dtype=str, keep_default_na=False)
            self.assertNotIn("outcome", held.columns)
            self.assertEqual(len(held), len(case.audit_actuals))
            self.assertEqual("Macro F1", case.metric)
            self.assertEqual({"strategy": "grouped"}, case.validation)

    def test_rejects_raw_duplicate_group_crossing_outer_boundary(self):
        data = comparison_dataset(builtin="wine")
        split = freeze_splits(data, seeds=(31415,), outer_folds=(0,))[0]
        left, right = split["outer_fit"][0], split["outer_held"][0]
        data["X"].iloc[right] = data["X"].iloc[left]
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            public_case(data, split, "wine")

    def test_temporal_public_case_rejects_future_training_rows(self):
        data = {"features": ["period", "measurement"], "numeric": ["period", "measurement"],
                "categorical": [], "X": pd.DataFrame({"period": [1, 2, 3, 4], "measurement": [5, 6, 7, 8]}),
                "ids": ["a", "b", "c", "d"], "y": np.asarray([1., 2., 3., 4.]),
                "profile": {"task_type": "regression", "metric": "RMSE"},
                "classes": (), "positive_class": None, "time_column": "period"}
        split = {"status": "ready", "outer_fit": [0, 1], "outer_held": [2, 3], "seed": 0, "id": "years"}
        case = public_case(data, split, "regression")
        self.assertEqual([1, 2], case.train["period"].tolist())
        broken = deepcopy(split)
        broken.update(outer_fit=[0, 3], outer_held=[1, 2])
        with self.assertRaisesRegex(ValueError, "past-only"):
            public_case(data, broken, "regression")

    def test_missing_checkpoint_never_counts_as_full_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = _full_refit_evidence(Path(temporary) / "task.md", {"output_valid": True})
        self.assertFalse(evidence["valid"])

    def test_dirty_solver_is_rejected_before_output_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            with patch("experiment_strong_tabular.benchmark.source_provenance", return_value={"worktree_dirty": True}):
                with self.assertRaisesRegex(ValueError, "clean immutable"):
                    run(output, solver_root=Path(temporary), cases=("linear_regression",))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
