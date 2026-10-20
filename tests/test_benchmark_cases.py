"""Deterministic synthetic benchmark inputs and audit-answer boundaries."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchmark_cases import AUDIT_ROWS, CASE_NAMES, TRAIN_ROWS, make_case, write_case
from agent.search import inner_split_rows
from agent.splits import split_rows
from agent.task_spec import load_task, load_task_data
from offline_baseline import read_csv


class BenchmarkCaseTests(unittest.TestCase):
    def test_all_six_cases_have_explicit_loadable_contracts_and_complete_inner_splits(self):
        self.assertEqual(len(CASE_NAMES), 6)
        with tempfile.TemporaryDirectory() as directory:
            for name in CASE_NAMES:
                with self.subTest(case=name):
                    case = make_case(name, 42)
                    task = write_case(case, Path(directory) / name)
                    spec = load_task(task)
                    train, test, *_, y, features, numeric, categorical, classification = load_task_data(spec)
                    self.assertEqual(len(train), TRAIN_ROWS)
                    self.assertEqual(len(test), AUDIT_ROWS)
                    self.assertEqual(len(case.audit_actuals), AUDIT_ROWS)
                    self.assertEqual(spec.metric.name, case.metric)
                    self.assertEqual(spec.classes, case.classes)
                    self.assertEqual(spec.positive_class, case.positive_class)
                    self.assertEqual(set(numeric), set(case.numeric))
                    self.assertEqual(set(categorical), set(case.categorical))
                    self.assertNotIn("outcome", test)
                    self.assertEqual(test["row_key"].tolist()[:3], ["000005", "NA", "audit c"])
                    fit, held, _ = split_rows(train, y, features, classification, case.validation)
                    inner_fit, inner_held, _ = inner_split_rows(
                        train, y, features, classification, case.validation, fit)
                    self.assertFalse(set(fit) & set(held))
                    self.assertFalse(set(inner_fit) & set(inner_held))
                    self.assertTrue(set(inner_fit) | set(inner_held) <= set(fit))

    def test_same_case_and_seed_produce_identical_bytes_and_audit_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in CASE_NAMES:
                with self.subTest(case=name):
                    first, second = make_case(name, 137), make_case(name, 137)
                    self.assertEqual(first.audit_actuals, second.audit_actuals)
                    roots = [Path(directory) / (name + suffix) for suffix in ("-one", "-two")]
                    for case, root in zip((first, second), roots):
                        write_case(case, root)
                    files = ["task.md", *["inputs/" + file for file in
                             ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")]]
                    for filename in files:
                        self.assertEqual((roots[0] / filename).read_bytes(), (roots[1] / filename).read_bytes())
                    self.assertNotEqual(first.audit_actuals, make_case(name, 2026).audit_actuals)

    def test_changing_only_audit_answers_cannot_change_any_written_solver_input(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in CASE_NAMES:
                with self.subTest(case=name):
                    case = make_case(name, 42)
                    replacement = tuple("host-only-canary" for _ in case.audit_actuals)
                    roots = [Path(directory) / (name + suffix) for suffix in ("-a", "-b")]
                    write_case(case, roots[0])
                    write_case(replace(case, audit_actuals=replacement), roots[1])
                    paths = sorted(path.relative_to(roots[0]) for path in roots[0].rglob("*") if path.is_file())
                    self.assertEqual(set(map(str, paths)), {"task.md", "inputs/train.csv", "inputs/test.csv",
                                     "inputs/data_dictionary.csv", "inputs/sample_submission.csv"})
                    for relative in paths:
                        self.assertEqual((roots[0] / relative).read_bytes(), (roots[1] / relative).read_bytes())
                        self.assertNotIn(b"host-only-canary", (roots[1] / relative).read_bytes())

    def test_temporal_audit_is_strictly_future_and_inner_outer_partitions_are_past_only(self):
        case = make_case("temporal_drift", 42)
        self.assertEqual(set(case.train["period"]), {1, 2, 3, 4, 5})
        self.assertEqual(set(case.test["period"]), {6})
        features = list(case.numeric + case.categorical)
        fit, held, _ = split_rows(case.train, case.train["outcome"], features, False, case.validation)
        self.assertLess(case.train["period"].iloc[fit].max(), case.train["period"].iloc[held].min())
        inner_fit, inner_held, _ = inner_split_rows(
            case.train, case.train["outcome"], features, False, case.validation, fit)
        self.assertLess(case.train["period"].iloc[inner_fit].max(), case.train["period"].iloc[inner_held].min())

    def test_nonlinear_case_uses_mae_for_nonnegative_skewed_outcomes(self):
        for seed in (42, 137, 2026):
            with self.subTest(seed=seed):
                case = make_case("nonlinear_regression", seed)
                values = np.concatenate((case.train["outcome"].to_numpy(), case.audit_actuals))
                self.assertEqual(case.metric, "MAE")
                self.assertTrue((values >= 0).all())
                self.assertGreater(np.quantile(values, .99), 3 * np.median(values))

    def test_duplicate_groups_are_real_and_do_not_overlap_audit_or_grouped_partitions(self):
        case = make_case("multiclass_duplicates", 42)
        features = list(case.numeric + case.categorical)
        groups = pd.util.hash_pandas_object(case.train[features], index=False).to_numpy()
        audit_groups = pd.util.hash_pandas_object(case.test[features], index=False).to_numpy()
        self.assertEqual(len(np.unique(groups)), TRAIN_ROWS // 3)
        self.assertFalse(set(groups) & set(audit_groups))
        fit, held, _ = split_rows(case.train, case.train["outcome"], features, True, case.validation)
        self.assertFalse(set(groups[fit]) & set(groups[held]))
        # Conflicting labels cannot justify splitting a duplicate feature group.
        self.assertTrue(any(len(set(case.train.loc[groups == group, "outcome"])) > 1
                            for group in np.unique(groups)))

    def test_probability_schema_preserves_literal_labels_and_unknown_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = make_case("imbalanced_binary", 2026)
            task = write_case(binary, Path(directory) / "binary")
            spec = load_task(task)
            self.assertEqual(spec.positive_class, "01")
            self.assertEqual(spec.probability_columns, (("outcome", "01"),))
            self.assertLess((binary.train["outcome"] == "01").mean(), .3)
            multiclass = make_case("categorical_shift", 2026)
            task = write_case(multiclass, Path(directory) / "multiclass")
            spec = load_task(task)
            self.assertEqual(spec.classes, ("01", "NA", "class c"))
            self.assertEqual(spec.output_columns, ("row_key", "class c", "01", "NA"))
            self.assertTrue(set(multiclass.test["entity"]) - set(multiclass.train["entity"]))
            self.assertIn("NA", set(read_csv(task.parent / "inputs/train.csv")["outcome"]))

    def test_invalid_seed_case_and_existing_destination_are_rejected(self):
        for value in (-1, 2 ** 32, True, "42", 42.0):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "seed"):
                make_case("linear_regression", value)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            make_case("not-a-case", 42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task"
            write_case(make_case("linear_regression", 42), root)
            original = (root / "inputs/train.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                write_case(make_case("linear_regression", 137), root)
            self.assertEqual((root / "inputs/train.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
