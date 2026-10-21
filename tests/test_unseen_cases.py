"""Pinned fixture provenance, supported schemas and audit-flow boundaries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import unseen_cases as cases
from agent.task_spec import load_task


class UnseenCaseTests(unittest.TestCase):
    def test_catalogue_covers_supported_metrics_cases_and_explicit_cost(self):
        catalog = cases.case_catalog()
        self.assertEqual(len(catalog), len({row["id"] for row in catalog}))
        self.assertGreaterEqual(sum(row["kind"] == "valid" for row in catalog), 18)
        self.assertGreaterEqual(sum(row["kind"] == "negative" for row in catalog), 6)
        self.assertGreaterEqual(sum(row["kind"] == "scale" for row in catalog), 3)
        metrics = {row["metric"] for row in catalog if row["full_run"] and row["kind"] != "negative"}
        self.assertEqual(metrics, {"RMSE", "MAE", "MSE", "R2", "Accuracy", "Macro F1",
                                   "Weighted F1", "ROC AUC", "LogLoss"})
        self.assertEqual(sum(row["default_cli"] for row in catalog), 3)
        self.assertTrue(all(row["timeout_seconds"] > 0 and row["offline_seconds"] > 0 for row in catalog))
        json.dumps(catalog, allow_nan=False)
        catalog[0]["validation"]["strategy"] = "corrupted"
        self.assertEqual(cases.case_catalog()[0]["validation"], {"strategy": "random"})

    def test_all_generators_are_deterministic_and_metadata_row_counts_are_exact(self):
        for definition in cases.case_catalog():
            with self.subTest(case=definition["id"]):
                first = cases.generate_case(definition["id"])
                second = cases.generate_case(definition["id"])
                pd.testing.assert_frame_equal(first.train, second.train)
                pd.testing.assert_frame_equal(first.test, second.test)
                self.assertEqual(first.audit_actuals, second.audit_actuals)
                self.assertEqual(first.task_text, second.task_text)
                self.assertEqual(len(first.train), definition["train_rows"])
                self.assertEqual(len(first.test), definition["audit_rows"])
                self.assertEqual(len(first.audit_actuals), len(first.test))
                self.assertEqual(first.seed, cases.DEFAULT_SEED)
                if first.contract["classes"]:
                    self.assertEqual(set(first.train[first.contract["target_column"]]),
                                     set(first.contract["classes"]))

    def test_materialization_writes_only_solver_inputs_and_preserves_literal_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = cases.materialize_case("probability_columns_permuted_logloss", root / "a")
            second = cases.materialize_case("probability_columns_permuted_logloss", root / "b")
            self.assertEqual(first["input_sha256"], second["input_sha256"])
            expected = {"task.md", "inputs/train.csv", "inputs/test.csv",
                        "inputs/data_dictionary.csv", "inputs/sample_submission.csv"}
            self.assertEqual(set(first["input_paths"]), expected)
            self.assertEqual({path.relative_to(root / "a").as_posix() for path in (root / "a").rglob("*") if path.is_file()}, expected)
            test = pd.read_csv(root / "a/inputs/test.csv", dtype=str, keep_default_na=False)
            template = pd.read_csv(root / "a/inputs/sample_submission.csv", dtype=str, keep_default_na=False)
            contract = first["contract"]
            self.assertNotIn(contract["target_column"], test.columns)
            self.assertEqual(test[contract["id_column"]].tolist(), first["test_ids"])
            self.assertEqual(template[contract["id_column"]].tolist(), first["test_ids"])
            for literal in ["000007", "NA", "строка 4", "id,comma", 'id "quoted"', " id padded "]:
                self.assertIn(literal, first["test_ids"])
            self.assertEqual(list(template.columns), contract["output_columns"])
            self.assertFalse(any("audit" in name or "answer" in name for name in first["input_paths"]))
            for name, digest in first["input_sha256"].items():
                self.assertEqual(hashlib.sha256((root / "a" / name).read_bytes()).hexdigest(), digest)
            self.assertNotIn(str(root), (root / "a/task.md").read_text(encoding="utf-8"))

    def test_all_positive_contracts_match_the_declared_supported_task_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            for definition in cases.case_catalog():
                if definition["kind"] == "negative":
                    continue
                with self.subTest(case=definition["id"]):
                    materialized = cases.materialize_case(definition["id"], Path(temporary) / definition["id"])
                    spec = load_task(Path(materialized["task_path"]))
                    contract = materialized["contract"]
                    self.assertEqual(spec.metric.name, definition["metric"])
                    self.assertEqual(list(spec.output_columns), contract["output_columns"])
                    self.assertEqual(list(spec.classes), contract["classes"])
                    self.assertEqual([list(pair) for pair in spec.probability_columns], contract["probability_columns"])
                    self.assertEqual(spec.positive_class, contract["positive_class"])
                    self.assertEqual(spec.prediction_kind, contract["prediction_kind"])

    def test_negative_cases_are_actual_contract_refusals_not_unlucky_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            for definition in cases.case_catalog():
                if definition["kind"] != "negative":
                    continue
                with self.subTest(case=definition["id"]):
                    materialized = cases.materialize_case(definition["id"], Path(temporary) / definition["id"])
                    self.assertEqual(materialized["expected_status"], "refusal")
                    with self.assertRaises((ValueError, csv.Error)):
                        load_task(Path(materialized["task_path"]))

    def test_grouped_and_temporal_cases_have_the_intended_structure(self):
        grouped = cases.generate_case("grouped_logic_macro_f1")
        features = grouped.train.drop(columns=["record id", "response"])
        counts = features.value_counts(dropna=False)
        self.assertTrue((counts == 4).all())
        self.assertTrue(grouped.train.groupby(list(features.columns), dropna=False)["response"].nunique().gt(1).any())
        temporal = cases.generate_case("future_inventory_mae")
        self.assertEqual(set(temporal.train["period"]), {1, 2, 3, 4, 5})
        self.assertEqual(set(temporal.test["period"]), {6})
        self.assertLess(temporal.train["period"].max(), temporal.test["period"].min())
        missing = cases.generate_case("numeric_all_missing_rmse")
        self.assertTrue(missing.train["empty channel"].eq("").all())
        self.assertTrue(missing.test["empty channel"].eq("").all())
        self.assertTrue(missing.train["empty category"].eq("").all())
        self.assertTrue(missing.test["empty category"].eq("").all())
        self.assertEqual(missing.train["constant"].nunique(), 1)

    def test_new_and_metamorphic_origins_are_explicit(self):
        catalog = cases.case_catalog()
        metamorphic = [row for row in catalog if row["kind"] == "metamorphic"]
        self.assertEqual(len(metamorphic), 3)
        self.assertEqual(sum(row["provenance"].startswith("benchmark_cases.") for row in metamorphic), 2)
        self.assertTrue(all(row["provenance"] != "new_generator" for row in metamorphic))
        self.assertTrue(all(row["provenance"] == "new_generator" for row in catalog if row["kind"] != "metamorphic"))
        self.assertEqual(cases.generate_case("reverse_positive_binary_auc").contract["positive_class"], "NA")
        labels = cases.generate_case("literal_labels_accuracy").contract["classes"]
        self.assertIn(" мягкий класс ", labels)

    def test_existing_task_directories_and_bad_identifiers_are_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "task"
            original = cases.materialize_case("flow_rmse", target)
            with self.assertRaises(FileExistsError):
                cases.materialize_case("flow_rmse", target)
            self.assertEqual({name: hashlib.sha256((target/name).read_bytes()).hexdigest()
                              for name in original["input_paths"]}, original["input_sha256"])
        for seed in (True, -1, 2**32, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                cases.generate_case("flow_rmse", seed)
        with self.assertRaises(ValueError):
            cases.generate_case("not-a-case")


if __name__ == "__main__":
    unittest.main()
