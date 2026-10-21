"""Fixture-only checks of real-data adapter boundaries; no network or model fit."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.task_spec import load_task
from research_lab.adapters import IneligibleCase, _technical_ids, prepare_case
from research_lab.integrity import file_hash, read_json


class ResearchLabDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lab data 駅 ")
        self.root = Path(self.temporary.name)
        self.serial = 0

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, **updates):
        value = {"dataset_id": "fixture", "target": "answer", "task_type": "regression", "metric": "RMSE",
                 "numeric": ["x"], "categorical": ["literal"], "feature_exclusions": [], "id_columns": [],
                 "classes": [], "positive_class": None, "id_rule": {"column": "id"},
                 "adapter": {"format": "csv", "delimiter": ",", "encoding": "utf-8"},
                 "group_rule": {"columns": [], "exact_features": True, "near_duplicate_decimals": 8},
                 "split_rule": {"kind": "source_order", "audit_fraction": .2, "gap_periods": 0}}
        value.update(updates)
        return value

    def rows(self, count=40):
        return [{"x": str(i), "literal": ["01", "NA", "", "N/A", "駅"][i % 5],
                 "answer": str(10000 + i)} for i in range(count)]

    def prepare(self, rows, record=None, seed=1729, *, receipt_sha=None, case_dir=None, audit_dir=None):
        self.serial += 1
        source = self.root / f"source-{self.serial}.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        receipt = {"sha256": file_hash(source) if receipt_sha is None else receipt_sha,
                   "source_bytes": source.stat().st_size, "dataset_id": "fixture"}
        case_dir = self.root / "templates" / f"case-{self.serial}" if case_dir is None else case_dir
        audit_dir = self.root / f"host-audit-{self.serial}" if audit_dir is None else audit_dir
        with patch("research_lab.fetch.cached_member", return_value=(source, receipt)):
            case = prepare_case(self.record() if record is None else record, self.root / "cache",
                                case_dir, audit_dir, seed)
        return case, read_json(case["audit_path"])

    @staticmethod
    def read(case, name):
        return pd.read_csv(Path(case["task_template"]) / "inputs" / name, dtype=str, keep_default_na=False)

    def test_literal_values_and_truth_boundary_native_preflight(self):
        case, audit = self.prepare(self.rows())
        train, test = self.read(case, "train.csv"), self.read(case, "test.csv")
        self.assertEqual(train["literal"].tolist()[:5], ["01", "NA", "", "N/A", "駅"])
        self.assertEqual(set(test), {"id", "x", "literal"})
        self.assertTrue(train["id"].str.match(r"r_[a-f0-9]{64}").all())
        self.assertFalse(train["id"].isin(test["id"]).any())
        self.assertEqual(test["id"].tolist(), audit["test_ids"])
        self.assertEqual(audit["audit_actuals"], [float(10000 + i) for i in range(32, 40)])
        template = Path(case["task_template"])
        self.assertEqual({p.relative_to(template).as_posix() for p in template.rglob("*") if p.is_file()},
                         {"task.md", "inputs/train.csv", "inputs/test.csv", "inputs/data_dictionary.csv", "inputs/sample_submission.csv"})
        all_bytes = b"\n".join(p.read_bytes() for p in template.rglob("*") if p.is_file())
        self.assertNotIn(b"10032", all_bytes)
        self.assertNotIn(str(Path(case["audit_path"]).parent).encode(), all_bytes)
        self.assertEqual(load_task(Path(case["task_path"])).to_dict(), case["spec"])
        self.assertEqual({name: file_hash(template / name) for name in case["input_hashes"]}, case["input_hashes"])
        self.assertEqual(file_hash(case["audit_path"]), case["audit_sha256"])

    def test_semantic_and_near_duplicate_conflicting_labels_stay_grouped(self):
        rows = []
        for i in range(30):
            rows.extend([{"x": f"{i}.000000001", "literal": "NA", "answer": "01"},
                         {"x": f"{i}.000000002", "literal": "NA", "answer": "NA"}])
        record = self.record(task_type="classification", metric="LogLoss", classes=["01", "NA"], positive_class="NA",
                             probability_order=["NA", "01"],
                             split_rule={"kind": "stratified_group", "audit_fraction": .2})
        case, audit = self.prepare(rows, record)
        train_ord, audit_ord = map(set, (audit["source_ordinals"]["train"], audit["source_ordinals"]["audit"]))
        for first in range(0, len(rows), 2):
            self.assertEqual(first in train_ord, first + 1 in train_ord)
            self.assertEqual(first in audit_ord, first + 1 in audit_ord)
        self.assertFalse(set(audit["group_ids"]["train"]) & set(audit["group_ids"]["audit"]))
        self.assertEqual(case["spec"]["classes"], ["01", "NA"])
        self.assertEqual(case["spec"]["probability_columns"], [["NA", "NA"], ["01", "01"]])
        self.assertEqual(case["spec"]["output_columns"], ["id", "NA", "01"])
        self.assertEqual(set(self.read(case, "train.csv")["answer"]), {"01", "NA"})

    def test_entity_or_feature_duplicates_union_transitively(self):
        rows = self.rows(40)
        for i, row in enumerate(rows):
            row["patient"] = f"p{i}"
        rows[1]["patient"] = rows[0]["patient"]
        rows[2]["x"], rows[2]["literal"] = rows[1]["x"], rows[1]["literal"]
        record = self.record(id_columns=["patient"], group_rule={"columns": ["patient"], "exact_features": True},
                             split_rule={"kind": "group_random", "audit_fraction": .2})
        case, audit = self.prepare(rows, record)
        train = set(audit["source_ordinals"]["train"])
        self.assertEqual(len(train & {0, 1, 2}) in {0, 3}, True)
        self.assertNotIn("patient", self.read(case, "train.csv"))

    def test_numeric_spelling_duplicates_are_semantic_without_near_rounding(self):
        rows = self.rows()
        rows[35]["x"], rows[35]["literal"] = "00.000e5", rows[0]["literal"]
        record = self.record(group_rule={"columns": [], "exact_features": True})
        _, audit = self.prepare(rows, record)
        self.assertEqual(audit["source_ordinals"]["purged"], [35])

    def test_source_order_purges_features_duplicate_after_exclusions(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["duration"] = str(i * 99)
        rows[35]["x"], rows[35]["literal"] = rows[0]["x"], rows[0]["literal"]
        record = self.record(feature_exclusions=["duration"])
        case, audit = self.prepare(rows, record)
        self.assertEqual(audit["source_ordinals"]["purged"], [35])
        self.assertEqual(case["metadata"]["split"]["cross_boundary_group_rows_purged_from_audit"], 1)
        self.assertNotIn("duration", self.read(case, "train.csv"))
        self.assertLess(max(audit["source_ordinals"]["train"]), min(audit["source_ordinals"]["audit"]))
        self.assertIn("proxy", Path(case["task_path"]).read_text())

    def test_time_split_complete_periods_gap_and_native_automatic_route(self):
        rows = [{"x": str(year * 10 + j), "model_year": str(year), "literal": "01", "car_name": f"car{year}-{j}",
                 "answer": str(year + j / 10)} for year in range(70, 83) for j in range(3)]
        # An entity observed in the past must be purged from the future audit.
        rows[-1]["car_name"] = rows[0]["car_name"]
        record = self.record(numeric=["x", "model_year"], id_columns=["car_name"],
                             group_rule={"columns": ["car_name"], "exact_features": True},
                             split_rule={"kind": "time", "time_column": "model_year", "audit_fraction": .2, "gap_periods": 1})
        case, audit = self.prepare(rows, record)
        train, test = self.read(case, "train.csv"), self.read(case, "test.csv")
        self.assertEqual(max(map(int, train["model_year"])), 78)
        self.assertEqual(min(map(int, test["model_year"])), 80)
        self.assertEqual(case["rows"]["purged"], 4)
        self.assertNotIn("car_name", test)
        self.assertEqual(audit["metadata"]["split"]["time_column"], "model_year")

    def test_bike_month_period_and_target_components_excluded(self):
        rows = [{"x": str(i), "literal": "NA", "yr": str(i // 12), "mnth": str(i % 12 + 1),
                 "dteday": f"201{1+i//12}-{i%12+1:02d}-01", "instant": str(i),
                 "casual": str(100+i), "registered": str(9000), "answer": str(9100+i)} for i in range(24)]
        record = self.record(numeric=["x", "yr", "mnth", "period"], id_columns=["instant"],
                             feature_exclusions=["dteday", "casual", "registered"],
                             adapter={"format": "csv", "derived_period": {"kind": "calendar_month", "column": "period", "year_column": "yr", "month_column": "mnth"}},
                             split_rule={"kind": "time", "time_column": "period", "audit_fraction": .2})
        case, _ = self.prepare(rows, record)
        train, test = self.read(case, "train.csv"), self.read(case, "test.csv")
        self.assertEqual(train["period"].tolist(), [str(i) for i in range(1, 20)])
        self.assertEqual(test["period"].tolist(), [str(i) for i in range(20, 25)])
        self.assertFalse({"dteday", "casual", "registered", "instant"} & set(test))

    def test_numeric_missing_rules_preserve_categorical_tokens(self):
        rows = self.rows()
        rows[0]["x"], rows[1]["x"], rows[2]["x"] = "?", "NaN", ""
        rows[0]["literal"], rows[1]["literal"] = "?", "NaN"
        record = self.record(adapter={"format": "csv", "missing_numeric": {"x": ["?", "NaN", ""]}})
        case, _ = self.prepare(rows, record)
        train = self.read(case, "train.csv")
        self.assertEqual(train["x"].tolist()[:3], ["", "", ""])
        self.assertEqual(train["literal"].tolist()[:2], ["?", "NaN"])
        record["adapter"]["missing_numeric"]["literal"] = ["?"]
        with self.assertRaisesRegex(IneligibleCase, "only declared numeric"):
            self.prepare(rows, record)

    def test_wine_static_red_filter_keeps_original_source_ordinals(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["color"] = "red" if i % 2 == 0 else "white"
        record = self.record(feature_exclusions=["color"], adapter={"format": "csv", "filter": {"column": "color", "value": "red"}})
        case, audit = self.prepare(rows, record)
        self.assertEqual(case["rows"]["selected"], 20)
        self.assertEqual(audit["source_ordinals"]["audit"], [32, 34, 36, 38])
        self.assertNotIn("color", self.read(case, "train.csv"))

    def test_binary_auc_positive_class_is_literal(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["answer"] = "01" if i % 2 else "NA"
        record = self.record(task_type="classification", metric="ROC AUC", classes=["01", "NA"], positive_class="01")
        case, audit = self.prepare(rows, record)
        self.assertEqual(case["spec"]["positive_class"], "01")
        self.assertEqual(case["spec"]["probability_columns"], [["answer", "01"]])
        self.assertEqual(audit["audit_actuals"][:2], ["NA", "01"])

    def test_multiclass_probability_order_is_declared_not_sorted(self):
        rows = self.rows(60)
        for i, row in enumerate(rows):
            row["answer"] = ["01", "NA", "駅"][i % 3]
        record = self.record(task_type="classification", metric="LogLoss", classes=["駅", "01", "NA"],
                             probability_order=["NA", "駅", "01"])
        case, _ = self.prepare(rows, record)
        self.assertEqual(case["spec"]["classes"], ["駅", "01", "NA"])
        self.assertEqual(case["spec"]["output_columns"], ["id", "NA", "駅", "01"])

    def test_empty_label_and_missing_split_class_support_are_ineligible_no_retry(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["answer"] = "" if i == 0 else "NA"
        record = self.record(task_type="classification", metric="Accuracy", classes=["", "NA"])
        with self.assertRaisesRegex(IneligibleCase, "empty classification labels"):
            self.prepare(rows, record)
        for i, row in enumerate(rows):
            row["answer"] = "01" if i == 0 else "NA"
        record["classes"] = ["01", "NA"]
        with self.assertRaisesRegex(IneligibleCase, "no seed retries"):
            self.prepare(rows, record)
        self.assertFalse(any((self.root / "templates").glob("case-*")))

    def test_declared_label_mapping_only_and_no_implicit_adult_period_alias(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["answer"] = "<=50K." if i % 2 else ">50K."
        record = self.record(task_type="classification", metric="Accuracy", classes=["<=50K", ">50K"])
        with self.assertRaisesRegex(IneligibleCase, "literal classes"):
            self.prepare(rows, record)
        record["label_rule"] = {"mapping": {"<=50K.": "<=50K", ">50K.": ">50K"}}
        case, _ = self.prepare(rows, record)
        self.assertEqual(set(self.read(case, "train.csv")["answer"]), {"<=50K", ">50K"})

    def test_targets_ids_unclassified_columns_and_source_drift_rejected(self):
        rows = self.rows()
        for updates, pattern in [({"numeric": ["x", "answer"]}, "Target must"),
                                 ({"id_columns": ["x"]}, "Excluded leakage/ID"),
                                 ({"categorical": []}, "Every source column"),
                                 ({"group_rule": {"columns": ["answer"], "exact_features": True}}, "Target cannot")]:
            with self.subTest(updates=updates), self.assertRaisesRegex(IneligibleCase, pattern):
                self.prepare(rows, self.record(**updates))
        with self.assertRaisesRegex(IneligibleCase, "source bytes changed"):
            self.prepare(rows, receipt_sha="0" * 64)

    def test_truth_directory_and_frozen_destination_not_reused(self):
        task = self.root / "unsafe-task"
        with self.assertRaisesRegex(IneligibleCase, "outside"):
            self.prepare(self.rows(), case_dir=task, audit_dir=task / "truth")
        case, _ = self.prepare(self.rows())
        with self.assertRaisesRegex(IneligibleCase, "must be new"):
            self.prepare(self.rows(), case_dir=Path(case["task_template"]))

    def test_unsupported_short_chronology_does_not_fall_back_to_random(self):
        rows = self.rows(12)
        for i, row in enumerate(rows):
            row["model_year"] = str(70 + i // 6)
        record = self.record(numeric=["x", "model_year"],
                             split_rule={"kind": "time", "time_column": "model_year", "audit_fraction": .5})
        with self.assertRaisesRegex(IneligibleCase, "2..100 training periods"):
            self.prepare(rows, record)

    def test_ragged_or_duplicate_source_headers_are_rejected(self):
        self.serial += 1
        source = self.root / "bad.csv"
        for raw, pattern in [("x,literal,answer\n1,NA,5\n2,NA\n", "Ragged"),
                             ("x,x,answer\n1,2,5\n2,3,6\n", "duplicate header")]:
            with self.subTest(raw=raw):
                source.write_text(raw, encoding="utf-8")
                receipt = {"sha256": file_hash(source)}
                with patch("research_lab.fetch.cached_member", return_value=(source, receipt)), self.assertRaisesRegex(IneligibleCase, pattern):
                    prepare_case(self.record(), self.root / "cache", self.root / "bad-task", self.root / "host-only", 1729)

    def test_static_id_rule_does_not_accept_label_argument(self):
        ids = _technical_ids("fixture", "f" * 64, [0, 1, 2])
        self.assertEqual(ids, _technical_ids("fixture", "f" * 64, [0, 1, 2]))
        self.assertEqual(len(set(ids)), 3)
        self.assertNotEqual(ids, _technical_ids("fixture", "e" * 64, [0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
