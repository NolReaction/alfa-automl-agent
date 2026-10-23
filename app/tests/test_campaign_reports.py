"""Campaign accounting/export tests; no filesystem inputs, fitting or API use."""
from copy import deepcopy
import csv
from io import BytesIO, StringIO
import json
import os
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from app.campaign_reports import (MAX_REPORT_BYTES, MAX_STEPS, RUNTIME_PACKAGES, _canonical,
                                  build_campaign_report, report_bytes)
import hashlib


def step(index, score=10., *, name="RMSE", higher=False, dataset="a" * 64,
         task="regression", started=None):
    identifier = f"{index + 1:032x}"
    split = {"seed": 42, "stratified": False, "validation_rows": 10,
             "validation_fraction": .2, "validation_hash": "b" * 64,
             "train_rows": 40, "strategy": "random", "shuffle": True}
    key = hashlib.sha256(_canonical({"dataset_fingerprint": dataset, "metric": name,
        "higher_is_better": higher, "split": split}).encode()).hexdigest()
    job = {"id": identifier, "task_id": task, "action": "automl", "source": "local", "status": "completed",
           "output_valid": True, "model": "RandomForest", "elapsed_seconds": 100.,
           "started_at": started or f"2026-10-24T00:{index:02d}:00+00:00",
           "finished_at": f"2026-10-24T00:{index:02d}:30+00:00",
           "metric": {"name": name, "value": score, "higher_is_better": higher},
           "split": split, "comparison_key": key, "dataset_fingerprint": dataset}
    return {"id": identifier, "job_id": identifier, "task_id": task, "action": "automl",
            "minutes": (index + 1) * 10, "label": f"Budget {index}", "status": "completed",
            "admissible": True, "job": job, "error": None}


def detail(steps=None):
    runtime = {"python_version": "3.12.2", "implementation": "CPython", "pointer_bits": 64,
               "packages": dict.fromkeys(RUNTIME_PACKAGES, "1.0")}
    return {"id": "c" * 32, "name": "Night campaign", "preset": "budget-curve", "status": "completed",
            "source_seal": {"sha": "d" * 40, "source_fingerprint": "e" * 64, "valid": True,
                            "checked_at": "2026-10-24T00:00:00Z"},
            "runtime_seal": {**runtime, "fingerprint": hashlib.sha256(_canonical(runtime).encode()).hexdigest(), "valid": True},
            "plan_sha256": "f" * 64, "task_fingerprints": {"regression": "a" * 64},
            "wall_budget_seconds": 36000, "elapsed_seconds": 1000,
            "worker_upper_bound_seconds": 3000, "steps": steps if steps is not None else []}


class CampaignReportTests(unittest.TestCase):
    def test_manager_detail_contract_exports_without_training_or_private_paths(self):
        from app.tests.test_campaigns import CampaignTests
        fixture = CampaignTests("test_serial_jobs_have_durable_distinct_owners_and_exact_comparison_groups")
        fixture.setUp()
        try:
            identifier, active = fixture.queued()
            snapshot = fixture.run_queue(identifier, active)
            report = build_campaign_report(snapshot)
            self.assertEqual(report["summary"]["comparison_eligible"], 2)
            self.assertEqual(len(report["groups"]), 1)
            self.assertEqual(report["groups"][0]["order_basis"], "persisted_plan_order")
            self.assertNotIn(str(fixture.root), report_bytes(report).decode())
            with ZipFile(BytesIO(report_bytes(report, "zip"))) as archive:
                self.assertIsNone(archive.testzip())
        finally:
            fixture.doCleanups()

    def test_chronological_control_best_gain_and_stable_tie(self):
        first, second, tied = step(0, 10), step(1, 8), step(2, 8)
        report = build_campaign_report(detail([tied, second, first]))
        group, = report["groups"]
        self.assertEqual(group["control"]["job_id"], first["job_id"])
        self.assertEqual(group["best"]["job_id"], second["job_id"])
        self.assertEqual((group["gain"], group["gain_percent"]), (2, 20))
        self.assertEqual(group["order_basis"], "started_at")
        self.assertTrue(group["comparison_available"])
        self.assertEqual([p["minutes"] for p in group["budget_curve"]], [10, 20, 30])
        self.assertEqual(report["summary"]["observed_job_elapsed_seconds"], 300)
        self.assertIn("not establish hidden-test", report["scope"])

    def test_same_time_tie_uses_stable_plan_order_and_missing_dates_are_honest(self):
        a, b = step(0, 10), step(1, 10)
        b["job"]["started_at"] = a["job"]["started_at"]
        report = build_campaign_report(detail([b, a]))
        self.assertEqual(report["groups"][0]["best"]["job_id"], b["job_id"])
        a["job"]["started_at"] = "bad timestamp"
        report = build_campaign_report(detail([a, b]))
        self.assertEqual(report["groups"][0]["order_basis"], "persisted_plan_order")
        self.assertEqual(report["groups"][0]["control"]["job_id"], a["job_id"])

    def test_zero_and_negative_scores_both_directions(self):
        for baseline, best, higher, gain, percent in ((0, -2, False, 2, None),
                (-4, -2, True, 2, 50), (-2, -4, False, 2, 100), (0, 0, True, 0, None)):
            with self.subTest(baseline=baseline, best=best, higher=higher):
                report = build_campaign_report(detail([step(0, baseline, higher=higher), step(1, best, higher=higher)]))
                self.assertEqual((report["groups"][0]["gain"], report["groups"][0]["gain_percent"]), (gain, percent))

    def test_groups_never_mix_task_dataset_metric_direction_or_split(self):
        records = [step(0), step(1, name="MAE"), step(2, higher=True),
                   step(3, task="another"), step(4, dataset="9" * 64, task="third")]
        sixth = step(5)
        sixth["job"]["split"]["seed"] = 23
        sixth["job"]["comparison_key"] = hashlib.sha256(_canonical({
            "dataset_fingerprint": "a" * 64, "metric": "RMSE", "higher_is_better": False,
            "split": sixth["job"]["split"]}).encode()).hexdigest()
        records.append(sixth)
        raw = detail(records)
        raw["task_fingerprints"].update(another="a" * 64, third="9" * 64)
        report = build_campaign_report(raw)
        self.assertEqual(len(report["groups"]), 6)
        self.assertTrue(all(group["count"] == 1 for group in report["groups"]))
        self.assertTrue(all(group["comparison_available"] is False for group in report["groups"]))
        self.assertNotIn("aggregate_gain", report["summary"])

    def test_corrupt_job_evidence_is_diagnostic_only(self):
        mutations = [lambda s: s["job"].update(output_valid=False),
                     lambda s: s["job"].update(source="imported"),
                     lambda s: s["job"].update(status="failed"),
                     lambda s: s.update(admissible=None),
                     lambda s: s["job"].update(task_id="other"),
                     lambda s: s["job"].update(action="baseline"),
                     lambda s: s["job"].update(id="0" * 32),
                     lambda s: s["job"].update(dataset_fingerprint="0" * 64),
                     lambda s: s["job"].update(comparison_key="0" * 64),
                     lambda s: s["job"]["split"].update(validation_rows=0),
                     lambda s: s["job"]["split"].update(strategy=[]),
                     lambda s: s["job"]["metric"].update(higher_is_better="false"),
                     lambda s: s["job"]["metric"].update(value=float("nan")),
                     lambda s: s["job"]["metric"].update(value=True),
                     lambda s: s["job"].update(metric=None)]
        for change in mutations:
            record = step(0)
            change(record)
            with self.subTest(change=change):
                report = build_campaign_report(detail([record]))
                self.assertFalse(report["groups"])
                self.assertFalse(report["steps"][0]["comparison_eligible"])
                self.assertTrue(report["steps"][0]["exclusion_reasons"])
                json.loads(report_bytes(report))

    def test_global_source_failure_or_unknown_retains_metrics_but_excludes_comparison(self):
        for seal in ({"valid": False}, {"valid": None}, {"valid": True}, {"valid": "true"}):
            raw = detail([step(0)])
            raw["source_seal"] = seal
            report = build_campaign_report(raw)
            self.assertEqual(report["steps"][0]["metric"]["value"], 10)
            self.assertFalse(report["groups"])
            self.assertIn("source_seal_not_verified", report["steps"][0]["exclusion_reasons"])

    def test_duplicate_bindings_cannot_fake_additional_runs(self):
        first = step(0)
        report = build_campaign_report(detail([first, deepcopy(first)]))
        self.assertFalse(report["groups"])
        self.assertIn("duplicate_job_id", report["steps"][0]["exclusion_reasons"])

    def test_unknown_drifted_or_corrupt_runtime_cannot_qualify(self):
        for seal in (None, {}, {"valid": False}, {"valid": True}, {**detail()["runtime_seal"], "pointer_bits": []},
                     {**detail()["runtime_seal"], "packages": {"numpy": "1.0", "secret": "private"}},
                     {**detail()["runtime_seal"], "packages": dict.fromkeys(RUNTIME_PACKAGES, "2.0")}):
            raw = detail([step(0)])
            raw["runtime_seal"] = seal
            report = build_campaign_report(raw)
            self.assertFalse(report["groups"])
            self.assertIn("runtime_seal_not_verified", report["steps"][0]["exclusion_reasons"])
            self.assertEqual(report["steps"][0]["metric"]["value"], 10)

    def test_counts_missing_results_and_unknown_elapsed(self):
        raw = detail([{"id": f"{i:032x}", "status": status, "error": "expected diagnostic"}
                      for i, status in enumerate(("failed", "skipped", "interrupted", "cancelled", "running"))])
        report = build_campaign_report(raw)
        self.assertEqual({k: report["summary"][k] for k in ("failed", "skipped", "interrupted", "cancelled", "unfinished")},
                         dict.fromkeys(("failed", "skipped", "interrupted", "cancelled", "unfinished"), 1))
        self.assertEqual(report["summary"]["elapsed_unknown_steps"], 5)
        self.assertIsNone(report["summary"]["observed_job_elapsed_seconds"])
        self.assertEqual(report["summary"]["comparison_eligible"], 0)
        self.assertEqual(build_campaign_report(detail())["summary"]["steps"], 0)
        with ZipFile(BytesIO(report_bytes(report, "zip"))) as archive:
            self.assertIn("Нет результатов, допущенных к сравнению".encode(), archive.read("report.md"))

    def test_downloads_have_fixed_names_csv_formula_defense_and_no_private_fields(self):
        record = step(0, -3)
        record["label"] = '\t=HYPERLINK("https://secret.example","open")'
        record["job"]["model"] = "+Formula"
        record["error"] = "api_key=secret-value /tmp/model.joblib C:\\models\\private.cbm sk-12345678"
        record["job"].update(train=[[1, 2]], actuals=["PRIVATE-LABEL"], model_path="PRIVATE-MODEL-PATH")
        raw = detail([record])
        raw.update(root="PRIVATE-SOURCE-PATH", api_key="SECRET-CREDENTIAL", raw_train="PRIVATE-TRAIN")
        raw["source_seal"]["root"] = "PRIVATE-SOURCE-PATH"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-value"}):
            report = build_campaign_report(raw)
        content = report_bytes(report, "json")
        for forbidden in (b"secret-value", b"PRIVATE-", b"secret.example", b"/tmp/model", b"C:\\\\models", b"sk-12345678"):
            self.assertNotIn(forbidden, content)
        with ZipFile(BytesIO(report_bytes(report, "zip"))) as archive:
            self.assertEqual(set(archive.namelist()), {"report.json", "report.csv", "report.md"})
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read("report.json")), report)
            rows = list(csv.DictReader(StringIO(archive.read("report.csv").decode("utf-8-sig"))))
            self.assertTrue(rows[0]["label"].startswith("'\t="))
            self.assertEqual(rows[0]["model"], "'+Formula")
            self.assertEqual(rows[0]["metric_value"], "-3")
            self.assertNotIn(b"<script>", archive.read("report.md"))

    def test_malformed_input_limits_and_provenance(self):
        for invalid in (None, {}, {"steps": None}, {"steps": [None]}, {"steps": [{}] * (MAX_STEPS + 1)}):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                build_campaign_report(invalid)
        raw = detail([step(0)])
        before = deepcopy(raw)
        report = build_campaign_report(raw)
        self.assertEqual(raw, before)
        self.assertEqual(report["campaign"]["source_seal"]["sha"], "d" * 40)
        self.assertEqual(report["campaign"]["plan_sha256"], "f" * 64)
        for kind in ("../../secret", "html", None, []):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                report_bytes(report, kind)
        with self.assertRaises(ValueError):
            report_bytes({"oversized": "x" * MAX_REPORT_BYTES})
        with self.assertRaises(ValueError):
            report_bytes({"nonfinite": float("inf")})

    def test_text_is_bounded_redacted_and_csv_controls_do_not_enable_formulas(self):
        for prefix in ("=", "+", "-", "@", "\t=", "\n@", "\ufeff=", "\u200b+"):
            record = step(0)
            record["label"] = prefix + "formula"
            record["error"] = "abc " + "x" * 10000
            with patch.dict(os.environ, {"OPENAI_API_KEY": "abc"}):
                report = build_campaign_report(detail([record]))
            self.assertNotIn("abc", report["steps"][0]["error"])
            self.assertLessEqual(len(report["steps"][0]["error"]), 2000)
            rows = list(csv.DictReader(StringIO(report_bytes(report, "csv").decode("utf-8-sig"))))
            self.assertTrue(rows[0]["label"].startswith("'"), prefix)


if __name__ == "__main__":
    unittest.main()
