"""Saved native evidence stays distinct from inferred success and new work."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_rescue_integration_report as reporting


class RescueIntegrationReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="rescue reports Пример ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_v3_extensions_metrics_and_estimates_remain_distinct(self):
        path = self.write("run.json", {
            "research_protocol_version": 3, "status": "completed", "reason": "local_only",
            "elapsed_seconds": 813.5, "api": {"attempts": 0, "responses": 0},
            "stages": [{"id": "inner_research", "status": "completed"}],
            "time_allocation": {"stage_seconds": {"candidate_refit": 80.0}},
            "local_research": {"warmup_attempted": 16, "adaptive_attempted": 24,
                               "extension_hook_called": True, "attempted": 40,
                               "extension_schedule": "before_warmup",
                               "extension_hook_started_remaining_seconds": 600.5,
                               "extension_hook_ended_remaining_seconds": 420.2},
            "research_extensions": {"status": "completed", "attempted": 1, "verified": 1, "accepted": 1,
                "schedule": "task_priority", "order": ["history", "frequency"],
                "trials": [{"kind": "frequency", "status": "skipped", "attempted": False, "reason": "metric_unsupported"},
                           {"kind": "history", "status": "completed", "attempted": True,
                            "code_hash": "c" * 64,
                            "feedback": {"verified": True, "accepted": True, "reason": "improved", "elapsed_seconds": 12.5,
                                         "recipe_hash": "r" * 64}}]},
            "evaluations": {"final_check": {"baseline": {"name": "RMSE", "value": 49.0},
                "candidate": {"name": "RMSE", "value": 44.5}, "selected": "candidate",
                "selection_uses_outer": True, "unbiased_estimate": False}},
            "frozen_baseline": {"code_hash": "b" * 64, "recipe_hash": "a" * 64},
            "frozen_candidate": {"code_hash": "c" * 64, "recipe_hash": "r" * 64},
            "candidate_receipts": {"outer": str(self.root / "absent-outer-receipt.json"),
                                   "refit": str(self.root / "absent-refit-receipt.json"),
                                   "code_hash": "c" * 64, "recipe_hash": "r" * 64},
            "publication": {"new_result_published": True, "recipe": "candidate"},
            "refit_execution": {"candidate": {"elapsed_seconds": 30.0, "host_elapsed_seconds": 31.0,
                "exit_code": 0, "resources": {"available": True, "peak_rss_bytes": 2048, "cpu_seconds": 23.0}}}})
        before = path.read_bytes()
        row = reporting.summarize_report(path)
        self.assertEqual(row["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(row["local_research"]["warmup_attempted"], 16)
        self.assertTrue(row["local_research"]["extension_hook_called"])
        self.assertEqual(row["local_research"]["extension_schedule"], "before_warmup")
        self.assertEqual(row["local_research"]["extension_hook_started_remaining_seconds"], 600.5)
        self.assertEqual(row["local_research"]["extension_hook_ended_remaining_seconds"], 420.2)
        self.assertEqual(row["research_extensions"]["schedule"], "task_priority")
        self.assertEqual(row["research_extensions"]["order"], ["history", "frequency"])
        self.assertTrue(row["research_extensions"]["trials"][1]["verified"])
        self.assertEqual(row["research_extensions"]["trials"][1]["code_hash"], "c" * 64)
        self.assertEqual(row["research_extensions"]["trials"][1]["recipe_hash"], "r" * 64)
        self.assertIsNone(row["research_extensions"]["trials"][0]["verified"])
        self.assertEqual(row["final_check"]["candidate"]["value"], 44.5)
        self.assertEqual(row["frozen_baseline"], {"code_hash": "b" * 64, "recipe_hash": "a" * 64})
        self.assertEqual(row["frozen_candidate"], {"code_hash": "c" * 64, "recipe_hash": "r" * 64})
        self.assertEqual(row["candidate_receipts"]["outer"], str(self.root / "absent-outer-receipt.json"))
        self.assertEqual(row["candidate_receipts"]["refit"], str(self.root / "absent-refit-receipt.json"))
        self.assertEqual(row["candidate_receipts"]["code_hash"], "c" * 64)
        self.assertEqual(row["candidate_receipts"]["recipe_hash"], "r" * 64)
        self.assertEqual(row["candidate_receipts"]["verification"], "not_performed")
        self.assertFalse((self.root / "absent-refit-receipt.json").exists())
        self.assertFalse(row["final_check"]["unbiased_estimate"])
        self.assertIsNone(row["timings"]["stages"][0]["elapsed_seconds"])
        self.assertEqual(row["timings"]["estimated_stage_seconds"], {"candidate_refit": 80.0})
        self.assertEqual(row["timings"]["execution_times"][0]["host_elapsed_seconds"], 31.0)
        self.assertIsNone(row["resources"]["observations"][0]["cleanup_complete"])
        self.assertEqual(row["resources"]["qualification"], "unknown")
        self.assertEqual(path.read_bytes(), before)

    def test_api_fallback_publishes_baseline_without_candidate_refit_claim(self):
        path = self.write("fallback.json", {"research_protocol_version": 3,
            "status": "fallback", "reason": "api_error", "api": {"attempts": 3, "responses": 1},
            "events": [{"kind": "api_error", "details": {"error_type": "InternalServerError", "status_code": 504}},
                       {"kind": "api_error", "details": {"error_type": "APITimeoutError", "cause_type": "ConnectTimeout"}}],
            "local_research": {"attempted": 0, "warmup_attempted": 0, "adaptive_attempted": 0, "extension_hook_called": False},
            "evaluations": {"final_check": {"selected": "baseline", "candidate": None}},
            "publication": {"recipe": "baseline", "new_result_published": True},
            "output_check": {"valid": True}})
        row = reporting.summarize_report(path)
        self.assertEqual((row["api"]["attempts"], row["api"]["responses"]), (3, 1))
        self.assertEqual(row["api"]["errors"][0]["status_code"], 504)
        self.assertEqual(row["api"]["errors"][1]["cause_type"], "ConnectTimeout")
        self.assertFalse(row["local_research"]["extension_hook_called"])
        self.assertEqual(row["publication"]["recipe"], "baseline")
        self.assertEqual(row["publication"]["refit_reported_recipes"], [])
        self.assertIsNone(row["final_check"]["candidate"])
        self.assertEqual(row["resources"]["observations"], [])

    def test_native_automl_calls_do_not_invent_attempts_or_v3_publication(self):
        path = self.write("automl_report.json", {"mode": "offline", "api": {"used": False, "calls": 0},
            "metric": {"name": "Macro F1", "value": 0.613, "higher_is_better": True}, "output_valid": True,
            "search": {"status": "time_limit", "best_candidate": "cb", "candidates": [
                {"id": "cb", "status": "completed", "selected": True, "metric": 0.613,
                 "durations": {"fit_seconds": 12, "predict_seconds": 1, "total_seconds": 13},
                 "resources": {"available": False, "peak_rss_bytes": None, "unavailable_reason": "not observed"}}]}})
        row = reporting.summarize_report(path)
        self.assertEqual(row["schema"], "automl")
        self.assertEqual(row["api"]["calls"], 0)
        self.assertIsNone(row["api"]["attempts"])
        self.assertIsNone(row["api"]["responses"])
        self.assertEqual(row["search"]["best_candidate"], "cb")
        self.assertEqual(row["search"]["candidates"][0]["durations"]["total_seconds"], 13)
        self.assertIsNone(row["publication"]["new_result_published"])
        self.assertEqual(row["publication"]["reported_best_metric"]["value"], 0.613)
        self.assertFalse(row["resources"]["observations"][0]["available"])
        self.assertIsNone(row["resources"]["observations"][0]["peak_rss_bytes"])

    def test_cli_reports_missing_bad_and_partial_evidence_without_new_work(self):
        partial = self.write("partial.json", {"status": "running", "api": {"attempts": True}})
        bad = self.root / "bad.json"
        bad.write_text("{broken", encoding="utf-8")
        missing = self.root / "missing.json"
        output = self.root / "out" / "summary.json"
        originals = {path: path.read_bytes() for path in (partial, bad)}
        with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")), \
                patch("socket.socket", side_effect=AssertionError("must not connect")), \
                redirect_stdout(io.StringIO()):
            code = reporting.main(["--reports", str(partial), str(bad), str(missing), "--output", str(output)])
        self.assertEqual(code, 2)
        rows = json.loads(output.read_text(encoding="utf-8"))["reports"]
        self.assertEqual([row["load_status"] for row in rows], ["loaded", "error", "error"])
        self.assertIsNone(rows[0]["api"]["attempts"])
        self.assertIsNone(rows[0]["local_research"]["extension_hook_called"])
        self.assertIsNone(rows[0]["local_research"]["extension_schedule"])
        self.assertIsNone(rows[0]["local_research"]["extension_hook_started_remaining_seconds"])
        self.assertIsNone(rows[0]["research_extensions"]["order"])
        self.assertIsNone(rows[0]["frozen_baseline"])
        self.assertIsNone(rows[0]["frozen_candidate"])
        self.assertIsNone(rows[0]["candidate_receipts"])
        self.assertEqual(rows[0]["timings"]["stages"], [])
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        self.assertFalse(missing.exists())

    def test_output_cannot_overwrite_source_report(self):
        path = self.write("run.json", {"status": "completed"})
        original = path.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(reporting.main(["--reports", str(path), "--output", str(path)]), 2)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
