"""Report arithmetic and evidence admission tests; no solver or downloads."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_lab.integrity import file_hash, json_hash, read_json, write_json
from research_lab import report as reporting


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="lab report Пример ")
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.plan = {"schema_version": 1, "plan_id": "fixture", "profile": "pilot", "policy_comparison": True,
                     "self_check": False, "tie_tolerance": 1e-12, "time_budget_seconds": 360,
                     "threads": 8, "memory_limit_bytes": 24 * 1024**3, "manifest_copy_sha256": "a" * 64,
                     "method_sha256": "b" * 64, "comparison_rule": "qualified paired CPU only",
                     "environment": {"platform": "unit fixture; no Windows execution"},
                     "solvers": {"incumbent": {"sha": "1" * 40}, "candidate": {"sha": "2" * 40}}, "cases": []}
        self.fresh = {"valid": True, "arms": [], "errors": []}
        for index, (metric, direction) in enumerate((("RMSE", "minimize"), ("Macro F1", "maximize"),
                                                    ("MAE", "minimize"), ("LogLoss", "minimize"))):
            case = {"key": "case" + str(index), "dataset_id": "dataset" + str(index), "seed": 701,
                    "tier": "cpu-360s", "metric": metric, "direction": direction,
                    "metadata": {"role": "development", "prior_exposure": "report unit fixture"}, "arms": {}}
            for name, value in (("incumbent", 0.6), ("candidate", 0.4)):
                relative = Path(case["key"]) / name / "state.json"
                write_json(self.output / relative, {"status": "finished", "execution": {
                    "worker_wall_seconds": 12.0, "cleanup_elapsed_seconds": 0.2,
                    "cleanup": {"complete": True}, "resources": {"peak_rss_bytes": None}}})
                case["arms"][name] = {"state_path": str(relative)}
                self.fresh["arms"].append({"case": case["key"], "arm": name, "valid": True,
                    "status": "completed", "quality_verified": True,
                    "audit": {"valid": True, "metric": {"name": metric, "value": value,
                               "higher_is_better": direction == "maximize"}, "per_class": []},
                    "resources": {"qualification": "observed", "fully_observed": True},
                    "provenance": {"outer": {"receipt": "outer"}, "refit": {"receipt": "refit"}}})
            self.plan["cases"].append(case)
        write_json(self.output / "plan.json", self.plan)

    def run_report(self, write=False):
        with patch.object(reporting, "_load_plan", return_value=deepcopy(self.plan)), \
                patch.object(reporting, "_re_audit", return_value=deepcopy(self.fresh)) as audit, \
                patch("subprocess.Popen", side_effect=AssertionError("Report must not launch workers")):
            result = reporting.report(self.output, write=write)
        audit.assert_called_once_with(self.output)
        return result

    def arm(self, case, name):
        return next(row for row in self.fresh["arms"] if row["case"] == "case" + str(case) and row["arm"] == name)

    def test_metric_direction_changes_signed_difference_without_averaging_metrics(self):
        result = self.run_report()
        self.assertAlmostEqual(result["pairs"][0]["signed_difference"], 0.2)
        self.assertAlmostEqual(result["pairs"][1]["signed_difference"], -0.2)
        self.assertEqual(result["summary"]["outcomes"]["win"], 3)
        self.assertEqual(result["summary"]["outcomes"]["loss"], 1)
        self.assertEqual(result["summary"]["denominator_pairs"], 4)
        self.assertEqual(result["summary"]["candidate_win_rate"], 0.75)
        self.assertFalse(any("average" in name for name in result["summary"]))

    def test_provisional_metrics_remain_visible_and_never_become_wins(self):
        self.arm(0, "candidate")["resources"] = {"qualification": "provisional_resource_observation", "fully_observed": False}
        result = self.run_report()
        row = result["pairs"][0]
        self.assertEqual(row["outcome"], "provisional_resource_observation")
        self.assertIsNone(row["signed_difference"])
        self.assertAlmostEqual(row["provisional_signed_difference"], 0.2)
        self.assertEqual(row["arms"]["candidate"]["metric"]["value"], 0.4)
        self.assertEqual(result["summary"]["outcomes"]["win"], 2)
        self.assertEqual(result["summary"]["denominator_pairs"], 4)

    def test_losses_failures_censors_skips_and_unknown_resources_are_retained(self):
        for index, status in ((0, "solver_failed"), (1, "censored"), (2, "skipped")):
            arm = self.arm(index, "candidate")
            arm.update(status=status, quality_verified=False, contract_errors=["intentional " + status])
        self.arm(3, "candidate")["resources"] = {"qualification": "provisional_resource_observation", "fully_observed": False}
        result = self.run_report()
        self.assertEqual([row["outcome"] for row in result["pairs"]],
                         ["failed", "censored", "skipped", "provisional_resource_observation"])
        self.assertEqual(result["summary"]["outcomes"]["win"], 0)
        self.assertEqual(len(result["pairs"]), 4)
        self.assertEqual(result["summary"]["denominator_pairs"], 4)
        self.assertEqual(result["pairs"][0]["arms"]["candidate"]["reason"], ["intentional solver_failed"])

    def test_missing_case_and_not_started_arms_keep_four_pair_pilot_denominator(self):
        self.plan["cases"] = self.plan["cases"][:3]
        for arm in self.fresh["arms"]:
            arm.update(status="not_started", quality_verified=False)
        result = self.run_report()
        self.assertEqual(result["summary"]["denominator_pairs"], 4)
        self.assertEqual(result["summary"]["outcomes"]["missing"], 1)
        self.assertEqual(result["summary"]["outcomes"]["not_completed"], 3)
        self.assertEqual(result["status"], "incomplete")

    def test_stale_embedded_state_audit_cannot_replace_fresh_reaudit(self):
        state_path = self.output / self.plan["cases"][0]["arms"]["candidate"]["state_path"]
        state = read_json(state_path)
        state["audit"] = {"quality_verified": True, "audit": {"valid": True, "metric": {"value": 0}}}
        write_json(state_path, state)
        self.arm(0, "candidate").update(valid=False, status="integrity_failed", quality_verified=False)
        self.fresh.update(valid=False, errors=[{"case": "case0", "arm": "candidate", "error": "tampered receipt"}])
        result = self.run_report()
        self.assertEqual(result["pairs"][0]["outcome"], "integrity_failed")
        self.assertIsNone(result["pairs"][0]["signed_difference"])
        self.assertEqual(result["status"], "integrity_failed")

    def test_mutation_after_reaudit_invalidates_the_state_metadata(self):
        arm = self.arm(0, "candidate")
        arm["state_sha256_after"] = "0" * 64
        result = self.run_report()
        self.assertEqual(result["pairs"][0]["outcome"], "integrity_failed")
        self.assertFalse(result["pairs"][0]["arms"]["candidate"]["qualified_cpu"])
        self.assertEqual(result["status"], "integrity_failed")

    def test_actual_api_zero_requires_pinned_report_counts_for_all_started_arms(self):
        result = self.run_report()
        self.assertIsNone(result["summary"]["actual_api"]["attempts"])
        for arm in self.fresh["arms"]:
            arm["actual_api"] = {"attempts": 0, "responses": 0, "source_report_sha256": "d" * 64}
        result = self.run_report()
        self.assertEqual(result["summary"]["actual_api"], {"attempts": 0, "responses": 0,
                                                            "verified_arms": 8, "started_arms": 8})
        self.arm(0, "candidate")["actual_api"]["attempts"] = None
        result = self.run_report()
        self.assertIsNone(result["summary"]["actual_api"]["attempts"])
        self.assertEqual(result["summary"]["actual_api"]["verified_arms"], 7)
        self.arm(0, "candidate")["actual_api"]["attempts"] = 0
        (self.output / self.plan["cases"][0]["arms"]["candidate"]["state_path"]).unlink()
        result = self.run_report()
        self.assertIsNone(result["summary"]["actual_api"]["attempts"])
        self.assertEqual(result["summary"]["actual_api"]["started_arms"], 8)

    def test_utc_timings_and_api_provenance_are_preserved(self):
        case = self.plan["cases"][0]
        state_path = self.output / case["arms"]["candidate"]["state_path"]
        state = read_json(state_path)
        state.update(started_at_utc="2026-10-21T12:00:00+00:00", finished_at_utc="2026-10-21T12:00:12+00:00")
        write_json(state_path, state)
        self.arm(0, "candidate")["actual_api"] = {"attempts": 0, "responses": 0, "source_report_sha256": "d" * 64}
        result = self.run_report()
        arm = result["pairs"][0]["arms"]["candidate"]
        self.assertEqual(arm["timing"]["started_at_utc"], state["started_at_utc"])
        self.assertEqual(arm["timing"]["finished_at_utc"], state["finished_at_utc"])
        self.assertEqual(arm["actual_api_source_report_sha256"], "d" * 64)

    def test_fixture_and_same_solver_checks_are_not_policy_wins(self):
        for profile, self_check, outcome in (("fixture-smoke", False, "fixture_smoke"), ("pilot", True, "self_check")):
            with self.subTest(profile=profile):
                self.plan.update(profile=profile, self_check=self_check, policy_comparison=False)
                result = self.run_report()
                self.assertTrue(all(row["outcome"] == outcome for row in result["pairs"]))
                self.assertEqual(result["summary"]["outcomes"]["win"], 0)
                self.assertFalse(result["quality_evidence"])

    def test_metric_name_direction_and_finite_value_must_match(self):
        original = deepcopy(self.fresh)
        for field, value in (("name", "MAE"), ("higher_is_better", True), ("value", float("nan"))):
            with self.subTest(field=field):
                self.fresh = deepcopy(original)
                self.arm(0, "candidate")["audit"]["metric"][field] = value
                result = self.run_report()
                self.assertNotEqual(result["pairs"][0]["outcome"], "win")
                self.assertIsNone(result["pairs"][0]["signed_difference"])

    def test_tie_uses_the_predeclared_tolerance(self):
        self.arm(0, "candidate")["audit"]["metric"]["value"] = 0.6 - 1e-13
        result = self.run_report()
        self.assertEqual(result["pairs"][0]["outcome"], "tie")
        self.assertEqual(result["tie_tolerance"], 1e-12)

    def test_mixed_budget_tiers_are_reported_separately(self):
        self.plan["cases"][0]["tier"] = "cpu-900s"
        result = self.run_report()
        self.assertEqual(set(result["tiers"]), {"cpu-360s", "cpu-900s"})
        self.assertIsNone(result["summary"]["candidate_win_rate"])
        self.assertEqual(result["tiers"]["cpu-900s"]["denominator_pairs"], 1)

    def test_duplicate_audit_rows_fail_closed(self):
        self.fresh["arms"].append(deepcopy(self.arm(0, "candidate")))
        result = self.run_report()
        self.assertEqual(result["pairs"][0]["outcome"], "integrity_failed")
        self.assertEqual(result["status"], "integrity_failed")

    def test_plan_drift_renders_failure_and_cannot_admit_saved_metrics(self):
        with patch.object(reporting, "_load_plan", side_effect=ValueError("Plan seal mismatch")), \
                patch.object(reporting, "_re_audit", return_value=self.fresh):
            result = reporting.report(self.output, write=False)
        self.assertEqual(result["status"], "integrity_failed")
        self.assertEqual(result["summary"]["qualified_pairs"], 0)
        self.assertFalse(result["quality_evidence"])

    def test_read_only_mode_does_not_write_or_repair_evidence(self):
        before = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.run_report(write=False)
        after = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_json_markdown_and_file_hashes_include_every_outcome(self):
        self.arm(0, "candidate").update(status="censored", quality_verified=False, reason="cleanup unknown | preserve")
        result = self.run_report(write=True)
        self.assertEqual(read_json(self.output / "report.json"), result)
        for report_name, hash_name in (("report.json", "report.sha256"), ("report.md", "report.md.sha256")):
            self.assertEqual((self.output / hash_name).read_text().strip(), file_hash(self.output / report_name))
        self.assertEqual(result["content_sha256"], json_hash({k: v for k, v in result.items() if k != "content_sha256"}))
        markdown = (self.output / "report.md").read_text()
        for case in self.plan["cases"]:
            self.assertIn(case["dataset_id"], markdown)
        self.assertIn("censored", markdown)
        self.assertIn("cleanup unknown \\| preserve", markdown)
        self.assertIn("planned pairs", markdown)


if __name__ == "__main__":
    unittest.main()
