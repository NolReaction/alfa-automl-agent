import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import summarize_quality as summary


def row(value, *, metric="MAE", arm="baseline", index=0, seed=211):
    return {"case": "test", "seed": seed, "mode": "automl", "input_sha256": "i", "audit_sha256": "a",
            "metric": metric, "validation": {}, "time_budget_seconds": 180, "memory_limit_bytes": 123,
            "status": "completed", "execution": {"exit_code": 0, "timed_out": False},
            "audit": {"valid": True, "metric": {"name": metric, "value": value, "higher_is_better": metric in summary.BOUNDED}},
            "elapsed_seconds": 2, "inputs_unchanged": True, "sources_unchanged": True,
            "arm": arm, "trial_index": index, "pair": 0,
            "refit_evidence": {"valid": True}}


class ComparisonTests(unittest.TestCase):
    def test_loss_and_score_are_not_same_gain_scale(self):
        loss = summary.compare(row(10), row(8, arm="candidate"), pair=0)
        score = summary.compare(row(.7, metric="Macro F1"), row(.72, metric="Macro F1", arm="candidate"), pair=1)
        self.assertAlmostEqual(loss["relative_loss_gain"], .2)
        self.assertEqual(loss["absolute_delta"], -2)
        self.assertIsNone(loss["bounded_score_gain"])
        self.assertAlmostEqual(score["bounded_score_gain"], .02)
        self.assertIsNone(score["relative_loss_gain"])
        self.assertEqual(score["outcome"], "win")

    def test_failed_or_missing_seed_is_retained(self):
        good = summary.compare(row(10), row(8), pair=0)
        badrow = row(7)
        badrow["status"] = "solver_failed"
        bad = summary.compare(row(10), badrow, pair=1)
        families = summary.family_summaries([good, bad])
        self.assertEqual(families[0]["planned_pairs"], 2)
        self.assertEqual(families[0]["comparable_pairs"], 1)
        self.assertIsNone(families[0]["median_gain"])
        missing = summary.compare(None, None, pair=2, planned={"case": "later", "seed": 1, "mode": "automl", "budget": 90})
        self.assertEqual(missing["case"], "later")
        self.assertEqual(missing["outcome"], "unpaired")

    def test_contract_and_nonfinite_rejected(self):
        other = row(8)
        other["time_budget_seconds"] = 90
        self.assertFalse(summary.compare(row(10), other, pair=0)["comparable"])
        self.assertFalse(summary.compare(row(10), row(float("nan")), pair=0)["comparable"])

    def test_fixed_median_tolerances(self):
        pairs = [summary.compare(row(10, seed=seed), row(val, seed=seed), pair=seed)
                 for seed, val in ((1, 9), (2, 9.8))]
        family = summary.family_summaries(pairs)[0]
        self.assertAlmostEqual(family["median_gain"], .06)
        self.assertTrue(family["practical_improvement"])
        self.assertTrue(family["regression_tolerance_passed"])
        bad = summary.family_summaries([summary.compare(row(10), row(10.31), pair=0)])[0]
        self.assertFalse(bad["regression_tolerance_passed"])

    def test_like_metric_groups_only(self):
        a = summary.compare(row(10), row(9), pair=0)
        b = summary.compare(row(.5, metric="Macro F1"), row(.6, metric="Macro F1"), pair=1)
        with self.assertRaises(ValueError):
            summary.family_summaries([a, b])

    def test_confirmation_gate_requires_two_families_and_no_hidden_loss(self):
        manifest = {"trial_plan": [{"case": case, "seed": seed, "arm": arm,
                                      "budget": 180, "mode": "automl"}
                                     for case in summary.CONFIRM_FAMILIES for seed in (211, 307)
                                     for arm in ("baseline", "candidate")],
                    "cases": [{"case": case, "seed": seed} for case in summary.CONFIRM_FAMILIES for seed in (211, 307)]}
        pairs = [{"comparable": True} for _ in range(12)]
        arms = {"candidate": {"valid_outputs": 12, "full_refit_v1_verified": 11}}
        families = [{"practical_improvement": i < 2, "regression_tolerance_passed": True} for i in range(6)]
        report = {"purpose": "confirmation", "status": "completed"}
        gate = summary.numeric_gate(report, manifest, pairs, families, arms)
        self.assertTrue(gate["quality_gate_passed"])
        self.assertFalse(gate["checks"]["all_candidate_refits_v1_verified"])
        self.assertNotEqual(gate["default_eligibility"], "passed")
        families[0]["practical_improvement"] = False
        self.assertFalse(summary.numeric_gate(report, manifest, pairs, families, arms)["quality_gate_passed"])
        families[0]["practical_improvement"] = True
        families[-1]["regression_tolerance_passed"] = False
        self.assertFalse(summary.numeric_gate(report, manifest, pairs, families, arms)["quality_gate_passed"])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {"trial_plan": [{"index": i, "pair": 0, "arm": arm, "case": "test", "seed": 211, "mode": "automl", "budget": 180}
                                        for i, arm in enumerate(("baseline", "candidate"))],
                         "sources": {"baseline": {}, "candidate": {}},
                         "cases": [{"case": "test", "seed": 211, "input_sha256": "i", "audit_sha256": "a"}]}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.manifest_hash = summary.digest(self.root / "manifest.json")
        (self.root / "seal.json").write_text(json.dumps({"manifest_sha256": self.manifest_hash}))
        self.report = {"sources": self.manifest["sources"], "manifest_sha256": self.manifest_hash,
                       "purpose": "development", "status": "completed", "trials": [row(10), row(8, arm="candidate", index=1)]}
        self.report_path = self.root / "report.json"
        self.write_report()

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def test_preserves_v1_unverified_and_binds_v2(self):
        self.report["trials"][1]["refit_evidence"] = {"valid": False, "reason": "stdout"}
        self.write_report()
        audit = {"version": 2, "kind": "night_quality_reaudit", "original_report_sha256": summary.digest(self.report_path),
                 "original_manifest_sha256": self.manifest_hash, "trials": []}
        for trial in self.report["trials"]:
            audit["trials"].append({"trial_index": trial["trial_index"], "arm": trial["arm"], "valid": True,
                                    "v1_refit_evidence": trial["refit_evidence"], "independent_audit": trial["audit"],
                                    "independent_audit_matches_original": True, "full_refit": {"valid": True}})
        path = self.root / "reaudit.json"
        path.write_text(json.dumps(audit))
        result = summary.summarize_report(self.report_path, path)
        self.assertEqual(result["arms"]["candidate"]["full_refit_v1_verified"], 0)
        self.assertEqual(result["arms"]["candidate"]["full_refit_v2_verified"], 1)
        audit["original_report_sha256"] = "bad"
        path.write_text(json.dumps(audit))
        with self.assertRaises(ValueError):
            summary.summarize_report(self.report_path, path)

    def test_seal_and_duplicate_detection(self):
        self.report["trials"].append(copy.deepcopy(self.report["trials"][0]))
        self.write_report()
        with self.assertRaises(ValueError):
            summary.summarize_report(self.report_path)
        (self.root / "manifest.json").write_text("{}")
        with self.assertRaises(ValueError):
            summary.summarize_report(self.report_path)

    def test_incomplete_report_does_not_hide_missing_arm(self):
        self.report["trials"].pop()
        self.write_report()
        result = summary.summarize_report(self.report_path)
        self.assertEqual(result["arms"]["candidate"]["statuses"], {"not_started": 1})
        self.assertEqual(result["counts"], {"unpaired": 1})


if __name__ == "__main__":
    unittest.main()
