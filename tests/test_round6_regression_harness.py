"""Zero-fit tests for development boundaries and consequential acceptance gates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_round6_regression_calibration as lab


SOURCE = {"head": "a" * 40, "dirty": False, "dependency_hashes": {},
          "dependency_sha256": lab.json_hash({})}


def write_plan(root):
    """A small frozen input manifest; no model/data-directory preparation."""
    folds = {}
    for period in lab.DEVELOPMENT:
        path = root / "inputs" / f"p{period}" / "fixture.csv"
        path.parent.mkdir(parents=True)
        path.write_text("value\n1\n", encoding="utf-8")
        folds[str(period)] = {"fit_rows": 1, "predict_rows": 1,
            "files": {str(path.relative_to(root)): lab.digest(path)}}
    frozen = {"version": 1, "round": 6, "base_sha": lab.BASE_SHA, "source": deepcopy(SOURCE),
        "plan_document_sha256": lab.digest(lab.PLAN_DOCUMENT),
        "plan_recipes_sha256": lab.digest(lab.PLAN_RECIPES), "folds": folds,
        "period_column": "period", "size_column": "value", "source_files": {},
        "screen": {"period": 4, "arms": list(lab.ARMS), "cap_seconds": lab.SCREEN_SECONDS},
        "confirmation": {"periods": list(lab.DEVELOPMENT), "maximum_finalists": 2,
                         "cap_seconds": lab.CONFIRM_SECONDS},
        "known5_evaluation": "forbidden_in_round6", "total_heavy_seconds": lab.TOTAL_SECONDS,
        "history": {"scope": "host_inner_fit", "max_cell_visits": 8000000,
                    "max_checked_configs": 48, "extended": False}}
    lab.write(root / "plan.json", frozen)
    return frozen


def decision_records(root, *, candidate_rmse=(9., 9., 9.), omit=None, control_cap=360):
    """Explicit synthetic score evidence for testing decisions, never ML claims."""
    lab.write(root / "promotion.json", {"finalists": ["S20"]})
    records = {}
    for period, rmse in zip(lab.DEVELOPMENT, candidate_rmse):
        for arm in ("I", "S20"):
            if (period, arm) == omit:
                continue
            path = root / "runs" / "confirmation" / f"p{period}" / arm / "summary.json"
            metric = 10. if arm == "I" else rmse
            record = {"status": "completed", "budget_seconds": control_cap if arm == "I" else 360,
                "recipe": {"history_config": None}, "selected_ablation": "calibrated",
                "metrics": {"calibrated": {"rmse": metric, "mae": metric,
                                            "sse": metric * metric * 1000, "rows": 1000}}}
            lab.write(path, record)
            lab.write(path.parent / "segments.json", {"rows": []})
            records[(period, arm)] = (record, path)

    def lookup(_output, stage, period, arm):
        if stage != "confirmation" or (period, arm) not in records:
            raise FileNotFoundError("Missing synthetic confirmation")
        return records[(period, arm)]
    return lookup


class Round6HarnessGates(unittest.TestCase):
    def test_post_diagnostic_deadline_writer_and_independent_reader_reject_overcap(self):
        record = {"status": "completed", "budget_seconds": 180,
            "host_total_seconds": 179., "quality_metric": 1.,
            "metrics": {"calibrated": {"rmse": 1.}},
            "physical_fit_count": 2, "completed_fit_count": 2}
        # The old checkpoint was within cap; diagnostics/verification consume
        # 22 more seconds before the success publication point.
        with patch.object(lab.time, "monotonic", return_value=301.):
            lab.finalize_deadline(record, started=100., cap=180)
        self.assertEqual(record["host_total_seconds"], 201.)
        self.assertNotEqual(record["status"], "completed")
        self.assertIsNone(record["quality_metric"])
        self.assertIsNone(record["metrics"])
        self.assertEqual(record["physical_fit_count"], 2)
        self.assertEqual(record["completed_fit_count"], 2)
        forged = {**record, "status": "completed", "quality_metric": 1.}
        with self.assertRaises(ValueError):
            lab.admit_completed(forged, expected_cap=180)

    def test_reader_rejects_nonfinite_time_and_incomplete_physical_fit_ledger(self):
        record = {"status": "completed", "budget_seconds": 180,
            "host_total_seconds": 179., "physical_fit_count": 2,
            "completed_fit_count": 2}
        lab.finalize_deadline(record, started=100., cap=180, now=279.)
        lab.admit_completed(record, expected_cap=180)
        for changed in ({"host_total_seconds": float("nan")},
                        {"host_total_seconds": float("inf")},
                        {"host_total_seconds": -1.},
                        {"host_total_seconds": 180.00001},
                        {"host_total_seconds": 201., "deadline_admissible": True,
                         "deadline_checked_monotonic": 301.},
                        {"budget_seconds": 360}, {"deadline_admissible": False},
                        {"original_deadline_monotonic": 999.}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                lab.admit_completed({**record, **changed}, expected_cap=180)

        # Admission to a paired comparison also reconciles the actual persisted
        # engine and analytic operations, and binds every source/recipe byte.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runs/screen/p4/S20/summary.json"
            frozen = {"source": deepcopy(SOURCE), "source_files": {},
                      "folds": {"4": {"fit_rows": 1000}}, "period_column": "period"}
            original, recipe = lab._recipe("S20", "period", None)
            fits = {"fits": [{"started": True, "completed": True}] * 3,
                    "analytic_fits": [{"started": True, "completed": True}]}
            record.update(source=deepcopy(SOURCE), input_hashes={}, split=frozen["folds"]["4"],
                arm="S20", stage="screen", period=4, recipe=recipe,
                recipe_sha256=lab.json_hash(recipe), original_source_sha256=lab.text_hash(original),
                code_sha256=lab.text_hash(lab.instrument_source(original)), physical_fit_count=3,
                completed_fit_count=3, actual_analytic_fit_count=1, completed_analytic_fit_count=1,
                total_fit_operations=4)
            for name, field in (("segments.json", "segments_sha256"), ("receipt.json", "receipt_sha256"),
                                ("physical_fits.json", "physical_fits_sha256"), ("submission.csv", "output_sha256")):
                lab.write(path.parent / name, fits if name == "physical_fits.json" else {})
                record[field] = lab.digest(path.parent / name)

            def publish(value):
                lab.atomic(path, value)
                lab.atomic(root / "attempts.json", {"attempts": [{"status": "completed",
                    "summary": str(path.relative_to(root)), "summary_sha256": lab.digest(path)}]})

            publish(record)
            with patch.object(lab, "plan", return_value=frozen), \
                    patch.object(lab, "schema", return_value={"selected_config": None}):
                self.assertEqual(lab._record(root, "screen", 4, "S20")[0], record)
                wrong_arm_recipe = {**recipe, "arm": "S50"}
                for changed in ({"physical_fit_count": 4}, {"completed_fit_count": 2},
                                {"actual_analytic_fit_count": 0}, {"completed_analytic_fit_count": 0},
                                {"total_fit_operations": 3}, {"code_sha256": "changed"},
                                {"source": {**SOURCE, "head": "b" * 40}},
                                {"recipe": wrong_arm_recipe, "recipe_sha256": lab.json_hash(wrong_arm_recipe)}):
                    publish({**record, **changed})
                    with self.subTest(changed=changed), self.assertRaises(ValueError):
                        lab._record(root, "screen", 4, "S20")

    def test_known5_and_changed_frozen_inputs_stop_before_any_computation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_plan(root)
            with patch.object(lab, "source_identity", return_value=deepcopy(SOURCE)):
                self.assertEqual(set(lab.plan(root)["folds"]), {"2", "3", "4"})
                with patch.object(lab, "_reserve") as reserve, patch.object(lab, "_run") as run:
                    for call in (lambda: lab.run(root, "screen", 5, "S20"),
                                 lambda: lab.run(root, "confirmation", 5, "S20"),
                                 lambda: lab.case_frames(root, 5), lambda: lab.schema(root, 5),
                                 lambda: lab.discover(root, 5)):
                        with self.assertRaises(ValueError):
                            call()
                    reserve.assert_not_called()
                    run.assert_not_called()
                (root / "inputs/p3/fixture.csv").write_text("value\n999999\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    lab.plan(root)
            self.assertFalse((root / "discovery").exists())
            self.assertFalse((root / "runs").exists())

    def test_equal_caps_and_complete_confirmation_are_required_before_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "unequal_caps"
            lookup = decision_records(root, control_cap=180)
            with patch.object(lab, "plan", return_value={"source": SOURCE, "period_column": "period"}), \
                    patch.object(lab, "source_identity", return_value=SOURCE), \
                    patch.object(lab, "_checked_promotion", return_value={"finalists": ["S20"]}), \
                    patch.object(lab, "_record", side_effect=lookup):
                with self.assertRaisesRegex(ValueError, "180-second.*360-second"):
                    lab.freeze(root)
            self.assertFalse((root / "frozen_decision.json").exists())
            root = Path(temporary) / "missing_period4"
            lookup = decision_records(root, omit=(4, "S20"))
            with patch.object(lab, "plan", return_value={"source": SOURCE, "period_column": "period"}), \
                    patch.object(lab, "source_identity", return_value=SOURCE), \
                    patch.object(lab, "_checked_promotion", return_value={"finalists": ["S20"]}), \
                    patch.object(lab, "_record", side_effect=lookup):
                result = lab.evaluate(root)
                self.assertEqual(result["recommended_arm"], "I")
                finding = result["findings"][0]
                self.assertFalse(finding["complete"])
                self.assertFalse(finding["eligible_for_freeze"])
                self.assertIsNone(finding["candidate_sse"])
                self.assertEqual(finding["folds"][-1]["gates"], "unknown")
                with self.assertRaisesRegex(ValueError, "retain incumbent"):
                    lab.freeze(root)
            self.assertFalse((root / "frozen_decision.json").exists())
            self.assertFalse((root / "frozen_winner_source.py").exists())

    def test_preflight_source_failure_retains_zero_fit_ledger_and_original_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_plan(root)
            lab.write(root / "attempts.json", {"attempts": []})
            changed_source = {**deepcopy(SOURCE), "head": "b" * 40}
            with patch.object(lab, "source_identity", return_value=changed_source), \
                    patch.object(lab, "_run") as run:
                failed = lab.run(root, "screen", 4, "S20")
                self.assertEqual(failed["status"], "preflight_failed")
                self.assertEqual(failed["physical_fit_count"], 0)
                self.assertEqual(failed["completed_fit_count"], 0)
                self.assertEqual(failed["actual_analytic_fit_count"], 0)
                self.assertIsNone(failed["metrics"])
                summary = root / "runs/screen/p4/S20/summary.json"
                original_hash = lab.digest(summary)
                blocked = lab.run(root, "screen", 4, "S20")
                self.assertEqual(blocked["status"], "preflight_failed")
                self.assertEqual(blocked["physical_fit_count"], 0)
                self.assertEqual(lab.digest(summary), original_hash)
                run.assert_not_called()
            attempts = lab.read(root / "attempts.json")["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["summary_sha256"], original_hash)
            self.assertEqual([item["status"] for item in attempts], ["preflight_failed"] * 2)
            self.assertEqual(attempts[1]["kind"], "preflight")
            self.assertTrue(all(item["elapsed_seconds"] >= 0 for item in attempts))
            self.assertTrue((root / "queue.guard").is_file())

    def test_one_large_fold_loss_vetoes_improved_total_and_majority_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lookup = decision_records(root, candidate_rmse=(9., 9., 10.03))
            with patch.object(lab, "plan", return_value={"source": SOURCE, "period_column": "period"}), \
                    patch.object(lab, "source_identity", return_value=SOURCE), \
                    patch.object(lab, "_checked_promotion", return_value={"finalists": ["S20"]}), \
                    patch.object(lab, "_record", side_effect=lookup):
                result = lab.evaluate(root)
            finding = result["findings"][0]
            self.assertTrue(finding["complete"])
            self.assertEqual(finding["practical_wins"], 2)
            self.assertLess(finding["candidate_sse"], finding["control_sse"])
            self.assertFalse(finding["eligible_for_freeze"])
            self.assertEqual(result["recommended_arm"], "I")
            self.assertEqual(finding["history_gate"], "unknown")

    def test_novel_category_segment_loss_blocks_even_when_cold_gate_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_path, candidate_path = root / "control/summary.json", root / "candidate/summary.json"
            control = {"selected_ablation": "calibrated", "metrics": {"calibrated": {"rows": 1000, "sse": 10000.}}}
            candidate = {**deepcopy(control), "recipe": {"history_config": None}}
            before = [{"arm": "calibrated", "partition": "category_novelty", "segment": "novel",
                       "n": 150, "sse": 400.},
                      {"arm": "calibrated", "partition": "history", "segment": "schema_unavailable",
                       "n": 1000, "sse": 10000.},
                      {"arm": "calibrated", "partition": "category:kind", "segment": "tiny",
                       "n": 2, "sse": 1.}]
            after = deepcopy(before)
            after[0]["sse"] = 600.  # +50%, absolute +2% of control fold SSE.
            after[1]["sse"] = 9000.
            after[2]["sse"] = 100.  # Reportable but too small to be a significant group.
            lab.write(control_path.parent / "segments.json", {"rows": before})
            lab.write(candidate_path.parent / "segments.json", {"rows": after})
            result = lab._segments_gate(control, candidate, control_path, candidate_path)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["cold_history_gate"], "unknown")
            self.assertEqual(len(result["failures"]), 1)
            self.assertEqual(result["failures"][0]["rows"], 150)
            self.assertEqual(result["failures"][0]["absolute_sse_change"], 200.)
            after[0]["n"] = 149
            lab.atomic(candidate_path.parent / "segments.json", {"rows": after})
            with self.assertRaisesRegex(ValueError, "membership changed"):
                lab._segments_gate(control, candidate, control_path, candidate_path)


if __name__ == "__main__":
    unittest.main()
