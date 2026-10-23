"""Zero-fit tests for development boundaries and consequential acceptance gates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_round5_regression as lab


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
    frozen = {"version": 1, "round": 5, "base_sha": lab.BASE_SHA, "source": deepcopy(SOURCE),
        "plan_document_sha256": lab.digest(lab.PLAN_DOCUMENT),
        "plan_recipes_sha256": lab.digest(lab.PLAN_RECIPES), "folds": folds,
        "period_column": "period", "size_column": "value", "source_files": {},
        "screen": {"period": 4, "arms": list(lab.ARMS), "cap_seconds": lab.SCREEN_SECONDS},
        "confirmation": {"periods": list(lab.DEVELOPMENT), "maximum_finalists": 2,
                         "cap_seconds": lab.CONFIRM_SECONDS},
        "known5_evaluation": "forbidden_in_round5", "total_heavy_seconds": lab.TOTAL_SECONDS,
        "history": {"scope": "host_inner_fit", "max_cell_visits": 8000000,
                    "max_checked_configs": 48, "extended": False}}
    lab.write(root / "plan.json", frozen)
    return frozen


def decision_records(root, *, candidate_rmse=(9., 9., 9.), omit=None, control_cap=360):
    """Explicit synthetic score evidence for testing decisions, never ML claims."""
    lab.write(root / "promotion.json", {"finalists": ["E50"]})
    records = {}
    for period, rmse in zip(lab.DEVELOPMENT, candidate_rmse):
        for arm in ("I", "E50"):
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


class Round5HarnessGates(unittest.TestCase):
    def test_known5_and_changed_frozen_inputs_stop_before_any_computation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_plan(root)
            with patch.object(lab, "source_identity", return_value=deepcopy(SOURCE)):
                self.assertEqual(set(lab.plan(root)["folds"]), {"2", "3", "4"})
                with patch.object(lab, "_reserve") as reserve, patch.object(lab, "_run") as run:
                    for call in (lambda: lab.run(root, "screen", 5, "E50"),
                                 lambda: lab.run(root, "confirmation", 5, "E50"),
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
                    patch.object(lab, "_record", side_effect=lookup):
                with self.assertRaisesRegex(ValueError, "180-second.*360-second"):
                    lab.freeze(root)
            self.assertFalse((root / "frozen_decision.json").exists())
            root = Path(temporary) / "missing_period4"
            lookup = decision_records(root, omit=(4, "E50"))
            with patch.object(lab, "plan", return_value={"source": SOURCE, "period_column": "period"}), \
                    patch.object(lab, "source_identity", return_value=SOURCE), \
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

    def test_one_large_fold_loss_vetoes_improved_total_and_majority_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lookup = decision_records(root, candidate_rmse=(9., 9., 10.6))
            with patch.object(lab, "plan", return_value={"source": SOURCE, "period_column": "period"}), \
                    patch.object(lab, "source_identity", return_value=SOURCE), \
                    patch.object(lab, "_record", side_effect=lookup):
                result = lab.evaluate(root)
            finding = result["findings"][0]
            self.assertTrue(finding["complete"])
            self.assertEqual(finding["practical_wins"], 2)
            self.assertLess(finding["candidate_sse"], finding["control_sse"])
            self.assertFalse(finding["every_fold_within_5pct"])
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
