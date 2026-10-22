"""Saved v3 baseline attempts exercise cleanup qualification without fitting."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_lab import audit
from research_lab.integrity import file_hash, read_json, write_json
from tests.test_research_lab_audit import fixture, terminal


def baseline_fixture(root, *, cleanup=True, available=True, observed=True, timed_out=True):
    case, plan, worker, run = fixture(root)
    task = Path(case["arms"]["incumbent"]["task_dir"])
    phase = run / "workspace" / ".agent_runs" / ("f" * 32) / "trial" / "selection"
    (phase / "execution").mkdir(parents=True)
    write_json(phase / "config.json", {"id": "trial", "phase": "selection",
                                       "candidate_dir": str(phase)})
    execution = {"exit_code": 15 if timed_out else 0, "timed_out": timed_out,
                 "memory_limit_exceeded": False, "error": None,
                 "resources": {"available": available, "cleanup_complete": cleanup,
                               "peak_rss_bytes": 192212992}}
    search = {"phase": "finished", "status": "completed", "candidates": [
        {"id": "trial", "phase": "selection", "status": "timed_out" if timed_out else "completed",
         "error": "Attempt time limit" if timed_out else None,
         "resources": {"available": available, "exit_code": execution["exit_code"],
                       "timed_out": timed_out, "memory_exceeded": False}}]}
    startup = run / "startup"
    write_json(startup / "automl_report.json", {"search": search})
    report = read_json(run / "run.json")
    report["baseline_search"] = {"search": search, "execution": {"exit_code": 0,
        "resources": {"available": True, "cleanup_complete": True, "peak_rss_bytes": 1000}}}
    write_json(run / "run.json", report)
    value = read_json(worker)
    value["run_report"]["sha256"] = file_hash(run / "run.json")
    if observed:
        record = {"phase": "automl_execution", "source": "baseline_subprocess",
                  "directory": phase.relative_to(task).as_posix(), "execution": execution}
        sidecar = startup / "automl_observations.json"
        write_json(sidecar, {"version": 1, "observations": [record], "pending": [], "errors": []})
        value["observations"].extend([record, {"phase": "automl_observer_complete",
            "path": sidecar.relative_to(task).as_posix(), "sha256": file_hash(sidecar)}])
    write_json(worker, value)
    write_json(task / "solver_output" / "research_benchmark.json", {
        "run_report": value["run_report"]["path"], "run_report_sha256": value["run_report"]["sha256"]})
    return case, plan, worker, run


class BaselineResourceTests(unittest.TestCase):
    def test_timed_out_selection_with_own_cleanup_proof_is_observed_and_reauditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, run = baseline_fixture(root)
            native_before = (run / "run.json").read_bytes()
            result = terminal(root, case, plan, worker)
            self.assertEqual(result["contract_errors"], [])
            self.assertEqual(result["status"], "completed", result)
            self.assertTrue(result["quality_verified"])
            self.assertEqual(result["resources"]["qualification"], "observed")
            self.assertEqual(result["resources"]["censored_cleanup_paths"], [])
            self.assertEqual(len(result["resources"]["execution_records"]), 4)
            self.assertEqual((run / "run.json").read_bytes(), native_before)
            before = {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}
            with patch.object(audit, "_load", return_value=plan):
                rechecked = audit.re_audit(root)
            self.assertTrue(rechecked["valid"], rechecked)
            self.assertEqual(before, {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()})

    def test_unknown_or_failed_child_cleanup_still_censors_despite_clean_outer_tree(self):
        for cleanup in (None, False):
            with self.subTest(cleanup=cleanup), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, _ = baseline_fixture(root, cleanup=cleanup)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertEqual(result["status"], "censored")
                self.assertFalse(result["quality_verified"])

    def test_unavailable_resources_and_old_missing_observations_are_never_qualified(self):
        for options, qualification in (({"available": False}, "provisional_resource_observation"),
                ({"observed": False}, "censored"),
                ({"observed": False, "timed_out": False}, "provisional_resource_observation")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, _ = baseline_fixture(root, **options)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertEqual(result["resources"]["qualification"], qualification)

    def test_missing_phase_is_not_hidden_by_an_observed_phase_of_same_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, run = baseline_fixture(root)
            phase = run / "workspace" / ".agent_runs" / ("f" * 32) / "trial" / "verification"
            (phase / "execution").mkdir(parents=True)
            write_json(phase / "config.json", {"id": "trial", "phase": "verification",
                                               "candidate_dir": str(phase)})
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertEqual(result["resources"]["qualification"], "censored")

    def test_invalid_or_incomplete_observer_evidence_cannot_admit_quality(self):
        for mutation in ("outside_run", "duplicate", "observer_error", "missing_seal", "sidecar_drift", "report_drift"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, run = baseline_fixture(root)
                value = read_json(worker)
                record = next(item for item in value["observations"] if item["phase"] == "automl_execution")
                if mutation == "outside_run":
                    record["directory"] = ".agent_runs/" + "e" * 32 + "/trial/selection"
                elif mutation == "duplicate":
                    value["observations"].append(dict(record))
                elif mutation == "observer_error":
                    value["observations"].append({"phase": "automl_observer_error", "error": "Interrupted trial"})
                elif mutation == "missing_seal":
                    value["observations"] = [item for item in value["observations"]
                                             if item["phase"] != "automl_observer_complete"]
                elif mutation == "sidecar_drift":
                    write_json(run / "startup" / "automl_observations.json", {"version": 2})
                else:
                    write_json(run / "startup" / "automl_report.json", {"search": {}})
                write_json(worker, value)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertFalse(result["quality_verified"], result)
                self.assertNotEqual(result["resources"]["qualification"], "observed")

    def test_true_launcher_deadline_is_not_turned_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, _ = baseline_fixture(root)
            state_path = Path(case["arms"]["incumbent"]["state_path"])
            state = read_json(state_path)
            state["execution"].update(exit_code=15, external_timeout=True)
            write_json(state_path, state)
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertEqual(result["status"], "solver_failed")
            self.assertFalse(result["quality_verified"])


if __name__ == "__main__":
    unittest.main()
