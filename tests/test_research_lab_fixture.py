"""Typed short native AutoML evidence, without running model fits."""
from __future__ import annotations

from dataclasses import replace
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.automl import data_fingerprint
from agent.search import seed_candidate
from agent.task_spec import load_task
from offline_baseline import read_csv
from research_lab import audit
from research_lab.integrity import file_hash, read_json, write_json
from tests.test_research_lab_audit import fixture, terminal


def automl_fixture(root, *, refit=False):
    """Saved trusted-code protocol fixture; no candidate is imported or fitted."""
    case, plan, worker, old_run = fixture(root)
    task = old_run.parent.parent
    spec = load_task(task / "task.md")
    shutil.rmtree(old_run)
    (task / "solver_output" / "research_benchmark.json").unlink()
    run = task / ".agent_runs" / ("d" * 32)
    identifier = "candidate_1" if refit else "seed"
    phase_name = "refit" if refit else "seed"
    phase = run / identifier / phase_name
    inputs = phase / "inputs"
    inputs.mkdir(parents=True)
    snapshot = run / "input_snapshot"
    shutil.copytree(spec.data_dir, snapshot)
    for path in spec.data_dir.iterdir():
        shutil.copyfile(path, inputs / path.name)
    (phase / "task.md").write_bytes((task / "task.md").read_bytes())
    recipe = seed_candidate(True)
    recipe["id"] = identifier
    fitting, held = list(range(40)), list(range(40, 60))
    local = replace(spec, task_path=phase / "task.md", data_dir=inputs)
    write_json(phase / "config.json", {**recipe, "candidate_dir": str(phase),
        "phase": phase_name, "task_spec": local.to_dict(), "fit_rows": fitting,
        "validation_rows": held})
    train = read_csv(inputs / "train.csv")
    held_frame = spec.holdout_frame(train[spec.id_column].iloc[held],
        train[spec.target_column].iloc[held], train[spec.target_column].iloc[held])
    held_frame.to_csv(phase / "holdout_predictions.csv", index=False)
    if refit:
        outer = run / identifier / "verification"
        outer.mkdir()
        (outer / "holdout_predictions.csv").write_bytes((phase / "holdout_predictions.csv").read_bytes())
    for destination in (phase / "submission.csv",):
        destination.write_bytes((task / "submission.csv").read_bytes())
    write_json(phase / "durations.json", {"fit_seconds": .1, "predict_seconds": .01,
                                         "total_seconds": .11})
    execution = phase / "execution"
    execution.mkdir()
    (execution / "0001.stdout.txt").write_text("seed: refitting on all 60 rows\n")
    checkpoint = run / "checkpoints" / ("e" * 32)
    checkpoint.mkdir(parents=True)
    for name in ("submission.csv", "holdout_predictions.csv"):
        (checkpoint / name).write_bytes((phase / name).read_bytes())
    (run / "holdout_predictions.csv").write_bytes((phase / "holdout_predictions.csv").read_bytes())
    write_json(run / "checkpoint.json", {"version": 1,
        "directory": checkpoint.relative_to(run).as_posix(),
        "submission_sha256": file_hash(checkpoint / "submission.csv"),
        "holdout_sha256": file_hash(checkpoint / "holdout_predictions.csv")})
    report_path = run / "automl_report.json"
    report = {"mode": "offline", "api": {"used": False, "calls": 0},
        "time_budget_seconds": 56, "elapsed_seconds": 10, "output_valid": True,
        "input_fingerprint": data_fingerprint(snapshot), "report_path": str(report_path),
        "metric": {"name": "Accuracy", "value": 1.0, "higher_is_better": True},
        "search": {"status": "completed", "phase": "finished", "best_candidate": identifier,
                   "candidates": [{"id": identifier, "resources": {"available": True, "exit_code": 0}}]},
        "candidate_configs": {identifier: recipe["model_config"]},
        "candidate_feature_configs": {identifier: recipe["feature_config"]}}
    write_json(report_path, report)
    config_path = worker.parent / "config.json"
    config = read_json(config_path)
    config.update(time_budget_seconds=60, require_adaptive=False)
    write_json(config_path, config)
    value = read_json(worker)
    value.update(config_sha256=file_hash(config_path), original_time_budget_seconds=60,
        cli_time_budget_seconds=58, observations=[{"phase": "automl_execution",
            "directory": phase.relative_to(task).as_posix(), "execution": {"exit_code": 0,
                "timed_out": False, "memory_limit_exceeded": False, "error": None,
                "resources": {"available": True, "cleanup_complete": True, "peak_rss_bytes": 1000}}}],
        run_report={"kind": "automl", "origin": "returned",
            "path": report_path.relative_to(task).as_posix(), "sha256": file_hash(report_path)})
    value["cli_argv"][-1] = "58"
    write_json(worker, value)
    plan.update(profile="fixture-smoke", policy_comparison=False, time_budget_seconds=60)
    return case, plan, worker, run


class FixtureRouteTests(unittest.TestCase):
    def test_selected_non_seed_full_refit_is_verified_from_its_exact_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            case, plan, worker, _ = automl_fixture(root, refit=True)
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertTrue(result["mechanism_verified"], result)
            self.assertEqual(result["provenance"]["refit"]["phase"], "refit")
            self.assertFalse(result["quality_verified"])

    def test_native_v3_fixture_is_also_mechanism_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            case, plan, worker, _ = fixture(root)
            plan["profile"] = "fixture-smoke"
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertTrue(result["mechanism_verified"], result)
            self.assertFalse(result["quality_verified"])

    def test_short_native_automl_is_mechanism_only_and_reaudit_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            case, plan, worker, _ = automl_fixture(root)
            result = terminal(root, case, plan, worker)
            self.assertEqual(result["status"], "completed", result)
            self.assertTrue(result["mechanism_verified"])
            self.assertFalse(result["quality_verified"])
            self.assertFalse(result["quality_admission"])
            self.assertEqual(result["provenance"]["refit"]["fit_rows"], 60)
            self.assertEqual(result["actual_api"]["attempts"], 0)
            self.assertEqual(result["resources"]["qualification"], "observed")
            before = {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}
            with patch.object(audit, "_load", return_value=plan):
                rechecked = audit.re_audit(root)
            self.assertTrue(rechecked["valid"], rechecked)
            self.assertTrue(rechecked["arms"][0]["mechanism_verified"])
            self.assertEqual(before, {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()})

    def test_missing_nested_observation_remains_provisional_and_unknown_failed_cleanup_censors(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                case, plan, worker, _ = automl_fixture(root)
                value = read_json(worker)
                if missing:
                    value["observations"] = []
                else:
                    execution = value["observations"][0]["execution"]
                    execution["timed_out"] = True
                    execution["resources"]["cleanup_complete"] = None
                write_json(worker, value)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertEqual(result["resources"]["qualification"],
                                 "provisional_resource_observation" if missing else "censored")
                self.assertEqual(result["mechanism_verified"], missing)
                self.assertFalse(result["quality_verified"])

    def test_automl_route_never_satisfies_real_adaptive_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            case, plan, worker, _ = automl_fixture(root)
            plan["profile"] = "pilot"
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertEqual(result["status"], "unverified_contract")
            self.assertFalse(result["quality_verified"])
            self.assertFalse(result.get("mechanism_verified", False))

    def test_short_report_and_model_chain_are_checked_without_glob_selection(self):
        for mutation in ("pointer", "checkpoint", "snapshot", "phase_inputs", "fit_rows", "metric", "api", "missing_api", "bool_api", "prior"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                case, plan, worker, run = automl_fixture(root)
                phase = run / "seed" / "seed"
                report_path = run / "automl_report.json"
                report = read_json(report_path)
                if mutation == "pointer":
                    value = read_json(worker); value["run_report"]["path"] = ".agent_runs/unknown/automl_report.json"
                    write_json(worker, value)
                elif mutation == "checkpoint":
                    (run / "checkpoints" / ("e" * 32) / "submission.csv").write_text("tampered")
                elif mutation in {"snapshot", "phase_inputs"}:
                    path = (run / "input_snapshot" if mutation == "snapshot" else phase / "inputs") / "train.csv"
                    path.write_bytes(path.read_bytes() + b"changed")
                elif mutation == "fit_rows":
                    config = read_json(phase / "config.json"); config["fit_rows"] = [0]
                    write_json(phase / "config.json", config)
                else:
                    if mutation == "metric": report["metric"]["value"] = .125
                    elif mutation == "api": report["api"]["calls"] = 1
                    elif mutation == "missing_api": report.pop("api")
                    elif mutation == "bool_api": report["api"]["calls"] = False
                    else: report["output_valid"] = False
                    write_json(report_path, report)
                    value = read_json(worker); value["run_report"]["sha256"] = file_hash(report_path)
                    write_json(worker, value)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertFalse(result.get("mechanism_verified", False), result)
                self.assertFalse(result["quality_verified"])


if __name__ == "__main__":
    unittest.main()
