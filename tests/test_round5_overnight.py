"""Saved receipt chains and fake native launches; no candidate is imported or fit."""
from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import experiment_round5_overnight as overnight
from agent.task_spec import load_task
from research_lab.integrity import file_hash, read_json, write_json
from tests.test_research_lab_audit import fixture
from tests.test_task_spec import make_task, write_csv


def receipt_job(root, *, profile="classification"):
    """Adapt the real audit fixture to the overnight worker's sealed layout."""
    case, _, worker_path, run = fixture(root / "saved")
    old_job = worker_path.parent
    run_relative = run.relative_to(old_job)
    job = root / "jobs" / profile
    job.parent.mkdir(parents=True)
    old_job.rename(job)

    def relocate(value):
        if isinstance(value, str):
            return value.replace(str(old_job), str(job))
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    for path in job.rglob("*.json"):
        # Frozen canonical recipe bytes must not gain write_json's newline.
        if path.name not in {"recipe.json", "research_recipe.json"}:
            write_json(path, relocate(read_json(path)))
    task, run = job / "task", job / run_relative
    flat = root / "flat"
    flat.mkdir()
    (flat / "solution.py").write_text('raise AssertionError("Never execute audit inputs")\n',
                                     encoding="utf-8")
    config = {"version": 1, "job_id": job.name, "profile": job.name,
              "task_dir": str(task), "task_path": str(task / "task.md"),
              "solver_root": str(flat), "worker_output": str(job / "worker_evidence.json"),
              "time_budget_seconds": 360}
    write_json(job / "config.json", config)
    spec = load_task(task / "task.md")
    contract = spec.to_dict()
    contract["prediction_kind"] = spec.prediction_kind
    seal = {"contract": contract, "metric": spec.metric.name, "test_ids": ["0005", "NA"],
            "input_sha256": overnight.hashes(task, case["input_hashes"])}
    write_json(job / "inputs.json", seal)
    write_json(job / "execution.json", read_json(job / "state.json")["execution"])
    worker = read_json(job / "worker.json")
    worker.update(config_sha256=file_hash(job / "config.json"), terminal_error=None,
                  original_deadline_monotonic=1360, native_started_monotonic=1000,
                  native_ended_monotonic=1011)
    write_json(job / "worker_evidence.json", worker)
    report = read_json(run / "run.json")
    selected_execution = {"exit_code": 0, "error": None, "timed_out": False,
                          "memory_limit_exceeded": False,
                          "resources": {"available": True, "cleanup_complete": True}}
    report["evaluations"]["final_check"]["execution"] = {"baseline": selected_execution}
    report["refit_execution"] = {"baseline": selected_execution}
    report["publication"]["recipe"] = "baseline"
    # Real default-OFF reports omit the rescue option and rescue report entirely.
    report["quality_options"] = {}
    save_report(job, run, report)
    plan = {"native_seconds_per_job": 360, "flat_root": str(flat),
            "flat_sha256": overnight.hashes(flat, ["solution.py"]),
            "release": {"fixture": "source identity is the only patched audit operation"},
            "job_seals": {job.name: {"config": file_hash(job / "config.json"),
                                     "inputs": file_hash(job / "inputs.json")}}}
    return job, plan, run


def save_report(job, run, report):
    """Refresh transport pointers, so metric mutations reach semantic auditing."""
    write_json(run / "run.json", report)
    worker = read_json(job / "worker_evidence.json")
    worker["run_report"]["sha256"] = file_hash(run / "run.json")
    write_json(job / "worker_evidence.json", worker)
    write_json(job / "task" / "solver_output" / "research_benchmark.json",
               {"run_report": worker["run_report"]["path"],
                "run_report_sha256": worker["run_report"]["sha256"]})


class OvernightAuditTests(unittest.TestCase):
    def audit(self, job, plan):
        with patch.object(overnight, "check_identity", return_value=plan["release"]):
            return overnight.audit_job(job, plan)

    def test_complete_receipt_chain_is_read_only_and_independently_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            job, plan, _ = receipt_job(root)
            before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            result = self.audit(job, plan)
            self.assertTrue(result["continuation_allowed"], result)
            self.assertEqual(result["qualification"], "requires_review")
            self.assertEqual(result["proof"]["outer_metric"]["value"], 1.0)
            self.assertEqual(result["proof"]["outer"]["fit_rows"], 40)
            self.assertEqual(result["proof"]["outer"]["predict_rows"], 20)
            self.assertEqual(result["proof"]["refit"]["fit_rows"], 60)
            self.assertEqual(result["proof"]["refit"]["predict_rows"], 2)
            self.assertEqual(before, {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()})

    def test_budget_search_must_match_sealed_config_and_observed_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            job, plan, run = receipt_job(Path(temporary).resolve())
            plan["research_search"] = "adaptive-budget"
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                self.audit(job, plan)
            config = read_json(job / "config.json")
            config["research_search"] = "adaptive-budget"
            write_json(job / "config.json", config)
            plan["job_seals"][job.name]["config"] = file_hash(job / "config.json")
            worker = read_json(job / "worker_evidence.json")
            worker["config_sha256"] = file_hash(job / "config.json")
            write_json(job / "worker_evidence.json", worker)
            self.assertFalse(self.audit(job, plan)["checks"]["one_adaptive_offline_invocation"])
            for item in worker["observations"]:
                if item["phase"] == "run_agent":
                    item["adaptive_budget"] = True
            write_json(job / "worker_evidence.json", worker)
            result = self.audit(job, plan)
            self.assertTrue(result["checks"]["one_adaptive_offline_invocation"])
            self.assertFalse(result["checks"]["budget_search_policy_recorded"])
            report = read_json(run / "run.json")
            report["quality_options"]["adaptive_budget"] = True
            report["coordinator"].update(search_policy="adaptive-budget", local_trial_limit=256,
                                         trial_limit_kind="safety")
            save_report(job, run, report)
            self.assertTrue(self.audit(job, plan)["continuation_allowed"])

    def test_receipt_and_exact_phase_input_tampering_cannot_pass(self):
        for mutation in ("receipt_nonce", "fit_value", "predict_value", "frozen_source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                job, plan, run = receipt_job(Path(temporary).resolve())
                report = read_json(run / "run.json")
                refit = Path(report["candidate_receipts"]["refit"]).parent
                if mutation == "receipt_nonce":
                    receipt = read_json(refit / "receipt.json")
                    receipt["nonce"] = "d" * 32
                    write_json(refit / "receipt.json", receipt)
                elif mutation == "frozen_source":
                    (run / "baseline_frozen" / "research_candidate.py").write_text("changed\n")
                else:
                    path = refit / ("fit.csv" if mutation == "fit_value" else "predict.csv")
                    with path.open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.reader(stream))
                    rows[1][1] = "changed-feature-value"
                    write_csv(path, rows)
                with self.assertRaises(ValueError):
                    self.audit(job, plan)

    def test_native_off_report_may_omit_rescue_but_on_must_record_request(self):
        for profile in ("regression_off", "regression_on"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                job, plan, _ = receipt_job(Path(temporary).resolve(), profile=profile)
                result = self.audit(job, plan)
                expected = profile == "regression_off"
                self.assertEqual(result["checks"]["requested_rescue_profile_recorded"], expected)
                self.assertEqual(result["continuation_allowed"], expected)

    def test_claimed_metric_is_recomputed_even_with_updated_report_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            job, plan, run = receipt_job(Path(temporary).resolve())
            report = read_json(run / "run.json")
            report["evaluations"]["final_check"]["baseline"]["value"] = .125
            save_report(job, run, report)
            with self.assertRaisesRegex(ValueError, "Claimed outer metric differs"):
                self.audit(job, plan)

    def test_input_bytes_and_flat_package_are_checked_against_original_seals(self):
        for mutation, check in (("task", "input_hashes_unchanged"),
                                ("flat_bytes", "flat_hashes_unchanged"),
                                ("flat_extra", "flat_hashes_unchanged")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                job, plan, _ = receipt_job(Path(temporary).resolve())
                if mutation == "task":
                    path = job / "task" / "task.md"
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                else:
                    name = "solution.py" if mutation == "flat_bytes" else "unexpected.py"
                    (Path(plan["flat_root"]) / name).write_text("# changed\n", encoding="utf-8")
                result = self.audit(job, plan)
                self.assertFalse(result["checks"][check])
                self.assertFalse(result["continuation_allowed"])

    def test_job_seals_reject_config_or_input_contract_replacement(self):
        for filename in ("config.json", "inputs.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                job, plan, _ = receipt_job(Path(temporary).resolve())
                value = read_json(job / filename)
                value["unexpected"] = "changed after preparation"
                write_json(job / filename, value)
                with self.assertRaisesRegex(ValueError, "Job input/config seals changed"):
                    self.audit(job, plan)

    def test_current_returned_report_pointer_never_falls_back_to_an_old_run(self):
        for mutation in ("missing_run", "hash", "origin"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                job, plan, _ = receipt_job(Path(temporary).resolve())
                worker = read_json(job / "worker_evidence.json")
                if mutation == "missing_run":
                    worker["run_report"]["path"] = ".agent_runs/" + "e" * 32 + "/run.json"
                else:
                    worker["run_report"]["sha256" if mutation == "hash" else "origin"] = "wrong"
                write_json(job / "worker_evidence.json", worker)
                with self.assertRaises(ValueError):
                    self.audit(job, plan)

    def test_unobserved_resources_and_native_contract_fail_closed(self):
        cases = (
            ("execution", ("resources", "available"), False, "tree_resources_observed"),
            ("execution", ("cleanup", "complete"), None, "owned_cleanup_complete"),
            ("execution", ("external_timeout",), True, "watchdog_not_reached"),
            ("worker", ("native_ended_monotonic",), 1361, "original_deadline_respected"),
            ("worker", ("config_sha256",), "wrong", "worker_config_unchanged"),
            ("report", ("api", "attempts"), 1, "zero_api"),
            ("report", ("publication", "new_result_published"), False, "fresh_full_refit_published"),
            ("report", ("evaluations", "final_check", "execution", "baseline", "timed_out"),
             True, "selected_outer_execution_clean"),
            ("report", ("refit_execution", "baseline", "resources", "cleanup_complete"),
             None, "selected_refit_execution_clean"),
        )
        for kind, keys, replacement, check in cases:
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                job, plan, run = receipt_job(Path(temporary).resolve())
                path = {"execution": job / "execution.json", "worker": job / "worker_evidence.json",
                        "report": run / "run.json"}[kind]
                value = read_json(path)
                target = value
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                if kind == "report":
                    save_report(job, run, value)
                else:
                    write_json(path, value)
                result = self.audit(job, plan)
                self.assertFalse(result["checks"][check], result)
                self.assertFalse(result["continuation_allowed"])


class OvernightStagingTests(unittest.TestCase):
    def test_two_fresh_copies_preserve_exact_inputs_and_literal_ids_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            task = make_task(source, labels=("00", "nan"), dtype="category")
            before = {p.relative_to(source).as_posix(): file_hash(p)
                      for p in source.rglob("*") if p.is_file()}
            left, right = root / "off", root / "on"
            off, on = overnight.stage_task(task, left), overnight.stage_task(task, right)
            self.assertEqual(off["input_sha256"], before)
            self.assertEqual(on["input_sha256"], before)
            self.assertEqual(off["test_ids"], ["0005", "NA"])
            self.assertEqual(on["test_ids"], ["0005", "NA"])
            for copy in (left, right):
                self.assertEqual({p.relative_to(copy).as_posix(): file_hash(p)
                                  for p in copy.rglob("*") if p.is_file()}, before)
            with self.assertRaises(FileExistsError):
                overnight.stage_task(task, left)
            self.assertEqual({p.relative_to(source).as_posix(): file_hash(p)
                              for p in source.rglob("*") if p.is_file()}, before)
            self.assertEqual({p.relative_to(left).as_posix(): file_hash(p)
                              for p in left.rglob("*") if p.is_file()}, before)


class OvernightQueueTests(unittest.TestCase):
    def fixture(self, root):
        queue = {"version": 1, "status": "running", "jobs": [
            {"profile": name, "status": "pending"} for name in overnight.PROFILES]}
        plan = {"native_seconds_per_job": 600}
        for item in queue["jobs"]:
            job = root / "jobs" / item["profile"]
            (job / "task").mkdir(parents=True)
            write_json(job / "config.json", {"profile": item["profile"]})
        return plan, queue

    def launch(self, events):
        def fake(argv, task, logs, timeout, memory, *, threads):
            profile = task.parent.name
            events.append(("launch", profile))
            self.assertEqual(Path(argv[4]), task.parent / "config.json")
            self.assertEqual(timeout, 630)
            self.assertEqual(memory, 28 * 1024**3)
            self.assertEqual(threads, 8)
            logs.mkdir()
            (logs / "child.stderr.txt").write_text("diagnostic: " + profile, encoding="utf-8")
            # Finishes much earlier than the 600 second cap. No sleep or solver.
            return {"exit_code": 0, "elapsed_seconds": .01, "early_finish": True}
        return fake

    def audited(self, events, *, failed_profile=None):
        def fake(job, plan):
            profile = job.name
            self.assertEqual(events[-1], ("launch", profile))
            self.assertTrue(read_json(job / "execution.json")["early_finish"])
            events.append(("audit", profile))
            permitted = profile != failed_profile
            return {"qualification": "requires_review", "continuation_allowed": permitted,
                    "checks": {"owned_cleanup_complete": permitted},
                    "proof": {"outer_metric": {"name": "RMSE", "value": 2 if profile == "regression_off" else 1},
                              "outer_split_sha256": "same-sealed-split"},
                    "rescue_regression": {"requested": profile == "regression_on"},
                    "diagnostic": "retain the original audit"}
        return fake

    def test_serial_queue_continues_after_early_finish_and_exports_actual_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, events = Path(temporary).resolve(), []
            plan, queue = self.fixture(root)
            with patch.object(overnight, "supervise", side_effect=self.launch(events)) as supervisor, \
                    patch.object(overnight, "audit_job", side_effect=self.audited(events)), \
                    patch.object(overnight.time, "monotonic", side_effect=[100, 200, 300]), \
                    redirect_stdout(io.StringIO()):
                overnight.run_queue(root, plan, queue)
            self.assertEqual(events, [(event, name) for name in overnight.PROFILES
                                       for event in ("launch", "audit")])
            self.assertEqual([float(call.args[0][-1]) for call in supervisor.call_args_list],
                             [700, 800, 900])
            saved = read_json(root / "queue.json")
            self.assertEqual(saved["status"], "completed")
            self.assertEqual([item["status"] for item in saved["jobs"]], ["completed"] * 3)
            pair = read_json(root / "comparison.json")
            self.assertEqual(pair["status"], "comparable")
            self.assertEqual(pair["on_minus_off"], -1)
            with zipfile.ZipFile(root / "evidence.zip") as bundle:
                self.assertIn("comparison.json", bundle.namelist())
                self.assertIn("bundle_manifest.json", bundle.namelist())
                for profile in overnight.PROFILES:
                    name = f"jobs/{profile}/execution/child.stderr.txt"
                    self.assertEqual(bundle.read(name).decode(), "diagnostic: " + profile)

    def test_failed_required_check_stops_remaining_jobs_and_preserves_original_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, events = Path(temporary).resolve(), []
            plan, queue = self.fixture(root)
            with patch.object(overnight, "supervise", side_effect=self.launch(events)), \
                    patch.object(overnight, "audit_job", side_effect=self.audited(
                        events, failed_profile="regression_on")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "owned_cleanup_complete"):
                    overnight.run_queue(root, plan, queue)
            self.assertEqual(events, [("launch", "regression_off"), ("audit", "regression_off"),
                                      ("launch", "regression_on"), ("audit", "regression_on")])
            saved = read_json(root / "queue.json")
            self.assertEqual(saved["status"], "stopped")
            self.assertEqual([item["status"] for item in saved["jobs"]],
                             ["completed", "failed", "pending"])
            audit = read_json(root / "jobs" / "regression_on" / "audit.json")
            self.assertEqual(audit["diagnostic"], "retain the original audit")
            self.assertFalse(audit["checks"]["owned_cleanup_complete"])
            self.assertEqual(read_json(root / "comparison.json")["status"], "incomplete")
            self.assertFalse((root / "jobs" / "classification" / "execution.json").exists())
            with zipfile.ZipFile(root / "evidence.zip") as bundle:
                self.assertIn(b"retain the original audit", bundle.read("jobs/regression_on/audit.json"))
                self.assertIn(b"regression_on", bundle.read("jobs/regression_on/execution/child.stderr.txt"))

    def test_receipt_audit_exception_is_preserved_with_logs_and_no_next_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, events = Path(temporary).resolve(), []
            plan, queue = self.fixture(root)
            with patch.object(overnight, "supervise", side_effect=self.launch(events)), \
                    patch.object(overnight, "audit_job", side_effect=ValueError("receipt nonce changed")), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "receipt nonce changed"):
                    overnight.run_queue(root, plan, queue)
            self.assertEqual(events, [("launch", "regression_off")])
            saved = read_json(root / "queue.json")
            self.assertEqual([item["status"] for item in saved["jobs"]], ["failed", "pending", "pending"])
            audit = read_json(root / "jobs" / "regression_off" / "audit.json")
            self.assertEqual(audit["error"], "ValueError: receipt nonce changed")
            self.assertFalse(audit["continuation_allowed"])
            with zipfile.ZipFile(root / "evidence.zip") as bundle:
                self.assertIn(b"receipt nonce changed", bundle.read("jobs/regression_off/audit.json"))
                self.assertIn("jobs/regression_off/execution/child.stderr.txt", bundle.namelist())


if __name__ == "__main__":
    unittest.main()
