"""Serial campaign lifecycle fixtures; no recipe/process/model is executed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app import campaigns
from app.profiling import DataCatalog
from tests.test_task_spec import make_task


class DormantThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args
    def start(self):
        pass
    def join(self, timeout=None):
        pass


class FakeJobs:
    def __init__(self, catalog):
        self.catalog = catalog
        self.owners, self.calls, self.stops = {}, [], []
        self.active = None
        self.auto_complete = True
        self.outcomes = []
        self.complete_hook = None
        self.raise_after_launch = False

    def active_job(self):
        return deepcopy(self.active)

    def start(self, task_id, action, **kwargs):
        owner = kwargs["campaign_owner"]
        pair = (owner["campaign_id"], owner["step_id"])
        if pair in self.owners:
            return deepcopy(self.owners[pair]["job"])
        identifier = f"{len(self.calls) + 1:032x}"
        split = {"seed": 1729, "validation_fraction": .2, "stratified": False,
                 "validation_rows": 12, "validation_hash": "c" * 64, "strategy": "random"}
        fingerprint = self.catalog.fingerprint(task_id)
        metric = {"name": "RMSE", "value": 3 + len(self.calls), "higher_is_better": False}
        key = campaigns._hash({"dataset_fingerprint": fingerprint, "metric": metric["name"],
                              "higher_is_better": False, "split": split})
        job = {"id": identifier, "task_id": task_id, "action": action, "status": "running", "stage": "training",
               "source": "local", "output_valid": True, "metric": metric, "split": split,
               "dataset_fingerprint": fingerprint, "comparison_key": key, "model": action,
               "elapsed_seconds": 1, "error": None}
        self.owners[pair] = {"job": job, "cleanup_confirmed": True, "resources_ok": True}
        self.calls.append((task_id, action, deepcopy(kwargs)))
        self.active = deepcopy(job)
        if self.raise_after_launch:
            raise OSError("crash after durable owner commit")
        return deepcopy(job)

    def campaign_job(self, campaign_id, step_id):
        capsule = self.owners.get((campaign_id, step_id))
        if capsule and capsule["job"]["status"] == "running" and self.auto_complete:
            outcome = self.outcomes.pop(0) if self.outcomes else {}
            capsule["job"].update(status=outcome.get("status", "completed"), stage="finished")
            capsule.update({key: value for key, value in outcome.items() if key != "status"})
            if capsule["job"]["status"] == "failed":
                capsule["job"].update(error="model failure", output_valid=False, metric=None)
            self.active = None
            if self.complete_hook:
                self.complete_hook()
        return deepcopy(capsule)

    def stop(self, task_id, job_id):
        self.stops.append((task_id, job_id))
        for capsule in self.owners.values():
            if capsule["job"]["id"] == job_id:
                capsule["job"].update(status="cancelled", stage="finished", output_valid=False)
                capsule["cleanup_confirmed"] = True
        self.active = None


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        make_task(self.data / "regression", metric="RMSE")
        make_task(self.data / "classification", metric="Macro F1")
        self.catalog = DataCatalog(self.data)
        self.jobs = FakeJobs(self.catalog)
        self.native = Mock()
        self.native.active_queue.return_value = None
        self.gate = threading.RLock()
        self.identity = {"sha": "a" * 40, "source_fingerprint": "b" * 64, "dirty": False}
        repo = self.root / "repo"
        repo.mkdir()
        pins = (campaigns.ROOT / "requirements.txt").read_text(encoding="utf-8")
        (repo / "requirements.txt").write_text(pins, encoding="utf-8")
        self.versions = dict(line.split("==", 1) for line in pins.splitlines()
                             if line.strip() and not line.lstrip().startswith("#"))
        self.enterContext(patch.object(campaigns.metadata, "version", side_effect=lambda name: self.versions[name]))
        self.disk_free = 100 * 1024**3
        self.enterContext(patch.object(campaigns.shutil, "disk_usage", side_effect=lambda path: SimpleNamespace(free=self.disk_free)))
        self.enterContext(patch.object(campaigns, "source_identity", side_effect=lambda root: dict(self.identity)))
        self.enterContext(patch.object(campaigns.threading, "Thread", DormantThread))
        self.manager = self.new_manager()
        self.addCleanup(self.manager.close)

    def new_manager(self):
        return campaigns.CampaignManager(self.root / "state", self.catalog, self.jobs,
            execution_gate=self.gate, native_queues=self.native, source_root=self.root / "repo")

    def body(self, count=2, **updates):
        result = {"preset": "custom", "task_ids": ["regression"], "name": "fixture", "max_hours": 2,
                  "steps": [{"task_id": "regression", "action": "automl", "minutes": 5} for _ in range(count)]}
        result.update(updates)
        return result

    def queued(self, body=None):
        body = body or self.body()
        preview = self.manager.preview(body)
        self.assertTrue(preview["ready"], preview)
        detail = self.manager.start({**body, "preview_token": preview["plan_sha256"]})
        return detail["id"], self.manager._active

    def run_queue(self, identifier, active):
        self.manager._run(identifier, active)
        return self.manager.detail(identifier)

    def test_presets_interleave_day_and_report_real_worker_caps(self):
        preview = self.manager.preview({"preset": "day", "task_ids": ["regression", "classification"]})
        self.assertTrue(preview["ready"], preview)
        self.assertEqual([(row["task_id"], row["minutes"]) for row in preview["steps"]],
            [(task, minutes) for minutes in (30, 90, 175) for task in ("regression", "classification")])
        self.assertEqual(preview["worker_upper_bound_seconds"], 590 * 60 + 6 * 30)
        self.assertFalse(preview["preparation_included"])
        models = self.manager.preview({"preset": "models", "task_ids": ["regression", "classification"]})
        self.assertEqual(len(models["steps"]), 10)
        self.assertEqual(models["worker_upper_bound_seconds"], 10 * campaigns.LOCAL_TIMEOUT)
        fixed = self.body(count=1)
        fixed["steps"][0].update(action="baseline", minutes=1)
        self.assertEqual(self.manager.preview(fixed)["steps"][0]["worker_timeout_seconds"], 1800)
        self.assertEqual(self.jobs.calls, [])

    def test_invalid_payload_and_stale_preview_never_launch(self):
        invalid = [self.body(task_ids=["regression", "regression"]), self.body(max_hours=True), self.body(max_hours=13),
                   self.body(command="bad"), self.body(steps=[{"task_id": "regression", "action": "agent", "minutes": 5}]),
                   self.body(steps=[{"task_id": "regression", "action": "automl", "minutes": 5, "path": "x"}]),
                   self.body(steps=[{"task_id": "regression", "action": "automl", "minutes": 5}] * 13),
                   self.body(max_hours=.01), self.body(preview_token="0" * 64)]
        for body in invalid:
            with self.subTest(body=body):
                self.assertFalse(self.manager.preview(body)["ready"])
                with self.assertRaises(ValueError):
                    self.manager.start(body)
        self.assertEqual(self.jobs.calls, [])
        with self.assertRaisesRegex(ValueError, "preview_token"):
            self.manager.start(self.body())

    def test_runtime_versions_are_frozen_and_drift_excludes_the_result(self):
        identifier, active = self.queued()
        detail = self.manager.detail(identifier)
        self.assertEqual(len(detail["runtime_seal"]["packages"]), 15)
        self.assertNotIn(str(self.root), json.dumps(detail["runtime_seal"]))
        first = next(iter(self.versions))
        self.jobs.complete_hook = lambda: self.versions.update({first: "999.0"})
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertFalse(detail["runtime_seal"]["valid"])
        self.assertFalse(detail["steps"][0]["admissible"])
        self.assertEqual(detail["comparisons"], [])

    def test_disk_estimate_covers_all_input_copies_and_stops_before_next_copy(self):
        preview = self.manager.preview(self.body())
        size = self.manager._input_size("regression")
        self.assertEqual(preview["disk_estimate"]["required_free_bytes"], 2 * size + 2 * 1024**3)
        self.assertFalse(preview["disk_estimate"]["model_size_known"])
        self.disk_free = 1
        self.assertFalse(self.manager.preview(self.body())["ready"])
        self.disk_free = 100 * 1024**3
        identifier, active = self.queued()
        self.jobs.complete_hook = lambda: setattr(self, "disk_free", 1)
        detail = self.run_queue(identifier, active)
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertEqual(detail["status"], "failed")

    def test_current_owner_and_resource_proofs_are_required_for_detail_admission(self):
        identifier, active = self.queued()
        detail = self.run_queue(identifier, active)
        first, second = detail["steps"]
        self.jobs.owners.pop((identifier, first["id"]))
        self.jobs.owners[(identifier, second["id"])]["resources_ok"] = None
        refreshed = self.manager.detail(identifier)
        self.assertEqual([step["admissible"] for step in refreshed["steps"]], [False, False])
        self.assertEqual(refreshed["comparisons"], [])

    def test_serial_jobs_have_durable_distinct_owners_and_exact_comparison_groups(self):
        identifier, active = self.queued()
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(len(self.jobs.calls), 2)
        owners = [row[2]["campaign_owner"] for row in self.jobs.calls]
        self.assertEqual({owner["campaign_id"] for owner in owners}, {identifier})
        self.assertEqual(len({owner["step_id"] for owner in owners}), 2)
        self.assertTrue(all(step["admissible"] for step in detail["steps"]))
        self.assertEqual(len(detail["comparisons"]), 1)
        self.assertEqual(detail["comparisons"][0]["best_job_id"], detail["steps"][0]["job_id"])
        self.assertIsNone(self.manager.active_campaign())

    def test_pause_waits_for_current_job_resume_only_runs_pending_steps(self):
        identifier, active = self.queued()
        self.jobs.complete_hook = lambda: self.manager.pause(identifier)
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "paused")
        self.assertEqual([step["status"] for step in detail["steps"]], ["completed", "pending"])
        self.assertEqual(self.jobs.stops, [])
        self.assertEqual(self.manager.active_campaign()["id"], identifier)
        self.jobs.complete_hook = None
        self.manager.resume(identifier)
        detail = self.run_queue(identifier, self.manager._active)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(len(self.jobs.calls), 2)

    def test_model_failure_can_continue_once_but_unknown_resources_stop(self):
        identifier, active = self.queued()
        self.jobs.outcomes = [{"status": "failed"}, {}]
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual([step["admissible"] for step in detail["steps"]], [False, True])
        identifier, active = self.queued()
        self.jobs.outcomes = [{"resources_ok": None}]
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(detail["steps"][1]["status"], "pending")

    def test_cleanup_unknown_reserves_campaign_and_retry_stop_does_not_launch(self):
        identifier, active = self.queued()
        self.jobs.outcomes = [{"cleanup_confirmed": None}]
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "cleanup_pending")
        self.assertEqual(self.manager.active_campaign()["id"], identifier)
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertFalse(self.manager.preview(self.body())["ready"])
        capsule = self.jobs.owners[(identifier, detail["steps"][0]["id"])]
        capsule["cleanup_confirmed"] = True
        self.assertEqual(self.manager.stop(identifier)["status"], "stopped")
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertEqual(self.jobs.stops, [])

    def test_start_exception_after_owner_commit_recovers_job_and_stops_only_it(self):
        identifier, active = self.queued()
        self.jobs.auto_complete = False
        self.jobs.raise_after_launch = True
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertEqual(detail["steps"][0]["job_id"], "1".zfill(32))
        self.assertEqual(self.jobs.stops, [("regression", "1".zfill(32))])

    def test_stop_never_kills_foreign_job_and_comparisons_reject_changed_receipt(self):
        identifier, active = self.queued()
        detail = self.run_queue(identifier, active)
        step = self.manager._records[identifier]["steps"][0]
        step["status"] = "running"
        self.manager._records[identifier]["status"] = "cleanup_pending"
        self.jobs.active = {"id": "f" * 32, "task_id": "classification", "status": "running"}
        self.manager.stop(identifier)
        self.assertEqual(self.jobs.stops, [])
        self.assertEqual(self.jobs.active["id"], "f" * 32)
        capsule = self.jobs.owners[(identifier, detail["steps"][1]["id"])]
        capsule["job"]["comparison_key"] = "0" * 64
        self.assertEqual(self.manager.detail(identifier)["comparisons"], [])

    def test_restore_quarantines_partial_json_and_tampered_frozen_plan(self):
        identifier, _ = self.queued()
        target = self.manager._folder(identifier) / "campaign.json"
        original = target.read_text(encoding="utf-8")
        for bad in ("[]", "null", '{"incomplete":'):
            target.write_text(bad, encoding="utf-8")
            restored = self.new_manager()
            self.assertEqual(restored.list(), [])
            restored.close()
        record = json.loads(original)
        record["steps"][0]["minutes"] = 175
        target.write_text(json.dumps(record), encoding="utf-8")
        restored = self.new_manager()
        self.assertEqual(restored.list(), [])
        restored.close()
        self.assertEqual(self.jobs.calls, [])

    def test_ambiguous_owner_on_restore_keeps_blocking_reservation(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record["steps"][0]["status"] = "launching"
        self.manager._save(record)
        with patch.object(self.jobs, "campaign_job", side_effect=RuntimeError("ambiguous owner")):
            restored = self.new_manager()
            self.assertEqual(restored.active_campaign()["status"], "cleanup_pending")
            self.assertIsNone(restored.detail(identifier)["source_seal"]["valid"])
            self.assertIsNone(restored.detail(identifier)["runtime_seal"]["valid"])
            saved = json.loads((restored._folder(identifier) / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "cleanup_pending")
            self.assertIsNone(saved["runtime_seal"]["valid"])
        self.addCleanup(restored.close)
        self.assertEqual(self.jobs.calls, [])
        self.assertEqual(self.jobs.stops, [])

    def test_source_and_data_drift_exclude_current_result_and_stop_future_jobs(self):
        for drift in ("source", "data"):
            with self.subTest(drift=drift):
                identifier, active = self.queued()
                calls = len(self.jobs.calls)
                if drift == "source":
                    self.jobs.complete_hook = lambda: self.identity.update(source_fingerprint="d" * 64)
                else:
                    task = self.data / "regression/task.md"
                    self.jobs.complete_hook = lambda: task.write_text(task.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
                detail = self.run_queue(identifier, active)
                self.assertEqual(detail["status"], "failed")
                self.assertFalse(detail["steps"][0]["admissible"])
                self.assertEqual(len(self.jobs.calls), calls + 1)
                self.assertEqual(detail["comparisons"], [])
                self.jobs.complete_hook = None

    def test_global_budget_never_extends_child_limit_and_stops_current_owned_job(self):
        identifier, active = self.queued()
        active["deadline"] = -1
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(self.jobs.calls, [])
        identifier, active = self.queued()
        self.jobs.auto_complete = False
        with patch.object(active["stop"], "wait", side_effect=lambda seconds: active.update(deadline=-1)):
            detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(len(self.jobs.stops), 1)
        self.assertEqual(self.jobs.calls[0][2]["minutes"], 5)

    def test_budget_expiry_inside_prelaunch_checks_never_dispatches(self):
        identifier, active = self.queued(self.body(count=1))
        original = self.manager._check_seals
        def expire(record):
            original(record)
            active["deadline"] = -1
        with patch.object(self.manager, "_check_seals", side_effect=expire):
            detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(self.jobs.calls, [])

    def test_budget_expiry_during_final_audit_excludes_last_result(self):
        identifier, active = self.queued(self.body(count=1))
        original = self.manager._check_seals
        calls = []
        def expire_after_job(record):
            original(record)
            calls.append(True)
            if len(calls) == 2:
                active["deadline"] = -1
        with patch.object(self.manager, "_check_seals", side_effect=expire_after_job):
            detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertFalse(detail["steps"][0]["admissible"])
        self.assertEqual(detail["comparisons"], [])
        self.assertEqual(len(self.jobs.calls), 1)

    def test_expired_campaign_cannot_become_paused(self):
        identifier, active = self.queued()
        self.manager.pause(identifier)
        active["deadline"] = -1
        detail = self.run_queue(identifier, active)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(self.jobs.calls, [])

    def test_recovery_reconciles_launch_owner_without_redo_or_foreign_kill(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        step = record["steps"][0]
        step["status"] = "launching"
        self.manager._save(record)
        self.jobs.auto_complete = False
        self.jobs.start(step["task_id"], step["action"], campaign_owner={"campaign_id": identifier, "step_id": step["id"]})
        restored = self.new_manager()
        self.addCleanup(restored.close)
        detail = restored.detail(identifier)
        self.assertEqual(detail["status"], "cleanup_pending")
        self.assertEqual(detail["steps"][0]["job_id"], self.jobs.calls and "1".zfill(32))
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertEqual(self.jobs.stops, [])
        restored.stop(identifier)
        self.assertEqual(len(self.jobs.stops), 1)
        self.assertEqual(restored.detail(identifier)["status"], "stopped")

    def test_restart_with_missing_launch_association_never_retries_that_step(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record["steps"][0]["status"] = "launching"
        self.manager._save(record)
        restored = self.new_manager()
        self.addCleanup(restored.close)
        self.assertEqual(restored.detail(identifier)["status"], "interrupted")
        self.assertEqual(restored.detail(identifier)["steps"][0]["status"], "interrupted")
        restored.resume(identifier)
        restored._run(identifier, restored._active)
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertEqual(self.jobs.calls[0][2]["campaign_owner"]["step_id"], record["steps"][1]["id"])

    def test_conflicts_and_expired_pause_prevent_launch(self):
        self.native.active_queue.return_value = {"id": "external"}
        self.assertFalse(self.manager.preview(self.body())["ready"])
        self.native.active_queue.return_value = None
        self.jobs.active = {"id": "foreign"}
        self.assertFalse(self.manager.preview(self.body())["ready"])
        self.jobs.active = None
        identifier, active = self.queued()
        self.manager.pause(identifier)
        self.run_queue(identifier, active)
        record = self.manager._records[identifier]
        record["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "истёк"):
            self.manager.resume(identifier)
        self.assertEqual(self.jobs.calls, [])


if __name__ == "__main__":
    unittest.main()
