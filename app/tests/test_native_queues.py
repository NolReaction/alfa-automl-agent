"""Native app lifecycle fixtures: no model is trained or native solver invoked."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from app import native_queues as native
from app.profiling import DataCatalog
from app.tests.test_profiling import make_task
from tests.test_analyze_data import regression_task
from research_lab.process import coordinator_lock
from research_lab import process as process_adapter


class DormantThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args
        self.join_timeouts = []
    def start(self):
        pass
    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


class FakeProcess:
    pid = 12345
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.terminated = False
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        if self.returncode is None:
            raise native.subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode
    def terminate(self):
        self.terminated = True
        self.returncode = -15
    def kill(self):
        self.terminate()


class CancelOnWait:
    def __init__(self):
        self.cancelled = False
    def is_set(self):
        return self.cancelled
    def set(self):
        self.cancelled = True
    def wait(self, timeout=None):
        self.cancelled = True
        return True


class NativeQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo, self.data = self.root / "repo", self.root / "data"
        self.repo.mkdir()
        (self.repo / "app").mkdir()
        for name in ("experiment_round5_overnight.py", "experiment_round5_overnight_worker.py",
                     "build_submission.py", "app/supervisor.py"):
            (self.repo / name).write_text("# fixture; never executed\n", encoding="utf-8")
        pins = (native.ROOT / "requirements.txt").read_text(encoding="utf-8")
        (self.repo / "requirements.txt").write_text(pins, encoding="utf-8")
        self.pins = dict(line.split("==", 1) for line in pins.splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
        classification = make_task(self.data)
        (classification / "task.md").write_text("Primary metric: Macro F1\n", encoding="utf-8")
        regression = regression_task(self.data)
        (regression / "task.md").write_text("Основная метрика: RMSE\n", encoding="utf-8")
        self.catalog = DataCatalog(self.data)
        self.ordinary = Mock()
        self.ordinary.active_job.return_value = None
        self.enterContext(patch.object(native, "WINDOWS", True))
        self.enterContext(patch.object(native.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True))
        self.enterContext(patch.object(native, "source_identity", return_value={"sha": "a" * 40, "dirty": False}))
        self.enterContext(patch.object(native.metadata, "version", side_effect=lambda name: self.pins[name]))
        self.enterContext(patch.object(native.psutil, "virtual_memory", return_value=SimpleNamespace(total=32 * 1024**3)))
        self.enterContext(patch.object(native.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)))
        self.enterContext(patch.object(native, "machine_lock_path", return_value=self.root / "fixture-cpu.lock"))
        self.enterContext(patch.object(native, "_identity", return_value="fixture-birth"))
        self.enterContext(patch.object(native.threading, "Thread", DormantThread))
        self.manager = self.new_manager()
        self.addCleanup(self.manager.close)

    def new_manager(self):
        return native.NativeQueueManager(self.data, self.root / "state", self.catalog,
            source_root=self.repo, runs_root=self.root / "external-runs", ordinary_jobs=self.ordinary)

    def queued(self, budget=600):
        detail = self.manager.start("regression", "classification", budget)
        return detail["id"], self.manager._active

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def successful_output(self, identifier, *, allow=True):
        output = self.manager._output(identifier)
        self.write(output / "queue.json", {"status": "completed", "jobs": [
            {"profile": profile, "status": "completed"} for profile in native.PROFILES]})
        for profile in native.PROFILES:
            self.write(output / "jobs" / profile / "audit.json", {
                "qualification": "requires_review", "continuation_allowed": allow,
                "checks": {"root_exit_zero": True, "owned_cleanup_complete": True},
                "native_elapsed_seconds": 57.5,
                "proof": {"outer_metric": {"name": "RMSE", "value": 35.0}}})

    def test_preflight_reports_runtime_pins_source_hardware_and_global_lane(self):
        report = self.manager.preflight()
        self.assertTrue(report["ready"], report)
        self.assertEqual((report["release_sha"], report["threads"], report["total_native_seconds"]),
                         ("a" * 40, 8, 31500))
        lane = self.root / "global.lock"
        with patch.object(native, "machine_lock_path", return_value=lane):
            with coordinator_lock(lane):
                blocked = self.manager.preflight()
                self.assertFalse(blocked["lane_available"])
                self.assertFalse(blocked["ready"])
                with self.assertRaisesRegex(ValueError, "CPU"):
                    self.manager.start("regression", "classification")
        self.assertFalse(self.manager.list())
        with patch.object(native, "WINDOWS", False):
            self.assertFalse(self.manager.preflight()["supported"])
        with patch.object(native, "source_identity", return_value={"sha": "a" * 40, "dirty": True}):
            self.assertFalse(self.manager.preflight()["ready"])
        with patch.object(native.metadata, "version", side_effect=metadata.PackageNotFoundError("catboost")):
            self.assertFalse(self.manager.preflight()["ready"])
        with patch.object(native.psutil, "virtual_memory", return_value=SimpleNamespace(total=8 * 1024**3)):
            self.assertFalse(self.manager.preflight()["ready"])

    def test_start_validates_task_roles_budget_and_ordinary_exclusion_before_spawn(self):
        for budget in (True, 300, 10501, 600.0):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.manager.start("regression", "classification", budget)
        with self.assertRaisesRegex(ValueError, "RMSE"):
            self.manager.start("classification", "regression", 600)
        self.ordinary.active_job.return_value = {"id": "ordinary"}
        with self.assertRaisesRegex(ValueError, "обычного"):
            self.manager.start("regression", "classification", 600)
        self.assertFalse(self.manager.list())

    def test_thread_start_failure_is_durable_and_does_not_leave_lane_active(self):
        with patch.object(DormantThread, "start", side_effect=RuntimeError("thread unavailable")), \
                self.assertRaisesRegex(ValueError, "поток"):
            self.manager.start("regression", "classification", 600)
        self.assertIsNone(self.manager.active_queue())
        history = self.manager.list()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "failed")
        restored = self.new_manager()
        self.addCleanup(restored.close)
        self.assertEqual(restored.list()[0]["error"], "thread unavailable")
        self.assertTrue(self.manager.preflight()["ready"])

    def test_native_cli_uses_pinned_coordinator_containment_and_does_not_precreate_output(self):
        identifier, active = self.queued()
        output = self.manager._output(identifier)
        self.assertFalse(output.exists())
        self.assertEqual(self.manager.active_queue()["total_native_seconds"], 1800)
        def launch(argv, **kwargs):
            self.assertIn(str(self.repo / "app/supervisor.py"), argv)
            self.assertIn(str(self.repo / "experiment_round5_overnight.py"), argv)
            self.assertNotIn("--cpu-lane", argv)  # the coordinator owns its lane
            self.assertNotIn(str(self.repo / "app/worker.py"), argv)
            self.assertEqual(argv[argv.index("--release-commit") + 1], "a" * 40)
            self.assertEqual(argv[argv.index("--time-budget") + 1], "600")
            self.assertEqual(argv[argv.index("--output") + 1], str(output))
            self.assertEqual(kwargs["creationflags"], native.subprocess.CREATE_NEW_PROCESS_GROUP)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["env"]["USERNAME"], "fixture-owner")
            # Windows has no getuid: the exact user name, not a generic fallback,
            # must scope both app preflight and the sanitized child coordinator.
            with patch.object(process_adapter, "os", SimpleNamespace(environ=os.environ)):
                parent_lane = process_adapter.machine_lock_path()
            with patch.object(process_adapter, "os", SimpleNamespace(environ=kwargs["env"])):
                child_lane = process_adapter.machine_lock_path()
            self.assertEqual(parent_lane, child_lane)
            self.assertFalse(output.exists())
            self.successful_output(identifier)
            return FakeProcess()
        with patch.dict(os.environ, {"USERNAME": "fixture-owner", "OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(native.subprocess, "Popen", side_effect=launch) as popen:
            self.manager._run(identifier, active)
        popen.assert_called_once()
        detail = self.manager.detail(identifier)
        self.assertEqual(detail["status"], "completed")
        self.assertTrue(all(p["qualification"] == "requires_review" and p["continuation_allowed"]
                            for p in detail["profiles"]))
        self.assertIsNone(self.manager.active_queue())

    def test_exit_zero_without_technical_audits_is_not_completed(self):
        identifier, active = self.queued()
        self.successful_output(identifier, allow=False)
        with patch.object(native.subprocess, "Popen", return_value=FakeProcess()):
            self.manager._run(identifier, active)
        self.assertEqual(self.manager.detail(identifier)["status"], "failed")
        self.assertIn("аудитов", self.manager.detail(identifier)["error"])

    def test_overall_watchdog_covers_all_three_caps_and_never_extends_child_budget(self):
        identifier, active = self.queued(600)
        process = FakeProcess(None)
        # 1000s is past one arm's cap, but still below 3*(600+30)+1800.
        times = iter((0., 1000., 4000.))
        with patch.object(native.subprocess, "Popen", return_value=process), \
                patch.object(native.time, "monotonic", side_effect=lambda: next(times)), \
                patch.object(native, "_kill_tree", side_effect=lambda pid: setattr(process, "returncode", -15)) as kill:
            self.manager._run(identifier, active)
        kill.assert_not_called()
        self.assertTrue(process.terminated)  # retained Windows HANDLE closes the Job Object
        self.assertIn("watchdog", self.manager.detail(identifier)["error"])
        self.assertEqual(self.manager.detail(identifier)["time_budget_seconds"], 600)

    def test_cancel_stops_owned_tree_and_never_launches_a_second_process(self):
        identifier, active = self.queued()
        active["cancel"] = CancelOnWait()
        process = FakeProcess(None)
        with patch.object(native.subprocess, "Popen", return_value=process) as popen, \
                patch.object(native, "_kill_tree", side_effect=lambda pid: setattr(process, "returncode", -15)) as kill:
            self.manager._run(identifier, active)
        self.assertEqual(self.manager.detail(identifier)["status"], "cancelled")
        self.assertEqual([p["status"] for p in self.manager.detail(identifier)["profiles"]], ["pending"] * 3)
        popen.assert_called_once()
        kill.assert_not_called()
        self.assertTrue(process.terminated)

    def test_reused_pid_is_not_tree_killed_and_windows_owned_handle_is_used(self):
        identifier, active = self.queued()
        active["cancel"] = CancelOnWait()
        process = FakeProcess(None)
        with patch.object(native.subprocess, "Popen", return_value=process), \
                patch.object(native, "_identity", side_effect=("original-birth", "reused-birth")), \
                patch.object(native, "_kill_tree") as kill:
            self.manager._run(identifier, active)
        kill.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertEqual(self.manager.detail(identifier)["status"], "cancelled")

    def test_birth_probe_failure_after_spawn_still_retains_handle_for_stop(self):
        identifier, active = self.queued()
        active["cancel"] = CancelOnWait()
        process = FakeProcess(None)
        with patch.object(native.subprocess, "Popen", return_value=process), \
                patch.object(native, "_identity", side_effect=OSError("birth access denied")), \
                patch.object(native, "_kill_tree") as kill:
            self.manager._run(identifier, active)
        kill.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertEqual(self.manager.detail(identifier)["status"], "cancelled")

    def test_recovery_interrupts_own_history_without_resume_or_pid_reuse_kill(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record.update(status="running", _runtime={"pid": 99999, "identity": "old-birth"},
                      output_dir=str(self.root / "must-not-use"))
        self.manager._save(record)
        with patch.object(native, "_identity", return_value="different-birth"), \
                patch.object(native, "_kill_tree") as kill, \
                patch.object(native.subprocess, "Popen") as popen:
            restored = self.new_manager()
        self.addCleanup(restored.close)
        kill.assert_not_called()
        popen.assert_not_called()
        self.assertIsNone(restored.active_queue())
        detail = restored.detail(identifier)
        self.assertEqual(detail["status"], "interrupted")
        self.assertEqual(detail["output_dir"], str(self.root / "external-runs" / identifier))
        self.assertNotIn("must-not-use", str(detail))
        self.assertFalse((self.root / "must-not-use").exists())

    def test_recovery_may_stop_only_a_matching_previously_owned_birth_marker(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record.update(status="running", _runtime={"pid": 99999, "identity": "fixture-birth"})
        self.manager._save(record)
        with patch.object(native, "_kill_tree") as kill, \
                patch.object(native, "_identity", side_effect=("fixture-birth", "different-birth")):
            restored = self.new_manager()
        self.addCleanup(restored.close)
        kill.assert_called_once_with(99999)
        self.assertEqual(restored.detail(identifier)["status"], "interrupted")

    def test_recovery_retains_ownership_after_noop_kill_and_stop_rechecks_birth(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record.update(status="running", _runtime={"pid": 99999, "identity": "fixture-birth"})
        self.manager._save(record)
        with patch.object(native, "_kill_tree") as kill, patch.object(native.time, "sleep"):
            restored = self.new_manager()
        self.addCleanup(restored.close)
        kill.assert_called_once_with(99999)
        self.assertEqual(restored.active_queue()["status"], "cleanup_pending")
        self.assertIsNone(restored.detail(identifier)["finished_at"])
        self.assertFalse(restored.preflight()["ready"])
        persisted = json.loads((restored._folder(identifier) / "registry.json").read_text())
        self.assertEqual((persisted["_runtime"]["pid"], persisted["_runtime"]["identity"]),
                         (99999, "fixture-birth"))
        with patch.object(native, "_identity", return_value="a-new-process-birth"), \
                patch.object(native, "_kill_tree") as reused_kill:
            result = restored.stop(identifier)
        reused_kill.assert_not_called()
        self.assertEqual(result["status"], "interrupted")
        self.assertIsNone(restored.active_queue())
        self.assertNotIn("_runtime", json.loads((restored._folder(identifier) / "registry.json").read_text()))

    def test_unknown_recovered_identity_stays_retryable_without_killing_arbitrary_pid(self):
        identifier, _ = self.queued()
        record = self.manager._records[identifier]
        record.update(status="running", _runtime={"pid": 99999, "identity": "fixture-birth"})
        self.manager._save(record)
        with patch.object(native, "_identity", return_value=None), \
                patch.object(native.psutil, "pid_exists", return_value=True), \
                patch.object(native, "_kill_tree") as kill:
            restored = self.new_manager()
            self.assertEqual(restored.stop(identifier)["status"], "cleanup_pending")
        self.addCleanup(restored.close)
        kill.assert_not_called()
        self.assertEqual(restored._records[identifier]["_runtime"]["identity"], "fixture-birth")
        with patch.object(native, "_identity", return_value=None), \
                patch.object(native.psutil, "pid_exists", return_value=False):
            self.assertEqual(restored.stop(identifier)["status"], "interrupted")

    def test_failed_live_handle_stop_keeps_durable_retry_and_close_waits_for_cleanup(self):
        identifier, active = self.queued()
        active["cancel"] = CancelOnWait()
        process = FakeProcess(None)
        with patch.object(native.subprocess, "Popen", return_value=process) as launch, \
                patch.object(process, "terminate", side_effect=OSError("HANDLE termination denied")):
            self.manager._run(identifier, active)
            self.assertEqual(self.manager.active_queue()["status"], "cleanup_pending")
            self.assertEqual(self.manager._records[identifier]["_runtime"]["identity"], "fixture-birth")
            self.manager.close()
            self.assertEqual(self.manager.detail(identifier)["status"], "cleanup_pending")
            self.assertIn(10, active["thread"].join_timeouts)
            self.assertIn("_runtime", json.loads((self.manager._folder(identifier) / "registry.json").read_text()))
        self.assertEqual(self.manager.stop(identifier)["status"], "cancelled")
        self.assertIsNone(self.manager.active_queue())
        self.assertTrue(process.terminated)
        launch.assert_called_once()

    def test_progress_is_bounded_safe_and_partial_or_shared_json_stays_unknown(self):
        identifier, _ = self.queued()
        output = self.manager._output(identifier)
        self.write(output / "queue.json", {"status": "running", "jobs": [
            {"profile": "regression_off", "status": "running"}]})
        job = output / "jobs/regression_off"
        self.write(job / "worker_evidence.json", {"worker_elapsed_seconds": 12,
            "run_report": {"path": "../../outside.json"}})
        (self.root / "outside.json").write_text('{"stage":"SECRET"}')
        self.assertNotIn("SECRET", str(self.manager.detail(identifier)))
        (job / "worker_evidence.json").write_text('{"native_started_monotonic":')
        detail = self.manager.detail(identifier)
        self.assertIsNone(detail["profiles"][0]["resources"])
        self.assertIsNone(detail["profiles"][0]["metric"])
        self.write(output / "queue.json", {"jobs": None})
        self.assertEqual(len(self.manager.detail(identifier)["profiles"]), 3)
        target = output / "queue.json"
        original = Path.open
        remaining = [2]
        def sharing(path, *args, **kwargs):
            if path == target and remaining[0]:
                remaining[0] -= 1
                raise PermissionError("Windows sharing lock")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", sharing), patch.object(native.time, "sleep"):
            self.assertIsNotNone(native._json_snapshot(target, output))
        self.assertEqual(remaining[0], 0)

    def test_current_native_stage_is_live_but_metric_needs_audit_and_comparison_preserves_unknown(self):
        identifier, _ = self.queued()
        output = self.manager._output(identifier)
        self.write(output / "queue.json", {"status": "running", "jobs": [
            {"profile": "regression_off", "status": "running"}]})
        job = output / "jobs/regression_off"
        run = ".agent_runs/" + "b" * 32
        self.write(job / "worker_evidence.json", {"observations": [
            {"phase": "prepare_research", "run_dir": run}], "native_started_monotonic": 100})
        self.write(job / "task" / run / "run.json", {"stage": "inner_research",
            "metric": {"name": "RMSE", "value": 0.001}, "group_hash": 2**64 - 1})
        self.write(output / "comparison.json", {"status": "incomplete"})
        with patch.object(native.time, "monotonic", return_value=142):
            detail = self.manager.detail(identifier)
        profile = detail["profiles"][0]
        self.assertEqual((profile["stage"], profile["elapsed_seconds"]), ("inner_research", 42))
        self.assertIsNone(profile["metric"])
        self.assertIsNone(profile["resources"])
        self.assertIsNone(detail["comparison"]["rescue_exercised"])
        self.write(job / "audit.json", {"proof": {"outer_metric": {"name": "RMSE", "value": 35}},
            "qualification": "requires_review", "continuation_allowed": True})
        self.write(output / "comparison.json", {"status": "incomplete", "rescue_exercised": False})
        detail = self.manager.detail(identifier)
        self.assertEqual(detail["profiles"][0]["metric"], {"name": "RMSE", "value": 35})
        self.assertFalse(detail["comparison"]["rescue_exercised"])

    def test_malformed_json_field_types_never_crash_progress_or_recovery(self):
        identifier, _ = self.queued()
        output = self.manager._output(identifier)
        self.write(output / "queue.json", {"jobs": [
            {"profile": "regression_off", "status": []}, {"profile": {}, "status": "running"}]})
        self.write(output / "comparison.json", {"status": {}})
        detail = self.manager.detail(identifier)
        self.assertEqual(detail["profiles"][0]["status"], "pending")
        self.assertEqual(detail["comparison"]["status"], "incomplete")
        with self.assertRaises(ValueError):
            self.manager.logs(identifier, kind=[])
        record = dict(self.manager._records[identifier], status=[])
        self.manager._save(record)
        restored = self.new_manager()
        self.addCleanup(restored.close)
        self.assertEqual(restored.list(), [])

    def test_logs_use_fixed_names_bounded_redacted_tail_and_child_finalization(self):
        identifier, _ = self.queued()
        coordinator = self.manager._folder(identifier) / "coordinator.txt"
        coordinator.write_text("x" * (native.LOG_TAIL_BYTES * 2) + " secret-fixture-key", encoding="utf-8")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-fixture-key"}):
            result = self.manager.logs(identifier)
        self.assertLessEqual(len(result["text"].encode()), native.LOG_TAIL_BYTES)
        self.assertNotIn("secret-fixture-key", result["text"])
        self.assertFalse(result["finalized"])
        with self.assertRaises(ValueError):
            self.manager.logs(identifier, "stdout", "../../escape")
        with self.assertRaises(ValueError):
            self.manager.detail("../escape")
        self.successful_output(identifier)
        path = self.manager._output(identifier) / "jobs/classification/execution/stdout.txt"
        path.parent.mkdir(parents=True)
        path.write_text("final child tail")
        result = self.manager.logs(identifier, "stdout", "classification")
        self.assertEqual(result["text"], "final child tail")
        self.assertTrue(result["finalized"])

    def test_native_evidence_is_its_own_zip_format_with_manifest_and_path_checks(self):
        identifier, _ = self.queued()
        output = self.manager._output(identifier)
        output.mkdir()
        archive = output / "evidence.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("bundle_manifest.json", '{"sha256":{}}')
            for index in range(40):  # not the existing reports.json/32-entry importer
                bundle.writestr(f"jobs/audit-{index}.json", "{}")
        self.assertTrue(self.manager.detail(identifier)["artifact_available"])
        self.assertEqual(self.manager.artifact(identifier), archive)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("bundle_manifest.json", "{}")
            bundle.writestr("../escape", "bad")
        with self.assertRaisesRegex(ValueError, "Небезопасная"):
            self.manager.artifact(identifier)
        archive.write_bytes(b"not a zip")
        self.assertFalse(self.manager.detail(identifier)["artifact_available"])
        with self.assertRaises(ValueError):
            self.manager.artifact(identifier)

    def test_outputs_inside_source_or_data_and_linked_output_paths_are_refused(self):
        for output in (self.repo / "app/.workspace/night", self.data / "night"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                native.NativeQueueManager(self.data, self.root / "other-state", self.catalog,
                                          source_root=self.repo, runs_root=output)
        if hasattr(os, "symlink"):
            link = self.root / "linked-runs"
            try:
                link.symlink_to(self.root / "external-runs", target_is_directory=True)
            except OSError:
                return
            with self.assertRaises(ValueError):
                native.NativeQueueManager(self.data, self.root / "other-state", self.catalog,
                                          source_root=self.repo, runs_root=link)


if __name__ == "__main__":
    unittest.main()
