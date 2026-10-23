"""Owned process and campaign evidence lifecycle; all launches/monitors are fake."""
from contextlib import nullcontext
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from app import jobs
from app.jobs import JobManager
from app.profiling import DataCatalog
from app.tests.test_jobs import csv_file


OWNER = {"campaign_id": "a" * 32, "step_id": "b" * 32}


class IdleThread:
    def __init__(self, *args, **kwargs):
        self.started = False
        self.joins = []

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joins.append(timeout)


class Process:
    pid = 99999

    def __init__(self, returncode=None, *, terminate_exits=True, kill_exits=True):
        self.returncode = returncode
        self.terminate_exits, self.kill_exits = terminate_exits, kill_exits
        self.calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_exits:
            self.returncode = -15

    def kill(self):
        self.calls.append("kill")
        if self.kill_exits:
            self.returncode = -9

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned fixture", timeout)
        return self.returncode


class Monitor:
    def __init__(self, process, *, available=True, over_limit=False, clean=True):
        self.process, self.available, self.over_limit, self.clean = process, available, over_limit, clean
        self.samples, self.cleanup = 0, None

    def sample(self):
        self.samples += 1
        if not self.over_limit:
            self.process.returncode = 0  # finish after one observed polling pass
        return self.over_limit

    def summary(self):
        return {"available": self.available, "peak_rss_bytes": (
                    jobs.DEFAULT_MEMORY_LIMIT_BYTES + 1 if self.over_limit else 1024) if self.available else None,
                "cpu_seconds": .25 if self.available else None,
                "memory_limit_bytes": jobs.DEFAULT_MEMORY_LIMIT_BYTES,
                "sample_count": self.samples, "tracked_processes": 1,
                "cleanup_complete": self.cleanup,
                "unavailable_reason": None if self.available else "fixture inventory unavailable"}

    def terminate(self):
        self.cleanup = self.clean if self.available else None
        if self.clean and self.process.returncode is None:
            self.process.returncode = -9
        return self.clean


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.state = self.root / "data", self.root / "state"
        task = self.data / "example"
        task.mkdir(parents=True)
        (task / "task.md").write_text("Основная метрика: RMSE", encoding="utf-8")
        inputs = task / "inputs"
        csv_file(inputs / "data_dictionary.csv", [["feature", "role", "dtype"],
            ["id", "id", "category"], ["x", "feature", "numeric"], ["answer", "target", "numeric"]])
        csv_file(inputs / "train.csv", [["id", "x", "answer"], *[[str(i), i, 2 * i] for i in range(60)]])
        csv_file(inputs / "test.csv", [["id", "x"], ["0005", 5], ["NA", 6]])
        csv_file(inputs / "sample_submission.csv", [["id", "answer"], ["0005", 0], ["NA", 0]])
        self.catalog = DataCatalog(self.data)
        self.manager = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(self.finish)
        self.enterContext(patch("research_lab.process.coordinator_lock", side_effect=lambda _: nullcontext()))
        self.enterContext(patch.object(jobs.psutil, "Process", return_value=Mock(status=lambda: "running")))

    def finish(self):
        # Fixtures never launch real processes. Avoid signalling invented PIDs
        # during unittest teardown, including after an assertion failure.
        self.manager._active = None
        for job in self.manager._jobs.values():
            if job["status"] in jobs.ACTIVE:
                job["status"] = "interrupted"
        self.manager.close()

    def queued(self, *, campaign=False):
        with patch.object(jobs.threading, "Thread", IdleThread):
            summary = self.manager.start("example", "automl" if campaign else "baseline", minutes=1,
                **({"campaign_owner": OWNER} if campaign else {}))
        return self.manager._jobs[summary["id"]], self.manager._active

    def running(self, process, *, campaign=False):
        job, active = self.queued(campaign=campaign)
        active.update(process=process, identity="owned-birth")
        job.update(status="running", _runtime={"pid": process.pid, "identity": "owned-birth"})
        self.manager._save(job)
        return job, active

    def crash_and_restore(self, job):
        self.manager._save(job)
        self.manager._active = None
        self.manager._release_lock()  # simulate a dead HTTP process, no real child exists
        self.manager = JobManager(self.data, self.state, self.catalog)
        return self.manager._jobs[job["id"]]

    def test_owner_is_durable_before_thread_start_and_duplicate_never_relaunches(self):
        checked = []

        def start(thread):
            saved = list((self.state / "jobs").glob("*/job.json"))
            checked.append(json.loads(saved[0].read_text(encoding="utf-8"))["_campaign_owner"])

        with patch.object(jobs.threading, "Thread", IdleThread), patch.object(IdleThread, "start", start):
            first = self.manager.start("example", "automl", minutes=1, campaign_owner=OWNER)
            second = self.manager.start("example", "automl", minutes=1, campaign_owner=OWNER)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(checked, [OWNER])
        self.assertNotIn("_campaign_owner", first)
        association = self.manager.campaign_job(**OWNER)
        self.assertEqual(association["job"]["id"], first["id"])
        self.assertIsNone(association["cleanup_confirmed"])
        self.assertIsNone(association["resources_ok"])

    def test_invalid_owner_and_ambiguous_existing_association_are_rejected(self):
        for owner in ({}, {**OWNER, "extra": "x"}, {**OWNER, "step_id": "../outside"}, True):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                self.manager.start("example", "automl", campaign_owner=owner)
        job, _ = self.queued(campaign=True)
        self.manager._jobs["c" * 32] = {**job, "id": "c" * 32}
        with self.assertRaisesRegex(RuntimeError, "несколько"):
            self.manager.campaign_job(**OWNER)

    def test_campaign_allows_fixed_local_recipes_but_never_agent_or_prepare(self):
        for index, action in enumerate(sorted(jobs.OFFLINE_ACTIONS)):
            owner = {**OWNER, "step_id": f"{index:032x}"}
            with self.subTest(action=action), patch.object(jobs.threading, "Thread", IdleThread):
                result = self.manager.start("example", action, campaign_owner=owner)
            job = self.manager._jobs[result["id"]]
            self.manager._settle(job, self.manager._active, "cancelled", "fixture, no launch")
        with patch.object(jobs, "_settings_values") as settings:
            for action in ("agent", "prepare", []):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    self.manager.start("example", action, campaign_owner=OWNER)
        settings.assert_not_called()

    def test_history_counts_are_not_truncated_to_the_public_list_limit(self):
        job, active = self.queued()
        self.manager._settle(job, active, "failed", "fixture")
        for index in range(220):
            self.manager._jobs[f"{index:032x}"] = {**job, "id": f"{index:032x}", "status": "completed"}
        self.assertEqual(self.manager.history_counts(), {"total": 221, "completed": 220, "failed": 1})

    def test_thread_start_failure_keeps_owner_and_terminal_failure(self):
        with patch.object(jobs.threading, "Thread", IdleThread), \
                patch.object(IdleThread, "start", side_effect=RuntimeError("thread unavailable")), \
                self.assertRaisesRegex(RuntimeError, "поток"):
            self.manager.start("example", "automl", campaign_owner=OWNER)
        result = self.manager.campaign_job(**OWNER)
        self.assertEqual(result["job"]["status"], "failed")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertIsNone(result["resources_ok"])
        self.assertIsNone(self.manager.active_job())
        self.manager.close()
        self.manager = JobManager(self.data, self.state, self.catalog)
        self.assertEqual(self.manager.campaign_job(**OWNER)["job"]["id"], result["job"]["id"])

    def test_windows_uses_handle_and_waits_after_kill_before_dropping_owner(self):
        process = Process(terminate_exits=False)
        job, active = self.running(process)
        with patch.object(jobs, "WINDOWS", True), patch.object(jobs, "_kill_tree") as kill:
            self.assertTrue(self.manager._settle(job, active, "cancelled", "requested"))
        kill.assert_not_called()
        self.assertEqual(process.calls, ["terminate", ("wait", 2), "kill", ("wait", 2)])
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("_runtime", job)
        self.assertIsNone(self.manager.active_job())

    def test_noop_handle_kill_keeps_pending_owner_lock_and_stop_can_retry_after_close(self):
        process = Process(terminate_exits=False, kill_exits=False)
        job, active = self.running(process)
        with patch.object(jobs, "WINDOWS", True):
            self.assertFalse(self.manager._settle(job, active, "cancelled", "requested"))
            self.assertEqual(self.manager.active_job()["status"], "cleanup_pending")
            self.assertEqual(job["_runtime"]["identity"], "owned-birth")
            self.assertIsNone(job["finished_at"])
            with self.assertRaises(RuntimeError):
                self.manager.start("example", "baseline")
            self.manager.close()
            with self.assertRaises(ValueError):
                JobManager(self.data, self.state, self.catalog)
            process.kill_exits = True
            stopped = self.manager.stop("example", job["id"])
        self.assertEqual(stopped["status"], "cancelled")
        self.assertIsNone(self.manager.active_job())
        self.assertIsNone(self.manager._lock_stream)

    def test_recovered_reused_pid_is_never_killed(self):
        job, _ = self.running(Process())
        with patch.object(jobs, "_identity", return_value="different-birth"), \
                patch.object(jobs, "_kill_tree") as kill:
            restored = self.crash_and_restore(job)
        kill.assert_not_called()
        self.assertEqual(restored["status"], "interrupted")
        self.assertNotIn("_runtime", restored)

    def test_recovered_noop_kill_retains_pid_and_retry_rechecks_birth(self):
        job, _ = self.running(Process(), campaign=True)
        with patch.object(jobs, "_identity", return_value="owned-birth"), \
                patch.object(jobs, "_kill_tree") as kill, patch.object(jobs.time, "sleep"):
            restored = self.crash_and_restore(job)
        kill.assert_called_once_with(99999)
        self.assertEqual(restored["status"], "cleanup_pending")
        self.assertFalse(self.manager.campaign_job(**OWNER)["cleanup_confirmed"])
        with patch.object(jobs, "_identity", return_value="replacement-birth"), \
                patch.object(jobs, "_kill_tree") as kill:
            stopped = self.manager.stop("example", job["id"])
        kill.assert_not_called()
        self.assertEqual(stopped["status"], "interrupted")
        self.assertIsNone(self.manager.campaign_job(**OWNER)["resources_ok"])

    def test_unknown_recovered_identity_blocks_without_signalling_unowned_pid(self):
        job, _ = self.running(Process())
        with patch.object(jobs, "_identity", return_value=None), \
                patch.object(jobs, "process_inventory_matches", return_value=True), \
                patch.object(jobs.psutil, "pid_exists", return_value=True), \
                patch.object(jobs, "_kill_tree") as kill:
            restored = self.crash_and_restore(job)
            self.assertEqual(self.manager.stop("example", job["id"])["status"], "cleanup_pending")
        kill.assert_not_called()
        self.assertEqual(restored["_runtime"]["pid"], 99999)

    def test_recovered_zombie_is_not_treated_as_still_executing(self):
        job, _ = self.running(Process())
        with patch.object(jobs, "_identity", return_value="owned-birth"), \
                patch.object(jobs.psutil, "Process", return_value=Mock(status=lambda: jobs.psutil.STATUS_ZOMBIE)), \
                patch.object(jobs, "_kill_tree") as kill:
            restored = self.crash_and_restore(job)
        kill.assert_not_called()
        self.assertEqual(restored["status"], "interrupted")

    def run_campaign(self, *, available=True, over_limit=False, clean=True):
        job, active = self.queued(campaign=True)
        process = Process()
        monitor = Monitor(process, available=available, over_limit=over_limit, clean=clean)
        with patch.object(jobs.subprocess, "Popen", return_value=process), \
                patch.object(jobs, "_identity", return_value="owned-birth"), \
                patch.object(jobs, "_descendants", return_value=[]), \
                patch.object(jobs, "ProcessTreeMonitor", return_value=monitor) as construct, \
                patch.object(jobs, "POLL_SECONDS", 0), patch.object(self.manager, "_collect"):
            self.manager._run(job["id"], active, 1, {})
        construct.assert_called_once_with(process, max_memory_bytes=28 * 1024**3)
        return job, active, monitor

    def test_campaign_monitor_records_observed_resources_and_confirmed_cleanup(self):
        job, _, monitor = self.run_campaign()
        result = self.manager.campaign_job(**OWNER)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(monitor.samples, 1)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertTrue(result["resources_ok"])
        self.assertEqual(result["job"]["resources"]["peak_rss_bytes"], 1024)
        self.assertEqual(result["job"]["resources"]["scope"], "sampled_owned_process_tree")
        self.assertNotIn("_execution", result["job"])

    def test_unavailable_monitor_can_confirm_root_exit_but_never_resource_success(self):
        job, _, _ = self.run_campaign(available=False)
        result = self.manager.campaign_job(**OWNER)
        self.assertEqual(job["status"], "completed")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertIsNone(result["resources_ok"])
        self.assertIsNone(result["job"]["resources"]["peak_rss_bytes"])
        self.assertIsNone(result["job"]["resources"]["cleanup_complete"])

    def test_memory_limit_stops_campaign_job_and_cannot_be_admitted(self):
        job, _, monitor = self.run_campaign(over_limit=True)
        result = self.manager.campaign_job(**OWNER)
        self.assertEqual(job["status"], "failed")
        self.assertIn("памяти", job["error"])
        self.assertTrue(result["cleanup_confirmed"])
        self.assertFalse(result["resources_ok"])
        self.assertTrue(job["_execution"]["memory_limit_exceeded"])
        self.assertTrue(monitor.cleanup)

    def test_monitor_cleanup_failure_blocks_next_job_until_successful_retry(self):
        job, _, monitor = self.run_campaign(clean=False)
        self.assertEqual(job["status"], "cleanup_pending")
        self.assertFalse(self.manager.campaign_job(**OWNER)["cleanup_confirmed"])
        with self.assertRaises(RuntimeError):
            self.manager.start("example", "baseline")
        monitor.clean = True
        self.assertEqual(self.manager.stop("example", job["id"])["status"], "completed")
        self.assertTrue(self.manager.campaign_job(**OWNER)["cleanup_confirmed"])

    def test_birth_probe_error_after_spawn_still_retains_handle_for_cleanup(self):
        job, active = self.queued()
        process = Process()
        active["cancel"].wait = lambda _: active["cancel"].set()
        with patch.object(jobs, "WINDOWS", True), \
                patch.object(jobs.subprocess, "Popen", return_value=process), \
                patch.object(jobs, "_identity", side_effect=OSError("birth denied")), \
                patch.object(jobs, "_descendants", return_value=[]), \
                patch.object(jobs, "_kill_tree") as kill:
            self.manager._run(job["id"], active, 1, {})
        kill.assert_not_called()
        self.assertIn("terminate", process.calls)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(self.manager.active_job())

    def test_terminal_save_failure_keeps_handle_and_association_for_retry(self):
        job, active = self.running(Process(returncode=0), campaign=True)
        with patch.object(self.manager, "_save", side_effect=OSError("disk temporarily unavailable")):
            self.assertFalse(self.manager._settle(job, active, "cancelled", "requested"))
        self.assertIs(self.manager._active, active)
        self.assertEqual(job["status"], "cleanup_pending")
        self.assertEqual(self.manager.stop("example", job["id"])["status"], "cancelled")
        self.assertEqual(self.manager.campaign_job(**OWNER)["job"]["id"], job["id"])

    def test_error_in_final_report_collection_does_not_bypass_cleanup(self):
        job, active = self.queued()
        job["action"] = "agent"
        process = Process(returncode=0)
        with patch.object(jobs.subprocess, "Popen", return_value=process), \
                patch.object(jobs, "_identity", return_value="owned-birth"), \
                patch.object(self.manager, "_collect"), \
                patch.object(self.manager, "_collect_agent", side_effect=ValueError("bad final report")):
            self.manager._run(job["id"], active, 1, {})
        self.assertEqual(job["status"], "failed")
        self.assertIn("bad final report", job["error"])
        self.assertTrue(job["_execution"]["cleanup_confirmed"])
        self.assertIsNone(self.manager.active_job())


if __name__ == "__main__":
    unittest.main()
