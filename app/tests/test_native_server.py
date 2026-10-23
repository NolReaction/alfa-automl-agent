"""Native queue HTTP and shared process-lane boundaries without training."""
from contextlib import closing
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from app import jobs, supervisor
from app.server import WorkspaceServer
from research_lab import process as lane
from tests.test_task_spec import make_task


class FakeNativeManager:
    def __init__(self, data_root, state_root, catalog, **kwargs):
        self.ordinary = kwargs["ordinary_jobs"]
        self.root = Path(state_root)
        self.records, self.active, self.calls = {}, None, []

    def preflight(self):
        return {"ready": True, "supported": True, "errors": [], "release_sha": "b" * 40,
                "threads": 8, "total_native_seconds": 31500, "lane_available": True}

    def list(self):
        return list(self.records.values())

    def active_queue(self):
        return self.active

    def start(self, regression_task_id, classification_task_id, time_budget_seconds=10500):
        if self.ordinary.active_job():
            raise RuntimeError("Ordinary job is active")
        if self.active:
            raise RuntimeError("Native queue is active")
        self.calls.append((regression_task_id, classification_task_id, time_budget_seconds))
        self.active = {"id": "a" * 32, "status": "running", "release_sha": "b" * 40,
                       "time_budget_seconds": time_budget_seconds, "profiles": []}
        self.records[self.active["id"]] = self.active
        return self.active

    def detail(self, identifier):
        return self.records[identifier]

    def stop(self, identifier):
        record = self.records[identifier]
        record["status"] = "cancelled"
        self.active = None
        return record

    def logs(self, identifier, kind="coordinator", profile=None):
        self.detail(identifier)
        return {"text": "bounded fixture", "kind": kind, "profile": profile, "finalized": False}

    def artifact(self, identifier):
        self.detail(identifier)
        path = self.root / "evidence.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("bundle_manifest.json", "{}")
        return path

    def close(self):
        pass


class NativeHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        make_task(data / "regression", metric="RMSE")
        make_task(data / "classification", metric="Macro F1")
        self.enterContext(patch("app.native_queues.NativeQueueManager", FakeNativeManager))
        self.server = WorkspaceServer(data, 0, self.root / "state")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, *, value=None, headers=None, method="GET"):
        request_headers = dict(headers or {})
        body = None
        if value is not None:
            method = "POST"
            body = json.dumps(value).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
            request_headers.setdefault("X-Alfa-Token", self.server.token)
        with closing(http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)) as client:
            client.request(method, path, body, request_headers)
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def start(self):
        status, _, raw = self.request("/api/native-queues", value={
            "regression_task_id": "regression", "classification_task_id": "classification",
            "time_budget_seconds": 600})
        self.assertEqual(status, 202, raw)
        return json.loads(raw)["queue"]["id"]

    def test_start_requires_local_origin_token_and_known_catalog_ids(self):
        payload = {"regression_task_id": "regression", "classification_task_id": "classification"}
        for headers in ({"X-Alfa-Token": "wrong"}, {"Origin": "https://foreign.invalid"}):
            status, _, _ = self.request("/api/native-queues", value=payload, headers=headers)
            self.assertEqual(status, 403)
        for additional in ({"command": "arbitrary"}, {"release_sha": "a" * 40}, {"runs_root": "/tmp"}):
            self.assertEqual(self.request("/api/native-queues", value={**payload, **additional})[0], 400)
        self.assertEqual(self.request("/api/native-queues", value={**payload, "regression_task_id": "../secret"})[0], 400)
        self.assertEqual(self.server.native_queues.calls, [])

    def test_lifecycle_progress_logs_stop_and_separate_zip_download(self):
        self.assertTrue(json.loads(self.request("/api/native-queues/preflight")[2])["ready"])
        identifier = self.start()
        self.assertEqual(self.server.native_queues.calls, [("regression", "classification", 600)])
        session = json.loads(self.request("/api/session")[2])
        self.assertEqual(session["active_queue"]["id"], identifier)
        listing = json.loads(self.request("/api/native-queues")[2])
        self.assertEqual(len(listing["queues"]), 1)
        details = json.loads(self.request(f"/api/native-queues/{identifier}")[2])
        self.assertEqual(details["queue"]["status"], "running")
        log = json.loads(self.request(f"/api/native-queues/{identifier}/logs?kind=stdout&profile=classification")[2])
        self.assertEqual((log["kind"], log["profile"], log["finalized"]), ("stdout", "classification", False))
        status, headers, raw = self.request(f"/api/native-queues/{identifier}/evidence")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertTrue(raw.startswith(b"PK"))
        status, headers, body = self.request(f"/api/native-queues/{identifier}/evidence", method="HEAD")
        self.assertEqual((status, body), (200, b""))
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(self.request(f"/api/native-queues/{identifier}/stop", value={"resume": True})[0], 400)
        status, _, raw = self.request(f"/api/native-queues/{identifier}/stop", value={})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["queue"]["status"], "cancelled")
        self.assertIsNone(self.server.native_queues.active_queue())

    def test_native_and_ordinary_starts_exclude_each_other(self):
        self.start()
        with patch.object(self.server.jobs, "start") as ordinary_start:
            self.assertEqual(self.request("/api/jobs", value={"task_id": "regression", "action": "baseline"})[0], 409)
            ordinary_start.assert_not_called()
        self.server.native_queues.active = None
        with patch.object(self.server.jobs, "active_job", return_value={"id": "ordinary"}):
            self.assertEqual(self.request("/api/native-queues", value={
                "regression_task_id": "regression", "classification_task_id": "classification"})[0], 409)

    def test_foreign_reads_traversal_and_ambiguous_log_queries_are_refused(self):
        identifier = self.start()
        self.assertEqual(self.request("/api/native-queues", headers={"Origin": "https://foreign.invalid"})[0], 403)
        self.assertEqual(self.request(f"/api/native-queues/{identifier}/../secret")[0], 400)
        self.assertEqual(self.request(f"/api/native-queues/{identifier}/logs?kind=stdout&kind=stderr")[0], 400)
        self.assertEqual(self.request(f"/api/native-queues/{identifier}/logs?filename=secret")[0], 404)
        self.assertEqual(self.request("/api/native-queues/not-a-queue/evidence")[0], 404)

    def test_external_cpu_lane_rejects_ordinary_training_before_job_creation(self):
        path = self.root / "external-cpu.lock"
        with patch.object(lane, "machine_lock_path", return_value=path), lane.coordinator_lock(path):
            with self.assertRaisesRegex(RuntimeError, "coordinator"):
                self.server.jobs.start("regression", "baseline")
        self.assertIsNone(self.server.jobs.active_job())
        self.assertEqual(self.server.jobs.list(), [])


class ContainedLaneTests(unittest.TestCase):
    def test_supervisor_owns_lane_during_child_and_releases_after_failure(self):
        command = [sys.executable, "-u", str(supervisor.ROOT / "app/worker.py"), "--config", "fixture.json"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cpu.lock"
            def child(*args, **kwargs):
                with self.assertRaises(RuntimeError), lane.coordinator_lock(path):
                    pass
                raise OSError("fixture launch failure")
            with patch.object(lane, "machine_lock_path", return_value=path), \
                    patch.object(supervisor, "_contain_windows"), \
                    patch.object(supervisor, "_contain_linux", return_value=False), \
                    patch.object(supervisor.subprocess, "call", side_effect=child):
                with self.assertRaisesRegex(OSError, "fixture launch failure"):
                    supervisor.main(["--cpu-lane", "--", *command])
            with lane.coordinator_lock(path):
                pass

    def test_busy_lane_never_spawns_another_model(self):
        command = [sys.executable, "-u", str(supervisor.ROOT / "app/worker.py"), "--config", "fixture.json"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cpu.lock"
            with patch.object(lane, "machine_lock_path", return_value=path), \
                    lane.coordinator_lock(path), patch.object(supervisor, "_contain_windows"), \
                    patch.object(supervisor, "_contain_linux", return_value=False), \
                    patch.object(supervisor.subprocess, "call") as launch:
                with self.assertRaisesRegex(RuntimeError, "coordinator"):
                    supervisor.main(["--cpu-lane", "--", *command])
                launch.assert_not_called()

    def test_windows_owner_is_preserved_without_copying_ambient_secrets(self):
        parent_env = {"USERNAME": "fixture-user", "OPENAI_API_KEY": "ambient-secret"}
        fake_os = SimpleNamespace(name="nt", environ=parent_env)
        with patch.object(jobs, "os", fake_os), \
                patch("agent.execution.child_environment", return_value={"PATH": "runtime-path"}):
            environment = jobs._job_environment({})
        self.assertEqual(environment["USERNAME"], "fixture-user")
        self.assertNotIn("OPENAI_API_KEY", environment)
        with patch.object(lane, "os", SimpleNamespace(environ=parent_env)):
            before = lane.machine_lock_path()
        with patch.object(lane, "os", SimpleNamespace(environ=environment)):
            after = lane.machine_lock_path()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
