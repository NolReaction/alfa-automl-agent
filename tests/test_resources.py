"""Scoped resource accounting and cleanup of detached experiment descendants."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from agent.execution import child_environment, run_python
from agent.resources import DEFAULT_MEMORY_LIMIT_BYTES, ProcessTreeMonitor


class FakeProcess:
    def __init__(self, pid, *, created=100.0, rss=10, cpu=1.0, parent=None):
        self.pid = pid
        self.created = created
        self.rss = rss
        self.cpu = cpu
        self.parent = os.getpid() if parent is None else parent
        self.alive = True
        self.descendants = []
        self.signals = []

    def create_time(self):
        return self.created

    def ppid(self):
        return self.parent

    def is_running(self):
        return self.alive

    def status(self):
        return psutil.STATUS_RUNNING

    def children(self, recursive=False):
        return [child for child in self.descendants if child.alive]

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0.5, children_user=1000, children_system=1000)

    def suspend(self):
        self.signals.append("suspend")

    def kill(self):
        self.signals.append("kill")
        self.alive = False


class FakePopen:
    def __init__(self, root):
        self.root = root
        self.pid = root.pid
        self.returncode = None

    def poll(self):
        if not self.root.alive:
            self.returncode = -9
        return self.returncode

    def kill(self):
        self.root.kill()

    def wait(self, timeout=None):
        if self.root.alive:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.poll()


class ResourceMonitorTests(unittest.TestCase):
    def setUp(self):
        namespace = patch("agent.resources._namespace_matches", return_value=True)
        namespace.start()
        self.addCleanup(namespace.stop)

    def test_tracking_survives_parent_exit_and_does_not_touch_unrelated_process(self):
        root = FakeProcess(100, rss=10, cpu=1)
        parent = FakeProcess(101, rss=20, cpu=2)
        detached = FakeProcess(102, rss=30, cpu=3)
        unrelated = FakeProcess(900, rss=100000)
        root.descendants = [parent, detached]
        parent.descendants = [detached]
        with patch("agent.resources.psutil.Process", return_value=root), \
                patch("agent.resources.os.getpgid", return_value=999, create=True):
            monitor = ProcessTreeMonitor(FakePopen(root), max_memory_bytes=100)
            self.assertFalse(monitor.sample())
            first = monitor.summary()
            self.assertEqual(first["peak_rss_bytes"], 60)
            self.assertEqual(first["cpu_seconds"], 7.5)
            # The root and intermediate parent disappear; the retained detached
            # worker can itself acquire a child before cleanup.
            root.alive = parent.alive = False
            root.descendants = []
            grandchild = FakeProcess(103, rss=50, cpu=4)
            detached.descendants = [grandchild]
            self.assertFalse(monitor.sample())
            self.assertTrue(monitor.terminate())
            self.assertTrue(monitor.summary()["cleanup_complete"])
        self.assertIn("kill", detached.signals)
        self.assertIn("kill", grandchild.signals)
        self.assertEqual(unrelated.signals, [])
        self.assertEqual(monitor.summary()["peak_rss_bytes"], 80)

    def test_reused_pid_is_not_measured_or_signalled(self):
        root = FakeProcess(100)
        descendant = FakeProcess(101, created=101)
        root.descendants = [descendant]
        with patch("agent.resources.psutil.Process", return_value=root), \
                patch("agent.resources.os.getpgid", return_value=999, create=True):
            monitor = ProcessTreeMonitor(FakePopen(root))
            monitor.sample()
            root.descendants = []
            descendant.created = 999  # a different process now owns that PID
            descendant.rss = 2 * DEFAULT_MEMORY_LIMIT_BYTES
            self.assertFalse(monitor.sample())
            self.assertTrue(monitor.terminate())
        self.assertEqual(descendant.signals, [])
        self.assertTrue(descendant.alive)

    def test_sum_rss_limit_and_cpu_own_times(self):
        root = FakeProcess(100, rss=60, cpu=1)
        root.descendants = [FakeProcess(101, rss=60, cpu=2)]
        with patch("agent.resources.psutil.Process", return_value=root):
            monitor = ProcessTreeMonitor(FakePopen(root), max_memory_bytes=100)
            self.assertTrue(monitor.sample())
            self.assertEqual(monitor.summary()["peak_rss_bytes"], 120)
            self.assertEqual(monitor.summary()["cpu_seconds"], 4)

    def test_unavailable_inspection_returns_none_instead_of_fake_zero(self):
        root = FakeProcess(100)
        with patch("agent.resources.psutil.Process", side_effect=psutil.AccessDenied(100)), \
                patch("agent.resources.os.getpgid", return_value=999, create=True):
            monitor = ProcessTreeMonitor(FakePopen(root))
            self.assertFalse(monitor.sample())
            report = monitor.summary()
            self.assertFalse(report["available"])
            self.assertIsNone(report["peak_rss_bytes"])
            self.assertIsNone(report["cpu_seconds"])
            self.assertIn("AccessDenied", report["unavailable_reason"])
            self.assertTrue(monitor.terminate())
            self.assertIsNone(monitor.summary()["cleanup_complete"])
        self.assertIn("kill", root.signals)

    def test_mismatched_process_namespace_never_tracks_that_pid(self):
        launched = FakeProcess(100)
        other_namespace_process = FakeProcess(100, parent=-1)
        with patch("agent.resources.psutil.Process", return_value=other_namespace_process), \
                patch("agent.resources.os.getpgid", return_value=999, create=True):
            monitor = ProcessTreeMonitor(FakePopen(launched))
            monitor.sample()
            self.assertEqual(monitor.summary()["tracked_processes"], 0)
            monitor.terminate()
        self.assertEqual(other_namespace_process.signals, [])
        self.assertIn("kill", launched.signals)

    def test_host_proc_mount_is_rejected_before_looking_up_any_pid(self):
        launched = FakeProcess(100)
        with patch("agent.resources._namespace_matches", return_value=False), \
                patch("agent.resources.psutil.Process") as inspect_pid, \
                patch("agent.resources.os.getpgid", return_value=999, create=True):
            monitor = ProcessTreeMonitor(FakePopen(launched))
            monitor.sample()
            monitor.terminate()
        inspect_pid.assert_not_called()
        self.assertFalse(monitor.summary()["available"])
        self.assertIsNone(monitor.summary()["cleanup_complete"])

    def test_child_environment_caps_numerical_threads_and_removes_api_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "not-for-workers", "OMP_NUM_THREADS": "99"}):
            environment = child_environment()
        self.assertNotIn("OPENAI_API_KEY", environment)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment[name], "8")


class RealProcessResourceTests(unittest.TestCase):
    def setUp(self):
        # Some workspaces expose /proc from a different PID namespace. The same
        # tests MUST execute on GitHub's Linux and Windows runners.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        monitor = ProcessTreeMonitor(child)
        try:
            monitor.sample()
            available = monitor.summary()["available"]
            reason = monitor.summary()["unavailable_reason"]
        finally:
            monitor.terminate()
        if not available:
            if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
                self.fail(f"Process resource monitoring unavailable on CI: {reason}")
            self.skipTest(f"Workspace cannot inspect its child PID namespace: {reason}")

    @staticmethod
    def alive(identity):
        try:
            process = psutil.Process(identity["pid"])
            return (process.create_time() == identity["created"] and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return False

    @staticmethod
    def cleanup(identity):
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["created"]:
                process.kill()
        except psutil.NoSuchProcess:
            pass

    def test_timeout_kills_detached_grandchild_but_keeps_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            nested = (
                "import json, os, subprocess, sys, time, psutil\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                "start_new_session=os.name != 'nt', "
                "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)\n"
                "Path('detached.json').write_text(json.dumps({'pid': child.pid, "
                "'created': psutil.Process(child.pid).create_time()}))\n"
                "time.sleep(30)\n"
            )
            code = (
                "import os, subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {nested!r}], "
                "start_new_session=os.name != 'nt', "
                "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)\n"
                "time.sleep(30)\n"
            )
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            identity = None
            try:
                result = run_python(code, task, task / "logs", 1, timeout=3)
                self.assertTrue((task / "detached.json").is_file(), result)
                identity = json.loads((task / "detached.json").read_text())
                self.assertTrue(result["timed_out"], result)
                self.assertTrue(result["resources"]["available"], result)
                self.assertTrue(result["resources"]["cleanup_complete"], result)
                self.assertGreaterEqual(result["resources"]["tracked_processes"], 3)
                self.assertFalse(self.alive(identity))
                self.assertIsNone(unrelated.poll())
            finally:
                if identity is not None:
                    self.cleanup(identity)
                unrelated.kill()
                unrelated.wait(timeout=5)

    def test_normal_root_exit_cleans_up_tracked_detached_child(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            code = (
                "import json, os, subprocess, sys, time, psutil\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                "start_new_session=os.name != 'nt', "
                "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)\n"
                "Path('detached.json').write_text(json.dumps({'pid': child.pid, "
                "'created': psutil.Process(child.pid).create_time()}))\n"
                "time.sleep(1)\n"
            )
            identity = None
            try:
                result = run_python(code, task, task / "logs", 1, timeout=5)
                identity = json.loads((task / "detached.json").read_text())
                self.assertEqual(result["exit_code"], 0, result)
                self.assertFalse(result["timed_out"], result)
                self.assertTrue(result["resources"]["cleanup_complete"], result)
                self.assertFalse(self.alive(identity))
            finally:
                if identity is not None:
                    self.cleanup(identity)

    def test_memory_limit_stops_experiment_and_reports_measured_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            limit = 48 * 1024 ** 2
            result = run_python(
                "import time; data = bytearray(80 * 1024 ** 2); time.sleep(10)",
                task, task / "logs", 1, timeout=5, max_memory_bytes=limit,
            )
            self.assertTrue(result["memory_limit_exceeded"], result)
            self.assertFalse(result["timed_out"], result)
            self.assertNotEqual(result["exit_code"], 0, result)
            self.assertTrue(result["resources"]["available"], result)
            self.assertGreater(result["resources"]["peak_rss_bytes"], limit)
            self.assertGreaterEqual(result["resources"]["cpu_seconds"], 0)
            self.assertLess(result["elapsed_seconds"], 5)


if __name__ == "__main__":
    unittest.main()
