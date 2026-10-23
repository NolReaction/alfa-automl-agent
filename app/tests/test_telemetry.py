"""Hardware dashboard telemetry is cached, bounded and never model evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from app import telemetry


GPU_OUTPUT = b"NVIDIA GeForce RTX 3080, 10240, 256, 37, 52\n"


class NvidiaParsingTests(unittest.TestCase):
    def test_measured_gpu_counters_are_finite_and_mib_are_converted_to_bytes(self):
        result = telemetry.parse_nvidia_csv(GPU_OUTPUT)
        self.assertEqual(result["status"], "measured")
        device = result["devices"][0]
        self.assertEqual(device["memory_total_bytes"], 10240 * 1024**2)
        self.assertEqual(device["memory_used_bytes"], 256 * 1024**2)
        self.assertEqual(device["utilization_percent"], 37)
        self.assertEqual(device["temperature_c"], 52)

    def test_unknown_nonfinite_and_impossible_values_are_not_reported_as_zero(self):
        for row in (b"GPU, N/A, N/A, N/A, N/A\n", b"GPU, inf, nan, NaN, -inf\n",
                    b"GPU, -1, -2, 101, 999\n"):
            with self.subTest(row=row):
                result = telemetry.parse_nvidia_csv(row)
                self.assertEqual(result["status"], "partial")
                self.assertTrue(result["available"])
                device = result["devices"][0]
                for name in ("memory_total_bytes", "memory_used_bytes", "utilization_percent", "temperature_c"):
                    self.assertIsNone(device[name])
        result = telemetry.parse_nvidia_csv(b"GPU, 1024, 2048, 0, 0\n")
        self.assertIsNone(result["devices"][0]["memory_used_bytes"])
        self.assertEqual(result["devices"][0]["utilization_percent"], 0)
        self.assertEqual(result["devices"][0]["temperature_c"], 0)

    def test_output_shape_device_count_and_bytes_are_bounded(self):
        for raw in (b"", b"not CSV", b'"unclosed, 1, 2, 3, 4', GPU_OUTPUT * 9,
                    b"x" * (telemetry.GPU_OUTPUT_BYTES + 1)):
            with self.subTest(size=len(raw)):
                result = telemetry.parse_nvidia_csv(raw)
                self.assertFalse(result["available"])
                self.assertEqual(result["devices"], [])
        self.assertEqual(len(telemetry.parse_nvidia_csv(GPU_OUTPUT * 8)["devices"]), 8)

    def test_subprocess_has_fixed_query_timeout_and_discards_stderr(self):
        process = Mock(stdout=io.BytesIO(GPU_OUTPUT))
        process.wait.return_value = 0
        process.poll.return_value = 0
        argv = ["nvidia-smi", *telemetry.GPU_QUERY]
        with patch.object(telemetry.subprocess, "Popen", return_value=process) as launch:
            result = telemetry._bounded_nvidia(argv)
        self.assertEqual(result["status"], "measured")
        args, options = launch.call_args
        self.assertEqual(args[0], argv)
        self.assertIs(options["shell"], False)
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        process.wait.assert_called_once_with(timeout=2.0)

    def test_oversized_stdout_is_killed_and_timeout_never_echoes_driver_diagnostics(self):
        process = Mock(stdout=io.BytesIO(b"x" * (telemetry.GPU_OUTPUT_BYTES + 100)))
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(telemetry.subprocess, "Popen", return_value=process):
            result = telemetry._bounded_nvidia(["nvidia-smi", *telemetry.GPU_QUERY])
        self.assertEqual(result["reason_code"], "gpu_output_too_large")
        process.kill.assert_called()
        timed_out = Mock(stdout=io.BytesIO(b""))
        timed_out.wait.side_effect = [subprocess.TimeoutExpired("private-command-secret", 2), 0]
        timed_out.poll.return_value = 0
        with patch.object(telemetry.subprocess, "Popen", return_value=timed_out):
            result = telemetry._bounded_nvidia(["nvidia-smi", *telemetry.GPU_QUERY])
        self.assertEqual(result["reason_code"], "nvidia_smi_timeout")
        self.assertNotIn("private-command-secret", json.dumps(result))
        timed_out.kill.assert_called()

    def test_nvidia_launch_error_does_not_expose_the_executable_path(self):
        with patch.object(telemetry.subprocess, "Popen", side_effect=OSError("private-executable-secret")):
            result = telemetry._bounded_nvidia(["private-executable-secret", *telemetry.GPU_QUERY])
        self.assertEqual(result["reason_code"], "nvidia_smi_unavailable")
        self.assertNotIn("private-executable-secret", json.dumps(result))


class SystemMonitorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.now = 0.0
        self.stack.enter_context(patch.object(telemetry.time, "monotonic", side_effect=lambda: self.now))
        self.stack.enter_context(patch.object(telemetry.platform, "processor", return_value="Test CPU"))
        self.stack.enter_context(patch.object(telemetry.psutil, "cpu_count", side_effect=lambda logical: 20 if logical else 10))
        self.cpu = self.stack.enter_context(patch.object(telemetry.psutil, "cpu_percent", return_value=40.0))
        self.memory = self.stack.enter_context(patch.object(telemetry.psutil, "virtual_memory", return_value=
            SimpleNamespace(total=32 * 1024**3, available=16 * 1024**3, percent=50)))
        self.disk = self.stack.enter_context(patch.object(telemetry.psutil, "disk_usage", return_value=
            SimpleNamespace(total=100 * 1024**3, free=70 * 1024**3, percent=30)))
        self.which = self.stack.enter_context(patch.object(telemetry.shutil, "which", return_value=None))
        self.monitor = telemetry.SystemMonitor(Path(self.directory) / "source-secret",
                                                Path(self.directory) / "private-results" / "not-created")
        self.addCleanup(self.monitor.close)

    def test_initial_cpu_is_unknown_then_measurement_uses_the_same_sampling_thread(self):
        sampling_threads = []
        def cpu(interval):
            self.assertIsNone(interval)
            sampling_threads.append(threading.get_ident())
            return 0 if len(sampling_threads) == 1 else 62.5
        self.cpu.side_effect = cpu
        first = self.monitor.snapshot()
        self.assertIsNone(first["cpu"]["percent"])
        self.assertEqual(first["cpu"]["measurement_status"], "initializing")
        self.now = 4
        with ThreadPoolExecutor(max_workers=1) as requests:
            second = requests.submit(self.monitor.snapshot).result(timeout=2)
        self.assertEqual(second["cpu"]["percent"], 62.5)
        self.assertEqual(second["cpu"]["measurement_status"], "measured")
        self.assertEqual(len(set(sampling_threads)), 1)
        self.assertNotEqual(sampling_threads[0], threading.get_ident())

    def test_system_and_gpu_have_separate_caches_and_snapshots_are_copies(self):
        self.which.return_value = "nvidia-smi"
        with patch.object(telemetry, "_bounded_nvidia", side_effect=lambda _: telemetry.parse_nvidia_csv(GPU_OUTPUT)) as gpu:
            first = self.monitor.snapshot()
            first["gpu"]["devices"][0]["name"] = "changed"
            cached = self.monitor.get()
            self.assertEqual(cached["gpu"]["devices"][0]["name"], "NVIDIA GeForce RTX 3080")
            self.assertEqual(self.cpu.call_count, 1)
            self.now = 4
            self.monitor.snapshot()
            self.assertEqual(self.cpu.call_count, 2)
            self.assertEqual(gpu.call_count, 1)
            self.now = 11
            self.monitor.snapshot()
            self.assertEqual(self.cpu.call_count, 3)
            self.assertEqual(gpu.call_count, 2)
            self.assertEqual(gpu.call_args.args[0], ["nvidia-smi", *telemetry.GPU_QUERY])

    def test_concurrent_requests_share_one_gpu_probe_and_one_system_sample(self):
        self.which.return_value = "nvidia-smi"
        barrier = threading.Barrier(5)
        def query(_):
            time.sleep(.03)
            return telemetry.parse_nvidia_csv(GPU_OUTPUT)
        def request():
            barrier.wait(timeout=2)
            return self.monitor.snapshot()
        with patch.object(telemetry, "_bounded_nvidia", side_effect=query) as gpu:
            with ThreadPoolExecutor(max_workers=5) as requests:
                results = list(requests.map(lambda _: request(), range(5)))
        self.assertEqual(gpu.call_count, 1)
        self.assertEqual(self.cpu.call_count, 1)
        self.assertEqual(self.memory.call_count, 1)
        self.assertEqual(len(results), 5)

    def test_missing_gpu_and_process_errors_are_unknown_without_secrets_or_fake_training(self):
        self.cpu.side_effect = psutil.NoSuchProcess(123, msg="source-secret")
        self.memory.side_effect = psutil.AccessDenied(123, msg="private-memory-secret")
        self.disk.side_effect = OSError("private-results-secret")
        with patch.object(telemetry, "_bounded_nvidia") as gpu:
            snapshot = self.monitor.snapshot()
        gpu.assert_not_called()
        self.assertIsNone(snapshot["cpu"]["percent"])
        self.assertIsNone(snapshot["memory"]["total_bytes"])
        self.assertIsNone(snapshot["disk"]["free_bytes"])
        self.assertEqual(snapshot["gpu"]["reason_code"], "nvidia_smi_missing")
        self.assertFalse(snapshot["training"]["gpu_enabled"])
        self.assertEqual(snapshot["training"]["cpu_threads"], 8)
        self.assertEqual(snapshot["scope"], "system_wide_not_training_process")
        raw = json.dumps(snapshot, allow_nan=False)
        for secret in (self.directory, "source-secret", "private-memory-secret", "private-results-secret"):
            self.assertNotIn(secret, raw)

    def test_nvidia_visibility_does_not_enable_gpu_training_and_close_stops_sampling(self):
        self.which.return_value = "nvidia-smi"
        with patch.object(telemetry, "_bounded_nvidia", return_value=telemetry.parse_nvidia_csv(GPU_OUTPUT)) as gpu:
            snapshot = self.monitor.snapshot()
            self.assertTrue(snapshot["gpu"]["available"])
            self.assertFalse(snapshot["training"]["gpu_enabled"])
            self.monitor.close()
            self.now = 100
            self.monitor.snapshot()
        self.assertEqual(self.cpu.call_count, 1)
        self.assertEqual(gpu.call_count, 1)

    def test_nvidia_lookup_error_does_not_break_the_hardware_snapshot(self):
        self.which.side_effect = OSError("private-search-path-secret")
        snapshot = self.monitor.snapshot()
        self.assertEqual(snapshot["gpu"]["reason_code"], "nvidia_smi_unavailable")
        self.assertTrue(snapshot["memory"]["available"])
        self.assertNotIn("private-search-path-secret", json.dumps(snapshot))


if __name__ == "__main__":
    unittest.main()
