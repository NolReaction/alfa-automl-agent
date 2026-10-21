"""Real process-tree checks without training or external API calls."""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from app import supervisor


SCRIPT = Path(supervisor.__file__)
SUPPORTED = os.name == "nt" or sys.platform == "linux"
FULL_CONTAINMENT = os.name == "nt" or (sys.platform == "linux" and supervisor.process_inventory_matches())


def alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int32
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2:].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


@unittest.skipUnless(SUPPORTED, "Supervisor supports Windows and Linux")
class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(FULL_CONTAINMENT, "Full lifecycle needs Windows or Linux /proc in the current PID namespace")
    def test_original_exit_code_and_logs(self):
        code = "import sys; print('ordinary output'); print('ordinary error',file=sys.stderr); sys.exit(23)"
        result = subprocess.run([sys.executable, str(SCRIPT), "--", sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(result.stdout.strip(), "ordinary output")
        self.assertEqual(result.stderr.strip(), "ordinary error")

    @unittest.skipUnless(FULL_CONTAINMENT, "Full lifecycle needs Windows or Linux /proc in the current PID namespace")
    def test_detached_child_of_fast_parent_is_terminated(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child.pid"
            child = "import time; time.sleep(30)"
            code = ("import os,subprocess,sys; from pathlib import Path; "
                    f"child=subprocess.Popen([sys.executable,'-c',{child!r}],"
                    "start_new_session=os.name!='nt',"
                    "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0); "
                    f"Path({str(marker)!r}).write_text(str(child.pid))")
            result = subprocess.run([sys.executable, str(SCRIPT), "--", sys.executable, "-c", code],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            pid = int(marker.read_text())
            try:
                deadline = time.monotonic() + 3
                while alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(alive(pid), "Detached child survived supervisor exit")
            finally:
                if alive(pid):
                    os.kill(pid, signal.SIGTERM)

    @unittest.skipUnless(FULL_CONTAINMENT, "Full lifecycle needs Windows or Linux /proc in the current PID namespace")
    def test_cancellation_reaches_adopted_child(self):
        from app.jobs import _kill_tree
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child.pid"
            grandchild = "import time; time.sleep(30)"
            intermediate = ("import os,subprocess,sys; from pathlib import Path; "
                            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],"
                            "start_new_session=os.name!='nt',"
                            "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0); "
                            f"Path({str(marker)!r}).write_text(str(p.pid))")
            parent = (f"import subprocess,sys,time; subprocess.run([sys.executable,'-c',{intermediate!r}]); "
                      "time.sleep(30)")
            process = subprocess.Popen([sys.executable, str(SCRIPT), "--", sys.executable, "-c", parent],
                                       start_new_session=os.name != "nt", stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            pid = None
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "Child did not start")
                pid = int(marker.read_text())
                self.assertTrue(alive(pid))
                _kill_tree(process.pid)
                process.wait(timeout=5)
                deadline = time.monotonic() + 3
                while alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(alive(pid), "Adopted child survived cancellation")
            finally:
                if process.poll() is None:
                    _kill_tree(process.pid)
                    process.wait(timeout=5)
                if pid and alive(pid):
                    os.kill(pid, signal.SIGTERM)

    def test_win32_structure_layout_on_64_bit(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(supervisor._BasicLimits), 64)
            self.assertEqual(ctypes.sizeof(supervisor._IOCounters), 48)
            self.assertEqual(ctypes.sizeof(supervisor._ExtendedLimits), 144)

    @unittest.skipUnless(sys.platform == "linux", "Linux process inventory")
    def test_process_inventory_requires_a_matching_readable_pid_namespace(self):
        with patch.object(supervisor.Path, "is_file", return_value=True), \
                patch.object(supervisor.os, "getpid", return_value=123):
            for target, matches in (("123", True), ("456", False), ("not-a-pid", False)):
                with self.subTest(target=target), \
                        patch.object(supervisor.os, "readlink", return_value=target):
                    self.assertIs(supervisor.process_inventory_matches(), matches)
            with patch.object(supervisor.os, "readlink", side_effect=OSError("unavailable")):
                self.assertFalse(supervisor.process_inventory_matches())
        with patch.object(supervisor.Path, "is_file", return_value=False), \
                patch.object(supervisor.os, "readlink") as read_link:
            self.assertFalse(supervisor.process_inventory_matches())
            read_link.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "Linux mismatched process inventory")
    def test_mismatched_proc_refuses_generated_command_before_prctl_or_spawn(self):
        with patch.object(supervisor.Path, "is_file", return_value=True), \
                patch.object(supervisor.os, "readlink", return_value="456"), \
                patch.object(supervisor.os, "getpid", return_value=123), \
                patch.object(supervisor.ctypes, "CDLL") as library, \
                patch.object(supervisor.subprocess, "call") as launch, \
                patch.object(supervisor, "_cleanup_linux") as cleanup:
            self.assertFalse(supervisor._contain_linux())
            with self.assertRaisesRegex(RuntimeError, "require Linux /proc"):
                supervisor.main(["--", sys.executable, "-c", "print('must not run')"])
            library.assert_not_called()
            launch.assert_not_called()
            cleanup.assert_not_called()

    def test_no_proc_allows_only_exact_fixed_recipes(self):
        worker = [sys.executable, "-u", str(supervisor.ROOT / "app" / "worker.py"), "--config", "job.json"]
        prepare = [sys.executable, "-u", str(supervisor.ROOT / "prepare_data.py"), "all", "--data-dir", "data"]
        self.assertTrue(supervisor._trusted_without_proc(worker))
        self.assertTrue(supervisor._trusted_without_proc(prepare))
        for command in ([sys.executable, "-c", "print('unsafe')"],
                        worker + ["--anything"], worker[:3] + ["--code", "unsafe"],
                        prepare[:3] + ["other", "--data-dir", "data"],
                        [sys.executable, "-u", str(supervisor.ROOT / "solution.py"), "--config", "job.json"],
                        [sys.executable, "-u", "/tmp/worker.py", "--config", "job.json"]):
            self.assertFalse(supervisor._trusted_without_proc(command), command)

    @unittest.skipUnless(sys.platform == "linux", "Linux no-proc fallback")
    def test_no_proc_refuses_arbitrary_command_before_spawn(self):
        with patch.object(supervisor, "_contain_linux", return_value=False), \
                patch.object(supervisor.subprocess, "call") as launch:
            with self.assertRaisesRegex(RuntimeError, "require Linux /proc"):
                supervisor.main(["--", sys.executable, "-c", "print('must not run')"])
            launch.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "Linux no-proc fallback")
    def test_no_proc_trusted_recipe_preserves_command_and_return_code(self):
        command = [sys.executable, "-u", str(supervisor.ROOT / "app" / "worker.py"), "--config", "job.json"]
        with patch.object(supervisor, "_contain_linux", return_value=False), \
                patch.object(supervisor.subprocess, "call", return_value=23) as launch, \
                patch.object(supervisor, "_cleanup_linux") as cleanup:
            self.assertEqual(supervisor.main(["--", *command]), 23)
            launch.assert_called_once_with(command, stdin=subprocess.DEVNULL)
            cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
