"""Real workers retain relative-file semantics in deeply nested task folders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from agent import execution
from agent.execution import child_environment, run_python


ROOT = Path(__file__).resolve().parent.parent


class ExecutionPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.deep = self.root / "task with spaces и кириллицей"
        while len(str(self.deep)) < 330:
            self.deep /= "another deeply nested experiment folder"
        self.deep.mkdir(parents=True)
        self.assertGreater(len(str(self.deep)), 320)
        self.parent_cwd = Path.cwd()

    def assert_worker_completed(self, result):
        self.assertEqual(result.get("exit_code"), 0, result)
        self.assertFalse(result.get("timed_out"), result)
        self.assertFalse(result.get("memory_limit_exceeded"), result)
        self.assertNotIn("error", result)
        self.assertEqual(Path.cwd(), self.parent_cwd)

    def test_deep_worker_uses_the_task_directory_for_relative_reads_and_writes(self):
        payload = "Literal input: кавычки ' and spaces.\n"
        (self.deep / "relative input.txt").write_text(payload, encoding="utf-8")
        code = (
            "import json\nfrom pathlib import Path\n"
            "value = Path('relative input.txt').read_text(encoding='utf-8')\n"
            "Path('relative output.txt').write_text(value + 'processed', encoding='utf-8')\n"
            "Path('observed cwd.json').write_text(json.dumps({'cwd': str(Path.cwd())}), encoding='utf-8')\n"
        )
        result = run_python(code, self.deep, self.deep / "execution logs", 1, timeout=10)
        self.assert_worker_completed(result)
        observed = json.loads((self.deep / "observed cwd.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(observed["cwd"]).resolve(), self.deep.resolve())
        self.assertEqual((self.deep / "relative output.txt").read_text(encoding="utf-8"), payload + "processed")
        self.assertFalse((self.root / "relative output.txt").exists())

    def test_worker_already_in_a_deep_directory_can_launch_another_deep_worker(self):
        nested = self.deep / "nested child task"
        nested.mkdir()
        (nested / "input.txt").write_text("nested input", encoding="utf-8")
        nested_code = (
            "import json\nfrom pathlib import Path\n"
            "value = Path('input.txt').read_text(encoding='utf-8')\n"
            "Path('output.txt').write_text(value + ' processed', encoding='utf-8')\n"
            "Path('cwd.json').write_text(json.dumps({'cwd': str(Path.cwd())}), encoding='utf-8')\n"
        )
        code = (
            "import json\nimport sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from agent.execution import run_python\n"
            "original_cwd = Path.cwd()\n"
            f"nested = Path({str(nested)!r})\n"
            f"result = run_python({nested_code!r}, nested, nested / 'execution logs', 2, timeout=10)\n"
            "assert Path.cwd() == original_cwd, 'Nested launch changed its parent working directory'\n"
            "Path('nested result.json').write_text(json.dumps(result), encoding='utf-8')\n"
        )
        result = run_python(code, self.deep, self.deep / "execution logs", 1, timeout=20)
        self.assert_worker_completed(result)
        nested_result = json.loads((self.deep / "nested result.json").read_text(encoding="utf-8"))
        self.assert_worker_completed(nested_result)
        self.assertEqual((nested / "output.txt").read_text(encoding="utf-8"), "nested input processed")
        observed = json.loads((nested / "cwd.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(observed["cwd"]).resolve(), nested.resolve())
        self.assertFalse((self.deep / "output.txt").exists())

    def test_timeout_stops_the_actual_worker_in_a_deep_directory(self):
        code = (
            "import time\nfrom pathlib import Path\n"
            "Path('started.txt').write_text('started', encoding='utf-8')\n"
            "counter = 0\n"
            "while True:\n"
            "    Path('heartbeat.txt').write_text(str(counter), encoding='utf-8')\n"
            "    counter += 1\n"
            "    time.sleep(.02)\n"
        )
        result = run_python(code, self.deep, self.deep / "execution logs", 1, timeout=1.5)
        self.assertTrue((self.deep / "started.txt").is_file(), result)
        self.assertTrue(result.get("timed_out"), result)
        self.assertIsNotNone(result.get("exit_code"), result)
        self.assertNotEqual(result["exit_code"], 0, result)
        self.assertNotIn("error", result)
        self.assertEqual(Path.cwd(), self.parent_cwd)
        # Popen has reaped the launched worker. Check its observable writes have
        # stopped too, without relying on a matching host PID namespace.
        stopped_heartbeat = (self.deep / "heartbeat.txt").read_bytes()
        time.sleep(.15)
        self.assertEqual((self.deep / "heartbeat.txt").read_bytes(), stopped_heartbeat)

    @unittest.skipUnless(os.name == "nt", "CreateProcessW initial-cwd limitation is Windows-specific")
    def test_raw_windows_process_creation_documents_the_initial_cwd_limit(self):
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", "print('raw-process-started')"],
                cwd=self.deep, env=child_environment(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as error:
            self.assertIn(getattr(error, "winerror", None), {267, 87}, repr(error))
        else:
            try:
                stdout, stderr = process.communicate(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.skipTest("This Windows version already accepts a long initial process cwd")


class BootstrapPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.deep = self.root / "long bootstrap task with spaces и кириллицей"
        while len(str(self.deep)) < 330:
            self.deep /= "another nested task directory"
        self.deep.mkdir(parents=True)
        self.parent_cwd = Path.cwd()

    def test_bootstrap_runs_in_the_real_directory_without_exposing_its_cwd_argument(self):
        # Force the Windows branch on POSIX too: ordinary deep POSIX launches
        # alone would never execute the bootstrap that fixes CreateProcessW.
        code = (
            "import json\nimport sys\nfrom pathlib import Path\n"
            "value = Path('input.txt').read_text(encoding='utf-8')\n"
            "Path('output.txt').write_text(value + ' processed', encoding='utf-8')\n"
            "Path('observed.json').write_text(json.dumps({'cwd': str(Path.cwd()), 'argv': sys.argv}), encoding='utf-8')\n"
        )
        (self.root / "input.txt").write_text("short input", encoding="utf-8")
        (self.deep / "input.txt").write_text("deep input", encoding="utf-8")
        with patch.object(execution, "WINDOWS", False):
            ordinary = run_python(code, self.root, self.root / "short logs", 1, timeout=10)
        with patch.object(execution, "WINDOWS", True):
            bootstrapped = run_python(code, self.deep, self.deep / "deep logs", 1, timeout=10)
        for result in (ordinary, bootstrapped):
            self.assertEqual(result.get("exit_code"), 0, result)
            self.assertFalse(result.get("timed_out"), result)
            self.assertNotIn("error", result)
        expected = json.loads((self.root / "observed.json").read_text(encoding="utf-8"))
        actual = json.loads((self.deep / "observed.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(actual["cwd"]).resolve(), self.deep.resolve())
        self.assertEqual((self.deep / "output.txt").read_text(encoding="utf-8"), "deep input processed")
        # Preserve the observable argv of the established runner. In particular,
        # the extra launch directory must be consumed before user code starts.
        self.assertEqual([Path(value).name for value in actual["argv"]],
                         [Path(value).name for value in expected["argv"]])
        self.assertFalse(any(Path(value).resolve() == self.deep.resolve() for value in actual["argv"]))
        self.assertEqual(Path.cwd(), self.parent_cwd)

    def test_short_windows_launch_preserves_the_existing_command_and_cwd(self):
        script = self.root / "script.py"
        script.write_text("pass\n", encoding="utf-8")
        with patch.object(execution, "WINDOWS", False):
            ordinary = execution._python_launch(script, self.root)
        with patch.object(execution, "WINDOWS", True):
            windows = execution._python_launch(script, self.root)
        self.assertEqual(windows, ordinary)
        self.assertEqual(Path.cwd(), self.parent_cwd)

    def test_missing_long_task_directory_fails_before_any_user_code_executes(self):
        missing = self.deep / "not created"
        marker = self.root / "must not be written.txt"
        code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        with patch.object(execution, "WINDOWS", True):
            with self.assertRaises(NotADirectoryError):
                run_python(code, missing, self.root / "missing task logs", 1, timeout=10)
        self.assertFalse(marker.exists())
        self.assertFalse(missing.exists())
        self.assertEqual(Path.cwd(), self.parent_cwd)

    def test_windows_length_uses_utf16_units_for_astral_characters(self):
        # Keep the Python character count below the initial-cwd limit while
        # crossing it in UTF-16, as Windows counts a non-BMP character twice.
        suffix = 240 - len(str(self.root)) - 60 - 30 - 3
        self.assertGreater(suffix, 0)
        astral = self.root / ("a" * 60) / ("😀" * 30) / ("b" * suffix)
        self.assertLess(len(str(astral)), 258)
        self.assertGreater(len(str(astral).encode("utf-16-le")) // 2, 258)
        astral.mkdir(parents=True)
        script = self.root / "astral probe.py"
        script.write_text("pass\n", encoding="utf-8")
        with patch.object(execution, "WINDOWS", True):
            _, launch_cwd = execution._python_launch(script, astral)
        self.assertNotEqual(launch_cwd, astral)
        self.assertLessEqual(len(str(launch_cwd).encode("utf-16-le")) // 2, 258)
        self.assertEqual(Path.cwd(), self.parent_cwd)


if __name__ == "__main__":
    unittest.main()
