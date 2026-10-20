"""Atomic replacement retries Windows locks without bypassing path guards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from app import file_io, worker


def locked_error(code=5):
    error = PermissionError(13, "Target is temporarily locked")
    error.winerror = code
    return error


class FileIOTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.source = self.root / "closed.tmp"
        self.destination = self.root / "stage.json"

    def test_success_replaces_the_whole_file_and_checks_paths_first(self):
        self.source.write_bytes(b'{"stage":"new"}\n')
        self.destination.write_bytes(b'{"stage":"old"}\n')
        check = Mock()
        file_io.replace_file(self.source, self.destination, check=check)
        check.assert_called_once_with()
        self.assertEqual(json.loads(self.destination.read_bytes()), {"stage": "new"})
        self.assertFalse(self.source.exists())

    def test_transient_windows_lock_codes_retry_the_same_atomic_replacement(self):
        check = Mock()
        errors = [locked_error(code) for code in (5, 32, 33)]
        with patch.object(file_io, "WINDOWS", True), \
                patch.object(file_io.os, "replace", side_effect=[*errors, None]) as replace, \
                patch.object(file_io.time, "sleep") as sleep:
            file_io.replace_file(self.source, self.destination, check=check)
        self.assertEqual(replace.call_count, 4)
        self.assertTrue(all(call.args == (self.source, self.destination) for call in replace.call_args_list))
        self.assertEqual(check.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertTrue(all(0 < call.args[0] <= file_io.REPLACE_MAX_DELAY for call in sleep.call_args_list))

    def test_non_windows_and_unrelated_errors_fail_immediately(self):
        for windows, error in ((False, locked_error()), (True, locked_error(2)),
                               (True, locked_error(87)), (True, PermissionError("No winerror"))):
            with self.subTest(windows=windows, winerror=getattr(error, "winerror", None)):
                check = Mock()
                with patch.object(file_io, "WINDOWS", windows), \
                        patch.object(file_io.os, "replace", side_effect=error) as replace, \
                        patch.object(file_io.time, "sleep") as sleep:
                    with self.assertRaises(OSError) as caught:
                        file_io.replace_file(self.source, self.destination, check=check)
                self.assertIs(caught.exception, error)
                replace.assert_called_once_with(self.source, self.destination)
                check.assert_called_once_with()
                sleep.assert_not_called()

    def test_retry_exhaustion_preserves_the_error_and_has_a_bounded_delay(self):
        error = locked_error()
        with patch.object(file_io, "WINDOWS", True), \
                patch.object(file_io.os, "replace", side_effect=error) as replace, \
                patch.object(file_io.time, "sleep") as sleep:
            with self.assertRaises(OSError) as caught:
                file_io.replace_file(self.source, self.destination)
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, file_io.REPLACE_ATTEMPTS)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1)

    def test_path_guard_can_reject_a_link_swap_before_the_next_replacement(self):
        rejected = ValueError("Destination became a link")
        check = Mock(side_effect=[None, rejected])
        with patch.object(file_io, "WINDOWS", True), \
                patch.object(file_io.os, "replace", side_effect=[locked_error(), None]) as replace, \
                patch.object(file_io.time, "sleep"):
            with self.assertRaises(ValueError) as caught:
                file_io.replace_file(self.source, self.destination, check=check)
        self.assertIs(caught.exception, rejected)
        self.assertEqual(check.call_count, 2)
        replace.assert_called_once_with(self.source, self.destination)

    def test_initial_path_rejection_never_attempts_replacement(self):
        with patch.object(file_io.os, "replace") as replace:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                file_io.replace_file(self.source, self.destination,
                                     check=Mock(side_effect=ValueError("Unsafe path")))
        replace.assert_not_called()

    def test_worker_exhausted_json_replacement_cleans_temp_and_keeps_old_json(self):
        old = b'{"stage":"old"}\n'
        self.destination.write_bytes(old)
        error = locked_error()
        with patch.object(file_io, "WINDOWS", True), patch.object(file_io, "REPLACE_ATTEMPTS", 3), \
                patch.object(file_io.os, "replace", side_effect=error) as replace, \
                patch.object(file_io.time, "sleep"):
            with self.assertRaises(OSError) as caught:
                worker._atomic_json(self.destination, {"stage": "new"})
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(self.destination.read_bytes(), old)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows open handles prevent destination replacement")
    def test_windows_reader_release_allows_atomic_json_replacement(self):
        old = {"stage": "old", "rows": list(range(20))}
        new = {"stage": "готово", "rows": list(range(1000))}
        self.destination.write_text(json.dumps(old), encoding="utf-8")
        self.source.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
        retry_started = threading.Event()
        reader_released = threading.Event()
        errors = []
        checks = 0

        def check():
            nonlocal checks
            checks += 1
            if checks == 2:
                retry_started.set()
                if not reader_released.wait(5):
                    raise TimeoutError("Reader-release barrier timed out")

        def write():
            try:
                file_io.replace_file(self.source, self.destination, check=check)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=write, daemon=True)
        try:
            with self.destination.open("rb") as reader:
                thread.start()
                self.assertTrue(retry_started.wait(5), "The open reader did not trigger a retry")
                self.assertEqual(json.load(reader), old)
        finally:
            reader_released.set()
            if thread.ident is not None:
                thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "The writer did not finish after reader release")
        self.assertEqual(errors, [])
        self.assertGreaterEqual(checks, 2)
        self.assertEqual(json.loads(self.destination.read_text(encoding="utf-8")), new)
        self.assertFalse(self.source.exists())


if __name__ == "__main__":
    unittest.main()
