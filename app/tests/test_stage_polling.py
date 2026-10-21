"""Progress polling tolerates bounded Windows reader/replacement races."""

from __future__ import annotations

import errno
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from app import jobs, worker


def denied(winerror=None):
    error = PermissionError(errno.EACCES, "Access temporarily denied")
    if winerror is not None:
        error.winerror = winerror
    return error


class StagePollingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "stage.json"
        self.old = b'{"stage":"validating"}\n'
        self.path.write_bytes(self.old)

    def test_windows_crt_or_known_lock_recovers_after_atomic_writer_publishes_next_stage(self):
        original_open = Path.open
        for error in (denied(), denied(32)):
            with self.subTest(winerror=getattr(error, "winerror", None)):
                self.path.write_bytes(self.old)
                reads = 0

                def read_open(path, *args, **kwargs):
                    nonlocal reads
                    if path == self.path and args and args[0] == "rb":
                        reads += 1
                        if reads == 1:
                            raise error
                    return original_open(path, *args, **kwargs)

                def release_and_publish(delay):
                    self.assertGreater(delay, 0)
                    # A denied read must leave the previous complete stage intact.
                    with original_open(self.path, "rb") as stream:
                        self.assertEqual(stream.read(), self.old)
                    worker._atomic_json(self.path, {"stage": "refitting"})

                with patch.object(jobs, "STAGE_READ_WINDOWS", True), \
                        patch.object(Path, "open", new=read_open), \
                        patch.object(jobs.time, "sleep", side_effect=release_and_publish) as sleep, \
                        patch.object(jobs, "_safe", wraps=jobs._safe) as guard:
                    result = jobs._stage_json(self.path, self.root)
                self.assertEqual(result, {"stage": "refitting"})
                self.assertEqual(reads, 2)
                self.assertEqual(sleep.call_count, 1)
                self.assertGreaterEqual(guard.call_count, 4)
                self.assertEqual(json.loads(self.path.read_text()), result)
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_existence_check_is_inside_the_same_guarded_retry(self):
        original_exists = Path.exists
        checks = 0

        def exists(path):
            nonlocal checks
            if path == self.path:
                checks += 1
                if checks == 1:
                    raise denied()
            return original_exists(path)

        with patch.object(jobs, "STAGE_READ_WINDOWS", True), \
                patch.object(Path, "exists", new=exists), patch.object(jobs.time, "sleep") as sleep:
            self.assertEqual(jobs._stage_json(self.path, self.root), {"stage": "validating"})
        self.assertEqual(checks, 2)
        sleep.assert_called_once()

    def test_permanent_crt_denial_propagates_after_bounded_retries_and_preserves_old_json(self):
        original_open = Path.open
        error = denied()
        reads = 0

        def read_open(path, *args, **kwargs):
            nonlocal reads
            if path == self.path and args and args[0] == "rb":
                reads += 1
                raise error
            return original_open(path, *args, **kwargs)

        with patch.object(jobs, "STAGE_READ_WINDOWS", True), \
                patch.object(Path, "open", new=read_open), patch.object(jobs.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as caught:
                jobs._stage_json(self.path, self.root)
        self.assertIs(caught.exception, error)
        self.assertEqual(reads, jobs.STAGE_READ_ATTEMPTS)
        self.assertEqual(sleep.call_count, jobs.STAGE_READ_ATTEMPTS - 1)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), .85 + 1e-12)
        self.assertEqual(self.path.read_bytes(), self.old)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_unrelated_errors_and_non_windows_denials_propagate_immediately(self):
        for windows, error in ((False, denied(32)), (False, denied()), (True, denied(87)),
                               (True, OSError(errno.EIO, "I/O failure")),
                               (True, PermissionError("No errno information"))):
            with self.subTest(windows=windows, error=error):
                with patch.object(jobs, "STAGE_READ_WINDOWS", windows), \
                        patch.object(jobs, "_json", side_effect=error) as read, \
                        patch.object(jobs.time, "sleep") as sleep:
                    with self.assertRaises(OSError) as caught:
                        jobs._stage_json(self.path, self.root)
                self.assertIs(caught.exception, error)
                read.assert_called_once()
                sleep.assert_not_called()

    def test_invalid_json_is_not_hidden_by_retry_or_replaced_with_empty_progress(self):
        self.path.write_text('{"stage":', encoding="utf-8")
        with patch.object(jobs, "STAGE_READ_WINDOWS", True), patch.object(jobs.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "Повреждён"):
                jobs._stage_json(self.path, self.root)
        sleep.assert_not_called()

    def test_link_swap_is_rejected_by_the_next_attempt_before_reopening(self):
        original_linked = jobs._linked
        stage_guards = 0

        def linked(path):
            nonlocal stage_guards
            if path == self.path:
                stage_guards += 1
                if stage_guards >= 3:
                    return True
            return original_linked(path)

        with patch.object(jobs, "STAGE_READ_WINDOWS", True), \
                patch.object(jobs, "_linked", side_effect=linked), \
                patch.object(Path, "open", side_effect=denied()) as read, \
                patch.object(jobs.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "Ссылки"):
                jobs._stage_json(self.path, self.root)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(stage_guards, 3)
        self.assertEqual(self.path.read_bytes(), self.old)

    def test_cancellation_interrupts_a_sharing_retry_without_another_read(self):
        cancel = threading.Event()

        def blocked_read(*args):
            cancel.set()
            raise denied()

        with patch.object(jobs, "STAGE_READ_WINDOWS", True), \
                patch.object(jobs, "_json", side_effect=blocked_read) as read, \
                patch.object(jobs.time, "sleep") as sleep:
            with self.assertRaisesRegex(InterruptedError, "остановлен"):
                jobs._stage_json(self.path, self.root, cancel)
        read.assert_called_once()
        sleep.assert_not_called()

    def test_not_yet_created_stage_is_the_only_empty_progress_result(self):
        self.path.unlink()
        with patch.object(jobs.time, "sleep") as sleep:
            self.assertIsNone(jobs._stage_json(self.path, self.root))
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
