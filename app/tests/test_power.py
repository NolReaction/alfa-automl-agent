import threading
import unittest
from unittest.mock import Mock

from app.power import CONTINUOUS, SYSTEM_REQUIRED, SleepGuard


class SleepGuardTests(unittest.TestCase):
    def test_idle_does_not_change_policy_and_active_request_is_released(self):
        setter = Mock(return_value=1)
        guard = SleepGuard(lambda: False, platform="win32", setter=setter, autostart=False)
        guard._tick(False)
        setter.assert_not_called()
        guard._tick(True)
        guard._tick(True)
        self.assertTrue(guard.snapshot()["active"])
        guard._tick(False)
        self.assertFalse(guard.snapshot()["active"])
        self.assertEqual([item.args[0] for item in setter.call_args_list],
                         [CONTINUOUS | SYSTEM_REQUIRED, CONTINUOUS])

    def test_rejected_request_remains_unknown_and_is_retried(self):
        setter = Mock(side_effect=[0, 1, OSError("unavailable"), 1])
        guard = SleepGuard(lambda: True, platform="win32", setter=setter, autostart=False)
        guard._tick(True)
        self.assertEqual((guard.snapshot()["requested"], guard.snapshot()["active"]), (True, False))
        guard._tick(True)
        guard._tick(False)
        self.assertTrue(guard.snapshot()["active"])
        guard._tick(False)
        self.assertFalse(guard.snapshot()["active"])

    def test_worker_releases_on_same_thread_when_closed(self):
        called = threading.Event()
        calls = []
        def setter(flags):
            calls.append((threading.get_ident(), flags))
            called.set()
            return 1
        guard = SleepGuard(lambda: True, platform="win32", setter=setter)
        self.assertTrue(called.wait(2))
        guard.close()
        self.assertFalse(guard._thread.is_alive())
        self.assertEqual([item[1] for item in calls], [CONTINUOUS | SYSTEM_REQUIRED, CONTINUOUS])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertNotEqual(calls[0][0], threading.get_ident())

    def test_unsupported_platform_does_not_create_worker_or_call_os(self):
        setter = Mock()
        guard = SleepGuard(lambda: True, platform="linux", setter=setter)
        guard._tick(True)
        guard.close()
        self.assertIsNone(guard._thread)
        self.assertFalse(guard.snapshot()["supported"])
        self.assertFalse(guard.snapshot()["active"])
        setter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
