from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.server import WorkspaceServer
from app.jobs import JobManager


class ServerLifecycleTests(unittest.TestCase):
    def test_report_store_failure_releases_acquired_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("app.server.ReportStore", side_effect=OSError("report failure")):
                with self.assertRaisesRegex(OSError, "report failure"):
                    WorkspaceServer(root, port=0, state_root=root / "state")
            # Reopening the same workspace proves the OS lock was released.
            server = WorkspaceServer(root, port=0, state_root=root / "state")
            server.server_close()

    def test_late_failure_closes_socket_and_other_owners_even_if_cleanup_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root, captured = Path(folder), {}
            def fail_guard(activity):
                captured["server"] = activity.__self__
                raise RuntimeError("original startup failure")
            with patch("app.server.SleepGuard", side_effect=fail_guard), \
                 patch("app.campaigns.CampaignManager.close", side_effect=RuntimeError("cleanup failure")), \
                 self.assertLogs(level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "original startup failure"):
                    WorkspaceServer(root, port=0, state_root=root / "state")
            failed = captured["server"]
            self.assertEqual(failed.socket.fileno(), -1)
            self.assertTrue(failed.jobs._closed)
            # No process was started, so the job lock must also be available.
            jobs = JobManager(root, root / "state", failed.catalog)
            jobs.close()


if __name__ == "__main__":
    unittest.main()
