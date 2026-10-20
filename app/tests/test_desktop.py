from pathlib import Path
import errno
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from app.desktop import close_server, open_server, project_paths


class DesktopTests(unittest.TestCase):
    def test_paths_are_checkout_relative(self):
        app = Path("/project с пробелами/app")
        self.assertEqual(project_paths(app), (app.parent / "data", app / ".workspace"))

    def test_busy_port_gets_own_server_not_another_application(self):
        factory = Mock(side_effect=[OSError(errno.EADDRINUSE, "busy"), "own server"])
        self.assertEqual(open_server(factory, Path("data"), Path("state")), "own server")
        self.assertEqual([call.args[1] for call in factory.call_args_list], [8765, 0])

    def test_workspace_lock_error_is_not_retried_on_another_port(self):
        factory = Mock(side_effect=RuntimeError("already running"))
        with self.assertRaisesRegex(RuntimeError, "already running"):
            open_server(factory, Path("data"), Path("state"))
        self.assertEqual(factory.call_count, 1)

    def test_shutdown_always_closes_job_manager_after_error(self):
        server = Mock()
        server.shutdown.side_effect = RuntimeError("stopping")
        server.jobs.active_job.return_value = None
        worker = Mock()
        worker.is_alive.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            close_server(server, worker)
        server.server_close.assert_called_once()
        worker.join.assert_called_once_with(timeout=5)

    def test_waits_for_experiment_cleanup_before_allowing_process_exit(self):
        server = Mock()
        server.jobs.active_job.side_effect = [{"id": "pending-history-write"}, None]
        worker = Mock()
        worker.is_alive.side_effect = [True, False]
        with patch("app.desktop.time.sleep") as wait:
            close_server(server, worker)
        wait.assert_called_once_with(0.1)
        server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
