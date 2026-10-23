from pathlib import Path
import errno
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from app.desktop import close_server, launch_paths, open_server, owned_work_active, project_paths


class DesktopTests(unittest.TestCase):
    def test_paths_are_checkout_relative(self):
        app = Path("/project с пробелами/app")
        self.assertEqual(project_paths(app), (app.parent / "data", app / ".workspace"))

    def test_external_data_and_history_do_not_follow_checkout(self):
        data, state = launch_paths(["--data-dir", "public inputs", "--state-dir", "persistent history"])
        self.assertEqual(data, Path("public inputs").absolute())
        self.assertEqual(state, Path("persistent history").absolute())
        self.assertEqual(launch_paths([]), project_paths())

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

    def test_owned_work_includes_native_series_and_unconfirmed_cleanup(self):
        server = Mock()
        server.jobs.active_job.return_value = None
        for status in ("running", "cleanup_pending"):
            with self.subTest(status=status):
                server.native_queues.active_queue.return_value = {"id": "night", "status": status}
                self.assertTrue(owned_work_active(server))
        server.native_queues.active_queue.return_value = None
        self.assertFalse(owned_work_active(server))
        server.jobs.active_job.return_value = {"id": "ordinary", "status": "running"}
        self.assertTrue(owned_work_active(server))

    def test_missing_or_unconfigured_native_manager_is_not_active_work(self):
        server = Mock()
        server.jobs.active_job.return_value = None
        self.assertFalse(owned_work_active(server))
        server.native_queues = None
        self.assertFalse(owned_work_active(server))

    def test_native_cleanup_failure_is_visible_and_does_not_wait_indefinitely(self):
        server = Mock()
        server.jobs.active_job.return_value = None
        server.native_queues.active_queue.return_value = {"id": "night", "status": "cleanup_pending"}
        worker = Mock()
        worker.is_alive.side_effect = [True, False]
        with patch("app.desktop.time.sleep") as wait:
            with self.assertRaisesRegex(RuntimeError, "Не удалось подтвердить остановку ночной серии"):
                close_server(server, worker)
        wait.assert_not_called()
        server.server_close.assert_called_once()
        worker.join.assert_called_once_with(timeout=5)
        # A later desktop retry accepts the manager's confirmed final cleanup.
        server.native_queues.active_queue.return_value = None
        close_server(server, None)
        self.assertEqual(server.server_close.call_count, 2)

    def test_ordinary_cleanup_failure_is_bounded_and_retryable(self):
        server = Mock()
        server.jobs.active_job.return_value = {"id": "owned", "status": "cleanup_pending"}
        with patch("app.desktop.time.monotonic", side_effect=[0, 11]), patch("app.desktop.time.sleep") as wait:
            with self.assertRaisesRegex(RuntimeError, "Остановка эксперимента"):
                close_server(server, None)
        wait.assert_not_called()
        server.jobs.active_job.return_value = None
        close_server(server, None)
        self.assertEqual(server.server_close.call_count, 2)

    def test_paused_campaign_is_owned_work_and_unfinished_close_stays_visible(self):
        server = Mock()
        server.jobs.active_job.return_value = None
        server.campaigns.active_campaign.return_value = {"id": "plan", "status": "paused"}
        self.assertTrue(owned_work_active(server))
        with self.assertRaisesRegex(RuntimeError, "Серия экспериментов"):
            close_server(server, None)


if __name__ == "__main__":
    unittest.main()
