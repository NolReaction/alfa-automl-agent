"""Entrypoint recovery preserves the original deadline and integrity failures."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import solution


class EntryRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.task = Path(temporary.name) / "task.md"
        self.task.write_text("Metric: RMSE", encoding="utf-8")
        self.environment = {"OPENAI_API_KEY": "entry-recovery-secret",
                            "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}
        self.clock = 100.0
        self.checkpoint = MagicMock()
        self.checkpoint.restore.return_value = True
        self.result = {"model": "seed", "metric": {"name": "RMSE", "value": 1.0},
                       "report_path": "report.json"}

    def invoke(self, *, client=None, research=None, offline=None, arguments=()):
        output, errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, self.environment, clear=True))
            stack.enter_context(patch("sys.argv", ["solution.py", str(self.task),
                                                   "--time-budget", "120", *arguments]))
            stack.enter_context(patch.object(solution.time, "monotonic", side_effect=lambda: self.clock))
            stack.enter_context(patch.object(solution, "load_local_env"))
            prepared = stack.enter_context(patch.object(solution, "_prepare_emergency", return_value=self.checkpoint))
            client_call = stack.enter_context(patch.object(solution, "make_client", side_effect=client))
            research_call = stack.enter_context(patch.object(solution, "run_agent", side_effect=research))
            offline_call = stack.enter_context(patch("agent.automl.run_automl", side_effect=offline,
                                                       return_value=self.result))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(errors))
            status = solution.main()
        self.assertNotIn(self.environment["OPENAI_API_KEY"], output.getvalue() + errors.getvalue())
        return status, output.getvalue(), errors.getvalue(), prepared, client_call, research_call, offline_call

    def test_constructor_failure_in_auto_runs_offline_using_only_the_remaining_budget(self):
        def fail():
            self.clock += 40
            raise OSError("Provider unavailable entry-recovery-secret")

        status, _, errors, prepared, _, research, offline = self.invoke(client=fail)
        self.assertEqual(status, 0, errors)
        self.assertEqual(prepared.call_args.args, (self.task, 220.0))
        self.assertAlmostEqual(offline.call_args.kwargs["time_budget"], 77.6)
        research.assert_not_called()
        self.checkpoint.restore.assert_not_called()

    def test_context_enter_failure_in_auto_also_uses_offline(self):
        manager = MagicMock()
        manager.__enter__.side_effect = ValueError("Provider setup failed")
        status, _, errors, _, _, research, offline = self.invoke(client=lambda: manager)
        self.assertEqual(status, 0, errors)
        research.assert_not_called()
        offline.assert_called_once()

    def test_explicit_llm_setup_failure_remains_diagnostic(self):
        status, _, errors, _, _, research, offline = self.invoke(
            client=ValueError("Missing API settings"), arguments=("--mode", "llm"))
        self.assertEqual(status, 1)
        self.assertIn("Missing API settings", errors)
        research.assert_not_called()
        offline.assert_not_called()
        self.checkpoint.restore.assert_not_called()

    def test_research_operational_failure_restores_a_verified_checkpoint(self):
        status, output, errors, _, _, _, offline = self.invoke(research=OSError("Worker stopped"))
        self.assertEqual(status, 0, errors)
        self.assertIn("quality has not been validated", output.lower())
        self.checkpoint.restore.assert_called_once_with(deadline=220.0)
        offline.assert_not_called()

    def test_research_integrity_errors_cannot_be_hidden_by_the_emergency_output(self):
        for error in (ValueError("Frozen candidate code changed"), RuntimeError("Integrity check failed")):
            with self.subTest(error=error):
                status, _, errors, _, _, _, offline = self.invoke(research=error)
                self.assertEqual(status, 1)
                self.assertIn(str(error), errors)
                self.checkpoint.restore.assert_not_called()
                offline.assert_not_called()

    def test_offline_failure_uses_the_same_verified_emergency_checkpoint(self):
        status, output, errors, _, client, _, _ = self.invoke(
            offline=RuntimeError("All model workers failed"), arguments=("--mode", "offline"))
        self.assertEqual(status, 0, errors)
        self.assertIn("target-prior CSV", output)
        client.assert_not_called()
        self.checkpoint.restore.assert_called_once_with(deadline=220.0)

    def test_unverified_or_changed_checkpoint_cannot_report_success(self):
        self.checkpoint.restore.return_value = False
        status, _, errors, *_ = self.invoke(research=OSError("Worker stopped"))
        self.assertEqual(status, 1)
        self.assertIn("Worker stopped", errors)

    def test_changed_original_inputs_prevent_even_client_failure_recovery(self):
        self.checkpoint.check_inputs.side_effect = ValueError("Original inputs changed")
        status, _, errors, _, _, _, offline = self.invoke(client=OSError("Provider unavailable"))
        self.assertEqual(status, 1)
        self.assertIn("Original inputs changed", errors)
        offline.assert_not_called()
        self.checkpoint.restore.assert_not_called()

    def test_expired_original_deadline_does_not_start_a_fresh_offline_budget(self):
        def fail():
            self.clock += 121
            raise OSError("Provider setup exhausted the deadline")

        self.checkpoint.restore.return_value = False
        status, _, errors, _, _, research, offline = self.invoke(client=fail)
        self.assertEqual(status, 1, errors)
        research.assert_not_called()
        offline.assert_not_called()
        self.checkpoint.restore.assert_called_once_with(deadline=220.0)

    def test_offline_snapshot_integrity_error_is_not_hidden_by_target_prior(self):
        status, _, errors, _, _, _, _ = self.invoke(
            offline=ValueError("Snapshot bytes changed"), arguments=("--mode", "offline"))
        self.assertEqual(status, 1)
        self.assertIn("Snapshot bytes changed", errors)
        self.checkpoint.restore.assert_not_called()

    def test_provider_close_error_preserves_the_finished_verified_solution(self):
        manager = MagicMock()
        manager.__exit__.side_effect = OSError("Provider close failed")
        with patch.object(solution, "validate_predictions", return_value={"valid": True}):
            status, _, errors, _, _, research, offline = self.invoke(client=lambda: manager)
        self.assertEqual(status, 0, errors)
        research.assert_called_once()
        offline.assert_not_called()
        self.checkpoint.restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
