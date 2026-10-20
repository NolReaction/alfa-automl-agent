"""Terminal failures retain bounded, redacted causes without more research."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.research_candidate import CandidateExecutionError
from agent.research_v3 import (
    FINAL_FAILURE_MESSAGE_LIMIT, FINAL_FAILURE_STDERR_LIMIT, run_agent_v3,
)
from tests.test_research import FakeClient, make_regression, response
from tests.test_team_integration import fixture_baseline


class ResearchFailureDiagnosticTests(unittest.TestCase):
    def run_failure(self, root, execute, *, key="configured-diagnostic-secret"):
        task = make_regression(root)
        client = FakeClient([response()])
        with patch.dict("os.environ", {"OPENAI_API_KEY": key}), \
                patch("agent.research_v3.prepare_baseline", side_effect=fixture_baseline), \
                patch("agent.research_v3.V3State.measure_inner",
                      side_effect=OSError("Fixture inner measurement unavailable")), \
                patch("agent.research_v3.run_candidate", side_effect=execute), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError) as raised:
                run_agent_v3(task, client, max_steps=1, time_budget=90)
        reports = list((task.parent / ".agent_runs").glob("*/run.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse((task.parent / "submission.csv").exists())
        self.assertEqual(len(client.received_messages), 1)
        self.assertNotIn("Injected terminal", json.dumps(client.received_messages))
        self.assertNotIn(key, json.dumps(report))
        self.assertNotIn(key, str(raised.exception))
        self.assertIn("No verified full-data predictions", str(raised.exception))
        self.assertIn(str(reports[0]), str(raised.exception))
        self.assertEqual(report["error"], str(raised.exception))
        self.assertEqual(report["finalization_failure"]["report_path"], str(reports[0]))
        return str(raised.exception), report

    def test_outer_file_error_before_child_execution_retains_the_underlying_path_failure(self):
        phases = []

        def execute(candidate, **kwargs):
            phases.append(kwargs["phase"])
            raise FileNotFoundError(2, "Injected terminal path unavailable",
                                    "C:/offline/context.json")

        with tempfile.TemporaryDirectory() as directory:
            message, report = self.run_failure(Path(directory), execute)
        self.assertEqual(phases, ["outer_validation"])
        self.assertIn("Injected terminal path unavailable", message)
        self.assertIn("C:/offline/context.json", message)
        cause, = report["finalization_failure"]["causes"]
        self.assertEqual(cause["phase"], "outer_validation")
        self.assertEqual(cause["recipe"], "baseline")
        self.assertEqual(cause["error_type"], "FileNotFoundError")
        self.assertIn("Injected terminal path unavailable",
                      report["evaluations"]["final_check"]["errors"]["baseline"])
        self.assertEqual(report["stages"][3]["status"], "failed")
        self.assertNotIn("execution", cause)

    def test_failed_refit_keeps_bounded_redacted_stderr_and_prints_its_actionable_tail(self):
        key = "configured-diagnostic-secret"
        phases = []
        child_error = CandidateExecutionError("Injected terminal child failed " + key, {
            "exit_code": 1, "timed_out": False, "memory_limit_exceeded": False,
            "error": "process error " + key,
            "stderr": "old traceback line\n" * 600
                      + "RuntimeError: Injected terminal refit failed " + key,
        })

        def execute(candidate, **kwargs):
            phases.append(kwargs["phase"])
            if kwargs["phase"] == "full_refit":
                raise child_error
            return SimpleNamespace(
                metric={"name": "RMSE", "value": 1.0, "higher_is_better": False},
                execution={"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False})

        with tempfile.TemporaryDirectory() as directory:
            message, report = self.run_failure(Path(directory), execute, key=key)
        self.assertEqual(phases, ["outer_validation", "full_refit"])
        self.assertIn("Last full_refit failure (baseline)", message)
        self.assertIn("RuntimeError: Injected terminal refit failed [REDACTED]", message)
        cause, = report["finalization_failure"]["causes"]
        self.assertEqual(cause["phase"], "full_refit")
        self.assertEqual(cause["execution"]["exit_code"], 1)
        self.assertLessEqual(len(cause["message"]), FINAL_FAILURE_MESSAGE_LIMIT)
        self.assertLessEqual(len(cause["execution"]["stderr_tail"]), FINAL_FAILURE_STDERR_LIMIT)
        self.assertTrue(cause["execution"]["stderr_tail"].startswith("[truncated]"))
        self.assertTrue(cause["execution"]["stderr_tail"].endswith("failed [REDACTED]"))
        self.assertEqual(report["evaluations"]["final_check"]["selected"], "baseline")
        self.assertEqual(report["stages"][4]["status"], "failed")

    def test_oversized_pre_execution_message_is_bounded_and_redacted_before_display(self):
        key = "configured-diagnostic-secret"

        def execute(candidate, **kwargs):
            raise OSError("Injected terminal " + key + " x" * 5000)

        with tempfile.TemporaryDirectory() as directory:
            message, report = self.run_failure(Path(directory), execute, key=key)
        cause, = report["finalization_failure"]["causes"]
        self.assertLessEqual(len(cause["message"]), FINAL_FAILURE_MESSAGE_LIMIT)
        self.assertIn("[truncated]", cause["message"])
        self.assertIn("[REDACTED]", cause["message"])
        self.assertLess(len(message), FINAL_FAILURE_MESSAGE_LIMIT * 2 + 400)


if __name__ == "__main__":
    unittest.main()
