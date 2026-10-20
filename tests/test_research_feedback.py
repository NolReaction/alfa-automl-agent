import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.research_feedback import ResearchLedger, error_analysis
from agent.task_spec import load_task
from tests import test_research as fixtures
from agent.automl import _commit_checkpoint
from agent.research import RunState, prepare_workspace, prepare_starting_output


class FeedbackTests(unittest.TestCase):
    def test_execution_failures_stop_repeated_code_without_claiming_money_cost(self):
        ledger = ResearchLedger()
        result = {"exit_code": 1, "stderr": "failed", "elapsed_seconds": 2,
                  "selection": {"accepted": False, "verified": False,
                                "reason": "execution_failed"}}
        ledger.record("raise ValueError('failed')", result, remaining=60)
        self.assertTrue(ledger.is_repeated_failure(" raise ValueError('failed') \n"))
        self.assertFalse(ledger.is_repeated_failure("print('fixed')"))
        context = json.loads(ledger.context(best=None, remaining=60, api={"total_tokens": 12}))
        self.assertIsNone(context["monetary_cost"])
        self.assertEqual(context["reported_token_usage"]["total_tokens"], 12)
        self.assertEqual(context["consecutive_execution_failures"], 1)
        ledger.record("print('analysis')", {"exit_code": 0}, remaining=59)
        self.assertEqual(ledger.failure_streak, 0)

    def test_regression_diagnostics_do_not_change_metric_or_use_claimed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            task = fixtures.make_regression(Path(directory))
            spec = load_task(task)
            held = Path(directory) / "held.csv"
            with held.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "actual", "prediction"])
                writer.writerows([[str(i), i, i + 3] for i in range(40)])
            result = error_analysis(spec, held, groups={"category": ["a"] * 20 + ["b"] * 20})
            self.assertEqual(result["regression"]["mean_error"], 3)
            self.assertEqual(result["regression"]["absolute_error_p99"], 3)
            self.assertEqual([row["value"] for row in result["groups"]], [3, 3])

    def test_repeated_failure_falls_back_and_journal_preserves_best(self):
        with tempfile.TemporaryDirectory() as directory:
            task = fixtures.make_regression(Path(directory))
            client = fixtures.FakeClient([fixtures.response(code="raise ValueError('intentional')")] * 3)
            runner = fixtures.ResearchTests()
            run_dir = runner.run_quietly(task, client, 3)
            report = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(report["reason"], "repeated_failures")
            # The first failure executes; identical subsequent failures are blocked.
            self.assertEqual(report["experiments"], 1)
            ledger = json.loads((run_dir / "research_journal.json").read_text())
            self.assertEqual(len(ledger["trials"]), 3)
            self.assertEqual(ledger["consecutive_execution_failures"], 3)
            self.assertEqual(report["best_metric"]["value"], 10)
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             (run_dir / "last_valid_submission.csv").read_bytes())

    def test_interrupted_startup_recovers_committed_pair_not_mixed_loose_files(self):
        with tempfile.TemporaryDirectory() as directory:
            task = fixtures.make_regression(Path(directory))
            state = RunState(task, max_steps=2, time_budget=90)
            prepare_workspace(state)
            gate = state.validator

            def interrupted(*args, **kwargs):
                startup = state.run_dir / "startup"
                startup.mkdir()
                complete = state.run_dir / "complete"
                complete.mkdir()
                fixtures.existing_output(complete)
                fixtures.write_csv(complete / "holdout_predictions.csv", [["id", "actual", "prediction"]] + [
                    [row_id, actual, float(actual) + 10]
                    for row_id, actual in zip(gate.expected_ids, gate.expected_actuals)])
                _commit_checkpoint(complete / "submission.csv", complete / "holdout_predictions.csv", startup)
                # Emulate a later interrupted publication with mismatched loose files.
                (state.workspace / "submission.csv").write_text("row_key,outcome\n0005,999\nNA,999\n")
                fixtures.write_csv(startup / "holdout_predictions.csv", [["id", "actual", "prediction"]] + [
                    [row_id, actual, actual] for row_id, actual in zip(gate.expected_ids, gate.expected_actuals)])
                return {"exit_code": -9, "timed_out": True, "stderr": "interrupted"}

            with patch("agent.research.run_python", side_effect=interrupted):
                prepare_starting_output(state)
            self.assertEqual(state.report["best_metric"]["value"], 10)
            self.assertTrue(state.report["events"][-1]["recovered_checkpoint"])
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             (state.run_dir / "complete/submission.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
