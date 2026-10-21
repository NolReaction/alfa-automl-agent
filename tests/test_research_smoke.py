"""Smoke harness preflight, shared budget and evidence summaries without API."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import experiment_research_smoke as smoke
from agent.task_spec import load_task, load_task_data


class ResearchSmokeTests(unittest.TestCase):
    def test_default_preflight_never_constructs_client_or_runs_research(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "secret-quoted-key", "OPENAI_BASE_URL": "https://private-endpoint/v1"}), \
                patch.object(smoke, "load_local_env"), patch.object(smoke, "make_client") as client, \
                patch.object(smoke, "run_agent") as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(smoke.main(["--output-dir", directory]), 0)
            client.assert_not_called()
            run.assert_not_called()
            reports = list(Path(directory).glob("*/smoke_report.json"))
            self.assertEqual(len(reports), 1)
            raw = reports[0].read_text(encoding="utf-8")
            report = json.loads(raw)
            self.assertEqual(report["mode"], "preflight")
            self.assertTrue(report["api_configured"])
            self.assertFalse(report["api_called"])
            self.assertEqual(report["budget"]["attempts"], 0)
            self.assertIsInstance(report["source_worktree_dirty"], bool)
            self.assertEqual(len(report["source_files_sha256"]), 64)
            self.assertEqual(len(report["harness_sha256"]), 64)
            self.assertFalse(report["coverage_summary"]["full_candidate_path_passed"])
            for secret in ("secret-quoted-key", "https://private-endpoint/v1"):
                self.assertNotIn(secret, raw + output.getvalue())

    def test_execute_missing_settings_is_blocked_without_client(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {}, clear=True), patch.object(smoke, "load_local_env"), \
                patch.object(smoke, "make_client") as client, redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(["--execute", "--output-dir", directory]), 2)
            client.assert_not_called()

    def test_generated_tasks_preserve_literal_class_labels_and_id_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = smoke.create_tasks(Path(directory))
            classification = load_task(tasks["classification"])
            self.assertEqual(classification.classes, smoke.CLASS_LABELS)
            self.assertEqual(classification.output_columns, ("row_key", "outcome"))
            train, test, *_ = load_task_data(classification)
            self.assertEqual(set(train["outcome"]), set(smoke.CLASS_LABELS))
            self.assertEqual(test["row_key"].tolist(), ["0005", "NA", "test c"])
            self.assertEqual(load_task(tasks["regression"]).metric.name, "RMSE")

    def test_shared_budget_blocks_ninth_request_and_reserves_output_tokens(self):
        budget = smoke.RequestBudget()
        for _ in range(8):
            budget.reserve({"messages": [], "max_tokens": 10})
        with self.assertRaisesRegex(RuntimeError, "ceiling"):
            budget.reserve({"messages": [], "max_tokens": 10})
        self.assertEqual(budget.attempts, 8)
        self.assertEqual(budget.estimated_tokens_reserved, 88)
        self.assertTrue(budget.blocked)

    def test_summary_reports_unknown_usage_and_does_not_copy_arbitrary_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            task = smoke.create_tasks(Path(directory))["regression"]
            report = {"research_protocol_version": 3, "status": "fallback", "reason": "api_error",
                      "successful_experiments": 1, "error": "do-not-copy-provider-body",
                      "api": {"attempts": 2, "responses": 2, "usage_reported_responses": 1,
                              "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                      "evaluations": {"selection": {"metric": {"name": "R2", "value": -2.5, "higher_is_better": True}},
                                      "final_check": {"selected": "baseline", "selection_uses_outer": True,
                                                      "unbiased_estimate": False}}}
            summary = smoke.summarize_run(task, report)
            self.assertEqual(summary["api"]["responses_without_usage"], 1)
            self.assertFalse(summary["api"]["usage_complete"])
            self.assertIsNone(summary["api"]["currency_cost"])
            self.assertEqual(summary["selection"]["metric"]["value"], -2.5)
            self.assertNotIn("do-not-copy-provider-body", json.dumps(summary))
            self.assertFalse(summary["output_check"]["valid"])

    def test_explicit_execute_shares_request_budget_and_records_both_tasks(self):
        class FakeSDK:
            def __init__(self):
                self.chat = self
                self.completions = self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def with_options(self, **kwargs):
                return self

            def create(self, **kwargs):
                return SimpleNamespace()

        calls = []

        def run(task, client, *, max_steps, time_budget):
            calls.append((task.parent.name, max_steps, time_budget))
            client.with_options(timeout=1).chat.completions.create(messages=[], max_tokens=10)
            with (task.parent / "inputs" / "sample_submission.csv").open(encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            smoke._write_csv(task.parent / "submission.csv", rows)
            result = task.parent / ".agent_runs" / "fixture"
            result.mkdir(parents=True)
            smoke._write_json(result / "run.json", {
                "research_protocol_version": 3, "status": "completed", "reason": "model_finished",
                "successful_experiments": 1, "api": {"attempts": 1, "responses": 1,
                    "usage_reported_responses": 0},
                "evaluations": {"final_check": {"baseline": {"name": "RMSE", "value": 1.2, "higher_is_better": False}}},
                "publication": {"new_result_published": True, "recipe": "baseline"},
            })
            return result

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://example.test/v1"}), \
                patch.object(smoke, "load_local_env"), patch.object(smoke, "make_client", return_value=FakeSDK()), \
                patch.object(smoke, "run_agent", side_effect=run), redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(["--execute", "--output-dir", directory, "--max-steps", "3",
                                         "--time-budget", "90"]), 0)
            report = json.loads(next(Path(directory).glob("*/smoke_report.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["budget"]["attempts"], 2)
            self.assertEqual(report["budget"]["max_requests"], 6)
            self.assertEqual(report["coverage_summary"]["scope"], "inner_research_and_baseline_finalization")
            self.assertFalse(report["full_candidate_path_passed"])
            self.assertEqual(calls, [("regression", 3, 90), ("classification", 3, 90)])
            self.assertEqual([task["api"]["responses_without_usage"] for task in report["tasks"]], [1, 1])

    def test_api_failure_does_not_start_second_task(self):
        def run(task, client, **kwargs):
            result = task.parent / ".agent_runs" / "fixture"
            result.mkdir(parents=True)
            smoke._write_json(result / "run.json", {"research_protocol_version": 3, "status": "fallback",
                "reason": "api_error", "successful_experiments": 0, "api": {}})
            return result

        sdk = MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://example.test/v1"}), \
                patch.object(smoke, "load_local_env"), patch.object(smoke, "make_client", return_value=sdk), \
                patch.object(smoke, "run_agent", side_effect=run) as agent, redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(["--execute", "--output-dir", directory]), 1)
            self.assertEqual(agent.call_count, 1)

    def test_completed_baseline_after_outer_candidate_failure_is_not_full_candidate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            task = smoke.create_tasks(Path(directory))["regression"]
            smoke._write_csv(task.parent / "submission.csv", [
                ["row_key", "outcome"], ["0005", 0], ["NA", 0], ["test c", 0]])
            report = {"research_protocol_version": 3, "status": "completed", "successful_experiments": 1,
                      "frozen_candidate": {"code_hash": "a" * 64, "recipe_hash": "b" * 64},
                      "evaluations": {"final_check": {"baseline": {"name": "RMSE", "value": 1.5, "higher_is_better": False},
                                                      "candidate": None, "selected": "baseline", "errors": {"candidate": "timeout"}}},
                      "publication": {"new_result_published": True, "recipe": "baseline"}}
            item = smoke.summarize_run(task, report, run_dir=task.parent / ".agent_runs" / "fixture")
            self.assertTrue(item["coverage"]["inner_trial_verified"])
            self.assertFalse(item["coverage"]["generated_candidate_outer_valid"])
            self.assertFalse(item["coverage"]["generated_candidate_published"])
            coverage = smoke.coverage_summary([item, {**item, "task": "classification"}])
            self.assertTrue(coverage["integration_passed"])
            self.assertFalse(coverage["full_candidate_path_passed"])
            self.assertEqual(coverage["scope"], "inner_research_and_baseline_finalization")

    def test_generated_candidate_requires_matching_outer_refit_receipts_and_published_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            task = smoke.create_tasks(Path(directory))["regression"]
            output = b"row_key,outcome\n0005,12\nNA,4.5\ntest c,7\n"
            (task.parent / "submission.csv").write_bytes(output)
            run_dir = task.parent / ".agent_runs" / "fixture"
            frozen = {"code_hash": "a" * 64, "recipe_hash": "b" * 64}
            for directory_name, phase in (("outer_evaluations", "outer_validation"), ("full_refit", "full_refit")):
                invocation = run_dir / directory_name / "candidate" / (phase + "_fixture")
                invocation.mkdir(parents=True)
                (invocation / "predictions.csv").write_bytes(output)
                smoke._write_json(invocation / "receipt.json", {"version": 3, "phase": phase, **frozen,
                    "output_hash": hashlib.sha256(output).hexdigest()})
            report = {"research_protocol_version": 3, "status": "completed", "successful_experiments": 1,
                      "frozen_candidate": frozen,
                      "evaluations": {"final_check": {"candidate": {"name": "RMSE", "value": 0, "higher_is_better": False},
                                                      "selected": "candidate"}},
                      "publication": {"new_result_published": True, "recipe": "candidate"}}
            item = smoke.summarize_run(task, report, run_dir=run_dir)
            self.assertTrue(item["coverage"]["generated_candidate_outer_valid"])
            self.assertTrue(item["coverage"]["generated_candidate_published"])
            self.assertTrue(smoke.coverage_summary([item, {**item, "task": "classification"}])["full_candidate_path_passed"])
            # A legitimate outer tie keeps baseline; it is not a complete
            # generated-candidate publication even if both scores are valid.
            report["publication"]["recipe"] = "baseline"
            report["evaluations"]["final_check"]["selected"] = "baseline"
            tied = smoke.summarize_run(task, report, run_dir=run_dir)
            self.assertTrue(tied["coverage"]["generated_candidate_outer_valid"])
            self.assertFalse(tied["coverage"]["generated_candidate_published"])
            report["publication"]["recipe"] = "candidate"
            report["evaluations"]["final_check"]["selected"] = "candidate"
            (task.parent / "submission.csv").write_bytes(output.replace(b",12", b",99"))
            self.assertFalse(smoke.summarize_run(task, report, run_dir=run_dir)["coverage"]["generated_candidate_published"])

    def test_source_provenance_fingerprints_current_bytes_and_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "solution.py"
            source.write_bytes(b"print('one')\n")
            with patch.object(smoke, "ROOT", root), patch.object(smoke, "source_files", return_value=[source]), \
                    patch.object(smoke, "_source_commit", return_value="c" * 40), \
                    patch.object(smoke.subprocess, "run", return_value=SimpleNamespace(stdout=" M solution.py\n")):
                first = smoke.source_provenance()
                source.write_bytes(b"print('two')\n")
                second = smoke.source_provenance()
            self.assertTrue(first["source_worktree_dirty"])
            self.assertEqual(first["source_commit"], "c" * 40)
            self.assertEqual(first["source_files"][0]["path"], "solution.py")
            self.assertEqual(first["source_files"][0]["sha256"], hashlib.sha256(b"print('one')\n").hexdigest())
            self.assertNotEqual(first["source_files_sha256"], second["source_files_sha256"])
            self.assertEqual(first["harness_sha256"], second["harness_sha256"])

    def test_task_and_request_limits_reject_oversized_values_before_execution(self):
        for arguments in (["--max-steps", "5"], ["--time-budget", "601"], ["--time-budget", "nan"]):
            with self.subTest(arguments=arguments), patch.object(smoke, "make_client") as client, \
                    redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                smoke.main(arguments)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
