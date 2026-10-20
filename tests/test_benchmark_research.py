"""Research failure mode accounting, current-run evidence and publication checks."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_agent as benchmark
from tests.test_benchmark_agent import fake_checkout, quiet_benchmark


FAKE_RESEARCH = '''from pathlib import Path
import hashlib
import json
import os
import shutil
from agent.llm import safe_error_details

def run_agent_v3(task, client, *, time_budget, **options):
    assert not any(name in os.environ for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"))
    try:
        client.with_options(timeout=2).chat.completions.create(model="fake", messages=[])
    except Exception as error:
        details = safe_error_details(error)
    run_dir = task.parent / ".agent_runs" / ("a" * 32)
    run_dir.mkdir(parents=True)
    shutil.copyfile(task.parent / "inputs" / "sample_submission.csv", task.parent / "submission.csv")
    frozen = {"code_hash": "b" * 64, "recipe_hash": "c" * 64}
    receipts = dict(frozen)
    for key, phase, directory in (("outer", "outer_validation", "outer_evaluations"),
                                  ("refit", "full_refit", "full_refit")):
        output = run_dir / directory / "baseline"
        output.mkdir(parents=True)
        shutil.copyfile(task.parent / "submission.csv", output / "predictions.csv")
        receipt = {"version": 3, "phase": phase, "nonce": "d" * 32, **frozen,
                   "output_hash": hashlib.sha256((output / "predictions.csv").read_bytes()).hexdigest()}
        (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        receipts[key] = str(output / "receipt.json")
    report = {"research_protocol_version": 3, "status": "fallback", "reason": "api_error", "stage": "result",
        "stages": [{"id": stage, "status": "completed"} for stage in
                   ("preparation", "inner_research", "candidate_frozen", "outer_validation", "full_refit", "result")],
        "api": {"attempts": 1, "responses": 0}, "experiments": 0, "successful_experiments": 0,
        "baseline_search": {"status": "searched", "execution": {"exit_code": 0, "timed_out": False}},
        "host_validation": {"split": {"strategy": "random"}},
        "local_experiments": 1, "local_successful_experiments": 1,
        "local_research": {"status": "completed", "origin": "host_local", "scope": "inner_selection",
                           "api_requests": 0, "attempted": 1, "verified": 1, "failed": 0},
        "publication": {"new_result_published": True, "reason": "api_error", "recipe": "baseline"},
        "evaluations": {"final_check": {"scope": "final_check", "selected": "baseline",
                        "baseline": {"name": "RMSE", "value": 1.25, "higher_is_better": False}}},
        "best_metric": {"name": "RMSE", "value": 1.25, "higher_is_better": False},
        "output_check": {"valid": True}, "time_budget_seconds": time_budget,
        "time_allocation": {"time_budget_seconds": time_budget}, "published_recipe": "baseline",
        "frozen_baseline": frozen, "candidate_receipts": receipts, "quality_options": options,
        "events": [{"kind": "api_error", "details": details}]}
    (run_dir / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir
'''


def fake_research_checkout(directory):
    solver = fake_checkout(directory)
    # The fake checkout deliberately reuses only the safe exception classifier.
    (solver / "agent" / "__init__.py").write_bytes((benchmark.ROOT / "agent" / "__init__.py").read_bytes())
    source = (benchmark.ROOT / "agent" / "llm.py").read_text(encoding="utf-8")
    (solver / "agent" / "llm.py").write_text(source, encoding="utf-8")
    (solver / "agent" / "research_v3.py").write_text(FAKE_RESEARCH, encoding="utf-8")
    return solver


class BenchmarkResearchTests(unittest.TestCase):
    def run_fixture(self, root):
        solver = fake_research_checkout(root / "solver")
        report = quiet_benchmark(root / "results", solver_root=solver,
            cases=("linear_regression",), modes=("research_api_failure",), time_budget=5,
            solver_options={"metric_recipes": True})
        row = report["trials"][0]
        return report, row, Path(row["task"]).parent

    def test_only_provider_is_replaced_and_simulated_attempt_is_not_an_external_api_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "OPENAI_API_KEY": "secret-not-inherited", "OPENAI_BASE_URL": "https://unused.invalid"}):
            report, row, task_dir = self.run_fixture(Path(directory))
            self.assertEqual(report["status"], "completed", row)
            self.assertEqual(row["research_contract_errors"], [])
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertEqual(report["simulation"]["recorded_simulated_api_calls"], 1)
            self.assertEqual(row["solver_report"]["api"], {"attempts": 1, "responses": 0})
            self.assertEqual(row["solver_report"]["status"], "fallback")
            self.assertTrue(row["solver_report"]["receipt_checks_valid"])
            self.assertEqual(row["solver_report"]["quality_options"], {"metric_recipes": True})
            self.assertNotIn("secret-not-inherited", json.dumps(report))
            driver = (task_dir / "execution" / "experiment_000.py").read_text(encoding="utf-8")
            self.assertNotIn("host_audit", driver)
            self.assertNotIn("audit_actuals", driver)
            self.assertNotIn("candidate_plan =", driver)
            self.assertNotIn("prepare_baseline =", driver)
            self.assertIn("run_agent_v3(task, client, time_budget=remaining", driver)

    def test_full_route_must_have_current_complete_publication_and_real_local_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, row, _ = self.run_fixture(Path(directory))
            summary = row["solver_report"]
            interventions = (
                ("benchmark_evidence", {"simulated_api_calls": 2}),
                ("api", {"attempts": 2}),
                ("api", {"attempts": True}),
                ("baseline_search", {"status": "seed_only"}),
                ("local_research", {"attempted": 0, "verified": 0, "failed": 0}),
                ("publication", {"new_result_published": False}),
                ("time_allocation", {"time_budget_seconds": 10000}),
            )
            for field, changes in interventions:
                with self.subTest(field=field, changes=changes):
                    altered = deepcopy(summary)
                    altered[field].update(changes)
                    self.assertTrue(benchmark._research_contract(altered, 5, {"strategy": "random"}))
            self.assertIn("automatic_validation_policy_mismatch", benchmark._research_contract(
                summary, 5, {"strategy": "time"}))
            altered = deepcopy(summary)
            altered["stages"][3]["status"] = "failed"
            self.assertIn("terminal_stages_incomplete", benchmark._research_contract(altered, 5))

    def test_startup_execution_errors_and_missing_or_inconsistent_outer_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, row, _ = self.run_fixture(Path(directory))
            summary = row["solver_report"]
            for failure in ({"error": "Process cleanup failed"}, {"memory_limit_exceeded": True}):
                with self.subTest(startup_failure=failure):
                    altered = deepcopy(summary)
                    altered["baseline_search"]["execution"].update(failure)
                    self.assertIn("real_startup_search_not_completed", benchmark._research_contract(altered, 5))
            for evaluations in (None, {}, {"final_check": {"scope": "selection", "selected": "baseline"}}):
                with self.subTest(evaluations=evaluations):
                    altered = deepcopy(summary)
                    altered["evaluations"] = evaluations
                    self.assertIn("host_outer_evaluation_not_verified", benchmark._research_contract(altered, 5))
            for changes in ({"value": float("nan")}, {"value": float("inf")}, {"value": True},
                            {"name": "MAE"}, {"higher_is_better": True}, {"value": 99}):
                with self.subTest(metric_changes=changes):
                    altered = deepcopy(summary)
                    altered["evaluations"]["final_check"]["baseline"].update(changes)
                    self.assertIn("host_outer_evaluation_not_verified", benchmark._research_contract(
                        altered, 5, metric_name="RMSE"))
            altered = deepcopy(summary)
            altered["evaluations"]["final_check"]["selected"] = "candidate"
            self.assertIn("host_outer_evaluation_not_verified", benchmark._research_contract(altered, 5))

    def test_stale_alternative_run_is_never_used_and_report_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, row, task_dir = self.run_fixture(Path(directory))
            evidence_path = task_dir / "solver_output" / "research_benchmark.json"
            evidence = json.loads(evidence_path.read_text())
            current = task_dir / evidence["run_report"]
            stale = task_dir / ".agent_runs" / ("f" * 32) / "run.json"
            stale.parent.mkdir()
            stale.write_bytes(current.read_bytes())
            current.unlink()
            self.assertIsNone(benchmark._research_summary(task_dir))
            current.write_bytes(stale.read_bytes() + b" ")
            self.assertIsNone(benchmark._research_summary(task_dir))
            evidence["run_report"] = "../outside/run.json"
            evidence_path.write_text(json.dumps(evidence))
            self.assertIsNone(benchmark._research_summary(task_dir))

    def test_refit_receipt_must_match_published_bytes_and_frozen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, row, task_dir = self.run_fixture(Path(directory))
            evidence = row["solver_report"]["benchmark_evidence"]
            current = task_dir / evidence["run_report"]
            report = json.loads(current.read_text())
            receipt_path = Path(report["candidate_receipts"]["refit"])
            receipt = json.loads(receipt_path.read_text())
            for key, value in (("code_hash", "f" * 64), ("output_hash", "e" * 64), ("phase", "inner_selection")):
                with self.subTest(field=key):
                    changed = {**receipt, key: value}
                    receipt_path.write_text(json.dumps(changed))
                    summary = benchmark._research_summary(task_dir)
                    self.assertFalse(summary["receipt_checks_valid"])
            receipt_path.write_text(json.dumps(receipt))
            (task_dir / "submission.csv").write_text("different bytes")
            self.assertFalse(benchmark._research_summary(task_dir)["receipt_checks_valid"])

    def test_malformed_research_fields_fail_closed_and_defaults_stay_offline(self):
        self.assertEqual(benchmark.DEFAULT_MODES, ("seed", "automl"))
        self.assertIn("research_api_failure", benchmark.MODES)
        for value in (None, {}, {"api": []}, {"benchmark_evidence": []}, {"stages": 1}):
            with self.subTest(value=value):
                self.assertTrue(benchmark._research_contract(value, 90))


if __name__ == "__main__":
    unittest.main()
