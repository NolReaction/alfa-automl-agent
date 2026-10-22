"""Public smoke evidence is provenance-linked and does not export native text."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import experiment_research_smoke as smoke
from agent.gateways import ALTERNATE_BASE_URL, PRIMARY_BASE_URL, client_gateways


METRIC = {"name": "RMSE", "value": 0.1, "higher_is_better": False}
QWEN = {"code_hash": "a" * 64, "recipe_hash": "b" * 64}
LOCAL = {"code_hash": "c" * 64, "recipe_hash": "d" * 64}
BASELINE = {"code_hash": "e" * 64, "recipe_hash": "f" * 64}
OUTPUT = b"row_key,outcome\n0005,12\nNA,4.5\ntest c,7\n"


def trial(identity, origin="llm"):
    feedback = {"verified": True, "scope": "inner_selection", "origin": origin,
                "metric": METRIC, **identity}
    return {"kind": "experiment" if origin == "llm" else "local_experiment",
            "result": {"exit_code": 0, "selection": feedback} if origin == "llm" else feedback}


class RescueSmokeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = smoke.create_tasks(Path(self.temporary.name))["regression"]
        self.run_dir = self.task.parent / ".agent_runs" / "fixture"
        self.report = {"research_protocol_version": 3, "status": "completed", "reason": "model_finished",
            "successful_experiments": 1, "experiments": 1, "frozen_candidate": QWEN,
            "frozen_baseline": BASELINE, "events": [trial(QWEN)],
            "started_at": "2026-10-22T12:00:00+00:00", "elapsed_seconds": 2,
            "api": {"attempts": 1, "responses": 1, "usage_reported_responses": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "evaluations": {"final_check": {"baseline": METRIC, "candidate": METRIC, "selected": "candidate"}},
            "publication": {"new_result_published": True, "recipe": "candidate", "reason": "model_finished"}}
        (self.task.parent / "submission.csv").write_bytes(OUTPUT)

    def receipts(self, identity, recipe="candidate", phases=("outer_validation", "full_refit")):
        for phase in phases:
            directory = "outer_evaluations" if phase == "outer_validation" else "full_refit"
            invocation = self.run_dir / directory / recipe / (phase + "_fixture")
            invocation.mkdir(parents=True, exist_ok=True)
            (invocation / "predictions.csv").write_bytes(OUTPUT)
            (invocation / "receipt.json").write_text(json.dumps({"version": 3, "phase": phase,
                **identity, "output_hash": hashlib.sha256(OUTPUT).hexdigest()}))

    def summary(self):
        return smoke.summarize_run(self.task, self.report, run_dir=self.run_dir)

    def test_qwen_full_path_requires_identity_receipts_and_published_bytes(self):
        self.receipts(QWEN)
        item = self.summary()
        coverage = item["coverage"]
        for name in ("api_response_received", "tool_call_valid", "candidate_executed", "inner_trial_verified",
                     "candidate_frozen", "full_refit_verified", "csv_valid", "candidate_published"):
            self.assertTrue(coverage[name], name)
        self.assertEqual(coverage["published_output_origin"], "llm")
        self.assertTrue(smoke.coverage_summary([item, {**item, "task": "classification"}])["full_candidate_path_passed"])
        (self.task.parent / "submission.csv").write_bytes(OUTPUT.replace(b",12", b",99"))
        self.assertFalse(self.summary()["coverage"]["candidate_published"])

    def test_local_winner_after_qwen_trial_is_not_a_qwen_candidate(self):
        self.report["frozen_candidate"] = LOCAL
        self.report["local_successful_experiments"] = 1
        self.report["events"].append(trial(LOCAL, "local"))
        self.receipts(LOCAL)
        coverage = self.summary()["coverage"]
        self.assertTrue(coverage["inner_trial_verified"])
        self.assertTrue(coverage["candidate_frozen"])
        self.assertTrue(coverage["finalization_verified"])
        self.assertTrue(coverage["full_refit_verified"])
        self.assertEqual(coverage["published_output_origin"], "local")
        self.assertFalse(coverage["qwen_full_refit_verified"])
        self.assertFalse(coverage["generated_candidate_outer_valid"])
        self.assertFalse(coverage["candidate_published"])

    def test_losing_qwen_trial_can_finalize_baseline_without_qwen_refit(self):
        self.report["publication"]["recipe"] = "baseline"
        self.report["evaluations"]["final_check"]["selected"] = "baseline"
        self.report["status"] = "fallback"
        self.report["reason"] = "step_limit"
        self.receipts(QWEN, phases=("outer_validation",))
        self.receipts(BASELINE, recipe="baseline")
        item = self.summary()
        self.assertTrue(item["coverage"]["finalization_verified"])
        self.assertFalse(item["coverage"]["qwen_full_refit_verified"])
        self.assertEqual(item["coverage"]["published_output_origin"], "baseline")
        coverage = smoke.coverage_summary([item, {**item, "task": "classification"}])
        self.assertTrue(coverage["integration_passed"])
        self.assertFalse(coverage["full_candidate_path_passed"])

    def test_missing_origin_evidence_is_not_inferred_from_zero_local_counts(self):
        self.report["events"] = []
        self.receipts(QWEN)
        item = self.summary()
        self.assertTrue(item["coverage"]["inner_trial_verified"])
        self.assertFalse(item["coverage"]["candidate_frozen"])
        self.assertEqual(item["coverage"]["frozen_candidate_origin"], "unknown")
        self.assertFalse(item["coverage"]["candidate_published"])
        self.assertFalse(smoke.coverage_summary([item, {**item, "task": "classification"}])["integration_passed"])

    def test_successful_python_tool_is_not_a_candidate_execution(self):
        self.report.update(successful_experiments=0, events=[{"kind": "experiment", "result": {"exit_code": 0}}])
        coverage = self.summary()["coverage"]
        self.assertTrue(coverage["tool_call_valid"])
        self.assertFalse(coverage["candidate_executed"])
        self.assertFalse(coverage["inner_trial_verified"])

    def test_unknown_usage_is_null_and_public_fields_drop_secrets(self):
        secret = "private-key-provider-body-C:\\private\\dataset.csv"
        self.report.update(status=secret, reason=secret, started_at=secret, elapsed_seconds=secret,
                           frozen_candidate={**QWEN, "recipe": secret}, error=secret)
        self.report["publication"].update(reason=secret, unexpected=secret)
        self.report["evaluations"]["final_check"].update(selected=secret, unbiased_estimate=secret)
        self.report["evaluations"]["selection"] = {"metric": {**METRIC, "name": secret}}
        self.report["api"]["prompt_tokens"] = secret
        self.report["events"].append({"kind": "api_error", "error": secret,
            "details": {"error_type": "InternalServerError", "status_code": 504,
                        "request_id": secret, "body": secret, "authorization": secret}})
        with patch.object(smoke, "validate_predictions", return_value={"valid": False, "errors": [secret]}):
            summary = self.summary()
        self.assertNotIn(secret, json.dumps(summary))
        self.assertEqual(summary["api"]["reported_tokens"],
                         {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})
        self.assertEqual(summary["api"]["responses_without_usage"], 1)
        self.assertEqual(summary["api"]["errors"], [{"error_type": "InternalServerError", "status_code": 504}])
        self.assertIsNone(summary["api"]["currency_cost"])

    def test_budget_wrapper_preserves_official_route_and_clones_shared_budget(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, request=request, json={"choices": []})
        with OpenAI(api_key="test-only", base_url=ALTERNATE_BASE_URL, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
            budget = smoke.RequestBudget(limit=1)
            client = smoke.BudgetClient(sdk, budget)
            self.assertEqual(client_gateways(client)[0][1], ALTERNATE_BASE_URL)
            clone = client.with_options(base_url=PRIMARY_BASE_URL)
            self.assertEqual(str(clone.base_url).rstrip("/"), PRIMARY_BASE_URL)
            clone.chat.completions.create(model="test", messages=[], max_tokens=1)
            with self.assertRaises(RuntimeError):
                client.chat.completions.create(model="test", messages=[], max_tokens=1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].url.host, "changellenge.up.railway.app")
            self.assertEqual(budget.attempts, 1)


if __name__ == "__main__":
    unittest.main()
