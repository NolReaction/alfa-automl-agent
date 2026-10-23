"""Round-4 protocol risks through the unchanged host; no live provider calls.

These are tiny protocol fixtures, not a comparison of full-main ML quality.
Baseline search is replaced by its existing cheap fixture, while candidate
processes, metrics, freeze/admission, refits and CSV checks remain real.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from agent.gateways import ALTERNATE_BASE_URL
from agent.research_candidate import CandidateExecutionError, run_candidate
from agent.research_v3 import V3State, _request_completion, prepare_research
from agent.search import seed_candidate
from experiment_research_smoke import summarize_run
from tests.test_research import make_regression, response, write_csv
from tests.test_research_candidate import data, frozen, make_spec
from tests import test_team_integration as lifecycle
from tests.test_team_integration import CONSTANT_CODE, candidate_tool, fixture_baseline
from validation import validate_predictions


class Round4HostRisks(unittest.TestCase):
    def test_bad_literal_labels_and_probabilities_are_rejected_by_real_child(self):
        cases = (
            ("Macro F1", "{'predictions': ['unrecognized', '01']}"),
            ("LogLoss", "{'predictions': [[0.9, 0.9], [0.5, 0.5]], 'classes': ['01', 'NA']}"),
        )
        for metric, expression in cases:
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fit, predict = data(classification=True)
                source = "def fit_predict(X_fit, y_fit, X_predict, context):\n    return " + expression + "\n"
                candidate = frozen(root, source, {"family": "invalid_output_fixture"})
                identity = (candidate.code_hash, candidate.recipe_hash)
                with self.assertRaises(CandidateExecutionError) as failure:
                    run_candidate(candidate, spec=make_spec(root, metric), fit_frame=fit,
                                  predict_frame=predict, actuals=["01", "NA"], phase="inner_selection",
                                  output_dir=root / "executions", timeout=20)
                self.assertNotEqual(failure.exception.execution.get("exit_code"), 0)
                self.assertFalse(list((root / "executions").glob("*/receipt.json")))
                candidate.verify()
                self.assertEqual((candidate.code_hash, candidate.recipe_hash), identity)

    def test_inner_success_outer_loser_publishes_baseline_without_loser_refit(self):
        # This fixed synthetic distribution shift is chosen before any fit.
        # The candidate learns the inner linear pattern from supplied arguments;
        # no phase branching or outer targets are exposed to generated code.
        split = {"fit_rows": list(range(80)), "validation_rows": list(range(80, 100)),
                 "choice": {"strategy": "random"}, "split": {"strategy": "random"}}
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            write_csv(task.parent / "regression/participants/train.csv",
                      [["row_key", "measurement", "outcome"]] + [
                          [f"train-{index:03d}", index, 3 * index + 2 if index < 80 else 120.5]
                          for index in range(100)])
            run_dir, report, client, executions, _ = lifecycle.TeamIntegrationTests().run_v3(
                task, [response(code=candidate_tool()), response()], frozen_split=split)
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["evaluations"]["final_check"]["selected"], "baseline")
            self.assertEqual(report["evaluations"]["final_check"]["candidate_acceptance"], "not_better")
            self.assertEqual(report["publication"]["recipe"], "baseline")
            identity = report["frozen_candidate"]
            phases = [item["phase"] for item in executions if
                      all(item[key] == identity[key] for key in ("code_hash", "recipe_hash"))]
            self.assertIn("inner_selection", phases)
            self.assertIn("outer_validation", phases)
            self.assertNotIn("full_refit", phases)
            summary = summarize_run(task, report, run_dir=run_dir)
            self.assertTrue(summary["coverage"]["inner_trial_verified"])
            self.assertTrue(summary["coverage"]["finalization_verified"])
            self.assertFalse(summary["coverage"]["candidate_published"])
            self.assertFalse(summary["coverage"]["qwen_full_refit_verified"])
            self.assertEqual(summary["coverage"]["published_output_origin"], "baseline")
            self.assertTrue(validate_predictions(task.parent)["valid"])
            self.assertEqual(len(client.received_messages), 2)
            # Real lifecycle helper also asserts there are no post-outer API calls.

    def test_native_local_winner_after_llm_trial_keeps_local_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.local_candidate_plan", return_value=[seed_candidate(False)]):
                run_dir, report, client, executions, _ = lifecycle.TeamIntegrationTests().run_v3(
                    task, [response(code=candidate_tool(CONSTANT_CODE, {"family": "llm_mean"}))])
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["local_successful_experiments"], 1)
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["publication"]["recipe"], "candidate")
            llm = next(event["result"]["selection"] for event in report["events"]
                       if event["kind"] == "experiment")
            local = next(event["result"] for event in report["events"] if event["kind"] == "local_experiment")
            self.assertTrue(llm["verified"])
            self.assertTrue(local["verified"])
            self.assertEqual(local["origin"], "local")
            for key in ("code_hash", "recipe_hash"):
                self.assertEqual(report["frozen_candidate"][key], local[key])
            self.assertNotEqual(llm["code_hash"], local["code_hash"])
            summary = summarize_run(task, report, run_dir=run_dir)
            coverage = summary["coverage"]
            self.assertTrue(coverage["inner_trial_verified"])
            self.assertTrue(coverage["finalization_verified"])
            self.assertEqual(coverage["published_output_origin"], "local")
            self.assertFalse(coverage["candidate_published"])
            self.assertFalse(coverage["qwen_full_refit_verified"])
            local_phases = [item["phase"] for item in executions if item["code_hash"] == local["code_hash"]]
            self.assertIn("full_refit", local_phases)

    def test_http_504_cannot_retry_after_original_research_time_is_exhausted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            state = V3State(task.resolve(), 4, 90)
            prepare_research(state)
            fixture_baseline(state)
            state.transition("inner_research")
            deadline = state.budget.deadline
            calls = []

            def handler(request):
                calls.append(request)
                state.budget._clock = lambda: deadline - state.budget.minimum_final_reserve + .1
                return httpx.Response(504, request=request, json={"error": {"message": "offline fixture gateway timeout"}})

            with OpenAI(api_key="offline-fixture", base_url=ALTERNATE_BASE_URL, max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
                result = _request_completion(state, sdk, [{"role": "user", "content": "INNER-only fixture"}])
            self.assertEqual(result, (None, "api_error"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(state.report["api"]["attempts"], 1)
            self.assertEqual(state.report["api"]["responses"], 0)
            self.assertEqual(state.report["api_recovery"]["retries"], 0)
            self.assertEqual(state.budget.deadline, deadline)
            self.assertEqual(state.report["events"][-1], {"kind": "api_retry_skipped", "reason": "time_limit"})
            error = next(event for event in state.report["events"] if event["kind"] == "api_error")
            self.assertEqual(error["details"]["error_type"], "InternalServerError")
            self.assertEqual(error["details"]["status_code"], 504)
            self.assertGreater(state.report["request_guard"]["estimated_tokens_reserved"], 8192)

    def test_truncated_tool_is_not_executed_and_baseline_receipt_matches_csv(self):
        truncated = response(code="raise AssertionError('truncated response executed')")
        truncated.choices[0].finish_reason = "length"
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.local_candidate_plan", return_value=[]):
                run_dir, report, client, executions, _ = lifecycle.TeamIntegrationTests().run_v3(task, [truncated])
            self.assertEqual(report["experiments"], 0)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["reason"], "truncated_response")
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["publication"]["recipe"], "baseline")
            self.assertTrue(report["publication"]["new_result_published"])
            self.assertIsNone(report["frozen_candidate"])
            # The host legitimately measures its fixed baseline on INNER before
            # the dialogue. No generated recipe may execute after truncation.
            baseline = report["frozen_baseline"]
            self.assertTrue(all(all(item[key] == baseline[key]
                                    for key in ("code_hash", "recipe_hash")) for item in executions))
            receipt = json.loads(Path(report["candidate_receipts"]["refit"]).read_text())
            self.assertEqual(receipt["output_hash"], hashlib.sha256((task.parent / "submission.csv").read_bytes()).hexdigest())
            summary = summarize_run(task, report, run_dir=run_dir)
            self.assertTrue(summary["coverage"]["finalization_verified"])
            self.assertFalse(summary["coverage"]["inner_trial_verified"])
            self.assertEqual(summary["coverage"]["published_output_origin"], "baseline")


if __name__ == "__main__":
    unittest.main()
