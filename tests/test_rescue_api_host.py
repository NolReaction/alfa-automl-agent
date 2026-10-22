"""Rescue evidence against the unchanged v3 host; all responses are local mocks.

The small candidate fits use the real process runner, host INNER scoring,
freezing and finalization. They are not live-Qwen evidence.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.research_v3 import V3State, _request_completion, prepare_research, run_agent_v3
from tests.test_research import FakeClient, make_regression, response
from tests.test_team_integration import candidate_tool, fixture_baseline
from validation import validate_predictions


class RescueHostEnvelopeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.task = make_regression(Path(directory.name))
        self.state = V3State(self.task.resolve(), 4, 90)
        # Prepare the permitted data view and a fixed cheap recipe, without
        # fitting. Preserve real request guards and original monotonic deadline.
        prepare_research(self.state)
        fixture_baseline(self.state)
        self.state.transition("inner_research")
        self.messages = [{"role": "user", "content": "inner-only fixture"}]

    def test_missing_usage_is_unknown_and_reservation_is_kept(self):
        client = FakeClient([response(content="OK")])
        completion, stopped = _request_completion(self.state, client, self.messages)
        self.assertIsNone(stopped)
        self.assertEqual(completion["content"], "OK")
        self.assertEqual(self.state.report["api"]["attempts"], 1)
        self.assertEqual(self.state.report["api"]["responses"], 1)
        self.assertEqual(self.state.report["api"]["usage_reported_responses"], 0)
        self.assertGreater(self.state.report["request_guard"]["estimated_tokens_reserved"], 4096)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)
        self.assertEqual(self.state.report["successful_experiments"], 0)

    def test_empty_choices_is_paid_response_without_retry_or_experiment(self):
        client = FakeClient([SimpleNamespace(choices=[]), response(content="unused")])
        self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
        self.assertEqual(len(client.received_messages), 1)
        self.assertEqual(self.state.report["api"]["responses"], 1)
        self.assertEqual(self.state.report["successful_experiments"], 0)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)

    def test_blank_assistant_envelope_is_not_a_verified_experiment(self):
        client = FakeClient([response(content="")])
        completion, stopped = _request_completion(self.state, client, self.messages)
        self.assertIsNone(stopped)
        self.assertEqual(completion["content"], "")
        self.assertEqual(completion["tool_calls"], [])
        self.assertEqual(self.state.report["successful_experiments"], 0)


class RescueHostLifecycleTests(unittest.TestCase):
    def run_host(self, task, replies, *, local_continuation=False):
        client = FakeClient(replies)
        with patch("agent.research_v3.prepare_baseline", side_effect=fixture_baseline), \
                patch("agent.research_v3.continue_locally") as local, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run_dir = run_agent_v3(task, client, max_steps=len(replies), time_budget=90)
        if local_continuation:
            local.assert_called_once()
        else:
            local.assert_not_called()
        report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        return run_dir, report, client

    def test_invalid_json_feedback_then_real_inner_trial_and_frozen_full_refit(self):
        broken = response(code="must never execute")
        broken.choices[0].message.tool_calls[0].function.arguments = '{"code":'
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            run_dir, report, client = self.run_host(task, [broken, response(code=candidate_tool()), response()])
            self.assertEqual(report["api"]["attempts"], 3)
            self.assertEqual(report["api"]["responses"], 3)
            self.assertEqual(report["api"]["usage_reported_responses"], 0)
            self.assertEqual(report["api_recovery"]["retries"], 0)
            self.assertEqual(report["experiments"], 1)
            self.assertEqual(report["successful_experiments"], 1)
            feedback = [item for item in client.received_messages[1] if item["role"] == "tool"]
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback[0]["tool_call_id"], "local-research-call")
            self.assertFalse(json.loads(feedback[0]["content"])["selection"]["verified"])
            verified = [event["result"]["selection"] for event in report["events"]
                        if event["kind"] == "experiment" and event["result"]["selection"].get("verified")]
            self.assertEqual(len(verified), 1)
            self.assertEqual(verified[0]["origin"], "llm")
            frozen = report["frozen_candidate"]
            for key in ("code_hash", "recipe_hash"):
                self.assertEqual(verified[0][key], frozen[key])
                self.assertEqual(report["candidate_receipts"][key], frozen[key])
            outer = json.loads(Path(report["candidate_receipts"]["outer"]).read_text(encoding="utf-8"))
            refit = json.loads(Path(report["candidate_receipts"]["refit"]).read_text(encoding="utf-8"))
            self.assertEqual(outer["phase"], "outer_validation")
            self.assertEqual(refit["phase"], "full_refit")
            for key in ("code_hash", "recipe_hash"):
                self.assertEqual(outer[key], refit[key])
            self.assertEqual(report["publication"]["recipe"], "candidate")
            self.assertTrue(report["publication"]["new_result_published"])
            self.assertTrue(validate_predictions(task.parent)["valid"])
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["reason"], "model_finished")

    def test_length_response_executes_no_tool_and_finalizes_baseline(self):
        truncated = response(code="raise AssertionError('must not execute truncated code')")
        truncated.choices[0].finish_reason = "length"
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            _, report, client = self.run_host(task, [truncated], local_continuation=True)
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["api"]["attempts"], 1)
            self.assertEqual(report["api"]["responses"], 1)
            self.assertEqual(report["experiments"], 0)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["research_stop"]["reason"], "truncated_response")
            self.assertIsNone(report["frozen_candidate"])
            self.assertEqual(report["publication"]["recipe"], "baseline")
            self.assertTrue(validate_predictions(task.parent)["valid"])


if __name__ == "__main__":
    unittest.main()
