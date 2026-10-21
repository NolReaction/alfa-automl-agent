"""Bounded transport recovery and integrity-checked accepted CSV recovery."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from agent.research_v3 import (
    MAX_COMPLETION_TOKENS, V3State, _request_completion, _retryable_api_error,
    prepare_research, run_agent_v3,
)
from tests.test_research import FakeClient, existing_output, make_regression, response
from tests.test_team_integration import fixture_baseline


SECRET = "resilience-test-private-key"


def connection_error():
    return APIConnectionError(request=httpx.Request("POST", "https://offline.invalid/v1"))


def status_error(status, body=None):
    request = httpx.Request("POST", "https://offline.invalid/v1")
    return APIStatusError(SECRET, response=httpx.Response(status, request=request), body=body)


class ObservedClient(FakeClient):
    def __init__(self, replies, *, after_call=None):
        super().__init__(replies)
        self.timeouts = []
        self.max_tokens = []
        self.after_call = after_call

    def with_options(self, **kwargs):
        self.timeouts.append(kwargs["timeout"])
        return self

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        try:
            return super().create(**kwargs)
        finally:
            if self.after_call is not None:
                self.after_call()


class ApiRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        task = make_regression(Path(temporary.name))
        self.state = V3State(task.resolve(), 5, 90)
        prepare_research(self.state)
        fixture_baseline(self.state)
        self.state.transition("inner_research")
        self.messages = [{"role": "user", "content": "Use only the inner research view."}]

    def test_output_allowance_is_charged_by_guard_and_matches_actual_request(self):
        payload = json.dumps(self.state.clean(self.messages), ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        self.assertIsNone(self.state.request_allowed(self.messages))
        guard = self.state.report["request_guard"]
        self.assertEqual(guard["estimated_tokens_reserved"],
                         math.ceil(len(payload) / 3) + MAX_COMPLETION_TOKENS)
        self.assertIn(str(MAX_COMPLETION_TOKENS), guard["estimation"])
        guard["estimated_tokens_reserved"] = 0
        client = ObservedClient([response()])
        self.assertIsNone(_request_completion(self.state, client, self.messages)[1])
        self.assertEqual(client.max_tokens, [MAX_COMPLETION_TOKENS])
        self.assertGreaterEqual(guard["estimated_tokens_reserved"], MAX_COMPLETION_TOKENS)

    def test_transport_failure_then_success_counts_both_requests_under_one_deadline(self):
        client = ObservedClient([connection_error(), response(content="Recovered")])
        deadline = self.state.budget.deadline
        completion, stopped = _request_completion(self.state, client, self.messages)
        self.assertIsNone(stopped)
        self.assertEqual(completion["content"], "Recovered")
        self.assertEqual(self.state.report["api"]["attempts"], 2)
        self.assertEqual(self.state.report["api"]["responses"], 1)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 1)
        self.assertEqual(self.state.budget.deadline, deadline)
        self.assertTrue(all(0 < value <= 90 for value in client.timeouts))
        kinds = [event["kind"] for event in self.state.report["events"]]
        self.assertEqual(kinds.count("api_error"), 1)
        self.assertIn("api_retry_recovered", kinds)
        self.assertTrue(all("api_error" not in json.dumps(item) for item in client.received_messages))

    def test_two_transport_failures_end_requests_for_local_fallback(self):
        client = ObservedClient([connection_error(), connection_error(), response()])
        completion, stopped = _request_completion(self.state, client, self.messages)
        self.assertIsNone(completion)
        self.assertEqual(stopped, "api_error")
        self.assertEqual(len(client.received_messages), 2)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 1)
        self.assertEqual(sum(event["kind"] == "api_error" for event in self.state.report["events"]), 2)

    def test_retry_limit_is_global_across_subsequent_research_steps(self):
        client = ObservedClient([connection_error(), response(), connection_error(), response()])
        self.assertIsNone(_request_completion(self.state, client, self.messages)[1])
        self.assertEqual(_request_completion(self.state, client, self.messages)[1], "api_error")
        self.assertEqual(len(client.received_messages), 3)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 1)

    def test_request_limit_one_does_not_make_a_retry(self):
        self.state.report["max_steps"] = 1
        client = ObservedClient([connection_error(), response()])
        self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
        self.assertEqual(len(client.received_messages), 1)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)
        self.assertEqual(self.state.report["events"][-1], {"kind": "api_retry_skipped", "reason": "step_limit"})

    def test_retry_does_not_spend_the_reserved_finalization_time(self):
        deadline = self.state.budget.deadline

        def use_spare_time():
            self.state.budget._clock = lambda: deadline - self.state.budget.minimum_final_reserve + .1

        client = ObservedClient([connection_error(), response()], after_call=use_spare_time)
        self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
        self.assertEqual(len(client.received_messages), 1)
        self.assertEqual(self.state.report["events"][-1]["reason"], "time_limit")
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)
        self.assertEqual(self.state.budget.deadline, deadline)

    def test_retry_obeys_the_context_budget_again(self):
        def exhaust_context():
            guard = self.state.report["request_guard"]
            guard["estimated_tokens_reserved"] = guard["estimated_token_limit"]

        client = ObservedClient([connection_error(), response()], after_call=exhaust_context)
        self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
        self.assertEqual(len(client.received_messages), 1)
        self.assertEqual(self.state.report["events"][-1]["reason"], "context_budget")
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)

    def test_timeout_expiring_after_guard_does_not_start_a_request(self):
        client = ObservedClient([response()])
        with patch.object(self.state, "budget_context", return_value={}), \
                patch.object(self.state, "request_allowed", return_value=None), \
                patch.object(self.state, "research_timeout", return_value=0):
            self.assertEqual(_request_completion(self.state, client, self.messages), (None, "time_limit"))
        self.assertEqual(client.received_messages, [])
        self.assertEqual(self.state.report["api"]["attempts"], 0)

    def test_malformed_paid_response_is_not_retried(self):
        client = ObservedClient([SimpleNamespace(choices=[]), response()])
        self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
        self.assertEqual(len(client.received_messages), 1)
        self.assertEqual(self.state.report["api"]["responses"], 1)
        self.assertEqual(self.state.report["api_recovery"]["retries"], 0)

    def test_retryable_types_are_typed_and_quota_or_account_errors_are_excluded(self):
        retryable = [connection_error(), APITimeoutError(request=httpx.Request("POST", "https://offline.invalid")),
                     *(status_error(status) for status in (429, 500, 502, 503, 504))]
        for error in retryable:
            with self.subTest(type=type(error).__name__, status=getattr(error, "status_code", None)):
                self.assertTrue(_retryable_api_error(error))
        rejected = [RuntimeError("Transient 503; retry please"), OSError("network"),
                    *(status_error(status) for status in (400, 401, 402, 403, 404, 422)),
                    status_error(429, {"code": "insufficient_quota"}),
                    status_error(429, {"error": {"type": "billing_not_active"}}),
                    status_error(503, {"type": "budget_exceeded"}),
                    status_error(429, {"error": {"code": "insufficient_balance"}})]
        for error in rejected:
            with self.subTest(type=type(error).__name__, body=getattr(error, "body", None)):
                self.assertFalse(_retryable_api_error(error))

    def test_auth_or_quota_does_not_trigger_an_actual_retry_or_expose_provider_text(self):
        for status, body in ((401, None), (403, None), (429, {"code": "insufficient_quota"})):
            with self.subTest(status=status), patch.dict("os.environ", {"OPENAI_API_KEY": SECRET}):
                self.state.report["api"]["attempts"] = 0
                client = ObservedClient([status_error(status, body), response()])
                self.assertEqual(_request_completion(self.state, client, self.messages), (None, "api_error"))
                self.assertEqual(len(client.received_messages), 1)
                self.assertEqual(self.state.report["api_recovery"]["retries"], 0)
                self.assertNotIn(SECRET, json.dumps(self.state.report))


class AcceptedCheckpointRecoveryTests(unittest.TestCase):
    def invoke(self, mutation=None, *, accepted=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        task = make_regression(root)
        original = existing_output(root) if accepted else None
        observed = []

        def fail_after_preparation(state):
            fixture_baseline(state)
            observed.append(state)
            if mutation is not None:
                mutation(state)
            if accepted:
                (root / "submission.csv").write_text("incomplete replacement", encoding="utf-8")
            raise RuntimeError("Injected unexpected failure " + SECRET)

        with patch.dict("os.environ", {"OPENAI_API_KEY": SECRET}), \
                patch("agent.research_v3.prepare_baseline", side_effect=fail_after_preparation), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                run_dir = run_agent_v3(task, FakeClient([]), max_steps=1, time_budget=90)
                error = None
            except RuntimeError as caught:
                run_dir, error = observed[0].run_dir, caught
        report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertNotIn(SECRET, json.dumps(report))
        return root, original, report, error

    def test_unexpected_error_restores_exact_accepted_bytes_and_reports_fallback(self):
        root, original, report, error = self.invoke()
        self.assertIsNone(error)
        self.assertEqual((root / "submission.csv").read_bytes(), original)
        self.assertEqual(report["status"], "fallback")
        self.assertEqual(report["reason"], "unexpected_error")
        self.assertTrue(report["output_check"]["valid"])
        self.assertEqual(report["recovery"]["status"], "restored")
        self.assertFalse(report["publication"]["new_result_published"])
        self.assertEqual(report["api"]["attempts"], 0)
        self.assertEqual(report["stage"], "result")
        self.assertNotEqual(report["stages"][0]["status"], "completed")

    def test_no_accepted_checkpoint_still_fails(self):
        root, _, report, error = self.invoke(accepted=False)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(report["status"], "failed")
        self.assertFalse((root / "submission.csv").exists())

    def test_new_schema_valid_backup_file_is_not_an_accepted_checkpoint(self):
        def unaccepted(state):
            state.backup.write_text("row_key,outcome\n0005,999\nNA,999\n", encoding="utf-8")
            self.assertFalse(state.recover())

        root, _, report, error = self.invoke(unaccepted, accepted=False)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(report["status"], "failed")
        self.assertFalse((root / "submission.csv").exists())

    def test_original_task_or_data_mutation_is_fatal_despite_accepted_backup(self):
        mutations = [
            lambda state: state.task_path.write_text("changed task", encoding="utf-8"),
            lambda state: (state.gate.source_dir / "train.csv").write_text("changed train", encoding="utf-8"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                _, _, report, error = self.invoke(mutation)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["recovery"]["status"], "rejected")

    def test_allowed_research_input_mutation_is_fatal(self):
        _, _, report, error = self.invoke(lambda state: (
            state.gate.work_data / "train.csv").write_text("changed inner inputs", encoding="utf-8"))
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(report["recovery"]["status"], "rejected")

    def test_frozen_baseline_code_or_recipe_mutation_is_fatal(self):
        for name in ("code_path", "recipe_path"):
            with self.subTest(name=name):
                _, _, report, error = self.invoke(lambda state: getattr(state.baseline, name).write_text(
                    "changed frozen artifact", encoding="utf-8"))
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["recovery"]["status"], "rejected")

    def test_schema_valid_but_changed_backup_is_not_recovered(self):
        def tamper(state):
            state.backup.write_text("row_key,outcome\n0005,999\nNA,999\n", encoding="utf-8")

        _, _, report, error = self.invoke(tamper)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(report["status"], "failed")
        self.assertIn("checkpoint bytes changed", report["recovery"]["error"])


if __name__ == "__main__":
    unittest.main()
