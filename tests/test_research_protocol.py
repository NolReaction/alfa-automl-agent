"""Retain v2 SDK protocol coverage; default v3 has dedicated integration tests."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import solution
from agent.llm import DEFAULT_MODEL
from agent.research import run_agent_v2
from tests.test_research import FakeClient, known_start, make_regression, response as fake_response
from validation import validate_predictions


HAS_PACKAGES = all(importlib.util.find_spec(name) is not None
                   for name in ("openai", "pandas", "sklearn"))
EXPECTED_OUTPUT = b"row_key,outcome\r\n0005,17.0000\r\nNA,20.0000\r\n"


def completion(message=None, *, finish="stop", usage=None, choices=None):
    body = {
        "id": "chatcmpl-protocol", "object": "chat.completion", "created": 1,
        "model": "provided-test-model", "choices": choices if choices is not None else [{
            "index": 0, "message": message if message is not None else {
                "role": "assistant", "content": "Finished.",
            }, "finish_reason": finish,
        }],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def tool_call(identifier="call-protocol", arguments=None):
    return {"id": identifier, "type": "function", "function": {
        "name": "run_python", "arguments": arguments,
    }}


def tool_completion(calls):
    return completion({"role": "assistant", "content": None, "tool_calls": calls},
                      finish="tool_calls")


@unittest.skipUnless(HAS_PACKAGES, "Install requirements.txt for SDK protocol tests")
class ResearchProtocolTests(unittest.TestCase):
    def run_case(self, replies, *, starter=True, max_steps=2, fake_client=None, on_request=None,
                 model="provided-test-model", response_status=200, emergency=True):
        """Exercise the real SDK parser; only startup training and tools are mocked."""
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                if on_request is not None:
                    on_request(task_dir)
                reply = replies[len(requests) - 1] if len(requests) <= len(replies) else completion()
                body = json.dumps(reply).encode("utf-8")
                self.send_response(response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-Id", "req-local-protocol")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory)
                task = make_regression(task_dir)
                env = {name: value for name, value in os.environ.items()
                       if not name.lower().endswith("_proxy")}
                env.update({
                    "OPENAI_API_KEY": "local-protocol-placeholder",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                })
                env.pop("OPENAI_MODEL", None)
                if model is not None:
                    env["OPENAI_MODEL"] = model
                output, errors = io.StringIO(), io.StringIO()
                with patch.dict(os.environ, env, clear=True), \
                        patch("solution.run_agent", side_effect=run_agent_v2), \
                        patch("sys.argv", ["solution.py", str(task), "--max-steps", str(max_steps),
                                           "--time-budget", "120"]), \
                        patch("agent.research.prepare_starting_output", side_effect=known_start if starter else None), \
                        patch("agent.research.run_python", side_effect=AssertionError("Invalid reply executed code")) as execute, \
                        (nullcontext() if emergency else patch("solution._prepare_emergency", return_value=None)), \
                        (patch("solution.make_client", return_value=nullcontext(fake_client))
                         if fake_client is not None else nullcontext()), \
                        redirect_stdout(output), redirect_stderr(errors):
                    status = solution.main()
                reports = list((task_dir / ".agent_runs").glob("*/run.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                result = task_dir / "submission.csv"
                emergency_reports = [json.loads(path.read_text(encoding="utf-8"))
                                     for path in (task_dir / ".agent_runs").glob("*/emergency.json")]
                return {
                    "status": status, "report": report, "requests": requests if fake_client is None else [
                        {"messages": messages} for messages in fake_client.received_messages],
                    "output": result.read_bytes() if result.exists() else None,
                    "stdout": output.getvalue(), "stderr": errors.getvalue(),
                    "executions": execute.call_count,
                    "output_valid": validate_predictions(task_dir)["valid"],
                    "emergency_reports": emergency_reports,
                }
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def assert_preserved_fallback(self, result, reason):
        self.assertEqual(result["status"], 0, result["stderr"])
        self.assertEqual(result["output"], EXPECTED_OUTPUT)
        self.assertEqual((result["report"]["status"], result["report"]["reason"]),
                         ("fallback", reason))
        self.assertEqual(result["executions"], 0)
        self.assertEqual(result["report"]["best_metric"]["value"], 10)

    def assert_emergency_prior_fallback(self, result, reason):
        self.assertEqual(result["status"], 0, result["stderr"])
        self.assertTrue(result["output_valid"])
        self.assertEqual((result["report"]["status"], result["report"]["reason"]),
                         ("fallback", reason))
        self.assertIsNone(result["report"]["best_metric"])
        self.assertEqual(result["report"]["successful_experiments"], 0)
        self.assertEqual(result["executions"], 0)
        prior, = result["emergency_reports"]
        self.assertEqual(prior["kind"], "target_prior")
        self.assertFalse(prior["quality_verified"])
        self.assertEqual(prior["model"]["strategy"], "mean")
        self.assertEqual(prior["model"]["fit_rows"], 100)
        self.assertEqual(prior["predict_rows"], 2)
        rows = list(csv.reader(io.StringIO(result["output"].decode("utf-8"))))
        self.assertEqual(rows[0], ["row_key", "outcome"])
        self.assertEqual([row[0] for row in rows[1:]], ["0005", "NA"])
        self.assertEqual([float(row[1]) for row in rows[1:]], [150.5, 150.5])

    def test_required_environment_alone_uses_default_model_in_request_and_report(self):
        for model in (None, "   ", "provided-test-model"):
            with self.subTest(model=model):
                result = self.run_case([completion()], max_steps=1, model=model)
                self.assert_preserved_fallback(result, "step_limit")
                expected = model if model and model.strip() else DEFAULT_MODEL
                self.assertEqual(result["requests"][0]["model"], expected)
                self.assertEqual(result["report"]["model"], expected)

    def test_rate_limit_has_no_retries_or_provider_body_in_output_and_report(self):
        reply = {"error": {"code": "insufficient_quota", "message":
                          "Authorization: Bearer local-protocol-placeholder; private-body-marker"}}
        for starter, emergency in ((True, True), (False, True), (False, False)):
            with self.subTest(starter=starter, emergency=emergency):
                result = self.run_case([reply], starter=starter, emergency=emergency,
                                       response_status=429, model=None)
                self.assertEqual(len(result["requests"]), 1)
                self.assertEqual(result["report"]["api"]["attempts"], 1)
                self.assertEqual(result["report"]["api"]["responses"], 0)
                if starter:
                    self.assert_preserved_fallback(result, "api_error")
                elif emergency:
                    self.assert_emergency_prior_fallback(result, "api_error")
                else:
                    self.assertEqual(result["status"], 1)
                    self.assertEqual(result["report"]["status"], "failed")
                    self.assertIsNone(result["output"])
                    self.assertEqual(result["emergency_reports"], [])
                evidence = (json.dumps(result["report"]) + json.dumps(result["emergency_reports"])
                            + result["stdout"] + result["stderr"])
                self.assertIn("RateLimitError", evidence)
                self.assertIn("HTTP 429", evidence)
                self.assertIn("req-local-protocol", evidence)
                self.assertIn("insufficient_quota", evidence)
                for forbidden in ("Authorization", "local-protocol-placeholder", "private-body-marker"):
                    self.assertNotIn(forbidden, evidence)

    def test_empty_choices_and_null_message_return_success_with_retained_csv(self):
        malformed = {
            "empty_choices": completion(choices=[]),
            "null_message": completion(choices=[{"index": 0, "message": None, "finish_reason": "stop"}]),
        }
        for name, reply in malformed.items():
            with self.subTest(reply=name):
                result = self.run_case([reply])
                self.assert_preserved_fallback(result, "api_error")
                self.assertEqual(len(result["requests"]), 1)
                self.assertTrue(any(event["kind"] == "api_error" for event in result["report"]["events"]))

    def test_nonstring_arguments_receive_tool_feedback_without_execution(self):
        for arguments in (None, [], {"code": "print('must not run')"}):
            with self.subTest(arguments=arguments):
                result = self.run_case([tool_completion([tool_call(arguments=arguments)]), completion()])
                self.assert_preserved_fallback(result, "step_limit")
                self.assertEqual(len(result["requests"]), 2)
                messages = result["requests"][1]["messages"]
                feedback = [item for item in messages if item["role"] == "tool"]
                self.assertEqual(len(feedback), 1)
                self.assertEqual(feedback[0]["tool_call_id"], "call-protocol")
                self.assertIn("error", json.loads(feedback[0]["content"]))
                replay = next(item for item in messages if item.get("tool_calls"))
                self.assertIsInstance(replay["tool_calls"][0]["function"]["arguments"], str)

    def test_unknown_or_inconsistent_finish_reason_never_executes_code(self):
        message = {"role": "assistant", "content": None,
                   "tool_calls": [tool_call(arguments=json.dumps({"code": "print('must not run')"}))]}
        for finish in ("content_filter", "unknown", "stop"):
            with self.subTest(finish=finish):
                result = self.run_case([completion(message, finish=finish)])
                self.assert_preserved_fallback(result, "api_error")
        result = self.run_case([completion(finish="tool_calls")])
        self.assert_preserved_fallback(result, "api_error")

    def test_invalid_call_id_rejects_the_whole_envelope_before_any_execution(self):
        valid = tool_call("valid-first", json.dumps({"code": "print('must not run')"}))
        for identifier in (None, ""):
            with self.subTest(identifier=identifier):
                bad = tool_call(identifier, json.dumps({"code": "print('also must not run')"}))
                result = self.run_case([tool_completion([valid, bad])])
                self.assert_preserved_fallback(result, "api_error")
                self.assertEqual(len(result["requests"]), 1)

    def test_invalid_or_incomplete_usage_is_ignored_and_next_reply_is_processed(self):
        valid = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        invalid = {
            "string": {"prompt_tokens": "not-a-count", "completion_tokens": 3, "total_tokens": 5},
            "negative": {"prompt_tokens": -2, "completion_tokens": 3, "total_tokens": 5},
            "missing_field": {"prompt_tokens": 2, "completion_tokens": 3},
            "empty": {},
        }
        for name, usage in invalid.items():
            with self.subTest(usage=name):
                result = self.run_case([completion(usage=usage), completion(usage=valid)])
                self.assert_preserved_fallback(result, "step_limit")
                self.assertEqual(len(result["requests"]), 2)
                self.assertEqual(result["report"]["api"], {
                    "attempts": 2, "responses": 2, "usage_reported_responses": 1, **valid,
                })

    def test_boolean_usage_in_provider_object_is_not_treated_as_one_token(self):
        # The real SDK coerces JSON true to int(1). A fake provider preserves a
        # Python bool so this boundary check exercises the object we actually see.
        valid = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        fake = FakeClient([
            fake_response(usage={"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 5}),
            fake_response(usage=valid),
        ])
        result = self.run_case([], fake_client=fake)
        self.assert_preserved_fallback(result, "step_limit")
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(result["report"]["api"], {
            "attempts": 2, "responses": 2, "usage_reported_responses": 1, **valid,
        })

    def test_changed_source_inputs_cannot_be_hidden_by_malformed_api_fallback(self):
        def mutate_source(task_dir):
            (task_dir / "regression/participants/train.csv").write_text("changed", encoding="utf-8")

        result = self.run_case([completion(choices=[])], on_request=mutate_source)
        self.assertEqual(result["status"], 1)
        self.assertEqual(result["report"]["status"], "failed")
        self.assertEqual(result["output"], EXPECTED_OUTPUT)
        self.assertEqual(result["executions"], 0)
        self.assertIn("Original task inputs changed", result["stderr"])

    def test_bad_envelope_without_accepted_predictions_still_returns_nonzero(self):
        result = self.run_case([completion(choices=[])], starter=False, emergency=False)
        self.assertEqual(result["status"], 1)
        self.assertEqual(result["report"]["status"], "failed")
        self.assertIsNone(result["output"])
        self.assertEqual(result["executions"], 0)
        self.assertEqual(len(result["requests"]), 1)
        self.assertIn("Error:", result["stderr"])

    def test_bad_envelope_with_only_a_fresh_emergency_prior_returns_honest_fallback(self):
        result = self.run_case([completion(choices=[])], starter=False)
        self.assert_emergency_prior_fallback(result, "api_error")
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["report"]["api"]["attempts"], 1)
        self.assertEqual(result["report"]["api"]["responses"], 1)
        self.assertTrue(any(event["kind"] == "api_error" for event in result["report"]["events"]))


if __name__ == "__main__":
    unittest.main()
