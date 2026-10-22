"""No-network regressions for the shared transport policy and public errors."""

from contextlib import redirect_stdout
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError, OpenAI

from agent import llm
from agent.gateways import (
    ALTERNATE_BASE_URL, PRIMARY_BASE_URL, configured_gateways,
    request_options, retryable_api_error,
)


SECRET = "not-the-configured-key-private-marker"


class RescueTransportTests(unittest.TestCase):
    def client(self, handler, base_url=ALTERNATE_BASE_URL):
        client = OpenAI(api_key="fake-configured-key", base_url=base_url,
                        max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    @staticmethod
    def completion(request):
        return httpx.Response(200, request=request, json={
            "id": "mock", "object": "chat.completion", "created": 0,
            "model": llm.DEFAULT_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
        })

    @staticmethod
    def state():
        report = {"max_steps": 6,
                  "api": {"attempts": 0, "responses": 0, "usage_reported_responses": 0,
                          "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                  "api_recovery": {"maximum_retries": 1, "retries": 0}, "events": []}
        state = SimpleNamespace(report=report)
        state.budget_context = lambda: {"remaining_seconds": 90}
        state.request_allowed = lambda messages: None
        state.research_timeout = lambda maximum=1800: min(90, maximum)
        state.clean = lambda value: value
        state.save = lambda: None
        state.event = lambda kind, **fields: report["events"].append({"kind": kind, **fields})
        return state

    def test_typed_transport_and_server_errors_retry_once_through_real_sdk(self):
        from agent.research_v3 import _request_completion

        for failure in (httpx.ConnectTimeout, httpx.ReadTimeout, 429, 500, 502, 503, 504):
            with self.subTest(failure=failure):
                requests = []

                def handler(request):
                    requests.append(request)
                    if len(requests) == 1:
                        if isinstance(failure, type):
                            raise failure(SECRET, request=request)
                        return httpx.Response(failure, request=request, json={
                            "error": {"code": "rate_limit_exceeded" if failure == 429 else "server_error",
                                      "message": SECRET}})
                    return self.completion(request)

                state = self.state()
                result, reason = _request_completion(state, self.client(handler), [])
                self.assertIsNone(reason)
                self.assertEqual(result["content"], "OK")
                self.assertEqual([r.url.host for r in requests], [
                    "api.alfahack-gateway26.ru", "changellenge.up.railway.app"])
                self.assertEqual(state.report["api"]["attempts"], 2)
                self.assertEqual(state.report["api"]["responses"], 1)
                self.assertEqual(state.report["api_recovery"]["retries"], 1)
                # Missing usage stays unknown: the reservation is independent,
                # and no response is falsely counted as having measured usage.
                self.assertEqual(state.report["api"]["usage_reported_responses"], 0)
                for request in requests:
                    self.assertEqual(request.extensions["timeout"]["connect"], 10)
                    self.assertEqual(request.extensions["timeout"]["read"], 90)
                self.assertNotIn(SECRET, json.dumps(state.report))

    def test_auth_permissions_schema_payment_and_quota_do_not_failover(self):
        from agent.research_v3 import _request_completion

        cases = [(status, {"code": "unsupported_schema"})
                 for status in (400, 401, 402, 403, 404, 422)]
        cases += [(429, {"code": code}) for code in (
            "insufficient_quota", "quota_exceeded", "daily_budget_exceeded",
            "insufficient_balance", "billing_not_active", "payment_required")]
        cases += [(503, {"type": "budget_exceeded"})]
        for status, fields in cases:
            with self.subTest(status=status, fields=fields):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(status, request=request,
                                          json={"error": {**fields, "message": SECRET}})

                state = self.state()
                self.assertEqual(_request_completion(state, self.client(handler), []),
                                 (None, "api_error"))
                self.assertEqual(len(requests), 1)
                self.assertEqual(state.report["api_recovery"]["retries"], 0)
                self.assertEqual(state.report["api_gateway"]["switches"], 0)
                self.assertNotIn(SECRET, json.dumps(state.report))

    def test_custom_credentials_stay_on_custom_host_for_both_paths(self):
        from agent.research_v3 import _request_completion

        endpoint = "https://custom.private.invalid/v1"
        for path in ("check", "research"):
            with self.subTest(path=path):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(504, request=request,
                                          json={"error": {"message": SECRET}})

                client = self.client(handler, endpoint)
                if path == "check":
                    with patch.object(llm, "make_client", return_value=client), \
                            redirect_stdout(io.StringIO()), self.assertRaises(Exception):
                        llm.check_api()
                else:
                    self.assertEqual(_request_completion(self.state(), client, []),
                                     (None, "api_error"))
                self.assertEqual(len(requests), 1 if path == "check" else 2)
                self.assertTrue(all(r.url.host == "custom.private.invalid" for r in requests))
                self.assertTrue(all(r.headers["authorization"] == "Bearer fake-configured-key"
                                    for r in requests))
                self.assertEqual(configured_gateways(endpoint), (("custom", endpoint),))

    def test_retry_policy_is_typed_and_never_parses_exception_messages(self):
        self.assertFalse(retryable_api_error(RuntimeError("HTTP 504 retry " + SECRET)))
        for body in ({"code": "insufficient_quota"},
                     {"error": {"type": "billing_not_active"}},
                     {"error": {"code": "insufficient_balance"}}):
            error = APIStatusError(SECRET, body=body, response=httpx.Response(
                429, request=httpx.Request("POST", "https://offline.invalid")))
            self.assertFalse(retryable_api_error(error))
        options = request_options(3.5, ("alternate", ALTERNATE_BASE_URL))
        self.assertEqual(options["timeout"].connect, 3.5)
        self.assertEqual(options["timeout"].read, 3.5)
        self.assertEqual(configured_gateways(PRIMARY_BASE_URL)[0][1], ALTERNATE_BASE_URL)


class RescuePublicDiagnosticTests(unittest.TestCase):
    def test_public_messages_drop_unrelated_secrets_even_valid_request_ids(self):
        request = httpx.Request("POST", "https://user:password@private.invalid/v1",
                                headers={"Authorization": "Bearer " + SECRET})
        response = httpx.Response(429, request=request,
                                  headers={"x-request-id": SECRET, "x-private": SECRET})
        error = APIStatusError(SECRET, response=response,
                               body={"code": "insufficient_quota", "message": SECRET})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "different-configured-key"}):
            message = llm.public_error_message(error)
        self.assertEqual(message, "Request failed (APIStatusError, HTTP 429, code=insufficient_quota).")
        for value in (SECRET, "request_id", "password", "Authorization", "private.invalid"):
            self.assertNotIn(value, message)

    def test_public_messages_never_read_arbitrary_exception_strings_or_class_names(self):
        def forbidden_string(self):
            raise AssertionError("Never stringify provider exceptions")

        error_type = type(SECRET, (Exception,), {"__str__": forbidden_string})
        error = error_type(SECRET)
        error.__cause__ = httpx.ReadTimeout(SECRET)
        self.assertEqual(llm.public_error_message(error),
                         "Request failed (Exception, cause_type=ReadTimeout, transport_reason=read_timeout).")
        self.assertNotIn(SECRET, llm.public_error_message(ValueError(SECRET)))


if __name__ == "__main__":
    unittest.main()
