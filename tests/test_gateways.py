"""Official host failover uses fake transport; no real requests or credentials."""

from contextlib import redirect_stdout
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from agent.gateways import (
    ALTERNATE_BASE_URL, PRIMARY_BASE_URL, OFFICIAL_GATEWAYS,
    client_gateways, configured_gateways, official_gateway_name,
)
from agent import llm


class GatewayConfigurationTests(unittest.TestCase):
    def test_either_official_address_prefers_the_new_host(self):
        for value in (PRIMARY_BASE_URL, PRIMARY_BASE_URL + "/", ALTERNATE_BASE_URL,
                      "https://API.ALFAHACK-GATEWAY26.RU:443/v1/"):
            with self.subTest(value=value):
                self.assertEqual(configured_gateways(value), OFFICIAL_GATEWAYS)

    def test_custom_and_deceptive_addresses_never_forward_credentials(self):
        values = (
            "http://api.alfahack-gateway26.ru/v1",
            "https://api.alfahack-gateway26.ru.evil.invalid/v1",
            "https://evil.invalid/api.alfahack-gateway26.ru/v1",
            "https://user:secret@api.alfahack-gateway26.ru/v1",
            "https://api.alfahack-gateway26.ru:444/v1",
            "https://api.alfahack-gateway26.ru/v1?token=private",
            "https://api.alfahack-gateway26.ru/v1#private",
            "https://api.alfahack-gateway26.ru/v2",
            "http://127.0.0.1:1/v1",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertIsNone(official_gateway_name(value))
                self.assertEqual(configured_gateways(value), (("custom", value),))

    def test_unknown_clients_and_empty_routes_are_not_stringified(self):
        class Untrusted:
            def __str__(self):
                raise AssertionError("Do not stringify arbitrary endpoint objects")

        client = type("Client", (), {"base_url": Untrusted()})()
        self.assertEqual(client_gateways(client), ())
        self.assertEqual(configured_gateways("  "), ())
        self.assertEqual(configured_gateways(None), ())
        self.assertIsNone(official_gateway_name("https://[invalid/v1"))

    def test_sdk_client_uses_alternative_and_bounded_connect_timeout(self):
        env = {"OPENAI_API_KEY": "test-only-key", "OPENAI_BASE_URL": PRIMARY_BASE_URL}
        with patch.dict(os.environ, env, clear=True), patch("openai.OpenAI") as constructor:
            llm.make_client()
        options = constructor.call_args.kwargs
        self.assertEqual(options["base_url"], ALTERNATE_BASE_URL)
        self.assertEqual(options["api_key"], "test-only-key")
        self.assertEqual(options["max_retries"], 0)
        self.assertEqual(options["timeout"].connect, 10)
        self.assertEqual(options["timeout"].read, 60)


class GatewayTransportCheckTests(unittest.TestCase):
    def client(self, handler):
        client = OpenAI(api_key="test-only-key", base_url=ALTERNATE_BASE_URL,
                        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    @staticmethod
    def completion(request):
        return httpx.Response(200, request=request, json={
            "id": "fake-completion", "object": "chat.completion", "created": 0,
            "model": "fake-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "OK"}}],
        })

    def test_check_fails_over_same_key_once_on_transport_failure(self):
        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("fake connection failure", request=request)
            return self.completion(request)

        output = io.StringIO()
        with patch.object(llm, "make_client", return_value=self.client(handler)), redirect_stdout(output):
            llm.check_api()
        self.assertEqual([r.url.host for r in requests],
                         ["api.alfahack-gateway26.ru", "changellenge.up.railway.app"])
        self.assertEqual([r.headers["authorization"] for r in requests], ["Bearer test-only-key"] * 2)
        self.assertIn("API gateway: primary", output.getvalue())
        self.assertNotIn("test-only-key", output.getvalue())
        self.assertEqual([json.loads(r.content)["max_tokens"] for r in requests], [64, 64])

    def test_authentication_and_shared_quota_never_try_the_other_host(self):
        for status, code in ((401, "invalid_api_key"), (403, "permission_denied"),
                             (429, "insufficient_quota"), (429, "daily_budget_exceeded")):
            with self.subTest(status=status, code=code):
                calls = []

                def handler(request):
                    calls.append(request)
                    return httpx.Response(status, request=request,
                        json={"error": {"message": "fake account error", "code": code}})

                with patch.object(llm, "make_client", return_value=self.client(handler)), \
                        redirect_stdout(io.StringIO()), self.assertRaises(Exception):
                    llm.check_api()
                self.assertEqual(len(calls), 1)

    def test_two_unavailable_hosts_end_after_two_network_attempts(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(503, request=request, json={"error": {"message": "fake overload"}})

        with patch.object(llm, "make_client", return_value=self.client(handler)), \
                redirect_stdout(io.StringIO()), self.assertRaises(Exception):
            llm.check_api()
        self.assertEqual(len(calls), 2)


class ResearchGatewayAccountingTests(unittest.TestCase):
    def state(self, limit=4):
        report = {"max_steps": limit,
                  "api": {"attempts": 0, "responses": 0, "prompt_tokens": 0,
                          "completion_tokens": 0, "total_tokens": 0},
                  "api_recovery": {"maximum_retries": 1, "retries": 0}, "events": []}
        state = SimpleNamespace(report=report, timeout=90)
        state.budget_context = lambda: {"remaining_seconds": 90}
        state.request_allowed = lambda messages: ("step_limit" if report["api"]["attempts"] >= limit else None)
        state.research_timeout = lambda maximum=1800: state.timeout
        state.clean = lambda value: value
        state.save = lambda: None
        state.event = lambda kind, **fields: report["events"].append({"kind": kind, **fields})
        return state

    def client(self, handler):
        client = OpenAI(api_key="test-only-key", base_url=PRIMARY_BASE_URL,
                        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    def test_failover_counts_actual_attempts_and_keeps_the_healthy_route(self):
        from agent.research_v3 import _request_completion

        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ConnectError("fake outage", request=request)
            return GatewayTransportCheckTests.completion(request)

        state, client = self.state(), self.client(handler)
        for _ in range(2):
            completion, stopped = _request_completion(state, client, [{"role": "user", "content": "test"}])
            self.assertIsNone(stopped)
            self.assertEqual(completion["content"], "OK")
        self.assertEqual([r.url.host for r in calls],
                         ["api.alfahack-gateway26.ru", "changellenge.up.railway.app", "changellenge.up.railway.app"])
        self.assertEqual(state.report["api"]["attempts"], 3)
        self.assertEqual(state.report["api"]["responses"], 2)
        self.assertEqual(state.report["api_recovery"]["retries"], 1)
        self.assertEqual(state.report["api_gateway"]["attempts"], {"alternate": 1, "primary": 2})
        self.assertEqual(state.report["api_gateway"]["switches"], 1)
        self.assertNotIn("test-only-key", json.dumps(state.report))
        self.assertNotIn("https://", json.dumps(state.report))

    def test_failover_cannot_bypass_the_original_request_or_time_guard(self):
        from agent.research_v3 import _request_completion

        for exhausted in ("requests", "time"):
            with self.subTest(exhausted=exhausted):
                state = self.state(limit=1 if exhausted == "requests" else 4)
                calls = []

                def handler(request):
                    calls.append(request)
                    if exhausted == "time":
                        state.timeout = 0
                    raise httpx.ConnectError("fake outage", request=request)

                result = _request_completion(state, self.client(handler), [{"role": "user", "content": "test"}])
                self.assertEqual(result, (None, "api_error"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(state.report["api"]["attempts"], 1)
                self.assertEqual(state.report["api_recovery"]["retries"], 0)
                self.assertEqual(state.report["api_gateway"]["switches"], 0)

    def test_invalid_paid_completion_does_not_trigger_another_gateway(self):
        from agent.research_v3 import _request_completion

        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, request=request, json={"choices": []})

        state = self.state()
        self.assertEqual(_request_completion(state, self.client(handler), [{"role": "user", "content": "test"}]),
                         (None, "api_error"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.report["api"]["responses"], 1)


if __name__ == "__main__":
    unittest.main()
