"""No-network evidence for the opt-in rescue probe and its aggregate ceilings."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from agent.api_probe import ProbeBudget, ProbeClient, ProbeLimitReached, request_size
from agent.gateways import ALTERNATE_BASE_URL, PRIMARY_BASE_URL
import experiment_rescue_api as probe


SECRET = "rescue-private-value-DO-NOT-EXPORT"


def completion(request, *, content="OK", code=None, arguments=None, finish=None):
    message = {"role": "assistant", "content": content}
    if code is not None or arguments is not None:
        message["tool_calls"] = [{"id": "probe-call", "type": "function", "function": {
            "name": "run_python", "arguments": arguments if arguments is not None else json.dumps({"code": code})}}]
    return httpx.Response(200, request=request, json={
        "id": SECRET, "object": "chat.completion", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish or ("tool_calls" if "tool_calls" in message else "stop")}],
    })


class ProbeTransportTests(unittest.TestCase):
    def client(self, handler, *, budget=None, base_url=ALTERNATE_BASE_URL):
        sdk = OpenAI(api_key=SECRET, base_url=base_url, max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(sdk.close)
        return ProbeClient(sdk, budget or ProbeBudget(deadline=time.monotonic() + 90))

    def test_failover_and_clones_share_exact_attempts_unknown_usage_and_safe_metadata(self):
        requests = []
        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ReadTimeout(SECRET, request=request)
            return completion(request)
        client = self.client(handler)
        recovery = {"retries": 0, "route_index": 0}
        item, valid = probe._diagnostic_request(client, client.budget, "text", recovery)
        self.assertTrue(valid)
        self.assertTrue(item["api_response_received"])
        client.with_options(base_url=PRIMARY_BASE_URL).with_options(timeout=1).create(**probe._arguments("text"))
        self.assertEqual([request.url.host for request in requests], [
            "api.alfahack-gateway26.ru", "changellenge.up.railway.app", "changellenge.up.railway.app"])
        summary = client.budget.summary()
        self.assertEqual(summary["attempts"], 3)
        self.assertEqual(summary["responses"], 2)
        self.assertEqual(summary["responses_without_usage"], 2)
        self.assertFalse(summary["usage_complete"])
        self.assertIsNone(summary["currency_cost"])
        self.assertGreater(summary["estimated_tokens_reserved"], 3 * 64)
        self.assertEqual(recovery["retries"], 1)
        raw = json.dumps({"budget": summary, "attempts": client.budget.attempts})
        self.assertNotIn(SECRET, raw)
        self.assertNotIn("https://", raw)
        self.assertEqual(client.budget.attempts[0]["error"]["cause_type"], "ReadTimeout")

    def test_retry_cannot_bypass_any_shared_ceiling(self):
        args = probe._arguments("text")
        cost = request_size(args)["estimated_tokens"]
        for exhausted in ("request_limit", "token_limit", "deadline"):
            with self.subTest(exhausted=exhausted):
                clock = [100.]
                budget = ProbeBudget(deadline=110., max_requests=1 if exhausted == "request_limit" else 6,
                                     max_tokens=cost if exhausted == "token_limit" else 30000,
                                     clock=lambda: clock[0])
                calls = []
                def handler(request):
                    calls.append(request)
                    if exhausted == "deadline":
                        clock[0] = 110.
                    raise httpx.ConnectTimeout(SECRET, request=request)
                client = self.client(handler, budget=budget)
                with self.assertRaises(ProbeLimitReached):
                    probe._diagnostic_request(client, budget, "text", {"retries": 0, "route_index": 0})
                self.assertEqual(len(calls), 1)
                self.assertEqual(budget.summary()["attempts"], 1)
                self.assertEqual(budget.reserved, cost)
                self.assertLessEqual(calls[0].extensions["timeout"]["connect"], 10)
                self.assertLessEqual(calls[0].extensions["timeout"]["read"], 10)

    def test_paid_malformed_truncated_empty_and_invalid_json_never_retry(self):
        variants = (
            ("text", lambda request: httpx.Response(200, request=request, json={"choices": []}), "malformed_response"),
            ("text", lambda request: completion(request, finish="length"), "truncated_response"),
            ("text", lambda request: completion(request, content=""), "empty_or_unexpected_response"),
            ("tool_schema", lambda request: completion(request, arguments='{"code":'), "invalid_tool_json"),
            ("tool_schema", lambda request: completion(request, code="print('untrusted')"), "unexpected_probe_code"),
        )
        for stage, reply, expected in variants:
            with self.subTest(expected=expected):
                calls = []
                def handler(request):
                    calls.append(request)
                    return reply(request)
                client = self.client(handler)
                recovery = {"retries": 0, "route_index": 0}
                item, valid = probe._diagnostic_request(client, client.budget, stage, recovery)
                self.assertFalse(valid)
                self.assertEqual(item["status"], expected)
                self.assertEqual(len(calls), 1)
                self.assertEqual(recovery["retries"], 0)
                self.assertEqual(client.budget.summary()["responses"], 1)

    def test_quota_error_is_terminal_and_custom_key_stays_on_custom_endpoint(self):
        for base in (ALTERNATE_BASE_URL, "https://custom.example.invalid/v1"):
            with self.subTest(base=base):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(429, request=request, headers={"x-request-id": SECRET},
                        json={"error": {"code": "insufficient_quota", "message": SECRET}})
                client = self.client(handler, base_url=base)
                recovery = {"retries": 0, "route_index": 0}
                with self.assertRaises(Exception):
                    probe._diagnostic_request(client, client.budget, "text", recovery)
                self.assertEqual(len(calls), 1)
                self.assertEqual(str(calls[0].url).split("/chat/")[0], base)
                self.assertEqual(recovery["retries"], 0)
                self.assertNotIn(SECRET, json.dumps(client.budget.attempts))


class ProbeWorkflowTests(unittest.TestCase):
    def test_default_preflight_and_missing_settings_never_construct_network_client(self):
        for execute in (False, True):
            with self.subTest(execute=execute), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {}, clear=True), patch.object(probe, "load_local_env"), \
                    patch.object(probe, "make_client") as client, patch.object(probe, "_supervise") as supervisor, \
                    redirect_stdout(io.StringIO()):
                args = ["--output-dir", directory] + (["--execute"] if execute else [])
                self.assertEqual(probe.main(args), 2 if execute else 0)
                client.assert_not_called()
                supervisor.assert_not_called()
                report = json.loads(next(Path(directory).glob("*/probe_report.json")).read_text())
                self.assertFalse(report["api_called"])
                self.assertIsNone(report["quota_remaining"])
                self.assertFalse(report["integration_passed"])

    def test_unexpected_tool_code_stops_research_and_never_executes(self):
        calls = []
        def handler(request):
            calls.append(request)
            return completion(request) if len(calls) == 1 else completion(request, code="raise RuntimeError('untrusted')")
        sdk = OpenAI(api_key=SECRET, base_url=ALTERNATE_BASE_URL, max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "make_client", return_value=sdk), \
                patch("agent.execution.run_python") as run, patch.object(probe, "run_agent_v3") as smoke:
            report = {}
            code = probe.execute_probe(Path(directory), report, deadline=time.monotonic() + 90,
                                       max_requests=6, max_tokens=30000)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "diagnostic_failed")
        self.assertEqual(len(calls), 2)
        run.assert_not_called()
        smoke.assert_not_called()

    def test_only_exact_constant_executes_then_real_v3_hook_shares_remaining_budget(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return completion(request)
            return completion(request, code=probe.TOOL_CODE if len(calls) == 2 else "never execute diagnostic research code")
        sdk = OpenAI(api_key=SECRET, base_url=ALTERNATE_BASE_URL, max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        observed = []
        def smoke(task, client, *, max_steps, time_budget):
            observed.append((max_steps, time_budget, client.budget.summary()["attempts"]))
            client.with_options(timeout=1).chat.completions.create(**probe._arguments("text"))
            native = task.parent / "fixture_native"
            native.mkdir()
            (native / "run.json").write_text("{}")
            return native
        summary = {"coverage": {"inner_trial_verified": False, "candidate_published": False},
                   "status": "fallback", "output_check": {"valid": True},
                   "publication": {"new_result_published": True, "recipe": "baseline"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "make_client", return_value=sdk), \
                patch("agent.execution.run_python", return_value={"exit_code": 0, "stdout": "rescue-api-tool-ok\n"}) as run, \
                patch.object(probe, "run_agent_v3", side_effect=smoke), \
                patch.object(probe, "summarize_run", return_value=summary):
            report = {}
            self.assertEqual(probe.execute_probe(Path(directory), report, deadline=time.monotonic() + 90,
                                                max_requests=4, max_tokens=30000), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], probe.TOOL_CODE)
        self.assertEqual(len(calls), 4)
        self.assertEqual(observed[0][0], 1)
        self.assertEqual(observed[0][2], 3)
        self.assertLessEqual(observed[0][1], 90)
        self.assertEqual(report["budget"]["attempts"], 4)
        self.assertEqual(report["attempts"][-1]["variant"], "v3_smoke")
        self.assertFalse(report["integration_passed"])
        self.assertEqual(report["status"], "integration_unverified")


if __name__ == "__main__":
    unittest.main()
