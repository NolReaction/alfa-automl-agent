"""Typed transport diagnostics never copy provider text or credentials."""

import json
from pathlib import Path
import socket
import ssl
import tempfile
import unittest

import httpx
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from agent.llm import safe_error_details, sanitize_error_details
import experiment_research_smoke as smoke


SECRET = "private-key-and-provider-body"
PRIVATE_URL = "https://user:password@private.example/v1?token=" + SECRET


class ApiDiagnosticsTests(unittest.TestCase):
    def connection_error(self, cause=None):
        error = APIConnectionError(message=PRIVATE_URL, request=httpx.Request("POST", PRIVATE_URL))
        error.__cause__ = cause
        return error

    def test_dns_is_identified_from_nested_type_without_copying_messages(self):
        transport = httpx.ConnectError("Authorization: Bearer " + SECRET)
        transport.__cause__ = socket.gaierror(-2, PRIVATE_URL)
        details = safe_error_details(self.connection_error(transport))
        self.assertEqual(details, {"error_type": "APIConnectionError", "cause_type": "gaierror",
                                   "transport_reason": "dns_error"})
        for secret in (SECRET, PRIVATE_URL, "Authorization", "password"):
            self.assertNotIn(secret, json.dumps(details))

    def test_generic_connect_message_does_not_claim_dns_or_tls(self):
        cause = httpx.ConnectError("DNS failed; TLS certificate expired; timeout " + SECRET)
        self.assertEqual(safe_error_details(self.connection_error(cause)), {
            "error_type": "APIConnectionError", "cause_type": "ConnectError",
            "transport_reason": "connect_error",
        })

    def test_timeout_stages_tls_and_refusal_are_distinguished_by_types(self):
        cases = [
            (httpx.ConnectTimeout(SECRET), "connect_timeout", "ConnectTimeout"),
            (httpx.ReadTimeout(SECRET), "read_timeout", "ReadTimeout"),
            (httpx.WriteTimeout(SECRET), "write_timeout", "WriteTimeout"),
            (httpx.PoolTimeout(SECRET), "pool_timeout", "PoolTimeout"),
            (ssl.SSLCertVerificationError(1, SECRET), "tls_certificate_error", "SSLCertVerificationError"),
            (ssl.SSLError(1, SECRET), "tls_error", "SSLError"),
            (ConnectionRefusedError(111, SECRET), "connection_refused", "ConnectionRefusedError"),
        ]
        for cause, reason, name in cases:
            with self.subTest(reason=reason):
                details = safe_error_details(self.connection_error(cause))
                self.assertEqual(details["transport_reason"], reason)
                self.assertEqual(details["cause_type"], name)
                self.assertNotIn(SECRET, json.dumps(details))
        error = APITimeoutError(request=httpx.Request("POST", PRIVATE_URL))
        self.assertEqual(safe_error_details(error), {"error_type": "APITimeoutError",
                                                   "transport_reason": "timeout"})

    def test_http_status_and_allowlisted_quota_code_survive_without_headers(self):
        response = httpx.Response(429, request=httpx.Request("POST", PRIVATE_URL),
                                  headers={"x-request-id": SECRET, "Authorization": "Bearer " + SECRET})
        error = RateLimitError(SECRET, response=response,
                               body={"code": "insufficient_quota", "message": PRIVATE_URL})
        self.assertEqual(safe_error_details(error), {"error_type": "RateLimitError", "status_code": 429,
                                                   "code": "insufficient_quota"})
        response = httpx.Response(401, request=response.request)
        error = AuthenticationError(SECRET, response=response, body={"code": SECRET})
        self.assertEqual(safe_error_details(error), {"error_type": "AuthenticationError", "status_code": 401})

    def test_unknown_exception_name_and_string_are_never_accessed(self):
        def forbidden_string(self):
            raise AssertionError("Exception text must not be read")

        custom_type = type(SECRET, (Exception,), {"__str__": forbidden_string})
        error = custom_type(PRIVATE_URL)
        error.__cause__ = custom_type(SECRET)
        self.assertEqual(safe_error_details(error), {"error_type": "Exception"})

    def test_cyclic_and_suppressed_chains_are_safe(self):
        error = self.connection_error()
        cause = httpx.ConnectTimeout(SECRET)
        cause.__cause__ = error
        error.__cause__ = cause
        self.assertEqual(safe_error_details(error)["transport_reason"], "connect_timeout")
        error = self.connection_error()
        error.__context__ = socket.gaierror(-2, SECRET)
        error.__suppress_context__ = True
        self.assertEqual(safe_error_details(error), {"error_type": "APIConnectionError",
                                                   "transport_reason": "connection_error"})

    def test_summary_revalidates_fields_and_drops_provider_text(self):
        with tempfile.TemporaryDirectory() as directory:
            task = smoke.create_tasks(Path(directory))["regression"]
            details = {"error_type": "APIConnectionError", "transport_reason": "dns_error",
                       "cause_type": "gaierror", "status_code": True, "code": SECRET,
                       "request_id": SECRET, "message": PRIVATE_URL, "headers": {"key": SECRET}}
            report = {"status": "fallback", "reason": "api_error", "events": [
                {"kind": "api_error", "error": PRIVATE_URL, "details": details},
                {"kind": "api_error", "error": SECRET, "details": {"error_type": SECRET}},
                {"kind": "other", "details": {"status_code": 500}},
            ]}
            summary = smoke.summarize_run(task, report)
            self.assertEqual(summary["api"]["errors"], [{"error_type": "APIConnectionError",
                              "transport_reason": "dns_error", "cause_type": "gaierror"}])
            self.assertNotIn(SECRET, json.dumps(summary))
            self.assertNotIn(PRIVATE_URL, json.dumps(summary))

    def test_summary_bounds_diagnostics_and_ignores_legacy_or_invalid_details(self):
        report = {"events": [{"kind": "api_error", "details": {"status_code": status}}
                             for status in range(400, 420)]}
        errors = smoke._api_errors(report)
        self.assertEqual(errors, [{"status_code": status} for status in range(412, 420)])
        self.assertEqual(smoke._api_errors({"events": SECRET}), [])
        self.assertEqual(smoke._api_errors({"events": [{"kind": "api_error", "error": SECRET}]}), [])
        for value in (SECRET, None, [SECRET], {"status_code": 600}, {"status_code": True},
                      {"transport_reason": [SECRET]}, {"cause_type": {"secret": SECRET}}):
            with self.subTest(value=value):
                self.assertEqual(sanitize_error_details(value), {})


if __name__ == "__main__":
    unittest.main()
