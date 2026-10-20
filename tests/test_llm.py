"""Required API credentials, optional model, and safe SDK error diagnostics."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AuthenticationError, RateLimitError

from agent import llm


REQUIRED_ENV = {
    "OPENAI_API_KEY": "unit-test-placeholder",
    "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
}


class ApiSettingsTests(unittest.TestCase):
    def test_two_required_settings_are_sufficient_and_model_defaults(self):
        for optional_model in (None, "", "  "):
            environment = dict(REQUIRED_ENV)
            if optional_model is not None:
                environment["OPENAI_MODEL"] = optional_model
            with self.subTest(model=optional_model), patch.dict(os.environ, environment, clear=True):
                llm.require_api_settings()
                self.assertEqual(llm.get_model(), "qwen/qwen3.8-27b")

    def test_supplied_model_is_used(self):
        with patch.dict(os.environ, {**REQUIRED_ENV, "OPENAI_MODEL": "team-provided-model"}, clear=True):
            self.assertEqual(llm.get_model(), "team-provided-model")

    def test_partial_credentials_are_rejected_without_values(self):
        for missing in REQUIRED_ENV:
            for missing_value in (None, "  "):
                environment = dict(REQUIRED_ENV)
                if missing_value is None:
                    del environment[missing]
                else:
                    environment[missing] = missing_value
                with self.subTest(missing=missing, value=missing_value), \
                        patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, missing) as error:
                        llm.require_api_settings()
                    self.assertNotIn(REQUIRED_ENV["OPENAI_API_KEY"], str(error.exception))

    def test_client_uses_runtime_credentials_and_disables_sdk_retries(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True), patch("openai.OpenAI") as client:
            self.assertIs(llm.make_client(), client.return_value)
        client.assert_called_once_with(api_key=REQUIRED_ENV["OPENAI_API_KEY"],
                                       base_url=REQUIRED_ENV["OPENAI_BASE_URL"],
                                       timeout=60, max_retries=0)

    def test_api_check_uses_default_or_supplied_model_in_one_request(self):
        for model in (None, "team-provided-model"):
            environment = dict(REQUIRED_ENV)
            if model is not None:
                environment["OPENAI_MODEL"] = model
            with self.subTest(model=model), patch.dict(os.environ, environment, clear=True), \
                    patch.object(llm, "make_client") as factory, redirect_stdout(io.StringIO()):
                client = factory.return_value.__enter__.return_value
                llm.check_api()
                client.chat.completions.create.assert_called_once_with(
                    model=model or llm.DEFAULT_MODEL,
                    messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=64,
                )

    def test_local_dotenv_fills_missing_settings_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "OPENAI_API_KEY=local-placeholder\nOPENAI_BASE_URL=http://local.example/v1\n"
                "OPENAI_MODEL=local-model\n", encoding="utf-8",
            )
            with patch.dict(os.environ, REQUIRED_ENV, clear=True), patch.object(llm, "ROOT", root):
                llm.load_local_env()
                self.assertEqual(os.environ["OPENAI_API_KEY"], REQUIRED_ENV["OPENAI_API_KEY"])
                self.assertEqual(os.environ["OPENAI_BASE_URL"], REQUIRED_ENV["OPENAI_BASE_URL"])
                self.assertEqual(llm.get_model(), "local-model")

    def test_sdk_errors_keep_safe_metadata_without_body_or_headers(self):
        response = httpx.Response(429, request=httpx.Request("POST", "http://local.example/v1"),
                                  headers={"x-request-id": "req-local-check", "x-private": "header-marker"})
        error = RateLimitError("Authorization: Bearer body-marker", response=response,
                               body={"code": "insufficient_quota", "message": "body-marker"})
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            detail = llm.safe_error_message(error)
        self.assertEqual(detail, "API request failed (RateLimitError, HTTP 429, "
                                 "request_id=req-local-check, code=insufficient_quota).")
        self.assertNotIn("body-marker", detail)
        self.assertNotIn("header-marker", detail)
        self.assertNotIn("Authorization", detail)

    def test_untrusted_error_metadata_is_not_echoed(self):
        response = httpx.Response(401, request=httpx.Request("POST", "http://local.example/v1"),
                                  headers={"x-request-id": "Authorization: Bearer header-marker"})
        error = AuthenticationError("body-marker", response=response,
                                    body={"code": "body-marker"})
        self.assertEqual(llm.safe_error_message(error), "API request failed (AuthenticationError, HTTP 401).")

    def test_non_sdk_errors_redact_the_configured_key(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            self.assertEqual(llm.safe_error_message(ValueError(REQUIRED_ENV["OPENAI_API_KEY"])),
                             "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
