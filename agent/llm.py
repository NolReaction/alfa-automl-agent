"""Qwen configuration, instructions and API access. Owner: Nika."""

from __future__ import annotations

import importlib.util
import os
import re

from agent import project_root


ROOT = project_root()
REQUIRED_API_FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
API_FIELDS = (*REQUIRED_API_FIELDS, "OPENAI_MODEL")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python in a new process in the task directory. Files persist "
            "between calls; Python variables do not. Returns stdout, stderr, "
            "exit_code and timed_out."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
SYSTEM_PROMPT = """You are an autonomous ML researcher. Read the supplied task
carefully and solve it using run_python. All relative paths are relative to the
directory containing task.md. Inspect the actual files, data dictionary and
submission template. Infer the target, task type, metric and output requirements
from those materials; do not assume specific column names or labels.

A starting baseline may already exist. Additional user context will describe
its status. Treat an existing valid baseline as a starting point; otherwise
train a fast, simple baseline and immediately save usable predictions in the
exact format and location requested by the task. In either case, perform a small
AutoResearch cycle: record a hypothesis, test it, measure its validation result,
and compare it with the best result. Keep a written experiment history in the
task directory. Reserve time for final training and writing predictions.

Use the same held-out validation rows when comparing experiments. Fit learned
preprocessing only on training rows. Do not use test targets, external data,
internet downloads or pretrained ML models. Available initial modelling tools
are numpy, pandas and scikit-learn. Set CPU parallelism to at most 8.
Never read .env files, API keys or credentials.

When replacing a useful prediction file, write to a temporary file first and
replace the destination atomically. A failed experiment must preserve the best
predictions. Check columns, row order, missing values and task-specific output
rules before finishing. Finish with a short factual account of completed work.
"""


def load_local_env() -> None:
    # Import only when installed: the offline check also works before setup.
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)


def require_api_settings() -> None:
    missing = [name for name in REQUIRED_API_FIELDS if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Missing required API settings: " + ", ".join(missing)
                         + ". Supply them in the environment (or .env for --mode llm).")


def get_model() -> str:
    """The contest proxy may replace this optional OpenAI request field."""
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def safe_error_message(error: Exception) -> str:
    """Keep useful SDK diagnostics without echoing response bodies or headers."""
    try:
        from openai import APIError
    except ImportError:
        APIError = ()

    if isinstance(error, APIError):
        details = [type(error).__name__]
        status = getattr(error, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            details.append(f"HTTP {status}")
        request_id = getattr(error, "request_id", None)
        if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            details.append(f"request_id={request_id}")
        code = getattr(error, "code", None)
        if isinstance(code, str) and code in {
            "rate_limit_exceeded", "rate_limit_error", "insufficient_quota",
            "quota_exceeded", "budget_exceeded", "budget_limit_exceeded",
            "daily_budget_exceeded", "insufficient_balance",
        }:
            details.append(f"code={code}")
        detail = "API request failed (" + ", ".join(details) + ")."
    else:
        detail = str(error)
    key = os.environ.get("OPENAI_API_KEY", "")
    return detail.replace(key, "[REDACTED]") if key else detail


def make_client():
    require_api_settings()
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=60, max_retries=0,
    )


def check_api() -> None:
    with make_client() as client:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=64,
        )
    print("API connection successful.")
    print("Reply:", response.choices[0].message.content or "(empty response)")
