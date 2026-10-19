"""Qwen configuration, instructions and API access. Owner: Nika."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
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
    missing = [name for name in API_FIELDS if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Fill in these settings in .env: " + ", ".join(missing))


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
            model=os.environ["OPENAI_MODEL"],
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=64,
        )
    print("API connection successful.")
    print("Reply:", response.choices[0].message.content or "(empty response)")
