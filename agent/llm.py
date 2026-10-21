"""Qwen configuration, instructions and API access. Owner: Nika."""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import ssl

from agent import project_root


ROOT = project_root()
REQUIRED_API_FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
API_FIELDS = (*REQUIRED_API_FIELDS, "OPENAI_MODEL")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
_API_ERROR_TYPES = (
    "APITimeoutError", "APIConnectionError", "BadRequestError", "AuthenticationError",
    "PermissionDeniedError", "NotFoundError", "ConflictError", "UnprocessableEntityError",
    "RateLimitError", "InternalServerError", "APIResponseValidationError", "APIStatusError", "APIError",
)
_HTTP_ERROR_TYPES = (
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
    "ConnectError", "ReadError", "WriteError", "CloseError", "NetworkError", "ProxyError",
    "RemoteProtocolError", "LocalProtocolError", "ProtocolError", "TransportError", "RequestError",
)
_BUILTIN_ERROR_TYPES = (
    socket.gaierror, ssl.SSLCertVerificationError, ssl.SSLError, ConnectionRefusedError,
    TimeoutError, ConnectionError, OSError, ValueError, RuntimeError, Exception,
)
_ERROR_TYPE_NAMES = frozenset((*_API_ERROR_TYPES, *_HTTP_ERROR_TYPES,
                               *(kind.__name__ for kind in _BUILTIN_ERROR_TYPES)))
_API_ERROR_CODES = frozenset({
    "rate_limit_exceeded", "rate_limit_error", "insufficient_quota", "quota_exceeded",
    "budget_exceeded", "budget_limit_exceeded", "daily_budget_exceeded", "insufficient_balance",
})
_TRANSPORT_REASONS = {
    "APITimeoutError": "timeout", "APIConnectionError": "connection_error",
    "ConnectTimeout": "connect_timeout", "ReadTimeout": "read_timeout",
    "WriteTimeout": "write_timeout", "PoolTimeout": "pool_timeout", "TimeoutException": "timeout",
    "ConnectError": "connect_error", "ReadError": "read_error", "WriteError": "write_error",
    "CloseError": "close_error", "NetworkError": "network_error", "ProxyError": "proxy_error",
    "RemoteProtocolError": "protocol_error", "LocalProtocolError": "protocol_error",
    "ProtocolError": "protocol_error", "TransportError": "transport_error",
    "gaierror": "dns_error", "SSLCertVerificationError": "tls_certificate_error",
    "SSLError": "tls_error", "ConnectionRefusedError": "connection_refused",
    "TimeoutError": "timeout", "ConnectionError": "connection_error",
}
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


def sanitize_error_details(details) -> dict:
    """Revalidate diagnostic metadata before copying it into a public summary.

    No free-form provider values, messages, request IDs, URLs or headers belong
    in this schema. Unknown fields and values are intentionally discarded.
    """
    if not isinstance(details, dict):
        return {}
    result = {}
    for field in ("error_type", "cause_type"):
        value = details.get(field)
        if isinstance(value, str) and value in _ERROR_TYPE_NAMES:
            result[field] = value
    status = details.get("status_code")
    if type(status) is int and 100 <= status <= 599:
        result["status_code"] = status
    code = details.get("code")
    if isinstance(code, str) and code in _API_ERROR_CODES:
        result["code"] = code
    reason = details.get("transport_reason")
    if isinstance(reason, str) and reason in _TRANSPORT_REASONS.values():
        result["transport_reason"] = reason
    return result


def safe_error_details(error: Exception) -> dict:
    """Describe a typed SDK/transport failure without parsing its message.

    For example, a generic ConnectError does not prove a DNS or TLS failure.
    Only an actual typed nested cause supports those more precise diagnoses.
    The cause chain is bounded and cycle-safe; no exception text is accessed.
    """
    known = []
    api_error = ()
    try:
        import openai
        api_error = openai.APIError
        known.extend((getattr(openai, name), name) for name in _API_ERROR_TYPES
                     if hasattr(openai, name))
    except ImportError:
        pass
    try:
        import httpx
        known.extend((getattr(httpx, name), name) for name in _HTTP_ERROR_TYPES
                     if hasattr(httpx, name))
    except ImportError:
        pass
    known.extend((kind, kind.__name__) for kind in _BUILTIN_ERROR_TYPES)

    def known_type(value):
        return next((name for kind, name in known if isinstance(value, kind)), "Exception")

    result = {"error_type": known_type(error)}
    if isinstance(error, api_error):
        result.update(status_code=getattr(error, "status_code", None), code=getattr(error, "code", None))
    seen = set()
    current = error
    for depth in range(8):
        if not isinstance(current, BaseException) or id(current) in seen:
            break
        seen.add(id(current))
        name = known_type(current)
        reason = _TRANSPORT_REASONS.get(name)
        if reason is not None:
            result["transport_reason"] = reason
            if depth:
                result["cause_type"] = name
        cause = current.__cause__
        current = cause if cause is not None else (
            None if current.__suppress_context__ else current.__context__)
    return sanitize_error_details(result)


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
        if isinstance(code, str) and code in _API_ERROR_CODES:
            details.append(f"code={code}")
        detail = "API request failed (" + ", ".join(details) + ")."
    else:
        detail = str(error)
    key = os.environ.get("OPENAI_API_KEY", "")
    return detail.replace(key, "[REDACTED]") if key else detail


def make_client():
    require_api_settings()
    from openai import OpenAI
    import httpx
    from agent.gateways import CONNECT_TIMEOUT_SECONDS, configured_gateways

    routes = configured_gateways(os.environ["OPENAI_BASE_URL"])

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=routes[0][1],
        timeout=(httpx.Timeout(60, connect=CONNECT_TIMEOUT_SECONDS)
                 if routes[0][0] != "custom" else 60), max_retries=0,
    )


def check_api() -> None:
    from agent.gateways import client_gateways, request_options, retryable_api_error

    with make_client() as client:
        routes = client_gateways(client) or (None,)
        for index, route in enumerate(routes):
            try:
                request_client = client if route is None else client.with_options(**request_options(30, route))
                response = request_client.chat.completions.create(
                    model=get_model(),
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    max_tokens=64,
                )
                if route is not None:
                    print("API gateway:", route[0])
                break
            except Exception as error:
                if index + 1 == len(routes) or not retryable_api_error(error):
                    raise
    print("API connection successful.")
    print("Reply:", response.choices[0].message.content or "(empty response)")
