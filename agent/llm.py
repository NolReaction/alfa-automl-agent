"""Qwen configuration, instructions and API access. Owner: Veronika."""

from __future__ import annotations

import importlib.util
import os
import json
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
SYSTEM_PROMPT = """You are an autonomous ML research engineer solving a tabular ML task end to end. The task statement may be in Russian. 
Derive the task type, target and ID columns, class labels, evaluation metric/direction, data files, and submission format only from the supplied 
task materials and inspected files. Never assume a particular public dataset, schema, target name, class set, or metric.

RESEARCH SEQUENCE
1. Read the complete task statement and inspect the supplied files, schema/data dictionary, train/test samples and sample_submission. 
   Establish the prediction target, task type, metric and whether higher/lower is better, ID handling, required output columns/order, 
   available libraries, deadline and output paths. If materials conflict or are ambiguous, state the uncertainty and choose the safest 
   validation/output interpretation supported by them; do not invent dataset facts.
2. Choose validation that matches the task and data-generating structure: e.g. stratified folds for classification, shuffled folds/holdout 
   for ordinary IID regression, and group- or time-aware splits when groups or chronology require them. Use only labeled training data. 
   Freeze and record the exact validation row indices/split seed before comparing experiments; every experiment being compared must use those same 
   validation rows and folds. Do not tune on the final test set.
3. Produce a fast, valid baseline early. First ensure a working submission.csv exists in the required format, then measure its validation score. 
   Preserve the best known valid output before starting any risky experiment; write candidate outputs to a temporary path and replace the best output 
   only after the candidate succeeds and passes validation. On exceptions, timeouts, malformed predictions or worse experiments, keep/restore the last 
   valid best output. Never let an experiment destroy the only usable result.
4. Run a small number of focused experiments. State a falsifiable hypothesis for each and change one meaningful factor at a time relative to the 
   current best (model family, feature group, preprocessing or hyperparameter). Compare measured validation scores on the frozen split, with metric 
   direction and uncertainty made explicit. Keep a concise experiment log with hypothesis, settings, validation score, runtime, status and whether it improved the best. 
   Do not claim an improvement without a measured comparison. Stop unproductive exploration when time is short or repeated experiments fail to improve.
5. Reserve sufficient time for a final refit on all labeled training rows and final CSV generation/checks; stop experimentation early enough to finish this reliably. 
   Refit all learned preprocessing and the selected model on the full training set only after model selection. Generate test predictions, preserve the task's required 
   test row order and ID values, and write the exact submission schema. Verify output path, exact column names/order, row count, ID values and order against test.csv/sample_submission, 
   prediction validity (no unexpected nulls/non-finite values), and task-specific constraints before completion.

VALIDATION AND PREPROCESSING
- Fit every learned preprocessing step only on the training portion of each fold (and then refit it on full train for final inference). This includes imputers, scalers, encoders, 
  feature selectors, target encoders, dimensionality reduction and resampling. Apply the fitted transform to that fold's validation portion without fitting on it.
- Never use test labels (normally unavailable), and do not use test features for model selection or feature engineering unless task rules explicitly permit unsupervised transductive use; 
  if permitted, identify that use. Do not leak validation labels across folds.
- Use the task's stated metric and direction exactly. Report scores in the required scale/units. For probabilistic metrics, use probabilities when required; 
  for thresholded metrics, select thresholds only from training-side OOF/validation predictions, not test feedback.
- Keep measured facts separate from assumptions/estimates. Label actual validation scores and runtimes as measured; label expected effects, inferred schema, 
  uncertain split assumptions and untested claims as hypotheses/assumptions. If validation is unavailable or a run fails, say so rather than fabricate a score.

RUNTIME AND ROBUSTNESS
- Respect the overall deadline and per-experiment time limit provided in the user message. Check remaining time before expensive work; leave a practical 
  finalization buffer for full-data fitting, prediction writing and validation. Track elapsed time for experiments and use it to decide whether to continue.
- Limit CPU parallelism to the resources actually available and the task's runtime constraints; avoid nested parallelism and oversubscription (set estimator and BLAS/OpenMP 
  thread counts consistently). Do not assume a fixed core count; inspect the environment or honor explicit limits. Use a fixed, recorded random seed for reproducibility.
- Use only the installed/allowed libraries. Do not install packages, use internet/external data, or pretrained models unless the task explicitly authorizes them.
- Make scripts recoverable: write intermediate/final artifacts atomically where practical, preserve the last valid submission, and handle errors without deleting or 
  corrupting it. Log concise actionable errors and continue only when safe.

OUTPUT CONTRACT FOR EACH EXPERIMENT
- Return exactly one fenced Python code block. Its first line is `# HYPOTHESIS: ...`, stating the single change and why it may help.
- The standalone script must read paths and task-specific configuration from the supplied task materials/user message, inspect actual columns/classes 
  instead of hardcoding them, run validation only on train, and print one final machine-readable line of the form `RESULT {"metric":"<task metric>","direction":"minimize|maximize","cv_mean":<float>,"cv_std":<float>,"runtime_seconds":<float>}`. 
  If the task requires a different metric representation, follow the task and document it. Never print a made-up or unmeasured score.
- Save a valid submission.csv in the exact requested location and format, including IDs and row order from the task files. Save OOF/test prediction arrays 
  only when paths are supplied and their alignment is unambiguous; document their row order. Do not overwrite the best usable output until the candidate has 
  completed and passed structural checks.
- Do not assume classification/regression, column names, class labels, metric, CV scheme, model availability or output format from another task; 
  derive each from the current task statement and files.

"""


METRIC_HINTS = {
    "macro_f1": (
        "Metric: Macro F1, higher is better. Use StratifiedKFold and class weights. After CV, tune per-class "
        "probability multipliers on the OOF predictions to maximize Macro F1. Rare classes weigh as much as frequent ones."
    ),
    "rmse": (
        "Metric: RMSE, higher is worse. Compare the raw target with log1p or sqrt transforms, but always score RMSE "
        "on the original scale. Inspect target outliers. Clip predictions to the train target range."
    ),
}


def load_local_env() -> None:
    # Import only when installed: the offline check also works before setup.
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)


def _metric_hint(task_text: str) -> str:
    """Return a hint only when the task materials explicitly identify that metric."""
    normalized = " ".join(str(task_text).lower().replace("-", " ").split())
    if "macro f1" in normalized or "macro f1 score" in normalized:
        return METRIC_HINTS["macro_f1"]
    if "rmse" in normalized or "root mean squared error" in normalized:
        return METRIC_HINTS["rmse"]
    return ""


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

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=60, max_retries=0,
    )


def check_api() -> bool:
    """Explicitly probe chat, tool-call, tool-result and continuation support."""
    missing = [name for name in API_FIELDS if not os.environ.get(name, "").strip()]
    if missing:
        print(f"API diagnostic: missing settings: {', '.join(missing)}")
        return False

    try:
        client_context = make_client()
    except Exception as error:
        print(f"API diagnostic: client configuration failed ({_safe_error(error)}).")
        return False

    try:
        with client_context as client:
            _check_text_response(client)
    except Exception as error:
        category = _classify_api_error(error)
        print(f"API diagnostic: {category}: {_safe_error(error)}")
        return False

    print("API diagnostic: text response verified.")
    try:
        with make_client() as client:
            _check_tool_protocol(client)
    except Exception as error:
        category = _classify_api_error(error)
        print(f"API diagnostic: {category}: {_safe_error(error)}")
        return False

    print("API diagnostic: success; text response and run_python tool protocol verified.")
    return True


def _safe_error(error: Exception) -> str:
    """Return a bounded diagnostic without exposing configured credentials."""
    message = str(error)
    for name in API_FIELDS:
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    return " ".join(message.split())[:400] or type(error).__name__


def _classify_api_error(error: Exception) -> str:
    try:
        import httpx
        if isinstance(error, httpx.TimeoutException):
            return "timeout"
    except ImportError:
        pass

    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    if status == 401 or status == 403 or "authentication" in name or "permission" in name:
        return "authorization rejected"
    if status == 404 or status == 400 or "notfound" in name or "badrequest" in name:
        return "model unavailable or invalid API address/request"
    if status == 408 or "timeout" in name:
        return "timeout"
    if isinstance(status, int) and status >= 500:
        return "server error"
    if isinstance(status, int):
        return f"API request failed (HTTP {status})"
    if isinstance(error, (TimeoutError,)):
        return "timeout"
    if "connection" in name or "connecterror" in name:
        return "invalid API address or connection failure"
    return "API diagnostic failed"


def _check_text_response(client) -> None:
    response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[{"role": "user", "content": "Reply with exactly: API_DIAGNOSTIC_OK"}],
        max_tokens=32,
        timeout=20,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("invalid response: no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty or invalid response: no text content")
    if content.strip() != "API_DIAGNOSTIC_OK":
        raise ValueError("invalid response: unexpected diagnostic text")


def _check_tool_protocol(client) -> None:
    diagnostic_result = {
        "stdout": "API_DIAGNOSTIC_TOOL_OK",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
    }
    first = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": (
                "For this diagnostic, call run_python exactly once with code that prints "
                "API_DIAGNOSTIC_TOOL_OK. Do not answer in text instead."
            )},
            {"role": "user", "content": "Check the run_python tool protocol."},
        ],
        tools=[TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
        max_tokens=128,
        timeout=20,
    )
    choices = getattr(first, "choices", None) or []
    if not choices:
        raise ValueError("invalid tool response: no choices")
    assistant = getattr(choices[0], "message", None)
    calls = getattr(assistant, "tool_calls", None) or []
    if len(calls) != 1:
        raise ValueError("tool protocol failed: expected exactly one tool call")
    call = calls[0]
    function = getattr(call, "function", None)
    if getattr(function, "name", None) != "run_python":
        raise ValueError("tool protocol failed: unexpected tool name")
    try:
        arguments = json.loads(getattr(function, "arguments", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("tool protocol failed: invalid tool arguments") from error
    if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
        raise ValueError("tool protocol failed: invalid run_python arguments")

    tool_call_id = getattr(call, "id", None)
    if not tool_call_id:
        raise ValueError("tool protocol failed: missing tool call id")
    second = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": (
                "For this diagnostic, after receiving the run_python result, reply with exactly: "
                "API_DIAGNOSTIC_CONTINUED"
            )},
            {"role": "user", "content": "Check the run_python tool protocol."},
            {
                "role": "assistant",
                "content": getattr(assistant, "content", None),
                "tool_calls": [call.model_dump(exclude_none=True)],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(diagnostic_result),
            },
        ],
        tools=[TOOL],
        parallel_tool_calls=False,
        max_tokens=32,
        timeout=20,
    )
    choices = getattr(second, "choices", None) or []
    if not choices:
        raise ValueError("invalid continuation response: no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or content.strip() != "API_DIAGNOSTIC_CONTINUED":
        raise ValueError("tool protocol failed: continuation response was empty or unexpected")


def build_user_prompt(task_text, profile, history, best_code, minutes_left, exp_limit_min, failed_in_row):
    rows = "\n".join(
        f"#{h['id']} | {h['hypothesis'][:90]} | cv={h['cv_mean']:.5f} | {h['status']} | {h['minutes']:.1f} min"
        for h in history[-12:]
    ) or "no experiments yet"
    hint = _metric_hint(task_text)
    shift = (
        "The last 2 hypotheses did not help: change direction (another model family or another feature group)."
        if failed_in_row >= 2 else ""
    )
    return (
        f"TASK\n{task_text[:6000]}\n\n"
        f"DATA PROFILE\n{profile[:3000]}\n\n"
        f"{hint}\n\n"
        f"EXPERIMENT LOG\n{rows}\n\n"
        f"BEST SCRIPT SO FAR\n<best_code>\n{best_code}\n</best_code>\n\n"
        f"Time left: {minutes_left:.0f} min. Limit for one experiment: {exp_limit_min} min. {shift}\n"
        "Write the next experiment. Change one thing and keep the output contract."
    )
