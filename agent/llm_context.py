"""Opt-in, pure preparation of the existing v3 INNER dialogue.

Not connected to research_v3 by default. This module never reads a state/report,
files, environment, answers or model outputs beyond the supplied messages. The
caller must supply the already-redacted, pre-freeze v3 dialogue. This is not an
outer-data detector or a public-report sanitizer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby
import json
import math


FIRST_TRIAL_HINT = (
    "For the next useful run_python call, write ONE small executable candidate "
    "and its settings: fresh research_candidate.py, research_recipe.json and "
    "research_hypothesis.json together. Define "
    "fit_predict(X_fit, y_fit, X_predict, context), using only its supplied "
    "training arguments and the declared allowed inputs. Use the schema and "
    "INNER feedback already supplied; if an essential schema fact is missing, "
    "inspect only that fact first. The host measures the INNER trial. "
    "A textual plan or a successful tool process is not a measured experiment. "
    "Do not repeat a failed approach unchanged."
)
LAST_TRIAL_HINT = (
    " Only one or two API attempts remain: prefer that small feasible trial "
    "to further broad analysis. Preserve the host's finalization reserve."
)


@dataclass(frozen=True)
class PreparedContext:
    messages: list[dict]
    summary: dict


def _bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _validate_pairs(messages):
    """Refuse to prepare an incomplete/mismatched tool transaction."""
    pending, seen = set(), set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Context messages must be dictionaries")
        role = message.get("role")
        if role == "tool":
            identifier = message.get("tool_call_id")
            if not isinstance(identifier, str) or identifier not in pending:
                raise ValueError("Tool result has no pending call")
            pending.remove(identifier)
            continue
        if pending:
            raise ValueError("Tool call is missing its result")
        calls = message.get("tool_calls")
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ValueError("Tool calls require an assistant message")
            for call in calls:
                identifier = call.get("id") if isinstance(call, dict) else None
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ValueError("Tool call IDs must be non-empty and unique")
                pending.add(identifier)
                seen.add(identifier)
    if pending:
        raise ValueError("Tool call is missing its result")


def _tool_result(message):
    if message.get("role") != "tool" or not isinstance(message.get("content"), str):
        return None
    try:
        value = json.loads(message["content"])
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    selection = value.get("selection")
    if isinstance(selection, dict) and selection.get("scope") != "inner_selection":
        raise ValueError("Only host INNER selection results belong in research context")
    return value


def _verified_inner(result):
    selection = result.get("selection")
    if not isinstance(selection, dict) or selection.get("verified") is not True:
        return False
    metric = selection.get("metric")
    if not isinstance(metric, dict):
        return False
    value = metric.get("value")
    return (selection.get("scope") == "inner_selection"
            and isinstance(metric.get("name"), str) and bool(metric["name"])
            and type(value) in {int, float} and math.isfinite(value)
            and type(metric.get("higher_is_better")) is bool)


def _compact_repeated_lines(value):
    """Keep every distinct line and annotate consecutive exact repetitions.

    Unlike a head/tail truncation, this keeps all unique facts. It runs only on
    stdout from a successful, host-verified INNER trial; stderr and failure
    details are never shortened. A replacement must actually save UTF-8 bytes.
    """
    output, changed_runs = [], 0
    for line, lines in groupby(value.splitlines(keepends=True)):
        count = sum(1 for _ in lines)
        original = line * count
        replacement = line + ("" if line.endswith(("\n", "\r")) else "\n")
        replacement += f"[host context: previous line repeated {count - 1} additional times]\n"
        if count > 1 and len(replacement.encode("utf-8")) < len(original.encode("utf-8")):
            output.append(replacement)
            changed_runs += 1
        else:
            output.append(original)
    return "".join(output), changed_runs


def compact_research_messages(messages, *, remaining_requests,
                              encourage_first_trial=True):
    """Copy a v3 dialogue, compact safe repetitions and add at most one hint.

    No messages, tool-call arguments, recipes, diagnostics, system constraints,
    unique stdout or errors are dropped. A host-verified INNER selection must
    exist before any stdout is compacted. Unknown/malformed tool contents are
    retained unchanged. Pair or explicit selection-scope violations raise.

    Use on the accumulated dialogue BEFORE appending the fresh host budget and
    BEFORE the existing request/token/deadline guards on every real attempt.
    Do not persist the returned hint into the accumulated dialogue. Metrics in
    ``summary`` are payload-size observations, not provider usage or ML quality.
    """
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a sequence")
    if type(remaining_requests) is not int or remaining_requests < 0:
        raise ValueError("remaining_requests must be a non-negative integer")
    if type(encourage_first_trial) is not bool:
        raise ValueError("encourage_first_trial must be a boolean")
    _validate_pairs(messages)
    original_bytes = _bytes(messages)
    prepared = deepcopy(list(messages))
    verified = False
    runs = 0
    for message in prepared:
        result = _tool_result(message)
        if result is None or not _verified_inner(result):
            continue
        verified = True
        if (result.get("exit_code") != 0 or result.get("error")
                or result.get("timed_out") or result.get("memory_limit_exceeded")
                or result.get("candidate_execution") is not None
                or not isinstance(result.get("stdout"), str)):
            continue
        stdout, count = _compact_repeated_lines(result["stdout"])
        if count:
            result["stdout"] = stdout
            message["content"] = json.dumps(result, ensure_ascii=False, allow_nan=False,
                                             separators=(",", ":"))
            runs += count
    hint_added = bool(encourage_first_trial and remaining_requests and not verified)
    if hint_added:
        hint = FIRST_TRIAL_HINT + (LAST_TRIAL_HINT if remaining_requests <= 2 else "")
        # Also safe if a caller accidentally re-prepares an already prepared
        # list: do not grow the context by appending identical hints.
        if not any(message.get("role") == "user" and message.get("content") == hint
                   for message in prepared):
            prepared.append({"role": "user", "content": hint})
        else:
            hint_added = False
    return PreparedContext(prepared, {
        "policy": "inner_repetition_v1", "original_bytes": original_bytes,
        "prepared_bytes": _bytes(prepared), "stdout_runs_compacted": runs,
        "first_trial_hint_added": hint_added, "verified_inner_seen": verified,
    })
