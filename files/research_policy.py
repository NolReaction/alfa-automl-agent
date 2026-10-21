"""Bounded dialogue policy for the Qwen research loop. Owner: Nika.

The host keeps the full, cleaned history in files. This module builds a bounded
*projection* of it for the next model request, compacts tool results into a small
feedback record and chooses the instruction for the next step. It never replaces
the host journal, the metric recomputation or the acceptance of results.

Contract (v1, agreed with Dima):

* pure and deterministic: no files, environment, clock, network or Python
  execution; inputs are never mutated;
* secrets are removed first (``redact``), then text is shortened, then the size is
  checked; sizes are UTF-8 bytes of compact JSON (``ensure_ascii=False``), not
  tokens and not money;
* host facts (``selection``, ``snapshot``, ``contract``) are the only source of
  ``accepted`` / ``verified`` / metric values; text printed by the model never
  changes them and is never summarised as "verified";
* v3 note: the policy does not guess where a metric comes from. The host puts
  only inner scores into the research snapshot and names their origin in
  ``contract["validation"]["scope"]``. Diagnostics are forwarded exactly as the
  host supplies them, so the host must not supply outer-holdout diagnostics.

``compact_tool_feedback`` keeps (allowlist): ``exit_code``, ``timed_out``,
``memory_limit_exceeded``, ``elapsed_seconds``, ``selection`` (``accepted``,
``verified``, ``metric``, ``reason``, ``hypothesis``, ``diagnostics``), ``error``,
``output_check`` (``valid``, first errors), ``previous_output_restored``, a small
numeric subset of ``resources``, ``stderr`` and ``stdout``. Unknown fields are
dropped and only counted. Added service fields: ``feedback_version`` (1),
``truncated`` (field name -> original size in bytes, also for fields dropped
entirely) and ``dropped_fields`` (number of dropped unknown fields).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math

__all__ = [
    "DECISION_ACTIONS", "DECISION_STAGES", "ERROR_KINDS", "FAILURE_STOP_STREAK",
    "LOW_REMAINING_REQUESTS", "LOW_REMAINING_SECONDS", "REASON_CODES",
    "PolicyBudgetExceeded", "PolicyContext", "PolicyDecision", "PolicyInputError",
    "PolicyLimits", "build_model_messages", "compact_tool_feedback",
    "decide_next_action", "json_size",
]

SNAPSHOT_STAGES = ("inspect", "research", "finalize")
DECISION_STAGES = ("inspect", "experiment", "repair", "finalize", "finished")
DECISION_ACTIONS = ("continue", "complete", "fallback", "stop")
ERROR_KINDS = ("execution", "timeout", "memory", "evidence", "repeated_failure", "api", "unknown")
REASON_CODES = (
    "inspect_without_baseline", "inspect_with_baseline", "first_hypothesis",
    "improved_continue", "not_better_continue", "unverified_continue",
    "repeated_code", "finalize_requested", "ready_to_finish", "finalize_low_resources",
    "repair_execution_error", "repair_consecutive_failures", "repair_resource_limit",
    "repair_evidence_contract", "repair_repeated_failure", "repair_unknown_error",
    "no_verified_result", "retry_after_api_error", "finish_too_early",
    "contract_fulfilled", "unverified_completion", "contract_unreachable",
    "resources_exhausted_keep_result", "resources_exhausted_no_result",
    "repeated_failures_keep_result", "repeated_failures_no_result",
)

# Advisory thresholds. The host keeps its own deadline and failure guard.
LOW_REMAINING_SECONDS = 300.0
LOW_REMAINING_REQUESTS = 1
FAILURE_STOP_STREAK = 3

_MAX_JSON_DEPTH = 12
_MAX_ASSISTANT_TEXT = 2048        # bytes kept from assistant text next to tool calls
_MAX_PLAIN_TEXT = 512             # bytes kept from the trailing plain assistant message
_TRIAL_HYPOTHESIS_CHARS = 240
_LATEST_HYPOTHESIS_CHARS = 500
_TRIAL_REASON_CHARS = 120
_LATEST_ERROR_BYTES = (0, 300, 1200)  # indexed by detail level 0/1/2


class PolicyInputError(ValueError):
    """The input violates the v1 interface; the message never echoes input values."""


class PolicyBudgetExceeded(ValueError):
    """A mandatory part does not fit into the requested limit."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


@dataclass(frozen=True)
class PolicyLimits:
    max_messages_bytes: int = 64 * 1024
    max_tool_feedback_bytes: int = 6 * 1024
    max_recent_trials: int = 6
    max_recent_tool_groups: int = 2

    def __post_init__(self):
        for name, minimum in (("max_messages_bytes", 1), ("max_tool_feedback_bytes", 1),
                              ("max_recent_trials", 0), ("max_recent_tool_groups", 0)):
            value = getattr(self, name)
            if not _is_int(value) or value < minimum:
                raise PolicyInputError(f"limits.{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class PolicyDecision:
    stage: str
    action: str
    instruction: str
    reason: str


@dataclass(frozen=True)
class PolicyContext:
    messages: list[dict]
    diagnostics: dict


# --------------------------------------------------------------------------- sizes

def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def json_size(value) -> int:
    """UTF-8 size of the compact JSON form used for every limit in this module."""
    return len(_dump(value).encode("utf-8"))


def _body_size(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _cut(text: str, keep: int, mode: str, original: int) -> str:
    if mode == "tail":
        return f"[…cut: kept the last part of {original} bytes]\n" + (text[-keep:] if keep else "")
    if mode == "head":
        return (text[:keep] if keep else "") + f"\n[…cut: kept the first part of {original} bytes]"
    head = keep // 3
    tail = keep - head
    return (text[:head] + f"\n[…cut: {original} bytes in total…]\n"
            + (text[-tail:] if tail else ""))


def _fit_body(text: str, cap: int, mode: str) -> tuple[str, bool]:
    """Largest piece of ``text`` whose JSON string body is at most ``cap`` bytes."""
    if _body_size(text) <= cap:
        return text, False
    original = len(text.encode("utf-8"))
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        piece = _cut(text, middle, mode, original)
        if _body_size(piece) <= cap:
            best, low = piece, middle + 1
        else:
            high = middle - 1
    return best, True


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:max(0, limit - 1)] + "…"


# ------------------------------------------------------------------- cleaning

def _sanitize(text: str) -> str:
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:  # lone surrogates cannot be sent to a provider
        return text.encode("utf-8", errors="replace").decode("utf-8")


def _check_redact(redact):
    if not callable(redact):
        raise PolicyInputError("redact must be a function str -> str")
    return redact


def _clean_text(value: str, redact) -> str:
    text = _sanitize(value)
    try:
        cleaned = redact(text)
    except Exception:  # the original error may quote the secret
        raise PolicyInputError("redact raised an error") from None
    if not isinstance(cleaned, str):
        raise PolicyInputError("redact must return str")
    return _sanitize(cleaned)


def _clean(value, redact):
    if isinstance(value, str):
        return _clean_text(value, redact)
    if isinstance(value, Mapping):
        return {_clean_text(key, redact): _clean(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item, redact) for item in value]
    return value


def _reject_constant(_name):
    raise ValueError("non-finite number")


def _clean_arguments(arguments: str, redact) -> str:
    """Redact decoded strings so the result stays valid JSON; never shorten it."""
    text = _sanitize(arguments)
    try:
        parsed = json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        return _clean_text(arguments, redact)  # already malformed; keep it as it was
    cleaned = _clean(parsed, redact)
    if cleaned == parsed and text == arguments:
        return arguments  # nothing to remove: keep the executed text byte for byte
    return _dump(cleaned)


# ----------------------------------------------------------------- validation

def _fail(path: str, problem: str):
    raise PolicyInputError(f"{path}: {problem}")


def _mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _need(mapping: Mapping, key: str, path: str):
    if key not in mapping:
        _fail(f"{path}.{key}", "is required")
    return mapping[key]


def _get_int(mapping, key, path, *, minimum=0, required=True, default=None):
    if key not in mapping and not required:
        return default
    value = _need(mapping, key, path)
    if not _is_int(value) or value < minimum:
        _fail(f"{path}.{key}", f"must be an integer >= {minimum}")
    return value


def _get_bool(mapping, key, path, *, required=True, default=False):
    if key not in mapping and not required:
        return default
    value = _need(mapping, key, path)
    if not isinstance(value, bool):
        _fail(f"{path}.{key}", "must be a boolean")
    return value


def _get_str(mapping, key, path, *, nonempty=False):
    value = _need(mapping, key, path)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        _fail(f"{path}.{key}", "must be a nonempty string" if nonempty else "must be a string")
    return value


def _get_opt_str(mapping, key, path):
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        _fail(f"{path}.{key}", "must be a string or null")
    return value


def _json_ok(value, path: str, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        _fail(path, "is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "keys must be strings")
            _json_ok(item, path + "[]", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_ok(item, path + "[]", depth + 1)
        return
    _fail(path, "must be JSON-compatible")


def _json_object(value, path: str) -> Mapping:
    _mapping(value, path)
    _json_ok(value, path)
    return value


def _metric(value, path: str) -> dict:
    mapping = _mapping(value, path)
    name = _need(mapping, "name", path)
    number = _need(mapping, "value", path)
    if not isinstance(name, str):
        _fail(path + ".name", "must be a string")
    if not _is_number(number):
        _fail(path + ".value", "must be a finite number")
    return {"name": name, "value": number,
            "higher_is_better": _get_bool(mapping, "higher_is_better", path)}


def _opt_metric(mapping: Mapping, key: str, path: str):
    value = mapping.get(key)
    return None if value is None else _metric(value, f"{path}.{key}")


def _trial(value, path: str) -> dict:
    mapping = _mapping(value, path)
    kind = mapping.get("error_kind")
    if kind is not None and kind not in ERROR_KINDS:
        _fail(path + ".error_kind", "is not a known error kind")
    trial = {
        "number": _get_int(mapping, "number", path, minimum=1),
        "code_hash": _get_opt_str(mapping, "code_hash", path),
        "hypothesis": _get_opt_str(mapping, "hypothesis", path),
        "accepted": _get_bool(mapping, "accepted", path),
        "verified": _get_bool(mapping, "verified", path),
        "metric": _opt_metric(mapping, "metric", path),
        "reason": _get_opt_str(mapping, "reason", path),
        "error_kind": kind,
        "error": _get_opt_str(mapping, "error", path),
    }
    for key in ("diagnostics", "resources"):
        if mapping.get(key) is not None:
            trial[key] = _json_object(mapping[key], f"{path}.{key}")
    return trial


def _snapshot(value) -> dict:
    path = "snapshot"
    mapping = _mapping(value, path)
    if _get_int(mapping, "version", path) != 1:
        _fail(path + ".version", "unsupported version")
    stage = _get_str(mapping, "stage", path)
    if stage not in SNAPSHOT_STAGES:
        _fail(path + ".stage", "is not a known stage")
    budget = _mapping(_need(mapping, "budget", path), path + ".budget")
    seconds = _need(budget, "remaining_seconds", path + ".budget")
    if not _is_number(seconds) or seconds < 0:
        _fail(path + ".budget.remaining_seconds", "must be a finite non-negative number")
    baseline = _mapping(_need(mapping, "baseline", path), path + ".baseline")
    counts = _mapping(_need(mapping, "counts", path), path + ".counts")
    trials = _need(mapping, "recent_trials", path)
    if isinstance(trials, (str, bytes)) or not isinstance(trials, (list, tuple)):
        _fail(path + ".recent_trials", "must be a list")
    last = mapping.get("last_trial")
    return {
        "version": 1, "stage": stage,
        "budget": {"remaining_seconds": seconds,
                   "remaining_requests": _get_int(budget, "remaining_requests", path + ".budget"),
                   "can_request": _get_bool(budget, "can_request", path + ".budget")},
        "baseline": {"status": _get_str(baseline, "status", path + ".baseline"),
                     "retained_output_valid": _get_bool(baseline, "retained_output_valid",
                                                        path + ".baseline"),
                     "best_verified_metric": _opt_metric(baseline, "best_verified_metric",
                                                         path + ".baseline")},
        "last_trial": None if last is None else _trial(last, path + ".last_trial"),
        "recent_trials": [_trial(item, path + ".recent_trials[]") for item in trials],
        "counts": {name: _get_int(counts, name, path + ".counts") for name in (
            "verified_trials", "unverified_trials", "consecutive_execution_failures")},
        "finish_requested": _get_bool(mapping, "finish_requested", path),
    }


def _contract(value) -> dict:
    path = "contract"
    mapping = _mapping(value, path)
    if _get_int(mapping, "version", path) != 1:
        _fail(path + ".version", "unsupported version")
    inputs = _need(mapping, "allowed_inputs", path)
    outputs = _need(mapping, "required_outputs", path)
    for name, items in (("allowed_inputs", inputs), ("required_outputs", outputs)):
        if isinstance(items, (str, bytes)) or not isinstance(items, (list, tuple)):
            _fail(f"{path}.{name}", "must be a list")
    validation = _mapping(_need(mapping, "validation", path), path + ".validation")
    completion = _mapping(_need(mapping, "completion", path), path + ".completion")
    allowed = []
    for item in inputs:
        item = _mapping(item, path + ".allowed_inputs[]")
        allowed.append({"path": _get_str(item, "path", path + ".allowed_inputs[]", nonempty=True),
                        "purpose": _get_str(item, "purpose", path + ".allowed_inputs[]")})
    required = []
    for item in outputs:
        item = _mapping(item, path + ".required_outputs[]")
        columns = _need(item, "columns", path + ".required_outputs[]")
        if (isinstance(columns, (str, bytes)) or not isinstance(columns, (list, tuple))
                or any(not isinstance(column, str) for column in columns)):
            _fail(path + ".required_outputs[].columns", "must be a list of strings")
        required.append({"path": _get_str(item, "path", path + ".required_outputs[]", nonempty=True),
                         "columns": list(columns)})
    return {
        "version": 1,
        "manifest_version": _get_int(mapping, "manifest_version", path, minimum=1),
        "manifest_path": _get_str(mapping, "manifest_path", path, nonempty=True),
        "host_instructions": _get_str(mapping, "host_instructions", path),
        "allowed_inputs": allowed, "required_outputs": required,
        "validation": {"supported": _get_bool(validation, "supported", path + ".validation"),
                       "metric": _opt_metric(validation, "metric", path + ".validation"),
                       "scope": _get_str(validation, "scope", path + ".validation")},
        "completion": {"required_verified_trials": _get_int(
            completion, "required_verified_trials", path + ".completion", minimum=1),
            "allow_unverified": _get_bool(completion, "allow_unverified", path + ".completion")},
    }


def _check_limits(limits) -> PolicyLimits:
    if not isinstance(limits, PolicyLimits):
        raise PolicyInputError("limits must be a PolicyLimits")
    limits.__post_init__()
    return limits


# ------------------------------------------------------------ tool feedback

_RESULT_KEYS = frozenset({
    "stdout", "stderr", "exit_code", "timed_out", "memory_limit_exceeded", "elapsed_seconds",
    "resources", "error", "output_check", "selection", "previous_output_restored",
})


def _read_result(result: Mapping, redact) -> dict:
    path = "result"
    exit_code = result.get("exit_code")
    if exit_code is not None and not _is_int(exit_code):
        _fail(path + ".exit_code", "must be an integer or null")
    elapsed = result.get("elapsed_seconds")
    if elapsed is not None and (not _is_number(elapsed) or elapsed < 0):
        _fail(path + ".elapsed_seconds", "must be a finite non-negative number")
    texts = {}
    for key in ("stdout", "stderr", "error"):
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            _fail(f"{path}.{key}", "must be a string or null")
        texts[key] = None if value is None else _clean_text(value, redact)
    selection = result.get("selection")
    if selection is not None:
        selection = _mapping(selection, path + ".selection")
        metric = _opt_metric(selection, "metric", path + ".selection")
        diagnostics = selection.get("diagnostics")
        selection = {
            "accepted": _get_bool(selection, "accepted", path + ".selection", required=False),
            "verified": _get_bool(selection, "verified", path + ".selection", required=False),
            "metric": metric,
            "reason": _get_opt_str(selection, "reason", path + ".selection"),
            "hypothesis": _get_opt_str(selection, "hypothesis", path + ".selection"),
            "diagnostics": None if diagnostics is None else _json_object(
                diagnostics, path + ".selection.diagnostics"),
        }
        for key in ("reason", "hypothesis"):
            if selection[key] is not None:
                selection[key] = _clean_text(selection[key], redact)
        selection["metric"] = _clean(metric, redact)
        if selection["diagnostics"] is not None:
            selection["diagnostics"] = _clean(selection["diagnostics"], redact)
    check = result.get("output_check")
    if check is not None:
        check = _mapping(check, path + ".output_check")
        errors = check.get("errors", [])
        if (not isinstance(errors, (list, tuple)) or any(not isinstance(item, str) for item in errors)):
            _fail(path + ".output_check.errors", "must be a list of strings")
        check = {"valid": _get_bool(check, "valid", path + ".output_check"),
                 "errors": [_clean_text(item, redact) for item in errors]}
    resources = result.get("resources")
    if resources is not None:
        resources = _json_object(resources, path + ".resources")
    return {
        "exit_code": exit_code,
        "timed_out": _get_bool(result, "timed_out", path, required=False),
        "oom": _get_bool(result, "memory_limit_exceeded", path, required=False),
        "elapsed": elapsed, "selection": selection, "output_check": check,
        "restored": (_get_bool(result, "previous_output_restored", path)
                     if "previous_output_restored" in result else None),
        "resources": resources, **texts,
        "dropped": sum(1 for key in result if key not in _RESULT_KEYS),
    }


def _with(feedback: dict, key: str, value, sub: str | None = None, cut=None) -> dict:
    out = dict(feedback)
    if sub is None:
        out[key] = value
    else:
        inner = dict(out.get(key, {}))
        inner[sub] = value
        out[key] = inner
    if cut is not None:
        marks = dict(out.get("truncated", {}))
        marks[cut[0]] = cut[1]
        out["truncated"] = marks
    return out


def _add_text(feedback, budget, key, sub, text, *, cap, mode, label):
    """Add a text field, shortening it to the bytes that are still available."""
    if not text:
        return feedback
    current = json_size(feedback)
    room = min(budget - current, cap)
    full = _with(feedback, key, text, sub)
    if json_size(full) - current <= room:
        return full
    original = len(text.encode("utf-8"))
    low, high, best = 0, len(text), None
    while low <= high:
        middle = (low + high) // 2
        trial = _with(feedback, key, _cut(text, middle, mode, original), sub, (label, original))
        if json_size(trial) <= budget and json_size(trial) - current <= room:
            best, low = trial, middle + 1
        else:
            high = middle - 1
    if best is not None:
        return best
    marker = _with(feedback, "truncated", {**feedback.get("truncated", {}), label: original})
    return marker if json_size(marker) <= budget else feedback


def _add_choice(feedback, budget, key, sub, candidates, *, cap, label=None, original=None):
    """Add the richest candidate that fits; a reduced one is reported as truncated."""
    current = json_size(feedback)
    room = min(budget - current, cap)
    for index, candidate in enumerate(candidates):
        cut = (label, original) if index > 0 and label is not None else None
        trial = _with(feedback, key, candidate, sub, cut)
        if json_size(trial) <= budget and json_size(trial) - current <= room:
            return trial
    if label is not None and original is not None:
        marker = _with(feedback, "truncated", {**feedback.get("truncated", {}), label: original})
        if json_size(marker) <= budget:
            return marker
    return feedback


def _classification_summary(value: Mapping, worst: int):
    try:
        labels, matrix, per_class = value["labels"], value["confusion_matrix"], value["per_class"]
        size = len(labels)
        if not (isinstance(labels, list) and isinstance(matrix, list) and isinstance(per_class, list)
                and size == len(matrix) == len(per_class) and 0 < size <= 200):
            return None
        ranked = sorted((float(per_class[i]["f1"]), i) for i in range(size))
        pairs = sorted(((matrix[i][j], i, j) for i in range(size) for j in range(size)
                        if i != j and matrix[i][j] > 0), key=lambda item: (-item[0], item[1], item[2]))
        summary = {key: value[key] for key in ("macro_f1", "accuracy", "rows")
                   if _is_number(value.get(key))}
        summary["worst_classes"] = [
            {"label": _clip(str(labels[i]), 60), "f1": f1, "support": per_class[i].get("support")}
            for f1, i in ranked[:worst]]
        summary["top_confusions"] = [
            {"actual": _clip(str(labels[i]), 60), "predicted": _clip(str(labels[j]), 60), "count": count}
            for count, i, j in pairs[:worst]]
        return summary
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _reduce_json(value, depth: int, items: int, chars: int):
    if isinstance(value, str):
        return _clip(value, chars)
    if isinstance(value, Mapping):
        if depth <= 0:
            return "…"
        out = {key: _reduce_json(value[key], depth - 1, items, chars) for key in list(value)[:items]}
        if len(value) > items:
            out["…omitted_keys"] = len(value) - items
        return out
    if isinstance(value, (list, tuple)):
        if depth <= 0:
            return "…"
        out = [_reduce_json(item, depth - 1, items, chars) for item in list(value)[:items]]
        if len(value) > items:
            out.append(f"…+{len(value) - items} more")
        return out
    return value


def _diagnostics_levels(diagnostics: Mapping) -> list:
    """Richest first: the full value, then ever smaller summaries of the same facts."""
    levels = [diagnostics]
    for worst, groups, items, chars in ((5, 6, 8, 80), (3, 3, 6, 60), (2, 1, 4, 40), (1, 0, 2, 24)):
        reduced = {}
        for key, value in diagnostics.items():
            if key == "classification" and isinstance(value, Mapping):
                summary = _classification_summary(value, worst)
                reduced[key] = summary if summary is not None else _reduce_json(value, 3, items, chars)
            elif key == "groups" and isinstance(value, (list, tuple)):
                if groups:
                    reduced[key] = [
                        {name: _clip(item[name], chars) if isinstance(item[name], str) else item[name]
                         for name in ("feature", "group", "rows", "metric", "value") if name in item}
                        for item in value[:groups] if isinstance(item, Mapping)]
            else:
                reduced[key] = _reduce_json(value, 3, items, chars)
        levels.append(reduced)
    return levels


def _assemble_feedback(facts: dict, budget: int) -> dict:
    feedback = {"feedback_version": 1, "exit_code": facts["exit_code"],
                "timed_out": facts["timed_out"], "memory_limit_exceeded": facts["oom"]}
    selection = facts["selection"]
    if selection is not None:
        feedback["selection"] = {"accepted": selection["accepted"], "verified": selection["verified"],
                                 "metric": selection["metric"]}
    if json_size(feedback) > budget:
        raise PolicyBudgetExceeded("the mandatory tool feedback does not fit into the limit")
    spare = budget - json_size(feedback)
    failed = (facts["error"] is not None or facts["exit_code"] != 0
              or facts["timed_out"] or facts["oom"])

    feedback = _add_text(feedback, budget, "error", None, facts["error"],
                         cap=min(1500, spare * 30 // 100), mode="tail", label="error")
    if selection is not None:
        feedback = _add_text(feedback, budget, "selection", "reason", selection["reason"],
                             cap=min(400, spare * 10 // 100), mode="head", label="selection.reason")
        feedback = _add_text(feedback, budget, "selection", "hypothesis", selection["hypothesis"],
                             cap=min(600, spare * 12 // 100), mode="head", label="selection.hypothesis")
    for key, value in (("elapsed_seconds", facts["elapsed"]),
                       ("previous_output_restored", facts["restored"])):
        if value is not None:
            feedback = _add_choice(feedback, budget, key, None, [value], cap=60)
    check = facts["output_check"]
    if check is not None:
        errors = [_clip(item, 160) for item in check["errors"][:3]]
        candidates = [{"valid": check["valid"]}]
        if errors:
            extra = {"more": len(check["errors"]) - 3} if len(check["errors"]) > 3 else {}
            candidates.insert(0, {"valid": check["valid"], "errors": errors, **extra})
        feedback = _add_choice(feedback, budget, "output_check", None, candidates,
                               cap=min(600, spare * 10 // 100))
    if selection is not None and selection["diagnostics"] is not None:
        diagnostics = selection["diagnostics"]
        feedback = _add_choice(
            feedback, budget, "selection", "diagnostics", _diagnostics_levels(diagnostics),
            cap=spare * 25 // 100, label="selection.diagnostics", original=json_size(diagnostics))
    if facts["resources"] is not None:
        subset = {name: facts["resources"][name] for name in
                  ("peak_rss_bytes", "cpu_seconds", "memory_limit_bytes")
                  if _is_number(facts["resources"].get(name))}
        if subset:
            feedback = _add_choice(feedback, budget, "resources", None, [subset],
                                   cap=min(200, spare * 5 // 100))
    room = budget - json_size(feedback)
    feedback = _add_text(feedback, budget, "stderr", None, facts["stderr"],
                         cap=room * (45 if failed else 15) // 100, mode="tail", label="stderr")
    feedback = _add_text(feedback, budget, "stdout", None, facts["stdout"],
                         cap=budget - json_size(feedback), mode="headtail", label="stdout")
    if facts["dropped"]:
        feedback = _add_choice(feedback, budget, "dropped_fields", None, [facts["dropped"]], cap=60)
    if "truncated" in feedback:  # service field last, for readability
        feedback["truncated"] = feedback.pop("truncated")
    if json_size(feedback) > budget:
        raise PolicyBudgetExceeded("the tool feedback does not fit into the limit")
    return feedback


def compact_tool_feedback(result: Mapping, *, limits: PolicyLimits,
                          redact: Callable[[str], str]) -> dict:
    """Bounded, allowlisted feedback for one tool result; host facts are copied unchanged."""
    _check_limits(limits)
    redact = _check_redact(redact)
    if not isinstance(result, Mapping):
        _fail("result", "must be an object")
    return _assemble_feedback(_read_result(result, redact), limits.max_tool_feedback_bytes)


def _text_feedback(text: str, budget: int, redact) -> dict:
    feedback = {"feedback_version": 1}
    feedback = _add_text(feedback, budget, "unstructured", None, _clean_text(text, redact),
                         cap=budget, mode="headtail", label="unstructured")
    if "truncated" in feedback:
        feedback["truncated"] = feedback.pop("truncated")
    return feedback


# ---------------------------------------------------------------- decisions

def _outputs_text(contract: Mapping) -> str:
    paths = [item["path"] for item in contract["required_outputs"]]
    return ", ".join(paths) if paths else "the outputs named in the host instructions"


def _inputs_text(contract: Mapping) -> str:
    paths = [item["path"] for item in contract["allowed_inputs"]]
    return ", ".join(paths) if paths else "the inputs named in the host instructions"


def _normalized(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _repeats_earlier(snapshot: Mapping, last: Mapping) -> bool:
    for trial in snapshot["recent_trials"]:
        if trial["number"] >= last["number"]:
            continue
        if last["code_hash"] is not None and trial["code_hash"] == last["code_hash"]:
            return True
        if _normalized(last["hypothesis"]) and _normalized(last["hypothesis"]) == _normalized(trial["hypothesis"]):
            return True
    return False


def _contract_met(snapshot: Mapping, contract: Mapping) -> bool:
    counts, completion = snapshot["counts"], contract["completion"]
    if contract["validation"]["supported"]:
        return counts["verified_trials"] >= completion["required_verified_trials"]
    return completion["allow_unverified"] and counts["unverified_trials"] >= 1


def _missing(snapshot: Mapping, contract: Mapping) -> str:
    counts, parts = snapshot["counts"], []
    if not snapshot["baseline"]["retained_output_valid"]:
        parts.append("no valid output is retained")
    if contract["validation"]["supported"]:
        needed = contract["completion"]["required_verified_trials"]
        if counts["verified_trials"] < needed:
            parts.append(f"{needed - counts['verified_trials']} more host-verified experiment(s) needed "
                         f"({counts['verified_trials']} of {needed} done)")
    elif counts["unverified_trials"] < 1:
        parts.append("at least one experiment with a valid output is needed")
    return "; ".join(parts) or "the host contract is not met"


def _is_low(snapshot: Mapping) -> bool:
    budget = snapshot["budget"]
    return (budget["remaining_seconds"] < LOW_REMAINING_SECONDS
            or budget["remaining_requests"] <= LOW_REMAINING_REQUESTS)


_BASE_STAGE = {"inspect": "inspect", "research": "experiment", "finalize": "finalize"}


def _repair(snapshot: Mapping, contract: Mapping, last: Mapping) -> PolicyDecision:
    kind = last["error_kind"]
    streak = snapshot["counts"]["consecutive_execution_failures"]
    outputs = _outputs_text(contract)
    resubmit = (f"Re-run with all required evidence ({outputs}) written in the same tool call, "
                "as the host instructions require.")
    if kind == "api":
        return PolicyDecision(_BASE_STAGE[snapshot["stage"]], "continue",
                              "The previous request failed on the provider side; no experiment state was "
                              "lost. Continue from the current state shown above.", "retry_after_api_error")
    if kind in ("timeout", "memory"):
        seconds = int(snapshot["budget"]["remaining_seconds"])
        what = "time" if kind == "timeout" else "memory"
        return PolicyDecision("repair", "continue",
                              f"The last run hit the {what} limit. Reduce its cost: fewer rows for exploratory "
                              "fits, smaller models, fewer folds or trees, compact dtypes, at most 8 CPU "
                              f"threads. About {seconds}s remain; size the run to finish well inside that "
                              f"and keep any final refit cheap. {resubmit}", "repair_resource_limit")
    if kind == "evidence":
        return PolicyDecision("repair", "continue",
                              "The host could not accept the evidence from the last run (see last_trial.reason). "
                              f"The contract requires: {outputs}, each with exactly the columns listed in "
                              f"required_outputs, in the row order given by {contract['manifest_path']}. "
                              "Do not alter inputs and do not guess missing details: read only the needed keys "
                              f"of {contract['manifest_path']}. {resubmit}", "repair_evidence_contract")
    if kind == "repeated_failure":
        return PolicyDecision("repair", "continue",
                              "Identical code already failed. Submitting it again, or a cosmetic edit of it, is "
                              "not a new idea. Diagnose the cause with a short check (print only the relevant "
                              f"shapes, dtypes or messages), then change the approach itself. {resubmit}",
                              "repair_repeated_failure")
    if kind in ("execution", "unknown") and streak >= 2:
        return PolicyDecision("repair", "continue",
                              f"{streak} runs failed in a row. Stop patching the same pipeline: reduce to the "
                              "simplest pipeline that already produced a valid result (or a plain baseline), "
                              f"verify each step with a short check, then add one change at a time. {resubmit}",
                              "repair_consecutive_failures")
    if kind == "unknown":
        return PolicyDecision("repair", "continue",
                              "The last run reported an unclassified error (see last_trial.error). Rely only on "
                              "what the message states, make the smallest safe change and test it with a short "
                              f"check before the full experiment. Do not guess a cause. {resubmit}",
                              "repair_unknown_error")
    return PolicyDecision("repair", "continue",
                          "The last run failed (see last_trial.error). Fix that specific cause with the smallest "
                          "change and keep the rest of the approach. Unchanged failed code is blocked by the "
                          f"host, so the code must differ. {resubmit}", "repair_execution_error")


def _next(snapshot: Mapping, contract: Mapping) -> PolicyDecision:
    last, stage = snapshot["last_trial"], snapshot["stage"]
    retained = snapshot["baseline"]["retained_output_valid"]
    outputs = _outputs_text(contract)
    if last is not None and last["error_kind"] is not None:
        return _repair(snapshot, contract, last)
    if stage == "finalize":
        done = _contract_met(snapshot, contract) and retained
        text = ("Finalize: refit the selected pipeline only as far as the host instructions allow, write the "
                f"final required outputs ({outputs}), re-check columns, row order and missing values, then "
                "reply with a short factual account and no tool call.")
        if done:
            return PolicyDecision("finalize", "continue", text, "ready_to_finish")
        return PolicyDecision("finalize", "continue", text, "finalize_requested")
    if _is_low(snapshot):
        return PolicyDecision(
            "finalize", "continue",
            "Resources are nearly exhausted: stop exploring. Make sure the best valid output is saved "
            f"({outputs}), re-check its columns and row order, then reply with a short factual account and "
            "no tool call.", "finalize_low_resources")
    if stage == "inspect" and last is not None:  # inspection ran: move on instead of inspecting again
        if retained:
            return PolicyDecision(
                "experiment", "continue",
                "Inspection is done; do not inspect again. Test one concrete hypothesis that differs from the "
                "starting result (another model family, regularization, fold-fitted features or a fixed blend), "
                f"fitting only on the declared fit data, and write all required evidence ({outputs}) in the "
                "same tool call.", "first_hypothesis")
        return PolicyDecision(
            "inspect", "continue",
            "Inspection is done; do not inspect again. No valid output is retained yet: train a fast baseline "
            f"and save {outputs} in the format the task requests.", "inspect_without_baseline")
    if stage == "inspect":
        base = (f"Inspect first. Read the task, the data dictionary, the submission template and only these "
                f"allowed inputs: {_inputs_text(contract)}. Print compact summaries only (shapes, dtypes, "
                "missing counts, target distribution, a few sample rows), never whole tables, the manifest "
                "or index arrays. Explore only the data representation the contract permits. End with a "
                "short schema and plan. ")
        if retained:
            return PolicyDecision(
                "inspect", "continue",
                base + "A valid starting output already exists: use at most one more call for inspection, then "
                f"test one concrete hypothesis and write the required evidence ({outputs}).",
                "inspect_with_baseline")
        return PolicyDecision(
            "inspect", "continue",
            base + f"No valid output is retained yet: next train a fast baseline and save {outputs} in the "
            "format the task requests.", "inspect_without_baseline")
    if last is not None and _repeats_earlier(snapshot, last):
        return PolicyDecision("experiment", "continue",
                              "This code or hypothesis repeats an earlier trial; an unchanged retry is not a new "
                              "idea. Choose a hypothesis that differs in kind (model family, feature handling, "
                              f"regularization, target handling or a fixed blend). Write the evidence ({outputs}) "
                              "in the same tool call.", "repeated_code")
    if last is None:
        return PolicyDecision("experiment", "continue",
                              "Test one concrete hypothesis that differs from the starting result (another model "
                              "family, regularization, fold-fitted features or a fixed blend). Fit only on the "
                              f"declared fit data and write all required evidence ({outputs}) in the same tool "
                              "call.", "first_hypothesis")
    if last["verified"] and last["accepted"]:
        return PolicyDecision("experiment", "continue",
                              "The host verified an improvement (best metric above). Keep that fact exactly as "
                              "stated and claim nothing beyond it. Next, test a different hypothesis that builds "
                              "on what worked, with all required evidence in the same tool call, or reply with a "
                              "short factual account if no promising hypothesis remains.", "improved_continue")
    if last["verified"]:
        return PolicyDecision("experiment", "continue",
                              "The last candidate was valid but not better; the best result is preserved. Do not "
                              "repeat or lightly tweak it. Use the diagnostics to pick a hypothesis that differs "
                              "in kind (model family, feature handling, regularization, class or target "
                              "handling, a fixed blend) and write all required evidence in the same tool call.",
                              "not_better_continue")
    if contract["validation"]["supported"] and snapshot["counts"]["verified_trials"] == 0:
        return PolicyDecision("experiment", "continue",
                              "No experiment has been host-verified yet; analysis-only calls do not count. Test "
                              "one concrete hypothesis that differs from the starting result (another model "
                              "family, regularization, fold-fitted features or a fixed blend), fitting only on "
                              f"the declared fit data, and write all required evidence ({outputs}) in the same "
                              "tool call.", "first_hypothesis")
    if contract["validation"]["supported"]:
        return PolicyDecision("experiment", "continue",
                              "The last call produced no host-verified experiment (see last_trial.reason). "
                              "Analysis-only calls are fine but do not count. To evaluate a concrete hypothesis "
                              f"write every required output ({outputs}) in the same tool call, exactly as the "
                              "contract describes.", "no_verified_result")
    return PolicyDecision("experiment", "continue",
                          "The host cannot score this metric: quality stays unverified. Design your own "
                          "validation, compare against your best measurement, keep the best output safe and "
                          "report numbers as your own measurements.", "unverified_continue")


def _decide(snapshot: dict, contract: dict) -> PolicyDecision:
    counts, budget = snapshot["counts"], snapshot["budget"]
    retained = snapshot["baseline"]["retained_output_valid"]
    supported = contract["validation"]["supported"]
    done = _contract_met(snapshot, contract) and retained
    resources = (budget["can_request"] and budget["remaining_seconds"] > 0
                 and budget["remaining_requests"] > 0)

    def stop(reason_keep: str, reason_none: str, why: str) -> PolicyDecision:
        if retained:
            return PolicyDecision("finished", "fallback",
                                  f"{why} Keep the retained valid output; make no further model request.",
                                  reason_keep)
        return PolicyDecision("finished", "stop",
                              f"{why} No valid output is retained: stop and report the failure.", reason_none)

    if snapshot["finish_requested"] and done:
        if supported:
            return PolicyDecision(
                "finished", "complete",
                f"The host contract is met with {counts['verified_trials']} host-verified experiment(s). "
                "The research dialogue is complete; the host takes over the recorded candidate.",
                "contract_fulfilled")
        return PolicyDecision(
            "finished", "complete",
            "The research dialogue is complete, but quality is NOT independently verified: the host cannot "
            "score this task's metric, so any number printed by the model is the model's own unverified "
            "measurement and must be reported as such.", "unverified_completion")
    if snapshot["finish_requested"] and not supported and not contract["completion"]["allow_unverified"]:
        return stop("contract_unreachable", "contract_unreachable",
                    "The host cannot verify quality here and does not allow unverified completion.")
    if counts["consecutive_execution_failures"] >= FAILURE_STOP_STREAK:
        return stop("repeated_failures_keep_result", "repeated_failures_no_result",
                    f"{counts['consecutive_execution_failures']} consecutive execution failures.")
    if not resources:
        return stop("resources_exhausted_keep_result", "resources_exhausted_no_result",
                    "No time or model requests remain.")
    decision = _next(snapshot, contract)
    if snapshot["finish_requested"]:
        return PolicyDecision(
            decision.stage, "continue",
            f"You asked to finish, but the host contract is not met: {_missing(snapshot, contract)}. "
            + decision.instruction, "finish_too_early")
    return decision


def decide_next_action(snapshot: Mapping, *, contract: Mapping) -> PolicyDecision:
    """Choose the stage, action and instruction for the next step from host facts only.

    The instruction may quote strings of the contract; the host cleans it before use
    (``build_model_messages`` does this itself).
    """
    return _decide(_snapshot(snapshot), _contract(contract))


# ------------------------------------------------------------------ messages

def _read_history(history) -> list[dict]:
    """Group assistant/tool messages; an unfinished or inconsistent group is an input error."""
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        _fail("history", "must be a list of messages")
    items: list[dict] = []
    group = None

    def close(final: bool) -> None:
        nonlocal group
        if group is not None and group["pending"]:
            _fail("history", "ends with an unfinished tool group; finish the group first"
                  if final else "contains an unfinished tool group; finish the group first")
        group = None

    for index, message in enumerate(history):
        path = f"history[{index}]"
        message = _mapping(message, path)
        role = message.get("role")
        if role == "assistant":
            close(False)
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                _fail(path + ".content", "must be a string or null")
            calls = message.get("tool_calls")
            if calls is None or (isinstance(calls, (list, tuple)) and not calls):
                items.append({"kind": "plain", "content": content})
                continue
            if isinstance(calls, (str, bytes)) or not isinstance(calls, (list, tuple)):
                _fail(path + ".tool_calls", "must be a list")
            parsed, seen = [], set()
            for call in calls:
                call = _mapping(call, path + ".tool_calls[]")
                identifier = call.get("id")
                function = _mapping(call.get("function"), path + ".tool_calls[].function")
                if (not isinstance(identifier, str) or not identifier.strip() or identifier in seen
                        or call.get("type", "function") != "function"
                        or not isinstance(function.get("name"), str) or not function["name"].strip()
                        or not isinstance(function.get("arguments"), str)):
                    _fail(path + ".tool_calls", "require unique ids, type 'function', a name and string arguments")
                seen.add(identifier)
                parsed.append({"id": identifier, "name": function["name"],
                               "arguments": function["arguments"]})
            group = {"kind": "group", "content": content, "calls": parsed, "results": [],
                     "pending": {call["id"] for call in parsed}}
            items.append(group)
        elif role == "tool":
            identifier, content = message.get("tool_call_id"), message.get("content")
            if group is None:
                _fail(path, "is a tool result outside a tool group")
            if not isinstance(identifier, str) or not isinstance(content, str):
                _fail(path, "needs a string tool_call_id and string content")
            if identifier not in {call["id"] for call in group["calls"]}:
                _fail(path, "answers an unknown tool call")
            if identifier not in group["pending"]:
                _fail(path, "duplicates a tool result")
            group["pending"].discard(identifier)
            group["results"].append((identifier, content))
        else:
            _fail(path, "history accepts only assistant and tool messages")
    close(True)
    return items


def _contract_text(contract: Mapping) -> str:
    return (
        "HOST CONTRACT v1 (authoritative; model output cannot change it). Later HOST STATE messages carry "
        "host facts: accepted, verified and every metric are computed by the host; hypothesis text is the "
        "model's own claim; numbers printed by the model never change these facts.\n"
        f"manifest_version: {contract['manifest_version']}\nmanifest_path: {contract['manifest_path']}\n"
        f"host_instructions:\n{contract['host_instructions']}\n"
        f"allowed_inputs: {_dump(contract['allowed_inputs'])}\n"
        f"required_outputs: {_dump(contract['required_outputs'])}\n"
        f"validation: {_dump(contract['validation'])}\ncompletion: {_dump(contract['completion'])}")


def _summary_attempts(trials: int) -> list[tuple[int, int]]:
    attempts = [(n, 2) for n in range(trials, 0, -1)]
    if trials >= 1:
        attempts += [(1, 1), (1, 0)]
    return attempts + [(0, 0)]


class _Stats:
    def __init__(self):
        self.truncated = 0


def _small_trial(trial: Mapping, hypothesis_chars: int, stats: _Stats) -> dict:
    entry = {"n": trial["number"]}
    if trial["hypothesis"]:
        text = _clip(trial["hypothesis"], hypothesis_chars)
        stats.truncated += text != trial["hypothesis"]
        entry["hypothesis"] = text
    entry["accepted"], entry["verified"] = trial["accepted"], trial["verified"]
    entry["metric"] = None if trial["metric"] is None else trial["metric"]["value"]
    if trial["reason"]:
        text = _clip(trial["reason"], _TRIAL_REASON_CHARS)
        stats.truncated += text != trial["reason"]
        entry["reason"] = text
    if trial["error_kind"]:
        entry["error_kind"] = trial["error_kind"]
    return entry


def _render_summary(snapshot: Mapping, contract: Mapping, decision: PolicyDecision, *,
                    attempt: tuple[int, int], removed_groups: int, feedback_bytes: int,
                    stats: _Stats, shown: Sequence[Mapping] = ()) -> tuple[dict, int]:
    count, detail = attempt
    last = snapshot["last_trial"]
    sequence = [t for t in snapshot["recent_trials"] if last is None or t["number"] != last["number"]]
    if last is not None:
        sequence.append(last)
    chosen = sequence[-count:] if count else []
    entries = []
    for position, trial in enumerate(chosen):
        latest = position == len(chosen) - 1
        entry = _small_trial(trial, _LATEST_HYPOTHESIS_CHARS if latest else _TRIAL_HYPOTHESIS_CHARS, stats)
        if latest and detail >= 1 and trial["error"]:
            # The newest kept tool result usually carries the same traceback: do not send it twice.
            needle = trial["error"][-120:]
            if any(needle in f"{fb.get('stderr', '')}\n{fb.get('error', '')}" for fb in shown):
                entry["error_shown_above"] = True
            else:
                text, cut = _fit_body(trial["error"], _LATEST_ERROR_BYTES[detail], "tail")
                stats.truncated += cut
                entry["error"] = text
        if latest and detail == 2:
            if trial.get("diagnostics"):
                if any("diagnostics" in (fb.get("selection") or {}) for fb in shown):
                    entry["diagnostics_shown_above"] = True
                else:
                    levels = _diagnostics_levels(trial["diagnostics"])
                    cap = max(200, feedback_bytes // 4)
                    chosen_level = next((v for v in levels if json_size(v) <= cap), None)
                    if chosen_level is not None:
                        stats.truncated += chosen_level is not levels[0]
                        entry["diagnostics"] = chosen_level
                    else:
                        stats.truncated += 1
            if (trial["error_kind"] in ("timeout", "memory") and trial.get("resources")
                    and not any("resources" in fb for fb in shown)):
                subset = {name: trial["resources"][name] for name in
                          ("peak_rss_bytes", "cpu_seconds", "memory_limit_bytes")
                          if _is_number(trial["resources"].get(name))}
                if subset:
                    entry["resources"] = subset
        entries.append(entry)
    budget = snapshot["budget"]
    state = {
        "stage": snapshot["stage"],
        "budget": {"remaining_seconds": round(budget["remaining_seconds"], 1),
                   "remaining_requests": budget["remaining_requests"], "can_request": budget["can_request"]},
        "baseline": snapshot["baseline"], "counts": snapshot["counts"],
        "finish_requested": snapshot["finish_requested"],
        "validation": {"supported": contract["validation"]["supported"],
                       "scope": contract["validation"]["scope"]},
        "recent_trials": entries,
    }
    if removed_groups:
        state["earlier_tool_groups_omitted"] = removed_groups
    text = ("HOST STATE v1\n" + _dump(state)
            + f"\nNEXT STEP [{decision.stage}/{decision.reason}]: {decision.instruction}")
    return {"role": "user", "content": text}, len(entries)


def build_model_messages(*, system_prompt: str, task_text: str, contract: Mapping,
                         history: Sequence[Mapping], snapshot: Mapping, limits: PolicyLimits,
                         redact: Callable[[str], str]) -> PolicyContext:
    """Bounded messages: static prefix, the newest tool groups and ONE host summary."""
    _check_limits(limits)
    redact = _check_redact(redact)
    if not isinstance(system_prompt, str) or not isinstance(task_text, str):
        _fail("prompt", "system_prompt and task_text must be strings")
    contract = _clean(_contract(contract), redact)
    snapshot = _clean(_snapshot(snapshot), redact)
    items = _read_history(history)
    decision = _decide(snapshot, contract)

    static = [{"role": "system", "content": _clean_text(system_prompt, redact)},
              {"role": "user", "content": _clean_text(task_text, redact)},
              {"role": "user", "content": _contract_text(contract)}]
    groups = [item for item in items if item["kind"] == "group"]
    plain = [item for item in items if item["kind"] == "plain"]
    trailing = items[-1] if items and items[-1]["kind"] == "plain" and items[-1]["content"] else None
    if trailing is not None and limits.max_recent_tool_groups == 0:
        trailing = None
    window = groups[-limits.max_recent_tool_groups:] if limits.max_recent_tool_groups else []
    feedback_cache: dict = {}

    def feedback_for(content: str, divisor: int) -> dict:
        key = (content, divisor)
        if key not in feedback_cache:
            budget = max(1, limits.max_tool_feedback_bytes // divisor)
            try:
                parsed = json.loads(content, parse_constant=_reject_constant)
            except ValueError:
                parsed = None
            result = None
            if isinstance(parsed, Mapping):
                try:
                    result = compact_tool_feedback(parsed, limits=replace(
                        limits, max_tool_feedback_bytes=budget), redact=redact)
                except PolicyInputError:
                    result = None
                except PolicyBudgetExceeded:
                    if divisor == 1:
                        raise
                    result = feedback_for(content, 1)
            feedback_cache[key] = result if result is not None else _text_feedback(content, budget, redact)
        return feedback_cache[key]

    def group_messages(group: Mapping, divisor: int, older: bool,
                       stats: _Stats) -> tuple[list[dict], list[dict]]:
        content = group["content"]
        if content is not None:
            content, cut = _fit_body(_clean_text(content, redact), _MAX_ASSISTANT_TEXT, "head")
            stats.truncated += cut
        messages = [{"role": "assistant", "content": content, "tool_calls": [
            {"id": call["id"], "type": "function",
             "function": {"name": call["name"], "arguments": _clean_arguments(call["arguments"], redact)}}
            for call in group["calls"]]}]
        feedbacks = []
        results = group["results"]
        for position, (identifier, result) in enumerate(results):
            # Only the last result of the newest group keeps the full budget.
            reduced = older or position < len(results) - 1
            feedback = feedback_for(result, divisor * (2 if reduced else 1))
            feedbacks.append(feedback)
            stats.truncated += len(feedback.get("truncated", {}))
            messages.append({"role": "tool", "tool_call_id": identifier, "content": _dump(feedback)})
        return messages, feedbacks

    def assemble(kept: int, divisor: int, attempt, with_plain: bool):
        stats = _Stats()
        messages, newest = list(static), []
        chosen_groups = window[-kept:] if kept else []
        for position, group in enumerate(chosen_groups):
            # Older kept groups only say what was tried; the newest one keeps the full budget.
            older = position < len(chosen_groups) - 1
            try:
                added, newest = group_messages(group, divisor, older, stats)
            except PolicyBudgetExceeded:
                return None
            messages += added
        kept_plain = 0
        if trailing is not None and with_plain:
            text, cut = _fit_body(_clean_text(trailing["content"], redact), _MAX_PLAIN_TEXT, "head")
            stats.truncated += cut
            messages.append({"role": "assistant", "content": text})
            kept_plain = 1
        summary, included = _render_summary(
            snapshot, contract, decision, attempt=attempt, removed_groups=len(groups) - kept,
            feedback_bytes=limits.max_tool_feedback_bytes, stats=stats, shown=newest)
        messages.append(summary)
        return messages, stats, kept, kept_plain, included

    # Uncompressed reference: the same request with the whole cleaned history.
    full = list(static)
    for item in items:
        if item["kind"] == "plain":
            if item["content"]:
                full.append({"role": "assistant", "content": _clean_text(item["content"], redact)})
            continue
        full.append({"role": "assistant",
                     "content": None if item["content"] is None else _clean_text(item["content"], redact),
                     "tool_calls": [{"id": call["id"], "type": "function", "function": {
                         "name": call["name"], "arguments": _clean_arguments(call["arguments"], redact)}}
                         for call in item["calls"]]})
        full += [{"role": "tool", "tool_call_id": identifier, "content": _clean_text(result, redact)}
                 for identifier, result in item["results"]]
    full_summary, _ = _render_summary(
        snapshot, contract, decision, attempt=(limits.max_recent_trials, 2), removed_groups=0,
        feedback_bytes=limits.max_tool_feedback_bytes, stats=_Stats())
    before = json_size(full + [full_summary])

    attempts = _summary_attempts(limits.max_recent_trials)
    plan = [(kept, 1, 0, True) for kept in range(len(window), 0, -1)]
    if window:
        # Bulky output yields first; hypotheses and rejection reasons are worth more bytes.
        one_trial = max(0, limits.max_recent_trials - 1)
        plan += [(1, divisor, 0, True) for divisor in (2, 4)]
        plan += [(1, 4, index, True) for index in range(1, one_trial + 1)]
    plan += [(0, 1, index, True) for index in range(len(attempts))]
    plan += [(0, 1, index, False) for index in range(len(attempts))]
    if not window:
        plan = [entry for entry in plan if entry[0] == 0]
    for kept, divisor, index, with_plain in plan:
        built = assemble(kept, divisor, attempts[index], with_plain)
        if built is None:
            continue
        messages, stats, kept, kept_plain, included = built
        size = json_size(messages)
        if size <= limits.max_messages_bytes:
            return PolicyContext(messages=messages, diagnostics={
                "version": 1, "before_bytes": before, "after_bytes": size,
                "removed_tool_groups": len(groups) - kept,
                "omitted_plain_messages": len(plain) - kept_plain,
                "truncated_fields": stats.truncated, "recent_trials_included": included})
    raise PolicyBudgetExceeded("the mandatory prompt, contract and host summary do not fit into the limit")
