"""Pure, inactive-by-default policy for the next bounded INNER action.

The caller supplies host-derived, already-cleaned pre-freeze inputs. This
module does not inspect state, reports, files, answers, environment or clocks.
A claimed scope or hash is never proof of provenance: the host must construct
these inputs from its permitted research view and verified candidate events.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
import math

from agent.llm_context import _tool_result, _validate_pairs
from agent.round4_llm_evidence import validate_inner_snapshot, validate_recipe_identity


POLICY_VERSION = "round4_inner_action_v1"
MAX_HISTORY_ROWS = 12
MAX_HISTORY_BYTES = 12 * 1024
MAX_ADDITION_BYTES = 32 * 1024
OUTCOMES = frozenset({"analysis_only", "no_candidate", "candidate_invalid", "execution_failed",
                      "narrow_inspection", "verified_inner", "truncated_response", "transport_error"})
UNPRODUCTIVE = frozenset({"analysis_only", "no_candidate", "candidate_invalid", "execution_failed"})
CONTEXT_SECTIONS = ("allowed_inputs", "features", "shape", "split", "installed_tools")
ESSENTIAL_FIELDS = CONTEXT_SECTIONS


@dataclass(frozen=True)
class PreparedPolicy:
    messages: list[dict]
    policy_version: str
    action: str
    stop_reason: str | None
    summary: dict


def _encoded(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Policy inputs require finite JSON data") from error


def _number(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _metric(value, task):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"name", "value", "higher_is_better"}:
        raise ValueError("History metric must be a host INNER metric")
    if (task["metric"] is None or value["name"] != task["metric"]["name"]
            or type(value["higher_is_better"]) is not bool
            or value["higher_is_better"] != task["metric"]["higher_is_better"]
            or type(value["value"]) not in {int, float} or not math.isfinite(value["value"])):
        raise ValueError("History metric differs from the INNER task contract")
    return deepcopy(value)


def validate_experiment_history(history, *, task):
    """Accept a short host-owned history, not a final report or model claims.

Identity uses the candidate source hash and the host's canonical recipe hash,
never the tool-script digest or model-family name. Optional recipe settings are
verified against that canonical recipe hash. Unknown settings remain absent.
"""
    if not isinstance(history, (list, tuple)) or len(history) > MAX_HISTORY_ROWS:
        raise ValueError("History must contain at most twelve host observations")
    allowed = {"scope", "outcome", "identity", "metric", "elapsed_seconds", "accepted", "inspected_field"}
    result = []
    for row in history:
        if (not isinstance(row, dict) or set(row) - allowed
                or row.get("scope") != "inner_selection"
                or not isinstance(row.get("outcome"), str) or row["outcome"] not in OUTCOMES):
            raise ValueError("History requires allowlisted INNER observations")
        item = {"scope": "inner_selection", "outcome": row["outcome"]}
        identity = row.get("identity")
        if identity is not None:
            item["identity"] = validate_recipe_identity(identity)
        metric = _metric(row.get("metric"), task)
        if metric is not None:
            item["metric"] = metric
        if "accepted" in row:
            if type(row["accepted"]) is not bool:
                raise ValueError("History acceptance must be a host boolean")
            item["accepted"] = row["accepted"]
        if "elapsed_seconds" in row:
            if row["elapsed_seconds"] is not None and not _number(row["elapsed_seconds"]):
                raise ValueError("History elapsed time must be finite or unknown")
            item["elapsed_seconds"] = row["elapsed_seconds"]
        if "inspected_field" in row:
            if row["outcome"] != "narrow_inspection" or row["inspected_field"] not in ESSENTIAL_FIELDS:
                raise ValueError("Only a narrow inspection can name an inspected field")
            item["inspected_field"] = row["inspected_field"]
        if row["outcome"] == "verified_inner" and (identity is None or metric is None):
            raise ValueError("A verified INNER observation requires metric and frozen identity")
        if row["outcome"] != "verified_inner" and (metric is not None or row.get("accepted") is True):
            raise ValueError("Unverified work cannot carry a verified metric or acceptance")
        result.append(item)
    if len(_encoded(result)) > MAX_HISTORY_BYTES:
        raise ValueError("History exceeds its 12 KiB context budget")
    return result


def _missing(snapshot):
    missing = [name for name in CONTEXT_SECTIONS if not snapshot.get(name)]
    shape = snapshot.get("shape") or {}
    if shape and any(shape.get(key) is None for key in ("fit_rows", "validation_rows", "feature_count")):
        missing.append("shape")
    split = snapshot.get("split") or {}
    if split and split.get("strategy") is None:
        missing.append("split")
    return list(dict.fromkeys(missing))


def _research_exhausted(snapshot):
    timing = snapshot.get("timing") or {}
    maximum = timing.get("maximum_experiment_seconds")
    if maximum is not None and maximum <= 2:
        return True
    remaining, reserve = timing.get("remaining_seconds"), timing.get("final_reserve_seconds")
    return remaining is not None and reserve is not None and remaining - reserve <= 2


def prepare_round4_messages(messages, *, remaining_requests, inner_snapshot, experiment_history):
    """Prepare a temporary append-only dialogue BEFORE every host API guard.

``stop_local`` is advice to the host adapter, never an executed action. All
original messages remain bytewise unchanged in the returned prefix. The
caller must not append the temporary addition to accumulated history.
"""
    if (not isinstance(messages, (list, tuple)) or type(remaining_requests) is not int
            or not 0 <= remaining_requests <= 128):
        raise ValueError("Policy requires a message sequence and bounded integer attempts")
    _validate_pairs(messages)
    for message in messages:
        if (message.get("role") not in {"system", "user", "assistant", "tool"}
                or not isinstance(message.get("content"), (str, type(None)))
                or (message.get("role") != "assistant" and not isinstance(message.get("content"), str))):
            raise ValueError("Policy requires text chat messages with supported roles")
        _tool_result(message)  # Reject explicit non-INNER selection scopes.
    original_bytes = len(_encoded(messages))
    snapshot = validate_inner_snapshot(inner_snapshot)
    history = validate_experiment_history(experiment_history, task=snapshot["task"])
    measured = [row for row in history if row["outcome"] == "verified_inner"]
    identities = Counter((row["identity"]["code_hash"], row["identity"]["recipe_hash"]) for row in measured)
    duplicates = sum(count - 1 for count in identities.values())
    unproductive_streak = 0
    for row in reversed(history):
        if row["outcome"] not in UNPRODUCTIVE:
            break
        unproductive_streak += 1
    missing = _missing(snapshot)
    stop = ("attempts_exhausted" if remaining_requests == 0 else
            "research_time_exhausted" if _research_exhausted(snapshot) else
            "verified_recipe_repeated" if duplicates else
            "unproductive_attempts" if unproductive_streak >= 2 else
            "essential_context_still_missing" if missing and any(
                row["outcome"] == "narrow_inspection" for row in history) else None)
    action = "stop_local" if stop else "inspect_one_field" if missing else "create_candidate"
    prepared = deepcopy(list(messages))
    if action != "stop_local":
        if action == "inspect_one_field":
            inspection = (
                "Check importlib.util.find_spec only for the permitted modelling packages; do not install, "
                "download or import data. " if missing[0] == "installed_tools" else
                "Read only research_validation.json for this aggregate; do not dump row-index arrays. "
            )
            instruction = (
                "One essential host aggregate is unavailable: " + missing[0] + ". "
                "Use at most ONE narrow run_python inspection to establish this field. " + inspection +
                "Do not print rows, IDs, individual targets, unique value lists or full CSVs. "
                "Do not train a separate model for profiling. If this cannot be established within the "
                "current host budget, stop the paid dialogue; do not guess or begin another inspection series."
            )
        else:
            instruction = (
                "Use the supplied host INNER evidence for ONE small concrete experiment now. In ONE "
                "run_python call create fresh research_hypothesis.json, research_recipe.json and "
                "research_candidate.py together. Explain the changed setting and why it may help. Define "
                "fit_predict(X_fit, y_fit, X_predict, context); fit the model and all learned transformations "
                "using only these fit arguments. Do not read original train inside the recipe, memorize "
                "row positions, branch on phase, or write submission.csv. Return predictions and exact "
                "class order when required. Choose only declared installed tools and size the experiment "
                "against measured fit cost and remaining research time. An analysis, valid Python exit, "
                "or syntactically valid recipe is not a verified fit. The host checks INNER predictions. "
                "Change the approach after invalid outputs. Do not remeasure any exact source+normalized "
                "recipe identity listed below as progress; different parameters remain different trials. "
                "Preserve the finalization reserve and finish if no feasible useful change remains."
            )
        payload = {"policy_version": POLICY_VERSION, "action": action,
                   "remaining_requests": remaining_requests, "inner_evidence": snapshot,
                   "host_observations": history}
        content = "Round4 host evidence (INNER only):\n" + _encoded(payload).decode("utf-8") + "\n" + instruction
        if len(content.encode("utf-8")) > MAX_ADDITION_BYTES:
            raise ValueError("Policy addition exceeds 32 KiB; preserve the original guards")
        prepared.append({"role": "user", "content": content})
    return PreparedPolicy(prepared, POLICY_VERSION, action, stop, {
        "policy_version": POLICY_VERSION, "action": action, "stop_reason": stop,
        "original_bytes": original_bytes, "prepared_bytes": len(_encoded(prepared)),
        "evidence_bytes": len(_encoded(snapshot)), "history_bytes": len(_encoded(history)),
        "verified_inner_observations": len(measured), "unique_verified_identities": len(identities),
        "duplicate_verified_observations": duplicates, "unproductive_streak": unproductive_streak,
        "missing_essential_fields": missing, "selected_inspection_field": missing[0] if action == "inspect_one_field" else None,
    })
