"""Optional paired inner rechecks of already frozen v3 research finalists."""

from __future__ import annotations

import math
import time

import pandas as pd

from agent.llm import safe_error_message
from agent.stability import additional_inner_splits, paired_stability_summary


TERMINAL_STAGES = {"candidate_frozen", "outer_validation", "full_refit", "result"}


def _value(metric, expected):
    if (not isinstance(metric, dict) or metric.get("name") != expected.name
            or metric.get("higher_is_better") is not expected.higher_is_better):
        raise ValueError("Stability requires the same host-scored inner metric for both finalists")
    value = metric.get("value")
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Stability requires a finite host-scored inner metric")
    return float(value)


def verify_research_finalists(state, splits=1):
    """Recheck baseline and selected candidate only before terminal freezing.

    ``state`` supplies host-frozen candidates, the approved outer-fit view,
    canonical inner scores and measured time estimates. No outer answers or
    outer evaluation results are read. Complete shared rounds choose by mean;
    ``best_inner`` always retains its meaning as the selected recipe's score
    on the original canonical split, never an average disguised as that score.

    This is opt-in. It does not extend the original deadline, retry a failed
    candidate, or call the model API. Frozen baseline code/recipe stay intact.
    """
    if type(splits) is not int or not 1 <= splits <= 3:
        raise ValueError("Research stability splits must be one, two or three")
    report = {
        "requested_splits": splits, "scope": "inner_selection", "status": "skipped",
        "outer_answers_used": False, "rounds": [], "changed_selection": False,
    }

    def finish(reason, *, status="skipped"):
        report.update(status=status, reason=reason)
        state.report["inner_stability"] = report
        state.save()
        return report

    if state.dialogue_closed or state.report.get("stage") in TERMINAL_STAGES:
        return finish("research_terminal")
    if splits == 1:
        return finish("disabled")
    gate = state.gate
    if gate is None or not gate.fit_rows or not gate.held_rows:
        return finish("inner_validation_unavailable")
    if gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return finish("task_semantics_unavailable")
    if state.baseline is None or state.selected_candidate is None:
        return finish("two_finalists_required")
    original_candidate = state.selected_candidate
    finalists = {"candidate": original_candidate, "baseline": state.baseline}
    if (original_candidate.code_hash, original_candidate.recipe_hash) == (
            state.baseline.code_hash, state.baseline.recipe_hash):
        return finish("identical_finalists")
    report["finalists"] = [{"id": name, "code_hash": candidate.code_hash,
                            "recipe_hash": candidate.recipe_hash}
                           for name, candidate in finalists.items()]
    try:
        gate.check_inputs()
        metric = gate.spec.metric
        canonical = {"candidate": _value(state.best_inner, metric),
                     "baseline": _value(state.baseline_inner, metric)}
        target = gate.research_train[gate.spec.target_column]
        if not gate.classification:
            target = pd.to_numeric(target, errors="raise")
        extra = additional_inner_splits(
            gate.research_train, target, gate.features, gate.classification, gate.choice,
            canonical=(gate.fit_rows, gate.held_rows), limit=splits - 1)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        report["error"] = safe_error_message(error)
        return finish("inner_stability_unavailable")
    rounds = [canonical]
    report["rounds"].append({"kind": "canonical", "status": "completed", "scores": canonical,
                             "split": gate.inner_split})
    reason = "completed" if len(extra) == splits - 1 else "fewer_admissible_splits"
    for fitting, held, metadata in extra:
        # Estimates use only host measurements of these exact frozen recipes.
        # Recompute after each round because the finalization reserve can grow.
        estimates = {name: state.estimate_research_cost(candidate, fit_rows=len(fitting),
                                                       predict_rows=len(held))
                     for name, candidate in finalists.items()}
        if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
               for value in estimates.values()):
            reason = "cost_estimate_unavailable"
            break
        available = state.research_timeout(maximum=3600)
        if (sum(estimates.values()) > available or available < len(finalists)
                or max(estimates.values()) > min(1800, available / len(finalists))):
            reason = "finalization_reserve"
            break
        round_deadline = time.monotonic() + available
        record = {"kind": "repeat", "status": "partial", "split": metadata,
                  "estimated_seconds": estimates, "scores": {}, "timeouts": {}}
        report["rounds"].append(record)
        scores = record["scores"]
        remaining = len(finalists)
        for name, candidate in finalists.items():
            # Equal remaining shares prevent the first model consuming all
            # spare time. measure_inner also checks the refreshed reserve.
            spare = min(max(0.0, round_deadline - time.monotonic()),
                        state.research_timeout(maximum=3600))
            timeout = min(1800.0, spare / remaining)
            if timeout < max(1.0, estimates[name]):
                record["reason"] = reason = "finalization_reserve"
                break
            record["timeouts"][name] = timeout
            try:
                result = state.measure_inner(candidate, rows=(fitting, held), maximum=timeout)
                gate.check_inputs()
                scores[name] = _value(result.metric, metric)
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                record["reason"] = reason = "repeat_failed"
                record["error"] = safe_error_message(error)
                break
            remaining -= 1
        rounds.append(dict(scores))
        if len(scores) != len(finalists):
            break
        record["status"] = "completed"

    summary = paired_stability_summary(list(finalists), rounds,
                                        higher_is_better=metric.higher_is_better)
    report["summary"] = summary
    # A single canonical round does not change a previously accepted winner.
    if summary["used_rounds"] > 1 and summary["ranking"][0] == "baseline":
        state.selected_candidate = None
        state.best_inner = state.baseline_inner
        report["changed_selection"] = True
        selection = state.report.get("evaluations", {}).get("selection")
        if isinstance(selection, dict):
            selection["metric"] = state.baseline_inner
    report["selected_recipe"] = "candidate" if state.selected_candidate is not None else "baseline"
    if summary["used_rounds"] > 1:
        status = "completed" if summary["used_rounds"] == splits else "partial"
    else:
        status = "skipped"
    return finish(reason, status=status)
