"""Optional paired inner rechecks of already frozen v3 research finalists."""

from __future__ import annotations

from copy import deepcopy
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
    if getattr(state, "finalist_selection", False):
        previous = getattr(state, "_checking_finalists", False)
        had_shortlist = hasattr(state, "_active_finalist_shortlist")
        previous_shortlist = getattr(state, "_active_finalist_shortlist", None)
        state._checking_finalists = True
        try:
            return _verify_measured_finalists(state, splits)
        finally:
            state._checking_finalists = previous
            if had_shortlist:
                state._active_finalist_shortlist = previous_shortlist
            elif hasattr(state, "_active_finalist_shortlist"):
                del state._active_finalist_shortlist
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


def _verify_measured_finalists(state, splits):
    """Compare a host-measured shortlist without exposing outer information.

    The shortlist and canonical scores are fixed before any shared repeat.
    Its prospective winners share the largest required finalization reserve;
    no additional search, API request or retry is performed here.
    """
    incumbent = state.selected_candidate
    incumbent_id = "candidate" if incumbent is not None else "baseline"
    selected_id = incumbent_id
    selected_metric = deepcopy(state.best_inner)
    selected = incumbent if incumbent is not None else state.baseline
    report = {"policy": "measured_finalists_v1", "requested_splits": splits,
              "scope": "inner_selection", "status": "skipped",
              "outer_answers_used": False, "rounds": [], "changed_selection": False,
              "excluded_finalists": []}

    def finish(reason, *, status="skipped"):
        report.update(status=status, reason=reason, selected_recipe=selected_id)
        if selected is not None:
            report["selected_frozen"] = {"code_hash": selected.code_hash,
                                         "recipe_hash": selected.recipe_hash}
            if selected_metric is not None:
                report["selected_frozen"]["metric"] = deepcopy(selected_metric)
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
    if state.baseline is None:
        return finish("two_finalists_required")
    recovery_baseline = state.baseline
    baseline_identity = (recovery_baseline.code_hash, recovery_baseline.recipe_hash)

    # Source/input/frozen-recipe failures are integrity failures, not optional
    # failures of an additional model fit. Keep these checks outside catches.
    def guard(entries):
        try:
            if (state.baseline is not recovery_baseline
                    or (recovery_baseline.code_hash, recovery_baseline.recipe_hash) != baseline_identity):
                raise ValueError("Stability changed the frozen recovery baseline")
            state.check_source()
            gate.check_inputs()
            recovery_baseline.verify()
            if incumbent is not None:
                incumbent.verify()
            for item in entries:
                item["candidate"].verify()
        except Exception:
            # Remember a known breach even if a later checker passes. Model
            # execution errors are handled separately and never set this latch.
            state._local_integrity_failed = True
            raise

    def invalid_shortlist(message):
        state._local_integrity_failed = True
        raise ValueError(message)

    guard(())
    if state.baseline_inner is None or state.best_inner is None:
        return finish("inner_baseline_unavailable")
    entries = state.finalist_shortlist()
    if (not isinstance(entries, (list, tuple)) or not 1 <= len(entries) <= 3
            or any(not isinstance(item, dict) or not {"id", "candidate", "metric", "family"} <= set(item)
                   for item in entries)):
        invalid_shortlist("Invalid host-measured finalist shortlist")
    # Detach mutable host metadata while retaining the immutable frozen files.
    entries = [{"id": item["id"], "candidate": item["candidate"],
                "metric": deepcopy(item["metric"]), "family": item["family"]}
               for item in entries]
    names = [item["id"] for item in entries]
    if (len(set(names)) != len(names) or any(name not in {"baseline", "candidate", "alternative"} for name in names)
            or "baseline" not in names or incumbent_id not in names):
        invalid_shortlist("Host-measured shortlist must retain the baseline and incumbent")
    by_id = {item["id"]: item for item in entries}
    for name, expected in (("baseline", state.baseline), (incumbent_id, selected)):
        actual = by_id[name]["candidate"]
        if (actual.code_hash, actual.recipe_hash) != (expected.code_hash, expected.recipe_hash):
            invalid_shortlist("Host-measured shortlist changed a protected frozen finalist")
    guard(entries)
    kept = []
    for item in entries:
        if item["id"] == "alternative" and not state.can_finalize_candidate(item["candidate"]):
            report["excluded_finalists"].append({"id": item["id"], "reason": "finalization_reserve"})
        else:
            kept.append(item)
    # Stable mean ties preserve the existing choice, including a baseline
    # incumbent. No repeat result influences this order or its membership.
    finalists = tuple(sorted(kept, key=lambda item: item["id"] != incumbent_id))
    state._active_finalist_shortlist = finalists
    if len(finalists) < 2:
        return finish("two_finalists_required")
    identities = [(item["candidate"].code_hash, item["candidate"].recipe_hash) for item in finalists]
    if len(set(identities)) != len(identities):
        invalid_shortlist("Host-measured finalist shortlist contains duplicate frozen identities")
    report["finalists"] = [{"id": item["id"], "family": item["family"],
                            "code_hash": item["candidate"].code_hash,
                            "recipe_hash": item["candidate"].recipe_hash,
                            "canonical_metric": deepcopy(item["metric"])}
                           for item in finalists]
    try:
        metric = gate.spec.metric
        canonical = {item["id"]: _value(item["metric"], metric) for item in finalists}
        # Baseline and incumbent must retain their previously accepted scores.
        if (canonical["baseline"] != _value(state.baseline_inner, metric)
                or canonical[incumbent_id] != _value(state.best_inner, metric)):
            invalid_shortlist("Host-measured shortlist changed a canonical metric")
        if hasattr(state, "finalist_inner_splits"):
            extra = state.finalist_inner_splits()
        else:
            target = gate.research_train[gate.spec.target_column]
            if not gate.classification:
                target = pd.to_numeric(target, errors="raise")
            extra = additional_inner_splits(gate.research_train, target, gate.features,
                gate.classification, gate.choice, canonical=(gate.fit_rows, gate.held_rows), limit=splits - 1)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        if getattr(state, "_local_integrity_failed", False):
            raise
        report["error"] = safe_error_message(error)
        guard(finalists)
        return finish("inner_stability_unavailable")
    guard(finalists)
    rounds = [canonical]
    report["rounds"].append({"kind": "canonical", "status": "completed", "scores": canonical,
                             "split": deepcopy(gate.inner_split)})
    reason = "completed" if len(extra) == splits - 1 else "fewer_admissible_splits"
    round_cap = 1800 * len(finalists)
    for fitting, held, metadata in extra:
        guard(finalists)
        estimates = {item["id"]: state.estimate_research_cost(item["candidate"],
                         fit_rows=len(fitting), predict_rows=len(held)) for item in finalists}
        if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
               for value in estimates.values()):
            reason = "cost_estimate_unavailable"
            break
        available = state.finalist_research_timeout(finalists, maximum=round_cap)
        if (sum(estimates.values()) > available or available < len(finalists)
                or max(estimates.values()) > min(1800, available / len(finalists))):
            reason = "finalization_reserve"
            break
        round_deadline = time.monotonic() + available
        record = {"kind": "repeat", "status": "partial", "split": deepcopy(metadata),
                  "estimated_seconds": estimates, "scores": {}, "timeouts": {}}
        report["rounds"].append(record)
        scores = record["scores"]
        remaining = len(finalists)
        for item in finalists:
            name, candidate = item["id"], item["candidate"]
            guard(finalists)
            spare = min(max(0.0, round_deadline - time.monotonic()),
                        state.finalist_research_timeout(finalists, maximum=round_cap))
            timeout = min(1800.0, spare / remaining)
            if timeout < max(1.0, estimates[name]):
                record["reason"] = reason = "finalization_reserve"
                break
            record["timeouts"][name] = timeout
            failed = False
            try:
                result = state.measure_inner(candidate, rows=(fitting, held), maximum=timeout)
                value = _value(result.metric, metric)
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                record["reason"] = reason = "repeat_failed"
                record["error"] = safe_error_message(error)
                failed = True
            guard(finalists)
            if failed:
                break
            scores[name] = value
            remaining -= 1
        rounds.append(dict(scores))
        if len(scores) != len(finalists):
            break
        record["status"] = "completed"

    summary = paired_stability_summary([item["id"] for item in finalists], rounds,
                                       higher_is_better=metric.higher_is_better)
    report["summary"] = summary
    guard(finalists)
    if summary["used_rounds"] > 1:
        winner = next(item for item in finalists if item["id"] == summary["ranking"][0])
        if winner["id"] != incumbent_id:
            feasible = (winner["id"] == "baseline"
                        or state.can_finalize_candidate(winner["candidate"]))
            guard(finalists)
            if feasible:
                selected_id = winner["id"]
                selected = winner["candidate"]
                selected_metric = deepcopy(winner["metric"])
                state.selected_candidate = None if selected_id == "baseline" else selected
                state.best_inner = deepcopy(selected_metric)
                report["changed_selection"] = True
                selection = state.report.get("evaluations", {}).get("selection")
                if isinstance(selection, dict):
                    selection["metric"] = deepcopy(selected_metric)
            else:
                report["selection_reason"] = "finalization_reserve"
    status = ("completed" if summary["used_rounds"] == splits else "partial"
              ) if summary["used_rounds"] > 1 else "skipped"
    return finish(reason, status=status)
