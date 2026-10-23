"""Inactive, fixed fit-only residual corrections before unchanged history.

A chooses the correction and the correction path's tree count. B admits one
proposal once. Rejection delegates the unchanged incumbent under the original
deadline; that explicitly separate legacy fallback retains its own stopping
contract. Accepted identity ablation is the A-only aligned base, not incumbent I.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from agent.boosting_candidate import _check_deadline, _deadline
from agent.regression_candidate import execute_regression_candidate, make_regression_recipe
from agent.regression_history import HistorySignals, apply_history, select_history
from agent.round6_regression_base import active_component, observed, prepare_inputs, stage_calibration, refit_full
from agent.round6_regression_calibration import (
    ARMS, CONSTANTS, split_calibration, fit_correction, apply_correction, gate_correction,
)


FAMILY = "round6_regression_calibration_v1"
MIN_CALIBRATION_HISTORY_ROWS = 64


def make_round6_recipe(period_column, arm, *, history_config=None):
    """Freeze one complete algorithm; external control I stays unchanged."""
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Choose one of S20, S50 or A20")
    incumbent = make_regression_recipe(period_column, history_config=history_config)
    return {"version": 1, "family": FAMILY, "arm": arm,
        "base": incumbent["base"], "history_config": incumbent["history_config"],
        "history_policy": "calibrated", "minimum_calibration_history_rows": MIN_CALIBRATION_HISTORY_ROWS,
        "calibration": {"split": "last_valid_period_inside_supplied_fit",
            "rules": deepcopy(CONSTANTS), "tree_selection": "strict_prefix_eval_A_only_B_absent",
            "tree_alignment": "always_one_paid_fresh_prefix_recheck_at_max_50_best",
            "minimum_final_iterations": 50,
            "recheck_early_stopping": False, "parameters_refit_after_B": False,
            "accepted_identity_ablation": "aligned_A_only_base_not_incumbent_I",
            "history_selection": "unchanged_on_corrected_A_plus_B_after_correction_policy_freeze",
            "composition": "freeze_correction_then_history_selection_then_fresh_full_base_then_correction_then_history",
            "target_scale": "original", "seed": 42},
        "resources": {"threads": 8, "device": "CPU", "execution": "sequential",
            "history_reserve_fraction": .05, "history_reserve_seconds_max": 8.0,
            "trial_fraction_of_initial_model_window": .4,
            "trial_includes": "preparation_A_selection_prefix_recheck_correction_fit_B_gate",
            "final_fit": "remaining_original_model_window",
            "deadline": "one_original_absolute_deadline_no_reset"},
        "fallback": {"no_calibration": "delegate_unchanged_incumbent_without_trial_fits",
            "insufficient_A_B_support": "delegate_unchanged_incumbent_without_trial_fits",
            "identity_parameters_or_B_rejection": "delegate_unchanged_incumbent_after_trial",
            "legacy_B_label_exception": "only_rejected_unchanged_incumbent_keeps_legacy_full_calibration_stopping",
            "incumbent_deadline": "same_original_deadline_with_existing_legacy_reserve",
            "failure_or_insufficient_time": "raise_no_partial_success",
            "no_history_schema": "corrected_base_or_exact_incumbent_base"}}


def _recipe(value):
    if not isinstance(value, dict) or value.get("family") != FAMILY:
        raise ValueError("A normalized round6 regression recipe is required")
    try:
        expected = make_round6_recipe(value["base"]["period_column"], value.get("arm"),
            history_config=value.get("history_config"))
        supplied = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        checked = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (KeyError, TypeError, ValueError, RecursionError):
        raise ValueError("Round6 recipe must be the complete finite frozen JSON") from None
    if supplied != checked:
        raise ValueError("Round6 recipe differs from the frozen algorithm or JSON types")
    return expected


def _position_hash(values):
    return hashlib.sha256(json.dumps([int(value) for value in values], separators=(",", ":")).encode()).hexdigest()


def _history_selection(prepared, calibration_predictions, config):
    selected = {"mode": "mean", "weight": .5, "shrinkage": 0}
    reason = "no_frozen_history_schema" if config is None else "insufficient_calibration_history_mean_fallback"
    details = None
    if config is not None:
        signals = HistorySignals(config).fit(prepared.X.iloc[prepared.fitting],
            prepared.y.iloc[prepared.fitting]).transform(prepared.X.iloc[prepared.held])
        if int(signals["eligible"].sum()) >= MIN_CALIBRATION_HISTORY_ROWS:
            details = select_history(signals, prepared.y.iloc[prepared.held], calibration_predictions)
            selected = {key: details[key] for key in ("mode", "weight", "shrinkage")}
            reason = "fit_only_calibrated_after_correction_freeze"
    return selected, reason, details


def _fallback(X_fit, y_fit, X_predict, context, recipe, correction, *, started,
              deadline, trial_deadline, reserve, preparation, calibration=None):
    """Run the actual unchanged incumbent, not an identity on a changed base."""
    _check_deadline(deadline)
    fallback_context = deepcopy(context)
    fallback_context["recipe"] = make_regression_recipe(recipe["base"]["period_column"],
        history_config=recipe["history_config"])
    fallback_context["deadline_monotonic"] = deadline
    # A None phase asks the observer to use the same fit-row inference as its
    # unchanged I arm. This avoids monkey-patching shared helpers for tracing.
    with observed(None, deadline, deadline, path="incumbent_fallback"):
        result, incumbent, alternatives = execute_regression_candidate(
            X_fit, y_fit, X_predict, fallback_context, return_evidence=True)
    _check_deadline(deadline)
    history = deepcopy(incumbent["history"])
    history["schema_available"] = recipe["history_config"] is not None
    if recipe["history_config"] is None:
        for key in ("eligible_rows", "ambiguous_rows", "no_history_rows"):
            history[key] = None
    expected_incumbent = 2 if incumbent["base"].get("internal_fit_rows", 0) else 1
    evidence = {"version": 1, "family": FAMILY, "arm": recipe["arm"], "status": "completed",
        "correction": deepcopy(correction), "base": deepcopy(incumbent["base"]),
        "preparation": deepcopy(preparation), "calibration": deepcopy(calibration),
        "history": history, "incumbent_fallback": True,
        "fallback": "unchanged_rescue_regression_v1",
        "fallback_B_label_exception": "legacy_selection_after_correction_rejection_only",
        "original_deadline_monotonic": deadline, "trial_deadline_monotonic": trial_deadline,
        "history_reserve_seconds": reserve, "full_fit_rows": len(X_fit), "predict_rows": len(X_predict),
        "fresh_full_refit": True, "elapsed_seconds": time.monotonic() - started,
        "api_requests": 0,
        "physical_fits_expected_completed": expected_incumbent + (2 if calibration is not None else 0),
        "physical_fit_count_source": "expected_successful_stages; actual starts/completions require observer"}
    values = {"identity": alternatives["base"].copy(), "base": alternatives["base"].copy(),
        "mean": alternatives["mean"].copy(), "calibrated": alternatives["calibrated"].copy()}
    return result, evidence, values


def execute_round6_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    if not isinstance(context, dict):
        raise ValueError("A declared host context is required")
    started = time.monotonic()
    recipe = _recipe(context.get("recipe"))
    deadline = _deadline(context)
    config = recipe["history_config"]
    if config is not None:
        if (not set(config["key_columns"] + [config["period_column"]]) <= set(context.get("features", []))
                or not set(config["numeric_columns"]) <= set(context.get("numeric", []))):
            raise ValueError("History configuration must use the declared raw feature roles")
    reserve = min(recipe["resources"]["history_reserve_seconds_max"],
        max(0.0, (deadline - started) * recipe["resources"]["history_reserve_fraction"]))
    base_deadline = deadline - reserve
    trial_deadline = started + max(0.0, base_deadline - started) * recipe["resources"]["trial_fraction_of_initial_model_window"]
    prepared = prepare_inputs(X_fit, y_fit, X_predict, context, recipe["base"])
    _check_deadline(deadline)
    correction = {"status": "no_calibration", "reason": prepared.evidence["reason"],
        "applied": False, "split": None, "policy": None, "B_gate": None,
        "parameter_refit_after_B": False, "B_labels_used_for_correction_trees": False,
        "analytic_parameter_fit_calls": 0, "parameter_fit_elapsed_seconds": 0.0}

    def fallback(calibration=None):
        answer = _fallback(X_fit, y_fit, X_predict, context, recipe, correction,
            started=started, deadline=deadline, trial_deadline=trial_deadline,
            reserve=reserve, preparation=prepared.evidence, calibration=calibration)
        return answer if return_evidence else answer[0]

    if not prepared.evidence["calibration_available"]:
        return fallback()
    a, b, split = split_calibration(prepared.X.iloc[prepared.held], prepared.features)
    correction["split"] = split
    _check_deadline(deadline)
    if not split["supported"]:
        correction.update(status="insufficient_support", reason=split["reason"])
        return fallback()
    _check_deadline(trial_deadline)
    aligned = stage_calibration(prepared, a, deadline=trial_deadline, original_deadline=deadline)
    calibration = {**aligned.evidence,
        "prefix_original_positions_sha256": _position_hash(prepared.order[prepared.fitting]),
        "calibration_original_positions_sha256": _position_hash(prepared.order[prepared.held]),
        "A_original_positions_sha256": _position_hash(prepared.order[prepared.held[a]]),
        "B_original_positions_sha256": _position_hash(prepared.order[prepared.held[b]])}
    parameter_started = time.monotonic()
    with observed("correction_fit_A", trial_deadline, deadline,
                  path="analytic_correction", engine="analytic_residual"):
        policy = fit_correction(recipe["arm"], prepared.y.iloc[prepared.held[a]], aligned.predictions[a])
    correction.update(policy=deepcopy(policy), analytic_parameter_fit_calls=1,
        parameter_fit_elapsed_seconds=time.monotonic() - parameter_started)
    _check_deadline(trial_deadline)
    if policy["status"] != "fitted":
        correction.update(status="identity", reason=policy["reason"])
        return fallback(calibration)
    gate_started = time.monotonic()
    gate = gate_correction(prepared.X.iloc[prepared.fitting], prepared.y.iloc[prepared.fitting],
        prepared.X.iloc[prepared.held[b]], prepared.y.iloc[prepared.held[b]], aligned.predictions[b], policy,
        period_column=recipe["base"]["period_column"], categorical=prepared.categorical, history_config=config)
    correction.update(B_gate=gate, gate_elapsed_seconds=time.monotonic() - gate_started)
    _check_deadline(trial_deadline)
    if not gate["accepted"]:
        correction.update(status="B_rejected", reason=gate["reason"])
        return fallback(calibration)
    correction.update(status="applied", reason=gate["reason"], applied=True)
    # A parameters and the one B decision are now immutable. History retains
    # its separate legacy calibration contract on the complete internal held
    # period, using the already corrected, aligned prefix predictions.
    history_started = time.monotonic()
    corrected_calibration = apply_correction(aligned.predictions, policy)
    selected, history_reason, history_calibration = _history_selection(prepared, corrected_calibration, config)
    history_calibration_seconds = time.monotonic() - history_started
    _check_deadline(base_deadline)
    identity, refit = refit_full(prepared, aligned.selected_iterations,
        deadline=base_deadline, original_deadline=deadline)
    base = apply_correction(identity, policy)
    history_started = time.monotonic()
    mean, calibrated = base.copy(), base.copy()
    coverage = {"schema_available": config is not None, "eligible_rows": None,
        "ambiguous_rows": None, "no_history_rows": None}
    if config is not None:
        signals = HistorySignals(config).fit(prepared.X, prepared.y).transform(prepared.predict)
        mean = apply_history(base, signals, mode="mean", weight=.5, shrinkage=0)
        calibrated = apply_history(base, signals, **selected)
        coverage.update(eligible_rows=int(signals["eligible"].sum()),
            ambiguous_rows=int(signals["ambiguous"].sum()), no_history_rows=int((signals["past_rows"] == 0).sum()))
    _check_deadline(deadline)
    evidence = {"version": 1, "family": FAMILY, "arm": recipe["arm"], "status": "completed",
        "correction": correction, "preparation": prepared.evidence, "calibration": calibration,
        "base": refit, "history": {"config": deepcopy(config), "policy": "calibrated",
            "selected": selected, "reason": history_reason, "calibration": history_calibration,
            "calibration_elapsed_seconds": history_calibration_seconds,
            "prediction_elapsed_seconds": time.monotonic() - history_started, **coverage},
        "incumbent_fallback": False, "fallback": "none", "original_deadline_monotonic": deadline,
        "base_deadline_monotonic": base_deadline, "trial_deadline_monotonic": trial_deadline,
        "history_reserve_seconds": reserve, "full_fit_rows": len(X_fit), "predict_rows": len(X_predict),
        "fresh_full_refit": True, "elapsed_seconds": time.monotonic() - started, "api_requests": 0,
        "identity_ablation": "aligned_A_only_uncorrected_base_not_incumbent_I",
        "physical_fits_expected_completed": 3,
        "physical_fit_count_source": "expected_successful_stages; actual starts/completions require observer"}
    answer = {"predictions": calibrated}
    alternatives = {"identity": identity.copy(), "base": base, "mean": mean, "calibrated": calibrated}
    return (answer, evidence, alternatives) if return_evidence else answer


def fit_predict(X_fit, y_fit, X_predict, context):
    return execute_round6_candidate(X_fit, y_fit, X_predict, context)


def candidate_source():
    """Flat-package-compatible entry point with data-free sidecars."""
    return '''import hashlib,json
from pathlib import Path
import pandas as pd
from agent.round6_regression_candidate import execute_round6_candidate

def fit_predict(X_fit,y_fit,X_predict,context):
    result,evidence,alternatives=execute_round6_candidate(
        X_fit,y_fit,X_predict,context,return_evidence=True)
    path=Path("regression_ablation.csv")
    with path.open("x",encoding="utf-8",newline="") as stream:
        pd.DataFrame(alternatives).to_csv(stream,index=False)
    evidence["ablation_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    with Path("regression_evidence.json").open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,sort_keys=True,allow_nan=False)
    return result
'''
