"""Pure, frozen fit-only residual correction and one-use admission on B.

The caller supplies calibration from the last period of its allowed fit and
base predictions made at one aligned tree count. This module cannot discover
history, fit an estimator, access files, or receive external validation labels.
Raw-feature groups, correction parameters and all diagnostic boundaries are
derived without consulting the prediction batch. Individual row/group hashes
and raw groups are never returned in evidence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np
import pandas as pd

from agent.building_history import history_keys, numeric_value
from agent.history_candidate import HistoryConfig
from agent.regression_base import _token
from agent.regression_history import HistorySignals


CONSTANTS = {
    "group_split": {
        "seed": 42,
        "feature_order": "lexically_sorted_declared_names",
        "encoding": "utf8_json_seed_and_name_typed_token_pairs_compact_ensure_ascii_false",
        "tokens": "unchanged_regression_base_token_v1",
        "digest": "sha256",
        "assignment": "int_full_hex_digest_modulo_100_lt_70_is_A_else_B",
        "labels_id_index": "excluded",
        "minimum_A_rows": 100,
        "minimum_B_rows": 100,
    },
    "numerics": {"epsilon": 1e-12, "std_ddof": 0, "ridge_penalty": 4.0,
        "shift_target_std_fraction": 0.1, "slope_min": 0.9, "slope_max": 1.1,
        "S20_factor": 0.2, "S50_factor": 0.5},
    "B_gate": {
        "total": "strict_SSE_improvement_over_identity",
        "tie": "gain_le_1e-12_times_max_1_identity_SSE_retains_identity",
        "minimum_rows": 100, "minimum_row_fraction": 0.01,
        "minimum_identity_SSE_fraction": 0.01,
        "history_relative_SSE_loss_max": 0.10,
        "category_relative_SSE_loss_max": 0.10,
        "category_absolute_SSE_loss_fraction_max": 0.01,
        "levels": "first_two_declared_categoricals_top3_by_prefix_count_then_lexical_else_other",
        "novelty": "new_then_rare_then_common_missing_ignored",
        "rare": "prefix_count_lt_max_20_ceil_0.001_prefix_rows",
        "boundary_fit": "strictly_earlier_model_fit_prefix_only",
        "history_groups": "literal_R5_eligible_ambiguous_valid_without_past_invalid",
        "invalid_diagnostic_partition": "reject_to_exact_incumbent",
        "insufficient_segment_support": "unknown_not_passed",
        "parameter_refit_after_B": False,
    },
}
EPSILON = 1e-12
MINIMUM_ROWS = 100
ARMS = ("S20", "S50", "A20")


class _DiagnosticContractError(ValueError):
    """The unchanged historical diagnostic masks cannot partition these rows."""


def _frame(value, name):
    if not isinstance(value, pd.DataFrame) or value.columns.duplicated().any():
        raise ValueError(f"{name} must be a frame with distinct columns")
    if any(not isinstance(column, str) or not column for column in value.columns):
        raise ValueError(f"{name} needs nonempty string column names")
    return value


def _names(value, X, name, *, allow_empty=False):
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a declared column sequence")
    if (any(not isinstance(column, str) or not column or column not in X for column in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must contain distinct existing columns")
    return list(value)


def _vector(values, length=None):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.ndim != 1 or (length is not None and len(array) != length) or not np.isfinite(array).all():
        return None
    return array


def _finite_sum(values):
    result = math.fsum(sorted(float(value) for value in values))
    if not math.isfinite(result):
        raise ValueError("Correction arithmetic is not finite")
    return result


def _mean(values):
    return _finite_sum(values) / len(values)


def _std(values, center):
    with np.errstate(over="raise", invalid="raise"):
        deviations = values - center
        return math.sqrt(_finite_sum(deviations * deviations) / len(values))


def _sse(errors):
    # Match the accepted host diagnostic's sorted scalar-square summation.
    return _finite_sum(float(value) ** 2 for value in errors)


def _sha(values):
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_calibration(X_calibration, feature_columns):
    """Return A/B iloc positions and aggregate evidence; no labels or ID input.

    Only columns explicitly listed in ``feature_columns`` are accessed, never
    the frame index or extra columns. The candidate validates these roles before
    calling. Equal typed raw rows always hash into the same part, regardless of
    row/column order or the presence of unrelated prediction rows.
    """
    X = _frame(X_calibration, "X_calibration")
    names = sorted(_names(feature_columns, X, "feature_columns"))
    group_hashes = []
    for row in X.loc[:, names].itertuples(index=False, name=None):
        group_hashes.append(_sha([42, [[name, list(_token(value))] for name, value in zip(names, row)]]))
    in_a = np.asarray([int(value, 16) % 100 < 70 for value in group_hashes], dtype=bool)
    a, b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
    a_hashes = sorted(group_hashes[int(position)] for position in a)
    b_hashes = sorted(group_hashes[int(position)] for position in b)
    available = len(a) >= MINIMUM_ROWS and len(b) >= MINIMUM_ROWS
    evidence = {"rows": len(X), "A_rows": len(a), "B_rows": len(b),
        "A_groups": len(set(a_hashes)), "B_groups": len(set(b_hashes)),
        "A_membership_sha256": _sha(a_hashes), "B_membership_sha256": _sha(b_hashes),
        "assignment_sha256": _sha([a_hashes, b_hashes]),
        "supported": available, "reason": "supported" if available else "insufficient_A_or_B_rows",
        "row_groups_exported": False, "labels_id_index_used": False}
    return a.astype(np.int64), b.astype(np.int64), evidence


def _identity(arm, reason, rows=0):
    return {"version": 1, "arm": arm, "status": "identity", "reason": reason,
        "center": 0.0, "offset": 0.0, "slope": 1.0, "fit_rows": int(rows),
        "A_target_mean": None, "A_target_std": None, "A_prediction_std": None,
        "unclipped_offset": None, "unclipped_slope": None,
        "ridge_intercept": None, "ridge_normalized_slope": None}


def fit_correction(arm, y_A, p_A):
    """Fit only the fixed arm's scalar policy on A, without estimator fitting.

    S20/S50 require target spread but permit constant base predictions. A20 also
    requires prediction standard deviation strictly greater than epsilon. The
    two-variable ridge objective penalizes the intercept and normalized slope.
    Floating-point sums are sorted so reordering A does not change the policy.
    """
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Choose one frozen correction arm")
    y = _vector(y_A)
    predictions = _vector(p_A, None if y is None else len(y))
    if y is None or predictions is None:
        return _identity(arm, "nonfinite_or_invalid_A_values")
    if len(y) < MINIMUM_ROWS:
        return _identity(arm, "insufficient_A_rows", len(y))
    try:
        mean_y, center = _mean(y), _mean(predictions)
        std_y, std_p = _std(y, mean_y), _std(predictions, center)
        if not math.isfinite(std_y) or std_y == 0:
            return _identity(arm, "zero_or_nonfinite_A_target_spread", len(y))
        maximum_offset = 0.1 * std_y
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residual = y - predictions
            if arm in ("S20", "S50"):
                intercept, normalized_slope = None, None
                raw_offset = (0.2 if arm == "S20" else 0.5) * _mean(residual)
                raw_slope = 1.0
            else:
                if not math.isfinite(std_p) or std_p <= EPSILON:
                    return _identity(arm, "insufficient_A_prediction_spread", len(y))
                z = (predictions - center) / max(std_p, EPSILON)
                mean_z = _mean(z)
                matrix = np.asarray([[5.0, mean_z], [mean_z, _mean(z * z) + 4.0]])
                rhs = np.asarray([_mean(residual), _mean(z * residual)])
                if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
                    return _identity(arm, "nonfinite_ridge_system", len(y))
                condition = float(np.linalg.cond(matrix))
                if not math.isfinite(condition) or condition >= 1.0 / EPSILON:
                    return _identity(arm, "degenerate_ridge_system", len(y))
                intercept, normalized_slope = map(float, np.linalg.solve(matrix, rhs))
                raw_offset, raw_slope = intercept, 1.0 + normalized_slope / max(std_p, EPSILON)
            offset = float(np.clip(raw_offset, -maximum_offset, maximum_offset))
            slope = float(np.clip(raw_slope, 0.9, 1.1))
        if not all(math.isfinite(value) for value in (center, offset, slope, raw_offset, raw_slope)):
            return _identity(arm, "nonfinite_correction_parameters", len(y))
        if offset == 0.0 and slope == 1.0:
            return _identity(arm, "zero_correction", len(y))
        return {"version": 1, "arm": arm, "status": "fitted", "reason": "fitted_on_A_only",
            "center": center, "offset": offset, "slope": slope, "fit_rows": len(y),
            "A_target_mean": mean_y, "A_target_std": std_y, "A_prediction_std": std_p,
            "unclipped_offset": float(raw_offset), "unclipped_slope": float(raw_slope),
            "ridge_intercept": intercept, "ridge_normalized_slope": normalized_slope}
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return _identity(arm, "nonfinite_or_degenerate_A_arithmetic", len(y))


def apply_correction(predictions, policy):
    """Apply frozen scalar parameters without reading query features or labels."""
    values = _vector(predictions)
    if values is None:
        raise ValueError("Correction requires finite one-dimensional predictions")
    if not isinstance(policy, dict) or policy.get("status") not in {"identity", "fitted"}:
        raise ValueError("Correction policy must be identity or fitted")
    if policy["status"] == "identity":
        return values.copy()
    if any(type(policy.get(name)) not in (int, float) or not math.isfinite(policy[name])
           for name in ("center", "offset", "slope")) or not 0.9 <= policy["slope"] <= 1.1:
        raise ValueError("Correction policy has invalid frozen coefficients")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = values + policy["offset"] + (policy["slope"] - 1.0) * (values - policy["center"])
        except FloatingPointError as error:
            raise ValueError("Corrected predictions are not finite") from error
    if not np.isfinite(result).all():
        raise ValueError("Corrected predictions are not finite")
    return result


def _text(value):
    missing = pd.isna(value)
    if not np.isscalar(missing):
        raise ValueError("Calibration diagnostics require scalar features")
    return None if missing or (isinstance(value, str) and not value.strip()) else str(value).strip()


def _statistics(errors, mask, total_sse):
    chosen = errors[mask]
    with np.errstate(over="raise", invalid="raise"):
        sse = _sse(chosen)
        absolute = _finite_sum(np.abs(chosen))
    n = len(chosen)
    return {"rows": n, "sse": sse, "mae": absolute / n if n else None,
        "rmse": math.sqrt(sse / n) if n else None,
        "row_share": n / len(errors) if len(errors) else None,
        "sse_share": sse / total_sse if total_sse else None}


def _masks(X_prefix, y_prefix, X_B, y_B, *, period_column, categorical, history_config):
    """Freeze the R5 veto partitions from the strictly earlier prefix only."""
    categorical = _names(categorical, X_prefix, "categorical", allow_empty=True)
    if list(X_prefix.columns) != list(X_B.columns):
        raise ValueError("B diagnostics require the same raw feature roles as the prefix")
    if period_column not in X_prefix:
        raise ValueError("The declared period column is required")
    size = len(X_B)
    masks = []

    def add(partition, segment, mask, *, veto=False, assessable=True, **metadata):
        value = np.asarray(mask, dtype=bool)
        if value.shape != (size,):
            raise ValueError("Diagnostic mask has incorrect shape")
        masks.append((partition, segment, value, veto, assessable, metadata))

    add("overall", "all", np.ones(size, dtype=bool))
    if history_config is None:
        add("history", "schema_unavailable", np.ones(size, dtype=bool), veto=True, assessable=False)
    else:
        parsed = HistoryConfig.from_json(history_config)
        if any(column not in X_prefix or column not in X_B for column in parsed.columns):
            raise _DiagnosticContractError("Frozen history columns are unavailable")
        signals = HistorySignals(parsed).fit(X_prefix, y_prefix).transform(X_B)
        keys = history_keys(X_B, parsed.key_columns, parsed.numeric_columns, parsed.positive_columns)
        periods = [numeric_value(value) for value in X_B[parsed.period_column]]
        invalid = np.asarray([key is None or period is None for key, period in zip(keys, periods)])
        past = signals["past_rows"].to_numpy() > 0
        ambiguous = signals["ambiguous"].to_numpy(dtype=bool)
        # Preserve the exact admitted R5 masks. An allow-policy repeated key
        # can overlap eligible/ambiguous; fail closed below rather than silently
        # changing the accepted diagnostic contract to make its gate pass.
        add("history", "eligible", signals["eligible"].to_numpy(dtype=bool), veto=True)
        add("history", "ambiguous", past & ambiguous, veto=True)
        add("history", "valid_without_past", ~invalid & ~past, veto=True)
        add("history", "incomplete_key_or_invalid_period", invalid, veto=True)

    any_new, any_rare = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
    rarity = max(20, math.ceil(0.001 * len(X_prefix)))
    for position, column in enumerate(categorical):
        prefix = [_text(value) for value in X_prefix[column]]
        held = [_text(value) for value in X_B[column]]
        counts = Counter(value for value in prefix if value is not None)
        any_new |= np.fromiter((value is not None and value not in counts for value in held), bool, size)
        any_rare |= np.fromiter((value is not None and 0 < counts[value] < rarity for value in held), bool, size)
        if position < 2:
            top = sorted(counts, key=lambda value: (-counts[value], value))[:3]
            for value in top:
                add("category:" + column, "level:" + value,
                    np.fromiter((item == value for item in held), bool, size), veto=True)
            add("category:" + column, "other", np.fromiter((value not in top for value in held), bool, size), veto=True)
    add("category_novelty", "new", any_new, veto=True, rare_count_threshold=rarity)
    add("category_novelty", "rare", ~any_new & any_rare, veto=True, rare_count_threshold=rarity)
    add("category_novelty", "common", ~any_new & ~any_rare, veto=True, rare_count_threshold=rarity)

    missing = np.zeros(size, dtype=np.int64)
    for column in X_prefix:
        missing += np.fromiter((_text(value) is None for value in X_B[column]), bool, size)
    add("missing_features", "0", missing == 0)
    add("missing_features", "1-5", (missing >= 1) & (missing <= 5))
    add("missing_features", "6+", missing >= 6)
    with np.errstate(over="raise", invalid="raise"):
        cutpoints = np.quantile(y_prefix, (0.5, 0.9, 0.99), method="linear")
    if not np.isfinite(cutpoints).all():
        raise ValueError("Prefix target diagnostic cutpoints must be finite")
    p50, p90, p99 = map(float, cutpoints)
    cuts = {"p50": p50, "p90": p90, "p99": p99}
    add("target", "<=p50", y_B <= p50, cutpoints=cuts)
    add("target", "(p50,p90]", (y_B > p50) & (y_B <= p90), cutpoints=cuts)
    add("target", "(p90,p99]", (y_B > p90) & (y_B <= p99), cutpoints=cuts)
    add("target", ">p99", y_B > p99, cutpoints=cuts)
    coverage = {}
    for partition, _, mask, _, _, _ in masks:
        coverage.setdefault(partition, np.zeros(size, dtype=np.int64))[:] += mask
    if any(not np.equal(counts, 1).all() for counts in coverage.values()):
        raise _DiagnosticContractError("Each calibration diagnostic partition must cover B exactly once")
    return masks


def gate_correction(X_prefix, y_prefix, X_B, y_B, p_B, policy, *, period_column,
                    categorical, history_config=None):
    """One frozen B decision: strict total SSE gain and unchanged R5 vetoes.

    B labels measure this one proposal only. They do not choose trees, fit
    parameters, change A/B assignments or refit the accepted policy. Boundaries
    and category membership reference the strictly earlier prefix. The caller
    must use exact incumbent output whenever ``accepted`` is false.
    """
    X_prefix, X_B = _frame(X_prefix, "X_prefix"), _frame(X_B, "X_B")
    prefix_y = _vector(y_prefix, len(X_prefix))
    truth, identity = _vector(y_B, len(X_B)), _vector(p_B, len(X_B))
    basic = {"accepted": False, "status": "rejected", "reason": None, "B_rows": len(X_B),
        "identity_sse": None, "corrected_sse": None, "sse_gain": None, "tie_tolerance": None,
        "segments": [], "violations": [], "unknown_segments": [],
        "history_gate": "unknown" if history_config is None else "not_evaluated",
        "parameter_refit_after_B": False}
    if len(X_B) < MINIMUM_ROWS:
        return {**basic, "status": "insufficient_support", "reason": "insufficient_B_rows"}
    if prefix_y is None or not len(prefix_y) or truth is None or identity is None:
        return {**basic, "reason": "nonfinite_or_invalid_B_or_prefix_values"}
    if not isinstance(policy, dict) or policy.get("status") != "fitted":
        return {**basic, "reason": "identity_policy_requires_no_B_selection"}
    try:
        corrected = apply_correction(identity, policy)
        with np.errstate(over="raise", invalid="raise"):
            before, after = truth - identity, truth - corrected
            total_before = _sse(before)
            total_after = _sse(after)
        tolerance = EPSILON * max(1.0, total_before)
        total_ok = total_before - total_after > tolerance
        basic.update(identity_sse=total_before, corrected_sse=total_after,
                     sse_gain=total_before - total_after, tie_tolerance=tolerance)
        minimum_rows = max(MINIMUM_ROWS, math.ceil(0.01 * len(X_B)))
        masks = _masks(X_prefix, prefix_y, X_B, truth, period_column=period_column,
                       categorical=categorical, history_config=history_config)
        history_statuses = []
        for partition, segment, mask, veto, assessable, metadata in masks:
            baseline = _statistics(before, mask, total_before)
            proposed = _statistics(after, mask, total_after)
            row = {"partition": partition, "segment": segment, "veto_partition": veto,
                "rows": baseline["rows"], "identity": baseline, "corrected": proposed,
                "status": "diagnostic_only", "reason": None, **metadata}
            if veto:
                significant = (assessable and baseline["rows"] >= minimum_rows
                    and baseline["sse"] >= 0.01 * total_before and baseline["sse"] > 0)
                if not significant:
                    row.update(status="unknown", reason="schema_unavailable" if not assessable else "insufficient_segment_support")
                    basic["unknown_segments"].append({"partition": partition, "segment": segment, "reason": row["reason"]})
                else:
                    delta = proposed["sse"] - baseline["sse"]
                    relative = delta / baseline["sse"]
                    violation = relative > 0.10 and (partition == "history" or delta > 0.01 * total_before)
                    row.update(status="failed" if violation else "passed", reason="significant_segment_veto" if violation else "within_frozen_veto",
                               absolute_sse_change=delta, relative_sse_change=relative)
                    if violation:
                        basic["violations"].append({"partition": partition, "segment": segment,
                            "rows": baseline["rows"], "identity_sse": baseline["sse"], "corrected_sse": proposed["sse"],
                            "absolute_sse_change": delta, "relative_sse_change": relative})
                if partition == "history":
                    history_statuses.append(row["status"])
            basic["segments"].append(row)
        basic["history_gate"] = ("failed" if "failed" in history_statuses else
            "unknown" if not history_statuses or "unknown" in history_statuses else "passed")
        accepted = bool(total_ok and not basic["violations"])
        reason = ("accepted_on_B_once" if accepted else "significant_segment_veto" if basic["violations"] else
                  "B_SSE_worsened" if total_after > total_before + tolerance else "B_practical_tie_identity")
        return {**basic, "accepted": accepted, "status": "accepted" if accepted else "rejected",
            "reason": reason, "minimum_segment_rows": minimum_rows,
            "segment_veto": "failed" if basic["violations"] else "no_known_violation_with_unknowns" if basic["unknown_segments"] else "passed"}
    except _DiagnosticContractError:
        return {**basic, "reason": "diagnostic_contract_failure", "history_gate": "unknown"}
    except (ArithmeticError, ValueError, TypeError):
        return {**basic, "reason": "nonfinite_or_invalid_B_diagnostics"}
