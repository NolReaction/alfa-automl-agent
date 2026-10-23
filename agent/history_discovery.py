"""Target-free, bounded screening before one history-schema proposal search.

Only the raw feature frame from INNER fitting rows belongs here. Numeric key
parts use exact Decimal strings, text keys retain literal values after strip.
Recurrence is structural evidence, never verified entity identity. A schema
still needs the host's ordinary INNER measurement before it can be selected.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json

import pandas as pd

from agent.history_schema_proposer import propose_history_schemas


VERSION = 1
SCOPE = "inner_fit_features_only"
POLICY = "past_unique_only"
MAX_ROWS = 250_000
MAX_FEATURES = 128
MAX_CELL_VISITS = 8_000_000
MAX_RAW_BYTES = 64 * 1024**2
MAX_VALUE_CHARS = 512
MAX_SELECTED_COLUMNS = 12
MAX_CHECKED_CONFIGS = 48
MAX_AUTO_KEY_WIDTH = 3


def _names(values, field):
    if not isinstance(values, (list, tuple)) or isinstance(values, str):
        raise ValueError(field + " must be a list or tuple of feature names")
    if any(not isinstance(name, str) or not name.strip() for name in values):
        raise ValueError(field + " must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(field + " contains duplicate names")
    return tuple(sorted(values))


def _number(value):
    if value is None or not value.strip():
        return None
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _text(value):
    return None if value is None else value.strip() or None


def _column_evidence(values, periods, *, numeric):
    groups = {}
    eligible = 0
    parse = _number if numeric else _text
    for raw, period in zip(values, periods):
        value = parse(raw)
        if value is not None and period is not None:
            period_counts = groups.get(value)
            if period_counts is None:
                period_counts = {}
                groups[value] = period_counts
            period_counts[period] = period_counts.get(period, 0) + 1
            eligible += 1
    clean = recurrent = recurrent_values = 0
    for counts in groups.values():
        if len(counts) >= 2:
            recurrent_values += 1
            recurrent += sum(counts.values())
            if all(count == 1 for count in counts.values()):
                clean += len(counts) - 1
    reason = ("constant" if len(groups) <= 1 else "all_unique" if len(groups) == eligible
              else "no_cross_period_recurrence" if recurrent == 0 else None)
    return {"eligible_rows": eligible, "distinct_values": len(groups),
            "clean_recurrent_rows": clean, "cross_period_recurrent_rows": recurrent,
            "cross_period_recurrent_values": recurrent_values, "excluded_reason": reason}


def discover_history_schema(X_fit, *, feature_columns, numeric_columns, period_column,
                            max_cell_visits=None, max_checked_configs=None, adaptive_checks=False):
    """Return finite JSON describing at most one fixed-policy schema.

    ``X_fit`` must contain exactly ``feature_columns`` and only raw ``str`` or
    ``None`` cells. Invalid contracts raise ValueError. Resource caps return
    ``limited`` without selecting a schema or using a sampled subset. Shape
    caps are checked before values. Work includes two full feature passes
    (raw validation and parsing/screening) plus the proposer's conservative
    planned visits. Byte counters describe observed input, not a claim that
    every value was read after an input cap stopped validation. Optional work/
    check limits are for an isolated opt-in experiment; the ordinary caller
    retains the original 8M/48 policy and result shape. ``adaptive_checks``
    reduces the requested check count to fit the total planned work allowance;
    it never activates an incomplete top-one search.
    """
    custom_limits = max_cell_visits is not None or max_checked_configs is not None or adaptive_checks is not False
    if type(adaptive_checks) is not bool:
        raise ValueError("adaptive_checks must be a boolean")
    for value, ceiling, name in ((max_cell_visits, 12_000_000, "max_cell_visits"),
                                  (max_checked_configs, 192, "max_checked_configs")):
        if value is not None and (type(value) is not int or not 1 <= value <= ceiling):
            raise ValueError(name + " exceeds its supported positive integer bounds")
    work_limit = MAX_CELL_VISITS if max_cell_visits is None else max_cell_visits
    requested_checks = MAX_CHECKED_CONFIGS if max_checked_configs is None else max_checked_configs
    if not isinstance(X_fit, pd.DataFrame):
        raise ValueError("X_fit must be a raw feature DataFrame")
    features = _names(feature_columns, "feature_columns")
    numeric = _names(numeric_columns, "numeric_columns")
    if not features or not isinstance(period_column, str) or period_column not in features:
        raise ValueError("period_column must be one of feature_columns")
    if not set(numeric) <= set(features):
        raise ValueError("numeric_columns must be feature columns")
    if any(not isinstance(name, str) for name in X_fit.columns) or X_fit.columns.has_duplicates:
        raise ValueError("X_fit needs unique string column names")
    if len(X_fit.columns) != len(features) or set(X_fit.columns) != set(features):
        raise ValueError("X_fit must contain exactly the declared feature columns")

    rows = len(X_fit)
    counts = {"rows": rows, "features": len(features), "raw_cells": rows * len(features),
        "raw_cells_scanned": 0, "raw_value_bytes_observed": 0, "max_value_chars_observed": 0,
        "screening_cell_visits_planned": 2 * rows * len(features),
        "proposer_cell_visits_planned": 0, "total_cell_visits_planned": 2 * rows * len(features),
        "valid_period_rows": 0, "distinct_periods": 0, "screened_columns": 0,
        "excluded_constant": 0, "excluded_all_unique": 0, "excluded_no_cross_period_recurrence": 0,
        "recurrent_columns": 0, "selected_columns": 0, "proposer_calls": 0}
    screening = {"counts": counts, "selected_columns": [], "columns": [],
        "caps": {"max_rows": MAX_ROWS, "max_features": MAX_FEATURES, "max_cell_visits": work_limit,
            "max_raw_value_bytes": MAX_RAW_BYTES, "max_value_chars": MAX_VALUE_CHARS,
            "max_selected_columns": MAX_SELECTED_COLUMNS}}
    if custom_limits:
        screening["check_budget"] = {"requested": requested_checks, "chosen": None,
                                      "adaptive": adaptive_checks}

    def result(status, reason, config=None, proposer=None):
        value = {"version": VERSION, "scope": SCOPE, "status": status, "reason": reason,
            "selected_config": config, "screening": screening, "proposer_result": proposer,
            "identity_verified": False}
        # Detach every container and reject any accidental non-finite/non-JSON
        # dependency output. Raw strings/Decimal keys are never emitted.
        return json.loads(json.dumps(value, allow_nan=False))

    if rows > MAX_ROWS:
        return result("limited", "row_limit")
    if len(features) > MAX_FEATURES:
        return result("limited", "feature_limit")
    if counts["total_cell_visits_planned"] > work_limit:
        return result("limited", "work_limit")
    if not rows:
        return result("abstained", "no_rows")

    # Validate every raw feature, including those later excluded. No coercion,
    # dropna, stringification, dtype inference or row sampling is permitted.
    for name in features:
        for value in X_fit[name].tolist():
            counts["raw_cells_scanned"] += 1
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError("Raw feature cells must be str or None")
            counts["max_value_chars_observed"] = max(counts["max_value_chars_observed"], len(value))
            if len(value) > MAX_VALUE_CHARS:
                return result("limited", "value_length_limit")
            try:
                counts["raw_value_bytes_observed"] += len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("Raw feature cells must be valid UTF-8 strings") from error
            if counts["raw_value_bytes_observed"] > MAX_RAW_BYTES:
                return result("limited", "raw_bytes_limit")

    periods = [_number(value) for value in X_fit[period_column].tolist()]
    counts["valid_period_rows"] = sum(value is not None for value in periods)
    counts["distinct_periods"] = len({value for value in periods if value is not None})
    if counts["distinct_periods"] < 2:
        return result("abstained", "no_valid_period" if not counts["valid_period_rows"] else "single_period")
    candidates = []
    for name in features:
        if name == period_column:
            continue
        evidence = _column_evidence(X_fit[name].tolist(), periods, numeric=name in numeric)
        entry = {"column": name, **evidence}
        screening["columns"].append(entry)
        counts["screened_columns"] += 1
        if evidence["excluded_reason"]:
            counts["excluded_" + evidence["excluded_reason"]] += 1
        else:
            candidates.append(entry)
    candidates.sort(key=lambda item: (-item["clean_recurrent_rows"], -item["cross_period_recurrent_rows"], item["column"]))
    selected = [item["column"] for item in candidates[:MAX_SELECTED_COLUMNS]]
    screening["selected_columns"] = selected
    counts.update(recurrent_columns=len(candidates), selected_columns=len(selected))
    if not selected:
        return result("abstained", "no_recurrent_key_columns")

    # Account for parsing, one-column bounds and every admitted width<=3 check.
    # Adaptive admission can lower the check cap, but the proposer must still
    # certify the entire top-one search; a partial best-so-far is never used.
    checks = requested_checks
    if adaptive_checks:
        fixed_work = counts["screening_cell_visits_planned"] + rows * (2 * len(selected) + 1)
        affordable_checks = (work_limit - fixed_work) // (rows * (MAX_AUTO_KEY_WIDTH + 1))
        checks = max(0, min(checks, affordable_checks))
    if custom_limits:
        screening["check_budget"]["chosen"] = checks
    if not checks:
        return result("limited", "work_limit")
    proposer_work = rows * (2 * len(selected) + 1 + checks * (MAX_AUTO_KEY_WIDTH + 1))
    counts["proposer_cell_visits_planned"] = proposer_work
    counts["total_cell_visits_planned"] += proposer_work
    if counts["total_cell_visits_planned"] > work_limit:
        return result("limited", "work_limit")
    raw = {name: X_fit[name].tolist() for name in sorted([period_column, *selected])}
    counts["proposer_calls"] = 1
    proposer = propose_history_schemas(raw, task_type="regression", feature_columns=list(features),
        allowed_period_columns=[period_column], allowed_key_columns=list(selected),
        numeric_key_columns=[name for name in selected if name in numeric],
        positive_key_columns=(), explicit_key_sets=(),
        limits={"max_rows": MAX_ROWS, "max_work": work_limit - counts["screening_cell_visits_planned"],
            "max_period_columns": 1, "max_key_columns": MAX_SELECTED_COLUMNS,
            "max_auto_key_width": MAX_AUTO_KEY_WIDTH, "max_explicit_key_width": MAX_AUTO_KEY_WIDTH,
            "max_checked_configs": checks, "max_proposals": 1})
    if (not isinstance(proposer, dict) or proposer.get("version") != VERSION or proposer.get("scope") != SCOPE
            or proposer.get("auto_activate") is not False
            or proposer.get("status") not in {"limited", "abstained", "proposed"}):
        raise ValueError("Schema proposer returned an invalid result")
    if proposer["status"] == "limited":
        return result("limited", "proposer_limited", proposer=proposer)
    if proposer["status"] == "abstained":
        return result("abstained", "proposer_" + str(proposer.get("reason") or "abstained"), proposer=proposer)
    proposals = proposer.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1 or not isinstance(proposals[0], dict):
        raise ValueError("Schema proposer must return exactly one top proposal")
    config = proposals[0].get("config")
    if (not isinstance(config, dict)
            or set(config) != {"key_columns", "period_column", "numeric_columns", "positive_columns", "ambiguity_policy"}
            or config.get("period_column") != period_column
            or config.get("ambiguity_policy") != POLICY or not isinstance(config.get("key_columns"), list)
            or not 1 <= len(config["key_columns"]) <= MAX_AUTO_KEY_WIDTH
            or any(not isinstance(name, str) for name in config["key_columns"])
            or len(set(config["key_columns"])) != len(config["key_columns"])
            or not set(config["key_columns"]) <= set(selected)
            or config.get("numeric_columns") != [name for name in config["key_columns"] if name in numeric]
            or config.get("positive_columns") != [] or proposals[0].get("identity_verified") is not False):
        raise ValueError("Schema proposer changed the fixed screening/policy contract")
    return result("proposed", "structurally_eligible_for_inner_measurement", config=config, proposer=proposer)
