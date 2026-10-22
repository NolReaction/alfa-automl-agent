"""Host-only error partitions for the preregistered regression rescue screen.

This module is never a runtime candidate. Held labels are used only for error
measurement and held-target diagnostic bands. All cutpoints, category counts,
top levels and reference history come from fit data. Multiple prediction arms
share one set of masks. Each partition is a complete, non-overlapping view of
the same rows; SSE must not be added across different partitions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from agent.building_history import numeric_value, period_json
from agent.history_candidate import HistoryConfig
from agent.regression_history import HistorySignals, MAX_ROWS


def _frame(X, name):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if X.columns.duplicated().any():
        raise ValueError(f"{name} must have unique columns")
    if len(X) > MAX_ROWS:
        raise ValueError(f"{name} exceeds the diagnostic row cap ({MAX_ROWS})")
    return X


def _vector(values, X, name):
    if isinstance(values, pd.Series) and not values.index.equals(X.index):
        raise ValueError(f"{name} index must match its feature frame")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have one finite value per feature row")
    return result


def _text(value):
    missing = pd.isna(value)
    if not np.isscalar(missing):
        raise ValueError("Diagnostic feature values must be scalars")
    return None if missing or (isinstance(value, str) and not value.strip()) else str(value).strip()


def _numbers(values):
    result = []
    for value in values:
        parsed = numeric_value(value)
        number = float(parsed) if parsed is not None else np.nan
        result.append(number if math.isfinite(number) else np.nan)
    return np.asarray(result, dtype=float)


def _quantiles(values, probabilities):
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    with np.errstate(over="raise", invalid="raise"):
        try:
            cutpoints = np.quantile(values, probabilities, method="linear")
        except FloatingPointError as error:
            raise ValueError("Fit diagnostic quantiles are not finite") from error
    if not np.isfinite(cutpoints).all():
        raise ValueError("Fit diagnostic quantiles are not finite")
    return tuple(float(value) for value in cutpoints)


def _statistics(errors, mask, total_sse):
    chosen = errors[mask]
    count = len(chosen)
    try:
        sse = math.fsum(sorted(float(value) ** 2 for value in chosen))
        absolute = math.fsum(sorted(abs(float(value)) for value in chosen))
    except OverflowError as error:
        raise ValueError("Diagnostic error totals overflowed") from error
    if not math.isfinite(sse) or not math.isfinite(absolute):
        raise ValueError("Diagnostic error totals must be finite")
    return {"n": count, "row_share": count / len(errors) if len(errors) else None,
            "sse": sse, "sse_share": sse / total_sse if total_sse else None,
            "rmse": math.sqrt(sse / count) if count else None,
            "mae": absolute / count if count else None}


def segment_rows(X_fit, y_fit, X_held, y_held, predictions, *, period_column, size_column,
                 categorical, reference_history_config) -> list[dict]:
    """Return JSON-safe metrics for all frozen error partitions.

    ``predictions`` is one vector, or a mapping arm name -> vector (the latter
    adds an ``arm`` field). All vectors are positional; Series indexes must
    match their frame, including duplicate indexes. Empty groups have zero n/SSE
    and null RMSE/MAE; SSE shares are null when total SSE is zero.

    Target/size bands are right-closed, with fit-only linear quantiles recorded
    as ``cutpoints``. Missing size is separate; if fit contains no valid size,
    held nonmissing sizes use ``unavailable_fit_cutpoints``. Category novelty
    ignores missing values: any new nonmissing value takes priority over any
    rare fit value, then common. Missingness has its own partition. Top levels
    exclude missing, ties sort lexically, and missing falls into ``other``.
    Recency ``one`` means 0 < numeric gap <= 1; ``two_or_more`` means > 1.
    """
    X_fit, X_held = _frame(X_fit, "X_fit"), _frame(X_held, "X_held")
    if not len(X_fit):
        raise ValueError("X_fit must contain at least one row")
    missing_columns = [column for column in X_fit if column not in X_held]
    if missing_columns:
        raise ValueError(f"X_held lacks fitted feature columns {missing_columns}")
    if not isinstance(categorical, (list, tuple)) or len(set(categorical)) != len(categorical):
        raise ValueError("categorical must be a list or tuple of distinct feature columns")
    declared = [column for column in (period_column, size_column, *categorical) if column is not None]
    if any(column not in X_fit for column in declared):
        raise ValueError("Declared diagnostic columns must exist in the fit features")
    fit_y, held_y = _vector(y_fit, X_fit, "y_fit"), _vector(y_held, X_held, "y_held")
    is_multi = isinstance(predictions, Mapping)
    arms = list(predictions.items()) if is_multi else [(None, predictions)]
    if not arms or any(is_multi and (not isinstance(name, str) or not name) for name, _ in arms):
        raise ValueError("Prediction mapping needs nonempty string arm names")
    validated_arms = [(name, _vector(values, X_held, f"predictions[{name}]")) for name, values in arms]
    size = len(X_held)
    masks = []

    def add(partition, segment, mask, **metadata):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (size,):
            raise ValueError("Internal diagnostic mask shape mismatch")
        masks.append((partition, segment, mask, metadata))

    add("overall", "all", np.ones(size, dtype=bool))
    if reference_history_config is None:
        rows = np.zeros(size, dtype=np.int64)
        ambiguous = np.zeros(size, dtype=bool)
        age = np.full(size, np.nan)
    else:
        config = reference_history_config
        if isinstance(config, HistoryConfig):
            config = config.to_json()
        signals = HistorySignals(config).fit(X_fit, fit_y).transform(X_held)
        rows = signals["past_rows"].to_numpy()
        ambiguous = signals["ambiguous"].to_numpy()
        age = signals["age"].to_numpy()
    present = rows > 0
    add("history", "eligible", present & ~ambiguous)
    add("history", "absent", ~present)
    add("history", "ambiguous", present & ambiguous)
    for value in (0, 1, 2):
        add("past_rows", str(value), rows == value)
    add("past_rows", "3+", rows >= 3)
    add("recency", "none", ~present)
    add("recency", "one", present & (age <= 1))
    add("recency", "two_or_more", present & (age > 1))

    target_cuts = _quantiles(fit_y, (0.5, 0.9, 0.99))
    p50, p90, p99 = target_cuts
    cuts = {"p50": p50, "p90": p90, "p99": p99}
    add("target", "<=p50", held_y <= p50, cutpoints=cuts)
    add("target", "(p50,p90]", (held_y > p50) & (held_y <= p90), cutpoints=cuts)
    add("target", "(p90,p99]", (held_y > p90) & (held_y <= p99), cutpoints=cuts)
    add("target", ">p99", held_y > p99, cutpoints=cuts)

    fit_size = _numbers(X_fit[size_column]) if size_column is not None else np.full(len(X_fit), np.nan)
    held_size = _numbers(X_held[size_column]) if size_column is not None else np.full(size, np.nan)
    size_cuts = _quantiles(fit_size, (0.5, 0.9))
    valid_size = np.isfinite(held_size)
    add("size", "missing", ~valid_size)
    if size_cuts is None:
        add("size", "unavailable_fit_cutpoints", valid_size)
    else:
        p50, p90 = size_cuts
        cuts = {"p50": p50, "p90": p90}
        add("size", "<=p50", valid_size & (held_size <= p50), cutpoints=cuts)
        add("size", "(p50,p90]", valid_size & (held_size > p50) & (held_size <= p90), cutpoints=cuts)
        add("size", ">p90", valid_size & (held_size > p90), cutpoints=cuts)

    missing_count = np.zeros(size, dtype=np.int64)
    for column in X_fit:
        missing_count += np.fromiter((_text(value) is None for value in X_held[column]), bool, size)
    add("missing_features", "0", missing_count == 0)
    add("missing_features", "1-5", (missing_count >= 1) & (missing_count <= 5))
    add("missing_features", "6+", missing_count >= 6)

    any_new, any_rare = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
    rarity_threshold = max(20, math.ceil(0.001 * len(X_fit)))
    for position, column in enumerate(categorical):
        fit_values = [_text(value) for value in X_fit[column]]
        held_values = [_text(value) for value in X_held[column]]
        counts = Counter(value for value in fit_values if value is not None)
        any_new |= np.fromiter((value is not None and value not in counts for value in held_values), bool, size)
        any_rare |= np.fromiter((value is not None and 0 < counts[value] < rarity_threshold for value in held_values), bool, size)
        if position < 2:
            top = sorted(counts, key=lambda value: (-counts[value], value))[:3]
            for value in top:
                add(f"category:{column}", f"level:{value}", np.fromiter((item == value for item in held_values), bool, size))
            add(f"category:{column}", "other", np.fromiter((value not in top for value in held_values), bool, size))
    add("category_novelty", "new", any_new, rare_count_threshold=rarity_threshold)
    add("category_novelty", "rare", ~any_new & any_rare, rare_count_threshold=rarity_threshold)
    add("category_novelty", "common", ~any_new & ~any_rare, rare_count_threshold=rarity_threshold)

    periods = [numeric_value(value) for value in X_held[period_column]] if period_column is not None else [None] * size
    for value in sorted({period for period in periods if period is not None}):
        add("period", str(period_json(value)), np.fromiter((period == value for period in periods), bool, size))
    add("period", "missing", np.fromiter((period is None for period in periods), bool, size))

    # Check our host partitions before assigning any arm's errors to them.
    coverage = {}
    for partition, _, mask, _ in masks:
        coverage.setdefault(partition, np.zeros(size, dtype=np.int64))[:] += mask
    if any(not np.equal(counts, 1).all() for counts in coverage.values()):
        raise ValueError("Diagnostic groups must partition every held row exactly once")
    result = []
    for name, prediction in validated_arms:
        with np.errstate(over="raise", invalid="raise"):
            try:
                errors = held_y - prediction
            except FloatingPointError as error:
                raise ValueError("Diagnostic residuals overflowed") from error
        total_sse = _statistics(errors, np.ones(size, dtype=bool), None)["sse"]
        for partition, segment, mask, metadata in masks:
            record = {"partition": partition, "segment": segment,
                      **_statistics(errors, mask, total_sse), **metadata}
            if is_multi:
                record["arm"] = name
            result.append(record)
    return result
