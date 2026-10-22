"""Bounded, strictly past-only regression history and calibration-only blending.

Pass raw feature values and an explicit ``HistoryConfig``; this module never
discovers keys, reads files, or fits a base model. Keys and periods use the exact
decimal semantics shared with ``building_history``. Missing keys/periods simply
have no history. A query at t sees only fitted periods < t, including when the
query is the fitting frame itself. Query batches never affect one another.

``decay`` weights each earlier observation by 0.5 ** (latest_past_period -
observation_period): these are numeric period gaps, not ranks of observed
periods. Gaps >= 1075 have zero floating-point weight. Normalizing relative to
the latest past period avoids underflow for a query far in the future. ``age``
is query period minus latest past period, saturated at the largest finite float.

``select_history`` is exclusively for predictions/labels from a caller-owned
calibration split inside fitting data. It cannot verify where those arrays came
from: never pass outer validation/test labels. Its fixed grid contains no
task-specific names or values, and always includes retaining the base model.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
import math
from numbers import Real

import numpy as np
import pandas as pd

from agent.building_history import history_keys, numeric_value
from agent.history_candidate import HistoryConfig


SIGNAL_COLUMNS = ("mean", "last", "decay", "past_rows", "past_periods", "age", "eligible", "ambiguous")
MODES = ("mean", "last", "decay")
MAX_ROWS = 500_000
MAX_KEY_COLUMNS = 32
MAX_KEY_CELLS = 8_000_000
MAX_OPTIONS = 24
MAX_DECAY_GAP = 1075.0
_FLOAT_MAX = float(np.finfo(np.float64).max)
DEFAULT_OPTIONS = ({"mode": "mean", "weight": 0.0, "shrinkage": 0},) + tuple(
    {"mode": mode, "weight": weight, "shrinkage": shrinkage}
    for mode in MODES for weight in (0.25, 0.5, 0.75) for shrinkage in (0, 1)
)


def _frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.duplicated().any():
        raise ValueError("X must have unique column names")
    if len(X) > MAX_ROWS:
        raise ValueError(f"History row cap exceeded ({MAX_ROWS})")
    return X


def _vector(values, size: int, name: str, index=None) -> np.ndarray:
    if isinstance(values, pd.Series) and index is not None and not values.index.equals(index):
        raise ValueError(f"The index of {name} differs from the frame index")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one-dimensional with {size} finite values")
    return result


def _sum(values) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("History arithmetic overflowed") from error
    if not math.isfinite(result):
        raise ValueError("History arithmetic must be finite")
    return result


def _gap(newer: Decimal, older: Decimal, cap: float) -> float:
    # Exact Decimal comparisons establish causality. This floating signal needs
    # guard digits both for near cancellation and differently scaled operands;
    # significant-digit lengths alone would round 1 - 1e-5 to 1. A fixed guard
    # avoids allocating precision proportional to an adversarial exponent gap.
    with localcontext() as context:
        context.prec = max(len(newer.as_tuple().digits), len(older.as_tuple().digits)) + 34
        context.Emax, context.Emin = MAX_EMAX, MIN_EMIN
        # Caller Decimal traps must not change this floating diagnostic. An
        # overflowed positive difference is necessarily above the finite cap.
        for signal in context.traps:
            context.traps[signal] = False
        difference = newer - older
        return cap if difference >= Decimal(str(cap)) else float(difference)


@dataclass(frozen=True)
class _Past:
    periods: tuple[Decimal, ...]
    mean: tuple[float, ...]
    last: tuple[float, ...]
    decay: tuple[float, ...]
    rows: tuple[int, ...]
    ambiguous: tuple[bool, ...]


class HistorySignals:
    """Past-only signals with the existing allow/past_unique_only policy.

    Missing required columns deactivate the affected frame rather than inventing
    a partial key. Ineligible rows retain diagnostic history when ambiguity is
    the cause; absent history has NaN means/age and zero counts. ``last`` is the
    latest past period's mean, so even the explicit ``allow`` policy is invariant
    to fit-row order. Under ``past_unique_only``, any repeated prior period makes
    the row ineligible, regardless of whether its repeated targets agree.
    """

    def __init__(self, config: HistoryConfig | dict):
        self.config = HistoryConfig.from_json(config.to_json() if isinstance(config, HistoryConfig) else config)
        if len(self.config.key_columns) > MAX_KEY_COLUMNS:
            raise ValueError(f"History key column cap exceeded ({MAX_KEY_COLUMNS})")
        self._past: dict | None = None

    def _rows(self, X: pd.DataFrame) -> tuple[list, list]:
        X = _frame(X)
        if len(X) * len(self.config.columns) > MAX_KEY_CELLS:
            raise ValueError(f"History key cell cap exceeded ({MAX_KEY_CELLS})")
        if any(column not in X for column in self.config.columns):
            return [None] * len(X), [None] * len(X)
        keys = history_keys(X, self.config.key_columns, self.config.numeric_columns, self.config.positive_columns)
        periods = [numeric_value(value) for value in X[self.config.period_column]]
        return keys, periods

    def fit(self, X: pd.DataFrame, y) -> "HistorySignals":
        keys, periods = self._rows(X)
        answers = _vector(y, len(X), "y", X.index)
        grouped = defaultdict(list)
        for key, period, answer in zip(keys, periods, answers):
            if key is not None and period is not None:
                grouped[(key, period)].append(float(answer))
        per_key = defaultdict(list)
        for (key, period), values in grouped.items():
            # Sorting ensures even adversarial cancellation follows exactly the
            # same arithmetic after any permutation of the fitting rows.
            per_key[key].append((period, _sum(sorted(values)), len(values)))
        past = {}
        for key, entries in per_key.items():
            entries.sort(key=lambda item: item[0])
            ordered, means, lasts, decays, counts, ambiguous = [], [], [], [], [], []
            # Non-overlapping partials preserve fsum's prefix accuracy without
            # quadratic rescanning of a key's complete earlier history.
            partials = []
            count, repeated = 0, False
            decay_sum, decay_count = 0.0, 0.0
            previous = None
            for period, total, size in entries:
                value = total
                position = 0
                for other in partials:
                    if abs(value) < abs(other):
                        value, other = other, value
                    high = value + other
                    if not math.isfinite(high):
                        raise ValueError("History arithmetic overflowed")
                    low = other - (high - value)
                    if low:
                        partials[position] = low
                        position += 1
                    value = high
                partials[position:] = [value]
                count += size
                repeated = repeated or size > 1
                gap = _gap(period, previous, MAX_DECAY_GAP) if previous is not None else MAX_DECAY_GAP
                factor = 0.0 if gap >= MAX_DECAY_GAP else math.exp2(-gap)
                decay_sum = _sum((decay_sum * factor, total))
                decay_count = _sum((decay_count * factor, size))
                ordered.append(period)
                means.append(_sum(partials) / count)
                lasts.append(total / size)
                decays.append(decay_sum / decay_count)
                counts.append(count)
                ambiguous.append(repeated)
                previous = period
            past[key] = _Past(tuple(ordered), tuple(means), tuple(lasts), tuple(decays), tuple(counts), tuple(ambiguous))
        self._past = past
        return self

    def transform(self, X_query: pd.DataFrame) -> pd.DataFrame:
        if self._past is None:
            raise RuntimeError("HistorySignals is not fitted")
        keys, periods = self._rows(X_query)
        size = len(X_query)
        values = {name: np.full(size, np.nan) for name in ("mean", "last", "decay", "age")}
        values.update({name: np.zeros(size, dtype=np.int64) for name in ("past_rows", "past_periods")})
        values.update({name: np.zeros(size, dtype=bool) for name in ("eligible", "ambiguous")})
        for row, (key, period) in enumerate(zip(keys, periods)):
            history = self._past.get(key) if key is not None and period is not None else None
            before = bisect_left(history.periods, period) if history else 0
            if not before:
                continue
            i = before - 1
            values["mean"][row], values["last"][row], values["decay"][row] = history.mean[i], history.last[i], history.decay[i]
            values["past_rows"][row], values["past_periods"][row] = history.rows[i], before
            values["age"][row] = _gap(period, history.periods[i], _FLOAT_MAX)
            values["ambiguous"][row] = history.ambiguous[i]
            values["eligible"][row] = self.config.ambiguity_policy == "allow" or not history.ambiguous[i]
        return pd.DataFrame(values, columns=SIGNAL_COLUMNS, index=X_query.index)

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def _option(mode, weight, shrinkage) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real) or not math.isfinite(weight) \
            or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and between 0 and 1")
    if isinstance(shrinkage, (bool, np.bool_)) or not isinstance(shrinkage, Real) or shrinkage not in (0, 1):
        raise ValueError("shrinkage must be 0 or 1")
    return {"mode": mode, "weight": float(weight), "shrinkage": int(shrinkage)}


def apply_history(base_predictions, signals: pd.DataFrame, mode="mean", weight=0.5, shrinkage=0) -> np.ndarray:
    """Blend eligible rows with weight * n/(n+shrinkage), n = past row count.

    ``eligible`` carries the configured ambiguity policy. Consequently strict
    signals keep ambiguous rows unchanged; ``allow`` explicitly permits them.
    All base predictions must be finite. Absent/ineligible signal NaNs are valid.
    """
    option = _option(mode, weight, shrinkage)
    signals = _frame(signals)
    required = {mode, "past_rows", "eligible"}
    if not required <= set(signals):
        raise ValueError(f"signals need columns {sorted(required)}")
    base = _vector(base_predictions, len(signals), "base_predictions", signals.index)
    result = base.copy()
    eligibility = signals["eligible"].to_numpy()
    if not all(isinstance(value, (bool, np.bool_)) for value in eligibility):
        raise ValueError("eligible must contain booleans")
    if not option["weight"]:
        return result
    applied = eligibility.astype(bool)
    if not applied.any():
        return result
    history = _vector(signals.loc[applied, mode], int(applied.sum()), mode)
    counts = _vector(signals.loc[applied, "past_rows"], int(applied.sum()), "past_rows")
    if (counts < 1).any() or (counts > MAX_ROWS).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Eligible past_rows must be positive bounded integers")
    effective = option["weight"] * counts / (counts + option["shrinkage"])
    with np.errstate(over="raise", invalid="raise"):
        try:
            result[applied] = (1.0 - effective) * base[applied] + effective * history
        except FloatingPointError as error:
            raise ValueError("The blended prediction is not finite") from error
    if not np.isfinite(result).all():
        raise ValueError("The blended prediction is not finite")
    return result


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            errors = y - prediction
        except FloatingPointError as error:
            raise ValueError("Calibration residuals overflowed") from error
    scale = float(np.max(np.abs(errors))) if len(errors) else 0.0
    if not scale:
        return 0.0
    # Scaling avoids overflow on squared residuals; sorted fsum also makes
    # calibration row permutations exactly invariant.
    return scale * math.sqrt(_sum(sorted(float(value) ** 2 for value in errors / scale)) / len(errors))


def select_history(calibration_signals: pd.DataFrame, calibration_y, basepred, options=None) -> dict:
    """Select from a small preregistered grid using caller-owned fit calibration.

    Ties retain no history, then favor mean/last/decay, lower weight, and no
    shrinkage in that order. Explicit options are dictionaries with mode, weight,
    shrinkage; weights must come from {0, .25, .5, .75}. The zero-weight option is
    always present, canonicalized to mean/no shrinkage. Returns JSON-compatible
    chosen parameters and calibration diagnostics; it never consumes query data.
    """
    signals = _frame(calibration_signals)
    if not len(signals):
        raise ValueError("Calibration must contain at least one row")
    y = _vector(calibration_y, len(signals), "calibration_y", signals.index)
    base = _vector(basepred, len(signals), "basepred", signals.index)
    supplied = DEFAULT_OPTIONS if options is None else options
    if not isinstance(supplied, (list, tuple)) or len(supplied) > MAX_OPTIONS:
        raise ValueError(f"options must be a list or tuple of at most {MAX_OPTIONS} choices")
    candidates = {(0.0, 0, 0): {"mode": "mean", "weight": 0.0, "shrinkage": 0}}
    for item in supplied:
        if not isinstance(item, dict) or set(item) != {"mode", "weight", "shrinkage"}:
            raise ValueError("Each option must contain exactly mode, weight, shrinkage")
        option = _option(**item)
        if option["weight"] not in (0.0, 0.25, 0.5, 0.75):
            raise ValueError("Selection weights must be in the preregistered grid: 0, .25, .5, .75")
        if option["weight"]:
            candidates[(option["weight"], MODES.index(option["mode"]), option["shrinkage"])] = option
    # Explicit sort makes both option order and calibration row order irrelevant.
    ordered = sorted(candidates.values(), key=lambda item: (item["weight"] != 0, MODES.index(item["mode"]),
                                                            item["weight"], item["shrinkage"]))
    baseline = _rmse(y, base)
    # Validate even a baseline-only explicit grid consistently.
    apply_history(base, signals, **ordered[0])
    best, best_score = ordered[0], baseline
    for option in ordered[1:]:
        score = _rmse(y, apply_history(base, signals, **option))
        if score < best_score:
            best, best_score = option, score
    if "eligible" not in signals:
        raise ValueError("signals need an eligible column")
    return {**best, "rmse": best_score, "baseline_rmse": baseline, "evaluated_options": len(ordered),
            "rows": len(signals), "eligible_rows": int(signals["eligible"].sum())}
