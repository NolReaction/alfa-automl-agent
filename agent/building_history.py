"""History features for a regression target: what earlier periods said about rows with the same key.

A key is a heuristic (for the building task: state, class, facility type, floor area, year built, elevation). Equal
keys are not proof of one object, so the features also tell how ambiguous the past matches are.

Rule against leakage: a row of period t sees only rows of periods strictly before t. This holds for the fitting rows
too, so their own answer, the answers of rows of the same period and all later periods are never used.

    encoder = HistoryEncoder(key_columns, period_column, numeric_columns=..., positive_columns=...)
    history_fit = encoder.fit_transform(X_fit, y_fit)       # features of the fitting rows
    history_new = encoder.transform(X_new)                  # features of new rows; their answers are not needed

Create a new encoder for every split and fit it on that split's fitting part only. The result holds only the history
features, in the order and with the index of X; the inputs are never changed.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import math
from typing import NamedTuple

import numpy as np
import pandas as pd


FEATURES = ("history_mean", "history_rows", "history_periods", "history_present", "history_ambiguous")


def _missing(value) -> bool:
    missing = pd.isna(value)
    if not np.isscalar(missing):
        raise ValueError("History key and period values must be scalars")
    return bool(missing) or (isinstance(value, str) and not value.strip())


def numeric_value(value) -> Decimal | None:
    """Parse a finite number without rounding strings, integers or Decimal inputs through float.

    Float inputs retain the precision already present in their string representation; precision lost before this
    function is called cannot be recovered. Equivalent decimal spellings identify the same numeric value.
    """
    if _missing(value):
        return None
    try:
        number = Decimal(int(value)) if isinstance(value, (bool, np.bool_)) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def period_json(value: Decimal | int) -> int | str:
    """JSON form of an exact numeric period (Decimal or int), never a binary float.

    An integral period becomes a JSON integer of any size ("5", "5.0" and "5E0" all give 5); any other period becomes
    its plain decimal text without trailing zeros ("2.50" gives "2.5"). Reading a period back: int, or Decimal(text).
    """
    value = Decimal(value)
    if value == value.to_integral_value():
        return int(value)
    return format(value.normalize(), "f")


def _add_exact(partials: list, value: float) -> None:
    """Add value to non-overlapping float partials (Shewchuk), keeping their sum exact.

    math.fsum(partials) is then the exact running total rounded once, the same number math.fsum gives for all values
    added so far, in O(len(partials)) instead of O(number of values). The list stays short for finite floats.
    """
    index = 0
    for other in partials:
        if abs(value) < abs(other):
            value, other = other, value
        high = value + other
        if not math.isfinite(high):
            raise ValueError("History target sums overflow the float range")
        low = other - (high - value)
        if low:
            partials[index] = low
            index += 1
        value = high
    partials[index:] = [value]


def _rounded(partials: list) -> float:
    try:
        return math.fsum(partials)
    except OverflowError as error:
        raise ValueError("History target sums overflow the float range") from error


def history_keys(X: pd.DataFrame, key_columns, numeric_columns=(), positive_columns=(), *, allow_missing=False) -> list:
    """Structured, exact keys shared by the encoder and the diagnostic checker.

    None is an internal missing component, distinct from every literal text value. Strict keys reject any missing
    component; the diagnostic lenient mode retains it inside the tuple so incomplete keys can match each other.
    """
    columns = []
    for name in key_columns:
        if name in numeric_columns:
            values = [numeric_value(value) for value in X[name]]
            if name in positive_columns:
                values = [value if value is not None and value > 0 else None for value in values]
        else:
            values = [None if _missing(value) else str(value).strip() for value in X[name]]
        columns.append(values)
    return [key if allow_missing or all(value is not None for value in key) else None for key in zip(*columns)]


class _Past(NamedTuple):
    """Running totals of one key after each of its periods (periods ascending)."""

    periods: list
    sums: list
    rows: list
    ambiguous: list


class HistoryEncoder:
    """Strict history features: a missing key part never creates a match.

    key_columns: columns that identify an object; period_column: numeric period of the row.
    numeric_columns: key columns compared as exact decimal values (without conversion to float); the others are compared as
    stripped text. positive_columns: numeric key columns where a value that is not positive means "unknown"
    (a year of construction), so such a row has no reliable key. Zero is valid in the other numeric columns.

    Features (no history: mean NaN, the others 0):
        history_mean       mean answer of the earlier rows with the same key
        history_rows       number of such rows
        history_periods    number of earlier periods in which the key occurs
        history_present    1 if there is any history
        history_ambiguous  1 if the key occurs more than once inside one earlier period
    """

    feature_names = FEATURES

    def __init__(self, key_columns, period_column, numeric_columns=(), positive_columns=()):
        self.key_columns = tuple(key_columns)
        self.period_column = period_column
        self.numeric_columns = tuple(numeric_columns)
        self.positive_columns = tuple(positive_columns)
        if not self.key_columns or len(set(self.key_columns)) != len(self.key_columns):
            raise ValueError("key_columns must be a non-empty list of different column names")
        if period_column in self.key_columns:
            raise ValueError("The period column must not be part of the key")
        if not set(self.numeric_columns) <= set(self.key_columns):
            raise ValueError("numeric_columns must be key columns")
        if not set(self.positive_columns) <= set(self.numeric_columns):
            raise ValueError("positive_columns must be numeric key columns")
        self._past: dict | None = None

    # ------------------------------------------------------------------ public

    def fit(self, X: pd.DataFrame, y) -> "HistoryEncoder":
        """Remember the answers per key and period. Replaces anything remembered before."""
        keys, periods = self._rows(X)
        answers = self._answers(X, y)
        per_period = defaultdict(list)
        for key, period, answer in zip(keys, periods, answers):
            if key is not None:
                per_period[(key, period)].append(answer)
        per_key = defaultdict(list)
        for (key, period), values in per_period.items():
            per_key[key].append((period, _rounded(values), len(values)))
        past = {}
        for key, rows in per_key.items():
            rows.sort()
            entry, partials, count, repeated = _Past([], [], [], []), [], 0, False
            for period, total, size in rows:
                # Running exact sum of the period totals: the same value as math.fsum over all earlier totals, in
                # linear instead of quadratic time for a key with a long history.
                _add_exact(partials, total)
                count += size
                repeated = repeated or size > 1
                entry.periods.append(period)
                entry.sums.append(_rounded(partials))
                entry.rows.append(count)
                entry.ambiguous.append(repeated)
            past[key] = entry
        self._past = past
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """History features of the rows of X from the remembered answers of strictly earlier periods."""
        if self._past is None:
            raise RuntimeError("HistoryEncoder is not fitted: call fit or fit_transform first")
        keys, periods = self._rows(X)
        size = len(keys)
        mean, rows = np.full(size, np.nan), np.zeros(size, dtype=np.int64)
        seen, ambiguous = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.int64)
        for i, (key, period) in enumerate(zip(keys, periods)):
            past = self._past.get(key) if key is not None else None
            before = bisect_left(past.periods, period) if past else 0  # periods strictly earlier than this one
            if before:
                mean[i] = past.sums[before - 1] / past.rows[before - 1]
                rows[i], seen[i], ambiguous[i] = past.rows[before - 1], before, int(past.ambiguous[before - 1])
        return pd.DataFrame({"history_mean": mean, "history_rows": rows, "history_periods": seen,
                             "history_present": (rows > 0).astype(np.int64), "history_ambiguous": ambiguous},
                            index=X.index)

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Fit on X and return the features of the same rows (each sees only strictly earlier periods)."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ inputs

    def _rows(self, X: pd.DataFrame) -> tuple[list, list]:
        """One structured key per row (tuple of values, None without a reliable key) and the period of each row."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing = [name for name in (*self.key_columns, self.period_column) if name not in X]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if X.columns.duplicated().any():
            raise ValueError("X must have unique column names")
        periods = [numeric_value(value) for value in X[self.period_column]]
        bad = sum(value is None for value in periods)
        if bad:
            raise ValueError(f"Period column {self.period_column!r} has {bad} missing or non-numeric values; "
                             "every row needs a period")
        return history_keys(X, self.key_columns, self.numeric_columns, self.positive_columns), periods

    @staticmethod
    def _answers(X: pd.DataFrame, y) -> np.ndarray:
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("The index of y differs from the index of X: align them first")
        try:
            answers = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain numbers") from error
        if answers.ndim != 1 or not np.isfinite(answers).all():
            raise ValueError("y must be one-dimensional with finite values only")
        return answers
