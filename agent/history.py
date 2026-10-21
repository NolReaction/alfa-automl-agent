"""Explicit past-only regression history features, independent of dataset names.

The caller supplies entity/time columns. No row ID or building identity is
inferred. Training rows exclude targets from their own and all later periods;
validation/test rows can use only earlier targets from the fitted training set.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _missing(value) -> bool:
    missing = pd.isna(value)
    if not np.isscalar(missing):
        raise ValueError("History entity/time values must be scalar values")
    return bool(missing) or (isinstance(value, str) and not value.strip())


def _add(group: dict, period, value: float) -> None:
    count, mean = group.get(period, (0, 0.0))
    updated = value if not count else mean * (count / (count + 1)) + value / (count + 1)
    if not np.isfinite(updated):
        raise ValueError("History target values produce a non-finite mean")
    group[period] = (count + 1, updated)


def _prefix(groups: dict):
    periods, counts, means, last = [], [], [], []
    cumulative_count, cumulative_mean = 0, 0.0
    for period in sorted(groups):
        count, mean = groups[period]
        total = cumulative_count + count
        cumulative_mean = (mean if not cumulative_count else
                           cumulative_mean * (cumulative_count / total) + mean * (count / total))
        cumulative_count = total
        periods.append(period)
        counts.append(cumulative_count)
        means.append(cumulative_mean)
        last.append(mean)
    return periods, counts, means, last


def _before(history, period):
    if period is None:
        return None
    position = bisect_left(history[0], period) - 1
    return position if position >= 0 else None


class PastOnlyHistory(TransformerMixin, BaseEstimator):
    """Add count/mean/latest-period mean using strictly earlier fitted rows.

    Missing entity values never identify a shared entity. An unseen/missing
    entity uses the global mean of strictly earlier fitted rows; without earlier
    rows, mean/last are zero. ``has_history`` denotes entity-specific history.
    Literal strings such as ``NA`` remain valid, exact entity-key components.
    Numeric periods retain their decimal ordering without float rounding.
    """

    def __init__(self, entity_columns, time_column, *, prefix="history", time_kind="numeric"):
        self.entity_columns = entity_columns
        self.time_column = time_column
        self.prefix = prefix
        self.time_kind = time_kind

    def _validate_parameters(self):
        if (not isinstance(self.entity_columns, (list, tuple)) or not self.entity_columns
                or any(not isinstance(name, str) or not name.strip() for name in self.entity_columns)
                or len(set(self.entity_columns)) != len(self.entity_columns)):
            raise ValueError("History requires explicit, unique entity_columns")
        if not isinstance(self.time_column, str) or not self.time_column.strip() or self.time_column in self.entity_columns:
            raise ValueError("History requires a time_column distinct from entity_columns")
        if not isinstance(self.prefix, str) or not self.prefix.strip():
            raise ValueError("History prefix must be a non-empty string")
        if not isinstance(self.time_kind, str) or self.time_kind not in {"numeric", "datetime"}:
            raise ValueError("History time_kind must be numeric or datetime")

    def _frame(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("History requires a DataFrame with unique named columns")
        required = set(self.entity_columns) | {self.time_column}
        if not required <= set(X.columns):
            raise ValueError("History entity/time columns are missing from the input")
        if set(self._output_names()) & set(X.columns):
            raise ValueError("History feature names would overwrite existing input columns")
        return X

    def _output_names(self):
        return tuple(f"{self.prefix}_{suffix}" for suffix in ("count", "mean", "last", "has_history"))

    def _period(self, value):
        if _missing(value):
            return None
        if self.time_kind == "numeric":
            try:
                result = Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
            return result if result.is_finite() else None
        if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:[T\s]|$)", value):
            # Reject locale-dependent date strings instead of guessing day/month.
            return None
        if not isinstance(value, (str, datetime, date, pd.Timestamp, np.datetime64)):
            return None
        try:
            result = pd.to_datetime(value, utc=True, errors="coerce")
            return None if pd.isna(result) else result.value
        except (ValueError, OverflowError):
            return None

    def _key(self, values):
        if any(_missing(value) for value in values):
            return None
        return tuple(str(value) for value in values)

    def fit(self, X, y):
        self._validate_parameters()
        X = self._frame(X)
        try:
            target = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("History requires numeric regression targets") from error
        if target.ndim != 1 or len(target) != len(X) or not len(X) or not np.isfinite(target).all():
            raise ValueError("History requires one finite numeric target per non-empty training row")
        entity_groups, global_groups = {}, {}
        selected = X.loc[:, [*self.entity_columns, self.time_column]]
        for values, answer in zip(selected.itertuples(index=False, name=None), target):
            period, key = self._period(values[-1]), self._key(values[:-1])
            if period is None:
                continue
            _add(global_groups, period, float(answer))
            if key is not None:
                _add(entity_groups.setdefault(key, {}), period, float(answer))
        self.global_history_ = _prefix(global_groups)
        self.entity_history_ = {key: _prefix(group) for key, group in entity_groups.items()}
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise ValueError("History does not accept additional fitting parameters")
        # transform's strict bisect_left excludes every target in the same
        # period, including the row's own target and same-period duplicates.
        return self.fit(X, y).transform(X)

    def transform(self, X):
        check_is_fitted(self, ["global_history_", "entity_history_"])
        X = self._frame(X)
        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError("History input columns differ from the fitted training columns")
        count, mean, latest, flag = [], [], [], []
        selected = X.loc[:, [*self.entity_columns, self.time_column]]
        for values in selected.itertuples(index=False, name=None):
            period, key = self._period(values[-1]), self._key(values[:-1])
            global_position = _before(self.global_history_, period)
            fallback = self.global_history_[2][global_position] if global_position is not None else 0.0
            history = self.entity_history_.get(key) if key is not None else None
            position = _before(history, period) if history is not None else None
            if position is None:
                count.append(0)
                mean.append(fallback)
                latest.append(fallback)
                flag.append(0)
            else:
                count.append(history[1][position])
                mean.append(history[2][position])
                latest.append(history[3][position])
                flag.append(1)
        output = X.copy(deep=True)
        for name, values in zip(self._output_names(), (count, mean, latest, flag)):
            output[name] = values
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        names = self.feature_names_in_.tolist() if input_features is None else list(input_features)
        if names != self.feature_names_in_.tolist():
            raise ValueError("History feature names differ from the fitted input")
        return np.asarray([*names, *self._output_names()], dtype=object)
