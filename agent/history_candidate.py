"""History candidate for AutoML: blend a regression prediction with the mean of earlier answers of the same key.

Contract v1 (docs/history_candidate_contract.md):

    candidate = fit_history_candidate(task_type="regression", metric_name="RMSE", feature_columns=features,
                                      X_raw=X_fit_raw, y=y_fit, config=config, blend_weight=0.5)
    result = candidate.apply(X_valid_raw, base_prediction)   # no answers, no refitting, no state change
    result.prediction, result.applied_mask, result.diagnostics

History is used only when the integrator passes an explicit config: the key columns and the numeric period column
are never guessed from column names, ids, file names or the target. A key match is a heuristic, not proof of one
object. A row of period t uses only fitted answers of periods strictly before t (HistoryEncoder), so its prediction
does not depend on the other rows of the query batch.

Prediction of an applied row: (1 - blend_weight) * base_prediction + blend_weight * past_mean, in the original target
scale (no rounding, thresholds or clipping). Every other row keeps the base prediction exactly. An inactive candidate
(no config, classification, unknown task type or metric, missing columns) returns the base prediction unchanged.

Raw values: pass the key and period columns as read, before model preprocessing converts them to float.
    numeric key parts and the period: exact decimals of str(value).strip() ("61242.0" equals "61242", nothing is
        rounded); a value that is not a finite number is missing; in positive columns a value <= 0 is missing too;
    text key parts: str(value).strip(), compared exactly; None, NaN and empty or blank text are missing, while literal
        text such as "NA" or "?" is an ordinary value (a float 1.0 in a text column becomes the text "1.0");
    a missing key part gives the row no key, so it never matches (no lenient matching);
    a missing or non-numeric period gives the row no history: when fitting, its answer is not remembered; when
        applying, the row keeps the base prediction. No row is dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd

from agent.building_history import HistoryEncoder, history_keys, numeric_value, period_json
from agent.metrics import get_metric


POLICIES = ("allow", "past_unique_only")
CONFIG_FIELDS = ("key_columns", "period_column", "numeric_columns", "positive_columns", "ambiguity_policy")


def _column_names(value, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of column names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{field} must contain non-empty strings only")
    duplicates = sorted(name for name, count in Counter(value).items() if count > 1)
    if duplicates:
        raise ValueError(f"{field} contains duplicate column names: {duplicates}")
    if not value and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    return tuple(value)


@dataclass(frozen=True)
class HistoryConfig:
    """The JSON configuration of the candidate; see the module docstring for the meaning of the columns.

    ambiguity_policy: "allow" blends every row with past history; "past_unique_only" blends only rows whose key never
    repeated inside one of the available earlier periods.
    """

    key_columns: tuple[str, ...]
    period_column: str
    numeric_columns: tuple[str, ...] = ()
    positive_columns: tuple[str, ...] = ()
    ambiguity_policy: str = "allow"

    @classmethod
    def from_json(cls, config) -> "HistoryConfig":
        if not isinstance(config, dict):
            raise ValueError("The history config must be a JSON object (dict) or None")
        unknown = sorted(set(config) - set(CONFIG_FIELDS))
        if unknown:
            raise ValueError(f"Unknown history config fields: {unknown}; allowed: {list(CONFIG_FIELDS)}")
        for field in ("key_columns", "period_column"):
            if field not in config:
                raise ValueError(f"The history config needs {field!r}")
        period = config["period_column"]
        if not isinstance(period, str) or not period.strip():
            raise ValueError("period_column must be a non-empty column name")
        result = cls(_column_names(config["key_columns"], "key_columns", allow_empty=False), period,
                     _column_names(config.get("numeric_columns", []), "numeric_columns", allow_empty=True),
                     _column_names(config.get("positive_columns", []), "positive_columns", allow_empty=True),
                     config.get("ambiguity_policy", "allow"))
        if result.ambiguity_policy not in POLICIES:
            raise ValueError(f"ambiguity_policy must be one of {list(POLICIES)}, got {result.ambiguity_policy!r}")
        if period in result.key_columns:
            raise ValueError("period_column must not be one of the key_columns")
        if not set(result.numeric_columns) <= set(result.key_columns):
            raise ValueError("numeric_columns must be key columns")
        if not set(result.positive_columns) <= set(result.numeric_columns):
            raise ValueError("positive_columns must be numeric key columns")
        return result

    @property
    def columns(self) -> list[str]:
        return [*self.key_columns, self.period_column]

    def to_json(self) -> dict:
        return {"key_columns": list(self.key_columns), "period_column": self.period_column,
                "numeric_columns": list(self.numeric_columns), "positive_columns": list(self.positive_columns),
                "ambiguity_policy": self.ambiguity_policy}

    def encoder(self) -> HistoryEncoder:
        return HistoryEncoder(self.key_columns, self.period_column, self.numeric_columns, self.positive_columns)


@dataclass(frozen=True)
class HistoryApplication:
    prediction: np.ndarray
    applied_mask: np.ndarray
    diagnostics: dict


def _check_weight(blend_weight) -> float:
    if isinstance(blend_weight, bool) or not isinstance(blend_weight, Real) or not math.isfinite(blend_weight) \
            or not 0 <= blend_weight <= 1:
        raise ValueError(f"blend_weight must be a finite number in [0, 1], got {blend_weight!r}")
    return float(blend_weight)


def _check_frame(X_raw) -> pd.DataFrame:
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("X_raw must be a pandas DataFrame of raw feature values")
    if X_raw.columns.duplicated().any():
        raise ValueError("X_raw must have unique column names")
    return X_raw


def _answers(X_raw: pd.DataFrame, y) -> np.ndarray:
    if len(y) != len(X_raw):
        raise ValueError(f"X_raw has {len(X_raw)} rows but y has {len(y)}")
    if isinstance(y, pd.Series) and not y.index.equals(X_raw.index):
        raise ValueError("The index of y differs from the index of X_raw: align them first")
    try:
        answers = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("A regression target must contain numbers") from error
    if answers.ndim != 1 or not np.isfinite(answers).all():
        raise ValueError("y must be one-dimensional with finite values only")
    return answers


def _valid_periods(X_raw: pd.DataFrame, config: HistoryConfig) -> np.ndarray:
    return np.fromiter((numeric_value(value) is not None for value in X_raw[config.period_column]), bool, len(X_raw))


def _keyed(X_raw: pd.DataFrame, config: HistoryConfig) -> list:
    return history_keys(X_raw, config.key_columns, config.numeric_columns, config.positive_columns)


class FittedHistoryCandidate:
    """Result of fit_history_candidate. Inactive candidates explain why in `reason` and change no prediction."""

    def __init__(self, *, task_type, metric_name, blend_weight: float, config: HistoryConfig | None,
                 reason: str | None, encoder: HistoryEncoder | None = None, fit_diagnostics: dict | None = None):
        self.task_type = task_type
        self.metric_name = metric_name
        self.blend_weight = blend_weight
        self.config = config
        self.reason = reason
        self._encoder = encoder
        self.fit_diagnostics = dict(fit_diagnostics or {})

    @property
    def active(self) -> bool:
        return self.reason is None

    def _base_diagnostics(self, rows: int) -> dict:
        return {"status": "active" if self.active else "inactive", "reason": self.reason,
                "task_type": self.task_type, "metric": self.metric_name, "blend_weight": self.blend_weight,
                "ambiguity_policy": self.config.ambiguity_policy if self.config else None, "rows": rows,
                "fit": self.fit_diagnostics}

    def apply(self, X_raw: pd.DataFrame, base_prediction: np.ndarray) -> HistoryApplication:
        """Blend base_prediction with the past mean where allowed; answers are not needed and nothing is refitted."""
        X_raw = _check_frame(X_raw)
        if not self.active:
            prediction = np.array(base_prediction, copy=True)
            if prediction.ndim == 0 or len(prediction) != len(X_raw):
                raise ValueError(f"base_prediction must have one entry per row of X_raw ({len(X_raw)})")
            return HistoryApplication(prediction, np.zeros(len(X_raw), dtype=bool), self._base_diagnostics(len(X_raw)))

        config = self.config
        missing = [name for name in config.columns if name not in X_raw]
        if missing:
            raise ValueError(f"X_raw lacks the history columns {missing}")
        try:
            base = np.asarray(base_prediction, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("A regression base_prediction must contain numbers") from error
        if base.shape != (len(X_raw),):
            raise ValueError(f"base_prediction must be one-dimensional with {len(X_raw)} values, got shape {base.shape}")

        rows = len(X_raw)
        periods = [numeric_value(value) for value in X_raw[config.period_column]]
        valid = np.fromiter((period is not None for period in periods), bool, rows)
        keys = _keyed(X_raw, config)
        mean = np.full(rows, np.nan)
        present = np.zeros(rows, dtype=bool)
        ambiguous = np.zeros(rows, dtype=bool)
        positions = np.flatnonzero(valid)
        if len(positions):
            features = self._encoder.transform(X_raw.loc[valid, config.columns])
            mean[positions] = features["history_mean"].to_numpy(float)
            present[positions] = features["history_present"].to_numpy() == 1
            ambiguous[positions] = features["history_ambiguous"].to_numpy() == 1
        allowed = present & (~ambiguous if config.ambiguity_policy == "past_unique_only" else True)
        applied = allowed & (self.blend_weight > 0)

        prediction = base.copy()
        if applied.any():
            if not np.isfinite(base[applied]).all():
                raise ValueError("base_prediction must be finite on the rows where history is applied")
            weight = self.blend_weight
            prediction[applied] = (1 - weight) * base[applied] + weight * mean[applied]
            if not np.isfinite(prediction[applied]).all():
                raise ValueError("The blended prediction is not finite")

        query = Counter((key, period) for key, period in zip(keys, periods) if key is not None and period is not None)
        repeated_in_query = sum(count for count in query.values() if count > 1)
        diagnostics = self._base_diagnostics(rows)
        diagnostics.update({
            "rows_with_valid_period": int(valid.sum()),
            "rows_with_full_key": int(sum(key is not None for key in keys)),
            "rows_with_past_history": int(present.sum()),
            "rows_with_ambiguous_past_history": int((present & ambiguous).sum()),
            "rows_allowed_by_policy": int(allowed.sum()),
            "rows_applied": int(applied.sum()),
            "rows_changed": int((prediction[applied] != base[applied]).sum()),
            "rows_repeated_key_in_query": int(repeated_in_query),  # diagnostic only: never used to decide a row
        })
        for name in ("rows_with_full_key", "rows_with_past_history", "rows_with_ambiguous_past_history", "rows_applied"):
            diagnostics["share" + name[4:]] = diagnostics[name] / rows if rows else None
        return HistoryApplication(prediction, applied, diagnostics)


def fit_history_candidate(
    *,
    task_type: str,
    metric_name: str,
    feature_columns: tuple[str, ...],
    X_raw: pd.DataFrame,
    y,
    config: dict | None,
    blend_weight: float,
) -> FittedHistoryCandidate:
    """Remember the answers of X_raw per key and period, or return an inactive candidate with the reason.

    Errors (ValueError/TypeError) are raised for programming mistakes: a malformed explicit config, duplicate column
    names, a weight outside [0, 1], an X_raw that is not a DataFrame, misaligned or non-finite regression answers.
    """
    weight = _check_weight(blend_weight)
    features = _column_names(feature_columns, "feature_columns", allow_empty=False)
    X_raw = _check_frame(X_raw)
    parsed = HistoryConfig.from_json(config) if config is not None else None

    def inactive(reason: str, **details) -> FittedHistoryCandidate:
        return FittedHistoryCandidate(task_type=task_type, metric_name=metric_name, blend_weight=weight,
                                      config=parsed, reason=reason, fit_diagnostics=details)

    if parsed is None:
        return inactive("no_config")
    if task_type == "classification":
        return inactive("classification_task")
    if task_type != "regression":
        return inactive("unknown_task_type")
    try:
        metric = get_metric(metric_name)
    except ValueError:
        return inactive("unsupported_metric")
    if metric.task_type != "regression":
        return inactive("metric_not_for_regression")
    not_features = [name for name in parsed.columns if name not in features]
    if not_features:
        return inactive("history_columns_not_in_features", columns=not_features)
    missing = [name for name in parsed.columns if name not in X_raw]
    if missing:
        return inactive("missing_columns", columns=missing)

    answers = _answers(X_raw, y)
    valid = _valid_periods(X_raw, parsed)
    keyed = np.fromiter((key is not None for key in _keyed(X_raw, parsed)), bool, len(X_raw))
    usable = valid & keyed
    details = {"rows": len(X_raw), "rows_with_valid_period": int(valid.sum()), "rows_with_full_key": int(keyed.sum()),
               "rows_remembered": int(usable.sum())}
    if not usable.any():
        return FittedHistoryCandidate(task_type=task_type, metric_name=metric.name, blend_weight=weight,
                                      config=parsed, reason="no_usable_fit_rows", fit_diagnostics=details)
    encoder = parsed.encoder().fit(X_raw.loc[usable, parsed.columns], answers[usable])
    periods = sorted({numeric_value(value) for value in X_raw.loc[usable, parsed.period_column]})
    details["periods"] = [period_json(value) for value in periods]
    return FittedHistoryCandidate(task_type=task_type, metric_name=metric.name, blend_weight=weight, config=parsed,
                                  reason=None, encoder=encoder, fit_diagnostics=details)
