"""Fold-local, target-free feature transformations for tabular model adapters.

The schema is explicit: date and text extraction never guess column roles.
Only rare-category vocabularies are learned, and only by ``fit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


MISSING_CATEGORY = "__ALFA_MISSING__"
RARE_CATEGORY = "__ALFA_RARE__"
_ESCAPE_PREFIX = "__ALFA_VALUE__"
_RESERVED_PREFIX = "__ALFA_"
_DATE_PARTS = ("year", "month", "day", "dayofweek", "hour")
_DEFAULTS = {
    "missing_indicators": False, "rare_min_count": 0,
    "date_columns": [], "text_columns": [], "signed_log": False,
}


def normalize_feature_config(value=None) -> dict:
    """Return a detached, JSON-compatible allowlist of feature options."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(_DEFAULTS):
        raise ValueError("Неизвестные настройки обработки признаков.")
    result = {}
    for name in ("missing_indicators", "signed_log"):
        flag = value.get(name, _DEFAULTS[name])
        if type(flag) is not bool:
            raise ValueError(f"Параметр {name} должен быть логическим значением.")
        result[name] = flag
    minimum = value.get("rare_min_count", 0)
    if type(minimum) is not int or not 0 <= minimum <= 1000:
        raise ValueError("rare_min_count должен быть целым числом от 0 до 1000.")
    result["rare_min_count"] = minimum
    for name in ("date_columns", "text_columns"):
        columns = value.get(name, [])
        if (not isinstance(columns, list) or len(columns) > 100
                or any(not isinstance(column, str) or not column.strip()
                       or len(column) > 500 or "\x00" in column for column in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f"{name} должен содержать не более 100 разных имён признаков.")
        result[name] = list(columns)
    if set(result["date_columns"]) & set(result["text_columns"]):
        raise ValueError("Один признак нельзя одновременно обрабатывать как дату и текст.")
    return result


def _columns(value, description):
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(name, str) or not name.strip() or "\x00" in name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Некорректный список {description} признаков.")
    return list(value)


def _schema(numeric, categorical, config):
    numeric = _columns(numeric, "числовых")
    categorical = _columns(categorical, "категориальных")
    if not numeric and not categorical or set(numeric) & set(categorical):
        raise ValueError("Нужны непересекающиеся числовые и категориальные признаки.")
    known = set(numeric) | set(categorical)
    expanded = set(config["date_columns"]) | set(config["text_columns"])
    if not expanded <= known:
        raise ValueError("Дата или текст должны быть явно объявленными входными признаками.")
    output_numeric = [name for name in numeric if name not in expanded]
    output_categorical = [name for name in categorical if name not in expanded]
    occupied = set(known)

    def add(name):
        if name in occupied:
            raise ValueError(f"Имя созданного признака уже занято: {name}")
        occupied.add(name)
        output_numeric.append(name)

    if config["missing_indicators"]:
        for name in numeric:
            if name not in expanded:
                add(name + "__missing")
    for name in config["date_columns"]:
        for part in _DATE_PARTS:
            add(name + "__" + part)
    for name in config["text_columns"]:
        for part in ("length", "word_count", "missing"):
            add(name + "__" + part)
    return numeric, categorical, output_numeric, output_categorical


def feature_schema(numeric, categorical, config=None):
    """Return deterministic numeric/categorical output names, without fitting."""
    _, _, output_numeric, output_categorical = _schema(
        numeric, categorical, normalize_feature_config(config))
    return output_numeric, output_categorical


def _missing(value):
    # The CSV contract marks an empty field as missing. A nonempty literal
    # category/text value, including whitespace or "NA", must remain distinct.
    if value is None or isinstance(value, str) and value == "":
        return True
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _category(value):
    if _missing(value):
        return MISSING_CATEGORY
    text = str(value)
    # Keep genuine categories distinct from the two generated sentinels.
    return _ESCAPE_PREFIX + text if text.startswith(_RESERVED_PREFIX) else text


class FeatureTransformer(TransformerMixin, BaseEstimator):
    """Prepare declared columns while learning category counts on fit rows only.

    ``y`` is deliberately ignored. The same transformer can be cloned/refitted
    for every validation fold and then fitted on full train for test inference.
    """

    def __init__(self, numeric, categorical, config=None):
        # Preserve constructor arguments exactly for sklearn.clone.
        self.numeric = numeric
        self.categorical = categorical
        self.config = config

    @staticmethod
    def _frame(X, numeric, categorical):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Для обработки нужна таблица с уникальными именами столбцов.")
        missing = set(numeric) | set(categorical)
        missing -= set(X.columns)
        if missing:
            raise ValueError("Не найдены объявленные признаки: " + ", ".join(sorted(missing)))
        return X

    def fit(self, X, y=None):
        config = normalize_feature_config(self.config)
        numeric, categorical, output_numeric, output_categorical = _schema(
            self.numeric, self.categorical, config)
        frame = self._frame(X, numeric, categorical)
        if frame.empty:
            raise ValueError("Нельзя обучить обработку признаков на пустой таблице.")
        self.config_ = config
        self.numeric_ = numeric
        self.categorical_ = categorical
        self.output_numeric_ = output_numeric
        self.output_categorical_ = output_categorical
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.allowed_categories_ = {}
        if config["rare_min_count"]:
            for name in output_categorical:
                counts = frame[name].map(_category).value_counts()
                self.allowed_categories_[name] = frozenset(
                    counts[counts >= config["rare_min_count"]].index) - {MISSING_CATEGORY}
        return self

    def transform(self, X):
        check_is_fitted(self, ("config_", "numeric_", "allowed_categories_"))
        frame = self._frame(X, self.numeric_, self.categorical_)
        # Build once: repeated DataFrame insertion fragments wide tables and
        # produced thousands of PerformanceWarnings in real search runs.
        output = {}
        config = self.config_
        expanded = set(config["date_columns"]) | set(config["text_columns"])
        for name in self.numeric_:
            if name in expanded:
                continue
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            values = np.where(np.isfinite(values), values, np.nan)
            if config["missing_indicators"]:
                output[name + "__missing"] = np.isnan(values).astype(float)
            output[name] = np.sign(values) * np.log1p(np.abs(values)) if config["signed_log"] else values
        for name in self.output_categorical_:
            values = frame[name].map(_category)
            if config["rare_min_count"]:
                allowed = self.allowed_categories_[name]
                values = values.where(values.eq(MISSING_CATEGORY) | values.isin(allowed), RARE_CATEGORY)
            output[name] = values
        for name in config["date_columns"]:
            values = frame[name].map(lambda value: None if _missing(value) else str(value))
            dates = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
            for part in _DATE_PARTS:
                output[name + "__" + part] = getattr(dates.dt, part).astype(float)
        for name in config["text_columns"]:
            missing = frame[name].map(_missing).astype(bool)
            values = frame[name].where(~missing, "").map(str).astype(object)
            output[name + "__length"] = values.str.len().astype(float)
            output[name + "__word_count"] = values.str.split().str.len().astype(float)
            output[name + "__missing"] = missing.astype(float)
        return pd.DataFrame(output, index=frame.index).loc[
            :, self.output_numeric_ + self.output_categorical_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_numeric_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Имена входных признаков не соответствуют обученной обработке.")
        return np.asarray(self.output_numeric_ + self.output_categorical_, dtype=object)
