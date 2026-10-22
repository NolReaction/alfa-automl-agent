"""Bounded target-free features fitted independently for every regression fit.

This adapter preserves the raw feature columns. It never inspects prediction
statistics, labels, identifiers, files or a host context. In particular, a new
instance must be fitted on the earlier internal rows before early stopping and
another fresh instance on all allowed rows before final refit.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from agent.features import _category


FAMILY = "round4_regression_features_v1"
_DEFAULTS = {"missing_indicators": False, "category_frequencies": False,
             "max_frequency_columns": 16, "max_categories_per_column": 512}


def normalize_feature_recipe(value=None):
    """Return explicit, detached quality options and bounded frequency caps."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(_DEFAULTS):
        raise ValueError("Invalid round4 regression feature recipe")
    result = {}
    for name in ("missing_indicators", "category_frequencies"):
        flag = value.get(name, _DEFAULTS[name])
        if type(flag) is not bool:
            raise ValueError(name + " must be a boolean")
        result[name] = flag
    for name in ("max_frequency_columns", "max_categories_per_column"):
        cap = value.get(name, _DEFAULTS[name])
        if type(cap) is not int or not 1 <= cap <= _DEFAULTS[name]:
            raise ValueError(name + " exceeds its supported integer bounds")
        result[name] = cap
    return result


def _names(value):
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(name, str) or not name or "\x00" in name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Expected unique declared feature names")
    return list(value)


def _schema(numeric, categorical, recipe):
    numeric, categorical = _names(numeric), _names(categorical)
    if not numeric and not categorical or set(numeric) & set(categorical):
        raise ValueError("Feature roles must be nonempty and disjoint")
    options = normalize_feature_recipe(recipe)
    additions = []
    missing = [(name, "__round4_missing__" + name) for name in numeric] if options["missing_indicators"] else []
    frequencies = [(name, "__round4_frequency__" + name)
                   for name in categorical[:options["max_frequency_columns"]]] if options["category_frequencies"] else []
    occupied = set(numeric) | set(categorical)
    for _, name in missing + frequencies:
        if name in occupied:
            raise ValueError("Derived regression feature name collides: " + name)
        occupied.add(name)
        additions.append(name)
    return options, numeric, categorical, missing, frequencies, additions


class RegressionFeatureAdapter(TransformerMixin, BaseEstimator):
    """Copy declared features and add missing flags or fit-only frequencies.

    Category frequencies are bounded relative counts. Unknown categories map
    to zero; a column exceeding the fit cardinality cap is disabled as a whole
    and its derived feature stays zero. The declared output schema therefore
    remains the same for internal fitting and fresh full refit.
    """

    def __init__(self, numeric, categorical, recipe=None):
        # Constructor values are retained exactly for sklearn.clone.
        self.numeric = numeric
        self.categorical = categorical
        self.recipe = recipe

    def output_schema(self):
        _, numeric, categorical, _, _, additions = _schema(self.numeric, self.categorical, self.recipe)
        return numeric + additions, categorical

    @staticmethod
    def _frame(X, numeric, categorical):
        if (not isinstance(X, pd.DataFrame) or not X.columns.is_unique
                or set(X.columns) != set(numeric) | set(categorical)):
            raise ValueError("Expected exactly the declared feature DataFrame")
        return X

    def fit(self, X, y=None):
        # A failed refit must not keep any previously learned category map.
        for name in ("config_", "frequency_maps_", "feature_names_in_", "evidence_"):
            self.__dict__.pop(name, None)
        options, numeric, categorical, missing, frequencies, additions = _schema(
            self.numeric, self.categorical, self.recipe)
        frame = self._frame(X, numeric, categorical)
        if frame.empty:
            raise ValueError("Cannot fit regression features on no rows")
        maps, reports = {}, []
        cap = options["max_categories_per_column"]
        for source, destination in frequencies:
            counts = {}
            for value in frame[source]:
                key = _category(value)
                counts[key] = counts.get(key, 0) + 1
                if len(counts) > cap:
                    break
            disabled = len(counts) > cap
            maps[source] = {} if disabled else {key: count / len(frame) for key, count in counts.items()}
            reports.append({"column": source, "feature": destination,
                "status": "disabled" if disabled else "enabled",
                "reason": "fit_cardinality_exceeds_cap" if disabled else "fit_relative_frequency",
                "fit_categories": None if disabled else len(counts),
                "fit_categories_lower_bound": cap + 1 if disabled else len(counts)})
        self.numeric_, self.categorical_ = numeric, categorical
        self.output_numeric_, self.output_categorical_ = numeric + additions, list(categorical)
        self.missing_columns_, self.frequency_columns_ = missing, frequencies
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.frequency_maps_ = maps
        self.config_ = options
        self.evidence_ = {"version": 1, "family": FAMILY, "config": deepcopy(options),
            "fit_rows": len(frame), "added_numeric_features": additions,
            "missing_indicator_columns": [name for name, _ in missing],
            "frequency_columns": reports,
            "frequency_columns_omitted_by_cap": categorical[options["max_frequency_columns"]:]
                if options["category_frequencies"] else [],
            "label_use": "none", "selection_scope": "this_model_fit_only"}
        return self

    def transform(self, X):
        check_is_fitted(self, ("config_", "frequency_maps_", "feature_names_in_"))
        frame = self._frame(X, self.numeric_, self.categorical_)
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction feature order differs from fitted schema")
        output = frame.copy(deep=True)
        for source, destination in self.missing_columns_:
            values = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            output[destination] = (~np.isfinite(values)).astype(float)
        for source, destination in self.frequency_columns_:
            mapping = self.frequency_maps_[source]
            output[destination] = frame[source].map(_category).map(mapping).fillna(0.0).astype(float)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Input names differ from fitted feature schema")
        return np.asarray([*self.feature_names_in_, *self.evidence_["added_numeric_features"]], dtype=object)
