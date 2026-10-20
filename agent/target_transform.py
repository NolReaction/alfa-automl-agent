"""Explicit fold-local log-target regression with a checked inverse transform."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_classifier
from sklearn.utils.validation import check_consistent_length, check_is_fitted


class Log1pTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit a fresh regressor on log1p(y); predict on the original target scale.

    Only the y passed to fit is inspected. Target values must be a nonempty
    finite nonnegative vector. The inverse is exact expm1, without clipping or
    fitting a correction from validation targets. Unrepresentable predictions
    fail explicitly instead of producing an invalid submission.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        # A failed refit must not leave a previously fitted regressor available.
        for name in ("regressor_", "n_features_in_", "feature_names_in_"):
            self.__dict__.pop(name, None)
        if is_classifier(self.estimator):
            raise ValueError("log1p target transformation supports regression only.")
        if {"eval_set", "validation_data"} & set(fit_params):
            raise ValueError("log1p target transformation accepts fit-fold targets only, not external validation targets.")
        try:
            raw = np.asarray(y)
        except (TypeError, ValueError) as error:
            raise ValueError("log1p targets must be a nonempty vector of finite nonnegative real numbers.") from error
        if raw.ndim != 1 or not len(raw) or np.iscomplexobj(raw):
            raise ValueError("log1p targets must be a nonempty vector of finite nonnegative real numbers.")
        try:
            values = np.array(raw, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("log1p targets must contain finite nonnegative real numbers.") from error
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("log1p targets must contain finite nonnegative real numbers.")
        check_consistent_length(X, values)
        regressor = clone(self.estimator)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            transformed = np.log1p(values)
        regressor.fit(X, transformed, **fit_params)
        self.regressor_ = regressor
        for name in ("n_features_in_", "feature_names_in_"):
            if hasattr(regressor, name):
                setattr(self, name, getattr(regressor, name))
        return self

    def predict(self, X):
        check_is_fitted(self, "regressor_")
        try:
            raw = np.asarray(self.regressor_.predict(X))
            if np.iscomplexobj(raw):
                raise ValueError("Complex predictions are unsupported.")
            values = raw.astype(float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("log1p regressor must return a finite real prediction vector.") from error
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("log1p regressor must return a finite real prediction vector.")
        try:
            check_consistent_length(X, values)
        except (TypeError, ValueError) as error:
            raise ValueError("log1p regressor must return one prediction per input row.") from error
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                predictions = np.expm1(values)
        except FloatingPointError as error:
            raise ValueError("log1p inverse overflow: predictions must be finite on the original target scale.") from error
        if not np.isfinite(predictions).all():
            raise ValueError("log1p inverse predictions must be finite on the original target scale.")
        return predictions
