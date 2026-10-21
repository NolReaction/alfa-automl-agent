"""Allowlisted tabular model adapters with fold-local feature preparation.

The old two recipes retain their exact preprocessing when no feature option is
selected. New adapters keep categorical features native where practical and
never learn a vocabulary or statistic from the validation/test batch.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from agent.features import FeatureTransformer, feature_schema, normalize_feature_config
from agent.model_config import make_model_config, validate_model_config
from agent.target_transform import Log1pTargetRegressor
from offline_baseline import SEED, make_pipeline, prepare_features


THREADS = 8
MAX_NATIVE_CATEGORIES = 4096
MAX_NATIVE_CATEGORY_CELLS = 5_000_000


class _LegacyInputCleaner(TransformerMixin, BaseEstimator):
    """Move the original target-free coercion inside the cloned pipeline."""

    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(self.numeric) + len(self.categorical)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Model inputs must be a DataFrame with unique feature columns")
        features = list(self.numeric) + list(self.categorical)
        if not set(features) <= set(X.columns):
            raise ValueError("Declared input feature columns are missing")
        return prepare_features(X, features, self.numeric, self.categorical)


def normalize_history_config(value=None):
    """Allow only explicit feature columns; target history is never inferred."""
    if value is None:
        return None
    required = {"entity_columns", "time_column"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"prefix", "time_kind"}):
        raise ValueError("History needs explicit entity_columns and time_column")
    entities, column = value["entity_columns"], value["time_column"]
    if (not isinstance(entities, list) or not 1 <= len(entities) <= 100
            or any(not isinstance(name, str) or not name.strip() or "\x00" in name
                   or len(name) > 500 for name in entities)
            or len(set(entities)) != len(entities)
            or not isinstance(column, str) or not column.strip() or "\x00" in column
            or len(column) > 500 or column in entities):
        raise ValueError("History key and time must be distinct declared feature columns")
    kind, prefix = value.get("time_kind", "numeric"), value.get("prefix", "history")
    if not isinstance(kind, str) or kind not in {"numeric", "datetime"}:
        raise ValueError("History time_kind must be numeric or datetime")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,60}", prefix):
        raise ValueError("History prefix must be a short ASCII identifier")
    return {"entity_columns": list(entities), "time_column": column,
            "time_kind": kind, "prefix": prefix}


class _LightGBMFrame(TransformerMixin, BaseEstimator):
    """Stable pandas categories; unseen categories become missing, never refit."""

    def __init__(self, categorical):
        self.categorical = categorical

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categories_ = {
            name: sorted(X[name].unique().tolist()) for name in self.categorical
        }
        # LightGBM rejects JSON punctuation in feature names. These internal
        # names are deterministic; the original schema remains in features.
        self.names_ = {name: f"feature_{index}" for index, name in enumerate(self.columns_)}
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError("Feature columns differ from the fitted LightGBM schema")
        result = X.copy()
        for name, values in self.categories_.items():
            result[name] = pd.Categorical(result[name], categories=values, ordered=False)
        return result.rename(columns=self.names_)


class _CatBoostAdapter(BaseEstimator):
    """Select native categories or a bounded ordinal fallback from fit only."""

    def __init__(self, parameters, categorical):
        self.parameters = parameters
        self.categorical = categorical

    def _frame(self, X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError("Feature columns differ from the fitted CatBoost schema")
        result = X.copy()
        if self.category_encoding_ == "ordinal":
            result.loc[:, self.categorical] = self.category_encoder_.transform(result.loc[:, self.categorical])
            for name in self.categorical:
                result[name] = result[name].astype(float)
        return result

    def fit(self, X, y, **fit_params):
        from catboost import CatBoostClassifier, CatBoostRegressor

        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("CatBoost needs a nonempty feature DataFrame")
        self.columns_ = list(X.columns)
        cardinality = sum(X[name].nunique(dropna=False) for name in self.categorical)
        native = (cardinality <= MAX_NATIVE_CATEGORIES
                  and len(X) * len(self.categorical) <= MAX_NATIVE_CATEGORY_CELLS)
        self.category_encoding_ = "native" if native else "ordinal"
        if not native:
            print("CatBoost: using fold-fitted ordinal categories because the native category budget was exceeded.", flush=True)
            self.category_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=float,
            ).fit(X.loc[:, self.categorical])
        constructor = CatBoostClassifier if self._estimator_type == "classifier" else CatBoostRegressor
        self.estimator_ = constructor(
            **self.parameters, cat_features=list(self.categorical) if native else [],
            random_seed=SEED, thread_count=THREADS, task_type="CPU",
            allow_writing_files=False, verbose=False,
        )
        self.estimator_.fit(self._frame(X), y, **fit_params)
        self.n_features_in_ = X.shape[1]
        if self._estimator_type == "classifier":
            self.classes_ = np.asarray(self.estimator_.classes_).astype(str)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(self._frame(X))).reshape(-1)


class CatBoostClassificationAdapter(ClassifierMixin, _CatBoostAdapter):
    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict_proba(self._frame(X)), dtype=float)


class CatBoostRegressionAdapter(RegressorMixin, _CatBoostAdapter):
    pass


def _encoded_preparation(numeric, categorical, *, linear=False):
    transformers = []
    if numeric:
        numeric_steps = [("missing", SimpleImputer(strategy="median", keep_empty_features=True))]
        if linear:
            numeric_steps.append(("scale", StandardScaler(with_mean=False)))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        encoder = (OneHotEncoder(handle_unknown="ignore", max_categories=256, sparse_output=True)
                   if linear else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        transformers.append(("category", Pipeline([
            ("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", encoder),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", n_jobs=1,
                             sparse_threshold=1.0 if linear else 0.0)


def build_pipeline(action, task_type, numeric, categorical, model_config=None, feature_config=None,
                   history_config=None):
    """Construct a fresh sklearn-compatible pipeline; fit learns on its fold.

    Configuration contains quality controls only. Seed, CPU limits, I/O flags
    and solver choices are fixed by this module and cannot come from JSON.
    """
    config = (make_model_config(action, task_type) if model_config is None else
              validate_model_config(model_config, action=action))
    expected = make_model_config(action, task_type)["model"]
    if config["model"] != expected:
        raise ValueError("Model configuration does not match task type")
    history_options = normalize_history_config(history_config)
    history = None
    if history_options is not None:
        if task_type != "regression":
            raise ValueError("Past target history is supported only for numeric regression")
        declared = set(numeric) | set(categorical)
        if not (set(history_options["entity_columns"]) | {history_options["time_column"]}) <= declared:
            raise ValueError("History columns must be input features, never an ID or target")
        additions = [history_options["prefix"] + "_" + name
                     for name in ("count", "mean", "last", "has_history")]
        if set(additions) & declared:
            raise ValueError("History feature names collide with declared input columns")
        from agent.history import PastOnlyHistory

        history = PastOnlyHistory(**history_options)
        numeric = list(numeric) + additions
    options = normalize_feature_config(feature_config)
    output_numeric, output_categorical = feature_schema(numeric, categorical, options)
    classification = task_type == "classification"
    parameters = dict(config["parameters"])
    if config.get("objective") == "mae":
        name, value = {"baseline": ("loss", "absolute_error"),
                       "catboost": ("loss_function", "MAE"),
                       "lightgbm": ("objective", "regression_l1")}[action]
        parameters[name] = value

    def finish(pipeline):
        if config.get("target_transform") == "log1p":
            pipeline.set_params(model=Log1pTargetRegressor(pipeline.named_steps["model"]))
        # Pipeline calls fit_transform(X, y) on training rows and transform(X)
        # for validation/test; the history component therefore receives no
        # validation answers. It must precede numeric coercion of entity keys.
        return Pipeline([("history", history), *pipeline.steps]) if history is not None else pipeline

    if action in {"baseline", "extra_trees"}:
        pipeline = make_pipeline(numeric, categorical, classification)
        if action == "extra_trees":
            constructor = ExtraTreesClassifier if classification else ExtraTreesRegressor
            pipeline.set_params(model=constructor(random_state=SEED, n_jobs=THREADS))
        pipeline.named_steps["model"].set_params(**parameters)
        if history is None and options == normalize_feature_config():
            return finish(Pipeline([("features", _LegacyInputCleaner(numeric, categorical)), *pipeline.steps]))
        return finish(Pipeline([
            ("features", FeatureTransformer(numeric, categorical, options)),
            ("prepare", _encoded_preparation(output_numeric, output_categorical)),
            ("model", pipeline.named_steps["model"]),
        ]))

    features = FeatureTransformer(numeric, categorical, options)
    if action == "catboost":
        constructor = CatBoostClassificationAdapter if classification else CatBoostRegressionAdapter
        return finish(Pipeline([("features", features), ("model", constructor(parameters, output_categorical))]))
    if action == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        constructor = LGBMClassifier if classification else LGBMRegressor
        model = constructor(**parameters, random_state=SEED, n_jobs=THREADS,
                            verbosity=-1, deterministic=True, force_col_wise=True)
        return finish(Pipeline([
            ("features", features), ("prepare", _LightGBMFrame(output_categorical)), ("model", model),
        ]))
    if action == "linear":
        # lbfgs uses BLAS threads inside the worker's eight-thread limit. A
        # second joblib process pool adds nested parallelism without helping
        # this solver's single multinomial fit.
        model = (LogisticRegression(**parameters, solver="lbfgs", n_jobs=1, random_state=SEED)
                 if classification else Ridge(**parameters, solver="lsqr", tol=0.0001, max_iter=2000))
        return finish(Pipeline([
            ("features", features), ("prepare", _encoded_preparation(output_numeric, output_categorical, linear=True)),
            ("model", model),
        ]))
    raise ValueError("Unknown model recipe")


def predict_for_metric(pipeline, X, spec):
    """Return labels/values or full probabilities in ``pipeline.classes_`` order.

    Probability consumers must preserve that class order when passing results
    to the shared scorer/output writer. No binary positive label is guessed.
    """
    from agent.metrics import get_metric

    metric = get_metric(getattr(spec, "metric", spec))
    if metric.prediction_kind == "probabilities":
        if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
            raise ValueError("This model does not provide class probabilities")
        probabilities = np.asarray(pipeline.predict_proba(X), dtype=float)
        classes = tuple(map(str, pipeline.classes_))
        if (probabilities.ndim != 2 or probabilities.shape != (len(X), len(classes))
                or len(classes) < 2 or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
            raise ValueError("Model returned invalid probability rows or class columns")
        return probabilities
    values = np.asarray(pipeline.predict(X))
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != len(X):
        raise ValueError("Model prediction shape does not match the input rows")
    if metric.prediction_kind == "labels":
        if any(value is None or not str(value).strip() for value in values):
            raise ValueError("Model returned empty class labels")
        return values.astype(str)
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Model returned non-finite regression predictions")
    return values
