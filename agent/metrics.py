"""Trusted metric definitions and strict scoring of labels or probabilities.

Training code may propose a model; it cannot supply the score used to select it.
All class names remain the literal strings from the training CSV.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
import re

@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    task_type: str
    prediction_kind: str

    @property
    def classification(self) -> bool:
        return self.task_type == "classification"

    def to_dict(self) -> dict:
        return asdict(self)


_REGISTRY = {
    "RMSE": MetricSpec("RMSE", False, "regression", "regression"),
    "MAE": MetricSpec("MAE", False, "regression", "regression"),
    "MSE": MetricSpec("MSE", False, "regression", "regression"),
    "R2": MetricSpec("R2", True, "regression", "regression"),
    "Macro F1": MetricSpec("Macro F1", True, "classification", "labels"),
    "Weighted F1": MetricSpec("Weighted F1", True, "classification", "labels"),
    "Accuracy": MetricSpec("Accuracy", True, "classification", "labels"),
    "ROC AUC": MetricSpec("ROC AUC", True, "classification", "probabilities"),
    "LogLoss": MetricSpec("LogLoss", False, "classification", "probabilities"),
}


def _key(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold().replace("²", "2")


_NAMES = {_key(name): name for name in _REGISTRY}
_NAMES.update({"f1macro": "Macro F1", "f1weighted": "Weighted F1",
               "logarithmicloss": "LogLoss", "crossentropy": "LogLoss"})


def get_metric(value: str | MetricSpec | dict) -> MetricSpec:
    """Resolve only allowlisted metrics, never trust flags supplied by a worker."""
    if isinstance(value, MetricSpec):
        name = value.name
    elif isinstance(value, dict):
        name = value.get("name")
    else:
        name = value
    if not isinstance(name, str) or _key(name) not in _NAMES:
        raise ValueError(f"Unsupported metric: {name!r}")
    spec = _REGISTRY[_NAMES[_key(name)]]
    if isinstance(value, MetricSpec) and value != spec:
        raise ValueError("Metric definition does not match the trusted registry")
    return spec


def registered_metrics() -> tuple[MetricSpec, ...]:
    return tuple(_REGISTRY.values())


def _vector(values, *, name: str) -> np.ndarray:
    import numpy as np

    result = np.asarray(values)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return result


def label_values(values, *, name: str = "Class labels") -> np.ndarray:
    """Preserve literal strings while rejecting real missing numeric values."""
    import numpy as np

    # Avoid NumPy converting a missing numeric value to the literal class "nan"
    # when a mixed input also contains text labels.
    result = _vector(np.asarray(values, dtype=object), name=name)
    if any(value is None or not str(value).strip()
           or (isinstance(value, Real) and not math.isfinite(value)) for value in result):
        raise ValueError(f"{name} contains missing or empty labels")
    return result.astype(str)


def infer_positive_class(classes) -> str | None:
    """Recognize only literal representations of the conventional numeric 0/1."""
    classes = tuple(map(str, classes))
    if len(classes) != 2:
        return None
    try:
        values = [float(label) for label in classes]
    except ValueError:
        return None
    return classes[values.index(1.0)] if set(values) == {0.0, 1.0} else None


def probability_matrix(predictions, *, classes, prediction_classes=None,
                       positive_class=None, rows=None) -> np.ndarray:
    """Return finite normalized probabilities in the exact requested class order.

    A one-column binary prediction denotes the explicitly identified positive
    class. For the standard labels 0/1, class 1 is the documented default.
    """
    import numpy as np

    classes = tuple(str(value) for value in classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Probability predictions require at least two unique classes")
    try:
        values = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Probabilities must be numeric") from error
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim == 1:
        if len(classes) != 2:
            raise ValueError("A single probability column is supported only for binary classification")
        positive_class = str(positive_class) if positive_class is not None else infer_positive_class(classes)
        if positive_class not in classes:
            raise ValueError("Binary probabilities require an unambiguous positive_class")
        if prediction_classes is not None and tuple(map(str, prediction_classes)) not in {
                (positive_class,), classes}:
            raise ValueError("Probability column class does not match positive_class")
        matrix = np.empty((len(values), 2), dtype=float)
        position = classes.index(positive_class)
        matrix[:, position], matrix[:, 1 - position] = values, 1.0 - values
    elif values.ndim == 2:
        supplied = tuple(map(str, prediction_classes)) if prediction_classes is not None else classes
        if len(supplied) != values.shape[1] or len(supplied) != len(classes) or set(supplied) != set(classes):
            raise ValueError("Probability columns do not match the complete class order")
        matrix = values[:, [supplied.index(label) for label in classes]]
    else:
        raise ValueError("Probabilities must be a vector or a two-dimensional matrix")
    if not len(matrix) or (rows is not None and len(matrix) != rows):
        raise ValueError("Probability row count does not match the expected rows")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    return matrix


def score_predictions(metric, actual, predictions, *, classes=(),
                      prediction_classes=None, positive_class=None) -> float:
    """Independently calculate one registered metric from fresh predictions."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
        r2_score, roc_auc_score,
    )

    metric = get_metric(metric)
    if metric.classification:
        actual = label_values(actual, name="Actual labels")
        classes = tuple(map(str, classes)) if len(classes) else tuple(sorted(set(actual)))
        if len(set(classes)) != len(classes) or not set(actual) <= set(classes):
            raise ValueError("Actual labels are outside the declared classes")
        if metric.prediction_kind == "labels":
            predicted = label_values(predictions, name="Predicted labels")
            if len(predicted) != len(actual) or not set(predicted) <= set(classes):
                raise ValueError("Predicted labels or row count do not match the task")
            if metric.name == "Accuracy":
                score = accuracy_score(actual, predicted)
            else:
                score = f1_score(actual, predicted,
                                 average="macro" if metric.name == "Macro F1" else "weighted",
                                 zero_division=0)
        else:
            matrix = probability_matrix(predictions, classes=classes,
                                        prediction_classes=prediction_classes,
                                        positive_class=positive_class, rows=len(actual))
            if metric.name == "LogLoss":
                ordered = tuple(sorted(classes))
                score = log_loss(actual, matrix[:, [classes.index(label) for label in ordered]],
                                 labels=list(ordered))
            elif len(classes) == 2:
                positive_class = str(positive_class) if positive_class is not None else infer_positive_class(classes)
                if positive_class not in classes:
                    raise ValueError("ROC AUC requires an unambiguous positive_class")
                if len(set(actual)) != 2:
                    raise ValueError("ROC AUC requires both classes in the validation rows")
                score = roc_auc_score(actual == positive_class, matrix[:, classes.index(positive_class)])
            else:
                ordered = tuple(sorted(classes))
                if set(actual) != set(classes):
                    raise ValueError("Multiclass ROC AUC requires every class in the validation rows")
                score = roc_auc_score(actual, matrix[:, [classes.index(label) for label in ordered]],
                                      labels=list(ordered), multi_class="ovr", average="macro")
    else:
        try:
            actual = _vector(actual, name="Actual values").astype(float)
            predicted = _vector(predictions, name="Predicted values").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Regression values must be numeric one-dimensional vectors") from error
        if len(actual) != len(predicted) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Regression values must be finite and have matching row counts")
        if metric.name == "MAE":
            score = mean_absolute_error(actual, predicted)
        elif metric.name == "R2":
            if len(actual) < 2:
                raise ValueError("R2 requires at least two validation rows")
            score = r2_score(actual, predicted)
        else:
            score = mean_squared_error(actual, predicted)
            if metric.name == "RMSE":
                score = math.sqrt(score)
    if not math.isfinite(float(score)):
        raise ValueError("Metric calculation returned a non-finite value")
    return float(score)
