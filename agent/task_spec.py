"""Explicit, immutable contract for the supported tabular competition tasks.

All four provided CSV files remain required. Ambiguous metrics or class/probability
semantics are rejected instead of being silently treated as a different task.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from agent.metrics import MetricSpec, get_metric, infer_positive_class, label_values, probability_matrix, score_predictions
from agent.task_contract import _DECLARATION, _description_lines, parse_task_metric
from offline_baseline import find_data, load_data, read_csv


def _declaration(text: str, label: str) -> str | None:
    values = []
    for line in _description_lines(text):
        match = re.match(rf"(?:{label})\s*[:=—–]\s*(.+)$", line, re.IGNORECASE)
        if match:
            values.append(match[1].strip())
    if len(set(values)) > 1:
        raise ValueError("Conflicting task declarations")
    return values[0] if values else None


def _class_list(text: str) -> tuple[str, ...] | None:
    value = _declaration(text, r"class[\s_-]*order|порядок\s+классов")
    if value is None:
        return None
    try:
        parsed = json.loads(value) if value.startswith("[") else next(csv.reader(io.StringIO(value)))
    except (ValueError, csv.Error) as error:
        raise ValueError("Class order must be a JSON array or a comma-separated list") from error
    if not isinstance(parsed, list) or not parsed or any(not isinstance(v, str) or not v.strip() for v in parsed):
        raise ValueError("Class order must contain non-empty literal string labels")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Class order contains duplicate labels")
    return tuple(parsed)


def _positive_class(text: str) -> str | None:
    value = _declaration(text, r"positive[\s_-]*class|положительный\s+класс")
    if value is None:
        return None
    if value.startswith('"'):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise ValueError("Positive class must be a literal class label") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Positive class must be a non-empty class label")
    return value


def _auc_declaration(text: str) -> str:
    """Read averaging from primary declarations, never from unrelated examples."""
    lines, values = _description_lines(text), []
    for index, line in enumerate(lines):
        matched = _DECLARATION.match(line)
        if matched is None:
            continue
        value = matched["value"].strip(":=—–- ")
        values.append(value or next((item for item in lines[index + 1:] if item), ""))
    return " ".join(values)


@dataclass(frozen=True)
class TaskSpec:
    task_path: Path
    data_dir: Path
    id_column: str
    target_column: str
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    classes: tuple[str, ...]
    output_columns: tuple[str, ...]
    probability_columns: tuple[tuple[str, str], ...]
    positive_class: str | None
    metric: MetricSpec

    def __post_init__(self):
        get_metric(self.metric)
        for name in ("features", "numeric", "categorical", "classes", "output_columns"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"Task specification {name} must contain immutable string values")
            if len(set(values)) != len(values):
                raise ValueError(f"Task specification {name} contains duplicate values")
        if not self.features or set(self.numeric) | set(self.categorical) != set(self.features) or set(self.numeric) & set(self.categorical):
            raise ValueError("Numeric and categorical columns must partition the input features")
        if self.id_column not in self.output_columns:
            raise ValueError("Task output schema must contain its ID column")
        if self.classification and len(self.classes) < 2:
            raise ValueError("Classification requires at least two literal classes")
        if not self.classification and self.classes:
            raise ValueError("Regression cannot declare classification labels")
        if self.prediction_kind != "probabilities":
            if len(self.output_columns) != 2 or set(self.output_columns) != {self.id_column, self.target_column}:
                raise ValueError("Label/regression output must contain its ID and target columns")
            if self.probability_columns:
                raise ValueError("Label/regression output cannot declare probability columns")
        else:
            if not isinstance(self.probability_columns, tuple) or any(
                    not isinstance(pair, tuple) or len(pair) != 2
                    or not all(isinstance(value, str) for value in pair) for pair in self.probability_columns):
                raise ValueError("Probability column mapping must be immutable string pairs")
            columns = [column for column, _ in self.probability_columns]
            labels = [label for _, label in self.probability_columns]
            if columns != list(self.prediction_columns) or not set(labels) <= set(self.classes):
                raise ValueError("Probability column mapping differs from the task output schema")
            if len(labels) == 1:
                if len(self.classes) != 2 or labels[0] != self.positive_class:
                    raise ValueError("Single-column probabilities require the declared binary positive class")
            elif len(labels) != len(self.classes) or set(labels) != set(self.classes):
                raise ValueError("Multicolumn probabilities must cover each class exactly once")

    @property
    def classification(self) -> bool:
        return self.metric.classification

    @property
    def prediction_kind(self) -> str:
        return self.metric.prediction_kind

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        return tuple(name for name in self.output_columns if name != self.id_column)

    @property
    def holdout_columns(self) -> tuple[str, ...]:
        if self.prediction_kind != "probabilities" or len(self.probability_columns) == 1:
            return ("id", "actual", "prediction")
        return ("id", "actual", *(f"probability:{label}" for label in self.classes))

    def to_dict(self) -> dict:
        return {
            "version": 1, "task_path": str(self.task_path), "data_dir": str(self.data_dir),
            "id_column": self.id_column, "target_column": self.target_column,
            "features": list(self.features), "numeric": list(self.numeric),
            "categorical": list(self.categorical), "classes": list(self.classes),
            "output_columns": list(self.output_columns),
            "probability_columns": [list(pair) for pair in self.probability_columns],
            "positive_class": self.positive_class, "metric": self.metric.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TaskSpec":
        if not isinstance(value, dict) or value.get("version", 1) != 1:
            raise ValueError("Unsupported task specification version")
        return cls(
            task_path=Path(value["task_path"]), data_dir=Path(value["data_dir"]),
            id_column=value["id_column"], target_column=value["target_column"],
            features=tuple(value["features"]), numeric=tuple(value["numeric"]),
            categorical=tuple(value["categorical"]), classes=tuple(value["classes"]),
            output_columns=tuple(value["output_columns"]),
            probability_columns=tuple(tuple(pair) for pair in value.get("probability_columns", ())),
            positive_class=value.get("positive_class"), metric=get_metric(value["metric"]),
        )

    def _probabilities(self, predictions, prediction_classes=None, rows=None):
        return probability_matrix(predictions, classes=self.classes,
                                  prediction_classes=prediction_classes,
                                  positive_class=self.positive_class, rows=rows)

    def prediction_frame(self, ids, predictions, *, prediction_classes=None) -> pd.DataFrame:
        ids = list(ids)
        output = {self.id_column: ids}
        if self.prediction_kind == "probabilities":
            matrix = self._probabilities(predictions, prediction_classes, len(ids))
            for column, label in self.probability_columns:
                output[column] = matrix[:, self.classes.index(label)]
        else:
            values = (label_values(predictions, name="Predicted labels")
                      if self.prediction_kind == "labels" else np.asarray(predictions))
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1 or len(values) != len(ids):
                raise ValueError("Prediction shape does not match the output template")
            output[self.target_column] = values
        frame = pd.DataFrame(output).loc[:, list(self.output_columns)]
        self.validate_frame(frame, ids)
        return frame

    def validate_frame(self, frame: pd.DataFrame, expected_ids) -> None:
        if list(frame.columns) != list(self.output_columns):
            raise ValueError("Submission columns or column order differ from sample_submission.csv")
        if frame[self.id_column].tolist() != list(expected_ids):
            raise ValueError("Submission IDs or their order differ from test.csv")
        if frame[self.id_column].duplicated().any() or frame[self.id_column].map(lambda v: not str(v).strip()).any():
            raise ValueError("Submission IDs must be non-empty and unique")
        if self.prediction_kind == "probabilities":
            columns = [column for column, _ in self.probability_columns]
            values = frame.loc[:, columns].to_numpy()
            provided = [label for _, label in self.probability_columns]
            self._probabilities(values, provided, len(frame))
        elif self.prediction_kind == "labels":
            values = label_values(frame[self.target_column].tolist(), name="Predicted labels")
            if not set(values) <= set(self.classes):
                raise ValueError("Submission contains an unknown or empty class label")
        else:
            values = pd.to_numeric(frame[self.target_column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Regression predictions must be finite numbers")

    def holdout_frame(self, ids, actual, predictions, *, prediction_classes=None) -> pd.DataFrame:
        ids, actual = list(ids), list(actual)
        if len(ids) != len(actual):
            raise ValueError("Holdout IDs and actual answers must have matching row counts")
        if self.classification:
            actual = label_values(actual, name="Actual labels").tolist()
            if not set(actual) <= set(self.classes):
                raise ValueError("Holdout answers contain an unknown class label")
        output = {"id": ids, "actual": actual}
        if self.prediction_kind == "probabilities":
            matrix = self._probabilities(predictions, prediction_classes, len(ids))
            if len(self.probability_columns) == 1:
                output["prediction"] = matrix[:, self.classes.index(self.positive_class)]
            else:
                for index, label in enumerate(self.classes):
                    output[f"probability:{label}"] = matrix[:, index]
        else:
            values = (label_values(predictions, name="Predicted labels")
                      if self.prediction_kind == "labels" else np.asarray(predictions))
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1 or len(values) != len(ids):
                raise ValueError("Holdout prediction shape does not match the validation rows")
            if self.prediction_kind == "labels" and not set(values) <= set(self.classes):
                raise ValueError("Holdout predictions contain an unknown class label")
            output["prediction"] = values
        return pd.DataFrame(output).loc[:, list(self.holdout_columns)]


def load_task(task_path: Path) -> TaskSpec:
    task_path = Path(task_path).resolve()
    if not task_path.is_file():
        raise ValueError(f"Task file does not exist: {task_path}")
    text = task_path.read_text(encoding="utf-8-sig")
    metric = parse_task_metric(text)
    data_dir = find_data(task_path.parent)
    (train, _, columns, id_column, target, y, features, numeric,
     categorical, classification) = load_data(data_dir, task_text=text)
    classes = tuple(sorted(set(map(str, y)))) if classification else ()
    declared_order = _class_list(text)
    if declared_order is not None:
        if not classification or set(declared_order) != set(classes):
            raise ValueError("Declared class order does not match the training labels")
        classes = declared_order
    declared_type = _declaration(text, r"task[\s_-]*type|тип\s+задачи")
    if declared_type is not None:
        is_class = re.search(r"classif|классифика", declared_type, re.IGNORECASE)
        is_reg = re.search(r"regress|регресси", declared_type, re.IGNORECASE)
        if (bool(is_class) == bool(is_reg)) or (classification != bool(is_class)):
            raise ValueError("Declared task type is unsupported or conflicts with its metric")
    positive = _positive_class(text)
    if positive is not None and positive not in classes:
        raise ValueError("Positive class does not match a training label")
    probability_columns = ()
    if metric.prediction_kind == "probabilities":
        output = [column for column in columns if column != id_column]
        if len(output) == 1:
            if len(classes) != 2:
                raise ValueError("A single probability output requires exactly two training classes")
            inferred = output[0] if output[0] in classes and output[0] != target else infer_positive_class(classes)
            if positive is None:
                positive = inferred
            if positive is None:
                raise ValueError("Single probability output requires an explicit Positive class declaration")
            if output[0] in classes and output[0] != positive:
                raise ValueError("Probability output column conflicts with the positive class")
            probability_columns = ((output[0], positive),)
        elif len(output) == len(classes) and set(output) == set(classes):
            probability_columns = tuple((column, column) for column in output)
            if len(classes) == 2 and positive is None:
                # Both complementary probabilities are present, so binary AUC is
                # identical whichever class is designated positive.
                positive = infer_positive_class(classes) or classes[-1]
        else:
            raise ValueError("Probability columns must name every training class exactly once")
        if metric.name == "ROC AUC" and len(classes) > 2:
            declaration = _auc_declaration(text)
            if (not (re.search(r"\bovr\b|one[ -]vs[ -]rest", declaration, re.IGNORECASE)
                     and re.search(r"\bmacro\b", declaration, re.IGNORECASE))
                    or re.search(r"\b(?:ovo|micro|weighted)\b|one[ -]vs[ -]one", declaration, re.IGNORECASE)):
                raise ValueError("Multiclass ROC AUC requires explicit OVR and macro averaging")
    return TaskSpec(task_path, data_dir, id_column, target, tuple(features), tuple(numeric),
                    tuple(categorical), classes, tuple(columns), probability_columns, positive, metric)


def load_task_data(spec: TaskSpec):
    """Return the familiar data tuple with task semantics overriding target dtype."""
    return load_data(spec.data_dir, task_text=spec.task_path.read_text(encoding="utf-8-sig"))


def score_holdout(path: Path, spec: TaskSpec, expected_ids, expected_actuals) -> float:
    frame = read_csv(Path(path))
    if list(frame.columns) != list(spec.holdout_columns):
        raise ValueError("Holdout prediction columns do not match the task specification")
    if frame["id"].tolist() != list(expected_ids) or frame["actual"].tolist() != list(expected_actuals):
        raise ValueError("Holdout IDs or actual answers differ from the frozen validation rows")
    if spec.prediction_kind == "probabilities" and len(spec.probability_columns) > 1:
        predictions = frame.loc[:, list(spec.holdout_columns[2:])].to_numpy()
        provided = spec.classes
    else:
        predictions = frame["prediction"].to_numpy()
        provided = None
    return score_predictions(spec.metric, frame["actual"].to_numpy(), predictions,
                             classes=spec.classes, prediction_classes=provided,
                             positive_class=spec.positive_class)
