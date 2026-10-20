"""Bounded classification errors derived from one verified local holdout.

Rows of the confusion matrix are actual labels; columns are predicted labels.
This module uses only the standard library so report validation does not load
the numerical training stack. A shared report is evidence, not an independent
verification that its predictions came from a particular model.
"""

from __future__ import annotations

import math


MAX_CLASSES = 50
MAX_LABEL_LENGTH = 200
MAX_ROWS = (1 << 31) - 1
_FIELDS = {"version", "kind", "labels", "rows", "confusion_matrix", "per_class",
           "accuracy", "macro_f1"}
_CLASS_FIELDS = {"precision", "recall", "f1", "support", "predicted"}
_TOLERANCE = 1e-10


def _count(value, field):
    if type(value) is not int or not 0 <= value <= MAX_ROWS:
        raise ValueError(f"Некорректное число в диагностике классификации: {field}.")
    return value


def _fraction(value, field):
    if (type(value) not in {int, float} or not 0 <= value <= 1
            or not math.isfinite(value)):
        raise ValueError(f"Некорректная доля в диагностике классификации: {field}.")
    return float(value)


def _label(value):
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _from_matrix(labels, matrix):
    """Return canonical values, computed solely from already checked counts."""
    supports = [sum(row) for row in matrix]
    predicted = [sum(row[column] for row in matrix) for column in range(len(labels))]
    total = sum(supports)
    if not 1 <= total <= MAX_ROWS:
        raise ValueError("Диагностика должна описывать непустую ограниченную проверочную выборку.")
    classes = []
    for index, support in enumerate(supports):
        true_positive = matrix[index][index]
        denominator = support + predicted[index]
        if denominator == 0:
            raise ValueError("Каждый класс должен встречаться в ответах или предсказаниях.")
        classes.append({
            "precision": true_positive / predicted[index] if predicted[index] else 0.0,
            "recall": true_positive / support if support else 0.0,
            "f1": 2 * true_positive / denominator if denominator else 0.0,
            "support": support,
            "predicted": predicted[index],
        })
    return {
        "version": 1, "kind": "classification", "labels": list(labels), "rows": total,
        "confusion_matrix": [list(row) for row in matrix], "per_class": classes,
        "accuracy": sum(matrix[index][index] for index in range(len(labels))) / total,
        "macro_f1": sum(row["f1"] for row in classes) / len(classes),
    }


def validate_classification_diagnostics(value, *, metric=None, validation_rows=None):
    """Validate and canonicalize a small confusion matrix and all its summaries.

    Labels include every actual or predicted class, matching sklearn's default
    Macro F1 label set. Optional enclosing metric/row counts must agree, so a
    report cannot attach another holdout's otherwise internally valid matrix.
    """
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Неполный или неизвестный формат диагностики классификации.")
    if type(value["version"]) is not int or value["version"] != 1 or value["kind"] != "classification":
        raise ValueError("Неизвестная версия диагностики классификации.")
    labels = value["labels"]
    if (not isinstance(labels, list) or not 1 <= len(labels) <= MAX_CLASSES
            or any(not _label(label) or len(label) > MAX_LABEL_LENGTH for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Некорректные или повторяющиеся классы в диагностике.")
    matrix = value["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != len(labels):
        raise ValueError("Размер матрицы ошибок не соответствует классам.")
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(labels):
            raise ValueError("Матрица ошибок должна быть квадратной.")
        for count in row:
            _count(count, "confusion_matrix")
    result = _from_matrix(labels, matrix)
    if _count(value["rows"], "rows") != result["rows"]:
        raise ValueError("Число строк не совпадает с матрицей ошибок.")
    rows = value["per_class"]
    if not isinstance(rows, list) or len(rows) != len(labels):
        raise ValueError("Число строк метрик не соответствует классам.")
    for row, expected in zip(rows, result["per_class"]):
        if not isinstance(row, dict) or set(row) != _CLASS_FIELDS:
            raise ValueError("Некорректный формат метрик отдельного класса.")
        for name in ("support", "predicted"):
            if _count(row[name], name) != expected[name]:
                raise ValueError("Число примеров класса не совпадает с матрицей ошибок.")
        for name in ("precision", "recall", "f1"):
            if abs(_fraction(row[name], name) - expected[name]) > _TOLERANCE:
                raise ValueError("Метрики класса не совпадают с матрицей ошибок.")
    for name in ("accuracy", "macro_f1"):
        if abs(_fraction(value[name], name) - result[name]) > _TOLERANCE:
            raise ValueError("Общая метрика не совпадает с матрицей ошибок.")
    if validation_rows is not None:
        if _count(validation_rows, "validation_rows") != result["rows"]:
            raise ValueError("Диагностика относится к другой проверочной выборке.")
    if metric is not None:
        if (not isinstance(metric, dict) or metric.get("name") != "Macro F1"
                or metric.get("higher_is_better") is not True
                or abs(_fraction(metric.get("value"), "metric.value") - result["macro_f1"]) > _TOLERANCE):
            raise ValueError("Диагностика не соответствует метрике запуска.")
    return result


def classification_diagnostics(actual, predicted):
    """Compute bounded, serializable errors; return None for unsupported sizes.

    Excessive class counts/label lengths only omit this optional detail. Empty,
    non-string or misaligned inputs are errors rather than silently coerced
    predictions. Labels such as ``NA`` retain their exact original spelling.
    """
    if isinstance(actual, (str, bytes)) or isinstance(predicted, (str, bytes)):
        raise ValueError("Передайте последовательности правильных и предсказанных классов.")
    try:
        actual, predicted = list(actual), list(predicted)
    except TypeError as error:
        raise ValueError("Передайте последовательности правильных и предсказанных классов.") from error
    if not actual or len(actual) != len(predicted):
        raise ValueError("Для диагностики нужны одинаковые непустые последовательности классов.")
    if len(actual) > MAX_ROWS:
        return None
    if any(not _label(label) for labels in (actual, predicted) for label in labels):
        raise ValueError("Классы должны быть непустыми строками без нулевых символов.")
    if any(len(label) > MAX_LABEL_LENGTH for labels in (actual, predicted) for label in labels):
        return None
    labels = sorted(set(actual) | set(predicted))
    if len(labels) > MAX_CLASSES:
        return None
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for truth, prediction in zip(actual, predicted):
        matrix[positions[truth]][positions[prediction]] += 1
    return _from_matrix(labels, matrix)
