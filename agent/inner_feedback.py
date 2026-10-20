"""Small host-derived error summaries from pinned inner-selection predictions.

No task, source dataset, or outer-validation file is opened here. Class labels
are preserved literally; row IDs and individual answers never enter feedback.
This optional diagnostic must not determine whether a recipe is accepted.
"""

from __future__ import annotations

from collections import Counter
from heapq import nsmallest
from io import BytesIO
import json
import math

import numpy as np
import pandas as pd

from agent.diagnostics import MAX_CLASSES, MAX_LABEL_LENGTH, MAX_ROWS
from agent.metrics import probability_matrix, score_predictions
from agent.research_candidate import CandidateResult
from agent.task_spec import TaskSpec


MAX_FEEDBACK_BYTES = 8 * 1024
MAX_CONFUSIONS = 8


def _encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _classification(actual, predicted, classes, max_classes):
    if any(not isinstance(label, str) or not label.strip() or "\x00" in label
           or len(label) > MAX_LABEL_LENGTH for labels in (actual, predicted) for label in labels):
        raise ValueError("Inner class diagnostics require literal labels of at most 200 characters")
    if not set(actual) <= set(classes) or not set(predicted) <= set(classes):
        raise ValueError("Inner class diagnostics contain undeclared labels")
    support, predicted_count = Counter(actual), Counter(predicted)
    correct = Counter(truth for truth, prediction in zip(actual, predicted) if truth == prediction)
    errors = Counter((truth, prediction) for truth, prediction in zip(actual, predicted)
                     if truth != prediction)
    positions = {label: index for index, label in enumerate(classes)}
    observed = set(support) | set(predicted_count)
    records = []
    for label in observed:
        count, tp = support[label], correct[label]
        denominator = count + predicted_count[label]
        records.append({"label": label, "support": count,
                        "recall": tp / count if count else 0.0,
                        "f1": 2 * tp / denominator if denominator else 0.0})
    selected = nsmallest(max_classes, records,
                         key=lambda row: (row["f1"], row["recall"], row["support"], positions[row["label"]]))
    confusions = nsmallest(MAX_CONFUSIONS, errors.items(), key=lambda item: (
        -item[1], positions[item[0][0]], positions[item[0][1]]))
    return {
        "classes_total": len(classes), "observed_classes": len(observed),
        "accuracy": sum(correct.values()) / len(actual),
        "per_class": selected,
        "worst_confusions": [{"actual_class": pair[0], "predicted_class": pair[1], "count": count}
                             for pair, count in confusions],
        "omitted_classes": len(observed) - len(selected),
        "omitted_confusions": len(errors) - len(confusions),
    }


def _bound_feedback(value):
    classification = value.get("classification")
    while _encoded_size(value) > MAX_FEEDBACK_BYTES:
        if classification is None:
            raise ValueError("Inner diagnostic exceeds its feedback size limit")
        rows, confusions = classification["per_class"], classification["worst_confusions"]
        # Keep the highest-priority errors and complete labels. Byte limits
        # apply after UTF-8 encoding, so long Unicode labels receive fewer rows.
        if confusions and (len(confusions) >= len(rows) or len(rows) <= 1):
            confusions.pop()
            classification["omitted_confusions"] += 1
        elif rows:
            rows.pop()
            classification["omitted_classes"] += 1
        else:
            raise ValueError("Inner diagnostic exceeds its feedback size limit")
    return value


def inner_feedback(spec: TaskSpec, candidate_result: CandidateResult, actuals, *,
                   max_classes: int = 12) -> dict:
    """Summarize verified inner predictions with host-owned validation answers.

    The caller passes only inner actuals. A result from any other phase is
    rejected before its pinned output is read. Errors propagate to the caller,
    which can omit this optional detail without changing the selected recipe.
    """
    if not isinstance(spec, TaskSpec) or not isinstance(candidate_result, CandidateResult):
        raise ValueError("Inner diagnostics require a task specification and verified candidate result")
    if candidate_result.phase != "inner_selection":
        raise ValueError("Inner diagnostics accept only the inner_selection phase")
    if type(max_classes) is not int or not 1 <= max_classes <= MAX_CLASSES:
        raise ValueError("max_classes must be an integer from 1 to 50")
    if isinstance(actuals, (str, bytes)):
        raise ValueError("Inner actuals must be a non-empty sequence")
    try:
        actual = list(actuals)
    except TypeError as error:
        raise ValueError("Inner actuals must be a non-empty sequence") from error
    if not 1 <= len(actual) <= MAX_ROWS:
        raise ValueError("Inner actuals must be a non-empty bounded sequence")
    output = candidate_result.read_output()
    frame = pd.read_csv(BytesIO(output), dtype=str, keep_default_na=False)
    if len(frame) != len(actual) or spec.id_column not in frame:
        raise ValueError("Inner predictions and host actuals have different row counts or schema")
    # IDs were already checked against host input when CandidateResult was
    # created. Revalidate shape/types here, and never return these IDs.
    spec.validate_frame(frame, frame[spec.id_column].tolist())
    provided_classes = None
    if spec.prediction_kind == "probabilities":
        provided_classes = tuple(label for _, label in spec.probability_columns)
        predictions = frame.loc[:, [column for column, _ in spec.probability_columns]].to_numpy()
        matrix = probability_matrix(predictions, classes=spec.classes,
                                    prediction_classes=provided_classes,
                                    positive_class=spec.positive_class, rows=len(actual))
        labels = np.asarray(spec.classes, dtype=object)[np.argmax(matrix, axis=1)].tolist()
    else:
        predictions = frame[spec.target_column].to_numpy()
        labels = predictions.tolist()
    if spec.classification:
        details = _classification(actual, labels, spec.classes, max_classes)
    else:
        try:
            truth, values = np.asarray(actual, dtype=float), np.asarray(predictions, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Inner regression values must be finite numeric vectors") from error
        if truth.ndim != 1 or not np.isfinite(truth).all() or not np.isfinite(values).all():
            raise ValueError("Inner regression values must be finite numeric vectors")
        with np.errstate(over="ignore", invalid="ignore"):
            errors = values - truth
        if not np.isfinite(errors).all():
            raise ValueError("Inner regression errors must be finite")
        absolute = np.abs(errors)
        details = {"mean_bias": math.fsum(float(value) / len(errors) for value in errors),
                   "mae": math.fsum(float(value) / len(errors) for value in absolute),
                   "absolute_error_p50": float(np.quantile(absolute, .5)),
                   "absolute_error_p90": float(np.quantile(absolute, .9)),
                   "absolute_error_p99": float(np.quantile(absolute, .99))}
        if not all(math.isfinite(value) for value in details.values()):
            raise ValueError("Inner regression diagnostic must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        score = score_predictions(spec.metric, actual, predictions,
                                  classes=spec.classes, prediction_classes=provided_classes,
                                  positive_class=spec.positive_class)
    result = {"version": 1, "scope": "inner_selection", "rows": len(actual),
              "metric": {"name": spec.metric.name, "value": score,
                         "higher_is_better": spec.metric.higher_is_better},
              "classification" if spec.classification else "regression": details}
    if spec.prediction_kind == "probabilities":
        result["classification"]["label_rule"] = "argmax_in_declared_class_order"
    return _bound_feedback(result)
