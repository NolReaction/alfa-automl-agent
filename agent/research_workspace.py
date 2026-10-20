"""Research v3 inputs: an outer-fit view, with host-only outer answers.

This controls the ordinary data path. Generated Python still runs under the
same OS user; separate directories are not an operating-system sandbox.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.automl import _atomic_json, choose_validation, data_fingerprint
from agent.metrics import score_predictions
from agent.search import inner_split_rows
from agent.splits import normalize_validation, split_rows
from agent.task_spec import load_task, load_task_data
from offline_baseline import read_csv


MANIFEST_NAME = "research_validation.json"


def _frozen_rows(value, count: int, name: str) -> np.ndarray:
    if (not isinstance(value, (list, tuple, np.ndarray)) or not len(value)
            or any(type(item) is not int and not isinstance(item, np.integer) for item in value)):
        raise ValueError(f"Frozen {name} must contain integer row positions")
    result = np.asarray(value, dtype=int)
    if (result.ndim != 1 or len(set(result.tolist())) != len(result)
            or (result < 0).any() or (result >= count).any()):
        raise ValueError(f"Frozen {name} contains repeated or out-of-range row positions")
    return result


def _additional_hypothesis(text: str) -> str | None:
    """Preserve the bounded explicit Alfa Lab hint, not arbitrary task prose."""
    marker = "## Дополнительная гипотеза для исследования\n"
    if marker not in text:
        return None
    hint = text.rsplit(marker, 1)[1].split("\n\nПроверь эту гипотезу,", 1)[0].strip()
    if not hint or len(hint) > 500:
        raise ValueError("Additional research hypothesis must contain 1..500 characters")
    return hint


def _task_text(spec, hypothesis=None) -> str:
    """Recreate required declarations without source paths or prior results."""
    metric = spec.metric.name
    if metric == "ROC AUC" and len(spec.classes) > 2:
        metric += " OVR macro"
    lines = [
        "# Research selection task",
        f"Primary metric: {metric}",
        f"Task type: {spec.metric.task_type}",
        "Use data/train.csv, data/test.csv, data/data_dictionary.csv and data/sample_submission.csv.",
        "The train contains only the host-approved outer-fit rows.",
        "The test is an inner-validation feature view, not the original competition test.",
        "Use research_validation.json for the frozen inner fit/validation positions.",
        "All reported measurements at this stage are selection measurements.",
        "Write an executable research_candidate.py and research_recipe.json for host validation.",
    ]
    if spec.classes:
        lines.append("Class order: " + json.dumps(list(spec.classes), ensure_ascii=False))
    if spec.positive_class is not None:
        lines.append("Positive class: " + json.dumps(spec.positive_class, ensure_ascii=False))
    if hypothesis is not None:
        lines.extend(["", "## Additional user hypothesis", hypothesis,
                      "Test this suggestion using only the approved inputs and inner validation."])
    return "\n".join(lines) + "\n"


class ResearchWorkspace:
    """Freeze host data once and expose only an outer-fit research view.

    ``frozen_split`` is a host-only replay contract with fit_rows,
    validation_rows, choice and split. It supports interventions after split
    selection; it must never come from generated code.
    """

    def __init__(self, task_path: Path, workspace: Path, *, frozen_split=None):
        self.task_path = Path(task_path).resolve()
        self.source_task_text = self.task_path.read_text(encoding="utf-8-sig")
        self.spec = load_task(self.task_path)
        self.source_dir = self.spec.data_dir
        self.source_fingerprint = data_fingerprint(self.source_dir)
        (train, test, _, _, _, y, features, numeric,
         categorical, classification) = load_task_data(self.spec)
        self.train, self.test, self.y = train.copy(deep=True), test.copy(deep=True), y.copy(deep=True)
        self.features, self.numeric, self.categorical = tuple(features), tuple(numeric), tuple(categorical)
        self.classification = classification
        if frozen_split is None:
            self.choice, self.validation_reason = choose_validation(
                train, test, features, numeric, classification)
            fitting, held, split = split_rows(train, y, features, classification, self.choice)
        else:
            if not isinstance(frozen_split, dict) or set(frozen_split) != {
                    "fit_rows", "validation_rows", "choice", "split"}:
                raise ValueError("Frozen split requires fit_rows, validation_rows, choice and split")
            fitting = _frozen_rows(frozen_split["fit_rows"], len(train), "fit_rows")
            held = _frozen_rows(frozen_split["validation_rows"], len(train), "validation_rows")
            self.choice = json.loads(json.dumps(frozen_split["choice"], allow_nan=False))
            split = json.loads(json.dumps(frozen_split["split"], allow_nan=False))
            self.validation_reason = "Host replay of a previously fixed split."
        self.choice = normalize_validation(self.choice)
        if (set(map(int, fitting)) & set(map(int, held))
                or set(map(int, fitting)) | set(map(int, held)) != set(range(len(train)))):
            raise ValueError("Frozen outer split must partition every training row exactly once")
        if self.choice["strategy"] == "time":
            periods = pd.to_numeric(train[self.choice["time_column"]], errors="coerce").to_numpy()
            if not np.isfinite(periods).all() or periods[fitting].max() >= periods[held].min():
                raise ValueError("Frozen time split must use strictly earlier fit periods")
        elif self.choice["strategy"] == "grouped":
            groups = pd.util.hash_pandas_object(train.loc[:, features], index=False).to_numpy()
            if set(groups[fitting]) & set(groups[held]):
                raise ValueError("Frozen grouped split must keep duplicate feature groups separate")
        self.outer_fit_rows = np.asarray(fitting, dtype=int)
        self.outer_holdout_rows = np.asarray(held, dtype=int)
        self.outer_fit_rows.setflags(write=False)
        self.outer_holdout_rows.setflags(write=False)
        self.split = json.loads(json.dumps(split, allow_nan=False))
        self.frozen_split = {
            "fit_rows": self.outer_fit_rows.tolist(),
            "validation_rows": self.outer_holdout_rows.tolist(),
            "choice": json.loads(json.dumps(self.choice)), "split": self.split.copy(),
        }
        self.expected_ids = train[self.spec.id_column].iloc[held].tolist()
        self.expected_actuals = train[self.spec.target_column].iloc[held].tolist()
        self.expected_test_ids = test[self.spec.id_column].tolist()
        self.outer_fit_frame = train.iloc[fitting].reset_index(drop=True).copy(deep=True)
        self.outer_holdout_frame = train.iloc[held].reset_index(drop=True).copy(deep=True)
        self.research_train = self.outer_fit_frame.copy(deep=True)
        self.inner_unavailable_reason = None
        try:
            inner_fit, inner_held, inner_metadata = inner_split_rows(
                self.research_train, y.iloc[fitting].reset_index(drop=True),
                features, classification, self.choice, np.arange(len(fitting)))
            self.fit_rows, self.held_rows = inner_fit.tolist(), inner_held.tolist()
            self.inner_split = inner_metadata
        except ValueError as error:
            self.fit_rows, self.held_rows, self.inner_split = [], [], None
            self.inner_unavailable_reason = str(error)
        self.inner_fit_rows, self.inner_validation_rows = self.fit_rows, self.held_rows
        self.expected_inner_ids = self.research_train[self.spec.id_column].iloc[self.held_rows].tolist()
        self.expected_inner_actuals = self.research_train[self.spec.target_column].iloc[self.held_rows].tolist()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.work_task = self.workspace / "task.md"
        self.work_data = self.workspace / "data"
        self.work_data.mkdir()
        self.task_text = _task_text(self.spec, _additional_hypothesis(self.source_task_text))
        self.work_task.write_text(self.task_text, encoding="utf-8")
        self.research_train.to_csv(self.work_data / "train.csv", index=False)
        # Keep a valid CSV task for the existing AutoML entry point. These rows
        # come exclusively from outer-fit and carry no target column.
        test_rows = self.held_rows if self.held_rows else [0]
        research_test = self.research_train.iloc[test_rows].drop(columns=[self.spec.target_column])
        research_test.to_csv(self.work_data / "test.csv", index=False)
        template = pd.DataFrame({name: [""] * len(research_test) for name in self.spec.output_columns})
        template[self.spec.id_column] = research_test[self.spec.id_column].tolist()
        template.to_csv(self.work_data / "sample_submission.csv", index=False)
        # Metadata declares the fixed schema, without any per-row outer values.
        read_csv(self.source_dir / "data_dictionary.csv").to_csv(
            self.work_data / "data_dictionary.csv", index=False)
        self.research_spec = replace(self.spec, task_path=self.work_task, data_dir=self.work_data)
        self.work_fingerprint = data_fingerprint(self.work_data)
        self.manifest = {
            "version": 3, "stage": "research", "purpose": "selection",
            "data_directory": "data", "training_data_scope": "outer_fit_only",
            "allowed_inputs": ["task.md", "data/train.csv", "data/test.csv",
                               "data/data_dictionary.csv", "data/sample_submission.csv", MANIFEST_NAME],
            "id_column": self.spec.id_column, "target_column": self.spec.target_column,
            "features": list(self.spec.features), "numeric": list(self.spec.numeric),
            "categorical": list(self.spec.categorical), "metric": self.spec.metric.to_dict(),
            "classes": list(self.spec.classes), "positive_class": self.spec.positive_class,
            "output_columns": list(self.spec.output_columns),
            "probability_columns": [list(pair) for pair in self.spec.probability_columns],
            "fit_rows": self.fit_rows, "validation_rows": self.held_rows,
            "split": self.inner_split, "validation_purpose": "selection",
            "holdout_columns": list(self.spec.output_columns), "holdout_file": "inner_predictions.csv",
            "candidate_file": "research_candidate.py", "recipe_file": "research_recipe.json",
            "hypothesis_file": "research_hypothesis.json",
            "required_outputs": ["research_candidate.py", "research_recipe.json", "research_hypothesis.json"],
            "candidate_contract": {
                "version": 3,
                "signature": "fit_predict(X_fit, y_fit, X_predict, context)",
                "arguments": {
                    "X_fit": "Raw string pandas.DataFrame with feature columns only; no ID or target.",
                    "y_fit": "Numeric regression pandas.Series or literal string class pandas.Series for X_fit only.",
                    "X_predict": "Raw string pandas.DataFrame with the same feature columns; no ID or target.",
                    "context": {
                        "protocol_version": 3,
                        "fields": ["version", "protocol_version", "phase", "nonce", "recipe",
                                   "code_hash", "recipe_hash", "features", "numeric", "categorical",
                                   "classes", "positive_class", "metric", "task_type", "prediction_kind"],
                        "phase_purposes": {"selection": ["selection", "inner_selection"],
                                           "final_check": ["final_check", "outer_validation"],
                                           "refit": ["refit", "full_refit"]},
                    },
                },
                "return": {
                    "type": "dict", "allowed_keys": ["predictions", "classes"],
                    "required_keys": ["predictions"],
                    "predictions": "One prediction per X_predict row, in that exact order; finite values, literal labels or normalized probabilities as declared by metric.",
                    "classes": "Required for probabilities: non-empty list or tuple of literal string labels in prediction-column order. Optional otherwise.",
                },
                "training_rule": "Use only supplied arguments for training; do not read files, memorize row positions, or branch on phase.",
                "host_owns": ["ID assignment", "prediction CSV", "frozen evaluation answers", "scoring", "publication"],
                "limits": {"code_bytes": 262144, "recipe_bytes": 65536},
            },
            "completion": "Freeze the selected recipe before host final_check; no research follows final_check.",
            "isolation": "allowed-input view; no operating-system sandbox",
        }
        if self.inner_unavailable_reason:
            self.manifest["inner_unavailable_reason"] = self.inner_unavailable_reason
        self.check_source()
        self.write_manifest()

    def check_source(self) -> None:
        if (self.task_path.read_text(encoding="utf-8-sig") != self.source_task_text
                or data_fingerprint(self.source_dir) != self.source_fingerprint):
            raise ValueError("Original task inputs changed during research; publication cancelled")

    def check_inputs(self) -> None:
        self.check_source()
        if (self.work_task.read_text(encoding="utf-8-sig") != self.task_text
                or data_fingerprint(self.work_data) != self.work_fingerprint):
            raise ValueError("Allowed research inputs changed; candidate cannot be accepted")

    def write_manifest(self) -> None:
        _atomic_json(self.workspace / MANIFEST_NAME, self.manifest)

    def evaluate_inner(self, path: Path) -> dict:
        """Score predictions-only rows against the host's frozen inner answers."""
        self.check_inputs()
        if not self.held_rows:
            raise ValueError("Inner validation is unavailable: " + str(self.inner_unavailable_reason))
        frame = read_csv(Path(path))
        self.spec.validate_frame(frame, self.expected_inner_ids)
        if self.spec.prediction_kind == "probabilities":
            columns = list(self.spec.prediction_columns)
            prediction = frame.loc[:, columns].to_numpy()
            classes = [label for _, label in self.spec.probability_columns]
            if len(columns) == 1:
                prediction, classes = prediction[:, 0], None
        else:
            prediction, classes = frame[self.spec.target_column].to_numpy(), None
        value = score_predictions(
            self.spec.metric, self.expected_inner_actuals, prediction,
            classes=self.spec.classes, prediction_classes=classes,
            positive_class=self.spec.positive_class)
        return {"name": self.spec.metric.name, "value": value,
                "higher_is_better": self.spec.metric.higher_is_better,
                "purpose": "selection", "split": "inner"}
