"""Frozen holdout evidence for generated research, independent of claimed scores.

This validates registered tabular metrics and their output contracts. It is not an
execution sandbox and does not prove that generated training code is leakage-free.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.automl import (
    _atomic_json, _json_hash, choose_validation, data_fingerprint,
)
from agent.splits import split_rows
from agent.search import inner_split_rows
from agent.task_spec import load_task, load_task_data, score_holdout
from offline_baseline import read_csv


HOLDOUT_NAME = "holdout_predictions.csv"
HYPOTHESIS_NAME = "research_hypothesis.json"
MANIFEST_NAME = "research_validation.json"


def read_hypothesis(directory: Path) -> str:
    path = Path(directory) / HYPOTHESIS_NAME
    if path.stat().st_size > 16 * 1024:
        raise ValueError("research_hypothesis.json exceeds 16 KiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    hypothesis = value.get("hypothesis") if isinstance(value, dict) else None
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 2000:
        raise ValueError("research_hypothesis.json requires a non-empty hypothesis (at most 2000 characters)")
    return hypothesis.strip()


class ResearchValidation:
    """Keep expected rows/answers in the parent, never reload them from LLM files."""

    def __init__(self, task_path: Path):
        self.task_path = Path(task_path)
        self.task_text = self.task_path.read_text(encoding="utf-8-sig")
        self.task_dir = self.task_path.parent
        self.spec = load_task(self.task_path)
        self.data_dir = self.spec.data_dir
        self.fingerprint = data_fingerprint(self.data_dir)
        (train, test, _, id_column, target, y, features, numeric,
         categorical, classification) = load_task_data(self.spec)
        choice, reason = choose_validation(train, test, features, numeric, classification)
        fit_rows, held_rows, split = split_rows(train, y, features, classification, choice)
        self.fit_rows = fit_rows.tolist()
        self.held_rows = held_rows.tolist()
        self.expected_ids = train[id_column].iloc[held_rows].tolist()
        self.expected_actuals = train[target].iloc[held_rows].tolist()
        self.expected_test_ids = test[id_column].tolist()
        self.classification = classification
        self.allowed_classes = y.unique().tolist() if classification else None
        self.metric_name = self.spec.metric.name
        self.groups = {column: train[column].iloc[held_rows].tolist()
                       for column in categorical
                       if 2 <= train[column].iloc[fit_rows].nunique() <= 15}
        split["validation_hash"] = _json_hash({
            "id_column": id_column, "target_column": target,
            "ids": self.expected_ids, "actuals": self.expected_actuals,
        })
        self.split = split
        self.manifest = {
            "version": 2, "data_directory": self.data_dir.relative_to(self.task_dir).as_posix(),
            "id_column": id_column, "target_column": target, "features": features,
            "numeric": numeric, "categorical": categorical,
            "fit_rows": self.fit_rows, "validation_rows": self.held_rows,
            "metric": self.spec.metric.to_dict(),
            "classes": list(self.spec.classes), "positive_class": self.spec.positive_class,
            "output_columns": list(self.spec.output_columns),
            "probability_columns": [list(pair) for pair in self.spec.probability_columns],
            "split": split, "validation_reason": reason,
            "holdout_columns": list(self.spec.holdout_columns),
            "holdout_file": HOLDOUT_NAME, "hypothesis_file": HYPOTHESIS_NAME,
        }
        try:
            inner_fit, inner_held, inner_split = inner_split_rows(
                train, y, features, classification, choice, fit_rows)
            self.manifest.update(inner_fit_rows=inner_fit.tolist(),
                                 inner_validation_rows=inner_held.tolist(), inner_split=inner_split)
        except ValueError as error:
            self.manifest.update(inner_fit_rows=[], inner_validation_rows=[],
                                 inner_unavailable_reason=str(error))
        self.check_inputs()
        self.write_manifest()

    def check_inputs(self) -> None:
        if (data_fingerprint(self.data_dir) != self.fingerprint
                or self.task_path.read_text(encoding="utf-8-sig") != self.task_text):
            raise ValueError("Research task or input data changed; candidate cannot be accepted")

    def write_manifest(self) -> None:
        # A generated script may overwrite the public copy; parent state is fixed.
        _atomic_json(self.task_dir / MANIFEST_NAME, self.manifest)

    def evaluate(self, holdout_path: Path, submission_dir: Path) -> dict:
        self.check_inputs()
        self.spec.validate_frame(read_csv(Path(submission_dir) / "submission.csv"),
                                 self.expected_test_ids)
        score = score_holdout(Path(holdout_path), self.spec, self.expected_ids, self.expected_actuals)
        return {"name": self.metric_name, "value": score,
                "higher_is_better": self.spec.metric.higher_is_better}
