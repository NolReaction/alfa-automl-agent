"""Frozen holdout evidence for generated research, independent of claimed scores.

This validates the documented tabular RMSE/Macro F1 contract. It is not an
execution sandbox and does not prove that generated training code is leakage-free.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.automl import (
    _atomic_json, _json_hash, _score_holdout, choose_validation, data_fingerprint,
)
from agent.splits import split_rows
from agent.task_contract import validate_task_metric
from offline_baseline import find_data, load_data
from validation import validate_predictions


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
        self.data_dir = find_data(self.task_dir)
        self.fingerprint = data_fingerprint(self.data_dir)
        (train, test, _, id_column, target, y, features, numeric,
         categorical, classification) = load_data(self.data_dir)
        validate_task_metric(self.task_text, classification)
        choice, reason = choose_validation(train, test, features, numeric, classification)
        fit_rows, held_rows, split = split_rows(train, y, features, classification, choice)
        self.fit_rows = fit_rows.tolist()
        self.held_rows = held_rows.tolist()
        self.expected_ids = train[id_column].iloc[held_rows].tolist()
        self.expected_actuals = train[target].iloc[held_rows].tolist()
        self.classification = classification
        self.allowed_classes = y.unique().tolist() if classification else None
        self.metric_name = "Macro F1" if classification else "RMSE"
        split["validation_hash"] = _json_hash({
            "id_column": id_column, "target_column": target,
            "ids": self.expected_ids, "actuals": self.expected_actuals,
        })
        self.split = split
        self.manifest = {
            "version": 1, "data_directory": self.data_dir.relative_to(self.task_dir).as_posix(),
            "id_column": id_column, "target_column": target, "features": features,
            "numeric": numeric, "categorical": categorical,
            "fit_rows": self.fit_rows, "validation_rows": self.held_rows,
            "metric": {"name": self.metric_name, "higher_is_better": classification},
            "split": split, "validation_reason": reason,
            "holdout_columns": ["id", "actual", "prediction"],
            "holdout_file": HOLDOUT_NAME, "hypothesis_file": HYPOTHESIS_NAME,
        }
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
        check = validate_predictions(Path(submission_dir))
        if not check["valid"]:
            raise ValueError("Output check failed: " + "; ".join(check["errors"]))
        score = _score_holdout(Path(holdout_path), self.expected_ids, self.expected_actuals,
                               self.classification, self.allowed_classes)
        return {"name": self.metric_name, "value": score,
                "higher_is_better": self.classification}
