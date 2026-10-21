"""Small, host-derived research history and error summaries for the next trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from agent.diagnostics import classification_diagnostics
from agent.metrics import probability_matrix, score_predictions
from offline_baseline import read_csv


def code_digest(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def error_analysis(spec, holdout_path: Path, *, groups=None) -> dict:
    """Call only after the frozen host validator accepted the evidence shape."""
    frame = read_csv(holdout_path)
    actual = frame["actual"].to_numpy()
    if spec.prediction_kind == "probabilities":
        raw = frame.loc[:, list(spec.holdout_columns[2:])].to_numpy()
        predictions = probability_matrix(raw, classes=spec.classes,
                                         positive_class=spec.positive_class)
        labels = np.asarray(spec.classes)[np.argmax(predictions, axis=1)]
    else:
        predictions = frame["prediction"].to_numpy()
        labels = predictions
    result = {"rows": len(frame), "scope": "outer_validation_diagnostics"}
    if spec.classification:
        details = classification_diagnostics(actual, labels)
        if details is not None:
            result["classification"] = details
    else:
        truth, values = np.asarray(actual, dtype=float), np.asarray(predictions, dtype=float)
        errors = values - truth
        result["regression"] = {
            "mean_error": float(np.mean(errors)),
            "absolute_error_p50": float(np.quantile(np.abs(errors), .5)),
            "absolute_error_p90": float(np.quantile(np.abs(errors), .9)),
            "absolute_error_p99": float(np.quantile(np.abs(errors), .99)),
        }
    slices = []
    for column, values in list((groups or {}).items())[:3]:
        values = np.asarray(values, dtype=str)
        if len(values) != len(actual) or not 2 <= len(set(values)) <= 15:
            continue
        for label in sorted(set(values)):
            mask = values == label
            if int(mask.sum()) < 20:
                continue
            try:
                value = score_predictions(spec.metric, actual[mask], predictions[mask],
                                          classes=spec.classes, positive_class=spec.positive_class)
            except ValueError:
                continue
            slices.append({"feature": column, "group": label[:200], "rows": int(mask.sum()),
                           "metric": spec.metric.name, "value": value})
    if slices:
        result["groups"] = slices
    return result


class ResearchLedger:
    """Never trusts scores, hypothesis quality, or costs stated by the model."""

    def __init__(self):
        self.trials = []
        self.failed_codes = set()
        self.failure_streak = 0

    def is_repeated_failure(self, code: str) -> bool:
        return code_digest(code) in self.failed_codes

    def record(self, code: str, result: dict, *, remaining: float) -> dict:
        selection = result.get("selection", {})
        execution_failed = ("error" in result or result.get("exit_code") != 0
                            or result.get("timed_out", False)
                            or result.get("memory_limit_exceeded", False))
        digest = code_digest(code)
        if execution_failed:
            self.failed_codes.add(digest)
            self.failure_streak += 1
        else:
            self.failure_streak = 0
        row = {
            "number": len(self.trials) + 1, "code_hash": digest,
            "hypothesis": selection.get("hypothesis"),
            "accepted": selection.get("accepted", False),
            "verified": selection.get("verified", False),
            "metric": selection.get("metric"), "reason": selection.get("reason"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "remaining_seconds": round(max(0, remaining), 3),
            "resources": result.get("resources"),
        }
        if selection.get("scope") is not None:
            row["evaluation_scope"] = selection["scope"]
            row["candidate_code_hash"] = selection.get("code_hash")
            row["recipe_hash"] = selection.get("recipe_hash")
            row["recipe"] = selection.get("recipe_summary")
        error = result.get("error") or (result.get("stderr") if execution_failed else None)
        if error:
            row["error"] = str(error)[-1500:]
        if "diagnostics" in selection:
            row["diagnostics"] = selection["diagnostics"]
        self.trials.append(row)
        return row

    def context(self, *, best, remaining: float, api: dict) -> str:
        value = {"best_verified": best, "remaining_seconds": round(max(0, remaining), 1),
                 "consecutive_execution_failures": self.failure_streak,
                 "recent_trials": self.trials[-6:],
                 "reported_token_usage": {key: api.get(key, 0) for key in (
                     "prompt_tokens", "completion_tokens", "total_tokens")},
                 "monetary_cost": None,
                 "next_trial_rule": "Use inner validation for parameter choices; run a new hypothesis. "
                     "Do not repeat failed code unchanged. Reserve time for full-data refit. "
                     "Outer diagnostics describe this holdout only, not hidden-test quality."}
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
