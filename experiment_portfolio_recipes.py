"""Bounded offline recipe smoke checks on synthetic mixed-type tables.

This checks execution and prediction contracts, not comparative model quality.
No public task data, API credentials or network requests are used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from agent.metrics import get_metric, score_predictions
from agent.model_config import ACTIONS
from agent.portfolio_recipes import portfolio_candidates

ROOT = Path(__file__).resolve().parent
SEED = 1729


def smoke_data(task_type, binary=False):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(SEED)
    number = rng.normal(size=160)
    frame = pd.DataFrame({"number": number, "empty": np.full(160, np.nan),
                          "category": ["NA", "blue", "red", ""] * 40})
    frame.loc[::11, "number"] = np.nan
    frame.loc[128:, "category"] = "holdout_only"
    if task_type == "classification":
        labels = np.asarray(["001", "NA"] if binary else ["001", "NA", "Attack"])
        target = labels[np.arange(len(frame)) % len(labels)]
    else:
        target = number * 2 + rng.normal(scale=0.1, size=len(frame))
    return frame.iloc[:128].copy(), target[:128], frame.iloc[128:].copy(), target[128:]


def _worker(candidate, task_type, binary, directory):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agent.models import build_pipeline, predict_for_metric

    fit, y_fit, held, actual = smoke_data(task_type, binary)
    model = build_pipeline(candidate["action"], task_type, ["number", "empty"], ["category"],
                           candidate["model_config"], candidate["feature_config"])
    with threadpool_limits(limits=8):
        model.fit(fit, y_fit)
        prediction = predict_for_metric(model, held, "LogLoss" if task_type == "classification" else "RMSE")
    table = pd.DataFrame({"id": [f"{i:05d}" for i in range(len(held))], "actual": actual})
    classes = tuple(map(str, model.classes_)) if task_type == "classification" else ()
    if classes:
        for i in range(len(classes)):
            table[f"probability_{i}"] = prediction[:, i]
    else:
        table["prediction"] = np.asarray(prediction)
    directory = Path(directory)
    table.to_csv(directory / "predictions.csv", index=False)
    (directory / "classes.json").write_text(json.dumps(classes), encoding="utf-8")


def score_saved(directory, task_type, metric, binary=False):
    """Re-read persisted predictions and verify IDs/answers against the fixture."""
    import numpy as np
    import pandas as pd

    _, y_fit, held, actual = smoke_data(task_type, binary)
    table = pd.read_csv(Path(directory) / "predictions.csv", dtype=str, keep_default_na=False)
    if table["id"].tolist() != [f"{i:05d}" for i in range(len(held))]:
        raise ValueError("Smoke prediction IDs do not match the frozen holdout")
    classes = tuple(json.loads((Path(directory) / "classes.json").read_text(encoding="utf-8")))
    if task_type == "classification":
        if table["actual"].tolist() != actual.tolist() or set(classes) != set(y_fit) or len(classes) != len(set(classes)):
            raise ValueError("Smoke class evidence does not match the fixture")
        probabilities = table[[f"probability_{i}" for i in range(len(classes))]].to_numpy(float)
        # Validate the matrix even when the selected metric consumes labels.
        from agent.metrics import probability_matrix
        probabilities = probability_matrix(probabilities, classes=classes, rows=len(held))
        prediction = (np.asarray(classes)[probabilities.argmax(axis=1)]
                      if get_metric(metric).prediction_kind == "labels" else probabilities)
        return score_predictions(metric, actual, prediction, classes=classes,
                                 prediction_classes=classes, positive_class="001" if binary else None)
    if classes or not np.array_equal(table["actual"].to_numpy(float), actual):
        raise ValueError("Smoke actual values do not match the fixture")
    return score_predictions(metric, actual, table["prediction"].to_numpy(float))


def run_smoke(profile, output, *, available_actions=ACTIONS, limit=8, binary=False,
              attempt_seconds=30.0, total_seconds=120.0):
    from agent.execution import run_python

    candidates = portfolio_candidates(profile, available_actions=available_actions, limit=limit)
    for value in (attempt_seconds, total_seconds):
        try:
            valid = type(value) in {int, float} and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Smoke time limits must be positive finite numbers")
    if type(binary) is not bool or (binary and profile["task_type"] != "classification"):
        raise ValueError("binary is supported only for classification")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "synthetic_execution_smoke", "seed": SEED, "profile": dict(profile),
              "binary": binary, "attempt_seconds": attempt_seconds, "total_seconds": total_seconds,
              "api_calls": 0, "candidates": candidates, "results": []}
    deadline = time.monotonic() + total_seconds
    for candidate in candidates:
        remaining = deadline - time.monotonic()
        item = {"id": candidate["id"], "status": "budget_skipped"}
        if remaining > 0:
            directory = output / candidate["id"]
            directory.mkdir()
            code = (f"import sys, json; sys.path.insert(0, {str(ROOT)!r})\n"
                    "from experiment_portfolio_recipes import _worker\n"
                    f"_worker(json.loads({json.dumps(candidate)!r}), {profile['task_type']!r}, "
                    f"{binary!r}, {str(directory)!r})\n")
            execution = run_python(code, directory, directory, 1,
                                   timeout=min(attempt_seconds, remaining), max_memory_bytes=8 * 1024**3)
            item.update(status="failed", execution=execution)
            if (execution.get("exit_code") == 0 and not execution.get("error")
                    and not execution.get("timed_out") and not execution.get("memory_limit_exceeded")):
                try:
                    item.update(score=score_saved(directory, profile["task_type"], profile["metric"], binary),
                                status="completed")
                except (OSError, ValueError, KeyError) as error:
                    item["error"] = str(error)
        report["results"].append(item)
        (output / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    if not candidates:
        (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", choices=("regression", "classification"), required=True)
    parser.add_argument("--metric", help="Default: RMSE for regression, LogLoss for classification")
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=list(ACTIONS))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--attempt-seconds", type=float, default=30)
    parser.add_argument("--total-seconds", type=float, default=120)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    try:
        report = run_smoke({"task_type": args.task_type,
                            "metric": args.metric or ("RMSE" if args.task_type == "regression" else "LogLoss")},
                           args.out_dir, available_actions=tuple(args.actions), limit=args.limit, binary=args.binary,
                           attempt_seconds=args.attempt_seconds, total_seconds=args.total_seconds)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for item in report["results"]:
        print(item["id"], item["status"], item.get("score", ""))
    return 0 if report["results"] and all(item["status"] == "completed" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
