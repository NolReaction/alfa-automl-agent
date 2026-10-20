"""Compare a predeclared portfolio on several frozen public-data holdouts.

This reports validation measurements only and never publishes submission.csv.
The output directory must be new; source datasets and prior evidence are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from agent.automl import _atomic_json, data_fingerprint
from agent.metrics import score_predictions
from agent.model_config import make_model_config
from agent.models import build_pipeline, predict_for_metric
from agent.task_spec import load_task, load_task_data
from offline_baseline import prepare_features


def validation_folds(spec, train, y, *, time_column=None):
    if spec.classification:
        groups = pd.util.hash_pandas_object(train.loc[:, list(spec.features)], index=False).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for number, (fit, held) in enumerate(splitter.split(train, y, groups)):
            if set(y.iloc[fit]) != set(spec.classes) or set(y.iloc[held]) != set(spec.classes):
                raise ValueError(f"Fold {number} does not contain every class in both parts")
            if set(groups[fit]) & set(groups[held]):
                raise ValueError("Duplicate-feature groups overlap")
            yield f"grouped_fold_{number}", fit, held
    else:
        if time_column not in spec.features:
            raise ValueError("Regression benchmark needs an explicit numeric --time-column")
        periods = pd.to_numeric(train[time_column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(periods).all():
            raise ValueError("Time periods must be finite")
        values = sorted(set(periods))
        if len(values) < 4:
            raise ValueError("At least four periods are needed for three rolling holdouts")
        for period in values[-3:]:
            yield f"period_{period:g}", np.flatnonzero(periods < period), np.flatnonzero(periods == period)


def benchmark(task, output, *, time_column=None, actions=None):
    spec = load_task(task)
    output = Path(output).resolve()
    # A new directory also prevents accidentally overwriting any existing input.
    output.mkdir(parents=True, exist_ok=False)
    train, _, _, _, _, y, features, numeric, categorical, classification = load_task_data(spec)
    X = prepare_features(train, features, numeric, categorical)
    actions = list(actions or ("baseline", "extra_trees", "catboost", "lightgbm", "linear"))
    report = {"version": 1, "task": str(spec.task_path), "task_spec": spec.to_dict(),
              "dataset_fingerprint": data_fingerprint(spec.data_dir), "seed": 42,
              "metric": spec.metric.to_dict(), "actions": actions,
              "purpose": "predeclared_cross_fold_comparison_not_platform_score", "trials": []}
    for name, fit, held in validation_folds(spec, train, y, time_column=time_column):
        actual = train[spec.target_column].iloc[held].tolist()
        ids = train[spec.id_column].iloc[held].tolist()
        digest = hashlib.sha256(json.dumps([fit.tolist(), held.tolist()], separators=(",", ":")).encode()).hexdigest()
        for action in actions:
            config = make_model_config(action, spec.metric.task_type)
            if action == "extra_trees" and not classification:
                config = make_model_config(action, "regression", {"n_estimators": 200, "min_samples_leaf": 1})
            row = {"fold": name, "split_hash": digest, "fit_rows": len(fit), "validation_rows": len(held),
                   "action": action, "model_config": config, "status": "running"}
            report["trials"].append(row)
            _atomic_json(output / "report.json", report)
            started = time.monotonic()
            try:
                pipeline = build_pipeline(action, spec.metric.task_type, numeric, categorical, config)
                with threadpool_limits(limits=8):
                    pipeline.fit(X.iloc[fit], y.iloc[fit])
                    predictions = predict_for_metric(pipeline, X.iloc[held], spec)
                classes = getattr(pipeline, "classes_", None)
                value = score_predictions(spec.metric, actual, predictions, classes=spec.classes,
                                          prediction_classes=classes, positive_class=spec.positive_class)
                spec.holdout_frame(ids, actual, predictions, prediction_classes=classes).to_csv(
                    output / f"{name}_{action}.csv", index=False)
                row.update(status="completed", value=value)
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _atomic_json(output / "report.json", report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if data_fingerprint(spec.data_dir) != report["dataset_fingerprint"]:
        raise ValueError("Source inputs changed during the benchmark")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-column")
    parser.add_argument("--actions", nargs="+", choices=("baseline", "extra_trees", "catboost", "lightgbm", "linear"))
    args = parser.parse_args()
    report = benchmark(args.task, args.output, time_column=args.time_column, actions=args.actions)
    return int(any(row["status"] != "completed" for row in report["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
