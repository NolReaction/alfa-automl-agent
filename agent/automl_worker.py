"""One isolated allowlisted phase; validation trials never refit the full data."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import numpy as np
from threadpoolctl import threadpool_limits

from agent.models import build_pipeline
from agent.search import align_predictions, fixed_blend, metric_predictions
from agent.task_spec import TaskSpec, load_task_data


def run_candidate(config_path: Path) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    directory = Path(config["candidate_dir"])
    spec = TaskSpec.from_dict(config["task_spec"])
    train, test, _, id_column, target, y, features, numeric, categorical, _ = load_task_data(spec)
    # Feature preparation is part of the pipeline and is fitted on each training
    # fold only. Do not learn vocabularies, imputations or statistics here.
    x, test_x = train.loc[:, features], test.loc[:, features]
    phase = config["phase"]
    if phase not in {"seed", "selection", "verification", "refit"}:
        raise ValueError("Unknown AutoML worker phase")
    started = time.monotonic()
    durations = {"fit_seconds": 0.0, "predict_seconds": 0.0, "total_seconds": 0.0}

    def fit_predict(recipe, fit_x, fit_y, predict_x):
        pipeline = build_pipeline(recipe["action"], spec.metric.task_type, numeric, categorical,
                                  recipe["model_config"], recipe.get("feature_config"))
        before = time.monotonic()
        pipeline.fit(fit_x, fit_y)
        durations["fit_seconds"] += time.monotonic() - before
        before = time.monotonic()
        if spec.classification:
            raw = align_predictions(spec, pipeline.predict_proba(predict_x), tuple(map(str, pipeline.classes_)))
        else:
            raw = align_predictions(spec, pipeline.predict(predict_x))
        durations["predict_seconds"] += time.monotonic() - before
        return raw

    with threadpool_limits(limits=8):
        if phase in {"seed", "selection", "verification"}:
            fitting, held = config["fit_rows"], config["validation_rows"]
            print(f"{phase}: fitting {config['model_config']['model']} on {len(fitting)} rows", flush=True)
            raw = fit_predict(config, x.iloc[fitting], y.iloc[fitting], x.iloc[held])
            ids, actual = train[id_column].iloc[held].tolist(), train[target].iloc[held].tolist()
            spec.holdout_frame(ids, actual, metric_predictions(spec, raw), prediction_classes=spec.classes).to_csv(
                directory / "holdout_predictions.csv", index=False, encoding="utf-8")
            np.savez_compressed(directory / "holdout_raw.npz", ids=np.asarray(ids, dtype=str),
                                actual=np.asarray(actual, dtype=str), predictions=raw,
                                classes=np.asarray(spec.classes, dtype=str))
        if phase in {"seed", "refit"}:
            print(f"{phase}: refitting on all {len(train)} rows", flush=True)
            components = config.get("components")
            if components is not None:
                if phase != "refit" or len(components) != 2 or config["model_config"]["parameters"]["weight"] != 0.5:
                    raise ValueError("Only the fixed two-component 50/50 blend is supported")
                left = fit_predict(components[0], x, y, test_x)
                right = fit_predict(components[1], x, y, test_x)
                raw = fixed_blend(spec, left, right, left_classes=spec.classes, right_classes=spec.classes)
            else:
                raw = fit_predict(config, x, y, test_x)
            spec.prediction_frame(test[id_column].tolist(), metric_predictions(spec, raw),
                                  prediction_classes=spec.classes).to_csv(
                directory / "submission.csv", index=False, encoding="utf-8")
    durations["total_seconds"] = time.monotonic() - started
    (directory / "durations.json").write_text(json.dumps(durations, allow_nan=False), encoding="utf-8")
    print(f"{phase} completed in {durations['total_seconds']:.2f}s; API calls: 0", flush=True)
