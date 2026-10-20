"""One isolated, allowlisted offline candidate; the parent verifies its evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from agent.model_config import validate_model_config
from offline_baseline import SEED, load_data, make_pipeline, prepare_features


def run_candidate(config_path: Path) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    candidate_dir = Path(config["candidate_dir"])
    (train, test, columns, id_column, target, y, features,
     numeric, categorical, classification) = load_data(candidate_dir / "inputs")
    model_config = validate_model_config(config["model_config"], action=config["action"])
    pipeline = make_pipeline(numeric, categorical, classification)
    if config["action"] == "extra_trees":
        constructor = ExtraTreesClassifier if classification else ExtraTreesRegressor
        pipeline.set_params(model=constructor(random_state=SEED, n_jobs=8))
    pipeline.named_steps["model"].set_params(**model_config["parameters"])
    x = prepare_features(train, features, numeric, categorical)
    test_x = prepare_features(test, features, numeric, categorical)
    fit_rows, held_rows = config["fit_rows"], config["validation_rows"]
    started = time.monotonic()
    print(f"Training {model_config['model']} on {len(fit_rows)} rows...", flush=True)
    with threadpool_limits(limits=8):
        pipeline.fit(x.iloc[fit_rows], y.iloc[fit_rows])
        predictions = pipeline.predict(x.iloc[held_rows])
    pd.DataFrame({
        "id": train[id_column].iloc[held_rows].to_numpy(),
        "actual": train[target].iloc[held_rows].to_numpy(),
        "prediction": predictions,
    }).to_csv(candidate_dir / "holdout_predictions.csv", index=False, encoding="utf-8")
    print(f"Refitting on all {len(train)} training rows...", flush=True)
    final_pipeline = clone(pipeline)
    with threadpool_limits(limits=8):
        final_pipeline.fit(x, y)
        predictions = final_pipeline.predict(test_x)
    pd.DataFrame({id_column: test[id_column], target: predictions}).loc[:, columns].to_csv(
        candidate_dir / "submission.csv", index=False, encoding="utf-8",
    )
    print(f"Candidate finished in {time.monotonic() - started:.2f}s. API calls: 0.", flush=True)
