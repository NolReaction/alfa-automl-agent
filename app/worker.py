"""Run one fixed offline modelling recipe on an isolated job's copied data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set these before importing numerical libraries; recipes also cap n_jobs at 8.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from offline_baseline import (
    SEED, VALIDATION_FRACTION, find_data, load_data, make_pipeline, prepare_features,
)
from validation import validate_predictions


ACTIONS = {"baseline", "extra_trees"}
OUTPUT_NAMES = ("stage.json", "result.json", "metrics.json", "holdout_predictions.csv")


def _plain_path(path: Path, root: Path | None = None, *, directory=False,
                required=False) -> Path:
    """Reject links before resolving, including Windows directory junctions."""
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Символические ссылки и junction-папки не поддерживаются.")
    if root is not None and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Файлы задания должны находиться внутри папки запуска.")
    if required and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Не найден файл или каталог: {path.name}")
    return path


def _json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    _plain_path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        _plain_path(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _plain_path(path)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        _plain_path(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(config_path: Path) -> dict:
    config_path = _plain_path(Path(config_path), required=True)
    with config_path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Файл настроек запуска слишком большой.")
    config = json.loads(raw.decode("utf-8-sig"))
    required = {"job_dir", "task_path", "action", "dataset_fingerprint"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Некорректные настройки запуска.")
    if any(not isinstance(config[key], str) for key in required):
        raise ValueError("Настройки запуска должны содержать строковые значения.")
    if config["action"] not in ACTIONS:
        raise ValueError("Неизвестный рецепт обучения.")
    if not re.fullmatch(r"[0-9a-f]{64}", config["dataset_fingerprint"]):
        raise ValueError("Некорректный отпечаток набора данных.")
    job_dir = _plain_path(Path(config["job_dir"]), directory=True, required=True)
    if config_path.parent.resolve() != job_dir.resolve():
        raise ValueError("Настройки должны находиться в папке запуска.")
    task_path = _plain_path(Path(config["task_path"]), job_dir, required=True)
    if task_path.suffix.lower() != ".md":
        raise ValueError("Необходимо передать Markdown-описание задачи.")
    if task_path.parent.resolve() == job_dir.resolve():
        raise ValueError("Копия задачи должна находиться в отдельной подпапке запуска.")
    for name in OUTPUT_NAMES:
        _plain_path(job_dir / name, job_dir)
    _plain_path(task_path.parent / "submission.csv", job_dir)
    config["job_dir"] = job_dir
    config["task_path"] = task_path
    return config


def _check_inputs(task_dir: Path, job_dir: Path) -> Path:
    # Check the small staged tree before find_data/validation recurse into it.
    count = 0
    for parent, directories, files in os.walk(task_dir, followlinks=False):
        for name in [*directories, *files]:
            count += 1
            if count > 10000:
                raise ValueError("В копии задачи слишком много файлов.")
            _plain_path(Path(parent) / name, job_dir)
    data_dir = find_data(task_dir)
    for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        path = _plain_path(data_dir / name, job_dir, required=True)
        if path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError(f"Файл {name} превышает лимит 512 МБ.")
    return data_dir


def run_job(config: dict) -> dict:
    """Execute an already validated config and return the compact job result."""
    started = time.monotonic()
    job_dir, task_path = config["job_dir"], config["task_path"]
    task_dir = task_path.parent
    data_dir = _check_inputs(task_dir, job_dir)
    (train, test, columns, id_column, target, y, features,
     numeric, categorical, classification) = load_data(data_dir)
    x = prepare_features(train, features, numeric, categorical)
    test_x = prepare_features(test, features, numeric, categorical)
    fit_rows, validation_rows = train_test_split(
        np.arange(len(train)), test_size=VALIDATION_FRACTION, random_state=SEED,
        shuffle=True, stratify=y if classification else None,
    )
    metric_name = "Macro F1" if classification else "RMSE"
    validation_hash = _json_hash({
        "id_column": id_column, "target_column": target,
        "ids": train[id_column].iloc[validation_rows].tolist(),
        "actuals": train[target].iloc[validation_rows].tolist(),
    })
    split = {
        "seed": SEED, "train_fraction": 1 - VALIDATION_FRACTION,
        "validation_fraction": VALIDATION_FRACTION, "shuffle": True,
        "stratified": classification, "train_rows": len(fit_rows),
        "validation_rows": len(validation_rows), "validation_hash": validation_hash,
    }
    # This comparison identity deliberately excludes the recipe/model.
    comparison_key = _json_hash({
        "dataset_fingerprint": config["dataset_fingerprint"],
        "metric": metric_name, "higher_is_better": classification, "split": split,
    })
    pipeline = make_pipeline(numeric, categorical, classification)
    if config["action"] == "extra_trees":
        model = (ExtraTreesClassifier(
            n_estimators=160, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=8, random_state=SEED,
        ) if classification else ExtraTreesRegressor(
            n_estimators=160, min_samples_leaf=2, n_jobs=8, random_state=SEED,
        ))
        pipeline.set_params(model=model)

    _atomic_json(job_dir / "stage.json", {"stage": "training"})
    fit_started = time.monotonic()
    with threadpool_limits(limits=8):
        pipeline.fit(x.iloc[fit_rows], y.iloc[fit_rows])
    fit_seconds = time.monotonic() - fit_started
    _atomic_json(job_dir / "stage.json", {"stage": "validating"})
    validation_started = time.monotonic()
    with threadpool_limits(limits=8):
        holdout_predictions = pipeline.predict(x.iloc[validation_rows])
    value = (f1_score(y.iloc[validation_rows], holdout_predictions,
                      average="macro", zero_division=0) if classification else
             np.sqrt(mean_squared_error(y.iloc[validation_rows], holdout_predictions)))
    if not math.isfinite(float(value)):
        raise ValueError("Получена неконечная метрика валидации.")
    validation_seconds = time.monotonic() - validation_started
    _atomic_csv(job_dir / "holdout_predictions.csv", pd.DataFrame({
        "id": train[id_column].iloc[validation_rows].to_numpy(),
        "actual": train[target].iloc[validation_rows].to_numpy(),
        "prediction": holdout_predictions,
    }))
    print(f"Local validation {metric_name}: {value:.6f}", flush=True)

    _atomic_json(job_dir / "stage.json", {"stage": "refitting"})
    refit_started = time.monotonic()
    final_pipeline = clone(pipeline)
    with threadpool_limits(limits=8):
        final_pipeline.fit(x, y)
    refit_seconds = time.monotonic() - refit_started
    prediction_started = time.monotonic()
    with threadpool_limits(limits=8):
        predictions = final_pipeline.predict(test_x)
    prediction_seconds = time.monotonic() - prediction_started
    submission = pd.DataFrame({id_column: test[id_column], target: predictions}).loc[:, columns]
    _atomic_csv(task_dir / "submission.csv", submission)
    output_check = validate_predictions(task_dir)
    if not output_check["valid"]:
        raise ValueError("Проверка предсказаний не пройдена: " + "; ".join(output_check["errors"]))

    elapsed = time.monotonic() - started
    metric = {"name": metric_name, "value": float(value), "higher_is_better": classification}
    model_name = type(pipeline.named_steps["model"]).__name__
    metrics = {
        "task_type": "classification" if classification else "regression",
        "seed": SEED, "split": split,
        "model": {"name": model_name, "parameters": pipeline.named_steps["model"].get_params()},
        "metric": {**metric, "source": "local_holdout_from_train.csv"},
        "shapes": {"train": list(train.shape), "test": list(test.shape),
                   "training_features": [len(fit_rows), len(features)],
                   "validation_features": [len(validation_rows), len(features)],
                   "submission": list(submission.shape)},
        "id_column": id_column, "target_column": target,
        "features": {"numeric": numeric, "categorical": categorical},
        "api": {"used": False, "calls": 0},
        "timing_seconds": {"training": fit_seconds, "validation": validation_seconds,
                           "refit": refit_seconds, "prediction": prediction_seconds,
                           "refit_and_predict": refit_seconds + prediction_seconds,
                           "total_before_save": elapsed},
        "elapsed_seconds": elapsed,
        "training_elapsed_seconds": fit_seconds + refit_seconds,
        "action": config["action"],
        "data_fingerprint": config["dataset_fingerprint"],
        "dataset_fingerprint": config["dataset_fingerprint"],
        "validation_signature": comparison_key,
        "comparison_key": comparison_key,
        "evaluation_source": "app_local_holdout",
        "holdout_predictions_path": "holdout_predictions.csv",
        "output_check": output_check,
    }
    result = {
        "model": model_name, "metric": metric, "split": split,
        "comparison_key": comparison_key, "output_valid": True,
    }
    _atomic_json(job_dir / "metrics.json", metrics)
    _atomic_json(job_dir / "result.json", result)
    _atomic_json(job_dir / "stage.json", {"stage": "finished"})
    print(f"Saved staged submission. Elapsed: {elapsed:.2f}s. API calls: 0.", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = None
    try:
        config = load_config(args.config)
        run_job(config)
        return 0
    except Exception as error:
        message = str(error)
        # Never load credentials; redact an inherited key if a library includes it.
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            message = message.replace(key, "[REDACTED]")
        if config is not None:
            try:
                _atomic_json(config["job_dir"] / "result.json",
                             {"output_valid": False, "error": message})
            except (OSError, ValueError):
                pass
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
