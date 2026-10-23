"""Run an offline model recipe or bounded search on an isolated job's copied data."""

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
from threadpoolctl import threadpool_limits

from agent.diagnostics import classification_diagnostics, validate_classification_diagnostics
from agent.features import normalize_feature_config
from agent.models import build_pipeline, predict_for_metric
from agent.task_spec import load_task, load_task_data, score_holdout
from app.file_io import replace_file
from app.model_config import make_model_config, model_task_type, validate_model_config, winner_feature_evidence
from app.model_artifacts import save_model_artifacts
from app.splits import normalize_validation, split_rows
from offline_baseline import SEED, find_data
from validation import validate_predictions


ACTIONS = {"baseline", "extra_trees", "catboost", "lightgbm", "linear", "automl"}
OUTPUT_NAMES = ("stage.json", "result.json", "metrics.json", "holdout_predictions.csv", "automl_report.json",
                "model.joblib", "model_manifest.json", "training_recipe.json")


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
        replace_file(temporary, path, check=lambda: (_plain_path(temporary), _plain_path(path)))
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _plain_path(path)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        replace_file(temporary, path, check=lambda: (_plain_path(temporary), _plain_path(path)))
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
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {"validation", "model_params", "model_config", "time_budget",
                                       "feature_config", "training_device"}):
        raise ValueError("Некорректные настройки запуска.")
    if any(not isinstance(config[key], str) for key in required):
        raise ValueError("Настройки запуска должны содержать строковые значения.")
    if config["action"] not in ACTIONS:
        raise ValueError("Неизвестный рецепт обучения.")
    from app.training_devices import normalize_training_device
    config["training_device"] = normalize_training_device(config.get("training_device"), action=config["action"])
    if "model_params" in config and "model_config" in config:
        raise ValueError("Передайте model_params или model_config, но не оба поля.")
    if config["action"] == "automl":
        if "model_params" in config or "model_config" in config or "feature_config" in config:
            raise ValueError("Автоподбор сам выбирает параметры моделей.")
        requested = config.get("validation")
        config["validation"] = ({"strategy": "auto"} if requested is None or requested == {"strategy": "auto"}
                                else normalize_validation(requested))
        budget = config.get("time_budget", 600)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 10500:
            raise ValueError("Бюджет автоподбора должен быть числом от 0 до 10500 секунд.")
        config["time_budget"] = budget
    else:
        if "time_budget" in config:
            raise ValueError("Бюджет времени доступен только для автоподбора.")
        config["validation"] = normalize_validation(config.get("validation"))
        config["feature_config"] = normalize_feature_config(config.get("feature_config"))
    if config.get("model_params") is not None and not isinstance(config["model_params"], dict):
        raise ValueError("Настройки модели должны быть объектом.")
    if "model_config" in config:
        config["model_config"] = validate_model_config(config["model_config"], action=config["action"])
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


def _run_automl_job(config: dict) -> dict:
    from agent.automl import run_automl
    from app.model_config import validate_model_config
    from app.search_summary import validate_search_summary

    job_dir, task_path = config["job_dir"], config["task_path"]
    _check_inputs(task_path.parent, job_dir)

    def progress(event):
        if not isinstance(event, dict) or event.get("stage") not in {
                "training", "validating", "refitting", "finished"}:
            raise ValueError("Некорректный этап автоподбора.")
        stage = {"stage": event["stage"]}
        if "search" in event:
            stage["search"] = validate_search_summary(event["search"], allow_running=True)
        _atomic_json(job_dir / "stage.json", stage)

    evidence = run_automl(
        task_path, time_budget=config.get("time_budget", 600),
        validation=config.get("validation"),
        dataset_fingerprint=config["dataset_fingerprint"], progress=progress, output_dir=job_dir,
    )
    result = {key: evidence[key] for key in (
        "model", "model_config", "metric", "split", "comparison_key", "output_valid", "search")}
    result["model_config"] = validate_model_config(result["model_config"])
    result["search"] = validate_search_summary(result["search"], metric=result["metric"], model=result["model"])
    result.update(winner_feature_evidence(evidence, result["search"]))
    result["training_device"] = "cpu"
    result["model_artifact"] = {"available": False, "reason":
        "Автоподбор сохранил рецепт и предсказания. Для файла обученной модели запустите выбранный рецепт отдельно."}
    if "classification_diagnostics" in evidence:
        result["classification_diagnostics"] = validate_classification_diagnostics(
            evidence["classification_diagnostics"], metric=result["metric"],
            validation_rows=result["split"]["validation_rows"])
    output_check = validate_predictions(task_path.parent)
    if result["output_valid"] is not True or not output_check["valid"]:
        raise ValueError("Предсказания автоподбора не прошли проверку формата.")
    metrics = {
        **result, "task_type": evidence["task_type"], "action": "automl", "seed": SEED,
        "model": {"name": result["model"], "parameters": result["model_config"]["parameters"]},
        "metric": {**result["metric"], "source": "local_holdout_from_train.csv"},
        "dataset_fingerprint": config["dataset_fingerprint"],
        "elapsed_seconds": evidence["elapsed_seconds"], "api": {"used": False, "calls": 0},
        "evaluation_source": "app_local_holdout", "holdout_predictions_path": "holdout_predictions.csv",
        "output_check": output_check,
    }
    _atomic_json(job_dir / "metrics.json", metrics)
    _atomic_json(job_dir / "result.json", result)
    progress({"stage": "finished", "search": result["search"]})
    print(f"AutoML selected {result['model']}. API calls: 0.", flush=True)
    return result


def run_job(config: dict) -> dict:
    """Execute an already validated config and return the compact job result."""
    if config["action"] == "automl":
        return _run_automl_job(config)
    started = time.monotonic()
    job_dir, task_path = config["job_dir"], config["task_path"]
    task_dir = task_path.parent
    _check_inputs(task_dir, job_dir)
    spec = load_task(task_path)
    (train, test, columns, id_column, target, y, features,
     numeric, categorical, classification) = load_task_data(spec)
    task_type = "classification" if classification else "regression"
    if "model_config" in config:
        if "model_params" in config:
            raise ValueError("Передайте model_params или model_config, но не оба поля.")
        model_config = validate_model_config(config["model_config"], action=config["action"])
        if model_task_type(model_config) != task_type:
            raise ValueError("Модель не соответствует типу задачи.")
    else:
        model_config = make_model_config(config["action"], task_type, config.get("model_params"))
    x = train.loc[:, features]
    test_x = test.loc[:, features]
    # Every adapter owns its input preparation. Keeping raw declared columns
    # here makes the saved fitted pipeline reusable directly on raw CSV rows.
    feature_config = normalize_feature_config(config.get("feature_config"))
    validation = normalize_validation(config.get("validation"))
    if validation["strategy"] == "time" and validation["time_column"] not in numeric:
        raise ValueError("Столбец периода должен быть числовым признаком по словарю данных.")
    fit_rows, validation_rows, split = split_rows(train, y, features, classification, validation)
    metric_name = spec.metric.name
    validation_hash = _json_hash({
        "id_column": id_column, "target_column": target,
        "ids": train[id_column].iloc[validation_rows].tolist(),
        "actuals": train[target].iloc[validation_rows].tolist(),
    })
    split["validation_hash"] = validation_hash
    # This comparison identity deliberately excludes the recipe/model.
    comparison_key = _json_hash({
        "dataset_fingerprint": config["dataset_fingerprint"],
        "metric": metric_name, "higher_is_better": spec.metric.higher_is_better, "split": split,
    })
    pipeline = build_pipeline(config["action"], spec.metric.task_type, numeric, categorical,
                              model_config=model_config, feature_config=feature_config)
    from app.training_devices import prepare_training_pipeline, confirm_training_device
    training_device = config.get("training_device", "cpu")
    pipeline, device_evidence = prepare_training_pipeline(
        pipeline, action=config["action"], training_device=training_device)

    _atomic_json(job_dir / "stage.json", {"stage": "training"})
    fit_started = time.monotonic()
    with threadpool_limits(limits=8):
        pipeline.fit(x.iloc[fit_rows], y.iloc[fit_rows])
    fit_seconds = time.monotonic() - fit_started
    _atomic_json(job_dir / "stage.json", {"stage": "validating"})
    validation_started = time.monotonic()
    with threadpool_limits(limits=8):
        holdout_predictions = predict_for_metric(pipeline, x.iloc[validation_rows], spec)
    held_ids = train[id_column].iloc[validation_rows].tolist()
    held_actuals = train[target].iloc[validation_rows].tolist()
    _atomic_csv(job_dir / "holdout_predictions.csv", spec.holdout_frame(
        held_ids, held_actuals, holdout_predictions,
        prediction_classes=getattr(pipeline, "classes_", None)))
    value = score_holdout(job_dir / "holdout_predictions.csv", spec, held_ids, held_actuals)
    validation_seconds = time.monotonic() - validation_started
    print(f"Local validation {metric_name}: {value:.6f}", flush=True)

    _atomic_json(job_dir / "stage.json", {"stage": "refitting"})
    refit_started = time.monotonic()
    final_pipeline = clone(pipeline)
    with threadpool_limits(limits=8):
        final_pipeline.fit(x, y)
    refit_seconds = time.monotonic() - refit_started
    device_evidence = confirm_training_device(final_pipeline, device_evidence)
    prediction_started = time.monotonic()
    with threadpool_limits(limits=8):
        predictions = predict_for_metric(final_pipeline, test_x, spec)
    prediction_seconds = time.monotonic() - prediction_started
    submission = spec.prediction_frame(test[id_column], predictions,
                                       prediction_classes=getattr(final_pipeline, "classes_", None))
    _atomic_csv(task_dir / "submission.csv", submission)
    output_check = validate_predictions(task_dir)
    if not output_check["valid"]:
        raise ValueError("Проверка предсказаний не пройдена: " + "; ".join(output_check["errors"]))

    _atomic_json(job_dir / "stage.json", {"stage": "saving_model"})
    export_started = time.monotonic()
    recipe = {"version": 1, "action": config["action"], "task_type": task_type,
              "model_config": model_config, "feature_config": feature_config,
              "validation": validation, "training_device": training_device}
    schema = {"features": list(features), "numeric": list(numeric), "categorical": list(categorical),
              "id_column": id_column, "target_column": target,
              "classes": [str(value) for value in getattr(final_pipeline, "classes_", [])]}
    try:
        model_artifact = save_model_artifacts(
            job_dir, final_pipeline, recipe=recipe, schema=schema, train_rows=len(train), split=split,
            dataset_fingerprint=config["dataset_fingerprint"], comparison_key=comparison_key)
    except Exception as error:
        # Prediction quality is already verified. A serialization/disk failure
        # must be visible without discarding the valid completed experiment.
        model_artifact = {"available": False, "reason":
            "Обучение завершено, но файл модели сохранить не удалось (" + type(error).__name__ + ")."}
        print("Trained model export unavailable: " + type(error).__name__, flush=True)
    export_seconds = time.monotonic() - export_started
    elapsed = time.monotonic() - started
    metric = {"name": metric_name, "value": float(value), "higher_is_better": spec.metric.higher_is_better}
    model_name = model_config["model"]
    fitted_model = pipeline.named_steps["model"]
    if model_config.get("target_transform") == "log1p":
        # The wrapper's get_params includes an estimator object, which cannot
        # be JSON evidence. Record the fitted underlying model's parameters;
        # the complete recipe above retains the explicit target transform.
        fitted_model = fitted_model.regressor_
    metrics = {
        "task_type": "classification" if classification else "regression",
        "seed": SEED, "split": split,
        "model": {"name": model_name, "parameters": fitted_model.get_params()},
        "model_config": model_config,
        "feature_config": feature_config,
        "model_artifact": model_artifact,
        "training_device": training_device,
        "training_device_evidence": device_evidence,
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
                           "model_export": export_seconds,
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
        "model_config": model_config,
        "feature_config": feature_config,
        "model_artifact": model_artifact,
        "training_device": training_device,
        "training_device_evidence": device_evidence,
        "comparison_key": comparison_key, "output_valid": True,
    }
    if metric_name == "Macro F1":
        diagnostics = classification_diagnostics(y.iloc[validation_rows].tolist(), holdout_predictions.tolist())
        if diagnostics is not None:
            diagnostics = validate_classification_diagnostics(
                diagnostics, metric=metric, validation_rows=len(validation_rows))
            result["classification_diagnostics"] = diagnostics
            metrics["classification_diagnostics"] = diagnostics
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
