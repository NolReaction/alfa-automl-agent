"""Bounded offline search for the documented RMSE/Macro F1 CSV task format.

Every recipe uses one fixed holdout. The parent recomputes the score from held
predictions and retains only a fully refitted, structurally valid winner.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from uuid import uuid4

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from agent.diagnostics import classification_diagnostics, validate_classification_diagnostics
from agent.execution import run_python
from agent.file_io import replace_file
from agent.model_config import make_model_config
from agent.splits import normalize_validation, split_rows
from agent.task_contract import validate_task_metric
from offline_baseline import find_data, load_data, read_csv
from validation import validate_predictions

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")


def _plain(path: Path) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Ссылки и junction-папки не поддерживаются в автоподборе.")
    return path


def _atomic_json(path: Path, value: dict) -> None:
    _plain(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        replace_file(temporary, path, check=lambda: (_plain(temporary), _plain(path)))
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    _plain(source)
    _plain(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        replace_file(temporary, destination, check=lambda: (_plain(temporary), _plain(destination)))
    finally:
        temporary.unlink(missing_ok=True)


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def data_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in INPUT_NAMES:
        digest.update(name.encode("utf-8") + b"\0")
        with (data_dir / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def candidate_plan(classification: bool) -> list[dict]:
    task_type = "classification" if classification else "regression"
    variants = (
        ("baseline", None, "Стартовая модель"),
        ("extra_trees", None, "ExtraTrees"),
        ("baseline", {"n_estimators": 160, "max_depth": 24, "min_samples_leaf": 1}
         if classification else {"max_iter": 250, "max_leaf_nodes": 63, "l2_regularization": 1.0},
         "Стартовая модель · больше ёмкость"),
        ("extra_trees", {"n_estimators": 200, "min_samples_leaf": 1}, "ExtraTrees · лист из 1 строки"),
    )
    return [{"id": f"candidate_{index}", "action": action, "label": label,
             "model_config": make_model_config(action, task_type, parameters)}
            for index, (action, parameters, label) in enumerate(variants, start=1)]


def choose_validation(train, test, features, numeric, classification, validation=None):
    if validation is not None and validation != {"strategy": "auto"}:
        chosen = normalize_validation(validation)
        if chosen["strategy"] == "time" and chosen["time_column"] not in numeric:
            raise ValueError("Столбец периода должен быть числовым признаком по словарю данных.")
        return chosen, "Схема проверки выбрана пользователем; все модели проверяются на одинаковых строках."
    if classification:
        if train.loc[:, features].duplicated().any():
            return {"strategy": "grouped"}, (
                "Найдены одинаковые строки признаков: группы дублей разделены между обучением и проверкой."
            )
        return {"strategy": "random"}, "Дубли признаков не найдены: выбран стратифицированный holdout 80/20."
    candidates = []
    for name in numeric:
        # Semantic hint plus actual distribution evidence. Avoid treating an
        # arbitrary continuous feature or row identifier as a time axis.
        if not re.search(r"year|period|time|год|период", name, re.IGNORECASE):
            continue
        periods = pd.to_numeric(train[name], errors="coerce")
        future = pd.to_numeric(test[name], errors="coerce")
        if (np.isfinite(periods.to_numpy(dtype=float)).all()
                and np.isfinite(future.to_numpy(dtype=float)).all()
                and 2 <= periods.nunique() <= 100 and future.min() > periods.max()):
            candidates.append(name)
    if len(candidates) > 1:
        raise ValueError("Несколько возможных столбцов периода: укажите time_column явно: "
                         + ", ".join(candidates))
    if candidates:
        return {"strategy": "time", "time_column": candidates[0]}, (
            f"В test более поздние значения {candidates[0]}: последний период train оставлен для проверки."
        )
    return {"strategy": "random"}, (
        "Однозначный числовой период с будущими значениями test не найден: выбран holdout 80/20; "
        "проверьте, соответствует ли он задаче."
    )


def _score_holdout(path: Path, expected_ids, expected_actuals, classification: bool,
                   allowed_classes=None) -> float:
    frame = read_csv(path)
    if list(frame.columns) != ["id", "actual", "prediction"]:
        raise ValueError("Некорректные столбцы проверочных предсказаний.")
    if frame["id"].tolist() != list(expected_ids) or frame["actual"].tolist() != list(expected_actuals):
        raise ValueError("Проверочные ID или ответы не совпадают с закреплённой выборкой.")
    if classification:
        if frame["prediction"].str.strip().eq("").any() or not set(frame["prediction"]) <= set(allowed_classes):
            raise ValueError("Неизвестные или пустые классы в проверочных предсказаниях.")
        value = f1_score(frame["actual"], frame["prediction"], average="macro", zero_division=0)
    else:
        actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError("Проверочные предсказания должны быть конечными числами.")
        value = np.sqrt(mean_squared_error(actual, predicted))
    if not math.isfinite(float(value)):
        raise ValueError("Получена неконечная метрика.")
    return float(value)


def _stage_inputs(data_dir: Path, destination: Path) -> None:
    destination.mkdir()
    for name in INPUT_NAMES:
        source = _plain(data_dir / name)
        try:
            os.link(source, destination / name)
        except OSError:
            shutil.copyfile(source, destination / name)


def _snapshot_inputs(data_dir: Path, destination: Path) -> str:
    """Copy once: every candidate must train on the same immutable input bytes."""
    before = data_fingerprint(data_dir)
    destination.mkdir()
    for name in INPUT_NAMES:
        shutil.copyfile(_plain(data_dir / name), destination / name)
    if data_fingerprint(destination) != before or data_fingerprint(data_dir) != before:
        raise ValueError("Исходные данные изменились при подготовке автоподбора. Повторите запуск.")
    return before


def run_automl(task_path: Path, *, time_budget=600, validation=None,
               dataset_fingerprint=None, progress=None, output_dir=None) -> dict:
    started = time.monotonic()
    if (type(time_budget) not in {int, float} or not math.isfinite(time_budget)
            or not 0 < time_budget <= 10500):
        raise ValueError("Время автоподбора должно быть в диапазоне (0, 10500] секунд.")
    deadline = started + time_budget
    task_path = _plain(Path(task_path))
    if not task_path.is_file() or task_path.suffix.lower() != ".md":
        raise ValueError("Передайте существующий task.md.")
    task_dir = task_path.parent
    data_dir = find_data(task_dir)
    for name in INPUT_NAMES:
        _plain(data_dir / name)
    _plain(task_dir / "submission.csv")
    task_text = task_path.read_text(encoding="utf-8-sig")
    run_dir = _plain(task_dir / ".agent_runs" / uuid4().hex)
    run_dir.mkdir(parents=True)
    snapshot_dir = run_dir / "input_snapshot"
    input_fingerprint = _snapshot_inputs(data_dir, snapshot_dir)
    (train, test, columns, id_column, target, y, features,
     numeric, categorical, classification) = load_data(snapshot_dir)
    validate_task_metric(task_text, classification)
    choice, reason = choose_validation(train, test, features, numeric, classification, validation)
    fit_rows, held_rows, split = split_rows(train, y, features, classification, choice)
    expected_ids = train[id_column].iloc[held_rows].tolist()
    expected_actuals = train[target].iloc[held_rows].tolist()
    split["validation_hash"] = _json_hash({"id_column": id_column, "target_column": target,
                                           "ids": expected_ids, "actuals": expected_actuals})
    metric_name = "Macro F1" if classification else "RMSE"
    if dataset_fingerprint is None:
        dataset_fingerprint = input_fingerprint
    if not isinstance(dataset_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint):
        raise ValueError("Некорректный отпечаток набора данных.")
    comparison_key = _json_hash({"dataset_fingerprint": dataset_fingerprint, "metric": metric_name,
                                "higher_is_better": classification, "split": split})
    output_dir = _plain(Path(output_dir)) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("automl_report.json", "holdout_predictions.csv"):
        _plain(output_dir / name)
    plan = candidate_plan(classification)
    search = {"status": "running", "planned": len(plan), "completed": 0, "best_candidate": None,
              "validation_reason": reason, "candidates": [
                  {"id": item["id"], "label": item["label"], "model": item["model_config"]["model"],
                   "status": "skipped", "metric": None, "selected": False,
                   "elapsed_seconds": 0.0, "error": None} for item in plan]}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": "offline",
              "task_type": "classification" if classification else "regression",
              "dataset_fingerprint": dataset_fingerprint, "input_fingerprint": input_fingerprint, "split": split,
              "comparison_key": comparison_key, "api": {"used": False, "calls": 0},
              "time_budget_seconds": time_budget, "search": search,
              "candidate_configs": {item["id"]: item["model_config"] for item in plan},
              "report_path": str(output_dir / "automl_report.json")}
    best = None

    def save(stage):
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(output_dir / "automl_report.json", report)
        if progress:
            progress({"stage": stage, "search": deepcopy(search)})

    print(f"Offline AutoML: {len(plan)} candidates; {metric_name}; {choice['strategy']} validation.", flush=True)
    print(reason, flush=True)
    save("training")
    try:
        for index, (spec, row) in enumerate(zip(plan, search["candidates"]), start=1):
            remaining = deadline - time.monotonic() - 2
            if remaining <= 1:
                search["status"] = "time_limit"
                break
            # Reserve a fair share for the remaining recipes, including their
            # full-data refit. Killing one slow candidate cannot consume the run.
            timeout = min(1800.0, remaining / (len(plan) - index + 1))
            if timeout <= 1:
                search["status"] = "time_limit"
                break
            candidate_started = time.monotonic()
            row["status"] = "running"
            try:
                candidate_dir = run_dir / spec["id"]
                candidate_dir.mkdir()
                _stage_inputs(snapshot_dir, candidate_dir / "inputs")
                config_path = candidate_dir / "config.json"
                _atomic_json(config_path, {"candidate_dir": str(candidate_dir), "action": spec["action"],
                                          "model_config": spec["model_config"], "fit_rows": fit_rows.tolist(),
                                          "validation_rows": held_rows.tolist()})
                save("training")
                code = ("import sys\n" f"sys.path.insert(0, {str(ROOT)!r})\n"
                        "from pathlib import Path\nfrom agent.automl_worker import run_candidate\n"
                        f"run_candidate(Path({str(config_path)!r}))\n")
                remaining = deadline - time.monotonic() - 2
                timeout = min(1800.0, remaining / (len(plan) - index + 1))
                if timeout <= 1:
                    row["status"] = "skipped"
                    # A terminal successful summary must already have a winner;
                    # without one the failure is recorded below after the loop.
                    search["status"] = "time_limit" if best is not None else "running"
                    break
                print(f"[{index}/{len(plan)}] {spec['label']} (limit {timeout:.0f}s)", flush=True)
                execution = run_python(code, candidate_dir, candidate_dir / "execution", index, timeout=timeout)
                if execution.get("timed_out"):
                    row.update(status="timed_out", error="Превышен лимит времени попытки.")
                    continue
                if execution.get("exit_code") != 0:
                    raise ValueError((execution.get("stderr") or execution.get("error") or
                                      "Процесс обучения завершился с ошибкой.")[-2000:])
                save("validating")
                value = _score_holdout(candidate_dir / "holdout_predictions.csv", expected_ids,
                                       expected_actuals, classification, y.unique())
                check = validate_predictions(candidate_dir)
                if not check["valid"]:
                    raise ValueError("Проверка CSV: " + "; ".join(check["errors"]))
                row.update(status="completed", metric=value)
                search["completed"] += 1
                better = best is None or (value > best["value"] if classification else value < best["value"])
                if better:
                    # Keep immutable candidate evidence until final publication;
                    # later attempts operate in other directories.
                    best = {"value": value, "spec": spec, "directory": candidate_dir}
                    search["best_candidate"] = spec["id"]
                    for item in search["candidates"]:
                        item["selected"] = item["id"] == spec["id"]
                print(f"  {metric_name}: {value:.6f}" + (" · best" if better else " · retained previous best"), flush=True)
            except (OSError, ValueError, RuntimeError) as error:
                row.update(status="failed", error=str(error)[-2000:])
                print(f"  Candidate failed: {row['error']}", flush=True)
            finally:
                row["elapsed_seconds"] = round(time.monotonic() - candidate_started, 3)
                save("training")
        if best is None:
            search["status"] = "failed"
            raise RuntimeError("Автоподбор не получил ни одного проверенного результата. "
                               "Прежний submission.csv сохранён; откройте журнал попыток.")
        if search["status"] == "running":
            search["status"] = "time_limit" if any(row["status"] == "timed_out" for row in search["candidates"]) else "completed"
        winning_dir = best["directory"]
        # Revalidate evidence before replacing a caller's existing submission.
        value = _score_holdout(winning_dir / "holdout_predictions.csv", expected_ids,
                               expected_actuals, classification, y.unique())
        if value != best["value"] or not validate_predictions(winning_dir)["valid"]:
            raise ValueError("Сохранённый результат победителя изменился; публикация отменена.")
        if (data_fingerprint(data_dir) != input_fingerprint
                or task_path.read_text(encoding="utf-8-sig") != task_text):
            raise ValueError("Исходная задача или данные изменились во время автоподбора; публикация отменена.")
        report.update(model=best["spec"]["model_config"]["model"], model_config=best["spec"]["model_config"],
                      metric={"name": metric_name, "value": value, "higher_is_better": classification},
                      output_valid=True)
        if classification:
            held = read_csv(winning_dir / "holdout_predictions.csv")
            diagnostics = classification_diagnostics(held["actual"], held["prediction"])
            if diagnostics is not None:
                report["classification_diagnostics"] = validate_classification_diagnostics(
                    diagnostics, metric=report["metric"], validation_rows=len(held_rows))
        _atomic_copy(winning_dir / "holdout_predictions.csv", output_dir / "holdout_predictions.csv")
        save("finished")
        _atomic_copy(winning_dir / "submission.csv", task_dir / "submission.csv")
        print(f"Best: {report['model']} · {metric_name}: {value:.6f}. API calls: 0.", flush=True)
        print(f"Saved: {task_dir / 'submission.csv'}", flush=True)
        print(f"Report: {output_dir / 'automl_report.json'}", flush=True)
        return deepcopy(report)
    except Exception:
        search["status"] = "failed"
        report["output_valid"] = False
        try:
            save("finished")
        except Exception:
            pass
        raise
