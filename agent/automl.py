"""Budgeted nested model search with early publication and verified checkpoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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

from agent import project_root
from agent.diagnostics import classification_diagnostics, validate_classification_diagnostics
from agent.execution import run_python
from agent.file_io import replace_file
from agent.features import normalize_feature_config
from agent.search import (align_predictions, exploration_plan, fixed_blend, inner_split_rows,
                          metric_predictions, refinement_plan, seed_candidate, select_shortlist)
from agent.task_spec import load_task, load_task_data, score_holdout
from agent.splits import normalize_validation, split_rows
from offline_baseline import read_csv

ROOT = project_root()
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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with _plain(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_accepted_checkpoint(output_dir: Path) -> Path | None:
    """Read an atomically committed CSV/evidence pair, including after a kill.

    Loose publication copies are for users only and are never recovery proof.
    A pointer exists only after both immutable files have been copied and hashed.
    """
    output_dir = _plain(Path(output_dir))
    pointer = _plain(output_dir / "checkpoint.json")
    if not pointer.exists():
        return None
    value = json.loads(pointer.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or set(value) != {"version", "directory", "submission_sha256", "holdout_sha256"}
            or value["version"] != 1 or not isinstance(value["directory"], str)
            or not re.fullmatch(r"checkpoints/[0-9a-f]{32}", value["directory"])):
        raise ValueError("Invalid accepted AutoML checkpoint pointer")
    directory = _plain(output_dir / value["directory"])
    for name, key in (("submission.csv", "submission_sha256"), ("holdout_predictions.csv", "holdout_sha256")):
        if _file_hash(directory / name) != value[key]:
            raise ValueError("Accepted AutoML checkpoint bytes changed")
    return directory


def _commit_checkpoint(submission: Path, held: Path, output_dir: Path) -> Path:
    relative = "checkpoints/" + uuid4().hex
    directory = _plain(output_dir / relative)
    directory.mkdir(parents=True)
    _atomic_copy(submission, directory / "submission.csv")
    _atomic_copy(held, directory / "holdout_predictions.csv")
    _atomic_json(output_dir / "checkpoint.json", {
        "version": 1, "directory": relative,
        "submission_sha256": _file_hash(directory / "submission.csv"),
        "holdout_sha256": _file_hash(directory / "holdout_predictions.csv"),
    })
    return directory


def data_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in INPUT_NAMES:
        digest.update(name.encode("utf-8") + b"\0")
        with (data_dir / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def candidate_plan(classification: bool) -> list[dict]:
    """Initial exploration; retained as a public extension/testing seam."""
    return exploration_plan(classification)


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


def _raw_evidence(directory, spec, expected_ids, expected_actuals):
    """Require the raw blend bundle to describe exactly the scored CSV rows."""
    with np.load(directory / "holdout_raw.npz", allow_pickle=False) as bundle:
        if set(bundle.files) != {"ids", "actual", "predictions", "classes"}:
            raise ValueError("Invalid raw prediction bundle")
        if (bundle["ids"].tolist() != list(expected_ids)
                or bundle["actual"].tolist() != list(expected_actuals)):
            raise ValueError("Raw prediction IDs or actuals differ from the frozen holdout")
        raw = align_predictions(spec, bundle["predictions"], bundle["classes"].tolist())
    regenerated = spec.holdout_frame(expected_ids, expected_actuals, metric_predictions(spec, raw),
                                      prediction_classes=spec.classes)
    saved = read_csv(directory / "holdout_predictions.csv")
    if list(saved.columns) != list(regenerated.columns) or len(saved) != len(regenerated):
        raise ValueError("Raw predictions do not match scored evidence")
    for column in saved.columns:
        if column in {"id", "actual"} or spec.prediction_kind == "labels":
            if saved[column].tolist() != regenerated[column].astype(str).tolist():
                raise ValueError("Raw predictions do not match scored evidence")
        elif not np.allclose(pd.to_numeric(saved[column]), pd.to_numeric(regenerated[column]),
                             rtol=1e-12, atol=1e-12):
            raise ValueError("Raw predictions do not match scored evidence")
    return raw


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
    task_dir, task_text = task_path.parent, task_path.read_text(encoding="utf-8-sig")
    spec = load_task(task_path)
    data_dir = _plain(spec.data_dir)
    for name in INPUT_NAMES:
        _plain(data_dir / name)
    _plain(task_dir / "submission.csv")
    run_dir = _plain(task_dir / ".agent_runs" / uuid4().hex)
    run_dir.mkdir(parents=True)
    snapshot_dir = run_dir / "input_snapshot"
    input_fingerprint = _snapshot_inputs(data_dir, snapshot_dir)
    snapshot_spec = replace(spec, data_dir=snapshot_dir)
    (train, test, _, id_column, target, y, features,
     numeric, categorical, classification) = load_task_data(snapshot_spec)
    choice, reason = choose_validation(train, test, features, numeric, classification, validation)
    fit_rows, held_rows, split = split_rows(train, y, features, classification, choice)
    expected_ids = train[id_column].iloc[held_rows].tolist()
    expected_actuals = train[target].iloc[held_rows].tolist()
    split["validation_hash"] = _json_hash({"id_column": id_column, "target_column": target,
                                           "ids": expected_ids, "actuals": expected_actuals})
    inner = None
    inner_error = None
    try:
        inner_fit, inner_held, inner_metadata = inner_split_rows(train, y, features, classification, choice, fit_rows)
        inner = (inner_fit, inner_held)
        inner_metadata["validation_hash"] = _json_hash({
            "id_column": id_column, "target_column": target,
            "ids": train[id_column].iloc[inner_held].tolist(), "actuals": train[target].iloc[inner_held].tolist(),
        })
    except ValueError as error:
        inner_metadata = None
        inner_error = str(error)
        reason += " Внутренняя проверка невозможна: сохранится только стартовая модель без подбора."
    if dataset_fingerprint is None:
        dataset_fingerprint = input_fingerprint
    if not isinstance(dataset_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint):
        raise ValueError("Некорректный отпечаток набора данных.")
    comparison_key = _json_hash({"dataset_fingerprint": dataset_fingerprint, "metric": spec.metric.name,
                                "higher_is_better": spec.metric.higher_is_better, "split": split})
    output_dir = _plain(Path(output_dir)) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("automl_report.json", "holdout_predictions.csv"):
        _plain(output_dir / name)
    exploration = candidate_plan(classification) if inner is not None else []
    plan = [seed_candidate(classification), *exploration]
    search = {"status": "running", "phase": "seed", "planned": len(plan), "completed": 0,
              "selection_completed": 0, "verified": 0, "best_candidate": None,
              "validation_reason": reason, "candidates": []}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": "offline",
              "task_type": spec.metric.task_type, "dataset_fingerprint": dataset_fingerprint,
              "input_fingerprint": input_fingerprint, "split": split, "inner_split": inner_metadata,
              "inner_unavailable": inner_error, "comparison_key": comparison_key,
              "api": {"used": False, "calls": 0}, "time_budget_seconds": time_budget,
              "search": search, "candidate_configs": {}, "candidate_feature_configs": {},
              "search_policy": {"inner_selection": True, "outer_shortlist_limit": 3,
                                "outer_shortlist_policy": "distinct_families_first",
                                "outer_execution_order": "measured_inner_elapsed_ascending",
                                "blend_weight": 0.5, "refit_reserve_fraction": 0.25},
              "report_path": str(output_dir / "automl_report.json")}
    rows, recipes, evidence = {}, {}, {}
    best = None
    phase_counter = 0

    def add_recipe(recipe, phase="selection"):
        recipe = deepcopy(recipe)
        recipe["feature_config"] = normalize_feature_config(recipe.get("feature_config"))
        recipes[recipe["id"]] = recipe
        row = {"id": recipe["id"], "label": recipe["label"], "model": recipe["model_config"]["model"],
               "model_config": recipe["model_config"], "feature_config": recipe["feature_config"],
               "status": "skipped", "phase": phase, "metric": None, "selection_metric": None,
               "verification_metric": None, "selected": False, "elapsed_seconds": 0.0,
               "error": None, "rejection_reason": None}
        if recipe.get("components"):
            row["component_feature_configs"] = [item["feature_config"] for item in recipe["components"]]
        rows[recipe["id"]] = row
        search["candidates"].append(row)
        search["planned"] = len(search["candidates"])
        report["candidate_configs"][recipe["id"]] = recipe["model_config"]
        report["candidate_feature_configs"][recipe["id"]] = recipe["feature_config"]

    for item in plan:
        add_recipe(item, "seed" if item["id"] == "seed" else "selection")

    def save(stage):
        search["completed"] = sum(row["status"] == "completed" for row in search["candidates"])
        search["selection_completed"] = sum(row["selection_metric"] is not None for row in search["candidates"])
        search["verified"] = sum(row["metric"] is not None for row in search["candidates"])
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(output_dir / "automl_report.json", report)
        if progress:
            progress({"stage": stage, "search": deepcopy(search)})

    def better(left, right):
        return left > right if spec.metric.higher_is_better else left < right

    def rank(items, key):
        return sorted(items, key=lambda item: rows[item["id"]][key], reverse=spec.metric.higher_is_better)

    def check_source():
        if (data_fingerprint(data_dir) != input_fingerprint
                or data_fingerprint(snapshot_dir) != input_fingerprint
                or task_path.read_text(encoding="utf-8-sig") != task_text):
            raise ValueError("Исходная задача или данные изменились во время автоподбора; публикация отменена.")

    def check_submission(directory):
        spec.validate_frame(read_csv(directory / "submission.csv"), test[id_column].tolist())

    def publish(recipe, record, submission_dir):
        nonlocal best
        value = score_holdout(record["directory"] / "holdout_predictions.csv", spec, expected_ids, expected_actuals)
        if value != record["value"]:
            raise ValueError("Сохранённый результат победителя изменился; публикация отменена.")
        check_submission(submission_dir)
        check_source()
        checkpoint_dir = _commit_checkpoint(submission_dir / "submission.csv",
                                            record["directory"] / "holdout_predictions.csv", output_dir)
        _atomic_copy(checkpoint_dir / "submission.csv", task_dir / "submission.csv")
        _atomic_copy(checkpoint_dir / "holdout_predictions.csv", output_dir / "holdout_predictions.csv")
        best = {"recipe": recipe, "record": record, "submission_dir": submission_dir, "value": value}
        search["best_candidate"] = recipe["id"]
        for row in search["candidates"]:
            row["selected"] = row["id"] == recipe["id"]
        report.update(model=recipe["model_config"]["model"], model_config=recipe["model_config"],
                      feature_config=recipe["feature_config"],
                      metric={"name": spec.metric.name, "value": value,
                              "higher_is_better": spec.metric.higher_is_better}, output_valid=True)
        if recipe.get("components"):
            report["component_feature_configs"] = [item["feature_config"] for item in recipe["components"]]
        else:
            report.pop("component_feature_configs", None)
        report.pop("classification_diagnostics", None)
        if spec.metric.name == "Macro F1":
            held = read_csv(record["directory"] / "holdout_predictions.csv")
            diagnostics = classification_diagnostics(held["actual"], held["prediction"])
            if diagnostics is not None:
                report["classification_diagnostics"] = validate_classification_diagnostics(
                    diagnostics, metric=report["metric"], validation_rows=len(held_rows))
        save("training")

    def run_phase(recipe, phase, phase_deadline, shares=1):
        nonlocal phase_counter
        row = rows[recipe["id"]]
        row.update(phase=phase, status="running", error=None, rejection_reason=None)
        search["phase"] = phase
        phase_started = time.monotonic()
        directory = run_dir / recipe["id"] / phase
        try:
            if min(phase_deadline, deadline - 2) - time.monotonic() <= 1:
                row.update(status="skipped", rejection_reason="insufficient_budget")
                return None
            directory.mkdir(parents=True)
            _stage_inputs(snapshot_dir, directory / "inputs")
            (directory / "task.md").write_text(task_text, encoding="utf-8")
            local_spec = replace(spec, data_dir=directory / "inputs", task_path=directory / "task.md")
            fitting, held = inner if phase == "selection" else (fit_rows, held_rows)
            config_path = directory / "config.json"
            _atomic_json(config_path, {**recipe, "candidate_dir": str(directory), "phase": phase,
                                      "task_spec": local_spec.to_dict(), "fit_rows": fitting.tolist(),
                                      "validation_rows": held.tolist()})
            save("training")
            # Staging and progress callbacks are charged to the same deadline.
            remaining = min(phase_deadline, deadline - 2) - time.monotonic()
            timeout = min(1800.0, remaining / max(1, shares))
            if timeout <= 1:
                row.update(status="skipped", rejection_reason="insufficient_budget")
                return None
            phase_counter += 1
            code = ("import sys\n" f"sys.path.insert(0, {str(ROOT)!r})\n"
                    "from pathlib import Path\nfrom agent.automl_worker import run_candidate\n"
                    f"run_candidate(Path({str(config_path)!r}))\n")
            print(f"{phase}: {recipe['label']} (limit {timeout:.1f}s)", flush=True)
            execution = run_python(code, directory, directory / "execution", phase_counter, timeout=timeout)
            telemetry = execution.get("resources") or {"available": False, "unavailable_reason": "No process telemetry"}
            previous = row.get("resources", {})
            available = bool(telemetry.get("available", False)) and previous.get("available", True)
            row["resources"] = {
                "available": available,
                "unavailable_reason": (previous.get("unavailable_reason") or telemetry.get("unavailable_reason"))
                                      if not available else None,
                "peak_rss_bytes": max(previous.get("peak_rss_bytes") or 0, telemetry.get("peak_rss_bytes") or 0)
                                  if available else None,
                "cpu_seconds": (previous.get("cpu_seconds") or 0) + (telemetry.get("cpu_seconds") or 0)
                               if available else None,
                "memory_limit_bytes": telemetry.get("memory_limit_bytes"),
                "exit_code": execution.get("exit_code"), "timed_out": bool(execution.get("timed_out")),
                "memory_exceeded": bool(execution.get("memory_limit_exceeded")),
            }
            if execution.get("memory_limit_exceeded"):
                row.update(status="failed", rejection_reason="memory_limit", error="Превышен лимит памяти попытки.")
                return None
            if execution.get("timed_out"):
                row.update(status="timed_out", rejection_reason="timeout", error="Превышен лимит времени попытки.")
                return None
            if execution.get("exit_code") != 0 or execution.get("error"):
                error = (execution.get("error") or execution.get("stderr") or "Процесс обучения завершился с ошибкой.")[-2000:]
                missing = "ModuleNotFoundError" in error or "ImportError" in error
                row.update(status="failed", rejection_reason="optional_dependency_unavailable" if missing else "execution_failed",
                           error=error)
                return None
            if data_fingerprint(directory / "inputs") != input_fingerprint:
                raise ValueError("Данные попытки изменились во время обучения.")
            if (directory / "durations.json").is_file():
                durations = json.loads((directory / "durations.json").read_text(encoding="utf-8"))
                if (set(durations) != {"fit_seconds", "predict_seconds", "total_seconds"}
                        or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
                               for value in durations.values())):
                    raise ValueError("Invalid worker timing evidence")
                row["durations"] = {key: row.get("durations", {}).get(key, 0) + value
                                    for key, value in durations.items()}
            record = {"directory": directory}
            if phase != "refit":
                ids, actual = train[id_column].iloc[held].tolist(), train[target].iloc[held].tolist()
                value = score_holdout(directory / "holdout_predictions.csv", spec, ids, actual)
                raw = _raw_evidence(directory, spec, ids, actual)
                record.update(value=value, raw=raw)
                if phase == "selection":
                    row["selection_metric"] = value
                else:
                    row.update(metric=value, verification_metric=value)
            if phase in {"seed", "refit"}:
                check_submission(directory)
            row["status"] = "completed"
            return record
        except (OSError, ValueError, RuntimeError) as error:
            row.update(status="failed", rejection_reason="invalid_evidence", error=str(error)[-2000:])
            return None
        finally:
            row["elapsed_seconds"] = round(row["elapsed_seconds"] + time.monotonic() - phase_started, 3)
            save("training")

    print(f"Offline AutoML: nested selection; {spec.metric.name}; {choice['strategy']} validation.", flush=True)
    print(reason, flush=True)
    save("training")
    try:
        seed = recipes["seed"]
        record = run_phase(seed, "seed", started + time_budget * 0.20)
        if record is not None:
            evidence["seed"] = record
            publish(seed, record, record["directory"])
        if inner is not None:
            # Plan six refinement slots up front in the budget, but choose their
            # families only after all initial INNER results have been observed.
            adaptive = len(exploration) >= 5
            for index, recipe in enumerate(exploration):
                run_phase(recipes[recipe["id"]], "selection", started + time_budget * 0.60,
                          len(exploration) - index + (6 if adaptive else 0))
            successful = [recipes[item["id"]] for item in exploration if rows[item["id"]]["selection_metric"] is not None]
            ranked_inner = rank(successful, "selection_metric")
            refinements = refinement_plan(classification, ranked_inner, list(recipes.values())) if adaptive else []
            for recipe in refinements:
                add_recipe(recipe)
            for index, recipe in enumerate(refinements):
                run_phase(recipes[recipe["id"]], "selection", started + time_budget * 0.60, len(refinements) - index)
            successful = [recipe for recipe in recipes.values() if rows[recipe["id"]]["selection_metric"] is not None]
            shortlist = select_shortlist(rank(successful, "selection_metric"))
            for recipe in successful:
                if recipe not in shortlist:
                    rows[recipe["id"]]["rejection_reason"] = "not_selected"
            # Freeze a stable order from INNER runtimes before verification adds
            # to row timings. Fast trials release their unused shares to slower
            # finalists; the selected families and outer deadline do not change.
            verification_order = sorted(shortlist, key=lambda recipe: rows[recipe["id"]]["elapsed_seconds"])
            for index, recipe in enumerate(verification_order):
                record = run_phase(recipe, "verification", started + time_budget * 0.75, len(verification_order) - index)
                if record is not None:
                    evidence[recipe["id"]] = record

        verified = rank([recipes[identifier] for identifier in evidence], "metric")
        if len(verified) >= 2 and time.monotonic() < started + time_budget * 0.75:
            left, right = verified[:2]
            blend = {"id": "blend", "action": "blend", "label": "Среднее двух моделей · 50/50",
                     "model_config": {"model": "FixedBlend", "parameters": {
                         "weight": 0.5, "components": [left["model_config"], right["model_config"]]}},
                     "feature_config": normalize_feature_config(), "components": [left, right]}
            add_recipe(blend, "blend")
            row, directory = rows["blend"], run_dir / "blend" / "blend"
            search["phase"] = "blend"
            directory.mkdir(parents=True)
            raw = fixed_blend(spec, evidence[left["id"]]["raw"], evidence[right["id"]]["raw"],
                              left_classes=spec.classes, right_classes=spec.classes)
            spec.holdout_frame(expected_ids, expected_actuals, metric_predictions(spec, raw),
                                prediction_classes=spec.classes).to_csv(
                directory / "holdout_predictions.csv", index=False, encoding="utf-8")
            value = score_holdout(directory / "holdout_predictions.csv", spec, expected_ids, expected_actuals)
            row.update(status="completed", metric=value, verification_metric=value)
            # A blend must strictly improve on BOTH components. Never replace a
            # component with an equal-scoring, more expensive full-data refit.
            if better(value, rows[left["id"]]["metric"]):
                evidence["blend"] = {"directory": directory, "value": value, "raw": raw}
                verified = rank([*verified, recipes["blend"]], "metric")
            else:
                row["rejection_reason"] = "not_improved"
            save("training")

        for recipe in verified:
            row = rows[recipe["id"]]
            if best is not None and not better(row["metric"], best["value"]):
                if not row["selected"] and row["rejection_reason"] is None:
                    row["rejection_reason"] = "not_improved"
                continue
            record = run_phase(recipe, "refit", deadline - 2)
            if record is None:
                if row["rejection_reason"] != "insufficient_budget":
                    row["rejection_reason"] = "refit_failed"
                continue
            publish(recipe, evidence[recipe["id"]], record["directory"])
        check_source()
        if best is None:
            raise RuntimeError("Автоподбор не получил ни одного проверенного результата. "
                               "Прежний submission.csv сохранён; откройте журнал попыток.")
        search["status"] = ("time_limit" if any(row["rejection_reason"] in {"timeout", "insufficient_budget"}
                                               or row.get("resources", {}).get("timed_out")
                                               for row in search["candidates"]) else "completed")
        search["phase"] = "finished"
        save("finished")
        print(f"Best: {report['model']} · {spec.metric.name}: {best['value']:.6f}. API calls: 0.", flush=True)
        print(f"Saved: {task_dir / 'submission.csv'}", flush=True)
        print(f"Report: {output_dir / 'automl_report.json'}", flush=True)
        return deepcopy(report)
    except Exception:
        search.update(status="failed", phase="finished")
        report["output_valid"] = False
        try:
            save("finished")
        except Exception:
            pass
        raise
