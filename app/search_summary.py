"""Bounded, internally consistent evidence from the offline model search."""

from __future__ import annotations

import math
import re


_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_MODELS = {"RandomForestClassifier", "ExtraTreesClassifier",
           "HistGradientBoostingRegressor", "ExtraTreesRegressor"}
_SUMMARY_FIELDS = {"status", "planned", "completed", "best_candidate",
                   "validation_reason", "candidates"}
_CANDIDATE_FIELDS = {"id", "label", "model", "status", "metric", "selected",
                     "elapsed_seconds", "error"}


def _text(value, field, limit, *, empty=False):
    if (not isinstance(value, str) or len(value) > limit or "\x00" in value
            or not empty and not value.strip()):
        raise ValueError(f"Некорректное поле автоподбора: {field}.")
    return value


def _number(value, field, *, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= (1 << 53) - 1):
        raise ValueError(f"Некорректное число в отчёте автоподбора: {field}.")
    return value


def validate_search_summary(raw, *, metric=None, model=None, allow_running=False):
    """Validate evidence; sharing it does not make an imported claim verified.

    In-flight progress may have no winner yet. A final successful search must
    identify one completed candidate and, when supplied, its enclosing metric
    and model must agree. Unknown keys are rejected rather than exposed.
    """
    if not isinstance(raw, dict) or set(raw) != _SUMMARY_FIELDS:
        raise ValueError("Неполный или неизвестный формат отчёта автоподбора.")
    statuses = {"completed", "time_limit"} | ({"running", "failed"} if allow_running else set())
    if not isinstance(raw["status"], str) or raw["status"] not in statuses:
        raise ValueError("Некорректный статус автоподбора.")
    if (type(raw["planned"]) is not int or not 1 <= raw["planned"] <= 8
            or type(raw["completed"]) is not int
            or not 0 <= raw["completed"] <= raw["planned"]):
        raise ValueError("Некорректное число попыток автоподбора.")
    reason = _text(raw["validation_reason"], "validation_reason", 2000, empty=True)
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) > raw["planned"]:
        raise ValueError("Некорректный список попыток автоподбора.")
    copied, identifiers, selected = [], set(), []
    candidate_statuses = {"completed", "failed", "timed_out", "skipped"}
    if allow_running:
        candidate_statuses |= {"running", "pending"}
    for row in candidates:
        if not isinstance(row, dict) or set(row) != _CANDIDATE_FIELDS:
            raise ValueError("Неполный или неизвестный формат попытки автоподбора.")
        identifier = row["id"]
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in identifiers:
            raise ValueError("Идентификаторы попыток должны быть уникальными.")
        identifiers.add(identifier)
        label = _text(row["label"], "candidate.label", 240)
        candidate_model = _text(row["model"], "candidate.model", 120)
        if candidate_model not in _MODELS:
            raise ValueError("Неизвестная модель в отчёте автоподбора.")
        status = row["status"]
        if not isinstance(status, str) or status not in candidate_statuses:
            raise ValueError("Некорректный статус попытки автоподбора.")
        if type(row["selected"]) is not bool:
            raise ValueError("Признак победителя автоподбора должен быть логическим.")
        value = row["metric"]
        if value is not None:
            _number(value, "candidate.metric")
        if status == "completed" and value is None:
            raise ValueError("У завершённой попытки отсутствует метрика.")
        elapsed = _number(row["elapsed_seconds"], "candidate.elapsed_seconds")
        error = row["error"]
        if error is not None:
            _text(error, "candidate.error", 2000, empty=True)
        if row["selected"] and status != "completed":
            raise ValueError("Победителем может быть только завершённая попытка.")
        candidate = {"id": identifier, "label": label, "model": candidate_model,
                     "status": status, "metric": value, "selected": row["selected"],
                     "elapsed_seconds": elapsed, "error": error}
        copied.append(candidate)
        if candidate["selected"]:
            selected.append(candidate)
    if raw["completed"] != sum(row["status"] == "completed" for row in copied):
        raise ValueError("Число завершённых попыток не совпадает с отчётом.")
    best = raw["best_candidate"]
    if best is None:
        if raw["status"] not in {"running", "failed"} or selected or raw["completed"]:
            raise ValueError("Не указан победитель завершённого автоподбора.")
    elif (not isinstance(best, str) or len(selected) != 1
          or selected[0]["id"] != best):
        raise ValueError("Победитель автоподбора не совпадает с выбранной попыткой.")
    if selected:
        winner = selected[0]
        if model is not None and model != winner["model"]:
            raise ValueError("Модель победителя не совпадает с отчётом запуска.")
        if metric is not None:
            if (not isinstance(metric, dict) or type(metric.get("higher_is_better")) is not bool
                    or winner["metric"] != metric.get("value")):
                raise ValueError("Метрика победителя не совпадает с отчётом запуска.")
            if any((row["metric"] > winner["metric"] if metric["higher_is_better"]
                    else row["metric"] < winner["metric"])
                   for row in copied if row["status"] == "completed"):
                raise ValueError("Выбранная попытка не имеет лучшую метрику.")
    return {"status": raw["status"], "planned": raw["planned"], "completed": raw["completed"],
            "best_candidate": best, "validation_reason": reason, "candidates": copied}
