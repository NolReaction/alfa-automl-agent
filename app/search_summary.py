"""Bounded, internally consistent evidence from the offline model search."""
from __future__ import annotations

import math
import re
from app.model_config import model_options, validate_model_config, feature_evidence

_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_BASE = {"status", "planned", "completed", "best_candidate", "validation_reason", "candidates"}
_OPTIONAL = {"phase", "selection_completed", "verified"}
_ROW = {"id", "label", "model", "status", "metric", "selected", "elapsed_seconds", "error"}
_ROW_OPTIONAL = {"phase", "model_config", "selection_metric", "verification_metric", "rejection_reason", "resources", "durations", "feature_config", "component_feature_configs"}
_PHASES = {"seed", "selection", "verification", "refit", "blend", "finished"}
_REASONS = {"timeout", "memory_limit", "execution_failed", "invalid_evidence", "not_selected", "not_improved", "insufficient_budget", "refit_failed", "optional_dependency_unavailable"}


def _text(value, field, limit, *, empty=False):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value or (not empty and not value.strip()):
        raise ValueError(f"Некорректное поле автоподбора: {field}.")
    return value


def _number(value, field, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= (1 << 53) - 1:
        raise ValueError(f"Некорректное число в отчёте автоподбора: {field}.")
    return value


def _metric(value, field):
    return None if value is None else _number(value, field, minimum=-((1 << 53) - 1))


def _resources(raw):
    fields = {"available", "unavailable_reason", "peak_rss_bytes", "cpu_seconds", "memory_limit_bytes", "exit_code", "timed_out", "memory_exceeded"}
    if not isinstance(raw, dict) or set(raw) - fields:
        raise ValueError("Неизвестные поля ресурсов попытки.")
    result = {}
    for key, value in raw.items():
        if key in {"available", "timed_out", "memory_exceeded"}:
            if type(value) is not bool:
                raise ValueError("Некорректный признак ресурсов попытки.")
        elif key == "unavailable_reason":
            if value is not None:
                _text(value, key, 500, empty=True)
        elif value is not None:
            _number(value, key, minimum=-255 if key == "exit_code" else 0)
        result[key] = value
    return result


def validate_search_summary(raw, *, metric=None, model=None, allow_running=False):
    """Accept legacy rows and separate selection evidence from outer verification."""
    if not isinstance(raw, dict) or not _BASE <= set(raw) or set(raw) - _BASE - _OPTIONAL:
        raise ValueError("Неполный или неизвестный формат отчёта автоподбора.")
    statuses = {"completed", "time_limit"} | ({"running", "failed"} if allow_running else set())
    if not isinstance(raw["status"], str) or raw["status"] not in statuses:
        raise ValueError("Некорректный статус автоподбора.")
    if type(raw["planned"]) is not int or not 1 <= raw["planned"] <= 21 or type(raw["completed"]) is not int or not 0 <= raw["completed"] <= raw["planned"]:
        raise ValueError("Некорректное число попыток автоподбора.")
    result = {key: raw[key] for key in _BASE - {"candidates"}}
    result["validation_reason"] = _text(raw["validation_reason"], "validation_reason", 2000, empty=True)
    if "phase" in raw:
        if not isinstance(raw["phase"], str) or raw["phase"] not in _PHASES:
            raise ValueError("Неизвестная фаза автоподбора.")
        result["phase"] = raw["phase"]
    rows = raw["candidates"]
    if not isinstance(rows, list) or len(rows) > raw["planned"]:
        raise ValueError("Некорректный список попыток автоподбора.")
    models = {option["model"] for task in ("regression", "classification") for option in model_options(task).values()} | {"FixedBlend"}
    statuses = {"completed", "failed", "timed_out", "skipped"} | ({"running", "pending"} if allow_running else set())
    copied, identifiers, selected = [], set(), []
    for row in rows:
        if not isinstance(row, dict) or not _ROW <= set(row) or set(row) - _ROW - _ROW_OPTIONAL:
            raise ValueError("Неполный или неизвестный формат попытки автоподбора.")
        identifier = row["id"]
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in identifiers:
            raise ValueError("Идентификаторы попыток должны быть уникальными.")
        identifiers.add(identifier)
        candidate = {"id": identifier, "label": _text(row["label"], "label", 240), "model": _text(row["model"], "model", 120), "status": row["status"], "metric": _metric(row["metric"], "metric"), "selected": row["selected"], "elapsed_seconds": _number(row["elapsed_seconds"], "elapsed_seconds"), "error": row["error"]}
        if candidate["model"] not in models or not isinstance(candidate["status"], str) or candidate["status"] not in statuses or type(candidate["selected"]) is not bool:
            raise ValueError("Неизвестная модель, статус или признак победителя.")
        if candidate["error"] is not None:
            _text(candidate["error"], "error", 2000, empty=True)
        for key in ("selection_metric", "verification_metric"):
            if key in row:
                candidate[key] = _metric(row[key], key)
        if "verification_metric" in candidate and candidate["verification_metric"] != candidate["metric"]:
            raise ValueError("Проверочная метрика не совпадает с метрикой попытки.")
        if candidate["status"] == "completed" and candidate["metric"] is None and candidate.get("selection_metric") is None:
            raise ValueError("У завершённой попытки отсутствует метрика.")
        if candidate["selected"] and (candidate["status"] != "completed" or candidate["metric"] is None):
            raise ValueError("Победителю нужна завершённая внешняя проверка.")
        if "phase" in row:
            if not isinstance(row["phase"], str) or row["phase"] not in _PHASES - {"finished"}:
                raise ValueError("Неизвестная фаза попытки.")
            candidate["phase"] = row["phase"]
        if "model_config" in row:
            candidate["model_config"] = validate_model_config(row["model_config"])
            if candidate["model_config"]["model"] != candidate["model"]:
                raise ValueError("Параметры попытки относятся к другой модели.")
        if "rejection_reason" in row:
            reason = row["rejection_reason"]
            if reason is not None and (not isinstance(reason, str) or reason not in _REASONS):
                raise ValueError("Неизвестная причина отклонения попытки.")
            candidate["rejection_reason"] = reason
        if "resources" in row:
            candidate["resources"] = _resources(row["resources"])
        if "durations" in row:
            durations = row["durations"]
            if not isinstance(durations, dict) or set(durations) - {"fit_seconds", "predict_seconds", "total_seconds"}:
                raise ValueError("Неизвестные поля длительности попытки.")
            candidate["durations"] = {key: _number(value, key) for key, value in durations.items()}
        candidate.update(feature_evidence(row, model=candidate["model"]))
        copied.append(candidate)
        if candidate["selected"]:
            selected.append(candidate)
    if raw["completed"] != sum(row["status"] == "completed" for row in copied):
        raise ValueError("Число завершённых попыток не совпадает с отчётом.")
    for key, count in (("selection_completed", sum(row.get("selection_metric") is not None for row in copied)), ("verified", sum(row["metric"] is not None for row in copied))):
        if key in raw:
            if type(raw[key]) is not int or raw[key] != count:
                raise ValueError("Число измеренных попыток не совпадает с отчётом.")
            result[key] = raw[key]
    best = raw["best_candidate"]
    if best is None:
        if raw["status"] not in {"running", "failed"} or selected:
            raise ValueError("Не указан победитель завершённого автоподбора.")
    elif not isinstance(best, str) or len(selected) != 1 or selected[0]["id"] != best:
        raise ValueError("Победитель автоподбора не совпадает с выбранной попыткой.")
    if selected:
        winner = selected[0]
        if model is not None and model != winner["model"]:
            raise ValueError("Модель победителя не совпадает с отчётом запуска.")
        if metric is not None:
            if not isinstance(metric, dict) or type(metric.get("higher_is_better")) is not bool or winner["metric"] != metric.get("value"):
                raise ValueError("Метрика победителя не совпадает с отчётом запуска.")
            # A fixed blend must improve both components before it is eligible
            # for refit. If a stronger component later fails to refit, that
            # already-rejected blend does not become an eligible fallback.
            eligible = (row for row in copied
                        if row["status"] == "completed" and row["metric"] is not None
                        and row.get("rejection_reason") not in {"refit_failed", "invalid_evidence", "insufficient_budget"}
                        and not (row["model"] == "FixedBlend"
                                 and row.get("rejection_reason") == "not_improved"))
            if any(row["metric"] > winner["metric"] if metric["higher_is_better"]
                   else row["metric"] < winner["metric"] for row in eligible):
                raise ValueError("Выбранная попытка не имеет лучшую проверочную метрику.")
    result["candidates"] = copied
    return result
