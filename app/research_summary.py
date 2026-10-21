"""Bounded presentation of research stages and the origin of reported scores."""

from __future__ import annotations

import math
import re


STAGES = ("preparation", "inner_research", "candidate_frozen", "outer_validation", "full_refit", "result")
STAGE_STATUSES = {"pending", "running", "completed", "skipped", "failed"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _metric(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or not isinstance(value.get("name"), str)
            or not 1 <= len(value["name"]) <= 120
            or type(value.get("higher_is_better")) is not bool
            or isinstance(value.get("value"), bool)
            or not isinstance(value.get("value"), (int, float))
            or not -((1 << 53) - 1) <= value["value"] <= (1 << 53) - 1
            or not math.isfinite(value["value"])):
        raise ValueError("Некорректная метрика исследования.")
    return {key: value[key] for key in ("name", "value", "higher_is_better")}


def _split(value):
    """Present split diagnostics without paths, row indices or target values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Некорректное разбиение исследования.")
    result = {}
    if "strategy" in value:
        if not isinstance(value["strategy"], str) or value["strategy"] not in {"random", "time", "grouped"}:
            raise ValueError("Неизвестное разбиение исследования.")
        result["strategy"] = value["strategy"]
    for name in ("train_rows", "validation_rows", "group_overlap"):
        if name in value:
            if type(value[name]) is not int or not 0 <= value[name] <= (1 << 31) - 1:
                raise ValueError("Некорректный размер разбиения исследования.")
            result[name] = value[name]
    return result


def validate_research_summary(value):
    """Validate an already projected job/import summary; discard unknown fields."""
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int:
        raise ValueError("Не указана версия протокола исследования.")
    version = value["protocol_version"]
    if version == 2:
        return {"protocol_version": 2}
    if version != 3:
        raise ValueError("Неизвестная версия протокола исследования.")
    result = {"protocol_version": 3}
    stage = value.get("stage")
    if stage is not None:
        if stage not in STAGES:
            raise ValueError("Неизвестный этап исследования.")
        result["stage"] = stage
    stages = value.get("stages", [])
    if not isinstance(stages, list) or len(stages) > len(STAGES):
        raise ValueError("Некорректный список этапов исследования.")
    result["stages"] = []
    seen = set()
    for item in stages:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or item["id"] not in STAGES or item["id"] in seen
                or not isinstance(item.get("status"), str) or item["status"] not in STAGE_STATUSES):
            raise ValueError("Некорректный этап исследования.")
        seen.add(item["id"])
        entry = {"id": item["id"], "status": item["status"]}
        if item.get("reason") is not None:
            if not isinstance(item["reason"], str) or len(item["reason"]) > 2000:
                raise ValueError("Некорректное пояснение этапа исследования.")
            entry["reason"] = item["reason"]
        result["stages"].append(entry)
    evaluations = value.get("evaluations", {})
    if not isinstance(evaluations, dict):
        raise ValueError("Некорректные оценки исследования.")
    result["evaluations"] = {}
    selection = evaluations.get("selection")
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError("Некорректная внутренняя оценка исследования.")
        if selection.get("scope", "inner_selection") != "inner_selection":
            raise ValueError("Назначение внутренней оценки исследования не совпадает с протоколом.")
        # A metric is optional while research is still preparing its baseline.
        selected_metric = _metric(selection) if "value" in selection else None
        result["evaluations"]["selection"] = {
            "scope": "inner_selection", "metric": selected_metric,
            "split": _split(selection.get("split")),
        }
        # Stored summaries wrap the metric to keep its provenance explicit.
        if "metric" in selection:
            result["evaluations"]["selection"]["metric"] = _metric(selection["metric"])
    final_check = evaluations.get("final_check")
    if final_check is not None:
        if not isinstance(final_check, dict):
            raise ValueError("Некорректная внешняя проверка исследования.")
        final = {name: _metric(final_check.get(name)) for name in ("baseline", "candidate")}
        selected = final_check.get("selected")
        if selected is not None and (not isinstance(selected, str) or selected not in {"baseline", "candidate"}):
            raise ValueError("Неизвестный результат приёмки исследования.")
        # The protocol uses outer for acceptance, so never promote a claimed
        # unbiased estimate from an imported or older report.
        final.update(selected=selected, selection_uses_outer=True, unbiased_estimate=False,
                     split=_split(final_check.get("split")))
        result["evaluations"]["final_check"] = final
    frozen = value.get("frozen_candidate")
    if frozen is not None:
        if not isinstance(frozen, dict):
            raise ValueError("Некорректный зафиксированный кандидат.")
        result["frozen_candidate"] = {}
        for name in ("code_hash", "recipe_hash"):
            if name in frozen:
                if not isinstance(frozen[name], str) or not _HASH.fullmatch(frozen[name]):
                    raise ValueError("Некорректный хеш зафиксированного кандидата.")
                result["frozen_candidate"][name] = frozen[name]
    return result


def research_summary(report):
    version = report.get("research_protocol_version", 2)
    return validate_research_summary({
        "protocol_version": version,
        **{name: report[name] for name in ("stage", "stages", "evaluations", "frozen_candidate")
           if name in report},
    })
