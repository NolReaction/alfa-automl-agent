"""Data-free campaign summaries from manager-supplied public snapshots only.

This module never opens task, model, prediction, source or report paths. Local
scores describe exploratory validation, not a hidden-test or platform score.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from io import BytesIO, StringIO
import json
import math
import os
import re
from zipfile import ZIP_DEFLATED, ZipFile

from app.jobs import SPLIT_FIELDS, _valid_split


MAX_STEPS = 128
MAX_REPORT_BYTES = 2 * 1024**2
MAX_SAFE_NUMBER = 2**53 - 1
HASH = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[0-9a-f]{32}\Z")
RUNTIME_PACKAGES = {"numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "openai", "httpx",
                    "pydantic", "tqdm", "python-dotenv", "catboost", "lightgbm", "optuna", "psutil"}
SCOPE = ("Local exploratory validation on known TRAIN. Reusing validation for "
         "selection does not establish hidden-test quality or platform points.")
STATUSES = {"pending", "queued", "running", "completed", "failed", "skipped",
            "interrupted", "cancelled", "cleanup_pending", "stopped", "launching", "paused", "stopping"}
CSV_FIELDS = ("step_id", "job_id", "task_id", "label", "status", "minutes",
              "worker_timeout_seconds", "elapsed_seconds", "metric_name", "metric_value", "higher_is_better",
              "model", "comparison_eligible", "comparison_group", "exclusion_reasons", "error")


def _number(value, *, nonnegative=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and abs(value) <= MAX_SAFE_NUMBER and (not nonnegative or value >= 0))
    except OverflowError:
        return False


def _text(value, limit=500):
    if not isinstance(value, str):
        return None
    # Read inherited values only; do not load any credential files.
    secrets = [secret for name, secret in os.environ.items() if secret and
               any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    # Enough lookahead to redact a credential beginning inside the retained
    # prefix, without scanning an unbounded malformed diagnostic string.
    value = value[:limit + max([4096, *(len(secret) for secret in secrets)])]
    for secret in secrets:
        value = value.replace(secret, "[redacted]")
    value = re.sub(r"(?i)\b(?:https?://)[^\s<>]+", "[redacted-url]", value)
    value = re.sub(r"(?i)\b(?:Bearer\s+|sk-)[A-Za-z0-9_.-]+", "[redacted]", value)
    value = re.sub(r"(?i)((?:api[_-]?key|password|secret|access[_-]?token|authorization)\s*[=:]\s*)[^\s,;]+",
                   r"\1[redacted]", value)
    value = re.sub(r"(?:[A-Za-z]:[\\/]|(?<!\w)/)[^\s<>\"']+", "[redacted-path]", value)
    return "".join(c for c in value if c in "\n\t" or ord(c) >= 32)[:limit]


def _hash(value):
    return value if isinstance(value, str) and HASH.fullmatch(value) else None


def _identifier(value):
    return value if isinstance(value, str) and ID.fullmatch(value) else None


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _split(value):
    if not isinstance(value, dict):
        return None
    try:
        if not _valid_split(value):
            return None
        result = {key: value[key] for key in sorted(SPLIT_FIELDS) if key in value}
        if any(not isinstance(v, (str, bool)) and not _number(v) for v in result.values()):
            return None
        if any(isinstance(v, str) and len(v) > 500 for v in result.values()):
            return None
        _canonical(result)
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def _metric(value):
    if (not isinstance(value, dict) or not isinstance(value.get("name"), str)
            or not value["name"].strip() or len(value["name"]) > 100
            or not _number(value.get("value")) or type(value.get("higher_is_better")) is not bool):
        return None
    return {"name": _text(value["name"], 100), "value": value["value"],
            "higher_is_better": value["higher_is_better"]}


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


def _source(value):
    value = value if isinstance(value, dict) else {}
    sha = value.get("sha")
    return {"sha": sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) else None,
            "source_fingerprint": _hash(value.get("source_fingerprint")),
            "valid": value.get("valid") if type(value.get("valid")) is bool else None,
            "checked_at": _timestamp(value.get("checked_at"))}


def _runtime(value):
    value = value if isinstance(value, dict) else {}
    packages = value.get("packages") if isinstance(value.get("packages"), dict) else {}
    versions = {name: version for name, version in packages.items() if name in RUNTIME_PACKAGES
                and isinstance(version, str) and re.fullmatch(r"[0-9][0-9A-Za-z.+!_-]{0,79}", version)
                and _text(version, 80) == version}
    python = value.get("python_version")
    implementation = value.get("implementation")
    result = {"python_version": python if isinstance(python, str) and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,79}", python) and _text(python, 80) == python else None,
            "implementation": implementation if isinstance(implementation, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.-]{0,39}", implementation) else None,
            "pointer_bits": value.get("pointer_bits") if type(value.get("pointer_bits")) is int and value["pointer_bits"] in {32, 64} else None,
            "packages": versions, "fingerprint": _hash(value.get("fingerprint")),
            "valid": value.get("valid") if type(value.get("valid")) is bool else None,
            "checked_at": _timestamp(value.get("checked_at"))}
    identity = {key: result[key] for key in ("python_version", "implementation", "pointer_bits", "packages")}
    result["integrity_verified"] = (bool(result["fingerprint"]) and bool(result["python_version"])
        and bool(result["implementation"]) and bool(result["pointer_bits"]) and set(versions) == RUNTIME_PACKAGES
        and hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest() == result["fingerprint"])
    return result


def _row(step, position, detail, source, runtime):
    job = step.get("job") if isinstance(step.get("job"), dict) else {}
    metric, split = _metric(job.get("metric")), _split(job.get("split"))
    fingerprint, key = _hash(job.get("dataset_fingerprint")), _hash(job.get("comparison_key"))
    status = step.get("status") if isinstance(step.get("status"), str) and step["status"] in STATUSES else "unknown"
    task = _text(step.get("task_id"), 128)
    job_id = _identifier(step.get("job_id"))
    expected = detail.get("task_fingerprints")
    expected = expected.get(step.get("task_id")) if isinstance(expected, dict) and isinstance(step.get("task_id"), str) else None
    reasons = []
    if status != "completed" or job.get("status") != "completed":
        reasons.append("not_completed")
    if job.get("source") != "local":
        reasons.append("not_local")
    if job.get("output_valid") is not True:
        reasons.append("output_not_verified")
    if source["valid"] is not True or not source["sha"] or not source["source_fingerprint"]:
        reasons.append("source_seal_not_verified")
    if runtime["valid"] is not True or runtime["integrity_verified"] is not True:
        reasons.append("runtime_seal_not_verified")
    if step.get("admissible") is not True:
        reasons.append("step_not_admitted")
    if (not task or job.get("task_id") != step.get("task_id") or not job_id or job.get("id") != job_id
            or not isinstance(step.get("action"), str) or job.get("action") != step.get("action")):
        reasons.append("job_binding_not_verified")
    if not fingerprint or fingerprint != _hash(expected):
        reasons.append("dataset_not_verified")
    if metric is None:
        reasons.append("metric_invalid")
    if split is None:
        reasons.append("split_invalid")
    if key is None:
        reasons.append("comparison_key_invalid")
    elif metric is not None and split is not None and fingerprint:
        identity = {"dataset_fingerprint": fingerprint, "metric": job["metric"]["name"],
                    "higher_is_better": metric["higher_is_better"], "split": split}
        if hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest() != key:
            reasons.append("comparison_key_mismatch")
    return {"position": position, "step_id": _identifier(step.get("id")), "job_id": job_id,
            "task_id": task, "action": _text(step.get("action"), 80), "label": _text(step.get("label")),
            "status": status, "minutes": step.get("minutes") if _number(step.get("minutes"), nonnegative=True) else None,
            "worker_timeout_seconds": step.get("worker_timeout_seconds") if _number(step.get("worker_timeout_seconds"), nonnegative=True) else None,
            "elapsed_seconds": job.get("elapsed_seconds") if _number(job.get("elapsed_seconds"), nonnegative=True) else None,
            "started_at": _timestamp(job.get("started_at")), "finished_at": _timestamp(job.get("finished_at")),
            "model": _text(job.get("model"), 160), "metric": metric, "split": split,
            "dataset_fingerprint": fingerprint, "comparison_key": key,
            "comparison_eligible": not reasons, "comparison_group": None,
            "exclusion_reasons": reasons, "error": _text(step.get("error") or job.get("error"), 2000)}


def _point(row):
    return {key: row[key] for key in ("step_id", "job_id", "position", "minutes", "worker_timeout_seconds", "elapsed_seconds", "started_at", "model", "metric")}


def build_campaign_report(detail):
    """Build a bounded, path-free report; malformed evidence cannot qualify."""
    if not isinstance(detail, dict) or not isinstance(detail.get("steps"), list):
        raise ValueError("Campaign detail must contain a steps list.")
    if len(detail["steps"]) > MAX_STEPS:
        raise ValueError("Campaign report exceeds the step limit.")
    if any(not isinstance(step, dict) for step in detail["steps"]):
        raise ValueError("Campaign steps must be objects.")
    fingerprints = detail.get("task_fingerprints")
    fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    if len(fingerprints) > 16:
        raise ValueError("Campaign report exceeds the task limit.")
    source = _source(detail.get("source_seal"))
    runtime = _runtime(detail.get("runtime_seal"))
    disk = detail.get("disk_estimate") if isinstance(detail.get("disk_estimate"), dict) else {}
    disk_estimate = {key: disk.get(key) if _number(disk.get(key), nonnegative=True) else None for key in
                     ("task_copy_bytes", "reserve_bytes", "required_free_bytes", "free_bytes")}
    disk_estimate.update({key: disk.get(key) if type(disk.get(key)) is bool else None
                          for key in ("model_size_known", "ready")})
    rows = [_row(step, index, detail, source, runtime) for index, step in enumerate(detail["steps"])]
    for field in ("step_id", "job_id"):
        seen = {}
        for row in rows:
            if row[field] is not None:
                seen.setdefault(row[field], []).append(row)
        for duplicates in seen.values():
            if len(duplicates) > 1:
                for row in duplicates:
                    row["exclusion_reasons"].append("duplicate_" + field)
                    row["comparison_eligible"] = False
    buckets = {}
    for row in rows:
        if row["comparison_eligible"]:
            identity = {key: row[key] for key in ("task_id", "dataset_fingerprint", "comparison_key", "split")}
            identity.update(metric_name=row["metric"]["name"], higher_is_better=row["metric"]["higher_is_better"])
            canonical = _canonical(identity)
            buckets.setdefault(canonical, (identity, []))[1].append(row)
    groups = []
    for canonical, (identity, members) in buckets.items():
        # Missing dates use persisted plan order. Equal scores keep the earliest
        # control/result, so downloading the report cannot change the winner.
        order_basis = "started_at" if all(row["started_at"] is not None for row in members) else "persisted_plan_order"
        members.sort(key=lambda r: (r["started_at"], r["position"]) if order_basis == "started_at" else (r["position"],))
        control = members[0]
        direction = 1 if identity["higher_is_better"] else -1
        best = max(members, key=lambda r: direction * r["metric"]["value"])
        gain = direction * (best["metric"]["value"] - control["metric"]["value"])
        gain = gain if _number(gain) else None
        denominator = abs(control["metric"]["value"])
        percent = 100 * (gain / denominator) if gain is not None and denominator else None
        percent = percent if _number(percent) else None
        group_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        for row in members:
            row["comparison_group"] = group_id
        groups.append({"id": group_id, **identity, "count": len(members), "comparison_available": len(members) > 1,
                       "control": _point(control),
                       "best": _point(best), "gain": gain, "gain_percent": percent,
                       "budget_curve": [_point(row) for row in members], "order_basis": order_basis, "scope": SCOPE})
    counts = {name: sum(row["status"] == name for row in rows) for name in
              ("completed", "failed", "skipped", "interrupted", "cancelled")}
    known_elapsed = [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] is not None]
    observed_elapsed = sum(known_elapsed)
    report = {"version": 1, "scope": SCOPE, "campaign": {
        "id": _identifier(detail.get("id")), "name": _text(detail.get("name"), 160),
        "preset": _text(detail.get("preset"), 120), "status": _text(detail.get("status"), 80),
        "created_at": _timestamp(detail.get("created_at")), "started_at": _timestamp(detail.get("started_at")),
        "finished_at": _timestamp(detail.get("finished_at")), "source_seal": source, "runtime_seal": runtime,
        "plan_sha256": _hash(detail.get("plan_sha256")),
        "disk_estimate": disk_estimate,
        "task_fingerprints": {task: _hash(value) for task, value in fingerprints.items()
                              if isinstance(task, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task)},
        "wall_budget_seconds": detail.get("wall_budget_seconds") if _number(detail.get("wall_budget_seconds"), nonnegative=True) else None,
        "elapsed_seconds": detail.get("elapsed_seconds") if _number(detail.get("elapsed_seconds"), nonnegative=True) else None,
        "worker_upper_bound_seconds": detail.get("worker_upper_bound_seconds") if _number(detail.get("worker_upper_bound_seconds"), nonnegative=True) else None,
        "error": _text(detail.get("error"), 2000)},
        "summary": {"steps": len(rows), **counts, "unfinished": len(rows) - sum(counts.values()),
                    "comparison_eligible": sum(row["comparison_eligible"] for row in rows),
                    "comparison_groups": len(groups), "observed_job_elapsed_seconds":
                        observed_elapsed if _number(observed_elapsed) and (known_elapsed or not rows) else None,
                    "elapsed_unknown_steps": len(rows) - len(known_elapsed)},
        "warnings": [_text(value) for value in detail.get("warnings", [])[:32] if isinstance(value, str)]
                    if isinstance(detail.get("warnings"), list) else [],
        "steps": rows, "groups": groups}
    _json_bytes(report)
    return report


def _json_bytes(report):
    try:
        content = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Campaign report contains invalid JSON values.") from error
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Campaign report exceeds 2 MiB.")
    return content


def _csv_cell(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    # Excel and LibreOffice can treat quoted text as a formula. Escape strings
    # even after whitespace/control characters; numeric metrics remain numbers.
    return "'" + value if re.match(r"^[\s\ufeff\u200b]*[=+\-@]", value) else value


def _csv_bytes(report):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in report["steps"]:
        metric = row.get("metric") or {}
        record = {key: row.get(key) for key in CSV_FIELDS}
        record.update(metric_name=metric.get("name"), metric_value=metric.get("value"),
                      higher_is_better=metric.get("higher_is_better"),
                      exclusion_reasons="; ".join(row["exclusion_reasons"]))
        writer.writerow({key: _csv_cell(value) for key, value in record.items()})
    return stream.getvalue().encode("utf-8-sig")


def _md(value):
    if type(value) is bool:
        value = "да" if value else "нет"
    text = escape(str(value if value is not None else "неизвестно"), quote=True).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_|{}\[\]()!#])", r"\\\1", text)


def _markdown_bytes(report):
    campaign, summary = report["campaign"], report["summary"]
    source, runtime = campaign["source_seal"], campaign["runtime_seal"]
    source_confirmed = None if source["valid"] is None else bool(source["valid"] and source["sha"] and source["source_fingerprint"])
    runtime_confirmed = None if runtime["valid"] is None else runtime["valid"] and runtime["integrity_verified"]
    reasons = {"not_completed": "запуск не завершён", "not_local": "результат получен вне этой лаборатории",
        "output_not_verified": "формат результата не подтверждён", "source_seal_not_verified": "целостность версии не подтверждена",
        "runtime_seal_not_verified": "не подтверждены Python и версии зависимостей",
        "step_not_admitted": "проверки задания не разрешили сравнение", "job_binding_not_verified": "связь задания и результата не подтверждена",
        "dataset_not_verified": "версия данных не совпадает или не подтверждена", "metric_invalid": "метрика некорректна или отсутствует",
        "split_invalid": "проверочные строки не подтверждены", "comparison_key_invalid": "отсутствует корректный отпечаток сравнения",
        "comparison_key_mismatch": "отпечаток сравнения не соответствует результату",
        "duplicate_step_id": "повторная запись задания", "duplicate_job_id": "повторная запись одного запуска"}
    lines = ["# Отчёт о серии экспериментов", "", _md(campaign["name"]), "",
             "Локальная исследовательская проверка на известных строках TRAIN. "
             "Выбор модели по этим строкам не доказывает качество на скрытом TEST и не означает баллы платформы.", "",
             f"Статус: {_md(campaign['status'])}. Целостность версии подтверждена: {_md(source_confirmed)}.",
             f"Версия исходного проекта: {_md(campaign['source_seal']['sha'])}.",
             f"Python: {_md(campaign['runtime_seal']['python_version'])}; "
             f"разрядность: {_md(campaign['runtime_seal']['pointer_bits'])}; "
             f"среда подтверждена: {_md(runtime_confirmed)}.",
             f"Заданий: {summary['steps']}; завершено: {summary['completed']}; ошибок: {summary['failed']}; "
             f"пропущено: {summary['skipped']}; прервано: {summary['interrupted']}; отменено: {summary['cancelled']}; "
             f"ещё не завершено: {summary['unfinished']}.",
             f"Допущено к сравнению: {summary['comparison_eligible']}; отдельных групп сравнения: {summary['comparison_groups']}.",
             f"Время всей серии, секунд: {_md(campaign['elapsed_seconds'])}; "
             f"общий лимит, секунд: {_md(campaign['wall_budget_seconds'])}.",
             f"Известное время отдельных запусков, секунд: {_md(summary['observed_job_elapsed_seconds'])}; "
             f"заданий с неизвестным временем: {summary['elapsed_unknown_steps']}.", "",
             "Контроль — первый сопоставимый результат по времени запуска; при неизвестных датах используется порядок плана. "
             "Положительный прирост означает улучшение. Процент делится на модуль контрольной метрики; при нулевом контроле он не вычисляется.",
             "Метрики разных задач не складываются. Таблица бюджетов использует те же известные проверочные строки и остаётся исследовательской.", ""]
    if campaign["error"]:
        lines += ["Ошибка серии: " + _md(campaign["error"]), ""]
    if report["warnings"]:
        lines += ["Предупреждения:", "", *["- " + _md(value) for value in report["warnings"]], ""]
    if not report["groups"]:
        lines += ["Нет результатов, допущенных к сравнению. Причины указаны по каждому заданию ниже.", ""]
    for group in report["groups"]:
        lines += [f"## {_md(group['task_id'])}: {_md(group['metric_name'])}", ""]
        if not group["comparison_available"]:
            lines += ["Есть только один допущенный запуск; сравнение разных бюджетов или моделей пока отсутствует.", ""]
        lines += [
                  f"Контроль: {_md(group['control']['metric']['value'])}; лучший результат: {_md(group['best']['metric']['value'])}; "
                  f"прирост: {_md(group['gain'])}; прирост, %: {_md(group['gain_percent'])}. "
                  f"Порядок контроля: {'по времени запуска' if group['order_basis'] == 'started_at' else 'по сохранённому плану'}.", "",
                  "| Бюджет, минут | Прошло, секунд | Метрика | Модель |", "| --- | --- | --- | --- |"]
        lines += ["| " + " | ".join(_md(v) for v in (point["minutes"], point["elapsed_seconds"], point["metric"]["value"], point["model"])) + " |"
                  for point in group["budget_curve"]]
        lines.append("")
    lines += ["## Диагностика заданий", "", "| Задание | Статус | Допущено к сравнению | Причины / ошибка |", "| --- | --- | --- | --- |"]
    lines += ["| " + " | ".join(_md(v) for v in (row["label"] or row["position"], row["status"], row["comparison_eligible"],
              "; ".join(reasons.get(reason, reason) for reason in row["exclusion_reasons"])
              + (": " + row["error"] if row["error"] else ""))) + " |" for row in report["steps"]]
    return ("\n".join(lines) + "\n").encode("utf-8")


def report_bytes(report, format="json"):
    """Return fixed downloads; accepts only a build_campaign_report result."""
    if not isinstance(format, str) or format not in {"json", "csv", "zip"}:
        raise ValueError("Supported campaign report formats: json, csv, zip.")
    content = _json_bytes(report)
    if format == "json":
        return content
    if format == "csv":
        return _csv_bytes(report)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, payload in (("report.json", content), ("report.csv", _csv_bytes(report)),
                              ("report.md", _markdown_bytes(report))):
            archive.writestr(name, payload)
    return output.getvalue()
