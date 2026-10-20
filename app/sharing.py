"""Exchange bounded, unverified experiment summaries without executable artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading


FORMAT = "alfa-lab-reports"
VERSION = 1
MAX_REPORTS = 200
MAX_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024
MAX_SAFE_INTEGER = (1 << 53) - 1
FINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
ACTIONS = {"baseline", "extra_trees", "agent"}
AGENT_STATUSES = {"running", "completed", "fallback", "failed"}
AGENT_REASONS = {"time_limit", "api_error", "truncated_response", "model_finished", "step_limit", "unknown"}
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")
_RECORD_FILE = re.compile(r"([0-9a-f]{64})\.json\Z")
_LOCK = threading.RLock()


def _link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _text(value, field: str, limit: int, *, default=""):
    if value is None:
        return default
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(f"Поле {field} должно быть строкой длиной до {limit} символов.")
    return value


def _hash(value, field: str, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"Поле {field} должно содержать SHA-256 данных.")
    return value.lower()


def _number(value, field: str, *, minimum=None, maximum=MAX_SAFE_INTEGER, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (integer and not isinstance(value, int))
            or not -MAX_SAFE_INTEGER <= value <= maximum
            or (isinstance(value, float) and not math.isfinite(value))
            or (minimum is not None and value < minimum)):
        raise ValueError(f"Поле {field} содержит недопустимое число.")
    return value


def _timestamp(value, field: str):
    if value is None:
        return None
    text = _text(value, field, 64)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Поле {field} содержит некорректную дату ISO 8601.") from error
    return text


def _metric(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Метрика должна быть объектом или null.")
    name = _text(value.get("name"), "metric.name", 60)
    if not name.strip() or not isinstance(value.get("higher_is_better"), bool):
        raise ValueError("У метрики нужны название и логическое направление улучшения.")
    return {"name": name,
            "value": _number(value.get("value"), "metric.value"),
            "higher_is_better": value["higher_is_better"]}


def _split(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Разбиение должно быть объектом или null.")
    result = {}
    if "seed" in value:
        result["seed"] = _number(value["seed"], "split.seed", minimum=0,
                                 maximum=(1 << 32) - 1, integer=True)
    if "validation_fraction" in value:
        fraction = _number(value["validation_fraction"], "split.validation_fraction",
                           minimum=0, maximum=1)
        if not 0 < fraction < 1:
            raise ValueError("Доля валидации должна быть больше 0 и меньше 1.")
        result["validation_fraction"] = fraction
    if "stratified" in value:
        if not isinstance(value["stratified"], bool):
            raise ValueError("Поле split.stratified должно быть логическим значением.")
        result["stratified"] = value["stratified"]
    if "validation_rows" in value:
        result["validation_rows"] = _number(value["validation_rows"], "split.validation_rows",
                                            minimum=1, maximum=(1 << 31) - 1, integer=True)
    if "validation_hash" in value:
        result["validation_hash"] = _hash(value["validation_hash"], "split.validation_hash")
    return result


def _summary(raw: dict, fingerprint: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Каждый отчёт должен быть JSON-объектом.")
    identifier = raw.get("id")
    # Re-exporting an imported record must preserve its original content identity.
    if raw.get("source") == "imported" and isinstance(identifier, str) and identifier.startswith("import-"):
        identifier = raw.get("origin_id")
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("У отчёта некорректный идентификатор.")
    if (not isinstance(raw.get("action"), str) or raw["action"] not in ACTIONS
            or not isinstance(raw.get("status"), str) or raw["status"] not in FINAL_STATUSES):
        raise ValueError("Можно передавать только завершённые запуски baseline, extra_trees или agent.")
    if _hash(raw.get("dataset_fingerprint"), "dataset_fingerprint") != fingerprint:
        raise ValueError("Отчёт относится к другим данным. Скачайте одинаковую версию задачи и экспортируйте заново.")
    output_valid = raw.get("output_valid")
    if output_valid is not None and not isinstance(output_valid, bool):
        raise ValueError("Поле output_valid должно быть логическим значением или null.")
    api_used = raw.get("api_used", raw["action"] == "agent")
    if not isinstance(api_used, bool):
        raise ValueError("Поле api_used должно быть логическим значением.")
    result = {
        "id": identifier, "action": raw["action"], "status": raw["status"],
        "stage": _text(raw.get("stage"), "stage", 120),
        "author": _text(raw.get("author"), "author", 80),
        "hypothesis": _text(raw.get("hypothesis"), "hypothesis", 500),
        "started_at": _timestamp(raw.get("started_at"), "started_at"),
        "finished_at": _timestamp(raw.get("finished_at"), "finished_at"),
        "model": _text(raw.get("model"), "model", 120, default=None),
        "metric": _metric(raw.get("metric")),
        "comparison_key": _hash(raw.get("comparison_key"), "comparison_key", optional=True),
        "dataset_fingerprint": fingerprint, "split": _split(raw.get("split")),
        "error": "",
        "output_valid": output_valid, "api_used": api_used,
    }
    # Raw process/API errors can contain credentials, paths or response bodies.
    # User-authored hypotheses remain exact; machine diagnostics stay local.
    _text(raw.get("error"), "error", 1000)
    if raw["status"] == "failed":
        result["error"] = "Запуск завершился с ошибкой"
    elif raw["status"] in {"cancelled", "interrupted"}:
        result["error"] = "Запуск прерван"
    if "elapsed_seconds" in raw:
        result["elapsed_seconds"] = _number(raw["elapsed_seconds"], "elapsed_seconds",
                                            minimum=0, maximum=10**9)
    if raw.get("exit_code") is not None:
        result["exit_code"] = _number(raw["exit_code"], "exit_code", minimum=-(1 << 31),
                                     maximum=(1 << 32) - 1, integer=True)
    if raw.get("agent_status") is not None:
        agent_status = raw["agent_status"]
        if not isinstance(agent_status, str) or agent_status not in AGENT_STATUSES:
            raise ValueError("У исследования Qwen некорректный статус.")
        result["agent_status"] = agent_status
    if raw.get("agent_reason") is not None:
        reason = _text(raw["agent_reason"], "agent_reason", 120)
        # Only stable reason codes travel; arbitrary machine diagnostics stay local.
        result["agent_reason"] = reason if reason in AGENT_REASONS else "unknown"
    return result


def _encoded(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("Отчёт содержит некорректные JSON-данные.") from error


def _check_tree(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 16 or count > 50_000:
            raise ValueError("Отчёт содержит слишком сложную структуру.")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Ключи JSON должны быть строками.")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            _number(item, "JSON")
        elif item is not None and not isinstance(item, (str, bool)):
            raise ValueError("Отчёт содержит неподдерживаемый тип данных.")
    if len(_encoded(value)) > MAX_BYTES:
        raise ValueError("Файл отчёта превышает допустимый размер 1 МБ.")


def _digest(summary: dict) -> str:
    return hashlib.sha256(_encoded(summary)).hexdigest()


class ReportStore:
    """Persist imported summaries only; never load paths or execute report content."""

    def __init__(self, state_root: Path):
        self.root = Path(state_root).absolute()
        self._safe(self.root)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Хранилище отчётов должно быть папкой.")

    def _safe(self, path: Path) -> Path:
        if ".." in path.parts:
            raise ValueError("Недопустимый путь хранилища отчётов.")
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if _link(current):
                raise ValueError("Ссылки и junction-папки в хранилище отчётов запрещены.")
            if current != path and current.exists() and not current.is_dir():
                raise ValueError("Родительский путь хранилища должен быть папкой.")
        if path != self.root and not path.is_relative_to(self.root):
            raise ValueError("Доступ за пределы хранилища отчётов запрещён.")
        return path

    def _folder(self, task_id: str, *, create=False) -> Path:
        task_id = _text(task_id, "task_id", 512)
        if not task_id:
            raise ValueError("Не указана задача для отчётов.")
        task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        folder = self._safe(self.root / "imports" / task_hash)
        if create:
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise ValueError("Не удалось создать папку для отчётов.") from error
            self._safe(folder)
        if folder.exists() and not folder.is_dir():
            raise ValueError("Хранилище отчётов должно быть папкой.")
        return folder

    def export(self, task_id: str, dataset_fingerprint: str, jobs: list[dict]) -> dict:
        task_id = _text(task_id, "task_id", 512)
        if not task_id:
            raise ValueError("Не указана задача для экспорта.")
        fingerprint = _hash(dataset_fingerprint, "dataset_fingerprint")
        if not isinstance(jobs, list):
            raise ValueError("Для экспорта нужен список запусков.")
        completed = [job for job in jobs if isinstance(job, dict)
                     and isinstance(job.get("status"), str) and job["status"] in FINAL_STATUSES
                     and isinstance(job.get("action"), str) and job["action"] in ACTIONS
                     and isinstance(job.get("dataset_fingerprint"), str)
                     and job["dataset_fingerprint"].lower() == fingerprint]
        if len(completed) > MAX_REPORTS:
            raise ValueError("За один раз можно экспортировать не более 200 отчётов.")
        reports = [_summary({**job, "error": ""}, fingerprint) for job in completed]
        result = {"format": FORMAT, "version": VERSION, "task_id": task_id,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "dataset_fingerprint": fingerprint, "reports": reports}
        _check_tree(result)
        return result

    def import_reports(self, task_id: str, dataset_fingerprint: str, envelope: dict) -> dict:
        fingerprint = _hash(dataset_fingerprint, "dataset_fingerprint")
        _check_tree(envelope)
        if (not isinstance(envelope, dict) or envelope.get("format") != FORMAT
                or type(envelope.get("version")) is not int or envelope["version"] != VERSION):
            raise ValueError("Нужен файл Alfa Lab Reports версии 1.")
        if _hash(envelope.get("dataset_fingerprint"), "dataset_fingerprint") != fingerprint:
            raise ValueError("Данные задачи отличаются. Скачайте одинаковую версию данных у отправителя и получателя.")
        if not _text(envelope.get("task_id"), "task_id", 512):
            raise ValueError("В файле не указана исходная задача.")
        if _timestamp(envelope.get("created_at"), "created_at") is None:
            raise ValueError("В файле не указана дата экспорта.")
        reports = envelope.get("reports")
        if not isinstance(reports, list) or len(reports) > MAX_REPORTS:
            raise ValueError("Файл должен содержать список из не более 200 отчётов.")
        # Validate every record before writing anything, preventing partial malformed imports.
        summaries = [_summary(raw, fingerprint) for raw in reports]
        records = []
        for summary in summaries:
            digest = _digest(summary)
            record = {**summary, "id": "import-" + digest, "origin_id": summary["id"],
                      "source": "imported", "published": False, "verified": False}
            records.append((digest, record))
        imported, skipped = 0, 0
        with _LOCK:
            folder = self._folder(task_id, create=bool(records))
            for digest, record in records:
                path = self._safe(folder / (digest + ".json"))
                if path.exists():
                    if not path.is_file():
                        raise ValueError("Файл отчёта занят другим объектом.")
                    existing = self._read(path, digest)
                    if existing == record:
                        skipped += 1
                        continue
                raw = _encoded(record)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(prefix="report-", suffix=".tmp", dir=folder,
                                                     delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(raw)
                    self._safe(folder)
                    self._safe(path)
                    os.replace(temporary, path)
                except OSError as error:
                    raise ValueError("Не удалось сохранить импортированный отчёт.") from error
                finally:
                    if temporary is not None:
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            pass
                imported += 1
        return {"imported": imported, "skipped": skipped}

    def _read(self, path: Path, digest: str) -> dict | None:
        self._safe(path)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_RECORD_BYTES + 1)
            if len(raw) > MAX_RECORD_BYTES:
                return None
            record = json.loads(raw.decode("utf-8"))
            _check_tree(record)
            if (not isinstance(record, dict) or record.get("source") != "imported"
                    or record.get("published") is not False or record.get("verified") is not False
                    or record.get("id") != "import-" + digest):
                return None
            fingerprint = _hash(record.get("dataset_fingerprint"), "dataset_fingerprint")
            summary = _summary(record, fingerprint)
            if _digest(summary) != digest:
                return None
            # Reconstruct the allowlist, so edited files cannot inject extra fields.
            return {**summary, "id": "import-" + digest, "origin_id": summary["id"],
                    "source": "imported", "published": False, "verified": False}
        except (OSError, ValueError, TypeError, RecursionError):
            return None

    def list(self, task_id: str) -> list[dict]:
        with _LOCK:
            folder = self._folder(task_id)
            if not folder.exists():
                return []
            candidates = []
            with os.scandir(folder) as entries:
                for entry in entries:
                    match = _RECORD_FILE.fullmatch(entry.name)
                    if not match:
                        continue
                    path = self._safe(Path(entry.path))
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        stamp = entry.stat(follow_symlinks=False).st_mtime_ns
                    except OSError:
                        continue
                    heapq.heappush(candidates, (stamp, path, match.group(1)))
                    if len(candidates) > MAX_REPORTS * 4:
                        heapq.heappop(candidates)
            result = []
            for _, path, digest in sorted(candidates, reverse=True):
                record = self._read(path, digest)
                if record is not None:
                    result.append(record)
                    if len(result) == MAX_REPORTS:
                        break
            return result
