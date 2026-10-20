"""Fixed local training actions, isolated workspaces and explicit publication."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from app.profiling import DataCatalog, _metric
from app.runs import RunReader, _check_json_tree
from app.splits import normalize_validation
from app.model_config import make_model_config, validate_model_config


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {"baseline", "extra_trees", "agent", "prepare"}
ACTIVE = {"queued", "running"}
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
API_FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
MAX_LOG_BYTES = 20 * 1024 * 1024
LOG_TAIL_BYTES = 128 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
LOCAL_TIMEOUT = 1800
PREPARE_TIMEOUT = 600
POLL_SECONDS = 0.2
PUBLIC_FIELDS = {"id", "task_id", "action", "status", "stage", "author", "hypothesis", "started_at",
                 "finished_at", "elapsed_seconds", "model", "metric", "comparison_key", "dataset_fingerprint",
                 "split", "error", "output_valid", "published", "source", "exit_code", "api_used",
                 "agent_status", "agent_reason", "agent_run_id", "api", "published_at", "validation", "model_config"}
REQUIRED_FIELDS = PUBLIC_FIELDS - {"agent_status", "agent_reason", "agent_run_id", "api", "published_at", "validation", "model_config"}
SPLIT_TEXT_FIELDS = {"time_column": 500, "train_period_min": 100, "train_period_max": 100,
                     "validation_period": 100}
SPLIT_COUNT_FIELDS = {"group_count", "train_groups", "validation_groups", "group_overlap", "n_splits", "fold"}
SPLIT_FIELDS = {"seed", "validation_fraction", "stratified", "validation_rows", "validation_hash",
                "train_fraction", "shuffle", "train_rows", "strategy", *SPLIT_TEXT_FIELDS, *SPLIT_COUNT_FIELDS}


def _valid_split(split):
    """Accept legacy holdouts and bounded evidence from the supported recipes."""
    if (not isinstance(split, dict) or type(split.get("seed")) is not int
            or type(split.get("stratified")) is not bool or type(split.get("validation_rows")) is not int
            or split["validation_rows"] < 1 or isinstance(split.get("validation_fraction"), bool)
            or not isinstance(split.get("validation_fraction"), (int, float))
            or not 0 < split["validation_fraction"] < 1
            or not isinstance(split.get("validation_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", split["validation_hash"])):
        return False
    if "train_fraction" in split and (isinstance(split["train_fraction"], bool)
            or not isinstance(split["train_fraction"], (int, float))
            or not 0 < split["train_fraction"] < 1):
        return False
    if "train_rows" in split and (type(split["train_rows"]) is not int
                                  or not 1 <= split["train_rows"] <= (1 << 31) - 1):
        return False
    if "shuffle" in split and type(split["shuffle"]) is not bool:
        return False
    if "strategy" in split and (not isinstance(split["strategy"], str)
                                 or split["strategy"] not in {"random", "time", "grouped"}):
        return False
    for name, limit in SPLIT_TEXT_FIELDS.items():
        if name in split and (not isinstance(split[name], str) or not split[name].strip()
                              or len(split[name]) > limit):
            return False
    for name in SPLIT_COUNT_FIELDS:
        if name in split and (type(split[name]) is not int or not 0 <= split[name] <= (1 << 31) - 1):
            return False
    if split.get("strategy") == "time" and not SPLIT_TEXT_FIELDS.keys() <= split.keys():
        return False
    if split.get("strategy") == "grouped":
        if (not SPLIT_COUNT_FIELDS <= split.keys() or not 2 <= split["n_splits"] <= 5
                or not 0 <= split["fold"] < split["n_splits"] or split["group_overlap"] != 0):
            return False
    return True


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _safe(path: Path, root: Path, *, required=False, directory=False):
    path, root = Path(path).absolute(), Path(root).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Путь находится за пределами рабочей папки.") from error
    current = root
    for part in (None, *relative.parts):
        current = current if part is None else current / part
        if part in {".", ".."} or _linked(current):
            raise ValueError("Ссылки и переходы за пределы папки запрещены.")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Путь находится за пределами рабочей папки.")
    if required and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Не найден файл или каталог: {path.name}")
    return path


def _json(path: Path, root: Path, limit=2 * 1024 * 1024):
    path = _safe(path, root, required=True)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Служебный отчёт превышает допустимый размер.")
    try:
        result = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Повреждён служебный отчёт запуска.") from error
    if not isinstance(result, dict):
        raise ValueError("Служебный отчёт должен быть JSON-объектом.")
    _check_json_tree(result)
    return result


def _write_json(path: Path, value, root: Path):
    _safe(path, root)
    temporary = path.with_name(path.name + ".tmp")
    _safe(temporary, root)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
    os.replace(temporary, path)


def _settings_values():
    values = {name: os.environ.get(name, "").strip() for name in API_FIELDS}
    local = ROOT / ".env"
    if local.is_file() and not _linked(local):
        from dotenv import dotenv_values
        configured = dotenv_values(local)
        for name in API_FIELDS:
            if name in configured:
                values[name] = str(configured[name] or "").strip()
    return values


def api_settings():
    values = _settings_values()
    missing = [name for name, value in values.items() if not value]
    return {"ready": not missing, "missing": missing,
            "model_configured": bool(values["OPENAI_MODEL"])}


def _redact(text, key=""):
    text = str(text)
    for secret in (key, os.environ.get("OPENAI_API_KEY", "")):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _identity(pid: int):
    """An OS process birth marker prevents killing a reused PID on restart."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            creation, exit_time, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(system), ctypes.byref(user)):
                return None
            return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def _descendants(pid: int):
    pairs = []
    proc = Path("/proc")
    if proc.is_dir():
        for path in proc.iterdir():
            if not path.name.isdigit():
                continue
            try:
                text = (path / "stat").read_text()
                fields = text[text.rfind(")") + 2:].split()
                pairs.append((int(path.name), int(fields[1])))
            except (OSError, ValueError, IndexError):
                continue
    else:
        try:
            result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True,
                                    text=True, timeout=1, check=False)
            for line in result.stdout.splitlines():
                child, parent = line.split()
                pairs.append((int(child), int(parent)))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return []
    found = {pid}
    while True:
        new = {child for child, parent in pairs if parent in found} - found
        if not new:
            return list(found - {pid})
        found.update(new)


def _kill_tree(pid: int):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    # The core agent creates a fresh session for each experiment. Enumerate the
    # descendant tree, not just the manager's original process group.
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    descendants = set()
    for _ in range(3):
        found = set(_descendants(pid))
        for child in found - descendants:
            try:
                os.kill(child, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        if found <= descendants:
            break
        descendants.update(found)
    for child in descendants:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class JobManager:
    """One active job; every training action works on its own input copies."""

    def __init__(self, data_root: Path, state_root: Path, catalog: DataCatalog):
        self.data_root = Path(data_root).absolute()
        self.state_root = Path(state_root).absolute()
        self.catalog = catalog
        for path in (self.data_root, self.state_root):
            for ancestor in (path, *path.parents):
                if _linked(ancestor):
                    raise ValueError("Рабочая папка не должна содержать символические ссылки.")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = _safe(self.state_root / "jobs", self.state_root)
        self.jobs_root.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._active = None
        self._closed = False
        self._jobs = {}
        self._publication_cache = {}
        self._secrets = {}
        self._lock_stream = None
        self._take_lock()
        try:
            self._restore()
        except Exception:
            self._release_lock()
            raise

    def _take_lock(self):
        path = _safe(self.state_root / "manager.lock", self.state_root)
        stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0, 2)
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise ValueError("Эта рабочая папка уже используется другим сервером приложения.") from error
        self._lock_stream = stream

    def _release_lock(self):
        if self._lock_stream is not None:
            if os.name == "nt":
                import msvcrt
                self._lock_stream.seek(0)
                msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
            self._lock_stream.close()
            self._lock_stream = None

    def _restore(self):
        for folder in self.jobs_root.iterdir():
            if not JOB_ID.fullmatch(folder.name) or _linked(folder) or not folder.is_dir():
                continue
            try:
                job = _json(folder / "job.json", self.state_root, 256 * 1024)
                if not self._valid_stored_job(job, folder.name):
                    continue
                job = {key: value for key, value in job.items() if key in PUBLIC_FIELDS or key in {
                    "_runtime", "_published_sha256", "_output_sha256", "_previous_submission"}}
                if job["status"] in ACTIVE:
                    runtime = job.get("_runtime") or {}
                    pid, identity = runtime.get("pid"), runtime.get("identity")
                    if isinstance(pid, int) and pid > 1 and identity and _identity(pid) == identity:
                        _kill_tree(pid)
                    job.update(status="interrupted", stage="finished", finished_at=_now(),
                               error="Сервер был перезапущен во время выполнения.")
                    job.pop("_runtime", None)
                    self._save(job)
                self._jobs[job["id"]] = job
            except (OSError, ValueError, TypeError):
                continue

    @staticmethod
    def _valid_stored_job(job, job_id):
        if (not REQUIRED_FIELDS <= job.keys() or job.get("id") != job_id
                or job.get("action") not in ACTIONS
                or job.get("status") not in ACTIVE | {"completed", "failed", "cancelled", "interrupted"}
                or job.get("source") != "local"):
            return False
        task_id = job.get("task_id")
        if (job["action"] == "prepare" and task_id is not None
                or job["action"] != "prepare" and (not isinstance(task_id, str)
                                                  or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id))):
            return False
        if "validation" in job:
            if job["action"] not in {"baseline", "extra_trees"}:
                return False
            try:
                if normalize_validation(job["validation"]) != job["validation"]:
                    return False
            except ValueError:
                return False
        if "model_config" in job:
            if job["action"] not in {"baseline", "extra_trees"}:
                return False
            try:
                config = validate_model_config(job["model_config"], action=job["action"])
                if config != job["model_config"] or job.get("model") not in (None, config["model"]):
                    return False
            except ValueError:
                return False
        for name, limit in (("author", 80), ("hypothesis", 500), ("stage", 40), ("started_at", 64)):
            if not isinstance(job.get(name), str) or len(job[name]) > limit:
                return False
        for name in ("finished_at", "model", "error", "agent_reason", "agent_status", "published_at"):
            if job.get(name) is not None and (not isinstance(job[name], str) or len(job[name]) > 2000):
                return False
        try:
            datetime.fromisoformat(job["started_at"])
            if job["finished_at"] is not None:
                datetime.fromisoformat(job["finished_at"])
        except ValueError:
            return False
        if (type(job["published"]) is not bool or type(job["api_used"]) is not bool
                or job["output_valid"] is not None and type(job["output_valid"]) is not bool
                or isinstance(job["elapsed_seconds"], bool)
                or not isinstance(job["elapsed_seconds"], (int, float)) or job["elapsed_seconds"] < 0
                or job["exit_code"] is not None and type(job["exit_code"]) is not int):
            return False
        for name in ("comparison_key", "dataset_fingerprint", "_published_sha256", "_output_sha256"):
            if job.get(name) is not None and (not isinstance(job[name], str) or not re.fullmatch(r"[a-f0-9]{64}", job[name])):
                return False
        metric = job.get("metric")
        if metric is not None and (not isinstance(metric, dict) or not isinstance(metric.get("name"), str)
                or type(metric.get("higher_is_better")) is not bool or isinstance(metric.get("value"), bool)
                or not isinstance(metric.get("value"), (int, float))
                or not -((1 << 53) - 1) <= metric["value"] <= (1 << 53) - 1):
            return False
        split = job.get("split")
        if split is not None and not _valid_split(split):
            return False
        if (split is not None and "validation" in job
                and split.get("strategy", "random") != job["validation"]["strategy"]):
            return False
        if (split is not None and job.get("validation", {}).get("strategy") == "time"
                and split.get("time_column") != job["validation"]["time_column"]):
            return False
        api = job.get("api")
        if api is not None:
            if not isinstance(api, dict) or api.get("state") not in {"none", "unknown", "partial", "reported"}:
                return False
            for name in ("attempts", "responses", "reported_responses", "prompt_tokens", "completion_tokens", "total_tokens"):
                if api.get(name) is not None and (type(api[name]) is not int or api[name] < 0):
                    return False
        if job.get("agent_run_id") is not None and (not isinstance(job["agent_run_id"], str) or not JOB_ID.fullmatch(job["agent_run_id"])):
            return False
        runtime = job.get("_runtime")
        if runtime is not None and (not isinstance(runtime, dict) or type(runtime.get("pid")) is not int
                or runtime["pid"] <= 1 or runtime.get("identity") is not None and not isinstance(runtime["identity"], str)):
            return False
        return True

    def _save(self, job):
        _write_json(self.jobs_root / job["id"] / "job.json", job, self.state_root)

    def _summary(self, job):
        summary = {key: deepcopy(value) for key, value in job.items() if key in PUBLIC_FIELDS}
        if isinstance(summary.get("metric"), dict):
            summary["metric"] = {key: value for key, value in summary["metric"].items()
                                 if key in {"name", "value", "higher_is_better"}}
        if isinstance(summary.get("split"), dict):
            summary["split"] = {key: value for key, value in summary["split"].items()
                                if key in SPLIT_FIELDS}
        if isinstance(summary.get("validation"), dict):
            summary["validation"] = {key: value for key, value in summary["validation"].items()
                                     if key in {"strategy", "time_column"}}
        if "model_config" in summary:
            try:
                summary["model_config"] = validate_model_config(summary["model_config"], action=job["action"])
            except ValueError:
                summary.pop("model_config")
        if isinstance(summary.get("api"), dict):
            summary["api"] = {key: value for key, value in summary["api"].items()
                              if key in {"state", "attempts", "responses", "reported_responses", "prompt_tokens",
                                         "completion_tokens", "total_tokens"}}
        if summary.get("published"):
            summary["published"] = self._publication_current(job)
        active = self._active
        if active and active["id"] == job["id"] and job["status"] in ACTIVE:
            summary["elapsed_seconds"] = round(time.monotonic() - active["started"], 3)
        return self.redact(job["task_id"], job["id"], summary)

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def _publication_current(self, job):
        try:
            if self.catalog.fingerprint(job["task_id"]) != job["dataset_fingerprint"]:
                return False
            path = _safe(self.catalog.resolve(job["task_id"]) / "submission.csv", self.data_root, required=True)
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
            cached = self._publication_cache.get(job["task_id"])
            if cached is None or cached[0] != signature:
                cached = (signature, self._file_hash(path))
                self._publication_cache[job["task_id"]] = cached
            return cached[1] == job.get("_published_sha256")
        except (OSError, ValueError):
            return False

    def active_job(self):
        with self._lock:
            return self._summary(self._jobs[self._active["id"]]) if self._active else None

    def list(self, task_id=None):
        with self._lock:
            jobs = [job for job in self._jobs.values() if task_id is None or job["task_id"] == task_id]
            return [self._summary(job) for job in sorted(jobs, key=lambda x: x["started_at"], reverse=True)[:200]]

    def _job(self, task_id, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError("Недопустимый идентификатор запуска.")
        job = self._jobs.get(job_id)
        if job is None or job.get("task_id") != task_id:
            raise ValueError("Запуск не найден для этой задачи.")
        _safe(self.jobs_root / job_id, self.state_root, required=True, directory=True)
        return job

    def start(self, task_id, action, author="", hypothesis="", minutes=175, validation=None, model_params=None):
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError("Неизвестное действие. Доступны baseline, extra_trees, agent и prepare.")
        if validation is not None and action not in {"baseline", "extra_trees"}:
            raise ValueError("Выбор валидации доступен только для baseline и ExtraTrees.")
        if model_params is not None and action not in {"baseline", "extra_trees"}:
            raise ValueError("Настройка модели доступна только для baseline и ExtraTrees.")
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("Параметры модели должны быть объектом.")
        validation = normalize_validation(validation) if action in {"baseline", "extra_trees"} else None
        model_config = None
        if not isinstance(author, str) or len(author) > 80 or not isinstance(hypothesis, str) or len(hypothesis) > 500:
            raise ValueError("Имя автора: до 80 символов; гипотеза: до 500 символов.")
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 175:
            raise ValueError("Бюджет агента должен быть целым числом от 1 до 175 минут.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Менеджер запусков остановлен.")
            if self._active:
                raise RuntimeError("Уже выполняется другой запуск. Дождитесь завершения или остановите его.")
            values = _settings_values() if action == "agent" else {}
            if action == "agent" and any(not values.get(name) for name in API_FIELDS):
                raise ValueError("Для Qwen заполните OPENAI_API_KEY, OPENAI_BASE_URL и OPENAI_MODEL в локальном .env.")
            if action == "prepare":
                if task_id is not None:
                    raise ValueError("Загрузка материалов запускается без выбранной задачи.")
                self._check_prepare_paths()
                fingerprint = None
            else:
                if not isinstance(task_id, str):
                    raise ValueError("Выберите задачу для запуска.")
                # Keep the catalog's path spelling for its lexical boundary
                # checks: resolve() expands Windows 8.3 names such as RUNNER~1.
                task_folder = self.catalog.resolve(task_id)
                task_location = task_folder.resolve()
                state_folder = self.state_root.resolve()
                if state_folder.is_relative_to(task_location) or task_location.is_relative_to(state_folder):
                    raise ValueError("Рабочая папка запусков должна находиться отдельно от исходной задачи.")
                if action in {"baseline", "extra_trees"}:
                    _, inputs = self.catalog._inputs(task_folder)
                    metadata, _, _, task_type = self.catalog._metadata(inputs)
                    expected = {"regression": "RMSE", "classification": "Macro F1"}.get(task_type)
                    if expected is None:
                        raise ValueError("Локальные рецепты поддерживают числовую регрессию и классификацию. Для другой задачи запустите агента.")
                    declared = _metric(self.catalog.task_text(task_id))["name"]
                    if declared not in {"Не определена", expected}:
                        raise ValueError(f"Этот рецепт оценивает {expected}, а в задаче указана {declared}. Для этой метрики запустите агента.")
                    if validation["strategy"] == "time":
                        if task_type != "regression":
                            raise ValueError("Хронологическая валидация доступна для регрессии.")
                        column = validation["time_column"]
                        if column not in metadata.index or metadata.loc[column, "role"] != "feature":
                            raise ValueError("Выберите существующий признак времени из словаря данных.")
                        if metadata.loc[column, "dtype"] not in {"numeric", "binary"}:
                            raise ValueError("Признак времени должен быть числовым.")
                    if validation["strategy"] == "grouped" and task_type != "classification":
                        raise ValueError("Групповая валидация доступна для классификации.")
                    model_config = make_model_config(action, task_type, model_params)
                fingerprint = self.catalog.fingerprint(task_id)
            job_id = uuid4().hex
            folder = _safe(self.jobs_root / job_id, self.state_root)
            folder.mkdir()
            job = {"id": job_id, "task_id": task_id, "action": action, "status": "queued",
                   "stage": "preparing", "author": author.strip(),
                   "hypothesis": hypothesis.strip() or ("Проверить Extra Trees на том же holdout" if action == "extra_trees" else ""),
                   "started_at": _now(), "finished_at": None, "elapsed_seconds": 0,
                   "model": None, "metric": None, "comparison_key": None,
                   "dataset_fingerprint": fingerprint, "split": None, "error": None,
                   "output_valid": None, "published": False, "source": "local",
                   "exit_code": None, "api_used": False}
            if validation is not None:
                job["validation"] = validation
            if model_config is not None:
                job["model_config"] = model_config
            self._jobs[job_id] = job
            self._save(job)
            active = {"id": job_id, "cancel": threading.Event(), "process": None,
                      "started": time.monotonic(), "key": values.get("OPENAI_API_KEY", ""), "children": {}}
            if active["key"]:
                self._secrets[job_id] = active["key"]
            self._active = active
            thread = threading.Thread(target=self._run, args=(job_id, active, minutes, values), daemon=True)
            active["thread"] = thread
            thread.start()
            return self._summary(job)

    def _check_prepare_paths(self):
        from prepare_data import TASKS
        if not self.data_root.exists():
            self.data_root.mkdir(parents=True)
        _safe(self.data_root, self.data_root, required=True, directory=True)
        for name, config in TASKS.items():
            folder = self.data_root / name
            for filename in config["files"]:
                target = folder / filename if filename == "task.md" else folder / config["folder"] / "participants" / filename
                _safe(target, self.data_root)
                _safe(target.with_name(target.name + ".download"), self.data_root)

    def _copy(self, source, destination, source_root, cancel):
        _safe(source, source_root, required=True)
        _safe(destination, self.state_root)
        if source.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("Входной файл превышает лимит 512 МБ.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, destination.open("xb") as dst:
            while block := src.read(1024 * 1024):
                if cancel.is_set():
                    raise InterruptedError("Запуск остановлен пользователем.")
                dst.write(block)

    def _stage(self, job, cancel):
        original = self.catalog.resolve(job["task_id"])
        _, inputs = self.catalog._inputs(original)
        folder = self.jobs_root / job["id"]
        stage = folder / "task"
        stage.mkdir()
        for source in (original / "task.md", *inputs.values()):
            self._copy(source, stage / source.relative_to(original), self.data_root, cancel)
        copied = DataCatalog(folder).fingerprint("task")
        if copied != job["dataset_fingerprint"] or self.catalog.fingerprint(job["task_id"]) != copied:
            raise ValueError("Данные изменились при подготовке. Начните запуск заново.")
        if job["action"] == "agent":
            reader = RunReader(original)
            existing = reader.result()
            if existing["submission"]["valid"] is True:
                self._copy(reader.submission_path(), stage / "submission.csv", self.data_root, cancel)
                if existing["baseline"] is not None:
                    _write_json(stage / "baseline_metrics.json", existing["baseline"], self.state_root)
            if job["hypothesis"]:
                with (stage / "task.md").open("a", encoding="utf-8") as stream:
                    stream.write("\n\n## Дополнительная гипотеза для исследования\n"
                                 + job["hypothesis"]
                                 + "\n\nПроверь эту гипотезу, соблюдая исходные ограничения задачи, "
                                   "честную валидацию и общий бюджет времени.\n")
        return stage / "task.md"

    def _command(self, job, task_path, minutes):
        if job["action"] == "prepare":
            return [sys.executable, "-u", str(ROOT / "prepare_data.py"), "all", "--data-dir", str(self.data_root)]
        if job["action"] == "agent":
            return [sys.executable, "-u", str(ROOT / "solution.py"), str(task_path),
                    "--max-steps", "12", "--time-budget", str(minutes * 60)]
        folder = self.jobs_root / job["id"]
        config = {"job_dir": str(folder), "task_path": str(task_path), "action": job["action"],
                  "dataset_fingerprint": job["dataset_fingerprint"],
                  "validation": normalize_validation(job.get("validation"))}
        if "model_config" in job:
            config["model_params"] = validate_model_config(job["model_config"], action=job["action"])["parameters"]
        _write_json(folder / "config.json", config, self.state_root)
        return [sys.executable, "-u", str(ROOT / "app" / "worker.py"), "--config", str(folder / "config.json")]

    def _set(self, job_id, **updates):
        with self._lock:
            job = self._jobs[job_id]
            job.update(updates)
            self._save(job)

    def _run(self, job_id, active, minutes, settings):
        job = self._jobs[job_id]
        folder = self.jobs_root / job_id
        process = None
        final_status, error = "failed", None
        try:
            task_path = None if job["action"] == "prepare" else self._stage(job, active["cancel"])
            if active["cancel"].is_set():
                raise InterruptedError("Запуск остановлен пользователем.")
            command = self._command(job, task_path, minutes)
            from agent.execution import child_environment
            environment = child_environment()
            environment.update(settings)
            timeout = PREPARE_TIMEOUT if job["action"] == "prepare" else minutes * 60 + 30 if job["action"] == "agent" else LOCAL_TIMEOUT
            stage = {"prepare": "downloading", "agent": "research"}.get(job["action"], "training")
            with (folder / "stdout.txt").open("xb") as stdout, (folder / "stderr.txt").open("xb") as stderr:
                with self._lock:
                    if self._closed or active["cancel"].is_set():
                        raise InterruptedError("Запуск остановлен пользователем.")
                    supervised = [sys.executable, "-u", str(ROOT / "app" / "supervisor.py"), "--", *command]
                    process = subprocess.Popen(supervised, cwd=ROOT, env=environment,
                                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               start_new_session=os.name != "nt",
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                    active["process"] = process
                    self._set(job_id, status="running", stage=stage,
                              _runtime={"pid": process.pid, "identity": _identity(process.pid)})
                launched = time.monotonic()
                while process.poll() is None:
                    if os.name != "nt":
                        for pid in _descendants(process.pid):
                            identity = _identity(pid)
                            if identity:
                                active["children"][pid] = identity
                    if active["cancel"].is_set():
                        raise InterruptedError("Запуск остановлен пользователем.")
                    if time.monotonic() - launched > timeout:
                        raise TimeoutError("Запуск остановлен: превышен лимит времени.")
                    for name in ("stdout.txt", "stderr.txt"):
                        path = _safe(folder / name, self.state_root, required=True)
                        if path.stat().st_size > MAX_LOG_BYTES:
                            raise ValueError("Запуск остановлен: журнал превысил лимит 20 МБ.")
                    stage_file = folder / "stage.json"
                    if job["action"] in {"baseline", "extra_trees"} and stage_file.exists():
                        current = _json(stage_file, self.state_root).get("stage")
                        if current in {"training", "validating", "refitting", "finished"} and current != job["stage"]:
                            self._set(job_id, stage=current)
                    if job["action"] == "agent":
                        with self._lock:
                            if self._collect_agent(job):
                                self._save(job)
                    active["cancel"].wait(POLL_SECONDS)
                if active["cancel"].is_set():
                    raise InterruptedError("Запуск остановлен пользователем.")
            self._set(job_id, exit_code=process.returncode)
            if any((folder / name).stat().st_size > MAX_LOG_BYTES for name in ("stdout.txt", "stderr.txt")):
                raise ValueError("Запуск остановлен: журнал превысил лимит 20 МБ.")
            if process.returncode != 0:
                raise ValueError("Процесс завершился с ошибкой. Подробности в журнале stderr.")
            self._collect(job)
            final_status = "completed"
        except InterruptedError as exception:
            final_status, error = "cancelled", str(exception)
        except Exception as exception:
            error = _redact(exception, active["key"])
        finally:
            if process is not None and process.poll() is None:
                _kill_tree(process.pid)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            for pid, identity in active["children"].items():
                if _identity(pid) == identity:
                    _kill_tree(pid)
            self._scrub_logs(folder, active["key"])
            if job["action"] == "agent":
                self._collect_agent(job)
            with self._lock:
                job.pop("_runtime", None)
                job.update(status=final_status, stage="finished", finished_at=_now(),
                           elapsed_seconds=round(time.monotonic() - active["started"], 3), error=error,
                           exit_code=process.returncode if process is not None else None)
                try:
                    self._save(job)
                finally:
                    if self._active is active:
                        self._active = None
                    if self._closed:
                        self._release_lock()

    def _scrub_logs(self, folder, key):
        for name in ("stdout.txt", "stderr.txt"):
            try:
                path = _safe(folder / name, self.state_root, required=True)
                with path.open("rb") as stream:
                    size = os.fstat(stream.fileno()).st_size
                    stream.seek(max(0, size - MAX_LOG_BYTES))
                    raw = stream.read(MAX_LOG_BYTES)
                cleaned = raw.replace(key.encode("utf-8"), b"[REDACTED]") if key else raw
                cleaned = cleaned[-MAX_LOG_BYTES:]
                if cleaned != raw or size > MAX_LOG_BYTES:
                    temporary = _safe(path.with_suffix(".redacted.tmp"), self.state_root)
                    temporary.write_bytes(cleaned)
                    os.replace(temporary, path)
            except (OSError, ValueError):
                continue

    def _collect(self, job):
        if job["action"] == "prepare":
            return
        folder = self.jobs_root / job["id"]
        reader = RunReader(folder / "task")
        result = reader.result()
        job["output_valid"] = result["submission"]["valid"]
        if job["output_valid"] is not True:
            raise ValueError("Предсказания не прошли проверку формата: " + "; ".join(result["submission"]["errors"]))
        job["_output_sha256"] = self._file_hash(reader.submission_path())
        if job["action"] == "agent":
            self._collect_agent(job)
            return
        evidence = _json(folder / "result.json", self.state_root)
        metric = evidence.get("metric")
        split = evidence.get("split")
        if (not isinstance(metric, dict) or not isinstance(metric.get("name"), str)
                or isinstance(metric.get("value"), bool) or not isinstance(metric.get("value"), (int, float))
                or not -((1 << 53) - 1) <= metric["value"] <= (1 << 53) - 1
                or not isinstance(metric.get("higher_is_better"), bool)
                or not _valid_split(split) or not isinstance(evidence.get("model"), str)
                or not isinstance(evidence.get("comparison_key"), str)):
            raise ValueError("Неполный отчёт о локальной валидации.")
        requested_validation = normalize_validation(job.get("validation"))
        if (split.get("strategy", "random") != requested_validation["strategy"]
                or requested_validation["strategy"] == "time"
                and split.get("time_column") != requested_validation["time_column"]):
            raise ValueError("Отчёт использует другую стратегию валидации.")
        if "model_config" in job:
            requested_config = validate_model_config(job["model_config"], action=job["action"])
            reported_config = validate_model_config(evidence.get("model_config"), action=job["action"])
            if reported_config != requested_config or evidence["model"] != reported_config["model"]:
                raise ValueError("Параметры обученной модели не совпадают с настройками запуска.")
        job.update(model=evidence["model"], metric=metric, split=split,
                   comparison_key=evidence["comparison_key"])

    def _collect_agent(self, job):
        try:
            reader = self.agent_reader(job["task_id"], job["id"])
            reports = [row for row in reader.list_runs() if row["status"] != "invalid"]
            if reports:
                detail = reader.detail(reports[0]["id"])
                updates = dict(agent_status=detail["report"]["status"], agent_reason=detail["report"].get("reason"),
                               api=detail["usage"], model=detail["report"].get("model"),
                               api_used=bool(detail["usage"].get("attempts")), agent_run_id=detail["id"])
                updates = self.redact(job["task_id"], job["id"], updates)
                changed = any(job.get(name) != value for name, value in updates.items())
                job.update(updates)
                return changed
        except (OSError, ValueError):
            pass
        return False

    def redact(self, task_id, job_id, value):
        with self._lock:
            self._job(task_id, job_id)
            key = self._secrets.get(job_id, "")
        try:
            current_key = _settings_values().get("OPENAI_API_KEY", "")
        except (OSError, ValueError):
            current_key = ""
        def clean(item):
            if isinstance(item, str):
                return _redact(_redact(item, key), current_key)
            if isinstance(item, list):
                return [clean(child) for child in item]
            if isinstance(item, dict):
                return {clean(name): clean(child) for name, child in item.items()}
            return item
        return clean(value)

    def detail(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            folder = self.jobs_root / job_id
            key = self._secrets.get(job_id, "")
            logs, truncated = {}, False
            for name in ("stdout", "stderr"):
                path = _safe(folder / (name + ".txt"), self.state_root)
                text = ""
                if path.is_file():
                    with path.open("rb") as stream:
                        size = os.fstat(stream.fileno()).st_size
                        stream.seek(max(0, size - LOG_TAIL_BYTES))
                        text = stream.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
                        truncated = truncated or size > LOG_TAIL_BYTES
                logs[name] = _redact(text, key)
            return self.redact(task_id, job_id, {"job": self._summary(job), **logs, "truncated": truncated})

    def agent_reader(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            if job["action"] != "agent":
                raise ValueError("Этот запуск не является исследованием Qwen.")
            stage = _safe(self.jobs_root / job_id / "task", self.state_root, required=True, directory=True)
            return RunReader(stage)

    def stop(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            active = self._active
            if not active or active["id"] != job_id:
                return self._summary(job)
            active["cancel"].set()
            thread = active["thread"]
        thread.join(timeout=4)
        with self._lock:
            return self._summary(job)

    def artifact(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            if job["status"] != "completed" or job["output_valid"] is not True or job["action"] == "prepare":
                raise ValueError("У завершённого запуска пока нет корректного CSV.")
            stage = _safe(self.jobs_root / job_id / "task", self.state_root, required=True, directory=True)
            reader = RunReader(stage)
            if reader.result()["submission"]["valid"] is not True:
                raise ValueError("CSV запуска изменён или больше не проходит проверку.")
            path = reader.submission_path()
            if self._file_hash(path) != job.get("_output_sha256"):
                raise ValueError("CSV был изменён после завершения запуска. Запустите эксперимент заново.")
            return path

    def publish(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            source = self.artifact(task_id, job_id)
            source_hash = self._file_hash(source)
            if self.catalog.fingerprint(task_id) != job["dataset_fingerprint"]:
                raise ValueError("Данные задачи изменились. Результат нельзя применить к другой версии данных.")
            task = self.catalog.resolve(task_id)
            destination = _safe(task / "submission.csv", self.data_root)
            backup = None
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size > MAX_INPUT_BYTES:
                    raise ValueError("Существующий submission.csv нельзя безопасно сохранить.")
                backups = _safe(self.state_root / "backups" / job_id, self.state_root)
                backups.mkdir(parents=True, exist_ok=True)
                backup = backups / (uuid4().hex + ".csv")
                shutil.copyfile(destination, backup)
            temporary = _safe(task / (".app-publish-" + uuid4().hex + ".tmp"), self.data_root)
            try:
                with source.open("rb") as src, temporary.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                if self._file_hash(temporary) != source_hash:
                    raise ValueError("CSV запуска изменился при копировании.")
                if self.catalog.fingerprint(task_id) != job["dataset_fingerprint"]:
                    raise ValueError("Данные изменились во время применения результата.")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            for other in self._jobs.values():
                if other["task_id"] == task_id and other.get("published") and other["id"] != job_id:
                    other["published"] = False
                    self._save(other)
            job.update(published=True, published_at=_now(), _published_sha256=source_hash)
            if backup is not None:
                job["_previous_submission"] = str(backup.relative_to(self.state_root))
            self._save(job)
            return self._summary(job)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = self._active
            if active:
                active["cancel"].set()
        if active:
            active["thread"].join(timeout=4)
        with self._lock:
            # Keep the process-wide lock while a cancelled thread is still
            # finishing a large input copy/hash/validation. Its finally releases
            # the lock, so another server cannot race the old writer.
            if self._active is None:
                self._release_lock()


def _now():
    return datetime.now(timezone.utc).isoformat()
