"""Local model training and search, isolated workspaces and explicit publication."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import errno
import io
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

import psutil

from app.profiling import DataCatalog, _task_semantics
from app.runs import RunReader, _check_json_tree
from app.splits import normalize_validation
from app.model_config import make_model_config, validate_model_config, model_task_type, winner_feature_evidence, feature_evidence
from app.file_io import replace_file
from app.search_summary import validate_search_summary
from app.research_summary import validate_research_summary
from app.supervisor import process_inventory_matches
from agent.diagnostics import classification_diagnostics, validate_classification_diagnostics
from agent.llm import API_FIELDS, DEFAULT_MODEL, REQUIRED_API_FIELDS
from agent.resources import ProcessTreeMonitor, DEFAULT_MEMORY_LIMIT_BYTES


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {"baseline", "extra_trees", "catboost", "lightgbm", "linear", "automl", "agent", "prepare"}
OFFLINE_ACTIONS = ACTIONS - {"agent", "prepare"}
ACTIVE = {"queued", "running", "cleanup_pending"}
WINDOWS = os.name == "nt"
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_LOG_BYTES = 20 * 1024 * 1024
LOG_TAIL_BYTES = 128 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
LOCAL_TIMEOUT = 1800
PREPARE_TIMEOUT = 600
POLL_SECONDS = 0.2
STAGE_READ_WINDOWS = os.name == "nt"
STAGE_READ_ATTEMPTS = 12
STAGE_READ_INITIAL_DELAY = 0.01
STAGE_READ_MAX_DELAY = 0.1
MAX_DIAGNOSTIC_BYTES = 16 * 1024 * 1024
MAX_DIAGNOSTIC_TRAIN_BYTES = 128 * 1024 * 1024
PUBLIC_FIELDS = {"id", "task_id", "action", "status", "stage", "author", "hypothesis", "started_at",
                 "finished_at", "elapsed_seconds", "model", "metric", "comparison_key", "dataset_fingerprint",
                 "split", "error", "output_valid", "published", "source", "exit_code", "api_used",
                 "agent_status", "agent_reason", "agent_run_id", "api", "published_at", "validation", "model_config", "search", "classification_diagnostics", "feature_config", "component_feature_configs", "research", "resources"}
REQUIRED_FIELDS = PUBLIC_FIELDS - {"agent_status", "agent_reason", "agent_run_id", "api", "published_at", "validation", "model_config", "search", "classification_diagnostics", "feature_config", "component_feature_configs", "research", "resources"}
SPLIT_TEXT_FIELDS = {"time_column": 500, "train_period_min": 100, "train_period_max": 100,
                     "validation_period": 100}
SPLIT_COUNT_FIELDS = {"group_count", "train_groups", "validation_groups", "group_overlap", "n_splits", "fold"}
SPLIT_FIELDS = {"seed", "validation_fraction", "stratified", "validation_rows", "validation_hash",
                "train_fraction", "shuffle", "train_rows", "strategy", *SPLIT_TEXT_FIELDS, *SPLIT_COUNT_FIELDS}


def _campaign_owner(value):
    if (not isinstance(value, dict) or set(value) != {"campaign_id", "step_id"}
            or any(not isinstance(item, str) or not JOB_ID.fullmatch(item) for item in value.values())):
        raise ValueError("Некорректная принадлежность запуска шагу кампании.")
    return dict(value)


def _resource_summary(value):
    """Only bounded observations from the trusted process monitor reach the UI."""
    if not isinstance(value, dict) or type(value.get("available")) is not bool:
        raise ValueError("Некорректные наблюдения ресурсов.")
    result = {"available": value["available"], "scope": "sampled_owned_process_tree"}
    for name in ("peak_rss_bytes", "cpu_seconds", "memory_limit_bytes", "sample_count", "tracked_processes"):
        number = value.get(name)
        if number is not None and (type(number) not in (int, float) or not math.isfinite(number)
                                   or not 0 <= number <= (1 << 53) - 1):
            raise ValueError("Некорректное измерение ресурсов.")
        if name != "cpu_seconds" and number is not None and type(number) is not int:
            raise ValueError("Некорректный счётчик ресурсов.")
        result[name] = number
    cleanup = value.get("cleanup_complete")
    if cleanup is not None and type(cleanup) is not bool:
        raise ValueError("Некорректное подтверждение очистки дерева.")
    result["cleanup_complete"] = cleanup
    reason = value.get("unavailable_reason")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 2000):
        raise ValueError("Некорректная причина недоступности наблюдений.")
    result["unavailable_reason"] = reason
    if result["available"] and (result["peak_rss_bytes"] is None or result["cpu_seconds"] is None
                                or not result["sample_count"] or not result["tracked_processes"]):
        raise ValueError("Доступность ресурсов не подтверждена измерениями.")
    return result


def _resources_ok(resources, execution):
    if execution.get("memory_limit_exceeded") or execution.get("external_timeout"):
        return False
    if not resources or resources.get("available") is not True:
        return None
    if resources.get("cleanup_complete") is not True:
        return resources.get("cleanup_complete")
    peak, limit = resources.get("peak_rss_bytes"), resources.get("memory_limit_bytes")
    if peak is None or limit is None:
        return None
    return peak <= limit


def _job_validation(action, value=None):
    if action == "automl" and (value is None or value == {"strategy": "auto"}):
        return {"strategy": "auto"}
    return normalize_validation(value)


def _config_action(action):
    return None if action == "automl" else action


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


def _stage_json(path: Path, root: Path, cancel=None):
    """Read progress despite brief Windows replacement/reader sharing locks.

    CPython's CRT open can expose EACCES without the original winerror. Retry
    that narrowly on Windows, along with known sharing/access lock codes. A
    permanent denial still propagates after at most .85 s of delays; malformed
    JSON and unsafe paths are never hidden. Recheck the path on every attempt.
    """
    for attempt in range(STAGE_READ_ATTEMPTS):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Запуск остановлен пользователем.")
        try:
            _safe(path, root)
            if not path.exists():
                return None
            return _json(path, root)
        except OSError as error:
            winerror = getattr(error, "winerror", None)
            crt_denial = (winerror is None and isinstance(error, PermissionError)
                          and error.errno == errno.EACCES)
            if (not STAGE_READ_WINDOWS or not (winerror in {5, 32, 33} or crt_denial)
                    or attempt == STAGE_READ_ATTEMPTS - 1):
                raise
            delay = min(STAGE_READ_INITIAL_DELAY * (2 ** attempt), STAGE_READ_MAX_DELAY)
            if cancel is None:
                time.sleep(delay)
            elif cancel.wait(delay):
                raise InterruptedError("Запуск остановлен пользователем.") from error


def _write_json(path: Path, value, root: Path):
    _safe(path, root)
    temporary = path.with_name(path.name + ".tmp")
    _safe(temporary, root)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
    replace_file(temporary, path,
                 check=lambda: (_safe(temporary, root, required=True), _safe(path, root)))


def _settings_values():
    values = {name: os.environ.get(name, "").strip() for name in API_FIELDS}
    local = ROOT / ".env"
    if local.is_file() and not _linked(local):
        from dotenv import dotenv_values
        configured = dotenv_values(local)
        for name in API_FIELDS:
            if name not in os.environ and name in configured:
                values[name] = str(configured[name] or "").strip()
    values["OPENAI_MODEL"] = values["OPENAI_MODEL"] or DEFAULT_MODEL
    return values


def api_settings():
    values = _settings_values()
    missing = [name for name in REQUIRED_API_FIELDS if not values.get(name)]
    return {"ready": not missing, "missing": missing,
            "model_configured": bool(values["OPENAI_MODEL"])}


def _job_environment(settings):
    from agent.execution import child_environment
    environment = child_environment()
    environment.update(settings)
    if os.name == "nt" and os.environ.get("USERNAME"):
        # The contained supervisor must use the same user-specific OS lane.
        environment["USERNAME"] = os.environ["USERNAME"]
    return environment


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
    if sys.platform == "linux" and not process_inventory_matches():
        return None
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def _descendants(pid: int):
    if sys.platform == "linux" and not process_inventory_matches():
        return []
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
        self._cleanup_lock = threading.RLock()
        self._active = None
        self._closed = False
        self._jobs = {}
        self._publication_cache = {}
        self._diagnostic_cache = {}
        self._diagnostic_catalogs = {}
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
                    "_runtime", "_published_sha256", "_output_sha256", "_previous_submission", "_holdout_sha256",
                    "_campaign_owner", "_execution"}}
                self._jobs[job["id"]] = job
                if job["status"] in ACTIVE:
                    runtime = job.get("_runtime") or {}
                    status = runtime.get("terminal_status", "interrupted")
                    error = runtime.get("terminal_error", "Сервер был перезапущен во время выполнения.")
                    self._settle(job, None, status, error)
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
            if job["action"] not in OFFLINE_ACTIONS:
                return False
            try:
                if _job_validation(job["action"], job["validation"]) != job["validation"]:
                    return False
            except ValueError:
                return False
        if "model_config" in job:
            if job["action"] not in OFFLINE_ACTIONS:
                return False
            try:
                config = validate_model_config(job["model_config"], action=_config_action(job["action"]))
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
        for name in ("comparison_key", "dataset_fingerprint", "_published_sha256", "_output_sha256", "_holdout_sha256"):
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
                and job["validation"]["strategy"] != "auto"
                and split.get("strategy", "random") != job["validation"]["strategy"]):
            return False
        if (split is not None and job.get("validation", {}).get("strategy") == "time"
                and split.get("time_column") != job["validation"]["time_column"]):
            return False
        if "search" in job:
            if job["action"] != "automl":
                return False
            try:
                validate_search_summary(job["search"], metric=job.get("metric"), model=job.get("model"),
                                        allow_running=job["status"] != "completed")
                winner_feature_evidence(job, job["search"])
            except ValueError:
                return False
        if "feature_config" in job or "component_feature_configs" in job:
            if job["action"] != "automl" or "search" not in job:
                return False
            try:
                winner_feature_evidence(job, job["search"])
            except ValueError:
                return False
        if "classification_diagnostics" in job:
            if job["action"] not in OFFLINE_ACTIONS:
                return False
            try:
                validate_classification_diagnostics(
                    job["classification_diagnostics"], metric=job.get("metric"),
                    validation_rows=(job.get("split") or {}).get("validation_rows"))
            except ValueError:
                return False
        if job["action"] == "automl" and job["status"] == "completed" and (
                "search" not in job or "model_config" not in job):
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
        if "research" in job:
            if job["action"] != "agent":
                return False
            try:
                validate_research_summary(job["research"])
            except ValueError:
                return False
        runtime = job.get("_runtime")
        if runtime is not None and (not isinstance(runtime, dict) or type(runtime.get("pid")) is not int
                or runtime["pid"] <= 1 or runtime.get("identity") is not None and not isinstance(runtime["identity"], str)):
            return False
        if runtime is not None and (runtime.get("terminal_status", "interrupted") not in {
                "completed", "failed", "cancelled", "interrupted"}
                or runtime.get("terminal_error") is not None and not isinstance(runtime["terminal_error"], str)):
            return False
        try:
            if "_campaign_owner" in job:
                _campaign_owner(job["_campaign_owner"])
            if "resources" in job:
                _resource_summary(job["resources"])
        except ValueError:
            return False
        execution = job.get("_execution")
        if execution is not None:
            if (not isinstance(execution, dict) or set(execution) - {
                    "cleanup_confirmed", "resources_ok", "memory_limit_exceeded", "external_timeout"}
                    or any(value is not None and type(value) is not bool for value in execution.values())):
                return False
            if execution.get("resources_ok") is True and _resources_ok(job.get("resources"), execution) is not True:
                return False
        return True

    def _save(self, job):
        _write_json(self.jobs_root / job["id"] / "job.json", job, self.state_root)

    def _summary(self, job):
        summary = {key: deepcopy(value) for key, value in job.items() if key in PUBLIC_FIELDS}
        if "resources" in summary:
            summary["resources"] = _resource_summary(summary["resources"])
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
                summary["model_config"] = validate_model_config(summary["model_config"], action=_config_action(job["action"]))
            except ValueError:
                summary.pop("model_config")
        if "search" in summary:
            try:
                summary["search"] = validate_search_summary(
                    summary["search"], metric=job.get("metric"), model=job.get("model"),
                    allow_running=job["status"] != "completed")
            except ValueError:
                summary.pop("search")
        if "feature_config" in summary or "component_feature_configs" in summary:
            try:
                summary.update(feature_evidence(summary, model=summary.get("model")))
            except ValueError:
                summary.pop("feature_config", None)
                summary.pop("component_feature_configs", None)
        if "classification_diagnostics" in summary:
            try:
                summary["classification_diagnostics"] = validate_classification_diagnostics(
                    summary["classification_diagnostics"], metric=job.get("metric"),
                    validation_rows=(job.get("split") or {}).get("validation_rows"))
            except ValueError:
                summary.pop("classification_diagnostics")
        if isinstance(summary.get("api"), dict):
            summary["api"] = {key: value for key, value in summary["api"].items()
                              if key in {"state", "attempts", "responses", "reported_responses", "prompt_tokens",
                                         "completion_tokens", "total_tokens"}}
        if "research" in summary:
            summary["research"] = validate_research_summary(summary["research"])
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
            if self._active:
                return self._summary(self._jobs[self._active["id"]])
            return next((self._summary(job) for job in self._jobs.values() if job.get("status") in ACTIVE), None)

    def campaign_job(self, campaign_id, step_id):
        owner = _campaign_owner({"campaign_id": campaign_id, "step_id": step_id})
        with self._lock:
            matches = [job for job in self._jobs.values() if job.get("_campaign_owner") == owner]
            if len(matches) > 1:
                raise RuntimeError("Найдено несколько запусков одного шага кампании; повторный запуск запрещён.")
            if not matches:
                return None
            job = matches[0]
            execution = job.get("_execution") or {}
            return {"job": self._summary(job), "cleanup_confirmed": execution.get("cleanup_confirmed"),
                    "resources_ok": execution.get("resources_ok")}

    def list(self, task_id=None):
        with self._lock:
            jobs = [job for job in self._jobs.values() if task_id is None or job["task_id"] == task_id]
            return [self._summary(job) for job in sorted(jobs, key=lambda x: x["started_at"], reverse=True)[:200]]

    def history_counts(self):
        with self._lock:
            return {"total": len(self._jobs),
                    "completed": sum(job.get("status") == "completed" for job in self._jobs.values()),
                    "failed": sum(job.get("status") == "failed" for job in self._jobs.values())}

    def _job(self, task_id, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError("Недопустимый идентификатор запуска.")
        job = self._jobs.get(job_id)
        if job is None or job.get("task_id") != task_id:
            raise ValueError("Запуск не найден для этой задачи.")
        _safe(self.jobs_root / job_id, self.state_root, required=True, directory=True)
        return job

    def start(self, task_id, action, author="", hypothesis="", minutes=None, validation=None, model_params=None,
              *, campaign_owner=None):
        owner = None if campaign_owner is None else _campaign_owner(campaign_owner)
        if owner is not None and (not isinstance(action, str) or action not in OFFLINE_ACTIONS):
            raise ValueError("Шаг кампании поддерживает только локальные модели и AutoML.")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError("Неизвестный рецепт запуска.")
        if validation is not None and action not in OFFLINE_ACTIONS:
            raise ValueError("Выбор валидации доступен только для локальных моделей и автоподбора.")
        if model_params is not None and action not in OFFLINE_ACTIONS - {"automl"}:
            raise ValueError("Настройка модели доступна только для baseline и ExtraTrees.")
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("Параметры модели должны быть объектом.")
        validation = _job_validation(action, validation) if action in OFFLINE_ACTIONS else None
        model_config = None
        if not isinstance(author, str) or len(author) > 80 or not isinstance(hypothesis, str) or len(hypothesis) > 500:
            raise ValueError("Имя автора: до 80 символов; гипотеза: до 500 символов.")
        if minutes is None:
            minutes = 10 if action == "automl" else 175
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 175:
            raise ValueError("Бюджет запуска должен быть целым числом от 1 до 175 минут.")
        with self._lock:
            if owner is not None:
                previous = self.campaign_job(**owner)
                if previous is not None:
                    if previous["job"]["task_id"] != task_id or previous["job"]["action"] != action:
                        raise ValueError("Этот шаг кампании уже связан с другим запуском.")
                    return previous["job"]
            if self._closed:
                raise RuntimeError("Менеджер запусков остановлен.")
            if self.active_job():
                raise RuntimeError("Уже выполняется другой запуск. Дождитесь завершения или остановите его.")
            if action != "prepare":
                from research_lab.process import coordinator_lock, machine_lock_path
                with coordinator_lock(machine_lock_path()):
                    pass
            values = _settings_values() if action == "agent" else {}
            if action == "agent" and any(not values.get(name) for name in REQUIRED_API_FIELDS):
                raise ValueError("Для Qwen заполните OPENAI_API_KEY и OPENAI_BASE_URL в окружении или локальном .env.")
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
                if action in OFFLINE_ACTIONS:
                    _, inputs = self.catalog._inputs(task_folder)
                    metadata, _, _, task_type = self.catalog._metadata(inputs)
                    from agent.task_spec import load_task
                    spec = load_task(task_folder / "task.md")
                    task_type = spec.metric.task_type
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
                    if action != "automl":
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
            if owner is not None:
                job["_campaign_owner"] = owner
                job["_execution"] = {"cleanup_confirmed": None, "resources_ok": None,
                                     "memory_limit_exceeded": False, "external_timeout": False}
            self._jobs[job_id] = job
            self._save(job)
            active = {"id": job_id, "cancel": threading.Event(), "process": None,
                      "started": time.monotonic(), "key": values.get("OPENAI_API_KEY", ""), "children": {}}
            if active["key"]:
                self._secrets[job_id] = active["key"]
            self._active = active
            try:
                thread = threading.Thread(target=self._run, args=(job_id, active, minutes, values), daemon=True)
                active["thread"] = thread
                thread.start()
            except Exception as error:
                self._settle(job, active, "failed", "Не удалось запустить поток: " + _redact(error, active["key"]))
                raise RuntimeError("Не удалось запустить поток; запись сохранена в истории.") from error
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
                    "--mode", "llm", "--max-steps", "12", "--time-budget", str(minutes * 60)]
        folder = self.jobs_root / job["id"]
        config = {"job_dir": str(folder), "task_path": str(task_path), "action": job["action"],
                  "dataset_fingerprint": job["dataset_fingerprint"],
                  "validation": _job_validation(job["action"], job.get("validation"))}
        if job["action"] == "automl":
            config["time_budget"] = minutes * 60
        if "model_config" in job and job["action"] != "automl":
            model_config = validate_model_config(job["model_config"], action=_config_action(job["action"]))
            if {"objective", "target_transform"} & set(model_config):
                config["model_config"] = model_config
            else:
                config["model_params"] = model_config["parameters"]
        _write_json(folder / "config.json", config, self.state_root)
        return [sys.executable, "-u", str(ROOT / "app" / "worker.py"), "--config", str(folder / "config.json")]

    def _set(self, job_id, **updates):
        with self._lock:
            job = self._jobs[job_id]
            job.update(updates)
            self._save(job)

    @staticmethod
    def _runtime_state(runtime):
        pid, birth = runtime.get("pid"), runtime.get("identity")
        if type(pid) is not int or pid <= 1:
            return "unknown"
        try:
            current = _identity(pid)
        except (OSError, RuntimeError):
            current = None
        if current:
            if not birth:
                return "unknown"
            if current != birth:
                return "gone"
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    return "gone"
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return "gone"
            except (OSError, psutil.AccessDenied):
                pass
            return "owned"
        if sys.platform == "linux" and not process_inventory_matches():
            return "unknown"
        try:
            return "unknown" if psutil.pid_exists(pid) else "gone"
        except (OSError, psutil.Error):
            return "unknown"

    def _cleanup_recovered(self, runtime):
        state = self._runtime_state(runtime)
        if state == "gone":
            return True
        if state != "owned":
            return False
        _kill_tree(runtime["pid"])
        for attempt in range(6):
            if self._runtime_state(runtime) == "gone":
                return True
            if attempt < 5:
                time.sleep(.05)
        return False

    def _terminate_active(self, active):
        process, monitor = active.get("process"), active.get("monitor")
        if process is None:
            return True
        if monitor is not None:
            stopped = monitor.terminate()
            return stopped and process.poll() is not None
        if process.poll() is None:
            birth = active.get("identity")
            if WINDOWS:
                process.terminate()  # retained Windows HANDLE, never a bare recovered PID
            elif birth and _identity(process.pid) == birth:
                _kill_tree(process.pid)
            else:
                # Popen still owns this unreaped child even with an unavailable
                # inventory. No guessed descendants or recovered PID are used.
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        clean = process.poll() is not None
        for pid, birth in active.get("children", {}).items():
            runtime = {"pid": pid, "identity": birth}
            clean = self._cleanup_recovered(runtime) and clean
        return clean

    def _settle(self, job, active, status, error):
        """Do not relinquish a process or durable job until cleanup is known."""
        with self._cleanup_lock:
            runtime = dict(job.get("_runtime") or {})
            execution = dict(job.get("_execution") or {})
            cleanup_error, clean, resources = None, False, job.get("resources")
            try:
                if active is not None:
                    process = active.get("process")
                    if process is not None:
                        runtime.update(pid=process.pid, identity=active.get("identity"))
                    execution.update(active.get("execution") or {})
                    clean = self._terminate_active(active)
                    if active.get("monitor") is not None:
                        resources = _resource_summary(active["monitor"].summary())
                elif runtime:
                    clean = self._cleanup_recovered(runtime)
                    # A restarted manager cannot reconstruct missed samples or
                    # certify descendants from a persisted root PID alone.
                    if resources is not None:
                        resources = {**resources, "cleanup_complete": None}
                else:
                    clean = True  # no retained process; never restart this job
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, psutil.Error) as failure:
                cleanup_error = _redact(failure, (active or {}).get("key", ""))
                if active is not None and active.get("monitor") is not None:
                    try:
                        resources = _resource_summary(active["monitor"].summary())
                    except (OSError, RuntimeError, ValueError, psutil.Error):
                        resources = None
            with self._lock:
                execution["cleanup_confirmed"] = clean
                execution["resources_ok"] = _resources_ok(resources, execution) if clean else False
                job["_execution"] = execution
                if resources is not None:
                    job["resources"] = resources
                if active is not None:
                    job["elapsed_seconds"] = round(time.monotonic() - active["started"], 3)
                    process = active.get("process")
                    job["exit_code"] = process.returncode if process is not None else None
                if clean:
                    job.pop("_runtime", None)
                    job.update(status=status, stage="finished", finished_at=_now(), error=error)
                else:
                    runtime.update(terminal_status=status, terminal_error=error)
                    if runtime.get("pid") is not None:
                        job["_runtime"] = runtime
                    job.update(status="cleanup_pending", stage="cleanup", finished_at=None,
                               error="Остановка принадлежащего запуску процесса ещё не подтверждена. "
                                     "Повторите остановку; новый запуск заблокирован."
                                     + (" " + cleanup_error if cleanup_error else ""))
                try:
                    self._save(job)
                except (OSError, ValueError) as failure:
                    # Keep the live handle/association if the terminal history
                    # write fails. A later Stop retries persistence as well.
                    if runtime.get("pid") is not None:
                        runtime.update(terminal_status=status, terminal_error=error)
                        job["_runtime"] = runtime
                    job.update(status="cleanup_pending", stage="cleanup", finished_at=None,
                               error="Не удалось сохранить завершение запуска: " + _redact(failure))
                    return False
                if clean and self._active is active:
                    self._active = None
                if self._closed and not self.active_job():
                    self._release_lock()
                return clean

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
            environment = _job_environment(settings)
            timeout = PREPARE_TIMEOUT if job["action"] == "prepare" else minutes * 60 + 30 if job["action"] in {"agent", "automl"} else LOCAL_TIMEOUT
            stage = {"prepare": "downloading", "agent": "research"}.get(job["action"], "training")
            with (folder / "stdout.txt").open("xb") as stdout, (folder / "stderr.txt").open("xb") as stderr:
                with self._lock:
                    if self._closed or active["cancel"].is_set():
                        raise InterruptedError("Запуск остановлен пользователем.")
                    lane_args = [] if job["action"] == "prepare" else ["--cpu-lane"]
                    supervised = [sys.executable, "-u", str(ROOT / "app" / "supervisor.py"),
                                  *lane_args, "--", *command]
                    process = subprocess.Popen(supervised, cwd=ROOT, env=environment,
                                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               start_new_session=os.name != "nt",
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                    active["process"] = process
                    active["identity"] = None
                    self._set(job_id, status="running", stage=stage,
                              _runtime={"pid": process.pid, "identity": None})
                    try:
                        active["identity"] = _identity(process.pid)
                    except (OSError, RuntimeError):
                        pass
                    self._set(job_id, _runtime={"pid": process.pid, "identity": active["identity"]})
                    if job.get("_campaign_owner") is not None:
                        active["monitor"] = ProcessTreeMonitor(process, max_memory_bytes=DEFAULT_MEMORY_LIMIT_BYTES)
                        active["execution"] = {"memory_limit_exceeded": False, "external_timeout": False}
                launched = time.monotonic()
                while process.poll() is None:
                    if active.get("monitor") is not None:
                        over_limit = active["monitor"].sample()
                        self._set(job_id, resources=_resource_summary(active["monitor"].summary()))
                        if over_limit:
                            active["execution"]["memory_limit_exceeded"] = True
                            raise MemoryError("Запуск остановлен: превышен лимит памяти 28 GiB.")
                    if os.name != "nt":
                        for pid in _descendants(process.pid):
                            identity = _identity(pid)
                            if identity:
                                active["children"][pid] = identity
                    if active["cancel"].is_set():
                        raise InterruptedError("Запуск остановлен пользователем.")
                    if time.monotonic() - launched > timeout:
                        active.setdefault("execution", {})["external_timeout"] = True
                        raise TimeoutError("Запуск остановлен: превышен лимит времени.")
                    for name in ("stdout.txt", "stderr.txt"):
                        path = _safe(folder / name, self.state_root, required=True)
                        if path.stat().st_size > MAX_LOG_BYTES:
                            raise ValueError("Запуск остановлен: журнал превысил лимит 20 МБ.")
                    stage_file = folder / "stage.json"
                    evidence = (_stage_json(stage_file, self.state_root, active["cancel"])
                                if job["action"] in OFFLINE_ACTIONS else None)
                    if evidence is not None:
                        current = evidence.get("stage")
                        updates = {}
                        if current in {"training", "validating", "refitting", "finished"} and current != job["stage"]:
                            updates["stage"] = current
                        if job["action"] == "automl" and "search" in evidence:
                            search = validate_search_summary(evidence["search"], allow_running=True)
                            if search != job.get("search"):
                                updates["search"] = search
                        if updates:
                            self._set(job_id, **updates)
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
            try:
                self._scrub_logs(folder, active["key"])
                if job["action"] == "agent":
                    self._collect_agent(job)
            except Exception as failure:
                final_status, error = "failed", _redact(failure, active["key"])
            finally:
                self._settle(job, active, final_status, error)

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
                    replace_file(temporary, path, check=lambda: (
                        _safe(temporary, self.state_root, required=True),
                        _safe(path, self.state_root, required=True)))
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
        from agent.task_spec import load_task
        spec = load_task(folder / "task" / "task.md")
        if metric["name"] != spec.metric.name or metric["higher_is_better"] != spec.metric.higher_is_better:
            raise ValueError("Метрика отчёта не соответствует основной метрике задачи.")
        requested_validation = _job_validation(job["action"], job.get("validation"))
        if (requested_validation["strategy"] != "auto"
                and split.get("strategy", "random") != requested_validation["strategy"]
                or requested_validation["strategy"] == "time"
                and split.get("time_column") != requested_validation["time_column"]):
            raise ValueError("Отчёт использует другую стратегию валидации.")
        if "model_config" in job and job["action"] != "automl":
            requested_config = validate_model_config(job["model_config"], action=_config_action(job["action"]))
            reported_config = validate_model_config(evidence.get("model_config"), action=job["action"])
            if reported_config != requested_config or evidence["model"] != reported_config["model"]:
                raise ValueError("Параметры обученной модели не совпадают с настройками запуска.")
        expected_key = hashlib.sha256(json.dumps({
            "dataset_fingerprint": job["dataset_fingerprint"], "metric": metric["name"],
            "higher_is_better": metric["higher_is_better"], "split": split,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        if evidence["comparison_key"] != expected_key:
            raise ValueError("Отпечаток сравнения не соответствует данным и проверочным строкам.")
        if job["action"] == "automl":
            reported_config = validate_model_config(evidence.get("model_config"))
            if evidence["model"] != reported_config["model"]:
                raise ValueError("Модель победителя не совпадает с параметрами отчёта.")
            if model_task_type(reported_config) != spec.metric.task_type:
                raise ValueError("Модель победителя не соответствует типу задачи.")
            search = validate_search_summary(evidence.get("search"), metric=metric, model=evidence["model"])
            job.update(model_config=reported_config, search=search,
                       **winner_feature_evidence(evidence, search))
        job.update(model=evidence["model"], metric=metric, split=split,
                   comparison_key=evidence["comparison_key"])
        if "classification_diagnostics" in evidence:
            job["classification_diagnostics"] = validate_classification_diagnostics(
                evidence["classification_diagnostics"], metric=metric,
                validation_rows=split["validation_rows"])
            held = _safe(folder / "holdout_predictions.csv", self.state_root, required=True)
            if held.stat().st_size <= MAX_DIAGNOSTIC_BYTES:
                job["_holdout_sha256"] = self._file_hash(held)

    def _collect_agent(self, job):
        try:
            reader = self.agent_reader(job["task_id"], job["id"])
            reports = [row for row in reader.list_runs() if row["status"] != "invalid"]
            if reports:
                detail = reader.detail(reports[0]["id"])
                updates = dict(agent_status=detail["report"]["status"], agent_reason=detail["report"].get("reason"),
                               api=detail["usage"], model=detail["report"].get("model"),
                               api_used=bool(detail["usage"].get("attempts")), agent_run_id=detail["id"],
                               research=detail["research"])
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

    def diagnostics_for(self, task_id, job_id):
        """Read a bounded, checked holdout; older jobs need no new training.

        Early versions did not hash the prediction file. Their first read is
        tied to the original staged dataset, held IDs/answers and saved Macro
        F1 before pinning its hash. Later reads reject any changed evidence.
        Missing diagnostics never make an otherwise usable CSV disappear.
        """
        with self._lock:
            job = self._job(task_id, job_id)
            metric, split = job.get("metric"), job.get("split")
            if (job["status"] != "completed" or job["action"] not in OFFLINE_ACTIONS
                    or not isinstance(metric, dict) or metric.get("name") != "Macro F1"
                    or metric.get("higher_is_better") is not True or not _valid_split(split)):
                return None
            try:
                folder = _safe(self.jobs_root / job_id, self.state_root, required=True, directory=True)
                held = _safe(folder / "holdout_predictions.csv", self.state_root, required=True)
                with held.open("rb") as stream:
                    raw = stream.read(MAX_DIAGNOSTIC_BYTES + 1)
                if len(raw) > MAX_DIAGNOSTIC_BYTES:
                    raise ValueError("Проверочные предсказания слишком велики для диагностики.")
                digest = hashlib.sha256(raw).hexdigest()
                if job.get("_holdout_sha256") not in (None, digest):
                    raise ValueError("Проверочные предсказания изменены после завершения запуска.")
                catalog = self._diagnostic_catalogs.get(job_id)
                if catalog is None:
                    catalog = DataCatalog(folder)
                    if len(self._diagnostic_catalogs) >= 8:
                        self._diagnostic_catalogs.pop(next(iter(self._diagnostic_catalogs)))
                    self._diagnostic_catalogs[job_id] = catalog
                fingerprint = catalog.fingerprint("task")
                if fingerprint != job["dataset_fingerprint"]:
                    raise ValueError("Копия данных запуска была изменена.")
                signature = (digest, fingerprint, metric["value"], split["validation_hash"],
                             split["validation_rows"])
                cached = self._diagnostic_cache.get(job_id)
                if cached and cached[0] == signature:
                    if cached[1] is None:
                        return None
                    if job.get("classification_diagnostics") != cached[1]:
                        job["classification_diagnostics"] = deepcopy(cached[1])
                        self._save(job)
                    return deepcopy(cached[1])
                rows = csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
                if next(rows, None) != ["id", "actual", "prediction"]:
                    raise ValueError("Некорректные столбцы проверочных предсказаний.")
                ids, actual, predicted = [], [], []
                for row in rows:
                    if len(row) != 3 or len(ids) >= split["validation_rows"]:
                        raise ValueError("Некорректное число проверочных строк.")
                    identifier, answer, prediction = row
                    if (not identifier.strip() or not answer.strip() or not prediction.strip()
                            or max(len(identifier), len(answer), len(prediction)) > 2000):
                        raise ValueError("Проверочные строки содержат пустые или слишком длинные значения.")
                    ids.append(identifier)
                    actual.append(answer)
                    predicted.append(prediction)
                if len(ids) != split["validation_rows"] or len(set(ids)) != len(ids):
                    raise ValueError("ID проверочных строк должны быть уникальными.")
                task = catalog.resolve("task")
                _, inputs = catalog._inputs(task)
                _, id_column, target, task_type = catalog._metadata(inputs)
                task_type, _ = _task_semantics(catalog.task_text("task"), task_type)
                if task_type != "classification":
                    raise ValueError("Диагностика предназначена для классификации.")
                identity = {"id_column": id_column, "target_column": target, "ids": ids, "actuals": actual}
                payload = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")
                if hashlib.sha256(payload).hexdigest() != split["validation_hash"]:
                    raise ValueError("Проверочные ID и ответы не совпадают с отчётом запуска.")
                train_path = _safe(inputs["train.csv"], self.state_root, required=True)
                if train_path.stat().st_size > MAX_DIAGNOSTIC_TRAIN_BYTES:
                    raise ValueError("Копия train слишком велика для диагностики.")
                expected = dict(zip(ids, actual))
                found, seen, labels = set(), set(), set()
                with train_path.open("rb") as stream:
                    train_raw = stream.read(MAX_DIAGNOSTIC_TRAIN_BYTES + 1)
                if len(train_raw) > MAX_DIAGNOSTIC_TRAIN_BYTES:
                    raise ValueError("Копия train слишком велика для диагностики.")
                with io.StringIO(train_raw.decode("utf-8-sig"), newline="") as stream:
                    train_rows = csv.reader(stream, strict=True)
                    header = next(train_rows)
                    id_index, target_index = header.index(id_column), header.index(target)
                    for row in train_rows:
                        if len(row) != len(header):
                            raise ValueError("Повреждена копия train запуска.")
                        identifier, answer = row[id_index], row[target_index]
                        if identifier in seen or not answer.strip():
                            raise ValueError("Повреждены ID или ответы копии train.")
                        seen.add(identifier)
                        labels.add(answer)
                        if identifier in expected:
                            if answer != expected[identifier]:
                                raise ValueError("Правильные ответы не совпадают с копией train.")
                            found.add(identifier)
                if found != set(ids) or not set(predicted) <= labels:
                    raise ValueError("Проверочные ID или классы отсутствуют в копии train.")
                diagnostics = classification_diagnostics(actual, predicted)
                if diagnostics is None:
                    if len(self._diagnostic_cache) >= 8:
                        self._diagnostic_cache.pop(next(iter(self._diagnostic_cache)))
                    self._diagnostic_cache[job_id] = (signature, None)
                    return None
                diagnostics = validate_classification_diagnostics(
                    diagnostics, metric=metric, validation_rows=split["validation_rows"])
                if catalog.fingerprint("task") != fingerprint or self._file_hash(held) != digest:
                    raise ValueError("Проверочные предсказания или данные изменились во время чтения.")
                if (job.get("classification_diagnostics") != diagnostics
                        or job.get("_holdout_sha256") != digest):
                    job["classification_diagnostics"] = diagnostics
                    job["_holdout_sha256"] = digest
                    self._save(job)
                if len(self._diagnostic_cache) >= 8:
                    self._diagnostic_cache.pop(next(iter(self._diagnostic_cache)))
                self._diagnostic_cache[job_id] = (signature, diagnostics)
                return deepcopy(diagnostics)
            except (OSError, ValueError, KeyError, IndexError, StopIteration, csv.Error, UnicodeError):
                self._diagnostic_cache.pop(job_id, None)
                if job.pop("classification_diagnostics", None) is not None:
                    try:
                        self._save(job)
                    except (OSError, ValueError):
                        pass
                return None

    def detail(self, task_id, job_id):
        with self._lock:
            job = self._job(task_id, job_id)
            self.diagnostics_for(task_id, job_id)
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
            active = active if active and active["id"] == job_id else None
            if active is not None:
                active["cancel"].set()
            thread = active.get("thread") if active else None
        if thread is not None:
            try:
                thread.join(timeout=4)
            except RuntimeError:
                pass  # a thread whose start failed has never owned a process
        with self._lock:
            pending = job["status"] == "cleanup_pending" or active is None and job["status"] in ACTIVE
            runtime = job.get("_runtime") or {}
            status = runtime.get("terminal_status", "interrupted")
            error = runtime.get("terminal_error")
        if pending:
            self._settle(job, active, status, error)
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
            self.diagnostics_for(task_id, job_id)
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
                replace_file(temporary, destination, check=lambda: (
                    _safe(temporary, self.data_root, required=True),
                    _safe(destination, self.data_root)))
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
            self._closed = True
            active = self._active
            pending = self._jobs[active["id"]] if active else next(
                (job for job in self._jobs.values() if job.get("status") in ACTIVE), None)
        if pending is not None:
            self.stop(pending["task_id"], pending["id"])
        with self._lock:
            # Keep the process-wide lock while a cancelled thread is still
            # finishing a large input copy/hash/validation. Its finally releases
            # the lock, so another server cannot race the old writer.
            if not self.active_job():
                self._release_lock()


def _now():
    return datetime.now(timezone.utc).isoformat()
