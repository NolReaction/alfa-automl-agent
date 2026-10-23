"""App adapter for the accepted, fixed three-job native overnight coordinator.

The coordinator owns the CPU lane, deadlines, flat package and receipt audit.
This module only owns its process, durable UI registry and bounded evidence reads.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4
import zipfile
import zlib

import psutil

from app.jobs import _identity, _kill_tree, _linked, _safe, _write_json
from app.runs import _redact
from research_lab.integrity import sanitized_environment, source_identity
from research_lab.process import coordinator_lock, machine_lock_path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
PROFILES = ("regression_off", "regression_on", "classification")
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
ACTIVE = {"queued", "running", "cleanup_pending"}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
PROFILE_STATES = {"pending", "running", "completed", "failed", "interrupted", "cancelled"}
MAX_HISTORY = 200
MAX_JSON_BYTES = 4 * 1024**2
MAX_LOG_BYTES = 20 * 1024**2
LOG_TAIL_BYTES = 128 * 1024
MAX_BUNDLE_BYTES = 270 * 1024**2
PREPARATION_AUDIT_MARGIN = 1800
POLL_SECONDS = .2


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plain(path):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or any(_linked(item) for item in (path, *path.parents)):
        raise ValueError("Ссылки и переходы в рабочих путях не поддерживаются.")
    return path.resolve()


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 9_007_199_254_740_991
    except OverflowError:
        return False


def _text(value, limit=2000):
    return _redact(str(value)[:limit]) if value is not None else None


def _json_snapshot(path, root):
    """Read atomic snapshots; sharing locks and incomplete writes stay unavailable.

    A transient progress read never turns missing evidence into a passed check.
    Unlike browser JSON, source reports may contain uint64 group hashes; only
    explicitly selected safe fields leave this reader.
    """
    for attempt in range(12):
        try:
            path = _safe(path, root)
            if not path.exists():
                return None
            with path.open("rb") as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                return None
            def unique(pairs):
                result = {}
                for name, value in pairs:
                    if name in result:
                        raise ValueError("Duplicate JSON key")
                    result[name] = value
                return result
            def invalid(value):
                raise ValueError("Nonfinite JSON")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
            return value if isinstance(value, dict) else None
        except PermissionError:
            if not WINDOWS or attempt == 11:
                return None
            time.sleep(min(.01 * (2 ** attempt), .1))
        except (OSError, ValueError, UnicodeError, RecursionError):
            return None
    return None


def _metric(value):
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not _finite(value.get("value")):
        return None
    result = {"name": value["name"][:100], "value": value["value"]}
    if type(value.get("higher_is_better")) is bool:
        result["higher_is_better"] = value["higher_is_better"]
    return result


class NativeQueueManager:
    """One fixed native queue; no arbitrary source, path, flags or resume API."""

    def __init__(self, data_root, state_root, catalog, *, source_root=ROOT,
                 runs_root=None, ordinary_jobs=None):
        self.data_root, self.state_root = _plain(data_root), _plain(state_root)
        self.source_root = _plain(source_root)
        default = Path("C:/aq-app") if WINDOWS else Path(tempfile.gettempdir()) / "alfa-app-native"
        self.runs_root = _plain(runs_root if runs_root is not None else default)
        if self.runs_root.is_relative_to(self.source_root):
            raise ValueError("Папка native-запусков должна находиться вне исходного проекта.")
        if (self.runs_root.is_relative_to(self.data_root) or self.data_root.is_relative_to(self.runs_root)):
            raise ValueError("Папка native-запусков должна быть отделена от исходных данных.")
        self.catalog, self.ordinary_jobs = catalog, ordinary_jobs
        self.registry = _safe(self.state_root / "natives", self.state_root)
        self.registry.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self._records, self._active = {}, None
        self._closed = False
        self._artifact_cache = {}
        self._restore()

    def _folder(self, identifier):
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise ValueError("Недопустимый идентификатор native-очереди.")
        return _safe(self.registry / identifier, self.state_root)

    def _output(self, identifier):
        self._folder(identifier)
        return _safe(self.runs_root / identifier, self.runs_root)

    def _save(self, record):
        _write_json(self._folder(record["id"]) / "registry.json", record, self.state_root)

    def _restore(self):
        folders = [p for p in self.registry.iterdir()
                   if IDENTIFIER.fullmatch(p.name) and not _linked(p) and p.is_dir()]
        folders.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for folder in folders[:MAX_HISTORY]:
            record = _json_snapshot(folder / "registry.json", self.state_root)
            if (not record or record.get("version") != 1 or record.get("id") != folder.name
                    or not isinstance(record.get("status"), str)
                    or record.get("status") not in ACTIVE | TERMINAL
                    or not isinstance(record.get("release_sha"), str)
                    or not SHA.fullmatch(record["release_sha"])
                    or type(record.get("time_budget_seconds")) is not int
                    or not 301 <= record["time_budget_seconds"] <= 10500
                    or not isinstance(record.get("created_at"), str)):
                continue
            # Retain only known metadata. Never trust a persisted arbitrary path.
            record = {key: record.get(key) for key in (
                "version", "id", "status", "created_at", "finished_at", "time_budget_seconds",
                "release_sha", "regression_task_id", "classification_task_id", "error", "_runtime")}
            self._records[record["id"]] = record
            if record["status"] in ACTIVE:
                self._settle(record, None, "interrupted",
                             "Сервер был перезапущен. Автоматическое продолжение обучения отключено.")

    def _prune_history(self):
        old = sorted(self._records.values(), key=lambda item: item["created_at"])
        for record in old:
            if len(self._records) <= MAX_HISTORY:
                break
            if record["status"] not in ACTIVE:
                self._records.pop(record["id"], None)

    def _record(self, identifier):
        self._folder(identifier)
        if identifier not in self._records:
            raise ValueError("Native-очередь не найдена.")
        return self._records[identifier]

    def _lane_available(self):
        try:
            with coordinator_lock(machine_lock_path()):
                pass
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def preflight(self):
        """Cheap launch checks only; no data scans, training or API requests."""
        errors, release, memory, free = [], None, None, None
        supported = WINDOWS and sys.version_info[:2] == (3, 12)
        if not supported:
            errors.append("Ночная native-проверка требует Windows и Python 3.12.")
        try:
            identity = source_identity(self.source_root)
            release = identity.get("sha")
            if not isinstance(release, str) or not SHA.fullmatch(release) or identity.get("dirty"):
                errors.append("Нужен чистый checkout точного принятого SHA.")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            errors.append("Не удалось определить версию исходного проекта.")
        for relative in ("experiment_round5_overnight.py", "experiment_round5_overnight_worker.py",
                         "app/supervisor.py", "build_submission.py"):
            try:
                _safe(self.source_root / relative, self.source_root, required=True)
            except ValueError:
                errors.append("В этой версии отсутствует компонент ночной проверки: " + relative)
        try:
            pins = (self.source_root / "requirements.txt").read_text(encoding="utf-8")
            count = 0
            for line in pins.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                name, expected = line.strip().split("==", 1)
                count += 1
                try:
                    if metadata.version(name) != expected:
                        errors.append("Не совпадает закреплённая версия пакета: " + name)
                except metadata.PackageNotFoundError:
                    errors.append("Не установлен пакет: " + name)
            if count != 15:
                errors.append("Список runtime-зависимостей отличается от принятого набора из 15 пакетов.")
        except (OSError, ValueError):
            errors.append("Не удалось проверить requirements.txt.")
        try:
            memory = psutil.virtual_memory().total
            if memory < 30_000_000_000:
                errors.append("Этот план предназначен для компьютера с 32 ГБ RAM.")
        except (OSError, RuntimeError):
            errors.append("Не удалось определить физическую память.")
        try:
            self.runs_root.mkdir(parents=True, exist_ok=True)
            _plain(self.runs_root)
            with tempfile.TemporaryFile(dir=self.runs_root):
                pass
            free = shutil.disk_usage(self.runs_root).free
            if free < 15 * 1024**3:
                errors.append("Для ночного плана нужно не менее 15 ГиБ свободного диска.")
        except (OSError, ValueError):
            errors.append("Внешняя папка результатов недоступна для записи.")
        lane = self._lane_available()
        if not lane:
            errors.append("CPU уже занят другой лабораторной очередью.")
        with self._lock:
            if self._active or any(item["status"] == "cleanup_pending" for item in self._records.values()):
                errors.append("Native-очередь выполняется или её остановка ещё не подтверждена.")
            if self._closed:
                errors.append("Менеджер native-очередей остановлен.")
        if self.ordinary_jobs is not None and self.ordinary_jobs.active_job():
            errors.append("Дождитесь завершения обычного запуска лаборатории.")
        return {"supported": supported, "ready": not errors, "errors": errors, "release_sha": release,
                "threads": 8, "total_native_seconds": 31500, "physical_ram_bytes": memory,
                "free_disk_bytes": free, "lane_available": lane}

    def start(self, regression_task_id, classification_task_id, time_budget_seconds=10500):
        if type(time_budget_seconds) is not int or not 301 <= time_budget_seconds <= 10500:
            raise ValueError("Лимит каждой native-задачи должен быть целым числом от 301 до 10500 секунд.")
        with self._lock:
            checked = self.preflight()
            if not checked["ready"]:
                raise ValueError(" ".join(checked["errors"]))
            from agent.task_spec import load_task
            paths = []
            for task_id, classification in ((regression_task_id, False), (classification_task_id, True)):
                if not isinstance(task_id, str):
                    raise ValueError("Выберите regression и classification из доступных задач.")
                folder = self.catalog.resolve(task_id)
                task = _safe(Path(folder) / "task.md", self.data_root, required=True)
                spec = load_task(task)
                if spec.classification is not classification or not classification and spec.metric.name != "RMSE":
                    raise ValueError("Ночной план требует RMSE-регрессию и задачу классификации.")
                paths.append(task)
            identifier = uuid4().hex
            folder = self._folder(identifier)
            folder.mkdir()
            if self._output(identifier).exists():
                raise ValueError("Каталог новой native-очереди уже существует.")
            record = {"version": 1, "id": identifier, "status": "queued", "created_at": _now(),
                      "finished_at": None, "time_budget_seconds": time_budget_seconds,
                      "release_sha": checked["release_sha"], "regression_task_id": regression_task_id,
                      "classification_task_id": classification_task_id, "error": None}
            self._records[identifier] = record
            self._prune_history()
            self._save(record)
            active = {"id": identifier, "cancel": threading.Event(), "process": None,
                      "identity": None, "paths": paths}
            thread = threading.Thread(target=self._run, args=(identifier, active), daemon=True)
            active["thread"] = thread
            self._active = active
            try:
                thread.start()
            except Exception as error:
                active["cancel"].set()
                self._active = None
                record.update(status="failed", finished_at=_now(), error=_text(error))
                self._save(record)
                raise ValueError("Не удалось запустить поток native-очереди; запись сохранена в истории.") from error
            return self.detail(identifier)

    def _terminate(self, active):
        process = active.get("process")
        if process is None or process.poll() is not None:
            return
        birth = active.get("identity")
        if WINDOWS:
            # Popen owns a Windows process HANDLE, unlike a persisted bare PID.
            # Terminate it immediately: closing its Job Object kills all members
            # without waiting for a best-effort taskkill or trusting a raw PID.
            process.terminate()
        elif birth and _identity(process.pid) == birth:
            _kill_tree(process.pid)
        else:
            raise RuntimeError("Не удалось подтвердить принадлежность дерева процесса.")
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            if WINDOWS:
                process.kill()  # retained HANDLE; never a recovered or reused PID
                process.wait(timeout=2)
            else:
                raise

    def _runtime_state(self, runtime):
        """Distinguish an owned process, a gone birth marker and unknown access."""
        pid, birth = runtime.get("pid"), runtime.get("identity")
        if type(pid) is not int or pid <= 1:
            return "unknown"
        try:
            current = _identity(pid)
        except (OSError, RuntimeError):
            current = None
        if isinstance(current, str) and current:
            if isinstance(birth, str) and birth:
                return "owned" if current == birth else "gone"
            return "unknown"
        try:
            return "unknown" if psutil.pid_exists(pid) else "gone"
        except (OSError, psutil.Error):
            return "unknown"

    def _cleanup_recovered(self, runtime):
        # A recovered PID has no retained HANDLE: act only on the same birth.
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

    def _settle(self, record, active, status, error):
        """Keep durable ownership until exit is confirmed; never resume a fit."""
        with self._cleanup_lock:
            runtime = record.get("_runtime")
            runtime = dict(runtime) if isinstance(runtime, dict) else {}
            clean, cleanup_error = False, None
            try:
                if active is not None:
                    self._terminate(active)
                    process = active.get("process")
                    clean = process is None or process.poll() is not None
                    if process is not None:
                        runtime.update(pid=process.pid, identity=active.get("identity"))
                elif not runtime and record["status"] == "queued":
                    clean = True  # persisted before the launch thread existed
                else:
                    clean = self._cleanup_recovered(runtime)
            except (OSError, RuntimeError, subprocess.SubprocessError, psutil.Error) as failure:
                cleanup_error = _text(failure)
            with self._lock:
                if clean:
                    record.update(status=status, finished_at=_now(), error=error)
                    record.pop("_runtime", None)
                    if self._active is active:
                        self._active = None
                else:
                    runtime.update(terminal_status=status, terminal_error=error)
                    record.update(status="cleanup_pending", finished_at=None, _runtime=runtime,
                        error="Остановка принадлежащего очереди процесса ещё не подтверждена. "
                              "Повторите остановку; новое обучение заблокировано."
                              + (" " + cleanup_error if cleanup_error else ""))
                    if active is not None:
                        active["cleanup_pending"] = True
                self._save(record)
            return clean

    def _run(self, identifier, active):
        record = self._records[identifier]
        status, error = "failed", None
        try:
            if active["cancel"].is_set():
                raise InterruptedError("Очередь остановлена до запуска.")
            regression, classification = active["paths"]
            argv = [sys.executable, "-u", str(self.source_root / "app/supervisor.py"), "--",
                    sys.executable, "-u", str(self.source_root / "experiment_round5_overnight.py"),
                    "--release-commit", record["release_sha"], "--regression-task", str(regression),
                    "--classification-task", str(classification), "--output", str(self._output(identifier)),
                    "--time-budget", str(record["time_budget_seconds"])]
            environment = sanitized_environment(8)
            # The Windows CPU lane is scoped by USERNAME. Keep that same owner
            # in the coordinator while still excluding API credentials.
            if "USERNAME" in os.environ:
                environment["USERNAME"] = os.environ["USERNAME"]
            log = self._folder(identifier) / "coordinator.txt"
            with log.open("xb") as stream:
                with self._lock:
                    if self._closed or active["cancel"].is_set():
                        raise InterruptedError("Очередь остановлена до запуска.")
                    process = subprocess.Popen(argv, cwd=self.source_root, env=environment,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=not WINDOWS,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0)
                    # Retain the HANDLE before probing identity: an access
                    # failure here must never orphan an already-started child.
                    active["process"] = process
                    try:
                        active["identity"] = _identity(process.pid)
                    except (OSError, RuntimeError):
                        active["identity"] = None
                    record.update(status="running", _runtime={"pid": process.pid, "identity": active["identity"]})
                    self._save(record)
                started = time.monotonic()
                watchdog = 3 * (record["time_budget_seconds"] + 30) + PREPARATION_AUDIT_MARGIN
                while process.poll() is None:
                    if active["cancel"].is_set():
                        raise InterruptedError("Очередь остановлена пользователем.")
                    if time.monotonic() - started > watchdog:
                        raise TimeoutError("Истёк общий watchdog очереди, включая подготовку и аудиты.")
                    if log.stat().st_size > MAX_LOG_BYTES:
                        raise ValueError("Журнал координатора превысил лимит 20 МиБ.")
                    active["cancel"].wait(POLL_SECONDS)
                if active["cancel"].is_set():
                    raise InterruptedError("Очередь остановлена пользователем.")
            if process.returncode != 0:
                raise ValueError("Координатор остановился с ошибкой; сохранённые аудиты и журнал доступны.")
            queue = _json_snapshot(self._output(identifier) / "queue.json", self._output(identifier))
            if (not queue or queue.get("status") != "completed" or
                    [item.get("profile") for item in queue.get("jobs", []) if isinstance(item, dict)] != list(PROFILES)):
                raise ValueError("Отсутствует полное подтверждение завершения трёх native-задач.")
            for item in queue["jobs"]:
                audit = _json_snapshot(self._output(identifier) / "jobs" / item["profile"] / "audit.json",
                                       self._output(identifier))
                if item.get("status") != "completed" or not audit or audit.get("continuation_allowed") is not True:
                    raise ValueError("Один из обязательных автоматических аудитов не пройден.")
            status = "completed"
        except InterruptedError as failure:
            status, error = "cancelled", _text(failure)
        except Exception as failure:
            error = _text(failure)
        finally:
            self._settle(record, active, status, error)

    def _profile(self, identifier, name, item, record):
        output = self._output(identifier)
        job = output / "jobs" / name
        raw_status = item.get("status")
        status = raw_status if isinstance(raw_status, str) and raw_status in PROFILE_STATES else "pending"
        if status == "running" and record["status"] in TERMINAL:
            status = "interrupted"
        worker = _json_snapshot(job / "worker_evidence.json", output) or {}
        audit = _json_snapshot(job / "audit.json", output) or {}
        execution = _json_snapshot(job / "execution.json", output) or {}
        native = {}
        relative = (worker.get("run_report") or {}).get("path") if isinstance(worker.get("run_report"), dict) else None
        if not isinstance(relative, str):
            observations = worker.get("observations", [])
            for observed in reversed(observations if isinstance(observations, list) else []):
                if isinstance(observed, dict) and observed.get("phase") == "prepare_research":
                    directory = observed.get("run_dir")
                    if isinstance(directory, str) and re.fullmatch(r"\.agent_runs/[0-9a-f]{32}", directory):
                        relative = directory + "/run.json"
                    break
        if isinstance(relative, str) and re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative):
            native = _json_snapshot(job / "task" / relative, output) or {}
        elapsed = audit.get("native_elapsed_seconds")
        if not _finite(elapsed):
            elapsed = worker.get("worker_elapsed_seconds")
        if status == "running" and _finite(worker.get("native_started_monotonic")):
            elapsed = max(0., time.monotonic() - worker["native_started_monotonic"])
        stage = native.get("stage")
        if not isinstance(stage, str) or len(stage) > 80:
            stage = "preparing" if status == "running" else "waiting" if status == "pending" else "finished"
        resources = execution.get("resources")
        if isinstance(resources, dict):
            cleanup = execution.get("cleanup") if isinstance(execution.get("cleanup"), dict) else {}
            resources = {"available": resources.get("available") if type(resources.get("available")) is bool else None,
                "peak_rss_bytes": resources.get("peak_rss_bytes") if _finite(resources.get("peak_rss_bytes")) else None,
                "cpu_seconds": resources.get("cpu_seconds") if _finite(resources.get("cpu_seconds")) else None,
                "cleanup_complete": cleanup.get("complete") if type(cleanup.get("complete")) is bool else None,
                "scope": _text(resources.get("scope"), 300)}
        else:
            resources = None
        checks = audit.get("checks")
        checks = ({str(k)[:120]: v for k, v in list(checks.items())[:128] if type(v) is bool or v is None}
                  if isinstance(checks, dict) else {})
        return {"profile": name, "status": status, "stage": stage, "elapsed_seconds": elapsed if _finite(elapsed) else None,
            "metric": _metric((audit.get("proof") or {}).get("outer_metric")) if isinstance(audit.get("proof"), dict) else None,
            "continuation_allowed": audit.get("continuation_allowed") if type(audit.get("continuation_allowed")) is bool else None,
            "qualification": "requires_review" if audit.get("qualification") == "requires_review" else None,
            "checks": checks, "error": _text(item.get("error") or audit.get("error")), "resources": resources}

    def detail(self, identifier):
        with self._lock:
            record = deepcopy(self._record(identifier))
        output = self._output(identifier)
        queue = _json_snapshot(output / "queue.json", output) or {}
        jobs = queue.get("jobs") if isinstance(queue.get("jobs"), list) else []
        items = {item.get("profile"): item for item in jobs
                 if isinstance(item, dict) and item.get("profile") in PROFILES}
        comparison = _json_snapshot(output / "comparison.json", output)
        if comparison:
            comparison = {"status": comparison.get("status") if isinstance(comparison.get("status"), str) and comparison.get("status") in {
                "comparable", "incompatible", "incomplete"} else "incomplete",
                "off": _metric(comparison.get("off")), "on": _metric(comparison.get("on")),
                "on_minus_off": comparison.get("on_minus_off") if _finite(comparison.get("on_minus_off")) else None,
                "rescue_exercised": comparison.get("rescue_exercised")
                    if type(comparison.get("rescue_exercised")) is bool else None,
                "note": _text(comparison.get("note"))}
        return {"id": identifier, "status": record["status"], "created_at": record["created_at"],
            "finished_at": record.get("finished_at"), "time_budget_seconds": record["time_budget_seconds"],
            "total_native_seconds": 3 * record["time_budget_seconds"], "release_sha": record["release_sha"],
            "profiles": [self._profile(identifier, name, items.get(name, {}), record) for name in PROFILES],
            "comparison": comparison, "error": _text(record.get("error")),
            "artifact_available": self._artifact_available(identifier), "output_dir": str(output)}

    def list(self):
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r["created_at"], reverse=True)[:MAX_HISTORY]
            identifiers = [record["id"] for record in records]
        return [self.detail(identifier) for identifier in identifiers]

    def active_queue(self):
        with self._lock:
            identifier = self._active["id"] if self._active else next((record["id"]
                for record in self._records.values() if record["status"] == "cleanup_pending"), None)
        return self.detail(identifier) if identifier else None

    def logs(self, identifier, kind="coordinator", profile=None):
        with self._lock:
            record = deepcopy(self._record(identifier))
        if kind == "coordinator":
            if profile is not None:
                raise ValueError("Журнал координатора не принимает profile.")
            root, path = self.state_root, self._folder(identifier) / "coordinator.txt"
            finalized = record["status"] in TERMINAL
        elif isinstance(kind, str) and kind in {"stdout", "stderr"} and profile in PROFILES:
            root = self._output(identifier)
            path = root / "jobs" / profile / "execution" / (kind + ".txt")
            detail = self.detail(identifier)
            state = next(item["status"] for item in detail["profiles"] if item["profile"] == profile)
            finalized = state in TERMINAL
        else:
            raise ValueError("Выберите coordinator или stdout/stderr известного профиля.")
        text = ""
        try:
            path = _safe(path, root)
            if path.is_file():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - LOG_TAIL_BYTES))
                    text = stream.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        except OSError:
            pass
        text = _redact(text).encode("utf-8")[-LOG_TAIL_BYTES:].decode("utf-8", errors="ignore")
        return {"text": text, "kind": kind, "profile": profile, "finalized": finalized}

    def _artifact_available(self, identifier):
        try:
            path = _safe(self._output(identifier) / "evidence.zip", self.runs_root)
            if not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES:
                return False
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            previous = self._artifact_cache.get(identifier)
            if previous and previous[0] == signature:
                return previous[1]
            valid = zipfile.is_zipfile(path)
            self._artifact_cache[identifier] = (signature, valid)
            return valid
        except (OSError, ValueError):
            return False

    def artifact(self, identifier):
        with self._lock:
            self._record(identifier)
        path = _safe(self._output(identifier) / "evidence.zip", self.runs_root, required=True)
        if not self._artifact_available(identifier):
            raise ValueError("Готовый архив evidence пока отсутствует или повреждён.")
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (len(entries) > 30000 or sum(item.file_size for item in entries) > 257 * 1024**2
                        or len([item for item in entries if item.filename == "bundle_manifest.json"]) != 1):
                    raise ValueError("Недопустимый состав или размер native evidence ZIP.")
                for entry in entries:
                    relative = Path(entry.filename)
                    if (relative.is_absolute() or relative.drive or ".." in relative.parts
                            or "\\" in entry.filename or entry.flag_bits & 1):
                        raise ValueError("Небезопасная запись native evidence ZIP.")
                if len({entry.filename for entry in entries}) != len(entries):
                    raise ValueError("Повторяющиеся записи native evidence ZIP.")
                if archive.testzip() is not None:
                    raise ValueError("Контроль целостности native evidence ZIP не пройден.")
        except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error) as error:
            raise ValueError("Не удалось проверить native evidence ZIP.") from error
        return path

    def stop(self, identifier):
        with self._lock:
            record = self._record(identifier)
            active = self._active
            active = active if active is not None and active["id"] == identifier else None
            if active is not None:
                active["cancel"].set()
                thread = active["thread"]
            else:
                thread = None
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            pending = record["status"] == "cleanup_pending"
            runtime = record.get("_runtime") if isinstance(record.get("_runtime"), dict) else {}
            status = runtime.get("terminal_status")
            status = status if isinstance(status, str) and status in TERMINAL else "interrupted"
            error = _text(runtime.get("terminal_error"))
        if pending:
            self._settle(record, active, status, error)
        return self.detail(identifier)

    def close(self):
        with self._lock:
            self._closed = True
            identifier = self._active["id"] if self._active else next((record["id"]
                for record in self._records.values() if record["status"] == "cleanup_pending"), None)
        if identifier:
            self.stop(identifier)
            with self._lock:
                active = self._active
            if active is not None and active["id"] == identifier:
                # HTTP Stop remains responsive; orderly server shutdown waits
                # beyond terminate/wait/kill's six-second HANDLE fallback.
                active["thread"].join(timeout=10)
