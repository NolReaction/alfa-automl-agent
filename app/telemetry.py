"""Cached machine telemetry, independent of any solver's resource evidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time

import psutil


SYSTEM_CACHE_SECONDS = 3.0
GPU_CACHE_SECONDS = 10.0
GPU_TIMEOUT_SECONDS = 2.0
GPU_OUTPUT_BYTES = 16 * 1024
GPU_DEVICE_LIMIT = 8
GPU_QUERY = ("--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits")


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value, low=0, high=None):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= low and (high is None or value <= high) else None


def _bytes(value):
    number = _number(value)
    return int(number) if number is not None else None


def _text(value, limit=160):
    if not isinstance(value, str):
        return None
    value = " ".join("".join(char for char in value if char.isprintable()).split())[:limit]
    return value or None


def _gpu_unavailable(code, reason):
    return {"available": False, "status": "unavailable", "reason_code": code,
            "reason": reason, "sampled_at": _utc(), "devices": []}


def parse_nvidia_csv(raw):
    """Parse at most eight fixed-query rows; unknown counters stay unknown."""
    if not isinstance(raw, bytes) or len(raw) > GPU_OUTPUT_BYTES:
        return _gpu_unavailable("gpu_output_too_large", "Ответ датчика GPU превышает допустимый размер.")
    try:
        rows = [row for row in csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")), strict=True)
                if any(value.strip() for value in row)]
    except csv.Error:
        return _gpu_unavailable("gpu_output_invalid", "Ответ датчика GPU не удалось разобрать.")
    if not rows or any(len(row) != 5 for row in rows):
        return _gpu_unavailable("gpu_output_invalid", "Ответ датчика GPU имеет неожиданную структуру.")
    if len(rows) > GPU_DEVICE_LIMIT:
        return _gpu_unavailable("gpu_device_limit", "Число GPU превышает поддерживаемый предел обзора.")
    devices = []
    for index, row in enumerate(rows):
        name = _text(row[0])
        if name and name.lower() in {"n/a", "nan", "unknown", "[not supported]"}:
            name = None
        total = _number(row[1], low=1, high=2**40)
        used = _number(row[2], high=2**40)
        if total is not None and used is not None and used > total:
            used = None
        devices.append({"index": index, "name": name,
                        "memory_total_bytes": int(total * 1024**2) if total is not None else None,
                        "memory_used_bytes": int(used * 1024**2) if used is not None else None,
                        "utilization_percent": _number(row[3], high=100),
                        "temperature_c": _number(row[4], low=-50, high=200)})
    partial = any(value is None for device in devices for key, value in device.items() if key != "index")
    return {"available": True, "status": "partial" if partial else "measured",
            "reason_code": "gpu_metrics_partial" if partial else None,
            "reason": "Часть показателей GPU недоступна." if partial else None,
            "sampled_at": _utc(), "devices": devices}


def _bounded_nvidia(argv):
    """Bound stdout to 16 KiB plus an overflow sentinel; never expose stderr."""
    process = None
    reader = None
    result = {}
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def collect():
            try:
                result["stdout"] = process.stdout.read(GPU_OUTPUT_BYTES + 1)
                if len(result["stdout"]) > GPU_OUTPUT_BYTES:
                    result["overflow"] = True
                    try:
                        process.kill()
                    except OSError:
                        pass
            except (OSError, ValueError):
                result["read_error"] = True

        reader = threading.Thread(target=collect, name="alfa-gpu-output", daemon=True)
        reader.start()
        try:
            exit_code = process.wait(timeout=GPU_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
            return _gpu_unavailable("nvidia_smi_timeout", "Датчик GPU не ответил за отведённое время.")
        reader.join(timeout=.25)
        if result.get("overflow"):
            return _gpu_unavailable("gpu_output_too_large", "Ответ датчика GPU превышает допустимый размер.")
        if reader.is_alive() or result.get("read_error"):
            return _gpu_unavailable("gpu_output_unavailable", "Не удалось прочитать показатели GPU.")
        if exit_code != 0:
            return _gpu_unavailable("nvidia_smi_failed", "Драйвер NVIDIA не предоставил показатели GPU.")
        return parse_nvidia_csv(result.get("stdout", b""))
    except (OSError, subprocess.SubprocessError):
        return _gpu_unavailable("nvidia_smi_unavailable", "Не удалось запустить датчик NVIDIA.")
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
            if reader is not None:
                reader.join(timeout=.25)
            # A pathological inherited pipe must not block an HTTP thread.
            if process.stdout is not None and (reader is None or not reader.is_alive()):
                try:
                    process.stdout.close()
                except OSError:
                    pass


class SystemMonitor:
    """One cached monitor; snapshots contain no paths, environment or process data.

    psutil CPU baselines are thread-local. A single lazy worker keeps samples
    comparable across ThreadingHTTPServer request threads without busy polling.
    """
    def __init__(self, source_root, state_root):
        self._source_root = Path(source_root)
        self._state_root = Path(state_root)
        self._lock = threading.RLock()
        self._cpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alfa-system-cpu")
        self._cpu_future = None
        self._cpu_primed = False
        self._closed = False
        self._system = self._gpu = None
        self._system_until = self._gpu_until = 0.0
        try:
            name = _text(platform.processor())
        except OSError:
            name = None
        self._cpu_name = name
        self._runtime = {"python_version": platform.python_version(),
                         "platform": f"{platform.system()} {platform.release()}".strip(),
                         "machine": _text(platform.machine())}

    @staticmethod
    def _count(logical):
        try:
            count = psutil.cpu_count(logical=logical)
            return count if type(count) is int and count > 0 else None
        except (psutil.Error, OSError):
            return None

    def _read_cpu(self):
        try:
            value = _number(psutil.cpu_percent(interval=None), high=100)
            if value is None:
                raise ValueError("invalid CPU sample")
            if not self._cpu_primed:
                self._cpu_primed = True
                return {"percent": None, "measurement_status": "initializing",
                        "reason_code": "initial_cpu_sample", "reason": "Ожидается второй замер нагрузки CPU."}
            return {"percent": value, "measurement_status": "measured", "reason_code": None, "reason": None}
        except (psutil.Error, OSError, ValueError):
            return {"percent": None, "measurement_status": "unavailable",
                    "reason_code": "cpu_read_unavailable", "reason": "Показатели нагрузки CPU недоступны."}

    def _sample_cpu(self):
        if self._cpu_future is None:
            self._cpu_future = self._cpu_worker.submit(self._read_cpu)
        try:
            value = self._cpu_future.result(timeout=.25)
            self._cpu_future = None
            return value
        except FutureTimeout:
            return {"percent": None, "measurement_status": "unavailable",
                    "reason_code": "cpu_read_timeout", "reason": "Замер нагрузки CPU ещё не завершён."}

    def _sample_system(self):
        cpu = {"name": self._cpu_name, "physical_cores": self._count(False),
               "logical_cores": self._count(True), **self._sample_cpu()}
        memory = {"total_bytes": None, "available_bytes": None, "percent": None,
                  "available": False, "reason_code": "memory_read_unavailable", "reason": "Показатели памяти недоступны."}
        try:
            measured = psutil.virtual_memory()
            total, available, percent = _bytes(measured.total), _bytes(measured.available), _number(measured.percent, high=100)
            if total and available is not None and available <= total and percent is not None:
                memory.update(total_bytes=total, available_bytes=available, percent=percent,
                              available=True, reason_code=None, reason=None)
        except (psutil.Error, OSError, ValueError):
            pass
        disk = {"label": "Диск рабочей папки приложения",
                "note": "Измеряется один диск рабочей папки; ночные результаты могут сохраняться на другом диске.",
                "total_bytes": None, "free_bytes": None, "percent": None,
                "available": False, "reason_code": "disk_read_unavailable", "reason": "Показатели диска рабочей папки недоступны."}
        try:
            directory = self._state_root.absolute()
            while not directory.exists() and directory.parent != directory:
                directory = directory.parent
            measured = psutil.disk_usage(str(directory))
            total, free, percent = _bytes(measured.total), _bytes(measured.free), _number(measured.percent, high=100)
            if total and free is not None and free <= total and percent is not None:
                disk.update(total_bytes=total, free_bytes=free, percent=percent,
                            available=True, reason_code=None, reason=None)
        except (psutil.Error, OSError, ValueError):
            pass
        return {"cpu": cpu, "memory": memory, "disk": disk, "sampled_at": _utc()}

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            if self._system is None or (not self._closed and now >= self._system_until):
                if self._closed:
                    raise RuntimeError("Монитор ресурсов закрыт.")
                self._system = self._sample_system()
                self._system_until = time.monotonic() + SYSTEM_CACHE_SECONDS
            if self._gpu is None or (not self._closed and now >= self._gpu_until):
                try:
                    executable = shutil.which("nvidia-smi")
                    self._gpu = (_bounded_nvidia([executable, *GPU_QUERY]) if executable else
                                 _gpu_unavailable("nvidia_smi_missing", "Датчик NVIDIA не установлен или недоступен."))
                except OSError:
                    self._gpu = _gpu_unavailable("nvidia_smi_unavailable", "Не удалось найти датчик NVIDIA.")
                self._gpu_until = time.monotonic() + GPU_CACHE_SECONDS
            return deepcopy({"schema_version": 1, "scope": "system_wide_not_training_process",
                "scope_note": "Показатели всего компьютера, а не конкретного процесса обучения.",
                **self._system, "gpu": self._gpu, "runtime": self._runtime,
                "training": {"cpu_threads": 8, "gpu_enabled": False,
                    "note": "Ночная серия использует до 8 потоков CPU. Обучение на GPU в ней не включено."}})

    get = snapshot

    def close(self):
        with self._lock:
            self._closed = True
            self._cpu_worker.shutdown(wait=False, cancel_futures=True)
