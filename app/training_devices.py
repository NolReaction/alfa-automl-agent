"""Explicit local CatBoost GPU training, separate from contest model recipes.

Device discovery never trains a model. The app's supervisor still owns the
single machine lock, deadline and process tree for CPU and GPU runs alike.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from agent.models import (
    CatBoostClassificationAdapter, CatBoostRegressionAdapter,
    MAX_NATIVE_CATEGORIES, MAX_NATIVE_CATEGORY_CELLS, THREADS,
)
from agent.target_transform import Log1pTargetRegressor
from offline_baseline import SEED


ROOT = Path(__file__).resolve().parents[1]
GPU_RAM_PART = .75
PROBE_TIMEOUT = 10
CAPABILITY_CACHE_SECONDS = 60
_capability_lock = threading.Lock()
_capability_cache = None
_capability_time = 0.0


def normalize_training_device(value=None, *, action=None):
    value = "cpu" if value is None else value
    if type(value) is not str or value not in {"cpu", "gpu"}:
        raise ValueError("Выберите CPU или GPU для обучения.")
    if value == "gpu" and action != "catboost":
        raise ValueError("Обучение на GPU доступно только для отдельной модели CatBoost.")
    return value


def _empty_gpu(reason):
    return {"available": False, "label": "GPU · CatBoost", "reason": reason,
            "device_count": 0, "device_name": None, "driver_version": None,
            "total_memory_mib": None, "catboost_version": None,
            "training_verified": False}


def _probe_in_process():
    """Fixed child entry point: query libraries/device only; never call fit."""
    import shutil
    result = _empty_gpu("CatBoost не обнаружил доступную видеокарту.")
    try:
        import catboost
        from catboost.utils import get_gpu_device_count
        result["catboost_version"] = str(catboost.__version__)[:80]
        count = get_gpu_device_count()
        if type(count) is not int or not 0 <= count <= 1024:
            raise ValueError("Invalid device count")
        result["device_count"] = count
        result["available"] = count > 0
        result["reason"] = None if count else result["reason"]
    except Exception as error:
        # Provider exception text can contain local paths; surface a bounded
        # category rather than arbitrary library diagnostics over HTTP.
        result["reason"] = "Проверка GPU через CatBoost не выполнена: " + type(error).__name__
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run([
                executable, "--query-gpu=name,driver_version,memory.total", "--id=0",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=3, check=False)
            if completed.returncode == 0:
                rows = list(csv.reader(io.StringIO(completed.stdout[:8192])))
                if len(rows) == 1 and len(rows[0]) == 3:
                    name, driver, memory = [part.strip() for part in rows[0]]
                    result["device_name"] = name[:200] or None
                    result["driver_version"] = driver[:80] or None
                    amount = float(memory)
                    if math.isfinite(amount) and 0 < amount <= 10_000_000:
                        result["total_memory_mib"] = amount
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass  # Device discovery can succeed without nvidia-smi metadata.
    return result


def _validate_capability(value):
    if not isinstance(value, dict) or set(value) != set(_empty_gpu(None)):
        raise ValueError("Invalid GPU capability response")
    if (type(value["available"]) is not bool or type(value["device_count"]) is not int
            or not 0 <= value["device_count"] <= 1024
            or value["available"] != (value["device_count"] > 0)
            or value["training_verified"] is not False
            or value["label"] != "GPU · CatBoost"):
        raise ValueError("Invalid GPU capability values")
    for name, limit in (("reason", 300), ("device_name", 200),
                        ("driver_version", 80), ("catboost_version", 80)):
        text = value[name]
        if text is not None and (not isinstance(text, str) or len(text) > limit
                                 or any(ord(char) < 32 for char in text)):
            raise ValueError("Invalid GPU capability text")
    memory = value["total_memory_mib"]
    if memory is not None and (type(memory) not in {int, float}
                              or not math.isfinite(memory) or not 0 < memory <= 10_000_000):
        raise ValueError("Invalid GPU memory metadata")
    return deepcopy(value)


def probe_gpu_capability():
    """Bounded, credential-free child probe; no package installs or model fit."""
    from agent.execution import child_environment
    environment = child_environment()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    code = ("import json; from app.training_devices import _probe_in_process; "
            "print(json.dumps(_probe_in_process(), ensure_ascii=True))")
    try:
        completed = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT,
            env=environment, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT,
            check=False)
        if completed.returncode != 0:
            return _empty_gpu("Процесс проверки GPU завершился с ошибкой.")
        lines = completed.stdout[-16384:].splitlines()
        if not lines:
            return _empty_gpu("Проверка GPU не вернула результат.")
        return _validate_capability(json.loads(lines[-1]))
    except subprocess.TimeoutExpired:
        return _empty_gpu("Проверка GPU превысила лимит 10 секунд.")
    except (OSError, ValueError, TypeError, KeyError):
        return _empty_gpu("Не удалось проверить доступность GPU.")


def training_device_capabilities(*, force=False):
    """Safe cached metadata for the form; available is not proof of training."""
    global _capability_cache, _capability_time
    with _capability_lock:
        now = time.monotonic()
        if force or _capability_cache is None or now - _capability_time >= CAPABILITY_CACHE_SECONDS:
            _capability_cache = probe_gpu_capability()
            _capability_time = time.monotonic()
        return {"default": "cpu", "supported_gpu_actions": ["catboost"],
                "cpu": {"available": True, "label": "CPU · 8 потоков"},
                "gpu": deepcopy(_capability_cache)}


class _LocalGPUCatBoostMixin:
    """Retain the core adapter's fold-fitted category handling on one GPU."""

    def fit(self, X, y, **fit_params):
        from catboost import CatBoostClassifier, CatBoostRegressor
        self.__dict__.pop("_gpu_fit_completed_", None)
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("CatBoost needs a nonempty feature DataFrame")
        self.columns_ = list(X.columns)
        cardinality = sum(X[name].nunique(dropna=False) for name in self.categorical)
        native = (cardinality <= MAX_NATIVE_CATEGORIES
                  and len(X) * len(self.categorical) <= MAX_NATIVE_CATEGORY_CELLS)
        self.category_encoding_ = "native" if native else "ordinal"
        if not native:
            print("CatBoost GPU: using fold-fitted ordinal categories.", flush=True)
            self.category_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=float,
            ).fit(X.loc[:, self.categorical])
        constructor = CatBoostClassifier if self._estimator_type == "classifier" else CatBoostRegressor
        self.estimator_ = constructor(
            **self.parameters, cat_features=list(self.categorical) if native else [],
            random_seed=SEED, thread_count=THREADS, task_type="GPU", devices="0",
            gpu_ram_part=GPU_RAM_PART, allow_writing_files=False, verbose=False,
        )
        self.estimator_.fit(self._frame(X), y, **fit_params)
        self.n_features_in_ = X.shape[1]
        if self._estimator_type == "classifier":
            self.classes_ = np.asarray(self.estimator_.classes_).astype(str)
        self._gpu_fit_completed_ = True
        return self


class LocalGPUCatBoostClassifier(_LocalGPUCatBoostMixin, CatBoostClassificationAdapter):
    pass


class LocalGPUCatBoostRegressor(_LocalGPUCatBoostMixin, CatBoostRegressionAdapter):
    pass


def _model_adapter(pipeline, *, fitted=False):
    model = pipeline.named_steps["model"]
    if isinstance(model, Log1pTargetRegressor):
        return model.regressor_ if fitted else model.estimator
    return model


def prepare_training_pipeline(pipeline, *, action, training_device=None):
    device = normalize_training_device(training_device, action=action)
    evidence = {"version": 1, "requested": device, "backend": device.upper(),
                "fit_completed": False, "backend_verified": False,
                "device_index": None, "device_name": "CPU", "driver_version": None,
                "gpu_device_count": 0, "catboost_version": None,
                "gpu_ram_part": None, "exploratory": device == "gpu"}
    model = _model_adapter(pipeline)
    if device == "cpu":
        if isinstance(model, _LocalGPUCatBoostMixin):
            raise ValueError("GPU-модель не соответствует запрошенному режиму CPU.")
        return pipeline, evidence
    capability = probe_gpu_capability()
    if not capability["available"]:
        raise ValueError(capability["reason"] or "CatBoost GPU недоступен; обучение не запускалось.")
    if isinstance(model, CatBoostClassificationAdapter):
        constructor = LocalGPUCatBoostClassifier
    elif isinstance(model, CatBoostRegressionAdapter):
        constructor = LocalGPUCatBoostRegressor
    else:
        raise ValueError("GPU поддерживает только штатный адаптер CatBoost.")
    replacement = constructor(deepcopy(model.parameters), deepcopy(model.categorical))
    if isinstance(pipeline.named_steps["model"], Log1pTargetRegressor):
        pipeline.set_params(model__estimator=replacement)
    else:
        pipeline.set_params(model=replacement)
    evidence.update(device_index=0, device_name=capability["device_name"] or "GPU 0 (CatBoost)",
        driver_version=capability["driver_version"], gpu_device_count=capability["device_count"],
        catboost_version=capability["catboost_version"], gpu_ram_part=GPU_RAM_PART)
    return pipeline, evidence


def confirm_training_device(pipeline, evidence):
    """Confirm the requested backend only after the final fit really returned."""
    result = validate_training_device_evidence(evidence)
    model = _model_adapter(pipeline, fitted=True)
    check_is_fitted(model)
    if result["requested"] == "gpu":
        if (not isinstance(model, _LocalGPUCatBoostMixin)
                or getattr(model, "_gpu_fit_completed_", False) is not True
                or model.estimator_.get_param("task_type") != "GPU"):
            raise ValueError("Запрошенное обучение CatBoost на GPU не подтверждено.")
    elif isinstance(model, _LocalGPUCatBoostMixin):
        raise ValueError("Фактическое обучение не соответствует режиму CPU.")
    result.update(fit_completed=True, backend_verified=True)
    return validate_training_device_evidence(result, require_fit=True)


def validate_training_device_evidence(value, *, training_device=None, require_fit=False):
    fields = {"version", "requested", "backend", "fit_completed", "backend_verified", "device_index",
              "device_name", "driver_version", "gpu_device_count", "catboost_version", "gpu_ram_part", "exploratory"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Некорректные сведения об устройстве обучения.")
    device = normalize_training_device(value["requested"], action="catboost")
    if (type(value["version"]) is not int or value["version"] != 1
            or training_device is not None and device != training_device
            or value["backend"] != device.upper()
            or any(type(value[key]) is not bool for key in ("fit_completed", "backend_verified", "exploratory"))
            or value["backend_verified"] != value["fit_completed"]
            or require_fit and value["fit_completed"] is not True
            or value["exploratory"] != (device == "gpu")
            or type(value["gpu_device_count"]) is not int or not 0 <= value["gpu_device_count"] <= 1024):
        raise ValueError("Не подтверждён режим устройства обучения.")
    for name, limit in (("device_name", 200), ("driver_version", 80), ("catboost_version", 80)):
        text = value[name]
        if (name == "device_name" and (not isinstance(text, str) or not text)
                or text is not None and (not isinstance(text, str) or len(text) > limit
                                        or any(ord(char) < 32 for char in text))):
            raise ValueError("Некорректное описание устройства обучения.")
    if device == "gpu":
        if type(value["device_index"]) is not int or value["device_index"] != 0 \
                or value["gpu_device_count"] < 1 or type(value["gpu_ram_part"]) not in {int, float} \
                or value["gpu_ram_part"] != GPU_RAM_PART:
            raise ValueError("Некорректные параметры CatBoost GPU.")
    elif (value["device_index"] is not None or value["gpu_device_count"] != 0
          or value["gpu_ram_part"] is not None or value["driver_version"] is not None
          or value["catboost_version"] is not None or value["device_name"] != "CPU"):
        raise ValueError("Сведения о GPU не соответствуют режиму CPU.")
    return deepcopy(value)
