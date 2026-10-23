"""Local fitted-model exports; inspecting or downloading never unpickles a file.

These artifacts are produced only by the trusted fixed-recipe worker.  A joblib
file is Python-specific and should only be loaded when its producer is trusted.
The HTTP/report import boundary does not accept or load trained model files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

from agent.features import normalize_feature_config
from agent.model_config import validate_model_config
from app.file_io import replace_file


MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_manifest.json"
RECIPE_FILENAME = "training_recipe.json"
MAX_MODEL_BYTES = 2 * 1024**3
MAX_METADATA_BYTES = 256 * 1024
MIN_FREE_BYTES = 64 * 1024**2
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SUMMARY_FIELDS = {"available", "format", "filename", "manifest_filename", "recipe_filename",
                   "sha256", "size_bytes", "trained_on", "validation_model_is_distinct"}
_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "catboost", "lightgbm")


def _path(path, *, required=False):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Путь экспорта модели не должен содержать переходы.")
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("Ссылки в экспорте модели не поддерживаются.")
    if required:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Артефакт модели должен быть обычным файлом.")
    elif path.exists() and not path.is_file():
        raise ValueError("Путь артефакта занят каталогом.")
    return path


def _hash_file(path):
    path = _path(path, required=True)
    if not 0 < path.stat().st_size <= MAX_MODEL_BYTES:
        raise ValueError("Размер файла модели вне допустимых границ.")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Артефакт модели должен быть обычным файлом.")
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_MODEL_BYTES:
                raise ValueError("Размер файла модели вне допустимых границ.")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if total != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Файл модели изменился во время проверки.")
    return digest.hexdigest()


def _payload(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _metadata_bytes(path):
    path = _path(path, required=True)
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Метаданные модели слишком велики.")
    return raw


def _read_json(path):
    raw = _metadata_bytes(path)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Метаданные модели должны быть объектом.")
    return value


class _CappedWriter:
    """Keep even compressed serialization below its original disk allowance."""

    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, value):
        if self.written + len(value) > self.limit:
            raise ValueError("Файл обученной модели превысил лимит экспорта.")
        written = self.stream.write(value)
        self.written += written
        return written

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _write_json(path, value):
    path = _path(path)
    payload = _payload(value)
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError("Метаданные модели слишком велики.")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".model-export-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        replace_file(temporary, path, check=lambda: (_path(temporary, required=True), _path(path)))
    finally:
        temporary.unlink(missing_ok=True)


def validate_artifact_summary(value):
    """Return the small allowlisted object safe to retain in job history."""
    if not isinstance(value, dict) or type(value.get("available")) is not bool:
        raise ValueError("Некорректное описание сохранённой модели.")
    if value["available"] is False:
        reason = value.get("reason")
        if (set(value) != {"available", "reason"} or not isinstance(reason, str)
                or not reason.strip() or len(reason) > 500 or "\x00" in reason):
            raise ValueError("Нужна причина отсутствия сохранённой модели.")
        return dict(value)
    if (set(value) != _SUMMARY_FIELDS or value.get("format") != "joblib"
            or value.get("filename") != MODEL_FILENAME or value.get("manifest_filename") != MANIFEST_FILENAME
            or value.get("recipe_filename") != RECIPE_FILENAME or value.get("trained_on") != "full_train"
            or value.get("validation_model_is_distinct") is not True
            or not isinstance(value.get("sha256"), str) or not _HASH.fullmatch(value["sha256"])
            or type(value.get("size_bytes")) is not int or not 0 < value["size_bytes"] <= MAX_MODEL_BYTES):
        raise ValueError("Некорректные свойства сохранённой модели.")
    return dict(value)


def _validate_manifest(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Неизвестная версия описания модели.")
    summary = validate_artifact_summary(value.get("artifact"))
    if summary["available"] is not True:
        raise ValueError("Описание экспорта не содержит модель.")
    recipe = value.get("recipe")
    if not isinstance(recipe, dict) or set(recipe) != {
            "version", "action", "task_type", "model_config", "feature_config", "validation", "training_device"}:
        raise ValueError("Повреждён рецепт сохранённой модели.")
    if recipe["version"] != 1 or recipe["task_type"] not in {"classification", "regression"}:
        raise ValueError("Неизвестный тип модели.")
    if recipe["training_device"] not in {"cpu", "gpu"} or (
            recipe["training_device"] == "gpu" and recipe["action"] != "catboost"):
        raise ValueError("Неизвестное устройство обучения модели.")
    if validate_model_config(recipe["model_config"], action=recipe["action"]) != recipe["model_config"]:
        raise ValueError("Изменились параметры модели.")
    if normalize_feature_config(recipe["feature_config"]) != recipe["feature_config"]:
        raise ValueError("Изменились настройки обработки признаков.")
    from app.splits import normalize_validation
    if normalize_validation(recipe["validation"]) != recipe["validation"]:
        raise ValueError("Изменились настройки проверочной выборки.")
    schema = value.get("schema")
    if (not isinstance(schema, dict) or set(schema) != {
            "features", "numeric", "categorical", "id_column", "target_column", "classes"}
            or any(not isinstance(schema.get(name), list) for name in ("features", "numeric", "categorical", "classes"))
            or any(not isinstance(column, str) or not column or "\x00" in column
                   for name in ("features", "numeric", "categorical") for column in schema[name])
            or not schema["features"] or len(set(schema["features"])) != len(schema["features"])
            or set(schema["numeric"]) & set(schema["categorical"])
            or set(schema["features"]) != set(schema["numeric"]) | set(schema["categorical"])
            or any(not isinstance(schema.get(name), str) or not schema[name] for name in ("id_column", "target_column"))):
        raise ValueError("Повреждена схема признаков модели.")
    fit = value.get("fit")
    if (not isinstance(fit, dict) or fit.get("scope") != "full_train"
            or type(fit.get("rows")) is not int or fit["rows"] < 2
            or fit.get("validation_metric_source") != "separate_holdout_fit"
            or not isinstance(fit.get("validation_split"), dict)):
        raise ValueError("Повреждено описание обучения модели.")
    for key in ("dataset_fingerprint", "comparison_key", "recipe_sha256"):
        if not isinstance(value.get(key), str) or not _HASH.fullmatch(value[key]):
            raise ValueError("Повреждены отпечатки модели.")
    if hashlib.sha256(_payload(recipe)).hexdigest() != value["recipe_sha256"]:
        raise ValueError("Рецепт модели не совпадает с отпечатком.")
    if not isinstance(value.get("libraries"), dict) or not isinstance(value.get("python"), str):
        raise ValueError("Неизвестно окружение сохранённой модели.")
    return summary


def artifact_download_path(job_dir, kind):
    """Verify an allowlisted local export, without loading executable pickle."""
    names = {"model": MODEL_FILENAME, "manifest": MANIFEST_FILENAME, "recipe": RECIPE_FILENAME}
    if not isinstance(kind, str) or kind not in names:
        raise ValueError("Неизвестный файл модели.")
    root = Path(job_dir).absolute()
    manifest = _read_json(root / MANIFEST_FILENAME)
    summary = _validate_manifest(manifest)
    model_path = _path(root / MODEL_FILENAME, required=True)
    if model_path.stat().st_size != summary["size_bytes"] or _hash_file(model_path) != summary["sha256"]:
        raise ValueError("Сохранённая модель изменилась после обучения.")
    recipe_path = _path(root / RECIPE_FILENAME, required=True)
    raw_recipe = _metadata_bytes(recipe_path)
    if (json.loads(raw_recipe.decode("utf-8")) != manifest["recipe"]
            or hashlib.sha256(raw_recipe).hexdigest() != manifest["recipe_sha256"]):
        raise ValueError("Сохранённый рецепт модели изменился.")
    return _path(root / names[kind], required=True)


def save_model_artifacts(job_dir, pipeline, *, recipe, schema, train_rows, split,
                         dataset_fingerprint, comparison_key):
    """Serialize an already full-data-fitted pipeline; this never calls fit."""
    import joblib

    root = Path(job_dir).absolute()
    model_path, manifest_path, recipe_path = [_path(root / name) for name in
                                             (MODEL_FILENAME, MANIFEST_FILENAME, RECIPE_FILENAME)]
    if any(path.exists() for path in (model_path, manifest_path, recipe_path)):
        raise ValueError("Экспорт модели уже существует; старые файлы не заменяются.")
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MIN_FREE_BYTES:
        raise ValueError("Для сохранения модели недостаточно свободного места.")
    temporary = None
    installed = []
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".model-export-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            joblib.dump(pipeline, _CappedWriter(stream, min(MAX_MODEL_BYTES, free_bytes - MIN_FREE_BYTES // 2)), compress=3)
            stream.flush()
            os.fsync(stream.fileno())
        model_hash = _hash_file(temporary)
        summary = validate_artifact_summary({
            "available": True, "format": "joblib", "filename": MODEL_FILENAME,
            "manifest_filename": MANIFEST_FILENAME, "recipe_filename": RECIPE_FILENAME,
            "sha256": model_hash, "size_bytes": temporary.stat().st_size,
            "trained_on": "full_train", "validation_model_is_distinct": True,
        })
        libraries = {}
        for name in _PACKAGES:
            try:
                libraries[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                libraries[name] = None
        manifest = {
            "version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact": summary, "recipe": recipe, "schema": schema,
            "fit": {"scope": "full_train", "rows": train_rows,
                    "validation_metric_source": "separate_holdout_fit", "validation_split": split},
            "dataset_fingerprint": dataset_fingerprint, "comparison_key": comparison_key,
            "recipe_sha256": hashlib.sha256(_payload(recipe)).hexdigest(),
            "python": sys.version.split()[0], "libraries": libraries,
            "usage": "Локальная обученная модель. Загружайте joblib только от доверенного автора "
                     "в совместимом окружении с модулями проекта; сервер файл не исполняет.",
        }
        _validate_manifest(manifest)
        replace_file(temporary, model_path, check=lambda: (_path(temporary, required=True), _path(model_path)))
        installed.append(model_path)
        _write_json(recipe_path, recipe)
        installed.append(recipe_path)
        # The manifest is the final commit marker: incomplete sets are never downloadable.
        _write_json(manifest_path, manifest)
        installed.append(manifest_path)
        artifact_download_path(root, "model")
        return summary
    except Exception:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
