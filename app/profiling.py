"""Read-only, bounded data discovery and profiling for the local dashboard."""

from __future__ import annotations

from collections import OrderedDict
import copy
import csv
import hashlib
from itertools import islice
import os
from pathlib import Path
import re
import threading

import numpy as np
import pandas as pd


MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_COLUMNS = 512
MAX_TEXT = 500
MAX_TASK_TEXT = 100_000
MAX_FOLDERS = 2000
PREVIEW_ROWS = 100
FILES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
NAMES = {"regression": "Энергопотребление зданий", "classification": "Сетевые атаки"}


def _short(value: object, limit: int = MAX_TEXT) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _task_id(name: str) -> str:
    if len(name) <= 128 and re.fullmatch(r"[A-Za-z0-9_-]+", name) and not name.startswith("task-"):
        return name
    return "task-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]


def _linked(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _safe(path: Path, root: Path, *, directory: bool = False) -> Path:
    """Reject symlinks and Windows junctions, including links inside the task."""
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ValueError("Путь находится за пределами папки данных.") from error
    current = root.absolute()
    if _linked(current):
        raise ValueError("Символические ссылки и junction-папки в данных не поддерживаются.")
    for part in relative.parts:
        current = current / part
        if _linked(current):
            raise ValueError("Символические ссылки и junction-папки в данных не поддерживаются.")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Путь находится за пределами папки данных.")
    if directory:
        if not current.is_dir():
            raise ValueError("Папка задачи недоступна.")
    else:
        if not current.is_file():
            raise ValueError(f"Не найден файл {_short(path.name)}.")
        if current.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Файл {_short(path.name)} превышает лимит 512 МБ.")
    return current


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if not header or len(header) > MAX_COLUMNS:
        raise ValueError(f"{path.name}: требуется от 1 до {MAX_COLUMNS} столбцов.")
    if any(not value.strip() or len(value) > MAX_TEXT for value in header):
        raise ValueError(f"{path.name}: пустое или слишком длинное имя столбца.")
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: повторяющиеся имена столбцов.")
    return header


def _read(path: Path) -> pd.DataFrame:
    header = _header(path)
    # Pandas may infer an index for a malformed first data row. Verify record
    # widths explicitly so a damaged file cannot silently become a valid table.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        next(reader)
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path.name}: неверное число полей в записи {number}.")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _distribution(series: pd.Series, dtype: str, *, target: bool = False) -> tuple[dict, dict, int]:
    missing = int(series.eq("").sum())
    present = series[series.ne("")]
    if dtype in {"numeric", "binary"}:
        numeric = pd.to_numeric(present, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        invalid = len(numeric) - len(finite)
        stats = {}
        bins = []
        if len(finite):
            with np.errstate(over="ignore", invalid="ignore"):
                raw_stats = {"min": np.min(finite), "max": np.max(finite),
                             "mean": np.mean(finite), "median": np.median(finite),
                             "std": np.std(finite)}
            stats = {key: float(value) if np.isfinite(value) else None
                     for key, value in raw_stats.items()}
            unique = np.unique(finite)
            if dtype == "binary" or len(unique) <= 5:
                bins = [{"label": f"{value:g}", "count": int(np.count_nonzero(finite == value))}
                        for value in unique[:100]]
                if len(unique) > 100:
                    bins.append({"label": "Остальные значения", "count": int(np.count_nonzero(finite > unique[99]))})
            else:
                # Scaling prevents edge overflow for otherwise finite numbers.
                scale = max(abs(float(finite.min())), abs(float(finite.max())), 1.0)
                counts, edges = np.histogram(finite / scale, bins=min(20, len(unique)))
                bins = [{"label": f"{edges[index] * scale:.4g} – {edges[index + 1] * scale:.4g}",
                         "count": int(count)} for index, count in enumerate(counts)]
            for item in bins:
                item["percent"] = 100 * item["count"] / len(finite)
        return {"kind": "histogram", "bins": bins, "missing_count": missing,
                "invalid_count": int(invalid)}, stats, int(invalid)
    counts = present.value_counts(dropna=False)
    limit = (100 if len(counts) <= 100 else 99) if target else 15
    bins = [{"label": _short(label), "count": int(count)} for label, count in counts.iloc[:limit].items()]
    if len(counts) > limit:
        bins.append({"label": "Остальные категории", "count": int(counts.iloc[limit:].sum())})
    for item in bins:
        item["percent"] = 100 * item["count"] / len(present)
    return {"kind": "categories", "bins": bins, "missing_count": missing}, {}, 0


def _finite_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series[series.ne("")], errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _numeric_shift(name: str, train: pd.Series, test: pd.Series) -> dict | None:
    training, testing = _finite_values(train), _finite_values(test)
    if not len(training) or not len(testing):
        return None
    # Work on a scaled training distribution to keep finite inputs finite even
    # when their raw IQR or variance would overflow a float.
    scale = max(abs(float(training.min())), abs(float(training.max())), 1.0)
    test_scale = max(abs(float(testing.min())), abs(float(testing.max())), 1.0)
    normalized = training / scale
    lower, median, upper = np.percentile(normalized, [25, 50, 75])
    train_median = float(median * scale)
    test_median = float(np.median(testing / test_scale) * test_scale)
    spread = float(upper - lower)
    if spread <= 0:
        spread = float(np.std(normalized))
    if spread > 0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw_score = np.abs(test_median / scale - median) / spread
        score = float(raw_score) if np.isfinite(raw_score) else None
    else:
        # A changed constant cannot be normalized by zero dispersion. Keep the
        # two medians visible and report an undefined score instead of infinity.
        score = 0.0 if train_median == test_median else None
    return {"name": name, "train_median": train_median, "test_median": test_median,
            "score": score}


def _quality(train: pd.DataFrame, test: pd.DataFrame, metadata: pd.DataFrame,
             id_column: str, target: str, task_type: str) -> dict:
    features = [name for name in train.columns if name not in {id_column, target}]
    train_ids = set(train.loc[train[id_column].ne(""), id_column])
    test_ids = set(test.loc[test[id_column].ne(""), id_column])
    result = {
        "train_duplicate_ids": int(train[id_column].duplicated().sum()),
        "test_duplicate_ids": int(test[id_column].duplicated().sum()),
        "id_overlap_count": len(train_ids & test_ids),
        "train_duplicate_feature_rows": int(train.duplicated(subset=features).sum()) if features else 0,
        "test_duplicate_feature_rows": int(test.duplicated(subset=features).sum()) if features else 0,
        "constant_features": [], "new_categories": [], "missing_changes": [],
        "numeric_shift": [], "class_balance": None,
    }
    for name in features:
        if train[name].nunique(dropna=False) <= 1:
            result["constant_features"].append(name)
        train_percent = float(train[name].eq("").sum()) / len(train) * 100
        test_percent = float(test[name].eq("").sum()) / len(test) * 100
        if abs(train_percent - test_percent) >= 1:
            result["missing_changes"].append({"name": name, "train_percent": train_percent,
                                               "test_percent": test_percent})
        dtype = metadata.loc[name, "dtype"]
        if dtype == "category":
            known = set(train.loc[train[name].ne(""), name])
            unseen = set(test.loc[test[name].ne(""), name]) - known
            if unseen:
                result["new_categories"].append({
                    "name": name, "count": len(unseen),
                    "examples": [_short(value, 59) for value in sorted(unseen)[:3]],
                })
        if dtype in {"numeric", "binary"}:
            shift = _numeric_shift(name, train[name], test[name])
            if shift is not None:
                result["numeric_shift"].append(shift)
    result["new_categories"].sort(key=lambda item: (-item["count"], item["name"]))
    result["new_categories"] = result["new_categories"][:20]
    result["missing_changes"].sort(key=lambda item: (-abs(item["train_percent"] - item["test_percent"]), item["name"]))
    result["missing_changes"] = result["missing_changes"][:10]
    result["numeric_shift"].sort(key=lambda item: (item["score"] is not None,
                                                   -(item["score"] or 0), item["name"]))
    result["numeric_shift"] = result["numeric_shift"][:20]
    if task_type == "classification":
        counts = train.loc[train[target].ne(""), target].value_counts()
        total = int(counts.sum())
        if total:
            largest, smallest = int(counts.max()), int(counts.min())
            result["class_balance"] = {
                "classes": [{"label": _short(label), "count": int(count), "percent": 100 * int(count) / total}
                            for label, count in counts.iloc[:100].items()],
                "largest_count": largest, "smallest_count": smallest,
                "ratio": largest / smallest,
            }
        else:
            result["class_balance"] = {"classes": [], "largest_count": 0,
                                       "smallest_count": 0, "ratio": None}
    return result


def _metric(text: str) -> dict:
    """Use the shared explicit metric contract without inferring from dtype."""
    from agent.task_contract import parse_task_metric
    try:
        spec = parse_task_metric(text)
    except ValueError:
        return {"name": "Не определена", "higher_is_better": None}
    return {"name": spec.name, "higher_is_better": spec.higher_is_better}


def _task_semantics(text: str, fallback: str):
    from agent.task_contract import parse_task_metric
    try:
        spec = parse_task_metric(text)
        return spec.task_type, spec.prediction_kind
    except ValueError:
        return fallback, None


def _sample_columns_valid(columns, id_column, target, prediction_kind):
    if prediction_kind == "probabilities":
        return len(columns) >= 2 and id_column in columns and len(columns) == len(set(columns))
    return len(columns) == 2 and set(columns) == {id_column, target}



class DataCatalog:
    """The HTTP layer addresses discovered tasks by ID, never by supplied path."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).expanduser().absolute()
        self._cache: OrderedDict[str, tuple[tuple, dict, dict]] = OrderedDict()
        self._fingerprints: OrderedDict[str, tuple[tuple, str]] = OrderedDict()
        self._errors: dict[str, tuple[tuple, str]] = {}
        self._lock = threading.RLock()

    def _folders(self) -> dict[str, Path]:
        if not self.data_root.is_dir() or _linked(self.data_root):
            return {}
        entries = list(islice(self.data_root.iterdir(), MAX_FOLDERS + 1))
        if len(entries) > MAX_FOLDERS:
            raise ValueError(f"В папке данных более {MAX_FOLDERS} объектов; выберите папку только с задачами.")
        result = {}
        for folder in sorted(entries, key=lambda path: path.name):
            if folder.name.startswith(".") or not folder.is_dir() or _linked(folder):
                continue
            if (folder / "task.md").exists() or _linked(folder / "task.md"):
                result[_task_id(folder.name)] = folder
        return result

    def resolve(self, task_id: str) -> Path:
        with self._lock:
            folder = self._folders().get(task_id)
            if folder is None:
                raise ValueError("Задача не найдена.")
            return _safe(folder, self.data_root, directory=True)

    def _inputs(self, folder: Path) -> tuple[Path, dict[str, Path]]:
        _safe(folder / "task.md", self.data_root)
        samples = []
        for visited, (parent, directories, filenames) in enumerate(os.walk(folder, followlinks=False), 1):
            if visited > MAX_FOLDERS:
                raise ValueError(f"В задаче более {MAX_FOLDERS} папок; поиск данных остановлен.")
            parent = Path(parent)
            directories[:] = [name for name in directories if not name.startswith(".")
                              and not _linked(parent / name)]
            if len(directories) + visited > MAX_FOLDERS:
                raise ValueError(f"В задаче более {MAX_FOLDERS} папок; поиск данных остановлен.")
            if "sample_submission.csv" in filenames:
                samples.append(parent / "sample_submission.csv")
        if len(samples) != 1:
            raise ValueError(f"Нужен один sample_submission.csv; найдено: {len(samples)}.")
        data_dir = samples[0].parent
        paths = {name: _safe(data_dir / name, self.data_root) for name in FILES}
        return data_dir, paths

    def _metadata(self, paths: dict[str, Path]) -> tuple[pd.DataFrame, str, str, str]:
        metadata = _read(paths["data_dictionary.csv"])
        if not {"feature", "role", "dtype"}.issubset(metadata.columns):
            raise ValueError("В словаре нужны столбцы feature, role, dtype.")
        if len(metadata) > MAX_COLUMNS or metadata["feature"].duplicated().any():
            raise ValueError("Словарь содержит слишком много или повторяющиеся признаки.")
        metadata["role"] = metadata["role"].str.strip().str.lower()
        metadata["dtype"] = metadata["dtype"].str.strip().str.lower()
        ids = metadata.loc[metadata["role"].eq("id"), "feature"].tolist()
        targets = metadata.loc[metadata["role"].eq("target"), "feature"].tolist()
        if len(ids) != 1 or len(targets) != 1:
            raise ValueError("Словарь должен объявлять один id и один target.")
        metadata = metadata.set_index("feature")
        target_dtype = metadata.loc[targets[0], "dtype"]
        task_type = {"numeric": "regression", "category": "classification"}.get(target_dtype, "unknown")
        return metadata, ids[0], targets[0], task_type

    @staticmethod
    def _signature(paths: dict[str, Path], folder: Path) -> tuple:
        return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                     for path in [folder / "task.md", *paths.values()])

    def discover(self) -> list[dict]:
        with self._lock:
            tasks = []
            for task_id, folder in self._folders().items():
                summary = {"id": task_id, "name": NAMES.get(folder.name, _short(folder.name)),
                           "task_type": "unknown", "target": None, "available": False,
                           "error": None, "has_baseline": False, "run_count": 0}
                baseline = folder / "baseline_metrics.json"
                summary["has_baseline"] = baseline.is_file() and not _linked(baseline)
                runs = folder / ".agent_runs"
                if runs.is_dir() and not _linked(runs):
                    summary["run_count"] = sum(path.is_dir() and not _linked(path)
                                               for path in runs.iterdir())
                try:
                    _, paths = self._inputs(folder)
                    _, id_column, target, task_type = self._metadata(paths)
                    task_type, prediction_kind = _task_semantics(self.task_text(task_id), task_type)
                    train_header = _header(paths["train.csv"])
                    test_header = _header(paths["test.csv"])
                    sample_header = _header(paths["sample_submission.csv"])
                    if id_column not in train_header or target not in train_header or id_column not in test_header:
                        raise ValueError("Заявленные id/target отсутствуют в таблицах.")
                    if not _sample_columns_valid(sample_header, id_column, target, prediction_kind):
                        raise ValueError("Столбцы sample_submission.csv не совпадают с id/target.")
                    signature = self._signature(paths, folder)
                    old_error = self._errors.get(task_id)
                    if old_error and old_error[0] == signature:
                        raise ValueError(old_error[1])
                    summary.update(task_type=task_type, target=target, available=True)
                except (OSError, ValueError, csv.Error, UnicodeError, pd.errors.ParserError) as error:
                    summary["error"] = _short(str(error), 500)
                tasks.append(summary)
            return tasks

    def task_text(self, task_id: str) -> str:
        folder = self.resolve(task_id)
        path = _safe(folder / "task.md", self.data_root)
        with path.open(encoding="utf-8-sig") as stream:
            text = stream.read(MAX_TASK_TEXT + 1)
        if len(text) > MAX_TASK_TEXT:
            return text[:MAX_TASK_TEXT] + "\n\n[Условие сокращено до 100 000 символов.]"
        return text

    def fingerprint(self, task_id: str) -> str:
        """Hash exact input bytes under stable names, independent of machine path.

        Length framing makes file boundaries unambiguous. The same five input
        files copied into another task folder produce the same SHA-256 value.
        """
        with self._lock:
            folder = self.resolve(task_id)
            _, paths = self._inputs(folder)
            signature = self._signature(paths, folder)
            cached = self._fingerprints.get(task_id)
            if cached and cached[0] == signature:
                self._fingerprints.move_to_end(task_id)
                return cached[1]
            digest = hashlib.sha256(b"alfa-automl-dataset-v1\0")
            canonical = {"task.md": folder / "task.md", **paths}
            for name in sorted(canonical):
                path = _safe(canonical[name], self.data_root)
                size = path.stat().st_size
                encoded_name = name.encode("utf-8")
                digest.update(len(encoded_name).to_bytes(4, "big"))
                digest.update(encoded_name)
                digest.update(size.to_bytes(8, "big"))
                received = 0
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        received += len(block)
                        if received > size:
                            raise ValueError("Данные изменились во время вычисления отпечатка. Повторите действие.")
                        digest.update(block)
                if received != size:
                    raise ValueError("Данные изменились во время вычисления отпечатка. Повторите действие.")
            if signature != self._signature(paths, folder):
                raise ValueError("Данные изменились во время вычисления отпечатка. Повторите действие.")
            value = digest.hexdigest()
            self._fingerprints[task_id] = (signature, value)
            self._fingerprints.move_to_end(task_id)
            while len(self._fingerprints) > 4:
                self._fingerprints.popitem(last=False)
            return value

    def _load(self, task_id: str) -> tuple[dict, dict]:
        folder = self.resolve(task_id)
        _, paths = self._inputs(folder)
        signature = self._signature(paths, folder)
        cached = self._cache.get(task_id)
        if cached and cached[0] == signature:
            self._cache.move_to_end(task_id)
            return cached[1], cached[2]
        try:
            profile, details = self._build(task_id, folder, paths)
        except (OSError, ValueError, csv.Error, UnicodeError, pd.errors.ParserError) as error:
            self._errors[task_id] = (signature, _short(str(error), 500))
            self._cache.pop(task_id, None)
            raise ValueError(_short(str(error), 500)) from error
        if signature != self._signature(paths, folder):
            raise ValueError("Данные изменились во время чтения. Обновите страницу.")
        self._errors.pop(task_id, None)
        self._cache[task_id] = (signature, profile, details)
        self._cache.move_to_end(task_id)
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return profile, details

    def profile(self, task_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._load(task_id)[0])

    def column(self, task_id: str, column_name: str) -> dict:
        with self._lock:
            details = self._load(task_id)[1]
            if column_name not in details:
                raise ValueError("Столбец не найден.")
            return copy.deepcopy(details[column_name])

    def _build(self, task_id: str, folder: Path, paths: dict[str, Path]) -> tuple[dict, dict]:
        metadata, id_column, target, task_type = self._metadata(paths)
        task_type, prediction_kind = _task_semantics(self.task_text(task_id), task_type)
        train = _read(paths["train.csv"])
        test = _read(paths["test.csv"])
        sample_header = _header(paths["sample_submission.csv"])
        if id_column not in train or target not in train or id_column not in test:
            raise ValueError("Заявленные id/target отсутствуют в таблицах.")
        if not _sample_columns_valid(sample_header, id_column, target, prediction_kind):
            raise ValueError("Столбцы sample_submission.csv не совпадают с id/target.")
        if target in test or set(train.columns) - {target} != set(test.columns):
            raise ValueError("Признаки train.csv и test.csv не совпадают.")
        if train.empty or test.empty:
            raise ValueError("Обучающая и тестовая таблицы должны содержать строки.")
        missing_metadata = [name for name in train.columns if name not in metadata.index]
        if missing_metadata:
            raise ValueError("В словаре не описаны столбцы: " + ", ".join(missing_metadata[:5]))
        warnings = []
        for name, frame in (("train.csv", train), ("test.csv", test)):
            if frame[id_column].eq("").any():
                warnings.append(f"{name}: есть пустые идентификаторы.")
            if frame[id_column].duplicated().any():
                warnings.append(f"{name}: есть повторяющиеся идентификаторы.")
        columns, details, missing = [], {}, []
        type_counts = {"numeric": 0, "category": 0, "binary": 0, "unknown": 0}
        for name in train.columns:
            row = metadata.loc[name]
            dtype = row["dtype"]
            role = row["role"]
            description = _short(row.get("description_ru", ""))
            count = int(train[name].eq("").sum())
            percent = count / len(train) * 100
            unique_count = int(train.loc[train[name].ne(""), name].nunique())
            info = {"name": name, "role": role, "dtype": dtype, "description": description,
                    "missing_count": count, "missing_percent": round(percent, 4), "unique_count": unique_count}
            columns.append(info)
            distribution_dtype = "category" if name == target and task_type == "classification" else dtype
            distribution, stats, invalid = _distribution(train[name], distribution_dtype, target=name == target)
            details[name] = {**info, "stats": stats, "distribution": distribution}
            if invalid:
                warnings.append(f"{name}: {invalid} непустых значений не являются конечными числами; они исключены из числовой статистики.")
            if name not in {id_column, target}:
                type_counts[dtype if dtype in type_counts else "unknown"] += 1
            if count:
                missing.append({"name": name, "count": count, "percent": round(percent, 4)})
        missing.sort(key=lambda item: (-item["count"], item["name"]))
        missing_cells = sum(column["missing_count"] for column in columns)
        preview_text_limit = max(8, min(200, 1_000_000 // (PREVIEW_ROWS * len(train.columns))))
        profile = {
            "id": task_id, "name": NAMES.get(folder.name, _short(folder.name)),
            "task_type": task_type, "id_column": id_column, "target": target,
            "metric": _metric(self.task_text(task_id)),
            "train_rows": len(train), "test_rows": len(test), "feature_count": len(train.columns) - 2,
            "type_counts": type_counts, "missing_cells": missing_cells,
            "missing_percent": round(100 * missing_cells / train.size, 4),
            "columns": columns,
            "preview": {"columns": list(train.columns), "rows": [[_short(value, preview_text_limit) for value in row]
                        for row in train.head(PREVIEW_ROWS).itertuples(index=False, name=None)],
                        "total_rows": len(train), "limit": PREVIEW_ROWS},
            "target_distribution": details[target]["distribution"], "missing": missing,
            "quality": _quality(train, test, metadata, id_column, target, task_type),
            "dataset_fingerprint": self.fingerprint(task_id),
            "warnings": warnings,
        }
        return profile, details
