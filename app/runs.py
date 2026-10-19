"""Read local research reports and predictions without executing experiment code."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import heapq
import json
import math
import os
from pathlib import Path
import re
import threading

from validation import validate_predictions


RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
EXPERIMENT_FILE = re.compile(r"experiment_([0-9]{1,12})\.(py|stdout\.txt|stderr\.txt)\Z")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 128 * 1024
MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_RUNS = 200
MAX_FILES = 500
MAX_SAFE_INTEGER = (1 << 53) - 1
_RESULT_CACHE: OrderedDict = OrderedDict()
_RESULT_LOCK = threading.RLock()


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _finite_number(value) -> bool:
    # Compare integers directly: float conversion can overflow for malformed JSON.
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
            and (isinstance(value, int) or math.isfinite(value)))


def _number(value, *, integer=False) -> bool:
    return (_finite_number(value) and value >= 0
            and (not integer or isinstance(value, int)))


def _redact(value):
    """Never load credentials, but hide an inherited key if it occurs in a log."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {_redact(k): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _bad_constant(value):
    raise ValueError("JSON содержит недопустимое числовое значение.")


def _check_json_tree(value):
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 64:
            raise ValueError("JSON содержит слишком много уровней вложенности.")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("JSON содержит недопустимое числовое значение.")
        if isinstance(item, int) and not isinstance(item, bool) and abs(item) > MAX_SAFE_INTEGER:
            raise ValueError("JSON содержит целое число вне точного диапазона браузера.")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


class RunReader:
    """Access only known files inside one previously discovered task directory."""

    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir).absolute()
        if _is_link(self.task_dir) or not self.task_dir.is_dir():
            raise ValueError("Папка задачи недоступна или является ссылкой.")
        self.root = self.task_dir.resolve()

    def _safe(self, path: Path, *, required=False, directory=False) -> Path:
        try:
            parts = path.relative_to(self.task_dir).parts
        except ValueError as error:
            raise ValueError("Доступ за пределы задачи запрещён.") from error
        current = self.task_dir
        if _is_link(current):
            raise ValueError("Чтение символических ссылок запрещено.")
        for part in parts:
            if part in {".", ".."}:
                raise ValueError("Недопустимый путь.")
            current = current / part
            if _is_link(current):
                raise ValueError("Чтение символических ссылок запрещено.")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Доступ за пределы задачи запрещён.")
        if required and not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f"Файл или папка не найдены: {path.name}")
        return path

    def _read_json(self, path: Path) -> dict:
        path = self._safe(path, required=True)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                raise ValueError(f"{path.name} превышает допустимый размер 2 МБ.")
            result = json.loads(raw.decode("utf-8-sig"), parse_constant=_bad_constant)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"Не удалось прочитать {path.name}: повреждённый JSON.") from error
        if not isinstance(result, dict):
            raise ValueError(f"{path.name}: ожидается JSON-объект.")
        _check_json_tree(result)
        return _redact(result)

    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Недопустимый идентификатор запуска.")
        return self._safe(self.task_dir / ".agent_runs" / run_id, required=True, directory=True)

    def _report(self, folder: Path) -> dict:
        report = self._read_json(folder / "run.json")
        if report.get("status") not in {"running", "completed", "fallback", "failed"}:
            raise ValueError("run.json: неизвестный или отсутствующий статус запуска.")
        for name in ("started_at", "task", "model", "reason", "baseline", "error"):
            if name in report and not isinstance(report[name], str):
                raise ValueError(f"run.json: поле {name} должно быть строкой.")
        for name in ("experiments", "successful_experiments", "max_steps"):
            if name in report and not _number(report[name], integer=True):
                raise ValueError(f"run.json: некорректный счётчик {name}.")
        for name in ("elapsed_seconds", "time_budget_seconds"):
            if name in report and not _number(report[name]):
                raise ValueError(f"run.json: некорректная длительность {name}.")
        events = report.get("events", [])
        if not isinstance(events, list) or any(not isinstance(item, dict) for item in events):
            raise ValueError("run.json: события должны быть списком объектов.")
        for event in events:
            if not isinstance(event.get("kind"), str):
                raise ValueError("run.json: у события отсутствует тип.")
            for name in ("step", "tool_index"):
                if name in event and not _number(event[name], integer=True):
                    raise ValueError(f"run.json: некорректный номер события {name}.")
            if "result" in event and not isinstance(event["result"], dict):
                raise ValueError("run.json: результат события должен быть объектом.")
            if "result" in event:
                result = event["result"]
                for name in ("stdout", "stderr", "error"):
                    if name in result and not isinstance(result[name], str):
                        raise ValueError(f"run.json: поле {name} в событии должно быть строкой.")
                if "exit_code" in result and result["exit_code"] is not None and (
                        isinstance(result["exit_code"], bool)
                        or not isinstance(result["exit_code"], int)):
                    raise ValueError("run.json: некорректный код завершения процесса.")
                for name in ("timed_out", "previous_output_restored"):
                    if name in result and not isinstance(result[name], bool):
                        raise ValueError(f"run.json: поле {name} должно быть логическим значением.")
                self._check_output(result.get("output_check"))
        self._check_output(report.get("output_check"))
        api = report.get("api", {})
        if not isinstance(api, dict):
            raise ValueError("run.json: сведения об API должны быть объектом.")
        for name in ("attempts", "responses", "usage_reported_responses", "prompt_tokens",
                     "completion_tokens", "total_tokens"):
            if name in api and not _number(api[name], integer=True):
                raise ValueError(f"run.json: некорректный счётчик API {name}.")
        attempts, responses = api.get("attempts"), api.get("responses")
        reported = api.get("usage_reported_responses")
        if (attempts is not None and responses is not None and responses > attempts
                or reported is not None and responses is not None and reported > responses
                or reported is not None and attempts is not None and reported > attempts):
            raise ValueError("run.json: счётчики ответов API противоречат друг другу.")
        report.setdefault("events", [])
        return report

    @staticmethod
    def _check_output(output):
        if output is None:
            return
        if (not isinstance(output, dict) or not isinstance(output.get("valid"), bool)
                or not isinstance(output.get("errors", []), list)
                or any(not isinstance(item, str) for item in output.get("errors", []))):
            raise ValueError("run.json: некорректный результат проверки CSV.")

    @staticmethod
    def _usage(report):
        api = report.get("api", {})
        attempts = api.get("attempts")
        responses = api.get("responses")
        reported = api.get("usage_reported_responses")
        if attempts == 0:
            state = "none"
        elif not reported:
            state = "unknown"
        elif attempts is None or reported < attempts:
            state = "partial"
        else:
            state = "reported"
        return {
            "state": state, "attempts": attempts, "responses": responses,
            "reported_responses": reported,
            **{key: (0 if state == "none" else api.get(key) if reported else None)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }

    def list_runs(self) -> list[dict]:
        directory = self._safe(self.task_dir / ".agent_runs")
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise ValueError("Путь .agent_runs не является папкой.")
        recent = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if not RUN_ID.fullmatch(entry.name):
                    continue
                path = Path(entry.path)
                if _is_link(path) or not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    stamp = entry.stat(follow_symlinks=False).st_mtime_ns
                except OSError:
                    continue
                heapq.heappush(recent, (stamp, entry.name))
                if len(recent) > MAX_RUNS:
                    heapq.heappop(recent)
        rows = []
        for _, run_id in sorted(recent, reverse=True):
            try:
                report = self._report(self._run_dir(run_id))
                row = {"id": run_id, "error": report.get("error")}
                row.update({name: report.get(name) for name in (
                    "status", "started_at", "elapsed_seconds", "experiments",
                    "successful_experiments", "reason", "baseline")})
            except (OSError, ValueError) as error:
                row = {"id": run_id, "status": "invalid", "error": str(error),
                       "started_at": None, "elapsed_seconds": None, "experiments": None,
                       "successful_experiments": None, "reason": None, "baseline": None}
            rows.append(row)
        return rows

    def detail(self, run_id: str) -> dict:
        directory = self._run_dir(run_id)
        report = self._report(directory)
        files, warnings = [], []
        with os.scandir(directory) as entries:
            for entry in entries:
                match = EXPERIMENT_FILE.fullmatch(entry.name)
                if not match or not entry.is_file(follow_symlinks=False) or _is_link(Path(entry.path)):
                    continue
                if len(files) >= MAX_FILES:
                    warnings.append("Показаны первые 500 файлов запуска.")
                    break
                suffix = match.group(2)
                files.append({"name": entry.name, "experiment": int(match.group(1)),
                              "kind": "code" if suffix == "py" else suffix.split(".")[0]})
        files.sort(key=lambda item: (item["experiment"], item["kind"]))
        usage = self._usage(report)
        if usage["state"] == "unknown":
            warnings.append("API не сообщил расход токенов. Нулевой расход не подтверждён.")
        elif usage["state"] == "partial":
            warnings.append("Расход токенов учтён только для части обращений к API.")
        if report["status"] == "running":
            warnings.append("Статус взят из журнала. Приложение не проверяет, работает ли процесс сейчас.")
        return {"id": run_id, "report": report, "files": files, "usage": usage, "warnings": warnings}

    def file(self, run_id: str, name: str) -> dict:
        if not isinstance(name, str) or not EXPERIMENT_FILE.fullmatch(name):
            raise ValueError("Доступен только код экспериментов и их stdout/stderr.")
        path = self._safe(self._run_dir(run_id) / name, required=True)
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if not name.endswith(".py") and size > MAX_TEXT_BYTES:
                stream.seek(size - MAX_TEXT_BYTES)
            raw = stream.read(MAX_TEXT_BYTES)
        return {"name": name, "text": _redact(raw.decode("utf-8", errors="replace")),
                "truncated": size > MAX_TEXT_BYTES}

    def submission_path(self) -> Path:
        path = self._safe(self.task_dir / "submission.csv", required=True)
        if path.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("submission.csv превышает допустимый размер 512 МБ.")
        return path

    def _validation_inputs(self) -> list[Path]:
        """Check every location the existing validator may read, before calling it."""
        samples = []
        visited = 0
        for directory, folders, filenames in os.walk(self.task_dir, followlinks=False):
            visited += 1
            if visited > 2000:
                raise ValueError("Слишком много вложенных папок для проверки результата.")
            parent = Path(directory)
            filtered = []
            for name in folders:
                if name == ".agent_runs" or name.startswith(".offline-baseline-"):
                    continue
                candidate = parent / name
                self._safe(candidate, required=True, directory=True)
                filtered.append(name)
            folders[:] = filtered
            if "sample_submission.csv" in filenames:
                samples.append(self._safe(parent / "sample_submission.csv", required=True))
        if len(samples) != 1:
            raise ValueError("Для проверки нужен ровно один sample_submission.csv в папке задачи.")
        data_dir = samples[0].parent
        files = [samples[0], *(data_dir / name for name in (
            "train.csv", "test.csv", "data_dictionary.csv")), self.submission_path()]
        for path in files:
            self._safe(path, required=True)
            if path.stat().st_size > MAX_CSV_BYTES:
                raise ValueError(f"{path.name} превышает допустимый размер 512 МБ.")
        return files

    def _validation(self) -> dict:
        files = self._validation_inputs()
        fingerprint = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        key = str(self.root)
        with _RESULT_LOCK:
            cached = _RESULT_CACHE.get(key)
            if cached and cached[0] == fingerprint:
                _RESULT_CACHE.move_to_end(key)
                return deepcopy(cached[1])
            result = validate_predictions(self.task_dir)
            _RESULT_CACHE[key] = (fingerprint, deepcopy(result))
            _RESULT_CACHE.move_to_end(key)
            while len(_RESULT_CACHE) > 8:
                _RESULT_CACHE.popitem(last=False)
            return result

    def _baseline(self) -> dict | None:
        path = self._safe(self.task_dir / "baseline_metrics.json")
        if not path.exists():
            return None
        report = self._read_json(path)
        for name in ("metric", "split", "model", "timing_seconds"):
            if not isinstance(report.get(name), dict):
                raise ValueError(f"baseline_metrics.json: отсутствует объект {name}.")
        metric = report["metric"]
        if (not isinstance(metric.get("name"), str)
                or not _finite_number(metric.get("value"))
                or not isinstance(metric.get("higher_is_better"), bool)):
            raise ValueError("baseline_metrics.json: некорректная метрика.")
        if not isinstance(report["model"].get("name"), str):
            raise ValueError("baseline_metrics.json: не указано имя модели.")
        if "parameters" in report["model"] and not isinstance(report["model"]["parameters"], dict):
            raise ValueError("baseline_metrics.json: параметры модели должны быть объектом.")
        for name in ("train_fraction", "validation_fraction"):
            if name in report["split"] and (not _number(report["split"][name])
                                             or report["split"][name] > 1):
                raise ValueError(f"baseline_metrics.json: некорректная доля {name}.")
        for name in ("train_rows", "validation_rows"):
            if name in report["split"] and not _number(report["split"][name], integer=True):
                raise ValueError(f"baseline_metrics.json: некорректное число строк {name}.")
        for name in ("shuffle", "stratified"):
            if name in report["split"] and not isinstance(report["split"][name], bool):
                raise ValueError(f"baseline_metrics.json: поле {name} должно быть логическим значением.")
        for name, value in report["timing_seconds"].items():
            if not _number(value):
                raise ValueError(f"baseline_metrics.json: некорректное время {name}.")
        return report

    def result(self) -> dict:
        baseline, baseline_error = None, None
        try:
            baseline = self._baseline()
        except (OSError, ValueError) as error:
            baseline_error = str(error)
        path = self.task_dir / "submission.csv"
        submission = {"exists": path.exists() or path.is_symlink(), "valid": None,
                      "errors": [], "path": str(path), "size_bytes": None}
        if submission["exists"]:
            try:
                safe_path = self.submission_path()
                submission["size_bytes"] = safe_path.stat().st_size
                submission.update(self._validation())
            except (OSError, ValueError) as error:
                submission["errors"] = [str(error)]
        return {"baseline": baseline, "baseline_error": baseline_error,
                "submission": submission,
                "quality_note": "Метрика относится к отдельному запуску baseline на локальной "
                                "валидации. Проверка текущего CSV подтверждает только формат, "
                                "а не качество или улучшение модели."}
