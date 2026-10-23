"""Durable, serial plans over JobManager's existing local recipes.

No model runs here. JobManager owns each process, deadline and resource proof;
the campaign owns the frozen plan, admission and explicit pause/resume policy.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import threading
import time
from uuid import uuid4

from agent.automl import choose_validation
from app.search_summary import normalize_search_strategy
from agent.features import normalize_feature_config
from agent.task_spec import load_task, load_task_data
from app.jobs import LOCAL_TIMEOUT, _json, _linked, _safe, _valid_split, _write_json
from app.model_config import make_model_config, model_task_type, validate_model_config, validate_feature_config
from app.runs import _check_json_tree, _finite_number, _redact
from app.splits import normalize_validation, split_rows
from research_lab.integrity import source_identity


ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"[0-9a-f]{32}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
ACTIONS = ("automl", "baseline", "extra_trees", "catboost", "lightgbm", "linear")
RESERVED = {"queued", "running", "pausing", "paused", "stopping", "cleanup_pending"}
TERMINAL = {"completed", "stopped", "failed", "interrupted"}
STEP_STATES = {"pending", "launching", "running", "completed", "failed", "cancelled", "interrupted", "cleanup_pending"}
JOB_ACTIVE = {"queued", "running", "cleanup_pending"}
MAX_HISTORY = 200
POLL_SECONDS = .2
PLAN_KEYS = ("preset", "name", "task_ids", "max_hours", "wall_budget_seconds",
             "worker_upper_bound_seconds", "steps", "task_fingerprints", "input_copy_bytes", "continue_on_error")
STEP_KEYS = ("id", "task_id", "action", "minutes", "validation", "model_params", "label", "worker_timeout_seconds")
OPTIONAL_PLAN_KEYS = ("source_job_id", "experiment", "execution_policy")
OPTIONAL_STEP_KEYS = ("model_config", "feature_config", "is_control", "comparison_key", "split", "search_strategy")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _text(value, limit=2000):
    return _redact(str(value)[:limit]) if value is not None else None


def _plan_hash(value):
    plan = {key: value[key] for key in PLAN_KEYS}
    plan.update({key: value[key] for key in OPTIONAL_PLAN_KEYS if key in value})
    plan["steps"] = [{**{key: step[key] for key in STEP_KEYS},
                      **{key: step[key] for key in OPTIONAL_STEP_KEYS if key in step}}
                     for step in value["steps"]]
    return _hash({**plan, "source_seal": {key: value["source_seal"][key]
                                         for key in ("sha", "source_fingerprint")},
                  "runtime_fingerprint": value["runtime_seal"]["fingerprint"]})


def _plain(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(_linked(item) for item in (path, *path.parents)):
        raise ValueError("Рабочие пути не должны содержать ссылки или переходы.")
    return path.resolve()


class CampaignManager:
    def __init__(self, state_root, catalog, jobs, *, execution_gate, native_queues, source_root=ROOT):
        self.state_root, self.source_root = _plain(state_root), _plain(source_root)
        self.catalog, self.jobs = catalog, jobs
        self.execution_gate, self.native_queues = execution_gate, native_queues
        self.root = _safe(self.state_root / "campaigns", self.state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records, self._active = {}, None
        self._closed = False
        self._restore()

    @staticmethod
    def presets():
        return {"presets": [
            {"id": "smoke", "name": "Короткая проверка", "description": "AutoML по 5 минут на задачу.",
             "actions": ["automl"], "budgets_minutes": [5], "max_tasks": 2, "max_steps": 2},
            {"id": "models", "name": "Сравнение моделей", "description": "Пять фиксированных рецептов с лимитом 30 минут каждый.",
             "actions": list(ACTIONS[1:]), "budgets_minutes": [30], "max_tasks": 2, "max_steps": 10},
            {"id": "lightgbm_trees", "name": "LightGBM: больше деревьев",
             "description": "Сравнить 500, 750 и 1000 деревьев или увеличить число деревьев сохранённого LightGBM.",
             "actions": ["lightgbm"], "budgets_minutes": [30], "max_tasks": 1, "max_steps": 3},
            {"id": "day", "name": "Дневная серия", "description": "AutoML 30, 90 и 175 минут, задачи чередуются по бюджету.",
             "actions": ["automl"], "budgets_minutes": [30, 90, 175], "max_tasks": 2, "max_steps": 6},
            {"id": "custom", "name": "Свой план", "description": "До 12 последовательных локальных запусков.",
             "actions": list(ACTIONS), "budgets_minutes": [], "max_tasks": 2, "max_steps": 12}],
            "actions": [{"id": action, "name": action, "minutes_min": 1, "minutes_max": 175,
                         "timeout_seconds": None, "timeout_from_minutes": True} for action in ACTIONS],
            "limits": {"max_steps": 12, "max_tasks": 2, "max_hours": 12},
            "scope": "Локальные проверки на известных строках TRAIN; это не баллы платформы и не обучение на GPU."}

    def _folder(self, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Недопустимый идентификатор кампании.")
        return _safe(self.root / identifier, self.state_root)

    def _save(self, record):
        _write_json(self._folder(record["id"]) / "campaign.json", record, self.state_root)

    def _record(self, identifier):
        self._folder(identifier)
        if identifier not in self._records:
            raise ValueError("Кампания не найдена.")
        return self._records[identifier]

    def _capsule(self, record, step):
        capsule = self.jobs.campaign_job(record["id"], step["id"])
        if capsule is None:
            return None
        if not isinstance(capsule, dict) or not isinstance(capsule.get("job"), dict):
            raise ValueError("Неизвестный формат подтверждения запуска кампании.")
        job = capsule["job"]
        _check_json_tree(job)
        if (not isinstance(job.get("id"), str) or not ID.fullmatch(job["id"])
                or job.get("task_id") != step["task_id"] or job.get("action") != step["action"]
                or step.get("job_id") not in (None, job["id"])
                or not isinstance(job.get("status"), str)
                or job["status"] not in JOB_ACTIVE | {"completed", "failed", "cancelled", "interrupted"}):
            raise ValueError("Нарушена связь запуска с шагом кампании.")
        return deepcopy(capsule)

    def _attach(self, record, step, capsule):
        job = capsule["job"]
        step.update(job_id=job["id"], job=job)
        status = job.get("status")
        step["status"] = ("launching" if status == "queued" else status
                          if isinstance(status, str) and status in STEP_STATES else "interrupted")

    def _restore(self):
        folders = [path for path in self.root.iterdir()
                   if ID.fullmatch(path.name) and not _linked(path) and path.is_dir()]
        folders.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for folder in folders[:MAX_HISTORY]:
            try:
                record = _json(folder / "campaign.json", self.state_root, 2 * 1024**2)
                self._validate_stored(record, folder.name)
                self._records[record["id"]] = record
                if record["status"] in RESERVED:
                    cleanup_pending = False
                    for step in record["steps"]:
                        if step["status"] in {"launching", "running", "cleanup_pending"}:
                            capsule = self._capsule(record, step)
                            if capsule is not None:
                                self._attach(record, step, capsule)
                                pending = (capsule["job"].get("status") in JOB_ACTIVE
                                    or capsule.get("cleanup_confirmed") is not True)
                                cleanup_pending |= pending
                                if pending:
                                    step["status"] = "cleanup_pending"
                            else:
                                step["status"] = "interrupted"
                            step["admissible"] = False
                            step["error"] = "Шаг прерван перезапуском; повторное обучение автоматически не запускается."
                    record.update(status="cleanup_pending" if cleanup_pending else "interrupted",
                        error="Сервер перезапущен. Для оставшихся шагов требуется явное продолжение.")
                    record["source_seal"]["valid"] = None
                    record["runtime_seal"]["valid"] = None
                    self._save(record)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                if folder.name in self._records:
                    # A valid plan with ambiguous ownership remains reserved.
                    # Dropping it would lose the user's path to retry Stop.
                    record.update(status="cleanup_pending", finished_at=None,
                                  error="Связь или остановка запуска требует проверки: " + _text(error))
                    record["source_seal"]["valid"] = None
                    record["runtime_seal"]["valid"] = None
                    try:
                        self._save(record)
                    except (OSError, ValueError):
                        pass  # keep the in-memory reservation even if persistence is unavailable

    @staticmethod
    def _validate_stored(record, identifier):
        if (not isinstance(record, dict) or record.get("version") != 1 or record.get("id") != identifier
                or not isinstance(record.get("status"), str) or record["status"] not in RESERVED | TERMINAL
                or not isinstance(record.get("created_at"), str) or not isinstance(record.get("name"), str)
                or not isinstance(record.get("source_seal"), dict)
                or not isinstance(record.get("runtime_seal"), dict)
                or not isinstance(record.get("task_fingerprints"), dict)
                or not isinstance(record.get("plan_sha256"), str) or not HASH.fullmatch(record["plan_sha256"])
                or not _finite_number(record.get("wall_budget_seconds")) or not 0 < record["wall_budget_seconds"] <= 43200
                or not isinstance(record.get("steps"), list) or not 1 <= len(record["steps"]) <= 12):
            raise ValueError("Повреждён план кампании.")
        seal = record["source_seal"]
        runtime = record["runtime_seal"]
        if (not isinstance(seal.get("sha"), str) or not SHA.fullmatch(seal["sha"])
                or not isinstance(seal.get("source_fingerprint"), str) or not HASH.fullmatch(seal["source_fingerprint"])
                or not isinstance(record.get("task_ids"), list) or not 1 <= len(record["task_ids"]) <= 2
                or any(not isinstance(item, str) for item in record["task_ids"])
                or set(record["task_ids"]) != set(record["task_fingerprints"])
                or any(not isinstance(value, str) or not HASH.fullmatch(value) for value in record["task_fingerprints"].values())
                or not _finite_number(record.get("elapsed_seconds")) or record["elapsed_seconds"] < 0
                or type(record.get("continue_on_error")) is not bool):
            raise ValueError("Повреждены зафиксированные отпечатки кампании.")
        runtime_fields = {key: runtime[key] for key in ("python_version", "implementation", "pointer_bits", "packages")}
        if (not isinstance(runtime.get("fingerprint"), str) or not HASH.fullmatch(runtime["fingerprint"])
                or not isinstance(runtime.get("packages"), dict) or len(runtime["packages"]) != 15
                or _hash(runtime_fields) != runtime["fingerprint"]):
            raise ValueError("Повреждён отпечаток runtime кампании.")
        dates = {name: datetime.fromisoformat(record[name]) for name in ("created_at", "started_at", "deadline_at")}
        if any(value.tzinfo is None for value in dates.values()) or abs(
                (dates["deadline_at"] - dates["started_at"]).total_seconds() - record["wall_budget_seconds"]) > 1:
            raise ValueError("Повреждён исходный срок кампании.")
        policy = record.get("execution_policy")
        if policy not in (None, "per_step_minutes_v1"):
            raise ValueError("Неизвестное правило лимитов кампании.")
        if "source_job_id" in record and (not isinstance(record["source_job_id"], str)
                or not ID.fullmatch(record["source_job_id"])):
            raise ValueError("Повреждена ссылка на исходный рецепт.")
        for step in record["steps"]:
            if (not isinstance(step, dict) or not isinstance(step.get("id"), str) or not ID.fullmatch(step["id"])
                    or not isinstance(step.get("status"), str) or step["status"] not in STEP_STATES
                    or step.get("action") not in ACTIONS or type(step.get("minutes")) is not int
                    or not 1 <= step["minutes"] <= 175 or step.get("task_id") not in record["task_fingerprints"]):
                raise ValueError("Повреждён шаг кампании.")
            normalize_validation(step.get("validation"))
            if "search_strategy" in step:
                if normalize_search_strategy(step["search_strategy"], action=step["action"]) != step["search_strategy"]:
                    raise ValueError("Повреждена стратегия поиска в плане.")
            if "model_config" in step:
                checked = validate_model_config(step["model_config"], action=step["action"])
                if checked != step["model_config"] or checked["parameters"] != step.get("model_params"):
                    raise ValueError("Повреждён рецепт модели в плане.")
            if "feature_config" in step:
                validate_feature_config(step["feature_config"])
            if "is_control" in step and type(step["is_control"]) is not bool:
                raise ValueError("Повреждена отметка контроля.")
            if "split" in step and not _valid_split(step["split"]):
                raise ValueError("Повреждена зафиксированная проверка.")
            if "comparison_key" in step and (not isinstance(step["comparison_key"], str)
                    or not HASH.fullmatch(step["comparison_key"])):
                raise ValueError("Повреждён отпечаток проверки.")
            cap = (step["minutes"] * 60 + 30 if step["action"] == "automl" else
                   step["minutes"] * 60 if policy == "per_step_minutes_v1" else LOCAL_TIMEOUT)
            if (step.get("worker_timeout_seconds") != cap or not isinstance(step.get("label"), str)
                    or len(step["label"]) > 100 or step.get("job_id") is not None and (
                        not isinstance(step["job_id"], str) or not ID.fullmatch(step["job_id"]))):
                raise ValueError("Повреждены лимиты или связь шага кампании.")
        if (len({step["id"] for step in record["steps"]}) != len(record["steps"])
                or sum(step["worker_timeout_seconds"] for step in record["steps"]) != record["worker_upper_bound_seconds"]
                or _plan_hash(record) != record["plan_sha256"]):
            raise ValueError("Зафиксированный план кампании изменился.")

    def _identity(self):
        identity = source_identity(self.source_root)
        if (identity.get("dirty") or not isinstance(identity.get("sha"), str) or not SHA.fullmatch(identity["sha"])
                or not isinstance(identity.get("source_fingerprint"), str)
                or not HASH.fullmatch(identity["source_fingerprint"])):
            raise ValueError("Для кампании нужен чистый checkout с точным SHA и отпечатком исходников.")
        return {"sha": identity["sha"], "source_fingerprint": identity["source_fingerprint"],
                "valid": True, "checked_at": _now()}

    def _runtime_identity(self):
        path = _safe(self.source_root / "requirements.txt", self.source_root, required=True)
        with path.open("r", encoding="utf-8") as stream:
            text = stream.read(8193)
        if len(text) > 8192:
            raise ValueError("Список runtime-зависимостей слишком велик.")
        versions = {}
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]{1,80})==([A-Za-z0-9_.+!-]{1,80})", line.strip())
            if not match or match[1] in versions:
                raise ValueError("Нужен точный список из 15 закреплённых runtime-пакетов.")
            try:
                actual = metadata.version(match[1])
            except metadata.PackageNotFoundError as error:
                raise ValueError("Не установлен runtime-пакет: " + match[1]) from error
            if actual != match[2]:
                raise ValueError("Версия runtime-пакета не совпадает с requirements.txt: " + match[1])
            versions[match[1]] = actual
        if len(versions) != 15:
            raise ValueError("Нужны все 15 закреплённых runtime-пакетов.")
        result = {"python_version": ".".join(str(item) for item in sys.version_info[:3]),
                  "implementation": sys.implementation.name, "pointer_bits": struct.calcsize("P") * 8,
                  "packages": versions}
        return {**result, "fingerprint": _hash(result), "valid": True, "checked_at": _now()}

    def _check_seals(self, record):
        try:
            identity = self._identity()
            if any(identity[key] != record["source_seal"][key] for key in ("sha", "source_fingerprint")):
                raise ValueError("Исходники изменились после фиксации плана.")
        except (OSError, ValueError, RuntimeError) as error:
            record["source_seal"].update(valid=False, checked_at=_now())
            raise ValueError("Изменение или недоступность исходников: " + str(error)) from error
        record["source_seal"].update(valid=True, checked_at=_now())
        try:
            runtime = self._runtime_identity()
            if runtime["fingerprint"] != record["runtime_seal"]["fingerprint"]:
                raise ValueError("Интерпретатор или версии runtime-пакетов изменились.")
        except (OSError, ValueError, RuntimeError) as error:
            record["runtime_seal"].update(valid=False, checked_at=_now())
            raise ValueError("Изменение или недоступность runtime: " + str(error)) from error
        record["runtime_seal"].update(valid=True, checked_at=_now())
        for task_id, fingerprint in record["task_fingerprints"].items():
            if self.catalog.fingerprint(task_id) != fingerprint:
                raise ValueError("Входные данные задачи изменились: " + task_id)

    def _validation(self, task_id, requested):
        spec = load_task(self.catalog.resolve(task_id) / "task.md")
        train, test, _, id_column, target, y, features, numeric, _, classification = load_task_data(spec)
        choice, _ = choose_validation(train, test, features, numeric, classification, requested)
        # Verify feasibility using the same holdout helper as the fixed worker.
        _, held, split = split_rows(train, y, features, classification, choice)
        split["validation_hash"] = _hash({"id_column": id_column, "target_column": target,
            "ids": train[id_column].iloc[held].tolist(), "actuals": train[target].iloc[held].tolist()})
        return choice, spec.metric.task_type, split, spec.metric.name, spec.metric.higher_is_better

    def _source_recipe(self, task_id, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Выберите сохранённый локальный запуск LightGBM.")
        job = self.jobs.detail(task_id, identifier).get("job")
        if (not isinstance(job, dict) or job.get("id") != identifier or job.get("task_id") != task_id
                or job.get("status") != "completed" or job.get("source") != "local"
                or job.get("output_valid") is not True or job.get("action") not in {"lightgbm", "automl"}):
            raise ValueError("Нужен успешно завершённый локальный запуск LightGBM на выбранной задаче.")
        if job.get("dataset_fingerprint") != self.catalog.fingerprint(task_id):
            raise ValueError("Данные изменились после исходного запуска; его рецепт нельзя сравнивать с этой версией.")
        config = validate_model_config(job.get("model_config"), action="lightgbm")
        if job.get("model") != config["model"] or "component_feature_configs" in job:
            raise ValueError("Исходный результат должен быть одной моделью LightGBM, а не ансамблем.")
        split = job.get("split")
        if not isinstance(split, dict) or not _valid_split(split):
            raise ValueError("У исходного запуска отсутствует подтверждённая схема проверки.")
        requested = {"strategy": split.get("strategy", "random")}
        if requested["strategy"] == "time":
            requested["time_column"] = split.get("time_column")
        requested = normalize_validation(requested)
        if job["action"] == "automl" and "feature_config" not in job:
            raise ValueError("У победителя AutoML не сохранены настройки признаков; выберите другой запуск.")
        features = validate_feature_config(job["feature_config"]) if "feature_config" in job else normalize_feature_config()
        return config, features, requested, deepcopy(split), job.get("comparison_key")

    def _input_size(self, task_id):
        folder = self.catalog.resolve(task_id)
        _, paths = self.catalog._inputs(folder)
        return sum(_safe(path, self.catalog.data_root, required=True).stat().st_size
                   for path in [folder / "task.md", *paths.values()])

    def _disk_estimate(self, sizes, steps):
        destination = _plain(getattr(self.jobs, "jobs_root", self.state_root / "jobs"))
        while not destination.exists():
            destination = destination.parent
        copy_bytes = sum(sizes[step["task_id"]] for step in steps)
        reserve = 2 * 1024**3
        free = shutil.disk_usage(destination).free
        return {"task_copy_bytes": copy_bytes, "reserve_bytes": reserve,
                "required_free_bytes": copy_bytes + reserve, "free_bytes": free,
                "model_size_known": False, "ready": free >= copy_bytes + reserve}

    def preview(self, body):
        warnings = ["Подготовка и копирование данных не входят в сумму лимитов обучения.",
                    "Общий лимит включает паузы; загрузка компьютера на весь срок не гарантируется.",
                    "Поле minutes ограничивает весь worker: обучение, проверку, полное обучение и сохранение модели.",
                    "Диск: учтены копии пяти входных файлов и резерв 2 ГиБ; размер моделей заранее неизвестен."]
        result = {"ready": False, "errors": [], "warnings": warnings, "steps": [],
                  "preparation_included": False}
        try:
            _check_json_tree(body)
            allowed = {"preset", "task_ids", "name", "max_hours", "steps", "validation_by_task",
                       "continue_on_error", "preview_token", "source_job_id", "minutes_per_step"}
            if not isinstance(body, dict) or set(body) - allowed:
                raise ValueError("Неизвестные поля плана кампании.")
            preset = body.get("preset", "day")
            if not isinstance(preset, str) or preset not in {"day", "smoke", "models", "custom", "lightgbm_trees"}:
                raise ValueError("Выберите готовый или свой план кампании.")
            task_ids = body.get("task_ids")
            if (not isinstance(task_ids, list) or not 1 <= len(task_ids) <= 2
                    or any(not isinstance(item, str) for item in task_ids) or len(set(task_ids)) != len(task_ids)):
                raise ValueError("Выберите одну или две разные задачи.")
            name = body.get("name", "Дневная кампания")
            hours = body.get("max_hours", 12)
            if not isinstance(name, str) or len(name) > 100:
                raise ValueError("Название кампании: до 100 символов.")
            if not _finite_number(hours) or not 0 < hours <= 12:
                raise ValueError("Общий лимит должен быть больше нуля и не больше 12 часов.")
            if type(body.get("continue_on_error", True)) is not bool:
                raise ValueError("continue_on_error должен быть логическим значением.")
            mapping = body.get("validation_by_task", {})
            if not isinstance(mapping, dict) or set(mapping) - set(task_ids):
                raise ValueError("Настройки проверки должны относиться к выбранным задачам.")
            source = self._identity()
            runtime = self._runtime_identity()
            experiment = None
            inherited = None
            if preset != "lightgbm_trees" and ("source_job_id" in body or "minutes_per_step" in body):
                raise ValueError("Исходный LightGBM и общий бюджет шага доступны в плане lightgbm_trees.")
            if preset == "custom":
                raw_steps = body.get("steps")
                if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 12:
                    raise ValueError("Свой план должен содержать от 1 до 12 шагов.")
            elif preset == "lightgbm_trees":
                if len(task_ids) != 1 or "steps" in body:
                    raise ValueError("Для сравнения числа деревьев выберите одну задачу и готовый план.")
                minutes = body.get("minutes_per_step", 30)
                if type(minutes) is not int or not 1 <= minutes <= 175:
                    raise ValueError("Бюджет одного варианта: от 1 до 175 минут.")
                task_id = task_ids[0]
                if "source_job_id" in body:
                    inherited = self._source_recipe(task_id, body["source_job_id"])
                    base_config, features, choice, _, _ = inherited
                    override = mapping.get(task_id)
                    if override is not None and override != {"strategy": "auto"} and normalize_validation(override) != choice:
                        raise ValueError("Для повторения рецепта сохраните исходную схему проверки.")
                    description = "Повторное обучение рецепта выбранного запуска; прежняя метрика не заменяет новый контроль."
                else:
                    spec = load_task(self.catalog.resolve(task_id) / "task.md")
                    base_config = make_model_config("lightgbm", spec.metric.task_type, {"n_estimators": 500})
                    features, choice = normalize_feature_config(), mapping.get(task_id)
                    description = "Стандартный рецепт LightGBM с 500 деревьями; это не восстановление ночного победителя."
                count = base_config["parameters"]["n_estimators"]
                counts = sorted({count, min(2000, (count * 3 + 1) // 2), min(2000, count * 2)})
                if len(counts) < 2:
                    raise ValueError("В исходном рецепте уже 2000 деревьев. Для других настроек используйте свой план.")
                raw_steps = []
                for index, count in enumerate(counts):
                    config = deepcopy(base_config)
                    config["parameters"]["n_estimators"] = count
                    raw_steps.append({"task_id": task_id, "action": "lightgbm", "minutes": minutes,
                        "validation": choice, "model_config": config, "feature_config": features,
                        "is_control": index == 0,
                        "label": ("Контроль" if index == 0 else "Вариант") + f" · LightGBM · {count} деревьев"})
                experiment = {"kind": "lightgbm_tree_count", "parameter": "n_estimators",
                    "scope": "local_recipe_study", "counts": counts, "control_description": description}
                warnings.append("Сравниваются рецепты на известных TRAIN-строках, а не две полные версии конкурсного агента.")
            else:
                if "steps" in body:
                    raise ValueError("Редактируемые шаги передаются только с preset=custom.")
                recipes = [("automl", value) for value in ([5] if preset == "smoke" else [30, 90, 175])]
                if preset == "models":
                    recipes = [(action, 30) for action in ACTIONS[1:]]
                raw_steps = [{"task_id": task_id, "action": action, "minutes": minutes}
                             for action, minutes in recipes for task_id in task_ids]
            fingerprints, sizes, validations = {}, {}, {}
            for task_id in task_ids:
                fingerprints[task_id] = self.catalog.fingerprint(task_id)
                sizes[task_id] = self._input_size(task_id)
            steps = []
            for index, item in enumerate(raw_steps):
                if not isinstance(item, dict) or set(item) - {"task_id", "action", "minutes", "validation", "model_params", "model_config", "feature_config", "is_control", "label", "search_strategy"}:
                    raise ValueError("Шаг содержит неизвестные поля; команды и пути запрещены.")
                task_id, action, minutes = item.get("task_id"), item.get("action"), item.get("minutes")
                if not isinstance(task_id, str) or task_id not in task_ids or action not in ACTIONS:
                    raise ValueError("Шаг должен использовать выбранную задачу и локальный рецепт.")
                if type(minutes) is not int or not 1 <= minutes <= 175:
                    raise ValueError("Бюджет шага: целое число от 1 до 175 минут.")
                search_strategy = normalize_search_strategy(item.get("search_strategy"), action=action)
                label = item.get("label", "")
                if not isinstance(label, str) or len(label) > 100:
                    raise ValueError("Подпись шага: до 100 символов.")
                # Resolve only this step's actual request. An explicit split
                # must not first pass an unused automatic split, and it must
                # never become the default for another step that omitted one.
                requested = item.get("validation", mapping.get(task_id))
                requested = (None if requested is None or requested == {"strategy": "auto"}
                             else normalize_validation(requested))
                validation_key = (task_id, _hash(requested))
                if validation_key not in validations:
                    validations[validation_key] = self._validation(task_id, requested)
                validation, task_type, split, metric, higher_is_better = validations[validation_key]
                comparison_key = _hash({"dataset_fingerprint": fingerprints[task_id], "metric": metric,
                    "higher_is_better": higher_is_better, "split": split})
                if inherited is not None and (split != inherited[3] or comparison_key != inherited[4]):
                    raise ValueError("Точные проверочные строки исходного запуска не воспроизводятся; сравнение отменено.")
                parameters = item.get("model_params")
                model_config = item.get("model_config")
                if parameters is not None and model_config is not None:
                    raise ValueError("Передайте model_params или model_config, но не оба поля.")
                if action == "automl" and (parameters is not None or model_config is not None or "feature_config" in item):
                    raise ValueError("AutoML сам выбирает параметры моделей и признаков.")
                if action != "automl":
                    config = (validate_model_config(model_config, action=action) if model_config is not None
                              else make_model_config(action, task_type, parameters))
                    if model_task_type(config) != task_type:
                        raise ValueError("Модель не соответствует типу выбранной задачи.")
                    parameters = config["parameters"]
                step = {"task_id": task_id, "action": action, "minutes": minutes,
                        "validation": deepcopy(validation), "model_params": parameters, "label": label,
                        "split": deepcopy(split), "comparison_key": comparison_key,
                        "worker_timeout_seconds": minutes * 60 + 30 if action == "automl" else minutes * 60}
                if action != "automl":
                    step.update(model_config=deepcopy(config), feature_config=(validate_feature_config(item["feature_config"])
                        if "feature_config" in item else normalize_feature_config()))
                else:
                    step["search_strategy"] = search_strategy
                if "is_control" in item:
                    if type(item["is_control"]) is not bool:
                        raise ValueError("Отметка контроля должна быть логическим значением.")
                    step["is_control"] = item["is_control"]
                step["id"] = _hash({"index": index, **step})[:32]
                steps.append(step)
            total = sum(step["worker_timeout_seconds"] for step in steps)
            plan = {"preset": preset, "name": name.strip() or "Кампания", "task_ids": task_ids,
                    "max_hours": hours, "wall_budget_seconds": hours * 3600,
                    "worker_upper_bound_seconds": total, "steps": steps,
                    "task_fingerprints": fingerprints, "input_copy_bytes": sizes,
                    "continue_on_error": body.get("continue_on_error", True),
                    "execution_policy": "per_step_minutes_v1"}
            if experiment is not None:
                plan["experiment"] = experiment
            if inherited is not None:
                plan["source_job_id"] = body["source_job_id"]
                plan["experiment"]["source_job_id"] = body["source_job_id"]
            plan_hash = _plan_hash({**plan, "source_seal": source, "runtime_seal": runtime})
            result.update(plan, source_seal=source, runtime_seal=runtime, plan_sha256=plan_hash)
            result["disk_estimate"] = self._disk_estimate(sizes, steps)
            if not result["disk_estimate"]["ready"]:
                result["errors"].append("На диске папки обычных запусков не хватает места на копии данных и резерв 2 ГиБ.")
            if total > hours * 3600:
                result["errors"].append("Сумма верхних границ worker превышает общий лимит кампании.")
            if "preview_token" in body and body["preview_token"] != plan_hash:
                result["errors"].append("План или исходные данные изменились; повторите предварительную проверку.")
            with self.execution_gate:
                if self.jobs.active_job() or self.native_queues.active_queue():
                    result["errors"].append("Сначала завершите текущий обычный или ночной запуск.")
                if self.active_campaign():
                    result["errors"].append("Сначала завершите текущую кампанию, включая приостановленную.")
                if self._closed:
                    result["errors"].append("Менеджер кампаний остановлен.")
            result["ready"] = not result["errors"]
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            result["errors"].append(_text(error))
        return result

    def _launch_thread(self, record):
        active = {"id": record["id"], "stop": threading.Event(), "pause": threading.Event(),
                  "started": time.monotonic(), "base_elapsed": record.get("elapsed_seconds", 0)}
        remaining = self._remaining(record)
        active["deadline"] = time.monotonic() + remaining
        thread = threading.Thread(target=self._run, args=(record["id"], active), daemon=True)
        active["thread"] = thread
        self._active = active
        try:
            thread.start()
        except Exception as error:
            self._active = None
            record.update(status="interrupted", error=_text(error))
            self._save(record)
            raise ValueError("Не удалось запустить монитор кампании; план сохранён.") from error

    def start(self, body):
        if (not isinstance(body, dict) or not isinstance(body.get("preview_token"), str)
                or not HASH.fullmatch(body["preview_token"])):
            raise ValueError("Сначала проверьте план и передайте preview_token этой проверки.")
        with self.execution_gate, self._lock:
            plan = self.preview(body)
            if not plan["ready"]:
                raise ValueError(" ".join(plan["errors"]))
            identifier, now = uuid4().hex, _now()
            record = {key: deepcopy(value) for key, value in plan.items() if key not in {"ready", "errors"}}
            record.update(version=1, id=identifier, status="queued", created_at=now, started_at=now,
                finished_at=None, elapsed_seconds=0, error=None,
                deadline_at=datetime.fromtimestamp(datetime.now(timezone.utc).timestamp()
                    + plan["wall_budget_seconds"], timezone.utc).isoformat())
            for step in record["steps"]:
                step.update(status="pending", job_id=None, job=None, admissible=None, error=None)
            self._folder(identifier).mkdir()
            self._records[identifier] = record
            while len(self._records) > MAX_HISTORY:
                old = next((item for item in sorted(self._records.values(), key=lambda row: row["created_at"])
                            if item["status"] not in RESERVED), None)
                if old is None:
                    break
                self._records.pop(old["id"])
            self._save(record)
            self._launch_thread(record)
            return self.detail(identifier)

    def _remaining(self, record):
        wall = (datetime.fromisoformat(record["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()
        return max(0., min(wall, record["wall_budget_seconds"] - record.get("elapsed_seconds", 0)))

    def _refresh(self, record, step):
        capsule = self._capsule(record, step)
        if capsule is None:
            raise ValueError("Сохранённый запуск кампании не найден; автоматический повтор запрещён.")
        self._attach(record, step, capsule)
        return capsule

    def _stop_owned(self, record, step):
        capsule = self._capsule(record, step)
        if capsule is None:
            return None
        active = self.jobs.active_job()
        if active and active.get("id") == capsule["job"]["id"]:
            self.jobs.stop(step["task_id"], capsule["job"]["id"])
        return self._refresh(record, step)

    def _stop_record(self, record, target="stopped"):
        pending_cleanup = False
        for step in record["steps"]:
            if step["status"] in {"launching", "running", "cleanup_pending"}:
                capsule = self._stop_owned(record, step)
                if capsule is not None:
                    pending_cleanup |= capsule["job"].get("status") in JOB_ACTIVE or capsule.get("cleanup_confirmed") is not True
                step["admissible"] = False
                if capsule is None:
                    step["status"] = "interrupted"
            elif step["status"] == "pending":
                step["status"] = "cancelled"
        record.update(status="cleanup_pending" if pending_cleanup else target,
                      finished_at=None if pending_cleanup else _now())
        if pending_cleanup:
            record["error"] = "Остановка запуска кампании ещё не подтверждена; повторите остановку."
        self._save(record)

    def _run(self, identifier, active):
        record = self._records[identifier]
        current = None
        try:
            while True:
                with self.execution_gate, self._lock:
                    record["elapsed_seconds"] = active["base_elapsed"] + time.monotonic() - active["started"]
                    if active["stop"].is_set() or self._closed:
                        self._stop_record(record)
                        return
                    if time.monotonic() >= active["deadline"]:
                        raise TimeoutError("Исчерпан общий лимит кампании, включая подготовку, проверки и паузы.")
                    if active["pause"].is_set():
                        record.update(status="paused", error=None)
                        self._save(record)
                        return
                    current = next((step for step in record["steps"] if step["status"] == "pending"), None)
                    if current is None:
                        record.update(status="completed", finished_at=_now())
                        self._save(record)
                        return
                    if active["deadline"] - time.monotonic() < current["worker_timeout_seconds"]:
                        raise TimeoutError("В общем лимите недостаточно времени для полного лимита следующего шага.")
                    self._check_seals(record)
                    record["disk_estimate"] = self._disk_estimate(record["input_copy_bytes"],
                        [step for step in record["steps"] if step["status"] == "pending"])
                    if not record["disk_estimate"]["ready"]:
                        raise ValueError("Свободное место в папке запусков исчерпано; новый шаг не запущен.")
                    if active["deadline"] - time.monotonic() < current["worker_timeout_seconds"]:
                        raise TimeoutError("После подготовки и проверок не осталось полного лимита следующего шага.")
                    if self.jobs.active_job() or self.native_queues.active_queue():
                        raise ValueError("CPU занят другим запуском; кампания остановлена без повторов.")
                    record["status"], current["status"] = "running", "launching"
                    self._save(record)  # write intent before JobManager atomically persists ownership
                    capsule = self._capsule(record, current)
                    if capsule is None:
                        options = {}
                        if current.get("model_config") is not None:
                            options["model_config"] = current["model_config"]
                        else:
                            options["model_params"] = current["model_params"]
                        if current.get("feature_config") is not None:
                            options["feature_config"] = current["feature_config"]
                        if current["action"] != "automl":
                            options["local_timeout_seconds"] = current["worker_timeout_seconds"]
                        else:
                            options["search_strategy"] = current.get("search_strategy", "legacy")
                        self.jobs.start(current["task_id"], current["action"], author="Campaign",
                            hypothesis=current["label"], minutes=current["minutes"], validation=current["validation"],
                            **options, campaign_owner={"campaign_id": identifier, "step_id": current["id"]})
                        capsule = self._capsule(record, current)
                    if capsule is None:
                        raise ValueError("JobManager не сохранил связь с шагом; повторный запуск запрещён.")
                    self._attach(record, current, capsule)
                    self._save(record)
                while True:
                    with self._lock:
                        record["elapsed_seconds"] = active["base_elapsed"] + time.monotonic() - active["started"]
                        capsule = self._refresh(record, current)
                        if active["stop"].is_set() or self._closed or time.monotonic() >= active["deadline"]:
                            if time.monotonic() >= active["deadline"]:
                                record["error"] = "Исчерпан общий лимит кампании, включая подготовку и паузы."
                            self._stop_record(record, "failed" if record.get("error") else "stopped")
                            return
                        if capsule["job"].get("status") == "cleanup_pending":
                            record.update(status="cleanup_pending", error="Очистка процесса не подтверждена; серия остановлена.")
                            current.update(status="cleanup_pending", admissible=False)
                            self._save(record)
                            return
                        if capsule["job"].get("status") not in JOB_ACTIVE:
                            if capsule.get("cleanup_confirmed") is not True:
                                record.update(status="cleanup_pending", error="Очистка процесса не подтверждена; серия остановлена.")
                                current.update(status="cleanup_pending", admissible=False)
                                self._save(record)
                                return
                            if capsule.get("resources_ok") is not True:
                                raise ValueError("Наблюдение ресурсов отсутствует или ресурсный лимит нарушен; серия остановлена.")
                            self._check_seals(record)
                            if time.monotonic() >= active["deadline"]:
                                raise TimeoutError("Общий лимит истёк во время итоговых проверок; результат не допущен.")
                            current["admissible"] = self._admitted(record, current)
                            current["error"] = _text(capsule["job"].get("error"))
                            self._save(record)
                            if not current["admissible"] and not record["continue_on_error"]:
                                raise ValueError("Шаг завершился без допустимого результата; продолжение отключено.")
                            break
                    active["stop"].wait(POLL_SECONDS)
        except Exception as error:
            with self._lock:
                record.update(status="failed", finished_at=_now(), error=_text(error))
                try:
                    if current is not None:
                        current["admissible"] = False
                        current["error"] = _text(error)
                        if current["status"] == "launching" and current.get("job_id") is None:
                            # start may have committed its owner before raising.
                            capsule = self._capsule(record, current)
                            if capsule:
                                self._attach(record, current, capsule)
                        if current["status"] == "pending":
                            current["status"] = "interrupted"
                        if current["status"] in {"launching", "running", "cleanup_pending"}:
                            self._stop_record(record, "failed")
                except (OSError, ValueError, TypeError, RuntimeError) as cleanup_error:
                    record.update(status="cleanup_pending", finished_at=None,
                        error=_text(error) + " Остановка или связь запуска требует проверки: " + _text(cleanup_error))
                    if current is not None:
                        current["status"] = "cleanup_pending"
                self._save(record)
        finally:
            with self._lock:
                record["elapsed_seconds"] = active["base_elapsed"] + time.monotonic() - active["started"]
                try:
                    self._save(record)
                finally:
                    if self._active is active:
                        self._active = None

    @staticmethod
    def _admitted(record, step):
        job = step.get("job") or {}
        metric, split = job.get("metric"), job.get("split")
        if (job.get("status") != "completed" or job.get("source") != "local" or job.get("output_valid") is not True
                or job.get("dataset_fingerprint") != record["task_fingerprints"].get(step["task_id"])
                or not isinstance(metric, dict) or not isinstance(metric.get("name"), str)
                or not _finite_number(metric.get("value")) or type(metric.get("higher_is_better")) is not bool
                or not isinstance(split, dict) or not _valid_split(split)):
            return False
        key = _hash({"dataset_fingerprint": job["dataset_fingerprint"], "metric": metric["name"],
                     "higher_is_better": metric["higher_is_better"], "split": split})
        if ("comparison_key" in step and step["comparison_key"] != key
                or "split" in step and step["split"] != split
                or "model_config" in step and step["model_config"] != job.get("model_config")
                or "feature_config" in step and step["feature_config"] != job.get("feature_config")):
            return False
        return (record["source_seal"].get("valid") is True and record["runtime_seal"].get("valid") is True
                and job.get("comparison_key") == key)

    def detail(self, identifier):
        with self._lock:
            record = deepcopy(self._record(identifier))
            active = self._active
            if active and active["id"] == identifier:
                record["elapsed_seconds"] = active["base_elapsed"] + time.monotonic() - active["started"]
            # Job summaries only: detail() would load diagnostics/logs on every poll.
            for step in record["steps"]:
                if step.get("job_id") or step["status"] == "launching" or step.get("admissible") is True:
                    try:
                        capsule = self._capsule(record, step)
                        if capsule:
                            self._attach(record, step, capsule)
                            if capsule.get("cleanup_confirmed") is not True or capsule.get("resources_ok") is not True:
                                step["admissible"] = False
                        else:
                            step["admissible"] = False
                            step["error"] = "Подтверждение сохранённого запуска недоступно."
                    except (OSError, ValueError, RuntimeError):
                        step["admissible"] = False
            groups = {}
            for step in record["steps"]:
                if step.get("admissible") is not True or not self._admitted(record, step):
                    continue
                job, metric = step["job"], step["job"]["metric"]
                key = (step["task_id"], job["comparison_key"])
                group = groups.setdefault(key, {"task_id": step["task_id"], "dataset_fingerprint": job["dataset_fingerprint"],
                    "comparison_key": job["comparison_key"], "metric_name": metric["name"],
                    "higher_is_better": metric["higher_is_better"], "split": job["split"], "jobs": []})
                group["jobs"].append({"job_id": job["id"], "step_id": step["id"], "value": metric["value"],
                    "action": step["action"], "minutes": step["minutes"], "label": step["label"],
                    "elapsed_seconds": job.get("elapsed_seconds"), "is_control": step.get("is_control") is True})
            for group in groups.values():
                members = group["jobs"]
                best = sorted(members, key=lambda row: row["value"], reverse=group["higher_is_better"])[0]
                group["best_job_id"] = best["job_id"]
                declared = [step for step in record["steps"] if step["task_id"] == group["task_id"]
                            and step.get("is_control") is True
                            and step.get("comparison_key") == group["comparison_key"]]
                controls = [row for row in members if row["is_control"]]
                control = controls[0] if len(controls) == 1 else members[0] if not declared else None
                reason = "multiple_controls" if len(declared) > 1 else "control_not_admitted" if control is None else None
                if reason:
                    control = None
                group.update(control_job_id=control["job_id"] if control else None,
                    control_value=control["value"] if control else None,
                    comparison_available=control is not None and len(members) > 1,
                    comparison_reason=reason or ("one_result" if len(members) < 2 else None))
                direction = 1 if group["higher_is_better"] else -1
                gain = direction * (best["value"] - control["value"]) if control else None
                group["gain"] = gain if _finite_number(gain) else None
                percent = 100 * gain / abs(control["value"]) if control and control["value"] and _finite_number(gain) else None
                group["gain_percent"] = percent if _finite_number(percent) else None
                for row in members:
                    difference = row["value"] - control["value"] if control else None
                    row["difference"] = difference if _finite_number(difference) else None
                    row["improvement"] = direction * difference if _finite_number(difference) else None
            record["comparisons"] = list(groups.values())
            return _redact({key: value for key, value in record.items() if key != "version" and not key.startswith("_")})

    def list(self):
        with self._lock:
            identifiers = [row["id"] for row in sorted(self._records.values(), key=lambda row: row["created_at"], reverse=True)[:MAX_HISTORY]]
        return [self.detail(identifier) for identifier in identifiers]

    def active_campaign(self):
        with self._lock:
            identifier = next((row["id"] for row in self._records.values() if row["status"] in RESERVED), None)
        return self.detail(identifier) if identifier else None

    def pause(self, identifier):
        with self._lock:
            record = self._record(identifier)
            if record["status"] == "paused":
                return self.detail(identifier)
            if not self._active or self._active["id"] != identifier or record["status"] not in {"queued", "running", "pausing"}:
                raise ValueError("Эту кампанию нельзя приостановить.")
            self._active["pause"].set()
            record["status"] = "pausing"
            self._save(record)
            return self.detail(identifier)

    def resume(self, identifier):
        with self.execution_gate, self._lock:
            record = self._record(identifier)
            if record["status"] not in {"paused", "interrupted"}:
                raise ValueError("Продолжить можно приостановленный или прерванный план.")
            if self._closed or self._active or self.jobs.active_job() or self.native_queues.active_queue():
                raise ValueError("Для продолжения освободите CPU и запустите работающий менеджер.")
            if any(row["id"] != identifier and row["status"] in RESERVED for row in self._records.values()):
                raise ValueError("Другая кампания уже занимает очередь.")
            self._check_seals(record)
            if self._remaining(record) <= 0:
                raise ValueError("Общий срок кампании уже истёк; старые лимиты не продлеваются.")
            if not any(step["status"] == "pending" for step in record["steps"]):
                raise ValueError("В плане не осталось незапущенных шагов.")
            record.update(status="queued", error=None, finished_at=None)
            self._save(record)
            self._launch_thread(record)
            return self.detail(identifier)

    def stop(self, identifier):
        with self._lock:
            record = self._record(identifier)
            active = self._active if self._active and self._active["id"] == identifier else None
            if active:
                active["stop"].set()
                record["status"] = "stopping"
                self._save(record)
            elif record["status"] in RESERVED | {"interrupted"}:
                self._stop_record(record)
        if active:
            active["thread"].join(timeout=5)
        return self.detail(identifier)

    def close(self):
        with self._lock:
            self._closed = True
            active = self._active
            identifier = active["id"] if active else next((row["id"] for row in self._records.values()
                if row["status"] in RESERVED), None)
        if identifier:
            self.stop(identifier)
        if active:
            active["thread"].join(timeout=10)
