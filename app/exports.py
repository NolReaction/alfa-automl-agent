"""Build fixed, bounded download packages without writing project outputs."""

from __future__ import annotations

import hashlib
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from zipfile import BadZipFile, LargeZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

import build_submission


MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024 * 1024
MAX_SOURCE_FILES = 512
MAX_IMPORT_BYTES = 20 * 1024 * 1024
MAX_IMPORT_REPORT_BYTES = 1024 * 1024
MAX_IMPORT_ENTRIES = 32


def report_bytes_from_archive(content: bytes) -> bytes:
    """Read one bounded reports.json; never extract or interpret other entries."""
    if len(content) > MAX_IMPORT_BYTES:
        raise ValueError("ZIP отчёта превышает 20 МБ.")
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_IMPORT_ENTRIES:
                raise ValueError("В ZIP слишком много файлов. Выберите пакет результата Alfa Lab.")
            reports = [entry for entry in entries if entry.filename == "reports.json"]
            if len(reports) != 1:
                raise ValueError("В ZIP нужен ровно один файл reports.json в корне архива.")
            report, = reports
            if report.orig_filename != "reports.json":
                raise ValueError("В ZIP недопустимое имя файла отчёта.")
            if report.flag_bits & 1 or report.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                raise ValueError("ZIP отчёта зашифрован или использует неподдерживаемое сжатие.")
            if report.file_size > MAX_IMPORT_REPORT_BYTES:
                raise ValueError("Файл reports.json в ZIP превышает 1 МБ.")
            with archive.open(report) as stream:
                raw = stream.read(MAX_IMPORT_REPORT_BYTES + 1)
            if len(raw) > MAX_IMPORT_REPORT_BYTES:
                raise ValueError("Файл reports.json в ZIP превышает 1 МБ.")
            return raw
    except (BadZipFile, LargeZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as error:
        raise ValueError("Не удалось прочитать ZIP. Выберите пакет результата Alfa Lab.") from error


def _regular_bytes(path: Path, limit: int) -> bytes:
    """Reject links, non-files and changing/oversized input before archiving."""
    path = path.absolute()
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("Ссылки и junction-папки в выгружаемых файлах запрещены.")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Для выгрузки нужен обычный файл.")
        if before.st_size > limit:
            raise ValueError("Размер файлов превышает лимит выгрузки.")
        content = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if len(content) > limit:
            raise ValueError("Размер файлов превышает лимит выгрузки.")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Файл изменился во время выгрузки. Повторите скачивание.")
        return content


def _archive(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def delivery_archive() -> bytes:
    """Use the competition builder's flat manifest with bounded source reads."""
    remaining, count = MAX_SOURCE_BYTES, 0

    def read_bytes(path: Path) -> bytes:
        nonlocal remaining, count
        count += 1
        if count > MAX_SOURCE_FILES:
            raise ValueError("Слишком много исходных файлов для сборки через панель.")
        content = _regular_bytes(path, remaining)
        remaining -= len(content)
        return content

    files = build_submission.archive_files(read_bytes=read_bytes)
    if len(files) > MAX_SOURCE_FILES:
        raise ValueError("Слишком много исходных файлов для сборки через панель.")
    if sum(len(content) for content in files.values()) > MAX_SOURCE_BYTES:
        raise ValueError("Размер файлов превышает лимит выгрузки.")
    package = _archive(files)
    if len(package) > build_submission.MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP для конкурса превышает 100 МБ.")
    return package


def _readable_report(report: dict) -> bytes:
    def text(value):
        return escape(str(value if value is not None else "—"), quote=True)

    action = {"baseline": "Стартовая модель", "extra_trees": "ExtraTrees", "catboost": "CatBoost", "lightgbm": "LightGBM", "linear": "Линейная модель",
              "automl": "Автоподбор моделей", "agent": "Исследование с Qwen"}.get(report["action"], "Эксперимент")
    fallback = report.get("agent_status") == "fallback"
    status = "Резервный CSV: исследование не завершилось штатно" if fallback else "Запуск завершён"
    metric = report.get("metric")
    if metric:
        metric_html = (f'<strong>{text(format(metric["value"], ".6g"))}</strong>'
                       f'<span>{text(metric["name"])}</span>'
                       f'<p>{"Больше — лучше" if metric["higher_is_better"] else "Меньше — лучше"}. '
                       'Локальная проверка; это не балл платформы.</p>')
    else:
        metric_html = ('<strong>Качество не измерено</strong><p>Структурированная метрика отсутствует. '
                       'Корректный формат CSV сам по себе не подтверждает качество предсказаний.</p>')
    split = report.get("split") or {}
    fraction = split.get("validation_fraction")
    validation = (f'{fraction * 100:g}% данных' if fraction is not None else "Не указана")
    random_scheme = (f'Случайная {(1 - fraction) * 100:g}/{fraction * 100:g}'
                     if fraction is not None else "Случайная")
    scheme = {"random": random_scheme, "time": "По последнему периоду",
              "grouped": "По группам одинаковых признаков"}.get(split.get("strategy"),
                  "Прежняя проверка 80/20" if split and report["action"] in {"baseline", "extra_trees"}
                  else "Схема не указана")
    evidence = []
    if split.get("strategy") == "time":
        evidence.append(("Периоды обучения → проверки",
                         f'{split.get("time_column", "—")}: {split.get("train_period_min", "—")}–'
                         f'{split.get("train_period_max", "—")} → {split.get("validation_period", "—")}'))
    elif split.get("strategy") == "grouped":
        evidence.extend([("Групп в обучении / проверке",
                          f'{split.get("train_groups", "—")} / {split.get("validation_groups", "—")}'),
                         ("Пересечение групп", split.get("group_overlap"))])
        if "fold" in split and "n_splits" in split:
            evidence.append(("Разбиение групп", f'{split["fold"] + 1} из {split["n_splits"]}'))
    configuration = []
    model_config = report.get("model_config")
    if model_config is not None:
        from app.model_config import model_options

        def quality_options(config):
            options = []
            if "objective" in config:
                value = "MAE" if config["objective"] == "mae" else config["objective"]
                options.append(f"Целевая функция: {value}")
            if "target_transform" in config:
                value = ("log1p (обратное преобразование expm1)"
                         if config["target_transform"] == "log1p" else config["target_transform"])
                options.append(f"Преобразование цели: {value}")
            return options

        def component_name(component):
            options = quality_options(component)
            return component["model"] + (" (" + "; ".join(options) + ")" if options else "")

        definition = next((option for task_type in ("classification", "regression")
                           for option in model_options(task_type).values()
                           if option["model"] == model_config["model"]), {})
        fields = {field["name"]: field for field in definition.get("fields", [])}
        parameters = []
        for name, value in model_config["parameters"].items():
            field = fields.get(name, {})
            if model_config["model"] == "FixedBlend" and name == "components":
                parameters.append("Модели: " + " + ".join(component_name(item) for item in value))
                continue
            if model_config["model"] == "FixedBlend" and name == "weight":
                parameters.append("Вес каждой модели: 50 %")
                continue
            selected = next((option for option in field.get("options", [])
                             if option["value"] == value), None)
            if selected is not None:
                value = selected["label"]
            elif value is None:
                value = "Без ограничения"
            elif isinstance(value, bool):
                value = "Да" if value else "Нет"
            elif isinstance(value, float):
                value = f"{value:g}"
            parameters.append(f'{field.get("label", name)}: {value}')
        parameters.extend(quality_options(model_config))
        configuration.append(("Параметры модели", "; ".join(parameters)))
    rows = [
        ("Модель", report.get("model") or (model_config or {}).get("model") or action), *configuration,
        ("Автор", report.get("author") or "Не указан"),
        ("Заметка / гипотеза", report.get("hypothesis") or "Не указана"),
        ("Начало", report.get("started_at")), ("Окончание", report.get("finished_at")),
        ("Время", f'{report["elapsed_seconds"]:g} с' if report.get("elapsed_seconds") is not None else None),
        ("Схема проверки", scheme), *evidence,
        ("Проверочная выборка", validation), ("Проверочных строк", split.get("validation_rows")),
        ("Seed разбиения", split.get("seed")), ("API использовался", "Да" if report.get("api_used") else "Нет"),
        ("Идентификатор запуска", report["id"]), ("SHA-256 данных", report["dataset_fingerprint"]),
    ]
    details = "".join(f"<dt>{text(label)}</dt><dd>{text(value)}</dd>" for label, value in rows)
    search_html = ""
    search = report.get("search")
    if search:
        states = {"completed": "Завершена", "failed": "Ошибка", "timed_out": "Лимит времени",
                  "skipped": "Пропущена", "running": "Выполняется", "pending": "Ожидает"}
        candidate_rows = []
        for candidate in search["candidates"]:
            value = candidate.get("metric")
            score = format(value, ".6g") if isinstance(value, (int, float)) else "—"
            selection = candidate.get("selection_metric")
            selection_score = format(selection, ".6g") if isinstance(selection, (int, float)) else "—"
            resources = candidate.get("resources", {})
            memory = resources.get("peak_rss_bytes")
            memory_text = f"{memory / (1024 * 1024):.1f} МБ" if isinstance(memory, (int, float)) else "—"
            state = "Победитель" if candidate.get("selected") else states.get(candidate["status"], candidate["status"])
            reason = {"timeout": "Лимит времени", "memory_limit": "Лимит памяти",
                      "execution_failed": "Ошибка обучения", "invalid_evidence": "Проверка не пройдена",
                      "not_selected": "Не прошёл отбор", "not_improved": "Без улучшения",
                      "insufficient_budget": "Не хватило времени", "refit_failed": "Ошибка финального обучения",
                      "optional_dependency_unavailable": "Библиотека не установлена"}.get(candidate.get("rejection_reason"))
            explanation = "".join(f"<small>{text(note)}</small>" for note in (reason, candidate.get("error")) if note)
            candidate_rows.append(
                f'<tr><td>{text(candidate["label"])}<small>{text(candidate["model"])}</small></td>'
                f'<td>{text(selection_score)}</td><td>{text(score)}</td><td>{text(state)}{explanation}</td>'
                f'<td>{text(candidate["elapsed_seconds"])} с</td><td>{text(memory_text)}</td></tr>')
        ending = ("Лимит времени: сохранён лучший завершённый результат."
                  if search["status"] == "time_limit" else "Сохранён лучший завершённый результат.")
        search_html = (f'<section><h2>Попытки автоподбора</h2><p>{text(ending)}</p>'
                       f'<p>Подбор измеряется внутри обучающей части; проверка — на отложенных строках.</p><p>{text(search["validation_reason"])}</p><div class="table-scroll"><table>'
                       '<thead><tr><th>Модель</th><th>Подбор</th><th>Проверка</th><th>Результат</th><th>Время</th><th>Пик RAM</th></tr></thead>'
                       f'<tbody>{"".join(candidate_rows)}</tbody></table></div></section>')
    diagnostics_html = ""
    diagnostics = report.get("classification_diagnostics")
    if diagnostics:
        class_rows = []
        for label, row in zip(diagnostics["labels"], diagnostics["per_class"]):
            class_rows.append(
                f'<tr><th scope="row">{text(label)}</th><td>{text(row["support"])}</td>'
                f'<td>{text(row["predicted"])}</td>'
                f'<td>{text(format(row["precision"], ".3f"))}</td>'
                f'<td>{text(format(row["recall"], ".3f"))}</td>'
                f'<td>{text(format(row["f1"], ".3f"))}</td></tr>')
        confused = sorted(
            ((count, diagnostics["labels"][actual], diagnostics["labels"][predicted])
             for actual, row in enumerate(diagnostics["confusion_matrix"])
             for predicted, count in enumerate(row) if actual != predicted and count),
            key=lambda item: (-item[0], item[1], item[2]),
        )[:5]
        confusions = ("<ul>" + "".join(
            f'<li>{text(actual)} → {text(predicted)}: {text(count)}</li>'
            for count, actual, predicted in confused) + "</ul>"
            if confused else "<p>Ошибок на проверочных строках нет.</p>")
        provenance = ("Отчёт участника; предсказания не проверены на этом компьютере."
                      if report.get("source") == "imported" else
                      "Локальная проверочная выборка; это не оценка скрытого теста.")
        diagnostics_html = (
            f'<section><h2>Качество по классам</h2><p>{text(provenance)}</p>'
            f'<p>Проверочных строк: {text(diagnostics["rows"])} · '
            f'Accuracy: {text(format(diagnostics["accuracy"], ".3f"))}</p>'
            '<div class="table-scroll"><table><thead><tr><th>Класс</th><th>Строк</th><th>Предсказано</th>'
            '<th>Precision</th><th>Recall</th><th>F1</th></tr></thead>'
            f'<tbody>{"".join(class_rows)}</tbody></table></div>'
            f'<h3>Частые ошибки: настоящий класс → предсказанный</h3>{confusions}</section>')
    return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Alfa Lab — результат эксперимента</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;color:#20232b;background:#f7f8fa;margin:0;padding:32px}}
main{{max-width:850px;margin:auto;background:white;padding:32px;border-radius:12px}}h1{{font-size:26px;margin:0}}
.status{{color:{"#986d11" if fallback else "#287b60"}}}.metric{{border-block:1px solid #e7e9ee;padding:24px 0;margin:24px 0}}
.metric strong{{display:block;font-size:32px}}.metric span{{color:#667085}}p{{margin:8px 0}}
dl{{display:grid;grid-template-columns:190px minmax(0,1fr);gap:12px 20px}}dt{{color:#667085}}dd{{margin:0;overflow-wrap:anywhere;white-space:pre-wrap}}
footer{{border-top:1px solid #e7e9ee;padding-top:20px;margin-top:24px;color:#667085}}
.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e7e9ee;vertical-align:top}}td small{{display:block;color:#667085;overflow-wrap:anywhere}}
@media(max-width:600px){{body{{padding:12px}}main{{padding:20px}}dl{{grid-template-columns:1fr;gap:4px}}dd{{margin-bottom:12px}}}}</style>
</head><body><main><h1>{text(action)}</h1><p class="status">{text(status)}</p>
<section class="metric">{metric_html}</section><dl>{details}</dl>{search_html}{diagnostics_html}
<footer><p><b>В этом пакете:</b> submission.csv — предсказания этого запуска;
reports.json — отчёт для кнопки «Добавить отчёт» в Alfa Lab;
report.html — эта сводка. В Alfa Lab можно добавить весь ZIP целиком.</p><p>CSV прошёл проверку формата. Данные обучения, модель, ключи,
код эксперимента и журналы в пакет не входят. Для конкурса архив агента solution.zip собирается отдельно.</p></footer>
</main></body></html>'''.encode("utf-8")


def result_archive(task_id: str, job_id: str, *, jobs, catalog, reports) -> bytes:
    """Package one owned local result, tying its CSV to its exact report."""
    artifact = jobs.artifact(task_id, job_id)
    job = next((item for item in jobs.list(task_id) if item["id"] == job_id), None)
    if job is None:
        raise ValueError("Запуск не найден для этой задачи.")
    fingerprint = catalog.fingerprint(task_id)
    if job["dataset_fingerprint"] != fingerprint:
        raise ValueError("Данные задачи изменились. Скачайте результат для текущей версии данных.")
    csv_bytes = _regular_bytes(artifact, MAX_RESULT_BYTES)
    # Detect edits between the existing artifact validation and the bounded read.
    checked = jobs.artifact(task_id, job_id)
    if hashlib.sha256(csv_bytes).hexdigest() != jobs._file_hash(checked):
        raise ValueError("CSV изменился во время выгрузки. Повторите скачивание.")
    if catalog.fingerprint(task_id) != fingerprint:
        raise ValueError("Данные задачи изменились во время выгрузки. Повторите скачивание.")
    envelope = reports.export(task_id, fingerprint, [job])
    report, = envelope["reports"]
    return _archive({
        "submission.csv": csv_bytes,
        "reports.json": json.dumps(envelope, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"),
        "report.html": _readable_report(report),
    })
