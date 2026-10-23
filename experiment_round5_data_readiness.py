"""Round 5 data readiness: full TRAIN/TEST feature shift and a fit-only OOF split audit.

Offline developer tool with two separate modes and scopes. It trains no model,
starts no training subprocess and makes no API request.

shift  One row per declared feature: the full TRAIN against TEST. Values come
       from existing calculations only: app.profiling.DataCatalog, its median
       shift applied to every numeric feature instead of the 20 kept in the
       profile, and analyze_data.profile_features (KS and other distributions,
       without the leakage and correlation analysis).
fit    audit_oof_fit(X_fit, y_fit, context) checks one explicitly passed fit,
       its labels and declared context against the measured round 4 OOF
       candidate: canonical grouped split, raw-feature groups, class support and
       the candidate's own size and support gates. It reads no files and no TEST.

Neither report is an INNER context for Qwen. A compatible full-TRAIN fit does
not prove that a future host INNER fit is compatible.

    python experiment_round5_data_readiness.py shift --data-dir data --task classification --output-dir ../r5/shift-classification
    python experiment_round5_data_readiness.py fit --task-md data/classification/task.md
        --dictionary data/classification/multiclass/participants/data_dictionary.csv
        --fit-csv data/classification/multiclass/participants/train.csv --output-dir ../r5/fit-classification
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import analyze_data
from agent.metrics import get_metric
from agent.oof_classification import OOFContractError, fit_oof_guarded_rule
from agent.round4_classification_candidate import checked_recipe, coverage, json_hash, make_recipe
from agent.round4_classification_probability import ProbabilitySkipError, fixed_incumbent_recipe
# Private, side-effect-free checks of the measured runtime; the integration
# request asks for public equivalents instead of copies kept here.
from agent.round4_classification_probability import _inputs as probability_inputs
from agent.splits import split_rows
from agent.task_contract import parse_task_metric
from agent.task_spec import _class_list
from app.profiling import DataCatalog, _numeric_shift
import experiment_round4_feature_passport as passport
from offline_baseline import read_csv


SCHEMA_VERSION = "round5-data-readiness/1"
SHIFT_SCOPE = "developer_full_train_vs_test_features"
FIT_SCOPE = "explicit_fit_only_oof_structure"
SHIFT_FILES = ("shift_report.json", "shift_features.csv", "shift_summary.md")
FIT_FILES = ("oof_fit_report.json", "oof_class_support.csv", "oof_fit_summary.md")
NUMERIC_TYPES = {"numeric", "binary"}
CODE_VALUES = 20          # analyze_data.profile_numeric reports unseen TEST codes only up to this many TRAIN values
MISSING_CHANGE_PP = 1.0   # the change from which DataCatalog quality.missing_changes reports a feature
README_TOP = 10
FIT_DECISIONS = ("structurally_compatible", "ineligible", "unknown")

SHIFT_FIELDS = {
    "name": "DataCatalog profile()['columns'][].name",
    "dtype": "DataCatalog profile()['columns'][].dtype",
    "train_missing_percent_empty": "DataCatalog profile()['columns'][].missing_percent",
    "test_missing_percent_empty": "DataCatalog profile()['quality']['missing_changes'][].test_percent",
    "train_missing_share_blank": "analyze_data.profile_features: missing_train",
    "test_missing_share_blank": "analyze_data.profile_features: missing_test",
    "train_placeholder_share": "analyze_data.profile_features: placeholder_train",
    "train_nonfinite_count": "DataCatalog column()['distribution']['invalid_count']",
    "train_unparseable_share": "analyze_data.profile_features: unparseable",
    "train_infinite_share": "analyze_data.profile_features: infinite",
    "test_nonfinite": "нет источника",
    "train_unique_nonempty": "DataCatalog profile()['columns'][].unique_count",
    "train_unique_analyze": "analyze_data.profile_features: n_unique",
    "is_constant": "DataCatalog profile()['quality']['constant_features']",
    "train_std_ddof0": "DataCatalog column()['stats']['std']",
    "train_std_ddof1": "analyze_data.profile_features: std",
    "train_median": "app.profiling._numeric_shift: train_median",
    "test_median": "app.profiling._numeric_shift: test_median",
    "median_shift_score": "app.profiling._numeric_shift: score",
    "median_shift_in_profile": "DataCatalog profile()['quality']['numeric_shift'][].name",
    "ks_train_test": "analyze_data.profile_features: ks_train_test",
    "test_out_of_range_share": "analyze_data.profile_features: test_out_of_range",
    "new_category_count": "DataCatalog profile()['quality']['new_categories'][].count",
    "new_category_row_share": "analyze_data.profile_features: unseen_test_share (category)",
    "numeric_code_unseen_share": "analyze_data.profile_features: unseen_test_share (numeric)",
    "tvd_train_test": "analyze_data.profile_features: tvd_train_test",
}
NUMERIC_ONLY = ("train_nonfinite_count", "train_unparseable_share", "train_infinite_share", "train_std_ddof0",
                "train_std_ddof1", "train_median", "test_median", "median_shift_score", "median_shift_in_profile",
                "ks_train_test", "test_out_of_range_share", "numeric_code_unseen_share")
CATEGORY_ONLY = ("new_category_count", "new_category_row_share", "tvd_train_test")
SHIFT_CSV_COLUMNS = (*SHIFT_FIELDS, "status", "null_reasons")
SHIFT_STATEMENT = ("Полный TRAIN против признаков TEST. Отчёт для разработчиков: это не INNER-контекст для Qwen "
                   "и не допуск модели. Признаки не удаляются и не преобразуются по найденному сдвигу.")
SHIFT_LIMITATIONS = [
    "*_percent_empty (DataCatalog): пропуск — только пустая ячейка, значение в процентах. "
    "*_share_blank (analyze_data): пропуск — пустая ячейка или только пробелы, значение в долях.",
    "test_missing_percent_empty известна только для quality.missing_changes: изменение не меньше 1 п.п., не больше 10 записей.",
    "train_nonfinite_count (DataCatalog) объединяет нечисловые строки и ±inf; analyze_data разделяет их на "
    "train_unparseable_share и train_infinite_share. Для TEST ни один источник этих значений не даёт.",
    "train_std_ddof0 — np.std по конечным значениям (DataCatalog); train_std_ddof1 — pandas std (analyze_data).",
    "median_shift_score — |медиана TEST − медиана TRAIN| в единицах IQR TRAIN (std, если IQR = 0). "
    "Тот же расчёт DataCatalog выполнен для каждого числового признака, а не только для 20 записей профиля.",
    "ks_train_test — двухвыборочная статистика KS по конечным значениям (analyze_data). Изменение формы "
    "распределения видно в KS, даже когда медиана на месте.",
    "new_category_count — число различных новых непустых значений TEST как есть (DataCatalog, не больше 20 записей). "
    "new_category_row_share — доля всех строк TEST с непустым значением без пробелов по краям, которого нет в TRAIN "
    "(analyze_data). Это разные показатели.",
    "numeric_code_unseen_share — доля конечных значений TEST, которых нет в TRAIN; analyze_data считает её только "
    "для числовых признаков с не больше чем 20 значениями в TRAIN.",
    "Сдвиг — повод исследовать признак на TRAIN-срезах, а не доказательство ухудшения предсказаний.",
]

FIT_STATEMENT = ("Только явно переданный fit, его labels и объявленный context. TEST и другие файлы не читаются. "
                 "Структурная совместимость не означает принятие модели. Совместимость полного TRAIN не доказывает "
                 "совместимость будущего host INNER; этот отчёт не является INNER-контекстом для Qwen.")
FIT_LIMITATIONS = [
    "Группа — хеш всех raw-признаков строки (pandas.util.hash_pandas_object), как в agent.splits.split_rows; "
    "копии строки с любыми labels остаются в одной группе. Дубли не считаются ошибкой.",
    "Разбиение и группы строит agent.round4_classification_candidate.coverage; аудит только пересчитывает их "
    "пересечения и покрытие.",
    "Gates поддержки — настоящая agent.oof_classification.fit_oof_guarded_rule с DEFAULT_GUARD_OPTIONS из recipe. "
    "Её callback отказывается обучать модель, поэтому прохождение gates означает только, что первый fit был бы начат.",
    "Ограничения размера — проверка входов адаптера вероятностей для каждого планового child-fit; "
    "predict-таблица финального fit в этот режим не входит.",
    "Прирост Macro F1, class-drop guard, допуск по времени, успешный полный refit и память без обучения неизвестны.",
]


def peak_memory() -> dict:
    """Peak memory of this process when the platform exposes it; otherwise unknown."""
    try:
        import psutil
    except (ImportError, OSError):
        pass
    else:
        try:
            peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
            if peak:
                return {"peak_bytes": int(peak), "method": "psutil peak_wset", "scope": "this_process_whole_run"}
        except (psutil.Error, OSError):
            # Restricted process inventories can hide this process or deny its
            # counters. This optional metadata must not prevent report export.
            pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak_bytes": int(value) * (1 if sys.platform == "darwin" else 1024),
                "method": "resource.getrusage ru_maxrss", "scope": "this_process_whole_run"}
    except (ImportError, OSError, ValueError):
        return {"peak_bytes": None, "method": None, "scope": "unknown"}


def check_output(output_dir: Path, protected: list[Path], names: tuple[str, ...]) -> Path:
    """Refuse any output inside a folder holding inputs; nothing is written here."""
    output = output_dir.expanduser().resolve()
    if any(output.is_relative_to(root.expanduser().resolve()) for root in protected):
        raise ValueError("--output-dir должна быть вне папок с входными данными: экспорт не пишет рядом с ними.")
    if output.exists() and not output.is_dir():
        raise ValueError("--output-dir указывает на существующий файл, а не на папку.")
    for name in names:
        if (output / name).is_dir():
            raise ValueError(f"В папке вывода {name} является папкой; выберите другую папку.")
    return output


def write_outputs(output: Path, names: tuple[str, ...], report: dict, texts: dict[str, str]) -> None:
    """Write the set; the JSON report goes last and vouches for the other files."""
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / names[0]
    # A report left from a previous run must not vouch for files this run replaces.
    report_path.unlink(missing_ok=True)
    try:
        for name in names[1:]:
            passport.write_text(output / name, texts[name])
        report["files"] = {name: passport.text_sha256(texts[name]) for name in names[1:]}
        passport.write_text(report_path, passport.to_json(report))
    except OSError as error:
        failed = {key: report[key] for key in ("schema_version", "mode", "scope")}
        failed.update(status="error", error=f"{type(error).__name__}: запись прервана, файлы набора недостоверны.")
        with contextlib.suppress(OSError):
            passport.write_text(report_path, passport.to_json(failed))
        raise


def generated(started: float) -> dict:
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - started, 3), "memory": peak_memory()}


def plain(text) -> str:
    """Own fixed texts and runtime reason codes: keep them readable, only protect the table."""
    return str(text).replace("|", r"\|").replace("\n", " ")


def fmt(value) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    return passport.fmt(value)


def render_csv(columns, rows) -> str:
    def cell(value):
        if value is None:
            return ""
        return ("true" if value else "false") if isinstance(value, bool) else value

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([cell(value) for value in row] for row in rows)
    return buffer.getvalue()


# ---------------------------------------------------------------- shift mode

def read_shift_inputs(data_dir: Path, task_id: str) -> tuple[dict, dict[str, dict], analyze_data.Task]:
    catalog = DataCatalog(data_dir)
    passport.select_task(catalog, task_id)
    profile, details = passport.read_profile(catalog, task_id)
    # analyze_data reads the same CSV text (dtype=str, keep_default_na=False, utf-8-sig) as DataCatalog.
    task = analyze_data.Task(catalog.resolve(task_id))
    if catalog.fingerprint(task_id) != profile["dataset_fingerprint"]:
        raise ValueError("Данные изменились во время чтения. Повторите экспорт.")
    if (list(task.train.columns) != [column["name"] for column in profile["columns"]]
            or len(task.train) != profile["train_rows"] or len(task.test) != profile["test_rows"]):
        raise ValueError("analyze_data и DataCatalog прочитали разные таблицы.")
    return profile, details, task


def full_numeric_shift(profile: dict, task: analyze_data.Task, names: list[str]) -> dict[str, dict | None]:
    """The profiler's own median shift for every numeric feature, checked against its 20 records."""
    dtypes = {column["name"]: column["dtype"] for column in profile["columns"]}
    result = {name: _numeric_shift(name, task.train[name], task.test[name])
              for name in names if dtypes[name] in NUMERIC_TYPES}
    for item in profile["quality"]["numeric_shift"]:
        if result.get(item["name"]) != item:
            raise ValueError("Полный расчёт сдвига медиан не совпал с записями профиля DataCatalog.")
    return result


def shift_row(column: dict, detail: dict, measured: dict, shift: dict | None, listed: dict, full: dict,
              constant: set[str]) -> dict:
    name, dtype = column["name"], column["dtype"]
    row, reasons = dict.fromkeys(SHIFT_FIELDS), {}

    def put(field: str, value, reason: str = "undefined_by_source") -> None:
        row[field] = value
        if value is None:
            reasons[field] = reason

    row.update(name=name, dtype=dtype, train_missing_percent_empty=column["missing_percent"],
               train_unique_nonempty=column["unique_count"], is_constant=name in constant)
    put("train_missing_share_blank", measured["missing_train"])
    put("test_missing_share_blank", measured["missing_test"])
    put("train_placeholder_share", measured["placeholder_train"])
    put("train_unique_analyze", None if measured["n_unique"] is None else int(measured["n_unique"]),
        "no_values_in_train")
    change = listed["missing_changes"].get(name)
    put("test_missing_percent_empty", None if change is None else change["test_percent"],
        "outside_source_cap" if full["missing_changes"] else "change_below_1pp_not_reported")
    put("test_nonfinite", None, "not_provided_by_sources")

    if dtype in NUMERIC_TYPES:
        no_part = "no_finite_values_in_a_part"
        put("train_nonfinite_count", detail["distribution"].get("invalid_count"))
        put("train_unparseable_share", measured["unparseable"])
        put("train_infinite_share", measured["infinite"])
        put("train_std_ddof0", detail["stats"].get("std"), "undefined_by_source" if detail["stats"] else "no_finite_train_values")
        put("train_std_ddof1", measured["std"], "undefined_by_source" if detail["stats"] else "no_finite_train_values")
        if shift is None:
            for field in ("train_median", "test_median", "median_shift_score", "median_shift_in_profile"):
                put(field, None, no_part)
        else:
            row.update(train_median=shift["train_median"], test_median=shift["test_median"],
                       median_shift_in_profile=name in listed["numeric_shift"])
            put("median_shift_score", shift["score"], "zero_train_spread_with_changed_median")
        put("ks_train_test", measured["ks_train_test"], no_part)
        put("test_out_of_range_share", measured["test_out_of_range"], no_part)
        many = measured["n_unique"] is not None and measured["n_unique"] > CODE_VALUES
        put("numeric_code_unseen_share", measured["unseen_test_share"],
            "more_than_20_train_values" if many else no_part)
        for field in CATEGORY_ONLY:
            put(field, None, "not_category_dtype")
    else:
        for field in NUMERIC_ONLY:
            put(field, None, "not_numeric_dtype")
        if dtype == "category":
            unseen = listed["new_categories"].get(name)
            put("new_category_count", unseen["count"] if unseen else None if full["new_categories"] else 0,
                "outside_source_cap")
            put("new_category_row_share", measured["unseen_test_share"], "no_nonempty_train_values")
            put("tvd_train_test", measured["tvd_train_test"], "no_nonempty_train_values")
        else:
            for field in CATEGORY_ONLY:
                put(field, None, "unsupported_dtype")
    row["status"] = "partial" if "outside_source_cap" in reasons.values() else "complete"
    row["null_reasons"] = reasons
    return row


def shift_summary(features: list[dict], quality: dict) -> dict:
    numeric = [row for row in features if row["dtype"] in NUMERIC_TYPES]

    def names(rows, value):
        return [row["name"] for row in sorted(rows, key=lambda row: (-value(row), row["name"]))]

    def ks(row):
        return row["ks_train_test"] or 0.0

    warning = [row for row in numeric if ks(row) >= analyze_data.SHIFT_WARNING]
    hidden = [row for row in numeric if row["median_shift_in_profile"] is False]
    same_median = [row for row in warning if row["train_median"] is not None and row["train_median"] == row["test_median"]]
    blank_change = [row for row in features if abs(row["test_missing_share_blank"] - row["train_missing_share_blank"])
                    * 100 >= MISSING_CHANGE_PP]
    return {
        "features": len(features), "numeric_features": len(numeric),
        "median_shift_computed": sum(row["train_median"] is not None for row in numeric),
        "median_shift_in_profile": len(quality["numeric_shift"]),
        "median_shift_outside_profile": len(hidden),
        "ks_thresholds": {"warning": analyze_data.SHIFT_WARNING, "critical": analyze_data.SHIFT_CRITICAL},
        "ks_at_least_warning": names(warning, ks),
        "ks_at_least_critical": names([row for row in numeric if ks(row) >= analyze_data.SHIFT_CRITICAL], ks),
        "ks_warning_outside_profile": names([row for row in hidden if ks(row) >= analyze_data.SHIFT_WARNING], ks),
        "ks_warning_with_unchanged_median": names(same_median, ks),
        "new_category_features": names([row for row in features if row["new_category_count"]],
                                       lambda row: row["new_category_count"]),
        "blank_missing_change_at_least_1pp": len(blank_change),
        "empty_missing_change_in_profile": len(quality["missing_changes"]),
        "constant_features": [row["name"] for row in features if row["is_constant"]],
        "train_nonfinite_features": names([row for row in features if row["train_nonfinite_count"]],
                                          lambda row: row["train_nonfinite_count"]),
    }


def build_shift_report(task_id: str, profile: dict, details: dict[str, dict], task: analyze_data.Task,
                       commit: str | None) -> dict:
    quality = profile["quality"]
    excluded = {profile["id_column"], profile["target"]}
    columns = [column for column in profile["columns"]
               if column["role"] not in {"id", "target"} and column["name"] not in excluded]
    names = [column["name"] for column in columns]
    if len(names) != profile["feature_count"]:
        raise ValueError("Число признаков в профиле не совпадает со списком столбцов.")
    # Distributions only: no leakage scores, correlations or target analysis.
    measured = {row["feature"]: row for row in analyze_data.jsonable(
        analyze_data.profile_features(task, names, {}).to_dict("records"))}
    shifts = full_numeric_shift(profile, task, names)
    listed = {key: {item["name"]: item for item in quality[key]} for key in passport.SOURCE_CAPS}
    full = {key: len(quality[key]) >= cap for key, cap in passport.SOURCE_CAPS.items()}
    constant = set(quality["constant_features"])
    features = [shift_row(column, details[column["name"]], measured[column["name"]], shifts.get(column["name"]),
                          listed, full, constant) for column in columns]
    caps = {key: {"cap": cap, "returned": len(quality[key]), "may_be_truncated": full[key]}
            for key, cap in passport.SOURCE_CAPS.items()}
    config = {"schema_version": SCHEMA_VERSION, "mode": "shift", "field_sources": SHIFT_FIELDS,
              "ks_thresholds": [analyze_data.SHIFT_WARNING, analyze_data.SHIFT_CRITICAL],
              "missing_change_pp": MISSING_CHANGE_PP, "profile_caps": passport.SOURCE_CAPS}
    warnings = [] if commit else ["source_commit недоступен: git не найден или инструмент запущен не из репозитория."]
    report = {
        "schema_version": SCHEMA_VERSION, "mode": "shift", "scope": SHIFT_SCOPE, "statement": SHIFT_STATEMENT,
        "task_id": task_id, "source_commit": commit, "inputs": {"dataset_fingerprint": profile["dataset_fingerprint"]},
        "config": config, "config_sha256": passport.digest(config),
        "train_rows": profile["train_rows"], "test_rows": profile["test_rows"], "feature_count": len(features),
        "status": "partial" if any(row["status"] == "partial" for row in features) else "complete",
        "source_caps": caps,
        "crosscheck": {"numeric_shift_profile_records": len(quality["numeric_shift"]), "matched_full_calculation": True},
        "summary": shift_summary(features, quality), "limitations": SHIFT_LIMITATIONS, "warnings": warnings,
        "features": features, "features_sha256": passport.digest(features),
    }
    # Time, memory and output-file hashes are added after this hash and never enter it.
    report["content_sha256"] = passport.digest(report)
    return report


def render_shift_csv(features: list[dict]) -> str:
    return render_csv(SHIFT_CSV_COLUMNS, [
        [row[field] for field in SHIFT_FIELDS] + [row["status"], ";".join(
            f"{field}={reason}" for field, reason in row["null_reasons"].items())] for row in features])


def render_shift_summary(report: dict) -> str:
    md, table = passport.md, passport.table
    rows = {row["name"]: row for row in report["features"]}
    summary = report["summary"]

    def section(title: str, names: list[str], header: list[str], cells) -> list[str]:
        result = [f"### {title}", ""]
        if not names:
            return result + ["Нет.", ""]
        result += table(["Признак", *header], [[md(name), *cells(rows[name])] for name in names[:README_TOP]])
        if len(names) > README_TOP:
            result += [f"Ещё {len(names) - README_TOP} — в shift_features.csv.", ""]
        return result

    def numeric_cells(row):
        return [fmt(row["ks_train_test"]), f"{fmt(row['train_median'])} → {fmt(row['test_median'])}",
                fmt(row["median_shift_score"])]

    numeric_header = ["KS", "Медиана TRAIN → TEST", "median_shift_score"]
    lines = [f"# Сдвиг TRAIN → TEST: {md(report['task_id'])}", "", report["statement"], ""]
    lines += table(["Поле", "Значение"], [
        ["scope", report["scope"]], ["Статус", report["status"]],
        ["dataset_fingerprint", f"`{report['inputs']['dataset_fingerprint']}`"],
        ["source_commit", f"`{report['source_commit']}`" if report["source_commit"] else "—"],
        ["Строк TRAIN / TEST", f"{report['train_rows']} / {report['test_rows']}"],
        ["Признаков (строк таблицы)", str(report["feature_count"])],
        ["Числовых: сдвиг медиан посчитан / в профиле", f"{summary['median_shift_computed']} / {summary['median_shift_in_profile']}"],
        ["content_sha256", f"`{report['content_sha256']}`"],
    ])
    lines += ["## Что скрывает текущий профиль", ""]
    lines += section(f"KS ≥ {summary['ks_thresholds']['warning']} у признаков вне 20 записей numeric_shift",
                     summary["ks_warning_outside_profile"], numeric_header, numeric_cells)
    lines += section(f"Медиана не изменилась, но KS ≥ {summary['ks_thresholds']['warning']}",
                     summary["ks_warning_with_unchanged_median"], numeric_header, numeric_cells)
    lines += ["## Все сигналы", ""]
    lines += section(f"Наибольший KS (≥ {summary['ks_thresholds']['warning']})", summary["ks_at_least_warning"],
                     [*numeric_header, "В профиле"],
                     lambda row: [*numeric_cells(row), fmt(row["median_shift_in_profile"])])
    lines += section("Новые категории в TEST", summary["new_category_features"],
                     ["Различных новых значений", "Доля строк TEST", "TVD"],
                     lambda row: [fmt(row["new_category_count"]), fmt(row["new_category_row_share"]),
                                  fmt(row["tvd_train_test"])])
    changed = sorted((row for row in report["features"]
                      if abs(row["test_missing_share_blank"] - row["train_missing_share_blank"]) * 100 >= MISSING_CHANGE_PP),
                     key=lambda row: (-abs(row["test_missing_share_blank"] - row["train_missing_share_blank"]), row["name"]))
    lines += section(f"Пропуски (пусто или пробелы) изменились не меньше чем на {MISSING_CHANGE_PP:g} п.п.",
                     [row["name"] for row in changed], ["TRAIN, доля", "TEST, доля"],
                     lambda row: [fmt(row["train_missing_share_blank"]), fmt(row["test_missing_share_blank"])])
    lines += section("Непустые нечисловые значения в числовых признаках TRAIN", summary["train_nonfinite_features"],
                     ["Значений", "Нечисловых, доля", "±inf, доля"],
                     lambda row: [fmt(row["train_nonfinite_count"]), fmt(row["train_unparseable_share"]),
                                  fmt(row["train_infinite_share"])])
    constant = summary["constant_features"]
    lines += ["### Константные признаки TRAIN", "",
              (f"{len(constant)}: " + ", ".join(md(name) for name in constant) + ".") if constant else "Нет.", ""]
    lines += ["## Ограничения", ""] + [f"- {text}" for text in report["limitations"]]
    lines += [f"- {text}" for text in report["warnings"]] + [""]
    return "\n".join(lines)


def export_shift(data_dir: Path, task_id: str, output_dir: Path) -> dict:
    started = time.perf_counter()
    if not data_dir.is_dir():
        raise ValueError("--data-dir не является папкой.")
    output = check_output(output_dir, [data_dir], SHIFT_FILES)
    profile, details, task = read_shift_inputs(data_dir, task_id)
    report = build_shift_report(task_id, profile, details, task, passport.source_commit())
    texts = {"shift_features.csv": render_shift_csv(report["features"]),
             "shift_summary.md": render_shift_summary(report)}
    report["generated"] = generated(started)
    write_outputs(output, SHIFT_FILES, report, texts)
    return report


# ---------------------------------------------------------------- fit mode

class TrainingRefused(RuntimeError):
    """Raised by the gate probe callback: the fit-only audit never fits a model."""


def _refuse_training(X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
    raise TrainingRefused("The fit-only audit never trains a model")


def _strings(value, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Отказ: context.{name} должен быть списком строк.")
    return list(value)


def checked_fit(X_fit, y_fit, context) -> tuple[dict, list[str], list[str], np.ndarray]:
    """Contract refusal before any split: schema, labels, class order and the frozen recipe."""
    if not isinstance(context, dict):
        raise ValueError("Отказ: context должен быть словарём.")
    try:
        recipe = checked_recipe(context.get("recipe"))
    except ValueError as error:
        raise ValueError(f"Отказ: recipe не совпадает с действующим кандидатом ({error}).") from error
    if recipe["variant"] != "oof":
        raise ValueError("Отказ: аудит проверяет только variant 'oof' действующего кандидата.")
    classes = recipe["probability_recipe"]["classes"]
    if _strings(context.get("classes"), "classes") != classes:
        raise ValueError("Отказ: порядок классов context не совпадает с recipe.")
    try:
        metric = get_metric(context.get("metric")).name
    except ValueError as error:
        raise ValueError(f"Отказ: {error}.") from error
    if context.get("task_type") != "classification" or context.get("prediction_kind") != "labels" or metric != "Macro F1":
        raise ValueError("Отказ: кандидат поддерживает только классификацию Macro F1 с метками.")
    features = _strings(context.get("features"), "features")
    numeric, categorical = _strings(context.get("numeric"), "numeric"), _strings(context.get("categorical"), "categorical")
    if not features or len(set(features)) != len(features):
        raise ValueError("Отказ: список признаков пуст или содержит повторы.")
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise ValueError("Отказ: роли numeric и categorical должны без пересечений покрывать все признаки.")
    if set(features) & {context.get("id_column"), context.get("target_column")}:
        raise ValueError("Отказ: ID и target не могут быть признаками.")
    if not isinstance(X_fit, pd.DataFrame) or not X_fit.columns.is_unique or list(X_fit.columns) != features:
        raise ValueError("Отказ: X_fit должен содержать ровно объявленные признаки в объявленном порядке.")
    labels = np.asarray(y_fit, dtype=object)
    if labels.ndim != 1 or len(labels) != len(X_fit) or not len(labels):
        raise ValueError("Отказ: y_fit должен быть непустым одномерным и выровненным с X_fit.")
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError("Отказ: labels должны быть непустыми строками.")
    undeclared = set(labels.tolist()) - set(classes)
    if undeclared:
        raise ValueError(f"Отказ: в labels {len(undeclared)} необъявленных значений; классы задаёт context.")
    return recipe, classes, features, labels


def _part(groups: np.ndarray, rows: np.ndarray) -> dict:
    return {"rows": int(len(rows)), "groups": int(len(np.unique(groups[rows])))}


def duplicate_summary(groups: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    frame = pd.DataFrame({"group": groups, "label": labels})
    sizes = frame.groupby("group")["label"].transform("size")
    conflicting = frame.groupby("group")["label"].transform("nunique") > 1
    duplicated_groups = frame.loc[sizes > 1, "group"].nunique()
    return {
        "policy": "same_raw_features_same_group_any_labels; duplicates_are_not_an_error",
        "duplicate_groups": int(duplicated_groups),
        "rows_in_duplicate_groups": int((sizes > 1).sum()),
        "extra_copies": int(len(frame) - frame["group"].nunique()),
        "conflicting_label_groups": int(frame.loc[conflicting, "group"].nunique()),
        "rows_in_conflicting_groups": int(conflicting.sum()),
        "conflicting_rows_by_class": {label: int((conflicting & (frame["label"] == label)).sum()) for label in classes},
    }


def partition_layout(groups: np.ndarray, partition: dict) -> dict:
    """Recount group overlap and held coverage of the canonical partition; nothing is trusted."""
    n = len(groups)
    A = np.asarray(partition["tuning_rows"], dtype=np.int64)
    B = np.asarray(partition["selection_rows"], dtype=np.int64)
    in_A = np.zeros(n, dtype=bool)
    in_A[A] = True
    held_times = np.zeros(n, dtype=np.int64)
    folds = []
    for fold in partition["folds"]:
        fit, held = (np.asarray(fold[key], dtype=np.int64) for key in ("fit_rows", "held_rows"))
        np.add.at(held_times, held, 1)
        folds.append({"fit": _part(groups, fit), "held": _part(groups, held),
                      "group_overlap": len(set(groups[fit].tolist()) & set(groups[held].tolist())),
                      "row_overlap": int(len(np.intersect1d(fit, held))),
                      "rows_outside_A": int((~in_A[fit]).sum() + (~in_A[held]).sum())})
    return {"A": _part(groups, A), "B": _part(groups, B),
            "ab_group_overlap": len(set(groups[A].tolist()) & set(groups[B].tolist())),
            "ab_row_overlap": int(len(np.intersect1d(A, B))),
            "rows_outside_ab": int(n - len(np.union1d(A, B))),
            "folds": folds,
            "held_coverage": {"A_rows": int(len(A)), "exactly_once": int((held_times[A] == 1).sum()),
                              "never": int((held_times[A] == 0).sum()),
                              "more_than_once": int((held_times[A] > 1).sum())}}


def oof_gates(labels: np.ndarray, classes: list[str], partition: dict, recipe: dict) -> tuple[str, str]:
    """Run the candidate's own pre-callback gates; the callback refuses to train."""
    try:
        result = fit_oof_guarded_rule(
            np.empty((len(labels), 0)), labels.tolist(), classes, groups=partition["groups"], split_policy="grouped",
            tuning_rows=partition["tuning_rows"], selection_rows=partition["selection_rows"], folds=partition["folds"],
            fit_predict=_refuse_training, fold_timeout_seconds=recipe["caps"]["callback_seconds"],
            deadline=time.monotonic() + 3600, clock=time.monotonic, guard_options=recipe["guard_options"])
    except OOFContractError as error:
        return "violated", f"oof_contract_error: {error}"
    fits = result["fits"]
    if fits and fits[0].get("error_type") == TrainingRefused.__name__ and result["completed_fits"] == 0:
        return "passed", "first_fit_reached_without_training"
    if not fits and result["reason"] != "deadline_exhausted":
        return "violated", result["reason"]
    return "unknown", result["reason"]


def child_inputs(X_fit, labels, context, recipe, fit_rows, predict_rows) -> tuple[str, str | None]:
    try:
        probability_inputs(X_fit.iloc[fit_rows], labels[fit_rows], X_fit.iloc[predict_rows], context,
                           recipe["probability_recipe"])
    except ProbabilitySkipError as error:
        return "violated", error.reason
    return "passed", None


def support_constraints(partition: dict, recipe: dict, classes: list[str], rows: int) -> list[dict]:
    """Observed margins next to each threshold. The decision itself comes from the runtime calls."""
    options, caps = recipe["guard_options"], recipe["probability_recipe"]["caps"]
    result = [{"id": "fit_rows", "source": "probability caps.max_fit_rows", "limit": "max",
               "threshold": caps["max_fit_rows"], "observed": rows},
              {"id": "declared_classes", "source": "probability caps.max_classes", "limit": "max",
               "threshold": caps["max_classes"], "observed": len(classes)}]
    support = partition["support"]
    if partition["available"]:
        def minimum(part):
            return min(support[part][label]["rows"] for label in classes)

        def below(part, threshold):
            return [label for label in classes if support[part][label]["rows"] < threshold]

        result += [
            {"id": "A_rows", "source": "DEFAULT_GUARD_OPTIONS.min_tuning_samples", "limit": "min",
             "threshold": options["min_tuning_samples"], "observed": len(partition["tuning_rows"])},
            {"id": "A_min_class_rows", "source": "DEFAULT_GUARD_OPTIONS.min_tuning_class_count", "limit": "min",
             "threshold": options["min_tuning_class_count"], "observed": minimum("A"),
             "classes_below": below("A", options["min_tuning_class_count"])},
            {"id": "fold_fit_min_class_rows", "source": "oof_classification: every class in each fold fit",
             "limit": "min", "threshold": 1,
             "observed": min(fold["fit"][label]["rows"] for fold in support["folds"] for label in classes)},
            {"id": "B_rows", "source": "DEFAULT_GUARD_OPTIONS.min_selection_samples", "limit": "min",
             "threshold": options["min_selection_samples"], "observed": len(partition["selection_rows"])},
            {"id": "B_min_class_rows", "source": "DEFAULT_GUARD_OPTIONS.min_selection_class_count", "limit": "min",
             "threshold": options["min_selection_class_count"], "observed": minimum("B"),
             "classes_below": below("B", options["min_selection_class_count"])},
        ]
    for item in result:
        item["satisfied"] = (item["observed"] >= item["threshold"] if item["limit"] == "min"
                             else item["observed"] <= item["threshold"])
    return result


def not_evaluated(recipe: dict) -> list[dict]:
    options, caps = recipe["guard_options"], recipe["probability_recipe"]["caps"]
    return [
        {"id": "selection_macro_f1_gain", "status": "unknown",
         "requirement": f"Δ Macro F1 на B ≥ {options['min_selection_gain']}",
         "reason": "нужны OOF-вероятности; аудит не обучает модели"},
        {"id": "selection_class_f1_drop", "status": "unknown",
         "requirement": f"падение F1 любого класса на B ≤ {options['max_selection_class_f1_drop']}",
         "reason": "нужны OOF-вероятности; аудит не обучает модели"},
        {"id": "time_admission", "status": "unknown",
         "requirement": "4·probe·1.5 + max(120, 2·probe) ≤ 600 с (docs/round4_classification_integration.md)",
         "reason": "нужен измеренный wall time identity probe на этом fit"},
        {"id": "fresh_full_refit", "status": "unknown", "requirement": "успешный финальный fit на всех строках fit",
         "reason": "проверяется только обучением"},
        {"id": "final_predict_caps", "status": "unknown",
         "requirement": f"≤ {caps['max_predict_rows']} predict-строк и ≤ {caps['max_matrix_cells']} ячеек",
         "reason": "predict-таблица host не входит в fit-only режим"},
        {"id": "memory", "status": "unknown",
         "requirement": f"RSS дерева процессов ≤ {recipe['caps']['memory_limit_bytes']} байт",
         "reason": "без обучения потребление памяти моделью не измеряется"},
        {"id": "host_inner_equivalence", "status": "unknown",
         "requirement": "переданный fit совпадает с будущим host INNER",
         "reason": "проверяется на fit, который передаст host"},
    ]


def audit_oof_fit(X_fit, y_fit, context) -> dict:
    """Aggregate fit-only OOF readiness of the measured round 4 candidate; never trains or reads files.

    ``context`` declares features, numeric/categorical roles, classes, task
    semantics and the frozen ``recipe`` exactly as ``execute_candidate``
    receives them. A malformed contract raises ``ValueError``. The result has
    no row IDs, row labels, split indices or group keys; only class names and
    counts. ``structurally_compatible`` only means the first OOF fit would start.
    """
    recipe, classes, features, labels = checked_fit(X_fit, y_fit, context)
    partition = coverage(X_fit, labels, context)
    groups = np.asarray(partition["groups"], dtype=np.uint64)
    checks = [{"id": "contract", "status": "passed", "reason": None,
               "source": "round4 checked_recipe, class order, task semantics, schema and labels"}]

    def check(check_id, status, reason, source):
        checks.append({"id": check_id, "status": status, "reason": reason, "source": source})

    layout = refusal = None
    if partition["available"]:
        check("grouped_split", "passed", partition["reason"], "round4_classification_candidate.coverage")
        layout = partition_layout(groups, partition)
        ab_clean = not (layout["ab_group_overlap"] or layout["ab_row_overlap"] or layout["rows_outside_ab"])
        folds_clean = len(layout["folds"]) == 3 and not any(
            fold["group_overlap"] or fold["row_overlap"] or fold["rows_outside_A"] for fold in layout["folds"])
        held_once = layout["held_coverage"]["exactly_once"] == layout["held_coverage"]["A_rows"]
        check("ab_disjoint", "passed" if ab_clean else "violated", None if ab_clean else "ab_overlap_or_gap",
              "recount of coverage A/B")
        check("folds_disjoint", "passed" if folds_clean else "violated", None if folds_clean else "fold_overlap",
              "recount of coverage folds")
        check("held_cover_A_once", "passed" if held_once else "violated", None if held_once else "held_coverage_broken",
              "recount of coverage folds")
        check("oof_support_gates", *oof_gates(labels, classes, partition, recipe),
              "oof_classification.fit_oof_guarded_rule (pre-callback gates)")
    else:
        # coverage() keeps only the error type; the canonical split states the reason itself.
        try:
            split_rows(X_fit, labels, features, True, {"strategy": "grouped"})
        except ValueError as error:
            refusal = str(error)
        check("grouped_split", "violated", refusal or partition.get("error_type"), "agent.splits.split_rows")
        for check_id in ("ab_disjoint", "folds_disjoint", "held_cover_A_once", "oof_support_gates"):
            check(check_id, "not_reached", "grouped_split_unavailable", "")

    planned = []
    if partition["available"]:
        planned = [(f"fold_{index}", fold["fit_rows"], fold["held_rows"])
                   for index, fold in enumerate(partition["folds"], 1)]
        planned.append(("selection", partition["tuning_rows"], partition["selection_rows"]))
    # Fit side only: the host predict table is outside this mode.
    planned.append(("final_fit_side", list(range(len(labels))), [0]))
    children = []
    for name, fit_rows, predict_rows in planned:
        status, reason = child_inputs(X_fit, labels, context, recipe, fit_rows, predict_rows)
        children.append({"child": name, "status": status, "reason": reason})
    violated = [child for child in children if child["status"] == "violated"]
    check("child_input_caps", "violated" if violated else "passed",
          ";".join(f"{child['child']}={child['reason']}" for child in violated) or None,
          "round4_classification_probability input checks per planned child fit")

    constraints = support_constraints(partition, recipe, classes, len(labels))
    gate = next(item for item in checks if item["id"] == "oof_support_gates")
    if gate["status"] == "passed" and not all(item["satisfied"] for item in constraints):
        # A displayed margin and the runtime gate disagree: do not claim compatibility.
        gate.update(status="unknown", reason="support_counts_disagree_with_gate")
    failed = [f"{item['id']}:{item['reason']}" for item in checks if item["status"] == "violated"]
    unclear = [f"{item['id']}:{item['reason']}" for item in checks if item["status"] == "unknown"]
    decision = "ineligible" if failed else "unknown" if unclear else "structurally_compatible"
    probability = recipe["probability_recipe"]
    components = len(probability["base_recipe"]["components"])
    return {
        "scope": FIT_SCOPE, "decision": decision, "reasons": failed or unclear,
        "model_acceptance": "not_evaluated",
        "rows": int(len(labels)), "feature_count": len(features), "raw_feature_groups": int(len(np.unique(groups))),
        "classes": classes, "duplicates": duplicate_summary(groups, labels, classes),
        "split": {"available": partition["available"], "reason": partition["reason"], "refusal": refusal,
                  "metadata": partition.get("metadata")},
        "layout": layout, "support": partition["support"], "checks": checks, "child_inputs": children,
        "constraints": constraints, "not_evaluated": not_evaluated(recipe),
        "planned_physical_fits": {"planned": 5 * components, "cap": recipe["caps"]["max_physical_fits"]},
        "recipe": {"variant": recipe["variant"], "sha256": json_hash(recipe), "split": recipe["split"],
                   "guard_options": recipe["guard_options"], "candidate_caps": recipe["caps"],
                   "probability_caps": probability["caps"]},
    }


def parse_classes(value: str | None) -> list[str] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value) if value.lstrip().startswith("[") else next(csv.reader([value]))
    except ValueError as error:
        raise ValueError("--classes: ожидается JSON-массив или список через запятую.") from error
    if (not isinstance(parsed, list) or not parsed
            or any(not isinstance(label, str) or not label.strip() for label in parsed)
            or len(set(parsed)) != len(parsed)):
        raise ValueError("--classes: нужны различные непустые метки.")
    return parsed


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fit_inputs(task_md: Path, dictionary: Path, fit_csv: Path, classes: str | None):
    """Read only the three named files; TEST is never opened."""
    paths = {"task.md": task_md, "data_dictionary.csv": dictionary, "fit.csv": fit_csv}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Файл {name} не найден.")
    before = {name: file_sha256(path) for name, path in paths.items()}
    text = task_md.read_text(encoding="utf-8-sig")
    metric = parse_task_metric(text)
    if metric.name != "Macro F1" or metric.prediction_kind != "labels":
        raise ValueError("fit-режим проверяет только классификацию Macro F1 с метками.")
    # Same dictionary rules as offline_baseline.load_data, which also reads TEST and cannot be used here.
    meta = read_csv(dictionary)
    if not {"feature", "role", "dtype"} <= set(meta.columns) or meta["feature"].duplicated().any():
        raise ValueError("data_dictionary.csv: нужны столбцы feature, role, dtype без повторов признаков.")
    meta = meta.set_index("feature")
    roles = meta["role"].str.strip().str.lower()
    dtypes = meta["dtype"].str.strip().str.lower()
    ids, targets = roles.index[roles == "id"].tolist(), roles.index[roles == "target"].tolist()
    if len(ids) != 1 or len(targets) != 1:
        raise ValueError("data_dictionary.csv: нужен ровно один id и один target.")
    id_column, target = ids[0], targets[0]
    fit = read_csv(fit_csv)
    if target not in fit:
        raise ValueError("В fit.csv нет столбца target.")
    features = [name for name in fit.columns if name not in {id_column, target}]
    for name in features:
        if name not in roles.index or roles[name] != "feature" or dtypes[name] not in {"numeric", "category", "binary"}:
            raise ValueError("fit.csv содержит столбец, не объявленный признаком numeric/category/binary.")
    labels = fit[target].to_numpy(dtype=object)
    declared, explicit = _class_list(text), parse_classes(classes)
    if declared is not None and explicit is not None and list(declared) != explicit:
        raise ValueError("--classes противоречит порядку классов в task.md.")
    if declared is not None:
        order, source = list(declared), "declared_in_task_md"
    elif explicit is not None:
        order, source = explicit, "command_line"
    else:
        order, source = sorted(set(map(str, labels))), "sorted_fit_labels"
    context = {"features": features, "numeric": [name for name in features if dtypes[name] in NUMERIC_TYPES],
               "categorical": [name for name in features if dtypes[name] == "category"], "classes": order,
               "task_type": "classification", "prediction_kind": metric.prediction_kind, "metric": metric.to_dict(),
               "id_column": id_column, "target_column": target,
               "recipe": make_recipe(fixed_incumbent_recipe(), order, variant="oof")}
    if {name: file_sha256(path) for name, path in paths.items()} != before:
        raise ValueError("Входные файлы изменились во время чтения. Повторите аудит.")
    inputs = {name: {"sha256": before[name], "bytes": path.stat().st_size} for name, path in paths.items()}
    return fit.loc[:, features], labels, context, source, inputs


def build_fit_report(audit: dict, inputs: dict, class_source: str, commit: str | None) -> dict:
    config = {"schema_version": SCHEMA_VERSION, "mode": "fit", "recipe_sha256": audit["recipe"]["sha256"],
              "split": audit["recipe"]["split"], "guard_options": audit["recipe"]["guard_options"],
              "candidate_caps": audit["recipe"]["candidate_caps"], "probability_caps": audit["recipe"]["probability_caps"]}
    warnings = []
    if class_source == "sorted_fit_labels":
        warnings.append("Порядок классов взят из отсортированных меток fit. Класс, которого нет в fit, так не виден; "
                        "для host INNER передайте --classes из TaskSpec.")
    if commit is None:
        warnings.append("source_commit недоступен: git не найден или инструмент запущен не из репозитория.")
    report = {"schema_version": SCHEMA_VERSION, "mode": "fit", "scope": FIT_SCOPE, "statement": FIT_STATEMENT,
              "source_commit": commit, "inputs": inputs, "class_order_source": class_source,
              "config": config, "config_sha256": passport.digest(config), "audit": audit,
              "limitations": FIT_LIMITATIONS, "warnings": warnings}
    report["content_sha256"] = passport.digest(report)
    return report


def support_parts(audit: dict) -> list[tuple[str, dict]]:
    support = audit["support"]
    parts = [("allowed", support["allowed"])]
    if "A" in support:
        parts += [("A", support["A"]), ("B", support["B"])]
        for index, fold in enumerate(support["folds"], 1):
            parts += [(f"fold_{index}_fit", fold["fit"]), (f"fold_{index}_held", fold["held"])]
    return parts


def render_support_csv(audit: dict) -> str:
    return render_csv(("part", "class", "rows", "groups"), [
        [part, label, counts[label]["rows"], counts[label]["groups"]]
        for part, counts in support_parts(audit) for label in audit["classes"]])


def render_fit_summary(report: dict) -> str:
    md, table = passport.md, passport.table
    audit = report["audit"]
    lines = [f"# OOF-готовность fit: {audit['decision']}", "", report["statement"], ""]
    lines += table(["Поле", "Значение"], [
        ["scope", report["scope"]], ["Решение", audit["decision"]],
        ["Причины", ", ".join(plain(reason) for reason in audit["reasons"]) or "—"],
        ["Принятие модели", audit["model_acceptance"]],
        ["Строк / признаков / raw-feature групп", f"{audit['rows']} / {audit['feature_count']} / {audit['raw_feature_groups']}"],
        ["Порядок классов", report["class_order_source"]],
        ["recipe_sha256", f"`{audit['recipe']['sha256']}`"],
        ["fit.csv sha256", f"`{report['inputs']['fit.csv']['sha256']}`"],
        ["content_sha256", f"`{report['content_sha256']}`"],
    ])
    lines += ["## Проверки", ""]
    lines += table(["Проверка", "Статус", "Причина"],
                   [[check["id"], check["status"], plain(check["reason"] or "—")] for check in audit["checks"]])
    if audit["split"]["refusal"]:
        lines += [f"Отказ канонического разбиения: {plain(audit['split']['refusal'])}", ""]
    lines += ["## Ограничения кандидата", ""]
    lines += table(["Ограничение", "Порог", "Наблюдение", "Выполнено", "Классы ниже порога"], [
        [item["id"], f"{item['limit']} {item['threshold']}", str(item["observed"]), fmt(item["satisfied"]),
         ", ".join(md(label) for label in item.get("classes_below", [])) or "—"] for item in audit["constraints"]])
    lines += ["## Поддержка классов (строки / группы)", ""]
    parts = support_parts(audit)
    header = [name for name, _ in parts if not name.endswith("_held")]
    lines += table(["Класс", *header], [[md(label), *(f"{counts[label]['rows']} / {counts[label]['groups']}"
                                                      for name, counts in parts if name in header)]
                                        for label in audit["classes"]])
    lines += ["Held-части folds — в oof_class_support.csv.", ""]
    if audit["layout"]:
        layout = audit["layout"]
        lines += ["## Границы разбиений", ""]
        lines += table(["Часть", "Строк", "Групп", "Пересечение групп", "Пересечение строк"], [
            ["A", str(layout["A"]["rows"]), str(layout["A"]["groups"]), str(layout["ab_group_overlap"]), str(layout["ab_row_overlap"])],
            ["B", str(layout["B"]["rows"]), str(layout["B"]["groups"]), "—", "—"],
            *([f"fold {index} fit/held", f"{fold['fit']['rows']} / {fold['held']['rows']}",
               f"{fold['fit']['groups']} / {fold['held']['groups']}", str(fold["group_overlap"]), str(fold["row_overlap"])]
              for index, fold in enumerate(layout["folds"], 1))])
        coverage_rows = layout["held_coverage"]
        lines += [f"Строк A в held-folds: ровно один раз {coverage_rows['exactly_once']} из {coverage_rows['A_rows']}, "
                  f"ни разу {coverage_rows['never']}, больше одного раза {coverage_rows['more_than_once']}.", ""]
    duplicates = audit["duplicates"]
    lines += ["## Дубли raw-признаков", "",
              f"Групп с копиями: {duplicates['duplicate_groups']}, строк в них: {duplicates['rows_in_duplicate_groups']}, "
              f"лишних копий: {duplicates['extra_copies']}. Групп с разными labels: {duplicates['conflicting_label_groups']}, "
              f"строк в них: {duplicates['rows_in_conflicting_groups']}. Политика: {duplicates['policy']}.", ""]
    lines += ["## Не оценено без обучения", ""]
    lines += table(["Пункт", "Требование", "Статус", "Почему"], [
        [item["id"], plain(item["requirement"]), item["status"], plain(item["reason"])] for item in audit["not_evaluated"]])
    lines += ["## Ограничения", ""] + [f"- {text}" for text in report["limitations"]]
    lines += [f"- {text}" for text in report["warnings"]] + [""]
    return "\n".join(lines)


def export_fit(task_md: Path, dictionary: Path, fit_csv: Path, output_dir: Path, classes: str | None = None) -> dict:
    started = time.perf_counter()
    output = check_output(output_dir, [task_md.parent, dictionary.parent, fit_csv.parent], FIT_FILES)
    X_fit, y_fit, context, source, inputs = read_fit_inputs(task_md, dictionary, fit_csv, classes)
    report = build_fit_report(audit_oof_fit(X_fit, y_fit, context), inputs, source, passport.source_commit())
    texts = {"oof_class_support.csv": render_support_csv(report["audit"]),
             "oof_fit_summary.md": render_fit_summary(report)}
    report["generated"] = generated(started)
    write_outputs(output, FIT_FILES, report, texts)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    shift = modes.add_parser("shift", help="полная таблица сдвига TRAIN → TEST по DataCatalog")
    shift.add_argument("--data-dir", type=Path, required=True, help="Папка с задачами (подпапки с task.md)")
    shift.add_argument("--task", required=True, help="task_id из DataCatalog.discover()")
    shift.add_argument("--output-dir", type=Path, required=True, help="Папка для трёх файлов, вне --data-dir")
    fit = modes.add_parser("fit", help="fit-only проверка разбиений OOF-кандидата")
    fit.add_argument("--task-md", type=Path, required=True, help="task.md: метрика и объявленный порядок классов")
    fit.add_argument("--dictionary", type=Path, required=True, help="data_dictionary.csv: роли и типы")
    fit.add_argument("--fit-csv", type=Path, required=True, help="Таблица fit с признаками и target")
    fit.add_argument("--classes", help="Порядок классов, если task.md его не объявляет: JSON или через запятую")
    fit.add_argument("--output-dir", type=Path, required=True, help="Папка для трёх файлов, вне папок входов")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic output on Windows consoles and pipes
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        if args.mode == "shift":
            report = export_shift(args.data_dir, args.task, args.output_dir)
            line = (f"Сдвиг {report['task_id']}: {report['status']}, признаков {report['feature_count']}, "
                    f"числовых со сдвигом медиан {report['summary']['median_shift_computed']}")
        else:
            report = export_fit(args.task_md, args.dictionary, args.fit_csv, args.output_dir, args.classes)
            line = (f"OOF fit: {report['audit']['decision']}, строк {report['audit']['rows']}, "
                    f"групп {report['audit']['raw_feature_groups']}")
    except (OSError, ValueError) as error:
        print(f"Экспорт не выполнен: {error}", file=sys.stderr)
        return 1
    print(f"{line}; content_sha256 {report['content_sha256'][:12]}…, файлы в {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
