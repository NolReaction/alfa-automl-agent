"""Export the existing DataCatalog feature profile as a portable developer passport.

Offline developer tool: it reads one discovered task through
app.profiling.DataCatalog and writes passport.json, features.csv and README.md
into an explicit folder outside the data directory. It trains nothing, never
writes next to the task files and is not part of the competition runtime.
Every value comes from the profiler. A field the profiler does not return
stays null with a reason instead of being recomputed here.

    python experiment_round4_feature_passport.py --data-dir data --task regression --output-dir ../passport-regression
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from app.profiling import DataCatalog


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "round4-feature-passport/1"
SCOPE = "developer_full_train_test_features"
SOURCE = "app.profiling.DataCatalog"
OUTPUT_FILES = ("passport.json", "features.csv", "README.md")
NUMERIC_TYPES = {"numeric", "binary"}
STATS = ("min", "max", "mean", "median", "std")
README_TOP = 10
FEW_VALUES = 10  # README question only: a shifted feature this coarse may encode a period or group
# List lengths hard-coded in app.profiling._quality. A list that reaches its cap
# may have been cut, so a feature missing from it is unknown rather than clean.
SOURCE_CAPS = {"new_categories": 20, "numeric_shift": 20, "missing_changes": 10}
CAP_FIELDS = {"new_categories": "new_category_count", "numeric_shift": "shift_score",
              "missing_changes": "test_missing_percent"}

# The whitelist. Each exported field and where DataCatalog returns it; profile
# keys that are not listed here (preview, target_distribution, class_balance,
# examples, descriptions, warnings, ...) never reach the passport.
FIELD_SOURCES = {
    "name": "profile()['columns'][].name",
    "role": "profile()['columns'][].role",
    "dtype": "profile()['columns'][].dtype",
    "train_missing_count": "profile()['columns'][].missing_count",
    "train_missing_percent": "profile()['columns'][].missing_percent",
    "train_nonfinite_count": "column()['distribution']['invalid_count']",
    "train_unique_count": "profile()['columns'][].unique_count",
    "is_constant": "profile()['quality']['constant_features']",
    **{f"train_{stat}": f"column()['stats']['{stat}']" for stat in STATS},
    "test_missing_percent": "profile()['quality']['missing_changes'][].test_percent",
    "new_category_count": "profile()['quality']['new_categories'][].count",
    "shift_train_median": "profile()['quality']['numeric_shift'][].train_median",
    "shift_test_median": "profile()['quality']['numeric_shift'][].test_median",
    "shift_score": "profile()['quality']['numeric_shift'][].score",
}
CSV_COLUMNS = (*FIELD_SOURCES, "status", "null_reasons")
LIMITATIONS = [
    "DataCatalog считает пропуски, уникальные значения, нечисловые значения и числовые сводки по отдельному признаку только на TRAIN.",
    "Доля пропусков в TEST известна только для признаков из quality.missing_changes: изменение не меньше 1 п.п., не больше 10 записей.",
    "Число нечисловых значений в TEST профайлер не возвращает.",
    "Пропуском считается только пустая ячейка; строки 'NA' и ' ' считаются значениями. "
    "is_constant учитывает пустую ячейку как отдельное значение.",
    "quality.new_categories и quality.numeric_shift ограничены 20 записями; признак вне заполненного списка получает null с причиной outside_source_cap.",
    "shift_score — расстояние между медианами TEST и TRAIN в единицах IQR TRAIN. Это повод исследовать признак, а не доказательство ухудшения предсказаний.",
    "Паспорт описывает полный TRAIN и признаки TEST. Это не INNER-контекст для Qwen: его строят отдельно на fit, переданном host.",
]
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~&])")


def digest(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_commit() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


# ---------------------------------------------------------------- validation and reading

def check_output(output_dir: Path, data_dir: Path) -> Path:
    """Refuse any output that could land among task files; nothing is written here."""
    output = output_dir.expanduser().resolve()
    root = data_dir.expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("--output-dir должна быть вне --data-dir: экспорт не пишет рядом с файлами задач.")
    if output.exists() and not output.is_dir():
        raise ValueError("--output-dir указывает на существующий файл, а не на папку.")
    for name in OUTPUT_FILES:
        if (output / name).is_dir():
            raise ValueError(f"В папке вывода {name} является папкой; выберите другую папку.")
    return output


def select_task(catalog: DataCatalog, task_id: str) -> None:
    tasks = catalog.discover()
    if not tasks:
        raise ValueError("В --data-dir не найдено задач (папок с task.md).")
    known = {task["id"]: task for task in tasks}
    if task_id not in known:
        raise ValueError(f"Задача {task_id!r} не найдена. Допустимые --task: {', '.join(sorted(known))}.")
    if not known[task_id]["available"]:
        raise ValueError(f"Задача {task_id!r} недоступна: {known[task_id]['error']}")


def read_profile(catalog: DataCatalog, task_id: str) -> tuple[dict, dict[str, dict]]:
    profile = catalog.profile(task_id)
    excluded = {profile["id_column"], profile["target"]}
    details = {column["name"]: catalog.column(task_id, column["name"]) for column in profile["columns"]
               if column["name"] not in excluded}
    # profile() and column() load separately; a file replaced in between would mix two datasets.
    if catalog.fingerprint(task_id) != profile["dataset_fingerprint"]:
        raise ValueError("Данные изменились во время чтения профиля. Повторите экспорт.")
    return profile, details


# ---------------------------------------------------------------- passport

def feature_row(column: dict, detail: dict, listed: dict[str, dict], full: dict[str, bool],
                constant: set[str]) -> dict:
    name, dtype = column["name"], column["dtype"]
    numeric = dtype in NUMERIC_TYPES
    row = dict.fromkeys(FIELD_SOURCES)
    reasons = {}

    def put(field: str, value, reason: str = "undefined_by_source") -> None:
        row[field] = value
        if value is None:
            reasons[field] = reason

    row.update(name=name, role=column["role"], dtype=dtype, train_missing_count=column["missing_count"],
               train_missing_percent=column["missing_percent"], train_unique_count=column["unique_count"],
               is_constant=name in constant)
    if numeric:
        put("train_nonfinite_count", detail["distribution"].get("invalid_count"))
        stats = detail["stats"]
        for stat in STATS:
            put(f"train_{stat}", stats.get(stat), "undefined_by_source" if stats else "no_finite_train_values")
    else:
        for field in ("train_nonfinite_count", *(f"train_{stat}" for stat in STATS)):
            put(field, None, "not_numeric_dtype")

    change = listed["missing_changes"].get(name)
    put("test_missing_percent", None if change is None else change["test_percent"],
        "outside_source_cap" if full["missing_changes"] else "change_below_1pp_not_reported")

    if dtype == "category":
        unseen = listed["new_categories"].get(name)
        if unseen is not None:
            put("new_category_count", unseen["count"])
        else:
            put("new_category_count", None if full["new_categories"] else 0, "outside_source_cap")
    else:
        put("new_category_count", None, "not_category_dtype")

    shift = listed["numeric_shift"].get(name)
    for field in ("shift_train_median", "shift_test_median", "shift_score"):
        if not numeric:
            put(field, None, "not_numeric_dtype")
        elif shift is not None:
            put(field, shift[field.removeprefix("shift_")])
        else:
            # Absent from an incomplete list: the profiler found no finite value in TRAIN or TEST.
            put(field, None, "outside_source_cap" if full["numeric_shift"] else "no_finite_values_in_a_part")

    row["status"] = "partial" if "outside_source_cap" in reasons.values() else "ok"
    row["null_reasons"] = reasons
    return row


def build_passport(task_id: str, profile: dict, details: dict[str, dict], commit: str | None) -> dict:
    quality = profile["quality"]
    excluded = {profile["id_column"], profile["target"]}
    listed = {key: {item["name"]: item for item in quality[key]} for key in SOURCE_CAPS}
    full = {key: len(quality[key]) >= cap for key, cap in SOURCE_CAPS.items()}
    constant = set(quality["constant_features"])
    # Roles come from the data dictionary through the profiler, never from column names.
    features = [feature_row(column, details[column["name"]], listed, full, constant)
                for column in profile["columns"]
                if column["role"] not in {"id", "target"} and column["name"] not in excluded]
    if len(features) != profile["feature_count"]:
        raise ValueError("Число признаков в профиле не совпадает со списком столбцов.")
    caps, warnings = {}, []
    for key, cap in SOURCE_CAPS.items():
        outside = sum(row["null_reasons"].get(CAP_FIELDS[key]) == "outside_source_cap" for row in features)
        caps[key] = {"cap": cap, "returned": len(quality[key]), "may_be_truncated": full[key],
                     "features_outside_cap": outside}
        if outside:
            warnings.append(f"quality.{key}: возвращено {len(quality[key])} записей при лимите {cap}; "
                            f"{outside} признаков получили null outside_source_cap.")
    if commit is None:
        warnings.append("source_commit недоступен: git не найден или инструмент запущен не из репозитория.")
    passport = {
        "schema_version": SCHEMA_VERSION, "task_id": task_id, "scope": SCOPE, "source": SOURCE,
        "source_commit": commit, "dataset_fingerprint": profile["dataset_fingerprint"],
        "train_rows": profile["train_rows"], "test_rows": profile["test_rows"],
        "feature_count": profile["feature_count"],
        "status": "partial" if any(row["status"] == "partial" for row in features) else "complete",
        "limitations": LIMITATIONS, "warnings": warnings, "source_caps": caps,
        "field_sources": FIELD_SOURCES, "features": features, "features_sha256": digest(features),
    }
    # Time, paths and rendered-file hashes are added after this hash and never enter it.
    passport["content_sha256"] = digest(passport)
    return passport


# ---------------------------------------------------------------- rendering

def render_csv(features: list[dict]) -> str:
    def cell(value):
        if value is None:
            return ""
        return ("true" if value else "false") if isinstance(value, bool) else value

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in features:
        reasons = ";".join(f"{field}={reason}" for field, reason in row["null_reasons"].items())
        writer.writerow([cell(row[field]) for field in FIELD_SOURCES] + [row["status"], reasons])
    return buffer.getvalue()


def md(value) -> str:
    """Escape an untrusted name for a Markdown table cell, keeping it on one line."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ↵ ")
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def missing_change(row: dict) -> float | None:
    if row["test_missing_percent"] is None:
        return None
    return abs(row["test_missing_percent"] - row["train_missing_percent"])


def ranked(features: list[dict], value) -> list[dict]:
    """Rows with a non-zero value, largest first; null values are skipped, not treated as zero."""
    return sorted((row for row in features if value(row)), key=lambda row: (-value(row), row["name"]))


def questions(passport: dict) -> list[str]:
    features = passport["features"]
    unseen = ranked(features, lambda row: row["new_category_count"])
    shifted = ranked(features, lambda row: row["shift_score"])
    few_values = [row for row in shifted if row["train_unique_count"] <= FEW_VALUES]
    constant = [row for row in features if row["is_constant"]]
    changed = ranked(features, missing_change)
    nonfinite = ranked(features, lambda row: row["train_nonfinite_count"])
    # Names are escaped plain text: a code span would show the escapes literally.
    result = []
    if unseen:
        result.append(f"Категориальных признаков со значениями TEST, которых нет в TRAIN: {len(unseen)} "
                      f"(больше всего у {md(unseen[0]['name'])}, новых значений: {unseen[0]['new_category_count']}). "
                      "Проверить безопасный unknown fallback в текущем transformer.")
    if shifted:
        result.append(f"Наибольший shift_score у {md(shifted[0]['name'])}: {fmt(shifted[0]['shift_score'])} IQR TRAIN. "
                      "Проверить вклад признака на TRAIN-срезах; сдвиг сам по себе не повод удалять признак.")
    if few_values:
        shown = ", ".join(f"{md(row['name'])} (уникальных значений: {row['train_unique_count']}, медиана "
                          f"{fmt(row['shift_train_median'])} → {fmt(row['shift_test_median'])})" for row in few_values[:3])
        result.append(f"Сдвинуты признаки с малым числом значений в TRAIN: {shown}. Проверить, не задают ли они "
                      "период, группу или смесь подтипов, по которым стоит строить валидацию.")
    if constant:
        result.append(f"Константных признаков в TRAIN: {len(constant)} (например {md(constant[0]['name'])}). "
                      "Проверить, исключает ли их текущий transformer.")
    if changed:
        row = changed[0]
        result.append(f"Признаков с изменением доли пропусков не меньше 1 п.п.: {len(changed)}; больше всего у "
                      f"{md(row['name'])}: {fmt(row['train_missing_percent'])}% → {fmt(row['test_missing_percent'])}%. "
                      "Проверить обработку пропуска и стабильность источника.")
    if nonfinite:
        result.append(f"Числовых признаков с непустыми нечисловыми значениями: {len(nonfinite)} "
                      f"(больше всего у {md(nonfinite[0]['name'])}: {nonfinite[0]['train_nonfinite_count']}). "
                      "Проверить, что они не превращаются молча в 0.")
    for key, cap in passport["source_caps"].items():
        if cap["features_outside_cap"]:
            result.append(f"Профайлер вернул quality.{key} только {cap['returned']} записей (лимит {cap['cap']}); "
                          f"для {cap['features_outside_cap']} признаков паспорт не знает значения. "
                          "Нужен ли расширенный лимит через integration_request к app/profiling.py?")
    return result[:5] or ["Заметных сигналов в пределах лимитов профайлера нет."]


def render_readme(passport: dict) -> str:
    features = passport["features"]
    lines = [f"# Паспорт признаков: {md(passport['task_id'])}", "",
             "Developer-экспорт профиля `app.profiling.DataCatalog`: полный TRAIN и признаки TEST. "
             "Паспорт не выбирает модель и не доказывает рост качества. Поля описаны в "
             "`docs/round4_feature_passport.md`, полная таблица — `features.csv`.", ""]
    lines += table(["Поле", "Значение"], [
        ["Статус", passport["status"]],
        ["dataset_fingerprint", f"`{passport['dataset_fingerprint']}`"],
        ["source_commit", f"`{passport['source_commit']}`" if passport["source_commit"] else "—"],
        ["Строк TRAIN / TEST", f"{passport['train_rows']} / {passport['test_rows']}"],
        ["Признаков", str(passport["feature_count"])],
        ["content_sha256", f"`{passport['content_sha256']}`"],
    ])

    def section(title: str, header: list[str], value, cells) -> list[str]:
        rows = ranked(features, value)
        if not rows:
            return [f"### {title}", "", "Нет в пределах лимитов профайлера.", ""]
        result = [f"### {title}", ""] + table(header, [[md(row["name"]), *cells(row)] for row in rows[:README_TOP]])
        if len(rows) > README_TOP:
            result += [f"Ещё {len(rows) - README_TOP} признаков — в features.csv.", ""]
        return result

    lines += ["## Заметные факты", ""]
    lines += section("Новые категории в TEST", ["Признак", "Новых значений"],
                     lambda row: row["new_category_count"], lambda row: [str(row["new_category_count"])])
    lines += section("Сдвиг медианы TRAIN → TEST по shift_score", ["Признак", "Медиана TRAIN", "Медиана TEST", "shift_score"],
                     lambda row: row["shift_score"], lambda row: [fmt(row["shift_train_median"]),
                                                                  fmt(row["shift_test_median"]), fmt(row["shift_score"])])
    lines += section("Изменение доли пропусков не меньше 1 п.п.", ["Признак", "TRAIN, %", "TEST, %"], missing_change,
                     lambda row: [fmt(row["train_missing_percent"]), fmt(row["test_missing_percent"])])
    lines += section("Непустые нечисловые значения в числовых признаках TRAIN", ["Признак", "Значений"],
                     lambda row: row["train_nonfinite_count"], lambda row: [str(row["train_nonfinite_count"])])
    constant = [row["name"] for row in features if row["is_constant"]]
    lines += ["### Константные признаки TRAIN", ""]
    if constant:
        shown = ", ".join(md(name) for name in constant[:README_TOP])
        extra = f" и ещё {len(constant) - README_TOP}" if len(constant) > README_TOP else ""
        lines += [f"{len(constant)}: {shown}{extra}.", ""]
    else:
        lines += ["Нет.", ""]
    lines += ["## Вопросы команде", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(questions(passport), 1)] + [""]
    lines += ["## Ограничения", ""] + [f"- {text}" for text in passport["limitations"]]
    lines += [f"- {text}" for text in passport["warnings"]] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- writing

def to_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_export(output: Path, passport: dict, csv_text: str, readme_text: str) -> None:
    """Write the three files; passport.json goes last and vouches for the other two."""
    output.mkdir(parents=True, exist_ok=True)
    passport_path = output / "passport.json"
    # A passport left from a previous run must not vouch for files this run replaces.
    passport_path.unlink(missing_ok=True)
    try:
        write_text(output / "features.csv", csv_text)
        write_text(output / "README.md", readme_text)
        passport["files"] = {"features.csv": text_sha256(csv_text), "README.md": text_sha256(readme_text)}
        write_text(passport_path, to_json(passport))
    except OSError as error:
        failed = {key: passport[key] for key in ("schema_version", "task_id", "scope", "dataset_fingerprint")}
        failed.update(status="error", error=f"{type(error).__name__}: запись экспорта прервана, файлы набора недостоверны.")
        with contextlib.suppress(OSError):
            write_text(passport_path, to_json(failed))
        raise


def export(data_dir: Path, task_id: str, output_dir: Path) -> dict:
    started = time.perf_counter()
    if not data_dir.is_dir():
        raise ValueError("--data-dir не является папкой.")
    output = check_output(output_dir, data_dir)
    catalog = DataCatalog(data_dir)
    select_task(catalog, task_id)
    profile, details = read_profile(catalog, task_id)
    passport = build_passport(task_id, profile, details, source_commit())
    csv_text, readme_text = render_csv(passport["features"]), render_readme(passport)
    passport["generated"] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "elapsed_seconds": round(time.perf_counter() - started, 3)}
    write_export(output, passport, csv_text, readme_text)
    return passport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Папка с задачами (подпапки с task.md)")
    parser.add_argument("--task", required=True, help="task_id из DataCatalog.discover()")
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка для трёх файлов, вне --data-dir")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic output on Windows consoles and pipes
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        passport = export(args.data_dir, args.task, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Экспорт паспорта не выполнен: {error}", file=sys.stderr)
        return 1
    print(f"Паспорт {passport['task_id']}: {passport['status']}, признаков {passport['feature_count']}, "
          f"fingerprint {passport['dataset_fingerprint'][:12]}…, файлы: {', '.join(OUTPUT_FILES)} "
          f"в {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
