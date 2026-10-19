"""Audit the organizers' task data: types, missing values, target, outliers, leakage, train/test shift.

Needs only numpy and pandas. Run `prepare_data.py all` first. Reports go to data/analysis/<task>/
(data/ is ignored by git, so nothing from the datasets reaches the repository).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TASKS = ("regression", "classification")
SEED = 42
NUMERIC_TYPES = {"numeric", "binary"}
# Tokens that often hide a missing value. Only reported: "NA" can also be a real class label.
PLACEHOLDERS = {"na", "n/a", "nan", "null", "none", "?", "-", "--", "inf", "-inf", "unknown", "undefined", "#n/a"}
NAME_HINT = re.compile(r"target|label|leak|answer|outcome|verdict|predict|future|result", re.IGNORECASE)
LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2}
LEVEL_TITLES = {"critical": "Критично", "warning": "Важно", "info": "К сведению"}

MIN_LEVEL_ROWS = 5          # categories rarer than this count as "rare" in the feature profile
SCORE_MIN_LEVEL_ROWS = 2    # a level seen fewer times in the fitting half is not trusted when scoring a single feature
RARE_CLASS_ROWS = 50
LEAK_CRITICAL = 0.9         # single feature reaches this R2 / Macro F1 on a holdout: almost surely a leak
LEAK_WARNING_R2 = 0.5
LEAK_WARNING_F1_LIFT = 0.3  # Macro F1 above the majority-class baseline
ORDER_WARNING_R2 = 0.05     # row order or ID that already explains this much
ORDER_WARNING_F1_LIFT = 0.1
SHIFT_WARNING, SHIFT_CRITICAL = 0.1, 0.3
CORRELATION_LIMIT = 0.98
MAX_CORRELATION_COLUMNS = 400

PROFILE_COLUMNS = [
    "feature", "declared", "kind", "missing_train", "missing_test", "placeholder_train", "n_unique", "top_share",
    "unparseable", "infinite", "min", "p01", "median", "p99", "max", "mean", "std", "skew", "zero_share",
    "negative_share", "integer_like", "outlier_1_5_iqr", "outlier_3_iqr", "robust_z_max", "test_out_of_range",
    "ks_train_test", "rare_levels", "unseen_test_share", "tvd_train_test", "whitespace_share",
    "case_collisions", "numeric_like_share", "corr_pearson", "corr_spearman",
]


# ---------------------------------------------------------------- loading

def read_csv(path: Path) -> pd.DataFrame:
    # Everything as text, exactly as written: IDs such as 001 and labels such as NA stay intact.
    if not path.is_file():
        raise ValueError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def find_data_dir(task_dir: Path) -> Path:
    samples = [path for path in task_dir.rglob("sample_submission.csv")
               if ".agent_runs" not in path.relative_to(task_dir).parts]
    if len(samples) != 1:
        raise ValueError(f"Expected exactly one sample_submission.csv inside {task_dir}; found {len(samples)}. "
                         "Run prepare_data.py first.")
    return samples[0].parent


class Task:
    def __init__(self, task_dir: Path):
        self.data_dir = find_data_dir(task_dir)
        meta = read_csv(self.data_dir / "data_dictionary.csv").apply(lambda column: column.str.strip())
        if not {"feature", "role", "dtype"}.issubset(meta.columns):
            raise ValueError("data_dictionary.csv must contain feature, role and dtype columns")
        if meta["feature"].duplicated().any():
            raise ValueError("data_dictionary.csv contains duplicate feature definitions")
        self.meta = meta.set_index("feature")
        roles = self.meta["role"].str.lower()
        ids, targets = roles.index[roles == "id"].tolist(), roles.index[roles == "target"].tolist()
        if len(ids) != 1 or len(targets) != 1:
            raise ValueError("data_dictionary.csv must declare exactly one id and one target")
        self.id_column, self.target = ids[0], targets[0]
        self.dtypes = self.meta["dtype"].str.lower().to_dict()
        if self.dtypes[self.target] not in {"numeric", "category"}:
            raise ValueError(f"Unsupported target dtype {self.dtypes[self.target]!r}")
        self.classification = self.dtypes[self.target] == "category"
        self.train = read_csv(self.data_dir / "train.csv")
        self.test = read_csv(self.data_dir / "test.csv")
        self.sample = read_csv(self.data_dir / "sample_submission.csv")
        for name, frame in (("train.csv", self.train), ("test.csv", self.test)):
            if self.id_column not in frame:
                raise ValueError(f"{name}: no ID column {self.id_column!r}")
        if self.target not in self.train:
            raise ValueError("train.csv: no target column")


# ---------------------------------------------------------------- helpers

def add(findings: list, level: str, area: str, message: str, columns=()) -> None:
    columns = list(columns)
    findings.append({"level": level, "area": area, "message": message,
                     "n_columns": len(columns), "columns": columns[:12]})


def to_number(column: pd.Series) -> pd.Series:
    return pd.to_numeric(column.str.strip(), errors="coerce")


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict("records"))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> float:
    matrix = np.bincount(y_true * classes + y_pred, minlength=classes * classes).reshape(classes, classes)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    present = denominator > 0
    return float((2 * np.diag(matrix)[present] / denominator[present]).mean())


def is_numeric_kind(declared: str, train: pd.Series) -> bool:
    if declared in NUMERIC_TYPES:
        return True
    if declared == "category":
        return False
    filled = train.str.strip()
    filled = filled[filled != ""]
    return len(filled) > 0 and to_number(filled).notna().mean() >= 0.95  # undeclared column: guess


# ---------------------------------------------------------------- schema and IDs

def audit_schema(task: Task, findings: list) -> tuple[list[str], dict]:
    train_features = [name for name in task.train.columns if name not in {task.id_column, task.target}]
    test_features = [name for name in task.test.columns if name != task.id_column]
    if set(train_features) != set(test_features):
        add(findings, "critical", "схема", "Признаки train.csv и test.csv различаются",
            sorted(set(train_features) ^ set(test_features)))
    elif train_features != test_features:
        add(findings, "warning", "схема", "Порядок столбцов в train.csv и test.csv разный")
    if task.target in task.test:
        add(findings, "critical", "схема", "Целевой столбец присутствует в test.csv")
    undeclared = [name for name in train_features if name not in task.dtypes]
    absent = [name for name, role in task.meta["role"].str.lower().items()
              if role == "feature" and name not in task.train]
    if undeclared:
        add(findings, "warning", "схема", "Признаки отсутствуют в data_dictionary.csv", undeclared)
    if absent:
        add(findings, "warning", "схема", "Признаки из data_dictionary.csv отсутствуют в train.csv", absent)
    sample_ok = (list(task.sample.columns) == [task.id_column, task.target]
                 and task.sample[task.id_column].tolist() == task.test[task.id_column].tolist())
    if not sample_ok:
        add(findings, "critical", "схема", "sample_submission.csv не совпадает с test.csv по столбцам или ID")
    features = [name for name in train_features if name in test_features]
    kinds = pd.Series([task.dtypes.get(name, "undeclared") for name in features]).value_counts().to_dict()
    return features, {"train_shape": list(task.train.shape), "test_shape": list(task.test.shape),
                      "n_features": len(features), "declared_kinds": kinds, "sample_matches_test": sample_ok,
                      "id_column": task.id_column, "target": task.target,
                      "task_type": "classification" if task.classification else "regression"}


def analyze_ids(task: Task, findings: list) -> dict:
    train_ids = task.train[task.id_column].str.strip()
    test_ids = task.test[task.id_column].str.strip()
    result = {"train_unique": bool(train_ids.is_unique), "test_unique": bool(test_ids.is_unique),
              "train_test_overlap": len(set(train_ids) & set(test_ids))}
    if not (result["train_unique"] and result["test_unique"]):
        add(findings, "critical", "ID", "ID в train.csv или test.csv повторяются")
    a, b = to_number(train_ids), to_number(test_ids)
    result["numeric"] = bool(a.notna().all() and b.notna().all())
    # IDs that are just 0..N-1 in each file overlap by construction; that is not a leak.
    counters = [bool(values.notna().all() and (values.to_numpy() == np.arange(len(values))).all()) for values in (a, b)]
    result["train_is_row_counter"], result["test_is_row_counter"] = counters
    if result["train_test_overlap"] and any(counters):
        add(findings, "info", "ID", f"ID — порядковые номера строк внутри файла (0…N−1), {result['train_test_overlap']} значений "
            "совпадают в train и test: ID не уникален между файлами и не несёт информации, не использовать как признак и ключ слияния")
    elif result["train_test_overlap"]:
        add(findings, "critical", "ID", f"{result['train_test_overlap']} ID одновременно в train и test")
    if result["numeric"]:
        result.update(train_range=[a.min(), a.max()], test_range=[b.min(), b.max()],
                      test_inside_train_range=float(((b >= a.min()) & (b <= a.max())).mean()),
                      train_sorted=bool(a.is_monotonic_increasing), test_sorted=bool(b.is_monotonic_increasing))
        if result["test_inside_train_range"] < 0.05 and not any(counters):
            add(findings, "info", "ID", "Диапазон ID в test не пересекается с train: "
                "вероятно, разбиение по номеру/времени — валидацию лучше делать так же, а не случайным shuffle")
    else:
        result["examples"] = train_ids.head(3).tolist()
    return result


# ---------------------------------------------------------------- features

def profile_numeric(train: pd.Series, test: pd.Series, cache: dict, name: str) -> dict:
    values, test_values = to_number(train), to_number(test)
    cache[name] = values.where(np.isfinite(values))
    filled = train.str.strip().ne("")
    finite = cache[name].dropna()
    result = {"unparseable": float((values.isna() & filled).mean()), "infinite": float(np.isinf(values).mean()),
              "n_unique": int(finite.nunique())}
    if finite.empty:
        return result
    q01, q25, median, q75, q99 = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = q75 - q25
    mad = (finite - median).abs().median()
    test_finite = test_values.where(np.isfinite(test_values)).dropna()
    result.update(
        min=finite.min(), p01=q01, median=median, p99=q99, max=finite.max(), mean=finite.mean(),
        std=finite.std(), skew=finite.skew(), zero_share=float((finite == 0).mean()),
        negative_share=float((finite < 0).mean()), integer_like=bool((finite % 1 == 0).all()),
        top_share=float(finite.value_counts(normalize=True).iloc[0]),
        # With IQR = 0 (mostly one value) every other value would count as an outlier: skip the share.
        outlier_1_5_iqr=float(((finite < q25 - 1.5 * iqr) | (finite > q75 + 1.5 * iqr)).mean()) if iqr > 0 else None,
        outlier_3_iqr=float(((finite < q25 - 3 * iqr) | (finite > q75 + 3 * iqr)).mean()) if iqr > 0 else None,
        robust_z_max=float((finite - median).abs().max() / (1.4826 * mad)) if mad > 0 else None,
        test_out_of_range=float(((test_finite < finite.min()) | (test_finite > finite.max())).mean())
        if len(test_finite) else None,
        ks_train_test=ks_statistic(finite.to_numpy(float), test_finite.to_numpy(float)),
    )
    if result["n_unique"] <= 20 and len(test_finite):  # code-like column: are the test values known from train?
        result["unseen_test_share"] = float((~test_finite.isin(finite.unique())).mean())
    return result


def profile_category(train: pd.Series, test: pd.Series) -> dict:
    text, test_text = train.str.strip(), test.str.strip()
    counts = text[text != ""].value_counts()
    test_filled = test_text[test_text != ""]
    test_counts = test_filled.value_counts()
    result = {"n_unique": int(counts.size), "rare_levels": int((counts < MIN_LEVEL_ROWS).sum()),
              "whitespace_share": float(((train != text) & (text != "")).mean())}
    if counts.empty:
        return result
    shares, test_shares = counts / counts.sum(), test_counts / max(test_counts.sum(), 1)
    union = shares.index.union(test_shares.index)
    result.update(
        top_share=float(shares.iloc[0]),
        unseen_test_share=float((~test_filled.isin(counts.index)).sum() / max(len(test_text), 1)),
        tvd_train_test=float(0.5 * (shares.reindex(union, fill_value=0) - test_shares.reindex(union, fill_value=0)).abs().sum()),
        case_collisions=int(pd.Series(counts.index).str.lower().duplicated().sum()),
        numeric_like_share=float(pd.to_numeric(text[text != ""], errors="coerce").notna().mean()),
    )
    return result


def profile_features(task: Task, features: list[str], cache: dict) -> pd.DataFrame:
    rows = []
    for name in features:
        train, test = task.train[name], task.test[name]
        declared = task.dtypes.get(name, "undeclared")
        numeric = is_numeric_kind(declared, train)
        row = {"feature": name, "declared": declared, "kind": "numeric" if numeric else "category",
               "missing_train": float(train.str.strip().eq("").mean()),
               "missing_test": float(test.str.strip().eq("").mean()),
               "placeholder_train": float(train.str.strip().str.lower().isin(PLACEHOLDERS).mean())}
        row.update(profile_numeric(train, test, cache, name) if numeric else profile_category(train, test))
        if declared == "binary":
            levels = train.str.strip()
            row["n_unique"] = int(levels[levels != ""].nunique())
        rows.append(row)
    profile = pd.DataFrame(rows).reindex(columns=PROFILE_COLUMNS)
    for column in PROFILE_COLUMNS:
        if column not in {"feature", "declared", "kind", "integer_like"}:
            profile[column] = pd.to_numeric(profile[column], errors="coerce")  # all-None columns arrive as object
    return profile


def duplicate_columns(cache: dict, task: Task, profile: pd.DataFrame) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for name in profile["feature"]:
        values = cache[name] if name in cache else task.train[name].str.strip()
        digest = hashlib.md5(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()).hexdigest()
        groups.setdefault(digest, []).append(name)
    return [names for names in groups.values() if len(names) > 1]


def correlated_pairs(cache: dict) -> list[dict]:
    if not 2 <= len(cache) <= MAX_CORRELATION_COLUMNS:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns have no correlation
        matrix = pd.DataFrame(cache).corr().abs()
    upper = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack()
    top = upper[upper >= CORRELATION_LIMIT].sort_values(ascending=False).head(20)
    return [{"a": a, "b": b, "abs_corr": float(value)} for (a, b), value in top.items()]


def feature_findings(profile: pd.DataFrame, duplicates: list, pairs: list, findings: list) -> None:
    def pick(mask: pd.Series, by: str) -> list[str]:
        return profile[mask].sort_values(by, ascending=False)["feature"].tolist()

    numeric, category = profile["kind"] == "numeric", profile["kind"] == "category"
    split_by = pick((profile["unseen_test_share"] >= 0.9) & (profile["n_unique"] <= 20), "unseen_test_share")
    if split_by:
        add(findings, "critical", "сдвиг", "Значения этих признаков в test почти целиком отсутствуют в train: test отделён от train "
            "по ним (например, по периоду). Случайный CV будет завышать качество — валидировать тем же разбиением "
            "(обучение на ранних значениях, проверка на последних)", split_by)
    empty = pick(profile["missing_train"] >= 0.5, "missing_train")
    some = pick(profile["missing_train"] > 0, "missing_train")
    drift = pick((profile["missing_test"] - profile["missing_train"]).abs() >= 0.05, "missing_test")
    if empty:
        add(findings, "warning", "пропуски", "Признаки, пустые более чем наполовину", empty)
    if some:
        add(findings, "info", "пропуски", f"Пропуски в train есть у {len(some)} из {len(profile)} признаков "
            "(пустая строка; в baseline заполняются медианой/__MISSING__)", some)
    if drift:
        add(findings, "warning", "пропуски", "Доля пропусков в test отличается от train на 5 п.п. и больше", drift)
    hidden = pick((profile["placeholder_train"] > 0) & category, "placeholder_train")
    if hidden:
        add(findings, "info", "пропуски", "Категории со значениями вроде NA/null/?/unknown — возможно, это пропуски", hidden)
    for mask, level, text in (
        (numeric & (profile["unparseable"] > 0), "warning", "В числовых признаках есть нечисловые значения"),
        (numeric & (profile["infinite"] > 0), "warning", "В числовых признаках есть inf"),
        (profile["n_unique"].fillna(0) <= 1, "warning", "Константные признаки (бесполезны)"),
        ((profile["declared"] == "binary") & (profile["n_unique"] > 2), "warning", "binary-признаки с более чем двумя значениями"),
        (category & (profile["whitespace_share"] > 0), "warning", "Категории с пробелами по краям значений"),
        (category & (profile["case_collisions"] > 0), "warning", "Категории, различающиеся только регистром"),
        (category & (profile["unseen_test_share"] >= 0.01), "warning", "В test ≥1% строк с категориями, которых нет в train"),
        (numeric & (profile["test_out_of_range"] >= 0.01), "warning", "В test ≥1% значений вне диапазона train"),
    ):
        names = pick(mask, "feature")
        if names:
            add(findings, level, "типы", text, names)
    for mask, text in (
        (profile["top_share"].fillna(0) >= 0.999, "Почти константные признаки (≥99.9% одного значения)"),
        (category & (profile["n_unique"] > 100), "Категории с более чем 100 значениями (нужен target/frequency encoding)"),
        (category & (profile["numeric_like_share"] >= 0.95) & (profile["n_unique"] > 20),
         "Категориальные по словарю, но выглядят числовыми"),
        (numeric & (profile["declared"] == "numeric") & profile["integer_like"].eq(True)
         & (profile["n_unique"] <= 10), "Числовые с ≤10 целыми значениями — возможно, категории"),
    ):
        names = pick(mask & (profile["n_unique"].fillna(0) > 1), "feature")
        if names:
            add(findings, "info", "типы", text, names)
    heavy = pick(numeric & (profile["outlier_3_iqr"] >= 0.01), "outlier_3_iqr")
    extreme = pick(numeric & (profile["robust_z_max"] >= 50), "robust_z_max")
    skewed = pick(numeric & (profile["skew"].abs() >= 3), "skew")
    if heavy:
        add(findings, "info", "выбросы", "≥1% значений дальше 3·IQR от квартилей", heavy)
    if extreme:
        add(findings, "info", "выбросы", "Одиночные экстремальные значения (robust z ≥ 50)", extreme)
    if skewed:
        add(findings, "info", "выбросы", "Сильно скошенные признаки |skew| ≥ 3 (кандидаты на log/клиппинг)", skewed)
    for column, label in (("ks_train_test", "KS"), ("tvd_train_test", "TVD")):
        critical = pick(profile[column] >= SHIFT_CRITICAL, column)
        warning = pick((profile[column] >= SHIFT_WARNING) & (profile[column] < SHIFT_CRITICAL), column)
        if critical:
            add(findings, "critical", "сдвиг", f"Сильный сдвиг распределения train→test ({label} ≥ {SHIFT_CRITICAL})", critical)
        if warning:
            add(findings, "warning", "сдвиг", f"Заметный сдвиг распределения train→test ({label} ≥ {SHIFT_WARNING})", warning)
    for names in duplicates:
        add(findings, "warning", "типы", "Столбцы с одинаковым содержимым (дубли)", names)
    if pairs:
        add(findings, "info", "типы", f"{len(pairs)} пар числовых признаков с |corr| ≥ {CORRELATION_LIMIT} (избыточность)",
            [f"{pair['a']}~{pair['b']}" for pair in pairs])


# ---------------------------------------------------------------- target

def encode_target(task: Task) -> tuple[np.ndarray, list[str]]:
    raw = task.train[task.target]
    if task.classification:
        codes, classes = pd.factorize(raw, sort=True)
        return codes.astype(int), [str(label) for label in classes]
    return to_number(raw).to_numpy(float), []


def analyze_target(task: Task, y: np.ndarray, classes: list[str], findings: list) -> dict:
    raw = task.train[task.target]
    result = {"empty": int(raw.str.strip().eq("").sum())}
    if result["empty"]:
        add(findings, "critical", "цель", f"Пустые значения целевой переменной: {result['empty']}")
    if not task.classification:
        return analyze_regression_target(y, result, findings)
    counts = raw.value_counts()
    shares = counts / counts.sum()
    k = len(counts)
    top = float(shares.iloc[0])
    result.update(
        n_classes=k, counts=counts.to_dict(), shares=shares.round(5).to_dict(),
        imbalance_ratio=float(counts.iloc[0] / counts.iloc[-1]), majority_share=top,
        majority_macro_f1=float(2 * top / (1 + top) / k), uniform_random_macro_f1=float(1 / k),
        normalized_entropy=float(-(shares * np.log(shares)).sum() / np.log(k)) if k > 1 else 0.0,
        rare_classes={label: int(n) for label, n in counts.items() if n < RARE_CLASS_ROWS},
        label_whitespace=int((raw != raw.str.strip()).sum()),
        label_case_collisions=int(pd.Series(counts.index).str.lower().duplicated().sum()),
    )
    if k < 2:
        add(findings, "critical", "цель", "В целевой переменной меньше двух классов")
    if result["imbalance_ratio"] >= 10:
        add(findings, "critical" if result["imbalance_ratio"] >= 100 else "warning", "дисбаланс",
            f"Дисбаланс классов {result['imbalance_ratio']:.0f}:1, мажоритарный класс {top:.1%}. "
            f"Macro F1 константного ответа ≈ {result['majority_macro_f1']:.3f}: нужны веса классов/пороги и стратификация")
    if result["rare_classes"]:
        smallest = int(counts.iloc[-1])
        add(findings, "critical" if smallest < MIN_LEVEL_ROWS else "warning", "дисбаланс",
            f"Редкие классы (<{RARE_CLASS_ROWS} строк; минимум {smallest}): стратифицированный CV может не собраться, "
            "Macro F1 сильно зависит от них", list(result["rare_classes"]))
    if result["label_whitespace"] or result["label_case_collisions"]:
        add(findings, "warning", "цель", "Метки классов с пробелами или различающиеся регистром")
    adjacent = float((y[1:] == y[:-1]).mean()) if len(y) > 1 else float("nan")
    expected = float((shares ** 2).sum())
    result["adjacent_same_class"], result["adjacent_same_class_expected"] = adjacent, expected
    if adjacent > expected + 0.1:
        add(findings, "warning", "утечка", f"Соседние строки train чаще одного класса ({adjacent:.2f} против {expected:.2f} "
            "при перемешивании): файл упорядочен — не использовать порядок строк, валидацию перемешивать")
    return result


def analyze_regression_target(y: np.ndarray, result: dict, findings: list) -> dict:
    series = pd.Series(y)
    bad = int(series.isna().sum() - result["empty"])
    if bad > 0:
        add(findings, "critical", "цель", f"Нечисловые значения целевой переменной: {bad}")
    series = series.dropna()
    series = series[np.isfinite(series)]
    if series.empty:
        add(findings, "critical", "цель", "В целевой переменной нет ни одного числового значения")
        return result
    deviation = (series - series.mean()) ** 2
    ordered = np.sort(deviation.to_numpy())[::-1]
    tail = {f"top_{share:g}": float(ordered[:max(1, int(len(ordered) * share))].sum() / ordered.sum())
            for share in (0.001, 0.01, 0.05)} if ordered.sum() > 0 else {}
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    result.update(
        n=len(series), mean=series.mean(), std=series.std(ddof=0), min=series.min(), max=series.max(),
        quantiles={f"p{int(p * 100):02d}": series.quantile(p) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        skew=series.skew(), kurtosis=series.kurt(), zero_share=float((series == 0).mean()),
        negative_share=float((series < 0).mean()), n_unique=int(series.nunique()),
        top_value_share=float(series.value_counts(normalize=True).iloc[0]),
        rmse_constant_mean=float(series.std(ddof=0)),
        outlier_3_iqr=float(((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).mean()) if iqr > 0 else None,
        squared_error_share_of_top_rows=tail,
        skew_after_log1p=float(np.log1p(series).skew()) if series.min() >= 0 else None,
    )
    if abs(result["skew"]) >= 2:
        message = f"Целевая скошена (skew {result['skew']:.1f}): RMSE определяют редкие большие значения"
        if result["skew_after_log1p"] is not None:
            message += f"; после log1p skew {result['skew_after_log1p']:.1f} — проверить обучение на log1p с обратным преобразованием"
        add(findings, "info", "цель", message)
    if tail.get("top_0.01", 0) >= 0.2:
        add(findings, "warning", "выбросы", f"1% самых далёких от среднего строк даёт {tail['top_0.01']:.0%} дисперсии цели: "
            "RMSE чувствителен к хвосту; клиппинг цели может ухудшить метрику, а выбросы стоит проверить")
    if result["top_value_share"] >= 0.1:
        add(findings, "info", "цель", f"{result['top_value_share']:.0%} строк имеют одно и то же значение цели (точечная масса, например ноль)")
    return result


# ---------------------------------------------------------------- leakage

def quantile_codes(numbers: pd.Series) -> np.ndarray:
    """One bin per value for up to 20 distinct values, otherwise 20 quantile bins; missing/inf get bin -1."""
    numbers = numbers.where(np.isfinite(numbers))
    if numbers.nunique() <= 20:  # qcut would merge neighbouring values of binary/low-cardinality features
        return pd.factorize(numbers, sort=True)[0]
    try:
        bins = pd.qcut(numbers, q=20, duplicates="drop", labels=False)
    except ValueError:  # a single distinct value: qcut cannot build edges
        bins = pd.Series(0.0, index=numbers.index).where(numbers.notna())
    return bins.fillna(-1).to_numpy(int)


def bin_codes(values: pd.Series, numeric: bool) -> np.ndarray:
    return quantile_codes(to_number(values)) if numeric else pd.factorize(values.str.strip())[0]


def holdout_score(codes: np.ndarray, y: np.ndarray, fit: np.ndarray, hold: np.ndarray, classes: int) -> dict:
    """Fit a per-bin lookup on one half of train and score it on the other half."""
    levels, inverse, counts = np.unique(codes[fit], return_inverse=True, return_counts=True)
    position = pd.Index(levels).get_indexer(codes[hold])
    usable = position >= 0
    usable[usable] = counts[position[usable]] >= SCORE_MIN_LEVEL_ROWS
    y_fit, y_hold = y[fit], y[hold]
    if not classes:
        base = y_fit.mean()
        level_mean = np.bincount(inverse, weights=y_fit, minlength=len(levels)) / counts
        prediction = np.full(len(hold), base)
        prediction[usable] = level_mean[position[usable]]
        total = ((y_hold - base) ** 2).sum()
        return {"score": float(1 - ((y_hold - prediction) ** 2).sum() / total) if total > 0 else float("nan")}
    table = np.bincount(inverse * classes + y_fit, minlength=len(levels) * classes).reshape(len(levels), classes)
    base = int(np.bincount(y_fit, minlength=classes).argmax())
    prediction = np.full(len(hold), base)
    prediction[usable] = table.argmax(axis=1)[position[usable]]
    return {"score": macro_f1(y_hold, prediction, classes), "accuracy": float((prediction == y_hold).mean()),
            "baseline": macro_f1(y_hold, np.full(len(hold), base), classes)}


def analyze_duplicates(task: Task, features: list[str], findings: list) -> dict:
    train_key = pd.util.hash_pandas_object(task.train[features], index=False)
    test_key = pd.util.hash_pandas_object(task.test[features], index=False)
    labels = task.train[task.target]
    distinct = labels.groupby(train_key).nunique()
    sizes = train_key.map(train_key.value_counts())
    unambiguous = set(distinct.index[distinct == 1])
    result = {
        "train_rows_in_duplicate_groups": int((sizes > 1).sum()),
        "train_extra_duplicate_rows": int(train_key.duplicated().sum()),
        "conflicting_groups": int((distinct > 1).sum()),
        "test_rows_seen_in_train": float(test_key.isin(set(train_key)).mean()),
        "test_rows_with_unambiguous_train_match": float(test_key.isin(unambiguous).mean()),
        "test_rows_with_twin_in_test": float(test_key.duplicated(keep=False).mean()),
    }
    order = np.random.default_rng(SEED).permutation(len(train_key))
    cut = int(len(order) * 0.8)
    result["random_80_20_holdout_rows_with_twin_in_fit"] = float(
        train_key.iloc[order[cut:]].isin(set(train_key.iloc[order[:cut]])).mean())
    if result["train_extra_duplicate_rows"]:
        inflated = result["random_80_20_holdout_rows_with_twin_in_fit"]
        add(findings, "warning" if inflated >= 0.1 else "info", "дубликаты",
            f"{result['train_extra_duplicate_rows']} полных дублей признаков в train. При случайном 80/20 у {inflated:.1%} "
            f"валидационных строк есть точная копия в обучающей части, а у строк test копия в train только у "
            f"{result['test_rows_seen_in_train']:.1%}: локальная оценка будет завышена — дедуплицировать перед split или "
            "делить по группам одинаковых признаков")
    if result["conflicting_groups"]:
        add(findings, "warning", "дубликаты", f"{result['conflicting_groups']} групп одинаковых признаков с разными ответами: "
            "шум в цели, потолок качества ниже идеала")
    if result["test_rows_with_unambiguous_train_match"] >= 0.01:
        add(findings, "warning", "утечка", f"{result['test_rows_with_unambiguous_train_match']:.1%} строк test полностью совпадают "
            "по признакам со строкой train с однозначным ответом: ответ можно подставлять поиском")
    return result


def analyze_leakage(task: Task, features: list[str], profile: pd.DataFrame, cache: dict,
                    y: np.ndarray, classes: list[str], findings: list) -> dict:
    k = len(classes)
    valid = np.flatnonzero(np.isfinite(y)) if not k else np.arange(len(y))
    order = np.random.default_rng(SEED).permutation(valid)
    fit, hold = order[: len(order) // 2], order[len(order) // 2:]
    kinds = profile.set_index("feature")["kind"]
    candidates = [(name, "feature", bin_codes(task.train[name], kinds[name] == "numeric")) for name in features]
    ids = to_number(task.train[task.id_column])
    if ids.notna().all():
        candidates.append((task.id_column, "id", quantile_codes(ids)))
    candidates.append(("__row_order__", "row_order", quantile_codes(pd.Series(np.arange(len(y), dtype=float)))))
    rows = []
    if len(fit) >= 20 and len(hold) >= 20:
        for name, kind, codes in candidates:
            rows.append({"feature": name, "kind": kind, **holdout_score(codes, y, fit, hold, k)})
    scores = pd.DataFrame(rows)
    result = {"holdout_rows": int(len(hold)), "metric": "Macro F1" if k else "R2"}
    if scores.empty:
        return result
    scores["lift"] = scores["score"] - (scores["baseline"] if k else 0.0)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    result["single_feature_scores"] = scores.head(20)
    real = scores[scores["kind"] == "feature"]
    if k:
        strong_mask = real["lift"] >= LEAK_WARNING_F1_LIFT
        strong_text = f"Macro F1 выше константного ответа на {LEAK_WARNING_F1_LIFT}"
    else:
        strong_mask = real["score"] >= LEAK_WARNING_R2
        strong_text = f"R² ≥ {LEAK_WARNING_R2}"
    critical = real.loc[real["score"] >= LEAK_CRITICAL, "feature"].tolist()
    strong = real.loc[strong_mask & (real["score"] < LEAK_CRITICAL), "feature"].tolist()
    if critical:
        add(findings, "critical", "утечка", f"Один признак почти полностью предсказывает цель на отложенной половине train "
            f"({result['metric']} ≥ {LEAK_CRITICAL}): вероятная утечка", critical)
    if strong:
        add(findings, "warning", "утечка", f"Один признак сильно предсказывает цель ({strong_text}): "
            "проверить смысл признака и доступность его в момент прогноза", strong)
    threshold = ORDER_WARNING_F1_LIFT if k else ORDER_WARNING_R2
    for _, row in scores[scores["kind"] != "feature"].iterrows():
        if row["lift"] >= threshold:
            what = "ID" if row["kind"] == "id" else "порядок строк"
            add(findings, "warning", "утечка", f"{what.capitalize()} несёт информацию о цели ({result['metric']} {row['score']:.2f}): "
                "в test это может не работать, не использовать как признак; валидацию перемешивать")
    hinted = [name for name in features if NAME_HINT.search(name)]
    if hinted:
        add(findings, "info", "утечка", "Названия признаков похожи на целевые/результирующие — проверить по словарю", hinted)
    result["duplicates"] = analyze_duplicates(task, features, findings)
    return result


def target_correlations(cache: dict, y: np.ndarray) -> pd.DataFrame:
    y_series = pd.Series(y)
    y_rank = y_series.rank()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"feature": list(cache),
                             "corr_pearson": [column.corr(y_series) for column in cache.values()],
                             "corr_spearman": [column.rank().corr(y_rank) for column in cache.values()]})


# ---------------------------------------------------------------- orchestration and output

def analyze(task_dir: Path) -> tuple[dict, pd.DataFrame]:
    task = Task(task_dir)
    findings: list = []
    features, schema = audit_schema(task, findings)
    cache: dict[str, pd.Series] = {}
    profile = profile_features(task, features, cache)
    duplicates = duplicate_columns(cache, task, profile)
    pairs = correlated_pairs(cache)
    feature_findings(profile, duplicates, pairs, findings)
    y, classes = encode_target(task)
    report = {"schema": schema, "ids": analyze_ids(task, findings),
              "target": analyze_target(task, y, classes, findings)}
    report["leakage"] = analyze_leakage(task, features, profile, cache, y, classes, findings)
    if not task.classification and cache:
        correlations = target_correlations(cache, y)
        profile = profile.drop(columns=["corr_pearson", "corr_spearman"]).merge(correlations, on="feature", how="left")
        strongest = correlations.assign(abs=correlations["corr_spearman"].abs()).sort_values("abs", ascending=False)
        report["leakage"]["target_correlations"] = strongest.drop(columns="abs").head(20)
    report["correlated_pairs"] = pairs
    report["findings"] = sorted(findings, key=lambda item: LEVEL_ORDER[item["level"]])
    return report, profile


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={fmt(item)}" for key, item in value.items())
    return str(value)


def table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in columns) + " |" for _, row in frame.iterrows()]
    return lines


def render_markdown(name: str, report: dict, profile: pd.DataFrame) -> str:
    schema, target, leakage = report["schema"], report["target"], report["leakage"]
    lines = [f"# Анализ данных: {name}", "",
             f"- Тип задачи: **{schema['task_type']}**, цель `{schema['target']}`, ID `{schema['id_column']}`",
             f"- train: {schema['train_shape'][0]} × {schema['train_shape'][1]}, test: {schema['test_shape'][0]} × {schema['test_shape'][1]}, "
             f"признаков: {schema['n_features']} ({fmt(schema['declared_kinds'])})", "", "## Найденные проблемы", ""]
    for level in LEVEL_ORDER:
        items = [item for item in report["findings"] if item["level"] == level]
        if items:
            lines += [f"### {LEVEL_TITLES[level]}", ""]
            for item in items:
                shown = ", ".join(f"`{c}`" for c in item["columns"])
                more = f" … (+{item['n_columns'] - len(item['columns'])})" if item["n_columns"] > len(item["columns"]) else ""
                lines.append(f"- **[{item['area']}]** {item['message']}" + (f": {shown}{more}" if shown else ""))
            lines.append("")
    lines += ["## Целевая переменная", ""]
    if schema["task_type"] == "classification":
        counts = pd.DataFrame({"класс": list(target["counts"]), "строк": list(target["counts"].values()),
                               "доля": list(target["shares"].values())})
        lines += table(counts.head(30), ["класс", "строк", "доля"])
        lines += ["", f"Классов: {target['n_classes']}. Macro F1 константного ответа ≈ {fmt(target['majority_macro_f1'])}, "
                  f"случайного ≈ {fmt(target['uniform_random_macro_f1'])}."]
    else:
        lines += [f"- mean {fmt(target['mean'])}, std {fmt(target['std'])} (RMSE константного ответа), min {fmt(target['min'])}, max {fmt(target['max'])}",
                  f"- квантили: {fmt(target['quantiles'])}", f"- skew {fmt(target['skew'])}, kurtosis {fmt(target['kurtosis'])}",
                  f"- доля квадрата отклонений в топ-строках: {fmt(target['squared_error_share_of_top_rows'])}"]
    lines += ["", "## Признаки: пропуски", ""]
    lines += table(profile.sort_values("missing_train", ascending=False).head(15), ["feature", "missing_train", "missing_test"])
    lines += ["", "## Признаки: выбросы (числовые)", ""]
    lines += table(profile.sort_values("outlier_3_iqr", ascending=False).head(15),
                   ["feature", "outlier_3_iqr", "robust_z_max", "skew", "min", "p99", "max"])
    lines += ["", "## Сдвиг train → test", ""]
    shifted = profile.assign(shift=profile["ks_train_test"].fillna(profile["tvd_train_test"])).sort_values("shift", ascending=False)
    lines += table(shifted.head(15), ["feature", "kind", "ks_train_test", "tvd_train_test", "unseen_test_share", "test_out_of_range"])
    lines += ["", f"## Утечки: предсказательная сила одного признака ({leakage.get('metric', '')} на отложенной половине train)", ""]
    if "single_feature_scores" in leakage:
        columns = ["feature", "kind", "score"] + (["baseline", "accuracy"] if schema["task_type"] == "classification" else [])
        lines += table(leakage["single_feature_scores"].head(15), columns)
    if "target_correlations" in leakage:
        lines += ["", "### Корреляция с целью", ""]
        lines += table(leakage["target_correlations"].head(15), ["feature", "corr_pearson", "corr_spearman"])
    lines += ["", f"Дубликаты и пересечение с test: {fmt(leakage.get('duplicates', {}))}", ""]
    return "\n".join(lines)


def write_reports(name: str, report: dict, profile: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    profile.to_csv(out_dir / "features.csv", index=False, encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(name, report, profile), encoding="utf-8")
    return out_dir


def print_summary(name: str, report: dict, out_dir: Path) -> None:
    schema = report["schema"]
    counts = {level: sum(item["level"] == level for item in report["findings"]) for level in LEVEL_ORDER}
    print(f"\n=== {name}: {schema['task_type']}, train {schema['train_shape']}, test {schema['test_shape']} ===")
    print("Находок: " + ", ".join(f"{LEVEL_TITLES[level]} {n}" for level, n in counts.items()))
    for item in report["findings"]:
        if item["level"] != "info":
            names = f" [{', '.join(item['columns'][:6])}{'…' if item['n_columns'] > 6 else ''}]" if item["columns"] else ""
            print(f"  {LEVEL_TITLES[item['level']]:<8} ({item['area']}) {item['message']}{names}")
    print(f"Отчёт: {out_dir / 'report.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, help="Where to write reports (default: <data-dir>/analysis)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic output on Windows consoles and pipes
    data_dir = args.data_dir.resolve()
    out_root = (args.out_dir or data_dir / "analysis").resolve()
    try:
        for name in TASKS if args.task == "all" else [args.task]:
            report, profile = analyze(data_dir / name)
            print_summary(name, report, write_reports(name, report, profile, out_root / name))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Data analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
