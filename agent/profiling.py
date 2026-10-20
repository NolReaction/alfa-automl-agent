"""Automatic data profile of a task, as compact text for the agent. Owner: Murad.

Needs only numpy and pandas. The profile reports sizes, types, missing values,
the target, outliers, train/test shift, duplicates and possible leakage, then
turns them into short findings and a validation suggestion.

    python -m agent.profiling data/regression/task.md [--json profile.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd


SEED = 42
MAX_CHARS = 4000
TIME_BUDGET = 60.0  # seconds; the optional duplicate and leakage checks are skipped after it
LEVELS = {"high": 0, "medium": 1, "low": 2}
PLACEHOLDERS = {"na", "n/a", "nan", "null", "none", "?", "unknown", "undefined", "missing", "#n/a"}
NAME_HINT = re.compile(r"target|label|leak|answer|outcome|verdict|predict|future|result", re.IGNORECASE)
LOW_CARDINALITY = 50           # columns with at most this many values can encode a period or a group
RARE_CLASS_ROWS = 50
SCORE_SAMPLE_ROWS = 100_000
SCORE_MAX_COLUMNS = 300
SCORE_MIN_LEVEL_ROWS = 2
LEAK_HIGH = 0.9                # single-feature R2 / Macro F1 on a holdout this high: probable leak
LEAK_MEDIUM_R2, LEAK_MEDIUM_F1_LIFT = 0.5, 0.3
ORDER_R2, ORDER_F1_LIFT = 0.05, 0.1
SHIFT_STRONG, SHIFT_NOTICEABLE = 0.3, 0.1
MAX_FEATURE_RECORDS = 1000
TABLE_COLUMNS = [
    "feature", "kind", "declared", "missing_train", "missing_test", "n_unique", "unparseable", "infinite", "min",
    "median", "max", "skew", "top_share", "outlier_3_iqr", "test_out_of_range", "ks", "unseen_test_share",
    "rare_levels", "whitespace_share", "placeholder_share", "tvd",
]


# ---------------------------------------------------------------- loading

def _read_csv(path: Path) -> pd.DataFrame:
    # Everything as text, exactly as written: IDs such as 001 and labels such as NA stay intact.
    if not path.is_file():
        raise ValueError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _find_data_dir(task_dir: Path) -> Path:
    samples = sorted(
        path for path in task_dir.rglob("sample_submission.csv")
        if path.is_file() and ".agent_runs" not in path.relative_to(task_dir).parts
        and not any(part.startswith(".offline-baseline-") for part in path.relative_to(task_dir).parts)
    )
    if len(samples) != 1:
        raise ValueError(f"Expected exactly one sample_submission.csv inside {task_dir}; found {len(samples)}")
    return samples[0].parent


def _number(column: pd.Series) -> pd.Series:
    """Text column to float; empty, non-numeric and infinite cells become NaN."""
    values = pd.to_numeric(column.str.strip(), errors="coerce")
    return values.where(np.isfinite(values))


class _Task:
    def __init__(self, task_path: Path):
        path = Path(task_path).resolve()
        if not path.exists():
            raise ValueError(f"Task path does not exist: {path}")
        self.data_dir = _find_data_dir(path if path.is_dir() else path.parent)
        self.train = _read_csv(self.data_dir / "train.csv")
        self.test = _read_csv(self.data_dir / "test.csv")
        self.sample = _read_csv(self.data_dir / "sample_submission.csv")
        dtypes, roles = self._dictionary()
        ids = [name for name, role in roles.items() if role == "id"]
        self.id_column = ids[0] if len(ids) == 1 else self.sample.columns[0]
        self.targets = [name for name in self.sample.columns if name != self.id_column]
        if not self.targets:
            raise ValueError("sample_submission.csv has no prediction column")
        self.target = self.targets[0]
        for name, frame in (("train.csv", self.train), ("test.csv", self.test)):
            if self.id_column not in frame:
                raise ValueError(f"{name}: no ID column {self.id_column!r}")
        if self.target not in self.train:
            raise ValueError(f"train.csv: no target column {self.target!r}")
        skipped = {self.id_column, *self.targets}
        train_features = [name for name in self.train.columns if name not in skipped]
        test_features = [name for name in self.test.columns if name != self.id_column]
        self.features = [name for name in train_features if name in test_features]
        self.only_train = [name for name in train_features if name not in test_features]
        self.only_test = [name for name in test_features if name not in train_features]
        self.dtypes = dtypes
        self.kinds = {name: self._kind(name) for name in self.features}
        self.classification = self._target_is_category()

    def _dictionary(self) -> tuple[dict, dict]:
        path = self.data_dir / "data_dictionary.csv"
        if not path.is_file():
            return {}, {}
        meta = _read_csv(path).apply(lambda column: column.str.strip())
        if not {"feature", "role", "dtype"}.issubset(meta.columns):
            return {}, {}
        meta = meta.drop_duplicates("feature").set_index("feature")
        return meta["dtype"].str.lower().to_dict(), meta["role"].str.lower().to_dict()

    def _kind(self, name: str) -> str:
        declared = self.dtypes.get(name)
        if declared in {"numeric", "binary"}:
            return "numeric"
        if declared == "category":
            return "category"
        filled = self.train[name].str.strip()
        filled = filled[filled != ""]
        return "numeric" if len(filled) and _number(filled).notna().mean() >= 0.95 else "category"

    def _target_is_category(self) -> bool:
        declared = self.dtypes.get(self.target)
        if declared in {"numeric", "category"}:
            return declared == "category"
        values = _number(self.train[self.target])
        return not (values.notna().mean() >= 0.95 and values.nunique() > 20)


def _finding(findings: list, level: str, area: str, text: str, columns=()) -> None:
    columns = list(columns)
    findings.append({"level": level, "area": area, "text": text,
                     "n_columns": len(columns), "columns": columns[:10]})


# ---------------------------------------------------------------- statistics

def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> float:
    matrix = np.bincount(y_true * classes + y_pred, minlength=classes * classes).reshape(classes, classes)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    present = denominator > 0
    return float((2 * np.diag(matrix)[present] / denominator[present]).mean())


def _numeric_row(train: pd.Series, test: pd.Series, numbers: pd.Series) -> dict:
    finite = numbers.dropna()
    test_finite = _number(test).dropna()
    raw = pd.to_numeric(train.str.strip(), errors="coerce")
    row = {"n_unique": int(finite.nunique()),
           "unparseable": float((raw.isna() & train.str.strip().ne("")).mean()),
           "infinite": float(np.isinf(raw).mean())}
    if finite.empty:
        return row
    q25, median, q75 = finite.quantile([0.25, 0.5, 0.75])
    iqr = q75 - q25
    row.update(
        min=float(finite.min()), median=float(median), max=float(finite.max()), skew=float(finite.skew()),
        top_share=float(finite.value_counts(normalize=True).iloc[0]),
        outlier_3_iqr=float(((finite < q25 - 3 * iqr) | (finite > q75 + 3 * iqr)).mean()) if iqr > 0 else 0.0,
        test_out_of_range=float(((test_finite < finite.min()) | (test_finite > finite.max())).mean())
        if len(test_finite) else float("nan"),
        ks=_ks(finite.to_numpy(float), test_finite.to_numpy(float)),
    )
    if row["n_unique"] <= LOW_CARDINALITY and len(test_finite):
        row["unseen_test_share"] = float((~test_finite.isin(finite.unique())).sum() / len(test))
    return row


def _category_row(train: pd.Series, test: pd.Series) -> dict:
    text, test_text = train.str.strip(), test.str.strip()
    counts = text[text != ""].value_counts()
    test_filled = test_text[test_text != ""]
    row = {"n_unique": int(counts.size),
           "rare_levels": int((counts < 5).sum()),
           "whitespace_share": float(((train != text) & (text != "")).mean()),
           "placeholder_share": float(text.str.lower().isin(PLACEHOLDERS).mean())}
    if counts.empty:
        return row
    shares = counts / counts.sum()
    test_shares = test_filled.value_counts(normalize=True)
    union = shares.index.union(test_shares.index)
    row.update(
        top_share=float(shares.iloc[0]),
        unseen_test_share=float((~test_filled.isin(counts.index)).sum() / len(test)),
        tvd=float(0.5 * (shares.reindex(union, fill_value=0) - test_shares.reindex(union, fill_value=0)).abs().sum()),
    )
    return row


def _feature_table(task: _Task) -> tuple[pd.DataFrame, dict]:
    rows, numbers = [], {}
    for name in task.features:
        train, test = task.train[name], task.test[name]
        row = {"feature": name, "kind": task.kinds[name], "declared": task.dtypes.get(name, ""),
               "missing_train": float(train.str.strip().eq("").mean()),
               "missing_test": float(test.str.strip().eq("").mean())}
        if task.kinds[name] == "numeric":
            numbers[name] = _number(train)
            row.update(_numeric_row(train, test, numbers[name]))
        else:
            row.update(_category_row(train, test))
        rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=TABLE_COLUMNS)
    for column in TABLE_COLUMNS[3:]:
        table[column] = pd.to_numeric(table[column], errors="coerce")  # all-empty columns arrive as object
    table["shift"] = table["ks"].fillna(table["tvd"])
    return table, numbers


def _duplicate_columns(task: _Task) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for name in task.features:
        digest = hashlib.md5(pd.util.hash_pandas_object(task.train[name], index=False).to_numpy().tobytes()).hexdigest()
        groups.setdefault(digest, []).append(name)
    return [names for names in groups.values() if len(names) > 1]


# ---------------------------------------------------------------- sections

def _id_section(task: _Task, findings: list) -> dict:
    train_ids, test_ids = task.train[task.id_column].str.strip(), task.test[task.id_column].str.strip()
    overlap = len(set(train_ids) & set(test_ids))
    result = {"train_unique": bool(train_ids.is_unique), "test_unique": bool(test_ids.is_unique), "overlap": overlap}
    counters = []
    for ids in (train_ids, test_ids):
        values = _number(ids)
        counters.append(bool(values.notna().all() and (values.to_numpy() == np.arange(len(values))).all()))
    result["row_counter"] = counters
    if not (result["train_unique"] and result["test_unique"]):
        _finding(findings, "high", "id", f"Duplicate `{task.id_column}` values inside train or test.")
    if overlap and any(counters):
        _finding(findings, "low", "id", f"`{task.id_column}` is a row counter (0..N-1) in each file, so {overlap} values repeat "
                 "between train and test. It carries no information: not a feature, not a merge key.")
    elif overlap:
        _finding(findings, "high", "id", f"{overlap} `{task.id_column}` values occur in both train and test.")
    return result


def _target_section(task: _Task, findings: list) -> dict:
    raw = task.train[task.target]
    result = {"kind": "classification" if task.classification else "regression", "name": task.target,
              "empty": int(raw.str.strip().eq("").sum())}
    if result["empty"]:
        _finding(findings, "high", "target", f"{result['empty']} train rows have an empty target.")
    if len(task.targets) > 1:
        _finding(findings, "low", "target", f"sample_submission.csv has several prediction columns: {task.targets}. "
                 f"This profile covers `{task.target}` only.")
    if task.classification:
        counts = raw[raw.str.strip() != ""].value_counts()
        shares = counts / counts.sum()
        k, top = len(counts), float(shares.iloc[0])
        ratio = float(counts.iloc[0] / counts.iloc[-1])
        result.update(n_classes=k, counts=counts.to_dict(), imbalance_ratio=ratio, majority_share=top,
                      majority_macro_f1=float(2 * top / (1 + top) / k),
                      rare_classes={str(label): int(n) for label, n in counts.items() if n < RARE_CLASS_ROWS})
        if k < 2:
            _finding(findings, "high", "target", "Fewer than two classes in the training target.")
        if ratio >= 10:
            _finding(findings, "high" if ratio >= 100 else "medium", "target",
                     f"Class imbalance {ratio:.0f}:1; the majority class is {top:.1%}. Always predicting it gives Macro F1 "
                     f"{result['majority_macro_f1']:.3f}. Use stratified validation, class weights or thresholds, "
                     "and report F1 per class.")
        if result["rare_classes"]:
            smallest = int(counts.iloc[-1])
            _finding(findings, "high" if smallest < 5 else "medium", "target",
                     f"Rare classes (<{RARE_CLASS_ROWS} rows, smallest has {smallest}): Macro F1 is noisy on them.",
                     result["rare_classes"])
        return result
    values = _number(raw).dropna()
    bad = int(len(raw) - result["empty"] - len(values))
    if bad:
        _finding(findings, "high", "target", f"{bad} train target values are not finite numbers.")
    if values.empty:
        return result
    deviation = np.sort(((values - values.mean()) ** 2).to_numpy())[::-1]
    tail = float(deviation[:max(1, len(deviation) // 100)].sum() / deviation.sum()) if deviation.sum() > 0 else 0.0
    quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    result.update(n=len(values), mean=float(values.mean()), std=float(values.std(ddof=0)), min=float(values.min()),
                  p01=float(quantiles[0.01]), p25=float(quantiles[0.25]), p50=float(quantiles[0.5]),
                  p75=float(quantiles[0.75]), p99=float(quantiles[0.99]), max=float(values.max()),
                  skew=float(values.skew()), zero_share=float((values == 0).mean()), top_1pct_variance_share=tail)
    if tail >= 0.2:
        _finding(findings, "medium", "target", f"Heavy tail: the top 1% of rows carry {tail:.0%} of the target's squared "
                 "deviation, so RMSE is dominated by a few rows. Do not clip the target blindly; a log transform "
                 "must be judged by RMSE on the original scale.")
    elif abs(result["skew"]) >= 2:
        _finding(findings, "low", "target", f"Target is skewed (skew {result['skew']:.1f}).")
    return result


def _feature_findings(table: pd.DataFrame, task: _Task, findings: list) -> list[str]:
    """Report problems found in the per-feature table. Returns columns that mark a train/test split."""
    def pick(mask: pd.Series, by: str = "feature", ascending: bool = False) -> list[str]:
        return table[mask.fillna(False)].sort_values(by, ascending=ascending)["feature"].tolist()

    numeric, category = table["kind"] == "numeric", table["kind"] == "category"
    if task.only_train or task.only_test:
        _finding(findings, "high", "schema", "Feature columns differ between train and test (listed: only in train, "
                 "then only in test).", task.only_train + task.only_test)
    empty = pick(table["missing_train"] >= 0.5, "missing_train")
    drift = pick((table["missing_test"] - table["missing_train"]).abs() >= 0.05, "missing_test")
    some = pick(table["missing_train"] > 0, "missing_train")
    if empty:
        _finding(findings, "medium", "missing", "Columns empty in at least half of the train rows.", empty)
    if drift:
        _finding(findings, "medium", "missing", "Share of missing values differs between train and test by 5 points or more. "
                 "Median-filling can shift predictions; keep NaN where the model allows, or add a missingness flag.", drift)
    if some:
        _finding(findings, "low", "missing", f"{len(some)} of {len(table)} features have empty cells.", some)
    for mask, level, text in (
        (numeric & (table["unparseable"] > 0), "medium", "Numeric columns with non-numeric cells."),
        (category & (table["placeholder_share"] > 0), "low", "Categories that look like missing markers (NA, null, ?, unknown)."),
        (category & (table["whitespace_share"] > 0), "low", "Categories with leading or trailing spaces."),
        (category & (table["unseen_test_share"] >= 0.01), "medium", "At least 1% of test rows have a category never seen in train."),
        (numeric & (table["test_out_of_range"] >= 0.01), "low", "At least 1% of test values lie outside the train range."),
        (category & (table["n_unique"] > 100), "low", "High-cardinality categories (over 100 values)."),
        ((table["declared"] == "binary") & (table["n_unique"] > 2), "low", "Declared binary but more than two distinct values."),
        (table["n_unique"] <= 1, "low", "Constant or empty columns."),
    ):
        names = pick(mask)
        if names:
            _finding(findings, level, "types", text, names)
    heavy = pick(numeric & ((table["outlier_3_iqr"] >= 0.01) | (table["skew"].abs() >= 3)), "outlier_3_iqr")
    if heavy:
        _finding(findings, "low", "outliers", "Heavy tails (outliers beyond 3 IQR or |skew| >= 3). Trees do not mind; "
                 "linear models and neural nets need a log or rank transform.", heavy)
    for names in _duplicate_columns(task):
        _finding(findings, "low", "types", "Identical columns.", names)
    split = pick(table["unseen_test_share"].ge(0.9) & table["n_unique"].between(2, LOW_CARDINALITY), "unseen_test_share")
    outside = pick(numeric & (table["n_unique"] > LOW_CARDINALITY) & table["test_out_of_range"].ge(0.9), "test_out_of_range")
    if split or outside:
        _finding(findings, "high", "shift", "Test values of these columns are (almost) all absent from train: the test set "
                 "is a separate period or group. A random validation split will overstate quality, and the column "
                 "itself cannot be learned as a feature.", split + outside)
    strong = pick(table["shift"] >= SHIFT_STRONG, "shift")
    noticeable = pick((table["shift"] >= SHIFT_NOTICEABLE) & (table["shift"] < SHIFT_STRONG), "shift")
    if strong:
        _finding(findings, "medium", "shift", f"{len(strong)} features differ strongly between train and test "
                 f"(KS or TVD >= {SHIFT_STRONG}); they may not transfer.", strong)
    if noticeable:
        _finding(findings, "low", "shift", f"{len(noticeable)} features differ noticeably between train and test "
                 f"(KS or TVD >= {SHIFT_NOTICEABLE}).", noticeable)
    return split + outside


def _row_keys(task: _Task) -> tuple[pd.Series, pd.Series]:
    return (pd.util.hash_pandas_object(task.train[task.features], index=False),
            pd.util.hash_pandas_object(task.test[task.features], index=False))


def _duplicates_section(task: _Task, findings: list) -> dict:
    train_key, test_key = _row_keys(task)
    labels = task.train[task.target]
    if not task.classification:
        labels = _number(labels)
    differing = labels.groupby(train_key).nunique()
    sizes = train_key.map(train_key.value_counts())
    unambiguous = set(differing.index[differing == 1])
    order = np.random.default_rng(SEED).permutation(len(train_key))
    cut = int(len(order) * 0.8)
    result = {
        "train_extra_copies": int(train_key.duplicated().sum()),
        "train_rows_with_a_twin": float((sizes > 1).mean()),
        "test_rows_seen_in_train": float(test_key.isin(set(train_key)).mean()),
        "test_rows_with_unambiguous_train_match": float(test_key.isin(unambiguous).mean()),
        "test_rows_with_twin_in_test": float(test_key.duplicated(keep=False).mean()),
        "random_80_20_holdout_rows_with_twin_in_fit": float(
            train_key.iloc[order[cut:]].isin(set(train_key.iloc[order[:cut]])).mean()),
        "rows_in_groups_with_different_targets": float((sizes[train_key.map(differing) > 1]).count() / len(train_key)),
    }
    twin, seen = result["random_80_20_holdout_rows_with_twin_in_fit"], result["test_rows_seen_in_train"]
    if result["train_extra_copies"] and twin >= 0.1:
        _finding(findings, "medium", "duplicates", f"{result['train_extra_copies']} repeated feature rows in train. With a random 80/20 "
                 f"split {twin:.1%} of validation rows have an exact copy in the fitting part, but only {seen:.1%} of "
                 "test rows have one in train. Local scores will be too high: split by groups of identical "
                 f"feature rows (hash of all feature columns). {result['test_rows_with_twin_in_test']:.1%} of test rows "
                 "also repeat inside test, so judge before dropping repeats from training.")
    elif result["train_extra_copies"]:
        _finding(findings, "low", "duplicates", f"{result['train_extra_copies']} repeated feature rows in train.")
    if result["rows_in_groups_with_different_targets"] >= 0.05:
        _finding(findings, "medium", "duplicates", f"{result['rows_in_groups_with_different_targets']:.1%} of train rows share "
                 "all features with rows that have a different target: label noise caps the achievable score.")
    if result["test_rows_with_unambiguous_train_match"] >= 0.01:
        _finding(findings, "low", "duplicates", f"{result['test_rows_with_unambiguous_train_match']:.1%} of test rows match a train "
                 "row on all features with a single known target.")
    return result


def _codes(values: pd.Series, numbers: pd.Series | None) -> np.ndarray:
    if numbers is None:
        return pd.factorize(values.str.strip())[0]
    if numbers.nunique() <= 20:  # qcut would merge neighbouring values of low-cardinality columns
        return pd.factorize(numbers, sort=True)[0]
    try:
        bins = pd.qcut(numbers, q=20, duplicates="drop", labels=False)
    except ValueError:
        bins = pd.Series(0.0, index=numbers.index).where(numbers.notna())
    return bins.fillna(-1).to_numpy(int)


def _holdout_score(codes: np.ndarray, y: np.ndarray, fit: np.ndarray, hold: np.ndarray, classes: int) -> dict:
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
        return {"score": float(1 - ((y_hold - prediction) ** 2).sum() / total) if total > 0 else 0.0, "baseline": 0.0}
    table = np.bincount(inverse * classes + y_fit, minlength=len(levels) * classes).reshape(len(levels), classes)
    base = int(np.bincount(y_fit, minlength=classes).argmax())
    prediction = np.full(len(hold), base)
    prediction[usable] = table.argmax(axis=1)[position[usable]]
    return {"score": _macro_f1(y_hold, prediction, classes),
            "baseline": _macro_f1(y_hold, np.full(len(hold), base), classes)}


def _leakage_section(task: _Task, numbers: dict, findings: list) -> dict:
    if task.classification:
        codes, classes = pd.factorize(task.train[task.target], sort=True)
        y, k = codes.astype(int), len(classes)
    else:
        y, k = _number(task.train[task.target]).to_numpy(float), 0
    rows = np.flatnonzero(np.isfinite(y)) if not k else np.arange(len(y))
    rng = np.random.default_rng(SEED)
    if len(rows) > SCORE_SAMPLE_ROWS:
        rows = rng.choice(rows, SCORE_SAMPLE_ROWS, replace=False)
    rows = rng.permutation(rows)
    fit, hold = rows[: len(rows) // 2], rows[len(rows) // 2:]
    result = {"metric": "Macro F1" if k else "R2", "holdout_rows": int(len(hold))}
    if len(fit) < 20 or len(hold) < 20:
        return result
    candidates = [(name, "feature", _codes(task.train[name], numbers.get(name)))
                  for name in task.features[:SCORE_MAX_COLUMNS]]
    ids = _number(task.train[task.id_column])
    if ids.notna().all():
        candidates.append((task.id_column, "id", _codes(task.train[task.id_column], ids)))
    candidates.append(("__row_order__", "row_order", _codes(task.train[task.id_column], pd.Series(np.arange(len(y), dtype=float)))))
    scores = pd.DataFrame([{"feature": name, "kind": kind, **_holdout_score(codes, y, fit, hold, k)}
                           for name, kind, codes in candidates])
    if not k:
        # Binned means underrate a monotone relation when the target has a heavy tail: add squared rank correlation.
        y_rank = pd.Series(y[rows]).rank()
        with np.errstate(invalid="ignore", divide="ignore"):  # a constant column has no correlation
            rank_r2 = {name: pd.Series(numbers[name].to_numpy()[rows]).rank().corr(y_rank) ** 2
                       for name in task.features[:SCORE_MAX_COLUMNS] if name in numbers}
        scores["score"] = np.fmax(scores["score"], scores["feature"].map(rank_r2).astype(float))
    scores["lift"] = scores["score"] - scores["baseline"]
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    result["single_feature_scores"] = scores.head(10)
    real = scores[scores["kind"] == "feature"]
    if k:
        medium = real["lift"] >= LEAK_MEDIUM_F1_LIFT
        medium_text = f"Macro F1 more than {LEAK_MEDIUM_F1_LIFT} above the constant answer"
    else:
        medium = real["score"] >= LEAK_MEDIUM_R2
        medium_text = f"R2 >= {LEAK_MEDIUM_R2}"
    high = real.loc[real["score"] >= LEAK_HIGH, "feature"].tolist()
    strong = real.loc[medium & (real["score"] < LEAK_HIGH), "feature"].tolist()
    if high:
        _finding(findings, "high", "leakage", f"One column alone almost reproduces the target on held-out train rows "
                 f"({result['metric']} >= {LEAK_HIGH}): probable leak. Check whether it exists at prediction time.", high)
    if strong:
        _finding(findings, "medium", "leakage", f"One column alone predicts the target strongly ({medium_text} on held-out "
                 "train rows): check its meaning and availability at prediction time.", strong)
    threshold = ORDER_F1_LIFT if k else ORDER_R2
    for _, row in scores[scores["kind"] != "feature"].iterrows():
        if row["lift"] >= threshold:
            what = "ID" if row["kind"] == "id" else "Row order"
            _finding(findings, "medium", "leakage", f"{what} carries information about the target ({result['metric']} "
                     f"{row['score']:.2f}). The test file may not follow the same order: do not use it as a feature "
                     "and shuffle validation rows.")
    hinted = [name for name in task.features if NAME_HINT.search(name)]
    if hinted:
        _finding(findings, "low", "leakage", "Column names that sound like outcomes: check the data dictionary.", hinted)
    return result


# ---------------------------------------------------------------- public API

def _validation_advice(findings: list, split_columns: list[str], classification: bool) -> str:
    if split_columns:
        names = ", ".join(f"`{name}`" for name in split_columns[:3])
        return (f"Test is separated from train by {names}. Validate the same way: fit on the lower values, "
                "check on the highest train values (a forward split), and do not use these columns as features.")
    if any(item["area"] == "duplicates" and item["level"] == "medium" and "split by groups" in item["text"] for item in findings):
        return ("Split by groups of identical feature rows (hash of all features) so that copies do not fall on both sides"
                + ("; stratify by class where possible." if classification else "."))
    return "A random split" + (" stratified by class" if classification else "") + " looks consistent with the test set."


def profile_task(task_path, *, time_budget: float = TIME_BUDGET) -> dict:
    """Profile the task located at `task_path` (task.md or its directory)."""
    started = time.monotonic()
    task = _Task(Path(task_path))
    findings: list[dict] = []
    table, numbers = _feature_table(task)
    if not task.sample[task.id_column].tolist() == task.test[task.id_column].tolist():
        _finding(findings, "high", "schema", "sample_submission.csv IDs differ from test.csv in value or order.")
    profile = {"task_dir": str(task.data_dir), "id_column": task.id_column, "target_column": task.target,
               "rows": {"train": len(task.train), "test": len(task.test)},
               "n_features": len(task.features), "feature_kinds": table["kind"].value_counts().to_dict()}
    profile["id"] = _id_section(task, findings)
    profile["target"] = _target_section(task, findings)
    split_columns = _feature_findings(table, task, findings)
    skipped = []
    for name, section in (("duplicates", lambda: _duplicates_section(task, findings)),
                          ("leakage", lambda: _leakage_section(task, numbers, findings))):
        if time.monotonic() - started > time_budget:
            skipped.append(name)
            continue
        profile[name] = section()
    if skipped:
        _finding(findings, "low", "profile", f"Skipped for time: {', '.join(skipped)}.")
    profile["validation_advice"] = _validation_advice(findings, split_columns, task.classification)
    profile["findings"] = sorted(findings, key=lambda item: LEVELS[item["level"]])
    profile["feature_table"] = table.head(MAX_FEATURE_RECORDS)
    profile["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return profile


def _g(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}g}"


def _blocks(profile: dict) -> tuple[list[str], list[list[str]]]:
    """Text split into a mandatory head and optional blocks in priority order."""
    rows, target = profile["rows"], profile["target"]
    kinds = ", ".join(f"{count} {name}" for name, count in profile["feature_kinds"].items())
    head = [f"DATA PROFILE (computed from the task files in {profile['elapsed_seconds']}s; local facts, not platform feedback)",
            f"Task: {target['kind']}, target `{target['name']}`, id `{profile['id_column']}`. "
            f"Rows: train {rows['train']}, test {rows['test']}. Features: {profile['n_features']} ({kinds}).",
            "Suggested validation: " + profile["validation_advice"]]
    if target["kind"] == "regression" and "mean" in target:
        head.append(f"Target: mean {_g(target['mean'])}, std {_g(target['std'])} (RMSE of a constant prediction), "
                    f"min {_g(target['min'])}, p50 {_g(target['p50'])}, p99 {_g(target['p99'])}, max {_g(target['max'])}.")
    elif target["kind"] == "classification" and "counts" in target:
        counts = ", ".join(f"{label} {n}" for label, n in list(target["counts"].items())[:12])
        head.append(f"Classes ({target['n_classes']}): {counts}.")
    findings = [[f"- [{item['level'].upper()}] {item['text']}" + (
        f" ({', '.join(f'`{c}`' for c in item['columns'][:6])}{', ...' if item['n_columns'] > 6 else ''})" if item["columns"] else "")]
        for item in profile["findings"]]
    return head, findings


def format_profile(profile: dict, max_chars: int = MAX_CHARS) -> str:
    """Render the profile as text of at most `max_chars`; findings are kept in priority order."""
    head, findings = _blocks(profile)
    lines = list(head)
    size = sum(len(line) + 1 for line in lines)
    if findings:
        lines.append("Findings, most important first:")
        size += len(lines[-1]) + 1
    shown = 0
    for block in findings:
        cost = sum(len(line) + 1 for line in block)
        if size + cost > max_chars - 60:
            break
        lines.extend(block)
        size += cost
        shown += 1
    if shown < len(findings):
        lines.append(f"(+{len(findings) - shown} lower-priority findings omitted; run `python -m agent.profiling <task.md>` for all)")
    elif not findings:
        lines.append("No problems found.")
    return "\n".join(lines)[:max_chars]


def profile_text(task_path, max_chars: int = MAX_CHARS) -> str:
    """Profile text for the agent prompt. Never raises: profiling must not stop the research loop."""
    try:
        return format_profile(profile_task(task_path), max_chars)
    except Exception as error:  # noqa: BLE001 - any failure here is reported, not propagated
        return f"DATA PROFILE unavailable: {type(error).__name__}: {error}"[:max_chars]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict("records"))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="Path to task.md or the directory containing this task's data")
    parser.add_argument("--json", type=Path, help="Also write the full profile to this JSON file")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        profile = profile_task(args.task)
        print(format_profile(profile, args.max_chars))
        if args.json:
            args.json.write_text(json.dumps(_jsonable(profile), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
        return 0
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
