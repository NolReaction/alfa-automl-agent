"""Explicit, reproducible holdouts for the fixed local training recipes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_validation(value=None) -> dict:
    """Accept only a fixed strategy and, for time, one declared column name."""
    if value is None:
        return {"strategy": "random"}
    if not isinstance(value, dict):
        raise ValueError("Настройки проверки должны быть объектом.")
    strategy = value.get("strategy")
    if not isinstance(strategy, str) or strategy not in {"random", "time", "grouped"}:
        raise ValueError("Выберите проверку: random, time или grouped.")
    expected = {"strategy", "time_column"} if strategy == "time" else {"strategy"}
    if set(value) != expected:
        raise ValueError("Неизвестные или отсутствующие параметры проверки.")
    result = {"strategy": strategy}
    if strategy == "time":
        column = value["time_column"]
        if not isinstance(column, str) or not column.strip() or len(column) > 500:
            raise ValueError("Для проверки по времени выберите столбец периода.")
        result["time_column"] = column
    return result


def split_rows(train, y, features, classification, validation=None):
    """Return one holdout, never a cross-validation average or a test-set score.

    Grouping uses the original feature values before preprocessing; IDs and
    targets are absent from ``features``. A conflicting target therefore does
    not let copies of the same input row cross the group boundary.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    from offline_baseline import SEED, VALIDATION_FRACTION

    options = normalize_validation(validation)
    strategy = options["strategy"]
    rows = np.arange(len(train))
    if len(train) < 2 or len(y) != len(train):
        raise ValueError("Для проверки нужны как минимум две строки с ответами.")
    if not features or any(name not in train.columns for name in features):
        raise ValueError("Не найдены признаки для проверки модели.")
    extra = {}
    if strategy == "random":
        fit_rows, validation_rows = train_test_split(
            rows, test_size=VALIDATION_FRACTION, random_state=SEED,
            shuffle=True, stratify=y if classification else None,
        )
        fraction = VALIDATION_FRACTION
        stratified, shuffle = bool(classification), True
    elif strategy == "time":
        if classification:
            raise ValueError("Проверка по времени доступна для рецептов регрессии.")
        column = options["time_column"]
        if column not in features:
            raise ValueError("Столбец периода должен быть признаком, а не ID или целью.")
        periods = pd.to_numeric(train[column], errors="coerce")
        if not np.isfinite(periods.to_numpy(dtype=float)).all():
            raise ValueError("Столбец периода должен содержать только числа без пропусков и бесконечностей.")
        latest = periods.max()
        fit_rows = rows[(periods < latest).to_numpy()]
        validation_rows = rows[(periods == latest).to_numpy()]
        if not len(fit_rows) or not len(validation_rows):
            raise ValueError("Для проверки по времени нужны хотя бы два разных периода.")
        extra = {
            "time_column": column,
            "train_period_min": str(periods.iloc[fit_rows].min()),
            "train_period_max": str(periods.iloc[fit_rows].max()),
            "validation_period": str(latest),
        }
        fraction = len(validation_rows) / len(train)
        stratified, shuffle = False, False
    else:
        if not classification:
            raise ValueError("Проверка по группам дублей доступна для рецептов классификации.")
        # The hash is calculated over all raw feature columns, without the
        # DataFrame index, and is independent of Python's randomized hash().
        groups = pd.util.hash_pandas_object(train.loc[:, features], index=False).to_numpy()
        group_count = len(np.unique(groups))
        labels = np.asarray(y)
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2 or group_count < 2:
            raise ValueError("Для групповой проверки нужны несколько классов и групп признаков.")
        if any(len(np.unique(groups[labels == label])) < 2 for label in classes):
            raise ValueError("Групповая проверка невозможна: один из классов встречается только в одной группе признаков.")
        chosen = None
        maximum = min(5, group_count, int(counts.min()))
        for n_splits in range(maximum, 1, -1):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for fold, (fitting, held_out) in enumerate(splitter.split(rows, labels, groups)):
                if (len(np.unique(labels[fitting])) == len(classes)
                        and len(np.unique(labels[held_out])) == len(classes)):
                    chosen = fitting, held_out, n_splits, fold
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError("Не удалось разделить группы так, чтобы все классы были и в обучении, и в проверке.")
        fit_rows, validation_rows, n_splits, fold = chosen
        fit_groups = np.unique(groups[fit_rows])
        held_out_groups = np.unique(groups[validation_rows])
        overlap = len(np.intersect1d(fit_groups, held_out_groups))
        if overlap:
            raise ValueError("Группы признаков пересеклись между обучением и проверкой.")
        extra = {
            "group_count": group_count, "train_groups": len(fit_groups),
            "validation_groups": len(held_out_groups), "group_overlap": overlap,
            "n_splits": n_splits, "fold": fold,
        }
        fraction = len(validation_rows) / len(train)
        stratified, shuffle = True, True
    metadata = {
        "strategy": strategy, "seed": SEED,
        "train_fraction": 1 - fraction, "validation_fraction": fraction,
        "shuffle": shuffle, "stratified": stratified,
        "train_rows": len(fit_rows), "validation_rows": len(validation_rows),
        **extra,
    }
    return fit_rows, validation_rows, metadata
