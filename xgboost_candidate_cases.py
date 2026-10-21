"""Fixed synthetic cases for the native XGBoost protocol smoke.

These cases check mechanics: the native subprocess protocol, class order,
missing values, unseen categories and the time/group boundaries. They are
written by the candidate's author and are not an independent quality audit.

Generation and writing are separate. Only the controller receives holdout
answers; test rows have no answers at all. The writer puts features and fit
targets into child inputs, never the holdout answers.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


VERSION = 1
SEED = 4027
ID_COLUMN = "row_id"
TARGET_COLUMN = "target"
NUMERIC = ("x1", "x2", "period")
CATEGORICAL = ("kind", "city")
FEATURES = NUMERIC + CATEGORICAL
KINDS = ("NA", "0007", "7", "", "class c", " Ω ")
KNOWN_CITIES = 150
NEW_CITIES = 20
HOLDOUT_FRACTION = 0.2
CASES = {
    "regression_time_rmse": {
        "metric": "RMSE", "classes": [], "positive_class": None,
        "validation": {"strategy": "time", "time_column": "period"},
        "configuration": "depth6_v1", "train_rows": 2400, "test_rows": 400,
        "duplicate_pairs": False, "host_holdout": "latest_period",
        "target": "2*x1 + 1.5*sin(x2) + kind effect + city effect + 0.3*period + N(0, 0.5)",
    },
    "binary_random_auc": {
        "metric": "ROC AUC", "classes": ["NA", "01"], "positive_class": "01",
        "validation": {"strategy": "random"},
        "configuration": "depth6_v1", "train_rows": 2400, "test_rows": 400,
        "duplicate_pairs": False, "host_holdout": "seeded_row_fraction",
        "target": "Bernoulli(sigmoid(latent score)) gives 01, otherwise NA",
    },
    "multiclass_grouped_macro_f1": {
        "metric": "Macro F1", "classes": ["01", "NA", "class c"], "positive_class": None,
        "validation": {"strategy": "grouped"},
        "configuration": "depth4_regularized_v1", "train_rows": 2400, "test_rows": 400,
        "duplicate_pairs": True, "host_holdout": "seeded_pair_fraction",
        "target": "latent score tertiles; 5% label replacement; identical pair copies",
    },
    "regression_random_mae": {
        "metric": "MAE", "classes": [], "positive_class": None,
        "validation": {"strategy": "random"},
        "configuration": "depth4_regularized_v1", "train_rows": 2400, "test_rows": 400,
        "duplicate_pairs": False, "host_holdout": "seeded_row_fraction",
        "target": "same latent score + Laplace(0, 0.7) noise",
    },
}


def case_definition(name):
    return deepcopy(CASES[name])


def _latent(random, rows, *, test, time_trend):
    x1 = random.normal(size=rows)
    x2 = random.uniform(-3, 3, size=rows)
    kind = random.choice(np.asarray(KINDS, dtype=object), size=rows)
    kind_effect = {value: effect for value, effect in zip(KINDS, (0.8, -0.6, 0.4, 0.0, -1.0, 1.2))}
    city_effects = np.linspace(-1, 1, KNOWN_CITIES)
    codes = random.integers(0, KNOWN_CITIES, size=rows)
    city = np.asarray([f"c{code:03d}" for code in codes], dtype=object)
    effect = city_effects[codes]
    if test:
        new = random.uniform(size=rows) < 0.1
        city[new] = [f"new_{code:03d}" for code in random.integers(0, NEW_CITIES, size=int(new.sum()))]
        effect = np.where(new, 0.0, effect)
    score = 2 * x1 + 1.5 * np.sin(x2) + np.asarray([kind_effect[value] for value in kind]) + effect
    if time_trend is not None:
        score = score + 0.3 * time_trend
    observed = x1.copy()
    observed[random.uniform(size=rows) < 0.05] = np.nan
    return {"x1": observed, "x2": x2, "kind": kind, "city": city}, score


def _frame(prefix, columns, period):
    rows = len(period)
    frame = pd.DataFrame({ID_COLUMN: [f"{prefix}-{index:05d}" for index in range(rows)]})
    frame["x1"] = ["" if np.isnan(value) else f"{value:.6f}" for value in columns["x1"]]
    frame["x2"] = [f"{value:.6f}" for value in columns["x2"]]
    frame["period"] = [str(int(value)) for value in period]
    frame["kind"] = list(columns["kind"])
    frame["city"] = list(columns["city"])
    return frame


def _targets(definition, random, score):
    metric = definition["metric"]
    if metric == "RMSE":
        return score + random.normal(scale=0.5, size=len(score))
    if metric == "MAE":
        return score + random.laplace(scale=0.7, size=len(score))
    classes = np.asarray(definition["classes"], dtype=object)
    if len(classes) == 2:
        positive = random.uniform(size=len(score)) < 1 / (1 + np.exp(-score))
        return np.where(positive, definition["positive_class"],
                        classes[classes != definition["positive_class"]][0])
    labels = classes[np.searchsorted(np.quantile(score, [1 / 3, 2 / 3]), score)]
    noisy = random.uniform(size=len(score)) < 0.05
    labels[noisy] = random.choice(classes, size=int(noisy.sum()))
    return labels


def generate(name, seed=SEED):
    """Return fit/holdout/test frames and host-only holdout answers; writes nothing."""
    definition = case_definition(name)
    random = np.random.default_rng([seed, list(CASES).index(name)])
    pairs = definition["duplicate_pairs"]
    units = definition["train_rows"] // 2 if pairs else definition["train_rows"]
    time_case = definition["validation"]["strategy"] == "time"
    periods = (1 + np.arange(units) * 6 // units if time_case
               else random.integers(1, 7, size=units))
    columns, score = _latent(random, units, test=False, time_trend=periods if time_case else None)
    targets = _targets(definition, random, score)
    if pairs:
        repeat = np.repeat(np.arange(units), 2)
        columns = {key: value[repeat] for key, value in columns.items()}
        periods, targets = periods[repeat], targets[repeat]
    train = _frame(name + "-train", columns, periods)
    train[TARGET_COLUMN] = [str(value) for value in targets] if definition["classes"] else targets
    test_periods = (np.full(definition["test_rows"], 7) if time_case
                    else random.integers(1, 7, size=definition["test_rows"]))
    test_columns, _ = _latent(random, definition["test_rows"], test=True,
                              time_trend=test_periods if time_case else None)
    test = _frame(name + "-test", test_columns, test_periods)
    if definition["host_holdout"] == "latest_period":
        held = periods == periods.max()
    elif definition["host_holdout"] == "seeded_pair_fraction":
        chosen = random.permutation(units)[:int(round(units * HOLDOUT_FRACTION))]
        held = np.isin(np.arange(len(train)) // 2, chosen)
    else:
        chosen = random.permutation(len(train))[:int(round(len(train) * HOLDOUT_FRACTION))]
        held = np.isin(np.arange(len(train)), chosen)
    holdout = train.loc[held].reset_index(drop=True)
    answers = holdout.pop(TARGET_COLUMN).tolist()
    return {"name": name, "definition": definition, "fit": train.loc[~held].reset_index(drop=True),
            "holdout": holdout, "holdout_answers": answers, "test": test}
