"""Predeclared synthetic CSV cases for the developer-only offline benchmark.

The task writer never receives an audit-answer path and never writes the audit
targets. The caller owns those targets separately. Ordinary subprocesses share
the host filesystem; this is a data-flow boundary, not an OS security sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


CASE_NAMES = (
    "linear_regression", "nonlinear_regression", "temporal_drift",
    "imbalanced_binary", "multiclass_duplicates", "categorical_shift",
)
TRAIN_ROWS = 600
AUDIT_ROWS = 150
MULTICLASS_LABELS = ("01", "NA", "class c")
BINARY_LABELS = ("01", "NA")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    seed: int
    metric: str
    classes: tuple[str, ...]
    positive_class: str | None
    validation: dict
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    train: pd.DataFrame
    test: pd.DataFrame
    audit_actuals: tuple
    description: str


def _ids(prefix: str, count: int) -> list[str]:
    values = [f"{prefix}-{index:06d}" for index in range(count)]
    if prefix == "audit":
        # The surrounding pipeline must preserve literal IDs, including NA.
        values[:3] = ["000005", "NA", "audit c"]
    return values


def _frames(features: pd.DataFrame, targets, *, train_rows=TRAIN_ROWS):
    train = features.iloc[:train_rows].reset_index(drop=True).copy()
    test = features.iloc[train_rows:].reset_index(drop=True).copy()
    train.insert(0, "row_key", _ids("train", len(train)))
    test.insert(0, "row_key", _ids("audit", len(test)))
    train["outcome"] = list(targets[:train_rows])
    audit = tuple(value.item() if isinstance(value, np.generic) else value
                  for value in targets[train_rows:])
    return train, test, audit


def _missing(frame: pd.DataFrame, rng, columns, *, fraction=.07):
    frame = frame.copy()
    for column in columns:
        values = frame[column].astype(object)
        values.iloc[np.flatnonzero(rng.random(len(frame)) < fraction)] = ""
        frame[column] = values
    return frame


def _regression(name: str, seed: int, rng) -> BenchmarkCase:
    count = TRAIN_ROWS + AUDIT_ROWS
    segment = rng.integers(0, 4, count)
    x1, x2 = rng.normal(size=(2, count))
    scale = rng.normal(0, 1000, count)
    offsets = np.asarray([-.7, .2, 1.1, -.2])[segment]
    if name == "linear_regression":
        target = 3 * x1 - 1.8 * x2 + .002 * scale + offsets + rng.normal(0, .3, count)
        description = ("Predict a continuous measurement from mixed numeric and categorical inputs. "
                       "Numeric columns use different scales; empty feature cells are missing observations.")
    else:
        x1, x2 = rng.uniform(-3, 3, count), rng.uniform(-2, 2, count)
        target = 7 * np.exp(.5 * np.sin(x1) + .22 * x2 ** 2 + .0002 * x1 * scale
                            + .2 * offsets + rng.normal(0, .1, count))
        target[rng.random(count) < .015] *= 12
        description = ("Predict a nonnegative continuous measurement using MAE. Outcomes have a "
                       "skewed distribution with occasional large valid observations; the relationship "
                       "may contain interactions and nonlinear effects. Empty feature cells are missing observations.")
    features = pd.DataFrame({"measurement": x1, "auxiliary": x2, "large_scale": scale,
                             "segment": [f"segment-{value}" for value in segment]})
    features = _missing(features, rng, ("measurement", "segment"))
    train, test, audit = _frames(features, target)
    return BenchmarkCase(name, seed, "RMSE" if name == "linear_regression" else "MAE",
                         (), None, {"strategy": "random"},
                         ("measurement", "auxiliary", "large_scale"), ("segment",),
                         train, test, audit, description)


def _temporal(seed: int, rng) -> BenchmarkCase:
    periods = np.concatenate([np.repeat(np.arange(1, 6), TRAIN_ROWS // 5),
                              np.repeat(6, AUDIT_ROWS)])
    count = len(periods)
    measurement = rng.normal(.2 * periods, 1, count)
    auxiliary = rng.normal(size=count)
    segment = rng.integers(0, 3, count)
    target = 1.8 * measurement + .35 * auxiliary + .55 * periods + .7 * segment
    target += rng.normal(0, .25, count)
    features = pd.DataFrame({"period": periods, "measurement": measurement,
                             "auxiliary": auxiliary,
                             "segment": [f"segment-{value}" for value in segment]})
    features = _missing(features, rng, ("measurement",), fraction=.04)
    train, test, audit = _frames(features, target)
    return BenchmarkCase("temporal_drift", seed, "RMSE", (), None,
                         {"strategy": "time", "time_column": "period"},
                         ("period", "measurement", "auxiliary"), ("segment",),
                         train, test, audit,
                         "Predict the measurement for a future period. Training contains periods 1 through 5; "
                         "test contains period 6. Validation must use earlier periods to predict a later period.")


def _binary(seed: int, rng) -> BenchmarkCase:
    count = TRAIN_ROWS + AUDIT_ROWS
    x1, x2, x3 = rng.normal(size=(3, count))
    segment = rng.integers(0, 3, count)
    score = 1.5 * x1 + .8 * x2 - .5 * x3 + .35 * segment - 2.7
    probability = 1 / (1 + np.exp(-score))
    targets = np.where(rng.random(count) < probability, "01", "NA")
    features = pd.DataFrame({"measurement": x1, "auxiliary": x2, "noise": x3,
                             "segment": [f"segment-{value}" for value in segment]})
    features = _missing(features, rng, ("measurement", "segment"), fraction=.05)
    train, test, audit = _frames(features, targets)
    return BenchmarkCase("imbalanced_binary", seed, "LogLoss", BINARY_LABELS, "01",
                         {"strategy": "random"}, ("measurement", "auxiliary", "noise"),
                         ("segment",), train, test, audit,
                         "Predict the probability of literal class 01. The classes are imbalanced. "
                         "The literal label NA is a valid class, not a missing answer.")


def _duplicates(seed: int, rng) -> BenchmarkCase:
    groups = TRAIN_ROWS // 3
    x1, x2 = rng.normal(size=(2, groups + AUDIT_ROWS))
    segment = rng.integers(0, 3, groups + AUDIT_ROWS)
    score = x1 + .6 * x2 + .25 * segment
    labels = np.asarray(MULTICLASS_LABELS)
    base_labels = labels[np.digitize(score, [-.4, .7])]
    group_features = pd.DataFrame({"measurement": x1, "auxiliary": x2,
                                   "segment": [f"segment-{value}" for value in segment]})
    repeated = group_features.iloc[:groups].loc[group_features.index[:groups].repeat(3)].reset_index(drop=True)
    features = pd.concat([repeated, group_features.iloc[groups:]], ignore_index=True)
    targets = np.concatenate([np.repeat(base_labels[:groups], 3), base_labels[groups:]])
    # A predeclared 10% of train groups has one contradictory outcome. Exact
    # copies remain in the same validation partition despite label conflicts.
    for group in range(0, groups, 10):
        current = MULTICLASS_LABELS.index(targets[group * 3])
        targets[group * 3 + 2] = MULTICLASS_LABELS[(current + 1) % len(MULTICLASS_LABELS)]
    train, test, audit = _frames(features, targets)
    return BenchmarkCase("multiclass_duplicates", seed, "Macro F1", MULTICLASS_LABELS, None,
                         {"strategy": "grouped"}, ("measurement", "auxiliary"), ("segment",),
                         train, test, audit,
                         "Predict a literal class label. Identical raw feature rows repeat in training "
                         "and sometimes have different outcomes. Keep identical-feature groups together "
                         "during validation; test contains new feature groups.")


def _categorical(seed: int, rng) -> BenchmarkCase:
    count = TRAIN_ROWS + AUDIT_ROWS
    entity_effects = rng.normal(0, .8, (96, 3))
    entities = np.concatenate([rng.integers(0, 80, TRAIN_ROWS), rng.integers(0, 80, AUDIT_ROWS)])
    # Exactly 30 audit rows use categories never exposed during training.
    entities[TRAIN_ROWS:TRAIN_ROWS + 30] = rng.integers(80, 96, 30)
    regions = rng.integers(0, 4, count)
    x1, x2 = rng.normal(size=(2, count))
    scores = np.column_stack((1.4 * x1 - .4 * x2, -.8 * x1 + 1.2 * x2, -.5 * x1 - .9 * x2))
    scores += entity_effects[entities] + rng.normal(0, .3, (count, 3))
    targets = np.asarray(MULTICLASS_LABELS)[scores.argmax(axis=1)]
    region_names = np.asarray([f"region-{value}" for value in regions], dtype=object)
    region_names[TRAIN_ROWS + 30:TRAIN_ROWS + 45] = "new-region"
    features = pd.DataFrame({"measurement": x1, "auxiliary": x2,
                             "entity": [f"entity-{value:03d}" for value in entities],
                             "region": region_names})
    features = _missing(features, rng, ("measurement", "region"), fraction=.04)
    train, test, audit = _frames(features, targets)
    return BenchmarkCase("categorical_shift", seed, "LogLoss", MULTICLASS_LABELS, None,
                         {"strategy": "random"}, ("measurement", "auxiliary"),
                         ("entity", "region"), train, test, audit,
                         "Predict probabilities for all three literal classes. Categorical features contain "
                         "many levels, and test includes previously unseen levels. Empty feature cells are "
                         "missing observations; literal NA remains a valid class.")


def make_case(name: str, seed: int) -> BenchmarkCase:
    """Generate fixed-size data from one explicit seed, without external data."""
    if name not in CASE_NAMES:
        raise ValueError("Unknown benchmark case: " + str(name))
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("Benchmark seed must be an integer from 0 through 4294967295")
    rng = np.random.default_rng(seed)
    if name in {"linear_regression", "nonlinear_regression"}:
        result = _regression(name, seed, rng)
    else:
        result = {"temporal_drift": _temporal, "imbalanced_binary": _binary,
                  "multiclass_duplicates": _duplicates, "categorical_shift": _categorical}[name](seed, rng)
    if result.classes and set(result.train["outcome"]) != set(result.classes):
        raise ValueError("Generated training data does not contain every declared class")
    return result


def write_case(case: BenchmarkCase, directory: Path) -> Path:
    """Write only declared solver inputs to a new task directory, never labels."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    case.train.to_csv(inputs / "train.csv", index=False, lineterminator="\n", float_format="%.12g")
    case.test.to_csv(inputs / "test.csv", index=False, lineterminator="\n", float_format="%.12g")
    metadata = [("row_key", "id", "category", "Literal row identifier")]
    for column in (*case.numeric, *case.categorical):
        metadata.append((column, "feature", "numeric" if column in case.numeric else "category",
                         "Observed input feature; empty cells represent missing observations"))
    metadata.append(("outcome", "target", "category" if case.classes else "numeric", "Prediction target"))
    pd.DataFrame(metadata, columns=["feature", "role", "dtype", "description"]).to_csv(
        inputs / "data_dictionary.csv", index=False, lineterminator="\n")
    template = pd.DataFrame({"row_key": case.test["row_key"].tolist()})
    if case.metric in {"LogLoss", "ROC AUC"} and len(case.classes) > 2:
        # Deliberately distinct schema/class orders test explicit alignment.
        for label in (case.classes[-1], *case.classes[:-1]):
            template[label] = 1 / len(case.classes)
    else:
        template["outcome"] = .5 if case.metric in {"LogLoss", "ROC AUC"} else case.classes[0] if case.classes else 0
    template.to_csv(inputs / "sample_submission.csv", index=False, lineterminator="\n", float_format="%.12g")
    lines = [f"# Synthetic benchmark: {case.name}", case.description,
             "Use inputs/train.csv, inputs/test.csv, inputs/data_dictionary.csv and inputs/sample_submission.csv.",
             "Predict outcome; preserve row_key exactly and save submission.csv beside task.md.",
             "Task type: " + ("classification" if case.classes else "regression"),
             "Primary metric: " + case.metric]
    if case.classes:
        lines.append("Class order: " + json.dumps(case.classes))
    if case.positive_class is not None:
        lines.append("Positive class: " + json.dumps(case.positive_class))
    task = directory / "task.md"
    task.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return task
