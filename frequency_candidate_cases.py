"""Fixed, fit-independent synthetic cases for the isolated frequency trial.

The signal cases use a *latent, predefined* category sampling prior.  Their
answers never depend on empirical train/test counts.  Generation and writing
are separate: only the caller receives audit answers; the task writer writes
the four permitted input CSVs and task description, never those answers.

Class cases contain pairs of identical raw feature rows in TRAIN.  The native
host's grouped validation must keep each such pair together.  The regression
case has unique feature rows and uses the native random policy, because the
current host does not support grouped regression validation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = 1
TRAIN_ROWS = 1200
AUDIT_ROWS = 300
CLASSES = ("01", "NA", "class c")
DEVELOPMENT_SEED = 701
CONFIRMATION_SEEDS = (1907, 7919)
DEVELOPMENT_CASES = (
    "category_prevalence_classification", "category_no_frequency_signal",
    "category_prevalence_regression", "numeric_only",
)
CONFIRMATION_CASES = DEVELOPMENT_CASES[:2]
CATEGORY_COLUMNS = ("habitat kind",)
NUMERIC_COLUMNS = ("measurement", "auxiliary", "noise", "scale")
KNOWN_CATEGORY_COUNT = 192
UNKNOWN_CATEGORY_COUNT = 16
PRIOR_EXPONENT = 1.0
SPECIAL_CATEGORIES = ("NA", "0007", "7", "", "class c", " Ω ")
CASE_DEFINITIONS = {
    "category_prevalence_classification": {
        "train_rows": TRAIN_ROWS, "audit_rows": AUDIT_ROWS,
        "train_raw_units": 600, "copies_per_train_unit": 2,
        "metric": "Macro F1", "validation": {"strategy": "grouped"},
        "categorical": list(CATEGORY_COLUMNS), "numeric": list(NUMERIC_COLUMNS),
        "latent_category_prior": "q[r] = (r+1)^-1 / sum((j+1)^-1, j=0..207)",
        "train_category_sampling": "q conditioned on ranks 0..191",
        "audit_category_sampling": "0.90 known conditioned q + 0.10 ranks 192..207 conditioned q",
        "target": "base label by latent rank: 01 if r<12, NA if 12<=r<128, class c if r>=128",
        "label_noise": "6% independent replacement, probabilities [0.70,0.25,0.05]",
        "signal": "predefined category prevalence, not an observed train/test count",
    },
    "category_no_frequency_signal": {
        "train_rows": TRAIN_ROWS, "audit_rows": AUDIT_ROWS,
        "train_raw_units": 600, "copies_per_train_unit": 2,
        "metric": "Macro F1", "validation": {"strategy": "grouped"},
        "categorical": list(CATEGORY_COLUMNS), "numeric": list(NUMERIC_COLUMNS),
        "latent_category_prior": "same fixed q as category_prevalence_classification",
        "audit_category_sampling": "same independent 90/10 known/unknown mixture",
        "target": "digitize(x0 + 0.65*x1 - 0.25*x2 + Normal(0,0.35), [-0.5,1.1])",
        "signal": "numeric-only conditional answer; category and its prevalence are independent",
    },
    "category_prevalence_regression": {
        "train_rows": TRAIN_ROWS, "audit_rows": AUDIT_ROWS,
        "train_raw_units": TRAIN_ROWS, "copies_per_train_unit": 1,
        "metric": "RMSE", "validation": {"strategy": "random"},
        "categorical": list(CATEGORY_COLUMNS), "numeric": list(NUMERIC_COLUMNS),
        "latent_category_prior": "same predefined q over ranks 0..207",
        "audit_category_sampling": "same independent 90/10 known/unknown mixture",
        "target": "3*log(q[r]) + 2*x0 + 0.8*sin(x1) + Normal(0,0.45)",
        "signal": "signed response depends on latent prior, never an empirical audit count",
    },
    "numeric_only": {
        "train_rows": TRAIN_ROWS, "audit_rows": AUDIT_ROWS,
        "train_raw_units": 600, "copies_per_train_unit": 2,
        "metric": "Macro F1", "validation": {"strategy": "grouped"},
        "categorical": [], "numeric": list(NUMERIC_COLUMNS),
        "target": "digitize(x0 + 0.65*x1 - 0.25*x2 + Normal(0,0.35), [-0.5,1.1])",
        "signal": "no declared categorical columns; frequency must skip without an extra fit",
    },
}


@dataclass(frozen=True)
class FrequencyCase:
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
    metadata: dict


def case_definitions() -> dict:
    """A detached JSON description to freeze before the first quality run."""
    return deepcopy(CASE_DEFINITIONS)


def suite_cases(stage: str | None = None) -> list[dict]:
    """Return exactly four development and four confirmation specifications."""
    if stage not in {None, "development", "confirmation"}:
        raise ValueError("stage must be development, confirmation or None")
    development = [{"name": name, "seed": DEVELOPMENT_SEED,
                    "stage": "development"} for name in DEVELOPMENT_CASES]
    confirmation = [{"name": name, "seed": seed, "stage": "confirmation"}
                    for name in CONFIRMATION_CASES for seed in CONFIRMATION_SEEDS]
    return development if stage == "development" else confirmation if stage == "confirmation" else development + confirmation


def _category_parameters(seed: int):
    total = KNOWN_CATEGORY_COUNT + UNKNOWN_CATEGORY_COUNT
    ranks = np.arange(total, dtype=float)
    prior = (ranks + 1.0) ** -PRIOR_EXPONENT
    prior /= prior.sum()
    # Ordinary spelling order does not encode the latent rank.  The six exact
    # special strings have fixed high-prevalence ranks so they are represented
    # in the fixed training cases, without inspecting/redrawing their counts.
    permutation = np.random.default_rng(seed + 101).permutation(KNOWN_CATEGORY_COUNT-len(SPECIAL_CATEGORIES))
    known = list(SPECIAL_CATEGORIES) + [f"kind-{value:04d}" for value in permutation]
    unknown = [f"unseen-kind-{value:03d}" for value in range(UNKNOWN_CATEGORY_COUNT)]
    return prior, np.asarray(known + unknown, dtype=object)


def _category_draws(rng, count: int, prior, *, audit: bool):
    known = prior[:KNOWN_CATEGORY_COUNT]
    known = known / known.sum()
    draws = rng.choice(KNOWN_CATEGORY_COUNT, size=count, p=known)
    if audit:
        # This mask and prior are set before observing the batch.  There is no
        # test frequency dictionary anywhere in the data-generating mechanism.
        mask = rng.random(count) < 0.10
        unknown = prior[KNOWN_CATEGORY_COUNT:]
        draws[mask] = KNOWN_CATEGORY_COUNT + rng.choice(
            UNKNOWN_CATEGORY_COUNT, size=int(mask.sum()), p=unknown/unknown.sum())
    return draws


def _unit_data(name: str, count: int, seed: int, *, audit: bool):
    # Independent streams ensure changing an audit batch cannot alter TRAIN.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 503 if audit else 211]))
    numeric = rng.normal(size=(count, len(NUMERIC_COLUMNS)))
    frame = pd.DataFrame(numeric, columns=NUMERIC_COLUMNS)
    frame["scale"] *= 1000.0
    if name != "numeric_only":
        prior, tokens = _category_parameters(seed)
        ranks = _category_draws(rng, count, prior, audit=audit)
        frame[CATEGORY_COLUMNS[0]] = tokens[ranks]
    if name == "category_prevalence_classification":
        labels = np.where(ranks < 12, 0, np.where(ranks < 128, 1, 2))
        noisy = rng.random(count) < 0.06
        labels[noisy] = rng.choice(3, size=int(noisy.sum()), p=[.70, .25, .05])
        target = np.asarray(CLASSES, dtype=object)[labels]
    elif name == "category_prevalence_regression":
        target = (3*np.log(prior[ranks]) + 2*numeric[:, 0]
                  + .8*np.sin(numeric[:, 1]) + rng.normal(0, .45, count))
    else:
        value = numeric[:, 0] + .65*numeric[:, 1] - .25*numeric[:, 2] + rng.normal(0, .35, count)
        target = np.asarray(CLASSES, dtype=object)[np.digitize(value, [-.5, 1.1])]
    return frame, target


def _ids(name: str, seed: int, count: int, *, audit: bool):
    prefix = "audit" if audit else "train"
    values = [f"{name}:{seed}:{prefix}-{index:06d}" for index in range(count)]
    if audit:
        values[:3] = ["000007", "NA", "audit c"]
    return values


def make_case(name: str, seed: int) -> FrequencyCase:
    """Generate only a predeclared case/seed; no fitting or adaptive retries."""
    if name not in CASE_DEFINITIONS or type(seed) is not int:
        raise ValueError("Use a predeclared frequency case and integer seed")
    allowed_seeds = (DEVELOPMENT_SEED,) + (CONFIRMATION_SEEDS if name in CONFIRMATION_CASES else ())
    if seed not in allowed_seeds:
        raise ValueError("This case/seed is not in the frozen frequency suite")
    definition = CASE_DEFINITIONS[name]
    copies = definition["copies_per_train_unit"]
    features, target = _unit_data(name, definition["train_raw_units"], seed, audit=False)
    if copies > 1:
        features = features.iloc[np.repeat(np.arange(len(features)), copies)].reset_index(drop=True)
        target = np.repeat(target, copies)
    test, audit = _unit_data(name, AUDIT_ROWS, seed, audit=True)
    train = features.copy(deep=True)
    train.insert(0, "row_key", _ids(name, seed, len(train), audit=False))
    train["outcome"] = target
    test.insert(0, "row_key", _ids(name, seed, len(test), audit=True))
    classes = CLASSES if definition["metric"] == "Macro F1" else ()
    if classes and set(train["outcome"]) != set(classes):
        raise ValueError("The fixed generation lacks a declared class; no resampling is permitted")
    raw_features = [*NUMERIC_COLUMNS, *definition["categorical"]]
    train_groups = pd.util.hash_pandas_object(train[raw_features], index=False).to_numpy()
    audit_groups = pd.util.hash_pandas_object(test[raw_features], index=False).to_numpy()
    if set(train_groups.tolist()) & set(audit_groups.tolist()):
        raise ValueError("Generated raw feature groups cross TRAIN and audit")
    description = (
        "Predict outcome from the declared observed numeric and categorical columns. "
        "Technical row_key is an identifier, not a feature. Category strings are exact, "
        "including NA, 0007, 7 and empty strings; some audit categories are unseen. "
        "No audit answers are provided. "
        + ("Keep identical raw feature groups together in validation." if classes else
           "Use a fixed random validation holdout; response may be signed."))
    metadata = {"version": VERSION, "definition": deepcopy(definition),
                "target_source": "fixed latent parameters and independent unit RNG; no empirical split counts",
                "numeric_rng": "SeedSequence([seed,211]) TRAIN; SeedSequence([seed,503]) audit",
                "token_permutation_seed": seed + 101,
                "raw_feature_groups_train": len(set(train_groups.tolist())),
                "raw_feature_groups_audit": len(set(audit_groups.tolist())),
                "train_audit_group_overlap": 0,
                "resampling_attempts": 0,
                "class_support_train": {label: int(np.count_nonzero(target == label)) for label in classes}}
    return FrequencyCase(name, seed, definition["metric"], classes, None,
                         deepcopy(definition["validation"]), NUMERIC_COLUMNS,
                         tuple(definition["categorical"]), train, test,
                         tuple(value.item() if isinstance(value, np.generic) else value for value in audit),
                         description, metadata)


def write_case(case: FrequencyCase, directory: Path) -> Path:
    """Create solver inputs in a new directory; audit answers are never written."""
    if not isinstance(case, FrequencyCase):
        raise ValueError("write_case requires a generated FrequencyCase")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    for name, frame in (("train.csv", case.train), ("test.csv", case.test)):
        frame.to_csv(inputs/name, index=False, lineterminator="\n", float_format="%.17g")
    records = [{"feature": "row_key", "role": "id", "dtype": "category",
                "description": "Literal identifier; never a model feature"}]
    for column in (*case.numeric, *case.categorical):
        records.append({"feature": column, "role": "feature",
                        "dtype": "numeric" if column in case.numeric else "category",
                        "description": ("Observed real measurement" if column in case.numeric else
                                        "Exact string category, including literal NA, 0007, 7 and empty string")})
    records.append({"feature": "outcome", "role": "target",
                    "dtype": "category" if case.classes else "numeric", "description": "Prediction target"})
    pd.DataFrame(records).to_csv(inputs/"data_dictionary.csv", index=False, lineterminator="\n")
    pd.DataFrame({"row_key": case.test["row_key"].tolist(),
                  "outcome": [case.classes[0] if case.classes else 0.0]*len(case.test)}).to_csv(
        inputs/"sample_submission.csv", index=False, lineterminator="\n")
    lines = ["# Frequency candidate experiment: " + case.name, case.description,
             "Use inputs/train.csv, inputs/test.csv, inputs/data_dictionary.csv and inputs/sample_submission.csv.",
             "Predict outcome; preserve row_key and test order exactly; save submission.csv beside task.md.",
             "Task type: " + ("classification" if case.classes else "regression"),
             "Primary metric: " + case.metric]
    if case.classes:
        lines.append("Class order: " + json.dumps(case.classes, ensure_ascii=False))
    task = directory / "task.md"
    task.write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    return task
