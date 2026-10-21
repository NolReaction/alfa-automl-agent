"""Small frozen regression mechanisms for the developer-only history audit.

The solver receives ``BenchmarkCase.train`` and feature-only ``test``. Audit
answers remain in ``audit_actuals`` and must stay outside its input directory.
These synthetic mechanisms test an explicit history option; they do not guess
entity keys or establish that any key identifies a physical object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_cases import BenchmarkCase


CASE_NAMES = (
    "entity_signal", "no_history_coverage", "colliding_keys", "iid_regression",
)
DEVELOPMENT_SEEDS = (131, 379)
CONFIRMATION_SEEDS = (2203, 4001)
TRAIN_ROWS = 240
AUDIT_ROWS = 60
ROWS_PER_PERIOD = 48


def _check_name(name):
    if name not in CASE_NAMES:
        raise ValueError("Unknown history runtime case: " + repr(name))


def explicit_history_config(name: str) -> dict:
    """Detached opt-in config, including for the deliberately ineligible IID case.

    The runtime, rather than the fixture generator, must reject history when
    its validation policy has no eligible time axis. No weight is tuned here;
    the experiment fixes the blend weight at 0.5 for every requested candidate.
    """
    _check_name(name)
    return {"key_columns": ["entity_key"], "period_column": "period",
            "numeric_columns": [], "positive_columns": [], "ambiguity_policy": "allow"}


def _entity_names(count):
    names = [f"entity-{i:03d}" for i in range(count)]
    # These are ordinary literal keys, not missing values or numeric aliases.
    names[:3] = ["NA", "0007", "7"]
    return np.asarray(names, dtype=object)


def make_history_case(name: str, seed: int) -> BenchmarkCase:
    """Generate one deterministic case without training or exposing audit labels.

    All temporal cases contain periods 1..5 in train and period 6 in test.
    Row order within train/test is independently shuffled, so temporal safety
    cannot rely on contiguous row positions. The IID case has the same period
    support 1..5 in both partitions and should use the native random policy.
    """
    _check_name(name)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, CASE_NAMES.index(name) + 1]))
    count = TRAIN_ROWS + AUDIT_ROWS
    periods = np.concatenate([np.repeat(np.arange(1, 6), ROWS_PER_PERIOD),
                              np.repeat(6, AUDIT_ROWS)])
    measurement, auxiliary = rng.normal(size=(2, count))

    if name == "entity_signal":
        names = _entity_names(ROWS_PER_PERIOD)
        entity_index = np.concatenate([np.tile(np.arange(ROWS_PER_PERIOD), 5),
                                       np.arange(ROWS_PER_PERIOD),
                                       rng.integers(0, ROWS_PER_PERIOD, AUDIT_ROWS - ROWS_PER_PERIOD)])
        keys = names[entity_index]
        intercept = rng.normal(0, 18, ROWS_PER_PERIOD)
        target = (intercept[entity_index] + 0.08 * measurement + 0.10 * periods
                  + rng.normal(0, 0.25, count))
        description = (
            "Predict a numeric outcome for future periods from observed features. "
            "An entity key recurs across earlier periods, but keys are not declared physical identifiers. "
            "Validate strictly on later periods using earlier periods only.")
    elif name == "no_history_coverage":
        # Every row has a different key, including all validation/test periods.
        keys = np.asarray([f"single-observation-{i:04d}" for i in range(count)], dtype=object)
        target = (4.0 * measurement - 0.5 * auxiliary + 0.20 * periods
                  + rng.normal(0, 0.20, count))
        description = (
            "Predict a numeric outcome for future periods. Keys are single-observation labels; "
            "a later-period key has no earlier observation. Validate on later periods.")
    elif name == "colliding_keys":
        # Four coarse keys repeat 12 times inside every training period. They
        # carry no persistent target effect; matching them is a collision test.
        keys = np.asarray([f"coarse-bucket-{i % 4}" for i in range(count)], dtype=object)
        target = (4.0 * measurement - 2.0 * auxiliary + 0.10 * periods
                  + rng.normal(0, 0.05, count))
        description = (
            "Predict a numeric outcome for future periods. Coarse keys may occur many times "
            "inside one period and are not unique entity identifiers. Validate on later periods.")
    else:
        # Keep a parseable numeric period feature while deliberately providing
        # no future-only test period. A history opt-in must not force time split.
        periods = np.concatenate([np.repeat(np.arange(1, 6), ROWS_PER_PERIOD),
                                  np.repeat(np.arange(1, 6), AUDIT_ROWS // 5)])
        keys = _entity_names(12)[rng.integers(0, 12, count)]
        target = 3.0 * measurement - 1.1 * auxiliary + rng.normal(0, 0.15, count)
        description = (
            "Predict a numeric outcome for IID observations. Numeric period values are repeated "
            "metadata; train and test cover the same periods, with no future-only test period.")

    features = pd.DataFrame({"period": periods, "measurement": measurement,
                             "auxiliary": auxiliary, "entity_key": keys})
    fit_order = rng.permutation(TRAIN_ROWS)
    test_order = TRAIN_ROWS + rng.permutation(AUDIT_ROWS)
    train = features.iloc[fit_order].reset_index(drop=True).copy()
    test = features.iloc[test_order].reset_index(drop=True).copy()
    train.insert(0, "row_key", [f"train-{i:05d}" for i in fit_order])
    test.insert(0, "row_key", [f"audit-{i - TRAIN_ROWS:05d}" for i in test_order])
    # Literal output IDs test the serialization boundary without changing keys.
    test.loc[:2, "row_key"] = ["000001", "NA", "audit c"]
    train["outcome"] = target[fit_order]
    audit = tuple(float(value) for value in target[test_order])
    validation = ({"strategy": "random"} if name == "iid_regression" else
                  {"strategy": "time", "time_column": "period"})
    return BenchmarkCase(
        name=name, seed=seed, metric="RMSE", classes=(), positive_class=None,
        validation=validation, numeric=("period", "measurement", "auxiliary"),
        categorical=("entity_key",), train=train, test=test, audit_actuals=audit,
        description=description)
