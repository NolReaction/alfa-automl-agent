"""Predeclared synthetic Macro F1 cases; no experiment outcome is an input."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupShuffleSplit

SEEDS = (73, 119, 2027)
CLASSES = ("01", "NA", "class c")
CASE_DEFINITIONS = {
    "imbalance_overlap": {
        "rows": 1125, "features": 10, "informative": 6, "redundant": 2,
        "weights": [0.74, 0.20, 0.06], "class_sep": 0.8, "flip_y": 0.04,
        "split": "raw_feature_groups_60_20_20",
    },
    "contradictory_duplicates": {
        "groups": 300, "copies_per_group": 4, "features": 6,
        "logits": "[0.8*x0,-0.2+0.8*x1,-0.6+0.7*x2]",
        "probabilities": "normalize(exp(logits-row_max)+0.15)",
        "target_sampling": "four independent categorical draws per identical feature group",
        "split": "raw_feature_groups_60_20_20",
    },
    "categorical_shift": {
        "rows": 1125, "numeric_features": 3,
        "known_categories": ["oak", "river", "stone"],
        "audit_unknown_fraction": 0.35, "unknown_category": "unseen",
        "audit_logit_shift": [0.0, -0.4, 0.4],
        "split": "chronological_fit675_calibration225_audit225",
    },
}


def _group_split(frame, seed):
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    rows = np.arange(len(frame))
    development, audit = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                                random_state=seed).split(rows, groups=groups))
    inner_fit, inner_cal = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                random_state=seed + 1).split(
                                                    development, groups=groups[development]))
    fit, calibration = development[inner_fit], development[inner_cal]
    return fit, calibration, audit


def validate_splits(frame, fit, calibration, audit, *, chronological=False):
    """Reject crossing rows/raw duplicates; never search for another split."""
    arrays = [np.asarray(part, dtype=int) for part in (fit, calibration, audit)]
    if any(not len(part) for part in arrays):
        raise ValueError("Every declared partition must contain observations.")
    joined = np.concatenate(arrays)
    if len(joined) != len(frame) or not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError("Partitions must cover every source row exactly once.")
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    group_sets = [set(groups[part].tolist()) for part in arrays]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Raw feature groups cross declared partitions.")
    result = {"row_overlap": 0, "raw_feature_group_overlap": 0,
              "group_counts": dict(zip(("fit", "calibration", "audit"), map(len, group_sets)))}
    if chronological:
        periods = frame["period"].to_numpy()
        ranges = [[int(periods[part].min()), int(periods[part].max())] for part in arrays]
        if not ranges[0][1] < ranges[1][0] <= ranges[1][1] < ranges[2][0]:
            raise ValueError("Chronological boundaries are not strictly forward.")
        result["period_ranges"] = dict(zip(("fit", "calibration", "audit"), ranges))
    return result


def make_case(name, seed):
    if name not in CASE_DEFINITIONS or seed not in SEEDS:
        raise ValueError("Use only the predeclared cases and seeds.")
    rng = np.random.default_rng(seed)
    if name == "imbalance_overlap":
        x, labels = make_classification(
            n_samples=1125, n_features=10, n_informative=6, n_redundant=2,
            n_classes=3, n_clusters_per_class=1, weights=[0.74, 0.20, 0.06],
            class_sep=0.8, flip_y=0.04, random_state=seed,
        )
        frame = pd.DataFrame(x, columns=[f"x{index}" for index in range(10)])
        split = _group_split(frame, seed)
        strategy = "grouped"
    elif name == "contradictory_duplicates":
        unique = rng.normal(size=(300, 6))
        logits = np.column_stack((0.8 * unique[:, 0], -0.2 + 0.8 * unique[:, 1],
                                  -0.6 + 0.7 * unique[:, 2]))
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)) + 0.15
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = np.array([rng.choice(3, p=probabilities[group])
                           for group in range(300) for _ in range(4)])
        frame = pd.DataFrame(np.repeat(unique, 4, axis=0), columns=[f"x{i}" for i in range(6)])
        split = _group_split(frame, seed)
        strategy = "grouped"
    else:
        x = rng.normal(size=(1125, 3))
        categories = rng.choice(["oak", "river", "stone"], size=1125).astype(object)
        categories[900:][rng.random(225) < 0.35] = "unseen"
        effects = {"oak": [0.7, -0.3, 0.0], "river": [-0.2, 0.6, -0.2],
                   "stone": [-0.3, -0.2, 0.7], "unseen": [0.0, 0.0, 0.0]}
        logits = np.column_stack((0.9*x[:, 0], -0.2+0.9*x[:, 1], -0.4+0.9*x[:, 2]))
        logits += np.array([effects[value] for value in categories])
        logits[900:] += np.array([0.0, -0.4, 0.4])
        labels = np.argmax(logits + rng.gumbel(size=(1125, 3)), axis=1)
        frame = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        frame["habitat"] = categories
        frame["period"] = np.arange(1125)
        split = np.arange(675), np.arange(675, 900), np.arange(900, 1125)
        strategy = "time"
    integrity = validate_splits(frame, *split, chronological=strategy == "time")
    return {
        "name": name, "seed": seed, "frame": frame,
        "labels": np.asarray(CLASSES)[labels],
        "ids": np.asarray([f"{name}-{seed}-{index:04d}" for index in range(len(frame))]),
        "classes": list(CLASSES), "fit_rows": split[0], "calibration_rows": split[1],
        "audit_rows": split[2], "split": {"strategy": strategy, "seed": seed, **integrity},
    }
