"""Frozen synthetic cases for the isolated OOF Macro F1 experiment.

These are new seeded realizations of previously studied *families*, not new
unknown task types.  The categorical case is an IID domain-shift experiment;
unlike the earlier decision experiment it has no chronological interpretation.
The host must not use random OOF for a genuinely temporal task.

Every main case has three disjoint raw-feature group sets: 1500 allowed rows,
400 outer-audit rows, and 400 production-audit rows.  After rule freezing, the
host can fit on allowed + outer to check transfer to the production audit.
Neither labels nor observed model scores are used to pick any partition.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


VERSION = 1
CLASSES = ("01", "NA", "class c")
DEVELOPMENT_SEEDS = (131, 379)
CONFIRMATION_SEEDS = (2203, 4001, 10009)
FAMILIES = ("imbalance_overlap", "contradictory_duplicates", "categorical_shift")
DEGENERATE_CASES = ("rare_support", "impossible_split")
ALLOWED_ROWS = 1500
OUTER_ROWS = 400
PRODUCTION_ROWS = 400
TOTAL_ROWS = ALLOWED_ROWS + OUTER_ROWS + PRODUCTION_ROWS
CASE_DEFINITIONS = {
    "imbalance_overlap": {
        "rows": TOTAL_ROWS, "features": 10, "informative": 6, "redundant": 2,
        "weights": [0.74, 0.20, 0.06], "class_sep": 0.8, "flip_y": 0.04,
        "mechanism": "sklearn.make_classification; one cluster per class",
    },
    "contradictory_duplicates": {
        "rows": TOTAL_ROWS, "groups": 575, "copies_per_group": 4, "features": 6,
        "logits": "[0.8*x0,-0.2+0.8*x1,-0.6+0.7*x2]",
        "probabilities": "normalize(exp(logits-row_max)+0.15)",
        "mechanism": "four independent label draws per identical raw-feature group",
    },
    "categorical_shift": {
        "rows": TOTAL_ROWS, "numeric_features": 3,
        "known_categories": ["oak", "river", "stone"],
        "outer_and_production_unknown_fraction": 0.35,
        "unknown_category": "unseen",
        "outer_and_production_logit_shift": [0.0, -0.4, 0.4],
        "mechanism": "IID rows within each domain; no time column or chronology",
    },
    "rare_support": {
        "rows": TOTAL_ROWS, "features": 4, "allowed_rare_class_count": 6,
        "mechanism": "binary sign signal plus exactly six allowed rare labels",
        "purpose": "explicit class-support identity without changing guard defaults",
    },
    "impossible_split": {
        "rows": TOTAL_ROWS, "features": 2, "allowed_raw_feature_groups": 1,
        "mechanism": "one identical-feature allowed group with all three labels",
        "purpose": "host refuses an impossible group-safe A/B split before fitting",
    },
}


def case_definitions():
    """Return a detached JSON-compatible snapshot for preregistration."""
    return deepcopy(CASE_DEFINITIONS)


def suite_cases(stage):
    """List the fixed cases without generating labels or evaluating models."""
    if stage == "development":
        seeds, families = DEVELOPMENT_SEEDS, FAMILIES
    elif stage == "confirmation":
        seeds, families = CONFIRMATION_SEEDS, FAMILIES
    elif stage == "degenerate":
        seeds, families = (131,), DEGENERATE_CASES
    else:
        raise ValueError("stage must be development, confirmation or degenerate")
    return [{"name": name, "seed": seed} for name in families for seed in seeds]


def raw_feature_groups(frame):
    """SHA-256 identity of exact feature values; the label/index is excluded.

    Float hex strings retain the raw binary value, strings retain punctuation
    and whitespace, and types are explicit.  The generated frames contain only
    real numeric and string feature values.  This avoids treating contradictory
    labels of an identical feature row as different groups.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("raw feature groups require a DataFrame")
    result = []
    for row in frame.itertuples(index=False, name=None):
        encoded = []
        for value in row:
            if isinstance(value, str):
                encoded.append(["str", value])
            elif isinstance(value, (float, np.floating)):
                encoded.append(["float", float(value).hex()])
            elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                encoded.append(["int", str(int(value))])
            else:
                raise ValueError("unsupported generated feature value type")
        payload = json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))
        result.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return np.asarray(result, dtype=object)


def _fixed_roles(unit_ids, seed, allowed_units, outer_units):
    """Assign feature units to roles once, using no labels and no retries."""
    units = np.unique(unit_ids)
    order = np.random.default_rng(seed + 11).permutation(units)
    allowed_units_set = set(order[:allowed_units].tolist())
    outer_units_set = set(order[allowed_units:allowed_units + outer_units].tolist())
    role = np.asarray([
        0 if unit in allowed_units_set else 1 if unit in outer_units_set else 2
        for unit in unit_ids
    ])
    return tuple(np.flatnonzero(role == value) for value in range(3))


def _validate_roles(frame, groups, parts):
    joined = np.concatenate(parts)
    if not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError("allowed/outer/production partitions must cover every row exactly once")
    sets = [set(groups[part].tolist()) for part in parts]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("raw feature duplicates cross experiment roles")
    if tuple(map(len, parts)) != (ALLOWED_ROWS, OUTER_ROWS, PRODUCTION_ROWS):
        raise ValueError("generated sizes differ from the frozen protocol")
    return {"row_overlap": 0, "raw_feature_group_overlap": 0,
            "group_counts": dict(zip(("allowed", "outer", "production"), map(len, sets)))}


def make_case(name, seed):
    """Generate one fixed realization, with no model or audit outcome inputs."""
    if type(seed) is not int:
        raise ValueError("seed must be a predeclared integer")
    if name in FAMILIES:
        if seed not in DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS:
            raise ValueError("use only the preregistered family seeds")
    elif name in DEGENERATE_CASES:
        if seed != 131:
            raise ValueError("degenerate cases have the fixed seed 131")
    else:
        raise ValueError("unknown OOF classification case")
    rng = np.random.default_rng(seed)
    roles = _fixed_roles(np.arange(TOTAL_ROWS), seed, ALLOWED_ROWS, OUTER_ROWS)
    if name == "imbalance_overlap":
        x, labels = make_classification(
            n_samples=TOTAL_ROWS, n_features=10, n_informative=6, n_redundant=2,
            n_classes=3, n_clusters_per_class=1, weights=[0.74, 0.20, 0.06],
            class_sep=0.8, flip_y=0.04, random_state=seed,
        )
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(10)])
    elif name == "contradictory_duplicates":
        unique = rng.normal(size=(575, 6))
        logits = np.column_stack((0.8 * unique[:, 0], -0.2 + 0.8 * unique[:, 1],
                                  -0.6 + 0.7 * unique[:, 2]))
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)) + 0.15
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = np.asarray([rng.choice(3, p=probabilities[group])
                             for group in range(575) for _ in range(4)])
        frame = pd.DataFrame(np.repeat(unique, 4, axis=0), columns=[f"x{i}" for i in range(6)])
        roles = _fixed_roles(np.repeat(np.arange(575), 4), seed, 375, 100)
    elif name == "categorical_shift":
        x = rng.normal(size=(TOTAL_ROWS, 3))
        categories = rng.choice(["oak", "river", "stone"], TOTAL_ROWS).astype(object)
        future = np.concatenate((roles[1], roles[2]))
        categories[future[rng.random(len(future)) < 0.35]] = "unseen"
        effects = {"oak": [0.7, -0.3, 0.0], "river": [-0.2, 0.6, -0.2],
                   "stone": [-0.3, -0.2, 0.7], "unseen": [0.0, 0.0, 0.0]}
        logits = np.column_stack((0.9*x[:, 0], -0.2+0.9*x[:, 1], -0.4+0.9*x[:, 2]))
        logits += np.asarray([effects[value] for value in categories])
        logits[future] += np.asarray([0.0, -0.4, 0.4])
        labels = np.argmax(logits + rng.gumbel(size=(TOTAL_ROWS, 3)), axis=1)
        frame = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        frame["habitat kind"] = categories
    elif name == "rare_support":
        x = rng.normal(size=(TOTAL_ROWS, 4))
        labels = (x[:, 0] + 0.4*x[:, 1] > 0).astype(int)
        labels[roles[0][:6]] = 2
        for rows in roles[1:]:
            labels[rows[np.arange(len(rows)) % 10 == 0]] = 2
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(4)])
    else:
        # Three role-specific constants keep external raw groups disjoint.
        x = np.empty((TOTAL_ROWS, 2))
        for value, rows in enumerate(roles):
            x[rows] = value
        frame = pd.DataFrame(x, columns=["x0", "x1"])
        labels = np.arange(TOTAL_ROWS) % len(CLASSES)
    groups = raw_feature_groups(frame)
    integrity = _validate_roles(frame, groups, roles)
    return {
        "name": name, "seed": seed, "version": VERSION,
        "frame": frame, "labels": np.asarray(CLASSES)[labels],
        "classes": list(CLASSES), "groups": groups,
        "ids": np.asarray([f"{name}:{seed}:row-{i:05d}" for i in range(TOTAL_ROWS)]),
        "allowed_rows": roles[0], "outer_rows": roles[1], "production_rows": roles[2],
        "split_policy": "grouped" if name in {"contradictory_duplicates", "impossible_split"} else "iid",
        "generator_metadata": {
            "definition": deepcopy(CASE_DEFINITIONS[name]),
            "role_split_seed": seed + 11, "role_assignment": "label-free feature units",
            "family_history": ("previously studied family; fresh seeded realization"
                               if name in FAMILIES else "new structural/support probe"),
            "categorical_scope": "IID domain shift, not chronology" if name == "categorical_shift" else None,
            **integrity,
        },
    }


def make_partitions(case):
    """Create a single fixed label-free A/B split and three OOF folds in A.

    Indices are local to allowed rows.  If the fixed group structure cannot
    support the protocol, report unavailable; never retry a seed or substitute
    an IID split.  Missing class support does not trigger resplitting here.
    """
    rows = np.asarray(case["allowed_rows"])
    groups = np.asarray(case["groups"], dtype=object)[rows]
    local = np.arange(len(rows), dtype=np.int64)
    seed = case["seed"] + 31
    metadata = {"split_seed": seed, "selection_group_fraction": 0.3,
                "fold_count": 3, "fold_strategy": "GroupKFold(shuffle=False)",
                "label_stratification": False, "resplit_attempts": 0}
    unavailable = {"status": "unavailable", "reason": None,
                   "tuning_rows": np.asarray([], dtype=np.int64),
                   "selection_rows": np.asarray([], dtype=np.int64),
                   "folds": [], "metadata": metadata}
    if len(set(groups.tolist())) < 2:
        return {**unavailable, "reason": "insufficient_groups_for_tuning_selection"}
    tuning, selection = next(GroupShuffleSplit(
        n_splits=1, test_size=0.3, random_state=seed,
    ).split(local, groups=groups))
    if len(set(groups[tuning].tolist())) < 3:
        return {**unavailable, "reason": "insufficient_tuning_groups_for_three_folds"}
    folds = []
    for fit, held in GroupKFold(n_splits=3).split(tuning, groups=groups[tuning]):
        folds.append({"fit_rows": tuning[fit].astype(np.int64),
                      "held_rows": tuning[held].astype(np.int64)})
    metadata["group_counts"] = {"allowed": len(set(groups.tolist())),
                                "tuning": len(set(groups[tuning].tolist())),
                                "selection": len(set(groups[selection].tolist()))}
    return {"status": "ready", "reason": None,
            "tuning_rows": tuning.astype(np.int64), "selection_rows": selection.astype(np.int64),
            "folds": folds, "metadata": metadata}
