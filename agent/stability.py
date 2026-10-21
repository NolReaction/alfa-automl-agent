"""Predetermined repeat checks confined to an already frozen outer-fit view.

These helpers do not run models or grant a time budget. Callers must freeze a
shortlist first and reserve enough time for a complete paired round before
starting it. A partial round never changes the ranking.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from agent.splits import normalize_validation, split_rows


def _indices(value, size):
    result = np.asarray(value)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or len(np.unique(result)) != len(result)
            or np.any(result < 0) or np.any(result >= size)):
        raise ValueError("Stability partitions require unique in-range integer row positions")
    return result.astype(int, copy=True)


def _identity(fitting, held):
    # Row positions are local to the supplied outer-fit view. Neither labels
    # nor any outer-holdout positions/answers are available to this helper.
    payload = json.dumps({"fit": sorted(map(int, fitting)), "held": sorted(map(int, held))},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def additional_inner_splits(outer_train, outer_y, features, classification,
                            validation=None, *, canonical=None, limit=2):
    """Return at most two extra ``(fit, held, metadata)`` inner partitions.

    All returned positions are LOCAL to ``outer_train``/``outer_y``. Pass only
    the frozen outer-fit view, never the full training frame. ``canonical`` is
    the existing inner ``(fit, held)`` pair in these same local coordinates;
    if omitted it is computed with :func:`agent.splits.split_rows`.

    Random checks use fixed seeds 43 and 44 (the usual seed 42 plus one/two).
    Grouped checks use the next admissible folds of the same deterministic
    StratifiedGroupKFold partition as the canonical split. Time checks walk
    backwards through earlier periods, fitting strictly before each held
    period. Later periods are omitted from that check, not put in its fit.

    Classification checks require every class present in the supplied
    outer-fit labels on both sides. An impossible extra partition is skipped;
    no fallback to random splitting or to the outer holdout is allowed.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    from offline_baseline import SEED, VALIDATION_FRACTION

    if type(limit) is not int or not 0 <= limit <= 2:
        raise ValueError("At most two additional inner stability splits are allowed")
    options = normalize_validation(validation)
    strategy = options["strategy"]
    labels = np.asarray(outer_y)
    size = len(outer_train)
    if size < 2 or labels.ndim != 1 or len(labels) != size:
        raise ValueError("Stability checks require an outer-fit frame and matching target vector")
    if (not features or len(set(features)) != len(features)
            or any(name not in outer_train.columns for name in features)):
        raise ValueError("Stability features must be distinct columns of the outer-fit view")
    classes = set(map(str, labels)) if classification else None
    if classification and (len(classes) < 2 or pd.isna(labels).any()):
        raise ValueError("Classification stability checks require at least two nonmissing classes")
    groups, periods = None, None
    if strategy == "grouped":
        if not classification:
            raise ValueError("Duplicate-group stability checks require classification")
        groups = pd.util.hash_pandas_object(outer_train.loc[:, features], index=False).to_numpy()
    elif strategy == "time":
        column = options["time_column"]
        if column not in features:
            raise ValueError("The declared stability time column must be a feature")
        periods = pd.to_numeric(outer_train[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(periods).all():
            raise ValueError("Stability time periods must be finite numeric values")

    def check(fitting, held):
        fitting, held = _indices(fitting, size), _indices(held, size)
        if len(np.intersect1d(fitting, held)):
            raise ValueError("Stability fit and held partitions overlap")
        if classification and (set(map(str, labels[fitting])) != classes
                               or set(map(str, labels[held])) != classes):
            raise ValueError("Every outer-fit class must occur on both sides of a stability split")
        if groups is not None and len(np.intersect1d(groups[fitting], groups[held])):
            raise ValueError("Duplicate feature groups cross a stability split")
        if periods is not None and not periods[fitting].max() < periods[held].min():
            raise ValueError("Stability training periods must be strictly earlier than held periods")
        return fitting, held

    if canonical is None:
        fitting, held, _ = split_rows(outer_train, outer_y, features, classification, options)
    else:
        if not isinstance(canonical, (tuple, list)) or len(canonical) != 2:
            raise ValueError("Canonical stability split must be a pair of local row arrays")
        fitting, held = canonical
    canonical_fit, canonical_held = check(fitting, held)
    seen = {_identity(canonical_fit, canonical_held)}
    result = []
    if limit == 0:
        return result

    def append(fitting, held, extra):
        try:
            fitting, held = check(fitting, held)
        except ValueError:
            return
        identity = _identity(fitting, held)
        if identity in seen:
            return
        seen.add(identity)
        metadata = {
            "strategy": strategy, "split_hash": identity,
            "repeat": len(result) + 1, "train_rows": len(fitting),
            "validation_rows": len(held), "excluded_rows": size - len(fitting) - len(held),
            "train_fraction": len(fitting) / size, "validation_fraction": len(held) / size,
            "stratified": bool(classification) and strategy != "time",
            "shuffle": strategy != "time", **extra,
        }
        result.append((fitting, held, metadata))

    all_rows = np.arange(size)
    if strategy == "random":
        for repeat in (1, 2):
            try:
                fitting, held = train_test_split(
                    all_rows, test_size=VALIDATION_FRACTION, random_state=SEED + repeat,
                    shuffle=True, stratify=labels if classification else None,
                )
            except ValueError:
                continue
            append(fitting, held, {"seed": SEED + repeat})
            if len(result) == limit:
                break
    elif strategy == "grouped":
        group_count = len(np.unique(groups))
        _, counts = np.unique(labels, return_counts=True)
        maximum = min(5, group_count, int(counts.min()))
        for n_splits in range(maximum, 1, -1):
            valid = []
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for fold, (fitting, held) in enumerate(splitter.split(all_rows, labels, groups)):
                try:
                    fitting, held = check(fitting, held)
                except ValueError:
                    continue
                valid.append((fold, fitting, held))
            if not valid:
                continue
            for fold, fitting, held in valid:
                append(fitting, held, {
                    "seed": SEED, "n_splits": n_splits, "fold": fold,
                    "grouping": "raw_feature_fingerprint", "group_count": group_count,
                    "train_groups": len(np.unique(groups[fitting])),
                    "validation_groups": len(np.unique(groups[held])), "group_overlap": 0,
                })
                if len(result) == limit:
                    break
            # This is the canonical partition's n_splits. Do not switch to a
            # second partition scheme merely because some folds are unusable.
            break
    else:
        canonical_start = periods[canonical_held].min()
        for period in sorted(np.unique(periods[periods < canonical_start]), reverse=True):
            fitting, held = all_rows[periods < period], all_rows[periods == period]
            if not len(fitting):
                continue
            append(fitting, held, {
                "seed": SEED, "time_column": options["time_column"],
                "train_period_min": str(periods[fitting].min()),
                "train_period_max": str(periods[fitting].max()), "validation_period": str(period),
            })
            if len(result) == limit:
                break
    return result


def paired_stability_summary(candidate_ids, rounds, *, higher_is_better):
    """Rank a frozen shortlist using only complete, shared sequential rounds.

    ``rounds[0]`` is the original inner split. Further dictionaries map the
    same candidate IDs to independently recomputed scores on fixed repeats.
    At the first incomplete/nonfinite round, that round and all later ones
    are ignored for *every* candidate. Exact ties keep ``candidate_ids`` order,
    so callers can place the existing winner first. Standard deviation is
    descriptive population dispersion, not a confidence interval: repeated
    training splits can overlap.
    """
    candidate_ids = list(candidate_ids)
    rounds = list(rounds)
    if (not 1 <= len(candidate_ids) <= 3 or len(set(candidate_ids)) != len(candidate_ids)
            or any(not isinstance(value, str) or not value for value in candidate_ids)):
        raise ValueError("Stability ranking requires one to three distinct candidate IDs")
    if len(rounds) > 3:
        raise ValueError("Stability ranking accepts at most three inner rounds")
    if type(higher_is_better) is not bool:
        raise ValueError("Stability metric direction must be explicit")
    complete = []
    for scores in rounds:
        if not isinstance(scores, dict) or set(scores) != set(candidate_ids):
            break
        values = {}
        for candidate in candidate_ids:
            value = scores[candidate]
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(value))):
                break
            values[candidate] = float(value)
        else:
            complete.append(values)
            continue
        break
    candidates = {}
    for candidate in candidate_ids:
        values = [scores[candidate] for scores in complete]
        # Normalize first so even individually finite, very large errors do
        # not overflow while summing/squaring them to describe dispersion.
        scale = max(map(abs, values), default=0.0)
        normalized = np.asarray(values, dtype=float) / scale if scale else np.zeros(len(values))
        candidates[candidate] = {
            "scores": values, "mean": float(np.mean(normalized)) * scale if values else None,
            "std": float(np.std(normalized, ddof=0)) * scale if values else None,
            "rounds": len(values),
        }
    ranking = sorted(candidate_ids, key=lambda candidate: candidates[candidate]["mean"],
                     reverse=higher_is_better) if complete else list(candidate_ids)
    return {
        "used_rounds": len(complete), "provided_rounds": len(rounds),
        "ignored_rounds": len(rounds) - len(complete),
        "higher_is_better": higher_is_better, "candidates": candidates, "ranking": ranking,
        "dispersion": "population_std_of_overlapping_inner_splits",
        "selection_policy": "complete_paired_prefix_mean_stable_ties",
    }
