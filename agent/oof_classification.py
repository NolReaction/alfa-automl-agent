"""Group-safe OOF calibration with an independent Macro F1 admission set.

The host fixes local A/B indices, three folds inside A, and a finite deadline.
Only the supplied callback fits models; it receives copied fitting features,
fitting labels and prediction features, never prediction labels. Its host must
enforce the timeout in a bounded process. An ordinary function cannot be
interrupted merely by passing ``timeout_seconds``.

The three held predictions form OOF_A; a fourth fit on all A predicts B. The
existing guarded decision module selects one candidate using A and admits or
rejects it using B. Neither outer/test labels nor a production refit belong in
this function. Apply the returned ``rule`` with ``predict_with_rule`` only for
Macro F1 labels; original probabilities remain untouched for AUC/LogLoss.

The result contains finite JSON data, not OOF matrices. Callback artifacts may
be retained by the host separately. ``completed_fits`` counts callbacks that
returned valid predictions, including a valid result returned after deadline;
such a late result is discarded and never used to fit a decision rule.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real

import numpy as np

from agent.classification_decision import fit_guarded_decision_rule


class OOFContractError(ValueError):
    """Malformed inputs, split structure, clock or callback predictions."""


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise OOFContractError("OOF result must contain finite JSON-compatible data") from error


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strings(values, name, *, minimum=0):
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise OOFContractError(f"{name} must be a one-dimensional string sequence") from error
    if raw.ndim != 1 or len(raw) < minimum:
        raise OOFContractError(f"{name} must be a one-dimensional string sequence")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise OOFContractError(f"{name} must contain exact nonempty string labels")
    return [str(value) for value in raw]


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise OOFContractError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise OOFContractError(f"{name} must be a finite real number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise OOFContractError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _rows(values, n_rows, name):
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise OOFContractError(f"{name} must be a one-dimensional integer sequence") from error
    if raw.ndim != 1 or any(isinstance(value, (bool, np.bool_))
                            or not isinstance(value, (int, np.integer)) for value in raw):
        raise OOFContractError(f"{name} must be a one-dimensional integer sequence")
    result = [int(value) for value in raw]
    if len(set(result)) != len(result) or any(not 0 <= value < n_rows for value in result):
        raise OOFContractError(f"{name} contains repeated or out-of-range rows")
    return np.asarray(result, dtype=np.int64)


def _groups(values, n_rows):
    if isinstance(values, (str, bytes)) or getattr(values, "ndim", 1) != 1:
        raise OOFContractError("groups must be a one-dimensional sequence")
    try:
        groups = list(values)
    except TypeError as error:
        raise OOFContractError("groups must be a one-dimensional sequence") from error
    if len(groups) != n_rows:
        raise OOFContractError("groups must match the allowed rows")
    # Canonical integer codes retain group equality without serializing raw
    # group IDs or converting literal string IDs to numbers.
    codes, positions = [], {}
    for value in groups:
        try:
            if value is None or not bool(value == value):
                raise ValueError("missing group")
            if isinstance(value, Real) and not math.isfinite(value):
                raise ValueError("nonfinite group")
            hash(value)
            if value not in positions:
                positions[value] = len(positions)
            codes.append(positions[value])
        except (TypeError, ValueError, OverflowError) as error:
            raise OOFContractError("groups must contain nonmissing hashable scalar IDs") from error
    return np.asarray(codes, dtype=np.int64)


def _features(values, n_rows):
    if hasattr(values, "iloc"):
        if getattr(values, "ndim", None) != 2 or len(values) != n_rows:
            raise OOFContractError("X_allowed must be a two-dimensional table matching y_allowed")
        return values
    try:
        result = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise OOFContractError("X_allowed must be a two-dimensional table") from error
    if result.ndim != 2 or len(result) != n_rows:
        raise OOFContractError("X_allowed must be a two-dimensional table matching y_allowed")
    return result


def _subset(values, rows):
    if hasattr(values, "iloc"):
        result = values.iloc[rows].copy(deep=True)
        # DataFrame.copy(deep=True) copies its arrays but not Python objects
        # stored inside object cells. Isolate those too before invoking host
        # code, so a callback cannot mutate a nested cell in the input table.
        for column, dtype in enumerate(result.dtypes):
            if getattr(dtype, "kind", None) == "O":
                for row in range(len(result)):
                    result.iat[row, column] = deepcopy(result.iat[row, column])
        return result
    return deepcopy(values[rows])


def _prediction(result, n_rows, classes):
    if not isinstance(result, dict) or not {"probabilities", "classes"} <= set(result):
        raise OOFContractError("Callback must return probabilities and exact column classes")
    supplied = _strings(result["classes"], "callback classes", minimum=2)
    if len(set(supplied)) != len(supplied) or set(supplied) != set(classes):
        raise OOFContractError("Callback classes must match every declared class exactly once")
    try:
        # Inspect before float coercion so a mixed Python bool/numeric list
        # cannot silently become real probabilities.
        raw = np.asarray(result["probabilities"], dtype=object)
        if raw.shape != (n_rows, len(supplied)):
            raise ValueError("wrong shape")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in raw.flat):
            raise ValueError("not real numeric")
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise OOFContractError("Callback probabilities must be a real matrix with matching rows/classes") from error
    if (not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()
            or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise OOFContractError("Callback probabilities must be finite in [0, 1] and sum to one")
    return matrix[:, [supplied.index(label) for label in classes]].copy()


def fit_oof_guarded_rule(
    X_allowed, y_allowed, classes, *, groups, split_policy,
    tuning_rows, selection_rows, folds, fit_predict,
    fold_timeout_seconds, deadline, clock, guard_options=None,
):
    """Fit a frozen, group-safe OOF Macro F1 rule under one host deadline.

    ``iid`` and ``grouped`` require explicit groups and exactly three folds.
    Every held row belongs to A exactly once; fit rows are subsets of A and
    exclude both held rows and their groups. A and B cover the allowed table
    without duplicate rows or shared groups. Indices are positional/local,
    independent of a DataFrame's index. ``time`` validates A/B but returns
    identity without interpreting ``folds`` or inventing a temporal OOF split.

    Missing class support and model/timeout failures return identity. Invalid
    split structure, clock values, guard options or predictions raise
    :class:`OOFContractError`. Partial OOF predictions are never calibrated.
    No exception text from a failing callback is copied into the JSON result.
    """
    timeout = _number(fold_timeout_seconds, "fold_timeout_seconds", positive=True)
    deadline = _number(deadline, "deadline")
    if not callable(clock) or not callable(fit_predict):
        raise OOFContractError("clock and fit_predict must be callable")
    last_clock = None

    def now():
        nonlocal last_clock
        value = _number(clock(), "clock()")
        if last_clock is not None and value < last_clock:
            raise OOFContractError("clock must not move backwards")
        last_clock = value
        return value

    started = now()
    classes = _strings(classes, "classes", minimum=2)
    if len(set(classes)) != len(classes):
        raise OOFContractError("classes must be unique")
    labels = _strings(y_allowed, "y_allowed")
    if not set(labels) <= set(classes):
        raise OOFContractError("y_allowed contains undeclared classes")
    n_rows = len(labels)
    features = _features(X_allowed, n_rows)
    group_codes = _groups(groups, n_rows)
    tuning = _rows(tuning_rows, n_rows, "tuning_rows")
    selection = _rows(selection_rows, n_rows, "selection_rows")
    joined = np.concatenate((tuning, selection))
    if len(joined) != n_rows or not np.array_equal(np.sort(joined), np.arange(n_rows)):
        raise OOFContractError("A/B must be disjoint and cover every allowed row")
    if set(group_codes[tuning]) & set(group_codes[selection]):
        raise OOFContractError("A/B groups must be disjoint")
    if not isinstance(split_policy, str) or split_policy not in {"iid", "grouped", "time"}:
        raise OOFContractError("split_policy must explicitly be iid, grouped or time")
    # The existing public API validates options and constructs schema-correct
    # identity rules; no private search or rule schema is duplicated here.
    try:
        empty_guard = fit_guarded_decision_rule(
            np.empty((0, len(classes))), [], classes,
            tuning_rows=[], selection_rows=[], options=deepcopy(guard_options),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise OOFContractError("Invalid guard_options") from error
    config = empty_guard["options"]
    identity = empty_guard["rule"]
    labels_array = np.asarray(labels, dtype=object)

    def counts(rows):
        return [int(np.count_nonzero(labels_array[rows] == label)) for label in classes]

    support = {"allowed": counts(np.arange(n_rows)), "tuning": counts(tuning),
               "selection": counts(selection), "folds": []}
    normalized_folds = []
    if split_policy != "time":
        if not isinstance(folds, (list, tuple)) or len(folds) != 3:
            raise OOFContractError("folds must contain exactly three fit_rows/held_rows dictionaries")
        tuning_set = set(tuning.tolist())
        held_all = []
        for index, fold in enumerate(folds):
            if not isinstance(fold, dict) or set(fold) != {"fit_rows", "held_rows"}:
                raise OOFContractError("Every fold must contain exactly fit_rows and held_rows")
            fit = _rows(fold["fit_rows"], n_rows, f"folds[{index}].fit_rows")
            held = _rows(fold["held_rows"], n_rows, f"folds[{index}].held_rows")
            if len(tuning) and not len(held):
                raise OOFContractError("Each of the three OOF folds must have nonempty held rows")
            if not set(fit.tolist()) <= tuning_set or not set(held.tolist()) <= tuning_set:
                raise OOFContractError("All fold rows must belong to A; B cannot enter any fit")
            if set(fit.tolist()) & set(held.tolist()):
                raise OOFContractError("Fold fit and held rows must be disjoint")
            if set(group_codes[fit]) & set(group_codes[held]):
                raise OOFContractError("Fold fit and held groups must be disjoint")
            held_all.extend(held.tolist())
            normalized_folds.append({"fit_rows": fit.tolist(), "held_rows": held.tolist()})
            support["folds"].append({"fit": counts(fit), "held": counts(held)})
        if len(held_all) != len(tuning) or sorted(held_all) != sorted(tuning.tolist()):
            raise OOFContractError("Held folds must cover every A row exactly once")
    split_record = {"policy": split_policy, "classes": classes,
                    "tuning_rows": tuning.tolist(), "selection_rows": selection.tolist(),
                    "folds": normalized_folds, "group_codes": group_codes.tolist()}
    split_hash = _hash(split_record)
    fits = []
    completed_fits = 0
    fit_elapsed = 0.0
    staging_elapsed = 0.0
    guard_elapsed = 0.0

    def finish(reason, guarded=None):
        finished = now()
        if reason == "accepted" and finished >= deadline:
            reason = "deadline_exhausted"
        chosen = (guarded["rule"] if guarded is not None and reason == "accepted" else identity)
        candidate = guarded["candidate_rule"] if guarded is not None else empty_guard["candidate_rule"]
        elapsed = finished - started
        return _json_copy({
            "version": 1, "kind": "oof_guarded_macro_f1",
            "status": "accepted" if reason == "accepted" else "identity",
            "reason": reason, "classes": classes,
            "rule": chosen, "candidate_rule": candidate, "guarded": guarded,
            "support": support, "completed_fits": completed_fits, "fits": fits,
            "timing": {"elapsed_seconds": elapsed, "calibration_fit_seconds": fit_elapsed,
                       "subset_copy_seconds": staging_elapsed, "guard_seconds": guard_elapsed},
            "hashes": {"split_sha256": split_hash, "rule_sha256": _hash(chosen),
                       "candidate_rule_sha256": _hash(candidate)},
        })

    if split_policy == "time":
        return finish("unsupported_time_policy")
    tuning_counts = support["tuning"]
    if not len(tuning):
        return finish("tuning_empty_calibration")
    if min(tuning_counts) == 0:
        return finish("tuning_missing_class")
    if len(tuning) < config["min_tuning_samples"]:
        return finish("tuning_insufficient_samples")
    if min(tuning_counts) < config["min_tuning_class_count"]:
        return finish("tuning_insufficient_class_support")
    if any(min(fold["fit"]) == 0 for fold in support["folds"]):
        return finish("fit_missing_class")
    if not len(selection):
        return finish("selection_insufficient_samples")

    oof = np.full((n_rows, len(classes)), np.nan)
    jobs = [(np.asarray(fold["fit_rows"], dtype=np.int64),
             np.asarray(fold["held_rows"], dtype=np.int64), "oof") for fold in normalized_folds]
    jobs.append((tuning, selection, "selection"))
    for fit_index, (fit, held, kind) in enumerate(jobs):
        before_copy = now()
        if before_copy >= deadline:
            return finish("deadline_exhausted")
        X_fit = _subset(features, fit)
        y_fit = labels_array[fit].copy()
        X_predict = _subset(features, held)
        # Copying a large table is host work too. Re-read the same original
        # deadline after staging, so no callback starts on a stale allowance.
        before = now()
        staging_elapsed += before - before_copy
        remaining = deadline - before
        if remaining <= 0:
            return finish("deadline_exhausted")
        allowed_timeout = min(timeout, remaining)
        record = {"fit_index": fit_index, "kind": kind, "status": "started",
                  "fit_rows": len(fit), "predict_rows": len(held),
                  "timeout_seconds": allowed_timeout, "elapsed_seconds": 0.0}
        fits.append(record)
        try:
            result = fit_predict(X_fit, y_fit, X_predict, timeout_seconds=allowed_timeout,
                                 fit_index=fit_index)
        except Exception as error:
            after = now()
            elapsed = after - before
            fit_elapsed += elapsed
            reason = "callback_timeout" if isinstance(error, TimeoutError) else "callback_error"
            record.update(status=reason, elapsed_seconds=elapsed, error_type=type(error).__name__)
            return finish("deadline_exhausted" if after >= deadline else reason)
        # A returned malformed matrix is a contract failure, not a model-error
        # fallback. Validation is deliberately outside the callback try block.
        aligned = _prediction(result, len(held), classes)
        after = now()
        elapsed = after - before
        fit_elapsed += elapsed
        completed_fits += 1
        record.update(status="completed", elapsed_seconds=elapsed)
        if after >= deadline:
            record["status"] = "deadline_exceeded"
            return finish("deadline_exhausted")
        if elapsed >= allowed_timeout:
            record["status"] = "callback_timeout"
            return finish("callback_timeout")
        oof[held] = aligned
    if not np.isfinite(oof[joined]).all():
        # Coverage validation should make this unreachable; never tune partial
        # probabilities even if a future implementation changes the loop.
        raise OOFContractError("Incomplete OOF/selection predictions")
    before_guard = now()
    if before_guard >= deadline:
        return finish("deadline_exhausted")
    guarded = fit_guarded_decision_rule(
        oof[joined], labels_array[joined], classes,
        tuning_rows=np.arange(len(tuning)),
        selection_rows=np.arange(len(tuning), n_rows), options=deepcopy(config),
    )
    after_guard = now()
    guard_elapsed = after_guard - before_guard
    if after_guard >= deadline:
        return finish("deadline_exhausted", guarded)
    return finish(guarded["diagnostics"]["reason"], guarded)
