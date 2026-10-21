"""Literal, declarative CSV adapters and frozen outer splits for public data.

This module never fits a transformer or model, downloads data, or exposes the
complete source table to a solver. Numeric canonicalisation is used only for
duplicate detection; the feature CSVs retain their original literal values.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from agent.metrics import get_metric
from agent.task_spec import TaskSpec, load_task
from research_lab.integrity import file_hash, json_hash, safe_path, write_json


ADAPTER_VERSION = "literal-csv-v1"
ID_COLUMN = "record_id"
INPUT_PATHS = ("task.md", "inputs/train.csv", "inputs/test.csv",
               "inputs/data_dictionary.csv", "inputs/sample_submission.csv")


class IneligibleCase(ValueError):
    """A declared source/split cannot satisfy the native task contract."""


def _need(condition, message):
    if not condition:
        raise IneligibleCase(message)


def _names(value, name, *, empty=False):
    _need(isinstance(value, list) and (empty or value)
          and all(isinstance(item, str) and item.strip() for item in value),
          f"{name} must be a list of nonempty literal column names")
    _need(len(set(value)) == len(value), f"{name} contains duplicate names")
    return list(value)


def _id_column(record):
    name = record.get("id_rule", {}).get("column", ID_COLUMN)
    _need(isinstance(name, str) and name.strip() and name != record["target"], "Invalid technical ID column")
    return name


def _number(value, *, context):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise IneligibleCase(f"Non-numeric {context}: {value!r}") from error
    _need(result.is_finite(), f"Nonfinite {context}: {value!r}")
    return result


def _numeric_key(value, decimals=None):
    if value == "":
        return "missing"
    number = _number(value, context="numeric duplicate key")
    if decimals is not None:
        # A fixed decimal grid is declared before splitting. It is never fitted
        # to the observed range, labels, or the desired validation outcome.
        with localcontext() as context:
            context.prec = max(50, len(number.as_tuple().digits) + abs(number.adjusted()) + decimals + 4)
            number = number.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
    if number == 0:
        return "0"
    with localcontext() as context:
        context.prec = max(50, len(number.as_tuple().digits) + abs(number.adjusted()) + 4)
        return str(number.normalize())


def _read_source(path, receipt, record):
    adapter = record.get("adapter", {})
    _need(adapter.get("format", "csv") == "csv", "Only declared CSV adapters are supported")
    delimiter = adapter.get("delimiter", ",")
    _need(isinstance(delimiter, str) and len(delimiter) == 1, "CSV delimiter must be one character")
    encoding = adapter.get("encoding", "utf-8")
    source_sha = receipt.get("sha256", receipt.get("source_sha256"))
    _need(isinstance(source_sha, str) and len(source_sha) == 64,
          "Verified cache receipt has no source SHA-256")
    _need(file_hash(path) == source_sha, "Verified source bytes changed before preparation")
    with Path(path).open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        header = next(reader, [])
        source_rows = 0
        for row in reader:
            _need(len(row) == len(header), "Ragged source CSV row")
            source_rows += 1
    _need(header and len(set(header)) == len(header), "Source CSV has an empty/duplicate header")
    _need(all(isinstance(name, str) and name.strip() for name in header), "Source CSV has an empty column name")
    # Do not infer NA values or numeric-coded categories/classes. This is the
    # only source-table read; the raw table remains in the host-owned cache.
    frame = pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str,
                        keep_default_na=False, na_filter=False)
    _need(list(frame.columns) == header and len(frame) == source_rows and len(frame) >= 2,
          "Source CSV schema/rows are unsupported")
    _need(not frame.isna().any().any(), "Ragged source CSV contains absent cells")
    return frame, source_sha


def _adapt(frame, record):
    """Apply static schema rules without learning anything from feature values."""
    frame = frame.copy()
    ordinals = np.arange(len(frame), dtype=np.int64)
    adapter = record.get("adapter", {})
    selection = adapter.get("filter")
    if selection is not None:
        _need(isinstance(selection, dict) and selection.get("column") in frame
              and isinstance(selection.get("value"), str), "Unsupported literal source filter")
        keep = frame[selection["column"]].eq(selection["value"]).to_numpy()
        frame, ordinals = frame.loc[keep].copy(), ordinals[keep]
    frame = frame.reset_index(drop=True)
    _need(len(frame) >= 2, "Static source filter produced too few rows")
    target = record["target"]
    _need(target in frame, f"Declared target is absent: {target}")
    numeric = _names(record.get("numeric"), "numeric", empty=True)
    categorical = _names(record.get("categorical"), "categorical", empty=True)
    exclusions = _names(record.get("feature_exclusions", []), "feature_exclusions", empty=True)
    identifiers = _names(record.get("id_columns", []), "id_columns", empty=True)
    _need(not set(numeric) & set(categorical), "Numeric/categorical columns overlap")
    _need(target not in set(numeric + categorical + exclusions + identifiers),
          "Target must be separate from feature/exclusion/ID columns")
    _need(not set(numeric + categorical) & set(exclusions + identifiers),
          "Excluded leakage/ID column appears in features")
    _need(_id_column(record) not in frame, "Source conflicts with the reserved technical ID column")
    missing = adapter.get("missing_numeric", {})
    _need(isinstance(missing, dict) and set(missing) <= set(numeric),
          "Missing-value rules may name only declared numeric features")
    for name, tokens in missing.items():
        _need(name in frame and isinstance(tokens, list) and all(isinstance(x, str) for x in tokens),
              "Numeric missing markers must be explicit literal strings")
        frame[name] = frame[name].map(lambda value: "" if value in tokens else value)
    period = adapter.get("derived_period")
    if period is not None:
        _need(isinstance(period, dict) and period.get("kind") == "calendar_month",
              "Unsupported static derived period rule")
        name = period.get("column", "period")
        year, month = period.get("year_column"), period.get("month_column")
        _need(name not in frame and name in numeric and year in frame and month in frame,
              "Derived month period requires new numeric column and declared year/month sources")
        years = [_number(value, context="year period") for value in frame[year]]
        months = [_number(value, context="month period") for value in frame[month]]
        _need(all(y == y.to_integral_value() for y in years)
              and all(m == m.to_integral_value() and 1 <= m <= 12 for m in months),
              "Calendar month fields must be integral, with month in 1..12")
        frame[name] = [str(int(y * 12 + m)) for y, m in zip(years, months)]
    features = [name for name in frame if name in set(numeric + categorical)]
    _need(features and set(features) == set(numeric + categorical), "Declared features do not match source schema")
    _need(set(frame) == set(features + exclusions + identifiers + [target]),
          "Every source column must have an explicit feature, target, exclusion, or ID role")
    _need(set(exclusions + identifiers) <= set(frame), "An excluded/ID column is absent from the source")
    for name in numeric:
        for value in frame[name].unique():
            if value != "":
                _number(value, context=f"feature {name}")
    label_rule = record.get("label_rule", {})
    if isinstance(label_rule, dict) and label_rule.get("mapping") is not None:
        mapping = label_rule["mapping"]
        _need(record.get("task_type") == "classification" and isinstance(mapping, dict)
              and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in mapping.items()),
              "Label mapping must explicitly map nonempty classification strings; empty labels remain unsupported")
        frame[target] = frame[target].map(lambda value: mapping.get(value, value))
    return frame, ordinals, features, numeric, categorical


class _Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _groups(frame, features, numeric, rule):
    _need(isinstance(rule, dict), "group_rule must declare duplicate/entity handling")
    columns = _names(rule.get("columns", []), "group columns", empty=True)
    _need(set(columns) <= set(frame) and not set(columns) & {rule.get("target")}, "Unknown entity group column")
    decimals = rule.get("near_duplicate_decimals")
    _need(decimals is None or (type(decimals) is int and 0 <= decimals <= 15),
          "near_duplicate_decimals must be a fixed integer in 0..15")
    _need(rule.get("exact_features") is True, "Exact semantic feature duplicate grouping is mandatory")
    components, numeric = _Components(len(frame)), set(numeric)
    # Entities and duplicate rows are unioned, rather than concatenating entity
    # and feature keys: either relationship must prevent crossing the split.
    mechanisms = [(columns, None)] if columns else []
    mechanisms.append((features, None))
    if decimals is not None:
        mechanisms.append((features, decimals))
    for names, rounding in mechanisms:
        seen = {}
        for index, values in enumerate(frame.loc[:, names].itertuples(index=False, name=None)):
            key = tuple(_numeric_key(value, rounding) if name in numeric else value
                        for name, value in zip(names, values))
            if key in seen:
                components.union(index, seen[key])
            else:
                seen[key] = index
    return np.asarray([components.find(index) for index in range(len(frame))], dtype=np.int64)


def _classes(frame, record, metric):
    if not metric.classification:
        _need(not record.get("classes"), "Regression source cannot declare classes")
        for value in frame[record["target"]].unique():
            number = _number(value, context="regression target")
            _need(math.isfinite(float(number)), "Regression target is outside the native finite numeric range")
        return []
    labels = frame[record["target"]].tolist()
    _need(all(isinstance(label, str) and label.strip() for label in labels),
          "Native TaskSpec does not support empty classification labels; source is ineligible")
    declared = record.get("classes")
    classes = sorted(set(labels)) if declared is None else _names(declared, "classes")
    _need(len(classes) >= 2 and set(classes) == set(labels), "Declared literal classes differ from source labels")
    positive = record.get("positive_class")
    _need(positive is None or positive in classes, "Positive class must be a literal declared class")
    return classes


def _split(frame, ordinals, groups, record, seed, classes):
    rule = record.get("split_rule", {})
    kind, fraction = rule.get("kind"), rule.get("audit_fraction")
    _need(isinstance(fraction, (float, int)) and not isinstance(fraction, bool) and 0 < fraction < 1,
          "Outer audit fraction must lie strictly between zero and one")
    indices = np.arange(len(frame))
    metadata = {"kind": kind, "audit_fraction": fraction, "gap_periods": rule.get("gap_periods", 0)}
    purged = []
    if kind in {"group_random", "stratified_group"}:
        _need(len(set(groups)) >= 2, "Too few independent groups for the declared outer split")
        if kind == "stratified_group":
            folds = round(1 / fraction)
            _need(classes and folds >= 2 and math.isclose(1 / folds, fraction, abs_tol=1e-12),
                  "Stratified group fraction must be the reciprocal of a declared fold count")
            _need(len(set(groups)) >= folds, "Too few groups for the declared stratified fold count")
            labels = frame[record["target"]].tolist()
            _need(all(sum(label == wanted for label in labels) >= folds for wanted in classes),
                  "Insufficient class rows for the declared stratified fold count; no seed retries are permitted")
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            train, audit = next(splitter.split(indices, labels, groups))
        else:
            train, audit = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                                .split(indices, groups=groups))
        metadata["seed_use"] = "fixed first group split; no retries"
    elif kind == "source_order":
        _need(rule.get("gap_periods", 0) == 0, "Source-order proxy does not define timestamp gap periods")
        boundary = int(math.floor(len(frame) * (1 - fraction)))
        train, audit = indices[:boundary], indices[boundary:]
        metadata.update({"seed_use": "model randomness only; outer split fixed",
                         "time_interpretation": "publisher source-row order proxy; timestamps and entity independence unavailable"})
    elif kind == "time":
        column, gap = rule.get("time_column"), rule.get("gap_periods", 0)
        _need(column in frame and column in record["numeric"], "Time split requires a declared numeric period feature")
        _need(type(gap) is int and gap >= 0, "Time gap_periods must be a nonnegative integer")
        values = [_number(value, context="time period") for value in frame[column]]
        periods = sorted(set(values))
        held_periods = max(1, int(math.ceil(len(periods) * fraction)))
        boundary = len(periods) - held_periods
        _need(boundary - gap >= 1, "Too few past periods for the declared chronological holdout/gap")
        train_periods, audit_periods = set(periods[:boundary-gap]), set(periods[boundary:])
        train = np.asarray([i for i in indices if values[i] in train_periods], dtype=np.int64)
        audit = np.asarray([i for i in indices if values[i] in audit_periods], dtype=np.int64)
        purged = [int(i) for i in indices if values[i] not in train_periods | audit_periods]
        # Native automatic time validation only handles 2..100 distinct training
        # periods. Explicitly reject, rather than silently running random CV.
        _need(2 <= len(train_periods) <= 100, "Native automatic time validation needs 2..100 training periods")
        train = np.asarray(sorted(train, key=lambda i: (values[i], int(ordinals[i]))), dtype=np.int64)
        audit = np.asarray(sorted(audit, key=lambda i: (values[i], int(ordinals[i]))), dtype=np.int64)
        metadata.update({"time_column": column, "train_max": str(max(train_periods)),
                         "audit_min": str(min(audit_periods)), "train_periods": len(train_periods),
                         "audit_periods": len(audit_periods), "seed_use": "model randomness only; outer split fixed"})
    else:
        raise IneligibleCase(f"Unsupported declared split kind: {kind!r}")
    if kind in {"time", "source_order"}:
        train_groups = set(groups[train])
        overlaps = [int(i) for i in audit if groups[i] in train_groups]
        purged.extend(overlaps)
        audit = np.asarray([i for i in audit if groups[i] not in train_groups], dtype=np.int64)
        metadata["cross_boundary_group_rows_purged_from_audit"] = len(overlaps)
    _need(len(train) and len(audit), "Frozen outer split became empty after duplicate/entity purge")
    _need(not set(groups[train]) & set(groups[audit]), "Group leakage across the frozen outer split")
    if kind == "time":
        actual_train, actual_audit = {values[i] for i in train}, {values[i] for i in audit}
        _need(max(actual_train) < min(actual_audit), "Chronological split is not strictly past-to-future")
        metadata.update(train_max=str(max(actual_train)), audit_min=str(min(actual_audit)),
                        train_periods=len(actual_train), audit_periods=len(actual_audit))
    metadata.update(train_groups=len(set(groups[train])), audit_groups=len(set(groups[audit])))
    if classes:
        target = record["target"]
        _need(set(frame.iloc[train][target]) == set(classes) and set(frame.iloc[audit][target]) == set(classes),
              "Frozen split lacks declared class support; no seed retries are permitted")
    return train, audit, np.asarray(sorted(set(purged)), dtype=np.int64), metadata


def _contract(record, classes, features, numeric, categorical, task_path):
    metric = get_metric(record["metric"])
    _need(record.get("task_type") == metric.task_type, "Declared task type conflicts with the metric")
    target, positive, id_column = record["target"], record.get("positive_class"), _id_column(record)
    pairs = []
    if metric.prediction_kind == "probabilities":
        order = record.get("probability_order")
        if order is None and len(classes) == 2 and metric.name == "ROC AUC":
            _need(positive in classes, "Binary ROC AUC requires an explicit positive class")
            pairs = [[target, positive]]
        else:
            order = classes if order is None else _names(order, "probability_order")
            _need(set(order) == set(classes), "Probability order must cover every literal class exactly once")
            _need(id_column not in classes, "Class label conflicts with the technical ID probability column")
            pairs = [[label, label] for label in order]
    output = [id_column, *(column for column, _ in pairs)] if pairs else [id_column, target]
    spec = TaskSpec(task_path, task_path.parent / "inputs", id_column, target, tuple(features),
                    tuple(name for name in features if name in numeric),
                    tuple(name for name in features if name in categorical), tuple(classes), tuple(output),
                    tuple(tuple(pair) for pair in pairs), positive, metric)
    contract = spec.to_dict()
    contract["prediction_kind"] = spec.prediction_kind
    return spec, contract


def _technical_ids(dataset_id, source_sha, ordinals):
    return ["r_" + hashlib.sha256(json.dumps([dataset_id, source_sha, int(index)],
                                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            for index in ordinals]


def _write_task(directory, frame, train_indices, audit_indices, ids, spec, record):
    inputs = directory / "inputs"
    inputs.mkdir()
    train = frame.iloc[train_indices].loc[:, list(spec.features)].copy()
    test = frame.iloc[audit_indices].loc[:, list(spec.features)].copy()
    train.insert(0, spec.id_column, [ids[i] for i in train_indices])
    test.insert(0, spec.id_column, [ids[i] for i in audit_indices])
    train[spec.target_column] = frame.iloc[train_indices][spec.target_column].tolist()
    train.to_csv(inputs / "train.csv", index=False, lineterminator="\n")
    test.to_csv(inputs / "test.csv", index=False, lineterminator="\n")
    descriptions = [{"feature": spec.id_column, "role": "id", "dtype": "category",
                     "description": "Opaque technical record identifier; preserve literally; never use as a feature"}]
    descriptions.extend({"feature": name, "role": "feature", "dtype": "numeric" if name in spec.numeric else "category",
                         "description": "Literal source value; numeric empty cells are missing; categorical values are literal"}
                        for name in spec.features)
    descriptions.append({"feature": spec.target_column, "role": "target",
                         "dtype": "category" if spec.classification else "numeric", "description": "Prediction target"})
    pd.DataFrame(descriptions).to_csv(inputs / "data_dictionary.csv", index=False, lineterminator="\n")
    sample = {spec.id_column: test[spec.id_column].tolist()}
    if spec.prediction_kind == "probabilities":
        for column, _ in spec.probability_columns:
            sample[column] = [1 / len(spec.classes)] * len(test)
    else:
        sample[spec.target_column] = [spec.classes[0] if spec.classes else "0"] * len(test)
    pd.DataFrame(sample).loc[:, list(spec.output_columns)].to_csv(inputs / "sample_submission.csv", index=False, lineterminator="\n")
    declaration = "ROC AUC (OVR macro)" if spec.metric.name == "ROC AUC" and len(spec.classes) > 2 else spec.metric.name
    lines = [f"# Public tabular task: {record['dataset_id']}",
             "Use inputs/train.csv, inputs/test.csv, inputs/data_dictionary.csv and inputs/sample_submission.csv.",
             f"Predict {spec.target_column}; preserve {spec.id_column} literally and keep the test row order.",
             "Task type: " + spec.metric.task_type, "Primary metric: " + declaration]
    if spec.classes:
        lines.append("Class order: " + json.dumps(spec.classes, ensure_ascii=False))
    if spec.positive_class is not None:
        lines.append("Positive class: " + json.dumps(spec.positive_class, ensure_ascii=False))
    kind = record["split_rule"]["kind"]
    if kind == "time":
        lines.append(f"Predict later periods using earlier periods. The numeric time column is {record['split_rule']['time_column']}. Keep validation chronological.")
    elif kind == "source_order":
        lines.append("Rows follow the publisher's chronological source-order proxy. Full timestamps and client identifiers are unavailable; do not claim exact client independence.")
    else:
        lines.append("Keep identical and declared near-duplicate raw feature rows together during validation, including contradictory labels.")
    if record.get("task_description"):
        _need(isinstance(record["task_description"], str), "task_description must be literal text")
        lines.append(record["task_description"])
    (directory / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_case(record, cache, case_dir, audit_dir, seed):
    """Prepare one immutable solver template and a separate host-only audit.

    ``cache`` must already contain verified source bytes. No network fallback is
    possible here. A failed fixed split is explicitly ineligible; seeds are
    never searched. Both labels and source ordinals remain solely in the audit.
    """
    from research_lab.fetch import cached_member

    _need(isinstance(record, dict) and isinstance(record.get("dataset_id"), str), "Dataset record is required")
    _need(type(seed) is int and 0 <= seed <= 2**32 - 1, "Seed must be a 32-bit unsigned integer")
    case_dir, audit_dir = safe_path(case_dir), safe_path(audit_dir)
    _need(not case_dir.exists(), "Task template directory must be new")
    _need(not audit_dir.is_relative_to(case_dir) and not case_dir.is_relative_to(audit_dir),
          "Audit truth must be outside the solver template tree")
    dataset_id = record["dataset_id"]
    _need(dataset_id and all(c.isalnum() or c in "_-" for c in dataset_id), "Unsafe dataset_id")
    audit_path = audit_dir / f"{dataset_id}__seed{seed}.json"
    _need(not audit_path.exists(), "Audit evidence already exists; do not overwrite a frozen case")
    source_path, receipt = cached_member(record, cache)
    source, source_sha = _read_source(source_path, receipt, record)
    source_rows = len(source)
    frame, ordinals, features, numeric, categorical = _adapt(source, record)
    del source
    metric = get_metric(record["metric"])
    classes = _classes(frame, record, metric)
    _need(record["target"] not in record["group_rule"].get("columns", []), "Target cannot be an entity group key")
    groups = _groups(frame, features, numeric, record["group_rule"])
    train, audit, purged, split_metadata = _split(frame, ordinals, groups, record, seed, classes)
    ids = _technical_ids(dataset_id, source_sha, ordinals)
    _need(len(set(ids)) == len(ids), "Technical record ID collision")
    task_path = case_dir / "task.md"
    expected, contract = _contract(record, classes, features, numeric, categorical, task_path)
    rows = {"source": source_rows, "selected": len(frame), "train": len(train), "audit": len(audit), "purged": len(purged)}
    split_hashes = {"train": json_hash([int(ordinals[i]) for i in train]),
                    "audit": json_hash([int(ordinals[i]) for i in audit]),
                    "groups": json_hash([[int(ordinal), int(group)] for ordinal, group in zip(ordinals, groups)])}
    metadata = {"adapter_version": ADAPTER_VERSION, "adapter_sha256": file_hash(Path(__file__)), "source_sha256": source_sha,
                "source_bytes": Path(source_path).stat().st_size, "source_receipt_sha256": json_hash(receipt),
                "record_sha256": json_hash(record), "split_rule": record["split_rule"],
                "group_rule": record["group_rule"], "split": split_metadata,
                "excluded_features": record.get("feature_exclusions", []),
                "id_columns_excluded": record.get("id_columns", []),
                "id_rule": "sha256(JSON [dataset_id, verified source SHA-256, zero-based original source ordinal]); no target feature",
                "class_rule": "declared literal order" if record.get("classes") is not None else "lexical literal source class names; no label coercion"}
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".lab-case-", dir=case_dir.parent))
    try:
        _write_task(temporary, frame, train, audit, ids, expected, record)
        native = load_task(temporary / "task.md")
        actual_contract = native.to_dict()
        expected_contract = expected.to_dict()
        for name in ("task_path", "data_dir"):
            actual_contract.pop(name)
            expected_contract.pop(name)
        _need(actual_contract == expected_contract, "Native TaskSpec preflight differs from declared adapter contract")
        if record["split_rule"]["kind"] == "time":
            from agent.automl import choose_validation
            train_frame = pd.read_csv(temporary / "inputs/train.csv", dtype=str, keep_default_na=False)
            test_frame = pd.read_csv(temporary / "inputs/test.csv", dtype=str, keep_default_na=False)
            choice, _ = choose_validation(train_frame, test_frame, list(native.features), list(native.numeric), native.classification)
            _need(choice == {"strategy": "time", "time_column": record["split_rule"]["time_column"]},
                  "Native automatic validation does not preserve the declared chronological period")
        temporary.rename(case_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    # No raw source table, original index, truth path, or audit targets have been
    # written into the template. Only this host-owned sidecar receives truth.
    hashes = {name: file_hash(case_dir / name) for name in INPUT_PATHS}
    answers = frame.iloc[audit][record["target"]].tolist()
    if not metric.classification:
        answers = [float(_number(value, context="audit target")) for value in answers]
    private = {"dataset_id": dataset_id, "seed": seed, "metric": metric.name, "contract": contract,
               "audit_actuals": answers, "test_ids": [ids[i] for i in audit],
               "input_paths": list(INPUT_PATHS), "input_sha256": hashes,
               "rows": rows, "split_hashes": split_hashes, "metadata": metadata,
               "source_ordinals": {"train": [int(ordinals[i]) for i in train],
                                   "audit": [int(ordinals[i]) for i in audit],
                                   "purged": [int(ordinals[i]) for i in purged]},
               "group_ids": {"train": [int(groups[i]) for i in train], "audit": [int(groups[i]) for i in audit]}}
    write_json(audit_path, private)
    return {"dataset_id": dataset_id, "seed": seed, "task_path": str(task_path),
            "task_template": str(case_dir), "audit_path": str(audit_path), "audit_sha256": file_hash(audit_path),
            "input_hashes": hashes, "spec": expected.to_dict(), "rows": rows,
            "split_hashes": split_hashes, "metadata": metadata}
