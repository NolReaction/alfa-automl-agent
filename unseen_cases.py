"""Versioned, predeclared tabular qualification fixtures (no solver imports).

Audit answers are returned in memory. ``materialize_case`` writes only the
condition and four solver input CSVs; ordinary processes still share the host
filesystem, so this is a data-flow protocol rather than an OS sandbox.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG_VERSION = 1
DEFAULT_SEED = 81221
DEFAULT_TRAIN_ROWS = 300
DEFAULT_AUDIT_ROWS = 80
LABELS = ("01", "NA", "κλάση", "class c")
_PROBABILITY_METRICS = {"ROC AUC", "LogLoss"}
_CLASSIFICATION_METRICS = _PROBABILITY_METRICS | {"Accuracy", "Macro F1", "Weighted F1"}


def _definition(identifier, metric, mechanism, *, family=None, kind="valid", full=False,
                default=False, validation=None, train_rows=300, audit_rows=80, **extra):
    return {"id": identifier, "kind": kind, "metric": metric,
            "expected_status": "refusal" if kind == "negative" else "model_or_prior",
            "full_run": bool(full), "default_cli": bool(default),
            "validation": validation or {"strategy": "random"},
            "train_rows": train_rows, "audit_rows": audit_rows,
            "offline_seconds": 35, "timeout_seconds": 110,
            "mechanism": mechanism, "family": family or identifier,
            "provenance": "new_generator", **extra}


# The complete catalogue and all sizes were declared before any solver score.
# Contract variants share an explicitly named family; they are not claimed to
# be independent statistical task families.
_DEFINITIONS = (
    _definition("flow_rmse", "RMSE", "signed pressure drop / resistance with an on/off valve",
                family="hydraulic_flow", full=True, default=True),
    _definition("inventory_mae", "MAE", "compound Poisson demand minus planned deliveries",
                family="inventory_counts", full=True),
    _definition("fatigue_mse", "MSE", "positive overload fatigue with a temperature denominator",
                family="fatigue_threshold", full=True),
    _definition("binding_r2", "R2", "saturating dose binding with a discrete inhibitor",
                family="binding_kinetics", full=True),
    _definition("logic_accuracy", "Accuracy", "three-valued Boolean routing circuit",
                family="routing_logic", full=True, default=True),
    _definition("rings_macro_f1", "Macro F1", "nonconvex concentric radial bands",
                family="radial_bands", full=True),
    _definition("cascade_weighted_f1", "Weighted F1", "hierarchical alarm cascade with a rare terminal state",
                family="alarm_cascade", full=True),
    _definition("survival_binary_auc", "ROC AUC", "Weibull event time observed before a fixed horizon",
                family="survival_hazard", full=True),
    _definition("count_binary_logloss", "LogLoss", "Poisson event count crossing a threshold",
                family="count_events", full=True),
    _definition("checker_multiclass_logloss", "LogLoss", "four-colour periodic checkerboard cells",
                family="checkerboard", full=True, default=True),
    _definition("checker_multiclass_auc", "ROC AUC", "checkerboard with explicit OVR macro averaging",
                family="checkerboard", full=True),
    _definition("signed_flow_unicode_schema", "RMSE", "hydraulic flow, Unicode columns, target-first output",
                family="hydraulic_flow", unicode_schema=True),
    _definition("missing_markers_inventory", "MAE", "inventory counts with literal missing-like categories",
                family="inventory_counts", missing_markers=True),
    _definition("literal_labels_accuracy", "Accuracy", "routing circuit with exact Unicode and padded labels",
                family="routing_logic", padded_labels=True),
    _definition("reverse_positive_binary_auc", "ROC AUC", "survival risk with NA explicitly the positive label",
                family="survival_hazard", reverse_positive=True, full=True),
    _definition("probability_columns_permuted_logloss", "LogLoss", "checkerboard, reordered declarations and probability columns",
                family="checkerboard", permuted_classes=True),
    _definition("future_inventory_mae", "MAE", "future-period inventory with cyclic weekday demand",
                family="temporal_inventory", validation={"strategy": "time", "time_column": "period"}),
    _definition("grouped_logic_macro_f1", "Macro F1", "repeated routing-circuit observations with contradictory copies",
                family="grouped_routing", validation={"strategy": "grouped"}),
    _definition("numeric_all_missing_rmse", "RMSE", "flow with a completely missing numeric channel and a constant",
                family="hydraulic_flow", all_missing=True, full=True),
    _definition("high_cardinality_logloss", "LogLoss", "checkerboard with many categorical equipment identifiers",
                family="checkerboard", kind="difficult", high_cardinality=True),
    _definition("tiny_minority_weighted_f1", "Weighted F1", "alarm cascade with one training observation of a minority class",
                family="alarm_cascade", kind="difficult", tiny_minority=True, train_rows=40, audit_rows=20),
    _definition("impossible_group_split", "Accuracy", "each literal class confined to one identical-feature group",
                family="grouped_routing", kind="difficult", impossible_groups=True,
                validation={"strategy": "grouped"}, train_rows=40, audit_rows=20),
    *(_definition(identifier, "LogLoss" if identifier in {"ambiguous_positive", "ambiguous_probability_mapping"} else "RMSE",
                  mechanism, kind="negative", full=True, family="negative_contract")
      for identifier, mechanism in (
          ("missing_metric", "no explicit primary metric"),
          ("conflicting_metric", "two contradictory primary declarations"),
          ("ambiguous_positive", "single binary probability column without a positive class"),
          ("ambiguous_probability_mapping", "multiclass columns do not identify their classes"),
          ("duplicate_test_ids", "two test records have the same literal ID"),
          ("missing_train_target", "the declared target column is absent from train"),
          ("test_schema_mismatch", "one declared feature is absent from test"),
          ("malformed_csv", "unterminated quoted CSV field"),
          ("missing_test_id", "one test ID is empty"),
          ("task_type_conflict", "regression metric with an explicit classification task type"),
      )),
    _definition("markdown_literal_labels_accuracy", "Accuracy", "metamorphic routing labels containing literal Markdown punctuation",
                kind="metamorphic", family="routing_logic", provenance="unseen_cases.logic_accuracy",
                markdown_labels=True),
    _definition("known_linear_literal_ids", "RMSE", "metamorphic exact-ID and row-order variant of benchmark linear_regression",
                kind="metamorphic", family="known_linear", train_rows=600, audit_rows=150,
                provenance="benchmark_cases.linear_regression"),
    _definition("known_binary_permuted_features", "LogLoss", "metamorphic train/test feature-order permutation of benchmark imbalanced_binary",
                kind="metamorphic", family="known_binary", train_rows=600, audit_rows=150,
                provenance="benchmark_cases.imbalanced_binary"),
    _definition("wide_numeric_probe", "RMSE", "flow with 80 nuisance numeric channels",
                family="hydraulic_flow", kind="scale", train_rows=1200, audit_rows=240,
                extra_numeric=80, offline_seconds=20, timeout_seconds=90),
    _definition("high_cardinality_probe", "LogLoss", "checkerboard with 800 equipment categories",
                family="checkerboard", kind="scale", high_cardinality=True,
                train_rows=1600, audit_rows=320, offline_seconds=20, timeout_seconds=90),
    _definition("censored_budget_probe", "Accuracy", "routing circuit under an explicitly short external observation window",
                family="routing_logic", kind="scale", train_rows=1600, audit_rows=320,
                extra_numeric=20, offline_seconds=15, timeout_seconds=8),
)
# Verify catalogue uniqueness at import, without generating or scoring data.
if len({item["id"] for item in _DEFINITIONS}) != len(_DEFINITIONS):
    raise RuntimeError("Qualification case IDs must be unique")
_DEFINITION_BY_ID = {item["id"]: item for item in _DEFINITIONS}


def case_catalog() -> list[dict]:
    """Return independent JSON-compatible catalogue copies in declared order."""
    return deepcopy(list(_DEFINITIONS))


@dataclass(frozen=True)
class UnseenCase:
    definition: dict
    seed: int
    train: pd.DataFrame
    test: pd.DataFrame
    audit_actuals: tuple
    contract: dict
    task_text: str

    @property
    def case_id(self):
        return self.definition["id"]


def _ids(prefix, count):
    values = [f"{prefix}-{index:06d}" for index in range(count)]
    if prefix == "test":
        specials = ["000007", "NA", "строка 4", "id,comma", 'id "quoted"', " id padded "]
        values[:min(count, len(specials))] = specials[:count]
    return values


def _numeric_family(family, count, rng):
    if family in {"hydraulic_flow", "fatigue_threshold", "binding_kinetics"}:
        pressure_a, pressure_b = rng.uniform(10, 90, (2, count))
        resistance = rng.uniform(.3, 3.5, count)
        valve = rng.integers(0, 2, count)
        x = pd.DataFrame({"pressure A": pressure_a, "pressure B": pressure_b,
                          "resistance": resistance, "valve": valve,
                          "tiny scale": resistance * 1e-8, "large scale": pressure_a * 1e7})
        if family == "hydraulic_flow":
            y = valve * (pressure_a - pressure_b) / (1 + resistance) - 4
            y += rng.normal(0, .12, count)
        elif family == "fatigue_threshold":
            overload = np.maximum(pressure_a - pressure_b, 0)
            y = overload ** 2 / (15 + 7 * resistance) - 8 + rng.uniform(-.2, .2, count)
        else:
            dose = pressure_a / 8
            y = (40 * dose / (dose + 4)) * (2 * valve - 1) - 3 * resistance
            y += rng.normal(0, .1, count)
        return x, y
    day = rng.integers(0, 7, count)
    promotion = rng.integers(0, 2, count)
    delivery = rng.integers(3, 17, count)
    x = pd.DataFrame({"weekday": day, "promotion": promotion, "delivery units": delivery,
                      "warehouse": rng.choice(["east", "west", "NA"], count)})
    intensity = 2 + 1.5 * (day >= 4) + 4 * promotion
    y = rng.poisson(intensity).astype(float) - delivery
    if family == "temporal_inventory":
        n_train = count - 80
        # The caller replaces periods for non-default sizes as needed.
        x["period"] = np.concatenate([np.repeat(np.arange(1, 6), n_train // 5),
                                      np.repeat(6, count - (n_train // 5) * 5)])
        y += .4 * x["period"].to_numpy()
    return x, y


def _classification_family(family, count, rng, classes):
    labels = np.asarray(classes)
    if family in {"survival_hazard", "count_events"}:
        load = rng.uniform(.1, 3.5, count)
        temperature = rng.uniform(5, 55, count)
        exposure = rng.integers(1, 6, count)
        x = pd.DataFrame({"load": load, "temperature": temperature, "exposure": exposure,
                          "instrument": rng.choice(["NA", "null", "sensor x"], count)})
        if family == "survival_hazard":
            scale = 3.5 / (load + .1) + temperature / 35
            event_time = rng.weibull(1.7, count) * scale
            state = (event_time <= 1.8).astype(int)
        else:
            events = rng.poisson(.35 * exposure + .7 * (temperature > 30) + .2 * load)
            state = (events >= 2).astype(int)
        return x, labels[state]
    if family == "radial_bands":
        a, b = rng.uniform(-2, 2, (2, count))
        radius = np.sqrt(a*a + b*b)
        state = np.digitize(radius, [.8, 1.4, 2.0])
        x = pd.DataFrame({"horizontal": a, "vertical": b,
                          "probe": rng.choice(["A", "B", "NA"], count)})
        return x, labels[state]
    if family == "checkerboard":
        a, b = rng.uniform(-2, 2, (2, count))
        state = (np.floor((a + 2) * 1.5).astype(int) + np.floor((b + 2) * 1.5).astype(int)) % 4
        x = pd.DataFrame({"ось x": a, "axis y": b,
                          "station": rng.choice(["NA", "N/A", "null", "None", "nan", ""], count)})
        return x, labels[state]
    a, b, c = rng.integers(0, 2, (3, count))
    if family == "alarm_cascade":
        a = (rng.random(count) < .55).astype(int)
        b = (rng.random(count) < .35).astype(int)
        c = (rng.random(count) < .25).astype(int)
        state = np.where(a == 0, 0, np.where((b + c) == 2, 2, 1))
    else:
        state = (np.logical_xor(a, b).astype(int) + c) % 3
    x = pd.DataFrame({"switch A": a, "switch B": b, "switch C": c,
                      "nuisance": rng.uniform(-1, 1, count)})
    return x, labels[state]


def _known_case(definition, seed):
    # Explicit metamorphic controls derived from an existing, already studied
    # benchmark. They are never labelled new independent generators.
    from benchmark_cases import make_case
    known = make_case("linear_regression" if definition["family"] == "known_linear" else "imbalanced_binary", seed)
    train, test = known.train.copy(), known.test.copy()
    rename = {"row_key": "record id", "outcome": "response"}
    train, test = train.rename(columns=rename), test.rename(columns=rename)
    test["record id"] = _ids("test", len(test))
    order = np.random.default_rng(seed + 31).permutation(len(test))
    test = test.iloc[order].reset_index(drop=True)
    actual = tuple(known.audit_actuals[index] for index in order)
    if definition["family"] == "known_binary":
        features = [name for name in test if name != "record id"]
        test = test.loc[:, ["record id", *features[::-1]]]
    return train, test, actual, known.classes, known.positive_class, "record id", "response"


def generate_case(case_id: str, seed: int | None = None) -> UnseenCase:
    """Generate one pinned fixture. No model, score, or audit result is an input."""
    if case_id not in _DEFINITION_BY_ID:
        raise ValueError("Unknown qualification case: " + str(case_id))
    seed = DEFAULT_SEED if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**32 - 32:
        raise ValueError("Seed must be an integer from 0 through 4294967263")
    definition = deepcopy(_DEFINITION_BY_ID[case_id])
    rng = np.random.default_rng(seed)
    n_train, n_audit = definition["train_rows"], definition["audit_rows"]
    metric, family = definition["metric"], definition["family"]
    classification = metric in _CLASSIFICATION_METRICS
    binary = family in {"survival_hazard", "count_events"} or case_id == "ambiguous_positive"
    classes = (("NA", "01") if binary else LABELS[:3] if family in {"routing_logic", "alarm_cascade", "grouped_routing"}
               else LABELS) if classification else ()
    if definition.get("padded_labels"):
        classes = ("01", "NA", " мягкий класс ")
    if definition.get("markdown_labels"):
        classes = ("__ALFA__", "**x**", "`y`")
    positive = classes[1] if binary else None
    if definition.get("reverse_positive"):
        # The event remains the second generated label, now literally NA.
        classes, positive = ("01", "NA"), "NA"
    id_column = "ключ записи" if definition.get("unicode_schema") else "record id"
    target = "ответ" if definition.get("unicode_schema") else "response"
    if definition["provenance"].startswith("benchmark_cases."):
        train, test, actual, classes, positive, id_column, target = _known_case(definition, seed)
    else:
        if definition["kind"] == "negative":
            if case_id == "ambiguous_positive":
                family, classes, positive = "survival_hazard", ("NA", "01"), None
            elif case_id == "ambiguous_probability_mapping":
                family, classes = "checkerboard", LABELS
            else:
                family = "hydraulic_flow"
        total = n_train + n_audit
        if classification:
            frame, y = _classification_family(family, total, rng, classes)
        else:
            frame, y = _numeric_family(family, total, rng)
        if definition.get("impossible_groups"):
            frame = pd.DataFrame({"group coordinate": np.tile([0, 1], total // 2)})
            classes = ("NA", "01")
            y = np.asarray(classes)[frame["group coordinate"].to_numpy()]
        if family == "grouped_routing" and not definition.get("impossible_groups"):
            groups = n_train // 4
            unique, base = _classification_family("routing_logic", groups + n_audit, rng, classes)
            repeated = unique.iloc[:groups].loc[unique.index[:groups].repeat(4)].reset_index(drop=True)
            frame = pd.concat([repeated, unique.iloc[groups:]], ignore_index=True)
            y = np.concatenate([np.repeat(base[:groups], 4), base[groups:]])
            for group in range(0, groups, 7):
                index = classes.index(y[group*4+3])
                y[group*4+3] = classes[(index+1) % len(classes)]
        if definition.get("tiny_minority"):
            classes = ("NA", "01")
            y = np.asarray(["NA"] * total, dtype=object)
            y[n_train-1] = "01"
            y[n_train::4] = "01"
        if definition.get("all_missing"):
            frame["empty channel"] = ""
            frame["empty category"] = ""
            frame["constant"] = 7
        if definition.get("missing_markers"):
            frame["literal category"] = np.resize(np.asarray(["NA", "N/A", "null", "None", "nan", "", "駅"], dtype=object), total)
        if definition.get("high_cardinality"):
            frame["equipment"] = [f"unit-{value:04d}" for value in rng.integers(0, max(100, n_train//2), total)]
            frame.loc[n_train:n_train+9, "equipment"] = "unseen-equipment"
        for column in range(definition.get("extra_numeric", 0)):
            frame[f"nuisance {column}"] = rng.normal(size=total)
        train, test = frame.iloc[:n_train].copy(), frame.iloc[n_train:].copy()
        train.insert(0, id_column, _ids("train", n_train))
        test.insert(0, id_column, _ids("test", n_audit))
        train[target] = list(y[:n_train])
        order = rng.permutation(n_audit)
        actual = tuple(value.item() if isinstance(value, np.generic) else value for value in np.asarray(y[n_train:])[order])
        test = test.iloc[order].reset_index(drop=True)
        train = train.reset_index(drop=True)
    if classes and set(train[target].tolist()) != set(classes):
        raise ValueError("Pinned generator must contain all declared training classes; no seed search is permitted")
    if definition.get("permuted_classes"):
        classes = tuple(classes[index] for index in (2, 0, 3, 1))
    probability = metric in _PROBABILITY_METRICS
    if probability and len(classes) > 2:
        probability_columns = [(label, label) for label in (classes[-1], *classes[:-1])]
        output_columns = [probability_columns[0][0], id_column, *[column for column, _ in probability_columns[1:]]]
    elif probability:
        probability_columns = [(target, positive)]
        output_columns = [id_column, target]
    else:
        probability_columns = []
        output_columns = [target, id_column] if definition.get("unicode_schema") else [id_column, target]
    contract = {"id_column": id_column, "target_column": target, "output_columns": output_columns,
                "classes": list(classes), "probability_columns": [list(pair) for pair in probability_columns],
                "positive_class": positive, "prediction_kind": "probabilities" if probability else "labels" if classes else "regression"}
    declaration = "ROC AUC (OVR macro)" if metric == "ROC AUC" and len(classes) > 2 else metric
    lines = [f"# Qualification {CATALOG_VERSION}: {case_id}", definition["mechanism"],
             "Use inputs/train.csv, inputs/test.csv, inputs/data_dictionary.csv and inputs/sample_submission.csv.",
             f"Predict {target}; preserve {id_column} literally and retain the test row order.",
             "Task type: " + ("classification" if classes else "regression"),
             "Primary metric: " + declaration]
    if classes:
        lines.append("Class order: " + json.dumps(classes, ensure_ascii=False))
    if positive is not None:
        lines.append("Positive class: " + json.dumps(positive, ensure_ascii=False))
    if definition["validation"]["strategy"] == "time":
        lines.append("Predict the future period. Validation must use earlier periods to predict later periods; do not use random split.")
    if definition["validation"]["strategy"] == "grouped":
        lines.append("Keep identical raw-feature rows together during validation, including contradictory labels.")
    if case_id == "missing_metric":
        lines = [line for line in lines if not line.startswith("Primary metric:")]
    elif case_id == "conflicting_metric":
        lines.append("Primary metric: MAE")
    elif case_id == "task_type_conflict":
        lines = ["Task type: classification" if line == "Task type: regression" else line for line in lines]
    return UnseenCase(definition, seed, train, test, actual, contract, "\n".join(lines) + "\n")


def _write_inputs(case: UnseenCase, task_dir: Path):
    inputs = task_dir / "inputs"
    inputs.mkdir()
    id_column, target = case.contract["id_column"], case.contract["target_column"]
    train, test = case.train.copy(), case.test.copy()
    if case.case_id == "missing_train_target":
        train = train.drop(columns=[target])
    if case.case_id == "duplicate_test_ids":
        test.loc[1, id_column] = test.loc[0, id_column]
    if case.case_id == "missing_test_id":
        test.loc[1, id_column] = ""
    if case.case_id == "test_schema_mismatch":
        test = test.drop(columns=[next(name for name in test if name != id_column)])
    train.to_csv(inputs / "train.csv", index=False, lineterminator="\n", float_format="%.17g")
    test.to_csv(inputs / "test.csv", index=False, lineterminator="\n", float_format="%.17g")
    if case.case_id == "malformed_csv":
        with (inputs / "train.csv").open("a", encoding="utf-8", newline="") as stream:
            stream.write('"unterminated qualification field\n')
    metadata = [{"feature": id_column, "role": "id", "dtype": "category", "description": "Literal record ID"}]
    for name in case.train:
        if name not in {id_column, target}:
            numeric = pd.api.types.is_numeric_dtype(case.train[name]) or name == "empty channel"
            metadata.append({"feature": name, "role": "feature", "dtype": "numeric" if numeric else "category",
                             "description": "Empty cells are missing; all nonempty categorical strings are literal values"})
    metadata.append({"feature": target, "role": "target", "dtype": "category" if case.contract["classes"] else "numeric",
                     "description": "Prediction target"})
    pd.DataFrame(metadata).to_csv(inputs / "data_dictionary.csv", index=False, lineterminator="\n")
    template = pd.DataFrame({id_column: case.test[id_column].tolist()})
    if case.contract["prediction_kind"] == "probabilities":
        for column, label in case.contract["probability_columns"]:
            template[column] = .5 if len(case.contract["probability_columns"]) == 1 else 1 / len(case.contract["classes"])
    else:
        template[target] = case.contract["classes"][0] if case.contract["classes"] else 0.0
    template = template.loc[:, case.contract["output_columns"]]
    if case.case_id == "ambiguous_probability_mapping":
        template = template.rename(columns={column: f"p_{index}" for index, (column, _) in enumerate(case.contract["probability_columns"])})
    template.to_csv(inputs / "sample_submission.csv", index=False, lineterminator="\n", float_format="%.17g")


def materialize_case(case_id: str, task_dir: Path, seed: int | None = None) -> dict:
    """Write a fresh solver-only directory and return the private audit in memory.

    The caller must store any returned audit answers outside ``task_dir``. No
    audit answer path appears in task.md, CSV metadata, or solver arguments.
    Existing directories, including old submissions, are never reused.
    """
    case = generate_case(case_id, seed)
    task_dir = Path(task_dir).expanduser().resolve()
    task_dir.mkdir(parents=True, exist_ok=False)
    _write_inputs(case, task_dir)
    task_path = task_dir / "task.md"
    task_path.write_text(case.task_text, encoding="utf-8", newline="\n")
    inputs = ["task.md", *(f"inputs/{name}" for name in
               ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"))]
    hashes = {name: hashlib.sha256((task_dir/name).read_bytes()).hexdigest() for name in inputs}
    return {"catalog_version": CATALOG_VERSION, **deepcopy(case.definition),
            "case_id": case.case_id, "seed": case.seed, "task_path": str(task_path),
            "contract": deepcopy(case.contract), "audit_actuals": list(case.audit_actuals),
            "test_ids": case.test[case.contract["id_column"]].tolist(),
            "input_paths": inputs, "input_sha256": hashes}
