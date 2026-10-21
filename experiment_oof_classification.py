"""Isolated, preregistered OOF Macro F1 experiment; no API or runtime integration.

Prepare all fixtures before scoring, execute once, then independently re-audit
saved probabilities without training. Every real fit uses execution.run_python
and receives only fitting features/labels and prediction features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from agent.classification_decision import (DEFAULT_GUARD_OPTIONS,
                                         fit_guarded_decision_rule, predict_with_rule)
from agent.execution import run_python
from agent.oof_classification import fit_oof_guarded_rule
from oof_classification_cases import case_definitions, make_case, make_partitions, suite_cases

ROOT = Path(__file__).resolve().parent
BASE_SHA = "664bd5a38c4ba8e1451163917bbdff93979b4b54"
MODEL_CONFIG = {"n_estimators": 64, "max_depth": 10, "min_samples_leaf": 2,
                "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 8}
MEMORY_CAP_BYTES = 8 * 1024 ** 3
SOURCES = ("agent/oof_classification.py", "agent/classification_decision.py",
           "experiment_oof_classification.py", "oof_classification_cases.py",
           "agent/execution.py", "agent/resources.py", "agent/file_io.py")
PUBLIC_SOURCES = ("experiment_classification_weights.py", "offline_baseline.py",
                  "agent/splits.py", "agent/task_spec.py", "agent/task_contract.py",
                  "agent/features.py", "agent/metrics.py")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def environment_versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "psutil")}}


def source_hashes(public=False):
    return {name: file_hash(ROOT / name) for name in SOURCES + (PUBLIC_SOURCES if public else ())}


def make_pipeline(frame, seed):
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    transformations = []
    if numeric:
        transformations.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        transformations.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([("features", ColumnTransformer(transformations)),
                     ("model", RandomForestClassifier(**MODEL_CONFIG, random_state=seed))])


def worker_fit(request_path):
    """Trusted child entry: there are no prediction labels in its request."""
    started = time.monotonic()
    request_path = Path(request_path)
    request = joblib.load(request_path)
    if set(request) != {"X_fit", "y_fit", "X_predict", "seed", "model_config"}:
        raise ValueError("Unexpected fit request fields")
    if request["model_config"] != MODEL_CONFIG:
        raise ValueError("Frozen model configuration differs")
    model = make_pipeline(request["X_fit"], request["seed"])
    fit_started = time.monotonic()
    # RF's joblib work is explicitly threads; it does not create worker processes.
    with joblib.parallel_backend("threading", n_jobs=8), threadpool_limits(limits=8):
        model.fit(request["X_fit"], request["y_fit"])
        fit_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        probabilities = model.predict_proba(request["X_predict"])
        predict_seconds = time.monotonic() - predict_started
    directory = request_path.parent
    np.save(directory / "probabilities.npy", probabilities, allow_pickle=False)
    joblib.dump(model, directory / "model.joblib", compress=3)
    preprocessing = {"numeric_statistics": {}, "categorical_categories": {}}
    for name, transformer, columns in model.named_steps["features"].transformers_:
        if name == "numeric":
            preprocessing["numeric_statistics"] = {
                column: float(value) if math.isfinite(float(value)) else None
                for column, value in zip(columns, transformer.statistics_)}
        elif name == "categorical":
            preprocessing["categorical_categories"] = {
                column: [str(value) for value in values]
                for column, values in zip(columns, transformer.named_steps["encode"].categories_)}
    memory = {"available": False, "peak_rss_bytes": None,
              "scope": "worker lifetime, excludes supervisor; not a tree RSS limit"}
    try:
        import resource
        memory.update(available=True,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                      * (1 if sys.platform == "darwin" else 1024),
                      method="resource.getrusage(RUSAGE_SELF).ru_maxrss")
    except (ImportError, OSError, ValueError):
        memory["reason"] = "resource measurement unavailable"
    metadata = {"classes": model.named_steps["model"].classes_.tolist(),
                "fit_source_rows": request["X_fit"].index.tolist(),
                "predict_source_rows": request["X_predict"].index.tolist(),
                "fit_rows": len(request["X_fit"]), "predict_rows": len(request["X_predict"]),
                "fit_label_support": {label: int(np.count_nonzero(request["y_fit"] == label))
                                      for label in model.named_steps["model"].classes_.tolist()},
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "preprocessing": preprocessing, "memory": memory,
                "request_sha256": file_hash(request_path),
                "model_sha256": file_hash(directory / "model.joblib"),
                "probabilities_sha256": file_hash(directory / "probabilities.npy")}
    write_json(directory / "metadata.json", metadata)


class BoundedFitPredict:
    """A fit callback with real subprocess enforcement, not just a timeout hint."""
    def __init__(self, directory, seed):
        self.directory, self.seed = Path(directory).resolve(), seed
        self.records = []

    def __call__(self, X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise TimeoutError("No time remains for fit")
        started = time.monotonic()
        directory = self.directory / f"fit-{fit_index}"
        directory.mkdir(parents=True, exist_ok=False)
        request_path = directory / "request.joblib"
        joblib.dump({"X_fit": X_fit.copy(deep=True), "y_fit": np.array(y_fit, copy=True),
                     "X_predict": X_predict.copy(deep=True), "seed": self.seed,
                     "model_config": MODEL_CONFIG}, request_path, compress=3)
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Request serialization exhausted fit timeout")
        code = ("import sys\n" + f"sys.path.insert(0, {str(ROOT)!r})\n"
                + "from experiment_oof_classification import worker_fit\n"
                + f"worker_fit({str(request_path)!r})\n")
        receipt = run_python(code, ROOT, directory / "execution", 0,
                             timeout=remaining, max_memory_bytes=MEMORY_CAP_BYTES)
        write_json(directory / "receipt.json", receipt)
        record = {"fit_index": fit_index, "directory": str(directory), "receipt": receipt,
                  "metadata": None, "wall_seconds": time.monotonic() - started}
        self.records.append(record)
        if receipt.get("timed_out"):
            raise TimeoutError("Child fit timeout")
        if receipt.get("exit_code") != 0 or receipt.get("error") or receipt.get("memory_limit_exceeded"):
            raise RuntimeError("Child fit failed; inspect its bounded receipt")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        record["metadata"] = metadata
        if (file_hash(directory / "model.joblib") != metadata["model_sha256"]
                or file_hash(directory / "probabilities.npy") != metadata["probabilities_sha256"]
                or file_hash(request_path) != metadata["request_sha256"]):
            raise ValueError("Worker artifact hash mismatch")
        probabilities = np.load(directory / "probabilities.npy", allow_pickle=False)
        if len(probabilities) != len(X_predict):
            raise ValueError("Worker probability row count differs")
        return {"probabilities": probabilities, "classes": metadata["classes"]}


def identity_wrapper(classes, reason):
    guarded = fit_guarded_decision_rule(np.empty((0, len(classes))), [], classes,
                                        tuning_rows=[], selection_rows=[])
    return {"version": 1, "kind": "host_identity", "status": "identity", "reason": reason,
            "classes": list(classes), "rule": guarded["rule"],
            "candidate_rule": guarded["candidate_rule"], "guarded": guarded,
            "completed_fits": 0, "fits": [],
            "hashes": {"rule_sha256": json_hash(guarded["rule"])}}


def _public_cases(task):
    from agent.task_spec import load_task
    from experiment_classification_weights import grouped_fold
    from offline_baseline import load_data, prepare_features
    spec = load_task(task)
    if spec.metric.name != "Macro F1":
        raise ValueError("Public sanity task must use Macro F1")
    (train, test, _, id_column, _, y, features,
     numeric, categorical, _) = load_data(
         spec.data_dir, task_text=spec.task_path.read_text(encoding="utf-8-sig"), classification=True)
    # Deterministic dictionary conversion is not a fitted preprocessing step.
    frame = prepare_features(train, features, numeric, categorical).reset_index(drop=True)
    test_frame = prepare_features(test, features, numeric, categorical).copy()
    test_frame.index = np.arange(len(train), len(train) + len(test))
    groups = pd.util.hash_pandas_object(train[features], index=False).astype(str).to_numpy()
    result = []
    for fold in (0, 1, 2):
        allowed, outer, metadata = grouped_fold(train, y, features, fold)
        result.append({"name": f"public_fold{fold}", "seed": 42, "frame": frame,
                       "labels": np.asarray(y, dtype=object), "classes": list(spec.classes),
                       "ids": train[id_column].to_numpy(), "groups": groups,
                       "allowed_rows": allowed, "outer_rows": outer,
                       "production_rows": np.array([], dtype=np.int64),
                       "production_frame": test_frame, "production_ids": test[id_column].to_numpy(),
                       "split_policy": "grouped", "generator_metadata": metadata})
    return result


def _support(case, rows):
    labels = np.asarray(case["labels"], dtype=object)[rows]
    return {label: int(np.count_nonzero(labels == label)) for label in case["classes"]}


def _plain_partitions(partitions):
    return {**partitions, "tuning_rows": partitions["tuning_rows"].tolist(),
            "selection_rows": partitions["selection_rows"].tolist(),
            "folds": [{key: value.tolist() for key, value in fold.items()}
                      for fold in partitions["folds"]]}


def prepare(output_dir, *, public_task=None):
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_relative_to(ROOT / "data/experiments/oof_classification"):
        raise ValueError("Raw artifacts must stay under data/experiments/oof_classification")
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    public_task = Path(public_task or ROOT / "data/classification/task.md").expanduser().resolve()
    if not public_task.is_file() and public_task != ROOT / "data/classification/task.md":
        raise FileNotFoundError("Explicit public task is missing")
    public = {"status": "planned" if public_task.is_file() else "skipped_missing_data",
              "task": str(public_task), "folds": [0, 1, 2], "input_sha256": {}}
    sources = source_hashes(public["status"] == "planned")
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    for stage in ("development", "confirmation", "degenerate"):
        for definition in suite_cases(stage):
            cases.append((stage, make_case(**definition)))
    if public["status"] == "planned":
        from offline_baseline import find_data
        directory = find_data(public_task.parent)
        public["input_sha256"] = {str(path): file_hash(path) for path in [public_task,
            *(directory / name for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"))]}
        cases.extend(("public_sanity", case) for case in _public_cases(public_task))
    entries = []
    for index, (stage, case) in enumerate(cases):
        key = f"{case['name']}-{case['seed']}"
        directory = output_dir / "cases" / key
        directory.mkdir(parents=True)
        partitions = _plain_partitions(make_partitions(case))
        allowed = case["allowed_rows"]
        record = {"partitions": partitions,
                  "roles": {name: case[name + "_rows"].tolist() for name in ("allowed", "outer", "production")},
                  "support": {name: _support(case, case[name + "_rows"])
                              for name in ("allowed", "outer", "production")},
                  "calibration_support": {name: _support(case, allowed[np.asarray(partitions[name + "_rows"], dtype=int)])
                                          for name in ("tuning", "selection")}}
        joblib.dump(case, directory / "source.joblib", compress=3)
        source_frame = case["frame"].copy(deep=True)
        id_field, actual_field = "source_id", "actual"
        while id_field in source_frame:
            id_field = "_" + id_field
        while actual_field in source_frame or actual_field == id_field:
            actual_field = "_" + actual_field
        source_frame.insert(0, id_field, case["ids"])
        source_frame[actual_field] = case["labels"]
        source_frame.to_csv(directory / "source.csv", index=False, float_format="%.17g")
        write_json(directory / "splits.json", record)
        entries.append({"key": key, "name": case["name"], "seed": case["seed"], "stage": stage,
                        "source_sha256": file_hash(directory / "source.joblib"),
                        "source_csv_sha256": file_hash(directory / "source.csv"),
                        "source_csv_metadata_fields": {"id": id_field, "actual": actual_field},
                        "split_sha256": file_hash(directory / "splits.json"),
                        "generator": case["generator_metadata"],
                        "arm_order": ["control", "oof"] if index % 2 == 0 else ["oof", "control"],
                        "arm_cap_seconds": 900 if stage == "public_sanity" else 180,
                        "calibration_fit_timeout_seconds": 120 if stage == "public_sanity" else 25,
                        "final_reserve_seconds": 300 if stage == "public_sanity" else 60})
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    plan = {"version": 1, "kind": "oof_macro_f1_preregistration", "base_sha": BASE_SHA,
            "source_commit": git_sha, "source_dirty": dirty, "source_sha256": sources,
            "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment_versions(),
            "model_config": MODEL_CONFIG, "guard_options": dict(DEFAULT_GUARD_OPTIONS),
            "generators": case_definitions(), "public": public, "cases": entries,
            "cpu_threads": 8, "memory_monitor_cap_bytes": MEMORY_CAP_BYTES,
            "series_cap_seconds": 7200, "api_calls": 0,
            "calibration_fit_count": 4, "total_oof_fits_including_production": 6,
            "control_total_fits_including_production": 2,
            "control_scope": "equal cap, fixed model; no AutoML search; no AutoML superiority claim",
            "confirmation": "fresh fixed seeds; known families; algorithm frozen across development and confirmation"}
    plan["plan_sha256"] = json_hash(plan)
    write_json(output_dir / "preregistration.json", plan)
    return plan


def load_plan(output_dir, *, check_environment=True):
    plan = json.loads((Path(output_dir) / "preregistration.json").read_text(encoding="utf-8"))
    expected = plan.pop("plan_sha256")
    if json_hash(plan) != expected:
        raise ValueError("Preregistration hash mismatch")
    plan["plan_sha256"] = expected
    if source_hashes(plan["public"]["status"] == "planned") != plan["source_sha256"]:
        raise ValueError("Runtime sources changed after preparation")
    if check_environment and environment_versions() != plan["environment"]:
        raise ValueError("Environment changed after preparation")
    for name, expected_hash in plan["public"]["input_sha256"].items():
        if file_hash(name) != expected_hash:
            raise ValueError("Public inputs changed")
    return plan


def load_case(output_dir, entry):
    directory = Path(output_dir) / "cases" / entry["key"]
    for name, key in (("source.joblib", "source_sha256"), ("source.csv", "source_csv_sha256"),
                      ("splits.json", "split_sha256")):
        if file_hash(directory / name) != entry[key]:
            raise ValueError("Frozen case artifact changed: " + name)
    return joblib.load(directory / "source.joblib"), json.loads((directory / "splits.json").read_text(encoding="utf-8"))


def _score(actual, predictions, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predictions, labels=classes, zero_division=0)
    return {"macro_f1": float(f1.mean()), "per_class": [
        {"class": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
        for label, p, r, f, n in zip(classes, precision, recall, f1, support)]}


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Original arm deadline exhausted")
    return value


def run_arm(output_dir, entry, arm, plan):
    case, splits = load_case(output_dir, entry)
    directory = Path(output_dir) / "cases" / entry["key"] / arm
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + entry["arm_cap_seconds"]
    frame, labels, classes = case["frame"], case["labels"], case["classes"]
    allowed, outer = case["allowed_rows"], case["outer_rows"]
    partitions = splits["partitions"]
    calibration = BoundedFitPredict(directory / "calibration", entry["seed"])
    final = BoundedFitPredict(directory / "final", entry["seed"])
    wrapper = identity_wrapper(classes, "control_identity")
    result = {"arm": arm, "case": entry["key"], "status": "running", "scores": {},
              "arm_cap_seconds": entry["arm_cap_seconds"], "api_calls": 0}
    try:
        if arm == "oof":
            if partitions["status"] != "ready":
                wrapper = identity_wrapper(classes, "host_" + partitions["reason"])
            else:
                wrapper = fit_oof_guarded_rule(
                    frame.iloc[allowed], labels[allowed], classes,
                    groups=case["groups"][allowed], split_policy=case["split_policy"],
                    tuning_rows=partitions["tuning_rows"], selection_rows=partitions["selection_rows"],
                    folds=partitions["folds"], fit_predict=calibration,
                    fold_timeout_seconds=entry["calibration_fit_timeout_seconds"],
                    deadline=deadline-entry["final_reserve_seconds"], clock=time.monotonic,
                    guard_options=plan["guard_options"])
        frozen = {"wrapper": wrapper, "wrapper_sha256": json_hash(wrapper),
                  "rule_sha256": json_hash(wrapper["rule"]), "model_config": MODEL_CONFIG,
                  "plan_sha256": plan["plan_sha256"], "source_sha256": entry["source_sha256"],
                  "split_sha256": entry["split_sha256"],
                  "frozen_before_outer_fit": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "frozen.json", frozen)
        frozen_hash = file_hash(directory / "frozen.json")
        prediction = final(frame.iloc[allowed], labels[allowed], frame.iloc[outer],
                           timeout_seconds=_remaining(deadline), fit_index=4)
        _remaining(deadline)
        production_fit = np.concatenate((allowed, outer))
        production_rows = case["production_rows"]
        production_X = (case["production_frame"] if "production_frame" in case
                        else frame.iloc[production_rows])
        production_prediction = final(frame.iloc[production_fit], labels[production_fit], production_X,
                                      timeout_seconds=_remaining(deadline), fit_index=5)
        _remaining(deadline)
        # Both audit labels enter scoring only after immutable rule and both refits.
        phases = [("outer", prediction, labels[outer], case["ids"][outer])]
        phases.append(("production", production_prediction,
                       labels[production_rows] if len(production_rows) else None,
                       case["ids"][production_rows] if len(production_rows) else case["production_ids"]))
        identity = identity_wrapper(classes, "identity_for_same_probability_control")["rule"]
        for phase, supplied, actual, ids in phases:
            unchanged = predict_with_rule(supplied["probabilities"], supplied["classes"], identity)
            changed = predict_with_rule(supplied["probabilities"], supplied["classes"], wrapper["rule"])
            predictions = pd.DataFrame({"id": ids, "identity": unchanged, "chosen": changed})
            if actual is not None:
                predictions["actual"] = actual
                result["scores"][phase] = {"identity": _score(actual, unchanged, classes),
                                           "chosen": _score(actual, changed, classes)}
                result["scores"][phase]["delta_macro_f1"] = (
                    result["scores"][phase]["chosen"]["macro_f1"]
                    - result["scores"][phase]["identity"]["macro_f1"])
            else:
                result["scores"][phase] = {"status": "not_scored_public_test_answers_unavailable"}
            predictions.to_csv(directory / (phase + "_predictions.csv"), index=False)
        if (json_hash(wrapper) != frozen["wrapper_sha256"]
                or file_hash(directory / "frozen.json") != frozen_hash):
            raise ValueError("Frozen rule changed after refit")
        _remaining(deadline)
        result.update(status="completed", wrapper=wrapper, frozen_sha256=frozen_hash,
                      immutable_rule_after_refits=True)
    except (TimeoutError, RuntimeError, MemoryError, OSError) as error:
        # Contract errors intentionally propagate instead of becoming a successful fallback.
        result.update(status="failed", error_type=type(error).__name__, error=str(error)[:500], wrapper=wrapper)
    result.update(elapsed_seconds=time.monotonic()-started,
                  calibration_wall_seconds=sum(row["wall_seconds"] for row in calibration.records),
                  calibration_fit_count=sum(row["metadata"] is not None for row in calibration.records),
                  total_fit_count=sum(row["metadata"] is not None for row in calibration.records + final.records),
                  fits=calibration.records + final.records)
    write_json(directory / "result.json", result)
    return result


def run_prepared(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    plan = load_plan(output_dir)
    with (output_dir / "run_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan["plan_sha256"]}, stream)
    started = time.monotonic()
    report = {"status": "running", "plan_sha256": plan["plan_sha256"], "pairs": [], "api_calls": 0}
    for index, entry in enumerate(plan["cases"]):
        load_plan(output_dir)
        pair = {"key": entry["key"], "stage": entry["stage"], "arms": {}}
        for arm in entry["arm_order"]:
            print(f"[{index+1}/{len(plan['cases'])}] {entry['key']} {arm}", flush=True)
            if time.monotonic()-started + entry["arm_cap_seconds"] > plan["series_cap_seconds"]:
                pair["arms"][arm] = {"status": "skipped_series_budget"}
            else:
                pair["arms"][arm] = run_arm(output_dir, entry, arm, plan)
            report["active_pair"] = pair
            write_json(output_dir / "report.json", report)
        # Final models/configs are identical across arms. Verify this empirically.
        if all(row.get("status") == "completed" for row in pair["arms"].values()):
            case_directory = output_dir / "cases" / entry["key"]
            equal, comparison = {}, {}
            for fit_index, phase in ((4, "outer"), (5, "production")):
                first = np.load(case_directory / "control/final" / f"fit-{fit_index}" / "probabilities.npy")
                second = np.load(case_directory / "oof/final" / f"fit-{fit_index}" / "probabilities.npy")
                equal[phase] = bool(np.array_equal(first, second))
                comparison[phase] = {"bitwise_equal": equal[phase],
                                     "max_abs_difference": float(np.max(np.abs(first-second))),
                                     "allclose_atol_1e12": bool(np.allclose(first, second, rtol=0, atol=1e-12)),
                                     "argmax_agreement": bool(np.array_equal(first.argmax(axis=1), second.argmax(axis=1)))}
            pair["same_final_probabilities"] = equal
            pair["final_probability_comparison"] = comparison
        report["pairs"].append(pair)
        report.pop("active_pair", None)
        write_json(output_dir / "report.json", report)
    report.update(status="completed" if all(a.get("status") == "completed" for p in report["pairs"]
                                           for a in p["arms"].values()) else "incomplete",
                  elapsed_seconds=time.monotonic()-started)
    write_json(output_dir / "report.json", report)
    return report


def _independent_metrics(actual, predicted, classes):
    result = []
    for label in classes:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        support = sum(a == label for a in actual)
        count = sum(p == label for p in predicted)
        result.append({"class": label, "support": support,
                       "precision": tp/count if count else 0.0,
                       "recall": tp/support if support else 0.0,
                       "f1": 2*tp/(support+count) if support+count else 0.0})
    return {"macro_f1": sum(row["f1"] for row in result)/len(classes), "per_class": result}


def _require(condition, message):
    # Unlike assert, audit checks must remain active under python -O.
    if not condition:
        raise ValueError("Re-audit: " + message)


def _audit_fit(record, directory, case, splits, entry, plan):
    index = record["fit_index"]
    _require(type(index) is int and 0 <= index <= 5, "invalid fit index")
    fit_dir = directory / ("calibration" if index < 4 else "final") / f"fit-{index}"
    _require(Path(record["directory"]).resolve() == fit_dir.resolve(), "fit artifact location")
    receipt = json.loads((fit_dir / "receipt.json").read_text(encoding="utf-8"))
    _require(receipt == record["receipt"], "receipt changed, including a failed fit")
    metadata = record["metadata"]
    if metadata is None:
        _require(receipt.get("exit_code") != 0 or receipt.get("timed_out")
                 or receipt.get("error") or receipt.get("memory_limit_exceeded"), "missing successful metadata")
        return None
    _require(receipt.get("exit_code") == 0 and not receipt.get("timed_out")
             and not receipt.get("error") and not receipt.get("memory_limit_exceeded"), "fit did not succeed")
    _require(json.loads((fit_dir / "metadata.json").read_text(encoding="utf-8")) == metadata, "metadata changed")
    for name, key in (("model.joblib", "model_sha256"), ("request.joblib", "request_sha256"),
                      ("probabilities.npy", "probabilities_sha256")):
        _require(file_hash(fit_dir / name) == metadata[key], name + " hash changed")
    request = joblib.load(fit_dir / "request.joblib")
    _require(set(request) == {"X_fit", "y_fit", "X_predict", "seed", "model_config"}, "request fields")
    _require(request["seed"] == entry["seed"] and request["model_config"] == plan["model_config"], "fit configuration")
    allowed, outer = case["allowed_rows"], case["outer_rows"]
    if index < 3:
        fold = splits["partitions"]["folds"][index]
        fit_rows, predict_rows = allowed[fold["fit_rows"]], allowed[fold["held_rows"]]
    elif index == 3:
        fit_rows = allowed[splits["partitions"]["tuning_rows"]]
        predict_rows = allowed[splits["partitions"]["selection_rows"]]
    elif index == 4:
        fit_rows, predict_rows = allowed, outer
    else:
        fit_rows, predict_rows = np.concatenate((allowed, outer)), case["production_rows"]
    expected_X = (case["production_frame"] if index == 5 and "production_frame" in case
                  else case["frame"].iloc[predict_rows])
    try:
        pd.testing.assert_frame_equal(request["X_fit"], case["frame"].iloc[fit_rows], check_exact=True)
        np.testing.assert_array_equal(request["y_fit"], case["labels"][fit_rows])
        pd.testing.assert_frame_equal(request["X_predict"], expected_X, check_exact=True)
    except AssertionError as error:
        raise ValueError("Re-audit: fitting/prediction inputs differ from fixed rows") from error
    _require(metadata["fit_source_rows"] == request["X_fit"].index.tolist()
             and metadata["predict_source_rows"] == request["X_predict"].index.tolist(), "fit row metadata")
    _require(metadata["fit_rows"] == len(fit_rows) and metadata["predict_rows"] == len(expected_X), "fit sizes")
    _require(math.isfinite(metadata["fit_seconds"]) and metadata["fit_seconds"] > 0, "no actual fit duration")
    probabilities = np.load(fit_dir / "probabilities.npy", allow_pickle=False)
    _require(len(probabilities) == len(expected_X), "prediction row count")
    predict_with_rule(probabilities, metadata["classes"], identity_wrapper(case["classes"], "audit")["rule"])
    model = joblib.load(fit_dir / "model.joblib")
    _require(model.named_steps["model"].classes_.tolist() == metadata["classes"], "model class alignment")
    with joblib.parallel_backend("threading", n_jobs=8), threadpool_limits(limits=8):
        recomputed = model.predict_proba(request["X_predict"])
    _require(np.allclose(recomputed, probabilities, rtol=0, atol=1e-12), "saved model does not reproduce probabilities")
    return probabilities[:, [metadata["classes"].index(label) for label in case["classes"]]]


def re_audit(output_dir):
    """Verify complete plan coverage, artifacts and independently computed F1; no fits."""
    output_dir = Path(output_dir).expanduser().resolve()
    plan = load_plan(output_dir)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    _require(report.get("plan_sha256") == plan["plan_sha256"], "report plan lineage")
    _require(report.get("status") in {"completed", "incomplete"}, "report is not terminal")
    planned = [entry["key"] for entry in plan["cases"]]
    _require(isinstance(report.get("pairs"), list) and bool(planned)
             and all(isinstance(pair, dict) for pair in report["pairs"])
             and [pair.get("key") for pair in report["pairs"]] == planned, "planned case coverage")
    checks = []
    entries = {entry["key"]: entry for entry in plan["cases"]}
    for pair in report["pairs"]:
        entry = entries[pair["key"]]
        _require(isinstance(pair.get("arms"), dict) and set(pair["arms"]) == {"control", "oof"}, "planned arm coverage")
        _require(pair.get("stage") == entry["stage"], "planned case stage")
        case, splits = load_case(output_dir, entry)
        classes, allowed, outer = case["classes"], case["allowed_rows"], case["outer_rows"]
        for arm, result in pair["arms"].items():
            if result["status"] != "completed":
                checks.append({"case": pair["key"], "arm": arm, "status": "not_completed"})
                continue
            directory = output_dir / "cases" / pair["key"] / arm
            _require(result["arm"] == arm and result["case"] == entry["key"], "result identity")
            _require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result, "result changed")
            frozen = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
            _require(frozen["plan_sha256"] == plan["plan_sha256"] and frozen["source_sha256"] == entry["source_sha256"]
                     and frozen["split_sha256"] == entry["split_sha256"] and frozen["model_config"] == plan["model_config"], "frozen lineage")
            wrapper = frozen["wrapper"]
            _require(file_hash(directory / "frozen.json") == result["frozen_sha256"], "frozen file hash")
            _require(json_hash(wrapper) == frozen["wrapper_sha256"] == json_hash(result["wrapper"]), "wrapper changed")
            _require(json_hash(wrapper["rule"]) == frozen["rule_sha256"], "rule changed")
            indices = [record["fit_index"] for record in result["fits"]]
            _require(len(indices) == len(set(indices)) and indices == sorted(indices), "fit receipt coverage")
            matrices = {record["fit_index"]: _audit_fit(record, directory, case, splits, entry, plan)
                        for record in result["fits"]}
            _require(4 in matrices and matrices[4] is not None and 5 in matrices and matrices[5] is not None, "missing final refits")
            successful = sum(matrix is not None for matrix in matrices.values())
            _require(successful == result["total_fit_count"]
                     and sum(matrix is not None for index, matrix in matrices.items() if index < 4)
                     == result["calibration_fit_count"], "fit counts differ")
            if wrapper.get("guarded") is not None and wrapper.get("kind") == "oof_guarded_macro_f1":
                _require(wrapper["completed_fits"] == 4 and all(matrices.get(i) is not None for i in range(4)), "incomplete OOF used by guard")
                p = np.empty((len(allowed), len(classes)))
                partitions = splits["partitions"]
                for index, fold in enumerate(partitions["folds"]):
                    p[fold["held_rows"]] = matrices[index]
                p[partitions["selection_rows"]] = matrices[3]
                joined = np.asarray(partitions["tuning_rows"] + partitions["selection_rows"], dtype=int)
                tuning_count = len(partitions["tuning_rows"])
                guarded = fit_guarded_decision_rule(p[joined], case["labels"][allowed][joined], classes,
                    tuning_rows=np.arange(tuning_count), selection_rows=np.arange(tuning_count, len(allowed)),
                    options=plan["guard_options"])
                _require(guarded == wrapper["guarded"], "guarded rule cannot be reproduced")
            for phase, index, rows in (("outer", 4, outer), ("production", 5, case["production_rows"])):
                p = matrices[index]
                csv = pd.read_csv(directory / (phase + "_predictions.csv"), dtype=str, keep_default_na=False)
                _require(csv.columns.tolist() == ["id", "identity", "chosen"] + (["actual"] if len(rows) else []), "prediction CSV schema")
                ids = case["ids"][rows] if len(rows) else case["production_ids"]
                _require(csv["id"].tolist() == ids.tolist(), "literal ID/order mismatch")
                for column, rule in (("identity", identity_wrapper(classes, "audit")["rule"]), ("chosen", wrapper["rule"])):
                    predicted = predict_with_rule(p, classes, rule).tolist()
                    _require(csv[column].tolist() == predicted, "saved labels do not match rule")
                    if len(rows):
                        actual = case["labels"][rows].tolist()
                        _require(csv["actual"].tolist() == actual, "audit targets changed")
                        recomputed = _independent_metrics(actual, predicted, classes)
                        saved = result["scores"][phase][column]
                        _require(abs(saved["macro_f1"]-recomputed["macro_f1"]) <= 1e-12, "Macro F1 mismatch")
                        _require(len(saved["per_class"]) == len(classes), "per-class coverage")
                        for x, y in zip(saved["per_class"], recomputed["per_class"]):
                            _require(x["class"] == y["class"] and x["support"] == y["support"], "per-class support")
                            _require(all(abs(x[key]-y[key]) <= 1e-12 for key in ("precision", "recall", "f1")), "per-class metric")
                if len(rows):
                    scores = result["scores"][phase]
                    _require(abs(scores["delta_macro_f1"]-(scores["chosen"]["macro_f1"]-scores["identity"]["macro_f1"])) <= 1e-12, "paired delta")
            checks.append({"case": pair["key"], "arm": arm, "status": "verified", "phases": 2,
                           "model_predictions_reproduced": successful})
    result = {"status": "verified" if all(row["status"] == "verified" for row in checks) else "incomplete",
              "plan_sha256": plan["plan_sha256"], "report_sha256": file_hash(output_dir / "report.json"),
              "checks": checks, "api_calls": 0, "model_fits": 0}
    write_json(output_dir / "reaudit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run-prepared", type=Path)
    action.add_argument("--re-audit", type=Path)
    action.add_argument("--worker-fit", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--public-task", type=Path)
    args = parser.parse_args()
    if args.prepare_only:
        if args.output_dir is None:
            parser.error("--prepare-only requires --output-dir")
        plan = prepare(args.output_dir, public_task=args.public_task)
        print(json.dumps({"status": "prepared", "pairs": len(plan["cases"]),
                          "plan_sha256": plan["plan_sha256"], "public": plan["public"]["status"]}))
    elif args.output_dir or args.public_task:
        parser.error("Preparation arguments cannot accompany run/re-audit/worker")
    elif args.worker_fit:
        worker_fit(args.worker_fit)
    elif args.run_prepared:
        report = run_prepared(args.run_prepared)
        print(json.dumps({"status": report["status"], "pairs": len(report["pairs"]),
                          "elapsed_seconds": report["elapsed_seconds"]}))
        return 0 if report["status"] == "completed" else 1
    else:
        report = re_audit(args.re_audit)
        print(json.dumps({"status": report["status"], "arms": len(report["checks"]), "model_fits": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
