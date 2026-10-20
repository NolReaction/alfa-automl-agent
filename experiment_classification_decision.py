"""Frozen, bounded paired Macro F1 experiments; API calls: 0.

Run --prepare-only first, then --run-prepared PATH. Preparation records source
hashes and all scenarios before any audit score. Prepared runs cannot be resumed
or silently overwritten. A fresh child process has a 600-second deadline for
each case; synthetic cases run serially. This is a data protocol, not an OS
security boundary against malicious Python.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "8"

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from agent.classification_decision import fit_decision_rule, predict_with_rule
from classification_decision_cases import CASE_DEFINITIONS, SEEDS, make_case, validate_splits

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = {
    "n_estimators": 160, "max_depth": 12, "min_samples_leaf": 2,
    "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 8,
}
RULE_OPTIONS = {
    "multiplier_grid": [0.5, 0.75, 1.0, 4 / 3, 2.0],
    "multiplier_min": 0.5, "multiplier_max": 2.0, "max_rounds": 3,
    "max_candidates": 128, "min_samples": 100, "min_class_count": 10,
}
SOURCES = ("agent/classification_decision.py", "experiment_classification_decision.py",
           "classification_decision_cases.py")
CASE_TIMEOUT_SECONDS = 600
MEMORY_LIMIT_BYTES = 32 * 1024**3


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def source_hashes(public=False):
    paths = list(SOURCES)
    if public:
        paths += ["experiment_classification_weights.py", "offline_baseline.py", "agent/splits.py",
                  "agent/task_contract.py", "agent/metrics.py"]
    return {name: file_hash(ROOT / name) for name in paths}


def public_input_hashes(task):
    from offline_baseline import find_data
    directory = Path(find_data(task.parent)).resolve()
    paths = [task, *(directory / name for name in
                     ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"))]
    return {str(path): file_hash(path) for path in paths}


def environment_versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")}}


def prepare(output_dir=None, public_task=None):
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "data/experiments/classification_decision" / f"{stamp}-{uuid4().hex[:8]}"
    output_dir = Path(output_dir).expanduser().resolve()
    if public_task is not None and not Path(public_task).expanduser().is_file():
        raise ValueError("An explicitly requested public task does not exist.")
    public_task = Path(public_task or ROOT / "data/classification/task.md").expanduser().resolve()
    has_public = public_task.is_file()
    if has_public and output_dir.is_relative_to(public_task.parent):
        raise ValueError("Outputs must be outside the public task input directory.")
    plan = {
        "format": "classification-decision-preregistration", "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes(has_public), "model": MODEL_CONFIG,
        "model_seed": "scenario seed; public seed 42", "decision_options": RULE_OPTIONS,
        "generators": CASE_DEFINITIONS, "seeds": list(SEEDS),
        "preprocessing": "fit-only numeric median + categorical most_frequent + onehot(ignore unknown)",
        "synthetic_cases": [{"name": name, "seed": seed} for name in CASE_DEFINITIONS for seed in SEEDS],
        "public": {"status": "planned" if has_public else "skipped_missing_data",
                   "task": str(public_task), "folds": [0, 1, 2],
                   "input_sha256": public_input_hashes(public_task) if has_public else {}},
        "resources": {"cpu_threads": 8, "per_case_wall_seconds": CASE_TIMEOUT_SECONDS,
                      "planned_workers": len(CASE_DEFINITIONS) * len(SEEDS) + (3 if has_public else 0),
                      "total_worker_timeout_budget_seconds": CASE_TIMEOUT_SECONDS * (len(CASE_DEFINITIONS) * len(SEEDS) + (3 if has_public else 0)),
                      "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                      "memory_enforcement": "RLIMIT_AS where supported, otherwise reported unavailable"},
        "no_refit": True, "audit_not_used_for_selection": True, "api_calls": 0,
        "environment": environment_versions(),
    }
    plan["plan_sha256"] = json_hash(plan)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "preregistration.json", plan)
    return output_dir, plan


def load_plan(output_dir):
    plan = json.loads((output_dir / "preregistration.json").read_text(encoding="utf-8"))
    supplied_hash = plan.pop("plan_sha256")
    if json_hash(plan) != supplied_hash:
        raise ValueError("Preregistration hash mismatch.")
    plan["plan_sha256"] = supplied_hash
    if environment_versions() != plan["environment"]:
        raise ValueError("Runtime environment changed after preparation; create a fresh preregistration.")
    if source_hashes(plan["public"]["status"] == "planned") != plan["source_sha256"]:
        raise ValueError("Code changed after preparation; do not relabel an altered algorithm as preregistered.")
    if plan["public"]["status"] == "planned":
        if public_input_hashes(Path(plan["public"]["task"])) != plan["public"]["input_sha256"]:
            raise ValueError("Public inputs changed after preparation.")
    return plan


def memory_state(*, enforce=False):
    result = {"measurement_available": False, "peak_rss_bytes": None,
              "memory_limit_enforced": False}
    try:
        import resource
    except ImportError:
        result["unavailable_reason"] = "resource module unavailable"
        return result
    if enforce:
        if not hasattr(resource, "RLIMIT_AS"):
            result["enforcement_unavailable_reason"] = "RLIMIT_AS unavailable"
        else:
            try:
                _, hard = resource.getrlimit(resource.RLIMIT_AS)
                limit = MEMORY_LIMIT_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_LIMIT_BYTES, hard)
                resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
                result["memory_limit_enforced"] = True
                result["enforced_virtual_memory_bytes"] = limit
            except (ValueError, OSError) as error:
                result["enforcement_unavailable_reason"] = type(error).__name__
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        multiplier = 1 if sys.platform == "darwin" else 1024
        result.update(measurement_available=True, peak_rss_bytes=int(usage * multiplier),
                      method="resource.getrusage(RUSAGE_SELF).ru_maxrss",
                      scope="current worker process lifetime; includes fit, excludes supervisor")
    except (ValueError, OSError) as error:
        result["unavailable_reason"] = type(error).__name__
    return result


def _public_case(plan, fold):
    from experiment_classification_weights import grouped_fold
    from offline_baseline import find_data, load_data, prepare_features
    from agent.task_contract import validate_task_metric
    task = Path(plan["public"]["task"])
    task_text = task.read_text(encoding="utf-8-sig")
    validate_task_metric(task_text, True)
    data = load_data(find_data(task.parent), task_text=task_text, classification=True)
    train, _, _, id_column, _, y, features, numeric, categorical, classification = data
    if not classification:
        raise ValueError("Public task must be classification.")
    development, audit, outer = grouped_fold(train, y, features, fold)
    frame = prepare_features(train, features, numeric, categorical)
    groups = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    fitting, calibration = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                  random_state=42).split(development, groups=groups[development]))
    fit, cal = development[fitting], development[calibration]
    # Validate raw values, not transformed/prepared values.
    integrity = validate_splits(train[features], fit, cal, audit)
    return {"name": f"public_fold{fold}", "seed": 42, "frame": frame,
            "labels": np.asarray(y, dtype=str), "ids": train[id_column].astype(str).to_numpy(),
            "classes": sorted(set(y.tolist())), "fit_rows": fit, "calibration_rows": cal,
            "audit_rows": audit, "split": {"strategy": "grouped", "outer": outer, **integrity}}


def make_pipeline(frame, seed):
    numeric = [name for name in frame if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in frame if name not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([("features", ColumnTransformer(transformers)),
                     ("model", RandomForestClassifier(**MODEL_CONFIG, random_state=seed))])


def diagnostics(actual, predictions, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predictions, labels=classes, zero_division=0)
    return {"macro_f1": float(f1_score(actual, predictions, labels=classes, average="macro", zero_division=0)),
            "per_class": [{"class": label, "precision": float(p), "recall": float(r),
                           "f1": float(f), "support": int(n)}
                          for label, p, r, f, n in zip(classes, precision, recall, f1, support)]}


def run_case(output_dir, spec):
    started = time.monotonic()
    plan = load_plan(output_dir)
    memory_start = memory_state(enforce=True)
    case = _public_case(plan, spec["fold"]) if "fold" in spec else make_case(spec["name"], spec["seed"])
    name, seed = case["name"], case["seed"]
    directory = output_dir / f"{name}-{seed}"
    directory.mkdir(exist_ok=False)
    frame, y, ids, classes = case["frame"], case["labels"], case["ids"], case["classes"]
    fit, cal, audit = (case[key] for key in ("fit_rows", "calibration_rows", "audit_rows"))
    source = frame.copy()
    source_columns = {}
    for role, values in (("id", ids), ("actual", y)):
        column = "source_id" if role == "id" else "actual"
        while column in source.columns:
            column = "_" + column
        source_columns[role] = column
        source[column] = values
    source.to_csv(directory / "source.csv", index=False)
    splits = {"fit": fit.tolist(), "calibration": cal.tolist(), "audit": audit.tolist()}
    write_json(directory / "splits.json", splits)
    supports = {part: {label: int(np.count_nonzero(y[rows] == label)) for label in classes}
                for part, rows in (("fit", fit), ("calibration", cal), ("audit", audit))}
    if set(y[fit].tolist()) != set(classes):
        raise ValueError("Fixed fitting split lacks a class; no alternative split is selected.")
    pipeline = make_pipeline(frame, seed)
    fit_started = time.monotonic()
    with threadpool_limits(limits=8):
        pipeline.fit(frame.iloc[fit], y[fit])
        calibration_probabilities = pipeline.predict_proba(frame.iloc[cal])
    fit_seconds = time.monotonic() - fit_started
    model_classes = pipeline.named_steps["model"].classes_.tolist()
    # The model emits sorted classes; pass the exact actual probability order.
    tuning_started = time.monotonic()
    rule = fit_decision_rule(calibration_probabilities, y[cal], model_classes, options=RULE_OPTIONS)
    tuning_seconds = time.monotonic() - tuning_started
    joblib.dump(pipeline, directory / "frozen_model.joblib", compress=3)
    calibration_frame = pd.DataFrame({"id": ids[cal], "actual": y[cal]})
    for index, label in enumerate(model_classes):
        calibration_frame[f"probability_{index}"] = calibration_probabilities[:, index]
    calibration_frame.to_csv(directory / "calibration_probabilities.csv", index=False)
    frozen = {
        "case": name, "seed": seed, "classes": model_classes, "model_config": {**MODEL_CONFIG, "random_state": seed},
        "model_sha256": file_hash(directory / "frozen_model.joblib"), "rule": rule,
        "rule_sha256": json_hash(rule), "plan_sha256": plan["plan_sha256"],
        "configuration_sha256": json_hash({"model": {**MODEL_CONFIG, "random_state": seed},
                                             "decision": RULE_OPTIONS}),
        "source_sha256": file_hash(directory / "source.csv"), "source_metadata_columns": source_columns,
        "split_sha256": json_hash(splits),
        "calibration_sha256": file_hash(directory / "calibration_probabilities.csv"),
        "ids_sha256": {part: json_hash(ids[rows].tolist()) for part, rows in
                       (("fit", fit), ("calibration", cal), ("audit", audit))},
        "supports": supports, "split": case["split"],
        "frozen_before_audit": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
    }
    write_json(directory / "frozen.json", frozen)
    frozen_hash = file_hash(directory / "frozen.json")
    # Audit labels first enter scoring only after the configuration/rule snapshot.
    audit_started = time.monotonic()
    with threadpool_limits(limits=8):
        probabilities = pipeline.predict_proba(frame.iloc[audit])
    control = np.asarray(model_classes)[np.argmax(probabilities, axis=1)]
    candidate = predict_with_rule(probabilities, model_classes, rule)
    scores = {"control": diagnostics(y[audit], control, classes),
              "candidate": diagnostics(y[audit], candidate, classes)}
    predicted_frame = pd.DataFrame({"id": ids[audit], "actual": y[audit],
                                    "control": control, "candidate": candidate})
    for index, _ in enumerate(model_classes):
        predicted_frame[f"probability_{index}"] = probabilities[:, index]
    predicted_frame.to_csv(directory / "audit_predictions.csv", index=False)
    if (file_hash(directory / "frozen.json") != frozen_hash
            or file_hash(directory / "frozen_model.joblib") != frozen["model_sha256"]
            or json_hash(rule) != frozen["rule_sha256"]):
        raise ValueError("Frozen experiment artifacts changed during audit.")
    difference = scores["candidate"]["macro_f1"] - scores["control"]["macro_f1"]
    memory_end = memory_state()
    for key in ("memory_limit_enforced", "enforced_virtual_memory_bytes", "enforcement_unavailable_reason"):
        if key in memory_start:
            memory_end[key] = memory_start[key]
    result = {
        "case": name, "seed": seed, "status": "completed", "rows": {key: len(value) for key, value in splits.items()},
        "supports": supports, "split": case["split"], "rule": rule, "scores": scores,
        "delta_macro_f1": difference,
        "outcome": "win" if difference > 1e-12 else "loss" if difference < -1e-12 else "tie",
        "hashes": {"frozen": frozen_hash, "source": frozen["source_sha256"],
                   "split": frozen["split_sha256"], "rule": frozen["rule_sha256"],
                   "control_predictions": json_hash(control.tolist()),
                   "candidate_predictions": json_hash(candidate.tolist()),
                   "audit_ids": json_hash(ids[audit].tolist()),
                   "audit_csv": file_hash(directory / "audit_predictions.csv")},
        "timing_seconds": {"fit_and_calibration_predict": fit_seconds, "decision_fit": tuning_seconds,
                           "audit": time.monotonic() - audit_started, "total": time.monotonic() - started},
        "resources": {"cpu_threads": 8, "wall_timeout_seconds": CASE_TIMEOUT_SECONDS,
                      "memory_limit_bytes": MEMORY_LIMIT_BYTES, **memory_end},
        "api_calls": 0,
    }
    write_json(directory / "result.json", result)
    return result


def run_prepared(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    plan = load_plan(output_dir)
    # Exclusive creation prevents accidentally rerunning previously inspected audits.
    with (output_dir / "run_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan["plan_sha256"]}, stream)
    specs = list(plan["synthetic_cases"])
    if plan["public"]["status"] == "planned":
        specs += [{"fold": fold} for fold in plan["public"]["folds"]]
    report = {"version": 1, "status": "running", "plan_sha256": plan["plan_sha256"],
              "api_calls": 0, "public": plan["public"], "results": [], "counts": {"win": 0, "tie": 0, "loss": 0}}
    started = time.monotonic()
    for index, spec in enumerate(specs):
        print(f"[{index + 1}/{len(specs)}] {spec}", flush=True)
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(index), "--run-prepared", str(output_dir)],
                cwd=ROOT, capture_output=True, text=True, timeout=CASE_TIMEOUT_SECONDS, check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"Worker exit {completed.returncode}: {completed.stderr[-4000:]}")
            name = f"public_fold{spec['fold']}-42" if "fold" in spec else f"{spec['name']}-{spec['seed']}"
            result = json.loads((output_dir / name / "result.json").read_text(encoding="utf-8"))
            report["results"].append(result)
            report["counts"][result["outcome"]] += 1
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            report["results"].append({"spec": spec, "status": "failed", "error": str(error)})
        write_json(output_dir / "report.json", report)
    report["status"] = "completed" if all(row["status"] == "completed" for row in report["results"]) else "failed"
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output_dir": str(output_dir), "API calls": 0}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--public-task", type=Path)
    parser.add_argument("--run-prepared", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_prepared:
        if args.prepare_only or args.output_dir or args.public_task:
            parser.error("--run-prepared cannot be combined with preparation options")
        if args.worker is not None:
            plan = load_plan(args.run_prepared)
            specs = list(plan["synthetic_cases"])
            if plan["public"]["status"] == "planned":
                specs += [{"fold": fold} for fold in plan["public"]["folds"]]
            run_case(args.run_prepared, specs[args.worker])
            return 0
        report = run_prepared(args.run_prepared)
        return 0 if report["status"] == "completed" else 1
    if args.worker is not None:
        parser.error("--worker requires --run-prepared")
    output, _ = prepare(args.output_dir, args.public_task)
    print(f"Prepared: {output}", flush=True)
    if args.prepare_only:
        return 0
    report = run_prepared(output)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
