"""Isolated, replayable full-v3 frequency trial; never imported by solution.py.

prepare performs schema/split preflight only. run-prepared is the sole training
entry point. re-audit reads saved evidence, never trains or repairs a run.
All raw files belong below data/experiments/frequency_candidate/. Audit answers
are controller-only, outside every solver task and absent from child configs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
BASE_SHA = "143708b91f50959052dceb15ac968ef279071f02"
RAW_ROOT = ROOT / "data" / "experiments" / "frequency_candidate"
ARMS = ("control", "frequency")
ARM_SECONDS = 90
SYNTHETIC_SECONDS = 1800
PUBLIC_ARM_SECONDS = 300
PUBLIC_SECONDS = 1400
INPUT_NAMES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
RUNTIME_FILES = ("agent/frequency_candidate.py", "frequency_candidate_cases.py", "experiment_frequency_candidate.py")
OPTIONAL_SOURCES = ("tests/test_frequency_candidate.py", "tests/test_frequency_experiment.py")
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "openai", "httpx",
            "pydantic", "tqdm", "python-dotenv", "catboost", "lightgbm", "optuna", "psutil", "xgboost")


def require(condition, message):
    """Active under -O; evidence failures are controlled ValueErrors."""
    if not condition:
        raise ValueError(message)


def evidence_validation(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError("Incomplete or malformed evidence: " + str(error)) from error
    return checked


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Evidence must be a regular file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON field: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    try:
        return json.loads(Path(path).read_bytes().decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Unreadable evidence: " + str(path)) from error


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(json_bytes(value) + b"\n")
    pending.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
    require(result.returncode == 0, "git metadata unavailable")
    return result.stdout.strip()


def source_hashes():
    names = set(_git("ls-files", "--", "*.py").splitlines())
    names.update(RUNTIME_FILES)
    names.add("requirements.txt")
    names.update(name for name in OPTIONAL_SOURCES if (ROOT / name).is_file())
    require(names, "No source files to seal")
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def environment_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "executable": str(Path(sys.executable).absolute()),
            "prefix": sys.prefix, "platform": sys.platform, "packages": versions,
            "threads": 8, "external_model_api_calls": 0, "platform_submissions": 0}


def _output_root(path):
    result = Path(path).resolve()
    require(result != RAW_ROOT and RAW_ROOT in result.parents,
            "Raw evidence must use a fresh child of data/experiments/frequency_candidate/")
    return result


def _input_hashes(task):
    from agent.task_spec import load_task
    spec = load_task(Path(task))
    return {"task.md": file_hash(task), **{name: file_hash(spec.data_dir / name) for name in INPUT_NAMES}}


def _schema(spec):
    value = spec.to_dict()
    value.pop("task_path", None)
    value.pop("data_dir", None)
    return value


def _gate_snapshot(gate):
    identifier = gate.spec.id_column
    outer = gate.outer_fit_rows.tolist()
    return {"frozen_split": gate.frozen_split, "schema": _schema(gate.spec),
            "inner_fit_rows": list(gate.fit_rows), "inner_validation_rows": list(gate.held_rows),
            "inner_split": gate.inner_split,
            "inner_fit_ids": gate.research_train.iloc[gate.fit_rows][identifier].tolist(),
            "inner_validation_ids": list(gate.expected_inner_ids),
            "outer_fit_ids": gate.train.iloc[outer][identifier].tolist(),
            "outer_validation_ids": list(gate.expected_ids),
            "full_fit_ids": gate.train[identifier].tolist(), "test_ids": list(gate.expected_test_ids),
            "semantics_allowed": gate.task_semantics["adaptive_research_allowed"]}


def _prepare_entry(output, descriptor, task, validation, actuals, metadata_value, index, cap):
    from agent.task_spec import load_task, load_task_data
    from agent.splits import split_rows
    from agent.research_workspace import ResearchWorkspace
    spec = load_task(task)
    train, test, _, _, _, y, features, _, _, classification = load_task_data(spec)
    fit, held, split = split_rows(train, y, features, classification, validation)
    frozen = {"fit_rows": fit.tolist(), "validation_rows": held.tolist(), "choice": validation, "split": split}
    key = descriptor["name"] + "__" + str(descriptor["seed"])
    gate = ResearchWorkspace(task, output / "cases" / key / "preflight", frozen_split=frozen)
    snapshot = _gate_snapshot(gate)
    require(snapshot["inner_fit_rows"] and snapshot["inner_validation_rows"], "Preflight lacks INNER split")
    audit_path = output / "audit" / (key + ".json")
    audit = {"version": 1, "case": key, "ids": test[spec.id_column].tolist(),
             "actuals": list(actuals), "schema": _schema(spec), "scope": "external_audit"}
    write_json(audit_path, audit)
    return {**descriptor, "key": key, "template": str(task.parent.relative_to(output)),
            "audit": str(audit_path.relative_to(output)), "audit_hash": file_hash(audit_path),
            "input_hashes": _input_hashes(task), "snapshot": snapshot, "snapshot_hash": json_hash(snapshot),
            "metadata": metadata_value, "arm_cap_seconds": cap,
            "order": list(ARMS if index % 2 == 0 else reversed(ARMS))}


def _prepare_public(output, public_task):
    """Known TRAIN grouped folds only; no competition TEST answers are used."""
    checked = [str(Path(public_task).resolve())] if public_task else [str(ROOT / "data/classification/task.md")]
    task = Path(checked[0])
    if not task.is_file():
        return {"status": "skipped_missing_data", "checked_paths": checked, "entries": []}
    from agent.task_spec import load_task, load_task_data
    from experiment_classification_weights import grouped_fold
    import pandas as pd
    candidate_directories = (task.parent, task.parent / "inputs", task.parent / "data")
    expected_paths = [str(directory / name) for directory in candidate_directories for name in INPUT_NAMES]
    checked.extend(expected_paths)
    if not any(all((directory / name).is_file() for name in INPUT_NAMES) for directory in candidate_directories):
        return {"status": "skipped_missing_data", "checked_paths": checked,
                "missing_paths": [name for name in expected_paths if not Path(name).is_file()], "entries": []}
    spec = load_task(task)
    checked += [str(spec.data_dir / name) for name in INPUT_NAMES]
    missing = [str(path) for path in [task, *(spec.data_dir / name for name in INPUT_NAMES)] if not path.is_file()]
    if missing:
        return {"status": "skipped_missing_data", "checked_paths": checked, "missing_paths": missing, "entries": []}
    train, _, _, _, _, y, features, _, _, classification = load_task_data(spec)
    require(classification, "Optional known TRAIN task must be classification")
    entries = []
    for fold in (0, 1):
        fitting, held, split = grouped_fold(train, y, features, fold)
        name = "known_train_grouped_fold"
        descriptor = {"name": name, "seed": fold, "stage": "known_public_sanity"}
        key = name + "__" + str(fold)
        directory = output / "cases" / key / "template"
        data = directory / "inputs"
        data.mkdir(parents=True)
        train.iloc[fitting].to_csv(data / "train.csv", index=False, lineterminator="\n")
        train.iloc[held].drop(columns=[spec.target_column]).to_csv(data / "test.csv", index=False, lineterminator="\n")
        shutil.copyfile(spec.data_dir / "data_dictionary.csv", data / "data_dictionary.csv")
        sample = pd.DataFrame({name: [""] * len(held) for name in spec.output_columns})
        sample[spec.id_column] = train.iloc[held][spec.id_column].tolist()
        for column in spec.output_columns:
            if column != spec.id_column:
                sample[column] = (spec.classes[0] if spec.prediction_kind == "labels" else 1 / len(spec.classes))
        sample.to_csv(data / "sample_submission.csv", index=False, lineterminator="\n")
        local_task = directory / "task.md"
        local_task.write_text(task.read_bytes().decode("utf-8-sig") + "\nUse inputs/train.csv and inputs/test.csv; save submission.csv.\n",
                              encoding="utf-8", newline="\n")
        entry = _prepare_entry(output, descriptor, local_task, {"strategy": "grouped"},
                               train.iloc[held][spec.target_column].tolist(),
                               {"source_task": str(task), "source_inputs": _input_hashes(task),
                                "known_train_fold": fold, "external_split": split,
                                "external_fit_positions": fitting.tolist(), "external_audit_positions": held.tolist(),
                                "confirmation": False}, fold, PUBLIC_ARM_SECONDS)
        entries.append(entry)
    return {"status": "prepared", "checked_paths": checked, "entries": entries}


def prepare(output_dir, public_task=None):
    """Seal exact data, native splits, source and environment; perform zero fits."""
    from frequency_candidate_cases import make_case, write_case, suite_cases, case_definitions
    output = _output_root(output_dir)
    require(not output.exists(), "prepare requires a new evidence directory")
    require(_git("merge-base", "HEAD", BASE_SHA) == BASE_SHA, "Unexpected base ancestry")
    output.mkdir(parents=True)
    entries = []
    for index, descriptor in enumerate(suite_cases()):
        case = make_case(descriptor["name"], descriptor["seed"])
        key = case.name + "__" + str(case.seed)
        task = write_case(case, output / "cases" / key / "template")
        entries.append(_prepare_entry(output, descriptor, task, case.validation, case.audit_actuals,
                                      case.metadata, index, ARM_SECONDS))
    public = _prepare_public(output, public_task)
    plan = {"version": 1, "created_at": utc_now(), "base_sha": BASE_SHA,
            "executable_sha": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "sources": source_hashes(), "environment": environment_versions(),
            "case_definitions": case_definitions(), "entries": entries, "public": public,
            "method": {"version": 1, "arm_seconds": ARM_SECONDS, "synthetic_seconds": SYNTHETIC_SECONDS,
                       "public_arm_seconds": PUBLIC_ARM_SECONDS, "public_seconds": PUBLIC_SECONDS,
                       "hook": "continue_locally: frequency_attempt then original continuation",
                       "admission": "actual _assess_candidate origin=frequency",
                       "native": "run_agent_v3(client=None,local_only=True)",
                       "confirmation_case_families": ["category_prevalence_classification", "category_no_frequency_signal"],
                       "development_seed": 701, "confirmation_seeds": [1907, 7919],
                       "frequency": {"max_columns": 16, "max_categories": 4096, "unknown": 0,
                                     "formula": "log1p(count_fit(category))", "missing": "genuine_missing_own_category"},
                       "external_model_api_calls": 0, "platform_submissions": 0}}
    plan["method_sha"] = json_hash({"sources": plan["sources"], "method": plan["method"],
                                    "cases": plan["case_definitions"], "entries": entries, "public": public})
    write_json(output / "plan.json", plan)
    write_json(output / "preflight.json", {"status": "prepared", "fits": 0, "plan_hash": file_hash(output / "plan.json"),
                                         "method_sha": plan["method_sha"], "pairs": len(entries)})
    return plan


@evidence_validation
def load_plan(output_dir, check_environment=True):
    output = _output_root(output_dir)
    plan = read_json(output / "plan.json")
    preflight = read_json(output / "preflight.json")
    require(plan.get("version") == 1 and plan.get("base_sha") == BASE_SHA, "Invalid sealed base/version")
    require(file_hash(output / "plan.json") == preflight.get("plan_hash"), "Plan drift")
    require(plan.get("sources") == source_hashes(), "Source drift after preparation")
    if check_environment:
        require(plan.get("environment") == environment_versions(), "Environment drift after preparation")
    expected_method = json_hash({"sources": plan["sources"], "method": plan["method"], "cases": plan["case_definitions"],
                                 "entries": plan["entries"], "public": plan["public"]})
    require(plan.get("method_sha") == expected_method == preflight.get("method_sha"), "Method seal drift")
    from frequency_candidate_cases import suite_cases
    require([{key: entry[key] for key in ("name", "seed", "stage")} for entry in plan["entries"]] == suite_cases(),
            "Incomplete or reordered sealed synthetic suite")
    for entry in [*plan["entries"], *plan["public"]["entries"]]:
        require(json_hash(entry["snapshot"]) == entry["snapshot_hash"], "Split snapshot drift")
        require(_input_hashes(output / entry["template"] / "task.md") == entry["input_hashes"], "Sealed input drift")
        require(file_hash(output / entry["audit"]) == entry["audit_hash"], "Audit answer drift")
    return plan


def _candidate_identity(candidate):
    if candidate is None:
        return None
    candidate.verify()
    return {"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash,
            "code_path": str(candidate.code_path), "recipe_path": str(candidate.recipe_path), "recipe": candidate.recipe}


def _baseline_guard(state, baseline, identity):
    state.check_source()
    state.gate.check_inputs()
    require(state.baseline is baseline, "Frequency hook replaced the frozen baseline object")
    require(_candidate_identity(baseline) == identity, "Frequency hook changed frozen baseline bytes/recipe")


def frequency_attempt(state):
    """One measured native trial; no hand-written admission or queue changes."""
    from agent import research_v3 as host
    from agent.frequency_candidate import (fit_frequency_transform, frequency_candidate_source,
                                           make_frequency_recipe, FrequencyContractError)
    from agent.research_candidate import freeze_candidate, CandidateExecutionError
    gate = state.gate
    report = {"version": 1, "status": "skipped", "attempted": 0, "scope": "inner_selection", "origin": "frequency"}
    state.report["frequency_experiment"] = report
    def skip(reason):
        report["reason"] = reason
        state.save()
        return report
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return skip("inner_research_required")
    if not gate.task_semantics.get("adaptive_research_allowed", True):
        return skip("task_semantics_incomplete")
    if not gate.fit_rows or not gate.held_rows or not gate.inner_split:
        return skip("inner_split_unavailable")
    if state.baseline is None or state.baseline_inner is None:
        return skip("measured_baseline_required")
    if not gate.categorical:
        return skip("no_categorical_features")
    if state.research_timeout() <= 2:
        return skip("finalization_reserve")
    baseline = state.baseline
    identity = _candidate_identity(baseline)
    _baseline_guard(state, baseline, identity)
    report["baseline"] = identity
    try:
        # Eligibility reads only the current INNER fitting feature rows. No fit
        # is launched for a limit/collision refusal; the execution fits afresh.
        fitting = gate.research_train.iloc[gate.fit_rows].loc[:, list(gate.features)]
        fit_frequency_transform(fitting, categorical_columns=list(gate.categorical), feature_columns=list(gate.features))
        source = state.run_dir / "frequency_source.py"
        source.write_text(frequency_candidate_source(baseline.code_path.read_bytes().decode("utf-8")),
                          encoding="utf-8", newline="\n")
        recipe = make_frequency_recipe(baseline.recipe, list(gate.categorical))
        candidate = freeze_candidate(source, recipe, state.run_dir / "frequency_frozen")
        report.update(status="running", attempted=1, candidate=_candidate_identity(candidate))
        state.save()
        feedback = host._assess_candidate(state, candidate, {"hypothesis":
            "Add only fitting-partition log1p(category count) numeric features to the exact frozen baseline recipe."},
            origin="frequency")
        report.update(status="completed", reason=feedback["reason"], feedback=feedback)
        state.event("frequency_experiment", result=feedback)
    except FrequencyContractError as error:
        report.update(status="skipped", reason="frequency_contract_refusal", error=str(error))
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        report.update(status="failed", reason="candidate_execution_failed", error=str(error))
        if isinstance(error, CandidateExecutionError):
            report["execution"] = error.execution
        state.event("frequency_experiment_failed", error=str(error))
    finally:
        _baseline_guard(state, baseline, identity)
        state.save()
    return report


def _invocation_files(directory):
    return {name: {"path": str(directory / name), "sha256": file_hash(directory / name)} for name in
            ("candidate.py", "recipe.json", "fit.csv", "predict.csv", "context.json", "predictions.csv", "receipt.json", "frequency_evidence.json")
            if (directory / name).is_file()}


@contextmanager
def installed_hooks(arm, frozen_split, observation_path):
    """Process-local injection/observation; all original native returns survive."""
    from agent import research_v3 as host
    require(arm in ARMS, "Unknown arm")
    original_workspace, original_run = host.ResearchWorkspace, host.run_candidate
    original_continue, original_assess = host.continue_locally, host._assess_candidate
    collector = {"version": 1, "arm": arm, "invocations": [], "admissions": [], "continuations": 0}
    def save():
        write_json(observation_path, collector)
    def workspace(task, directory, **kwargs):
        require(not kwargs, "Unexpected native workspace arguments")
        gate = original_workspace(task, directory, frozen_split=frozen_split)
        collector["gate"] = _gate_snapshot(gate)
        collector["manifest"] = {"path": str(gate.workspace / "research_validation.json"),
                                 "sha256": file_hash(gate.workspace / "research_validation.json")}
        save()
        return gate
    def run(candidate, *args, **kwargs):
        directory = Path(kwargs["output_dir"])
        before = set(directory.iterdir()) if directory.exists() else set()
        record = {"index": len(collector["invocations"]), "phase": kwargs.get("phase"),
                  "candidate": _candidate_identity(candidate), "verified": False, "started": time.monotonic(),
                  "timeout_seconds": kwargs.get("timeout"), "fit_rows": len(kwargs["fit_frame"]),
                  "predict_rows": len(kwargs["predict_frame"])}
        collector["invocations"].append(record)
        save()
        try:
            result = original_run(candidate, *args, **kwargs)
            result.verify()
            record.update(verified=True, metric=result.metric, execution=result.execution,
                          code_hash=result.code_hash, recipe_hash=result.recipe_hash, phase=result.phase,
                          output_path=str(result.output_path), receipt_path=str(result.receipt_path),
                          output_hash=result.output_hash, receipt_hash=result.receipt_hash,
                          output_size=result.output_size)
            return result  # EXACT native object, not a reconstructed/faked result.
        except BaseException as error:
            record.update(error=type(error).__name__ + ": " + str(error))
            if hasattr(error, "execution"):
                record["execution"] = error.execution
            raise
        finally:
            after = set(directory.iterdir()) if directory.exists() else set()
            record["invocations"] = [{"directory": str(path), "files": _invocation_files(path)}
                                     for path in sorted(after - before) if path.is_dir()]
            record["ended"] = time.monotonic()
            save()
    def assess(state, candidate, hypothesis, *, origin):
        feedback = original_assess(state, candidate, hypothesis, origin=origin)
        require(type(feedback.get("accepted")) is bool and feedback.get("verified") is True
                and feedback.get("scope") == "inner_selection" and feedback.get("origin") == origin,
                "Native admission returned invalid typed feedback")
        require(feedback.get("code_hash") == candidate.code_hash and feedback.get("recipe_hash") == candidate.recipe_hash,
                "Admission identity mismatch")
        collector["admissions"].append({"origin": origin, "candidate": _candidate_identity(candidate), "feedback": feedback})
        save()
        return feedback  # Preserve original verdict object and normal host mutation.
    def continuation(state):
        collector["continuations"] += 1
        require(collector["continuations"] == 1, "Unexpected repeated local continuation")
        baseline = state.baseline
        identity = _candidate_identity(baseline)
        collector["baseline_before"] = identity
        collector["baseline_inner"] = state.baseline_inner.metric if hasattr(state.baseline_inner, "metric") else state.baseline_inner
        save()
        try:
            if arm == "frequency":
                collector["frequency_attempt"] = frequency_attempt(state)
                save()
            return original_continue(state)
        finally:
            _baseline_guard(state, baseline, identity)
            collector["baseline_after"] = _candidate_identity(baseline)
            save()
    host.ResearchWorkspace, host.run_candidate = workspace, run
    host.continue_locally, host._assess_candidate = continuation, assess
    try:
        save()
        yield collector
    finally:
        host.ResearchWorkspace, host.run_candidate = original_workspace, original_run
        host.continue_locally, host._assess_candidate = original_continue, original_assess
        save()


def child_bootstrap(config_path):
    return ("import time\nnative_started=time.monotonic()\nimport sys\n"
            f"sys.path.insert(0,{str(ROOT)!r})\n"
            "from experiment_frequency_candidate import worker_run\n"
            f"worker_run({str(config_path)!r},native_started)\n")


def worker_run(config_path, native_started):
    """Real v3 worker. The bootstrap clock precedes this module's imports."""
    config = read_json(config_path)
    directory = Path(config_path).parent
    summary = {"version": 1, "arm": config["arm"], "native_started": native_started,
               "arm_cap_seconds": config["arm_cap_seconds"], "status": "failed",
               "config_hash": file_hash(config_path), "environment": environment_versions(),
               "sources": source_hashes(), "external_model_api_calls": 0, "platform_submissions": 0}
    try:
        require(summary["environment"] == config["environment"], "Child environment differs from sealed environment")
        require(summary["sources"] == config["sources"], "Child source drift")
        # Native host imports count against the original deadline.
        from agent.research_v3 import run_agent_v3
        deadline = native_started + config["arm_cap_seconds"]
        with installed_hooks(config["arm"], config["snapshot"]["frozen_split"], directory / "observation.json") as observer:
            host_started = time.monotonic()
            remaining = deadline - host_started
            require(remaining > 0, "Native imports exhausted the complete arm budget")
            summary.update(host_started=host_started, host_budget_seconds=remaining)
            run_dir = run_agent_v3(Path(config["task"]), client=None, local_only=True, time_budget=remaining)
            summary.update(run_dir=str(run_dir), status="returned", host_run_hash=file_hash(Path(run_dir) / "run.json"))
            require(observer.get("gate") == config["snapshot"], "Actual native split/schema differs from sealed preflight")
        summary["observation_hash"] = file_hash(directory / "observation.json")
    except BaseException as error:
        summary["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        summary["native_ended"] = time.monotonic()
        summary["native_elapsed_seconds"] = summary["native_ended"] - native_started
        write_json(directory / "worker_summary.json", summary)


def _regular_under(path, directory):
    original, directory = Path(path), Path(directory).resolve()
    require(not original.is_symlink(), "Evidence must not be a symlink")
    path = original.resolve()
    require(directory in path.parents, "Evidence path escapes its arm")
    require(not any(parent.is_symlink() for parent in original.parents if parent != directory),
            "Evidence ancestors must not be symlinks")
    require(path.is_file() and not path.is_symlink(), "Missing regular evidence: " + str(path))
    return path


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _metric_from_frame(frame, actuals, schema):
    """Audit scoring from the saved CSV; no solver/printed scores are trusted."""
    classes = schema["classes"]
    target = schema["target_column"]
    metric_name = schema["metric"]["name"]
    require(len(frame) == len(actuals) and len(frame) > 0, "Audit answer row count mismatch")
    if metric_name == "Macro F1":
        predictions = frame[target].tolist()
        actuals = list(map(str, actuals))
        require(set(predictions) <= set(classes) and set(actuals) <= set(classes), "Unknown audit class")
        per_class = {}
        for label in classes:
            tp = sum(a == label and p == label for a, p in zip(actuals, predictions))
            fp = sum(a != label and p == label for a, p in zip(actuals, predictions))
            fn = sum(a == label and p != label for a, p in zip(actuals, predictions))
            denominator = 2 * tp + fp + fn
            per_class[label] = {"f1": 2 * tp / denominator if denominator else 0.0,
                                "support": sum(a == label for a in actuals), "tp": tp, "fp": fp, "fn": fn}
        active = set(actuals) | set(predictions)
        return {"name": metric_name, "value": sum(per_class[label]["f1"] for label in active) / len(active),
                "higher_is_better": True}, per_class
    if metric_name == "RMSE":
        predictions = [float(item) for item in frame[target]]
        answers = [float(item) for item in actuals]
        require(all(math.isfinite(item) for item in predictions + answers), "Nonfinite audit regression values")
        value = math.sqrt(math.fsum((predicted - actual) ** 2 for predicted, actual in zip(predictions, answers)) / len(answers))
        return {"name": "RMSE", "value": value, "higher_is_better": False}, {}
    # Optional public datasets may declare another existing native metric. Its
    # implementation is sealed, inputs remain external CSV/audit JSON only.
    from agent.metrics import get_metric, score_predictions
    specification = get_metric(metric_name)
    probabilities = schema["probability_columns"]
    if specification.prediction_kind == "probabilities":
        predictions = frame[[column for column, _ in probabilities]].to_numpy()
        provided = tuple(label for _, label in probabilities)
    else:
        predictions, provided = frame[target].to_numpy(), None
    value = score_predictions(specification, actuals, predictions, classes=tuple(classes),
                              prediction_classes=provided, positive_class=schema["positive_class"])
    return {"name": metric_name, "value": value, "higher_is_better": specification.higher_is_better}, {}


def _same_metric(actual, expected):
    require(isinstance(actual, dict) and set(actual) == {"name", "value", "higher_is_better"}, "Malformed measured metric")
    require(actual["name"] == expected["name"] and type(actual["higher_is_better"]) is bool
            and actual["higher_is_better"] == expected["higher_is_better"], "Measured metric semantics differ")
    require(type(actual["value"]) in (int, float) and math.isfinite(actual["value"])
            and math.isclose(actual["value"], expected["value"], rel_tol=1e-10, abs_tol=1e-12), "Measured metric differs from saved CSV")


def _independent_frequency_trace(trace, fit, predict, context, baseline):
    """Reconstruct dictionaries and augmented bytes without calling transform."""
    columns = context["features"]
    categorical = context["categorical"]
    added = ["__frequency_log1p_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16] for name in categorical]
    require(0 < len(categorical) <= 16 and not set(added) & set(columns), "Invalid frequency schema")
    expected_columns, dictionaries = [], []
    for name, output in zip(categorical, added):
        counts = Counter(fit[name].tolist())
        require(len(counts) <= 4096, "Frequency evidence exceeds fixed cardinality limit")
        dictionaries.append(counts)
        expected_columns.append({"source_column": name, "output_column": output,
                                 "counts": [{"kind": "string", "value": value, "count": counts[value]} for value in sorted(counts)]})
    transform = {"version": 1, "kind": "fit_only_frequency_transform", "formula": "log1p(count_fit(category))",
                 "missing_policy": "genuine_missing_own_category", "unknown_value": 0.0,
                 "feature_columns": columns, "categorical_columns": categorical,
                 "added_columns": added, "fit_rows": len(fit), "limits": {"max_columns": 16, "max_categories": 4096},
                 "columns": expected_columns}
    require(trace.get("transform") == transform, "Frequency dictionary is not exactly fitting-partition counts")
    def augmented_hash(frame):
        digest = hashlib.sha256()
        digest.update(b'{"columns":')
        digest.update(json_bytes(columns + added))
        digest.update(b',"rows":[')
        for row_index, row in enumerate(frame.loc[:, columns].itertuples(index=False, name=None)):
            if row_index:
                digest.update(b",")
            token_row = [["str", value] for value in row]
            for name, counts in zip(categorical, dictionaries):
                value = row[columns.index(name)]
                token_row.append(["float", math.log1p(counts[value]).hex() if value in counts else 0.0.hex()])
            digest.update(json_bytes(token_row))
        digest.update(b"]}")
        return digest.hexdigest()
    recipe = context["recipe"]
    require(recipe.get("family") == "frequency_candidate_v1" and recipe.get("base_recipe") == baseline["recipe"],
            "Frequency wrapper changed the frozen baseline recipe")
    require(recipe.get("frequency") == {"categorical_columns": categorical, "max_columns": 16, "max_categories": 4096,
            "missing_policy": "genuine_missing_own_category", "feature_name_prefix": "__frequency_log1p_"},
            "Frequency wrapper options drift")
    expected = {"version": 1, "phase": context["phase"], "nonce": context["nonce"],
                "code_hash": context["code_hash"], "recipe_hash": context["recipe_hash"],
                "base_recipe_hash": baseline["recipe_hash"], "base_source_hash": baseline["code_hash"],
                "transform": transform, "features": columns + added, "numeric": context["numeric"] + added,
                "categorical": categorical, "fit_rows": len(fit), "predict_rows": len(predict),
                "augmented_fit_sha256": augmented_hash(fit), "augmented_predict_sha256": augmented_hash(predict)}
    require(trace == expected, "Frequency augmented-frame/context evidence differs from independent reconstruction")
    return {"phase": context["phase"], "fit_rows": len(fit), "predict_rows": len(predict),
            "dictionary_hash": json_hash(expected_columns), "added_columns": added}


def _check_execution(execution, label):
    require(isinstance(execution, dict) and type(execution.get("exit_code")) is int and execution["exit_code"] == 0,
            label + " did not exit zero")
    require(execution.get("timed_out") is False and execution.get("memory_limit_exceeded") is False
            and not execution.get("error"), label + " was censored or failed")


@evidence_validation
def audit_arm(output_dir, entry, arm, summary, plan):
    """Reject unsupported success; independently verify every native artifact."""
    from agent.task_spec import TaskSpec
    output = _output_root(output_dir).resolve()
    require(arm in ARMS and summary.get("arm") == arm and summary.get("case") == entry["key"], "Arm identity mismatch")
    directory = output / summary["directory"]
    require(directory.resolve() == (output / "runs" / entry["key"] / arm).resolve(), "Unexpected arm directory")
    proof = {}
    def pin(path, expected=None):
        path = _regular_under(path, directory)
        digest = file_hash(path)
        if expected is not None:
            require(digest == expected, "Evidence hash drift: " + str(path))
        proof[str(path.relative_to(output))] = digest
        return path
    config_path = pin(directory / "config.json", summary["config_hash"])
    config = read_json(config_path)
    require(set(config) == {"version", "arm", "task", "arm_cap_seconds", "snapshot", "sources", "environment", "method_sha"},
            "Worker configuration has unsealed or audit fields")
    require(config["version"] == 1 and config["arm"] == arm and config["snapshot"] == entry["snapshot"]
            and config["sources"] == plan["sources"] and config["environment"] == plan["environment"]
            and config["method_sha"] == plan["method_sha"] and config["arm_cap_seconds"] == entry["arm_cap_seconds"],
            "Worker configuration drift")
    task = Path(config["task"])
    require(task.resolve() == (directory / "task/task.md").resolve(), "Solver task is outside its own arm")
    require(_input_hashes(task) == entry["input_hashes"], "Solver input mutation")
    launcher = read_json(pin(directory / "launcher.json", summary["launcher_hash"]))
    _check_execution(launcher, "Native arm process")
    worker = read_json(pin(directory / "worker_summary.json", summary["worker_summary_hash"]))
    require(worker.get("status") == "returned" and not worker.get("error") and worker.get("arm") == arm,
            "Native worker never returned successfully")
    require(worker.get("sources") == plan["sources"] and worker.get("environment") == plan["environment"]
            and worker.get("config_hash") == summary["config_hash"], "Native worker environment/source/config drift")
    cap = entry["arm_cap_seconds"]
    started, host_started, ended = (worker.get(key) for key in ("native_started", "host_started", "native_ended"))
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (started, host_started, ended)), "Missing native timing proof")
    require(started <= host_started <= ended and 0 < worker["host_budget_seconds"] <= cap
            and math.isclose(host_started + worker["host_budget_seconds"], started + cap, abs_tol=1e-6, rel_tol=0),
            "Imports were outside the original arm budget")
    require(ended - started <= cap and math.isclose(worker["native_elapsed_seconds"], ended - started, abs_tol=1e-6, rel_tol=0),
            "Native wall-time cap was exceeded")
    require(worker.get("external_model_api_calls") == 0 and worker.get("platform_submissions") == 0, "Unexpected external actions")
    observation_path = pin(directory / "observation.json", summary["observation_hash"])
    require(file_hash(observation_path) == worker.get("observation_hash"), "Worker observation receipt mismatch")
    observation = read_json(observation_path)
    require(observation.get("arm") == arm and observation.get("gate") == entry["snapshot"], "Native row/split/schema proof differs")
    require(observation.get("continuations") == 1, "Native local continuation did not run exactly once")
    baseline = observation.get("baseline_before")
    require(baseline and baseline == observation.get("baseline_after"), "Frozen baseline changed across continuation")
    for name, digest in (("code_path", baseline["code_hash"]), ("recipe_path", baseline["recipe_hash"])):
        pin(baseline[name], digest)
    require(read_json(baseline["recipe_path"]) == baseline["recipe"], "Baseline recipe object/bytes differ")
    require(json_hash(baseline["recipe"]) == baseline["recipe_hash"], "Baseline recipe is not canonical frozen JSON")
    from agent.research_v3 import BASELINE_SOURCE
    native_source = BASELINE_SOURCE.replace("\n", os.linesep).encode("utf-8")
    require(Path(baseline["code_path"]).read_bytes() == native_source, "Frozen baseline is not the actual native baseline source")
    manifest = read_json(pin(observation["manifest"]["path"], observation["manifest"]["sha256"]))
    require(manifest.get("fit_rows") == entry["snapshot"]["inner_fit_rows"]
            and manifest.get("validation_rows") == entry["snapshot"]["inner_validation_rows"]
            and manifest.get("split") == entry["snapshot"]["inner_split"], "Native INNER manifest drift")
    run_dir = Path(worker["run_dir"])
    require((directory / "task/.agent_runs").resolve() in run_dir.resolve().parents, "Host run path escaped arm")
    host = read_json(pin(run_dir / "run.json", summary["host_run_hash"]))
    require(worker.get("host_run_hash") == summary["host_run_hash"], "Native host report receipt drift")
    require(host.get("research_protocol_version") == 3 and host.get("status") == "completed"
            and host.get("publication", {}).get("new_result_published") is True, "Fallback/old/partial output cannot be completed")
    require(host.get("api", {}).get("attempts") == 0 and host.get("api", {}).get("responses") == 0, "Model API was used")
    require(host.get("time_budget_seconds") == worker["host_budget_seconds"], "Host received a fresh or shortened time budget")
    require(host.get("frozen_baseline") == {key: baseline[key] for key in ("code_hash", "recipe_hash")}, "Host frozen baseline differs")
    require(host.get("quality_options", {}).get("search_profile", "legacy") == "legacy", "Control search profile changed")
    schema = entry["snapshot"]["schema"]
    spec = TaskSpec.from_dict({**schema, "task_path": str(task), "data_dir": str(task.parent / "inputs")})
    source_train = _read_csv(spec.data_dir / "train.csv")
    source_test = _read_csv(spec.data_dir / "test.csv")
    for path in [task, *(spec.data_dir / name for name in INPUT_NAMES)]:
        pin(path)
    identifier, target, features = schema["id_column"], schema["target_column"], schema["features"]
    frames = {"inner_selection": (entry["snapshot"]["inner_fit_ids"], entry["snapshot"]["inner_validation_ids"]),
              "outer_validation": (entry["snapshot"]["outer_fit_ids"], entry["snapshot"]["outer_validation_ids"]),
              "full_refit": (entry["snapshot"]["full_fit_ids"], entry["snapshot"]["test_ids"])}
    train_by_id = source_train.set_index(identifier, drop=False)
    test_by_id = source_test.set_index(identifier, drop=False)
    verified, frequency_traces, nonces = [], [], set()
    all_records = observation.get("invocations")
    require(isinstance(all_records, list) and all_records, "No actual native candidate invocations")
    for index, record in enumerate(all_records):
        require(record.get("index") == index and type(record.get("verified")) is bool, "Malformed invocation record")
        invocations = record.get("invocations", [])
        require(isinstance(invocations, list) and len(invocations) <= 1, "Ambiguous native invocation")
        if not record["verified"]:
            # Retain/pin failures too; none can prove a full-data result.
            for invocation in invocations:
                for artifact in invocation["files"].values():
                    pin(artifact["path"], artifact["sha256"])
            continue
        require(len(invocations) == 1, "Verified result lacks fresh invocation directory")
        invocation = invocations[0]
        paths = {name: pin(artifact["path"], artifact["sha256"]) for name, artifact in invocation["files"].items()}
        required = {"candidate.py", "recipe.json", "fit.csv", "predict.csv", "context.json", "predictions.csv", "receipt.json"}
        require(required <= set(paths), "Verified invocation lacks required receipt/input/output files")
        require(all(path.parent == Path(invocation["directory"]).resolve() for path in paths.values()), "Invocation artifact directory mismatch")
        context, recipe, receipt = (read_json(paths[name]) for name in ("context.json", "recipe.json", "receipt.json"))
        phase = record["phase"]
        require(phase in frames and context.get("phase") == phase and receipt.get("phase") == phase, "Wrong invocation phase")
        nonce = context.get("nonce")
        require(isinstance(nonce, str) and nonce and nonce not in nonces, "Missing/reused invocation nonce")
        nonces.add(nonce)
        candidate = record["candidate"]
        require(record.get("code_hash") == candidate["code_hash"] == file_hash(paths["candidate.py"])
                and record.get("recipe_hash") == candidate["recipe_hash"] == file_hash(paths["recipe.json"]), "Frozen candidate hash drift")
        pin(candidate["code_path"], candidate["code_hash"])
        pin(candidate["recipe_path"], candidate["recipe_hash"])
        require(recipe == candidate["recipe"] == context.get("recipe") and json_hash(recipe) == candidate["recipe_hash"], "Frozen recipe/context drift")
        output_hash = file_hash(paths["predictions.csv"])
        require(receipt == {"version": 3, "nonce": nonce, "phase": phase, "code_hash": candidate["code_hash"],
                            "recipe_hash": candidate["recipe_hash"], "output_hash": output_hash}, "Native execution receipt mismatch")
        require(record.get("output_hash") == output_hash and record.get("receipt_hash") == file_hash(paths["receipt.json"])
                and record.get("output_size") == paths["predictions.csv"].stat().st_size,
                "Native result output/receipt identity differs")
        require(Path(record["output_path"]).resolve() == paths["predictions.csv"]
                and Path(record["receipt_path"]).resolve() == paths["receipt.json"], "Result paths differ from fresh receipt")
        require(context.get("version") == 3 and context.get("protocol_version") == 3
                and context.get("code_hash") == candidate["code_hash"] and context.get("recipe_hash") == candidate["recipe_hash"],
                "Candidate context identity differs")
        for name in ("features", "numeric", "categorical", "classes", "positive_class", "metric"):
            require(context.get(name) == schema[name], "Native candidate schema changed: " + name)
        require(context.get("task_type") == schema["metric"]["task_type"]
                and context.get("prediction_kind") == schema["metric"]["prediction_kind"], "Native prediction semantics changed")
        context_schema = dict(context["task_spec"])
        context_schema.pop("task_path", None)
        context_schema.pop("data_dir", None)
        require(context_schema == schema, "Native serialized task contract drift")
        fit_ids, predict_ids = frames[phase]
        expected_fit = train_by_id.loc[fit_ids, [identifier, *features, target]].reset_index(drop=True)
        predict_source = test_by_id if phase == "full_refit" else train_by_id
        expected_predict = predict_source.loc[predict_ids, [identifier, *features]].reset_index(drop=True)
        fit, predict, predictions = (_read_csv(paths[name]) for name in ("fit.csv", "predict.csv", "predictions.csv"))
        require(fit.equals(expected_fit) and predict.equals(expected_predict), "Actual invocation fit/predict rows or values differ from sealed split")
        require(record.get("fit_rows") == len(fit_ids) and record.get("predict_rows") == len(predict_ids), "Invocation row telemetry differs")
        spec.validate_frame(predictions, predict_ids)
        _check_execution(record.get("execution"), "Candidate invocation")
        if phase == "full_refit":
            require(record.get("metric") is None, "Audit answers leaked into native full refit")
        else:
            actuals = train_by_id.loc[predict_ids, target].tolist()
            metric, _ = _metric_from_frame(predictions, actuals, schema)
            _same_metric(record.get("metric"), metric)
        is_frequency = recipe.get("family") == "frequency_candidate_v1"
        if is_frequency:
            from agent.frequency_candidate import frequency_candidate_source
            expected_source = frequency_candidate_source(Path(baseline["code_path"]).read_bytes().decode("utf-8"))
            require(hashlib.sha256(expected_source.encode("utf-8")).hexdigest() == candidate["code_hash"],
                    "Frequency source does not wrap the unchanged actual baseline")
            require(arm == "frequency" and "frequency_evidence.json" in paths, "Frequency fit lacks its own phase evidence")
            trace = _independent_frequency_trace(read_json(paths["frequency_evidence.json"]), fit, predict, context, baseline)
            trace["code_hash"], trace["recipe_hash"] = candidate["code_hash"], candidate["recipe_hash"]
            frequency_traces.append(trace)
        else:
            require("frequency_evidence.json" not in paths, "Unexpected transform evidence in unchanged candidate")
        verified.append({"phase": phase, "code_hash": candidate["code_hash"], "recipe_hash": candidate["recipe_hash"],
                         "receipt": str(paths["receipt.json"]), "output": str(paths["predictions.csv"]),
                         "metric": record.get("metric"), "frequency": is_frequency})
    baseline_inner = [row for row in verified if row["phase"] == "inner_selection"
                      and row["code_hash"] == baseline["code_hash"] and row["recipe_hash"] == baseline["recipe_hash"]]
    require(len(baseline_inner) == 1, "Actual baseline INNER measurement missing/duplicated")
    _same_metric(observation.get("baseline_inner"), baseline_inner[0]["metric"])
    admissions = observation.get("admissions")
    require(isinstance(admissions, list), "Missing typed admission journal")
    best_inner = baseline_inner[0]["metric"]
    accepted_identity = None
    for admission in admissions:
        candidate, feedback = admission["candidate"], admission["feedback"]
        require(type(feedback.get("accepted")) is bool and feedback.get("verified") is True
                and feedback.get("origin") == admission["origin"] and feedback.get("scope") == "inner_selection",
                "Invalid typed native admission")
        measured = [row for row in verified if row["phase"] == "inner_selection"
                    and (row["code_hash"], row["recipe_hash"]) == (candidate["code_hash"], candidate["recipe_hash"])]
        require(len(measured) == 1 and feedback.get("code_hash") == candidate["code_hash"]
                and feedback.get("recipe_hash") == candidate["recipe_hash"], "Admission has no matching measured INNER receipt")
        _same_metric(feedback.get("metric"), measured[0]["metric"])
        timing = feedback.get("timing", {})
        require(type(timing.get("feasible_estimate")) is bool and type(timing.get("within_stage_limit")) is bool
                and timing.get("guaranteed") is False, "Admission lacks measured refit feasibility")
        estimated, available = timing.get("estimated_finalization_seconds"), timing.get("remaining_seconds")
        require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in (estimated, available)),
                "Admission has invalid measured cost/remaining time")
        # The native feasibility decision precedes serialization of remaining
        # time. Permit only a 1 ms logging boundary; this never changes any
        # execution deadline or the original native verdict.
        if abs(estimated - available) > .001:
            require(timing["feasible_estimate"] == (estimated <= available and timing["within_stage_limit"]),
                    "Admission feasibility disagrees with measured reserve")
        require(not timing["feasible_estimate"] or timing["within_stage_limit"], "Admission violated native stage limit")
        feasible = timing["feasible_estimate"] and timing["within_stage_limit"]
        metric = measured[0]["metric"]
        better = metric["value"] > best_inner["value"] if metric["higher_is_better"] else metric["value"] < best_inner["value"]
        require(feedback["accepted"] == (better and feasible)
                and feedback.get("reason") == (("improved" if feasible else "refit_budget") if better else "not_better"),
                "Native admission does not match measured improvement and refit feasibility")
        if feedback["accepted"]:
            best_inner = metric
            accepted_identity = {key: candidate[key] for key in ("code_hash", "recipe_hash")}
    require(host.get("frozen_candidate") == accepted_identity, "Frozen finalist differs from native accepted INNER trials")
    frequency = observation.get("frequency_attempt")
    frequency_admissions = [row for row in admissions if row["origin"] == "frequency"]
    if arm == "control":
        require(frequency is None and not frequency_admissions and not frequency_traces
                and "frequency_experiment" not in host, "Control was changed by frequency injection")
    else:
        require(isinstance(frequency, dict) and frequency == host.get("frequency_experiment"), "Frequency host/hook reports differ")
        require(frequency.get("attempted") in (0, 1) and frequency.get("status") in ("skipped", "failed", "completed"), "Frequency attempt status invalid")
        if not schema["categorical"]:
            require(frequency.get("reason") == "no_categorical_features" and frequency["attempted"] == 0
                    and not frequency_traces and not frequency_admissions, "Numeric-only case launched an extra fit")
        if frequency["status"] == "completed":
            require(frequency["attempted"] == 1 and len(frequency_admissions) == 1
                    and frequency["feedback"] == frequency_admissions[0]["feedback"], "Frequency admission missing/changed")
        else:
            require(not frequency_admissions, "Failed/skipped trial claimed successful admission")
    # Native startup must be genuinely invoked whenever its own guarded budget
    # permits it. A bounded startup failure is retained, never called success.
    search = host.get("baseline_search", {})
    require(search.get("scope") == "outer_fit_only" and search.get("status") in ("seed_only", "searched"), "Missing native startup baseline evidence")
    if "execution" in search:
        startup_execution = run_dir / "startup_execution"
        require(startup_execution.is_dir(), "Native startup execution logs are missing")
        for path in sorted(startup_execution.rglob("*")):
            if path.is_file():
                pin(path)
    if search.get("status") == "searched":
        _check_execution(search.get("execution"), "Native startup")
        startup_report_path = run_dir / "startup/automl_report.json"
        pin(startup_report_path)
        startup = read_json(startup_report_path)
        require(startup.get("search") == search.get("search"), "Native startup report differs from host")
        from agent.research_v3 import _recipe_from_report
        expected_baseline_recipe = _recipe_from_report(startup, schema["metric"]["task_type"])
    else:
        from agent.search import seed_candidate
        expected_baseline_recipe = seed_candidate(spec.classification)
    require(baseline["recipe"] == expected_baseline_recipe, "Frozen baseline recipe differs from actual native startup winner")
    receipts = host.get("candidate_receipts", {})
    published_name = host.get("published_recipe")
    require(published_name in ("baseline", "candidate") and host["publication"].get("recipe") == published_name,
            "Host did not name its newly published recipe")
    published_identity = host["frozen_baseline"] if published_name == "baseline" else host.get("frozen_candidate")
    require(isinstance(published_identity, dict)
            and {key: receipts.get(key) for key in ("code_hash", "recipe_hash")} == published_identity,
            "Publication identity differs from frozen finalist")
    matched = {}
    for phase, receipt_key in (("outer_validation", "outer"), ("full_refit", "refit")):
        rows = [row for row in verified if row["phase"] == phase
                and Path(row["receipt"]).resolve() == Path(receipts.get(receipt_key, "")).resolve()
                and {key: row[key] for key in ("code_hash", "recipe_hash")} == published_identity]
        require(len(rows) == 1, "Published recipe lacks its exact outer/full-refit receipt")
        matched[phase] = rows[0]
    publication = pin(task.parent / "submission.csv")
    checkpoint = pin(run_dir / "last_valid_submission.csv")
    refit_output = Path(matched["full_refit"]["output"])
    require(file_hash(publication) == file_hash(checkpoint) == file_hash(refit_output),
            "Published CSV/checkpoint differs from actual full-refit bytes")
    final_check = host.get("evaluations", {}).get("final_check", {})
    require(final_check.get("scope") == "final_check" and final_check.get("selection_uses_outer") is True
            and final_check.get("unbiased_estimate") is False, "Outer selection provenance missing")
    _same_metric(final_check.get(published_name), matched["outer_validation"]["metric"])
    audit_path = output / entry["audit"]
    require(file_hash(audit_path) == entry["audit_hash"] and task.parent.resolve() not in audit_path.resolve().parents,
            "Audit answers drifted or are inside solver task")
    audit = read_json(audit_path)
    require(audit.get("ids") == entry["snapshot"]["test_ids"] and audit.get("schema") == schema
            and audit.get("case") == entry["key"] and audit.get("scope") == "external_audit", "External audit contract drift")
    frame = _read_csv(publication)
    spec.validate_frame(frame, audit["ids"])
    metric, per_class = _metric_from_frame(frame, audit["actuals"], schema)
    return {"status": "completed", "metric": metric, "per_class": per_class,
            "baseline": {key: baseline[key] for key in ("code_hash", "recipe_hash")},
            "baseline_search_status": search["status"], "frequency": frequency,
            "frequency_phases": frequency_traces, "local_admissions": admissions,
            "published_recipe": published_name, "publication_hash": file_hash(publication),
            "host_status": host["status"], "new_result_published": True,
            "native_elapsed_seconds": worker["native_elapsed_seconds"],
            "launcher_elapsed_seconds": launcher.get("elapsed_seconds"),
            "resources": launcher.get("resources"), "proof_hashes": proof}


def _uncertain_censors(value, path="root"):
    found = []
    if isinstance(value, dict):
        looks_execution = (any(key in value for key in ("resources", "timed_out", "memory_limit_exceeded", "exit_code"))
                           or path == "launcher" or path.endswith(".execution"))
        resources = value.get("resources")
        cleanup = resources.get("cleanup_complete") if isinstance(resources, dict) else None
        abnormal = (value.get("timed_out") or value.get("memory_limit_exceeded") or value.get("error")
                    or value.get("exit_code") not in (None, 0))
        if looks_execution and abnormal and cleanup is not True:
            found.append(path)
        for key, child in value.items():
            found.extend(_uncertain_censors(child, path + "." + str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_uncertain_censors(child, path + "[" + str(index) + "]"))
    return found


def _artifact_index(directory):
    return {str(path.relative_to(directory)): file_hash(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "arm_summary.json"}


def run_arm(output_dir, entry, arm, plan):
    """One arm, also usable for a separately labelled native protocol smoke."""
    from agent.execution import run_python
    output = _output_root(output_dir)
    require(arm in ARMS, "Unknown arm")
    directory = output / "runs" / entry["key"] / arm
    require(not directory.exists(), "An arm cannot be retrained/repaired in place")
    directory.mkdir(parents=True)
    shutil.copytree(output / entry["template"], directory / "task")
    task = directory / "task/task.md"
    require(_input_hashes(task) == entry["input_hashes"], "Copied solver input differs from sealed template")
    config = {"version": 1, "arm": arm, "task": str(task), "arm_cap_seconds": entry["arm_cap_seconds"],
              "snapshot": entry["snapshot"], "sources": plan["sources"], "environment": plan["environment"],
              "method_sha": plan["method_sha"]}
    config_path = directory / "config.json"
    write_json(config_path, config)
    summary = {"version": 1, "case": entry["key"], "arm": arm, "status": "failed", "started_at": utc_now(),
               "directory": str(directory.relative_to(output)), "config_hash": file_hash(config_path),
               "hard_cap_seconds": entry["arm_cap_seconds"]}
    try:
        launcher = run_python(child_bootstrap(config_path), task.parent, directory / "launcher", 0,
                              timeout=entry["arm_cap_seconds"])
        write_json(directory / "launcher.json", launcher)
        summary["launcher_hash"] = file_hash(directory / "launcher.json")
        summary["launcher"] = {key: launcher.get(key) for key in ("exit_code", "timed_out", "memory_limit_exceeded", "elapsed_seconds", "error", "resources")}
        uncertain_censors = _uncertain_censors(launcher, "launcher")
        for name, key in (("worker_summary.json", "worker_summary_hash"), ("observation.json", "observation_hash")):
            if (directory / name).is_file():
                summary[key] = file_hash(directory / name)
        if (directory / "worker_summary.json").is_file():
            worker = read_json(directory / "worker_summary.json")
            uncertain_censors.extend(_uncertain_censors(worker, "worker"))
            if worker.get("run_dir") and (Path(worker["run_dir"]) / "run.json").is_file():
                host_path = Path(worker["run_dir"]) / "run.json"
                summary["host_run_hash"] = file_hash(host_path)
                uncertain_censors.extend(_uncertain_censors(read_json(host_path), "host"))
        if (directory / "observation.json").is_file():
            uncertain_censors.extend(_uncertain_censors(read_json(directory / "observation.json"), "observation"))
        # The host may raise before worker_run receives/records run_dir. Its
        # already-written journal still contains real startup failure receipts.
        for host_path in sorted((directory / "task/.agent_runs").glob("*/run.json")):
            uncertain_censors.extend(_uncertain_censors(read_json(host_path), "host_journal." + host_path.parent.name))
        summary["uncertain_censor_paths"] = sorted(set(uncertain_censors))
        summary["abort_series"] = bool(uncertain_censors)
        if launcher.get("timed_out"):
            summary.update(status="censored", reason="external_hard_timeout")
        elif launcher.get("memory_limit_exceeded"):
            summary.update(status="censored", reason="external_memory_limit")
        elif launcher.get("exit_code") != 0 or launcher.get("error"):
            summary.update(reason="native_process_failed")
        elif uncertain_censors:
            summary.update(status="censored", reason="nested_censored_cleanup_unverified")
        else:
            summary["audit"] = audit_arm(output, entry, arm, summary, plan)
            summary.update(status="completed", reason="independently_verified_new_full_refit")
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        summary.update(status="failed", reason="evidence_verification_failed", error=type(error).__name__ + ": " + str(error))
    # Include partial/failed artifacts, logs and startup evidence as well as the
    # successful receipt chain. Re-audit must detect drift in failures too.
    summary["artifact_hashes"] = _artifact_index(directory)
    summary["ended_at"] = utc_now()
    write_json(directory / "arm_summary.json", summary)
    return summary


def _pair_summary(entry, arms):
    comparable = set(arms) == set(ARMS) and all(arms[arm].get("status") == "completed" for arm in ARMS)
    pair = {"case": entry["key"], "name": entry["name"], "seed": entry["seed"], "stage": entry["stage"],
            "order": entry["order"], "status": "completed" if comparable else "not_comparable", "comparable": comparable,
            "arms": arms}
    if comparable:
        control, frequency = (arms[arm]["audit"] for arm in ARMS)
        require(control["metric"]["name"] == frequency["metric"]["name"], "Pair metric names differ")
        pair["baseline_hashes_match"] = control["baseline"] == frequency["baseline"]
        pair["interpretation"] = ("full_agent_same_baseline_time_allocation_may_differ" if pair["baseline_hashes_match"]
                                  else "full_agent_comparison_frequency_effect_not_isolated")
        pair["metric"] = control["metric"]["name"]
        pair["delta_frequency_minus_control"] = frequency["metric"]["value"] - control["metric"]["value"]
        pair["signed_improvement"] = pair["delta_frequency_minus_control"] * (1 if control["metric"]["higher_is_better"] else -1)
        pair["per_class_delta"] = {label: frequency["per_class"][label]["f1"] - row["f1"]
                                   for label, row in control["per_class"].items()}
        pair["per_class_harm"] = {label: delta for label, delta in pair["per_class_delta"].items() if delta < 0}
    return pair


def _skipped_arm(entry, arm, reason):
    return {"version": 1, "case": entry["key"], "arm": arm, "status": "not_started", "reason": reason}


def _report_paths(output, suite):
    require(suite in ("synthetic", "public"), "suite must be synthetic or public")
    return output / ("report.json" if suite == "synthetic" else "public_report.json"), output / (suite + "_completion_seal.json")


def _arm_summary_paths(output, pairs):
    """Use the saved arm structure, preserving the host's sealed path format."""
    return [str((output / arm["directory"] / "arm_summary.json").relative_to(output))
            for pair in pairs for arm in pair["arms"].values() if arm.get("directory")]


def _confirmation_seal(output, plan, pairs):
    require(len(pairs) == 4 and all(pair["stage"] == "development" for pair in pairs), "Confirmation seal requires all four development records")
    load_plan(output)
    seal = {"version": 1, "created_at": utc_now(), "plan_hash": file_hash(output / "plan.json"),
            "method_sha": plan["method_sha"], "sources": source_hashes(), "environment": environment_versions(),
            "development_pairs_hash": json_hash(pairs),
            "development_summary_hashes": {path: file_hash(output / path) for path in _arm_summary_paths(output, pairs)},
            "method_changed_after_development": False}
    path = output / "confirmation_seal.json"
    require(not path.exists(), "Confirmation seal already exists")
    write_json(path, seal)
    return file_hash(path)


def run_prepared(output_dir, suite="synthetic"):
    """Sequential equal-cap pairs; never shorten/retry an arm after results."""
    output = _output_root(output_dir)
    plan = load_plan(output)
    report_path, completed_path = _report_paths(output, suite)
    require(not report_path.exists() and not completed_path.exists(), "A prepared series cannot be rerun or repaired")
    entries = plan["entries"] if suite == "synthetic" else plan["public"]["entries"]
    cap = SYNTHETIC_SECONDS if suite == "synthetic" else PUBLIC_SECONDS
    report = {"version": 1, "suite": suite, "started_at": utc_now(), "plan_hash": file_hash(output / "plan.json"),
              "method_sha": plan["method_sha"], "base_sha": plan["base_sha"], "executable_sha": plan["executable_sha"],
              "status": "running", "coordinator_cap_seconds": cap, "pairs": [],
              "external_model_api_calls": 0, "platform_submissions": 0,
              "quality_evidence": True, "scope": "synthetic_development_and_confirmation" if suite == "synthetic" else "known_public_train_sanity"}
    started = time.monotonic()
    abort_reason = None
    write_json(report_path, report)
    if suite == "public" and not entries:
        report.update(status="skipped_missing_data", public=plan["public"], active_elapsed_seconds=time.monotonic() - started)
    else:
        for index, entry in enumerate(entries):
            load_plan(output)
            if entry["stage"] == "confirmation" and index == 4 and abort_reason is None:
                report["confirmation_seal_hash"] = _confirmation_seal(output, plan, report["pairs"])
            remaining = cap - (time.monotonic() - started)
            if abort_reason is None and remaining < 2 * entry["arm_cap_seconds"]:
                abort_reason = "coordinator_cannot_reserve_two_full_arms"
            arms = {}
            for arm in entry["order"]:
                if abort_reason is not None:
                    arms[arm] = _skipped_arm(entry, arm, abort_reason)
                else:
                    load_plan(output)
                    arms[arm] = run_arm(output, entry, arm, plan)
                    if arms[arm].get("abort_series"):
                        abort_reason = "censored_process_cleanup_unverified"
            report["pairs"].append(_pair_summary(entry, arms))
            report["active_elapsed_seconds"] = time.monotonic() - started
            write_json(report_path, report)
        report["status"] = "completed" if all(pair["comparable"] for pair in report["pairs"]) else "incomplete"
        report["abort_reason"] = abort_reason
    report["ended_at"] = utc_now()
    report["active_elapsed_seconds"] = time.monotonic() - started
    report["comparable_pairs"] = sum(pair["comparable"] for pair in report["pairs"])
    report["completed_arms"] = sum(arm["status"] == "completed" for pair in report["pairs"] for arm in pair["arms"].values())
    report["arm_summary_hashes"] = {path: file_hash(output / path) for path in _arm_summary_paths(output, report["pairs"])}
    write_json(report_path, report)
    write_json(completed_path, {"version": 1, "suite": suite, "report_hash": file_hash(report_path),
                                "plan_hash": report["plan_hash"], "terminal_status": report["status"], "ended_at": report["ended_at"]})
    return report


@evidence_validation
def re_audit(output_dir, suite="synthetic"):
    """Read-only verification. No worker launch, fitting, or artifact repair."""
    output = _output_root(output_dir)
    plan = load_plan(output)
    report_path, seal_path = _report_paths(output, suite)
    report, seal = read_json(report_path), read_json(seal_path)
    require(seal.get("version") == 1 and seal.get("suite") == suite
            and seal.get("report_hash") == file_hash(report_path)
            and seal.get("plan_hash") == file_hash(output / "plan.json") == report.get("plan_hash"), "Series completion seal drift")
    require(report.get("suite") == suite and report.get("version") == 1
            and report.get("method_sha") == plan["method_sha"] and report.get("status") == seal.get("terminal_status"), "Series metadata drift")
    require(report.get("external_model_api_calls") == 0 and report.get("platform_submissions") == 0, "External calls in series")
    entries = plan["entries"] if suite == "synthetic" else plan["public"]["entries"]
    if suite == "public" and not entries:
        require(report.get("status") == "skipped_missing_data" and not report.get("pairs")
                and report.get("public") == plan["public"], "Missing public data mislabeled as completed")
        return {"version": 1, "status": "skipped_missing_data", "verified": True, "suite": suite,
                "checked_paths": plan["public"]["checked_paths"], "fits": 0}
    pairs = report.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == len(entries), "Incomplete saved pair coverage")
    require(report.get("status") in ("completed", "incomplete"), "Partial/running report is not a terminal series")
    for path, digest in report.get("arm_summary_hashes", {}).items():
        require(file_hash(output / path) == digest, "Saved arm summary drift")
    verified_pairs, observed_summary_paths = [], set()
    for entry, pair in zip(entries, pairs):
        require(pair.get("case") == entry["key"] and pair.get("name") == entry["name"]
                and pair.get("seed") == entry["seed"] and pair.get("stage") == entry["stage"]
                and pair.get("order") == entry["order"], "Saved pair identity/order drift")
        require(isinstance(pair.get("arms"), dict) and set(pair["arms"]) == set(ARMS), "Incomplete pair arms")
        fresh_arms = {}
        for arm in ARMS:
            summary = pair["arms"][arm]
            require(summary.get("arm") == arm and summary.get("case") == entry["key"], "Saved arm identity drift")
            if summary.get("directory"):
                summary_path = output / summary["directory"] / "arm_summary.json"
                relative = str(summary_path.relative_to(output))
                observed_summary_paths.add(relative)
                require(report["arm_summary_hashes"].get(relative) == file_hash(summary_path), "Missing arm summary seal")
                require(read_json(summary_path) == summary, "Report/arm summary differ")
                require(isinstance(summary.get("artifact_hashes"), dict)
                        and summary["artifact_hashes"] == _artifact_index(summary_path.parent),
                        "Saved successful/failed/partial artifact index drift")
            if summary.get("status") == "completed":
                fresh = audit_arm(output, entry, arm, summary, plan)
                require(fresh == summary.get("audit"), "Saved audit verdict differs from independent re-audit")
            else:
                require(summary.get("status") in ("failed", "censored", "not_started") and not summary.get("audit"),
                        "Unverified/partial arm was promoted to completed")
            fresh_arms[arm] = summary
        fresh_pair = _pair_summary(entry, fresh_arms)
        require(fresh_pair == pair, "Saved paired comparison differs from re-audit")
        verified_pairs.append({"case": entry["key"], "comparable": fresh_pair["comparable"]})
    require(observed_summary_paths == set(report.get("arm_summary_hashes", {})), "Extra or missing arm summary indices")
    if suite == "synthetic":
        any_confirmation_started = any(arm.get("directory") for pair in pairs[4:] for arm in pair["arms"].values())
        if any_confirmation_started:
            confirmation_path = output / "confirmation_seal.json"
            require(file_hash(confirmation_path) == report.get("confirmation_seal_hash"), "Confirmation seal missing/drifted")
            confirmation = read_json(confirmation_path)
            require(confirmation.get("plan_hash") == report["plan_hash"] and confirmation.get("method_sha") == plan["method_sha"]
                    and confirmation.get("sources") == plan["sources"] and confirmation.get("environment") == plan["environment"]
                    and confirmation.get("development_pairs_hash") == json_hash(pairs[:4])
                    and confirmation.get("method_changed_after_development") is False, "Method/environment changed before confirmation")
            expected_dev = {path: report["arm_summary_hashes"][path] for path in _arm_summary_paths(output, pairs[:4])}
            require(confirmation.get("development_summary_hashes") == expected_dev, "Development evidence changed after confirmation seal")
    comparable = sum(pair["comparable"] for pair in verified_pairs)
    completed = sum(arm["status"] == "completed" for pair in pairs for arm in pair["arms"].values())
    require(report.get("comparable_pairs") == comparable and report.get("completed_arms") == completed, "Saved completion counts differ")
    require((report["status"] == "completed") == (comparable == len(entries)), "Incomplete series claimed completed")
    require(type(report.get("active_elapsed_seconds")) in (int, float)
            and report["active_elapsed_seconds"] <= report["coordinator_cap_seconds"], "Coordinator cap exceeded")
    return {"version": 1, "status": report["status"], "verified": True, "suite": suite, "fits": 0,
            "comparable_pairs": comparable, "completed_arms": completed, "pairs": verified_pairs,
            "report_hash": file_hash(report_path), "method_sha": plan["method_sha"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run-prepared", "re-audit"):
        command = commands.add_parser(name)
        command.add_argument("output_dir", type=Path)
        if name == "prepare":
            command.add_argument("--public-task", type=Path)
        else:
            command.add_argument("--suite", choices=("synthetic", "public"), default="synthetic")
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        result = prepare(arguments.output_dir, arguments.public_task)
        print(json.dumps({"status": "prepared", "fits": 0, "pairs": len(result["entries"]),
                          "method_sha": result["method_sha"], "public_status": result["public"]["status"]}))
    elif arguments.command == "run-prepared":
        result = run_prepared(arguments.output_dir, arguments.suite)
        print(json.dumps({key: result.get(key) for key in ("status", "comparable_pairs", "completed_arms", "active_elapsed_seconds")}))
    else:
        result = re_audit(arguments.output_dir, arguments.suite)
        # A new verification document is allowed; existing evidence is never changed.
        output = _output_root(arguments.output_dir)
        destination = output / (arguments.suite + "_reaudit_" + uuid4().hex + ".json")
        write_json(destination, result)
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
