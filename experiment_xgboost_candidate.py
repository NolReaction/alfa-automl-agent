"""Native subprocess smoke for the isolated XGBoost candidate; never imported by solution.py.

prepare writes the catalog, recipes, inputs and source/environment hashes and
performs zero fits. run is the only training entry point: cases run one after
another, each in a fresh directory, through the host's v3 run_candidate
protocol. audit re-reads saved bytes and rescores; it never trains or repairs.
Raw files stay below the ignored data/experiments/xgboost_candidate/. Holdout
answers live in host_only/ and never enter a child invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from agent import xgboost_candidate as candidate
from agent.metrics import get_metric, score_predictions
from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
from agent.resources import DEFAULT_MEMORY_LIMIT_BYTES
from agent.task_spec import TaskSpec
import xgboost_candidate_cases as cases


ROOT = Path(__file__).resolve().parent
AUTHOR_BASE_SHA = "184408390c83ef951e5a570cf9d2d6c3e9e72f98"
RAW_ROOT = ROOT / "data" / "experiments" / "xgboost_candidate"
INVOCATION_SECONDS = 600
PHASES = ("outer_validation", "full_refit")
SOURCES = ("agent/xgboost_candidate.py", "xgboost_candidate_cases.py", "experiment_xgboost_candidate.py",
           "agent/research_candidate.py", "agent/execution.py", "agent/resources.py",
           "agent/features.py", "agent/models.py", "agent/splits.py", "agent/metrics.py",
           "agent/task_spec.py", "offline_baseline.py", "requirements.platform.txt")
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "psutil", "xgboost")
TAIL_CHARACTERS = 4000
PLAN_VERSION = 2


def require(condition, message):
    """Active under -O; evidence failures are controlled ValueErrors."""
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Evidence must be a regular file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    try:
        return json.loads(Path(path).read_bytes().decode("utf-8"), parse_constant=invalid)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Unreadable evidence: " + str(path)) from error


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value) + b"\n")


def write_json_replace(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(json_bytes(value) + b"\n")
    pending.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments):
    try:
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    names = set(SOURCES) | {"requirements.txt", "solution.py", "validation.py"}
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "agent").rglob("*.py"))
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "executable": str(Path(sys.executable).absolute()),
            "platform": sys.platform, "packages": versions, "os_cpu_count": os.cpu_count(),
            "candidate_max_threads": candidate.THREADS,
            "configured_memory_cap_bytes": DEFAULT_MEMORY_LIMIT_BYTES,
            "memory_cap_is_measured_peak": False, "external_api_calls": 0}


def _output_root(path):
    result = Path(path).resolve()
    require(RAW_ROOT.resolve() in result.parents,
            "Raw evidence must use a child of data/experiments/xgboost_candidate/")
    return result


def _relative(output, path):
    return Path(path).resolve().relative_to(output).as_posix()


def _path(output, relative):
    require(isinstance(relative, str) and relative, "Missing evidence path")
    path = (output / relative).resolve()
    require(path.is_relative_to(output), "Evidence path escapes output")
    return path


def artifact_hashes(output):
    excluded = {"results.json", "results.seal.json", "results.partial.json"}
    result = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "Linked evidence file")
        if path.is_file():
            name = path.relative_to(output).as_posix()
            if name not in excluded:
                result[name] = file_hash(path)
    return result


def make_spec(output, name, definition):
    metric = get_metric(definition["metric"])
    classes = tuple(definition["classes"])
    if metric.prediction_kind == "probabilities":
        outputs = (cases.ID_COLUMN, *classes)
        probability_columns = tuple((label, label) for label in classes)
    else:
        outputs, probability_columns = (cases.ID_COLUMN, cases.TARGET_COLUMN), ()
    return TaskSpec(output / "inputs" / name / "task.md", output / "inputs" / name,
                    cases.ID_COLUMN, cases.TARGET_COLUMN, cases.FEATURES, cases.NUMERIC,
                    cases.CATEGORICAL, classes, outputs, probability_columns,
                    definition["positive_class"], metric)


def prepare(output_dir, names=None):
    """Seal catalog, recipes, inputs, sources and environment; perform zero fits."""
    output = _output_root(output_dir)
    require(not output.exists(), "prepare needs a fresh output directory")
    names = list(cases.CASES) if names is None else list(names)
    require(names and len(set(names)) == len(names) and set(names) <= set(cases.CASES),
            "Unknown or duplicate case names")
    output.mkdir(parents=True)
    source = output / "candidate_source.py"
    source.write_text(candidate.candidate_source(), encoding="utf-8")
    entries = []
    for name in names:
        generated = cases.generate(name)
        definition = generated["definition"]
        task_type = get_metric(definition["metric"]).task_type
        recipe = candidate.make_xgboost_recipe(
            task_type, definition["metric"], validation=definition["validation"],
            parameters=candidate.FIXED_CONFIGURATIONS[definition["configuration"]])
        files = {}
        for label in ("fit", "holdout", "test"):
            path = output / "inputs" / name / (label + ".csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            generated[label].to_csv(path, index=False)
            files[_relative(output, path)] = file_hash(path)
        answers = output / "host_only" / name / "holdout_answers.json"
        write_json_new(answers, {"answers": generated["holdout_answers"]})
        files[_relative(output, answers)] = file_hash(answers)
        entries.append({"name": name, "definition": definition, "recipe": recipe,
                        "recipe_sha256": json_hash(recipe), "files": files,
                        "rows": {label: len(generated[label]) for label in ("fit", "holdout", "test")}})
    plan = {"version": PLAN_VERSION, "created_at": utc_now(), "author_base_sha": AUTHOR_BASE_SHA,
            "head": _git("rev-parse", "HEAD"), "sources": source_hashes(),
            "environment": environment(), "candidate_source_sha256": file_hash(source),
            "cases_version": cases.VERSION, "seed": cases.SEED, "phases": list(PHASES),
            "invocation_seconds": INVOCATION_SECONDS, "cases": entries,
            "quality_audit": False, "note": "Protocol smoke on author-written synthetic cases"}
    write_json_new(output / "plan.json", plan)
    write_json_new(output / "seal.json", {"version": PLAN_VERSION,
                                         "plan_sha256": file_hash(output / "plan.json")})
    return plan


def load_plan(output, *, execution=True):
    plan = read_json(output / "plan.json")
    require(plan.get("version") == PLAN_VERSION
            and plan.get("author_base_sha") == AUTHOR_BASE_SHA, "Unsupported plan")
    seal = read_json(output / "seal.json")
    require(seal.get("version") == PLAN_VERSION
            and seal.get("plan_sha256") == file_hash(output / "plan.json"), "Plan seal changed")
    require(source_hashes() == plan["sources"], "Sources changed after prepare")
    require(file_hash(output / "candidate_source.py") == plan["candidate_source_sha256"],
            "Candidate source changed after prepare")
    if execution:
        require(environment() == plan["environment"], "Environment changed after prepare")
    for entry in plan["cases"]:
        for path, digest in entry["files"].items():
            require(file_hash(_path(output, path)) == digest, "Input changed after prepare: " + path)
        require(json_hash(entry["recipe"]) == entry["recipe_sha256"], "Recipe changed after prepare")
    return plan


def _read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _execution(execution, *, failed):
    keys = ("exit_code", "timed_out", "memory_limit_exceeded", "elapsed_seconds",
            "host_elapsed_seconds", "host_deadline_exceeded", "error")
    result = {key: execution[key] for key in keys if key in execution}
    resources = execution.get("resources")
    if isinstance(resources, dict):
        result["resources"] = json.loads(json_bytes(resources))
    if failed:
        for key in ("stdout", "stderr"):
            result[key + "_tail"] = str(execution.get(key, ""))[-TAIL_CHARACTERS:]
    return result


def _evidence(output, directory, invocation=None):
    if invocation is not None:
        path = Path(invocation) / candidate.EVIDENCE_NAME
        found = [path] if path.is_file() else []
    else:
        # A failed invocation has no returned result pointer. Retain diagnostic
        # evidence only when its fresh phase directory has exactly one match.
        found = sorted(Path(directory).glob("*/" + candidate.EVIDENCE_NAME))
    if len(found) != 1:
        return {"count": len(found)}
    record = {"count": 1, "path": _relative(output, found[0]), "sha256": file_hash(found[0])}
    try:
        content = read_json(found[0])
        require(isinstance(content, dict), "Candidate evidence must be a JSON object")
    except ValueError as error:
        # A killed child may leave a partially written diagnostic file. Preserve
        # its exact bytes/hash without losing the recorded failure or denominator.
        record["parse_error"] = str(error)[-TAIL_CHARACTERS:]
    else:
        record["content"] = content
    return record


def invoke(output, frozen, spec, fit, predict, directory, phase, actuals=None):
    """One native invocation; failures and timeouts are saved, never retried."""
    started = time.monotonic()
    record = {"phase": phase, "fit_rows": len(fit), "predict_rows": len(predict)}
    result = None
    try:
        result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=predict,
                               output_dir=directory, phase=phase, timeout=INVOCATION_SECONDS,
                               actuals=actuals)
    except CandidateExecutionError as error:
        record.update(status="failed", error=str(error),
                      execution=_execution(error.execution, failed=True))
    except Exception as error:  # noqa: BLE001 - host-side errors are saved evidence too
        record.update(status="error", error=type(error).__name__ + ": " + str(error))
    else:
        record.update(status="completed", execution=_execution(result.execution, failed=False),
                      metric=result.metric, output=_relative(output, result.output_path),
                      output_sha256=result.output_hash, receipt=_relative(output, result.receipt_path),
                      receipt_sha256=result.receipt_hash, code_hash=result.code_hash,
                      recipe_hash=result.recipe_hash)
    record["host_seconds"] = time.monotonic() - started
    record["evidence"] = _evidence(output, directory,
                                    result.receipt_path.parent if result is not None else None)
    return record, result


def rescore(output_bytes, spec, answers, expected_ids=None):
    """Score saved CSV bytes with host-only answers; no child value is trusted."""
    frame = pd.read_csv(io.BytesIO(output_bytes), dtype=str, keep_default_na=False)
    spec.validate_frame(frame, frame[spec.id_column].tolist() if expected_ids is None else expected_ids)
    if spec.prediction_kind == "probabilities":
        columns = [column for column, _ in spec.probability_columns]
        values = frame.loc[:, columns].to_numpy(dtype=float)
        prediction_classes = [label for _, label in spec.probability_columns]
    else:
        values, prediction_classes = frame[spec.target_column].to_numpy(), None
    return score_predictions(spec.metric, answers, values, classes=spec.classes,
                             prediction_classes=prediction_classes, positive_class=spec.positive_class)


def constant_reference(spec, fit_targets, answers):
    """A sanity floor from fit answers only; not a competing model."""
    rows = len(answers)
    if not spec.classification:
        values = pd.to_numeric(pd.Series(fit_targets), errors="raise")
        constant = values.median() if spec.metric.name == "MAE" else values.mean()
        return score_predictions(spec.metric, answers, [float(constant)] * rows)
    counts = pd.Series(fit_targets).value_counts()
    if spec.prediction_kind == "labels":
        return score_predictions(spec.metric, answers, [str(counts.index[0])] * rows, classes=spec.classes)
    priors = [float(counts.get(label, 0)) / len(fit_targets) for label in spec.classes]
    return score_predictions(spec.metric, answers, [priors] * rows, classes=spec.classes,
                             prediction_classes=spec.classes, positive_class=spec.positive_class)


def _content(record):
    evidence = record.get("evidence", {})
    return evidence.get("content") if evidence.get("count") == 1 else None


def _boundary(evidence):
    validation = evidence["validation"]
    if validation["strategy"] == "time":
        return float(validation["train_period_max"]) < float(validation["validation_period"])
    if validation["strategy"] == "grouped":
        return validation["group_overlap"] == 0
    return "not_applicable"


def _child_inputs_exclude_holdout(directory, holdout_ids, target, *, failed=False):
    try:
        if not Path(directory).is_dir():
            return False
        invocations = [path for path in Path(directory).iterdir() if path.is_dir()]
        if len(invocations) != 1:
            return False
        fit = _read_csv(invocations[0] / "fit.csv")
        predict = _read_csv(invocations[0] / "predict.csv")
        return not set(fit[cases.ID_COLUMN]) & set(holdout_ids) and target not in predict.columns
    except (OSError, UnicodeError, ValueError, KeyError):
        if not failed:
            raise
        # Failed host staging need not have produced complete CSV inputs.
        return False


def _not_started(phase, reason):
    return {"phase": phase, "status": "not_started", "reason": reason, "evidence": {"count": 0}}


def _stop_for_cleanup(record):
    if record["status"] != "completed" and (
            record.get("execution", {}).get("resources", {}).get("cleanup_complete") is not True):
        record.update(failure_status=record["status"], status="censored",
                      reason="failed_invocation_cleanup_unverified", stop_series=True)
        return True
    return False


def _not_started_case(entry, reason):
    definition = entry["definition"]
    return {"name": entry["name"], "metric": definition["metric"],
            "configuration": definition["configuration"], "validation": definition["validation"],
            "phases": [_not_started(phase, reason) for phase in PHASES],
            "checks": {"started": False}, "all_checks_passed": False, "reason": reason}


def run_case(output, entry, source):
    name, definition = entry["name"], entry["definition"]
    spec = make_spec(output, name, definition)
    inputs = output / "inputs" / name
    fit, holdout, test = (_read_csv(inputs / (label + ".csv")) for label in ("fit", "holdout", "test"))
    answers = read_json(output / "host_only" / name / "holdout_answers.json")["answers"]
    case_dir = output / "runs" / name
    case_dir.mkdir(parents=True, exist_ok=False)
    frozen = freeze_candidate(source, entry["recipe"], case_dir / "frozen")
    outer, outer_result = invoke(output, frozen, spec, fit, holdout, case_dir / "outer_validation",
                                 "outer_validation", answers)
    if _stop_for_cleanup(outer):
        refit = _not_started("full_refit", "prior_cleanup_unverified")
    else:
        full = pd.concat([fit, holdout.assign(**{cases.TARGET_COLUMN: answers})], ignore_index=True)
        refit, _ = invoke(output, frozen, spec, full, test, case_dir / "full_refit", "full_refit")
        _stop_for_cleanup(refit)
    record = {"name": name, "metric": definition["metric"], "configuration": definition["configuration"],
              "validation": definition["validation"], "phases": [outer, refit],
              "constant_reference": constant_reference(spec, fit[cases.TARGET_COLUMN].tolist(), answers)}
    checks = {"outer_child_inputs_exclude_holdout": _child_inputs_exclude_holdout(
        case_dir / "outer_validation", holdout[cases.ID_COLUMN].tolist(), cases.TARGET_COLUMN,
        failed=outer["status"] != "completed")}
    if outer_result is not None:
        record["outer_rescore"] = rescore(outer_result.read_output(), spec, answers)
        checks["outer_rescore_matches"] = record["outer_rescore"] == outer["metric"]["value"]
    for phase, expected_fit, expected_predict in ((outer, len(fit), len(holdout)),
                                                  (refit, len(fit) + len(holdout), len(test))):
        evidence = _content(phase)
        label = phase["phase"]
        checks[label + "_completed"] = phase["status"] == "completed"
        checks[label + "_evidence_completed"] = isinstance(evidence, dict) and evidence.get("status") == "completed"
        if evidence is None or evidence.get("status") != "completed":
            continue
        checks[label + "_rows"] = (evidence["full_fit_rows"], evidence["predict_rows"]) == (
            expected_fit, expected_predict)
        checks[label + "_refit_trees_equal_selected"] = (
            evidence["trees"]["refit_trained"] == evidence["trees"]["selected"])
        checks[label + "_internal_boundary"] = _boundary(evidence)
        checks[label + "_phase_recorded"] = evidence["phase"] == label
    record["checks"] = checks
    record["all_checks_passed"] = all(value is True or value == "not_applicable" for value in checks.values())
    record["stop_series"] = any(phase.get("stop_series") for phase in record["phases"])
    return record


def run(output_dir):
    """Sequential cases under one sealed plan; never shorten or retry after results."""
    output = _output_root(output_dir)
    plan = load_plan(output)
    final = output / "results.json"
    require(not final.exists() and not (output / "runs").exists(), "run needs an unused prepared plan")
    results = {"version": PLAN_VERSION, "plan_sha256": file_hash(output / "plan.json"), "started_at": utc_now(),
               "external_api_calls": 0, "cases": []}
    stop_reason = None
    for entry in plan["cases"]:
        if stop_reason:
            record = _not_started_case(entry, stop_reason)
        else:
            load_plan(output)
            record = run_case(output, entry, output / "candidate_source.py")
            if record.get("stop_series"):
                stop_reason = "prior_cleanup_unverified"
        results["cases"].append(record)
        write_json_replace(output / "results.partial.json", results)
    phases = [phase for case in results["cases"] for phase in case["phases"]]
    results.update(finished_at=utc_now(), stop_reason=stop_reason, summary={
        "cases": len(results["cases"]),
        "completed_phases": sum(phase["status"] == "completed" for phase in phases),
        "unsuccessful_phases": sum(phase["status"] != "completed" for phase in phases),
        "all_checks_passed": all(case["all_checks_passed"] for case in results["cases"])})
    (output / "results.partial.json").unlink()
    results["artifact_hashes"] = artifact_hashes(output)
    write_json_new(final, results)
    write_json_new(output / "results.seal.json", {"version": PLAN_VERSION,
                                                "results_sha256": file_hash(final)})
    return results


def _same_saved_frame(path, expected, columns):
    actual = _read_csv(path)
    require(list(actual.columns) == list(columns)
            and actual.values.tolist() == expected.loc[:, list(columns)].astype(str).values.tolist(),
            "Invocation rows/values differ from sealed host inputs: " + str(path))


def _audit_completed_phase(output, plan, entry, phase, spec, fit, predict, answers):
    output_path, receipt_path = (_path(output, phase[key]) for key in ("output", "receipt"))
    invocation = receipt_path.parent
    require(output_path == invocation / "predictions.csv" and receipt_path.name == "receipt.json",
            "Outputs do not share the exact current invocation")
    require(invocation.parent == output / "runs" / entry["name"] / phase["phase"],
            "Invocation is outside the planned case/phase")
    context = read_json(invocation / "context.json")
    recipe = read_json(invocation / "recipe.json")
    code_hash, recipe_hash = file_hash(invocation / "candidate.py"), file_hash(invocation / "recipe.json")
    require(code_hash == plan["candidate_source_sha256"] == phase["code_hash"]
            and recipe_hash == json_hash(entry["recipe"]) == phase["recipe_hash"]
            and recipe == entry["recipe"], "Frozen invocation source/recipe differs")
    require(context.get("version") == 3 and context.get("protocol_version") == 3
            and context.get("phase") == phase["phase"] and context.get("recipe") == recipe
            and context.get("code_hash") == code_hash and context.get("recipe_hash") == recipe_hash
            and context.get("metric") == spec.metric.to_dict()
            and context.get("features") == list(spec.features)
            and context.get("numeric") == list(spec.numeric)
            and context.get("categorical") == list(spec.categorical)
            and context.get("classes") == list(spec.classes)
            and context.get("positive_class") == spec.positive_class,
            "Saved invocation context differs from the sealed method")
    _same_saved_frame(invocation / "fit.csv", fit, [spec.id_column, *spec.features, spec.target_column])
    _same_saved_frame(invocation / "predict.csv", predict, [spec.id_column, *spec.features])
    output_hash = file_hash(output_path)
    receipt = read_json(receipt_path)
    require(receipt == {"version": 3, "nonce": context["nonce"], "phase": phase["phase"],
                        "code_hash": code_hash, "recipe_hash": recipe_hash, "output_hash": output_hash},
            "Native receipt differs from the exact current invocation")
    frame = _read_csv(output_path)
    expected_ids = predict[spec.id_column].tolist()
    spec.validate_frame(frame, expected_ids)
    evidence = phase.get("evidence", {})
    evidence_path = invocation / candidate.EVIDENCE_NAME
    require(evidence.get("count") == 1 and _path(output, evidence["path"]) == evidence_path
            and file_hash(evidence_path) == evidence.get("sha256")
            and read_json(evidence_path) == evidence.get("content"), "Candidate evidence changed or is absent")
    content = evidence["content"]
    require(content.get("status") == "completed" and content.get("phase") == phase["phase"]
            and content.get("recipe") == entry["recipe"]
            and content.get("classes") == list(spec.classes)
            and content.get("cpu", {}).get("device") == "cpu"
            and content.get("full_fit_rows") == len(fit) and content.get("predict_rows") == len(predict)
            and content.get("fresh_full_refit") is True
            and content.get("trees", {}).get("refit_trained") == content.get("trees", {}).get("selected")
            and type(content.get("trees", {}).get("selected")) is int
            and content["trees"]["selected"] > 0
            and _boundary(content) in (True, "not_applicable"), "Candidate full-refit evidence is invalid")
    checks = {"output_sha256_matches": output_hash == phase["output_sha256"],
              "receipt_sha256_matches": file_hash(receipt_path) == phase["receipt_sha256"],
              "evidence_matches": True, "sealed_inputs_match": True, "receipt_context_matches": True}
    if phase["phase"] == "outer_validation":
        value = rescore(output_path.read_bytes(), spec, answers, expected_ids)
        checks["rescore_matches"] = (phase["metric"] == {"name": spec.metric.name, "value": value,
                                                       "higher_is_better": spec.metric.higher_is_better})
    else:
        checks["no_test_score"] = phase["metric"] is None
    return checks


def audit(output_dir):
    """Re-hash saved outputs/receipts and rescore saved CSV bytes; no training."""
    output = _output_root(output_dir)
    plan = load_plan(output, execution=False)
    results = read_json(output / "results.json")
    seal = read_json(output / "results.seal.json")
    results_digest = file_hash(output / "results.json")
    seal_digest = file_hash(output / "results.seal.json")
    require(results.get("version") == PLAN_VERSION, "Unsupported results")
    require(seal.get("version") == PLAN_VERSION and seal.get("results_sha256") == results_digest,
            "Results seal changed")
    require(results["plan_sha256"] == file_hash(output / "plan.json"), "Plan changed after run")
    before = artifact_hashes(output)
    require(before == results.get("artifact_hashes"), "Saved evidence changed after run")
    entries = {entry["name"]: entry for entry in plan["cases"]}
    require([case["name"] for case in results["cases"]] == list(entries), "Planned case denominator changed")
    report = {"cases": [], "trained": False, "planned_cases": len(entries)}
    for case in results["cases"]:
        entry = entries[case["name"]]
        spec = make_spec(output, case["name"], entry["definition"])
        answers = read_json(output / "host_only" / case["name"] / "holdout_answers.json")["answers"]
        inputs = output / "inputs" / case["name"]
        fit, holdout, test = (_read_csv(inputs / (label + ".csv")) for label in ("fit", "holdout", "test"))
        full = pd.concat([fit, holdout.assign(**{cases.TARGET_COLUMN: answers})], ignore_index=True)
        require([phase["phase"] for phase in case["phases"]] == list(PHASES), "Planned phases changed")
        item = {"name": case["name"], "phases": []}
        for phase in case["phases"]:
            state = {"phase": phase["phase"], "status": phase["status"]}
            if phase["status"] == "completed":
                phase_fit, phase_predict = (fit, holdout) if phase["phase"] == "outer_validation" else (full, test)
                state.update(_audit_completed_phase(output, plan, entry, phase, spec, phase_fit, phase_predict, answers))
            item["phases"].append(state)
        report["cases"].append(item)
    phases = [phase for case in report["cases"] for phase in case["phases"]]
    report["completed_phases"] = sum(phase["status"] == "completed" for phase in phases)
    report["unsuccessful_phases"] = len(phases) - report["completed_phases"]
    report["all_matched"] = bool(phases) and all(
        phase["status"] == "completed" and all(value is True for key, value in phase.items()
                                               if key not in {"phase", "status"})
        for phase in phases)
    require(artifact_hashes(output) == before and file_hash(output / "results.json") == results_digest
            and file_hash(output / "results.seal.json") == seal_digest
            and source_hashes() == plan["sources"],
            "Evidence changed while auditing")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path,
                             help="Child directory of data/experiments/xgboost_candidate/")
    prepare_command = commands.choices["prepare"]
    prepare_command.add_argument("--case", action="append", choices=list(cases.CASES),
                                 help="Case to include; repeat it, default all cases")
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        plan = prepare(arguments.output, arguments.case)
        value = {"prepared": [entry["name"] for entry in plan["cases"]],
                 "plan_sha256": file_hash(_output_root(arguments.output) / "plan.json")}
    elif arguments.command == "run":
        value = run(arguments.output)["summary"]
    else:
        value = audit(arguments.output)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 2 if value.get("all_matched") is False or value.get("all_checks_passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
