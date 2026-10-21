"""Preregistered, API-free, paired functional audit of long v3 CPU search.

Both arms run the same checked-out solver in fresh supervised processes. Only
adaptive_search differs. Audit labels stay in the host's sibling directory;
the ordinary file boundary is not an OS sandbox. This small synthetic audit
does not qualify a production feature or estimate a private contest score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import time
from uuid import uuid4

from agent.execution import run_python
from benchmark_agent import audit_submission, input_fingerprint, source_provenance, _research_summary
from benchmark_cases import ALL_CASE_NAMES, make_case, write_case


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ("nonlinear_regression", "multiclass_duplicates")
ARMS = ("control", "adaptive")
DEFAULT_CAP = 360.0
HOST_SOURCES = ("benchmark_maximize.py", "benchmark_agent.py", "benchmark_cases.py")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evidence must be an ordinary file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Expected bounded ordinary evidence JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def regular_under(root, relative):
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts):
        raise ValueError("Evidence path must be a relative POSIX path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    if (path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Evidence file escaped its output directory")
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise ValueError("Evidence parent must not be linked")
        current = current.parent
    return path


def environment():
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "optuna", "psutil",
                 "autogluon.tabular"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def host_sources():
    return {name: file_hash(ROOT / name) for name in HOST_SOURCES}


def prepare(output, *, solver_root=ROOT, cases=DEFAULT_CASES, seeds=(42,),
            time_budget=DEFAULT_CAP, memory_mib=4096):
    """Seal sources, environment, both task copies and audit labels before fits."""
    if (not isinstance(cases, (list, tuple)) or not cases or len(set(cases)) != len(cases)
            or not set(cases) <= set(ALL_CASE_NAMES)):
        raise ValueError("Select distinct known synthetic cases")
    if (not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if (type(time_budget) not in {int, float} or not math.isfinite(time_budget)
            or not 300 < time_budget <= 10500):
        raise ValueError("Long-run budget must be in (300, 10500] seconds")
    if type(memory_mib) is not int or not 128 <= memory_mib <= 28 * 1024:
        raise ValueError("Memory cap must be an integer from 128 through 28672 MiB")
    output = Path(output).expanduser().resolve()
    solver_root = Path(solver_root).expanduser().resolve()
    source = source_provenance(solver_root)
    for name in ("agent/research_v3.py", "agent/research_candidate.py"):
        if not (solver_root / name).is_file() or (solver_root / name).is_symlink():
            raise ValueError("Selected solver must provide the real v3 interface")
    method = {"version": 1, "kind": "paired_long_v3_functional_audit",
              "cases": list(cases), "seeds": list(seeds), "arms": list(ARMS),
              "time_budget_seconds": float(time_budget), "memory_limit_bytes": memory_mib * 1024 ** 2,
              "max_steps": 12, "local_only": True, "source_sha256": source["source_sha256"],
              "host_sources": host_sources(), "environment": environment(),
              "fit_protocol": "same_v3_control_vs_adaptive_end_to_end",
              "quality_admission": False, "synthetic_only": True}
    output.mkdir(parents=True, exist_ok=False)
    (output / "host_audit").mkdir()
    entries = []
    for number, (name, seed) in enumerate((name, seed) for name in cases for seed in seeds):
        case = make_case(name, seed)
        key = name + "_" + str(seed)
        audit = {"case": name, "seed": seed, "scope": "host_external_audit",
                 "actuals": list(case.audit_actuals), "ids": case.test["row_key"].tolist()}
        audit_path = output / "host_audit" / (key + ".json")
        write_json(audit_path, audit)
        tasks = {}
        fingerprints = []
        for arm in ARMS:
            task = write_case(case, output / "runs" / key / arm)
            fingerprints.append(input_fingerprint(task))
            tasks[arm] = task.relative_to(output).as_posix()
        if len(set(fingerprints)) != 1:
            raise ValueError("Paired task input bytes differ")
        entries.append({"key": key, "case": name, "seed": seed, "metric": case.metric,
                        "validation": case.validation, "input_sha256": fingerprints[0], "tasks": tasks,
                        "audit": audit_path.relative_to(output).as_posix(), "audit_sha256": file_hash(audit_path),
                        "arm_order": list(ARMS if number % 2 == 0 else reversed(ARMS))})
    plan = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "method": method, "method_sha256": json_hash(method), "solver": source, "entries": entries,
            "input_boundary": "ordinary allowed inputs, not an OS sandbox"}
    write_json(output / "plan.json", plan)
    write_json(output / "plan_seal.json", {"plan_sha256": file_hash(output / "plan.json")})
    return plan


def load_plan(output):
    output = Path(output).expanduser().resolve()
    plan = read_json(output / "plan.json")
    seal = read_json(output / "plan_seal.json")
    if seal != {"plan_sha256": file_hash(output / "plan.json")}:
        raise ValueError("Preregistration seal changed")
    if (plan.get("version") != 1 or not isinstance(plan.get("method"), dict)
            or plan.get("method_sha256") != json_hash(plan["method"])):
        raise ValueError("Preregistered method changed")
    method = plan["method"]
    if (method.get("kind") != "paired_long_v3_functional_audit" or method.get("arms") != list(ARMS)
            or method.get("local_only") is not True or method.get("quality_admission") is not False
            or method.get("max_steps") != 12 or method.get("synthetic_only") is not True):
        raise ValueError("Unsupported long-run audit contract")
    source = source_provenance(Path(plan["solver"]["root"]))
    if source["source_sha256"] != method["source_sha256"]:
        raise ValueError("Solver source bytes changed since preregistration")
    if host_sources() != method.get("host_sources") or environment() != method.get("environment"):
        raise ValueError("Host method or environment changed since preregistration")
    for entry in plan["entries"]:
        if entry.get("arm_order") not in [list(ARMS), list(reversed(ARMS))]:
            raise ValueError("Invalid fixed arm order")
        if file_hash(regular_under(output, entry["audit"])) != entry["audit_sha256"]:
            raise ValueError("Host audit answers changed")
        for arm in ARMS:
            if input_fingerprint(regular_under(output, entry["tasks"][arm])) != entry["input_sha256"]:
                raise ValueError("Paired task inputs changed")
    return output, plan


def driver(solver_root, task, arm, budget, max_steps=12):
    """No audit path/labels/client/env access occurs in the solver worker."""
    if arm not in ARMS:
        raise ValueError("Unknown paired arm")
    # Observation wrappers preserve arguments/results, add no modelling choices,
    # and consume the same original wall-clock cap in both arms.
    return f'''import time
started = time.monotonic()
import sys
sys.path.insert(0, {str(solver_root)!r})
from pathlib import Path
import hashlib
import json
import functools
import re
from agent import research_v3 as research

task = Path({str(task)!r})
output = task.parent / "solver_output"
output.mkdir()
observations = []
current_run = {{}}
def observed(name, function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        begin = time.monotonic()
        row = {{"phase": name}}
        try:
            if name == "prepare_research":
                state = args[0] if args else kwargs.get("state")
                current_run["directory"] = getattr(state, "run_dir", None)
            value = function(*args, **kwargs)
            execution = getattr(value, "execution", None)
            if isinstance(execution, dict):
                row["execution"] = {{key: execution.get(key) for key in (
                    "exit_code", "timed_out", "memory_limit_exceeded", "elapsed_seconds", "resources", "error")}}
            return value
        except Exception as error:
            row["error_type"] = type(error).__name__
            execution = getattr(error, "execution", None)
            if isinstance(execution, dict):
                row["execution"] = {{key: execution.get(key) for key in (
                    "exit_code", "timed_out", "memory_limit_exceeded", "elapsed_seconds", "resources", "error")}}
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - begin
            observations.append(row)
    return call
for name in ("prepare_research", "prepare_baseline", "finalize"):
    function = getattr(research, name, None)
    if callable(function):
        setattr(research, name, observed(name, function))
native = getattr(research, "run_candidate", None)
if callable(native):
    def native_observed(*args, **kwargs):
        return observed("candidate_" + str(kwargs.get("phase")), native)(*args, **kwargs)
    research.run_candidate = native_observed
remaining = {budget!r} - (time.monotonic() - started) - 2.0
evidence = {{"version": 2, "mode": "paired_maximize", "arm": {arm!r}, "external_api_calls": 0,
            "requested_time_budget_seconds": remaining, "run_report": None,
            "run_report_origin": None, "solver_returned": False, "terminal_error_type": None}}
try:
    if remaining <= 300:
        raise TimeoutError("Startup left too little time to exercise long-run search")
    directory = research.run_agent_v3(task, None, max_steps={max_steps!r}, time_budget=remaining,
                                     local_only=True, adaptive_search={arm == 'adaptive'!r})
    current_run["returned_directory"] = directory
    evidence["solver_returned"] = True
except Exception as error:
    evidence["terminal_error_type"] = type(error).__name__[:80]
    raise
finally:
    directory = current_run.get("returned_directory", current_run.get("directory"))
    if directory is not None:
        try:
            root = task.parent.resolve()
            directory = Path(directory)
            if not directory.is_absolute():
                directory = root / directory
            path = directory / "run.json"
            relative = path.relative_to(root).as_posix()
            if (not re.fullmatch(r"\\.agent_runs/[0-9a-f]{{32}}/run\\.json", relative)
                    or path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink()
                    or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024
                    or path.resolve().relative_to(root).as_posix() != relative):
                raise ValueError("Current report is not an ordinary exact run path")
            evidence["run_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            evidence["run_report"] = relative
            evidence["run_report_origin"] = ("returned" if evidence["solver_returned"]
                                             else "observed_state_after_failure")
        except (OSError, ValueError, TypeError) as error:
            evidence["run_report_error_type"] = type(error).__name__[:80]
    evidence["worker_elapsed_seconds"] = time.monotonic() - started
    evidence["observations"] = observations
    (output / "research_benchmark.json").write_text(json.dumps(evidence, allow_nan=False), encoding="utf-8")
'''


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def summarize_research(task_dir, arm, budget):
    try:
        return _summarize_research(task_dir, arm, budget)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None, ["malformed_research_evidence"]


def _summarize_research(task_dir, arm, budget):
    task_dir = Path(task_dir).resolve()
    evidence = read_json(regular_under(task_dir, "solver_output/research_benchmark.json"))
    relative = evidence.get("run_report") if isinstance(evidence, dict) else None
    if not isinstance(relative, str) or not re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative):
        return None, ["missing_or_invalid_current_report"]
    report_path = regular_under(task_dir, relative)
    if report_path.stat().st_size > 64 * 1024 * 1024:
        return None, ["missing_or_invalid_current_report"]
    content = report_path.read_bytes()
    if (len(content) > 64 * 1024 * 1024
            or hashlib.sha256(content).hexdigest() != evidence.get("run_report_sha256")):
        return None, ["missing_or_invalid_current_report"]
    raw = json.loads(content)
    if not isinstance(raw, dict):
        return None, ["missing_or_invalid_current_report"]
    # A failed invocation supplies diagnostic data only, never a receipt claim.
    # Its missing/stale publication artifacts must not erase the pinned report.
    summary = ({"receipt_checks_valid": False} if evidence.get("solver_returned") is False
               else _research_summary(task_dir))
    if summary is None:
        return None, ["missing_or_invalid_current_report"]
    errors = []
    requested = evidence.get("requested_time_budget_seconds")
    if (evidence.get("version") != 2 or evidence.get("mode") != "paired_maximize"
            or evidence.get("arm") != arm or evidence.get("external_api_calls") != 0
            or type(requested) not in {int, float} or not 300 < requested <= budget
            or raw.get("time_budget_seconds") != requested):
        errors.append("original_offline_deadline_contract")
    if evidence.get("solver_returned") is not True:
        errors.append("solver_did_not_return")
    if ((evidence.get("solver_returned") is True and evidence.get("run_report_origin") != "returned")
            or (evidence.get("solver_returned") is False
                and evidence.get("run_report_origin") != "observed_state_after_failure")
            or type(evidence.get("solver_returned")) is not bool):
        errors.append("current_report_origin_contract")
    api = raw.get("api") or {}
    if api.get("attempts") != 0 or api.get("responses") != 0 or raw.get("mode") != "local_only":
        errors.append("offline_api_contract")
    if (raw.get("research_protocol_version") != 3 or raw.get("stage") != "result"
            or raw.get("status") not in {"completed", "fallback"}
            or (raw.get("publication") or {}).get("new_result_published") is not True
            or (raw.get("output_check") or {}).get("valid") is not True
            or summary.get("receipt_checks_valid") is not True):
        errors.append("verified_full_refit_contract")
    if arm == "adaptive" and (raw.get("coordinator") or {}).get("active") is not True:
        errors.append("adaptive_coordinator_not_active")
    if arm == "control" and (raw.get("coordinator") or {}).get("active") is True:
        errors.append("control_adaptive_contamination")
    elapsed, remaining = raw.get("elapsed_seconds"), (raw.get("time_allocation") or {}).get("remaining_seconds")
    if (type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0
            or type(remaining) not in {int, float} or not math.isfinite(remaining) or remaining < 0
            or (type(requested) in {int, float} and elapsed > requested + .1)):
        errors.append("invalid_or_exhausted_host_time_accounting")
    published_family = None
    if summary.get("receipt_checks_valid") is True:
        receipt = Path(raw["candidate_receipts"]["refit"])
        if not receipt.is_absolute():
            receipt = task_dir / receipt
        recipe_path = receipt.parent / "recipe.json"
        recipe_relative = recipe_path.resolve().relative_to(task_dir.resolve()).as_posix()
        recipe_path = regular_under(task_dir, recipe_relative)
        frozen = raw.get("frozen_" + str(raw.get("published_recipe"))) or {}
        if file_hash(recipe_path) != frozen.get("recipe_hash"):
            errors.append("published_recipe_bytes_changed")
        else:
            recipe = read_json(recipe_path)
            published_family = recipe.get("family") or recipe.get("action")
            if published_family is None and isinstance(recipe.get("components"), list):
                published_family = "equal_blend:" + "+".join(str(item.get("action", "custom"))
                    for item in recipe["components"] if isinstance(item, dict))
    local = raw.get("local_research") or {}
    extensions = raw.get("research_extensions") or {}
    evaluations = raw.get("evaluations") or {}
    measurements = (raw.get("time_allocation") or {}).get("measurements") or {}
    measured_fits = sum(len(values) for values in measurements.values() if isinstance(values, list))
    families = Counter(row.get("family", "unspecified") for row in local.get("trials", []) if isinstance(row, dict))
    observations = evidence.get("observations") or []
    phase_times = {}
    invocation_counts = Counter()
    for row in observations:
        if (isinstance(row, dict) and isinstance(row.get("phase"), str)
                and type(row.get("elapsed_seconds")) in {int, float}
                and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0):
            phase_times[row["phase"]] = phase_times.get(row["phase"], 0) + row["elapsed_seconds"]
            if row["phase"].startswith("candidate_"):
                invocation_counts[row["phase"]] += 1
    result = {"status": raw.get("status"), "stage": raw.get("stage"), "stop_reason": raw.get("reason"),
              "solver_returned": evidence.get("solver_returned"),
              "run_report_origin": evidence.get("run_report_origin"),
              "terminal_error_type": evidence.get("terminal_error_type"),
              "research_stop": raw.get("research_stop"), "local_stop_reason": local.get("reason"),
              "coordinator": raw.get("coordinator"), "api_attempts": api.get("attempts"),
              "api_responses": api.get("responses"), "elapsed_seconds": raw.get("elapsed_seconds"),
              "remaining_seconds": (raw.get("time_allocation") or {}).get("remaining_seconds"),
              "requested_time_budget_seconds": requested, "published_recipe": raw.get("published_recipe"),
              "published_family": published_family,
              "publication": raw.get("publication"), "inner_best_metric": evaluations.get("selection"),
              "outer_final_check": evaluations.get("final_check"),
              "local_attempted": _integer(local.get("attempted")), "local_verified": _integer(local.get("verified")),
              "adaptive_attempted": _integer(local.get("adaptive_attempted")),
              "extension_attempted": _integer(extensions.get("attempted")),
              "recorded_recipe_fit_measurements": measured_fits, "local_families": dict(families),
              "observed_candidate_invocations": dict(invocation_counts),
              "observed_phase_wall_seconds": phase_times,
              "phase_times_note": "Wrapper wall times overlap: finalize includes outer/refit; do not sum all phases.",
              "receipt_checks_valid": summary["receipt_checks_valid"], "observations": observations,
              "raw_report": evidence["run_report"], "raw_report_sha256": evidence["run_report_sha256"]}
    return result, errors


def resource_observation(value):
    """Unknown inventory remains unknown; failures with unknown cleanup censor."""
    records, censors = [], []

    def walk(item, path="root"):
        if isinstance(item, dict):
            resources = item.get("resources")
            launcher = path == "root/launcher"
            execution_like = launcher or any(key in item for key in
                                 ("resources", "exit_code", "timed_out", "memory_limit_exceeded"))
            unobserved_candidate_error = (isinstance(item.get("phase"), str)
                and item["phase"].startswith("candidate_") and bool(item.get("error_type"))
                and not isinstance(item.get("execution"), dict))
            if execution_like or unobserved_candidate_error:
                resources = resources if isinstance(resources, dict) else {}
                records.append({"path": path, "available": resources.get("available"),
                                "cleanup_complete": resources.get("cleanup_complete"),
                                "peak_rss_bytes": resources.get("peak_rss_bytes")})
                invalid_exit = ((launcher and "exit_code" not in item)
                    or ("exit_code" in item
                        and (type(item["exit_code"]) is not int or item["exit_code"] != 0)))
                failed = bool(invalid_exit or item.get("timed_out") or item.get("memory_limit_exceeded")
                              or item.get("error") or item.get("error_type") or unobserved_candidate_error)
                if (resources.get("cleanup_complete") is False
                        or (failed and resources.get("cleanup_complete") is not True)):
                    censors.append(path)
            for key, child in item.items():
                walk(child, path + "/" + str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, path + "/" + str(index))

    walk(value)
    complete = bool(records) and all(row["available"] is True and row["cleanup_complete"] is True for row in records)
    return {"fully_observed": complete, "execution_records": records,
            "censored_cleanup_paths": censors,
            "qualification": "censored" if censors else "observed" if complete else "provisional_resource_observation"}


def run_arm(output, plan, entry, arm):
    # Recheck all sealed inputs and method immediately before every child.
    load_plan(output)
    task = regular_under(output, entry["tasks"][arm])
    if (task.parent / "execution").exists() or (task.parent / "solver_output").exists():
        raise FileExistsError("A paired arm cannot be overwritten or resumed")
    budget = plan["method"]["time_budget_seconds"]
    started = time.monotonic()
    try:
        execution = run_python(driver(Path(plan["solver"]["root"]), task, arm, budget),
                               task.parent, task.parent / "execution", 0, timeout=budget,
                               max_memory_bytes=plan["method"]["memory_limit_bytes"])
    except (OSError, ValueError, RuntimeError) as error:
        execution = {"exit_code": None, "error_type": type(error).__name__, "resources": {"available": False}}
    elapsed = time.monotonic() - started
    integrity = True
    try:
        load_plan(output)
    except (OSError, ValueError, KeyError, TypeError):
        integrity = False
    research, errors = summarize_research(task.parent, arm, budget)
    try:
        audit_data = read_json(regular_under(output, entry["audit"]))
        audit = audit_submission(task, tuple(audit_data["actuals"]))
    except (OSError, ValueError, TypeError, KeyError):
        audit = {"valid": False, "metric": None, "error": "host_audit_unavailable"}
    resources = resource_observation({"launcher": execution,
        "worker": research.get("observations", []) if research else [],
        "report": read_json(regular_under(task.parent, research["raw_report"])) if research else {}})
    if not integrity:
        status = "integrity_failed"
        audit["metric"] = None
    elif resources["censored_cleanup_paths"]:
        status = "censored"
    elif (execution.get("exit_code") != 0 or execution.get("timed_out")
          or execution.get("memory_limit_exceeded") or execution.get("error") or execution.get("error_type")):
        status = "solver_failed"
    elif elapsed > budget + .25:
        status = "deadline_exceeded"
    elif errors:
        status = "unverified_contract"
    elif not audit["valid"]:
        status = "invalid_output"
    else:
        status = "completed"
    result = {"case": entry["case"], "seed": entry["seed"], "arm": arm,
              "status": status, "time_budget_seconds": budget, "elapsed_seconds": elapsed,
              "execution": execution, "research": research, "contract_errors": errors,
              "integrity_unchanged": integrity, "audit": audit, "resources": resources,
              "quality_admission": False}
    write_json(task.parent / "arm_summary.json", result)
    return result


def pair_result(entry, arms):
    completed = all(arms.get(arm, {}).get("status") == "completed" for arm in ARMS)
    result = {"case": entry["case"], "seed": entry["seed"], "arms": arms,
              "functionally_comparable": completed, "quality_admission": False,
              "outcome": None, "improvement": None,
              "qualification": "incomplete"}
    if completed:
        left, right = (arms[arm]["audit"]["metric"] for arm in ARMS)
        if left["name"] != right["name"] or left["higher_is_better"] != right["higher_is_better"]:
            raise ValueError("Paired metric identity changed")
        delta = right["value"] - left["value"]
        gain = delta if left["higher_is_better"] else -delta
        result.update(improvement=gain, outcome="tie" if abs(gain) <= 1e-12 else "win" if gain > 0 else "loss",
                      qualification="functional_only" if all(arms[a]["resources"]["fully_observed"] for a in ARMS)
                                    else "provisional_resource_observation")
    return result


def markdown_report(report):
    lines = ["# Long-run v3 functional audit", "",
             "Synthetic end-to-end comparison. No production quality admission or private-score estimate.", "",
             "| Case / seed | Arm | Status | Audit metric | Seconds | Local / adaptive attempts | Stop | Resources |",
             "| --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for pair in report["pairs"]:
        for arm in ARMS:
            row = pair["arms"].get(arm)
            if row is None:
                lines.append(f"| {pair['case']} / {pair['seed']} | {arm} | not_started | — | — | — | — | — |")
                continue
            research, metric = row.get("research") or {}, row["audit"].get("metric")
            label = "—" if metric is None else f"{metric['name']}={metric['value']:.8g}"
            lines.append(f"| {pair['case']} / {pair['seed']} | {arm} | {row['status']} | {label} | "
                         f"{row['elapsed_seconds']:.3f} | {research.get('local_attempted')} / {research.get('adaptive_attempted')} | "
                         f"{research.get('local_stop_reason') or research.get('stop_reason') or research.get('terminal_error_type')} | "
                         f"{row['resources']['qualification']} |")
    lines.extend(["", "Observed phase wall times overlap; finalize contains outer/refit work. CPU time and RAM enforcement "
                  "are not inferred from elapsed time. Missing process inventory remains explicitly unknown.", "",
                  f"Status: {report['status']}. API requests: 0. Quality admission: false."])
    return "\n".join(lines) + "\n"


def run_prepared(output):
    output, plan = load_plan(output)
    if (output / "report.json").exists():
        raise FileExistsError("Completed or partial audit report already exists")
    report = {"version": 1, "kind": "paired_long_v3_functional_audit",
              "method_sha256": plan["method_sha256"], "plan_sha256": file_hash(output / "plan.json"),
              "status": "running", "pairs": [], "api_requests": 0, "quality_admission": False,
              "scope": "known synthetic end-to-end functional evidence, not independent feature qualification"}
    write_json(output / "report.json", report)
    stopped = False
    for entry in plan["entries"]:
        arms = {}
        for arm in entry["arm_order"]:
            row = run_arm(output, plan, entry, arm)
            arms[arm] = row
            print(json.dumps({"case": entry["case"], "seed": entry["seed"], "arm": arm,
                              "status": row["status"], "elapsed_seconds": round(row["elapsed_seconds"], 3),
                              "audit": row["audit"].get("metric")}, ensure_ascii=False), flush=True)
            if row["status"] in {"integrity_failed", "censored"}:
                stopped = True
                break
        report["pairs"].append(pair_result(entry, arms))
        write_json(output / "report.json", report)
        if stopped:
            break
    report["status"] = ("completed" if len(report["pairs"]) == len(plan["entries"])
                        and all(p["functionally_comparable"] for p in report["pairs"]) else "incomplete")
    report["outcomes"] = dict(Counter(p["outcome"] for p in report["pairs"] if p["outcome"] is not None))
    report["planned_arms"] = len(plan["entries"]) * 2
    report["executed_arms"] = sum(len(pair["arms"]) for pair in report["pairs"])
    report["aggregate_metric"] = None
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    write_json(output / "completion_seal.json", {"report_sha256": file_hash(output / "report.json"),
                                               "markdown_sha256": file_hash(output / "report.md")})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--solver-root", type=Path, default=ROOT)
    prepare_parser.add_argument("--cases", nargs="+", choices=ALL_CASE_NAMES, default=DEFAULT_CASES)
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    prepare_parser.add_argument("--time-budget", type=float, default=DEFAULT_CAP)
    prepare_parser.add_argument("--memory-mib", type=int, default=4096)
    run_parser = commands.add_parser("run-prepared")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output_dir, solver_root=args.solver_root, cases=args.cases, seeds=args.seeds,
                             time_budget=args.time_budget, memory_mib=args.memory_mib)
            print(json.dumps({"prepared": True, "planned_arms": len(result["entries"]) * 2,
                              "method_sha256": result["method_sha256"]}))
            return 0
        result = run_prepared(args.output_dir)
        print(json.dumps({"status": result["status"], "api_requests": 0,
                          "report": str(args.output_dir.resolve() / "report.json")}))
        return 0 if result["status"] == "completed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        # All worker calls are offline; no exception/provider body is printed.
        print(json.dumps({"status": "error", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
