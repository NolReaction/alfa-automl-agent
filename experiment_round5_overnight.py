"""Serial public native runs with sealed inputs and independent receipt audits.

Developer tool only; this file is not part of the flat competition package.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

from research_lab.integrity import (check_identity, file_hash, json_hash, read_json,
                                    require, safe_path, source_identity, utc_now, write_json)
from research_lab.process import coordinator_lock, machine_lock_path, supervise

ROOT = Path(__file__).resolve().parent
PROFILES = ("regression_off", "regression_on", "classification")
MEMORY_BYTES = 28 * 1024**3
THREADS = 8
SCOPE = ("Known public TRAIN development. Outer scores participate in selection, "
         "so they are not unbiased estimates or hidden TEST scores. No default promotion.")


def hashes(root, names):
    return {name: file_hash(Path(root) / name) for name in names}


def stage_task(source, destination):
    """Copy only organizer inputs, preserving task-relative paths and literal IDs."""
    from agent.task_spec import load_task
    from research_lab.audit import _csv, _project
    source, destination = safe_path(source), safe_path(destination)
    spec = load_task(source)
    data = safe_path(spec.data_dir)
    require(data.is_relative_to(source.parent), "Task data must be beneath task.md's directory")
    names = ["task.md", *[(data.relative_to(source.parent) / name).as_posix() for name in
             ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")]]
    original = hashes(source.parent, names)
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, target)
    require(hashes(destination, names) == original, "Input copy changed")
    spec = load_task(destination / "task.md")
    header, test = _csv(Path(spec.data_dir) / "test.csv")
    require(spec.target_column not in header, "TEST must not contain target answers")
    ids = [row[0] for row in _project(header, test, [spec.id_column])]
    contract = spec.to_dict()
    contract["prediction_kind"] = spec.prediction_kind
    return {"input_sha256": original, "contract": contract, "metric": spec.metric.name,
            "test_ids": ids, "scope": SCOPE}


def preflight(expected_sha, output, budget):
    import psutil
    require(os.name == "nt" and sys.version_info[:2] == (3, 12),
            "This qualification requires Windows and Python 3.12")
    require(re.fullmatch(r"[0-9a-fA-F]{40}", expected_sha), "Use an exact 40-character release SHA")
    require(type(budget) in (int, float) and math.isfinite(budget) and 300 < budget <= 10500,
            "Adaptive native budget must be >300 and <=10500 seconds")
    identity = source_identity(ROOT)
    require(identity["sha"].lower() == expected_sha.lower() and not identity["dirty"],
            "Release SHA must match the clean checkout")
    require(not output.exists() and not output.is_relative_to(ROOT),
            "Output must be a new directory outside the checkout")
    packages = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, expected = line.strip().split("==", 1)
            packages[name] = metadata.version(name)
            require(packages[name] == expected, "Pinned dependency mismatch: " + name)
    memory = psutil.virtual_memory()
    require(memory.total >= 30_000_000_000, "This plan requires the intended 32 GB workstation")
    require(shutil.disk_usage(output.parent).free >= 15 * 1024**3,
            "Keep at least 15 GiB free for fresh task copies, models and logs")
    return {"release": identity, "python": sys.version, "platform": platform.platform(),
            "packages": {item.metadata["Name"]: item.version for item in metadata.distributions()},
            "pinned_runtime_packages": packages, "physical_ram_bytes": memory.total,
            "available_ram_bytes": memory.available, "logical_cpus": os.cpu_count()}


def selected_execution_ok(execution):
    resources = execution.get("resources") or {}
    return (execution.get("exit_code") == 0 and not execution.get("error")
            and execution.get("timed_out") is False
            and execution.get("memory_limit_exceeded") is False
            and resources.get("available") is True and resources.get("cleanup_complete") is True)


def audit_job(job, plan):
    """Fail closed on missing evidence; do not run/import candidate code here."""
    from research_lab.audit import _quality_chain
    from validation import validate_predictions
    task = job / "task"
    seal, config = read_json(job / "inputs.json"), read_json(job / "config.json")
    execution, worker = read_json(job / "execution.json"), read_json(job / "worker_evidence.json")
    cap = plan["native_seconds_per_job"]
    checks = {
        "root_exit_zero": execution.get("exit_code") == 0,
        "watchdog_not_reached": execution.get("external_timeout") is False,
        "memory_limit_not_reached": execution.get("memory_limit_exceeded") is False,
        "tree_resources_observed": execution.get("resources", {}).get("available") is True,
        "owned_cleanup_complete": execution.get("cleanup", {}).get("complete") is True,
        "input_hashes_unchanged": hashes(task, seal["input_sha256"]) == seal["input_sha256"],
        "flat_hashes_unchanged": (hashes(plan["flat_root"], plan["flat_sha256"]) == plan["flat_sha256"]
            and {p.name for p in Path(plan["flat_root"]).iterdir() if p.is_file()} == set(plan["flat_sha256"])),
        "worker_config_unchanged": worker.get("config_sha256") == file_hash(job / "config.json"),
        "worker_returned": worker.get("solver_returned") is True
            and worker.get("cli_exit_code") == 0 and worker.get("terminal_error") is None,
        "observer_complete": not any(str(item.get("phase", "")).endswith("observer_error")
                                    for item in worker.get("observations", [])),
    }
    checks["source_identity_unchanged"] = check_identity(plan["release"]) is not None
    require(config["profile"] == job.name and config["time_budget_seconds"] == cap
            and config["task_dir"] == str(task) and config["solver_root"] == plan["flat_root"]
            and config.get("research_search", "adaptive") == plan.get("research_search", "adaptive"),
            "Job configuration differs from sealed plan")
    require(plan["job_seals"][job.name] == {"config": file_hash(job / "config.json"),
                                            "inputs": file_hash(job / "inputs.json")},
            "Job input/config seals changed")
    pointer = worker.get("run_report") or {}
    relative = pointer.get("path", "")
    require(isinstance(relative, str) and re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative),
            "Missing exact current-run report pointer")
    report_path = safe_path(task / relative)
    require(report_path.is_relative_to(task) and file_hash(report_path) == pointer.get("sha256")
            and pointer.get("origin") == "returned" and pointer.get("kind") == "research_v3",
            "Current-run pointer/hash mismatch")
    native = read_json(report_path)
    number = lambda x: type(x) in (int, float) and math.isfinite(x)
    native_budget, elapsed = native.get("time_budget_seconds"), native.get("elapsed_seconds")
    observed = [item for item in worker.get("observations", []) if item.get("phase") == "run_agent"]
    checks["one_adaptive_offline_invocation"] = (len(observed) == 1
        and observed[0].get("local_only") is True and observed[0].get("adaptive_search") is True
        and observed[0].get("adaptive_budget", False) is (plan.get("research_search", "adaptive") == "adaptive-budget")
        and observed[0].get("time_budget_seconds") == native_budget
        and native.get("coordinator", {}).get("active") is True)
    if plan.get("research_search", "adaptive") == "adaptive-budget":
        coordinator = native.get("coordinator", {})
        checks["budget_search_policy_recorded"] = (
            native.get("quality_options", {}).get("adaptive_budget") is True
            and coordinator.get("search_policy") == "adaptive-budget"
            and coordinator.get("local_trial_limit") == 256
            and coordinator.get("trial_limit_kind") == "safety")
    cli_budget = worker.get("cli_time_budget_seconds")
    end, deadline = worker.get("native_ended_monotonic"), worker.get("original_deadline_monotonic")
    checks["original_deadline_respected"] = (number(cli_budget) and number(native_budget)
        and number(elapsed) and number(end) and number(deadline)
        and worker.get("original_time_budget_seconds") == cap
        and 0 < native_budget <= cli_budget <= cap and 0 <= elapsed <= native_budget and end <= deadline)
    checks["zero_api"] = native.get("api", {}).get("attempts") == 0 and native.get("api", {}).get("responses") == 0
    selected = native.get("published_recipe")
    checks["fresh_full_refit_published"] = (native.get("status") == "completed"
        and native.get("publication", {}).get("new_result_published") is True
        and native.get("publication", {}).get("recipe") == selected)
    checks["selected_outer_execution_clean"] = selected_execution_ok(
        native.get("evaluations", {}).get("final_check", {}).get("execution", {}).get(selected, {}))
    checks["selected_refit_execution_clean"] = selected_execution_ok(native.get("refit_execution", {}).get(selected, {}))
    rescue = native.get("rescue_regression") or {}
    requested = job.name == "regression_on"
    checks["requested_rescue_profile_recorded"] = (
        native.get("quality_options", {}).get("rescue_regression", False) is requested
        and (rescue.get("requested") is True if requested else not rescue.get("requested", False)))
    validation = validate_predictions(task)
    checks["output_structure_valid"] = validation.get("valid") is True
    _, proof = _quality_chain(task, report_path, native, worker, seal)
    checks["outer_and_full_refit_receipts_valid"] = True
    return {"version": 1, "qualification": "requires_review", "checks": checks,
            "continuation_allowed": all(checks.values()), "proof": proof,
            "native_report": relative, "native_status": native.get("status"),
            "native_reason": native.get("reason"), "native_elapsed_seconds": elapsed,
            "rescue_regression": rescue, "output_validation": validation, "scope": SCOPE,
            "resource_scope": "Sampled root process tree and selected outer/refit executions; not an OS sandbox or proof of unsampled orphans."}


def comparison(output, queue):
    """Only compare two technically admitted runs from the predeclared pair."""
    audits = {item["profile"]: read_json(output / "jobs" / item["profile"] / "audit.json")
              for item in queue["jobs"] if item["status"] == "completed"}
    if not all(name in audits for name in PROFILES[:2]):
        return {"status": "incomplete", "scope": SCOPE}
    off, on = (audits[name] for name in PROFILES[:2])
    a, b = off["proof"]["outer_metric"], on["proof"]["outer_metric"]
    same = (a["name"] == b["name"] and off["proof"]["outer_split_sha256"] == on["proof"]["outer_split_sha256"])
    return {"status": "comparable" if same else "incompatible", "off": a, "on": b,
            "on_minus_off": b["value"] - a["value"] if same else None,
            "rescue_status": on["rescue_regression"],
            "rescue_exercised": on["rescue_regression"].get("attempted") is True,
            "note": "Requested ON may skip rescue. This single development pair does not establish a general quality gain.", "scope": SCOPE}


def export_summary(output):
    """Small diagnostic bundle; large data/model files stay in the original run."""
    names = []
    for path in output.rglob("*"):
        if path.is_file() and not path.is_symlink() and (path.suffix in {".json", ".txt"}
                or path.name in {"submission.csv", "predictions.csv", "candidate.py", "research_candidate.py"}):
            if "flat" not in path.relative_to(output).parts:
                names.append(path)
    limit, total, included, skipped = 256 * 1024**2, 0, {}, []
    target = output / "evidence.zip"
    temporary = output / "evidence.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(names):
            name = path.relative_to(output).as_posix()
            size = path.stat().st_size
            if size > 32 * 1024**2 or total + size > limit:
                skipped.append(name)
                continue
            included[name] = file_hash(path)
            archive.write(path, name)
            total += size
        import json
        archive.writestr("bundle_manifest.json", json.dumps({"sha256": included, "skipped": skipped,
            "note": "Diagnostic subset. Preserve the full original run for replay-free receipt/input review; fit/predict inputs and models omitted."}, sort_keys=True))
    os.replace(temporary, target)
    return target


def run_queue(output, plan, queue):
    """Caller holds the single machine lane for the entire queue."""
    for item in queue["jobs"]:
        job = output / "jobs" / item["profile"]
        item.update(status="running", started_at=utc_now())
        write_json(output / "queue.json", queue)
        print(f"Starting {item['profile']}; cap {plan['native_seconds_per_job']}s; logs: {job / 'execution'}", flush=True)
        try:
            deadline = time.monotonic() + plan["native_seconds_per_job"]
            argv = [sys.executable, "-I", "-u", str(ROOT / "experiment_round5_overnight_worker.py"),
                    str(job / "config.json"), "--deadline", str(deadline)]
            execution = supervise(argv, job / "task", job / "execution",
                plan["native_seconds_per_job"] + 30, MEMORY_BYTES, threads=THREADS)
            write_json(job / "execution.json", execution)
            result = audit_job(job, plan)
            write_json(job / "audit.json", result)
            require(result["continuation_allowed"], "Required automatic checks failed: " +
                    ", ".join(key for key, value in result["checks"].items() if not value))
            item.update(status="completed", finished_at=utc_now(), outer_metric=result["proof"]["outer_metric"])
            print(f"Completed {item['profile']}: {result['proof']['outer_metric']}; requires review", flush=True)
        except BaseException as error:
            if not (job / "audit.json").exists():
                write_json(job / "audit.json", {"version": 1, "qualification": "requires_review",
                    "continuation_allowed": False, "error": type(error).__name__ + ": " + str(error)[-4000:]})
            item.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                        finished_at=utc_now(), error=type(error).__name__ + ": " + str(error)[-4000:])
            queue.update(status="stopped", stopped_at=utc_now())
            write_json(output / "queue.json", queue)
            write_json(output / "comparison.json", comparison(output, queue))
            export_summary(output)
            raise
        write_json(output / "queue.json", queue)
    queue.update(status="completed", finished_at=utc_now())
    write_json(output / "queue.json", queue)
    write_json(output / "comparison.json", comparison(output, queue))
    print("Evidence:", export_summary(output), flush=True)


def start(expected_sha, regression, classification, output, budget=10500, *, research_search="adaptive"):
    require(research_search in {"adaptive", "adaptive-budget"}, "Unknown research search mode")
    if research_search == "adaptive-budget":
        require(type(budget) in (int, float) and math.isfinite(budget) and 360 <= budget <= 10500,
                "Budget search requires 360–10500 seconds, including bootstrap and finalization reserves")
    output = safe_path(output)
    with coordinator_lock(machine_lock_path()):
        plan = preflight(expected_sha, output, budget)
        output.mkdir(parents=True, exist_ok=False)
        plan.update(version=1, created_at=utc_now(), profiles=list(PROFILES), threads=THREADS,
                    native_seconds_per_job=budget, sampled_tree_rss_limit_bytes=MEMORY_BYTES,
                    watchdog_seconds_per_job=budget + 30, scope=SCOPE, job_seals={},
                    research_search=research_search)
        with (output / "build.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, str(ROOT / "build_submission.py"), "--output", str(output / "solution.zip")],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        flat = output / "flat"
        with zipfile.ZipFile(output / "solution.zip") as archive:
            require(all(Path(name).name == name for name in archive.namelist()), "Expected a flat ZIP")
            archive.extractall(flat)
        plan.update(flat_root=str(flat), zip_sha256=file_hash(output / "solution.zip"),
                    flat_sha256=hashes(flat, [path.name for path in flat.iterdir() if path.is_file()]))
        for profile in PROFILES:
            job = output / "jobs" / profile
            task = job / "task"
            seal = stage_task(classification if profile == "classification" else regression, task)
            require((seal["contract"]["metric"]["task_type"] == "classification") == (profile == "classification"),
                    "Task type differs from queue profile")
            if profile != "classification":
                require(seal["metric"] == "RMSE", "Regression comparison requires public RMSE")
            write_json(job / "inputs.json", seal)
            config = {"version": 1, "job_id": profile, "profile": profile, "task_dir": str(task),
                      "task_path": str(task / "task.md"), "solver_root": str(flat),
                      "worker_output": str(job / "worker_evidence.json"), "time_budget_seconds": budget,
                      "research_search": research_search}
            write_json(job / "config.json", config)
            plan["job_seals"][profile] = {"config": file_hash(job / "config.json"), "inputs": file_hash(job / "inputs.json")}
        off = read_json(output / "jobs" / "regression_off" / "inputs.json")
        on = read_json(output / "jobs" / "regression_on" / "inputs.json")
        require(off["input_sha256"] == on["input_sha256"], "Paired regression inputs differ")
        plan["plan_sha256"] = json_hash(plan)
        write_json(output / "plan.json", plan)
        queue = {"version": 1, "status": "running", "plan_sha256": plan["plan_sha256"],
                 "jobs": [{"profile": profile, "status": "pending"} for profile in PROFILES]}
        write_json(output / "queue.json", queue)
        run_queue(output, plan, queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--regression-task", type=Path, required=True)
    parser.add_argument("--classification-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=int, default=10500)
    parser.add_argument("--research-search", choices=("adaptive", "adaptive-budget"), default="adaptive")
    args = parser.parse_args()
    try:
        start(args.release_commit, args.regression_task, args.classification_task, args.output, args.time_budget,
              research_search=args.research_search)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("Queue stopped:", error, "Preserve:", args.output, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
