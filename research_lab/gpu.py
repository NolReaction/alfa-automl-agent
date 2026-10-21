"""Opt-in CatBoost GPU baseline on a sealed CPU pilot's development splits.

This is exploration, never CPU qualification or a candidate policy backend.
Importing, preparing or probing this module does not fit a model. LightGBM's
presence is reported separately; an import is not proof of a working GPU build.
"""
from __future__ import annotations

import time
_NATIVE_STARTED = time.monotonic()

import argparse
from contextlib import contextmanager
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.integrity import (ROOT, file_hash, file_index, json_hash, read_json,
                                    require, safe_path, sanitized_environment, under,
                                    utc_now, write_json)


VERSION = 1
THREADS = 8
MEMORY_BYTES = 24 * 1024**3
PROBE_SECONDS = 20
INPUTS = ("task.md", "inputs/train.csv", "inputs/test.csv", "inputs/data_dictionary.csv", "inputs/sample_submission.csv")
FIXED_PARAMETERS = {"iterations": 256, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0,
                    "task_type": "GPU", "devices": "0", "thread_count": THREADS,
                    "gpu_ram_part": 0.8, "boosting_type": "Plain", "verbose": False,
                    "allow_writing_files": False}


def _input_hashes(task):
    return {name: file_hash(under(task, Path(task) / name)) for name in INPUTS}


@contextmanager
def _gpu_environment():
    """Only the GPU child gains device 0; restore the trusted adapter on error."""
    import qualify_unseen_tasks as trusted
    from research_lab.process import _ADAPTER_LOCK
    with _ADAPTER_LOCK:
        original = trusted.sanitized_environment
        environment = original()
        environment.update({name: str(THREADS) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        trusted.sanitized_environment = lambda: dict(environment)
        try:
            yield
        finally:
            trusted.sanitized_environment = original


def _supervise(argv, cwd, output, timeout):
    import qualify_unseen_tasks as trusted
    require(type(timeout) in {int, float} and math.isfinite(timeout) and timeout > 0, "Invalid remaining GPU budget")
    with _gpu_environment():
        original_stop, cleanup_seconds = trusted._OwnedProcesses.stop, 0.0
        def stop(owned, *args, **kwargs):
            nonlocal cleanup_seconds
            started = time.monotonic()
            try:
                return original_stop(owned, *args, **kwargs)
            finally:
                cleanup_seconds += time.monotonic() - started
        trusted._OwnedProcesses.stop = stop
        try:
            result = trusted.run_solver(argv, cwd, output, timeout, MEMORY_BYTES)
        finally:
            trusted._OwnedProcesses.stop = original_stop
        result.update(cleanup_elapsed_seconds=cleanup_seconds,
                      worker_wall_seconds=max(0.0, result["elapsed_seconds"] - cleanup_seconds),
                      memory_limit_bytes=MEMORY_BYTES, threads=THREADS)
        return result


def probe_worker(path):
    """Read hardware/driver/library capability, without training a tiny model."""
    result = {"version": VERSION, "kind": "read_only_gpu_capability", "platform": sys.platform,
              "python": sys.version, "executable": sys.executable, "libraries": {}, "hardware": [],
              "catboost_gpu_devices": None, "available": False, "reason": None, "errors": []}
    for name in ("catboost", "lightgbm", "numpy", "pandas", "psutil"):
        try:
            result["libraries"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["libraries"][name] = None
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            process = subprocess.run([executable, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8, check=False)
            result["nvidia_smi_exit_code"] = process.returncode
            result["hardware"] = process.stdout[:16384].splitlines() if process.returncode == 0 else []
            if process.returncode != 0: result["errors"].append("nvidia-smi: " + process.stderr[-2000:])
        except (OSError, subprocess.SubprocessError) as error:
            result["errors"].append(type(error).__name__ + ": " + str(error)[-2000:])
    else:
        result["errors"].append("nvidia-smi unavailable; device name/driver/VRAM unknown")
    result["lightgbm"] = {"version": result["libraries"]["lightgbm"], "gpu_verified": False,
                           "status": "skipped", "reason": "No read-only GPU backend API; no LightGBM fit probe or backend substitution"}
    if result["libraries"]["catboost"] is None:
        result["reason"] = "catboost_not_installed"
    else:
        try:
            from catboost.utils import get_gpu_device_count
            count = get_gpu_device_count()
            require(type(count) is int and count >= 0, "CatBoost returned invalid device count")
            result["catboost_gpu_devices"] = count
            result["available"] = count > 0
            result["reason"] = "catboost_cuda_device_available" if count else "no_usable_catboost_cuda_device"
        except Exception as error:
            result["reason"] = "catboost_cuda_probe_failed"
            result["errors"].append(type(error).__name__ + ": " + str(error)[-2000:])
    write_json(path, result)
    return result


def _pilot_gate(cpu_output, plan):
    """Observe terminal CPU work, not its scores, before allowing a GPU fit."""
    if plan.get("profile") != "pilot" or len(plan.get("cases", [])) != 4:
        return {"allowed": False, "reason": "requires_four_case_cpu_development_pilot", "terminal_arms": 0}
    terminal, states = 0, {}
    for case in plan["cases"]:
        for arm in case["arms"].values():
            path = under(cpu_output, Path(cpu_output) / arm["state_path"])
            state = read_json(path)
            if state.get("status") not in {"finished", "failed", "completed"}:
                return {"allowed": False, "reason": "cpu_pilot_incomplete_or_censored", "terminal_arms": terminal}
            marker = path.with_suffix(".sha256")
            require(safe_path(marker).read_text(encoding="ascii").strip() == file_hash(path), "CPU pilot terminal state seal changed")
            if state.get("execution", {}).get("cleanup", {}).get("complete") is not True:
                return {"allowed": False, "reason": "cpu_pilot_cleanup_unverified", "terminal_arms": terminal}
            states[path.relative_to(cpu_output).as_posix()] = file_hash(path)
            terminal += 1
    require(terminal == 8, "CPU pilot must have exactly eight terminal arms")
    from research_lab.audit import re_audit
    audit = re_audit(cpu_output)
    require(audit.get("valid") is True, "CPU pilot independent re-audit failed")
    return {"allowed": True, "reason": "cpu_pilot_terminal_and_reaudited", "terminal_arms": terminal,
            "state_sha256": states, "audit_sha256": json_hash(audit)}


def _features(frame, spec):
    """Static conversion only: CatBoost learns its own fit-only transforms."""
    import numpy as np
    import pandas as pd
    require(list(frame.columns) == list(spec.features), "GPU feature schema/order differs")
    result = frame.copy(deep=True)
    for name in spec.numeric:
        result[name] = pd.to_numeric(result[name].replace("", np.nan), errors="raise").astype(float)
        require(not np.isinf(result[name].to_numpy()).any(), "Infinite numeric GPU input")
    for name in spec.categorical:
        require(all(isinstance(value, str) for value in result[name]), "Categorical GPU values must remain literal strings")
    return result


def _fit_predict(train, test, spec, seed):
    """One fixed exploratory baseline. No audit validation or policy search."""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, CatBoostRegressor
    x_fit = _features(train.loc[:, list(spec.features)], spec)
    x_test = _features(test.loc[:, list(spec.features)], spec)
    parameters = {**FIXED_PARAMETERS, "random_seed": seed,
                  "loss_function": ("Logloss" if len(spec.classes) == 2 else "MultiClass") if spec.classification else "RMSE"}
    model = (CatBoostClassifier if spec.classification else CatBoostRegressor)(**parameters)
    y = train[spec.target_column].tolist() if spec.classification else pd.to_numeric(train[spec.target_column], errors="raise").to_numpy(dtype=float)
    if not spec.classification: require(np.isfinite(y).all(), "Nonfinite GPU regression targets")
    model.fit(x_fit, y, cat_features=list(spec.categorical))
    classes = list(map(str, model.classes_)) if spec.classification else None
    if spec.classification: require(set(classes) == set(spec.classes), "GPU model classes differ from the frozen contract")
    if spec.prediction_kind == "probabilities":
        values = model.predict_proba(x_test)
    else:
        values = np.asarray(model.predict(x_test)).reshape(-1)
    return spec.prediction_frame(test[spec.id_column].tolist(), values, prediction_classes=classes), parameters, classes


def worker(config_path, deadline):
    """Bounded child entry. Its config contains no audit labels or truth paths."""
    config_path = safe_path(config_path)
    config = read_json(config_path)
    require(set(config) == {"version", "run_id", "task_dir", "input_hashes", "seed", "source_sha256", "parameters"}, "Unexpected GPU worker fields")
    require(config["version"] == VERSION and config["parameters"] == FIXED_PARAMETERS, "GPU baseline configuration drift")
    require(file_hash(__file__) == config["source_sha256"], "GPU worker source drift")
    require(math.isfinite(deadline) and time.monotonic() < deadline, "Original GPU deadline expired during imports")
    task = safe_path(config["task_dir"])
    require(_input_hashes(task) == config["input_hashes"], "GPU task input drift")
    path = task / "submission.csv"
    require(not path.exists(), "GPU refuses a stale submission")
    from agent.task_spec import load_task
    import pandas as pd
    spec = load_task(task / "task.md")
    train = pd.read_csv(spec.data_dir / "train.csv", dtype=str, keep_default_na=False, na_filter=False)
    test = pd.read_csv(spec.data_dir / "test.csv", dtype=str, keep_default_na=False, na_filter=False)
    require(spec.target_column not in test, "GPU test contains a target")
    require(time.monotonic() < deadline, "Original GPU deadline expired before fitting")
    output, parameters, classes = _fit_predict(train, test, spec, config["seed"])
    require(time.monotonic() < deadline, "GPU fit exceeded the original deadline")
    require(_input_hashes(task) == config["input_hashes"], "GPU inputs mutated while fitting")
    output.to_csv(path, index=False, lineterminator="\n")
    receipt = {"version": VERSION, "run_id": config["run_id"], "config_sha256": file_hash(config_path),
               "source_sha256": config["source_sha256"], "input_hashes": config["input_hashes"],
               "output_sha256": file_hash(path), "parameters": parameters, "prediction_classes": classes,
               "native_started": _NATIVE_STARTED, "native_ended": time.monotonic(), "deadline": deadline,
               "api_attempts": 0, "label": "exploratory_gpu"}
    require(receipt["native_ended"] < deadline, "GPU publication exceeded original deadline")
    write_json(config_path.parent / "receipt.json", receipt)
    return receipt


def _execution_ok(execution):
    return (execution.get("exit_code") == 0 and not execution.get("external_timeout")
            and not execution.get("memory_limit_exceeded") and not execution.get("error"))


def _run_case(output, cpu_output, case, plan):
    directory = output / "cases" / case["key"]
    task = directory / "task"
    shutil.copytree(under(cpu_output, cpu_output / case["task_template"]), task)
    require(_input_hashes(task) == case["input_hashes"], "GPU split differs from CPU prepared inputs")
    config = {"version": VERSION, "run_id": uuid4().hex, "task_dir": str(task), "seed": case["seed"],
              "input_hashes": case["input_hashes"], "source_sha256": plan["source_sha256"], "parameters": FIXED_PARAMETERS}
    config_path = directory / "config.json"
    write_json(config_path, config)
    started = time.monotonic()
    deadline = started + plan["time_budget_seconds"]
    argv = [plan["python"], "-I", str(Path(__file__).resolve()), "--worker", str(config_path), "--deadline", str(deadline)]
    row = {"case": case["key"], "label": "exploratory_gpu", "status": "started", "run_id": config["run_id"],
           "started_at_utc": utc_now(), "started_monotonic": started, "deadline": deadline, "argv": argv,
           "config_sha256": file_hash(config_path), "seed": case["seed"], "metric": None, "cpu_qualification": False}
    write_json(directory / "state.json", row)
    try:
        execution = _supervise(argv, task, directory / "execution", deadline - time.monotonic())
        row["execution"] = execution
        if not _execution_ok(execution):
            row.update(status="failed", reason="gpu_worker_failed_or_capped")
        else:
            row.update(_audit_case(directory, cpu_output, case, plan))
        if execution.get("cleanup", {}).get("complete") is not True:
            row.update(status="censored", reason="gpu_cleanup_unverified", stop_series=True)
        elif row["status"] == "completed" and execution.get("resources", {}).get("available") is not True:
            row["status"] = "provisional_resource_observation"
    except (Exception, KeyboardInterrupt) as error:
        row.update(status="censored", reason="gpu_supervisor_or_evidence_failed", error=type(error).__name__ + ": " + str(error)[-2000:], stop_series=True)
    row["ended_at_utc"] = utc_now()
    row["artifact_hashes"] = file_index(directory, exclude=("state.json",))
    write_json(directory / "state.json", row)
    return row


def _audit_case(directory, cpu_output, case, plan):
    from qualify_unseen_tasks import audit_submission
    config_path = directory / "config.json"
    config, receipt = read_json(config_path), read_json(directory / "receipt.json")
    state = read_json(directory / "state.json")
    require(receipt.get("version") == VERSION and receipt.get("run_id") == config["run_id"]
            and receipt.get("config_sha256") == file_hash(config_path) and receipt.get("source_sha256") == plan["source_sha256"]
            and receipt.get("input_hashes") == case["input_hashes"] == _input_hashes(directory / "task")
            and receipt.get("output_sha256") == file_hash(directory / "task/submission.csv")
            and receipt.get("api_attempts") == 0 and receipt.get("label") == "exploratory_gpu", "GPU receipt identity differs")
    audit_path = under(cpu_output, cpu_output / case["audit_path"])
    require(file_hash(audit_path) == case["audit_sha256"], "GPU external audit changed")
    audit = read_json(audit_path)
    classes = audit["contract"]["classes"]
    provided = receipt.get("prediction_classes")
    require((isinstance(provided, list) and len(provided) == len(classes) and set(provided) == set(classes))
            if classes else provided is None, "GPU receipt prediction class order is invalid")
    loss = ("Logloss" if len(classes) == 2 else "MultiClass") if classes else "RMSE"
    require(config["parameters"] == plan["parameters"] == FIXED_PARAMETERS and receipt["parameters"] == {
        **FIXED_PARAMETERS, "random_seed": case["seed"], "loss_function": loss}, "GPU parameters drift")
    require(state["run_id"] == config["run_id"] and state["config_sha256"] == file_hash(config_path)
            and receipt["deadline"] == state["deadline"] == state["started_monotonic"] + plan["time_budget_seconds"]
            and state["started_monotonic"] <= receipt["native_started"] <= receipt["native_ended"] < state["deadline"],
            "GPU native deadline or state binding differs")
    result = audit_submission(directory / "task/submission.csv", audit)
    require(result.get("valid") is True, "GPU independent CSV audit failed: " + str(result.get("error")))
    return {"status": "completed", "metric": result["metric"], "audit": result,
            "receipt_sha256": file_hash(directory / "receipt.json"), "output_sha256": receipt["output_sha256"]}


def run_gpu(output, cpu_plan, time_budget_seconds=120, enabled=False):
    """Run fixed GPU exploration only after an eight-arm CPU development pilot.

    ``cpu_plan`` is the CPU output directory, not an unverified plan dictionary.
    Disabled/unavailable/blocked cases remain explicit rows in the denominator.
    """
    require(type(enabled) is bool, "GPU opt-in must be a Boolean")
    require(type(time_budget_seconds) is int and 1 <= time_budget_seconds <= 600, "GPU budget must be 1..600 seconds per case")
    output = safe_path(output)
    require(not output.exists(), "GPU evidence requires a fresh directory; no automatic retries")
    output.mkdir(parents=True)
    result = {"version": VERSION, "label": "exploratory_gpu", "enabled": enabled, "cpu_qualification": False,
              "source_sha256": file_hash(__file__),
              "status": "skipped", "reason": "explicit_opt_in_required", "cases": [], "api_attempts": 0,
              "threads": THREADS, "memory_limit_bytes": MEMORY_BYTES, "time_budget_seconds": time_budget_seconds,
              "limitation": "GPU baseline evidence does not qualify a CPU contest policy; sampled RSS is not an OS allocation cap. GPU reductions may be nondeterministic."}
    if not enabled:
        write_json(output / "gpu_report.json", result)
        write_json(output / "gpu_report_seal.json", {"report_sha256": file_hash(output / "gpu_report.json")})
        return result
    from research_lab.planning import load_plan
    from research_lab.process import coordinator_lock, machine_lock_path
    cpu_output = safe_path(cpu_plan)
    with coordinator_lock(output / ".coordinator.lock"), coordinator_lock(machine_lock_path()):
        cpu = load_plan(cpu_output, execution=True)
        gate = _pilot_gate(cpu_output, cpu)
        result["cpu_pilot"] = gate
        probe_path = output / "capability.json"
        code = "import sys;sys.path.insert(0," + repr(str(ROOT)) + ");from research_lab.gpu import probe_worker;probe_worker(" + repr(str(probe_path)) + ")"
        probe_execution = _supervise([cpu["python"], "-I", "-c", code], output, output / "probe_execution", PROBE_SECONDS)
        write_json(output / "probe_execution.json", probe_execution)
        capability = read_json(probe_path) if probe_path.is_file() else {"available": False, "reason": "capability_probe_failed"}
        result.update(capability=capability, capability_sha256=file_hash(probe_path) if probe_path.is_file() else None)
        reason = None if gate["allowed"] else gate["reason"]
        if not _execution_ok(probe_execution): reason = "capability_probe_failed"
        if not capability.get("available"): reason = reason or capability.get("reason", "gpu_unavailable")
        if probe_execution.get("cleanup", {}).get("complete") is not True and capability.get("available"):
            reason = "capability_probe_cleanup_unverified"
        plan = {"version": VERSION, "label": "exploratory_gpu", "cpu_output": str(cpu_output),
                "cpu_plan_sha256": file_hash(cpu_output / "plan.json"), "cpu_plan_id": cpu["plan_id"],
                "source_sha256": file_hash(__file__), "capability_sha256": result["capability_sha256"],
                "probe_execution_sha256": file_hash(output / "probe_execution.json"),
                "python": cpu["python"], "environment": cpu["environment"], "parameters": FIXED_PARAMETERS,
                "time_budget_seconds": time_budget_seconds, "threads": THREADS, "memory_limit_bytes": MEMORY_BYTES,
                "cpu_pilot": gate, "cases": cpu["cases"], "created_at_utc": utc_now()}
        write_json(output / "gpu_plan.json", plan)
        write_json(output / "gpu_seal.json", {"plan_sha256": file_hash(output / "gpu_plan.json")})
        stopped = reason
        for case in cpu["cases"]:
            load_plan(cpu_output, execution=True)
            for name, digest in gate.get("state_sha256", {}).items():
                require(file_hash(under(cpu_output, cpu_output / name)) == digest, "CPU pilot state changed during GPU exploration")
            if stopped:
                row = {"case": case["key"], "status": "skipped", "reason": stopped, "label": "exploratory_gpu", "metric": None}
            else:
                row = _run_case(output, cpu_output, case, plan)
                if row.get("stop_series"): stopped = row["reason"]
            result["cases"].append(row)
            write_json(output / "gpu_report.json", result)
        result.update(status="skipped" if reason else "censored" if stopped else "completed", reason=stopped,
                      gpu_plan_sha256=file_hash(output / "gpu_plan.json"))
        write_json(output / "gpu_report.json", result)
        write_json(output / "gpu_report_seal.json", {"report_sha256": file_hash(output / "gpu_report.json")})
        return result


def re_audit_gpu(output):
    """Recompute saved GPU scores without fitting, querying hardware or repair."""
    output = safe_path(output)
    report = read_json(output / "gpu_report.json")
    require(file_hash(output / "gpu_report.json") == read_json(output / "gpu_report_seal.json")["report_sha256"], "GPU report drift")
    require(report["source_sha256"] == file_hash(__file__) and report["cpu_qualification"] is False, "GPU report source/scope drift")
    if report["enabled"] is False:
        require(report["status"] == "skipped" and not report["cases"] and report["reason"] == "explicit_opt_in_required", "Disabled GPU lane claimed an outcome")
        return {"version": VERSION, "status": "skipped", "verified": True, "label": "exploratory_gpu",
                "cpu_qualification": False, "fits": 0, "cases": 0, "report_sha256": file_hash(output / "gpu_report.json")}
    from research_lab.planning import load_plan
    plan, seal = (read_json(output / name) for name in ("gpu_plan.json", "gpu_seal.json"))
    require(file_hash(output / "gpu_plan.json") == seal["plan_sha256"] == report["gpu_plan_sha256"], "GPU plan seal drift")
    require(plan["source_sha256"] == file_hash(__file__) and plan["parameters"] == FIXED_PARAMETERS, "GPU source/config drift")
    require(file_hash(output / "probe_execution.json") == plan["probe_execution_sha256"], "GPU probe execution drift")
    cpu_output = safe_path(plan["cpu_output"])
    cpu = load_plan(cpu_output, execution=False)
    require(plan["cpu_plan_sha256"] == file_hash(cpu_output / "plan.json") and plan["cpu_plan_id"] == cpu["plan_id"]
            and plan["cases"] == cpu["cases"], "GPU/CPU prepared plan differs")
    require(len(report["cases"]) == len(plan["cases"]), "GPU cases are missing")
    if plan["capability_sha256"] is not None:
        require(file_hash(output / "capability.json") == plan["capability_sha256"] == report["capability_sha256"], "GPU capability drift")
    for name, digest in plan["cpu_pilot"].get("state_sha256", {}).items():
        require(file_hash(under(cpu_output, cpu_output / name)) == digest, "CPU pilot terminal state drift")
    for case, row in zip(plan["cases"], report["cases"]):
        require(row["case"] == case["key"], "GPU case order/identity differs")
        if row["status"] == "skipped":
            require(row.get("metric") is None and bool(row.get("reason")), "GPU skip claimed a score")
            continue
        directory = under(output, output / "cases" / case["key"])
        require(read_json(directory / "state.json") == row, "GPU report/state differ")
        require(file_index(directory, exclude=("state.json",)) == row["artifact_hashes"], "GPU artifact drift")
        if row["status"] in {"completed", "provisional_resource_observation"}:
            require(plan["cpu_pilot"]["allowed"] is True and report["capability"].get("available") is True,
                    "GPU fit lacked pilot/capability authorization")
            require(_execution_ok(row["execution"]) and row["execution"].get("cleanup", {}).get("complete") is True,
                    "GPU score lacks successful execution and cleanup")
            fresh = _audit_case(directory, cpu_output, case, plan)
            require(fresh["audit"] == row["audit"] and fresh["metric"] == row["metric"], "GPU CSV score differs from saved audit")
        else:
            require(row["status"] in {"failed", "censored"}, "Unknown or interrupted GPU outcome")
    return {"version": VERSION, "status": report["status"], "verified": True, "label": "exploratory_gpu",
            "cpu_qualification": False, "fits": 0, "cases": len(plan["cases"]), "report_sha256": file_hash(output / "gpu_report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--deadline", type=float)
    arguments = parser.parse_args()
    require(arguments.worker and arguments.deadline is not None, "Use the laboratory's explicit gpu command")
    worker(arguments.worker, arguments.deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
