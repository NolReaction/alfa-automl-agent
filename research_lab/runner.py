"""Serial, restart-safe coordination of sealed external benchmark arms.

Candidate modules only execute in worker children. Preparation, source checks,
the process supervisor and the external auditor remain from this lab checkout.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

from .integrity import (check_identity, file_hash, read_json, require, safe_path,
                        under, utc_now, write_json)
from .process import coordinator_lock, machine_lock_path, supervise


_TERMINAL = {"finished", "failed", "censored", "interrupted", "completed"}


def _path(output, value):
    value = Path(value)
    return under(output, value if value.is_absolute() else output / value)


def _seal_state(path, state):
    """The state cannot include its own digest; store an adjacent atomic seal."""
    write_json(path, state)
    marker = path.with_suffix(".sha256")
    with tempfile.NamedTemporaryFile(dir=marker.parent, prefix=".state-seal-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((file_hash(path) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _check_seal(path):
    marker = safe_path(path.with_suffix(".sha256"))
    require(marker.is_file() and marker.stat().st_size == 65
            and marker.read_text(encoding="ascii").strip() == file_hash(path),
            "Terminal state seal is missing or changed")


def _missing_worker(config, config_path):
    # Evidence of a launch failure/interruption, never a model-quality receipt.
    return {"version": 1, "run_id": config["run_id"], "case_key": config["case_key"],
            "arm": config["arm"], "config_sha256": file_hash(config_path),
            "original_time_budget_seconds": config["time_budget_seconds"],
            "cli_time_budget_seconds": None, "cli_exit_code": None, "solver_returned": False,
            "run_report": None, "observations": [],
            "terminal_error": {"type": "MissingWorkerEvidence", "message": "Child evidence is absent"}}


def _worker_evidence(output, config, config_path):
    path = _path(output, config["worker_output"])
    return path if path.is_file() else _missing_worker(config, config_path)


def _audit_and_seal(output, plan, case, arm, state, state_path, config, config_path):
    from .audit import audit_arm

    write_json(state_path, state)
    result = audit_arm(output, case, arm, _worker_evidence(output, config, config_path), plan=plan)
    state["audit"] = result
    state["artifact_hashes"] = result["artifact_hashes"]
    state["finished_at_utc"] = utc_now()
    resource = result.get("resources") or {}
    if result.get("status") == "censored" or resource.get("qualification") == "censored":
        state["status"], state["outcome"] = "censored", "censored"
    elif result.get("quality_verified") is True:
        state["outcome"] = ("verified" if resource.get("qualification") == "observed"
                            else "provisional_resource_observation")
    else:
        state["outcome"] = result.get("status", "unverified")
    _seal_state(state_path, state)
    return state


def _report(output, plan, stop_reason=None):
    rows = []
    for case in plan["cases"]:
        for arm in case["arm_order"]:
            entry = case["arms"][arm]
            state = read_json(_path(output, entry["state_path"]))
            rows.append({"case": case["key"], "arm": arm, "run_id": state["run_id"],
                         "status": "not_started" if state["status"] == "prepared" else state["status"],
                         "outcome": state.get("outcome"), "audit": state.get("audit")})
    counts = {key: sum(row["status"] == key for row in rows)
              for key in sorted({row["status"] for row in rows})}
    result = {"version": 1, "status": "stopped" if stop_reason else "completed",
              "stop_reason": stop_reason, "arms": rows, "counts": counts,
              "planned_arms": len(rows), "quality_admission": False}
    write_json(output / "run_report.json", result)
    return result


def _resume_state(output, plan, case, arm, entry):
    state_path, config_path = (_path(output, entry[name]) for name in ("state_path", "config_path"))
    state, config = read_json(state_path), read_json(config_path)
    require(state.get("run_id") == entry["run_id"] == config.get("run_id"), "Arm run identity differs")
    require(state.get("status") in _TERMINAL | {"prepared", "started"}, "Unknown arm state")
    if state["status"] == "started":
        # The coordinator died after durable ownership transfer. There is no
        # fresh budget and no safe inference that its descendants are gone.
        state.update(status="interrupted", outcome="censored", reason="interrupted_started_arm")
        _audit_and_seal(output, plan, case, arm, state, state_path, config, config_path)
        return "Interrupted arm has no proven process-tree cleanup; remaining arms were not started"
    if state["status"] in _TERMINAL:
        _check_seal(state_path)
        audit = state.get("audit") or {}
        if state["status"] in {"censored", "interrupted"} or (audit.get("resources") or {}).get("qualification") == "censored":
            return "A prior arm is censored; remaining arms were not started"
    return None


def _run_arm(output, plan, case, arm, entry):
    state_path, config_path = (_path(output, entry[name]) for name in ("state_path", "config_path"))
    state, config = read_json(state_path), read_json(config_path)
    require(state.get("status") == "prepared", "Only untouched prepared arms may start")
    require(state.get("run_id") == entry["run_id"] == config.get("run_id"), "Arm run identity differs")
    task_dir = _path(output, entry["task_dir"])
    require(config.get("task_dir") == str(task_dir), "Config task root differs")
    worker_output = _path(output, config["worker_output"])
    execution_dir = config_path.parent / "execution"
    require(not worker_output.exists() and not execution_dir.exists()
            and not (task_dir / "submission.csv").exists() and not (task_dir / ".agent_runs").exists(),
            "Prepared arm contains previous execution artifacts")
    identity = config.get("solver_identity") or plan["solvers"][arm]
    check_identity(identity)
    cap = plan["time_budget_seconds"]
    require(cap == config["time_budget_seconds"], "Arm budget differs from sealed plan")
    require(plan["threads"] == config["threads"] and plan["memory_limit_bytes"] == config["memory_limit_bytes"],
            "Arm resource lane differs from sealed plan")
    started = time.monotonic()
    deadline = started + cap
    argv = [str(plan["python"]), "-I", str(Path(__file__).with_name("worker.py")), str(config_path),
            "--deadline", str(deadline)]
    state.update(status="started", started_at_utc=utc_now(), started_monotonic=started,
                 original_deadline_monotonic=deadline, argv=argv, config_sha256=file_hash(config_path))
    # This replace must complete BEFORE Popen. An interruption from this point
    # onward consumes this arm permanently.
    write_json(state_path, state)
    try:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Original deadline expired before child launch")
        execution = supervise(argv, task_dir, execution_dir, remaining, plan["memory_limit_bytes"],
                              threads=plan["threads"])
    except (Exception, KeyboardInterrupt) as error:
        execution = {"argv": argv, "exit_code": None, "external_timeout": isinstance(error, TimeoutError),
                     "memory_limit_exceeded": False, "error_type": type(error).__name__,
                     "error": str(error)[-2000:], "elapsed_seconds": time.monotonic() - started,
                     "cleanup_elapsed_seconds": None, "worker_wall_seconds": time.monotonic() - started,
                     "cleanup": {"complete": None, "inventory_available": False,
                                 "unavailable_reason": "Supervisor did not return cleanup evidence"},
                     "resources": {"available": False, "peak_rss_bytes": None, "cpu_seconds": None}}
    elapsed = time.monotonic() - started
    execution["inclusive_elapsed_seconds"] = elapsed
    cleanup = execution.get("cleanup_elapsed_seconds")
    execution["worker_wall_seconds"] = elapsed - cleanup if type(cleanup) in {int, float} else elapsed
    state["execution"] = execution
    state["status"] = "failed" if (execution.get("exit_code") != 0 or execution.get("external_timeout")
                                   or execution.get("memory_limit_exceeded") or execution.get("error_type")) else "finished"
    # A changed source cannot acquire quality status even if it produced a CSV.
    try:
        check_identity(identity)
    except (ValueError, OSError) as error:
        state["execution"].update(error_type=type(error).__name__, error=str(error)[-2000:])
        state["status"] = "failed"
    return _audit_and_seal(output, plan, case, arm, state, state_path, config, config_path)


def run_plan(output, resume=False):
    """Execute each originally planned arm at most once, in the sealed order."""
    from .planning import load_plan

    output = safe_path(output)
    require(output.is_dir(), "Prepare a sealed plan before running it")
    with coordinator_lock(output / ".coordinator.lock"), coordinator_lock(machine_lock_path()):
        plan = load_plan(output, execution=True)
        states = [read_json(_path(output, entry["state_path"]))
                  for case in plan["cases"] for entry in case["arms"].values()]
        require(resume or all(state.get("status") == "prepared" for state in states),
                "This run has already started; use resume to inspect and continue untouched arms")
        for case in plan["cases"]:
            for arm in case["arm_order"]:
                reason = _resume_state(output, plan, case, arm, case["arms"][arm])
                if reason:
                    return _report(output, plan, reason)
        try:
            for case in plan["cases"]:
                for arm in case["arm_order"]:
                    entry = case["arms"][arm]
                    if read_json(_path(output, entry["state_path"]))["status"] != "prepared":
                        continue
                    # Recheck the full sealed lane between arms. A previous child
                    # must not silently change another arm's inputs/config/cache.
                    load_plan(output, execution=True)
                    state = _run_arm(output, plan, case, arm, entry)
                    if state["status"] == "censored":
                        return _report(output, plan, "Process-tree cleanup is unknown after a failed arm")
        except Exception as error:
            # The complete planned denominator remains inspectable on drift or
            # another coordinator failure, while the CLI still exits in error.
            _report(output, plan, type(error).__name__ + ": " + str(error)[-1000:])
            raise
        return _report(output, plan)
