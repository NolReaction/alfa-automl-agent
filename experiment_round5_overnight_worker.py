"""Observe one isolated native CLI run for the serial overnight experiment.

This development driver imports no coordinator or audit code. The parent owns
the machine lock, resource limits, immutable inputs, and independent audits.
"""
import time

_NATIVE_STARTED = time.monotonic()

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


_FIELDS = {"version", "job_id", "task_dir", "task_path", "solver_root", "worker_output",
           "time_budget_seconds", "profile"}
_COMMON_ARGS = ["--mode", "offline", "--research-search", "adaptive", "--search-profile", "legacy"]
_PROFILE_ARGS = {
    "regression_off": ["--no-rescue-regression"],
    "regression_on": ["--rescue-regression"],
    "classification": ["--no-rescue-regression"],
}


def _plain(value):
    path = Path(value).absolute()
    if any(part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction())
           for part in (path, *path.parents)):
        raise ValueError("Linked worker paths are unsupported")
    return path.resolve()


def _digest(path):
    digest = hashlib.sha256()
    with _plain(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = _plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".overnight-worker-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _load_config(path):
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Missing/oversized worker configuration")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate worker configuration key")
            result[key] = value
        return result
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if (not isinstance(config, dict) or set(config) != _FIELDS
            or type(config["version"]) is not int or config["version"] != 1):
        raise ValueError("Worker requires the exact version-1 configuration fields")
    if (not isinstance(config["job_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", config["job_id"])):
        raise ValueError("Invalid worker job identity")
    if not isinstance(config["profile"], str) or config["profile"] not in _PROFILE_ARGS:
        raise ValueError("Unknown overnight profile")
    cap = config["time_budget_seconds"]
    if type(cap) not in {int, float} or not math.isfinite(cap) or not 0 < cap <= 10500:
        raise ValueError("Invalid original budget")
    return config


def execute(config_path, deadline):
    """Call the packaged main once; observers retain original returned objects."""
    config_path = _plain(config_path)
    config = _load_config(config_path)
    cap = config["time_budget_seconds"]
    if (type(deadline) not in {int, float} or not math.isfinite(deadline)
            or deadline > _NATIVE_STARTED + cap + 1):
        raise ValueError("Invalid original monotonic deadline")
    task_dir = _plain(config["task_dir"])
    task = _plain(config["task_path"])
    root = _plain(config["solver_root"])
    output = _plain(config["worker_output"])
    pointer = task_dir / "solver_output" / "research_benchmark.json"
    if (task != task_dir / "task.md" or output.parent != task_dir.parent
            or output.name != "worker_evidence.json"):
        raise ValueError("Worker paths must remain in the current job")
    if (not task.is_file() or not (root / "solution.py").is_file()
            or output.exists() or pointer.exists() or (task_dir / ".agent_runs").exists()):
        raise ValueError("Worker requires a fresh task, evidence, and existing packaged solver")
    evidence = {
        "version": 1, "job_id": config["job_id"], "profile": config["profile"],
        "config_sha256": _digest(config_path), "native_started_monotonic": _NATIVE_STARTED,
        "native_ended_monotonic": None, "original_deadline_monotonic": deadline,
        "original_time_budget_seconds": cap, "cli_time_budget_seconds": None,
        "cli_argv": None, "cli_exit_code": None, "solver_returned": False,
        "run_report": None, "terminal_error": None, "observations": [],
    }
    selected = None
    observed_state = None

    def persist():
        evidence["worker_elapsed_seconds"] = time.monotonic() - _NATIVE_STARTED
        _write(output, evidence)

    def observation_error(phase, error):
        evidence["observations"].append({"phase": phase,
            "error": type(error).__name__ + ": " + str(error)[-2000:]})

    def current_report(path, origin):
        path = _plain(path)
        if not path.is_relative_to(task_dir) or not path.is_file():
            raise ValueError("Returned report is absent or outside the current task")
        relative = path.relative_to(task_dir).as_posix()
        if not re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative):
            raise ValueError("Research report is not an exact current-run path")
        return {"path": relative, "sha256": _digest(path), "origin": origin, "kind": "research_v3"}

    def observe_state(state):
        nonlocal observed_state
        observed_state = state
        gate = state.gate
        evidence["observations"].append({
            "phase": "prepare_research", "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
            "run_dir": _plain(state.run_dir).relative_to(task_dir).as_posix(),
            "frozen_split": _copy_json(gate.frozen_split), "inner_split": _copy_json(gate.inner_split),
        })

    def observe_publication():
        if observed_state is None:
            raise ValueError("No original prepared research state was observed")
        state = observed_state
        name = state.report.get("published_recipe")
        candidate = getattr(state, "baseline" if name == "baseline" else "selected_candidate", None)
        if name not in {"baseline", "candidate"} or candidate is None:
            raise ValueError("No original frozen published recipe was observed")
        code, recipe = _plain(candidate.code_path), _plain(candidate.recipe_path)
        if not code.is_relative_to(task_dir) or not recipe.is_relative_to(task_dir):
            raise ValueError("Original frozen candidate escapes the current task")
        evidence["observations"].append({
            "phase": "frozen_publication", "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
            "run_dir": _plain(state.run_dir).relative_to(task_dir).as_posix(), "published_recipe": name,
            "frozen_source": {"code_path": code.relative_to(task_dir).as_posix(),
                "recipe_path": recipe.relative_to(task_dir).as_posix(), "code_hash": _digest(code),
                "recipe_hash": _digest(recipe), "declared_code_hash": candidate.code_hash,
                "declared_recipe_hash": candidate.recipe_hash},
        })

    persist()
    try:
        if not sys.flags.isolated:
            raise ValueError("Launch this development worker with Python -I")
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="backslashreplace")
        os.chdir(task_dir)
        # Preserve stdlib/site-packages, including nested venv dependencies.
        # Only the coordinator checkout's import entry is excluded.
        auditor_root = Path(__file__).resolve().parent
        sys.path[:] = [value for value in sys.path if value and Path(value).resolve() != auditor_root]
        sys.path.insert(0, str(root))
        module_spec = importlib.util.spec_from_file_location("solution", root / "solution.py")
        solution = importlib.util.module_from_spec(module_spec)
        sys.modules["solution"] = solution
        module_spec.loader.exec_module(solution)
        import agent.research_v3 as research

        original_prepare = research.prepare_research
        @functools.wraps(original_prepare)
        def prepare(*args, **kwargs):
            nonlocal observed_state
            observed_state = args[0] if args else kwargs["state"]
            try:
                return original_prepare(*args, **kwargs)
            finally:
                try:
                    observe_state(observed_state)
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    observation_error("prepare_observer_error", error)
                persist()
        research.prepare_research = prepare

        original_agent = solution.run_agent
        @functools.wraps(original_agent)
        def run_agent(*args, **kwargs):
            nonlocal selected
            evidence["observations"].append({
                "phase": "run_agent", "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
                "time_budget_seconds": kwargs.get("time_budget"), "local_only": kwargs.get("local_only"),
                "adaptive_search": kwargs.get("adaptive_search"),
                "rescue_regression": kwargs.get("rescue_regression"),
            })
            persist()
            result = original_agent(*args, **kwargs)
            evidence["solver_returned"] = True
            try:
                selected = current_report(Path(result) / "run.json", "returned")
                observe_publication()
            except (OSError, ValueError, TypeError, AttributeError) as error:
                observation_error("frozen_observer_error", error)
            persist()
            return result
        solution.run_agent = run_agent

        for name, module in tuple(sys.modules.items()):
            if name == "agent" or name.startswith("agent."):
                location = getattr(module, "__file__", None)
                if location is None or not _plain(location).is_relative_to(root):
                    raise ValueError("Solver agent module imported outside the packaged root")
        remaining = deadline - time.monotonic() - 2.0
        if remaining <= 0:
            raise TimeoutError("Original deadline exhausted during bootstrap/imports")
        evidence["cli_time_budget_seconds"] = remaining
        sys.argv = [str(root / "solution.py"), str(task), *_COMMON_ARGS,
                    *_PROFILE_ARGS[config["profile"]], "--time-budget", str(remaining)]
        evidence["cli_argv"] = list(sys.argv)
        persist()
        code = solution.main()
        evidence["cli_exit_code"] = int(code or 0)
        if not evidence["solver_returned"] or selected is None:
            raise ValueError("Native main did not return an exact research-v3 report")
        if any(item["phase"].endswith("observer_error") for item in evidence["observations"]):
            raise ValueError("Native observation is incomplete; independent audit is required")
    except BaseException as error:
        evidence["terminal_error"] = {"type": type(error).__name__, "message": str(error)[-4000:]}
        evidence["cli_exit_code"] = 1
    finally:
        # A failure-state report is diagnostic evidence, never a returned-success proof.
        if selected is None and observed_state is not None:
            try:
                selected = current_report(Path(observed_state.run_dir) / "run.json", "observed_state_after_failure")
            except (OSError, ValueError, TypeError, AttributeError) as error:
                observation_error("report_unavailable", error)
        evidence["run_report"] = selected
        if selected is not None:
            try:
                if pointer.exists():
                    raise ValueError("Current-run pointer already exists")
                _write(pointer, {"run_report": selected["path"], "run_report_sha256": selected["sha256"]})
            except (OSError, ValueError, TypeError) as error:
                evidence["terminal_error"] = {"type": type(error).__name__, "message": str(error)[-4000:]}
                evidence["cli_exit_code"] = 1
        evidence["native_ended_monotonic"] = time.monotonic()
        persist()
    return evidence["cli_exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    return execute(args.config, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
