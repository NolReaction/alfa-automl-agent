"""One isolated full-solver invocation. This file deliberately imports no lab code.

The supervisor supplies a deadline started before Popen.  Keep the native clock
above every solver import: importing solution itself can take meaningful time.
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


_FIELDS = {"version", "run_id", "case_key", "arm", "task_dir", "task_path", "solver_root",
           "solver_identity", "source_fingerprint", "runtime_code_hash", "time_budget_seconds",
           "threads", "memory_limit_bytes", "require_adaptive", "worker_output", "solver_args"}

_COMMON_ARGS = ["--mode", "offline", "--research-search", "adaptive", "--search-profile", "legacy"]
_CANDIDATE_ARGS = _COMMON_ARGS + ["--finalist-selection", "--stability-splits", "2"]


def _plain(value):
    path = Path(value).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Linked worker path")
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
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".worker-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def execute(config_path, deadline):
    """Execute main with its real defaults; hooks only observe returned objects."""
    config_path = _plain(config_path)
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError("Missing/oversized worker configuration")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) - _FIELDS:
        raise ValueError("Unexpected worker fields (oracle data must remain outside the child)")
    task_dir = _plain(config["task_dir"])
    task = _plain(config.get("task_path", task_dir / "task.md"))
    root = _plain(config["solver_root"])
    output = _plain(config.get("worker_output", task_dir.parent / "worker_evidence.json"))
    if task.parent != task_dir or not output.is_relative_to(task_dir.parent):
        raise ValueError("Worker paths escape their arm")
    if not task.is_file() or not (root / "solution.py").is_file() or output.exists():
        raise ValueError("Worker requires an untouched arm and an existing solver/task")
    solver_args = config.get("solver_args", _COMMON_ARGS)
    if solver_args not in (_COMMON_ARGS, _CANDIDATE_ARGS):
        raise ValueError("Solver arguments differ from the frozen laboratory CLI")
    cap = config["time_budget_seconds"]
    if type(cap) not in {int, float} or not math.isfinite(cap) or not 0 < cap <= 10500:
        raise ValueError("Invalid original budget")
    if not math.isfinite(deadline) or deadline > _NATIVE_STARTED + cap + 1:
        raise ValueError("Invalid original monotonic deadline")
    evidence = {"version": 1, "run_id": config["run_id"], "case_key": config["case_key"],
                "arm": config["arm"], "config_sha256": _digest(config_path),
                "native_started_monotonic": _NATIVE_STARTED, "native_ended_monotonic": None,
                "original_deadline_monotonic": deadline,
                "original_time_budget_seconds": cap, "cli_time_budget_seconds": None,
                "cli_exit_code": None, "solver_returned": False, "run_report": None,
                "terminal_error": None, "observations": [], "emergency": None}
    selected = None
    observed_state = None

    def persist():
        evidence["worker_elapsed_seconds"] = time.monotonic() - _NATIVE_STARTED
        _write(output, evidence)

    def report(path, origin, kind):
        path = _plain(path)
        if not path.is_relative_to(task_dir) or not path.is_file():
            raise ValueError("Returned report is absent or outside the current task")
        relative = path.relative_to(task_dir).as_posix()
        if kind == "research_v3" and not re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative):
            raise ValueError("Research report is not an exact current-run path")
        return {"path": relative, "sha256": _digest(path), "origin": origin, "kind": kind}

    def observe_state(state):
        nonlocal observed_state
        observed_state = state
        item = {"phase": "prepare_research", "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
                "run_dir": _plain(state.run_dir).relative_to(task_dir).as_posix()}
        gate = getattr(state, "gate", None)
        if gate is not None:
            for name in ("frozen_split", "inner_split"):
                if hasattr(gate, name):
                    item[name] = _copy_json(getattr(gate, name))
        evidence["observations"].append(item)
        persist()

    def observe_publication():
        if observed_state is None:
            return
        state = observed_state
        name = getattr(state, "report", {}).get("published_recipe")
        candidate = getattr(state, "baseline" if name == "baseline" else "selected_candidate", None)
        if name not in {"baseline", "candidate"} or candidate is None:
            return
        code, recipe = _plain(candidate.code_path), _plain(candidate.recipe_path)
        if not code.is_relative_to(task_dir) or not recipe.is_relative_to(task_dir):
            raise ValueError("Original frozen candidate escapes current task")
        evidence["observations"].append({"phase": "frozen_publication",
            "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
            "run_dir": _plain(state.run_dir).relative_to(task_dir).as_posix(), "published_recipe": name,
            "frozen_source": {"code_path": code.relative_to(task_dir).as_posix(),
                "recipe_path": recipe.relative_to(task_dir).as_posix(), "code_hash": _digest(code),
                "recipe_hash": _digest(recipe), "declared_code_hash": candidate.code_hash,
                "declared_recipe_hash": candidate.recipe_hash}})

    persist()
    try:
        os.chdir(task_dir)
        # -I excludes ambient PYTHONPATH/cwd injection. Preserve the selected
        # interpreter's libraries, including a venv inside the auditor checkout.
        # Remove only auditor source entries before adding the selected solver.
        trusted_root = Path(__file__).resolve().parents[1]
        auditor_sources = {trusted_root, Path(__file__).resolve().parent}
        sys.path[:] = [value for value in sys.path
                       if value and Path(value).resolve() not in auditor_sources]
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("solution", root / "solution.py")
        solution = importlib.util.module_from_spec(spec)
        sys.modules["solution"] = solution
        spec.loader.exec_module(solution)

        original_agent = getattr(solution, "run_agent", None)
        if original_agent is not None:
            @functools.wraps(original_agent)
            def run_agent(*args, **kwargs):
                nonlocal selected
                evidence["observations"].append({"phase": "run_agent",
                    "elapsed_seconds": time.monotonic() - _NATIVE_STARTED,
                    "time_budget_seconds": kwargs.get("time_budget"),
                    "local_only": kwargs.get("local_only"),
                    "adaptive_search": kwargs.get("adaptive_search")})
                persist()
                result = original_agent(*args, **kwargs)
                selected = report(Path(result) / "run.json", "returned", "research_v3")
                evidence["solver_returned"] = True
                try:
                    observe_publication()
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    evidence["observations"].append({"phase": "frozen_observer_error",
                                                   "error": type(error).__name__ + ": " + str(error)[-1000:]})
                return result
            solution.run_agent = run_agent

        # Importing observer modules also consumes the same original deadline.
        # Optional modules are absent only in tiny non-ML protocol fixtures.
        try:
            import agent.research_v3 as research
        except ModuleNotFoundError as error:
            if error.name not in {"agent", "agent.research_v3"}:
                raise
        else:
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
                        # Preserve the native prepare exception and return value.
                        evidence["observations"].append({"phase": "prepare_observer_error",
                                                       "error": type(error).__name__ + ": " + str(error)[-1000:]})
            research.prepare_research = prepare

        try:
            import agent.automl as automl
        except ModuleNotFoundError as error:
            if error.name not in {"agent", "agent.automl"}:
                raise
        else:
            original_automl = automl.run_automl
            @functools.wraps(original_automl)
            def run_automl(*args, **kwargs):
                nonlocal selected
                result = original_automl(*args, **kwargs)
                # Preserve the exact returned dictionary, including identity.
                selected = report(result["report_path"], "returned", "automl")
                evidence["solver_returned"] = True
                return result
            automl.run_automl = run_automl

        for name, module in tuple(sys.modules.items()):
            if name == "agent" or name.startswith("agent."):
                location = getattr(module, "__file__", None)
                if location is None or not _plain(location).is_relative_to(root):
                    raise ValueError("Solver agent module was imported outside the selected root")

        original_emergency = getattr(solution, "_prepare_emergency", None)
        if original_emergency is not None:
            @functools.wraps(original_emergency)
            def prepare_emergency(*args, **kwargs):
                result = original_emergency(*args, **kwargs)
                if result is not None and getattr(result, "report_path", None):
                    path = _plain(result.report_path)
                    if path.is_file() and path.is_relative_to(task_dir):
                        evidence["emergency"] = {"report_path": path.relative_to(task_dir).as_posix(),
                                                 "report_sha256": _digest(path),
                                                 "quality_verified": False}
                return result
            solution._prepare_emergency = prepare_emergency

        remaining = deadline - time.monotonic() - 2.0
        if remaining <= 0:
            raise TimeoutError("Original deadline exhausted during bootstrap/imports")
        if config.get("require_adaptive", False) and remaining <= 300:
            raise TimeoutError("Bootstrap left <=300 seconds; full adaptive route cannot be tested")
        evidence["cli_time_budget_seconds"] = remaining
        sys.argv = [str(root / "solution.py"), str(task), *solver_args, "--time-budget", str(remaining)]
        evidence["cli_argv"] = list(sys.argv)
        persist()
        code = solution.main()
        evidence["cli_exit_code"] = int(code or 0)
    except BaseException as error:
        evidence["terminal_error"] = {"type": type(error).__name__, "message": str(error)[-4000:]}
        evidence["cli_exit_code"] = 1
    finally:
        if selected is None and observed_state is not None:
            try:
                selected = report(Path(observed_state.run_dir) / "run.json",
                                  "observed_state_after_failure", "research_v3")
            except (OSError, ValueError) as error:
                evidence["observations"].append({"phase": "report_unavailable", "error": str(error)[-2000:]})
        evidence["run_report"] = selected
        if selected is not None and selected["kind"] == "research_v3":
            pointer = task_dir / "solver_output" / "research_benchmark.json"
            if pointer.exists():
                evidence["terminal_error"] = {"type": "ValueError", "message": "Current-run pointer already exists"}
                evidence["cli_exit_code"] = 1
            else:
                _write(pointer, {"run_report": selected["path"], "run_report_sha256": selected["sha256"]})
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
