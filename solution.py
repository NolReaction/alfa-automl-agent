"""Run the competition agent, or explicitly select offline AutoML."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from agent.execution import child_environment, run_python, stop_process_tree, tail
from agent.llm import (
    API_FIELDS, REQUIRED_API_FIELDS, ROOT, SYSTEM_PROMPT, TOOL, check_api, get_model, load_local_env,
    make_client, require_api_settings, safe_error_message,
)
from agent.research import run_agent
from validation import validate_predictions


REQUIRED_PACKAGES = ("openai", "dotenv", "numpy", "pandas", "sklearn", "catboost",
                     "lightgbm", "optuna", "psutil")
DEFAULT_TIME_BUDGET = 10500


def select_mode(requested: str) -> str:
    """Auto uses the judge's environment, never a developer's local .env file."""
    if requested != "auto":
        return requested
    present = [name for name in API_FIELDS if os.environ.get(name, "").strip()]
    if not present:
        print("No API settings supplied: offline fallback (budgeted AutoML; no LLM research).", flush=True)
        return "offline"
    missing = [name for name in REQUIRED_API_FIELDS if name not in present]
    if missing:
        print("Incomplete competition API environment; missing: " + ", ".join(missing)
              + ". Starting offline fallback without API calls.", flush=True)
        return "offline"
    print("Competition API environment supplied: starting LLM research.", flush=True)
    return "llm"


def check_setup() -> bool:
    python_ok = sys.version_info[:2] == (3, 12)
    packages = {name: importlib.util.find_spec(name) is not None
                for name in REQUIRED_PACKAGES}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        result = run_python(
            "import numpy, pandas, sklearn, openai, catboost, lightgbm, optuna, psutil; print('python-child-ok')",
            folder, folder / "runs", 1, 30,
        )
    worker_ok = result["exit_code"] == 0 and "python-child-ok" in result["stdout"]
    report = {
        "python": sys.version.split()[0], "python_3_12": python_ok,
        "packages": packages, "python_child": worker_ok,
        "python_child_error": result["stderr"] if not worker_ok else "",
        "api_settings_present": {name: bool(os.environ.get(name, "").strip())
                                 for name in API_FIELDS},
        "api_required_settings": list(REQUIRED_API_FIELDS),
        "api_model": get_model(),
        "api_called": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return python_ok and all(packages.values()) and worker_ok


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("The original solution time budget has been exhausted")
    return remaining


def _offline_solution(task_path, *, deadline, validation, quality_options):
    if (quality_options.get("history_config") is not None or quality_options.get("history_auto")
            or quality_options.get("adaptive_search")):
        # History uses the same v3 INNER/outer/refit protocol as LLM
        # research, with no client construction or paid requests.
        if validation != {"strategy": "auto"}:
            raise ValueError("Host-local research uses v3 auto validation; validation overrides are unavailable.")
        directory = run_agent(task_path, None, time_budget=_remaining(deadline),
                              local_only=True, **quality_options)
        print(f"Host-local research report: {directory / 'run.json'}. API calls: 0.")
        return
    from agent.automl import run_automl

    result = run_automl(task_path, time_budget=_remaining(deadline),
                       validation=validation, **quality_options)
    if result.get("status") == "fallback":
        print("Offline search interrupted; recovered its accepted model checkpoint.", flush=True)
    metric = result["metric"]
    print(f"Best model: {result['model']}; {metric['name']}: {metric['value']:.6f}")
    print(f"Saved: {task_path.resolve().parent / 'submission.csv'}. API calls: 0.")
    print(f"Run report: {result['report_path']}")


def _prepare_emergency(task_path, deadline):
    checkpoint = None
    try:
        from agent.emergency import EmergencyCheckpoint

        checkpoint = EmergencyCheckpoint(task_path)
        if checkpoint.prepare(timeout=min(30.0, _remaining(deadline) * .05), deadline=deadline):
            print("Emergency full-data checkpoint ready; model: target prior; quality unverified.", flush=True)
            print(f"Emergency report: {checkpoint.report_path}", flush=True)
            return checkpoint
        if checkpoint.inputs_frozen:
            checkpoint.check_inputs(deadline=deadline)
        print("Emergency checkpoint unavailable: " + str(checkpoint.metadata.get("error", "preparation incomplete")),
              file=sys.stderr)
    except Exception as error:
        if checkpoint is not None and checkpoint.inputs_frozen:
            checkpoint.check_inputs(deadline=deadline)
        print("Emergency checkpoint unavailable: " + safe_error_message(error), file=sys.stderr)
    return None


def _restore_emergency(checkpoint, error, *, deadline):
    if checkpoint is not None and checkpoint.restore(deadline=deadline):
        print("Normal solver unavailable: " + safe_error_message(error), file=sys.stderr)
        print("Recovered a verified target-prior CSV for the original test rows. "
              "Predictive quality has not been validated.", flush=True)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--mode", choices=("auto", "offline", "llm"), default="auto",
                        help="auto (default): use supplied competition API, otherwise offline fallback; "
                             "offline: budgeted AutoML without API; llm: Qwen research with local .env support")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline environment check; no API calls")
    mode.add_argument("--check-api", action="store_true",
                      help="Check the API with up to two short requests; retry a transient official gateway failure via its reserve")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--time-budget", type=float, default=None,
                        help="Seconds; default 10500 in every mode; maximum 10500")
    parser.add_argument("--search-profile", choices=("legacy", "strong_v1"), default="legacy",
                        help="CPU search plan; strong_v1 is an experimental deeper boosting plan")
    parser.add_argument("--research-search", choices=("adaptive", "legacy"), default="adaptive",
                        help="adaptive (default): use spare long-run CPU time for measured local research; "
                             "legacy keeps the original finite search; budgets <=300s retain short-run behavior")
    parser.add_argument("--validation", choices=("auto", "random", "grouped", "time"), default=None,
                        help="Offline holdout strategy; defaults to auto")
    parser.add_argument("--time-column", help="Numeric period feature for --validation time")
    parser.add_argument("--stability-splits", type=int, choices=(1, 2, 3), default=1,
                        help="Experimental paired inner checks of finalists; default 1 (disabled)")
    parser.add_argument("--metric-recipes", action="store_true",
                        help="Experimentally add metric-aware regression recipes to startup AutoML")
    parser.add_argument("--history-config", help="Experimental inline JSON or @JSON-file key/period configuration; time regression only")
    parser.add_argument("--history-auto", action="store_true",
                        help="Experimentally discover one history schema from INNER-fit features; fixed weight 0.5")
    parser.add_argument("--history-weight", type=float, default=None,
                        help="Fixed history blend weight in [0,1], default 0.5; requires --history-config")
    args = parser.parse_args()
    try:
        if args.check:
            load_local_env()
            return 0 if check_setup() else 1
        if args.check_api:
            load_local_env()
            check_api()
            return 0
        selected_mode = select_mode(args.mode)
        quality_options = {}
        if args.stability_splits != 1:
            quality_options["stability_splits"] = args.stability_splits
        if args.metric_recipes:
            quality_options["metric_recipes"] = True
        if args.search_profile != "legacy":
            quality_options["search_profile"] = args.search_profile
        if args.history_auto:
            if args.history_config is not None or args.history_weight is not None:
                raise ValueError("--history-auto cannot be combined with --history-config or --history-weight")
            if args.validation is not None or args.time_column is not None:
                raise ValueError("Automatic history uses v3 auto validation; omit --validation and --time-column.")
            quality_options["history_auto"] = True
        if args.history_config is not None:
            from agent.history_runtime import normalize_history_options
            raw_config = args.history_config
            if raw_config.startswith("@"):
                config_path = Path(raw_config[1:])
                if config_path.is_symlink() or not config_path.is_file() or config_path.stat().st_size > 64 * 1024:
                    raise ValueError("History config must be a regular JSON file under 64 KiB")
                raw_config = config_path.read_text(encoding="utf-8-sig")
            history = normalize_history_options(json.loads(raw_config),
                args.history_weight if args.history_weight is not None else 0.5)
            if history is None:
                raise ValueError("--history-config requires a JSON object, not null")
            quality_options.update(history_config=history["history_config"], history_weight=history["blend_weight"])
            if args.validation is not None or args.time_column is not None:
                raise ValueError("Explicit history uses v3 auto validation; omit --validation and --time-column.")
        elif args.history_weight is not None:
            raise ValueError("--history-weight requires --history-config")
        time_budget = args.time_budget if args.time_budget is not None else DEFAULT_TIME_BUDGET
        if args.max_steps < 1 or not 0 < time_budget <= 10500:
            raise ValueError("max-steps must be positive; time-budget must be in (0, 10500].")
        if selected_mode == "llm" and (args.validation is not None or args.time_column is not None):
            raise ValueError("--validation and --time-column are available only in offline mode.")
        if selected_mode == "offline" and (args.validation == "time") != (args.time_column is not None):
            raise ValueError("Use --validation time and --time-column together.")
        task_path = Path(args.task)
        if not task_path.is_file():
            raise FileNotFoundError(f"Task file not found: {task_path}")
        if (args.research_search == "adaptive" and time_budget > 300
                and args.validation in {None, "auto"} and args.time_column is None):
            from agent.task_contract import parse_task_metric

            try:
                parse_task_metric(task_path.read_text(encoding="utf-8-sig"))
            except ValueError:
                # Optional routing must not add an entry-point requirement to
                # legacy research. Its own parser/validator remains responsible
                # for unsupported, missing or conflicting task declarations.
                print("Adaptive search unavailable for this task metric; retaining the legacy solver.", flush=True)
            else:
                quality_options["adaptive_search"] = True
        deadline = time.monotonic() + time_budget
        checkpoint = _prepare_emergency(task_path, deadline)
        # Hold a small part of the same deadline for checking and restoring a
        # prepared CSV. Recovery never starts another training time budget.
        work_deadline = deadline - min(5.0, time_budget * .02)
        validation = {"strategy": args.validation or "auto"}
        if args.time_column is not None:
            validation["time_column"] = args.time_column
        if selected_mode == "offline":
            try:
                _offline_solution(task_path, deadline=work_deadline, validation=validation,
                                  quality_options=quality_options)
            except (OSError, TimeoutError, MemoryError, RuntimeError) as error:
                if not _restore_emergency(checkpoint, error, deadline=deadline):
                    raise
        else:
            if args.mode == "llm":
                load_local_env()
            research_started = solver_completed = False
            try:
                with make_client() as client:
                    research_started = True
                    run_agent(task_path, client, max_steps=args.max_steps,
                              time_budget=_remaining(work_deadline), **quality_options)
                    solver_completed = True
            except Exception as error:
                if checkpoint is not None:
                    checkpoint.check_inputs(deadline=deadline)
                if solver_completed and validate_predictions(task_path.resolve().parent)["valid"]:
                    print("Provider cleanup failed after verified predictions were saved: "
                          + safe_error_message(error), file=sys.stderr)
                elif not research_started and args.mode == "auto":
                    print("API client unavailable; starting offline fallback: "
                          + safe_error_message(error), file=sys.stderr)
                    try:
                        _offline_solution(task_path, deadline=work_deadline, validation=validation,
                                          quality_options=quality_options)
                    except (OSError, TimeoutError, MemoryError, RuntimeError) as offline_error:
                        if not _restore_emergency(checkpoint, offline_error, deadline=deadline):
                            raise
                elif research_started and isinstance(error, (OSError, TimeoutError, MemoryError)):
                    if not _restore_emergency(checkpoint, error, deadline=deadline):
                        raise
                else:
                    raise
        return 0
    except Exception as error:
        print(f"Error: {safe_error_message(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
