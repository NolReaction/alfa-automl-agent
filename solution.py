"""Run the competition agent, or explicitly select offline AutoML."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from agent.execution import child_environment, run_python, stop_process_tree, tail
from agent.llm import (
    API_FIELDS, ROOT, SYSTEM_PROMPT, TOOL, check_api, load_local_env,
    make_client, require_api_settings,
)
from agent.research import run_agent
from validation import validate_predictions


REQUIRED_PACKAGES = ("openai", "dotenv", "numpy", "pandas", "sklearn", "catboost",
                     "lightgbm", "optuna", "psutil")


def select_mode(requested: str) -> str:
    """Auto uses the judge's environment, never a developer's local .env file."""
    if requested != "auto":
        return requested
    present = [name for name in API_FIELDS if os.environ.get(name, "").strip()]
    if not present:
        print("No API settings supplied: offline fallback (budgeted AutoML; no LLM research).", flush=True)
        return "offline"
    missing = [name for name in API_FIELDS if name not in present]
    if missing:
        raise ValueError("Incomplete competition API environment; missing: " + ", ".join(missing)
                         + ". Use --mode offline for an intentional local run without API.")
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
        "api_called": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return python_ok and all(packages.values()) and worker_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--mode", choices=("auto", "offline", "llm"), default="auto",
                        help="auto (default): use supplied competition API, otherwise offline fallback; "
                             "offline: budgeted AutoML without API; llm: Qwen research with local .env support")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline environment check; no API calls")
    mode.add_argument("--check-api", action="store_true", help="Make one short request to the configured API")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--time-budget", type=float, default=None,
                        help="Seconds; default 600 offline / 10500 llm; maximum 10500")
    parser.add_argument("--validation", choices=("auto", "random", "grouped", "time"), default=None,
                        help="Offline holdout strategy; defaults to auto")
    parser.add_argument("--time-column", help="Numeric period feature for --validation time")
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
        time_budget = args.time_budget if args.time_budget is not None else (600 if selected_mode == "offline" else 10500)
        if args.max_steps < 1 or not 0 < time_budget <= 10500:
            raise ValueError("max-steps must be positive; time-budget must be in (0, 10500].")
        if selected_mode == "llm" and (args.validation is not None or args.time_column is not None):
            raise ValueError("--validation and --time-column are available only in offline mode.")
        if selected_mode == "offline" and (args.validation == "time") != (args.time_column is not None):
            raise ValueError("Use --validation time and --time-column together.")
        task_path = Path(args.task)
        if not task_path.is_file():
            raise FileNotFoundError(f"Task file not found: {task_path}")
        if selected_mode == "offline":
            from agent.automl import run_automl

            validation = {"strategy": args.validation or "auto"}
            if args.time_column is not None:
                validation["time_column"] = args.time_column
            result = run_automl(task_path, time_budget=time_budget, validation=validation)
            metric = result["metric"]
            print(f"Best model: {result['model']}; {metric['name']}: {metric['value']:.6f}")
            print(f"Saved: {task_path.resolve().parent / 'submission.csv'}. API calls: 0.")
            print(f"Run report: {result['report_path']}")
        else:
            if args.mode == "llm":
                load_local_env()
            with make_client() as client:
                run_agent(task_path, client, max_steps=args.max_steps, time_budget=time_budget)
        return 0
    except Exception as error:
        detail = str(error)
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            detail = detail.replace(key, "[REDACTED]")
        print(f"Error: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
