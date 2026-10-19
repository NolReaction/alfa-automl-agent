"""Competition entry point: python solution.py /path/to/task.md."""

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


REQUIRED_PACKAGES = ("openai", "dotenv", "numpy", "pandas", "sklearn")


def check_setup() -> bool:
    python_ok = sys.version_info[:2] == (3, 12)
    packages = {name: importlib.util.find_spec(name) is not None
                for name in REQUIRED_PACKAGES}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        result = run_python(
            "import numpy, pandas, sklearn, openai; print('python-child-ok')",
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline environment check; no API calls")
    mode.add_argument("--check-api", action="store_true", help="Make one short request to the configured API")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--time-budget", type=float, default=10500, help="Seconds; maximum 10500")
    args = parser.parse_args()
    try:
        load_local_env()
        if args.check:
            return 0 if check_setup() else 1
        if args.check_api:
            check_api()
            return 0
        if args.max_steps < 1 or not 0 < args.time_budget <= 10500:
            raise ValueError("max-steps must be positive; time-budget must be in (0, 10500].")
        task_path = Path(args.task)
        if not task_path.is_file():
            raise FileNotFoundError(f"Task file not found: {task_path}")
        with make_client() as client:
            run_agent(task_path, client, max_steps=args.max_steps, time_budget=args.time_budget)
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
