"""Small starting agent. Interface: python solution.py /path/to/task.md."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from validation import validate_predictions


ROOT = Path(__file__).resolve().parent
REQUIRED_PACKAGES = ("openai", "dotenv", "numpy", "pandas", "sklearn")
API_FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python in a new process in the task directory. Files persist "
            "between calls; Python variables do not. Returns stdout, stderr, "
            "exit_code and timed_out."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
SYSTEM_PROMPT = """You are an autonomous ML researcher. Read the supplied task
carefully and solve it using run_python. All relative paths are relative to the
directory containing task.md. Inspect the actual files, data dictionary and
submission template. Infer the target, task type, metric and output requirements
from those materials; do not assume specific column names or labels.

First train a fast, simple baseline and immediately save usable predictions in
the exact format and location requested by the task. Then perform a small
AutoResearch cycle: record a hypothesis, test it, measure its validation result,
and compare it with the best result. Keep a written experiment history in the
task directory. Reserve time for final training and writing predictions.

Use the same held-out validation rows when comparing experiments. Fit learned
preprocessing only on training rows. Do not use test targets, external data,
internet downloads or pretrained ML models. Available initial modelling tools
are numpy, pandas and scikit-learn. Set CPU parallelism to at most 8.

When replacing a useful prediction file, write to a temporary file first and
replace the destination atomically. A failed experiment must preserve the best
predictions. Check columns, row order, missing values and task-specific output
rules before finishing. Finish with a short factual account of completed work.
"""


def load_local_env() -> None:
    # Import only when installed: the offline check also works before setup.
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)


def require_api_settings() -> None:
    missing = [name for name in API_FIELDS if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Fill in these settings in .env: " + ", ".join(missing))


def child_environment() -> dict[str, str]:
    # Keep Windows system paths, but do not inherit the agent's API credentials.
    allowed = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
        "HOME", "USERPROFILE", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    })
    return env


def stop_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def tail(path: Path, limit: int = 12000) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def run_python(code: str, task_dir: Path, run_dir: Path, step: int,
               timeout: float = 1800) -> dict:
    if not isinstance(code, str) or not code.strip():
        return {"error": "code must be a non-empty string"}
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / f"experiment_{step:03d}.py"
    stdout_path = script.with_suffix(".stdout.txt")
    stderr_path = script.with_suffix(".stderr.txt")
    script.write_text(code, encoding="utf-8")
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
             str(script.resolve())],
            cwd=task_dir, env=child_environment(), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process_tree(process)
    return {
        "stdout": tail(stdout_path), "stderr": tail(stderr_path),
        "exit_code": process.returncode, "timed_out": timed_out,
    }


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


def make_client():
    require_api_settings()
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=60, max_retries=0,
    )


def check_api() -> None:
    with make_client() as client:
        response = client.chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=64,
        )
    print("API connection successful.")
    print("Reply:", response.choices[0].message.content or "(empty response)")


def run_agent(task_path: Path, client, *, max_steps: int = 12,
              time_budget: float = 10500) -> Path:
    task_path = task_path.resolve()
    task = task_path.read_text(encoding="utf-8")
    deadline = time.monotonic() + time_budget
    run_dir = task_path.parent / ".agent_runs" / uuid4().hex
    run_dir.mkdir(parents=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            if validate_predictions(task_path.parent)["valid"]:
                print("Time budget reached; a structurally valid submission.csv was preserved.")
                return run_dir
            raise TimeoutError(f"Agent time budget reached. Experiments: {run_dir}")
        print(f"Step {step}/{max_steps}; remaining {int(remaining)}s", file=sys.stderr)
        response = client.with_options(timeout=min(120, remaining)).chat.completions.create(
            model=os.environ["OPENAI_MODEL"], messages=messages,
            tools=[TOOL], parallel_tool_calls=False,
            max_tokens=4096,
        )
        message = response.choices[0].message
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("Model response was truncated; no code from it was executed.")
        if message.tool_calls:
            messages.append({
                "role": "assistant", "content": message.content,
                "tool_calls": [call.model_dump(exclude_none=True)
                               for call in message.tool_calls],
            })
            for index, call in enumerate(message.tool_calls, start=1):
                try:
                    if call.function.name != "run_python":
                        raise ValueError("Unknown tool: " + call.function.name)
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict) or set(arguments) != {"code"}:
                        raise ValueError("run_python requires exactly one argument: code")
                    remaining = deadline - time.monotonic()
                    if remaining <= 5:
                        raise TimeoutError("No time remains for another experiment")
                    result = run_python(
                        arguments["code"], task_path.parent, run_dir,
                        step * 100 + index, timeout=min(1800, remaining - 2),
                    )
                except (ValueError, OSError, TimeoutError) as error:
                    result = {"error": str(error)}
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            validation = validate_predictions(task_path.parent)
            if validation["valid"]:
                print(message.content or "Agent finished.")
                print("submission.csv passed the structural check.")
                return run_dir
            messages.extend([
                {"role": "assistant", "content": message.content or "Finished."},
                {"role": "user", "content": (
                    "The output check failed. Use run_python to fix the output before finishing: "
                    + json.dumps(validation, ensure_ascii=False)
                )},
            ])
    if validate_predictions(task_path.parent)["valid"]:
        print("Step limit reached; a structurally valid submission.csv was preserved.")
        return run_dir
    raise RuntimeError(f"Step limit reached. Existing output files were preserved. Logs: {run_dir}")


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
        # SDK exceptions may include server response text. Never echo our key.
        detail = str(error)
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            detail = detail.replace(key, "[REDACTED]")
        print(f"Error: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
