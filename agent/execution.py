"""Run generated Python experiments with bounded output and process timeouts."""

from __future__ import annotations

import os
from pathlib import Path
import math
import subprocess
import sys
import time

from agent.resources import DEFAULT_MEMORY_LIMIT_BYTES, POLL_SECONDS, ProcessTreeMonitor


WINDOWS = os.name == "nt"
WINDOWS_PROCESS_CWD_LIMIT = 258
WINDOWS_LAUNCH_CWD_LIMIT = 240
PYTHON_RUNNER = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
PYTHON_CWD_RUNNER = ("import os, runpy, sys; os.chdir(sys.argv.pop(2)); "
                     "runpy.run_path(sys.argv[1], run_name='__main__')")


def _windows_path_units(path: Path) -> int:
    return len(str(path).encode("utf-16-le", errors="surrogatepass")) // 2


def _python_launch(script: Path, task_dir: Path) -> tuple[list[str], Path]:
    """Keep CreateProcess's initial cwd short, then enter the real task in Python.

    Windows long-file-path support does not extend to CreateProcessW's cwd.
    Only long Windows task directories need the bootstrap; the script still
    executes in its actual directory, with the same argv as the usual runner.
    The parent process's current directory is never changed.
    """
    command = [sys.executable, "-u", "-c", PYTHON_RUNNER, str(script.resolve())]
    if not WINDOWS:
        return command, task_dir
    actual_task_dir = Path(task_dir).resolve()
    if _windows_path_units(actual_task_dir) <= WINDOWS_PROCESS_CWD_LIMIT:
        return command, task_dir
    if not actual_task_dir.is_dir():
        raise NotADirectoryError(f"Task working directory does not exist: {actual_task_dir}")
    for ancestor in actual_task_dir.parents:
        if (_windows_path_units(ancestor) <= WINDOWS_LAUNCH_CWD_LIMIT
                and ancestor.is_dir()):
            return ([sys.executable, "-u", "-c", PYTHON_CWD_RUNNER,
                     str(script.resolve()), str(actual_task_dir)], ancestor)
    raise OSError("Cannot launch Windows Python: the long task directory has no existing "
                  f"ancestor within {WINDOWS_LAUNCH_CWD_LIMIT} UTF-16 code units. "
                  "Use a shorter task root or share path.")


def child_environment() -> dict[str, str]:
    # Keep Windows system paths, but do not inherit the agent's API credentials.
    allowed = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
        "HOME", "USERPROFILE", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        # The case files and research manifest are UTF-8. This also sets the
        # default encoding for open()/Path.read_text() on Windows, unlike stdio.
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
    })
    return env


def stop_process_tree(process: subprocess.Popen, tracker: ProcessTreeMonitor | None = None) -> None:
    (tracker if tracker is not None else ProcessTreeMonitor(process)).terminate()


def tail(path: Path, limit: int = 12000) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def run_python(code: str, task_dir: Path, run_dir: Path, step: int,
               timeout: float = 1800, *,
               max_memory_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES) -> dict:
    if not isinstance(code, str) or not code.strip():
        return {"error": "code must be a non-empty string"}
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        return {"error": "timeout must be a positive finite number"}
    if max_memory_bytes is not None and (
        isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int)
        or max_memory_bytes <= 0
    ):
        return {"error": "max_memory_bytes must be a positive integer or None"}
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / f"experiment_{step:03d}.py"
    stdout_path = script.with_suffix(".stdout.txt")
    stderr_path = script.with_suffix(".stderr.txt")
    script.write_text(code, encoding="utf-8")
    timed_out = False
    memory_limit_exceeded = False
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        command, launch_dir = _python_launch(script, task_dir)
        process = subprocess.Popen(
            command, cwd=launch_dir, env=child_environment(), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        tracker = ProcessTreeMonitor(process, max_memory_bytes=max_memory_bytes)
        try:
            while True:
                exceeded = tracker.sample()
                if process.poll() is not None:
                    break
                if exceeded:
                    memory_limit_exceeded = True
                    break
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(POLL_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Clean up tracked descendants even when their root exits normally.
            stopped = tracker.terminate()
    result = {
        "stdout": tail(stdout_path), "stderr": tail(stderr_path),
        "exit_code": process.returncode, "timed_out": timed_out,
        "memory_limit_exceeded": memory_limit_exceeded,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "resources": tracker.summary(),
    }
    if not stopped:
        result["error"] = "Experiment processes could not be fully stopped within the cleanup deadline"
    return result
