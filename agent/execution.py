"""Run generated Python experiments with bounded output and process timeouts."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys


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
