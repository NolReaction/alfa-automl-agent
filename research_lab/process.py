"""Scoped adapter around the trusted process-tree supervisor, not a new runner."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import threading
import time

from research_lab.integrity import safe_path


_ADAPTER_LOCK = threading.Lock()


@contextmanager
def coordinator_lock(path):
    """An OS-owned lock releases on coordinator death, on POSIX and Windows."""
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another laboratory coordinator holds the lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another laboratory coordinator holds the lock") from error
        yield
    finally:
        # Closing an owned file descriptor releases either platform's lock.
        stream.close()


def machine_lock_path():
    """One CPU lane for this user, including runs in different output trees."""
    owner = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "local")
    return Path(tempfile.gettempdir()) / ("alfa-research-lab-cpu-" + owner + ".lock")


def supervise(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes, *, threads=8):
    """Retain trusted cleanup semantics while measuring cleanup independently."""
    import qualify_unseen_tasks as trusted

    with _ADAPTER_LOCK:
        original_environment, original_stop = trusted.sanitized_environment, trusted._OwnedProcesses.stop
        cleanup_seconds = 0.0
        cleanup_started = None

        def environment():
            # Call the saved function: integrity's environment helper imports
            # trusted.sanitized_environment and would recurse while patched.
            result = original_environment()
            result.update({name: str(threads) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")})
            result["CUDA_VISIBLE_DEVICES"] = ""
            return result

        def stop(owned, *args, **kwargs):
            nonlocal cleanup_seconds, cleanup_started
            started = time.monotonic()
            cleanup_started = started
            try:
                return original_stop(owned, *args, **kwargs)
            finally:
                cleanup_seconds += time.monotonic() - started

        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("threads must be 1..8")
        trusted.sanitized_environment, trusted._OwnedProcesses.stop = environment, stop
        started = time.monotonic()
        try:
            result = trusted.run_solver(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes)
        finally:
            trusted.sanitized_environment, trusted._OwnedProcesses.stop = original_environment, original_stop
        # trusted.run_solver waits for the root and finalizes bounded logs after
        # stop(). Include that finalization in cleanup rather than training.
        cleanup_elapsed = time.monotonic() - cleanup_started if cleanup_started is not None else cleanup_seconds
        result["owned_stop_elapsed_seconds"] = cleanup_seconds
        result["cleanup_elapsed_seconds"] = cleanup_elapsed
        result["cleanup_scope"] = "Owned-tree stop, root wait, bounded log finalization"
        result["worker_wall_seconds"] = max(0.0, result["elapsed_seconds"] - cleanup_elapsed)
        result["supervisor_elapsed_seconds"] = time.monotonic() - started
        result["memory_limit_bytes"] = memory_limit_bytes
        result["threads"] = threads
        return result
