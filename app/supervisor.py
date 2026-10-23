"""Contain one fixed job command and clean up its background descendants."""

from __future__ import annotations

import ctypes
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
        ("flags", ctypes.c_uint32), ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t), ("active_processes", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32),
    ]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "read_operations", "write_operations", "other_operations",
        "read_bytes", "write_bytes", "other_bytes",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits), ("io", _IOCounters),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


def _contain_windows():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, boolean, dword = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], handle),
        "SetInformationJobObject": ([handle, ctypes.c_int, ctypes.c_void_p, dword], boolean),
        "GetCurrentProcess": ([], handle),
        "AssignProcessToJobObject": ([handle, handle], boolean),
        "CloseHandle": ([handle], boolean),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = _ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    # The supervisor is itself a member. Keep the non-inheritable handle open
    # until OS process exit, which kills all remaining members, even new groups.
    return job


def process_inventory_matches() -> bool:
    """Check that /proc describes the PID namespace used by os.kill/getpid.

    A host-mounted /proc can expose an unrelated process under the same numeric
    PID as this supervisor. Never derive a process tree from that inventory.
    """
    if sys.platform != "linux":
        return False
    try:
        return (Path("/proc/self/stat").is_file()
                and int(os.readlink("/proc/self")) == os.getpid())
    except (OSError, ValueError):
        return False


def _contain_linux():
    if not process_inventory_matches():
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return True


def _trusted_without_proc(command):
    """Allow only fixed host-controlled entry points without tree inspection.

    Offline search can itself launch recipe processes. Without a matching
    inventory this path cannot promise full descendant supervision; generated
    Python and arbitrary entry points remain unavailable.
    """
    if (len(command) not in {5, 6} or command[1] != "-u"
            or Path(command[0]).resolve() != Path(sys.executable).resolve()):
        return False
    script = Path(command[2]).resolve()
    if script == ROOT / "app" / "worker.py":
        # worker.load_config independently validates the fixed offline actions.
        return len(command) == 5 and command[3] == "--config"
    return (script == ROOT / "prepare_data.py" and len(command) == 6
            and command[3:5] == ["all", "--data-dir"])


def _descendants():
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            parents[int(path.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {os.getpid()}
    while True:
        children = {pid for pid, parent in parents.items() if parent in found} - found
        if not children:
            return found - {os.getpid()}
        found.update(children)


def _signal(pid, value):
    try:
        os.kill(pid, value)
    except ProcessLookupError:
        pass


def _cleanup_linux():
    # Orphans are adopted by this subreaper, so a short-lived parent cannot
    # evade cleanup. Freeze parents before the next scan to close fork races.
    stopped = set()
    while True:
        remaining = _descendants()
        for pid in remaining - stopped:
            _signal(pid, signal.SIGSTOP)
        if remaining <= stopped:
            break
        stopped.update(remaining)
    for pid in stopped:
        _signal(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if not pid:
            if time.monotonic() >= deadline:
                raise RuntimeError("Background processes did not exit after termination.")
            time.sleep(0.01)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    cpu_lane = bool(arguments and arguments[0] == "--cpu-lane")
    if cpu_lane:
        arguments.pop(0)
    if not arguments or arguments[0] != "--" or len(arguments) < 2:
        raise ValueError("Expected -- followed by the internal job command.")
    inventory = False
    if os.name == "nt":
        job_handle = _contain_windows()
    elif sys.platform == "linux":
        inventory = _contain_linux()
        if not inventory and not _trusted_without_proc(arguments[1:]):
            raise RuntimeError("Qwen and arbitrary commands require Linux /proc process supervision in the current PID namespace.")
    else:
        raise RuntimeError("Job supervision supports Windows and Linux only.")
    lane = ExitStack()
    try:
        if cpu_lane:
            # Own the OS lock in the contained process, so losing the app's
            # HTTP process cannot release the lane while training continues.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from research_lab.process import coordinator_lock, machine_lock_path
            lane.enter_context(coordinator_lock(machine_lock_path()))
        return subprocess.call(arguments[1:], stdin=subprocess.DEVNULL)
    finally:
        try:
            if inventory:
                _cleanup_linux()
        finally:
            lane.close()


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"Job supervisor: {error}", file=sys.stderr, flush=True)
        code = 1
    if code < 0 and os.name != "nt":
        os.kill(os.getpid(), -code)
    raise SystemExit(code)
