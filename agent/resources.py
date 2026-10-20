"""Sample and terminate only processes belonging to one launched experiment.

Descendants are retained by PID plus creation time, including descendants that
start a new session or outlive their immediate parent. Polling cannot discover a
process that forks and becomes orphaned entirely between two samples; this is
best-effort process monitoring, not an OS sandbox or a hard memory allocation cap.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import subprocess
import sys
import time

import psutil


DEFAULT_MEMORY_LIMIT_BYTES = 28 * 1024 ** 3
POLL_SECONDS = 0.05


def _namespace_matches() -> bool:
    if sys.platform != "linux":
        return True
    try:
        # /proc can be mounted from a host PID namespace while Popen uses
        # container-local PIDs. Never inspect/signal those numerically matching
        # host PIDs, even if an accidental PPID collision looks plausible.
        return int(os.readlink(os.path.join(psutil.PROCFS_PATH, "self"))) == os.getpid()
    except (OSError, ValueError):
        return False


@dataclass
class _OwnedProcess:
    process: psutil.Process
    created: float
    cpu_seconds: float = 0.0


class ProcessTreeMonitor:
    """Keep scoped process identities, sampled tree RSS and accumulated CPU time.

RSS is summed across live owned processes, so shared pages may be counted more
than once. CPU time sums each tracked process's own user/system time, excluding
the child-time fields that would double-count descendants. Values are sampled;
very short-lived processes can disappear before they can be measured.
"""

    def __init__(self, process: subprocess.Popen, *,
                 max_memory_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES):
        if max_memory_bytes is not None and (
            isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int)
            or max_memory_bytes <= 0
        ):
            raise ValueError("max_memory_bytes must be a positive integer or None")
        self.process = process
        self.memory_limit_bytes = max_memory_bytes
        self._owned: dict[tuple[int, float], _OwnedProcess] = {}
        self._peak_rss = 0
        self._samples = 0
        self._unavailable_reason: str | None = None
        self._cleanup_complete: bool | None = None
        try:
            if not _namespace_matches():
                raise RuntimeError("Process namespace does not match the mounted process inventory")
            root = psutil.Process(process.pid)
            # Detect inaccessible/mismatched PID namespaces before treating any
            # psutil PID as one of our children. Popen remains the fallback owner.
            if root.ppid() != os.getpid():
                raise RuntimeError("Process namespace does not expose the launched child")
            self._remember(root)
        except (psutil.Error, OSError, RuntimeError) as error:
            self._unavailable_reason = type(error).__name__ + ": process inspection unavailable"

    def _remember(self, process: psutil.Process) -> None:
        created = process.create_time()
        key = (process.pid, created)
        if key not in self._owned:
            self._owned[key] = _OwnedProcess(process=process, created=created)

    @staticmethod
    def _alive(owned: _OwnedProcess) -> bool:
        try:
            # psutil is_running and all signalling methods recheck PID reuse.
            # An explicit identity comparison also protects fake/custom Process
            # providers and documents the invariant used by this tracker.
            return (owned.process.is_running()
                    and owned.process.create_time() == owned.created
                    and owned.process.status() != psutil.STATUS_ZOMBIE)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False

    def _unavailable(self, error: Exception) -> None:
        self._unavailable_reason = type(error).__name__ + ": process inspection unavailable"

    def _discover(self) -> None:
        # Scan descendants of all retained live owners, not just the root: a
        # nested worker may have a new session or a now-exited immediate parent.
        for owned in tuple(self._owned.values()):
            try:
                if not self._alive(owned):
                    continue
                for child in owned.process.children(recursive=True):
                    try:
                        self._remember(child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, OSError) as error:
                self._unavailable(error)

    def sample(self) -> bool:
        """Sample owned processes; return whether summed RSS exceeds the limit."""
        self._discover()
        total_rss = 0
        measured = False
        for owned in tuple(self._owned.values()):
            try:
                if not self._alive(owned):
                    continue
                rss = owned.process.memory_info().rss
                cpu = owned.process.cpu_times()
                owned.cpu_seconds = max(owned.cpu_seconds, cpu.user + cpu.system)
                total_rss += rss
                measured = True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, OSError) as error:
                self._unavailable(error)
        if measured:
            self._samples += 1
            self._peak_rss = max(self._peak_rss, total_rss)
        return self.memory_limit_bytes is not None and total_rss > self.memory_limit_bytes

    def summary(self) -> dict:
        available = self._samples > 0 and self._unavailable_reason is None
        return {
            "available": available,
            "peak_rss_bytes": self._peak_rss if available else None,
            "cpu_seconds": round(sum(item.cpu_seconds for item in self._owned.values()), 6)
                           if available else None,
            "memory_limit_bytes": self.memory_limit_bytes,
            "unavailable_reason": (None if available else self._unavailable_reason
                                   or "Process exited before resource sampling"),
            "sample_count": self._samples,
            "tracked_processes": len(self._owned),
            "cleanup_complete": self._cleanup_complete,
        }

    def terminate(self) -> bool:
        """Freeze and kill retained descendants, then reap the launched child.

        Freezing parents before the final discovery closes ordinary fork races
        during cleanup. No global process list is signalled, and stale identities
        never receive signals. The root Popen handle is still used when process
        inspection is unavailable.
        """
        frozen: set[tuple[int, float]] = set()
        deadline = time.monotonic() + 1.0
        while True:
            self._discover()
            pending = [(key, owned) for key, owned in self._owned.items() if key not in frozen]
            if not pending:
                break
            for key, owned in pending:
                try:
                    if self._alive(owned):
                        owned.process.suspend()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                except (psutil.AccessDenied, OSError) as error:
                    self._unavailable(error)
                frozen.add(key)
            if time.monotonic() >= deadline:
                break
        # Signal the original process group while its root still pins that group
        # ID. Afterwards the root may already be reaped and its PID reusable.
        if self.process.poll() is None and os.name != "nt":
            try:
                if os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for owned in reversed(tuple(self._owned.values())):
            try:
                if self._alive(owned):
                    owned.process.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, OSError) as error:
                self._unavailable(error)

        # Popen tracks the actual child (including its Windows process handle).
        # The group and individual descendants were handled above.
        if self.process.poll() is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._cleanup_complete = False
            return False

        # Descendants are not necessarily waitable by us, and zombies have
        # already stopped executing. Bound cleanup instead of blocking forever.
        deadline = time.monotonic() + 2.0
        while True:
            remaining = False
            for owned in self._owned.values():
                try:
                    remaining = self._alive(owned) or remaining
                except (psutil.AccessDenied, OSError) as error:
                    self._unavailable(error)
                    remaining = True
            if not remaining or time.monotonic() >= deadline:
                # With inaccessible process inventory we can stop/reap Popen,
                # but cannot honestly certify that all descendants were found.
                self._cleanup_complete = (False if remaining else
                                          None if self._unavailable_reason else True)
                return not remaining
            time.sleep(0.01)
