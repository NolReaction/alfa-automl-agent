"""Stdlib bootstrap for observing AutoML inside the native baseline subprocess.

Only returned execution metadata is copied. Native arguments, return objects,
exceptions, model recipes and training allowances retain their original values.
"""
import functools
import json
import os
from pathlib import Path
import tempfile


_EXECUTION_FIELDS = ("exit_code", "timed_out", "memory_limit_exceeded", "error",
                     "error_type", "elapsed_seconds", "resources", "cleanup")


def _plain(value):
    path = Path(value).absolute()
    if any(part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction())
           for part in (path, *path.parents)):
        raise ValueError("Linked AutoML observer path")
    return path.resolve()


def _write(path, value):
    _plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".automl-observer-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install(sidecar, task_root):
    """Observe this subprocess's AutoML phases without modifying solver files."""
    import agent.automl as automl

    sidecar, task_root = _plain(sidecar), _plain(task_root)
    if not sidecar.is_relative_to(task_root) or sidecar.exists():
        raise ValueError("AutoML observer requires a new sidecar inside this task")
    value = {"version": 1, "observations": [], "pending": [], "errors": []}
    _write(sidecar, value)
    original = automl.run_python

    def record_error(error, directory=None):
        value["errors"].append({"directory": directory,
                                "error": type(error).__name__ + ": " + str(error)[-1000:]})
        try:
            _write(sidecar, value)
        except (OSError, ValueError, TypeError):
            # A stale snapshot/pending record remains unknown to the auditor.
            # Observation I/O must never alter the native solver result.
            pass

    @functools.wraps(original)
    def observed(*args, **kwargs):
        directory = None
        try:
            current = _plain(args[1] if len(args) > 1 else kwargs["task_dir"])
            if not current.is_relative_to(task_root):
                raise ValueError("Observed AutoML child escapes this task")
            directory = str(current)
            value["pending"].append(directory)
            _write(sidecar, value)
        except (OSError, ValueError, TypeError, KeyError) as error:
            record_error(error, directory)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            # Keep pending evidence and the exact original exception.
            record_error(error, directory)
            raise
        try:
            if directory is None or not isinstance(result, dict):
                raise ValueError("AutoML child returned no recordable execution")
            execution = json.loads(json.dumps(
                {key: result[key] for key in _EXECUTION_FIELDS if key in result}, allow_nan=False))
            value["observations"].append({"directory": directory, "execution": execution})
            value["pending"].remove(directory)
            _write(sidecar, value)
        except (OSError, ValueError, TypeError, KeyError) as error:
            record_error(error, directory)
        return result

    automl.run_python = observed
