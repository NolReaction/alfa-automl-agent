"""Small shared evidence primitives; no solver imports from supplied roots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 32 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def safe_path(value):
    path = Path(value).expanduser().absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Linked paths are unsupported")
    return path.resolve()


def under(root, value):
    root, path = safe_path(root), safe_path(value)
    require(path.is_relative_to(root), "Evidence path escapes its root")
    return path


def file_hash(path):
    path = safe_path(path)
    require(path.is_file(), "Missing regular evidence file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES, "Missing/oversized JSON evidence")
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    value = json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    require(isinstance(value, dict), "JSON evidence must be an object")
    return value


def write_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes(value) + b"\n"
    require(len(raw) <= MAX_JSON_BYTES, "JSON evidence exceeds size limit")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lab-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(not path.is_symlink(), "Linked evidence destination")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def source_identity(root):
    # The imported auditor implementation is trusted and independently sealed.
    # solver_identity only reads the supplied checkout; it never imports it.
    from qualify_unseen_tasks import solver_identity
    return solver_identity(safe_path(root))


def check_identity(identity, *, strict_sha=True):
    actual = source_identity(identity["root"])
    require(actual["source_hashes"] == identity["source_hashes"]
            and actual["source_fingerprint"] == identity["source_fingerprint"], "Source drift")
    if strict_sha:
        require(actual["sha"] == identity["sha"] and not actual["dirty"], "Solver checkout/dirty status drift")
    return actual


def sanitized_environment(threads=8):
    require(type(threads) is int and 1 <= threads <= 8, "threads must be 1..8")
    from qualify_unseen_tasks import sanitized_environment as trusted_environment
    result = trusted_environment()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        result[name] = str(threads)
    result["CUDA_VISIBLE_DEVICES"] = ""
    return result


def environment_versions(python):
    packages = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "openai", "httpx",
                "pydantic", "tqdm", "python-dotenv", "catboost", "lightgbm", "optuna", "psutil",
                "autogluon.tabular", "xgboost"]
    code = ("import json,sys,platform,importlib.metadata as m\n"
            "d={'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'packages':{}}\n"
            f"for name in {packages!r}:\n"
            " try: d['packages'][name]=m.version(name)\n"
            " except m.PackageNotFoundError: d['packages'][name]=None\n"
            "print(json.dumps(d))\n")
    # Preserve a venv's invocation path: resolving bin/python's symlink would
    # select the base interpreter and lose that venv's installed packages.
    interpreter = Path(python).expanduser().absolute()
    require(interpreter.is_file(), "Missing selected interpreter")
    process = subprocess.run([str(interpreter), "-I", "-c", code], capture_output=True, text=True,
                             encoding="utf-8", env=sanitized_environment(), timeout=30, check=False)
    require(process.returncode == 0, "Interpreter preflight failed")
    require(len(process.stdout) <= 128 * 1024, "Oversized environment output")
    value = json.loads(process.stdout)
    require(value["python"].startswith("3.12."), "Laboratory CPU lane requires Python 3.12")
    return value


def is_transport_temp(relative):
    """Reserved delivery staging files are never model/evidence inputs."""
    return ".rsync-tmp" in Path(relative).parts


def transient_artifact(relative, directory):
    relative = Path(relative)
    if is_transport_temp(relative):
        return True
    # The trusted supervisor publishes bounded stdout.txt/stderr.txt and then
    # removes its raw spools. A synchronized filesystem can deliver a stale
    # spool after that deletion. Only those two known spools with a published
    # canonical tail are excluded; other raw/failure logs remain indexed.
    if relative.name in {".stdout.raw", ".stderr.raw"} and relative.parent.name in {"execution", "probe_execution"}:
        tail = "stdout.txt" if relative.name == ".stdout.raw" else "stderr.txt"
        return (Path(directory) / relative.parent / tail).is_file()
    return False


def file_index(directory, *, exclude=()):
    directory = safe_path(directory)
    excluded = {str(value) for value in exclude}
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Linked artifact in evidence tree")
        name = path.relative_to(directory).as_posix()
        if path.is_file() and name not in excluded and not transient_artifact(name, directory):
            result[name] = file_hash(path)
    return result


# Explicit aliases for adapters; all use the same strict implementation.
load_json = read_json
atomic_json = write_json
sha256_json = json_hash
file_sha256 = file_hash
