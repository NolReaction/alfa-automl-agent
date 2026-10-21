"""Independent, finite qualification of tabular solver checkouts; no API calls.

Default invocation prepares a new, sealed run without starting a solver. Use
--run DIR to execute it and --re-audit DIR to recalculate saved predictions.
The auditor never imports the selected solver checkout. Audit answers live in
an auditor-owned sibling directory, not the task or solver argv. This is a
data-flow boundary, not an operating-system sandbox.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CSV_BYTES = 128 * 1024 * 1024
METRIC_DIRECTIONS = {"RMSE": False, "MAE": False, "MSE": False, "R2": True,
                     "Accuracy": True, "Macro F1": True, "Weighted F1": True,
                     "ROC AUC": True, "LogLoss": False}
CONTRACT_CODE = (
    "import json,sys\nfrom pathlib import Path\n"
    "sys.path.insert(0,sys.argv[1])\n"
    "from agent.task_spec import load_task\n"
    "try:\n spec=load_task(Path(sys.argv[2]))\n"
    "except Exception as e:\n print(json.dumps({'contract_error':{'type':type(e).__name__,'message':str(e)}},"
    "ensure_ascii=False),file=sys.stderr)\n sys.exit(2)\n"
    "print(json.dumps(spec.to_dict(),ensure_ascii=False,allow_nan=False))\n"
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular evidence/input file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Missing, linked or oversized JSON evidence")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON evidence must be an object")
    return value


def _path(path):
    result = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (result, *result.parents)):
        raise ValueError("Linked paths are not supported for qualification")
    return result.resolve()


def _inside(root, value):
    path = _path(value)
    if not path.is_relative_to(_path(root)):
        raise ValueError("Evidence path escapes its task directory")
    return path


def _git(root, *args):
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                               text=True, timeout=15, check=False)
    if completed.returncode:
        raise ValueError("Solver root must be an accessible Git checkout")
    return completed.stdout.strip()


def solver_identity(root):
    root = _path(root)
    if not (root / "solution.py").is_file():
        raise ValueError("solver-root must contain solution.py")
    names = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    hashes = {}
    for name in sorted(set(names)):
        if name.endswith(".py") or Path(name).name in {"requirements.txt", "requirements.platform.txt"}:
            path = _inside(root, root / name)
            if path.is_file():
                hashes[name] = file_hash(path)
    if "solution.py" not in hashes:
        raise ValueError("solution.py is absent from the source manifest")
    return {"root": str(root), "sha": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(_git(root, "status", "--porcelain")),
            "source_hashes": hashes, "source_fingerprint": json_hash(hashes)}


def sanitized_environment():
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
               "LD_LIBRARY_PATH", "SYSTEMDRIVE", "COMSPEC", "PATHEXT"}
    result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    result.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONNOUSERSITE="1",
                  OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                  NUMEXPR_NUM_THREADS="8", VECLIB_MAXIMUM_THREADS="8")
    return result


def safe_text(value, limit=8000):
    value = str(value)
    value = re.sub(r"(?i)\b(?:https?://)[^\s\"']+", "[redacted-url]", value)
    value = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [redacted]", value)
    value = re.sub(r"(?i)((?:api[_-]?key|authorization|password|secret|access[_-]?token)\s*[=:]\s*)[^\s,;]+",
                   r"\1[redacted]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted-key]", value)
    return value[-limit:]


def environment_versions(python):
    code = ("import json,platform,importlib.metadata as m\n"
            "v={'python':platform.python_version(),'platform':platform.platform()}\n"
            "for n in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','psutil']:\n"
            " try: v[n]=m.version(n)\n except m.PackageNotFoundError: v[n]=None\n"
            "print(json.dumps(v))\n")
    completed = subprocess.run([str(python), "-c", code], env=sanitized_environment(),
                               capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode:
        raise ValueError("Selected Python failed its dependency-version preflight")
    return json.loads(completed.stdout)


def input_hashes(task_dir, relative_paths):
    return {name: file_hash(_inside(task_dir, Path(task_dir) / name)) for name in relative_paths}


def prepare_run(solver_root, output_dir, *, python=sys.executable, cases=None,
                modes=("offline",), seed=81221, solver_seconds=30, timeout_seconds=90,
                total_seconds=3600, memory_gib=8):
    from unseen_cases import CATALOG_VERSION, case_catalog, materialize_case

    identity = solver_identity(solver_root)
    # Venv Python is normally a symlink. Preserve its lexical path: resolving
    # it to the base interpreter would silently discard the selected venv.
    python = Path(python).expanduser().absolute()
    if not python.is_file():
        raise ValueError("Selected Python interpreter does not exist")
    output_dir = _path(output_dir)
    if output_dir.exists():
        raise ValueError("Qualification output directory must be new")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (not modes or len(set(modes)) != len(modes)
            or any(mode not in {"offline", "default", "contract"} for mode in modes)):
        raise ValueError("Choose distinct offline, default or contract modes")
    for name, value, maximum in (("solver_seconds", solver_seconds, 10500),
                                 ("timeout_seconds", timeout_seconds, 10800),
                                 ("total_seconds", total_seconds, 7200), ("memory_gib", memory_gib, 16)):
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"Invalid finite qualification budget: {name}")
    catalog = case_catalog()
    known = {row["id"]: row for row in catalog}
    chosen = list(known) if cases is None else list(cases)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(known):
        raise ValueError("cases must contain distinct known case IDs")
    output_dir.mkdir(parents=True)
    (output_dir / "tasks").mkdir()
    (output_dir / "audit").mkdir()
    entries = []
    for name in chosen:
        for mode in modes:
            entry_id = name + "--" + mode
            task_dir = output_dir / "tasks" / entry_id
            metadata = materialize_case(name, task_dir, seed=seed)
            # The task writer returns targets to the auditor; it does not write
            # these answers anywhere inside the solver's task directory.
            audit = {key: metadata[key] for key in
                     ("case_id", "seed", "kind", "metric", "expected_status", "contract",
                      "audit_actuals", "test_ids", "validation", "input_paths", "provenance")}
            audit_path = output_dir / "audit" / (entry_id + ".json")
            write_json(audit_path, audit)
            entries.append({"id": entry_id, "case_id": name, "mode": mode,
                            "task_path": str(_inside(task_dir, metadata["task_path"])),
                            "task_dir": str(task_dir), "audit_path": str(audit_path),
                            "audit_sha256": file_hash(audit_path),
                            "input_sha256": input_hashes(task_dir, metadata["input_paths"]),
                            "definition": known[name], "status": "preflight"})
    plan = {"format": "unseen-qualification", "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(), "solver": identity,
            "python": str(python), "environment": environment_versions(python),
            "auditor_source": {name: file_hash(ROOT / name) for name in
                               ("qualify_unseen_tasks.py", "unseen_cases.py", "benchmark_cases.py")},
            "catalog_version": CATALOG_VERSION, "seed": seed, "entries": entries,
            "budgets": {"solver_seconds": solver_seconds, "timeout_seconds": timeout_seconds,
                        "total_seconds": total_seconds, "memory_limit_bytes": int(memory_gib * 1024**3),
                        "cpu_threads": 8}, "api_calls": 0, "platform_submissions": 0,
            "boundary": "Separate data paths and sanitized argv/environment; not an OS sandbox"}
    plan["plan_sha256"] = json_hash(plan)
    write_json(output_dir / "manifest.json", plan)
    write_json(output_dir / "summary.json", {"status": "preflight", "plan_sha256": plan["plan_sha256"],
                                           "entries": [{"id": x["id"], "status": "preflight"} for x in entries],
                                           "api_calls": 0})
    return plan


def load_plan(output_dir, *, execution=False):
    output_dir = _path(output_dir)
    plan = read_json(output_dir / "manifest.json")
    digest = plan.pop("plan_sha256", None)
    if plan.get("version") != VERSION or json_hash(plan) != digest:
        raise ValueError("Qualification manifest is modified or unsupported")
    plan["plan_sha256"] = digest
    for entry in plan["entries"]:
        task = _inside(output_dir / "tasks", entry["task_path"])
        if task.parent != _inside(output_dir / "tasks", entry["task_dir"]):
            raise ValueError("Task path differs from its declared directory")
        audit = _inside(output_dir / "audit", entry["audit_path"])
        if file_hash(audit) != entry["audit_sha256"]:
            raise ValueError("Pinned audit answers or contract changed")
        if input_hashes(task.parent, entry["input_sha256"]) != entry["input_sha256"]:
            raise ValueError("Original task inputs changed")
    if execution:
        current = solver_identity(plan["solver"]["root"])
        if (current["sha"] != plan["solver"]["sha"]
                or current["source_fingerprint"] != plan["solver"]["source_fingerprint"]):
            raise ValueError("Solver source changed after preflight")
        if environment_versions(plan["python"]) != plan["environment"]:
            raise ValueError("Solver Python environment changed after preflight")
        if any(file_hash(ROOT / name) != digest for name, digest in plan["auditor_source"].items()):
            raise ValueError("Auditor/generator source changed after preflight")
    return plan


def read_csv(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Missing, linked or oversized CSV")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if (not header or any(not cell.strip() for cell in header)
                or len(set(header)) != len(header)):
            raise ValueError("CSV header is absent, empty or duplicated")
        rows = []
        for index, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {index} has wrong field count")
            rows.append(dict(zip(header, row)))
    return header, rows


def audit_submission(path, audit):
    """Validate exact CSV bytes/IDs/schema, then score without solver imports."""
    try:
        import numpy as np
        from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                                     mean_squared_error, r2_score, roc_auc_score,
                                     precision_recall_fscore_support)

        contract, actual, expected_ids = audit["contract"], audit["audit_actuals"], audit["test_ids"]
        metric = audit["metric"]
        if metric not in METRIC_DIRECTIONS:
            raise ValueError("Auditor does not guess unsupported metrics")
        header, rows = read_csv(path)
        if header != contract["output_columns"]:
            raise ValueError("Submission header/order differs from the frozen sample schema")
        ids = [row[contract["id_column"]] for row in rows]
        if (not ids or ids != expected_ids or len(set(ids)) != len(ids)
                or any(not value.strip() for value in ids) or len(actual) != len(ids)):
            raise ValueError("Submission IDs/count/order differ from original test")
        classes = list(contract["classes"])
        kind = contract["prediction_kind"]
        if kind == "probabilities":
            mapping = contract["probability_columns"]
            values = np.asarray([[float(row[column]) for column, _ in mapping] for row in rows])
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError("Probabilities must be finite in [0, 1]")
            matrix = np.empty((len(rows), len(classes)))
            if len(mapping) == 1:
                positive = contract["positive_class"]
                if len(classes) != 2 or positive not in classes or mapping[0][1] != positive:
                    raise ValueError("Single probability column has ambiguous positive class")
                position = classes.index(positive)
                matrix[:, position], matrix[:, 1-position] = values[:, 0], 1-values[:, 0]
            else:
                supplied = [label for _, label in mapping]
                if len(supplied) != len(classes) or set(supplied) != set(classes):
                    raise ValueError("Probability columns do not cover the declared classes")
                matrix = values[:, [supplied.index(label) for label in classes]]
                if not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-6):
                    raise ValueError("Probability rows do not sum to one")
            ordered = sorted(classes)
            if metric == "LogLoss":
                value = log_loss(actual, matrix[:, [classes.index(x) for x in ordered]], labels=ordered)
            elif metric == "ROC AUC" and len(classes) == 2:
                positive = contract["positive_class"]
                if positive not in classes:
                    raise ValueError("Binary ROC AUC positive class is ambiguous")
                value = roc_auc_score(np.asarray(actual) == positive, matrix[:, classes.index(positive)])
            elif metric == "ROC AUC":
                value = roc_auc_score(actual, matrix[:, [classes.index(x) for x in ordered]],
                                      labels=ordered, multi_class="ovr", average="macro")
            else:
                raise ValueError("Probability output supplied for a label/regression metric")
            predicted = np.asarray(classes)[matrix.argmax(axis=1)]
        elif kind == "labels":
            predicted = [row[contract["target_column"]] for row in rows]
            if not set(predicted) <= set(classes) or not set(actual) <= set(classes):
                raise ValueError("Literal labels are outside the frozen class set")
            if metric == "Accuracy":
                value = accuracy_score(actual, predicted)
            elif metric in {"Macro F1", "Weighted F1"}:
                value = f1_score(actual, predicted, average="macro" if metric == "Macro F1" else "weighted",
                                 zero_division=0)
            else:
                raise ValueError("Label output supplied for an incompatible metric")
        elif kind == "regression":
            predicted = np.asarray([float(row[contract["target_column"]]) for row in rows])
            actual = np.asarray(actual, dtype=float)
            if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
                raise ValueError("Regression values must be finite")
            if metric == "MAE":
                value = mean_absolute_error(actual, predicted)
            elif metric == "R2":
                value = r2_score(actual, predicted)
            elif metric in {"RMSE", "MSE"}:
                value = mean_squared_error(actual, predicted)
                if metric == "RMSE":
                    value = math.sqrt(value)
            else:
                raise ValueError("Regression output supplied for an incompatible metric")
        else:
            raise ValueError("Unsupported prediction representation")
        if not math.isfinite(float(value)):
            raise ValueError("Independent metric is nonfinite")
        result = {"valid": True, "error": None, "rows": len(rows),
                  "submission_sha256": file_hash(path),
                  "metric": {"name": metric, "value": float(value),
                             "higher_is_better": METRIC_DIRECTIONS[metric]}}
        if classes:
            p, r, f, n = precision_recall_fscore_support(actual, predicted, labels=classes, zero_division=0)
            result["per_class"] = [{"class": c, "precision": float(a), "recall": float(b),
                                    "f1": float(d), "support": int(e)}
                                   for c, a, b, d, e in zip(classes, p, r, f, n)]
        return result
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as error:
        return {"valid": False, "error": safe_text(f"{type(error).__name__}: {error}", 700),
                "rows": None, "submission_sha256": None, "metric": None}


def _record_file(path, root):
    path = _inside(root, path)
    return {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}


def inspect_model_evidence(task_dir, published_hash, audit):
    """Require a matching accepted checkpoint plus actual full-data phase files.

    Offline AutoML has a checkpoint receipt rather than v3 candidate receipts.
    We verify both copied CSVs, phase config, durations, original staged input
    hashes and the full-data worker log. This proves the claimed trusted-code
    path, not OS isolation or cryptographic authenticity against the same user.
    """
    root = _path(task_dir)
    evidence, errors = [], []
    for pointer in sorted(root.glob(".agent_runs/*/checkpoint.json")):
        try:
            pointer = _inside(root, pointer)
            value = read_json(pointer)
            if value.get("version") != 1 or value.get("submission_sha256") != published_hash:
                continue
            relative = value.get("directory", "")
            if not re.fullmatch(r"checkpoints/[0-9a-f]{32}", relative):
                raise ValueError("Invalid checkpoint directory")
            directory = _inside(pointer.parent, pointer.parent / relative)
            output, held = directory / "submission.csv", directory / "holdout_predictions.csv"
            if (file_hash(output) != published_hash or file_hash(held) != value.get("holdout_sha256")):
                raise ValueError("Checkpoint CSV hashes differ")
            report_path = pointer.parent / "automl_report.json"
            report = read_json(report_path)
            selected = report.get("search", {}).get("best_candidate")
            if not isinstance(selected, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", selected):
                raise ValueError("Selected candidate ID is invalid")
            score = report.get("metric", {}).get("value")
            if (report.get("output_valid") is not True or type(score) not in {int, float}
                    or not math.isfinite(score)):
                raise ValueError("Model report does not establish an accepted metric")
            for config_path in pointer.parent.glob(selected + "/*/config.json"):
                config_path = _inside(root, config_path)
                config = read_json(config_path)
                if config.get("phase") not in {"seed", "refit"}:
                    continue
                phase_dir = config_path.parent
                if file_hash(phase_dir / "submission.csv") != published_hash:
                    continue
                durations = read_json(phase_dir / "durations.json")
                if not all(type(durations.get(k)) in {int, float} and math.isfinite(durations[k])
                           and durations[k] >= 0 for k in ("fit_seconds", "predict_seconds", "total_seconds")):
                    raise ValueError("Invalid worker duration evidence")
                if durations["fit_seconds"] <= 0:
                    raise ValueError("Worker did not record an actual fit")
                spec = config["task_spec"]
                staged = _inside(root, spec["data_dir"])
                staged_header, staged_train = read_csv(staged / "train.csv")
                del staged_header
                expected_count = len(staged_train)
                logs = [_inside(root, path) for path in (phase_dir / "execution").glob("*.stdout.txt")]
                marker = f"refitting on all {expected_count} rows"
                if not any(marker in path.read_text(encoding="utf-8", errors="replace") for path in logs):
                    raise ValueError("Full-data refit marker is absent from worker evidence")
                original = {Path(name).name: digest for name, digest in audit["input_sha256"].items()}
                if any(file_hash(staged / name) != original[name] for name in
                       ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")):
                    raise ValueError("Full-data staged inputs differ from original task")
                paths = [pointer, report_path, output, held, config_path, phase_dir / "durations.json",
                         phase_dir / "submission.csv", *logs,
                         *(staged / name for name in ("train.csv", "test.csv", "data_dictionary.csv",
                                                      "sample_submission.csv"))]
                evidence.extend(_record_file(path, root) for path in paths)
                return {"kind": "full_model_result", "verified": True,
                        "full_train_rows": expected_count, "phase": config["phase"],
                        "candidate": selected, "artifacts": evidence}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as error:
            errors.append(safe_text(error, 300))
    for path in sorted(root.glob(".agent_runs/*/emergency.json")):
        try:
            path = _inside(root, path)
            value = read_json(path)
            if (value.get("kind") != "target_prior" or value.get("status") != "ready"
                    or value.get("quality_verified") is not False
                    or value.get("submission_sha256") != published_hash):
                continue
            prior = path.parent / "prior.csv"
            if file_hash(prior) != published_hash:
                raise ValueError("Prior receipt and bytes differ")
            original = {Path(name).name: digest for name, digest in audit["input_sha256"].items()}
            if value.get("inputs") != original:
                raise ValueError("Prior input fingerprints differ")
            return {"kind": "target_prior", "verified": True, "quality_verified": False,
                    "artifacts": [_record_file(path, root), _record_file(prior, root)]}
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(safe_text(error, 300))
    return {"kind": "unverified_output", "verified": False, "artifacts": [], "errors": errors[:5]}


def _process_inventory():
    """Reject a host /proc inventory whose PIDs cannot identify our children."""
    try:
        import psutil
        if (sys.platform.startswith("linux")
                and int(Path("/proc/self/stat").read_text(encoding="utf-8").split()[0]) != os.getpid()):
            raise RuntimeError("PID namespace differs")
        psutil.Process(os.getpid()).create_time()
        return psutil, None
    except Exception as error:
        return None, type(error).__name__ + ": process inventory unavailable"


def _stop_tree(process):
    # Only signal a group while its launched root is still alive. Once reaped,
    # its numeric PID/PGID could have been reused by an unrelated process.
    if process.poll() is None:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        if process.poll() is None:
            process.kill()
    process.wait(timeout=5)


class _OwnedProcesses:
    """Retain sampled descendant identities across session and parent changes.

    This is best effort: polling cannot see a child orphaned entirely between
    samples. No global process list is signalled, and PID reuse never grants
    ownership. An unavailable inventory cannot certify descendant cleanup.
    """

    def __init__(self, process, monitor, unavailable_reason=None):
        self.process, self.monitor = process, monitor
        self.owned = {}
        self.unavailable_reason = unavailable_reason
        if monitor is not None:
            try:
                root = monitor.Process(process.pid)
                if root.ppid() != os.getpid():
                    raise RuntimeError("Launched child is outside process inventory")
                self._remember(root)
            except (monitor.Error, OSError, RuntimeError) as error:
                self._unavailable(error)

    def _unavailable(self, error):
        self.unavailable_reason = type(error).__name__ + ": process inventory unavailable"

    def _remember(self, member):
        self.owned.setdefault((member.pid, member.create_time()), member)

    def _alive(self, identity, member):
        try:
            return (member.is_running() and member.create_time() == identity[1]
                    and member.status() != self.monitor.STATUS_ZOMBIE)
        except (self.monitor.NoSuchProcess, self.monitor.ZombieProcess):
            return False

    def discover(self):
        if self.monitor is None:
            return
        for identity, member in tuple(self.owned.items()):
            try:
                if self._alive(identity, member):
                    for child in member.children(recursive=True):
                        try:
                            self._remember(child)
                        except (self.monitor.NoSuchProcess, self.monitor.ZombieProcess):
                            pass
            except (self.monitor.NoSuchProcess, self.monitor.ZombieProcess):
                pass
            except (self.monitor.Error, OSError) as error:
                self._unavailable(error)

    def sample(self):
        self.discover()
        members = []
        for identity, member in tuple(self.owned.items()):
            try:
                if self._alive(identity, member):
                    members.append(member)
            except (self.monitor.Error, OSError) as error:
                self._unavailable(error)
        return members

    def stop(self):
        # Freeze each retained owner, then discover again so ordinary forks
        # cannot escape between the final inventory and termination.
        frozen = set()
        deadline = time.monotonic() + 1
        if self.monitor is not None:
            while True:
                self.discover()
                pending = [(key, member) for key, member in self.owned.items() if key not in frozen]
                if not pending:
                    break
                for identity, member in pending:
                    try:
                        if self._alive(identity, member):
                            member.suspend()
                    except (self.monitor.NoSuchProcess, self.monitor.ZombieProcess):
                        pass
                    except (self.monitor.Error, OSError) as error:
                        self._unavailable(error)
                    frozen.add(identity)
                if time.monotonic() >= deadline:
                    self.unavailable_reason = "Process discovery exceeded cleanup deadline"
                    break
            for identity, member in reversed(tuple(self.owned.items())):
                try:
                    if self._alive(identity, member):
                        member.kill()
                except (self.monitor.NoSuchProcess, self.monitor.ZombieProcess):
                    pass
                except (self.monitor.Error, OSError) as error:
                    self._unavailable(error)
        root_stopped = True
        try:
            _stop_tree(self.process)
        except (OSError, subprocess.SubprocessError) as error:
            root_stopped = False
            self._unavailable(error)
        remaining = self.sample()
        deadline = time.monotonic() + 2
        while remaining and time.monotonic() < deadline:
            time.sleep(.01)
            remaining = self.sample()
        available = self.monitor is not None and self.unavailable_reason is None
        complete = False if remaining or not root_stopped else True if available else None
        return {"inventory_available": available, "tracked_processes": len(self.owned),
                "complete": complete, "unavailable_reason": self.unavailable_reason,
                "scope": "Retained descendants; polling cannot prove absence of unsampled orphans"}


def run_solver(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes):
    """Run one child, monitor compatible process namespaces, stop its tree."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    stdout_path, stderr_path = output_dir / ".stdout.raw", output_dir / ".stderr.raw"
    started = time.monotonic()
    telemetry = {"available": False, "peak_rss_bytes": None, "cpu_seconds": None,
                 "scope": "solver process tree; sampled, not a hard OS limit",
                 "cpu_scope": "Maximum sampled sum of live process CPU; exited children may be omitted"}
    monitor, unavailable_reason = _process_inventory()
    if unavailable_reason:
        telemetry["unavailable_reason"] = unavailable_reason
    timed_out = memory_exceeded = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=sanitized_environment(), stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != "nt")
        owned = _OwnedProcesses(process, monitor, unavailable_reason)
        try:
            while process.poll() is None:
                if monitor is not None:
                    try:
                        members = owned.sample()
                        if members:
                            rss = sum(item.memory_info().rss for item in members)
                            cpu = sum(sum(item.cpu_times()[:2]) for item in members)
                            telemetry.update(available=True, peak_rss_bytes=max(telemetry["peak_rss_bytes"] or 0, rss),
                                             cpu_seconds=max(telemetry["cpu_seconds"] or 0, cpu))
                            if rss > memory_limit_bytes:
                                memory_exceeded = True
                                break
                    except (monitor.Error, OSError) as error:
                        owned._unavailable(error)
                if time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    break
                time.sleep(0.05)
        finally:
            cleanup = owned.stop()
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            returncode = None
            cleanup.update(complete=False, unavailable_reason="Root process did not exit within cleanup deadline")
    logs = {}
    for role, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            logs[role] = safe_text(stream.read().decode("utf-8", errors="replace"))
        (output_dir / (role + ".txt")).write_text(logs[role], encoding="utf-8")
        path.unlink()
    if owned.unavailable_reason:
        telemetry.update(available=False, unavailable_reason=owned.unavailable_reason)
    if not telemetry["available"]:
        telemetry.setdefault("unavailable_reason", "No compatible process telemetry samples")
    return {"argv": argv, "exit_code": returncode, "external_timeout": timed_out, "cleanup": cleanup,
            "memory_limit_exceeded": memory_exceeded, "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": timeout_seconds, "resources": telemetry, **logs}


def _argv(plan, entry, audit):
    python, root, task = plan["python"], plan["solver"]["root"], entry["task_path"]
    if entry["mode"] == "default":
        return [python, str(Path(root) / "solution.py"), task]
    if entry["mode"] == "contract":
        return [python, "-c", CONTRACT_CODE, root, task]
    args = [python, str(Path(root) / "solution.py"), task, "--mode", "offline",
            "--time-budget", str(plan["budgets"]["solver_seconds"])]
    validation = audit.get("validation") or {}
    if validation.get("strategy") in {"random", "grouped", "time"}:
        args += ["--validation", validation["strategy"]]
        if validation["strategy"] == "time":
            args += ["--time-column", validation["time_column"]]
    return args


def _semantic_refusal(entry, execution):
    text = ""
    # Match the actual terminal diagnostic, never traceback filenames, source
    # snippets or the task directory name (e.g. malformed_csv--contract).
    for line in reversed(execution.get("stderr", "").splitlines()):
        try:
            structured = json.loads(line)
        except (ValueError, TypeError):
            structured = None
        if isinstance(structured, dict) and isinstance(structured.get("contract_error"), dict):
            error = structured["contract_error"]
            if error.get("type") in {"ValueError", "ParserError", "Error"} and isinstance(error.get("message"), str):
                text = error["message"]
            break
        terminal = re.match(r"^(?:Error|ValueError|ParserError):\s*(.+)$", line.strip())
        if terminal:
            text = terminal[1]
            break
    if not text:
        return False
    if re.search(r"ModuleNotFoundError|ImportError|No module named|PermissionError|FileNotFoundError|"
                 r"out of memory|could not launch", text, re.IGNORECASE):
        return False
    patterns = {
        "missing_metric": r"metric|метрик",
        "conflicting_metric": r"conflict|contradict|противореч|несколько",
        "ambiguous_positive": r"positive|положитель|probabilit|вероятност",
        "ambiguous_probability_mapping": r"probabilit|class|колонк|вероятност|класс",
        "duplicate_test_ids": r"duplicat|повтор|уникаль|\bIDs?\b.*\bunique\b",
        "missing_train_target": r"target|response|цель|таргет|missing|отсутств",
        "test_schema_mismatch": r"feature|schema|признак|столб|column|missing",
        "malformed_csv": r"csv|token|field|quot|pars|строк|unexpected end",
        "missing_test_id": r"\bid\b|empty|missing|пуст",
        "task_type_conflict": r"type|тип|conflict|противореч|classification|regression|metric",
    }
    pattern = patterns.get(entry.get("case_id"))
    return bool(pattern and re.search(pattern, text, re.IGNORECASE))


def _classify(entry, execution, checked, provenance, unchanged):
    expected_refusal = entry["definition"]["expected_status"] == "refusal"
    if not unchanged:
        return "input_mutation", False
    if execution["external_timeout"]:
        return "external_timeout", False
    if execution["memory_limit_exceeded"]:
        return "external_memory_limit", False
    if expected_refusal:
        if execution["exit_code"] != 0 and _semantic_refusal(entry, execution):
            return "expected_refusal", True
        if execution["exit_code"] != 0:
            return "solver_error", False
        return "unexpected_acceptance", False
    if execution["exit_code"] != 0:
        return "solver_error", False
    if entry["mode"] == "contract":
        return ("contract_valid", True) if checked.get("valid") else ("contract_mismatch", False)
    if not checked.get("valid"):
        return "invalid_output", False
    if provenance["kind"] == "target_prior":
        return "valid_prior_fallback", True
    if provenance["kind"] == "full_model_result":
        return "full_result", True
    return "valid_output_unverified", False


def execute_run(output_dir):
    output_dir = _path(output_dir)
    plan = load_plan(output_dir, execution=True)
    with (output_dir / "run_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"plan_sha256": plan["plan_sha256"], "started_at": datetime.now(timezone.utc).isoformat()}, stream)
    started = time.monotonic()
    report = {"format": "unseen-qualification-result", "version": VERSION, "status": "running",
              "plan_sha256": plan["plan_sha256"], "results": [], "api_calls": 0,
              "quality_scope": "External fixture answers; not a platform/private score"}
    aborted_mutation = aborted_cleanup = False
    for entry in plan["entries"]:
        if aborted_mutation:
            report["results"].append({"id": entry["id"], "case_id": entry["case_id"], "mode": entry["mode"],
                                      "status": "skipped_input_mutation", "qualified": False})
            continue
        if aborted_cleanup:
            report["results"].append({"id": entry["id"], "case_id": entry["case_id"], "mode": entry["mode"],
                                      "status": "skipped_cleanup_unverified", "qualified": False})
            continue
        remaining = plan["budgets"]["total_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            report["results"].append({"id": entry["id"], "case_id": entry["case_id"], "mode": entry["mode"],
                                      "status": "skipped_total_budget", "qualified": False})
            continue
        load_plan(output_dir, execution=True)
        if (entry["mode"] == "offline"
                and min(remaining, plan["budgets"]["timeout_seconds"]) < plan["budgets"]["solver_seconds"]):
            inventory, reason = _process_inventory()
            if inventory is None:
                result = {"id": entry["id"], "case_id": entry["case_id"], "mode": entry["mode"],
                          "status": "skipped_cleanup_unavailable", "qualified": False,
                          "reason": "Offline budget would be externally censored without descendant inventory",
                          "cleanup_capability": {"inventory_available": False, "unavailable_reason": reason}}
                report["results"].append(result)
                write_json(output_dir / ("result-" + entry["id"] + ".json"), result)
                write_json(output_dir / "summary.json", report)
                continue
        task_dir = Path(entry["task_dir"])
        if (task_dir / "submission.csv").exists():
            raise ValueError("A stale submission exists before solver invocation")
        audit = read_json(entry["audit_path"])
        audit["input_sha256"] = entry["input_sha256"]
        argv = _argv(plan, entry, audit)
        print(f"[{len(report['results'])+1}/{len(plan['entries'])}] {entry['id']}", flush=True)
        try:
            execution = run_solver(argv, task_dir, output_dir / ("execution-" + entry["id"]),
                                   min(remaining, plan["budgets"]["timeout_seconds"]),
                                   plan["budgets"]["memory_limit_bytes"])
        except (OSError, subprocess.SubprocessError) as error:
            execution = {"argv": argv, "exit_code": None, "external_timeout": False,
                         "memory_limit_exceeded": False, "elapsed_seconds": None,
                         "stdout": "", "stderr": safe_text(error, 700),
                         "launch_error": type(error).__name__, "resources": {"available": False}}
        input_error = None
        try:
            unchanged = input_hashes(task_dir, entry["input_sha256"]) == entry["input_sha256"]
        except (OSError, ValueError) as error:
            unchanged = False
            input_error = safe_text(error, 500)
        checked = audit_submission(task_dir / "submission.csv", audit) if entry["mode"] != "contract" else {}
        provenance = {"kind": "not_applicable", "verified": False, "artifacts": []}
        if checked.get("valid"):
            provenance = inspect_model_evidence(task_dir, checked["submission_sha256"], audit)
        if entry["mode"] == "contract" and execution["exit_code"] == 0:
            try:
                contract = json.loads(execution["stdout"].splitlines()[-1])
                if not isinstance(contract, dict):
                    raise ValueError("Contract probe JSON must be an object")
                expected = audit["contract"]
                expected_metric = {"name": audit["metric"],
                                   "higher_is_better": METRIC_DIRECTIONS[audit["metric"]],
                                   "task_type": "regression" if expected["prediction_kind"] == "regression" else "classification",
                                   "prediction_kind": expected["prediction_kind"]}
                if (any(contract.get(k) != expected[k] for k in ("id_column", "target_column", "output_columns",
                                                                 "classes", "probability_columns", "positive_class"))
                        or contract.get("metric") != expected_metric):
                    checked = {"valid": False, "error": "Loaded task contract differs from frozen expected semantics"}
                else:
                    checked = {"valid": True, "contract_only": True}
            except (ValueError, IndexError, KeyError):
                checked = {"valid": False, "error": "Contract probe did not return valid JSON"}
        status, qualified = _classify(entry, execution, checked, provenance, unchanged)
        aborted_mutation = not unchanged
        aborted_cleanup = (bool(execution["external_timeout"] or execution["memory_limit_exceeded"])
                           and execution.get("cleanup", {}).get("complete") is not True)
        result = {"id": entry["id"], "case_id": entry["case_id"], "mode": entry["mode"],
                  "status": status, "qualified": qualified, "inputs_unchanged": unchanged,
                  "input_check_error": input_error,
                  "execution": execution, "audit": checked, "provenance": provenance,
                  "censored": bool(execution["external_timeout"]),
                  "timeout_interpretation": "External benchmark censoring; not proof of exceeding the platform's 3h"
                                            if execution["external_timeout"] else None}
        if checked.get("valid") and entry["mode"] != "contract":
            raw = output_dir / ("predictions-" + entry["id"] + ".csv")
            raw.write_bytes((task_dir / "submission.csv").read_bytes())
            result["raw_predictions"] = {"path": raw.name, "sha256": file_hash(raw)}
        write_json(output_dir / ("result-" + entry["id"] + ".json"), result)
        report["results"].append(result)
        write_json(output_dir / "summary.json", report)
    report.update(status=("aborted_input_mutation" if aborted_mutation else
                          "aborted_cleanup_unverified" if aborted_cleanup else "completed"),
                  elapsed_seconds=time.monotonic()-started)
    report["qualification_passed"] = all(row.get("qualified", False) for row in report["results"])
    report["counts"] = {status: sum(row["status"] == status for row in report["results"])
                        for status in sorted({row["status"] for row in report["results"]})}
    report["full_run_metrics"] = sorted({row["audit"]["metric"]["name"] for row in report["results"]
                                         if row["status"] == "full_result"})
    write_json(output_dir / "summary.json", report)
    return report


def re_audit(output_dir):
    output_dir = _path(output_dir)
    plan = load_plan(output_dir)
    report = read_json(output_dir / "summary.json")
    entries = {entry["id"]: entry for entry in plan["entries"]}
    results = []
    for row in report.get("results", []):
        if "raw_predictions" not in row:
            results.append({"id": row["id"], "status": "not_scored", "original_status": row["status"]})
            continue
        entry = entries[row["id"]]
        raw = _inside(output_dir, output_dir / row["raw_predictions"]["path"])
        if file_hash(raw) != row["raw_predictions"]["sha256"]:
            raise ValueError("Saved raw predictions changed")
        for artifact in row["provenance"].get("artifacts", []):
            path = _inside(entry["task_dir"], Path(entry["task_dir"]) / artifact["path"])
            if file_hash(path) != artifact["sha256"]:
                raise ValueError("Pinned model/prior evidence changed")
        checked = audit_submission(raw, read_json(entry["audit_path"]))
        if checked != row["audit"]:
            raise ValueError("Independent re-audit differs from saved scores/CSV evidence")
        results.append({"id": row["id"], "status": "verified", "metric": checked["metric"]})
    result = {"status": "verified", "plan_sha256": plan["plan_sha256"], "results": results,
              "models_refitted": 0, "api_calls": 0,
              "verified_predictions": sum(row["status"] == "verified" for row in results),
              "not_scored": sum(row["status"] == "not_scored" for row in results)}
    write_json(output_dir / "reaudit.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", type=Path, help="Execute an existing untouched preflight directory")
    action.add_argument("--re-audit", type=Path, help="Recalculate saved predictions without solver execution")
    parser.add_argument("--solver-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("offline", "default", "contract"), default=["offline"])
    parser.add_argument("--seed", type=int, default=81221)
    parser.add_argument("--solver-seconds", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--total-seconds", type=float, default=3600)
    parser.add_argument("--memory-gib", type=float, default=8)
    args = parser.parse_args(argv)
    try:
        if args.run:
            result = execute_run(args.run)
        elif args.re_audit:
            result = re_audit(args.re_audit)
        else:
            if args.solver_root is None:
                parser.error("--solver-root is required for preflight")
            output = args.out_dir or ROOT / "data/experiments/night_unseen" / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
            plan = prepare_run(args.solver_root, output, python=args.python, cases=args.cases,
                               modes=args.modes, seed=args.seed, solver_seconds=args.solver_seconds,
                               timeout_seconds=args.timeout, total_seconds=args.total_seconds,
                               memory_gib=args.memory_gib)
            result = {"status": "preflight", "output_dir": str(output), "entries": len(plan["entries"]),
                      "plan_sha256": plan["plan_sha256"], "api_calls": 0}
        print(json.dumps(result if result.get("status") == "preflight" else
                         {key: result.get(key) for key in ("status", "qualification_passed", "counts", "full_run_metrics", "api_calls")},
                         ensure_ascii=False, allow_nan=False))
        return 1 if result.get("qualification_passed") is False else 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        print("Qualification error: " + safe_text(error, 700), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
