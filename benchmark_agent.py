"""Developer-only, API-free audit of AutoML and research failure recovery.

Each task/mode runs in a fresh process and receives only its task and four CSV
inputs. Audit answers belong to this harness, outside the solver task directory.
This ordinary input boundary is not an operating-system security sandbox.
Synthetic audit scores are not public/private contest scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import time
from uuid import uuid4

from agent.execution import run_python
from agent.metrics import get_metric, score_predictions
from agent.task_spec import load_task
from benchmark_cases import ALL_CASE_NAMES, CASE_NAMES, make_case, write_case
from offline_baseline import read_csv


ROOT = Path(__file__).resolve().parent
FULL_SEEDS = (42, 137, 2026)
DEFAULT_MODES = ("seed", "automl")
MODES = (*DEFAULT_MODES, "research_api_failure")
INPUT_NAMES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
MAX_MEMORY_MIB = 28 * 1024


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _json_hash(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _git(root: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *arguments], check=True,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_provenance(root: Path) -> dict:
    """Read known source bytes only; never load another checkout's Python/.env."""
    root = Path(root).resolve()
    required = ("solution.py", "offline_baseline.py", "validation.py", "agent/__init__.py",
                "agent/automl.py")
    for name in required:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Solver checkout is missing a regular {name}")
    files = {root / name for name in required}
    for name in ("requirements.txt", "requirements.platform.txt"):
        if (root / name).is_file():
            files.add(root / name)
    for path in (root / "agent").rglob("*.py"):
        if "__pycache__" not in path.parts:
            files.add(path)
    digest, entries = hashlib.sha256(), []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Solver sources must be regular files inside the selected checkout")
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + str(len(content)).encode("ascii") + b"\0" + content)
        entries.append({"path": relative, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    status = _git(root, ["status", "--porcelain", "--untracked-files=normal"])
    return {"root": str(root), "commit": _git(root, ["rev-parse", "HEAD"]),
            "worktree_dirty": None if status is None else bool(status),
            "source_sha256": digest.hexdigest(), "files": entries}


def input_fingerprint(task: Path) -> str:
    digest = hashlib.sha256()
    for name, path in [("task.md", task), *[("inputs/" + name, task.parent / "inputs" / name)
                                            for name in INPUT_NAMES]]:
        content = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + str(len(content)).encode("ascii") + b"\0" + content)
    return digest.hexdigest()


def _driver(solver_root: Path, task: Path, mode: str, budget: float,
            validation: dict, solver_options: dict) -> str:
    # Audit labels/paths are deliberately absent. A fresh interpreter loads the
    # selected checkout, not modules already imported by the benchmark host.
    if mode == "research_api_failure":
        return _research_driver(solver_root, task, budget, solver_options)
    return (
        "import time\nstarted = time.monotonic()\nimport sys\n"
        f"sys.path.insert(0, {str(solver_root)!r})\n"
        "from pathlib import Path\nfrom agent import automl\n"
        + ("automl.candidate_plan = lambda classification: []\n"
           "if hasattr(automl, 'metric_candidates'):\n"
           "    automl.metric_candidates = lambda *args, **kwargs: []\n" if mode == "seed" else "")
        + f"remaining = {budget!r} - (time.monotonic() - started)\n"
        "if remaining <= 0:\n    raise TimeoutError('Worker startup exhausted the benchmark budget')\n"
        f"automl.run_automl(Path({str(task)!r}), time_budget=remaining, "
        f"validation={validation!r}, output_dir=Path({str(task.parent / 'solver_output')!r}), "
        f"**{solver_options!r})\n"
    )


def _research_driver(solver_root: Path, task: Path, budget: float, solver_options: dict) -> str:
    """Replace only the provider client; all research/model code stays real."""
    return (
        "import time\nstarted = time.monotonic()\nimport sys\n"
        f"sys.path.insert(0, {str(solver_root)!r})\n"
        "from pathlib import Path\nimport hashlib\nimport json\nimport httpx\n"
        "from openai import APIConnectionError\n"
        "from agent.research_v3 import run_agent_v3\n"
        "class FailingClient:\n"
        "    def __init__(self):\n"
        "        self.calls = 0\n        self.chat = self\n        self.completions = self\n"
        "    def with_options(self, **kwargs):\n        return self\n"
        "    def create(self, **kwargs):\n"
        "        self.calls += 1\n"
        "        request = httpx.Request('POST', 'https://offline.invalid/v1/chat/completions')\n"
        "        raise APIConnectionError(request=request) from httpx.ConnectError('Synthetic offline failure')\n"
        "client = FailingClient()\n"
        f"task = Path({str(task)!r})\n"
        f"output = Path({str(task.parent / 'solver_output')!r})\noutput.mkdir()\n"
        f"remaining = {budget!r} - (time.monotonic() - started)\n"
        "if remaining <= 0:\n    raise TimeoutError('Worker startup exhausted the benchmark budget')\n"
        "evidence = {'version': 1, 'mode': 'research_api_failure', 'external_api_calls': 0, "
        "'simulated_api_calls': 0, 'requested_time_budget_seconds': remaining, 'run_report': None}\n"
        "try:\n"
        f"    run_dir = run_agent_v3(task, client, time_budget=remaining, **{solver_options!r})\n"
        "    report_path = (run_dir / 'run.json').resolve()\n"
        "    evidence['run_report'] = report_path.relative_to(task.parent.resolve()).as_posix()\n"
        "    evidence['run_report_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()\n"
        "finally:\n"
        "    evidence['simulated_api_calls'] = client.calls\n"
        "    (output / 'research_benchmark.json').write_text(json.dumps(evidence, allow_nan=False), encoding='utf-8')\n"
    )


def audit_submission(task: Path, actuals: tuple) -> dict:
    """Validate the exact published bytes, then recompute an independent score."""
    try:
        spec = load_task(task)
        expected = read_csv(spec.data_dir / "test.csv")[spec.id_column].tolist()
        submission = read_csv(task.parent / "submission.csv")
        spec.validate_frame(submission, expected)
        if len(actuals) != len(expected):
            raise ValueError("Host audit answers differ from the audit row count")
        if spec.prediction_kind == "probabilities":
            columns, labels = zip(*spec.probability_columns)
            predictions = submission.loc[:, list(columns)].astype(float).to_numpy()
            prediction_classes = tuple(labels)
        else:
            predictions = submission[spec.target_column].tolist()
            prediction_classes = None
        value = score_predictions(spec.metric, actuals, predictions, classes=spec.classes,
                                  prediction_classes=prediction_classes, positive_class=spec.positive_class)
        content = (task.parent / "submission.csv").read_bytes()
        return {"valid": True, "error": None, "rows": len(expected),
                "submission_sha256": hashlib.sha256(content).hexdigest(),
                "metric": {"name": spec.metric.name, "value": value,
                           "higher_is_better": spec.metric.higher_is_better}}
    except (OSError, ValueError, TypeError, KeyError) as error:
        # These files are produced by our fixed offline solver; no credentials
        # are inherited and no arbitrary provider response is copied here.
        return {"valid": False, "error": f"{type(error).__name__}: {error}"[:500],
                "rows": None, "submission_sha256": None, "metric": None}


def _solver_summary(directory: Path) -> dict | None:
    path = directory / "solver_output" / "automl_report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            return None
        search = report.get("search") or {}
        if not isinstance(search, dict) or not isinstance(search.get("candidates", []), list):
            return None
        return {"api": report.get("api"), "model": report.get("model"),
                "metric": report.get("metric"), "output_valid": report.get("output_valid"),
                "search_status": search.get("status"), "best_candidate": search.get("best_candidate"),
                "candidate_count": len(search.get("candidates", [])),
                "inner_split": report.get("inner_split"), "split": report.get("split"),
                "search_policy": report.get("search_policy")}
    except (OSError, ValueError, TypeError):
        return None


def _research_summary(directory: Path) -> dict | None:
    """Read only the exact report returned by this driver, never a glob result."""
    try:
        evidence_path = directory / "solver_output" / "research_benchmark.json"
        if evidence_path.is_symlink():
            return None
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            return None
        relative = evidence.get("run_report")
        if not isinstance(relative, str) or not re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", relative):
            return None
        report_path = directory / relative
        run_dir = report_path.parent.resolve()
        if (report_path.is_symlink() or report_path.parent.is_symlink() or report_path.parent.parent.is_symlink()
                or not run_dir.is_relative_to(directory.resolve())
                or report_path.resolve().parent != run_dir):
            return None
        content = report_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != evidence.get("run_report_sha256"):
            return None
        report = json.loads(content)
        if not isinstance(report, dict):
            return None
        receipt_valid = False
        receipts = report.get("candidate_receipts")
        selected = report.get("published_recipe")
        frozen = report.get("frozen_" + str(selected))
        if isinstance(receipts, dict) and selected in {"baseline", "candidate"} and isinstance(frozen, dict):
            receipt_valid = all(receipts.get(key) == frozen.get(key) and
                                isinstance(frozen.get(key), str) and
                                re.fullmatch("[0-9a-f]{64}", frozen[key])
                                for key in ("code_hash", "recipe_hash"))
            for key, phase, subfolder in (("outer", "outer_validation", "outer_evaluations"),
                                          ("refit", "full_refit", "full_refit")):
                raw = receipts.get(key)
                if not isinstance(raw, str):
                    receipt_valid = False
                    break
                path = Path(raw)
                if not path.is_absolute():
                    path = directory / path
                if path.is_symlink() or not path.resolve().is_relative_to(run_dir / subfolder):
                    receipt_valid = False
                    break
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(receipt, dict):
                    receipt_valid = False
                    break
                prediction_path = path.parent / "predictions.csv"
                if prediction_path.is_symlink() or not prediction_path.resolve().is_relative_to(run_dir):
                    receipt_valid = False
                    break
                prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                if (receipt.get("version") != 3 or receipt.get("phase") != phase
                        or any(receipt.get(field) != frozen.get(field) for field in ("code_hash", "recipe_hash"))
                        or receipt.get("output_hash") != prediction_hash
                        or (key == "refit" and prediction_hash != hashlib.sha256(
                            (directory / "submission.csv").read_bytes()).hexdigest())):
                    receipt_valid = False
        fields = ("research_protocol_version", "status", "reason", "stage", "stages", "api",
                  "experiments", "successful_experiments", "local_experiments", "local_successful_experiments",
                  "local_research", "baseline_search", "evaluations", "publication", "output_check",
                  "time_budget_seconds", "elapsed_seconds", "time_allocation", "published_recipe", "quality_options",
                  "host_validation", "best_metric")
        result = {key: report.get(key) for key in fields}
        result.update(benchmark_evidence=evidence, receipt_checks_valid=bool(receipt_valid),
                      api_errors=[event.get("details") for event in report.get("events", [])
                                  if isinstance(event, dict) and event.get("kind") == "api_error"])
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _research_contract(summary: dict | None, budget: float, validation: dict | None = None,
                       metric_name: str | None = None) -> list[str]:
    """Explain incomplete coverage without calling a valid fallback an LLM success."""
    if summary is None:
        return ["missing_or_invalid_current_research_report"]
    try:
        return _research_contract_errors(summary, budget, validation, metric_name)
    except (AttributeError, TypeError, ValueError, IndexError):
        return ["malformed_research_contract"]


def _research_contract_errors(summary: dict, budget: float, validation: dict | None,
                              metric_name: str | None) -> list[str]:
    errors = []
    evidence = summary.get("benchmark_evidence") or {}
    if (evidence.get("version") != 1 or evidence.get("mode") != "research_api_failure"
            or type(evidence.get("external_api_calls")) is not int or evidence["external_api_calls"] != 0
            or type(evidence.get("simulated_api_calls")) is not int or evidence["simulated_api_calls"] != 1):
        errors.append("simulated_client_contract")
    api = summary.get("api") or {}
    if (type(api.get("attempts")) is not int or api.get("attempts") != 1
            or type(api.get("responses")) is not int or api.get("responses") != 0
            or summary.get("experiments") != 0 or summary.get("successful_experiments") != 0
            or len(summary.get("api_errors") or []) != 1
            or summary["api_errors"][0] != {"error_type": "APIConnectionError", "cause_type": "ConnectError",
                                           "transport_reason": "connect_error"}):
        errors.append("api_failure_not_observed_once")
    if (summary.get("research_protocol_version") != 3 or summary.get("status") != "fallback"
            or summary.get("reason") != "api_error" or summary.get("stage") != "result"):
        errors.append("research_failure_recovery_not_completed")
    expected_stages = ("preparation", "inner_research", "candidate_frozen", "outer_validation", "full_refit", "result")
    stages = summary.get("stages")
    if (not isinstance(stages, list) or [item.get("id") for item in stages if isinstance(item, dict)] != list(expected_stages)
            or any(not isinstance(item, dict) or item.get("status") != "completed" for item in stages)):
        errors.append("terminal_stages_incomplete")
    startup = summary.get("baseline_search") or {}
    execution = startup.get("execution") or {}
    if (startup.get("status") != "searched" or execution.get("exit_code") != 0
            or execution.get("timed_out") or execution.get("memory_limit_exceeded") or execution.get("error")):
        errors.append("real_startup_search_not_completed")
    if validation is not None:
        split = (summary.get("host_validation") or {}).get("split") or {}
        if split.get("strategy") != validation.get("strategy"):
            errors.append("automatic_validation_policy_mismatch")
        elif validation.get("strategy") == "time" and split.get("time_column") != validation.get("time_column"):
            errors.append("automatic_time_column_mismatch")
    local = summary.get("local_research") or {}
    attempted, verified = local.get("attempted"), local.get("verified")
    if (type(attempted) is not int or type(verified) is not int or not 1 <= attempted <= 12
            or not 1 <= verified <= attempted or local.get("failed") != attempted - verified
            or local.get("status") != "completed" or local.get("origin") != "host_local"
            or local.get("scope") != "inner_selection" or local.get("api_requests") != 0
            or summary.get("local_experiments", 0) != attempted
            or (summary.get("local_successful_experiments") or 0) != verified):
        errors.append("local_continuation_not_exercised")
    publication = summary.get("publication") or {}
    if (publication.get("new_result_published") is not True or publication.get("reason") != "api_error"
            or (summary.get("output_check") or {}).get("valid") is not True
            or summary.get("receipt_checks_valid") is not True):
        errors.append("verified_full_refit_not_published")
    final_check = (summary.get("evaluations") or {}).get("final_check") or {}
    selected = final_check.get("selected")
    selected_metric = final_check.get(selected) if selected in {"baseline", "candidate"} else None
    metric_valid = False
    if isinstance(selected_metric, dict):
        expected_metric = get_metric(metric_name if metric_name is not None else selected_metric)
        value = selected_metric.get("value")
        metric_valid = (selected_metric.get("name") == expected_metric.name
                        and selected_metric.get("higher_is_better") is expected_metric.higher_is_better
                        and type(value) in {int, float} and math.isfinite(value)
                        and summary.get("best_metric") == selected_metric)
    if (final_check.get("scope") != "final_check" or not metric_valid
            or selected != summary.get("published_recipe") or publication.get("recipe") != selected):
        errors.append("host_outer_evaluation_not_verified")
    requested, actual = evidence.get("requested_time_budget_seconds"), summary.get("time_budget_seconds")
    if (type(requested) not in {int, float} or not math.isfinite(requested) or not 0 < requested <= budget
            or actual != requested or (summary.get("time_allocation") or {}).get("time_budget_seconds") != requested):
        errors.append("original_deadline_not_preserved")
    return errors


def run_trial(case, mode: str, directory: Path, *, solver_root: Path, budget: float,
              memory_mib: int, solver_options: dict, expected_source_hash: str) -> dict:
    task = write_case(case, directory)
    before = input_fingerprint(task)
    row = {"case": case.name, "seed": case.seed, "mode": mode,
           "task": str(task), "input_sha256": before, "audit_sha256": _json_hash(case.audit_actuals),
           "metric": case.metric, "validation": case.validation, "time_budget_seconds": budget,
           "memory_limit_bytes": memory_mib * 1024 ** 2, "solver_options": solver_options,
           "search_disabled_by_harness": mode == "seed", "status": "running"}
    started = time.monotonic()
    try:
        execution = run_python(_driver(solver_root, task, mode, budget, case.validation, solver_options),
                               task.parent, task.parent / "execution", 0, timeout=budget,
                               max_memory_bytes=memory_mib * 1024 ** 2)
    except (OSError, ValueError, RuntimeError) as error:
        execution = {"exit_code": None, "timed_out": False, "memory_limit_exceeded": False,
                     "error": f"{type(error).__name__}: {error}"[:500], "resources": {"available": False}}
    row.update(elapsed_seconds=round(time.monotonic() - started, 6), execution=execution)
    try:
        inputs_unchanged = input_fingerprint(task) == before
    except OSError:
        inputs_unchanged = False
    try:
        sources_unchanged = source_provenance(solver_root)["source_sha256"] == expected_source_hash
    except (OSError, ValueError):
        sources_unchanged = False
    row.update(inputs_unchanged=inputs_unchanged, sources_unchanged=sources_unchanged,
               solver_report=(_research_summary(task.parent) if mode == "research_api_failure"
                              else _solver_summary(task.parent)),
               audit=audit_submission(task, case.audit_actuals))
    if mode == "research_api_failure":
        row["research_contract_errors"] = _research_contract(row["solver_report"], budget, case.validation, case.metric)
    api = (row["solver_report"] or {}).get("api")
    if not sources_unchanged:
        row["status"] = "source_changed"
    elif not inputs_unchanged:
        row["status"] = "input_changed"
    elif execution.get("memory_limit_exceeded"):
        row["status"] = "memory_limit"
    elif execution.get("timed_out"):
        row["status"] = "timeout"
    elif execution.get("exit_code") != 0 or execution.get("error"):
        row["status"] = "solver_failed"
    elif mode == "research_api_failure" and row["research_contract_errors"]:
        row["status"] = "unverified_research_contract"
    elif mode != "research_api_failure" and api != {"used": False, "calls": 0}:
        row["status"] = "unverified_offline_contract"
    elif not row["audit"]["valid"]:
        row["status"] = "invalid_output"
    else:
        row["status"] = "completed"
    if not inputs_unchanged or not sources_unchanged:
        row["audit"]["metric"] = None
    return row


def paired_comparisons(left: list[dict], right: list[dict], *, kind: str) -> dict:
    """Pair only the same task/data/budget; never average unlike metrics."""
    def key(row):
        return (row.get("case"), row.get("seed"), row.get("mode") if kind == "reference" else None)

    reference = {key(row): row for row in left}
    if len(reference) != len(left) or len({key(row) for row in right}) != len(right):
        raise ValueError("Duplicate trial keys cannot be compared")
    pairs = []
    for row in right:
        other = reference.get(key(row))
        if other is None:
            continue
        pair = {"case": row["case"], "seed": row["seed"], "control_mode": other["mode"],
                "candidate_mode": row["mode"], "comparable": False}
        comparable_fields = ("input_sha256", "audit_sha256", "metric", "validation",
                             "time_budget_seconds", "memory_limit_bytes")
        if any(row.get(field) != other.get(field) for field in comparable_fields):
            pair["reason"] = "contract_or_budget_mismatch"
        elif row.get("status") != "completed" or other.get("status") != "completed":
            pair["reason"] = "incomplete_trial"
        else:
            before = other["audit"]["metric"]
            after = row["audit"]["metric"]
            if before["name"] != after["name"] or before["higher_is_better"] != after["higher_is_better"]:
                pair["reason"] = "metric_mismatch"
            else:
                raw_delta = after["value"] - before["value"]
                improvement = raw_delta if after["higher_is_better"] else -raw_delta
                pair.update(comparable=True, metric=after["name"], control=before["value"],
                            candidate=after["value"], delta=raw_delta, improvement=improvement,
                            outcome="win" if improvement > 1e-12 else "loss" if improvement < -1e-12 else "tie",
                            elapsed_delta_seconds=row["elapsed_seconds"] - other["elapsed_seconds"])
        pairs.append(pair)
    counts = Counter(pair.get("outcome", "unpaired") for pair in pairs)
    return {"kind": kind, "pairs": pairs, "counts": dict(counts),
            "aggregate_metric": None, "note": "Scores with different metrics are never averaged; not a contest score."}


def _versions() -> dict:
    values = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "psutil"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def benchmark(output: Path, *, solver_root: Path = ROOT, cases=CASE_NAMES, seeds=(42,),
              modes=DEFAULT_MODES, time_budget=45.0, memory_mib=4096, solver_options=None,
              compare_report: Path | None = None) -> dict:
    output, solver_root = Path(output).resolve(), Path(solver_root).resolve()
    solver_options = dict(solver_options or {})
    if type(time_budget) not in {int, float} or not math.isfinite(time_budget) or not 0 < time_budget <= 10500:
        raise ValueError("time_budget must be finite and in (0, 10500]")
    if type(memory_mib) is not int or not 1 <= memory_mib <= MAX_MEMORY_MIB:
        raise ValueError(f"memory_mib must be 1..{MAX_MEMORY_MIB}")
    cases, seeds, modes = tuple(cases), tuple(seeds), tuple(modes)
    if (not cases or len(set(cases)) != len(cases) or not set(cases) <= set(ALL_CASE_NAMES)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1 for seed in seeds)
            or not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES)):
        raise ValueError("Provide unique supported cases/modes and integer seeds in 0..2**32-1")
    if set(solver_options) - {"stability_splits", "metric_recipes"}:
        raise ValueError("Unknown solver options")
    if "stability_splits" in solver_options and (type(solver_options["stability_splits"]) is not int
                                                or solver_options["stability_splits"] not in {1, 2, 3}):
        raise ValueError("stability_splits must be 1, 2 or 3")
    if "metric_recipes" in solver_options and type(solver_options["metric_recipes"]) is not bool:
        raise ValueError("metric_recipes must be boolean")
    source = source_provenance(solver_root)
    reference = None
    if compare_report is not None:
        reference = json.loads(Path(compare_report).read_text(encoding="utf-8"))
        if reference.get("kind") != "offline_agent_audit" or reference.get("version") != 1:
            raise ValueError("Reference report does not use this benchmark contract")
    if output.is_relative_to(solver_root) and not output.is_relative_to(solver_root / "data"):
        raise ValueError("Inside the solver checkout, put benchmark outputs under gitignored data/")
    output.mkdir(parents=True, exist_ok=False)
    audit_dir, runs_dir, host_dir = output / "host_audit", output / "runs", output / "host_source"
    audit_dir.mkdir()
    runs_dir.mkdir()
    host_dir.mkdir()
    host_bytes = {name: (ROOT / name).read_bytes() for name in ("benchmark_agent.py", "benchmark_cases.py")}
    for name, content in host_bytes.items():
        (host_dir / name).write_bytes(content)
    report = {"version": 1, "kind": "offline_agent_audit", "synthetic_only": True,
              "purpose": "Independent synthetic audit, not public/private contest scores or LLM quality",
              "input_boundary": "Audit answers stay outside task directories and are not passed to the solver; not an OS sandbox",
              "started_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "harness_sha256": hashlib.sha256(host_bytes["benchmark_agent.py"]).hexdigest(),
              "case_generator_sha256": hashlib.sha256(host_bytes["benchmark_cases.py"]).hexdigest(),
              "mode_definition": {"seed": "Full run_automl lifecycle with all exploration/metric recipe generators disabled",
                                  "automl": "Unmodified run_automl exploration for the selected solver options",
                                  "research_api_failure": "Real run_agent_v3 with real startup, automatic validation, local queue, outer/refit; only the provider client is replaced by one typed connection failure. External API calls remain zero."},
              "environment": _versions(), "cases": list(cases), "seeds": list(seeds), "modes": list(modes),
              "time_budget_seconds_per_trial": float(time_budget), "memory_limit_mib": memory_mib,
              "solver_options": solver_options, "api": {"used": False, "calls": 0},
              "status": "running", "trials": []}
    path = output / "report.json"
    _write_json(path, report)
    aborted = False
    for case_name in cases:
        for seed in seeds:
            case = make_case(case_name, seed)
            _write_json(audit_dir / f"{case_name}_{seed}.json", {
                "case": case_name, "seed": seed, "actuals": case.audit_actuals,
                "sha256": _json_hash(case.audit_actuals)})
            for mode in modes:
                directory = runs_dir / f"{case_name}_{seed}_{mode}"
                row = run_trial(case, mode, directory, solver_root=solver_root, budget=float(time_budget),
                                memory_mib=memory_mib, solver_options=solver_options,
                                expected_source_hash=source["source_sha256"])
                report["trials"].append(row)
                _write_json(path, report)
                print(json.dumps({"case": case_name, "seed": seed, "mode": mode, "status": row["status"],
                                  "audit": row["audit"]["metric"], "seconds": row["elapsed_seconds"]}), flush=True)
                if row["status"] in {"source_changed", "input_changed"}:
                    aborted = True
                    break
            if aborted:
                break
        if aborted:
            break
    trials = report["trials"]
    report["within_solver"] = paired_comparisons(
        [row for row in trials if row["mode"] == "seed"],
        [row for row in trials if row["mode"] == "automl"], kind="seed_vs_automl")
    if "research_api_failure" in modes:
        report["research_vs_automl"] = paired_comparisons(
            [row for row in trials if row["mode"] == "automl"],
            [row for row in trials if row["mode"] == "research_api_failure"], kind="automl_vs_research_api_failure")
        report["simulation"] = {"kind": "typed_connection_failure_only",
            "external_api_calls": 0,
            "recorded_simulated_api_calls": sum(
                count for row in trials if row["mode"] == "research_api_failure"
                for count in [((row.get("solver_report") or {}).get("benchmark_evidence") or {}).get("simulated_api_calls")]
                if type(count) is int and count >= 0),
            "note": "A simulated provider failure is not a successful real LLM request."}
    if reference is not None:
        report["reference"] = {"path": str(Path(compare_report).resolve()), "source": reference["source"],
                               "environment_matches": reference.get("environment") == report["environment"],
                               **paired_comparisons(reference["trials"], trials, kind="reference")}
    report.update(status="completed" if all(row["status"] == "completed" for row in trials) else "incomplete",
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  completed_trials=sum(row["status"] == "completed" for row in trials),
                  planned_trials=len(cases) * len(seeds) * len(modes))
    _write_json(path, report)
    print(f"Offline benchmark: {report['status']}; API calls: 0. Report: {path}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-root", type=Path, default=ROOT, help="Checkout used by fresh solver processes")
    parser.add_argument("--output", type=Path, help="New output directory; default is under data/experiments/agent_benchmark")
    parser.add_argument("--cases", nargs="+", choices=ALL_CASE_NAMES, default=CASE_NAMES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seeds", nargs="+", type=int, help="Fixed dataset seeds; default 42")
    selection.add_argument("--full", action="store_true", help="Use the three predeclared seeds 42,137,2026")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    parser.add_argument("--time-budget", type=float, default=45, help="Whole subprocess wall limit per task/mode, seconds")
    parser.add_argument("--memory-mib", type=int, default=4096, help="Sampled process-tree RSS limit, when telemetry is available")
    parser.add_argument("--stability-splits", type=int, choices=(1, 2, 3), help="Optional new-solver feature; omitted for old control")
    parser.add_argument("--metric-recipes", action="store_true", help="Optional new-solver feature; omitted for old control")
    parser.add_argument("--compare-report", type=Path, help="Pair with a previous report using identical cases/data/budgets")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "data" / "experiments" / "agent_benchmark" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    options = {}
    if args.stability_splits is not None:
        options["stability_splits"] = args.stability_splits
    if args.metric_recipes:
        options["metric_recipes"] = True
    try:
        report = benchmark(output, solver_root=args.solver_root, cases=args.cases,
                           seeds=FULL_SEEDS if args.full else tuple(args.seeds or (42,)), modes=args.modes,
                           time_budget=args.time_budget, memory_mib=args.memory_mib,
                           solver_options=options, compare_report=args.compare_report)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
