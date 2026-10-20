"""Bounded synthetic v3 smoke. Default preflight never requests the API.

Run ``python experiment_research_smoke.py --execute`` only when the team's
shared API budget is available. These tiny tasks check integration, not public
or private competition quality. Generated datasets and reports stay in data/.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agent.llm import API_FIELDS, REQUIRED_API_FIELDS, ROOT, load_local_env, make_client, safe_error_message
from agent.research import run_agent
from agent.task_spec import load_task
from build_submission import source_files
from validation import validate_predictions


MAX_REQUESTS = 8
MAX_ESTIMATED_TOKENS = 100000
TASK_NAMES = ("regression", "classification")
CLASS_LABELS = ("01", "NA", "class c")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _redact(value):
    secrets = [os.environ[name] for name in REQUIRED_API_FIELDS if os.environ.get(name)]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_redact(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def create_tasks(directory: Path):
    """Create deterministic, deliberately simple data with literal IDs/labels."""
    result = {}
    for name in TASK_NAMES:
        task_dir = directory / name
        inputs = task_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=False)
        classification = name == "classification"
        _write_csv(inputs / "data_dictionary.csv", [
            ["feature", "role", "dtype"], ["row_key", "id", "category"],
            ["measurement", "feature", "numeric"], ["segment", "feature", "category"],
            ["outcome", "target", "category" if classification else "numeric"],
        ])
        rows = [["row_key", "measurement", "segment", "outcome"]]
        for index in range(120):
            label_index = index % 3
            x = (label_index * 10 + index / 100) if classification else (index - 60) / 10
            target = CLASS_LABELS[label_index] if classification else 2.5 * x + 7
            rows.append([f"train-{index:04d}", x, f"segment-{label_index}", target])
        _write_csv(inputs / "train.csv", rows)
        _write_csv(inputs / "test.csv", [
            ["row_key", "measurement", "segment"],
            ["0005", 0.2 if classification else 2, "segment-0"],
            ["NA", 10.5 if classification else -1, "segment-1"],
            ["test c", 21 if classification else 0, "segment-2"],
        ])
        _write_csv(inputs / "sample_submission.csv", [
            ["row_key", "outcome"], ["0005", "01" if classification else 0],
            ["NA", "NA" if classification else 0], ["test c", "class c" if classification else 0],
        ])
        text = (f"# Synthetic {name} integration smoke\n"
                "Use inputs/*.csv and predict outcome. Save submission.csv here.\n"
                f"Task type: {name}\nPrimary metric: {'Macro F1' if classification else 'RMSE'}\n")
        if classification:
            text += "Class order: " + json.dumps(list(CLASS_LABELS)) + "\n"
        task = task_dir / "task.md"
        task.write_text(text, encoding="utf-8")
        # Fail before any paid request if a synthetic schema drifts.
        load_task(task)
        result[name] = task
    return result


class RequestBudget:
    """A shared ceiling for both sequential tasks, independent of model claims."""

    def __init__(self, limit=MAX_REQUESTS, token_limit=MAX_ESTIMATED_TOKENS):
        self.limit, self.token_limit = limit, token_limit
        self.attempts = 0
        self.estimated_tokens_reserved = 0
        self.blocked = False

    def reserve(self, arguments):
        messages = json.dumps(arguments.get("messages", []), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
        output = arguments.get("max_tokens", 4096)
        if type(output) is not int or output < 1:
            raise ValueError("Smoke output reservation must be a positive integer")
        estimated = math.ceil(len(messages) / 3) + output
        if self.attempts >= self.limit or self.estimated_tokens_reserved + estimated > self.token_limit:
            self.blocked = True
            raise RuntimeError("Shared synthetic smoke request/token ceiling reached; no further request sent")
        self.attempts += 1
        self.estimated_tokens_reserved += estimated

    def summary(self):
        return {"attempts": self.attempts, "max_requests": self.limit,
                "estimated_tokens_reserved": self.estimated_tokens_reserved,
                "estimated_token_limit": self.token_limit, "blocked": self.blocked,
                "estimate_method": "ceil(UTF-8 message bytes / 3) + max_tokens output reserve",
                "currency_cost": None}


class BudgetClient:
    """Apply the same budget even when the SDK clones request timeout options."""

    def __init__(self, client, budget):
        self.client, self.budget = client, budget
        self.chat = self
        self.completions = self

    def with_options(self, **kwargs):
        return BudgetClient(self.client.with_options(**kwargs), self.budget)

    def create(self, **kwargs):
        self.budget.reserve(kwargs)
        return self.client.chat.completions.create(**kwargs)


def _metric(value):
    if not isinstance(value, dict):
        return None
    if (not isinstance(value.get("name"), str) or type(value.get("value")) not in {int, float}
            or not math.isfinite(value["value"]) or type(value.get("higher_is_better")) is not bool):
        return None
    return {key: value[key] for key in ("name", "value", "higher_is_better")}


def _candidate_receipt(run_dir, directory, phase, frozen):
    if run_dir is None or not isinstance(frozen, dict):
        return None
    hashes = {key: frozen.get(key) for key in ("code_hash", "recipe_hash")}
    if any(not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
           for value in hashes.values()):
        return None
    for path in (run_dir / directory / "candidate").glob(phase + "_*/receipt.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            output = path.parent / "predictions.csv"
            if (receipt.get("version") == 3 and receipt.get("phase") == phase
                    and all(receipt.get(key) == value for key, value in hashes.items())
                    and receipt.get("output_hash") == hashlib.sha256(output.read_bytes()).hexdigest()):
                return receipt
        except (OSError, ValueError, TypeError):
            pass
    return None


def _coverage(task, report, run_dir):
    frozen = report.get("frozen_candidate")
    final_check = (report.get("evaluations") or {}).get("final_check") or {}
    outer = _candidate_receipt(run_dir, "outer_evaluations", "outer_validation", frozen)
    refit = _candidate_receipt(run_dir, "full_refit", "full_refit", frozen)
    outer_valid = _metric(final_check.get("candidate")) is not None and outer is not None
    publication = report.get("publication") or {}
    published = (outer_valid and refit is not None and publication.get("new_result_published") is True
                 and publication.get("recipe") == "candidate" and final_check.get("selected") == "candidate")
    if published:
        try:
            published = hashlib.sha256((task.parent / "submission.csv").read_bytes()).hexdigest() == refit["output_hash"]
        except OSError:
            published = False
    return {"inner_trial_verified": report.get("successful_experiments", 0) > 0,
            "generated_candidate_outer_valid": outer_valid,
            "generated_candidate_published": bool(published)}


def summarize_run(task: Path, report: dict, *, run_dir: Path | None = None):
    """Copy evidence fields only; arbitrary generated text/logs are not summary data."""
    api = report.get("api", {})
    responses = api.get("responses", 0)
    reported = api.get("usage_reported_responses", 0)
    evaluations = report.get("evaluations", {})
    selection = evaluations.get("selection") or {}
    final_check = evaluations.get("final_check") or {}
    return {
        "task": task.parent.name, "synthetic_only": True,
        "research_protocol_version": report.get("research_protocol_version"),
        "status": report.get("status"), "reason": report.get("reason"),
        "measured_trials": report.get("successful_experiments", 0),
        "unverified_trials": report.get("unverified_experiments", 0),
        "started_at": report.get("started_at"), "elapsed_seconds": report.get("elapsed_seconds"),
        "selection": {"scope": "inner_selection", "metric": _metric(selection.get("metric"))},
        "final_check": {"baseline": _metric(final_check.get("baseline")),
                        "candidate": _metric(final_check.get("candidate")),
                        "selected": final_check.get("selected"),
                        "selection_uses_outer": final_check.get("selection_uses_outer"),
                        "unbiased_estimate": final_check.get("unbiased_estimate")},
        "output_check": validate_predictions(task.parent),
        "api": {"attempts": api.get("attempts", 0), "responses": responses,
                "usage_reported_responses": reported,
                "responses_without_usage": max(0, responses - reported),
                "usage_complete": responses > 0 and reported == responses,
                "reported_tokens": {key: api.get(key) for key in
                                    ("prompt_tokens", "completion_tokens", "total_tokens")},
                "currency_cost": None},
        "frozen_candidate": report.get("frozen_candidate"),
        "publication": report.get("publication"),
        "coverage": _coverage(task, report, run_dir),
    }


def _source_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_provenance():
    """Fingerprint current source bytes without building an archive."""
    commit = _source_commit()
    try:
        state = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5, check=True)
        dirty = bool(state.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    entries = []
    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        entries.append({"path": relative, "sha256": file_hash, "bytes": len(content)})
        digest.update(relative.encode("utf-8") + b"\0" + str(len(content)).encode("ascii") + b"\0" + content)
    return {"source_commit": commit, "source_worktree_dirty": dirty,
            "source_files_sha256": digest.hexdigest(), "source_files": entries,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fingerprint_method": "SHA-256 of sorted source_files() relative path, NUL, byte size, NUL, content"}


def coverage_summary(summaries):
    complete = (len(summaries) == len(TASK_NAMES)
        and {item.get("task") for item in summaries} == set(TASK_NAMES) and all(
        item.get("research_protocol_version") == 3 and item.get("coverage", {}).get("inner_trial_verified")
        and item.get("status") == "completed" and item.get("output_check", {}).get("valid")
        and (item.get("publication") or {}).get("new_result_published") is True
        and _metric((item.get("final_check") or {}).get((item.get("publication") or {}).get("recipe"))) is not None
        for item in summaries))
    full = complete and all(item.get("coverage", {}).get("generated_candidate_outer_valid")
                            and item.get("coverage", {}).get("generated_candidate_published") for item in summaries)
    return {"integration_passed": complete, "full_candidate_path_passed": bool(full),
            "scope": ("full_generated_candidate_path" if full else "inner_research_and_baseline_finalization"
                      if complete else "partial_or_unverified"),
            "tasks_with_verified_inner_trial": sum(bool(item.get("coverage", {}).get("inner_trial_verified")) for item in summaries),
            "tasks_with_valid_generated_outer": sum(bool(item.get("coverage", {}).get("generated_candidate_outer_valid")) for item in summaries),
            "tasks_with_published_generated_candidate": sum(bool(item.get("coverage", {}).get("generated_candidate_published")) for item in summaries)}


def _positive_steps(value):
    result = int(value)
    if not 1 <= result <= 4:
        raise argparse.ArgumentTypeError("max-steps must be 1..4 per task (8 requests total at most)")
    return result


def _budget_seconds(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 600:
        raise argparse.ArgumentTypeError("time-budget must be positive and at most 600 seconds per task")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Make explicitly authorized real API requests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments" / "research_smoke")
    parser.add_argument("--max-steps", type=_positive_steps, default=4)
    parser.add_argument("--time-budget", type=_budget_seconds, default=300)
    args = parser.parse_args(argv)
    output_root = args.output_dir.expanduser().resolve()
    if output_root.is_relative_to(ROOT.resolve()) and not output_root.is_relative_to((ROOT / "data").resolve()):
        parser.error("Use gitignored data/ for output inside this repository, or a directory outside the repository")
    load_local_env()
    present = {name: bool(os.environ.get(name, "").strip()) for name in API_FIELDS}
    configured = all(present[name] for name in REQUIRED_API_FIELDS)
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = create_tasks(run_dir)
    summary = {
        "version": 1, "kind": "synthetic_research_v3_smoke", "synthetic_only": True,
        "purpose": "integration evidence; not public/private ML quality or platform score",
        **source_provenance(), "started_at": _now(),
        "mode": "execute" if args.execute else "preflight", "api_settings_present": present,
        "api_configured": configured, "max_steps_per_task": args.max_steps,
        "time_budget_seconds_per_task": args.time_budget, "tasks": [],
        "status": "ready" if configured else "missing_api_settings",
        "coverage_summary": coverage_summary([]),
        "full_candidate_path_passed": False,
    }
    budget = RequestBudget(limit=min(MAX_REQUESTS, 2 * args.max_steps))
    summary_path = run_dir / "smoke_report.json"
    _write_json(summary_path, summary)
    if not args.execute:
        summary.update(finished_at=_now(), budget=budget.summary(), api_called=False)
        _write_json(summary_path, summary)
        print(f"Preflight: {summary['status']}; API requests: 0. Report: {summary_path}")
        return 0
    if not configured:
        summary.update(finished_at=_now(), budget=budget.summary(), api_called=False)
        _write_json(summary_path, summary)
        print(f"Execution blocked: required API settings are missing. Report: {summary_path}")
        return 2
    try:
        with make_client() as sdk:
            client = BudgetClient(sdk, budget)
            for name, task in tasks.items():
                output, errors = io.StringIO(), io.StringIO()
                report = None
                result_dir = None
                try:
                    with redirect_stdout(output), redirect_stderr(errors):
                        result_dir = run_agent(task, client, max_steps=args.max_steps, time_budget=args.time_budget)
                    report = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
                except Exception as error:
                    paths = list((task.parent / ".agent_runs").glob("*/run.json"))
                    if paths:
                        result_dir = paths[0].parent
                        report = json.loads(paths[0].read_text(encoding="utf-8"))
                    summary["error"] = safe_error_message(error)
                (task.parent / "execution.log").write_text(_redact(output.getvalue() + errors.getvalue()), encoding="utf-8")
                if report is None:
                    summary["tasks"].append({"task": name, "status": "failed", "measured_trials": 0})
                    break
                item = summarize_run(task, report, run_dir=result_dir)
                summary["tasks"].append(item)
                _write_json(summary_path, summary)
                # API errors (including quota) are terminal for the shared smoke.
                # Do not launch the second task as an independent retry series.
                if budget.blocked or report.get("reason") == "api_error" or report.get("status") == "failed":
                    break
    except Exception as error:
        summary["error"] = safe_error_message(error)
    summaries = summary["tasks"]
    coverage = coverage_summary(summaries)
    success = coverage["integration_passed"]
    summary.update(status="passed" if success else "incomplete", finished_at=_now(),
                   budget=budget.summary(), api_called=budget.attempts > 0, coverage_summary=coverage,
                   full_candidate_path_passed=coverage["full_candidate_path_passed"])
    _write_json(summary_path, summary)
    print(f"Synthetic smoke: {summary['status']} ({coverage['scope']}); API requests: {budget.attempts}. Report: {summary_path}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
