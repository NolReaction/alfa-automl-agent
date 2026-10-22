"""Prepare and measure one unmodified-main offline classification incumbent.

This is development evidence on a grouped slice of public TRAIN, not an
independent test or a paired challenger comparison. Answers live outside the
solver task. Resource enforcement uses the existing sampled process monitor;
it is not an OS sandbox or a hard allocation limit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE_SHA = "d05572846369f3b1e9cfa33f4008dda028db4fe0"
INPUTS = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "catboost", "lightgbm",
            "xgboost", "joblib", "threadpoolctl", "optuna", "psutil", "openai", "httpx")
for _name in THREAD_KEYS:
    os.environ[_name] = "8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(source, *args):
    return subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def source_snapshot(source):
    require(not git(source, "status", "--porcelain", "--untracked-files=no"),
            "Incumbent checkout must have no tracked changes")
    sha = git(source, "rev-parse", "HEAD")
    require(sha == BASE_SHA, "Incumbent must be the preregistered main SHA " + BASE_SHA)
    names = git(source, "ls-files", "--", "*.py", "requirements*.txt").splitlines()
    return {"sha": sha, "files": {name: file_hash(source / name) for name in names}}


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    limits = {}
    for name in ("memory.max", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            limits[name] = path.read_text().strip()
    return {"python": sys.version, "executable": str(Path(sys.executable).absolute()),
            "packages": versions, "threads": 8, "host_cpu_count": os.cpu_count(),
            "cgroup": limits, "platform": sys.platform}


def task_hashes(task):
    from agent.task_spec import load_task
    spec = load_task(task)
    return {"task.md": file_hash(task), **{name: file_hash(spec.data_dir / name) for name in INPUTS}}


def prepare(output, public_task, source):
    import pandas as pd
    from agent.splits import split_rows
    from agent.task_spec import load_task, load_task_data
    require(not output.exists(), "Prepare requires a fresh output directory")
    source_record = source_snapshot(source)
    inputs_before = task_hashes(public_task)
    spec = load_task(public_task)
    train, _, _, _, _, y, features, _, _, classification = load_task_data(spec)
    require(classification and spec.prediction_kind == "labels", "Expected label classification")
    fitting, held, split = split_rows(train, y, features, True, {"strategy": "grouped"})
    require(split["seed"] == 42, "Unexpected shared grouped-split seed")
    groups = pd.util.hash_pandas_object(train.loc[:, features], index=False).to_numpy()
    require(not (set(groups[fitting]) & set(groups[held])), "External raw-feature groups overlap")
    task_dir = output / "task"
    data = task_dir / "inputs"
    data.mkdir(parents=True)
    train.iloc[fitting].to_csv(data / "train.csv", index=False, lineterminator="\n")
    train.iloc[held].drop(columns=[spec.target_column]).to_csv(data / "test.csv", index=False, lineterminator="\n")
    shutil.copyfile(spec.data_dir / "data_dictionary.csv", data / "data_dictionary.csv")
    sample = pd.DataFrame({spec.id_column: train.iloc[held][spec.id_column].tolist(),
                           spec.target_column: [spec.classes[0]] * len(held)})
    sample.to_csv(data / "sample_submission.csv", index=False, lineterminator="\n")
    # Retain task semantics, but make every declared path and row count accurate.
    text = public_task.read_text(encoding="utf-8-sig")
    text = text.replace("multiclass/participants/", "inputs/")
    text = text.replace("87 670", f"{len(fitting):,}".replace(",", " "))
    text = text.replace("41 166", f"{len(held):,}".replace(",", " "))
    text += "\nThis local development task uses a grouped held-out slice of public TRAIN as test; no original competition TEST labels are available.\n"
    task = task_dir / "task.md"
    task.write_text(text, encoding="utf-8", newline="\n")
    host = output / "host_only"
    host.mkdir()
    answers = train.iloc[held][[spec.id_column, spec.target_column]]
    answers.to_csv(host / "answers.csv", index=False, lineterminator="\n")
    write_json(host / "split.json", {"fitting": fitting.tolist(), "held": held.tolist(), "metadata": split})
    require(task_hashes(public_task) == inputs_before, "Original inputs changed during preparation")
    require(load_task(task).classes == spec.classes, "Derived class declaration changed")
    plan = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "public_train_grouped_development_incumbent_only", "source_root": str(source),
            "source": source_record, "harness_sha256": file_hash(__file__),
            "environment": environment(), "public_task": str(public_task), "source_inputs": inputs_before,
            "derived_inputs": task_hashes(task), "answers_sha256": file_hash(host / "answers.csv"),
            "split_sha256": file_hash(host / "split.json"), "split": split,
            "classes": list(spec.classes), "id_column": spec.id_column, "target_column": spec.target_column,
            "support": {"fitting": dict(Counter(y.iloc[fitting])), "held": dict(Counter(y.iloc[held]))},
            "time_budget_seconds": 900, "supervisor_timeout_seconds": 930, "memory_limit_bytes": 6 * 1024 ** 3,
            "mode": "offline", "research_search": "adaptive", "order": ["incumbent"],
            "api_expected": {"attempts": 0, "responses": 0},
            "limitations": ["8 GiB host in this experiment, not official 32 GiB resource",
                            "930s supervisor allows 30s startup/cleanup beyond native 900s solver budget",
                            "process-tree memory cap is sampled; allowed-input isolation is not an OS sandbox"]}
    write_json(output / "plan.json", plan)
    write_json(output / "seal.json", {"plan_sha256": file_hash(output / "plan.json")})
    return plan


def load_plan(output, *, execution=False):
    plan = read_json(output / "plan.json")
    require(file_hash(output / "plan.json") == read_json(output / "seal.json")["plan_sha256"], "Plan drift")
    require(task_hashes(output / "task/task.md") == plan["derived_inputs"], "Derived input drift")
    require(file_hash(output / "host_only/answers.csv") == plan["answers_sha256"], "Answer drift")
    require(file_hash(output / "host_only/split.json") == plan["split_sha256"], "Split drift")
    if execution:
        require(source_snapshot(Path(plan["source_root"])) == plan["source"], "Incumbent source drift")
        require(file_hash(__file__) == plan["harness_sha256"], "Harness changed since prepare")
        require(environment() == plan["environment"], "Environment changed since prepare")
    return plan


def audit(output):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    plan = load_plan(output)
    require(task_hashes(Path(plan["public_task"])) == plan["source_inputs"],
            "Original CSV/task changed since preparation")
    execution = read_json(output / "execution.json")
    task_dir = output / "task"
    reports = list((task_dir / ".agent_runs").glob("*/run.json"))
    require(len(reports) <= 1, "Multiple native runs in fresh task")
    native = read_json(reports[0]) if reports else {}
    api = native.get("api", {})
    resources = execution.get("resources", {})
    memory_max = plan.get("environment", {}).get("cgroup", {}).get("memory.max")
    hard_memory = int(memory_max) if isinstance(memory_max, str) and memory_max.isdigit() else None
    result = {"version": 1, "scope": plan["scope"], "source_sha": plan["source"]["sha"],
              "plan_sha256": file_hash(output / "plan.json"), "execution": execution,
              "audit_harness_sha256": file_hash(__file__),
              "run_harness_sha256": plan.get("harness_sha256"),
              "resource_limit_evidence": {
                  "requested_tree_rss_limit_bytes": plan.get("memory_limit_bytes"),
                  "requested_limit_enforcement": "sampled" if resources.get("available") is True else "unavailable",
                  "peak_rss_bytes": resources.get("peak_rss_bytes") if resources.get("available") is True else None,
                  "unavailable_reason": resources.get("unavailable_reason"),
                  "host_cgroup_hard_limit_bytes_at_prepare": hard_memory,
                  "official_32_gib_resource_confirmed": False},
              "native_report": str(reports[0].relative_to(output)) if reports else None,
              "native_report_sha256": file_hash(reports[0]) if reports else None,
              "api": {"attempts": api.get("attempts"), "responses": api.get("responses")},
              "api_zero_verified": api.get("attempts") == 0 and api.get("responses") == 0,
              "candidate_reached": bool(native.get("candidate_receipts")),
              "candidate_receipts": native.get("candidate_receipts"),
              "outer_acceptance": native.get("evaluations", {}).get("final_check"),
              "refit_execution": native.get("refit_execution"), "publication": native.get("publication"),
              "published_recipe": native.get("published_recipe"), "scores": None,
              "incumbent_only": True, "challenger_reached": False}
    receipt_paths = native.get("candidate_receipts") or {}
    checked_receipts = {}
    for name, phase in (("outer", "outer_validation"), ("refit", "full_refit")):
        if receipt_paths.get(name):
            path = Path(receipt_paths[name]).resolve()
            require(task_dir in path.parents, "Native receipt escaped task directory")
            receipt = read_json(path)
            require(receipt.get("phase") == phase
                    and isinstance(receipt_paths.get("code_hash"), str)
                    and isinstance(receipt_paths.get("recipe_hash"), str)
                    and receipt.get("code_hash") == receipt_paths["code_hash"]
                    and receipt.get("recipe_hash") == receipt_paths["recipe_hash"],
                    "Native " + name + " receipt identity differs from publication")
            checked_receipts[name] = receipt
    result["outer_receipt"] = checked_receipts.get("outer")
    result["outer_and_refit_receipts_verified"] = set(checked_receipts) == {"outer", "refit"}
    result["candidate_reached"] = bool(checked_receipts)
    submission = task_dir / "submission.csv"
    if submission.is_file():
        prediction = pd.read_csv(submission, dtype=str, keep_default_na=False)
        answers = pd.read_csv(output / "host_only/answers.csv", dtype=str, keep_default_na=False)
        identifier, target = plan["id_column"], plan["target_column"]
        require(list(prediction.columns) == [identifier, target], "Invalid submission columns")
        require(prediction[identifier].tolist() == answers[identifier].tolist(), "Submission IDs/order differ")
        require(set(prediction[target]) <= set(plan["classes"]), "Unknown predicted labels")
        precision, recall, f1, support = precision_recall_fscore_support(
            answers[target], prediction[target], labels=plan["classes"], zero_division=0)
        result["scores"] = {"macro_f1": float(f1.mean()), "classes": plan["classes"],
                            "per_class": [{"class": c, "precision": float(p), "recall": float(r),
                                           "f1": float(f), "support": int(n)}
                                          for c, p, r, f, n in zip(plan["classes"], precision, recall, f1, support)],
                            "confusion_matrix": confusion_matrix(answers[target], prediction[target],
                                                                  labels=plan["classes"]).tolist()}
        result["submission_sha256"] = file_hash(submission)
        result["refit_output_hash_matches"] = False
        if "refit" in checked_receipts:
            receipt = checked_receipts["refit"]
            result["refit_receipt"] = receipt
            result["refit_output_hash_matches"] = receipt.get("output_hash") == file_hash(submission)
    result["status"] = ("completed" if execution["exit_code"] == 0 and not execution["timed_out"]
                        and not execution["memory_limit_exceeded"] and result["scores"] is not None
                        and result["api_zero_verified"] and (result["publication"] or {}).get("new_result_published")
                        and result["outer_and_refit_receipts_verified"]
                        and result.get("refit_output_hash_matches") else "incomplete_or_fallback")
    write_json(output / "report.json", result)
    return result


def run_prepared(output):
    from agent.resources import ProcessTreeMonitor
    plan = load_plan(output, execution=True)
    require(not (output / "execution.json").exists() and not (output / "started.json").exists(), "Run is single-use")
    task = output / "task/task.md"
    require(not (task.parent / "submission.csv").exists() and not (task.parent / ".agent_runs").exists(),
            "Solver task already contains outputs")
    command = [plan["environment"]["executable"], str(Path(plan["source_root"]) / "solution.py"), str(task),
               "--mode", "offline", "--time-budget", "900", "--research-search", "adaptive"]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({name: "8" for name in THREAD_KEYS})
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    write_json(output / "started.json", {"command": command, "utc": datetime.now(timezone.utc).isoformat()})
    timed_out = memory_exceeded = False
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=plan["source_root"], env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != "nt")
        monitor = ProcessTreeMonitor(process, max_memory_bytes=plan["memory_limit_bytes"])
        try:
            while process.poll() is None:
                memory_exceeded = monitor.sample()
                timed_out = time.monotonic() - started >= plan["supervisor_timeout_seconds"]
                if memory_exceeded or timed_out:
                    break
                time.sleep(.05)
            monitor.sample()
        finally:
            monitor.terminate()
            write_json(output / "execution.json", {"command": command, "exit_code": process.returncode,
                       "elapsed_seconds": time.monotonic() - started, "timed_out": timed_out,
                       "memory_limit_exceeded": memory_exceeded, "resources": monitor.summary()})
    require(source_snapshot(Path(plan["source_root"])) == plan["source"], "Source changed while running")
    return audit(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true", help="Freeze data/split/source/environment; no training")
    actions.add_argument("--run-prepared", action="store_true", help="Run one native 900-second incumbent")
    actions.add_argument("--audit", action="store_true", help="Recompute saved submission metrics without training")
    parser.add_argument("--output", type=Path, required=True, help="Fresh evidence directory at preparation")
    parser.add_argument("--public-task", type=Path, default=ROOT / "data/classification/task.md")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "classification-incumbent")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        result = prepare(output, args.public_task.resolve(), args.source_root.resolve())
        print(json.dumps({"status": "prepared", "split": result["split"], "plan": str(output / "plan.json")}))
    else:
        result = run_prepared(output) if args.run_prepared else audit(output)
        print(json.dumps({"status": result["status"], "scores": result["scores"], "report": str(output / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
