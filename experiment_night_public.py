"""Prepared paired production-search sanity checks on public TRAIN holdouts.

Original public test labels are unknown and unused. These scores are neither
platform scores nor an independent confirmation set: public train was already
researched. While this file is staged outside the repo, set PYTHONPATH to the
trusted audit checkout containing experiment_night_quality.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import benchmark_agent as benchmark
from benchmark_cases import BenchmarkCase, write_case
from experiment_portfolio_recipes import comparison_dataset, freeze_splits
import experiment_night_quality as night


SCRIPT = Path(__file__).resolve()
HOST_ROOT = Path(night.__file__).resolve().parent
HARNESSES = (SCRIPT, *(HOST_ROOT / name for name in (
    "experiment_night_quality.py", "experiment_portfolio_recipes.py",
    "benchmark_agent.py", "benchmark_cases.py")))


def _metadata(dataset):
    return {key: dataset[key] for key in ("profile", "features", "numeric", "categorical", "classes",
                                         "positive_class", "time_column", "fingerprint", "fingerprints", "provenance")}


def case_from_split(dataset, split, label):
    """Expose only outer-fit labels; audit labels stay with the host."""
    if split["status"] != "ready":
        raise ValueError("Cannot materialize an impossible public split")
    features = dataset["features"]
    if {"row_key", "outcome"} & set(features):
        raise ValueError("Reserved benchmark column collides with public feature")
    fitting, held = split["outer_fit"], split["outer_held"]
    if not fitting or not held or set(fitting) & set(held):
        raise ValueError("Invalid frozen public outer indices")
    train = dataset["X"].iloc[fitting].loc[:, features].reset_index(drop=True).copy()
    test = dataset["X"].iloc[held].loc[:, features].reset_index(drop=True).copy()
    train.insert(0, "row_key", [dataset["ids"][index] for index in fitting])
    test.insert(0, "row_key", [dataset["ids"][index] for index in held])
    train["outcome"] = dataset["y"][fitting].tolist()
    actuals = tuple(dataset["y"][held].tolist())
    classification = dataset["profile"]["task_type"] == "classification"
    validation = {"strategy": "grouped"} if classification else {
        "strategy": "time", "time_column": dataset["time_column"]}
    return BenchmarkCase(name=f"public_{label}_{split['id']}", seed=split["seed"] or 0,
                         metric=dataset["profile"]["metric"], classes=tuple(dataset["classes"]),
                         positive_class=dataset["positive_class"], validation=validation,
                         numeric=tuple(dataset["numeric"]), categorical=tuple(dataset["categorical"]),
                         train=train, test=test, audit_actuals=actuals,
                         description=("Production-search sanity on a frozen public training holdout. "
                                      "Predict outcome using the observed input features. "
                                      + ("Keep identical raw feature groups together during validation."
                                         if classification else
                                         "Use earlier Year_Factor periods to validate predictions for later periods.")))


def _check_originals(manifest):
    for data in manifest["datasets"].values():
        for name, digest in data["fingerprints"].items():
            if night._hash(Path(name).read_bytes()) != digest:
                raise ValueError("Original public input changed: " + name)


def _check_serialized_outer(task, case):
    """CSV formatting must not introduce overlap absent in the raw split."""
    import pandas as pd

    fitting = benchmark.read_csv(task.parent / "inputs/train.csv")
    held = benchmark.read_csv(task.parent / "inputs/test.csv")
    features = [column for column in held.columns if column != "row_key"]
    if case.classes:
        left = set(pd.util.hash_pandas_object(fitting[features], index=False))
        right = set(pd.util.hash_pandas_object(held[features], index=False))
        if left & right:
            raise ValueError("Serialized raw feature groups overlap in public outer split")
    else:
        column = case.validation["time_column"]
        if pd.to_numeric(fitting[column]).max() >= pd.to_numeric(held[column]).min():
            raise ValueError("Serialized public outer split is not past-only")


def prepare(output, *, baseline_root, candidate_root, regression_task=None, classification_task=None,
            classification_seeds=(1729, 2718), regression_years=(4, 5), budget=180,
            memory_mib=4096, max_total_seconds=None, allow_dirty=False):
    output = Path(output).resolve()
    tasks = {label: Path(path).resolve() for label, path in (
        ("regression", regression_task), ("classification", classification_task)) if path is not None}
    if not tasks:
        raise ValueError("Provide at least one public task")
    if type(budget) not in {int, float} or not math.isfinite(budget) or not 0 < budget <= 10500:
        raise ValueError("budget must be finite seconds in (0, 10500]")
    if type(memory_mib) is not int or not 1 <= memory_mib <= benchmark.MAX_MEMORY_MIB:
        raise ValueError("Invalid memory_mib")
    if type(allow_dirty) is not bool:
        raise ValueError("allow_dirty must be boolean")
    roots = {"baseline": Path(baseline_root).resolve(), "candidate": Path(candidate_root).resolve()}
    if roots["baseline"] == roots["candidate"]:
        raise ValueError("Use distinct baseline and candidate checkouts")
    sources = {arm: benchmark.source_provenance(root) for arm, root in roots.items()}
    host = benchmark.source_provenance(HOST_ROOT)
    if not allow_dirty and any(s["worktree_dirty"] is not False for s in [*sources.values(), host]):
        raise ValueError("Commit clean solver/auditor checkouts before preparing")
    for root in roots.values():
        if output.is_relative_to(root) and not output.is_relative_to(root / "data"):
            raise ValueError("Inside solver checkout use gitignored data/")
    loaded = {label: comparison_dataset(task) for label, task in tasks.items()}
    frozen = []
    for label, dataset in loaded.items():
        if dataset["profile"]["task_type"] != label:
            raise ValueError("Public task type does not match its CLI argument")
        splits = freeze_splits(dataset, seeds=tuple(classification_seeds), outer_folds=(0,),
                               outer_years=tuple(regression_years))
        frozen.extend({"dataset": label, "split": split} for split in splits)
    ready = [item for item in frozen if item["split"]["status"] == "ready"]
    if not ready:
        raise ValueError("No requested public outer split is possible")
    trial_caps = float(2 * len(ready) * budget)
    if max_total_seconds is None:
        max_total_seconds = trial_caps + 120
    if (type(max_total_seconds) not in {int, float} or not math.isfinite(max_total_seconds)
            or not trial_caps <= max_total_seconds <= 24 * 3600):
        raise ValueError("Total allowance must cover finite trial caps and be at most 24 hours")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("prepared_inputs", "host_audit", "runs"):
        (output / name).mkdir()
    cases = []
    trials = []
    for pair, item in enumerate(ready):
        case = case_from_split(loaded[item["dataset"]], item["split"], item["dataset"])
        key = f"{case.name}_{case.seed}"
        task = write_case(case, output / "prepared_inputs" / key)
        _check_serialized_outer(task, case)
        audit_path = output / "host_audit" / f"{key}.json"
        night._write(audit_path, {"case": case.name, "seed": case.seed, "actuals": case.audit_actuals})
        cases.append({**item, "name": case.name, "seed": case.seed, "train_rows": len(case.train),
                      "test_rows": len(case.test), "input_sha256": benchmark.input_fingerprint(task),
                      "audit_sha256": night._hash(night._bytes(case.audit_actuals)),
                      "audit_file_sha256": night._hash(audit_path.read_bytes())})
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for arm in order:
            trials.append({"index": len(trials), "pair": pair, "arm": arm})
    manifest = {"version": 1, "kind": "night_public_prepared", "sources": sources, "host_source": host,
                "script_sources": {str(path): night._hash(path.read_bytes()) for path in HARNESSES},
                "environment": night._environment(), "tasks": {k: str(v) for k, v in tasks.items()},
                "datasets": {k: _metadata(v) for k, v in loaded.items()}, "requested_splits": frozen,
                "cases": cases, "trial_plan": trials, "planned_trials": len(trials),
                "budget_seconds": float(budget), "memory_mib": memory_mib,
                "planned_trial_wall_caps_seconds": trial_caps, "max_total_seconds": float(max_total_seconds),
                "solver_options": {"baseline": {}, "candidate": {"search_policy": "adaptive_v1"}},
                "prepared_at": datetime.now(timezone.utc).isoformat(), "external_api_calls": 0,
                "scope": "Previously researched public TRAIN outer holdouts, not original test/platform/private scores",
                "production_internal_validation": "grouped for classification, past-only Year_Factor for regression; reference inner split indices are not injected into the solver",
                "input_boundary": "Host retains outer answers outside solver task directories; not an OS sandbox"}
    _check_originals(manifest)
    night._write(output / "manifest.json", manifest)
    night._write(output / "seal.json", {"manifest_sha256": night._hash((output / "manifest.json").read_bytes())})
    return manifest


def read_plan(output):
    output = Path(output).resolve()
    content = (output / "manifest.json").read_bytes()
    seal = json.loads((output / "seal.json").read_text(encoding="utf-8"))
    if seal != {"manifest_sha256": night._hash(content)}:
        raise ValueError("Prepared public manifest changed")
    manifest = json.loads(content)
    if manifest.get("version") != 1 or manifest.get("kind") != "night_public_prepared":
        raise ValueError("Unsupported public plan")
    if manifest["environment"] != night._environment():
        raise ValueError("Prepared environment changed")
    for name, digest in manifest["script_sources"].items():
        if night._hash(Path(name).read_bytes()) != digest:
            raise ValueError("Prepared public script or harness changed")
    night._check_sources(manifest)
    _check_originals(manifest)
    for case in manifest["cases"]:
        key = f"{case['name']}_{case['seed']}"
        if benchmark.input_fingerprint(output / "prepared_inputs" / key / "task.md") != case["input_sha256"]:
            raise ValueError("Prepared public solver inputs changed")
        if night._hash((output / "host_audit" / f"{key}.json").read_bytes()) != case["audit_file_sha256"]:
            raise ValueError("Prepared public audit answers changed")
    return manifest


def execute_prepared(output):
    output = Path(output).resolve()
    manifest = read_plan(output)
    loaded = {label: comparison_dataset(Path(task)) for label, task in manifest["tasks"].items()}
    for label, data in loaded.items():
        if night._bytes(_metadata(data)) != night._bytes(manifest["datasets"][label]):
            raise ValueError("Reloaded public dataset differs from frozen metadata")
    if any((output / "runs").iterdir()) or (output / "report.json").exists():
        raise ValueError("Public execution directory is not fresh")
    with (output / "execution_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"manifest_sha256": night._hash((output / "manifest.json").read_bytes())}, stream)
    started = time.monotonic()
    report = {"version": 1, "kind": "night_public_comparison", "scope": manifest["scope"],
              "sources": manifest["sources"], "manifest_sha256": night._hash((output / "manifest.json").read_bytes()),
              "external_api_calls": 0, "status": "running", "trials": []}
    abort_reason = None
    night._write(output / "report.json", report)
    for planned in manifest["trial_plan"]:
        if time.monotonic() - started + manifest["budget_seconds"] > manifest["max_total_seconds"]:
            abort_reason = "series_budget_exhausted"
            break
        try:
            night._check_sources(manifest)
            _check_originals(manifest)
            frozen = manifest["cases"][planned["pair"]]
            case = case_from_split(loaded[frozen["dataset"]], frozen["split"], frozen["dataset"])
            if night._hash(night._bytes(case.audit_actuals)) != frozen["audit_sha256"]:
                raise ValueError("Regenerated public audit differs from prepared split")
            arm = planned["arm"]
            source = manifest["sources"][arm]
            row = benchmark.run_trial(case, "automl", output / "runs" / f"trial_{planned['index']:04d}_{arm}",
                                      solver_root=Path(source["root"]), budget=manifest["budget_seconds"],
                                      memory_mib=manifest["memory_mib"], solver_options=manifest["solver_options"][arm],
                                      expected_source_hash=source["source_sha256"])
            row.update(arm=arm, pair=planned["pair"], trial_index=planned["index"],
                       public_outer_split_hash=frozen["split"]["split_hash"])
            if row["input_sha256"] != frozen["input_sha256"]:
                row["status"] = "input_changed"
                row["audit"]["metric"] = None
            row["refit_evidence"] = night.refit_evidence(row, frozen["train_rows"])
            report["trials"].append(row)
            report["summary"] = night.summarize(report["trials"], manifest["planned_trials"])
            night._write(output / "report.json", report)
            print(json.dumps({"trial": planned["index"], "arm": arm, "case": case.name,
                              "status": row["status"], "metric": row["audit"].get("metric"),
                              "seconds": row["elapsed_seconds"], "full_refit_verified": row["refit_evidence"]["valid"]}), flush=True)
            if row["status"] in {"source_changed", "input_changed"}:
                abort_reason = row["status"]
                break
        except (OSError, ValueError, RuntimeError) as error:
            abort_reason = f"{type(error).__name__}: {error}"[:500]
            break
    try:
        read_plan(output)
    except (OSError, ValueError) as error:
        abort_reason = f"{type(error).__name__}: {error}"[:500]
    requested_ready = all(item["split"]["status"] == "ready" for item in manifest["requested_splits"])
    completed = (requested_ready and len(report["trials"]) == manifest["planned_trials"]
                 and all(row["status"] == "completed" for row in report["trials"]))
    report.update(status="aborted" if abort_reason else "completed" if completed else "incomplete",
                  abort_reason=abort_reason, elapsed_seconds=round(time.monotonic() - started, 6),
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  summary=night.summarize(report["trials"], manifest["planned_trials"]))
    night._write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true", help="Default, no training")
    action.add_argument("--run-prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, default=HOST_ROOT)
    parser.add_argument("--regression-task", type=Path)
    parser.add_argument("--classification-task", type=Path)
    parser.add_argument("--classification-seeds", nargs="+", type=int, default=(1729, 2718))
    parser.add_argument("--regression-years", nargs="+", type=float, default=(4, 5))
    parser.add_argument("--budget", type=float, default=180)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--max-total-seconds", type=float)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.run_prepared:
        report = execute_prepared(args.run_prepared)
        print(f"Public TRAIN sanity: {report['status']}; external API calls: 0; report: {args.run_prepared / 'report.json'}")
        return 0 if report["status"] == "completed" else 1
    if args.output is None or args.baseline_root is None:
        parser.error("Preparation requires --output and --baseline-root")
    manifest = prepare(args.output, baseline_root=args.baseline_root, candidate_root=args.candidate_root,
                       regression_task=args.regression_task, classification_task=args.classification_task,
                       classification_seeds=args.classification_seeds, regression_years=args.regression_years,
                       budget=args.budget, memory_mib=args.memory_mib, max_total_seconds=args.max_total_seconds,
                       allow_dirty=args.allow_dirty)
    print(f"Prepared public TRAIN sanity: {manifest['planned_trials']} trials; fits/API calls: 0; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
