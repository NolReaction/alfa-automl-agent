"""Paired production searches, with host-only audit answers and no API calls.

Both arms use the same immutable solver checkout and native ``search_profile``
argument; only the profile differs. Public TRAIN holdouts are development
evidence, never private/platform scores or an independent confirmation set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import benchmark_agent as benchmark
from benchmark_cases import ALL_CASE_NAMES, BenchmarkCase, make_case
from experiment_portfolio_recipes import comparison_dataset, freeze_splits


ROOT = Path(__file__).resolve().parent
PROFILES = ("legacy", "strong_v1")


def public_case(dataset, split, label):
    """Materialize earlier/group-disjoint rows; audit targets stay with host."""
    if split["status"] != "ready":
        raise ValueError("Requested public outer split is impossible")
    features = list(dataset["features"])
    if {"row_key", "outcome"} & set(features):
        raise ValueError("Benchmark field collides with an input feature")
    fitting, held = split["outer_fit"], split["outer_held"]
    if not fitting or not held or set(fitting) & set(held):
        raise ValueError("Public outer rows must be nonempty and disjoint")
    train = dataset["X"].iloc[fitting].loc[:, features].reset_index(drop=True).copy()
    test = dataset["X"].iloc[held].loc[:, features].reset_index(drop=True).copy()
    train.insert(0, "row_key", [dataset["ids"][i] for i in fitting])
    test.insert(0, "row_key", [dataset["ids"][i] for i in held])
    train["outcome"] = dataset["y"][fitting].tolist()
    classification = dataset["profile"]["task_type"] == "classification"
    if classification:
        import pandas as pd
        left = set(pd.util.hash_pandas_object(train[features], index=False))
        right = set(pd.util.hash_pandas_object(test[features], index=False))
        if left & right:
            raise ValueError("Raw feature groups overlap in the public outer split")
        if set(train["outcome"]) != set(dataset["classes"]) or set(dataset["y"][held]) != set(dataset["classes"]):
            raise ValueError("Every class must occur in both public outer partitions")
    else:
        import pandas as pd
        column = dataset["time_column"]
        fitting_periods, held_periods = pd.to_numeric(train[column]), pd.to_numeric(test[column])
        if fitting_periods.max() >= held_periods.min():
            raise ValueError("Public outer periods are not past-only")
    validation = ({"strategy": "grouped"} if classification else
                  {"strategy": "time", "time_column": dataset["time_column"]})
    return BenchmarkCase(
        name=f"public_{label}_{split['id']}", seed=split["seed"] or 0,
        metric=dataset["profile"]["metric"], classes=tuple(dataset["classes"]),
        positive_class=dataset["positive_class"], validation=validation,
        numeric=tuple(dataset["numeric"]), categorical=tuple(dataset["categorical"]),
        train=train, test=test, audit_actuals=tuple(dataset["y"][held].tolist()),
        description=("Predict outcome from the input features. This is a frozen public TRAIN outer holdout. "
                     + ("Keep identical raw feature groups together in validation." if classification else
                        "Validate future periods using earlier periods only.")))


def _full_refit_evidence(task, summary):
    """Check the committed CSV/holdout pair, not just the presence of a file."""
    from agent.automl import read_accepted_checkpoint
    from agent.task_spec import load_task, load_task_data, score_holdout
    from agent.splits import split_rows
    from offline_baseline import read_csv

    task = Path(task)
    try:
        checkpoint = read_accepted_checkpoint(task.parent / "solver_output")
        if checkpoint is None or summary is None or summary.get("output_valid") is not True:
            raise ValueError("No accepted full-data checkpoint")
        submitted = (task.parent / "submission.csv").read_bytes()
        if submitted != (checkpoint / "submission.csv").read_bytes():
            raise ValueError("Published CSV differs from the accepted checkpoint")
        spec = load_task(task)
        train, test, _, identifier, target, y, features, _, _, classification = load_task_data(spec)
        choice = {key: summary["split"][key] for key in ("strategy", "time_column") if key in summary["split"]}
        _, held, _ = split_rows(train, y, features, classification, choice)
        score = score_holdout(checkpoint / "holdout_predictions.csv", spec,
                              train[identifier].iloc[held].tolist(), train[target].iloc[held].tolist())
        if score != summary["metric"]["value"]:
            raise ValueError("Checkpoint holdout metric differs from the solver report")
        spec.validate_frame(read_csv(checkpoint / "submission.csv"), test[identifier].tolist())
        return {"valid": True, "checkpoint": str(checkpoint), "internal_holdout_score": score,
                "submission_sha256": hashlib.sha256(submitted).hexdigest()}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"valid": False, "error": f"{type(error).__name__}: {error}"}


def run(output, *, solver_root, cases=(), public_tasks=(), seeds=(31415,), years=(4, 5),
        budget=360, memory_mib=8192):
    """Run sequential equal-budget pairs after freezing all cases and sources."""
    if type(budget) not in {int, float} or not math.isfinite(budget) or not 0 < budget <= 10500:
        raise ValueError("budget must be finite seconds in (0, 10500]")
    if type(memory_mib) is not int or not 1 <= memory_mib <= benchmark.MAX_MEMORY_MIB:
        raise ValueError("Invalid memory limit")
    cases, seeds, public_tasks, years = tuple(cases), tuple(seeds), tuple(public_tasks), tuple(years)
    if not cases and not public_tasks:
        raise ValueError("Provide synthetic cases and/or public tasks")
    if not set(cases) <= set(ALL_CASE_NAMES) or len(set(cases)) != len(cases):
        raise ValueError("Provide distinct supported synthetic cases")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct uint32 seeds")
    output, solver_root = Path(output).resolve(), Path(solver_root).resolve()
    source = benchmark.source_provenance(solver_root)
    if source["worktree_dirty"] is not False:
        raise ValueError("Commit a clean immutable solver checkout before benchmarking")
    frozen = [(make_case(name, seed), None) for name in cases for seed in seeds]
    public_evidence = []
    for task in public_tasks:
        dataset = comparison_dataset(Path(task))
        label = dataset["profile"]["task_type"]
        splits = freeze_splits(dataset, seeds=seeds, outer_folds=(0,), outer_years=years)
        public_evidence.append({"task": str(Path(task).resolve()), "fingerprint": dataset["fingerprint"],
                                "fingerprints": dataset["fingerprints"], "profile": dataset["profile"],
                                "provenance": dataset["provenance"], "splits": splits})
        for split in splits:
            if split["status"] == "ready":
                frozen.append((public_case(dataset, split, label), split["split_hash"]))
    if not frozen or len({(case.name, case.seed) for case, _ in frozen}) != len(frozen):
        raise ValueError("Cases must be ready and uniquely named")
    output.mkdir(parents=True, exist_ok=False)
    (output / "host_audit").mkdir()
    harnesses = [Path(__file__).resolve(), ROOT / "benchmark_agent.py", ROOT / "benchmark_cases.py",
                 ROOT / "experiment_portfolio_recipes.py"]
    harness_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in harnesses}
    report = {"version": 1, "kind": "strong_tabular_paired_production", "source": source,
              "harness_sha256": harness_hashes, "environment": benchmark._versions(),
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "api_calls": 0,
              "scope": "Synthetic audit and previously researched public TRAIN holdouts; not platform/private scores",
              "input_boundary": "Host audit answers outside each task directory; not an OS sandbox",
              "profiles": list(PROFILES), "budget_seconds_per_trial": budget, "memory_mib": memory_mib,
              "planned_trials": 2 * len(frozen), "public": public_evidence, "trials": [],
              "unavailable_public_splits": [{"task": item["task"], "split": split}
                  for item in public_evidence for split in item["splits"] if split["status"] != "ready"]}
    report_path = output / "report.json"
    benchmark._write_json(report_path, report)
    started = time.monotonic()
    for index, (case, split_hash) in enumerate(frozen):
        audit_path = output / "host_audit" / f"{case.name}_{case.seed}.json"
        benchmark._write_json(audit_path, {"actuals": case.audit_actuals, "sha256": benchmark._json_hash(case.audit_actuals)})
        order = PROFILES if index % 2 == 0 else tuple(reversed(PROFILES))
        for profile in order:
            row = benchmark.run_trial(case, "automl", output / "runs" / f"{case.name}_{case.seed}_{profile}",
                solver_root=solver_root, budget=float(budget), memory_mib=memory_mib,
                solver_options={"search_profile": profile}, expected_source_hash=source["source_sha256"])
            row.update(profile=profile, public_outer_split_sha256=split_hash,
                       full_refit=_full_refit_evidence(row["task"], row.get("solver_report")))
            actual_profile = ((row.get("solver_report") or {}).get("search_policy") or {}).get("search_profile")
            row["native_profile_verified"] = actual_profile == profile
            if row["status"] == "completed" and not row["native_profile_verified"]:
                row["status"] = "profile_not_exercised"
            if row["status"] == "completed" and not row["full_refit"]["valid"]:
                row["status"] = "unverified_full_refit"
            report["trials"].append(row)
            benchmark._write_json(report_path, report)
            print(json.dumps({"case": case.name, "seed": case.seed, "profile": profile,
                              "status": row["status"], "audit": row["audit"]["metric"],
                              "seconds": row["elapsed_seconds"], "full_refit": row["full_refit"]}), flush=True)
            if row["status"] in {"source_changed", "input_changed"}:
                raise ValueError("Immutable benchmark boundary changed")
    for name, digest in harness_hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Benchmark harness changed during execution")
    for item in public_evidence:
        for name, digest in item["fingerprints"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("Original public inputs changed during execution")
    report.update(status="completed" if not report["unavailable_public_splits"]
                  and all(row["status"] == "completed" for row in report["trials"]) else "incomplete",
                  elapsed_seconds=round(time.monotonic() - started, 3),
                  finished_at=datetime.now(timezone.utc).isoformat())
    benchmark._write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", choices=ALL_CASE_NAMES, default=[])
    parser.add_argument("--public-tasks", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[31415])
    parser.add_argument("--years", type=float, nargs="+", default=[4, 5])
    parser.add_argument("--time-budget", type=float, default=360)
    parser.add_argument("--memory-mib", type=int, default=8192)
    args = parser.parse_args(argv)
    report = run(args.out_dir, solver_root=args.solver_root, cases=args.cases, public_tasks=args.public_tasks,
                 seeds=args.seeds, years=args.years, budget=args.time_budget, memory_mib=args.memory_mib)
    print(f"Strong tabular comparison: {report['status']}; API requests: 0; {args.out_dir / 'report.json'}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
