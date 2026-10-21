"""Prepared, sequential, API-free comparisons of two frozen solver checkouts.

The existing benchmark supplies fresh worker execution and independent audits.
This runner adds a sealed two-checkout plan, alternating execution order and a
finite whole-series allowance. Synthetic scores are not contest scores. Audit
answers are absent from solver task directories; this is not an OS sandbox.
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
import time

import benchmark_agent as benchmark
from benchmark_cases import ALL_CASE_NAMES, CASE_NAMES, make_case, write_case
from agent.task_spec import TaskSpec


ROOT = Path(__file__).resolve().parent
VERSION = 1
HARNESS_FILES = ("experiment_night_quality.py", "benchmark_agent.py", "benchmark_cases.py")
POLICIES = ("legacy", "adaptive_v1")
MODES = ("automl", "research_api_failure")


def _bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_bytes(value) + b"\n")
    temporary.replace(path)


def _environment() -> dict:
    value = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "psutil"):
        try:
            value[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value[name] = None
    return value


def _validate_sequence(values, *, allowed=None, integer=False, name="values") -> tuple:
    values = tuple(values)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and unique")
    if integer and any(type(v) is not int or not 0 <= v < 2 ** 32 for v in values):
        raise ValueError(f"{name} must contain integer seeds in 0..2**32-1")
    if allowed is not None and not set(values) <= set(allowed):
        raise ValueError(f"Unsupported {name}")
    return values


def prepare(output: Path, *, baseline_root: Path, candidate_root: Path,
            cases=CASE_NAMES, seeds=(42, 73), budgets=(90,), modes=("automl",),
            memory_mib=4096, candidate_policy="adaptive_v1", max_total_seconds=None,
            purpose="development", allow_dirty=False) -> dict:
    """Freeze a finite plan and its inputs without launching a solver or API."""
    output = Path(output).resolve()
    roots = {"baseline": Path(baseline_root).resolve(), "candidate": Path(candidate_root).resolve()}
    if roots["baseline"] == roots["candidate"]:
        raise ValueError("Use distinct baseline and candidate checkouts")
    cases = _validate_sequence(cases, allowed=ALL_CASE_NAMES, name="cases")
    seeds = _validate_sequence(seeds, integer=True, name="seeds")
    modes = _validate_sequence(modes, allowed=MODES, name="modes")
    budgets = _validate_sequence(budgets, name="budgets")
    if any(type(v) not in {int, float} or not math.isfinite(v) or not 0 < v <= 10500 for v in budgets):
        raise ValueError("budgets must be finite seconds in (0, 10500]")
    if type(memory_mib) is not int or not 1 <= memory_mib <= benchmark.MAX_MEMORY_MIB:
        raise ValueError("memory_mib is outside the benchmark limit")
    if candidate_policy not in POLICIES or purpose not in {"development", "confirmation", "integration"}:
        raise ValueError("Unsupported policy or purpose")
    if type(allow_dirty) is not bool:
        raise ValueError("allow_dirty must be boolean")
    sources = {label: benchmark.source_provenance(root) for label, root in roots.items()}
    host_source = benchmark.source_provenance(ROOT)
    if not allow_dirty and any(source["worktree_dirty"] is not False
                               for source in [*sources.values(), host_source]):
        raise ValueError("Commit both checkouts before preparing, or explicitly allow dirty development")
    if purpose == "confirmation" and allow_dirty:
        raise ValueError("Confirmation requires committed clean checkouts")
    for root in roots.values():
        if output.is_relative_to(root) and not output.is_relative_to(root / "data"):
            raise ValueError("Inside a solver checkout, use gitignored data/ for outputs")
    planned_seconds = float(2 * len(cases) * len(seeds) * len(modes) * sum(budgets))
    if max_total_seconds is None:
        max_total_seconds = planned_seconds + max(60, 2 * len(cases) * len(seeds) * len(modes) * len(budgets))
    if (type(max_total_seconds) not in {int, float} or not math.isfinite(max_total_seconds)
            or not planned_seconds <= max_total_seconds <= 24 * 3600):
        raise ValueError("max_total_seconds must cover all trial wall caps and be at most 24 hours")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("host_source", "host_audit", "prepared_inputs", "runs"):
        (output / name).mkdir()
    harness = {}
    for name in HARNESS_FILES:
        content = (ROOT / name).read_bytes()
        harness[name] = _hash(content)
        (output / "host_source" / name).write_bytes(content)
    frozen_cases = []
    for name in cases:
        for seed in seeds:
            case = make_case(name, seed)
            key = f"{name}_{seed}"
            task = write_case(case, output / "prepared_inputs" / key)
            audit = {"case": name, "seed": seed, "actuals": case.audit_actuals}
            _write(output / "host_audit" / f"{key}.json", audit)
            frozen_cases.append({"case": name, "seed": seed,
                                 "input_sha256": benchmark.input_fingerprint(task),
                                 "audit_sha256": _hash(_bytes(case.audit_actuals)),
                                 "audit_file_sha256": _hash((output / "host_audit" / f"{key}.json").read_bytes()),
                                 "train_rows": len(case.train), "test_rows": len(case.test)})
    trials = []
    pair = 0
    for budget in budgets:
        for mode in modes:
            for name in cases:
                for seed in seeds:
                    order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                    for arm in order:
                        trials.append({"index": len(trials), "pair": pair, "arm": arm,
                                       "case": name, "seed": seed, "mode": mode, "budget": float(budget)})
                    pair += 1
    manifest = {"version": VERSION, "kind": "night_quality_prepared", "purpose": purpose,
                "prepared_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
                "host_source": host_source,
                "harness_sha256": harness, "environment": _environment(), "cases": frozen_cases,
                "trial_plan": trials, "planned_trials": len(trials),
                "planned_trial_wall_caps_seconds": planned_seconds,
                "max_total_seconds": float(max_total_seconds), "memory_mib": memory_mib,
                "solver_options": {"baseline": {}, "candidate": {"search_policy": candidate_policy}},
                "candidate_policy": candidate_policy, "allow_dirty": allow_dirty,
                "api": {"external_calls": 0},
                "input_boundary": "Audit answers are outside task directories; ordinary processes are not OS-sandboxed."}
    _write(output / "manifest.json", manifest)
    _write(output / "seal.json", {"manifest_sha256": _hash((output / "manifest.json").read_bytes())})
    return manifest


def _read_plan(output: Path) -> dict:
    content = (output / "manifest.json").read_bytes()
    seal = json.loads((output / "seal.json").read_text(encoding="utf-8"))
    if seal != {"manifest_sha256": _hash(content)}:
        raise ValueError("Prepared manifest changed")
    manifest = json.loads(content)
    if manifest.get("version") != VERSION or manifest.get("kind") != "night_quality_prepared":
        raise ValueError("Unsupported prepared plan")
    if not manifest.get("trial_plan") or manifest["planned_trials"] != len(manifest["trial_plan"]):
        raise ValueError("Empty or inconsistent prepared trial plan")
    if manifest.get("environment") != _environment():
        raise ValueError("Prepared environment changed")
    for name, digest in manifest["harness_sha256"].items():
        if (_hash((ROOT / name).read_bytes()) != digest
                or _hash((output / "host_source" / name).read_bytes()) != digest):
            raise ValueError("Prepared harness or generator changed")
    _check_sources(manifest)
    for item in manifest["cases"]:
        key = f"{item['case']}_{item['seed']}"
        if benchmark.input_fingerprint(output / "prepared_inputs" / key / "task.md") != item["input_sha256"]:
            raise ValueError("Prepared case inputs changed")
        audit = output / "host_audit" / f"{key}.json"
        if _hash(audit.read_bytes()) != item["audit_file_sha256"]:
            raise ValueError("Prepared audit answers changed")
    return manifest


def _check_sources(manifest: dict) -> None:
    for expected in [*manifest["sources"].values(), manifest["host_source"]]:
        actual = benchmark.source_provenance(Path(expected["root"]))
        for key in ("source_sha256", "commit", "worktree_dirty"):
            if actual[key] != expected[key]:
                raise ValueError(f"Frozen solver {key} changed: {expected['root']}")


def refit_evidence(row: dict, train_rows: int) -> dict:
    """Check source-bound worker artifacts; do not call a valid CSV a refit."""
    if (row.get("status") != "completed" or row.get("sources_unchanged") is not True
            or row.get("inputs_unchanged") is not True):
        return {"valid": False, "kind": "trial_contract_incomplete", "train_rows": None}
    if row["mode"] == "research_api_failure":
        report = row.get("solver_report") or {}
        valid = (row["status"] == "completed" and not row.get("research_contract_errors")
                 and report.get("receipt_checks_valid") is True)
        return {"valid": valid, "kind": "research_receipts", "train_rows": train_rows if valid else None}
    task_dir = Path(row["task"]).parent
    try:
        directory = task_dir / "solver_output"
        report = json.loads((directory / "automl_report.json").read_text(encoding="utf-8"))
        search = report["search"]
        identifier = search["best_candidate"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Invalid selected candidate identifier")
        selected = [item for item in search["candidates"] if item.get("selected") is True]
        if (len(selected) != 1 or selected[0]["id"] != identifier
                or selected[0].get("status") != "completed"
                or selected[0].get("phase") not in {"seed", "refit"}):
            raise ValueError("No completed full-data phase for selected candidate")
        phase = selected[0]["phase"]
        pointer = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        relative = pointer["directory"]
        if not isinstance(relative, str) or not re.fullmatch(r"checkpoints/[0-9a-f]{32}", relative):
            raise ValueError("Invalid checkpoint path")
        for filename, key in (("submission.csv", "submission_sha256"),
                              ("holdout_predictions.csv", "holdout_sha256")):
            if _hash((directory / relative / filename).read_bytes()) != pointer[key]:
                raise ValueError("Checkpoint bytes changed")
        if _hash((task_dir / "submission.csv").read_bytes()) != pointer["submission_sha256"]:
            raise ValueError("Published CSV is not accepted checkpoint")
        paths = list((task_dir / ".agent_runs").glob(f"*/{identifier}/{phase}/config.json"))
        matches = []
        for config_path in paths:
            candidate_dir = config_path.parent
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("id") != identifier or config.get("phase") != phase:
                continue
            spec = TaskSpec.from_dict(config["task_spec"])
            if (Path(config["candidate_dir"]).resolve() != candidate_dir.resolve()
                    or spec.task_path.resolve() != (candidate_dir / "task.md").resolve()
                    or spec.data_dir.resolve() != (candidate_dir / "inputs").resolve()
                    or spec != benchmark.load_task(candidate_dir / "task.md")
                    or config.get("model_config") != selected[0].get("model_config")):
                raise ValueError("Full-data phase configuration does not match selected worker")
            # run_phase uses the native newline when rewriting task.md. Its
            # logical UTF-8 text can match while LF/CRLF bytes differ on Windows.
            same_data = all((candidate_dir / "inputs" / name).read_bytes()
                            == (task_dir / "inputs" / name).read_bytes()
                            for name in benchmark.INPUT_NAMES)
            same_description = ((candidate_dir / "task.md").read_text(encoding="utf-8-sig")
                                == (task_dir / "task.md").read_text(encoding="utf-8-sig"))
            if not same_data or not same_description:
                raise ValueError("Full-data phase inputs differ from original task")
            if _hash((candidate_dir / "submission.csv").read_bytes()) != pointer["submission_sha256"]:
                continue
            durations = json.loads((candidate_dir / "durations.json").read_text(encoding="utf-8"))
            if (set(durations) != {"fit_seconds", "predict_seconds", "total_seconds"}
                    or any(type(v) not in {int, float} or not math.isfinite(v) or v < 0
                           for v in durations.values())):
                raise ValueError("Invalid completed worker timing evidence")
            logs = "\n".join(p.read_text(encoding="utf-8")
                             for p in (candidate_dir / "execution").glob("*.stdout.txt"))
            if f"refitting on all {train_rows} rows" not in logs or f"{phase} completed in " not in logs:
                raise ValueError("No source-bound full-train worker completion evidence")
            matches.append(candidate_dir)
        if len(matches) != 1:
            raise ValueError("Full-data phase is absent or ambiguous")
        return {"valid": True, "kind": "source_bound_worker_artifacts", "train_rows": train_rows,
                "phase": phase, "candidate": identifier,
                "phase_directory": str(matches[0]), "submission_sha256": pointer["submission_sha256"]}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        return {"valid": False, "kind": "source_bound_worker_artifacts", "train_rows": None,
                "reason": f"{type(error).__name__}: {error}"[:500]}


def summarize(trials: list[dict], planned_trials: int) -> dict:
    comparisons = []
    groups = sorted({(row["time_budget_seconds"], row["mode"]) for row in trials})
    for budget, mode in groups:
        rows = [row for row in trials if row["time_budget_seconds"] == budget and row["mode"] == mode]
        result = benchmark.paired_comparisons([r for r in rows if r["arm"] == "baseline"],
                                             [r for r in rows if r["arm"] == "candidate"], kind="reference")
        by_arm = {arm: {(r["case"], r["seed"]) for r in rows if r["arm"] == arm}
                  for arm in ("baseline", "candidate")}
        for case, seed in sorted(by_arm["baseline"] ^ by_arm["candidate"]):
            result["pairs"].append({"case": case, "seed": seed, "comparable": False,
                                    "reason": "missing_candidate_trial" if (case, seed) in by_arm["baseline"]
                                    else "missing_baseline_trial"})
        result["counts"] = dict(Counter(pair.get("outcome", "unpaired") for pair in result["pairs"]))
        for pair in result["pairs"]:
            pair.update(time_budget_seconds=budget, mode=mode)
            if pair.get("comparable") and not pair.get("metric", "").endswith("F1"):
                if pair["metric"] in {"RMSE", "MAE", "MSE", "LogLoss"}:
                    pair["relative_loss_improvement"] = pair["improvement"] / max(abs(pair["control"]), 1e-12)
        comparisons.append({"time_budget_seconds": budget, "mode": mode, **result})
    arms = {}
    for arm in ("baseline", "candidate"):
        rows = [row for row in trials if row["arm"] == arm]
        arms[arm] = {"status_counts": dict(Counter(row["status"] for row in rows)),
                     "valid_outputs": sum((row.get("audit") or {}).get("valid") is True for row in rows),
                     "verified_full_refits": sum((row.get("refit_evidence") or {}).get("valid") is True for row in rows),
                     "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 6),
                     "resource_measurement_available": sum(
                         ((row.get("execution") or {}).get("resources") or {}).get("available") is True for row in rows)}
    recorded_pairs = sum(len(group["pairs"]) for group in comparisons)
    return {"planned_trials": planned_trials, "recorded_trials": len(trials), "arms": arms,
            "planned_pairs": planned_trials // 2, "recorded_pairs": recorded_pairs,
            "not_started_pairs": max(0, planned_trials // 2 - recorded_pairs),
            "comparisons": comparisons, "aggregate_metric": None,
            "note": "Unlike metrics are never averaged. Valid CSVs and verified full refits are reported separately."}


def execute_prepared(output: Path) -> dict:
    output = Path(output).resolve()
    # Exclusive marker prevents stale CSVs or accidental replay of partial runs.
    manifest = _read_plan(output)
    marker = output / "execution_started.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"manifest_sha256": _hash((output / "manifest.json").read_bytes()),
                   "started_at": datetime.now(timezone.utc).isoformat()}, stream)
    if any((output / "runs").iterdir()) or (output / "report.json").exists():
        raise ValueError("Prepared execution directory is not fresh")
    started = time.monotonic()
    report = {"version": VERSION, "kind": "night_quality_comparison", "purpose": manifest["purpose"],
              "manifest_sha256": _hash((output / "manifest.json").read_bytes()),
              "sources": manifest["sources"], "environment": manifest["environment"],
              "status": "running", "trials": [], "external_api_calls": 0}
    report_path = output / "report.json"
    _write(report_path, report)
    frozen = {(case["case"], case["seed"]): case for case in manifest["cases"]}
    abort_reason = None
    for planned in manifest["trial_plan"]:
        if time.monotonic() - started + planned["budget"] > manifest["max_total_seconds"]:
            abort_reason = "series_budget_exhausted"
            break
        try:
            _check_sources(manifest)
            case = make_case(planned["case"], planned["seed"])
            expected = frozen[(planned["case"], planned["seed"])]
            if _hash(_bytes(case.audit_actuals)) != expected["audit_sha256"]:
                raise ValueError("Regenerated audit answers differ from prepared case")
            arm = planned["arm"]
            source = manifest["sources"][arm]
            row = benchmark.run_trial(case, planned["mode"], output / "runs" / f"trial_{planned['index']:04d}_{arm}",
                                      solver_root=Path(source["root"]), budget=planned["budget"],
                                      memory_mib=manifest["memory_mib"], solver_options=manifest["solver_options"][arm],
                                      expected_source_hash=source["source_sha256"])
            row.update(arm=arm, pair=planned["pair"], trial_index=planned["index"])
            if row["input_sha256"] != expected["input_sha256"]:
                row.update(status="input_changed")
                row["audit"]["metric"] = None
            row["refit_evidence"] = refit_evidence(row, expected["train_rows"])
            report["trials"].append(row)
            report["summary"] = summarize(report["trials"], manifest["planned_trials"])
            _write(report_path, report)
            print(json.dumps({"trial": planned["index"], "arm": arm, "case": case.name, "seed": case.seed,
                              "mode": row["mode"], "budget": planned["budget"], "status": row["status"],
                              "audit": row["audit"].get("metric"), "seconds": row["elapsed_seconds"],
                              "full_refit_verified": row["refit_evidence"]["valid"]}), flush=True)
            if row["status"] in {"source_changed", "input_changed"}:
                abort_reason = row["status"]
                break
        except (OSError, ValueError, RuntimeError) as error:
            abort_reason = f"{type(error).__name__}: {error}"[:500]
            break
    try:
        _read_plan(output)
    except (OSError, ValueError) as error:
        abort_reason = f"{type(error).__name__}: {error}"[:500]
    complete = (len(report["trials"]) == manifest["planned_trials"]
                and all(row["status"] == "completed" for row in report["trials"]))
    report.update(status="aborted" if abort_reason else "completed" if complete else "incomplete",
        abort_reason=abort_reason, elapsed_seconds=round(time.monotonic() - started, 6),
        finished_at=datetime.now(timezone.utc).isoformat(),
        summary=summarize(report["trials"], manifest["planned_trials"]))
    _write(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true", help="Default: freeze inputs and source hashes; no fits")
    action.add_argument("--run-prepared", type=Path, metavar="DIRECTORY", help="Execute a previously frozen fresh plan")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--cases", nargs="+", choices=ALL_CASE_NAMES, default=CASE_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 73))
    parser.add_argument("--budgets", nargs="+", type=float, default=(90,))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=("automl",))
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--candidate-policy", choices=POLICIES, default="adaptive_v1")
    parser.add_argument("--max-total-seconds", type=float)
    parser.add_argument("--purpose", choices=("development", "confirmation", "integration"), default="development")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only; source bytes still remain frozen")
    args = parser.parse_args(argv)
    if args.run_prepared is not None:
        report = execute_prepared(args.run_prepared)
        print(f"Night quality: {report['status']}; external API calls: 0; report: {args.run_prepared / 'report.json'}")
        return 0 if report["status"] == "completed" else 1
    if args.output is None or args.baseline_root is None:
        parser.error("Preparing requires --output and --baseline-root; use --run-prepared DIRECTORY to execute")
    manifest = prepare(args.output, baseline_root=args.baseline_root, candidate_root=args.candidate_root,
                       cases=args.cases, seeds=args.seeds, budgets=args.budgets, modes=args.modes,
                       memory_mib=args.memory_mib, candidate_policy=args.candidate_policy,
                       max_total_seconds=args.max_total_seconds, purpose=args.purpose, allow_dirty=args.allow_dirty)
    print(f"Prepared {manifest['planned_trials']} trials; fits/API calls: 0; "
          f"trial wall caps: {manifest['planned_trial_wall_caps_seconds']:.0f}s; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
