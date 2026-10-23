"""Prepare and verify immutable paired plans, independently of model scores."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import time
import uuid

from research_lab import INCUMBENT_SHA
from research_lab.integrity import (ROOT, require, safe_path, under, source_identity,
    check_identity, environment_versions, file_hash, file_index, json_hash, read_json, write_json, utc_now)
from research_lab.catalog import load_manifest, select_records
from research_lab.fetch import cached_member

CANDIDATE_SHA = "3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee"
COMMON_ARGS = ["--mode", "offline", "--research-search", "adaptive", "--search-profile", "legacy"]
CANDIDATE_ARGS = ["--finalist-selection", "--stability-splits", "2"]
INPUTS = ("task.md", "inputs/train.csv", "inputs/test.csv", "inputs/data_dictionary.csv", "inputs/sample_submission.csv")


def estimate(profile="pilot", *, manifest=ROOT / "research_lab/datasets.json", cache=None,
             time_budget_seconds=360, seeds=None, threads=8, memory_gib=24):
    require(profile in {"pilot", "suite", "confirmation", "final", "fixture-smoke"}, "Unknown execution profile")
    require(type(threads) is int and threads == 8,
            "Frozen native CPU solvers require exactly 8 threads")
    require(type(time_budget_seconds) is int and 1 <= time_budget_seconds <= 10500, "Invalid time budget")
    catalog = load_manifest(manifest)
    records = [] if profile == "fixture-smoke" else select_records(catalog, "suite" if profile == "final" else profile)
    if profile == "final":
        records = [r for r in records if r["dataset_id"] in {"bank_marketing", "wine_quality_red"}]
    count = 1 if profile == "fixture-smoke" else len(records)
    seeds = list(seeds) if seeds is not None else ([1729] if profile in {"pilot", "final", "fixture-smoke"} else [1729, 1730])
    require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int and 0 <= s < 2**32 for s in seeds), "Invalid seeds")
    require(profile != "pilot" or len(seeds) == 1,
            "Pilot requires exactly one seed for its frozen four-pair denominator; use suite for multiple seeds")
    arms = count * len(seeds) * 2
    cache_bytes, missing = 0, []
    for record in records:
        try:
            _, receipt = cached_member(record, cache) if cache is not None else (None, None)
            if receipt is None:
                missing.append(record["dataset_id"])
            else:
                cache_bytes += receipt["source_bytes"]
        except (OSError, ValueError):
            missing.append(record["dataset_id"])
    return {"profile": profile, "datasets": count, "seeds": seeds, "arms": arms,
            "time_budget_seconds": time_budget_seconds, "training_upper_bound_seconds": arms * time_budget_seconds,
            "cleanup_allowance_seconds_per_arm": 20, "cleanup_upper_bound_seconds": arms * 20,
            "preparation_seconds": "measured separately; no training during preparation",
            "threads": threads, "memory_limit_bytes": int(memory_gib * 1024**3),
            "memory_enforcement": "sampled child-tree RSS, not an OS memory sandbox",
            "verified_cached_source_bytes": cache_bytes, "uncached_dataset_ids": missing,
            "uncached_download_upper_bound_bytes": len(missing) * catalog["max_download_bytes"],
            "disk_note": "Task copies and trained models are additional; model footprint is not predictable before fitting",
            "direct_rental_cost": 0, "electricity_cost": None,
            "stage": "A mechanism only" if profile == "fixture-smoke" else "B development" if profile == "pilot" else "explicit C/D, never automatic"}


def _hardware():
    import psutil
    return {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(), "total_ram_bytes": psutil.virtual_memory().total,
            "gpu": "CPU lane; source/config CPU enforcement, CUDA visibility disabled, no OS GPU sandbox"}


def _fixture(template, audit_dir, seed):
    from unseen_cases import materialize_case
    from agent.task_spec import load_task
    audit = materialize_case("flow_rmse", template, seed)
    audit_path = audit_dir / "audit.json"
    write_json(audit_path, audit)
    spec = load_task(template / "task.md")
    # Native TaskSpec return shapes have differed; keep only its public dict.
    return {"dataset_id": "fixture_flow_rmse", "seed": seed, "task_template": str(template),
            "task_path": str(template / "task.md"), "audit_path": str(audit_path),
            "audit_sha256": file_hash(audit_path), "input_hashes": audit["input_sha256"],
            "spec": spec.to_dict(), "rows": {"train": 300, "audit": 80},
            "split_hashes": {"fixture": json_hash(audit["test_ids"])},
            "metadata": {"role": "mechanism_fixture", "quality_evidence": False, "metric": "RMSE", "direction": "minimize"}}


def prepare(*, output, manifest, profile, incumbent_root, candidate_root, python, cache,
            time_budget_seconds=360, threads=8, memory_gib=24, seeds=None,
            candidate_commit=CANDIDATE_SHA, self_check=False):
    preparation_started = time.monotonic()
    output, cache, manifest = safe_path(output), safe_path(cache), safe_path(manifest)
    require(not output.exists(), "Use a new output directory; old plans are never overwritten")
    require(not output.is_relative_to(safe_path(incumbent_root)) and not output.is_relative_to(safe_path(candidate_root)),
            "Keep task outputs outside solver checkouts")
    projection = estimate(profile, manifest=manifest, cache=cache, time_budget_seconds=time_budget_seconds,
                          seeds=seeds, threads=threads, memory_gib=memory_gib)
    require(1 <= memory_gib <= 24, "Configured child RSS budget must be 1..24 GiB")
    require(profile == "fixture-smoke" or time_budget_seconds > 300, "Adaptive experiments need original budget >300 s")
    incumbent, candidate = source_identity(incumbent_root), source_identity(candidate_root)
    require(incumbent["sha"] == INCUMBENT_SHA and not incumbent["dirty"], "Incumbent must be exact clean a45861d checkout")
    require(not candidate["dirty"] and candidate["sha"] == candidate_commit, "Candidate must match exact clean freeze")
    require(candidate_commit != INCUMBENT_SHA or self_check, "Equal checkouts require explicit mechanism self-check")
    require(not self_check or candidate_commit == INCUMBENT_SHA, "Self-check must compare two incumbent checkouts")
    interpreter = str(Path(python).expanduser().absolute())
    environment = environment_versions(interpreter)
    # Validate the common official subset before any split or fit. Optional
    # AutoGluon/XGBoost absence is equally visible to both arms.
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            package, version = line.strip().split("==", 1)
            require(environment["packages"].get(package) == version, "Runtime package mismatch: " + package)
    auditor = source_identity(ROOT)
    require(not auditor["dirty"], "Commit laboratory code before sealing a plan")
    catalog = load_manifest(manifest)
    records = [] if profile == "fixture-smoke" else select_records(catalog, "suite" if profile == "final" else profile)
    if profile == "final":
        records = [r for r in records if r["dataset_id"] in {"bank_marketing", "wine_quality_red"}]
    # Verify every required source before producing any executable plan.
    sources = []
    for record in records:
        require(record["status"] == "ready", "Unavailable source blocks plan: " + record["dataset_id"])
        member, receipt = cached_member(record, cache)
        receipt_path = cache / "receipts" / record["dataset_id"] / (record["version"] + ".json")
        sources.append({"dataset_id": record["dataset_id"], "object_path": str(member),
                        "sha256": receipt["sha256"], "receipt_path": str(receipt_path),
                        "receipt_sha256": file_hash(receipt_path), "receipt": receipt})
    output.mkdir(parents=True)
    write_json(output / "manifest.json", catalog)
    plan = {"schema_version": 1, "plan_id": uuid.uuid4().hex, "created_at_utc": utc_now(),
            "profile": profile, "self_check": self_check, "policy_comparison": profile != "fixture-smoke" and not self_check,
            "python": interpreter, "environment": environment, "hardware": _hardware(),
            "solvers": {"incumbent": incumbent, "candidate": candidate}, "auditor": auditor,
            "method_sha256": auditor["source_fingerprint"], "manifest_path": str(manifest),
            "manifest_sha256": file_hash(manifest), "manifest_copy_sha256": file_hash(output / "manifest.json"),
            "cache": str(cache), "sources": sources, "time_budget_seconds": time_budget_seconds,
            "threads": threads, "memory_limit_bytes": projection["memory_limit_bytes"], "estimate": projection,
            "comparison_rule": "positive signed difference means candidate better; only two verified CPU arms count as win/tie/loss; no metric averaging",
            "tie_tolerance": 1e-12, "cleanup_allowance_seconds": 20,
            "confirmation_boundary": "pilot is development; confirmation sources/splits sealed before scores; no policy tuning by evaluator",
            "fixture_boundary": "short native AutoML or native v3 is mechanism only; fixture_gate requires independently verified model artifacts, observed resources and zero API; no quality admission",
            "cases": []}
    from research_lab.adapters import prepare_case, ADAPTER_VERSION
    plan["adapter_version"], plan["adapter_sha256"] = ADAPTER_VERSION, file_hash(ROOT / "research_lab/adapters.py")
    for record in records or ([None] if profile == "fixture-smoke" else []):
        for seed in projection["seeds"]:
            dataset_id = record["dataset_id"] if record else "fixture_flow_rmse"
            key = dataset_id + "__" + str(seed)
            directory = output / "cases" / key
            template, audit_dir = directory / "template", directory / "audit"
            case = prepare_case(record, cache, template, audit_dir, seed) if record else _fixture(template, audit_dir, seed)
            case.update(key=key, tier="cpu-" + str(time_budget_seconds) + "s", arms={},
                        metric=record["metric"] if record else "RMSE",
                        direction=record["direction"] if record else "minimize",
                        role=record["role"] if record else "mechanism_fixture",
                        prior_exposure=record["prior_exposure"] if record else "Known synthetic project fixture; mechanism only",
                        arm_order=["incumbent", "candidate"] if len(plan["cases"]) % 2 == 0 else ["candidate", "incumbent"])
            for field in ("task_path", "task_template", "audit_path"):
                case[field] = safe_path(case[field]).relative_to(output).as_posix()
            for arm, solver in plan["solvers"].items():
                arm_dir = directory / "arms" / arm
                task_dir = arm_dir / "task"
                shutil.copytree(template, task_dir)
                run_id = uuid.uuid4().hex
                config_path, state_path = arm_dir / "config.json", arm_dir / "state.json"
                config = {"version": 1, "run_id": run_id, "case_key": key, "arm": arm,
                          "task_dir": str(task_dir), "task_path": str(task_dir / "task.md"),
                          "solver_root": solver["root"], "solver_identity": solver,
                          "source_fingerprint": solver["source_fingerprint"],
                          "runtime_code_hash": file_hash(ROOT / "research_lab/worker.py"),
                          "time_budget_seconds": time_budget_seconds, "threads": threads,
                          "memory_limit_bytes": plan["memory_limit_bytes"], "require_adaptive": profile != "fixture-smoke",
                          "worker_output": str(arm_dir / "worker_evidence.json"),
                          "solver_args": COMMON_ARGS + (CANDIDATE_ARGS if arm == "candidate" and not self_check else [])}
                write_json(config_path, config)
                write_json(state_path, {"status": "prepared", "run_id": run_id, "case_key": key, "arm": arm})
                case["arms"][arm] = {"run_id": run_id, "task_dir": task_dir.relative_to(output).as_posix(),
                    "config_path": config_path.relative_to(output).as_posix(), "config_sha256": file_hash(config_path),
                    "state_path": state_path.relative_to(output).as_posix()}
            plan["cases"].append(case)
    require(len(plan["cases"]) * 2 == projection["arms"], "Unexpected case denominator")
    plan["preparation_elapsed_seconds"] = time.monotonic() - preparation_started
    write_json(output / "plan.json", plan)
    write_json(output / "seal.json", {"schema_version": 1, "plan_sha256": file_hash(output / "plan.json"),
                                      "plan_id": plan["plan_id"]})
    load_plan(output)
    return plan


def load_plan(output, execution=True):
    output = safe_path(output)
    plan, seal = read_json(output / "plan.json"), read_json(output / "seal.json")
    require(plan.get("schema_version") == 1 and seal.get("plan_id") == plan.get("plan_id")
            and seal.get("plan_sha256") == file_hash(output / "plan.json"), "Plan seal mismatch")
    check_identity(plan["auditor"], strict_sha=False)
    require(safe_path(plan["auditor"]["root"]) == ROOT, "Run using the separately sealed trusted auditor root")
    for solver in plan["solvers"].values():
        check_identity(solver)
    require(file_hash(plan["manifest_path"]) == plan["manifest_sha256"]
            and file_hash(output / "manifest.json") == plan["manifest_copy_sha256"], "Manifest drift")
    catalog = load_manifest(output / "manifest.json")
    for source in plan["sources"]:
        record = next(r for r in catalog["datasets"] if r["dataset_id"] == source["dataset_id"])
        member, receipt = cached_member(record, plan["cache"])
        require(str(member) == source["object_path"] and receipt == source["receipt"]
                and file_hash(source["receipt_path"]) == source["receipt_sha256"], "Cache receipt drift")
    for case in plan["cases"]:
        require(file_hash(under(output, output / case["audit_path"])) == case["audit_sha256"], "Audit labels/split drift")
        require(file_index(under(output, output / case["task_template"])) == case["input_hashes"], "Template file set drift")
        for name, digest in case["input_hashes"].items():
            require(file_hash(under(output, output / case["task_template"] / name)) == digest, "Template input drift")
            for arm in case["arms"].values():
                require(file_hash(under(output, output / arm["task_dir"] / name)) == digest, "Arm input drift")
        for arm in case["arms"].values():
            require(file_hash(under(output, output / arm["config_path"])) == arm["config_sha256"], "Worker config drift")
            task = under(output, output / arm["task_dir"])
            expected_inputs = {name.removeprefix("inputs/"): digest for name, digest in case["input_hashes"].items() if name.startswith("inputs/")}
            require(file_index(task / "inputs") == expected_inputs, "Input directory file set drift")
            state = read_json(under(output, output / arm["state_path"]))
            if state.get("status") == "prepared":
                require(file_index(task) == case["input_hashes"], "Untouched task contains extra artifacts")
    if execution:
        require(environment_versions(plan["python"]) == plan["environment"], "Selected interpreter/dependencies changed")
    return plan
