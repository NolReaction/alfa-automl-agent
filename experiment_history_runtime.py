"""Frozen paired, API-free v3 history experiments with host-only audit answers.

Development and confirmation are separate execution stages. Confirmation needs
an explicit method seal written before its first execution. This is ordinary
subprocess isolation, not an operating-system sandbox or platform qualification.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

import benchmark_agent as benchmark
from benchmark_cases import BenchmarkCase, write_case
from history_runtime_cases import CASE_NAMES, explicit_history_config, make_history_case
from offline_baseline import read_csv


ROOT = Path(__file__).resolve().parent
ARMS = ("control", "history")
DEVELOPMENT_SEEDS = (131,)
CONFIRMATION_SEEDS = (2203, 4001)
CONFIRMATION_CASES = ("entity_signal", "colliding_keys")
WEIGHT = 0.5
HARNESSES = ("experiment_history_runtime.py", "history_runtime_cases.py", "benchmark_agent.py", "benchmark_cases.py")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular artifact: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    benchmark._write_json(Path(path), value)


def _read(path):
    _hash(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _within(base, value):
    base, path = Path(base).resolve(), Path(value)
    path = path if path.is_absolute() else base / path
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("Artifact points outside its invocation")
    current = base
    for part in path.absolute().relative_to(base).parts:
        if part in {".", ".."}:
            raise ValueError("Artifact path contains a traversal")
        current = current / part
        if current.is_symlink():
            raise ValueError("Linked artifacts are not evidence")
    _hash(resolved)
    return resolved


def preflight(solver_root):
    """Read source/signature/dependency availability; never construct an API client."""
    solver_root = Path(solver_root).resolve()
    source = benchmark.source_provenance(solver_root)
    tree = ast.parse((solver_root / "agent/research_v3.py").read_text(encoding="utf-8"))
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_agent_v3"), None)
    arguments = set() if function is None else {a.arg for a in (*function.args.args, *function.args.kwonlyargs)}
    missing = sorted({"local_only", "history_config", "history_weight", "time_budget"} - arguments)
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "pandas", "sklearn", "psutil", "openai", "catboost", "lightgbm")}
    return {"ready": not missing and source["worktree_dirty"] is False and all(packages.values()),
            "source": source, "missing_interface": missing, "packages": packages,
            "api_calls": 0, "note": "Source readiness only; no training or API availability/quality test."}


def _split(task):
    from agent.automl import choose_validation
    from agent.search import inner_split_rows
    from agent.splits import split_rows
    from agent.task_spec import load_task, load_task_data

    spec = load_task(Path(task))
    train, test, _, _, _, y, features, numeric, _, classification = load_task_data(spec)
    choice, _ = choose_validation(train, test, features, numeric, classification)
    fitting, held, metadata = split_rows(train, y, features, classification, choice)
    inner_fit, inner_held, inner_metadata = inner_split_rows(train, y, features, classification, choice, fitting)
    # inner_split_rows above returns global positions; v3's work view uses the
    # equivalent local positions, while the frozen hash retains literal IDs.
    signature = benchmark._json_hash({"choice": choice, "outer_fit": train[spec.id_column].iloc[fitting].tolist(),
        "outer_held": train[spec.id_column].iloc[held].tolist(),
        "inner_fit": train[spec.id_column].iloc[inner_fit].tolist(),
        "inner_held": train[spec.id_column].iloc[inner_held].tolist()})
    return {"spec": spec, "train": train, "test": test, "choice": choice, "metadata": metadata,
            "inner_metadata": inner_metadata, "outer_fit": fitting, "outer_held": held,
            "inner_fit": inner_fit, "inner_held": inner_held, "signature": signature}


def _public_case(task):
    """Known public TRAIN year5 sanity only; key choice stays in this experiment."""
    from agent.task_spec import load_task, load_task_data

    spec = load_task(Path(task))
    train, _, _, _, _, y, features, numeric, categorical, classification = load_task_data(spec)
    keys = ["State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION"]
    if classification or spec.metric.name != "RMSE" or not set([*keys, "Year_Factor"]) <= set(features):
        raise ValueError("Optional public sanity expects the explicitly documented regression schema")
    import pandas as pd
    periods = pd.to_numeric(train["Year_Factor"], errors="raise")
    fitting, held = np.flatnonzero(periods < 5), np.flatnonzero(periods == 5)
    if not len(fitting) or not len(held):
        raise ValueError("Public year5 split is empty")
    fit = train.iloc[fitting].loc[:, features].reset_index(drop=True).copy()
    test = train.iloc[held].loc[:, features].reset_index(drop=True).copy()
    fit.insert(0, "row_key", train[spec.id_column].iloc[fitting].tolist())
    test.insert(0, "row_key", train[spec.id_column].iloc[held].tolist())
    fit["outcome"] = y.iloc[fitting].tolist()
    case = BenchmarkCase("known_public_year5", 0, "RMSE", (), None,
        {"strategy": "time", "time_column": "Year_Factor"}, tuple(numeric), tuple(categorical),
        fit, test, tuple(map(float, y.iloc[held])), "Known public TRAIN sanity. Validate strictly on later periods.")
    config = {"key_columns": keys, "period_column": "Year_Factor",
              "numeric_columns": ["floor_area", "year_built", "ELEVATION"],
              "positive_columns": ["year_built"], "ambiguity_policy": "allow"}
    inputs = {str(path.resolve()): _hash(path) for path in [spec.task_path, *sorted(spec.data_dir.glob("*.csv"))]}
    return case, config, inputs


def prepare(output, *, solver_root, budget=90, memory_mib=6144, total_seconds=1800, public_task=None, smoke=False):
    """Freeze cases/method/source before fitting. No audit scores are computed."""
    for value, limit in ((budget, 10500), (total_seconds, 24 * 3600)):
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError("Budgets must be positive finite seconds within limits")
    if type(memory_mib) is not int or not 1 <= memory_mib <= benchmark.MAX_MEMORY_MIB:
        raise ValueError("Invalid configured memory cap")
    if type(smoke) is not bool or (smoke and public_task is not None):
        raise ValueError("Smoke is a separate synthetic-only plan")
    check = preflight(solver_root)
    if not check["ready"]:
        raise ValueError("Preflight is not ready; publish a clean solver with the required native interface")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("prepared_inputs", "host_audit", "runs", "results"):
        (output / name).mkdir()
    definitions = []
    originals = {}
    if smoke:
        definitions.append(("development", make_history_case("entity_signal", 131), explicit_history_config("entity_signal")))
    elif public_task is None:
        for stage, names, seeds in (("development", CASE_NAMES, DEVELOPMENT_SEEDS),
                                   ("confirmation", CONFIRMATION_CASES, CONFIRMATION_SEEDS)):
            for name in names:
                for seed in seeds:
                    definitions.append((stage, make_history_case(name, seed), explicit_history_config(name)))
    else:
        case, config, originals = _public_case(public_task)
        definitions.append(("public_sanity", case, config))
    cases, pairs = [], []
    for index, (stage, case, config) in enumerate(definitions):
        key = f"{case.name}_{case.seed}"
        task = write_case(case, output / "prepared_inputs" / key)
        split = _split(task)
        if split["choice"] != case.validation:
            raise ValueError("Native validation differs from the predeclared case policy")
        audit = output / "host_audit" / (key + ".json")
        _write(audit, {"actuals": case.audit_actuals, "sha256": benchmark._json_hash(case.audit_actuals)})
        cases.append({"key": key, "name": case.name, "seed": case.seed, "stage": stage,
            "task": task.relative_to(output).as_posix(), "input_sha256": benchmark.input_fingerprint(task),
            "audit": audit.relative_to(output).as_posix(), "audit_file_sha256": _hash(audit),
            "audit_sha256": benchmark._json_hash(case.audit_actuals), "history_config": config,
            "validation": case.validation, "split_sha256": split["signature"],
            "train_rows": len(case.train), "audit_rows": len(case.test)})
        pairs.append({"case": key, "stage": stage, "arms": list(ARMS if index % 2 == 0 else reversed(ARMS))})
    source = check["source"]
    method = {"version": 1, "source_sha256": source["source_sha256"],
        "harness_sha256": {name: _hash(ROOT / name) for name in HARNESSES},
        "history_weight": WEIGHT, "local_only": True, "search_profile": "legacy", "smoke_only": smoke,
        "stability_splits": 1, "metric_recipes": False, "budget_seconds_per_trial": float(budget),
        "memory_mib": memory_mib, "total_seconds": float(total_seconds),
        "development_seeds": list(DEVELOPMENT_SEEDS), "reserved_seed": 379,
        "confirmation_cases": list(CONFIRMATION_CASES), "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "comparison": "Equal walltime native v3 local_only; baseline-match flag, not a pure-effect guarantee",
        "history_configs": {case["key"]: case["history_config"] for case in cases}}
    manifest = {"version": 1, "kind": "prepared_history_runtime", "prepared_at": _now(),
        "source": source, "host_source": benchmark.source_provenance(ROOT), "method": method,
        "method_sha256": benchmark._json_hash(method), "environment": benchmark._versions(),
        "original_inputs": originals, "cases": cases, "pairs": pairs,
        "scope": "Synthetic developer experiment; optional previously researched public TRAIN sanity, not platform/private score",
        "input_boundary": "Host audit labels outside task directories; no OS sandbox", "api_calls": 0}
    _write(output / "manifest.json", manifest)
    _write(output / "seal.json", {"manifest_sha256": _hash(output / "manifest.json")})
    _write(output / "budget.json", {"version": 1, "consumed_seconds": 0.0, "in_progress": None})
    return manifest


def read_plan(output):
    output = Path(output).resolve()
    if _read(output / "seal.json") != {"manifest_sha256": _hash(output / "manifest.json")}:
        raise ValueError("Prepared manifest changed")
    manifest = _read(output / "manifest.json")
    if manifest.get("kind") != "prepared_history_runtime" or manifest.get("version") != 1:
        raise ValueError("Unknown experiment plan")
    if benchmark._json_hash(manifest["method"]) != manifest["method_sha256"]:
        raise ValueError("Method changed")
    if benchmark._versions() != manifest["environment"]:
        raise ValueError("Frozen package/runtime versions changed")
    for source in (manifest["source"], manifest["host_source"]):
        if benchmark.source_provenance(Path(source["root"]))["source_sha256"] != source["source_sha256"]:
            raise ValueError("Frozen solver/auditor source changed")
    for name, digest in manifest["method"]["harness_sha256"].items():
        if _hash(ROOT / name) != digest:
            raise ValueError("Experiment harness changed")
    for name, digest in manifest["original_inputs"].items():
        if _hash(name) != digest:
            raise ValueError("Original public input changed")
    for case in manifest["cases"]:
        task = _within(output, case["task"])
        if benchmark.input_fingerprint(task) != case["input_sha256"] or _hash(_within(output, case["audit"])) != case["audit_file_sha256"]:
            raise ValueError("Prepared input or host audit changed")
    return manifest


def freeze_confirmation(output):
    """Seal the unchanged method before accessing confirmation model outcomes."""
    output = Path(output).resolve()
    plan = read_plan(output)
    path = output / "confirmation_freeze.json"
    if path.exists() or (output / "results/confirmation.json").exists():
        raise ValueError("Confirmation already frozen/started; never replace its method seal")
    development = _read(output / "results/development.json")
    if development["status"] != "completed":
        raise ValueError("Complete development before freezing confirmation")
    value = {"version": 1, "method_sha256": plan["method_sha256"],
        "manifest_sha256": _hash(output / "manifest.json"), "development_report_sha256": _hash(output / "results/development.json"),
        "frozen_at": _now()}
    _write(path, value)
    return value


def _check_confirmation_freeze(output, plan):
    frozen = _read(output / "confirmation_freeze.json")
    if (frozen.get("version") != 1 or frozen.get("method_sha256") != plan["method_sha256"]
            or frozen.get("manifest_sha256") != _hash(output / "manifest.json")
            or frozen.get("development_report_sha256") != _hash(output / "results/development.json")):
        raise ValueError("Confirmation method/development report changed after freeze")


def _driver(solver_root, task):
    return ("import time\nstarted=time.monotonic()\nimport sys,json,hashlib\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(solver_root)!r})\nfrom agent.research_v3 import run_agent_v3\n"
        f"task=Path({str(task)!r})\nrequest=json.loads((task.parent/'driver_request.json').read_text(encoding='utf-8'))\n"
        "evidence={'version':1,'mode':'local_only','api_calls':0,'run_report':None}\ntry:\n"
        "    remaining=request['time_budget_seconds']-(time.monotonic()-started)\n"
        "    if remaining<=0: raise TimeoutError('Import preparation exhausted the original trial budget')\n"
        "    run_dir=run_agent_v3(task,None,time_budget=remaining,local_only=True,"
        "history_config=request['history_config'],history_weight=request['history_weight'])\n"
        "    report=(run_dir/'run.json').resolve()\n"
        "    evidence.update(run_report=report.relative_to(task.parent.resolve()).as_posix(),"
        "run_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())\nfinally:\n"
        "    (task.parent/'driver_evidence.json').write_text(json.dumps(evidence,allow_nan=False),encoding='utf-8')\n")


def _receipt(invocation, frozen, *, spec, fitting, predicting, actuals, phase):
    """Recheck input rows, nonce/code/recipe/output receipt and host score."""
    from agent.metrics import score_predictions

    invocation = Path(invocation)
    names = ("candidate.py", "recipe.json", "context.json", "fit.csv", "predict.csv", "predictions.csv", "receipt.json")
    artifacts = {str(invocation / name): _hash(_within(invocation, name)) for name in names}
    context, receipt = _read(invocation / "context.json"), _read(invocation / "receipt.json")
    if artifacts[str(invocation / "candidate.py")] != frozen["code_hash"] or artifacts[str(invocation / "recipe.json")] != frozen["recipe_hash"]:
        raise ValueError("Frozen implementation/recipe changed")
    expected = {"version": 3, "nonce": context["nonce"], "phase": phase, **frozen,
                "output_hash": artifacts[str(invocation / "predictions.csv")]}
    if receipt != expected or context["phase"] != phase or any(context[k] != frozen[k] for k in frozen):
        raise ValueError("Receipt does not identify this exact invocation")
    if context["recipe"] != _read(invocation / "recipe.json") or context["features"] != list(spec.features):
        raise ValueError("Context recipe/schema differs from the frozen task")
    fit_columns, predict_columns = [spec.id_column, *spec.features, spec.target_column], [spec.id_column, *spec.features]
    if not read_csv(invocation / "fit.csv").equals(fitting.loc[:, fit_columns].reset_index(drop=True)):
        raise ValueError("Fit rows/targets differ from the frozen split")
    if not read_csv(invocation / "predict.csv").equals(predicting.loc[:, predict_columns].reset_index(drop=True)):
        raise ValueError("Prediction rows differ from the frozen split")
    frame = read_csv(invocation / "predictions.csv")
    spec.validate_frame(frame, predicting[spec.id_column].tolist())
    metric = None if actuals is None else {"name": spec.metric.name, "higher_is_better": spec.metric.higher_is_better,
        "value": score_predictions(spec.metric, actuals, frame[spec.target_column].tolist())}
    return {"metric": metric, "artifacts": artifacts, "output_hash": receipt["output_hash"]}


def inspect_trial(task, case, request, actuals):
    """Audit a returned v3 run without executing any candidate or fitting models."""
    task = Path(task)
    if benchmark.input_fingerprint(task) != case["input_sha256"]:
        raise ValueError("Solver task inputs changed after preparation")
    if _read(task.parent / "driver_request.json") != request:
        raise ValueError("Driver request changed")
    evidence = _read(task.parent / "driver_evidence.json")
    if evidence.get("mode") != "local_only" or evidence.get("api_calls") != 0 or not re.fullmatch(r"\.agent_runs/[0-9a-f]{32}/run\.json", evidence.get("run_report") or ""):
        raise ValueError("No exact current local-only run report")
    report_path = _within(task.parent, evidence["run_report"])
    if _hash(report_path) != evidence.get("run_report_sha256"):
        raise ValueError("Run report changed after the driver returned")
    run, report = report_path.parent, _read(report_path)
    if report.get("research_protocol_version") != 3 or report.get("mode") != "local_only" or any(report.get("api", {}).get(k) != 0 for k in ("attempts", "responses")):
        raise ValueError("Not a verified API-free native v3 local-only run")
    startup = report.get("baseline_search", {})
    if startup.get("status") != "searched" or startup.get("scope") != "outer_fit_only":
        raise ValueError("Real native AutoML baseline preparation was not exercised")
    split = _split(task)
    if split["signature"] != case["split_sha256"] or report["host_validation"]["split"] != split["metadata"]:
        raise ValueError("Runtime split differs from the prepared native split")
    publication = report.get("publication", {})
    selected = report.get("published_recipe")
    if not publication.get("new_result_published") or selected not in {"baseline", "candidate"} or publication.get("recipe") != selected:
        raise ValueError("No newly trained full-data result was published")
    frozen = report["frozen_" + selected]
    refs = report["candidate_receipts"]
    if any(refs.get(k) != frozen[k] for k in ("code_hash", "recipe_hash")):
        raise ValueError("Published receipts differ from the frozen winner")
    outer_path = _within(run / "outer_evaluations" / selected, refs["outer"])
    refit_path = _within(run / "full_refit" / selected, refs["refit"])
    spec, train, test = split["spec"], split["train"], split["test"]
    outer = _receipt(outer_path.parent, frozen, spec=spec, fitting=train.iloc[split["outer_fit"]],
        predicting=train.iloc[split["outer_held"]], actuals=train[spec.target_column].iloc[split["outer_held"]].tolist(), phase="outer_validation")
    refit = _receipt(refit_path.parent, frozen, spec=spec, fitting=train, predicting=test, actuals=None, phase="full_refit")
    if outer["metric"] != report["best_metric"] or outer["metric"] != report["evaluations"]["final_check"][selected]:
        raise ValueError("Outer metric differs from independent re-scoring")
    published = _hash(task.parent / "submission.csv")
    if published != refit["output_hash"] or _hash(run / "last_valid_submission.csv") != published:
        raise ValueError("Publication/checkpoint differs from the full refit receipt")
    audit = benchmark.audit_submission(task, tuple(actuals))
    if not audit["valid"]:
        raise ValueError("External audit CSV invalid: " + str(audit["error"]))
    history = report.get("history_research", {})
    requested = request["history_config"] is not None
    if history.get("version") != 1 or history.get("policy") != "explicit_v1" or history.get("scope") != "inner_selection" or history.get("origin") != "host_history" or history.get("requested") is not requested:
        raise ValueError("History request/origin accounting is missing or incorrect")
    if history.get("blend_weight") != WEIGHT or history.get("history_config") != request["history_config"]:
        raise ValueError("History controls changed")
    eligible = split["choice"]["strategy"] == "time"
    if not requested and (history.get("status") != "disabled" or history.get("attempted") != 0):
        raise ValueError("Control arm unexpectedly executed history")
    if requested and not eligible and (history.get("status") != "skipped" or history.get("attempted") != 0 or history.get("active") is not False):
        raise ValueError("History was not skipped on an ineligible IID policy")
    artifacts = {str(report_path): _hash(report_path), str(task.parent / "driver_request.json"): _hash(task.parent / "driver_request.json"),
        str(task.parent / "driver_evidence.json"): _hash(task.parent / "driver_evidence.json"),
        str(task.parent / "submission.csv"): published, str(run / "last_valid_submission.csv"): published,
        **outer["artifacts"], **refit["artifacts"]}
    history_verified = False
    if requested and history.get("status") == "completed":
        if history.get("attempted") != 1 or not history.get("feedback", {}).get("verified") or history.get("baseline") != report["frozen_baseline"]:
            raise ValueError("History trial is not tied to the original frozen baseline")
        matches = []
        for path in (run / "inner_evaluations").glob("*/receipt.json"):
            receipt = _read(_within(run / "inner_evaluations", path))
            if all(receipt.get(k) == history["candidate"][k] for k in ("code_hash", "recipe_hash")):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError("History requires one exact inner-trial receipt")
        inner = _receipt(matches[0].parent, history["candidate"], spec=spec, fitting=train.iloc[split["inner_fit"]],
            predicting=train.iloc[split["inner_held"]], actuals=train[spec.target_column].iloc[split["inner_held"]].tolist(), phase="inner_selection")
        if inner["metric"] != history["feedback"]["metric"]:
            raise ValueError("History inner feedback differs from its prediction receipt")
        artifacts.update(inner["artifacts"])
        history_verified = True
    for name in ("research_candidate.py", "research_recipe.json", "frozen_candidate.json"):
        path = _within(run / "baseline_frozen", name)
        artifacts[str(path)] = _hash(path)
    if _hash(run / "baseline_frozen/research_candidate.py") != report["frozen_baseline"]["code_hash"] or _hash(run / "baseline_frozen/research_recipe.json") != report["frozen_baseline"]["recipe_hash"]:
        raise ValueError("Recovery baseline changed")
    return {"audit": audit, "run_report": report_path.relative_to(task.parent).as_posix(),
        "baseline": report["frozen_baseline"], "history": history, "history_verified": history_verified,
        "history_expected_eligible": eligible, "full_refit_verified": True,
        "local_research": report.get("local_research"), "final_reason": report.get("reason"),
        "artifacts": {str(Path(name).relative_to(task.parent)): digest for name, digest in artifacts.items()}}


def _trial(output, plan, case, arm):
    directory = output / "runs" / (case["key"] + "_" + arm)
    shutil.copytree(output / Path(case["task"]).parent, directory)
    task = directory / "task.md"
    request = {"time_budget_seconds": plan["method"]["budget_seconds_per_trial"], "local_only": True,
        "history_config": case["history_config"] if arm == "history" else None, "history_weight": WEIGHT}
    _write(directory / "driver_request.json", request)
    row = {"case": case["key"], "arm": arm, "task": task.relative_to(output).as_posix(), "request": request,
        "input_sha256": case["input_sha256"], "split_sha256": case["split_sha256"], "status": "running"}
    started = time.monotonic()
    execution = benchmark.run_python(_driver(Path(plan["source"]["root"]), task), directory, directory / "execution", 0,
        timeout=request["time_budget_seconds"], max_memory_bytes=plan["method"]["memory_mib"] * 1024**2)
    row.update(execution=execution, elapsed_seconds=time.monotonic() - started, status="failed")
    if execution.get("exit_code") == 0 and not any(execution.get(k) for k in ("error", "timed_out", "memory_limit_exceeded")):
        try:
            audit = _read(output / case["audit"])
            if audit["sha256"] != case["audit_sha256"] or benchmark._json_hash(audit["actuals"]) != case["audit_sha256"]:
                raise ValueError("Host audit answers changed")
            row["verified"] = inspect_trial(task, case, request, audit["actuals"])
            row["status"] = "completed"
        except (OSError, ValueError, TypeError, KeyError) as error:
            row["verification_error"] = f"{type(error).__name__}: {error}"
    if benchmark.input_fingerprint(task) != case["input_sha256"]:
        raise ValueError("Solver task inputs changed")
    return row


def run_stage(output, stage="development"):
    output = Path(output).resolve()
    plan = read_plan(output)
    pairs = [pair for pair in plan["pairs"] if pair["stage"] == stage]
    if not pairs:
        raise ValueError("No requested stage in the frozen plan")
    if stage == "confirmation":
        _check_confirmation_freeze(output, plan)
    report_path = output / "results" / (stage + ".json")
    if report_path.exists():
        raise ValueError("Stage already started; preserve it and re-audit rather than overwriting")
    budget = _read(output / "budget.json")
    if budget["in_progress"] is not None:
        raise ValueError("An interrupted coordinator cannot reset its budget")
    report = {"version": 1, "stage": stage, "method_sha256": plan["method_sha256"], "started_at": _now(),
        "status": "running", "api_calls": 0, "trials": [], "censored_pairs": []}
    budget["in_progress"] = stage
    _write(output / "budget.json", budget)
    _write(report_path, report)
    started = time.monotonic()
    try:
        for pair in pairs:
            read_plan(output)
            remaining = plan["method"]["total_seconds"] - budget["consumed_seconds"] - (time.monotonic() - started)
            if remaining < 2 * plan["method"]["budget_seconds_per_trial"] + 5:
                report["censored_pairs"].append({"case": pair["case"], "reason": "coordinator_budget"})
                continue
            case = next(item for item in plan["cases"] if item["key"] == pair["case"])
            for arm in pair["arms"]:
                row = _trial(output, plan, case, arm)
                report["trials"].append(row)
                _write(report_path, report)
                read_plan(output)
                print(json.dumps({"stage": stage, "case": case["key"], "arm": arm, "status": row["status"],
                    "seconds": row["elapsed_seconds"], "audit": row.get("verified", {}).get("audit", {}).get("metric")}), flush=True)
        report["status"] = "completed" if not report["censored_pairs"] and len(report["trials"]) == 2 * len(pairs) and all(row["status"] == "completed" for row in report["trials"]) else "incomplete"
    finally:
        elapsed = time.monotonic() - started
        report.update(elapsed_seconds=elapsed, finished_at=_now())
        if report["status"] == "running":
            report["status"] = "interrupted"
        budget.update(consumed_seconds=budget["consumed_seconds"] + elapsed, in_progress=None)
        _write(output / "budget.json", budget)
        _write(report_path, report)
    return report


def reaudit(output, stage="development"):
    """Re-read all persisted proof; never train or silently repair artifacts."""
    output = Path(output).resolve()
    plan = read_plan(output)
    if stage == "confirmation":
        _check_confirmation_freeze(output, plan)
    report = _read(output / "results" / (stage + ".json"))
    rows, pairs = [], []
    for row in report["trials"]:
        item = {"case": row["case"], "arm": row["arm"], "status": row["status"], "verified": False}
        if row["status"] == "completed":
            case = next(c for c in plan["cases"] if c["key"] == row["case"])
            try:
                task = _within(output, row["task"])
                current = inspect_trial(task, case, row["request"], _read(output / case["audit"])["actuals"])
                if current != row["verified"]:
                    raise ValueError("Current proof differs from the committed trial evidence")
                item.update(verified=True, metric=current["audit"]["metric"], baseline=current["baseline"],
                    history_status=current["history"].get("status"), history_verified=current["history_verified"],
                    history_expected_eligible=current["history_expected_eligible"], final_reason=current["final_reason"],
                    seconds=row["elapsed_seconds"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                item["error"] = f"{type(error).__name__}: {error}"
        rows.append(item)
    for key in dict.fromkeys(row["case"] for row in rows):
        members = {row["arm"]: row for row in rows if row["case"] == key}
        pair = {"case": key, "comparable_end_to_end": all(members.get(arm, {}).get("verified") for arm in ARMS)}
        if pair["comparable_end_to_end"]:
            control, history = members["control"], members["history"]
            pair.update(control=control["metric"], history=history["metric"],
                signed_improvement=control["metric"]["value"] - history["metric"]["value"],
                baseline_recipe_match=control["baseline"] == history["baseline"],
                effect_scope="Equal-walltime end-to-end; differing local admissions may also contribute")
        pairs.append(pair)
    return {"version": 1, "stage": stage, "method_sha256": plan["method_sha256"], "report_sha256": _hash(output / "results" / (stage + ".json")),
        "status": "completed" if report["status"] == "completed" and all(row["verified"] for row in rows) else "incomplete",
        "api_calls": 0, "rows": rows, "pairs": pairs,
        "resource_note": "Configured memory cap is not measured RSS or proof of enforcement when process inventory is unavailable. Local cgroup is not platform qualification."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight"); check.add_argument("--solver-root", type=Path, required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--solver-root", type=Path, required=True); preparation.add_argument("--out-dir", type=Path, required=True)
    preparation.add_argument("--time-budget", type=float, default=90); preparation.add_argument("--memory-mib", type=int, default=6144)
    preparation.add_argument("--total-seconds", type=float, default=1800); preparation.add_argument("--public-task", type=Path)
    preparation.add_argument("--smoke", action="store_true", help="One synthetic pair for protocol execution only")
    freeze = sub.add_parser("freeze-confirmation"); freeze.add_argument("--out-dir", type=Path, required=True)
    for name in ("run", "audit"):
        command = sub.add_parser(name); command.add_argument("--out-dir", type=Path, required=True)
        command.add_argument("--stage", choices=("development", "confirmation", "public_sanity"), default="development")
    args = parser.parse_args(argv)
    if args.command == "preflight": result = preflight(args.solver_root)
    elif args.command == "prepare": result = prepare(args.out_dir, solver_root=args.solver_root, budget=args.time_budget,
        memory_mib=args.memory_mib, total_seconds=args.total_seconds, public_task=args.public_task, smoke=args.smoke)
    elif args.command == "freeze-confirmation": result = freeze_confirmation(args.out_dir)
    elif args.command == "run": result = run_stage(args.out_dir, args.stage)
    else: result = reaudit(args.out_dir, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result.get("ready") is False or result.get("status") in {"incomplete", "interrupted"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
