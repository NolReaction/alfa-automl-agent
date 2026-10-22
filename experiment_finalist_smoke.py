"""Native finalist protocol smoke on two known synthetic cases, without API.

The deliberately weak mean/majority prior is a controlled protocol fixture,
not a quality baseline. No audit labels are written or scored. This is neither
a quality benchmark nor permission to enable finalist selection by default.
An existing output directory is never reused, resumed, or retried.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from agent.features import normalize_feature_config
from agent.llm import safe_error_message
from agent.model_config import make_model_config
from agent.research_candidate import freeze_candidate
from agent.research_stability import verify_research_finalists
from agent.research_v3 import BASELINE_SOURCE, V3State, _assess_candidate, finalize, prepare_research
from benchmark_cases import make_case, write_case
from validation import validate_predictions


ROOT = Path(__file__).resolve().parent
CASES = ("linear_regression", "multiclass_duplicates")
PRIOR_SOURCE = '''from collections import Counter
from statistics import fmean

def fit_predict(X_fit, y_fit, X_predict, context):
    if context["task_type"] == "classification":
        counts = Counter(map(str, y_fit))
        label = min(counts, key=lambda value: (-counts[value], value))
        return {"predictions": [label] * len(X_predict), "classes": context["classes"]}
    return {"predictions": [fmean(map(float, y_fit))] * len(X_predict)}
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def manifest(paths, base):
    return {path.relative_to(base).as_posix(): digest(path) for path in sorted(paths)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def provenance():
    files = manifest([*ROOT.glob("*.py"), *ROOT.joinpath("agent").rglob("*.py")], ROOT)
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": packages}
    git = {}
    for key, arguments in (("head", ["rev-parse", "HEAD"]),
                           ("status", ["status", "--porcelain", "--untracked-files=normal"])):
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
        git[key] = result.stdout.strip() if result.returncode == 0 else None
    git["dirty"] = bool(git["status"]) if git["status"] is not None else None
    encoded = lambda value: json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    return {"git": git, "sources": files, "sources_sha256": hashlib.sha256(encoded(files)).hexdigest(),
            "environment": environment, "environment_sha256": hashlib.sha256(encoded(environment)).hexdigest()}


def identity(candidate):
    candidate.verify()
    return candidate.code_hash + ":" + candidate.recipe_hash


def checked_receipt(path, expected_phase, expected_identity):
    path = Path(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(receipt["phase"] == expected_phase, "Receipt phase mismatch")
    require(receipt["code_hash"] + ":" + receipt["recipe_hash"] == expected_identity,
            "Receipt frozen identity mismatch")
    predictions = digest(path.parent / "predictions.csv")
    require(predictions["sha256"] == receipt["output_hash"], "Receipt output hash mismatch")
    return {"path": str(path), "receipt": digest(path), "predictions": predictions, "contents": receipt}


def run_case(name, output, time_budget, entry):
    task = write_case(make_case(name, 42), output / name)
    input_paths = [task, *task.parent.joinpath("inputs").glob("*.csv")]
    inputs = manifest(input_paths, task.parent)
    entry.update(status="running", inputs=inputs, task=str(task))
    state = V3State(task, max_steps=1, time_budget=time_budget,
                    finalist_selection=True, stability_splits=2)
    entry["run_report"] = str(state.run_dir / "run.json")
    state.report.update(mode="local_only", quality_options={"finalist_selection": True,
        "stability_splits": 2, "metric_recipes": False, "search_profile": "legacy"},
        functional_smoke={"known_synthetic": True, "controlled_weak_prior": True,
                          "quality_benchmark": False, "quality_admission": False})
    try:
        prepare_research(state)
        source = state.run_dir / "controlled_prior.py"
        source.write_text(PRIOR_SOURCE, encoding="utf-8")
        state.baseline = freeze_candidate(source, {"family": "controlled_prior", "fixture": "functional_smoke"},
                                           state.run_dir / "controlled_prior_frozen")
        state.report["baseline_search"] = {"status": "controlled_prior", "scope": "outer_fit_only"}
        state.transition("inner_research")
        baseline_result = state.measure_inner(state.baseline)
        state.baseline_inner = deepcopy(baseline_result.metric)
        state.best_inner = deepcopy(baseline_result.metric)
        require(state.can_finalize_candidate(state.baseline), "Controlled prior finalization is infeasible")
        state.finalist_pool.record(state.baseline, baseline_result.metric)
        measured = {identity(state.baseline): deepcopy(baseline_result.metric)}
        source = state.run_dir / "ordinary_source.py"
        source.write_text(BASELINE_SOURCE, encoding="utf-8")
        classification = state.gate.spec.classification
        values = (.1, 10.0) if classification else (1.0, 10.0)
        for index, value in enumerate(values):
            parameters = {"C": value, "max_iter": 800} if classification else {"alpha": value}
            recipe = {"action": "linear", "model_config": make_model_config("linear",
                state.gate.spec.metric.task_type, parameters), "feature_config": normalize_feature_config()}
            candidate = freeze_candidate(source, recipe, state.run_dir / ("ordinary_" + str(index)))
            feedback = _assess_candidate(state, candidate,
                {"hypothesis": "Controlled native functional smoke; fixed regularization alternative."}, origin="local")
            require(feedback["verified"] and feedback["timing"]["feasible_estimate"],
                    "A real ordinary trial failed measurement or finalization admission")
            measured[identity(candidate)] = deepcopy(feedback["metric"])
        finalists = state.finalist_shortlist()
        require(len(finalists) == 3 and len(measured) == 3 and len(state.finalist_pool) == 3,
                "Expected three distinct measured finalists including the weak prior")
        require({identity(item["candidate"]) for item in finalists} == set(measured), "Shortlist lost a measured identity")
        require(state.selected_candidate is not None, "Ordinary recipes did not beat the disclosed controlled prior")
        entry["canonical_finalists"] = [{"id": item["id"], "identity": identity(item["candidate"]),
                                         "metric": deepcopy(item["metric"])} for item in finalists]
        stability = verify_research_finalists(state, splits=2)
        entry["stability"] = deepcopy(stability)
        require(stability["status"] == "completed" and stability["summary"]["used_rounds"] == 2,
                "The complete three-finalist paired repeat did not finish")
        require(len(stability["finalists"]) == 3 and not stability["outer_answers_used"],
                "Stability did not compare exactly three INNER finalists")
        compared = {item["code_hash"] + ":" + item["recipe_hash"]: item["canonical_metric"]
                    for item in stability["finalists"]}
        require(compared == measured, "Stability changed frozen membership or canonical scores")
        selected = state.selected_candidate or state.baseline
        require(state.best_inner == measured[identity(selected)], "Canonical selected score was replaced by a mean")
        entry["selected_canonical"] = {"identity": identity(selected), "metric": deepcopy(state.best_inner)}
        finalize(state, "local_only")
        report = json.loads((state.run_dir / "run.json").read_text(encoding="utf-8"))
        require(report["status"] == "completed" and report["publication"]["new_result_published"],
                "Normal outer-validation/full-refit publication did not finish")
        require(report["api"]["attempts"] == 0 and report["api"]["responses"] == 0, "Unexpected API activity")
        require(report["evaluations"]["selection"]["metric"] == entry["selected_canonical"]["metric"],
                "Persisted selection score differs from the canonical selected score")
        require(manifest(input_paths, task.parent) == inputs, "Original synthetic inputs changed")
        state.check_source()
        state.gate.check_inputs()
        for item in finalists:
            item["candidate"].verify()
            require(item["metric"] == measured[identity(item["candidate"])], "A canonical finalist score changed")
        validation = validate_predictions(task.parent, spec=state.gate.spec)
        require(validation["valid"], "Published CSV failed structural validation")
        published = state.baseline if report["published_recipe"] == "baseline" else state.selected_candidate
        receipts = report["candidate_receipts"]
        outer = checked_receipt(receipts["outer"], "outer_validation", identity(published))
        refit = checked_receipt(receipts["refit"], "full_refit", identity(published))
        with (Path(receipts["refit"]).parent / "fit.csv").open(encoding="utf-8", newline="") as stream:
            require(sum(1 for _ in csv.DictReader(stream)) == len(state.gate.train), "Full refit omitted training rows")
        submission = digest(task.parent / "submission.csv")
        require(submission == refit["predictions"], "Published CSV differs from the verified full-refit output")
        entry.update(status="passed", outer_receipt=outer, refit_receipt=refit, submission=submission,
                     validation=validation, api_attempts=0, inputs_unchanged=True,
                     all_frozen_verified=True, full_refit_rows=len(state.gate.train))
    except Exception as error:
        state.report["functional_smoke_error"] = safe_error_message(error)
        state.save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory; existing paths are refused")
    parser.add_argument("--time-budget", default=180.0, type=float, help="Original per-case wall-clock budget in seconds")
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.time_budget) or not 0 < arguments.time_budget <= 10500:
        parser.error("--time-budget must be finite, positive, and at most 10500 seconds")
    output = arguments.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output already exists; smoke evidence is never reused or retried")
    started = time.monotonic()
    report = {"version": 1, "kind": "native_finalist_functional_smoke", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "quality_benchmark": False,
        "quality_admission": False, "known_synthetic": True, "controlled_weak_prior": True,
        "audit_labels_scored": False, "sequential": True, "per_case_time_budget_seconds": arguments.time_budget,
        "api_requests": 0, "provenance": None, "cases": [],
        "resources": {"host_cpu_count": os.cpu_count(), "peak_threads": None,
                      "peak_memory_bytes": None, "hard_resource_limits_verified": False}}
    destination = output / "functional_report.json"
    write_json(destination, report)
    try:
        report["provenance"] = provenance()
        for name in CASES:
            entry = {"name": name, "seed": 42, "status": "starting"}
            report["cases"].append(entry)
            write_json(destination, report)
            run_case(name, output, arguments.time_budget, entry)
            write_json(destination, report)
        report["sources_unchanged"] = provenance()["sources"] == report["provenance"]["sources"]
        require(report["sources_unchanged"], "Implementation source bytes changed during the smoke")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=safe_error_message(error))
        if report["cases"]:
            report["cases"][-1].update(status="failed", error=safe_error_message(error))
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(destination, report)
    print(f"Functional smoke {report['status']}: {destination}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
