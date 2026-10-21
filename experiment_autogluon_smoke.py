"""Two tiny native AutoGluon execution proofs; not a quality benchmark.

Run in an environment that already contains platform-pinned AutoGluon 1.3.1.
This script never installs dependencies or calls an LLM. Its hard supervisor
budget covers both sequential invocations; each candidate itself performs an
internal-policy selection fit and full-data refit of one RandomForest family.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import pandas as pd

from agent.autogluon_candidate import availability, candidate_source, make_autogluon_recipe
from agent.metrics import get_metric
from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
from agent.task_spec import TaskSpec


ROOT = Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(case):
    if case == "classification_grouped":
        classes = ("01", "NA", "class c")
        fitting = pd.DataFrame({"row_key": [f"fit-{index:04d}" for index in range(180)],
            "x": [str(index // 3) for index in range(180)],
            "kind": [classes[(index // 3) % 3] for index in range(180)],
            "answer": [classes[(index // 3) % 3] for index in range(180)]})
        prediction = pd.DataFrame({"row_key": ["0005", "NA", "q2", "q3", "q4", "q5"],
                                   "x": ["60", "61", "62", "63", "64", "65"],
                                   "kind": list(classes) * 2})
        return fitting, prediction, classes, "Macro F1", {"strategy": "grouped"}, list(classes) * 2
    fitting = pd.DataFrame({"row_key": [f"fit-{index:04d}" for index in range(180)],
        "x": [str(index // 30 + 1) for index in range(180)],
        "kind": ["NA", "0007", "7"] * 60,
        "answer": [str(2 * (index // 30 + 1) + index % 3) for index in range(180)]})
    prediction = pd.DataFrame({"row_key": ["0005", "NA", "q2", "q3", "q4", "q5"],
                               "x": ["7"] * 6, "kind": ["NA", "0007", "7"] * 2})
    return fitting, prediction, (), "RMSE", {"strategy": "time", "time_column": "x"}, None


def run(output):
    readiness = availability()
    if not readiness["available"]:
        raise RuntimeError("AutoGluon native smoke unavailable: " + readiness["reason"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, limit = time.monotonic(), 90.0
    deadline = started + limit
    source_paths = [ROOT / "agent/autogluon_candidate.py", ROOT / "experiment_autogluon_smoke.py",
                    ROOT / "agent/research_candidate.py", ROOT / "agent/splits.py"]
    sources = {str(path.relative_to(ROOT)): _sha(path) for path in source_paths}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    report = {"kind": "native_contract_smoke", "quality_benchmark": False,
              "api_requests": 0, "source_commit": commit, "sources": sources,
              "hard_total_seconds": limit, "cases": [], "status": "running",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("autogluon.tabular", "numpy", "pandas", "scikit-learn", "psutil")}}
    try:
        for case in ("classification_grouped", "regression_time"):
            directory = output / case
            directory.mkdir()
            fitting, prediction, classes, metric, policy, actuals = _inputs(case)
            spec = TaskSpec(directory / "task.md", directory, "row_key", "answer",
                ("x", "kind"), ("x",), ("kind",), classes, ("row_key", "answer"),
                (), None, get_metric(metric))
            recipe = make_autogluon_recipe(spec.metric.task_type, metric, time_limit=30,
                validation=policy, num_cpus=2, memory_limit=2, model_families=("RF",))
            source = directory / "proposal.py"
            source.write_text(candidate_source(), encoding="utf-8")
            frozen = freeze_candidate(source, recipe, directory / "frozen")
            fitting.to_csv(directory / "fit_input.csv", index=False)
            prediction.to_csv(directory / "predict_input.csv", index=False)
            phase = "inner_selection" if actuals is not None else "full_refit"
            result = run_candidate(frozen, spec=spec, fit_frame=fitting,
                predict_frame=prediction, output_dir=directory / "execution", phase=phase,
                timeout=min(45.0, deadline - time.monotonic()), deadline=deadline,
                actuals=actuals, max_memory_bytes=2 * 1024**3)
            result.verify()
            evidence_path = result.output_path.parent / "autogluon_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if evidence["full_fit_rows"] != len(fitting) or not evidence["full_refit"]:
                raise RuntimeError("Native wrapper did not prove the required full-data refit")
            rows = pd.read_csv(result.output_path, dtype=str, keep_default_na=False)
            if rows["row_key"].tolist() != prediction["row_key"].tolist():
                raise RuntimeError("Prediction literal IDs changed")
            report["cases"].append({"case": case, "phase": phase, "status": "completed",
                "output": str(result.output_path.relative_to(output)),
                "output_sha256": result.output_hash, "receipt_sha256": result.receipt_hash,
                "execution": result.execution, "evidence": evidence,
                "metric": result.metric})
        if sources != {str(path.relative_to(ROOT)): _sha(path) for path in source_paths}:
            raise RuntimeError("Smoke source changed during execution")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        if isinstance(error, CandidateExecutionError):
            report["failed_execution"] = error.execution
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    return output / "report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.output)
    value = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps({"status": value["status"], "kind": value["kind"],
        "quality_benchmark": False, "api_requests": value["api_requests"],
        "elapsed_seconds": value["elapsed_seconds"], "versions": value["versions"],
        "cases": [{"case": row["case"], "phase": row["phase"], "status": row["status"],
                   "full_fit_rows": row["evidence"]["full_fit_rows"],
                   "internal_train_rows": row["evidence"]["internal_train_rows"],
                   "internal_tuning_rows": row["evidence"]["internal_tuning_rows"],
                   "full_refit": row["evidence"]["full_refit"],
                   "best_model": row["evidence"]["best_model"],
                   "refit_model": row["evidence"]["refit_model"],
                   "output_sha256": row["output_sha256"], "receipt_sha256": row["receipt_sha256"],
                   "resources": row["execution"].get("resources")}
                  for row in value["cases"]]}, ensure_ascii=False, allow_nan=False))
    print("AutoGluon native contract smoke completed; API requests 0. Report:", report)


if __name__ == "__main__":
    main()
