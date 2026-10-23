"""Small serial development runner; never imported by the competition solution.

Reuses the existing execution/frozen-candidate hosts. No new scheduler, API,
platform-score estimate or competition TEST label lookup is implemented here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from agent.execution import run_python
from agent.task_spec import load_task

ROOT = Path(__file__).resolve().parent
CSV = {"dtype": str, "keep_default_na": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
            "root": str(root)}


def prepare(task, output, period_column, size_column):
    output, task = Path(output).resolve(), Path(task).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    spec = load_task(task)
    if spec.metric.name != "RMSE" or period_column not in spec.numeric or size_column not in spec.numeric:
        raise ValueError("Need regression RMSE and explicitly declared numeric period/size")
    train = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    periods = pd.to_numeric(train[period_column], errors="raise")
    relative = spec.data_dir.relative_to(task.parent)
    entries = {}
    for year in (2, 3, 4, 5):
        fitting, held = train.loc[periods < year].copy(), train.loc[periods == year].copy()
        if fitting.empty or held.empty:
            raise ValueError("Predeclared fold is empty")
        case = output / "inputs" / f"y{year}"
        data = case / relative
        data.mkdir(parents=True)
        shutil.copyfile(task, case / "task.md")
        shutil.copyfile(spec.data_dir / "data_dictionary.csv", data / "data_dictionary.csv")
        fitting.to_csv(data / "train.csv", index=False)
        held.drop(columns=spec.target_column).to_csv(data / "test.csv", index=False)
        pd.DataFrame({spec.id_column: held[spec.id_column], spec.target_column: "0"}).to_csv(
            data / "sample_submission.csv", index=False)
        answers = output / "host_only" / f"y{year}.csv"
        answers.parent.mkdir(exist_ok=True)
        held.loc[:, [spec.id_column, spec.target_column]].to_csv(answers, index=False)
        entries[str(year)] = {"fit_rows": len(fitting), "predict_rows": len(held),
            "fit_id_sha256": hashlib.sha256(json.dumps(fitting[spec.id_column].tolist()).encode()).hexdigest(),
            "predict_id_sha256": hashlib.sha256(json.dumps(held[spec.id_column].tolist()).encode()).hexdigest(),
            "files": {str(p.relative_to(output)): digest(p) for p in [*sorted(case.rglob("*.csv")), case / "task.md", answers]}}
    write(output / "plan.json", {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "period_column": period_column, "size_column": size_column, "development": [4, 2, 3],
        "known_compatibility": [5], "source_task": str(task),
        "source_files": {p.name: digest(p) for p in [task, *sorted(spec.data_dir.glob("*.csv"))]},
        "plan_document_sha256": digest(ROOT / "docs/rescue_regression_plan.md"), "folds": entries,
        "scope": "Previously exposed public TRAIN development, not independent platform evaluation"})


def plan(output):
    output = Path(output).resolve()
    result = read(output / "plan.json")
    for entry in result["folds"].values():
        for path, sha in entry["files"].items():
            if digest(output / path) != sha:
                raise ValueError("Prepared input changed: " + path)
    return result


def score_csv(task, answers, prediction):
    spec = load_task(task)
    truth, frame = pd.read_csv(answers, **CSV), pd.read_csv(prediction, **CSV)
    spec.validate_frame(frame, truth[spec.id_column].tolist())
    actual = pd.to_numeric(truth[spec.target_column]).to_numpy(dtype=float)
    predicted = pd.to_numeric(frame[spec.target_column]).to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite regression values")
    error = predicted - actual
    return {"rows": len(actual), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "sse": float(np.dot(error, error)),
        "ids_exact_order": True, "finite": True, "prediction_sha256": digest(prediction),
        "answers_sha256": digest(answers)}


def native(output, checkout, year, budget):
    output, checkout = Path(output).resolve(), Path(checkout).resolve()
    plan(output)
    if year == 5 and not (output / "frozen_decision.json").is_file():
        raise ValueError("Freeze the development decision before known year 5")
    before = source(checkout)
    if before["dirty"]:
        raise ValueError("Native incumbent requires a clean immutable checkout")
    directory = output / "native" / f"y{year}"
    shutil.copytree(output / "inputs" / f"y{year}", directory)
    task = directory / "task.md"
    args = [str(checkout / "solution.py"), str(task), "--mode", "offline", "--time-budget", str(budget)]
    code = ("import runpy,sys\n" + f"sys.path.insert(0,{str(checkout)!r})\n"
            + f"sys.argv={args!r}\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    started = time.monotonic()
    execution = run_python(code, directory, directory / "launcher", 0,
                           timeout=budget + 30, max_memory_bytes=32 * 1024**3)
    after = source(checkout)
    if before != after:
        raise ValueError("Native source changed during measurement")
    prediction = directory / "submission.csv"
    reports = {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*.json")
               if p.name in {"run.json", "automl_report.json", "receipt.json", "publication_receipt.json"}}
    result = {"source": before, "budget_seconds": budget, "elapsed_seconds": time.monotonic() - started,
        "execution": execution, "arguments": args[2:], "native_reports": reports,
        "metrics": score_csv(task, output / "host_only" / f"y{year}.csv", prediction) if prediction.is_file() else None,
        "submission": str(prediction), "api_mode": "offline", "api_calls": 0,
        "comparison": "Native incumbent reproduction, isolated-candidate pairing remains pending"}
    write(directory / "execution_summary.json", result)
    print(json.dumps({"year": year, "metrics": result["metrics"], "elapsed": result["elapsed_seconds"]}), flush=True)
    return result


def case_frames(output, year):
    output = Path(output).resolve()
    spec = load_task(output / "inputs" / f"y{year}" / "task.md")
    fitting = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    held = pd.read_csv(spec.data_dir / "test.csv", **CSV)
    answers = pd.read_csv(output / "host_only" / f"y{year}.csv", **CSV)
    if held[spec.id_column].tolist() != answers[spec.id_column].tolist():
        raise ValueError("Host answers lost their frozen row order")
    return spec, fitting, held, answers


def discover(output, year):
    from agent.regression_discovery import discover_regression_schema_bounded
    output = Path(output).resolve()
    frozen = plan(output)
    spec, fitting, _, _ = case_frames(output, year)
    periods = pd.to_numeric(fitting[frozen["period_column"]], errors="raise")
    # Exactly the direct candidate's fit-internal earlier rows. A native host
    # must supply ITS INNER-fit view instead; it may be smaller due to nesting.
    inner = fitting.loc[periods < periods.max(), list(spec.features)].copy()
    for extended in (False, True):
        name = "extended" if extended else "original"
        result = discover_regression_schema_bounded(inner, feature_columns=list(spec.features),
            numeric_columns=list(spec.numeric), period_column=frozen["period_column"],
            output_dir=output / "discovery" / f"y{year}" / name, extended=extended)
        print(json.dumps({"year": year, "policy": name, "fit_rows": len(inner),
            "status": result["status"], "reason": result["reason"], "config": result["selected_config"]}), flush=True)


def diagnose(output, year, discovery_file):
    from experiment_rescue_regression_diagnostics import segment_rows
    output = Path(output).resolve()
    frozen = plan(output)
    spec, fitting, held, answers = case_frames(output, year)
    predictions = pd.read_csv(output / "native" / f"y{year}" / "submission.csv", **CSV)
    spec.validate_frame(predictions, held[spec.id_column].tolist())
    config = read(discovery_file).get("selected_config")
    rows = segment_rows(fitting.loc[:, list(spec.features)], pd.to_numeric(fitting[spec.target_column]),
        held.loc[:, list(spec.features)], pd.to_numeric(answers[spec.target_column]),
        {"native_main": pd.to_numeric(predictions[spec.target_column])},
        period_column=frozen["period_column"], size_column=frozen["size_column"],
        categorical=list(spec.categorical), reference_history_config=config)
    write(output / "diagnostics" / f"y{year}_native.json", {"rows": rows,
        "reference_config": config, "discovery_sha256": digest(discovery_file),
        "note": "Structural history masks shared by arms; not a claim native history was enabled"})
    return rows


def screen(output, year, engine, budget, discovery_file):
    from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
    from agent.regression_candidate import candidate_source, make_regression_recipe
    output = Path(output).resolve()
    frozen = plan(output)
    before = source(ROOT)
    if before["dirty"]:
        raise ValueError("Seal candidate code in a clean commit before measuring")
    if year == 5 and read(output / "frozen_decision.json")["engine"] != engine:
        raise ValueError("Known compatibility allows only the frozen development engine")
    if not (output / "diagnostics" / f"y{year}_native.json").is_file():
        raise ValueError("Measure and diagnose the incumbent before screening")
    spec, fitting, held, _ = case_frames(output, year)
    directory = output / "screen" / f"y{year}" / engine
    directory.mkdir(parents=True, exist_ok=False)
    source_path = directory / "source.py"
    source_path.write_text(candidate_source(), encoding="utf-8")
    recipe = make_regression_recipe(frozen["period_column"], engine=engine,
        history_config=read(discovery_file).get("selected_config"))
    candidate = freeze_candidate(source_path, recipe, directory / "frozen")
    record = {"source": before, "year": year, "engine": engine, "recipe": recipe,
        "code_sha256": candidate.code_hash, "recipe_sha256": candidate.recipe_hash,
        "discovery_sha256": digest(discovery_file), "budget_seconds": budget,
        "comparison": "Isolated screen; native equal-scheduler candidate on/off pending integration"}
    try:
        result = run_candidate(candidate, spec=spec, fit_frame=fitting, predict_frame=held,
            output_dir=directory, phase="full_refit", timeout=budget, max_memory_bytes=32 * 1024**3)
        result.verify()
        result.copy_output(directory / "submission.csv")
        evidence = read(result.output_path.parent / "regression_evidence.json")
        ablation = result.output_path.parent / "regression_ablation.csv"
        if digest(ablation) != evidence["ablation_sha256"]:
            raise ValueError("Ablation output changed after execution")
        alternatives = pd.read_csv(ablation, **CSV)
        if list(alternatives) != ["base", "mean", "calibrated"] or len(alternatives) != len(held):
            raise ValueError("Ablation output schema differs from the frozen contract")
        scores = {}
        for name in alternatives:
            frame = spec.prediction_frame(held[spec.id_column].tolist(), pd.to_numeric(alternatives[name]).to_numpy())
            target = directory / (name + ".csv")
            frame.to_csv(target, index=False)
            scores[name] = score_csv(spec.task_path, output / "host_only" / f"y{year}.csv", target)
        # The public result must be exactly the calibrated branch written by
        # this same invocation, not a prediction from an earlier model.
        public = pd.read_csv(directory / "submission.csv", **CSV)
        if not np.array_equal(pd.to_numeric(public[spec.target_column]), pd.to_numeric(alternatives["calibrated"])):
            raise ValueError("Published and ablation predictions differ")
        record.update(status="completed", execution=result.execution, evidence=evidence, metrics=scores,
            receipt=str(result.receipt_path), receipt_sha256=result.receipt_hash,
            submission_sha256=result.output_hash, prediction_rows=len(held), full_fit_rows=len(fitting))
    except CandidateExecutionError as error:
        record.update(status="failed", error=str(error), execution=error.execution)
    after = source(ROOT)
    if before != after:
        raise ValueError("Candidate source changed during measurement")
    write(directory / "summary.json", record)
    print(json.dumps({"year": year, "engine": engine, "status": record["status"],
                     "rmse": {k: v["rmse"] for k, v in record.get("metrics", {}).items()}}), flush=True)
    return record


def freeze_decision(output, engine):
    output = Path(output).resolve()
    plan(output)
    identity = source(ROOT)
    if identity["dirty"]:
        raise ValueError("Freeze a clean measured source")
    records = {}
    for year in (2, 3, 4):
        for family in ("catboost", "lightgbm"):
            path = output / "screen" / f"y{year}" / family / "summary.json"
            record = read(path)
            if record["source"] != identity:
                raise ValueError("Development results do not match this source")
            records[f"y{year}/{family}"] = {"sha256": digest(path), "status": record["status"],
                                                   "metrics": record.get("metrics")}
    write(output / "frozen_decision.json", {"source": identity, "engine": engine,
        "policy": "calibrated", "created_at": datetime.now(timezone.utc).isoformat(), "development": records,
        "status": "integration_pending", "note": "Frozen before known year5; native admission not established"})


def full_refit(output, budget):
    from agent.regression_candidate import candidate_source, make_regression_recipe
    from agent.research_candidate import freeze_candidate, run_candidate
    output = Path(output).resolve()
    frozen = plan(output)
    decision = read(output / "frozen_decision.json")
    before = source(ROOT)
    if before != decision["source"] or before["dirty"]:
        raise ValueError("Full refit requires the frozen measured source")
    spec = load_task(Path(frozen["source_task"]))
    for filename in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        if digest(spec.data_dir / filename) != frozen["source_files"][filename]:
            raise ValueError("Original full-refit input changed")
    fitting, held = pd.read_csv(spec.data_dir / "train.csv", **CSV), pd.read_csv(spec.data_dir / "test.csv", **CSV)
    # This config was discovered only on years1–3 before the known year5
    # candidate ran. Do not discover again on complete TRAIN.
    config = read(output / "discovery" / "y5" / "extended" / "discovery_report.json")["selected_config"]
    directory = output / "full_refit"
    directory.mkdir(exist_ok=False)
    path = directory / "source.py"
    path.write_text(candidate_source(), encoding="utf-8")
    recipe = make_regression_recipe(frozen["period_column"], engine=decision["engine"], history_config=config)
    candidate = freeze_candidate(path, recipe, directory / "frozen")
    result = run_candidate(candidate, spec=spec, fit_frame=fitting, predict_frame=held,
        output_dir=directory, phase="full_refit", timeout=budget, max_memory_bytes=32 * 1024**3)
    result.copy_output(directory / "submission.csv")
    frame = pd.read_csv(directory / "submission.csv", **CSV)
    spec.validate_frame(frame, held[spec.id_column].tolist())
    if not np.isfinite(pd.to_numeric(frame[spec.target_column])).all() or source(ROOT) != before:
        raise ValueError("Full-refit predictions/source invalid")
    evidence = read(result.output_path.parent / "regression_evidence.json")
    write(directory / "summary.json", {"source": before, "recipe": recipe, "budget_seconds": budget,
        "status": "completed", "execution": result.execution, "evidence": evidence,
        "fit_rows": len(fitting), "predict_rows": len(held), "ids_exact_order": True, "finite": True,
        "prediction_sha256": result.output_hash, "receipt_sha256": result.receipt_hash,
        "code_sha256": result.code_hash, "recipe_sha256": result.recipe_hash,
        "quality_metric": None, "quality_note": "Competition TEST has no available labels"})
    print(json.dumps({"full_refit": "completed", "fit_rows": len(fitting), "predict_rows": len(held)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period-column", required=True)
    p.add_argument("--size-column", required=True)
    p = sub.add_parser("native")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--years", type=int, nargs="+", choices=(2, 3, 4, 5), default=[4, 2, 3])
    p.add_argument("--budget", type=float, default=900)
    p = sub.add_parser("diagnose")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--year", type=int, choices=(2, 3, 4, 5), required=True)
    p.add_argument("--discovery", type=Path, required=True)
    p = sub.add_parser("discover")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--year", type=int, choices=(2, 3, 4, 5), required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine", choices=("catboost", "lightgbm"), required=True)
    p = sub.add_parser("full-refit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=float, default=360)
    p = sub.add_parser("screen")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--year", type=int, choices=(2, 3, 4, 5), required=True)
    p.add_argument("--engines", nargs="+", choices=("catboost", "lightgbm"), default=["catboost", "lightgbm"])
    p.add_argument("--budget", type=float, default=360)
    p.add_argument("--discovery", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.task, args.output, args.period_column, args.size_column)
    elif args.command == "native":
        for year in args.years:
            native(args.output, args.checkout, year, args.budget)
    elif args.command == "diagnose":
        diagnose(args.output, args.year, args.discovery)
    elif args.command == "screen":
        for engine in args.engines:
            screen(args.output, args.year, engine, args.budget, args.discovery)
    elif args.command == "discover":
        discover(args.output, args.year)
    elif args.command == "freeze":
        freeze_decision(args.output, args.engine)
    elif args.command == "full-refit":
        full_refit(args.output, args.budget)


if __name__ == "__main__":
    main()
