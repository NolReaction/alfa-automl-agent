"""Estimate contest points on public TRAIN holdouts; never imported by solution.py.

The contest converts a metric to points relative to the organizers' baseline B
on the hidden test:

    regression:     S = 100 * max(0, 1 - RMSE / B)
    classification: S = 100 * max(0, (F1 - B) / (1 - B))

Here B and the metric are both measured on one local holdout cut from public
TRAIN, so the points are an estimate, not a platform score. prepare writes the
derived tasks and keeps holdout answers in host_only/; run and stage give each
arm a fresh copy; score reads only the saved submission.csv files. Competition
TEST answers are never available and never used. Raw files stay below the
ignored data/experiments/score_meter/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from agent.metrics import get_metric, score_predictions
from agent.splits import split_rows
from agent.task_spec import load_task


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "experiments" / "score_meter"
CASES = {
    "regression_last_period": {"source": "regression", "metric": "RMSE", "split": "last_period",
                               "time_column": "Year_Factor"},
    "classification_grouped": {"source": "classification", "metric": "Macro F1", "split": "grouped"},
}
CSV = {"dtype": str, "keep_default_na": False}


def require(condition, message):
    """Active under -O; evidence failures are controlled ValueErrors."""
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Evidence must be a regular file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value) + b"\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def points(metric, value, baseline):
    """Contest formula for one task; at or below the baseline the task gets zero."""
    name = get_metric(metric).name
    for number in (value, baseline):
        require(type(number) in {int, float} and math.isfinite(number), "Metric values must be finite")
    if name == "RMSE":
        require(value >= 0 and baseline > 0, "RMSE and its baseline must be nonnegative, B > 0")
        return 100 * max(0.0, 1 - value / baseline)
    if name == "Macro F1":
        require(0 <= value <= 1 and 0 <= baseline < 1, "Macro F1 and its baseline must be in [0, 1), B < 1")
        return 100 * max(0.0, (value - baseline) / (1 - baseline))
    raise ValueError("The known contest formula covers only RMSE and Macro F1")


def _output(path):
    result = Path(path).resolve()
    require(RAW_ROOT.resolve() in result.parents, "Output must be a child of data/experiments/score_meter/")
    return result


def _holdout(train, spec, case):
    if case["split"] == "last_period":
        column = case["time_column"]
        require(column in spec.features, "The period column must be a declared feature")
        periods = pd.to_numeric(train[column], errors="raise")
        latest = periods.max()
        held = (periods == latest).to_numpy()
        require(held.any() and (~held).any(), "Need at least two periods")
        return held, {"strategy": "last_period", "time_column": column,
                      "fit_period_max": str(periods[~held].max()), "holdout_period": str(latest)}
    labels = train[spec.target_column]
    fitting, held_rows, meta = split_rows(train, labels, list(spec.features), True, {"strategy": "grouped"})
    held = pd.Series(False, index=train.index)
    held.iloc[held_rows] = True
    require(len(fitting) + len(held_rows) == len(train), "Grouped split lost rows")
    return held.to_numpy(), {"strategy": "grouped", **{key: meta[key] for key in (
        "seed", "n_splits", "fold", "group_count", "train_groups", "validation_groups", "group_overlap")}}


def prepare(output_dir, sources):
    """Write derived tasks, host-only answers and hashes; performs zero fits."""
    output = _output(output_dir)
    require(not output.exists(), "prepare needs a fresh output directory")
    require(set(sources) == {case["source"] for case in CASES.values()}, "Need regression and classification task.md")
    output.mkdir(parents=True)
    entries = {}
    for name, case in CASES.items():
        task = Path(sources[case["source"]]).resolve()
        spec = load_task(task)
        require(spec.metric.name == case["metric"], name + " source task has another metric")
        relative = spec.data_dir.relative_to(task.parent)
        train = pd.read_csv(spec.data_dir / "train.csv", **CSV)
        held, split = _holdout(train, spec, case)
        fit, holdout = train.loc[~held].reset_index(drop=True), train.loc[held].reset_index(drop=True)
        inputs = output / "inputs" / name
        data = inputs / relative
        data.mkdir(parents=True)
        shutil.copyfile(task, inputs / "task.md")
        shutil.copyfile(spec.data_dir / "data_dictionary.csv", data / "data_dictionary.csv")
        fit.to_csv(data / "train.csv", index=False)
        holdout.drop(columns=[spec.target_column]).to_csv(data / "test.csv", index=False)
        sample = pd.DataFrame({spec.id_column: holdout[spec.id_column],
                               spec.target_column: "0.0" if case["metric"] == "RMSE" else spec.classes[0]})
        sample.to_csv(data / "sample_submission.csv", index=False)
        answers = output / "host_only" / name / "answers.csv"
        answers.parent.mkdir(parents=True)
        holdout.loc[:, [spec.id_column, spec.target_column]].to_csv(answers, index=False)
        derived = load_task(inputs / "task.md")
        require((derived.metric, derived.features, derived.id_column, derived.target_column)
                == (spec.metric, spec.features, spec.id_column, spec.target_column),
                name + " derived task schema differs from its source")
        entries[name] = {"case": case, "split": split, "rows": {"fit": len(fit), "holdout": len(holdout)},
                         "source_task_sha256": file_hash(task),
                         "source_train_sha256": file_hash(spec.data_dir / "train.csv"),
                         "data_dir": relative.as_posix(),
                         "files": {path.relative_to(output).as_posix(): file_hash(path)
                                   for path in sorted(inputs.rglob("*")) + [answers] if path.is_file()}}
    plan = {"version": 1, "created_at": utc_now(), "cases": entries,
            "note": "Local public TRAIN holdouts; task.md row counts describe the original files"}
    write_json_new(output / "plan.json", plan)
    return plan


def load_plan(output):
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    for entry in plan["cases"].values():
        for path, digest in entry["files"].items():
            require(file_hash(output / path) == digest, "Prepared file changed: " + path)
    return plan


def stage(output_dir, arm):
    """Give one arm fresh copies of the sealed inputs; returns task.md paths."""
    output = _output(output_dir)
    plan = load_plan(output)
    require(arm and arm.replace("_", "").replace("-", "").isalnum(), "Arm names use letters, digits, _ and -")
    tasks = {}
    for name in plan["cases"]:
        target = output / "arms" / arm / name
        require(not target.exists(), "Arm already staged: " + arm + "/" + name)
        shutil.copytree(output / "inputs" / name, target)
        tasks[name] = target / "task.md"
    return tasks


def run(output_dir, arm, time_budget, extra=(), cases=None, work_root=None):
    """Run this checkout's solution.py offline per case, one after another; no API.

    The agent nests its run directories deeply. Windows without long paths
    (MAX_PATH 260) then fails every trial, so each case runs in a fresh short
    directory; only submission.csv and run.log are copied back to the arm.
    """
    require(type(time_budget) in {int, float} and 30 <= time_budget <= 10500, "time budget is 30..10500 s")
    tasks = stage(output_dir, arm)
    cases = list(tasks) if cases is None else list(cases)
    require(set(cases) <= set(tasks), "Unknown case")
    records = {}
    for name in cases:
        staged = tasks[name].parent
        work = Path(tempfile.mkdtemp(prefix="sm", dir=work_root)) / "c"
        shutil.copytree(staged, work)
        command = [sys.executable, str(ROOT / "solution.py"), str(work / "task.md"), "--mode", "offline",
                   "--time-budget", str(time_budget), *extra]
        started = time.monotonic()
        with (work / "run.log").open("wb") as log:
            try:
                completed = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, timeout=time_budget + 900, check=False)
                status = {"exit_code": completed.returncode, "timed_out": False}
            except subprocess.TimeoutExpired:
                status = {"exit_code": None, "timed_out": True}
        for file in ("submission.csv", "run.log"):
            if (work / file).is_file():
                shutil.copyfile(work / file, staged / file)
        records[name] = {**status, "seconds": time.monotonic() - started, "work_dir": str(work),
                         "arguments": command[3:], "head": _git_head()}
        write_json_new(staged / "run.json", records[name])
    return records


def _git_head():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def score_submission(output, name, submission):
    """Validate one saved submission against the derived task and host answers."""
    spec = load_task(output / "inputs" / name / "task.md")
    answers = pd.read_csv(output / "host_only" / name / "answers.csv", **CSV)
    frame = pd.read_csv(submission, **CSV)
    spec.validate_frame(frame, answers[spec.id_column].tolist())
    actual = answers[spec.target_column].tolist()
    predicted = frame[spec.target_column].to_numpy()
    if not spec.classification:
        actual = pd.to_numeric(pd.Series(actual)).tolist()
    return score_predictions(spec.metric, actual, predicted, classes=spec.classes)


def score(output_dir, baseline_arm=None, assumed=None):
    """Score every staged arm; points use the baseline arm's metric or assumed B values."""
    output = _output(output_dir)
    plan = load_plan(output)
    assumed = dict(assumed or {})
    require(set(assumed) <= set(plan["cases"]), "Unknown case in assumed baselines")
    metrics = {}
    for arm_dir in sorted((output / "arms").iterdir()) if (output / "arms").is_dir() else []:
        for name in plan["cases"]:
            submission = arm_dir / name / "submission.csv"
            state = {"submission_sha256": file_hash(submission)} if submission.is_file() else {"status": "missing"}
            if submission.is_file():
                try:
                    state.update(status="scored", metric=score_submission(output, name, submission))
                except ValueError as error:
                    state.update(status="invalid", error=str(error))
            metrics.setdefault(arm_dir.name, {})[name] = state
    baselines = {}
    for name, entry in plan["cases"].items():
        if name in assumed:
            baselines[name] = {"source": "assumed", "value": float(assumed[name])}
        elif baseline_arm is not None:
            state = metrics.get(baseline_arm, {}).get(name, {})
            require(state.get("status") == "scored", "Baseline arm has no valid " + name + " submission")
            baselines[name] = {"source": "arm:" + baseline_arm, "value": state["metric"]}
    for arm, cases in metrics.items():
        for name, state in cases.items():
            if state.get("status") == "scored" and name in baselines:
                state["points"] = points(plan["cases"][name]["case"]["metric"], state["metric"],
                                         baselines[name]["value"])
        values = [state.get("points") for state in cases.values()]
        cases["public_mean_points"] = (sum(values) / len(values) if values and None not in values
                                       and len(values) == len(plan["cases"]) else None)
    result = {"version": 1, "scored_at": utc_now(), "baselines": baselines, "arms": metrics,
              "estimate_not_platform_score": True}
    (output / "scores.json").write_bytes(json_bytes(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "stage", "run", "score"):
        commands.add_parser(name).add_argument("--output", required=True, type=Path,
                                               help="Child directory of data/experiments/score_meter/")
    commands.choices["prepare"].add_argument("--regression-task", required=True, type=Path)
    commands.choices["prepare"].add_argument("--classification-task", required=True, type=Path)
    for name in ("stage", "run"):
        commands.choices[name].add_argument("--arm", required=True)
    commands.choices["run"].add_argument("--time-budget", required=True, type=float)
    commands.choices["run"].add_argument("--case", action="append", choices=list(CASES))
    commands.choices["run"].add_argument("--extra", action="append", default=[],
                                         help="One extra solution.py argument; repeat it")
    commands.choices["run"].add_argument("--work-root", type=Path,
                                         help="Short directory for agent runs; default: system temp")
    commands.choices["score"].add_argument("--baseline-arm")
    commands.choices["score"].add_argument("--assume", action="append", default=[], metavar="CASE=B",
                                           help="Assumed baseline metric for one case")
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        plan = prepare(arguments.output, {"regression": arguments.regression_task,
                                          "classification": arguments.classification_task})
        value = {name: {"rows": entry["rows"], "split": entry["split"]} for name, entry in plan["cases"].items()}
    elif arguments.command == "stage":
        value = {name: str(path) for name, path in stage(arguments.output, arguments.arm).items()}
    elif arguments.command == "run":
        value = run(arguments.output, arguments.arm, arguments.time_budget, arguments.extra, arguments.case,
                    arguments.work_root)
    else:
        assumed = {}
        for item in arguments.assume:
            case, separator, number = item.partition("=")
            require(separator and case in CASES, "--assume uses CASE=B")
            assumed[case] = float(number)
        value = score(arguments.output, arguments.baseline_arm, assumed)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
