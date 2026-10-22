"""Five fixed classification challengers through the existing isolated worker.

Prepare seals inputs, sources, recipes and three grouped development splits.
Screen runs five recipes on seed 42; repeat runs only the two screen finalists
on seeds 43/44. Every invocation includes internal early stopping and fresh
refit. This is isolated evidence, not a claim about the native search route.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
INCUMBENT_SHA = "d05572846369f3b1e9cfa33f4008dda028db4fe0"
SEEDS = (42, 43, 44)
MINIMUM_GAIN = .005
CANDIDATES = ("catboost_none", "catboost_sqrt", "lightgbm_none", "xgboost_depth6", "xgboost_depth4")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    paths = sorted((ROOT / "agent").glob("*.py"))
    paths += [ROOT / name for name in ("offline_baseline.py", "requirements.platform.txt", __file__.split(os.sep)[-1])]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def environment():
    packages = {}
    for line in (ROOT / "requirements.platform.txt").read_text().splitlines():
        match = re.match(r"([\w.-]+)(?:\[[^]]+\])?==([^\s#]+)", line)
        if not match:
            continue
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = {"required": expected, "installed": actual, "matches": actual == expected}
    limits = {}
    for name in ("memory.max", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        limits[name] = path.read_text().strip() if path.exists() else None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "cgroup": limits, "packages": packages,
            "competition_resource_confirmed": False}


@contextmanager
def outer_seed(seed):
    # split_rows imports this value inside the function. Restore before workers
    # start so the candidate's internal split and model seeds remain 42.
    import offline_baseline
    previous = offline_baseline.SEED
    offline_baseline.SEED = seed
    try:
        yield
    finally:
        offline_baseline.SEED = previous


def grouped_splits(train, spec):
    import numpy as np
    import pandas as pd
    from agent.splits import split_rows
    groups = pd.util.hash_pandas_object(train.loc[:, list(spec.features)], index=False).to_numpy()
    results = []
    for seed in SEEDS:
        with outer_seed(seed):
            fitting, held, metadata = split_rows(train, train[spec.target_column], list(spec.features), True,
                                                 {"strategy": "grouped"})
        if len(np.intersect1d(groups[fitting], groups[held])):
            raise ValueError("Grouped development split overlaps")
        results.append({"seed": seed, "fit": fitting.tolist(), "held": held.tolist(), "metadata": metadata,
                        "fit_support": support(train.iloc[fitting][spec.target_column], spec.classes),
                        "held_support": support(train.iloc[held][spec.target_column], spec.classes)})
    return results


def support(labels, classes):
    counts = labels.value_counts()
    return {label: int(counts.get(label, 0)) for label in classes}


def diagnostics(train, spec):
    import numpy as np
    import pandas as pd
    features = train.loc[:, list(spec.features)]
    # Exact tuple factorization, independently of the host's hash grouping.
    group_codes, unique = pd.factorize(pd.MultiIndex.from_frame(features), sort=False)
    grouped = pd.DataFrame({"group": group_codes, "label": train[spec.target_column].to_numpy()})
    counts = grouped.groupby("group").size()
    unique_labels = grouped.groupby("group")["label"].nunique()
    conflicts = unique_labels[unique_labels > 1].index
    columns = {}
    for name in spec.features:
        values = features[name]
        numeric = pd.to_numeric(values, errors="coerce") if name in spec.numeric else None
        columns[name] = {"role": "numeric" if name in spec.numeric else "categorical",
                         "storage_dtype": str(values.dtype), "cardinality": int(values.nunique(dropna=False)),
                         "empty_cells": int(values.eq("").sum()), "null_cells": int(values.isna().sum()),
                         "numeric_nonfinite_or_missing": int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
                         if numeric is not None else None}
    duplicate_columns = [[a, b] for index, a in enumerate(spec.features)
                         for b in spec.features[index + 1:] if features[a].equals(features[b])]
    examples = [{"group": int(group), "rows": int(counts.loc[group]),
                 "labels": grouped.loc[grouped["group"].eq(group), "label"].value_counts().to_dict()}
                for group in conflicts[:20]]
    return {"rows": len(train), "features": len(spec.features), "class_support": support(train[spec.target_column], spec.classes),
            "columns": columns, "exact_duplicate_columns": duplicate_columns,
            "unique_feature_groups": len(unique), "duplicate_excess_rows": int(len(train) - len(unique)),
            "rows_in_duplicate_groups": int(counts[counts > 1].sum()),
            "conflicting_groups": len(conflicts), "rows_in_conflicting_groups": int(counts.loc[conflicts].sum()),
            "conflicting_rows_by_class": support(grouped.loc[grouped["group"].isin(conflicts), "label"], spec.classes),
            "conflicting_group_examples": examples}


def recipe_sources():
    from agent.boosting_candidate import candidate_source, make_boosting_recipe
    from agent.model_config import make_model_config
    from agent.xgboost_candidate import FIXED_CONFIGURATIONS, candidate_source as xgb_source, make_xgboost_recipe
    output = {}
    for name, action, parameters in (
        ("catboost_none", "catboost", {"depth": 6, "l2_leaf_reg": 3., "auto_class_weights": None}),
        ("catboost_sqrt", "catboost", {"depth": 6, "l2_leaf_reg": 3., "auto_class_weights": "SqrtBalanced"}),
        ("lightgbm_none", "lightgbm", {"num_leaves": 31, "min_child_samples": 20, "reg_lambda": 1., "class_weight": None}),
    ):
        base = {"action": action, "model_config": make_model_config(action, "classification", parameters)}
        output[name] = (candidate_source(), make_boosting_recipe(base, {"strategy": "grouped"}))
    for name, key in (("xgboost_depth6", "depth6_v1"), ("xgboost_depth4", "depth4_regularized_v1")):
        output[name] = (xgb_source(), make_xgboost_recipe("classification", "Macro F1", validation={"strategy": "grouped"},
                                                       parameters=FIXED_CONFIGURATIONS[key]))
    return output


def prepare(args):
    from agent.task_spec import load_task, load_task_data
    spec = load_task(args.task)
    if spec.metric.name != "Macro F1":
        raise ValueError("This experiment requires label classification with Macro F1")
    train, test, *_ = load_task_data(spec)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = recipe_sources()
    recipes = {}
    for name in CANDIDATES:
        source, recipe = sources[name]
        path = output / (name + ".py")
        path.write_text(source, encoding="utf-8")
        recipes[name] = {"source": path.name, "source_sha256": digest(path), "recipe": recipe}
    paths = [spec.task_path, *[spec.data_dir / name for name in
                             ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")]]
    plan = {"version": 1, "mode": "offline", "api_attempts": 0, "api_responses": 0,
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes": source_hashes(), "input_sha256": {str(path): digest(path) for path in paths},
            "task": str(spec.task_path), "environment": environment(), "classes": list(spec.classes),
            "seeds": list(SEEDS), "internal_seed": 42, "minimum_useful_gain": MINIMUM_GAIN,
            "maximum_slice_drop": .01, "maximum_class_f1_drop": .05,
            "invocation_seconds": args.timeout, "memory_limit_bytes": int(args.memory_gib * 1024 ** 3),
            "threads": 8, "screen_order": list(CANDIDATES), "maximum_screen_model_fits": 10,
            "repeat_policy": "top_two_by_seed42_macro_f1_then_fixed_screen_order_tiebreak",
            "control_replay_policy": "freeze_actual_native_seed42_published_code_recipe_then_replay_seeds43_44_at_300s_not_scheduler_comparison",
            "split_scope": "grouped_development_not_independent_test", "splits": grouped_splits(train, spec),
            "recipes": recipes, "diagnostics": diagnostics(train, spec), "test_rows": len(test)}
    write_new(output / "plan.json", plan)
    write_new(output / "plan.sha256.json", {"sha256": digest(output / "plan.json")})
    print(json.dumps({"prepared": str(output), "source_sha": plan["source_sha"],
                      "diagnostics": plan["diagnostics"], "splits": [x["metadata"] for x in plan["splits"]]}, ensure_ascii=False))


def load_plan(output, *, execution=True):
    from agent.task_spec import load_task, load_task_data
    output = output.resolve()
    if digest(output / "plan.json") != read_json(output / "plan.sha256.json")["sha256"]:
        raise ValueError("Sealed plan changed")
    plan = read_json(output / "plan.json")
    if execution and any(not (ROOT / path).is_file() or digest(ROOT / path) != expected
                         for path, expected in plan["source_hashes"].items()):
        raise ValueError("Experiment sources changed after preparation; prepare a new evidence directory")
    if execution and environment() != plan["environment"]:
        raise ValueError("Execution environment changed after preparation")
    for path, expected in plan["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Official input changed: " + path)
    for item in plan["recipes"].values():
        if digest(output / item["source"]) != item["source_sha256"]:
            raise ValueError("Frozen source changed")
    spec = load_task(Path(plan["task"]))
    train, test, *_ = load_task_data(spec)
    return plan, spec, train, test


def score_saved(raw, spec, expected_ids, actuals):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from agent.metrics import score_predictions
    frame = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
    spec.validate_frame(frame, list(expected_ids))
    labels = frame[spec.target_column].tolist()
    score = score_predictions(spec.metric, list(actuals), labels, classes=spec.classes)
    return {"macro_f1": score, "per_class": classification_report(list(actuals), labels,
             labels=list(spec.classes), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(list(actuals), labels, labels=list(spec.classes)).tolist(),
            "class_order": list(spec.classes)}


def invoke(output, plan, spec, train, test, name, *, split=None, control_plan_sha256=None):
    from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
    phase = "inner_selection" if split is not None else "full_refit"
    key = name + "_seed" + str(split["seed"]) if split is not None else name + "_full_refit"
    directory = output / "trials" / key
    directory.mkdir(parents=True, exist_ok=False)
    item = plan["recipes"][name]
    frozen = freeze_candidate(output / item["source"], item["recipe"], directory / "frozen")
    fit = train.iloc[split["fit"]].copy() if split is not None else train.copy()
    predict = train.iloc[split["held"]].drop(columns=[spec.target_column]).copy() if split is not None else test.copy()
    actuals = train.iloc[split["held"]][spec.target_column].tolist() if split is not None else None
    started = time.monotonic()
    record = {"candidate": name, "seed": split["seed"] if split else None, "phase": phase,
              "source_sha": plan["source_sha"], "plan_sha256": digest(output / "plan.json"),
              "input_sha256": plan["input_sha256"], "code_hash": frozen.code_hash, "recipe_hash": frozen.recipe_hash,
              "recipe": item["recipe"], "fit_rows": len(fit), "predict_rows": len(predict),
              "fit_support": support(fit[spec.target_column], spec.classes),
              "held_support": split["held_support"] if split else None,
              "api_attempts": 0, "api_responses": 0, "api_counter_basis": "structural_no_provider_path",
              "candidate_reached": False, "refit_status": "not_confirmed",
              "publication_status": "not_attempted", "native_route": "isolated_worker_only",
              "timeout_seconds": plan["invocation_seconds"], "memory_limit_bytes": plan["memory_limit_bytes"]}
    try:
        result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=predict, output_dir=directory / "worker",
                               phase=phase, timeout=plan["invocation_seconds"], actuals=actuals,
                               max_memory_bytes=plan["memory_limit_bytes"])
        record.update(status="completed", execution=result.execution, host_metric=result.metric,
                      output_path=str(result.output_path), output_sha256=result.output_hash,
                      receipt_path=str(result.receipt_path), receipt_sha256=result.receipt_hash)
        raw = result.read_output()
        if split is not None:
            metrics = score_saved(raw, spec, predict[spec.id_column].tolist(), actuals)
            if not math.isclose(metrics["macro_f1"], result.metric["value"], abs_tol=1e-12):
                raise ValueError("Independent saved-prediction rescore differs from host")
            record.update(metrics)
        else:
            destination = directory / "final_predictions.csv"
            result.copy_output(destination)
            record.update(publication_status="local_verified_csv_written", published_path=str(destination),
                          published_sha256=digest(destination))
    except CandidateExecutionError as error:
        record.update(status="failed", error=str(error), execution=error.execution)
    except Exception as error:
        record.update(status="error", error=type(error).__name__ + ": " + str(error))
    record["elapsed_seconds"] = time.monotonic() - started
    evidence = []
    for path in sorted((directory / "worker").glob("*/*_evidence.json")):
        item = {"path": str(path), "sha256": digest(path)}
        try:
            item["content"] = read_json(path)
        except (ValueError, OSError) as error:
            item["parse_error"] = str(error)
        evidence.append(item)
    record["candidate_evidence"] = evidence
    record["candidate_reached"] = bool(evidence) or record["status"] == "completed"
    record["candidate_reached_basis"] = ("diagnostic_receipt_present" if evidence else
                                           "host_verified_receipt" if record["status"] == "completed" else "not_confirmed")
    record["candidate_trained"] = any(x.get("content", {}).get("status") == "completed" for x in evidence)
    if record["status"] == "completed" and any(x.get("content", {}).get("fresh_full_refit") is True for x in evidence):
        record["refit_status"] = "confirmed_fresh_full_fit"
    resources = record.get("execution", {}).get("resources", {})
    record["peak_rss_bytes"] = resources.get("peak_rss_bytes")
    record["resource_monitor_available"] = resources.get("available", False)
    record["timed_out"] = record.get("execution", {}).get("timed_out", False)
    record["memory_limit_exceeded"] = record.get("execution", {}).get("memory_limit_exceeded", False)
    if control_plan_sha256 is not None:
        record.update(control_plan_sha256=control_plan_sha256,
                      native_route="frozen_incumbent_recipe_control_replay_not_native_scheduler")
        if record["status"] == "completed":
            record.update(candidate_trained=True, refit_status="native_worker_full_fit_verified")
    write_new(directory / "record.json", record)
    print(json.dumps({key: record.get(key) for key in
                      ("candidate", "seed", "status", "macro_f1", "elapsed_seconds", "peak_rss_bytes", "candidate_reached", "refit_status", "error")}), flush=True)
    return record


def records(output):
    return [read_json(path) for path in sorted((output / "trials").glob("*/record.json"))]


def finalists(rows):
    primary = {row["candidate"]: row for row in rows if row["candidate"] in CANDIDATES
               and row["phase"] == "inner_selection" and row["seed"] == SEEDS[0]}
    if set(primary) != set(CANDIDATES):
        raise ValueError("All five planned screen outcomes must be recorded before selecting finalists")
    completed = [row for row in primary.values() if row["status"] == "completed" and row.get("macro_f1") is not None]
    completed.sort(key=lambda row: (-row["macro_f1"], CANDIDATES.index(row["candidate"])))
    return [row["candidate"] for row in completed[:2]]


def run(args):
    output = args.output.resolve()
    plan, spec, train, test = load_plan(output)
    if args.command == "screen":
        planned = [(name, plan["splits"][0]) for name in CANDIDATES]
    else:
        selected = finalists(records(output))
        if not selected:
            raise ValueError("No successful screen finalist")
        if args.command == "full-refit":
            if args.candidate not in selected:
                raise ValueError("Full refit requires --candidate naming one of the two measured screen finalists")
            planned = [(args.candidate, None)]
        else:
            planned = [(name, split) for split in plan["splits"][1:] for name in selected]
    for name, split in planned:
        key = name + "_seed" + str(split["seed"]) if split else name + "_full_refit"
        destination = output / "trials" / key
        if (destination / "record.json").exists():
            print(json.dumps({"preserved_existing_outcome": key}), flush=True)
            continue
        if destination.exists():
            raise ValueError("Interrupted invocation exists without a completed record: " + key)
        invoke(output, plan, spec, train, test, name, split=split)
    export_table(output)


def export_table(output):
    # Derived table can be rebuilt; immutable per-invocation records are the evidence.
    rows = records(output)
    path = output / "experiments.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def check_incumbent_runtime(native_plan, experiment_plan):
    """Separate the measured incumbent commit from the experiment's own HEAD.

    Committing this harness advances HEAD without changing candidate runtime.
    Both sealed runtime hashes and the current files must still match main.
    """
    if native_plan["source"]["sha"] != INCUMBENT_SHA:
        raise ValueError("Native control is not the preregistered incumbent SHA")
    for relative, expected in native_plan["source"]["files"].items():
        if relative.startswith("agent/") or relative == "offline_baseline.py":
            if (experiment_plan["source_hashes"].get(relative) != expected
                    or digest(ROOT / relative) != expected):
                raise ValueError("Control runtime differs from incumbent source: " + relative)


def compare_incumbent(args):
    """Replay a frozen published control, never reselect or retune the main agent."""
    import pandas as pd
    from agent.task_spec import TaskSpec
    output = args.output.resolve()
    plan, spec, train, test = load_plan(output)
    if args.incumbent_output is None:
        raise ValueError("compare-incumbent requires --incumbent-output")
    native_output = args.incumbent_output.resolve()
    native_plan = read_json(native_output / "plan.json")
    if digest(native_output / "plan.json") != read_json(native_output / "seal.json")["plan_sha256"]:
        raise ValueError("Native incumbent plan changed")
    native_report = read_json(native_output / "report.json")
    if (native_report.get("status") != "completed" or not native_report.get("api_zero_verified")
            or not native_report.get("refit_output_hash_matches")):
        raise ValueError("A verified new publication by the exact incumbent SHA is required")
    check_incumbent_runtime(native_plan, plan)
    native_split = read_json(native_output / "host_only/split.json")
    primary = plan["splits"][0]
    if native_split["fitting"] != primary["fit"] or native_split["held"] != primary["held"]:
        raise ValueError("Native seed42 partition differs from the planned development partition")
    receipt_path = Path(native_report["candidate_receipts"]["refit"]).resolve()
    if native_output / "task" not in receipt_path.parents:
        raise ValueError("Native refit receipt escaped its task")
    native_receipt = read_json(receipt_path)
    invocation = receipt_path.parent
    source_path, recipe_path = invocation / "candidate.py", invocation / "recipe.json"
    if (digest(source_path) != native_receipt["code_hash"] or digest(recipe_path) != native_receipt["recipe_hash"]
            or digest(native_output / "task/submission.csv") != native_receipt["output_hash"]
            or digest(native_output / native_report["native_report"]) != native_report["native_report_sha256"]):
        raise ValueError("Published incumbent artifact or native report differs from receipt")
    expected_fit = train.iloc[primary["fit"]].loc[:, [spec.id_column, *spec.features, spec.target_column]]
    expected_predict = train.iloc[primary["held"]].loc[:, [spec.id_column, *spec.features]]
    actual_fit = pd.read_csv(invocation / "fit.csv", dtype=str, keep_default_na=False)
    actual_predict = pd.read_csv(invocation / "predict.csv", dtype=str, keep_default_na=False)
    if (not actual_fit.equals(expected_fit.reset_index(drop=True))
            or not actual_predict.equals(expected_predict.reset_index(drop=True))):
        raise ValueError("Published incumbent was not refitted on the same fit/predict rows")
    control = output / "control"
    control_plan_path = control / "plan.json"
    if not control_plan_path.exists():
        control.mkdir(exist_ok=False)
        shutil.copyfile(source_path, control / "incumbent.py")
        shutil.copyfile(recipe_path, control / "incumbent_recipe.json")
        control_plan = {"version": 1, "scope": "frozen_incumbent_recipe_control_replay_not_native_scheduler",
                        "screen_plan_sha256": digest(output / "plan.json"), "native_output": str(native_output),
                        "native_plan_sha256": digest(native_output / "plan.json"),
                        "native_report_sha256": digest(native_output / "report.json"),
                        "native_receipt_path": str(receipt_path), "native_receipt_sha256": digest(receipt_path),
                        "source_sha256": digest(source_path), "recipe_sha256": digest(recipe_path),
                        "native_submission_sha256": digest(native_output / "task/submission.csv"),
                        "seeds": list(SEEDS), "replay_timeout_seconds": plan["invocation_seconds"],
                        "selection_budget_seconds_seed42": native_plan["time_budget_seconds"],
                        "replay_selection": "none; exact frozen recipe", "seed42_retrained": False}
        write_new(control_plan_path, control_plan)
        write_new(control / "seal.json", {"sha256": digest(control_plan_path)})
    control_plan = read_json(control_plan_path)
    if (digest(control_plan_path) != read_json(control / "seal.json")["sha256"]
            or control_plan["native_report_sha256"] != digest(native_output / "report.json")
            or control_plan["source_sha256"] != digest(control / "incumbent.py")
            or control_plan["recipe_sha256"] != digest(control / "incumbent_recipe.json")):
        raise ValueError("Frozen incumbent control changed")
    name = "incumbent_control"
    native_record_dir = output / "trials" / (name + "_seed42")
    if not (native_record_dir / "record.json").exists():
        raw = (native_output / "task/submission.csv").read_bytes()
        metrics = score_saved(raw, spec, expected_predict[spec.id_column].tolist(),
                              train.iloc[primary["held"]][spec.target_column].tolist())
        context = read_json(invocation / "context.json")
        original_spec = TaskSpec.from_dict(context["task_spec"])
        if original_spec.classes != spec.classes:
            raise ValueError("Native incumbent class order differs")
        native_record = {"candidate": name, "seed": 42, "phase": "inner_selection", "status": "completed",
                         "source_sha": plan["source_sha"], "plan_sha256": digest(output / "plan.json"),
                         "input_sha256": plan["input_sha256"], "code_hash": native_receipt["code_hash"],
                         "recipe_hash": native_receipt["recipe_hash"], "recipe": read_json(recipe_path),
                         "fit_rows": len(expected_fit), "predict_rows": len(expected_predict),
                         "fit_support": primary["fit_support"], "held_support": primary["held_support"],
                         "api_attempts": 0, "api_responses": 0, "api_counter_basis": "native_report",
                         "candidate_reached": True, "candidate_reached_basis": "native_refit_receipt",
                         "candidate_trained": True, "refit_status": "native_full_refit_verified",
                         "publication_status": "native_verified_csv", "candidate_evidence": [],
                         "native_route": "whole_scheduler_seed42_only; frozen_recipe_control_on_other_seeds",
                         "control_plan_sha256": digest(control_plan_path),
                         "elapsed_seconds": native_report["execution"]["elapsed_seconds"],
                         "execution": native_report["execution"],
                         "output_path": str(native_output / "task/submission.csv"), "output_sha256": native_receipt["output_hash"],
                         "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path),
                         "peak_rss_bytes": native_report["execution"]["resources"].get("peak_rss_bytes"),
                         **metrics}
        write_new(native_record_dir / "record.json", native_record)
    plan["recipes"][name] = {"source": "control/incumbent.py", "recipe": read_json(control / "incumbent_recipe.json")}
    for split in plan["splits"][1:]:
        record_path = output / "trials" / (name + "_seed" + str(split["seed"])) / "record.json"
        if record_path.exists():
            continue
        invoke(output, plan, spec, train, test, name, split=split,
               control_plan_sha256=digest(control_plan_path))
    export_table(output)


def audit(args):
    import pandas as pd
    output = args.output.resolve()
    plan, spec, train, test = load_plan(output, execution=False)
    verified = 0
    for row in records(output):
        if row["plan_sha256"] != digest(output / "plan.json"):
            raise ValueError("Recorded outcome belongs to a different plan")
        if "control_plan_sha256" in row:
            control = output / "control"
            control_plan = read_json(control / "plan.json")
            if (digest(control / "plan.json") != row["control_plan_sha256"]
                    or digest(control / "plan.json") != read_json(control / "seal.json")["sha256"]
                    or digest(control / "incumbent.py") != control_plan["source_sha256"]
                    or digest(control / "incumbent_recipe.json") != control_plan["recipe_sha256"]
                    or digest(control_plan["native_receipt_path"]) != control_plan["native_receipt_sha256"]
                    or digest(Path(control_plan["native_output"]) / "report.json") != control_plan["native_report_sha256"]):
                raise ValueError("Frozen incumbent provenance changed")
        for evidence in row.get("candidate_evidence", []):
            if digest(evidence["path"]) != evidence["sha256"]:
                raise ValueError("Saved candidate diagnostic evidence changed")
            if "content" in evidence and read_json(evidence["path"]) != evidence["content"]:
                raise ValueError("Recorded diagnostic evidence differs from saved receipt")
        if row["status"] != "completed":
            continue
        path = Path(row["output_path"])
        if digest(path) != row["output_sha256"] or digest(row["receipt_path"]) != row["receipt_sha256"]:
            raise ValueError("Saved native output or receipt changed")
        if row["phase"] == "inner_selection":
            split = next(x for x in plan["splits"] if x["seed"] == row["seed"])
            held = train.iloc[split["held"]]
            rescored = score_saved(path.read_bytes(), spec, held[spec.id_column].tolist(), held[spec.target_column].tolist())
            if any(rescored[name] != row[name] for name in rescored):
                raise ValueError("Saved score no longer reproduces")
        else:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            spec.validate_frame(frame, test[spec.id_column].tolist())
            if (digest(row["published_path"]) != row["published_sha256"]
                    or row["published_sha256"] != row["output_sha256"]):
                raise ValueError("Published full-refit predictions changed")
        evidence = [item["content"] for item in row["candidate_evidence"] if "content" in item]
        if row["refit_status"] == "confirmed_fresh_full_fit" and not any(
                item.get("fresh_full_refit") is True and item.get("full_fit_rows") == row["fit_rows"]
                and item.get("predict_rows") == row["predict_rows"] and item.get("classes") == list(spec.classes)
                for item in evidence):
            raise ValueError("Full-fit receipt does not substantiate the recorded refit")
        verified += 1
    export_table(output)
    print(json.dumps({"audited_completed_invocations": verified, "recorded_outcomes": len(records(output)),
                      "scope": "isolated_worker_evidence"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "repeat", "full-refit", "audit", "compare-incumbent"))
    parser.add_argument("--mode", choices=("offline",), default="offline", help="No API provider is initialized")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory for prepare; same directory afterwards")
    parser.add_argument("--task", type=Path, default=ROOT / "data/classification/task.md")
    parser.add_argument("--timeout", type=float, default=300, help="Per invocation seconds, frozen by prepare")
    parser.add_argument("--memory-gib", type=float, default=6, help="Supervised per-process-tree limit, frozen by prepare")
    parser.add_argument("--candidate", choices=CANDIDATES, help="Required for full-refit")
    parser.add_argument("--incumbent-output", type=Path, help="Verified native incumbent evidence directory for compare-incumbent")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 10 <= args.timeout <= 3600:
        parser.error("--timeout must be between 10 and 3600 seconds")
    if not math.isfinite(args.memory_gib) or not 1 <= args.memory_gib <= 28:
        parser.error("--memory-gib must be between 1 and 28")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "compare-incumbent":
        compare_incumbent(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
