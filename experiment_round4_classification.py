"""Sealed Round 4 fixed-recipe experiments; offline isolated v3 workers only.

Prepare before training. Screen has one reference, one identity contract probe,
and five hypotheses; confirmation freezes one finalist on seeds 43/44/137.
This is a fixed-recipe comparison against current main, not a scheduler replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from experiment_rescue_classification import (
    diagnostics, digest, environment, outer_seed, read_json, score_saved, support, write_new,
)

ROOT = Path(__file__).resolve().parent
BASE_SHA = "e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb"
SEEDS = (42, 43, 44, 137)
HYPOTHESES = ("oof", "frequency", "asymmetry", "group_weight", "extra_trees_asymmetry")
SCREEN = ("native_fixed", "identity", *HYPOTHESES)
PLANNED_FITS = dict(zip(SCREEN, (2, 2, 10, 2, 2, 2, 1)))
CAPS = {"screen": 24, "confirm": 36, "full-refit": 10}
ASYMMETRY_PAIRS = [
    {"forward": "sbytes", "reverse": "dbytes", "quantity": "data_volume", "unit": "bytes"},
    {"forward": "spkts", "reverse": "dpkts", "quantity": "packet_count", "unit": "packets"},
    {"forward": "sload", "reverse": "dload", "quantity": "bit_rate", "unit": "bit/s"},
]
# Exact plain recipe previously published by native main, copied before this
# experiment. It is retrained here using e62 runtime and receives no old scores.
BASE_RECIPE = {"components": [
    {"action": "lightgbm", "feature_config": {"date_columns": [], "missing_indicators": False,
        "rare_min_count": 0, "signed_log": False, "text_columns": []},
     "model_config": {"model": "LGBMClassifier", "parameters": {"class_weight": "balanced",
        "colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": -1, "min_child_samples": 20,
        "n_estimators": 400, "num_leaves": 15, "reg_lambda": 3.0}}},
    {"action": "baseline", "feature_config": {"date_columns": [], "missing_indicators": False,
        "rare_min_count": 0, "signed_log": False, "text_columns": []},
     "model_config": {"model": "RandomForestClassifier", "parameters": {
        "class_weight": "balanced_subsample", "max_depth": 18, "max_features": "sqrt",
        "min_samples_leaf": 2, "n_estimators": 120}}},
]}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = sorted((ROOT / "agent").glob("*.py")) + [ROOT / name for name in (
        "offline_baseline.py", "requirements.platform.txt", "experiment_rescue_classification.py",
        "experiment_round4_classification.py", "docs/round4_classification_plan.md")]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def grouped_splits(train, spec):
    import numpy as np
    import pandas as pd
    from agent.splits import split_rows
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(train.loc[:, list(spec.features)]), sort=False)
    result = []
    for seed in SEEDS:
        with outer_seed(seed):
            fit, held, metadata = split_rows(train, train[spec.target_column], list(spec.features), True,
                                             {"strategy": "grouped"})
        if len(np.intersect1d(groups[fit], groups[held])):
            raise ValueError("Feature groups overlap across development boundary")
        partition = {"fit": fit.tolist(), "held": held.tolist()}
        result.append({"seed": seed, **partition, "sha256": json_hash(partition), "metadata": metadata,
                       "fit_support": support(train.iloc[fit][spec.target_column], spec.classes),
                       "held_support": support(train.iloc[held][spec.target_column], spec.classes)})
    return result


def validate_coverage(value, fitting, spec):
    """Bind a cached no-model preflight to these raw rows and labels."""
    import numpy as np
    import pandas as pd
    unsealed = {key: item for key, item in value.items() if key != "split_sha256"}
    groups = pd.util.hash_pandas_object(fitting.loc[:, list(spec.features)], index=False).to_numpy()
    if value["split_sha256"] != json_hash(unsealed) or value["groups"] != [int(group) for group in groups]:
        raise ValueError("Cached coverage differs from raw feature groups")
    labels = fitting[spec.target_column].to_numpy()
    def checked_support(rows):
        if any(type(i) is not int or not 0 <= i < len(fitting) for i in rows) or len(set(rows)) != len(rows):
            raise ValueError("Invalid coverage row positions")
        rows = np.asarray(rows, dtype=int)
        return {label: {"rows": int(np.count_nonzero(labels[rows] == label)),
                        "groups": len(np.unique(groups[rows][labels[rows] == label]))} for label in spec.classes}
    expected = {"allowed": checked_support(list(range(len(fitting))))}
    if value["available"]:
        A, B = value["tuning_rows"], value["selection_rows"]
        if set(A) & set(B) or set(A) | set(B) != set(range(len(fitting))) or set(groups[A]) & set(groups[B]):
            raise ValueError("Cached A/B coverage leaks or omits rows")
        expected.update(A=checked_support(A), B=checked_support(B), folds=[])
        held_rows = []
        for fold in value["folds"]:
            fit, held = fold["fit_rows"], fold["held_rows"]
            if not set(fit) <= set(A) or not set(held) <= set(A) or set(groups[fit]) & set(groups[held]):
                raise ValueError("Cached fold coverage leaks groups")
            expected["folds"].append({"fit": checked_support(fit), "held": checked_support(held)})
            held_rows.extend(held)
        if len(value["folds"]) != 3 or sorted(held_rows) != sorted(A):
            raise ValueError("Cached folds do not cover A exactly once")
    if value["support"] != expected:
        raise ValueError("Cached coverage support differs from fitting labels")


def prepare(args):
    from agent.research_v3 import BASELINE_SOURCE
    from agent.round4_classification_candidate import candidate_source, coverage, make_recipe
    from agent.task_spec import load_task, load_task_data
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT, check=True)
    reference_hashes = {}
    for name in ("agent/models.py", "agent/features.py", "agent/model_config.py", "agent/metrics.py",
                 "agent/research_v3.py", "agent/research_candidate.py", "agent/execution.py",
                 "agent/resources.py", "offline_baseline.py"):
        base_bytes = subprocess.check_output(["git", "show", BASE_SHA + ":" + name], cwd=ROOT)
        reference_hashes[name] = hashlib.sha256(base_bytes).hexdigest()
        if digest(ROOT / name) != reference_hashes[name]:
            raise ValueError("Reference runtime differs from preregistered main: " + name)
    spec = load_task(args.task)
    if spec.metric.name != "Macro F1" or spec.prediction_kind != "labels":
        raise ValueError("Round 4 requires Macro F1 classification labels")
    train, test, *_ = load_task_data(spec)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipes = {}
    for name in SCREEN:
        source = BASELINE_SOURCE if name == "native_fixed" else candidate_source()
        recipe = deepcopy(BASE_RECIPE) if name == "native_fixed" else make_recipe(
            deepcopy(BASE_RECIPE), list(spec.classes), variant=name,
            asymmetry_pairs=deepcopy(ASYMMETRY_PAIRS) if name in ("asymmetry", "extra_trees_asymmetry") else ())
        path = output / (name + ".py")
        path.write_text(source, encoding="utf-8")
        recipes[name] = {"source": path.name, "source_sha256": digest(path), "recipe": recipe,
                         "recipe_sha256": json_hash(recipe), "maximum_physical_fits": PLANNED_FITS[name]}
    splits = grouped_splits(train, spec)
    answers = {}
    for split in splits:
        path = output / "host_only" / ("answers_seed" + str(split["seed"]) + ".csv")
        path.parent.mkdir(exist_ok=True)
        train.iloc[split["held"]].loc[:, [spec.id_column, spec.target_column]].to_csv(path, index=False)
        answers[str(path)] = digest(path)
    context = {"features": list(spec.features), "numeric": list(spec.numeric),
               "categorical": list(spec.categorical), "classes": list(spec.classes),
               "positive_class": spec.positive_class, "task_type": "classification",
               "prediction_kind": "labels", "metric": spec.metric.to_dict(), "recipe": recipes["oof"]["recipe"]}
    cached_coverage = ROOT / "data/round4_coverage_preflight.json"
    if cached_coverage.exists():
        coverage_rows = read_json(cached_coverage)
        if [(row["seed"], row["outer_fit"], row["outer_held"]) for row in coverage_rows] != [
                (split["seed"], len(split["fit"]), len(split["held"])) for split in splits]:
            raise ValueError("Prepared coverage does not match sealed outer partitions")
    else:
        coverage_rows = []
        for split in splits:
            fitting = train.iloc[split["fit"]]
            coverage_rows.append({"seed": split["seed"], "outer_fit": len(fitting), "outer_held": len(split["held"]),
                                  "coverage": coverage(fitting.loc[:, list(spec.features)], fitting[spec.target_column], context)})
    for split, row in zip(splits, coverage_rows):
        validate_coverage(row["coverage"], train.iloc[split["fit"]], spec)
    write_new(output / "coverage_preflight.json", coverage_rows)
    preflight = coverage_rows[0]["coverage"]
    inputs = [spec.task_path, *[spec.data_dir / name for name in
                              ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")]]
    counts = support(train[spec.target_column], spec.classes)
    plan = {"version": 1, "base_sha": BASE_SHA,
            "experiment_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes": source_hashes(), "reference_runtime_hashes": reference_hashes,
            "input_sha256": {str(path): digest(path) for path in inputs}, "host_only_answer_hashes": answers,
            "task": str(spec.task_path), "environment": environment(), "classes": list(spec.classes),
            "mode": "offline", "api_attempts": 0, "api_responses": 0,
            "api_counter_basis": "structural_no_provider_path", "threads": 8,
            "scope": "fixed_recipe_current_main_runtime_not_whole_scheduler", "integration_status": "integration_pending",
            "seconds": {"ordinary": 300, "oof": 600}, "memory_limit_bytes_requested": 6 * 1024 ** 3,
            "memory_monitor_availability": "unknown_until_execution", "competition_resources_confirmed": False,
            "seeds": list(SEEDS), "internal_seed": 42, "splits": splits, "screen_order": list(SCREEN),
            "physical_fit_caps": CAPS, "planned_screen_fits": sum(PLANNED_FITS.values()),
            "finalist_policy": "highest_successful_hypothesis_screen_f1_then_fixed_order_even_if_negative_gain; oof_requires_applied_rule",
            "cost_admission": "4 * identity_wall_seconds * 1.5 + max(120, 2 * identity_wall_seconds) <= 600",
            "confirmation_policy": "one_frozen_finalist_and_native_fixed_on_43_44_137; no_B_tuning",
            "acceptance": {"mean_gain": .003, "minimum_slice_gain": -.01, "minimum_class_gain": -.05,
                           "mean_contribution_excluding_rarest_must_be_positive": True},
            "rarest_class": min(spec.classes, key=lambda label: counts[label]),
            "recipes": recipes, "coverage_preflight_sha256": digest(output / "coverage_preflight.json"),
            "screen_oof_preflight": preflight, "diagnostics": diagnostics(train, spec),
            "test_rows": len(test)}
    write_new(output / "plan.json", plan)
    write_new(output / "seal.json", {"plan_sha256": digest(output / "plan.json")})
    print(json.dumps({"prepared": str(output), "base_sha": BASE_SHA, "plan_sha256": digest(output / "plan.json"),
                      "screen_maximum_fits": plan["planned_screen_fits"], "coverage_reason": preflight.get("reason")}))


def load_plan(output, *, execution=True):
    from agent.task_spec import load_task, load_task_data
    if digest(output / "plan.json") != read_json(output / "seal.json")["plan_sha256"]:
        raise ValueError("Sealed plan changed")
    plan = read_json(output / "plan.json")
    if plan["base_sha"] != BASE_SHA:
        raise ValueError("Wrong preregistered base")
    if execution and (source_hashes() != plan["source_hashes"] or environment() != plan["environment"]):
        raise ValueError("Sources or environment changed after preparation")
    for path, expected in plan["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Official input changed: " + path)
    for path, expected in plan["host_only_answer_hashes"].items():
        if digest(path) != expected:
            raise ValueError("Frozen host-only answers changed")
    if digest(output / "coverage_preflight.json") != plan["coverage_preflight_sha256"]:
        raise ValueError("Frozen coverage changed")
    for item in plan["recipes"].values():
        if digest(output / item["source"]) != item["source_sha256"] or json_hash(item["recipe"]) != item["recipe_sha256"]:
            raise ValueError("Frozen source or recipe changed")
    for split in plan["splits"]:
        if json_hash({key: split[key] for key in ("fit", "held")}) != split["sha256"]:
            raise ValueError("Frozen split changed")
    spec = load_task(Path(plan["task"]))
    train, test, *_ = load_task_data(spec)
    return plan, spec, train, test


def metrics_saved(raw, spec, ids, actuals):
    result = score_saved(raw, spec, ids, actuals)
    matrix = result["confusion_matrix"]
    result["class_counts"] = {}
    for index, label in enumerate(spec.classes):
        tp = matrix[index][index]
        fn, fp = sum(matrix[index]) - tp, sum(row[index] for row in matrix) - tp
        f1 = result["per_class"][label]["f1-score"]
        result["class_counts"][label] = {"TP": tp, "FP": fp, "FN": fn, "support": tp + fn,
                                         "f1": f1, "macro_contribution": f1 / len(spec.classes)}
    return result


def trial_key(name, split):
    return name + ("_seed" + str(split["seed"]) if split is not None else "_full_refit")


def records(output):
    return [read_json(path) for path in sorted((output / "trials").glob("*/record.json"))]


def collect_evidence(directory):
    evidence = []
    for pattern in ("round4_evidence.json", "events.jsonl"):
        for path in sorted((directory / "worker").rglob(pattern)):
            item = {"path": str(path), "sha256": digest(path)}
            try:
                if pattern.endswith("jsonl"):
                    item["content"] = []
                    for line in path.read_text().splitlines():
                        if line.strip():
                            item["content"].append(json.loads(line))
                else:
                    item["content"] = read_json(path)
            except (ValueError, OSError) as error:
                item["parse_error"] = str(error)
            evidence.append(item)
    return evidence


def fit_accounting(name, status, evidence):
    events = [event for item in evidence if item["path"].endswith("events.jsonl")
              for event in item.get("content", [])]
    started = sum(event.get("status") == "started" for event in events)
    completed = sum(event.get("status") == "completed" for event in events)
    cancelled = sum(event.get("status") == "cancelled_before_fit" for event in events)
    actual_started = started - cancelled
    summaries = [item["content"] for item in evidence if item["path"].endswith("round4_evidence.json")
                 and isinstance(item.get("content"), dict)]
    if name == "native_fixed" and status == "completed":
        return {"actual_started": 2, "actual_completed": 2, "budget_charge": 2,
                "basis": "inferred_fixed_recipe_two_components_successful_host_receipt_no_fit_events"}
    if summaries and not any("parse_error" in item for item in evidence):
        summary = summaries[0]
        if (summary.get("physical_fits_started") != actual_started or summary.get("physical_fits_completed") != completed
                or completed > actual_started or started > PLANNED_FITS[name]):
            raise ValueError("Physical fit summary disagrees with child start/complete events")
        return {"actual_started": actual_started, "actual_completed": completed, "budget_charge": started,
                "basis": "adapter_start_complete_events", "observed_started": started, "observed_completed": completed,
                "cancelled_before_fit": cancelled}
    # A killed parent can leave no final summary. Every known start counts, plus
    # one conservative two-component child reservation; actual totals stay null.
    return {"actual_started": None, "actual_completed": None,
            "budget_charge": min(PLANNED_FITS[name], started + 2), "basis": "interrupted_unknown_reserved_two",
            "observed_started": started, "observed_completed": completed, "cancelled_before_fit": cancelled}


def invoke(output, plan, spec, train, test, name, *, split, stage):
    from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
    directory = output / "trials" / trial_key(name, split)
    directory.mkdir(parents=True, exist_ok=False)
    item = plan["recipes"][name]
    frozen = freeze_candidate(output / item["source"], item["recipe"], directory / "frozen")
    fit = train.iloc[split["fit"]].copy() if split is not None else train.copy()
    predict = train.iloc[split["held"]].drop(columns=[spec.target_column]).copy() if split is not None else test.copy()
    actuals = train.iloc[split["held"]][spec.target_column].tolist() if split is not None else None
    seconds = plan["seconds"]["oof" if name == "oof" else "ordinary"]
    phase = "inner_selection" if split is not None else "full_refit"
    started = time.monotonic()
    row = {"candidate": name, "seed": split["seed"] if split is not None else None, "stage": stage, "phase": phase,
           "plan_sha256": digest(output / "plan.json"), "split_sha256": split["sha256"] if split else None,
           "code_hash": frozen.code_hash, "recipe_hash": frozen.recipe_hash,
           "fit_rows": len(fit), "predict_rows": len(predict), "timeout_seconds": seconds,
           "memory_limit_bytes_requested": plan["memory_limit_bytes_requested"], "api_attempts": 0, "api_responses": 0,
           "api_counter_basis": "structural_no_provider_path", "candidate_reached": False}
    try:
        result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=predict,
                               output_dir=directory / "worker", phase=phase, timeout=seconds,
                               actuals=actuals, max_memory_bytes=plan["memory_limit_bytes_requested"])
        row.update(status="completed", execution=result.execution, output_path=str(result.output_path),
                   output_sha256=result.output_hash, receipt_path=str(result.receipt_path), receipt_sha256=result.receipt_hash)
        if split is not None:
            measured = metrics_saved(result.read_output(), spec, predict[spec.id_column].tolist(), actuals)
            if not math.isclose(measured["macro_f1"], result.metric["value"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Independent saved CSV score differs from host")
            row.update(measured)
        else:
            publication = directory / "final_predictions.csv"
            result.copy_output(publication)
            row.update(published_path=str(publication), published_sha256=digest(publication))
    except CandidateExecutionError as error:
        row.update(status="failed", error=str(error), execution=error.execution)
    except Exception as error:
        row.update(status="error", error=type(error).__name__ + ": " + str(error))
    row["elapsed_seconds"] = time.monotonic() - started
    row["candidate_evidence"] = collect_evidence(directory)
    row["candidate_reached"] = bool(row["candidate_evidence"]) or row["status"] == "completed"
    try:
        row["physical_fits"] = fit_accounting(name, row["status"], row["candidate_evidence"])
    except ValueError as error:
        row.update(status="error", error=str(error))
        row["physical_fits"] = {"actual_started": None, "actual_completed": None,
                                "budget_charge": PLANNED_FITS[name], "basis": "invalid_evidence_reserved_planned_maximum"}
    summaries = [x["content"] for x in row["candidate_evidence"] if x["path"].endswith("round4_evidence.json")
                 and isinstance(x.get("content"), dict)]
    row["decision"] = summaries[0].get("decision") if summaries else None
    row["oof"] = summaries[0].get("oof") if summaries else None
    resources = row.get("execution", {}).get("resources", {})
    row.update(resource_monitor_available=resources.get("available", False), peak_rss_bytes=resources.get("peak_rss_bytes"),
               memory_limit_enforced=bool(resources.get("available", False)),
               resource_limitation=None if resources.get("available", False) else "requested 6GiB sampled tree cap unavailable; cgroup bound is environment evidence; tree RSS/cleanup unknown")
    row["invocation_inputs"] = {str(path): digest(path) for path in (directory / "worker").glob("*/*")
                                if path.name in ("candidate.py", "recipe.json", "context.json", "fit.csv", "predict.csv")}
    write_new(directory / "record.json", row)
    print(json.dumps({key: row.get(key) for key in ("candidate", "seed", "status", "macro_f1", "elapsed_seconds", "physical_fits", "error")}), flush=True)
    return row


def skip(output, plan, name, split, stage, reason):
    row = {"candidate": name, "seed": split["seed"] if split else None, "phase": "inner_selection" if split else "full_refit",
           "stage": stage, "plan_sha256": digest(output / "plan.json"), "status": "skipped", "reason": reason,
           "physical_fits": {"actual_started": 0, "actual_completed": 0, "budget_charge": 0, "basis": "no_worker_started"}}
    write_new(output / "trials" / trial_key(name, split) / "record.json", row)
    return row


def admit_cost(wall):
    return math.isfinite(wall) and wall >= 0 and 4 * wall * 1.5 + max(120, 2 * wall) <= 600


def identity_agreement(rows):
    import pandas as pd
    from io import BytesIO
    primary = {row["candidate"]: row for row in rows if row.get("stage") == "screen"}
    pair = [primary.get(name, {}) for name in ("native_fixed", "identity")]
    if any(row.get("status") != "completed" for row in pair):
        return False
    frames = []
    for row in pair:
        if digest(row["output_path"]) != row["output_sha256"]:
            raise ValueError("Identity probe output changed")
        frames.append(pd.read_csv(BytesIO(Path(row["output_path"]).read_bytes()), dtype=str, keep_default_na=False))
    return frames[0].equals(frames[1])


def choose_finalist(rows):
    primary = {row["candidate"]: row for row in rows if row.get("stage") == "screen"}
    if set(primary) != set(SCREEN):
        raise ValueError("All seven planned screen outcomes are required")
    if not identity_agreement(rows):
        return None
    eligible = [name for name in HYPOTHESES if primary[name]["status"] == "completed"
                and (name != "oof" or (primary[name].get("decision") or {}).get("rules_applied", 0) > 0)]
    return min(eligible, key=lambda name: (-primary[name]["macro_f1"], HYPOTHESES.index(name))) if eligible else None


def freeze_selection(output, plan):
    path = output / "confirmation_selection.json"
    if path.exists():
        if digest(path) != read_json(output / "confirmation_selection.sha256.json")["sha256"]:
            raise ValueError("Confirmation selection changed")
        selection = read_json(path)
    else:
        selected = choose_finalist(records(output))
        selection = {"plan_sha256": digest(output / "plan.json"), "candidate": selected,
                     "recipes": {name: plan["recipes"][name] for name in ("native_fixed", selected) if name},
                     "seeds": list(SEEDS[1:]), "policy": "frozen_before_confirmation; no_B_tuning",
                     "screen_record_hashes": {str(p): digest(p) for p in sorted((output / "trials").glob("*_seed42/record.json"))}}
        write_new(path, selection)
        write_new(output / "confirmation_selection.sha256.json", {"sha256": digest(path)})
    if (selection["plan_sha256"] != digest(output / "plan.json")
            or selection["candidate"] != choose_finalist(records(output))
            or any(digest(path) != expected for path, expected in selection["screen_record_hashes"].items())
            or any(item != plan["recipes"][name] for name, item in selection["recipes"].items())):
        raise ValueError("Frozen finalist provenance differs from measured screen")
    return selection


def run(args):
    output = args.output.resolve()
    plan, spec, train, test = load_plan(output)
    stage = args.command
    if stage == "screen":
        planned = [(name, plan["splits"][0]) for name in SCREEN]
    else:
        selected = freeze_selection(output, plan)["candidate"]
        if selected is None:
            export_table(output, plan)
            print(json.dumps({"decision": "retain_incumbent", "reason": "no_eligible_screen_finalist"}))
            return
        planned = ([(selected, None)] if stage == "full-refit" else
                   [(name, split) for split in plan["splits"][1:] for name in ("native_fixed", selected)])
    for name, split in planned:
        directory = output / "trials" / trial_key(name, split)
        if (directory / "record.json").exists():
            continue
        if directory.exists():
            raise ValueError("Interrupted invocation without record; preserve evidence and audit before continuing: " + str(directory))
        rows = records(output)
        if stage == "screen" and name in HYPOTHESES:
            if not identity_agreement(rows):
                skip(output, plan, name, split, stage, "identity_contract_failed")
                continue
            probe = next(row for row in rows if row["candidate"] == "identity" and row["stage"] == "screen")
            if name == "oof" and not admit_cost(probe["elapsed_seconds"]):
                skip(output, plan, name, split, stage, "cost_admission")
                continue
        charged = sum(row["physical_fits"]["budget_charge"] for row in rows if row["stage"] == stage)
        if charged + PLANNED_FITS[name] > plan["physical_fit_caps"][stage]:
            skip(output, plan, name, split, stage, "physical_fit_cap")
            continue
        invoke(output, plan, spec, train, test, name, split=split, stage=stage)
    export_table(output, plan)


def paired_table(rows, plan):
    controls = {row["seed"]: row for row in rows if row["candidate"] == "native_fixed" and row["status"] == "completed"}
    paired = []
    for row in rows:
        control = controls.get(row.get("seed"))
        if row["candidate"] in ("native_fixed", "identity") or row["status"] != "completed" or control is None:
            continue
        contributions = {label: (row["class_counts"][label]["f1"] - control["class_counts"][label]["f1"]) / len(plan["classes"])
                         for label in plan["classes"]}
        paired.append({"candidate": row["candidate"], "seed": row["seed"], "stage": row["stage"],
                       "macro_delta": row["macro_f1"] - control["macro_f1"], "class_macro_contributions": contributions,
                       "class_f1_deltas": {label: value * len(plan["classes"]) for label, value in contributions.items()},
                       "macro_delta_excluding_rarest_contribution": sum(value for label, value in contributions.items()
                                                                         if label != plan["rarest_class"])})
    confirmed = [row for row in paired if row["stage"] == "confirm"]
    result = {"status": "confirmation_incomplete", "recommendation": "retain_incumbent"}
    if {row["seed"] for row in confirmed} == set(SEEDS[1:]):
        mean = sum(row["macro_delta"] for row in confirmed) / 3
        other = sum(row["macro_delta_excluding_rarest_contribution"] for row in confirmed) / 3
        criteria = plan["acceptance"]
        accepted = (mean >= criteria["mean_gain"] and other > 0
                    and all(row["macro_delta"] >= criteria["minimum_slice_gain"] for row in confirmed)
                    and all(value >= criteria["minimum_class_gain"] for row in confirmed for value in row["class_f1_deltas"].values()))
        result = {"status": "confirmed", "mean_macro_delta": mean, "mean_other_class_contribution": other,
                  "recommendation": "integration_pending" if accepted else "retain_incumbent"}
    return {"paired": paired, "confirmation": result}


def export_table(output, plan):
    rows = records(output)
    table = {"plan_sha256": digest(output / "plan.json"), "scope": plan["scope"], "outcomes": rows,
             "physical_fit_budget_charges": {stage: sum(row["physical_fits"]["budget_charge"] for row in rows if row["stage"] == stage)
                                             for stage in CAPS}, **paired_table(rows, plan)}
    (output / "experiments.json").write_text(json.dumps(table, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return table


def audit(args):
    import pandas as pd
    output = args.output.resolve()
    plan, spec, train, test = load_plan(output, execution=False)
    rows = records(output)
    if {row["candidate"] for row in rows if row["stage"] == "screen"} != set(SCREEN):
        raise ValueError("Missing planned screen outcomes")
    selection = freeze_selection(output, plan) if (output / "confirmation_selection.json").exists() else None
    if selection and selection["candidate"]:
        expected = {(name, seed) for seed in SEEDS[1:] for name in ("native_fixed", selection["candidate"])}
        if {(row["candidate"], row["seed"]) for row in rows if row["stage"] == "confirm"} != expected:
            raise ValueError("Missing or unexpected confirmation outcomes")
    verified = 0
    for row in rows:
        if row["plan_sha256"] != digest(output / "plan.json"):
            raise ValueError("Outcome belongs to a different plan")
        for item in row.get("candidate_evidence", []):
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Child fit evidence changed")
        if row["status"] == "skipped":
            if row["physical_fits"]["budget_charge"] != 0:
                raise ValueError("Skipped outcome has inconsistent fit accounting")
            continue
        if collect_evidence(output / "trials" / trial_key(row["candidate"], {"seed": row["seed"]} if row["seed"] is not None else None)) != row["candidate_evidence"]:
            raise ValueError("Recorded child evidence differs from saved evidence")
        if fit_accounting(row["candidate"], row["status"], row["candidate_evidence"]) != row["physical_fits"]:
            raise ValueError("Physical fit accounting does not reproduce")
        for path, expected in row["invocation_inputs"].items():
            if digest(path) != expected:
                raise ValueError("Worker input changed")
        if row["status"] != "completed":
            continue
        path = Path(row["output_path"])
        invocation = path.parent
        context, receipt = read_json(invocation / "context.json"), read_json(row["receipt_path"])
        item = plan["recipes"][row["candidate"]]
        expected = {"version": 3, "nonce": context["nonce"], "phase": row["phase"],
                    "code_hash": item["source_sha256"], "recipe_hash": item["recipe_sha256"], "output_hash": row["output_sha256"]}
        if (receipt != expected or digest(row["receipt_path"]) != row["receipt_sha256"]
                or digest(path) != row["output_sha256"] or digest(invocation / "candidate.py") != expected["code_hash"]
                or digest(invocation / "recipe.json") != expected["recipe_hash"]
                or any(context[key] != expected[key] or row[key] != expected[key] for key in ("phase", "code_hash", "recipe_hash"))):
            raise ValueError("Receipt nonce/phase/code/recipe/output identity mismatch")
        split = next((s for s in plan["splits"] if s["seed"] == row["seed"]), None)
        fit = train.iloc[split["fit"]] if split else train
        predict = train.iloc[split["held"]] if split else test
        for filename, frame, columns in (("fit.csv", fit, [spec.id_column, *spec.features, spec.target_column]),
                                          ("predict.csv", predict, [spec.id_column, *spec.features])):
            actual = pd.read_csv(invocation / filename, dtype=str, keep_default_na=False)
            if not actual.equals(frame.loc[:, columns].reset_index(drop=True)):
                raise ValueError("Worker fit/predict rows differ from sealed partition")
        for evidence in row["candidate_evidence"]:
            if not evidence["path"].endswith("round4_evidence.json"):
                continue
            value = evidence["content"]
            if (any(value.get(key) != expected[key] for key in ("nonce", "phase", "code_hash", "recipe_hash"))
                    or value.get("status") != "completed" or value.get("fresh_full_refit") is not True
                    or value.get("full_fit_rows") != len(fit) or value.get("predict_rows") != len(predict)
                    or value.get("classes") != list(spec.classes) or value.get("decision") != row.get("decision")
                    or value.get("oof") != row.get("oof")):
                raise ValueError("Adapter evidence does not substantiate fresh fit or invocation identity")
        if row["candidate"] != "native_fixed" and not any(x["path"].endswith("round4_evidence.json") for x in row["candidate_evidence"]):
            raise ValueError("Adapter success lacks physical fit evidence")
        if split:
            rescored = metrics_saved(path.read_bytes(), spec, predict[spec.id_column].tolist(), predict[spec.target_column].tolist())
            if any(row[key] != value for key, value in rescored.items()):
                raise ValueError("Saved CSV metrics do not reproduce")
        else:
            spec.validate_frame(pd.read_csv(path, dtype=str, keep_default_na=False), test[spec.id_column].tolist())
            if digest(row["published_path"]) != row["published_sha256"] or row["published_sha256"] != row["output_sha256"]:
                raise ValueError("Full-refit publication differs from verified output")
        verified += 1
    table = export_table(output, plan)
    if any(value > plan["physical_fit_caps"][stage] for stage, value in table["physical_fit_budget_charges"].items()):
        raise ValueError("Physical fit cap exceeded")
    report = {"verified_completed_invocations": verified, "recorded_outcomes": len(rows),
              "identity_label_agreement": identity_agreement(rows), "audit_harness_sha256": digest(Path(__file__)),
              "confirmation": table["confirmation"], "official_inputs_unchanged": True}
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "confirm", "full-refit", "audit"))
    parser.add_argument("--mode", choices=("offline",), default="offline")
    parser.add_argument("--output", type=Path, default=ROOT / "data/r4c")
    parser.add_argument("--task", type=Path, default=ROOT / "data/classification/task.md")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "audit":
        audit(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
