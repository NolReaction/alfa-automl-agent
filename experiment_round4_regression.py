"""Serial round-4 regression screening using the existing frozen-candidate host.

This developer tool never enters the competition runtime. Development answers
stay in the parent; the worker receives only fit labels and prediction features.
The original native scheduler is deliberately not rerun for each fixed recipe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from experiment_rescue_regression_screen import (
    CSV, ROOT, case_frames, digest, plan as checked_inputs, prepare as prepare_inputs,
    read, score_csv, source, write,
)
from agent.task_spec import load_task

BASE_SHA = "e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb"
PLAN_DOCUMENT = ROOT / "docs/round4_regression_plan.md"
MEMORY_BYTES = 8 * 1024**3


def _text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value):
    return _text_hash(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def prepare(task, output, period_column, size_column):
    # Reuse the already-audited public TRAIN fold writer. This research tool's
    # y2/y3/y4/y5 names are not production conditions or feature inference.
    if not PLAN_DOCUMENT.is_file():
        raise ValueError("Write the round-4 plan before preparing experiments")
    prepare_inputs(task, output, period_column, size_column)
    path = Path(output).resolve() / "plan.json"
    value = read(path)
    value.update(round=4, incumbent_sha=BASE_SHA,
        plan_document=str(PLAN_DOCUMENT), plan_document_sha256=digest(PLAN_DOCUMENT),
        comparison="Fixed current rescue algorithm on identical fits; not a full scheduler replay",
        history_policy="Host INNER-fit only; original 8M/48 discovery; no runtime rediscovery",
        budget_seconds=180, screening_year=4, confirmation_years=[2, 3], seed=42,
        maximum_configurations=11, maximum_finalists=2)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


def plan(output):
    value = checked_inputs(output)
    if value.get("round") != 4 or value.get("incumbent_sha") != BASE_SHA:
        raise ValueError("Not a round-4 incumbent plan")
    if digest(PLAN_DOCUMENT) != value["plan_document_sha256"]:
        raise ValueError("Predeclared round-4 plan changed")
    return value


def discover(output, year):
    from agent.regression_discovery import discover_regression_schema_bounded
    output = Path(output).resolve()
    frozen = plan(output)
    if year == 5 and not (output / "frozen_decision.json").is_file():
        raise ValueError("Freeze the development decision before known Year5")
    spec = load_task(output / "inputs" / f"y{year}" / "task.md")
    fit = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    period = frozen["period_column"]
    values = pd.to_numeric(fit[period], errors="raise")
    # External development t is the host OUTER. Its fit excludes t; the
    # host INNER-fit also excludes the latest remaining period t-1. Candidate
    # internal calibration is another fit-only operation, not discovery input.
    mask = values < values.max()
    inner = fit.loc[mask, list(spec.features)].copy()
    directory = output / "discovery" / f"y{year}"
    report = discover_regression_schema_bounded(inner,
        feature_columns=list(spec.features), numeric_columns=list(spec.numeric),
        period_column=period, output_dir=directory, extended=False)
    record = {"version": 1, "year": year, "period_column": period,
        "outer_fit_rows": len(fit), "host_inner_fit_rows": len(inner),
        "host_inner_fit_ids_sha256": _json_hash(fit.loc[mask, spec.id_column].tolist()),
        "fit_input_sha256": report["fit_input_sha256"],
        "discovery_report_sha256": digest(directory / "discovery_report.json"),
        "selected_config": report["selected_config"], "status": report["status"],
        "reason": report["reason"], "max_cell_visits": 8_000_000,
        "max_checked_configs": 48,
        "nesting": "development outer t; host INNER held latest fit period; candidate calibration inside its supplied fit"}
    write(directory / "host_schema.json", record)
    return record


def schema(output, year):
    directory = Path(output) / "discovery" / f"y{year}"
    record = read(directory / "host_schema.json")
    if record["discovery_report_sha256"] != digest(directory / "discovery_report.json"):
        raise ValueError("Frozen discovery report changed")
    report = read(directory / "discovery_report.json")
    if (report["selected_config"] != record["selected_config"]
            or report["max_cell_visits"] != 8_000_000 or report["max_checked_configs"] != 48):
        raise ValueError("Only the frozen original host discovery is allowed")
    if report["fit_input_sha256"] is not None:
        if digest(directory / "fit_features.csv") != report["fit_input_sha256"]:
            raise ValueError("Discovery input changed")
    return record


def check_recipe(recipe, frozen, discovered):
    if not isinstance(recipe, dict) or recipe.get("history_config") != discovered["selected_config"]:
        raise ValueError("Every arm must use the same frozen host history schema")
    if recipe.get("base", {}).get("period_column") != frozen["period_column"]:
        raise ValueError("Candidate period differs from the declared host period")
    if discovered["period_column"] != frozen["period_column"]:
        raise ValueError("Discovery period differs from the declared host period")


def algorithm(recipe):
    result = deepcopy(recipe)
    result["history_config"] = None
    return result


def instrument_source(original):
    """Observe physical model fits inside each isolated candidate process.

    Both arms get exactly the same observer. It neither changes model inputs
    nor intercepts predictions. Each fit start is persisted before fitting so
    a timed-out process cannot turn an unfinished fit into a fictitious zero.
    """
    return original + r'''

_round4_original_fit_predict = fit_predict

def fit_predict(X_fit, y_fit, X_predict, context):
    import json, time
    from pathlib import Path
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    ledger = []
    path = Path("physical_fits.json")
    def persist():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "fits": ledger}, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    def observed(original, engine):
        def wrapped(model, *args, **kwargs):
            X = args[0] if args else kwargs.get("X")
            count = X.shape[0] if hasattr(X, "shape") else (X.num_row() if hasattr(X, "num_row") else None)
            item = {"engine": engine, "rows": int(count) if count is not None else None,
                    "started": True, "completed": False, "elapsed_seconds": None,
                    "exception_type": None}
            ledger.append(item)
            persist()
            started = time.monotonic()
            try:
                result = original(model, *args, **kwargs)
                item["completed"] = True
                return result
            except BaseException as error:
                item["exception_type"] = type(error).__name__
                raise
            finally:
                item["elapsed_seconds"] = time.monotonic() - started
                persist()
        return wrapped
    original_cat, original_lgb = CatBoostRegressor.fit, LGBMRegressor.fit
    persist()
    CatBoostRegressor.fit = observed(original_cat, "catboost")
    LGBMRegressor.fit = observed(original_lgb, "lightgbm")
    try:
        return _round4_original_fit_predict(X_fit, y_fit, X_predict, context)
    finally:
        CatBoostRegressor.fit, LGBMRegressor.fit = original_cat, original_lgb
        persist()
'''


def diagnostics(fit, held, answers, predictions, spec, frozen, config):
    from experiment_rescue_regression_diagnostics import segment_rows, _statistics
    from agent.building_history import history_keys, numeric_value
    from agent.regression_history import HistorySignals
    from agent.history_candidate import HistoryConfig
    X, query = fit.loc[:, list(spec.features)], held.loc[:, list(spec.features)]
    y, truth = pd.to_numeric(fit[spec.target_column]), pd.to_numeric(answers[spec.target_column])
    rows = segment_rows(X, y, query, truth, predictions,
        period_column=frozen["period_column"], size_column=frozen["size_column"],
        categorical=list(spec.categorical), reference_history_config=config)
    rows = [row for row in rows if row["partition"] != "history"]
    if config is None:
        masks = {"schema_unavailable": np.ones(len(query), dtype=bool)}
    else:
        parsed = HistoryConfig.from_json(config)
        signals = HistorySignals(config).fit(X, y).transform(query)
        keys = history_keys(query, parsed.key_columns, parsed.numeric_columns, parsed.positive_columns)
        periods = [numeric_value(value) for value in query[parsed.period_column]]
        invalid = np.asarray([key is None or period is None for key, period in zip(keys, periods)])
        masks = {"eligible": signals["eligible"].to_numpy(dtype=bool),
            "ambiguous": (signals["past_rows"].to_numpy() > 0) & signals["ambiguous"].to_numpy(dtype=bool),
            "valid_without_past": ~invalid & (signals["past_rows"].to_numpy() == 0),
            "incomplete_key_or_invalid_period": invalid}
    if not np.equal(np.sum(list(masks.values()), axis=0), 1).all():
        raise ValueError("History groups must partition held rows")
    for arm, values in predictions.items():
        errors = truth.to_numpy(dtype=float) - np.asarray(values, dtype=float)
        total = _statistics(errors, np.ones(len(errors), dtype=bool), None)["sse"]
        for name, mask in masks.items():
            rows.append({"arm": arm, "partition": "history", "segment": name,
                "schema_available": config is not None, **_statistics(errors, mask, total)})
    return rows


def _clean_identity():
    identity = source(ROOT)
    if identity["dirty"]:
        raise ValueError("Seal candidate code in a clean commit before measuring")
    return identity


def _run(output, directory, fit, held, spec, frozen, recipe, original, budget, *, answers=None):
    from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 360:
        raise ValueError("Budget must be finite, positive and at most 360 seconds")
    started = time.monotonic()
    identity = _clean_identity()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "source.py"
    path.write_text(instrument_source(original), encoding="utf-8")
    (directory / "original_source.py").write_text(original, encoding="utf-8")
    candidate = freeze_candidate(path, recipe, directory / "frozen")
    record = {"version": 1, "status": "running", "source": identity,
        "incumbent_sha": BASE_SHA, "recipe": recipe, "recipe_sha256": candidate.recipe_hash,
        "original_source_sha256": _text_hash(original), "code_sha256": candidate.code_hash,
        "budget_seconds": budget, "fit_rows": len(fit), "predict_rows": len(held),
        "api_calls": 0, "metrics": None, "physical_fits": None,
        "physical_fits_reason": "observer not yet available", "quality_scope": "development" if answers is not None else "unlabelled_competition_test"}
    try:
        # Candidate host preparation and validation share this same allowance.
        result = run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=held,
            output_dir=directory, phase="full_refit", timeout=budget,
            deadline=started + budget, max_memory_bytes=MEMORY_BYTES)
        result.verify()
        result.copy_output(directory / "submission.csv")
        work = result.output_path.parent
        evidence = read(work / "regression_evidence.json")
        ablation_path = work / "regression_ablation.csv"
        if digest(ablation_path) != evidence["ablation_sha256"]:
            raise ValueError("Candidate ablation hash mismatch")
        alternatives = pd.read_csv(ablation_path, **CSV)
        if list(alternatives) != ["base", "mean", "calibrated"] or len(alternatives) != len(held):
            raise ValueError("Candidate ablation contract changed")
        scores, predictions = {}, {}
        for name in alternatives:
            values = pd.to_numeric(alternatives[name], errors="raise").to_numpy(dtype=float)
            frame = spec.prediction_frame(held[spec.id_column].tolist(), values)
            target = directory / (name + ".csv")
            frame.to_csv(target, index=False)
            predictions[name] = values
            if answers is not None:
                scores[name] = score_csv(spec.task_path, answers, target)
        public = pd.read_csv(directory / "submission.csv", **CSV)
        selected = "base" if recipe["history_policy"] == "off" else recipe["history_policy"]
        if not np.array_equal(pd.to_numeric(public[spec.target_column]), predictions[selected]):
            raise ValueError("Published predictions differ from declared policy")
        if answers is not None:
            answer_frame = pd.read_csv(answers, **CSV)
            segments = diagnostics(fit, held, answer_frame, predictions, spec, frozen, recipe["history_config"])
            write(directory / "segments.json", {"rows": segments,
                "note": "Partitions overlap each other; each individual partition is disjoint. No schema means key validity is unknown."})
        record.update(status="completed", execution=result.execution, evidence=evidence,
            metrics=scores if answers is not None else None, receipt_sha256=result.receipt_hash,
            submission_sha256=result.output_hash, ids_exact_order=True, finite=True,
            quality_metric=None if answers is None else scores[selected]["rmse"])
    except CandidateExecutionError as error:
        record.update(status="failed", error=str(error), execution=error.execution)
    except (ValueError, OSError, TimeoutError) as error:
        record.update(status="failed", error=str(error), execution=None)
    ledgers = sorted(directory.glob("full_refit_*/physical_fits.json"))
    if len(ledgers) == 1:
        ledger = read(ledgers[0])
        record.update(physical_fits=ledger["fits"], physical_fits_reason=None,
            physical_fit_count=len(ledger["fits"]), completed_fit_count=sum(f["completed"] for f in ledger["fits"]),
            physical_fits_sha256=digest(ledgers[0]))
    else:
        record.update(physical_fit_count=None, completed_fit_count=None,
            physical_fits_reason="observer unavailable before model fits or process terminated")
    record["host_total_seconds"] = time.monotonic() - started
    if source(ROOT) != identity:
        raise ValueError("Candidate source changed during measurement")
    write(directory / "summary.json", record)
    return record


def run(output, year, arm, budget=180, source_file=None, recipe_file=None):
    from agent.regression_candidate import candidate_source, make_regression_recipe
    output = Path(output).resolve()
    frozen = plan(output)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", arm):
        raise ValueError("Use a simple arm name")
    if (source_file is None) != (recipe_file is None):
        raise ValueError("Provide both candidate source and recipe, or neither")
    if arm == "incumbent" and source_file is not None:
        raise ValueError("The incumbent always uses the current unchanged rescue-v1 factory")
    if year == 5:
        if not (output / "frozen_decision.json").is_file():
            raise ValueError("Freeze the development decision before known Year5")
        decision = read(output / "frozen_decision.json")
        if _clean_identity() != decision["source"]:
            raise ValueError("Known Year5 requires the frozen tested source")
        if arm not in {"incumbent", decision["winner"]}:
            raise ValueError("Known Year5 permits only incumbent and frozen winner")
    discovered = schema(output, year)
    spec, fit, held, _ = case_frames(output, year)
    if source_file is None and arm == "incumbent":
        original = candidate_source()
        recipe = make_regression_recipe(frozen["period_column"], history_config=discovered["selected_config"])
    elif source_file is None and year == 5:
        original = (output / "frozen_winner_source.py").read_text(encoding="utf-8")
        recipe = deepcopy(decision["algorithm"])
        recipe["history_config"] = discovered["selected_config"]
    elif source_file is None:
        raise ValueError("A new arm needs explicit source and recipe files")
    else:
        original = Path(source_file).read_text(encoding="utf-8")
        recipe = read(recipe_file)
    check_recipe(recipe, frozen, discovered)
    if year == 5 and arm != "incumbent":
        if algorithm(recipe) != decision["algorithm"] or _text_hash(original) != decision["original_source_sha256"]:
            raise ValueError("Known Year5 must use the frozen development algorithm")
    record = _run(output, output / "runs" / f"y{year}" / arm, fit, held, spec, frozen,
        recipe, original, budget, answers=output / "host_only" / f"y{year}.csv")
    print(json.dumps({"year": year, "arm": arm, "status": record["status"], "metrics": record["metrics"]}), flush=True)
    return record


def freeze(output, winner, finalists):
    output = Path(output).resolve()
    plan(output)
    identity = _clean_identity()
    if not finalists or len(set(finalists)) != len(finalists) or len(finalists) > 2 or winner not in finalists:
        raise ValueError("Freeze one winner from at most two distinct finalists")
    records, template, original_hash = {}, None, None
    for year in (2, 3, 4):
        for arm in ("incumbent", *finalists):
            path = output / "runs" / f"y{year}" / arm / "summary.json"
            record = read(path)
            if record["status"] != "completed" or record["source"] != identity:
                raise ValueError("Every finalist and incumbent needs completed results from this clean source on all development folds")
            records[f"y{year}/{arm}"] = {"summary_sha256": digest(path), "metrics": record["metrics"]}
            if arm == winner:
                current = algorithm(record["recipe"])
                if template is not None and (current != template or record["original_source_sha256"] != original_hash):
                    raise ValueError("Winner algorithm changed across development folds")
                template, original_hash = current, record["original_source_sha256"]
    original = (output / "runs/y4" / winner / "original_source.py").read_text(encoding="utf-8")
    if _text_hash(original) != original_hash:
        raise ValueError("Winner source changed")
    (output / "frozen_winner_source.py").write_text(original, encoding="utf-8")
    decision = {"version": 1, "source": identity, "created_at": datetime.now(timezone.utc).isoformat(),
        "winner": winner, "finalists": finalists, "algorithm": template,
        "original_source_sha256": original_hash, "development": records,
        "status": "isolated_finalist_frozen_native_admission_pending"}
    write(output / "frozen_decision.json", decision)
    return decision


def full_refit(output, budget=180):
    output = Path(output).resolve()
    frozen = plan(output)
    decision = read(output / "frozen_decision.json")
    if _clean_identity() != decision["source"]:
        raise ValueError("Full refit requires the frozen tested source")
    known = output / "runs/y5" / decision["winner"]
    measured = read(known / "summary.json")
    if measured["status"] != "completed":
        raise ValueError("Complete the single known-Year5 compatibility check first")
    original = (known / "original_source.py").read_text(encoding="utf-8")
    recipe = measured["recipe"]
    if (_text_hash(original) != decision["original_source_sha256"]
            or algorithm(recipe) != decision["algorithm"]):
        raise ValueError("Final source/recipe differs from frozen winner")
    check_recipe(recipe, frozen, schema(output, 5))
    spec = load_task(frozen["source_task"])
    for name, expected in frozen["source_files"].items():
        path = spec.task_path if name == "task.md" else spec.data_dir / name
        if digest(path) != expected:
            raise ValueError("Full-refit input changed")
    fit = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    held = pd.read_csv(spec.data_dir / "test.csv", **CSV)
    return _run(output, output / "full_refit", fit, held, spec, frozen, recipe, original, budget)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--period-column", required=True)
    p.add_argument("--size-column", required=True)
    for name in ("discover", "run"):
        p = commands.add_parser(name)
        p.add_argument("--year", type=int, choices=(2, 3, 4, 5), required=True)
        if name == "run":
            p.add_argument("--arm", required=True)
            p.add_argument("--source-file", type=Path)
            p.add_argument("--recipe-file", type=Path)
            p.add_argument("--budget", type=float, default=180)
    p = commands.add_parser("freeze")
    p.add_argument("--winner", required=True)
    p.add_argument("--finalists", nargs="+", required=True)
    p = commands.add_parser("full-refit")
    p.add_argument("--budget", type=float, default=180)
    for p in commands.choices.values():
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.task, args.output, args.period_column, args.size_column)
    elif args.command == "discover":
        result = discover(args.output, args.year)
    elif args.command == "run":
        result = run(args.output, args.year, args.arm, args.budget, args.source_file, args.recipe_file)
    elif args.command == "freeze":
        result = freeze(args.output, args.winner, args.finalists)
    else:
        result = full_refit(args.output, args.budget)
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
