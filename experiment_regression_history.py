"""Does the building history help the regression models? A fixed matrix of experiments, no search.

For each validation year (3, 4, 5) the models are fitted on the rows of all earlier years and checked on that year;
every variant sees exactly the same fitting and validation rows. Six trained variants:

    1 HGB, all features                      4 ExtraTrees, all features
    2 HGB without Year_Factor                5 ExtraTrees without Year_Factor
    3 HGB without Year_Factor + history      6 ExtraTrees without Year_Factor + history

and four fixed blends of variants 1, 2, 4, 5: where the key has history 0.5 * model + 0.5 * history mean, elsewhere
the model. The weight 0.5 is fixed beforehand and never tuned. The history features come from a new HistoryEncoder per
year (fit_transform on the fitting rows, transform on the validation rows) and are computed before Year_Factor is
removed from the model features; the blends are made by the history adapter (agent/history_candidate.py). Models are
the project's offline_baseline pipeline; only the model step of the ExtraTrees variants is replaced.

    python experiment_regression_history.py [--data-dir DIR] [--out-dir DIR]

Results go to the output folder (default data/experiments/regression_history, ignored by Git): metrics.json,
metrics.csv, predictions_year<N>.csv, input_hashes.json. The input files are hashed before and after the run.

Other modes (default folders data/experiments/murad-history-v2/<mode>):

    --mode nested    for each outer year t: inner sequential splits inside the years before t choose the adapter
                     weight from WEIGHT_GRID by the task metric (base model and adapter refitted on each inner past);
                     then both are refitted on all years before t and scored once on year t. Every base model is
                     fitted in its own bounded process (--fit-timeout, --memory-gib, overall --deadline). Paired
                     bootstrap intervals against the model alone resample key clusters (rows as a sensitivity check).
    --mode controls  the same nested procedure on synthetic data with and without a persistent object level
    --mode scaling   time and traced memory of HistoryEncoder.fit for one key with a growing history, next to the
                     former quadratic summation, and the encoder cost on the official data
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import offline_baseline as baseline  # sets the thread environment variables to 8 before the numeric libraries load

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from agent import building_history
from agent.building_history import FEATURES as HISTORY_FEATURES, HistoryEncoder, history_keys, numeric_value, period_json
from agent.execution import run_python
from agent.history_candidate import POLICIES, fit_history_candidate
from agent.metrics import get_metric, score_predictions


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "regression" / "regression" / "participants"
DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "regression_history"
KEY_COLUMNS = ("State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION")
NUMERIC_KEY_COLUMNS = ("floor_area", "year_built", "ELEVATION")
POSITIVE_KEY_COLUMNS = ("year_built",)
PERIOD = "Year_Factor"
TARGET = "site_eui"
VALIDATION_YEARS = (3, 4, 5)
SEED = 42
THREADS = 8
BLEND_WEIGHT = 0.5
MIN_ROWS_FOR_SUBGROUP_RMSE = 30
EXTRA_TREES = {"n_estimators": 200, "min_samples_leaf": 1, "max_depth": None, "max_features": 1.0}
REFERENCE_RMSE = {"hgb_all": 49.206622, "et_all": 47.789053}  # year 5, given by the team captain
METRIC = "RMSE"  # the declared metric of the public regression task

# Nested mode, fixed before its first run: weight grid with 0 and 0.5, ties go to the smaller weight (closer to the
# model alone), fewer than two periods in an outer training part means weight 0.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TIE_RELATIVE = 1e-12
NESTED_BASE, NESTED_REFERENCE = "et_no_period", "et_all"
FIT_TIMEOUT_SECONDS = 300
RUN_DEADLINE_SECONDS = 7200
MEMORY_LIMIT_BYTES = 8 * 1024 ** 3  # summed RSS of one fit's process tree
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_V2_DIR = ROOT / "data" / "experiments" / "murad-history-v2"

# name, model, period removed from the model features, history features added
VARIANTS = (
    ("hgb_all", "hgb", False, False),
    ("hgb_no_period", "hgb", True, False),
    ("hgb_no_period_history", "hgb", True, True),
    ("et_all", "et", False, False),
    ("et_no_period", "et", True, False),
    ("et_no_period_history", "et", True, True),
)
BLENDS = (("blend_hgb_all", "hgb_all"), ("blend_hgb_no_period", "hgb_no_period"),
          ("blend_et_all", "et_all"), ("blend_et_no_period", "et_no_period"))
SUBSETS = ("all", "history", "no_history", "ambiguous")


# ---------------------------------------------------------------- safety: input files must not change

def protected_inputs(data_dir: Path) -> list[Path]:
    """Input files and nearby submissions, including symbolic links to files outside the data folder."""
    paths = [path for path in Path(data_dir).iterdir()
             if not path.is_dir() and (path.is_file() or path.is_symlink())]
    for folder in (Path(data_dir), Path(data_dir).parent, Path(data_dir).parent.parent):
        candidate = folder / "submission.csv"
        if not candidate.is_dir() and (candidate.is_file() or candidate.is_symlink()) and candidate not in paths:
            paths.append(candidate)
    return sorted(paths)


def file_hashes(data_dir: Path) -> dict[str, str]:
    """SHA-256 of every input file and nearby submission, keyed by its resolved target."""
    result = {}
    for path in protected_inputs(data_dir):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        result[str(path.resolve())] = digest.hexdigest()
    return result


def checked_out_dir(out_dir: Path, data_dir: Path) -> Path:
    """The output folder must not be, or lie inside, the input folder."""
    target, source = Path(out_dir).resolve(), Path(data_dir).resolve()
    if target == source or source in target.parents:
        raise ValueError(f"The output folder must not be inside the input folder: {out_dir}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"The output path is not a folder: {out_dir}")
    return target


def check_output_aliases(out_dir: Path, data_dir: Path, years, prefix: str = "") -> None:
    """Reject every planned destination that aliases an input by path, symlink or hard link, before any fitting."""
    inputs = protected_inputs(data_dir)
    outputs = [out_dir / name for name in (f"{prefix}metrics.json", f"{prefix}metrics.csv", "input_hashes.json")]
    outputs += [out_dir / f"{prefix}predictions_year{year}.csv" for year in years]
    for output in outputs:
        for source in inputs:
            if (output.resolve() == source.resolve()
                    or (output.exists() and source.exists() and os.path.samefile(output, source))):
                raise ValueError(f"The output file must not alias an input file: {output} -> {source}")


def verify_input_hashes(data_dir: Path, before: dict[str, str]) -> dict[str, str]:
    after = file_hashes(data_dir)
    if before != after:
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        raise RuntimeError(f"Input files changed during the run: {changed}")
    return after


def write_file(path: Path, write, binary: bool = False) -> None:
    """Write through a temporary file and replace the target in one step; a failure keeps the old file."""
    handle, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with (os.fdopen(handle, "wb") if binary else os.fdopen(handle, "w", encoding="utf-8", newline="")) as file:
            write(file)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def write_json(path: Path, value) -> None:
    write_file(path, lambda file: file.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))


# ---------------------------------------------------------------- data and models

def load(data_dir: Path):
    """Raw feature values, target and string ids; numeric model conversion must follow exact history encoding."""
    train, _, _, id_column, _, y, features, numeric, categorical, classification = baseline.load_data(Path(data_dir))
    if classification:
        raise ValueError("This experiment is for a regression task")
    missing = [name for name in (*KEY_COLUMNS, PERIOD) if name not in features]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    x = train.loc[:, features].copy()
    ids = train[id_column].astype(str)
    return x, y.to_numpy(float), ids, features, numeric, categorical


def make_model(kind: str, numeric: list[str], categorical: list[str]):
    """The project's pipeline; ExtraTrees replaces only its model step."""
    pipeline = baseline.make_pipeline(numeric, categorical, False)
    if kind == "et":
        pipeline.set_params(model=ExtraTreesRegressor(n_jobs=THREADS, random_state=SEED, **EXTRA_TREES))
    return pipeline


def model_features(spec, features, numeric, categorical):
    _, kind, drop_period, with_history = spec
    names = [name for name in features if not (drop_period and name == PERIOD)]
    numbers = [name for name in numeric if name in names] + (list(HISTORY_FEATURES) if with_history else [])
    return names, numbers, [name for name in categorical if name in names]


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan


def subset_masks(history: pd.DataFrame) -> dict[str, np.ndarray]:
    has = history["history_present"].to_numpy() == 1
    return {"all": np.ones(len(history), dtype=bool), "history": has, "no_history": ~has,
            "ambiguous": has & (history["history_ambiguous"].to_numpy() == 1)}


def error_summary(y: np.ndarray, prediction: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Rows, squared-error sum and RMSE for all rows and each subset (RMSE of a small subset is not reported)."""
    result = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        sse = float(np.square(y[mask] - prediction[mask]).sum())
        small = name != "all" and n < MIN_ROWS_FOR_SUBGROUP_RMSE
        result[name] = {"rows": n, "sse": sse, "rmse": None if n == 0 or small else math.sqrt(sse / n)}
    return result


def history_config(policy: str = "allow") -> dict:
    return {"key_columns": list(KEY_COLUMNS), "period_column": PERIOD, "numeric_columns": list(NUMERIC_KEY_COLUMNS),
            "positive_columns": list(POSITIVE_KEY_COLUMNS), "ambiguity_policy": policy}


def fit_candidate(x: pd.DataFrame, y: np.ndarray, rows: np.ndarray, features, policy: str, weight: float):
    """The history adapter fitted on the given rows only (raw values, before model preparation)."""
    return fit_history_candidate(task_type="regression", metric_name=METRIC, feature_columns=tuple(features),
                                 X_raw=x.loc[rows], y=y[rows], config=history_config(policy), blend_weight=weight)


# ---------------------------------------------------------------- one validation year

def run_year(year: int, x: pd.DataFrame, y: np.ndarray, ids: pd.Series, features, numeric, categorical):
    period = np.asarray([numeric_value(value) for value in x[PERIOD]], dtype=object)
    if any(value is None for value in period):
        raise ValueError(f"Period column {PERIOD!r} must contain finite numeric values")
    fit, val = period < year, period == year
    if not fit.any() or not val.any():
        raise ValueError(f"Year {year}: need rows before it and rows in it")
    y_fit, y_val = y[fit], y[val]
    key_columns = [*KEY_COLUMNS, PERIOD]

    started = time.perf_counter()
    encoder = HistoryEncoder(KEY_COLUMNS, PERIOD, NUMERIC_KEY_COLUMNS, POSITIVE_KEY_COLUMNS)
    history_fit = encoder.fit_transform(x.loc[fit, key_columns], y_fit)
    history_val = encoder.transform(x.loc[val, key_columns])
    history_seconds = time.perf_counter() - started
    masks = subset_masks(history_val)
    # Model preparation may round to float; raw CSV key and period values above must not pass through that path.
    prepared = baseline.prepare_features(x, features, numeric, categorical)
    x_fit, x_val = prepared[fit], prepared[val]

    fold = {"year": year, "fit_rows": int(fit.sum()), "validation_rows": int(val.sum()),
            "fit_years": [period_json(v) for v in sorted(set(period[fit]))], "history_features_seconds": history_seconds,
            "rows_with_history": masks["history"].sum().item(), "share_with_history": float(masks["history"].mean()),
            "rows_ambiguous": masks["ambiguous"].sum().item(),
            "fit_rows_with_history": int(history_fit["history_present"].sum())}
    records, predictions = [], {}
    for spec in VARIANTS:
        name, kind, _, with_history = spec
        names, numbers, texts = model_features(spec, features, numeric, categorical)
        train_part = pd.concat([x_fit[names], history_fit], axis=1) if with_history else x_fit[names]
        valid_part = pd.concat([x_val[names], history_val], axis=1) if with_history else x_val[names]
        model = make_model(kind, numbers, texts)
        with threadpool_limits(limits=THREADS):
            started = time.perf_counter()
            model.fit(train_part, y_fit)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = np.asarray(model.predict(valid_part), dtype=float)
            predict_seconds = time.perf_counter() - started
        predictions[name] = prediction
        records.append({
            "variant": name, "kind": "model", "year": year, "fit_rows": fold["fit_rows"], "validation_rows": fold["validation_rows"],
            "features": [*names, *(HISTORY_FEATURES if with_history else ())], "model": type(model[-1]).__name__,
            "parameters": {key: value for key, value in model[-1].get_params().items()
                           if isinstance(value, (str, int, float, bool, type(None)))},
            "history_features_seconds": history_seconds if with_history else 0.0,
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "errors": error_summary(y_val, prediction, masks)})
    candidate = fit_candidate(x, y, fit, features, "allow", BLEND_WEIGHT)
    for name, source in BLENDS:
        prediction = candidate.apply(x.loc[val], predictions[source]).prediction
        predictions[name] = prediction
        records.append({"variant": name, "kind": "blend", "year": year, "fit_rows": fold["fit_rows"],
                        "validation_rows": fold["validation_rows"], "source": source, "weight_history": BLEND_WEIGHT,
                        "errors": error_summary(y_val, prediction, masks)})

    table = pd.DataFrame({"id": ids[val].to_numpy(), "y_true": y_val, "history_mean": history_val["history_mean"].to_numpy(),
                          "history_rows": history_val["history_rows"].to_numpy(),
                          "history_periods": history_val["history_periods"].to_numpy(),
                          "history_present": history_val["history_present"].to_numpy(),
                          "history_ambiguous": history_val["history_ambiguous"].to_numpy()})
    for name, prediction in predictions.items():
        table[name] = prediction
    return fold, records, table


# ---------------------------------------------------------------- all years, combined metric, files

def combine(records: list[dict]) -> list[dict]:
    """Over all validation years: sqrt(sum(SSE) / sum(N)) per variant and subset (not a mean of yearly RMSE)."""
    result = []
    for name in dict.fromkeys(record["variant"] for record in records):
        own = [record for record in records if record["variant"] == name]
        row = {"variant": name, "years": [record["year"] for record in own]}
        for subset in SUBSETS:
            rows = sum(record["errors"][subset]["rows"] for record in own)
            sse = sum(record["errors"][subset]["sse"] for record in own)
            row[subset] = {"rows": rows, "rmse": math.sqrt(sse / rows) if rows else None}
        result.append(row)
    return result


def flat_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        row = {"variant": record["variant"], "kind": record["kind"], "year": record["year"],
               "fit_rows": record["fit_rows"], "validation_rows": record["validation_rows"],
               "history_features_seconds": record.get("history_features_seconds"),
               "fit_seconds": record.get("fit_seconds"), "predict_seconds": record.get("predict_seconds")}
        for subset in SUBSETS:
            row[f"rows_{subset}"] = record["errors"][subset]["rows"]
            row[f"rmse_{subset}"] = record["errors"][subset]["rmse"]
        rows.append(row)
    return rows


def run_experiment(data_dir: Path, out_dir: Path, years=VALIDATION_YEARS, log=lambda text: None) -> dict:
    data_dir = Path(data_dir)
    years = tuple(years)
    if (not years or any(not isinstance(year, (int, np.integer)) or isinstance(year, bool) for year in years)
            or len(set(years)) != len(years)):
        raise ValueError("Validation years must be a non-empty sequence of distinct integers")
    years = tuple(int(year) for year in years)
    out_dir = checked_out_dir(out_dir, data_dir)
    check_output_aliases(out_dir, data_dir, years)
    before = file_hashes(data_dir)
    x, y, ids, features, numeric, categorical = load(data_dir)
    folds, records, tables = [], [], {}
    for year in years:
        log(f"year {year}: fitting the {len(VARIANTS)} variants")
        fold, year_records, table = run_year(year, x, y, ids, features, numeric, categorical)
        folds.append(fold)
        records += year_records
        tables[year] = table
    after = verify_input_hashes(data_dir, before)
    check_output_aliases(out_dir, data_dir, years)

    summary = {"command": "python experiment_regression_history.py", "seed": SEED, "threads": THREADS,
               "blend_weight": BLEND_WEIGHT, "validation_years": list(years), "key_columns": list(KEY_COLUMNS),
               "period_column": PERIOD, "extra_trees": EXTRA_TREES,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__},
               "folds": folds, "records": records, "combined": combine(records),
               "reference_year_5": {name: {"given": value, "measured": next(
                   (r["errors"]["all"]["rmse"] for r in records if r["variant"] == name and r["year"] == 5), None)}
                   for name, value in REFERENCE_RMSE.items()},
               "input_hashes": {"before": before, "after": after, "unchanged": True}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, table in tables.items():  # ids stay text: read them back with dtype=str
        write_file(out_dir / f"predictions_year{year}.csv",
                   lambda file, table=table: table.to_csv(file, index=False, lineterminator="\n"))
    write_file(out_dir / "metrics.json", lambda file: file.write(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))
    write_file(out_dir / "metrics.csv", write_metrics_csv(records))
    write_file(out_dir / "input_hashes.json", lambda file: file.write(
        json.dumps(summary["input_hashes"], indent=2) + "\n"))
    # The serialized hash values remain valid only if this check succeeds after every output has been written.
    try:
        verify_input_hashes(data_dir, before)
    except BaseException:
        # These success manifests were produced by this run; do not leave a false unchanged=true claim behind.
        for name in ("metrics.json", "input_hashes.json"):
            with contextlib.suppress(OSError):
                (out_dir / name).unlink()
        raise
    return summary


def write_metrics_csv(records: list[dict]):
    rows = flat_rows(records)

    def write(file) -> None:
        writer = csv.DictWriter(file, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return write


def render(summary: dict) -> str:
    def number(value):
        return "-" if value is None else f"{value:.4f}"

    years = summary["validation_years"]
    lines = ["Validation RMSE (lower is better); combined = sqrt(sum SSE / sum N) over the years",
             f"{'variant':<24}" + "".join(f"{'year ' + str(y):>10}" for y in years) + f"{'combined':>10}{'hist rows':>11}{'no hist':>10}"]
    combined = {row["variant"]: row for row in summary["combined"]}
    for name in dict.fromkeys(record["variant"] for record in summary["records"]):
        yearly = {record["year"]: record for record in summary["records"] if record["variant"] == name}
        lines.append(f"{name:<24}" + "".join(f"{number(yearly[y]['errors']['all']['rmse']):>10}" for y in years)
                     + f"{number(combined[name]['all']['rmse']):>10}{number(combined[name]['history']['rmse']):>11}"
                     f"{number(combined[name]['no_history']['rmse']):>10}")
    lines.append("")
    for fold in summary["folds"]:
        lines.append(f"year {fold['year']}: fit {fold['fit_rows']} rows, validation {fold['validation_rows']} rows, "
                     f"with history {fold['rows_with_history']} ({100 * fold['share_with_history']:.1f}%), "
                     f"ambiguous {fold['rows_ambiguous']}, history features {fold['history_features_seconds']:.2f} s")
    reference = summary["reference_year_5"]
    lines.append("reference (year 5): " + ", ".join(
        f"{name} given {ref['given']:.6f} measured {number(ref['measured'])}" for name, ref in reference.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- nested mode: the weight is chosen inside each training part

RECIPES = ("et_all", "et_no_period", "fixed_0.5_allow", "fixed_0.5_past_unique_only", "nested_allow",
           "nested_past_unique_only", "control_shuffled_0.5_allow")


def shuffled_within_periods(y: np.ndarray, period: np.ndarray, rows: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Answers of the given rows permuted inside each period: the period's values stay, their link to keys is gone."""
    rng = np.random.default_rng(seed)
    result = y.copy()
    for value in sorted(set(period[rows])):
        positions = np.flatnonzero(rows & (period == value))
        result[positions] = y[rng.permutation(positions)]
    return result


def exact_periods(x: pd.DataFrame) -> np.ndarray:
    period = np.asarray([numeric_value(value) for value in x[PERIOD]], dtype=object)
    if any(value is None for value in period):
        raise ValueError(f"Period column {PERIOD!r} must contain finite numeric values")
    return period


def select_weight(scores: dict, higher_is_better: bool) -> float:
    """Best inner score; scores within TIE_RELATIVE of the best are equal and the smallest of their weights wins."""
    best = max(scores.values()) if higher_is_better else min(scores.values())
    tolerance = TIE_RELATIVE * max(1.0, abs(best))
    return min(weight for weight, score in scores.items() if abs(score - best) <= tolerance)


def choose_weights(x, y, period, outer_period, features, fit_base):
    """Sequential inner splits inside the periods before outer_period; per policy the weight with the best pooled
    inner score of the task metric. Every inner split refits the base model and the adapter on its own past only."""
    metric = get_metric(METRIC)
    inner = []
    for inner_period in sorted(set(period[period < outer_period]))[1:]:
        fit_rows, val_rows = period < inner_period, period == inner_period
        base = fit_base(NESTED_BASE, fit_rows, val_rows,
                        f"outer {period_json(outer_period)}, inner {period_json(inner_period)}")
        if base is None:
            return None, []
        inner.append((inner_period, fit_rows, val_rows, base))
    splits = [{"period": period_json(p), "fit_periods": [period_json(v) for v in sorted(set(period[f]))],
               "fit_rows": int(f.sum()), "validation_rows": int(v.sum())} for p, f, v, _ in inner]
    selections = {}
    for policy in POLICIES:
        if not inner:
            selections[policy] = {"weight": 0.0, "scores": {},
                                  "reason": "fewer than two periods before the outer period: history off"}
            continue
        scores = {}
        for weight in WEIGHT_GRID:
            predictions = [fit_candidate(x, y, f, features, policy, weight).apply(x.loc[v], base).prediction
                           for _, f, v, base in inner]
            scores[weight] = score_predictions(metric, np.concatenate([y[v] for _, _, v, _ in inner]),
                                               np.concatenate(predictions))
        selections[policy] = {"weight": select_weight(scores, metric.higher_is_better),
                              "scores": {str(weight): score for weight, score in scores.items()},
                              "reason": f"best pooled inner {metric.name}"}
    return selections, splits


def history_masks(x, y, fit_rows, val_rows, features) -> dict[str, np.ndarray]:
    """Rows with past history and rows whose past history is ambiguous, as the adapter sees them."""
    zero = np.zeros(int(val_rows.sum()))
    has = fit_candidate(x, y, fit_rows, features, "allow", 1.0).apply(x.loc[val_rows], zero).applied_mask
    unique = fit_candidate(x, y, fit_rows, features, "past_unique_only", 1.0).apply(x.loc[val_rows], zero).applied_mask
    return {"all": np.ones(len(zero), dtype=bool), "history": has, "no_history": ~has, "ambiguous": has & ~unique}


def nested_outer(outer_period, x, y, period, ids, features, fit_base):
    """One outer period: choose weights inside the earlier periods, refit on all of them, score the outer rows once."""
    fit_rows, val_rows = period < outer_period, period == outer_period
    started = time.perf_counter()
    result = {"year": period_json(outer_period), "fit_rows": int(fit_rows.sum()), "validation_rows": int(val_rows.sum()),
              "fit_periods": [period_json(v) for v in sorted(set(period[fit_rows]))]}
    if not fit_rows.any() or not val_rows.any():
        raise ValueError(f"Outer period {outer_period}: need rows before it and rows in it")
    selections, result["inner_splits"] = choose_weights(x, y, period, outer_period, features, fit_base)
    result["selection"] = selections
    base = reference = None
    if selections is not None:
        base = fit_base(NESTED_BASE, fit_rows, val_rows, f"outer {result['year']}")
        reference = fit_base(NESTED_REFERENCE, fit_rows, val_rows, f"outer {result['year']}") if base is not None else None
    if reference is None:
        result.update(status="incomplete: a fit failed or the deadline was reached", seconds=time.perf_counter() - started)
        return result, None
    predictions, applications = {NESTED_REFERENCE: reference, NESTED_BASE: base}, {}
    for policy in POLICIES:
        for name, weight in ((f"fixed_0.5_{policy}", 0.5), (f"nested_{policy}", selections[policy]["weight"])):
            application = fit_candidate(x, y, fit_rows, features, policy, weight).apply(x.loc[val_rows], base)
            predictions[name] = application.prediction
            applications[name] = {"weight": weight, **{key: value for key, value in application.diagnostics.items()
                                                        if key.startswith(("rows", "share"))}}
    # Control, not a recipe: the same blend after the fitting answers lost their link to the keys. What remains of the
    # gain here is shrinkage towards earlier values of the period, not knowledge of the object.
    shuffled = shuffled_within_periods(y, period, fit_rows)
    predictions["control_shuffled_0.5_allow"] = fit_candidate(x, shuffled, fit_rows, features, "allow", 0.5).apply(
        x.loc[val_rows], base).prediction
    masks = history_masks(x, y, fit_rows, val_rows, features)
    y_val = y[val_rows]
    result["records"] = [{"variant": name, "kind": "model" if name in (NESTED_BASE, NESTED_REFERENCE) else "adapter",
                          "year": result["year"], "fit_rows": result["fit_rows"], "validation_rows": result["validation_rows"],
                          "errors": error_summary(y_val, predictions[name], masks)} for name in RECIPES]
    result.update(status="complete", applications=applications, seconds=time.perf_counter() - started)
    table = pd.DataFrame({"id": ids[val_rows].to_numpy(), "y_true": y_val, "history": masks["history"].astype(int),
                          "ambiguous": masks["ambiguous"].astype(int)})
    for name in RECIPES:
        table[name] = predictions[name]
    return result, table


def cluster_ids(keys: list) -> np.ndarray:
    """One cluster per strict key, so a key's rows of every outer year stay together; a row without a key is alone."""
    mapping = {}
    return np.asarray([mapping.setdefault(key if key is not None else object(), len(mapping)) for key in keys],
                      dtype=np.int64)


def paired_bootstrap(actual, predictions: dict, baseline: str, clusters, resamples: int, seed: int = SEED) -> dict:
    """RMSE(recipe) - RMSE(baseline) with a 95% percentile interval; all recipes share the resampled clusters.

    Negative differences favour the recipe. The interval is conditional on these predictions: it ignores refitting
    noise, the choice of the evaluated years and the recipes already tried on them.
    """
    names = list(predictions)
    _, compact = np.unique(clusters, return_inverse=True)
    size = int(compact.max()) + 1
    sse = np.stack([np.bincount(compact, weights=(actual - predictions[name]) ** 2, minlength=size) for name in names], 1)
    count = np.bincount(compact, minlength=size).astype(float)
    point = np.sqrt(sse.sum(axis=0) / count.sum())
    base = names.index(baseline)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(resamples, 2_000_000 // size))
    differences = []
    for start in range(0, resamples, chunk):
        draws = rng.integers(0, size, size=(min(chunk, resamples - start), size))
        weights = np.stack([np.bincount(row, minlength=size) for row in draws]).astype(float)
        rmse = np.sqrt((weights @ sse) / (weights @ count)[:, None])
        differences.append(rmse - rmse[:, [base]])
    differences = np.concatenate(differences)
    return {"clusters": size, "rows": int(count.sum()), "resamples": resamples, "baseline": baseline,
            "recipes": {name: {"rmse": float(point[i]), "difference": float(point[i] - point[base]),
                               "interval_95": [float(v) for v in np.percentile(differences[:, i], [2.5, 97.5])],
                               "share_of_resamples_better": float(np.mean(differences[:, i] < 0))}
                        for i, name in enumerate(names) if i != base}}


def save_array(path: Path, array: np.ndarray) -> None:
    write_file(path, lambda file: np.save(file, array, allow_pickle=False), binary=True)


def bounded_call(work_dir: Path, step: int, function: str, spec: dict, timeout: float, memory_limit: int) -> dict:
    """Run experiment.<function>(spec file) in a child process with the project's time and process-tree RSS limits."""
    spec_path = work_dir / f"call_{step:03d}.json"
    write_json(spec_path, spec)
    code = f"import experiment_regression_history as experiment\nexperiment.{function}({str(spec_path)!r})\n"
    return run_python(code, ROOT, work_dir, step, timeout=timeout, max_memory_bytes=memory_limit)


def call_failed(result: dict) -> str | None:
    if result.get("error"):
        return result["error"]
    if result["timed_out"]:
        return "timeout"
    if result["memory_limit_exceeded"]:
        return "memory limit exceeded"
    if result["exit_code"] != 0:
        return f"exit code {result['exit_code']}"
    return None


class BoundedFits:
    """Each base model is fitted in its own bounded process, one after another, never past the run deadline."""

    def __init__(self, data_dir: Path, rows: int, work_dir: Path, deadline: float, fit_timeout: float,
                 memory_limit: int, log):
        self.data_dir, self.rows, self.work_dir = Path(data_dir).resolve(), rows, work_dir
        self.deadline, self.fit_timeout, self.memory_limit, self.log = deadline, fit_timeout, memory_limit, log
        self.records = []

    def __call__(self, model: str, fit_rows: np.ndarray, predict_rows: np.ndarray, label: str):
        step = len(self.records) + 1
        record = {"step": step, "label": label, "model": model, "fit_rows": int(fit_rows.sum()),
                  "predict_rows": int(predict_rows.sum())}
        self.records.append(record)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            record["status"] = "skipped: run deadline reached"
            return None
        prefix = self.work_dir / f"fit_{step:03d}"
        files = {name: f"{prefix}_{name}.npy" for name in ("fit_positions", "predict_positions", "prediction")}
        save_array(Path(files["fit_positions"]), np.flatnonzero(fit_rows))
        save_array(Path(files["predict_positions"]), np.flatnonzero(predict_rows))
        self.log(f"  fit {step}: {label}, {model} on {record['fit_rows']} rows")
        result = bounded_call(self.work_dir, step, "fit_worker",
                              {"data_dir": str(self.data_dir), "rows": self.rows, "model": model, **files,
                               "timing": f"{prefix}_timing.json"},
                              min(self.fit_timeout, remaining), self.memory_limit)
        resources = result.get("resources") or {}
        record.update(elapsed_seconds=result.get("elapsed_seconds"), peak_rss_bytes=resources.get("peak_rss_bytes"),
                      cpu_seconds=resources.get("cpu_seconds"))
        failure = call_failed(result)
        prediction = None if failure else np.load(files["prediction"], allow_pickle=False)
        if not failure and (prediction.shape != (record["predict_rows"],) or not np.isfinite(prediction).all()):
            failure = "prediction has the wrong shape or non-finite values"
        if failure:
            record.update(status=f"failed: {failure}", stderr_tail=(result.get("stderr") or "")[-3000:])
            self.log(f"  fit {step} failed: {failure}")
            return None
        record.update(json.loads(Path(f"{prefix}_timing.json").read_text(encoding="utf-8")), status="complete")
        return prediction


def fit_worker(spec_path: str) -> None:
    """Child process of BoundedFits: fit one base model on the given rows, save its predictions and timings."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    x, y, _, features, numeric, categorical = load(Path(spec["data_dir"]))
    if len(x) != spec["rows"]:
        raise ValueError("The training data changed after the experiment started")
    variant = next(item for item in VARIANTS if item[0] == spec["model"])
    if variant[3]:
        raise ValueError("The nested base models take no history features")
    fit_rows = np.load(spec["fit_positions"], allow_pickle=False)
    predict_rows = np.load(spec["predict_positions"], allow_pickle=False)
    names, numbers, texts = model_features(variant, features, numeric, categorical)
    prepared = baseline.prepare_features(x, features, numeric, categorical)
    model = make_model(variant[1], numbers, texts)
    with threadpool_limits(limits=THREADS):
        started = time.perf_counter()
        model.fit(prepared.iloc[fit_rows][names], y[fit_rows])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = np.asarray(model.predict(prepared.iloc[predict_rows][names]), dtype=float)
        predict_seconds = time.perf_counter() - started
    save_array(Path(spec["prediction"]), prediction)
    write_json(Path(spec["timing"]), {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                                      "features": names, "model": type(model[-1]).__name__})


def summarize_recipes(results: list[dict], tables: dict, keys: dict, resamples: int) -> dict:
    """Combined errors and paired bootstrap intervals over the complete outer periods."""
    complete = [result for result in results if result["status"] == "complete"]
    if not complete:
        return {"complete_years": [], "combined": [], "bootstrap": {}}
    years = [result["year"] for result in complete]
    table = pd.concat([tables[year] for year in years], ignore_index=True)
    clusters = cluster_ids([key for year in years for key in keys[year]])
    actual = table["y_true"].to_numpy(float)
    history = table["history"].to_numpy() == 1
    predictions = {name: table[name].to_numpy(float) for name in RECIPES}
    bootstrap = {}
    for subset, mask in (("all", np.ones(len(table), dtype=bool)), ("history", history)):
        if not mask.any():
            continue
        part = {name: value[mask] for name, value in predictions.items()}
        bootstrap[subset] = {
            "key_clusters": paired_bootstrap(actual[mask], part, NESTED_BASE, clusters[mask], resamples),
            "rows_sensitivity": paired_bootstrap(actual[mask], part, NESTED_BASE, np.arange(int(mask.sum())), resamples)}
    return {"complete_years": years, "combined": combine([r for result in complete for r in result["records"]]),
            "bootstrap": bootstrap}


def machine_memory_limit(requested: int) -> tuple[int, dict]:
    import psutil

    memory = psutil.virtual_memory()
    limit = min(requested, memory.total)
    return limit, {"requested_bytes": requested, "used_limit_bytes": limit, "machine_total_bytes": memory.total,
                   "available_at_start_bytes": memory.available,
                   "note": None if limit == requested else "lowered to the machine's physical memory"}


def run_nested(data_dir: Path, out_dir: Path, years=VALIDATION_YEARS, *, fit_timeout: float = FIT_TIMEOUT_SECONDS,
               deadline_seconds: float = RUN_DEADLINE_SECONDS, memory_limit: int = MEMORY_LIMIT_BYTES,
               resamples: int = BOOTSTRAP_RESAMPLES, log=lambda text: None) -> dict:
    data_dir, years = Path(data_dir), tuple(years)
    if (not years or any(not isinstance(year, (int, np.integer)) or isinstance(year, bool) for year in years)
            or len(set(years)) != len(years)):
        raise ValueError("Outer periods must be a non-empty sequence of distinct integers")
    years = tuple(int(year) for year in years)
    out_dir = checked_out_dir(out_dir, data_dir)
    check_output_aliases(out_dir, data_dir, years, prefix="nested_")
    started = time.monotonic()
    limit, memory = machine_memory_limit(memory_limit)
    before = file_hashes(data_dir)
    x, y, ids, features, _, _ = load(data_dir)
    period = exact_periods(x)
    work_dir = out_dir / "fits"
    work_dir.mkdir(parents=True, exist_ok=True)
    fits = BoundedFits(data_dir, len(x), work_dir, started + deadline_seconds, fit_timeout, limit, log)
    results, tables, keys = [], {}, {}
    for year in years:
        log(f"outer year {year}")
        result, table = nested_outer(year, x, y, period, ids, features, fits)
        results.append(result)
        if table is not None:
            tables[result["year"]] = table
            keys[result["year"]] = history_keys(x.loc[period == year], KEY_COLUMNS, NUMERIC_KEY_COLUMNS,
                                                POSITIVE_KEY_COLUMNS)
    summary = {"command": "python experiment_regression_history.py --mode nested", "metric": METRIC, "seed": SEED,
               "threads": THREADS, "weight_grid": list(WEIGHT_GRID), "tie_rule": f"relative {TIE_RELATIVE}, smaller weight",
               "base_model": NESTED_BASE, "reference_model": NESTED_REFERENCE, "extra_trees": EXTRA_TREES,
               "history_config": history_config(), "outer_years": list(years),
               "limits": {"fit_timeout_seconds": fit_timeout, "run_deadline_seconds": deadline_seconds, "memory": memory},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__},
               "outer": [{key: value for key, value in result.items() if key != "records"} for result in results],
               "records": [record for result in results for record in result.get("records", [])],
               "fits": fits.records, **summarize_recipes(results, tables, keys, resamples),
               "seconds": time.monotonic() - started}
    after = verify_input_hashes(data_dir, before)
    summary["input_hashes"] = {"before": before, "after": after, "unchanged": True}
    for year, table in tables.items():
        write_file(out_dir / f"nested_predictions_year{year}.csv",
                   lambda file, table=table: table.to_csv(file, index=False, lineterminator="\n"))
    write_json(out_dir / "nested_metrics.json", summary)
    if summary["records"]:
        write_file(out_dir / "nested_metrics.csv", write_metrics_csv(summary["records"]))
    write_json(out_dir / "input_hashes.json", summary["input_hashes"])
    try:
        verify_input_hashes(data_dir, before)
    except BaseException:
        for name in ("nested_metrics.json", "input_hashes.json"):
            with contextlib.suppress(OSError):
                (out_dir / name).unlink()
        raise
    return summary


def render_nested(summary: dict) -> str:
    def number(value, digits=4):
        return "-" if value is None else f"{value:.{digits}f}"

    years = summary["complete_years"]
    lines = [f"Nested history weights, metric {summary['metric']} (lower is better), complete outer years {years}",
             f"{'recipe':<28}" + "".join(f"{'year ' + str(y):>10}" for y in years)
             + f"{'combined':>10}{'history':>10}{'no hist':>10}{'ambig.':>10}"]
    combined = {row["variant"]: row for row in summary["combined"]}
    for name in RECIPES:
        if name not in combined:
            continue
        yearly = {r["year"]: r for r in summary["records"] if r["variant"] == name}
        row = combined[name]
        lines.append(f"{name:<28}" + "".join(f"{number(yearly[y]['errors']['all']['rmse']):>10}" for y in years)
                     + "".join(f"{number(row[s]['rmse']):>10}" for s in ("all", "history", "no_history", "ambiguous")))
    lines.append("")
    for outer in summary["outer"]:
        weights = {policy: (item or {}).get("weight") for policy, item in (outer.get("selection") or {}).items()}
        lines.append(f"year {outer['year']}: {outer['status']}; fit {outer['fit_rows']} rows, validation "
                     f"{outer['validation_rows']}; inner periods {[s['period'] for s in outer['inner_splits']]}; "
                     f"chosen weights {weights}")
    for subset, parts in summary.get("bootstrap", {}).items():
        boot = parts["key_clusters"]
        lines.append(f"paired key-cluster bootstrap, {subset} rows ({boot['rows']} rows, {boot['clusters']} clusters), "
                     f"RMSE - RMSE({boot['baseline']}), 95% interval:")
        for name, item in boot["recipes"].items():
            low, high = item["interval_95"]
            lines.append(f"  {name:<28}{item['difference']:>+9.4f}  [{low:+.4f}; {high:+.4f}]")
    failed = [fit for fit in summary["fits"] if fit.get("status") != "complete"]
    lines.append(f"fits: {len(summary['fits'])}, not complete: {len(failed)}; "
                 f"peak RSS of one fit {max((f.get('peak_rss_bytes') or 0) for f in summary['fits']) / 2**30:.2f} GiB; "
                 f"total {summary['seconds']:.0f} s")
    return "\n".join(lines)


# ---------------------------------------------------------------- synthetic controls and the cost of the encoder

def synthetic_task(seed: int, persistent: bool, objects: int = 400, periods: int = 5):
    """Regression rows shaped like the public task. persistent: each object keeps its level in every period (history
    carries a real signal); otherwise the level is drawn anew per period, so past answers of a key say nothing."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "State_Factor": rng.choice(["State_1", "State_2"], objects), "building_class": rng.choice(["Commercial", "Residential"], objects),
        "facility_type": rng.choice(["Office", "Retail", "School"], objects),
        "floor_area": [str(5000 + 37.5 * i) for i in range(objects)], "year_built": [str(1940 + i % 70) for i in range(objects)],
        "ELEVATION": [str((i % 6) * 1.5) for i in range(objects)]})
    data.loc[:9, "year_built"] = ""  # unknown year_built: no reliable key
    data.loc[11, list(KEY_COLUMNS)] = data.loc[10, list(KEY_COLUMNS)].to_numpy()  # two objects share one key
    level = rng.uniform(20, 200, objects)
    parts = []
    for number in range(1, periods + 1):
        present = rng.random(objects) < 0.75
        part = data[present].copy()
        temp = rng.normal(size=len(part))
        part[PERIOD], part["temp"] = str(number), [repr(value) for value in temp]
        current = level[present] if persistent else rng.uniform(20, 200, len(part))
        part[TARGET] = current + 5 * temp + rng.normal(scale=5, size=len(part))
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    features = [*KEY_COLUMNS, PERIOD, "temp"]
    numeric, categorical = ["floor_area", "year_built", "ELEVATION", PERIOD, "temp"], list(KEY_COLUMNS[:3])
    ids = pd.Series([f"{i:06d}" for i in range(len(rows))])
    return rows[features], rows[TARGET].to_numpy(float), ids, features, numeric, categorical


def in_process_fits(x, y, features, numeric, categorical, deadline: float):
    """Base model fits for the synthetic controls; the whole controls run is one bounded child process."""
    prepared = baseline.prepare_features(x, features, numeric, categorical)

    def fit_base(model, fit_rows, predict_rows, label):
        if time.monotonic() > deadline:
            return None
        variant = next(item for item in VARIANTS if item[0] == model)
        names, numbers, texts = model_features(variant, features, numeric, categorical)
        pipeline = make_model(variant[1], numbers, texts)
        with threadpool_limits(limits=THREADS):
            pipeline.fit(prepared.loc[fit_rows, names], y[fit_rows])
            return np.asarray(pipeline.predict(prepared.loc[predict_rows, names]), dtype=float)

    return fit_base


def controls_worker(spec_path: str) -> None:
    """Positive control (persistent object level) and null control (no persistent level), nested exactly as above."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    deadline = time.monotonic() + spec["seconds"]
    report = {}
    for name, persistent in (("positive", True), ("null", False)):
        x, y, ids, features, numeric, categorical = synthetic_task(spec["seed"], persistent)
        period = exact_periods(x)
        fit_base = in_process_fits(x, y, features, numeric, categorical, deadline)
        results, tables, keys = [], {}, {}
        for year in VALIDATION_YEARS:
            result, table = nested_outer(year, x, y, period, ids, features, fit_base)
            results.append(result)
            if table is not None:
                tables[result["year"]] = table
                keys[result["year"]] = history_keys(x.loc[period == year], KEY_COLUMNS, NUMERIC_KEY_COLUMNS,
                                                    POSITIVE_KEY_COLUMNS)
        report[name] = {"rows": len(x), "outer": [{k: v for k, v in r.items() if k != "records"} for r in results],
                        "records": [record for r in results for record in r.get("records", [])],
                        **summarize_recipes(results, tables, keys, spec["resamples"])}
    write_json(Path(spec["output"]), report)


def _quadratic_sums(partials: list, value: float) -> None:
    partials.append(value)  # with _rounded(partials) this is the former fit: math.fsum over all earlier totals


def scaling_worker(spec_path: str) -> None:
    """Time and peak traced memory of HistoryEncoder.fit for one key with a growing history, the former quadratic
    summation alongside while it stays within the budget, and the encoder on the official data."""
    import tracemalloc

    import psutil

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    deadline = time.monotonic() + spec["seconds"]
    official = {}
    if spec.get("data_dir"):
        train, test, *_ = baseline.load_data(Path(spec["data_dir"]))
        columns = [*KEY_COLUMNS, PERIOD]
        y = pd.to_numeric(train[TARGET]).to_numpy(float)
        make = lambda: HistoryEncoder(KEY_COLUMNS, PERIOD, NUMERIC_KEY_COLUMNS, POSITIVE_KEY_COLUMNS)
        process = psutil.Process()
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        fitted = make().fit(train[columns], y)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fitted.transform(test[columns])
        transform_seconds = time.perf_counter() - started
        rss_after = process.memory_info().rss
        tracemalloc.start()
        make().fit(train[columns], y).transform(test[columns])
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        official = {"train_rows": len(train), "test_rows": len(test), "fit_seconds": fit_seconds,
                    "transform_test_seconds": transform_seconds, "traced_peak_bytes": peak,
                    "rss_growth_bytes": rss_after - rss_before, "keys": len(fitted._past),
                    "longest_history_periods": max(len(entry.periods) for entry in fitted._past.values())}

    rng = np.random.default_rng(spec["seed"])
    encoder = lambda: HistoryEncoder(["key"], "period")

    def timed_fit(frame, answers, summation=None):
        original = building_history._add_exact
        building_history._add_exact = summation or original
        try:
            started = time.perf_counter()
            fitted = encoder().fit(frame, answers)
            return time.perf_counter() - started, fitted
        finally:
            building_history._add_exact = original

    rows, old_allowed, size = [], True, spec["start"]
    while size <= spec["largest"] and time.monotonic() < deadline:
        values = rng.choice([1e16, -1e16, 1.0, 3.3, -7.25e15], size=size) * rng.uniform(0.5, 1.5, size)
        frame = pd.DataFrame({"key": "k", "period": np.arange(1, size + 1)})
        seconds, fitted = timed_fit(frame, values)
        tracemalloc.start()
        encoder().fit(frame, values)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        means = fitted.transform(frame)["history_mean"].to_numpy()[1:]
        naive = np.cumsum(values)[:-1] / np.arange(1, size)
        row = {"periods_of_one_key": size, "fit_seconds": seconds, "traced_peak_bytes": peak,
               "max_relative_error_of_plain_cumsum": float(np.max(np.abs(naive - means) / np.maximum(np.abs(means), 1e-300)))}
        if old_allowed:
            old_seconds, old = timed_fit(frame, values, _quadratic_sums)
            row.update(former_fit_seconds=old_seconds, max_abs_difference_to_former=float(np.max(np.abs(
                old.transform(frame)["history_mean"].to_numpy()[1:] - means))))
            old_allowed = old_seconds * 4.5 < min(spec["former_limit_seconds"], deadline - time.monotonic())
        rows.append(row)
        if seconds * 12 > deadline - time.monotonic():  # the next size costs about 2x for each of its fits
            break
        size *= 2
    for previous, current in zip(rows, rows[1:]):
        current["time_ratio_to_half_size"] = current["fit_seconds"] / previous["fit_seconds"]
        if "former_fit_seconds" in current and "former_fit_seconds" in previous:
            current["former_time_ratio_to_half_size"] = current["former_fit_seconds"] / previous["former_fit_seconds"]
    write_json(Path(spec["output"]), {"synthetic_one_key": rows, "official_data": official})


def run_child_mode(mode: str, data_dir: Path, out_dir: Path, *, seconds: float, memory_limit: int,
                   resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """controls or scaling: one bounded child process; its JSON report is copied into the summary."""
    out_dir = checked_out_dir(out_dir, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit, memory = machine_memory_limit(memory_limit)
    before = file_hashes(data_dir) if Path(data_dir).is_dir() else {}
    output = out_dir / f"{mode}_report.json"
    spec = {"seed": SEED, "seconds": seconds * 0.9, "output": str(output.resolve())}
    if mode == "controls":
        spec["resamples"] = resamples
    else:
        spec.update(start=1000, largest=1_024_000, former_limit_seconds=120, data_dir=str(Path(data_dir).resolve()))
    result = bounded_call(out_dir, 1, f"{mode}_worker", spec, seconds, limit)
    failure = call_failed(result)
    summary = {"mode": mode, "command": f"python experiment_regression_history.py --mode {mode}",
               "limits": {"seconds": seconds, "memory": memory}, "elapsed_seconds": result.get("elapsed_seconds"),
               "resources": result.get("resources"), "status": f"failed: {failure}" if failure else "complete",
               "stderr_tail": (result.get("stderr") or "")[-3000:] if failure else None,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__}}
    if not failure:
        summary["report"] = json.loads(output.read_text(encoding="utf-8"))
    if before:
        summary["input_hashes"] = {"before": before, "after": verify_input_hashes(data_dir, before), "unchanged": True}
    write_json(out_dir / f"{mode}_summary.json", summary)
    return summary


def render_child(summary: dict) -> str:
    lines = [f"{summary['mode']}: {summary['status']} in {summary['elapsed_seconds']:.0f} s"]
    report = summary.get("report") or {}
    if summary["mode"] == "controls":
        for name, part in report.items():
            combined = {row["variant"]: row["all"]["rmse"] for row in part["combined"]}
            weights = [{p: s["weight"] for p, s in (o.get("selection") or {}).items()} for o in part["outer"]]
            boot = part["bootstrap"]["all"]["key_clusters"]["recipes"]
            lines.append(f"{name}: chosen weights per outer year {weights}")
            lines.append("  combined RMSE " + ", ".join(f"{k} {v:.3f}" for k, v in combined.items()))
            lines.append("  vs et_no_period, 95% key-cluster interval: " + ", ".join(
                f"{k} {v['difference']:+.3f} [{v['interval_95'][0]:+.3f}; {v['interval_95'][1]:+.3f}]"
                for k, v in boot.items() if k.startswith(("fixed", "nested"))))
    else:
        for row in report.get("synthetic_one_key", []):
            lines.append(f"periods {row['periods_of_one_key']:>8}: fit {row['fit_seconds']:.3f} s "
                         f"(x{row.get('time_ratio_to_half_size', float('nan')):.2f}), traced peak "
                         f"{row['traced_peak_bytes'] / 2**20:.1f} MiB" + (
                             f", former {row['former_fit_seconds']:.3f} s (x{row.get('former_time_ratio_to_half_size', float('nan')):.2f}),"
                             f" max diff {row['max_abs_difference_to_former']:.1e}" if "former_fit_seconds" in row else ""))
        official = report.get("official_data") or {}
        if official:
            lines.append(f"official data: fit {official['fit_seconds']:.2f} s on {official['train_rows']} rows, "
                         f"transform test {official['transform_test_seconds']:.2f} s, traced peak "
                         f"{official['traced_peak_bytes'] / 2**20:.1f} MiB, longest history {official['longest_history_periods']} periods")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("matrix", "nested", "controls", "scaling"), default="matrix")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, help=f"default: {DEFAULT_OUT_DIR} (matrix), {DEFAULT_V2_DIR}/<mode>")
    parser.add_argument("--fit-timeout", type=float, default=FIT_TIMEOUT_SECONDS, help="seconds per fit (nested)")
    parser.add_argument("--deadline", type=float, default=RUN_DEADLINE_SECONDS, help="seconds for the whole run")
    parser.add_argument("--memory-gib", type=float, default=MEMORY_LIMIT_BYTES / 2**30, help="RSS limit of a fit")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out_dir = args.out_dir or (DEFAULT_OUT_DIR if args.mode == "matrix" else DEFAULT_V2_DIR / args.mode)
    log = lambda text: print(text, flush=True)
    try:
        if min(args.fit_timeout, args.deadline, args.memory_gib) <= 0 or args.bootstrap < 1:
            raise ValueError("Limits and the number of bootstrap resamples must be positive")
        started = time.perf_counter()
        memory = int(args.memory_gib * 2**30)
        if args.mode == "matrix":
            print(render(run_experiment(args.data_dir, out_dir, log=log)))
        elif args.mode == "nested":
            print(render_nested(run_nested(args.data_dir, out_dir, fit_timeout=args.fit_timeout,
                                           deadline_seconds=args.deadline, memory_limit=memory,
                                           resamples=args.bootstrap, log=log)))
        else:
            summary = run_child_mode(args.mode, args.data_dir, out_dir, seconds=min(args.deadline, 1800),
                                     memory_limit=memory, resamples=args.bootstrap)
            print(render_child(summary))
            if summary["status"] != "complete":
                print(summary["stderr_tail"], file=sys.stderr)
                return 1
        print(f"Done in {time.perf_counter() - started:.0f} s. Results: {Path(out_dir).resolve()}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
