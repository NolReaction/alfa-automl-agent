"""Does the building history help the regression models? A fixed matrix of experiments, no search.

For each validation year (3, 4, 5) the models are fitted on the rows of all earlier years and checked on that year;
every variant sees exactly the same fitting and validation rows. Six trained variants:

    1 HGB, all features                      4 ExtraTrees, all features
    2 HGB without Year_Factor                5 ExtraTrees without Year_Factor
    3 HGB without Year_Factor + history      6 ExtraTrees without Year_Factor + history

and four fixed blends of variants 1, 2, 4, 5: where the key has history 0.5 * model + 0.5 * history mean, elsewhere
the model. The weight 0.5 is fixed beforehand and never tuned. The history features come from a new HistoryEncoder per
year (fit_transform on the fitting rows, transform on the validation rows) and are computed before Year_Factor is
removed from the model features. Models are the project's offline_baseline pipeline; only the model step of the
ExtraTrees variants is replaced.

    python experiment_regression_history.py [--data-dir DIR] [--out-dir DIR]

Results go to the output folder (default data/experiments/regression_history, ignored by Git): metrics.json,
metrics.csv, predictions_year<N>.csv, input_hashes.json. The input files are hashed before and after the run.
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

from agent.building_history import FEATURES as HISTORY_FEATURES, HistoryEncoder, numeric_value


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "regression" / "regression" / "participants"
DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "regression_history"
KEY_COLUMNS = ("State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION")
NUMERIC_KEY_COLUMNS = ("floor_area", "year_built", "ELEVATION")
POSITIVE_KEY_COLUMNS = ("year_built",)
PERIOD = "Year_Factor"
VALIDATION_YEARS = (3, 4, 5)
SEED = 42
THREADS = 8
BLEND_WEIGHT = 0.5
MIN_ROWS_FOR_SUBGROUP_RMSE = 30
EXTRA_TREES = {"n_estimators": 200, "min_samples_leaf": 1, "max_depth": None, "max_features": 1.0}
REFERENCE_RMSE = {"hgb_all": 49.206622, "et_all": 47.789053}  # year 5, given by the team captain

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


def check_output_aliases(out_dir: Path, data_dir: Path, years) -> None:
    """Reject every planned destination that aliases an input by path, symlink or hard link, before any fitting."""
    inputs = protected_inputs(data_dir)
    outputs = [out_dir / name for name in ("metrics.json", "metrics.csv", "input_hashes.json")]
    outputs += [out_dir / f"predictions_year{year}.csv" for year in years]
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


def write_file(path: Path, write) -> None:
    """Write through a temporary file and replace the target in one step; a failure keeps the old file."""
    handle, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            write(file)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


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


def blend(prediction: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    has = history["history_present"].to_numpy() == 1
    mean = history["history_mean"].to_numpy(float)
    return np.where(has, (1 - BLEND_WEIGHT) * prediction + BLEND_WEIGHT * np.where(has, mean, 0.0), prediction)


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
            "fit_years": sorted(int(v) for v in np.unique(period[fit])), "history_features_seconds": history_seconds,
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
    for name, source in BLENDS:
        prediction = blend(predictions[source], history_val)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        started = time.perf_counter()
        summary = run_experiment(args.data_dir, args.out_dir, log=lambda text: print(text, flush=True))
        print(render(summary))
        print(f"Done in {time.perf_counter() - started:.0f} s. Results: {Path(args.out_dir).resolve()}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
