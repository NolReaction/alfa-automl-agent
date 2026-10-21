"""Check whether a building key can link rows of the same building across years (regression task).

Questions answered from the public CSV files only:
  1. How reliable is the key: repeats inside one year, effect of missing year_built.
  2. How many validation rows (year 5, history in years 1-4) and test rows (history in years 1-5) have history.
  3. Does the mean of the past target values of a key beat a constant on year 5, history taken from years 1-4 only?

A key match is never proof of one building. Rows whose key repeats inside one year cannot all be the same building;
rows whose key is unique per year are only consistent with one building. Nothing in the training code is touched.

    python building_history_check.py [--data-dir data/regression/regression/participants] [--json report.json]
    python building_history_check.py --with-baseline   # also compare with the project's baseline model (needs scikit-learn)
"""

from __future__ import annotations

import argparse
import contextlib
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from agent.building_history import history_keys, numeric_value, period_json


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "regression" / "regression" / "participants"
KEY_COLUMNS = ["State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION"]
NUMERIC_KEY_COLUMNS = ("floor_area", "year_built", "ELEVATION")
YEAR, TARGET = "Year_Factor", "site_eui"
VARIANTS = ("strict", "lenient")
VALIDATION_YEARS = (3, 4, 5)  # year 5 is the main check; 3 and 4 only show stability
SEED = 42
BOOTSTRAP = 2000


# ---------------------------------------------------------------- loading and keys

def typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Text columns as read from CSV to the types used below: exact year, float target.

    The year stays the exact decimal number written in the file (Decimal objects, compared as numbers): no rounding
    through float and no truncation of a fractional period to an integer. JSON reports write it with period_json.
    """
    missing = [name for name in (*KEY_COLUMNS, YEAR) if name not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame.copy()
    periods = [numeric_value(value) for value in frame[YEAR]]
    bad = sum(value is None for value in periods)
    if bad:
        raise ValueError(f"Column {YEAR!r} has {bad} missing or non-numeric values; every row needs a period")
    frame[YEAR] = pd.Series(periods, index=frame.index, dtype=object)
    if TARGET in frame:
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(float)
    return frame


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("train.csv", "test.csv"):
        path = Path(data_dir) / name
        if not path.is_file():
            raise ValueError(f"Required file not found: {path}")
        frames.append(typed(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")))
    train, test = frames
    if TARGET not in train:
        raise ValueError(f"train.csv has no target column {TARGET!r}")
    return train, test


def unknown_year_built(frame: pd.DataFrame) -> pd.Series:
    """year_built that is empty, not a number or not positive is not a real year."""
    return frame["year_built"].map(lambda value: (number := numeric_value(value)) is None or number <= 0)


def make_keys(frame: pd.DataFrame, variant: str) -> pd.Series:
    """One structured key per row, None where the row has no usable key.

    strict: a row with any unknown key part (in practice year_built) gets no key and never matches.
    lenient: an internal missing marker lets such rows match each other on the remaining parts.
    Numbers are compared as numbers: '61242.0' and '61242' give the same key.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    keys = history_keys(frame, KEY_COLUMNS, NUMERIC_KEY_COLUMNS, ("year_built",), allow_missing=variant == "lenient")
    return pd.Series(keys, index=frame.index, dtype=object)


# ---------------------------------------------------------------- 1. reliability of the key

def year_table(frame: pd.DataFrame, key: pd.Series) -> list[dict]:
    """Per year: rows without a key and keys that repeat inside the year (cannot be one building each)."""
    rows = []
    unknown = unknown_year_built(frame)
    for year, part in frame.assign(_key=key, _unknown=unknown).groupby(YEAR):
        keyed = part[part["_key"].notna()]
        sizes = keyed.groupby("_key").size()
        repeated = sizes[sizes > 1].index
        in_repeated = keyed["_key"].isin(repeated)
        row = {"year": year, "rows": len(part), "without_key": len(part) - len(keyed),
               "unknown_year_built": int(part["_unknown"].sum()), "distinct_keys": int(sizes.size),
               "keys_repeated": int(len(repeated)), "rows_in_repeated_keys": int(in_repeated.sum()),
               "share_in_repeated_keys": float(in_repeated.mean()) if len(keyed) else math.nan}
        known_year = keyed[~keyed["_unknown"]]
        unknown_year = keyed[keyed["_unknown"]]
        row["share_in_repeated_keys_known_year_built"] = float(known_year["_key"].isin(repeated).mean()) if len(known_year) else math.nan
        row["share_in_repeated_keys_unknown_year_built"] = float(unknown_year["_key"].isin(repeated).mean()) if len(unknown_year) else math.nan
        if TARGET in part:
            spread = keyed[in_repeated].groupby("_key")[TARGET].nunique()
            row["repeated_keys_with_identical_target"] = int((spread == 1).sum())
        rows.append(row)
    return rows


def key_summary(train: pd.DataFrame, key: pd.Series) -> dict:
    keyed = pd.DataFrame({"key": key, "year": train[YEAR]}).dropna(subset=["key"])
    years_per_key = keyed.groupby("key")["year"].nunique()
    return {"rows": len(train), "keyed_rows": len(keyed), "distinct_keys": int(years_per_key.size),
            "keys_in_two_or_more_years": int((years_per_key >= 2).sum()),
            "rows_in_keys_of_two_or_more_years": int(keyed["key"].map(years_per_key).ge(2).sum())}


# ---------------------------------------------------------------- history lookups

class History:
    """Rows with a key from earlier years: mean target, years seen, rows per year."""

    def __init__(self, key: pd.Series, year: pd.Series, target: pd.Series | None = None):
        rows = pd.DataFrame({"key": key, "year": year})
        if target is not None:
            rows["y"] = target
        rows = rows.dropna(subset=["key"])
        per_year = rows.groupby(["key", "year"]).size()
        self.max_rows_per_year = per_year.groupby(level=0).max()
        self.years = per_year.groupby(level=0).size()
        self.mean = rows.groupby("key")["y"].mean() if target is not None else None
        self.keys_by_year = {y: set(part["key"]) for y, part in rows.groupby("year")}

    def lookup(self, key: pd.Series) -> dict:
        """For rows of one later year (validation or test): history flag and 'clean' flag.

        clean = the key is unique inside the history years and inside this year, so the match is consistent with
        one building (not proven). Any other match is ambiguous.
        """
        in_year = key.map(key.value_counts())
        has = key.map(self.years).notna().to_numpy()
        clean = has & (key.map(self.max_rows_per_year).to_numpy() == 1) & (in_year.to_numpy() == 1)
        return {"has": has, "clean": clean, "years": key.map(self.years).fillna(0).to_numpy(int)}


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan


# ---------------------------------------------------------------- 2. coverage

def validation_coverage(train: pd.DataFrame, key: pd.Series, val_year: int) -> dict:
    fit, val = train[YEAR] < val_year, train[YEAR] == val_year
    found = History(key[fit], train.loc[fit, YEAR]).lookup(key[val])
    n = int(val.sum())
    return {"validation_year": val_year, "rows": n, "with_history": int(found["has"].sum()),
            "share_with_history": float(found["has"].mean()), "share_clean": float(found["clean"].mean()),
            "share_ambiguous": float((found["has"] & ~found["clean"]).mean())}


def test_coverage(train: pd.DataFrame, test: pd.DataFrame, train_key: pd.Series, test_key: pd.Series) -> dict:
    history = History(train_key, train[YEAR])
    found = history.lookup(test_key)
    by_year = {y: float(test_key.isin(keys).mean()) for y, keys in sorted(history.keys_by_year.items())}
    years = pd.Series(found["years"]).value_counts(normalize=True).sort_index()
    return {"rows": len(test), "share_with_history": float(found["has"].mean()), "share_clean": float(found["clean"].mean()),
            "share_ambiguous": float((found["has"] & ~found["clean"]).mean()),
            "share_with_key_in_year": by_year, "share_by_number_of_years_with_history": {int(k): float(v) for k, v in years.items()}}


# ---------------------------------------------------------------- 3. does the history mean help

def paired_bootstrap(se_constant: np.ndarray, se_history: np.ndarray, seed: int, resamples: int) -> list[float]:
    """95% interval of RMSE(constant) - RMSE(history) over resampled rows (positive: history is better)."""
    n = len(se_constant)
    if n == 0 or resamples < 1:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    chunk = max(1, min(resamples, 4_000_000 // n))
    gains = []
    for start in range(0, resamples, chunk):
        index = rng.integers(0, n, size=(min(chunk, resamples - start), n))
        gains.append(np.sqrt(se_constant[index].mean(axis=1)) - np.sqrt(se_history[index].mean(axis=1)))
    low, high = np.percentile(np.concatenate(gains), [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_fold(train: pd.DataFrame, key: pd.Series, val_year: int, bootstrap: int = BOOTSTRAP) -> dict:
    """Predict val_year from years before it. No history: mean target of the fitting years."""
    fit, val = train[YEAR] < val_year, train[YEAR] == val_year
    y_fit, y_val = train.loc[fit, TARGET].to_numpy(float), train.loc[val, TARGET].to_numpy(float)
    history = History(key[fit], train.loc[fit, YEAR], train.loc[fit, TARGET])
    found = history.lookup(key[val])
    constant = float(y_fit.mean())
    past_mean = key[val].map(history.mean).to_numpy(float)
    has, clean = found["has"], found["clean"]
    ambiguous = has & ~clean
    prediction = np.where(has, past_mean, constant)
    prediction_clean_only = np.where(clean, past_mean, constant)
    se_constant, se_history = (y_val - constant) ** 2, (y_val - prediction) ** 2

    def part(mask: np.ndarray) -> dict:
        return {"rows": int(mask.sum()), "share": float(mask.mean()),
                "rmse_constant": rmse((y_val - constant)[mask]), "rmse_history_mean": rmse((y_val - prediction)[mask])}

    everything = np.ones(len(y_val), dtype=bool)
    result = {"validation_year": val_year, "fit_rows": int(fit.sum()), "constant": constant,
              "all_rows": part(everything), "with_history": part(has), "clean": part(clean),
              "ambiguous": part(ambiguous), "no_history": part(~has),
              "rmse_history_mean_only_when_clean": rmse(y_val - prediction_clean_only)}
    result["bootstrap_gain_all_rows"] = paired_bootstrap(se_constant, se_history, SEED, bootstrap)
    result["bootstrap_gain_with_history"] = paired_bootstrap(se_constant[has], se_history[has], SEED, bootstrap)
    return result


def baseline_reference(data_dir: Path, train: pd.DataFrame, key: pd.Series, val_year: int = 5,
                       bootstrap: int = BOOTSTRAP) -> list[dict]:
    """Optional: the project's offline baseline model against the history mean on the same validation rows.

    The model is fitted on years before val_year, exactly like offline_baseline.make_pipeline. The blend is a plain
    50/50 average of model and history mean on rows with history; its weight is not tuned on the validation year.
    """
    import offline_baseline as baseline  # read-only reuse, needs scikit-learn
    from threadpoolctl import threadpool_limits

    raw, _, _, _, _, y, features, numeric, categorical, _ = baseline.load_data(Path(data_dir))
    x = baseline.prepare_features(raw, features, numeric, categorical)
    fit, val = (train[YEAR] < val_year).to_numpy(), (train[YEAR] == val_year).to_numpy()
    y_val = y[val].to_numpy(float)
    history = History(key[fit], train.loc[fit, YEAR], train.loc[fit, TARGET])
    has = history.lookup(key[val])["has"]
    past_mean = key[val].map(history.mean).to_numpy(float)
    results = []
    for label, dropped in (("all features, as in the project baseline", ()), ("without Year_Factor", (YEAR,))):
        names = [name for name in features if name not in dropped]
        pipeline = baseline.make_pipeline([n for n in numeric if n in names], [n for n in categorical if n in names], False)
        with threadpool_limits(limits=8):
            pipeline.fit(x.loc[fit, names], y[fit])
            model = pipeline.predict(x.loc[val, names])
        blend = np.where(has, (model + past_mean) / 2, model)
        history_else_model = np.where(has, past_mean, model)
        results.append({
            "features": label, "validation_year": val_year, "rows_with_history": int(has.sum()),
            "rmse_model_all": rmse(y_val - model), "rmse_model_with_history": rmse((y_val - model)[has]),
            "rmse_history_with_history": rmse((y_val - past_mean)[has]), "rmse_blend_with_history": rmse((y_val - blend)[has]),
            "rmse_model_without_history": rmse((y_val - model)[~has]),
            "rmse_all_history_else_model": rmse(y_val - history_else_model), "rmse_all_blend": rmse(y_val - blend),
            "bootstrap_gain_blend_all_rows": paired_bootstrap((y_val - model) ** 2, (y_val - blend) ** 2, SEED, bootstrap)})
    return results


def consecutive_year_correlation(train: pd.DataFrame, key: pd.Series) -> dict:
    """Statistical evidence for the key: target of key-unique-per-year rows in year y vs y+1, against a control that
    pairs each row with a random row of the same facility_type and building_class from the next year."""
    rows = pd.DataFrame({"key": key, "year": train[YEAR], "y": train[TARGET],
                         "group": train["facility_type"] + "|" + train["building_class"]}).dropna(subset=["key"])
    rows = rows[rows.groupby(["key", "year"])["y"].transform("size") == 1]
    rng = np.random.default_rng(SEED)
    pairs = []
    periods = sorted(rows["year"].unique())
    for year, following in zip(periods, periods[1:]):  # next period present in the data, not year + 1
        now, later = rows[rows["year"] == year], rows[rows["year"] == following]
        both = now.merge(later, on="key", suffixes=("", "_next"))
        if both.empty:
            continue
        control = both["y_next"].copy()
        for group, index in both.groupby("group").groups.items():
            pool = later.loc[later["group"] == group, "y"].to_numpy()
            if len(pool):
                control.loc[index] = rng.choice(pool, size=len(index))
        pairs.append(both.assign(control=control))
    if not pairs:
        return {"pairs": 0}
    pairs = pd.concat(pairs)
    return {"pairs": len(pairs),
            "spearman_same_key": float(pairs["y"].corr(pairs["y_next"], method="spearman")),
            "pearson_same_key": float(pairs["y"].corr(pairs["y_next"])),
            "spearman_control_same_facility_type": float(pairs["y"].corr(pairs["control"], method="spearman")),
            "pearson_control_same_facility_type": float(pairs["y"].corr(pairs["control"]))}


# ---------------------------------------------------------------- orchestration and output

def analyze(train: pd.DataFrame, test: pd.DataFrame, bootstrap: int = BOOTSTRAP) -> dict:
    report = {"rows": {"train": len(train), "test": len(test)},
              "years": {"train": sorted(train[YEAR].unique()), "test": sorted(test[YEAR].unique())},
              "key_columns": KEY_COLUMNS, "variants": {}}
    for variant in VARIANTS:
        train_key, test_key = make_keys(train, variant), make_keys(test, variant)
        report["variants"][variant] = {
            "key_summary": key_summary(train, train_key),
            "train_years": year_table(train, train_key), "test_years": year_table(test, test_key),
            "validation_coverage": [validation_coverage(train, train_key, y) for y in VALIDATION_YEARS],
            "test_coverage": test_coverage(train, test, train_key, test_key),
            "folds": [evaluate_fold(train, train_key, y, bootstrap) for y in VALIDATION_YEARS],
            "consecutive_years": consecutive_year_correlation(train, train_key),
        }
    return report


def pct(value) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{100 * value:.1f}%"


def num(value, digits: int = 2) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def render(report: dict) -> str:
    years = {part: [period_json(y) for y in values] for part, values in report["years"].items()}
    lines = [f"Key: {', '.join(report['key_columns'])}",
             f"Rows: train {report['rows']['train']} (years {years['train']}), "
             f"test {report['rows']['test']} (years {years['test']})",
             "strict  = rows with any unknown key part get no key; lenient = missing parts match each other"]
    for variant, data in report["variants"].items():
        summary = data["key_summary"]
        lines += ["", f"==================== {variant.upper()} ====================",
                  f"Train: {summary['keyed_rows']} of {summary['rows']} rows have a key, {summary['distinct_keys']} distinct keys, "
                  f"{summary['keys_in_two_or_more_years']} keys occur in 2+ years ({summary['rows_in_keys_of_two_or_more_years']} rows)",
                  "", "1. Repeats of a key inside ONE year (cannot all be one building)",
                  "year   rows  no_key  unk_yb  keys_repeated  rows_in_repeated  share  share(known yb)  share(unknown yb)  same_target"]
        for row in [*data["train_years"], *data["test_years"]]:
            lines.append(f"{period_json(row['year'])!s:>4} {row['rows']:>6} {row['without_key']:>7} {row['unknown_year_built']:>7} "
                         f"{row['keys_repeated']:>14} {row['rows_in_repeated_keys']:>17} {pct(row['share_in_repeated_keys']):>6} "
                         f"{pct(row['share_in_repeated_keys_known_year_built']):>15} {pct(row['share_in_repeated_keys_unknown_year_built']):>18} "
                         f"{row.get('repeated_keys_with_identical_target', '-'):>11}")
        lines += ["", "2. Coverage: share of rows whose key has history in earlier years",
                  "validation year (history = all earlier train years): rows, with history, clean, ambiguous"]
        for row in data["validation_coverage"]:
            lines.append(f"  year {row['validation_year']}: {row['rows']} rows, {pct(row['share_with_history'])} with history "
                         f"(clean {pct(row['share_clean'])}, ambiguous {pct(row['share_ambiguous'])})")
        cov = data["test_coverage"]
        lines.append(f"test (history = train years 1-5): {cov['rows']} rows, {pct(cov['share_with_history'])} with history "
                     f"(clean {pct(cov['share_clean'])}, ambiguous {pct(cov['share_ambiguous'])})")
        lines.append("  test rows whose key occurs in train year: " + ", ".join(
            f"{period_json(y)}: {pct(v)}" for y, v in cov["share_with_key_in_year"].items()))
        lines.append("  test rows by number of train years with the key: " + ", ".join(
            f"{k}: {pct(v)}" for k, v in cov["share_by_number_of_years_with_history"].items()))
        lines += ["", "3. Mean of the key's past target vs constant (constant = mean target of the fitting years; no history -> constant)"]
        for fold in data["folds"]:
            lines.append(f"validation year {fold['validation_year']} (fit rows {fold['fit_rows']}, constant {num(fold['constant'])})"
                         + ("   <- main check" if fold["validation_year"] == 5 else ""))
            lines.append("  rows group        rows   share   RMSE constant   RMSE history mean")
            for name in ("all_rows", "with_history", "clean", "ambiguous", "no_history"):
                p = fold[name]
                lines.append(f"  {name:<15}{p['rows']:>7} {pct(p['share']):>7} {num(p['rmse_constant']):>15} {num(p['rmse_history_mean']):>19}")
            low, high = fold["bootstrap_gain_all_rows"]
            low_h, high_h = fold["bootstrap_gain_with_history"]
            lines.append(f"  RMSE gain (constant - history), 95% bootstrap interval: all rows [{num(low)}; {num(high)}], "
                         f"rows with history [{num(low_h)}; {num(high_h)}]")
            lines.append(f"  history used only for clean matches: RMSE on all rows {num(fold['rmse_history_mean_only_when_clean'])}")
        cons = data["consecutive_years"]
        if cons["pairs"]:
            lines += ["", f"Same-key rows in consecutive years (key unique in both years), {cons['pairs']} pairs: target correlation "
                      f"Spearman {num(cons['spearman_same_key'], 3)} / Pearson {num(cons['pearson_same_key'], 3)}; control with a random row of the "
                      f"same facility_type and building_class: Spearman {num(cons['spearman_control_same_facility_type'], 3)} / "
                      f"Pearson {num(cons['pearson_control_same_facility_type'], 3)}"]
    if report.get("baseline_reference"):
        lines += ["", "==================== REFERENCE: project baseline model (strict key, validation year 5) ===================="]
        for ref in report["baseline_reference"]:
            low, high = ref["bootstrap_gain_blend_all_rows"]
            lines += [f"Model features: {ref['features']}",
                      f"  all rows:            model {num(ref['rmse_model_all'], 4)} | 50/50 blend where history {num(ref['rmse_all_blend'])} "
                      f"| history mean where history, else model {num(ref['rmse_all_history_else_model'])}",
                      f"  rows with history ({ref['rows_with_history']}): model {num(ref['rmse_model_with_history'])} | history mean "
                      f"{num(ref['rmse_history_with_history'])} | 50/50 blend {num(ref['rmse_blend_with_history'])}",
                      f"  rows without history: model {num(ref['rmse_model_without_history'])}",
                      f"  RMSE gain of the blend over the model on all rows, 95% bootstrap interval: [{num(low)}; {num(high)}]"]
    return "\n".join(lines)


def jsonable(value):
    """Strict JSON values; exact periods (Decimal) as period_json gives them, also when they are dictionary keys."""
    if isinstance(value, dict):
        return {str(period_json(key) if isinstance(key, Decimal) else key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return period_json(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def checked_report_path(path: Path, data_dir: Path) -> Path:
    """Where the JSON report may go: a .json file that is not one of the input files, also through links."""
    target = Path(path).resolve()
    for name in (Path(path), target):
        if name.suffix.lower() != ".json":
            raise ValueError(f"The report file must have the .json extension: {path}")
    if target.is_dir():
        raise ValueError(f"The report path is a folder: {path}")
    folder = Path(data_dir)
    inputs = [item for item in folder.iterdir() if item.is_file()] if folder.is_dir() else []
    if target.exists() and any(os.path.samefile(target, item) for item in inputs):
        raise ValueError(f"The report must not overwrite an input file: {path}")
    return target


def write_json_atomically(target: Path, report: dict) -> None:
    """Write the whole report into a temporary file next to the target, then replace the target in one step.

    The text is built first, so a serialization error or a failed write leaves the previous file untouched.
    """
    text = json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json", type=Path, help="Also write all numbers to this .json file (never over an input file)")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP, help="Bootstrap resamples (0 disables the interval)")
    parser.add_argument("--with-baseline", action="store_true",
                        help="Also compare with the project's baseline model on year 5 (needs scikit-learn)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        report_path = checked_report_path(args.json, args.data_dir) if args.json else None
        train, test = load(args.data_dir)
        report = analyze(train, test, args.bootstrap)
        if args.with_baseline:
            report["baseline_reference"] = baseline_reference(args.data_dir, train, make_keys(train, "strict"),
                                                              bootstrap=args.bootstrap)
        print(render(report))
        if report_path:
            write_json_atomically(report_path, report)
        return 0
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
