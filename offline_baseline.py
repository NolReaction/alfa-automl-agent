"""Local ML baseline, without LLM/API calls. Not the competition entry point."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# HistGradientBoosting uses OpenMP; RandomForest also has an explicit n_jobs cap.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from validation import validate_predictions


SEED = 42
VALIDATION_FRACTION = 0.2


def csv_header(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Required file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if not header or any(not name.strip() for name in header):
        raise ValueError(f"Missing or empty column names: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names: {path}")
    return header


def read_csv(path: Path) -> pd.DataFrame:
    csv_header(path)
    # Preserve IDs such as 001 and class labels such as NA exactly as written.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def find_data(task_dir: Path) -> Path:
    samples = sorted(
        path for path in task_dir.rglob("sample_submission.csv")
        if path.is_file() and ".agent_runs" not in path.relative_to(task_dir).parts
        and not any(part.startswith(".offline-baseline-")
                    for part in path.relative_to(task_dir).parts)
    )
    if len(samples) != 1:
        found = ", ".join(str(path.relative_to(task_dir)) for path in samples)
        raise ValueError(
            f"Expected exactly one sample_submission.csv inside {task_dir}; "
            f"found {len(samples)}" + (f": {found}" if found else ". Download the task data first.")
        )
    return samples[0].parent


def load_data(data_dir: Path):
    metadata = read_csv(data_dir / "data_dictionary.csv")
    if not {"feature", "role", "dtype"}.issubset(metadata.columns):
        raise ValueError("data_dictionary.csv must contain feature, role and dtype columns")
    if metadata["feature"].duplicated().any():
        raise ValueError("data_dictionary.csv contains duplicate feature definitions")
    metadata = metadata.set_index("feature")
    roles = metadata["role"].str.strip().str.lower()
    dtypes = metadata["dtype"].str.strip().str.lower().to_dict()
    id_fields = roles.index[roles == "id"].tolist()
    targets = roles.index[roles == "target"].tolist()
    if len(id_fields) != 1 or len(targets) != 1:
        raise ValueError("data_dictionary.csv must declare exactly one id and one target")
    id_column, target = id_fields[0], targets[0]
    target_dtype = dtypes[target]
    if target_dtype not in {"numeric", "category"}:
        raise ValueError(f"Unsupported target dtype {target_dtype!r}; expected numeric or category")
    classification = target_dtype == "category"
    columns = csv_header(data_dir / "sample_submission.csv")
    if len(columns) != 2 or set(columns) != {id_column, target}:
        raise ValueError("sample_submission.csv must contain the declared id and target only")
    train, test = read_csv(data_dir / "train.csv"), read_csv(data_dir / "test.csv")
    if id_column not in train or target not in train or id_column not in test:
        raise ValueError("Declared id/target columns are missing from train.csv or test.csv")
    if target in test:
        raise ValueError("test.csv unexpectedly contains the target column")
    features = [name for name in train.columns if name not in {id_column, target}]
    if not features or set(features) != set(test.columns) - {id_column}:
        raise ValueError("Train/test feature columns must match and must not be empty")
    for name in features:
        if name not in metadata.index or roles[name] != "feature":
            raise ValueError(f"Feature {name!r} is not declared with role=feature in the dictionary")
        if dtypes[name] not in {"numeric", "category", "binary"}:
            raise ValueError(f"Unsupported dtype for feature {name!r}: {dtypes[name]!r}")
    if len(train) < 2 or test.empty:
        raise ValueError("Need at least two training rows and one test row")
    for name, frame in (("train.csv", train), ("test.csv", test)):
        if frame[id_column].str.strip().eq("").any() or frame[id_column].duplicated().any():
            raise ValueError(f"{name}: IDs must be nonempty and unique")
    if train[target].str.strip().eq("").any():
        raise ValueError("train.csv contains missing target values")
    y = train[target] if classification else pd.to_numeric(train[target], errors="coerce")
    if not classification and not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Regression target must contain finite numeric values only")
    if classification and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    numeric = [name for name in features if dtypes[name] in {"numeric", "binary"}]
    categorical = [name for name in features if dtypes[name] == "category"]
    return train, test, columns, id_column, target, y, features, numeric, categorical, classification


def prepare_features(frame, features, numeric, categorical):
    result = frame.loc[:, features].copy()
    for name in numeric:
        result[name] = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        result[name] = result[name].replace("", np.nan)
    return result


def make_pipeline(numeric, categorical, classification):
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        transformers.append(("category", Pipeline([
            ("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical))
    if classification:
        model = RandomForestClassifier(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=8, random_state=SEED,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.1,
            early_stopping=False, categorical_features=None, random_state=SEED,
        )
    return Pipeline([
        ("prepare", ColumnTransformer(transformers, remainder="drop", n_jobs=1)),
        ("model", model),
    ])


def save_outputs(task_dir: Path, data_dir: Path, submission, report, overwrite: bool):
    output = task_dir / "submission.csv"
    # Validate in a temporary task directory so a bad candidate cannot erase an old result.
    with tempfile.TemporaryDirectory(prefix=".offline-baseline-", dir=task_dir) as directory:
        staged = Path(directory)
        inputs = staged / "inputs"
        inputs.mkdir()
        for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
            try:
                os.link(data_dir / name, inputs / name)
            except OSError:
                shutil.copyfile(data_dir / name, inputs / name)
        candidate = staged / "submission.csv"
        submission.to_csv(candidate, index=False, encoding="utf-8")
        result = validate_predictions(staged)
        if not result["valid"]:
            raise ValueError("Output validation failed: " + "; ".join(result["errors"]))
        metrics = staged / "baseline_metrics.json"
        metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if overwrite:
            os.replace(candidate, output)
        elif os.name == "nt":
            # Windows rename is atomic and fails if the destination exists.
            # It also works on Windows filesystems without hardlink support.
            os.rename(candidate, output)
        else:
            # Atomic no-clobber publication, including if another process wrote meanwhile.
            os.link(candidate, output)
        os.replace(metrics, task_dir / "baseline_metrics.json")


def run_baseline(task_dir: Path, overwrite: bool = False) -> None:
    started = time.monotonic()
    output = task_dir / "submission.csv"
    if (output.exists() or output.is_symlink()) and not overwrite:
        raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")
    data_dir = find_data(task_dir)
    (train, test, columns, id_column, target, y, features,
     numeric, categorical, classification) = load_data(data_dir)
    x = prepare_features(train, features, numeric, categorical)
    test_x = prepare_features(test, features, numeric, categorical)
    try:
        fit_rows, validation_rows = train_test_split(
            np.arange(len(train)), test_size=VALIDATION_FRACTION, random_state=SEED,
            shuffle=True, stratify=y if classification else None,
        )
    except ValueError as error:
        raise ValueError(f"Cannot make the fixed 80/20 split (stratified for classification): {error}") from error
    pipeline = make_pipeline(numeric, categorical, classification)
    fit_started = time.monotonic()
    with threadpool_limits(limits=8):
        pipeline.fit(x.iloc[fit_rows], y.iloc[fit_rows])
        validation_predictions = pipeline.predict(x.iloc[validation_rows])
    value = (f1_score(y.iloc[validation_rows], validation_predictions, average="macro", zero_division=0)
             if classification else np.sqrt(mean_squared_error(y.iloc[validation_rows], validation_predictions)))
    metric = "Macro F1" if classification else "RMSE"
    validation_seconds = time.monotonic() - fit_started
    print(f"Local validation {metric}: {value:.6f} ({validation_seconds:.1f}s)", flush=True)
    print(f"Refitting on all {len(train)} training rows...", flush=True)
    refit_started = time.monotonic()
    final_pipeline = clone(pipeline)
    with threadpool_limits(limits=8):
        final_pipeline.fit(x, y)
        predictions = final_pipeline.predict(test_x)
    submission = pd.DataFrame({id_column: test[id_column], target: predictions}).loc[:, columns]
    report = {
        "task_type": "classification" if classification else "regression",
        "seed": SEED,
        "split": {"train_fraction": 0.8, "validation_fraction": VALIDATION_FRACTION,
                  "shuffle": True, "stratified": classification,
                  "train_rows": len(fit_rows), "validation_rows": len(validation_rows)},
        "model": {"name": type(pipeline.named_steps["model"]).__name__,
                  "parameters": pipeline.named_steps["model"].get_params()},
        "metric": {"name": metric, "value": float(value),
                   "higher_is_better": classification, "source": "local_holdout_from_train.csv"},
        "shapes": {"train": list(train.shape), "test": list(test.shape),
                   "training_features": [len(fit_rows), len(features)],
                   "validation_features": [len(validation_rows), len(features)],
                   "submission": list(submission.shape)},
        "id_column": id_column, "target_column": target,
        "features": {"numeric": numeric, "categorical": categorical},
        "api": {"used": False, "calls": 0},
        "timing_seconds": {"validation": validation_seconds,
                           "refit_and_predict": time.monotonic() - refit_started,
                           "total_before_save": time.monotonic() - started},
    }
    save_outputs(task_dir, data_dir, submission, report, overwrite)
    print(f"Saved: {output}")
    print(f"Metrics: {task_dir / 'baseline_metrics.json'}")
    print(f"Total time: {time.monotonic() - started:.1f}s. API calls: 0.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Path to task.md or the directory containing this task's data")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing submission.csv")
    args = parser.parse_args()
    try:
        path = Path(args.task).resolve()
        if not path.exists() or (path.is_file() and path.suffix.lower() != ".md"):
            raise ValueError("Pass an existing task.md file or a task directory")
        run_baseline(path if path.is_dir() else path.parent, overwrite=args.overwrite)
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
