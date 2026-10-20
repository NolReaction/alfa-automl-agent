"""Compare five predeclared class-weight recipes on one grouped holdout.

This research command never refits on all data or writes submission.csv. Outputs
are placed in a new directory outside the task's input tree. It uses no API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "8"

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from agent.diagnostics import classification_diagnostics
from agent.file_io import replace_file
from agent.model_config import make_model_config
from agent.splits import split_rows
from agent.task_contract import validate_task_metric
from offline_baseline import SEED, find_data, load_data, make_pipeline, prepare_features


ROOT = Path(__file__).resolve().parent
INPUT_NAMES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
MAX_INPUT_BYTES = 512 * 1024 * 1024


def _plain(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Symbolic links and junctions are not supported in experiment paths.")
    return path.resolve()


def _hash_file(path: Path) -> str:
    path = _plain(path)
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Missing or oversized input file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(data_dir: Path, task_path: Path) -> dict:
    return {"task.md": _hash_file(task_path),
            **{name: _hash_file(data_dir / name) for name in INPUT_NAMES}}


def _json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    _plain(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        replace_file(temporary, path, check=lambda: (_plain(temporary), _plain(path)))
    finally:
        temporary.unlink(missing_ok=True)


def candidate_plan() -> list[dict]:
    """Keep each ablation different from its control in class_weight only."""
    definitions = (
        ("rf_control", "RandomForest · balanced_subsample (контроль)", "baseline", None),
        ("et_control", "ExtraTrees · balanced (контроль)", "extra_trees", None),
        ("rf_none", "RandomForest · без весов", "baseline", {"class_weight": None}),
        ("rf_balanced", "RandomForest · balanced", "baseline", {"class_weight": "balanced"}),
        ("et_none", "ExtraTrees · без весов", "extra_trees", {"class_weight": None}),
    )
    return [{"id": identifier, "label": label, "action": action,
             "model_config": make_model_config(action, "classification", parameters)}
            for identifier, label, action, parameters in definitions]


def grouped_fold(train, y, features, fold_index=0):
    """Select an explicit fold of the shared grouped scheme, without fallback.

    The app's splitter establishes the feasible n_splits. An explicitly chosen
    fold must contain every class in both parts; it is never silently replaced
    with a more favorable fold. Fold zero reproduces the public reference.
    """
    if type(fold_index) is not int or not 0 <= fold_index <= 4:
        raise ValueError("fold-index must be an integer from 0 to 4.")
    _, _, split = split_rows(train, y, features, True, {"strategy": "grouped"})
    n_splits = split["n_splits"]
    if fold_index >= n_splits:
        raise ValueError(f"This dataset supports {n_splits} folds; requested fold {fold_index}.")
    groups = pd.util.hash_pandas_object(train.loc[:, features], index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fit_rows, held_rows = list(splitter.split(list(range(len(train))), y, groups))[fold_index]
    classes = set(y.tolist())
    if set(y.iloc[fit_rows].tolist()) != classes or set(y.iloc[held_rows].tolist()) != classes:
        raise ValueError(f"Requested fold {fold_index} does not contain all classes in both parts.")
    fitting_groups, held_groups = set(groups[fit_rows]), set(groups[held_rows])
    overlap = len(fitting_groups & held_groups)
    if overlap:
        raise ValueError("Feature groups overlap between fitting and validation rows.")
    fraction = len(held_rows) / len(train)
    split.update(train_rows=len(fit_rows), validation_rows=len(held_rows),
                 train_fraction=1 - fraction, validation_fraction=fraction,
                 group_count=len(set(groups)), train_groups=len(fitting_groups),
                 validation_groups=len(held_groups), group_overlap=overlap, fold=fold_index)
    return fit_rows, held_rows, split


def select_recipes(recipes=None):
    plan = candidate_plan()
    if recipes is None:
        return plan
    known = {row["id"] for row in plan}
    if (not isinstance(recipes, (list, tuple)) or not recipes
            or any(not isinstance(name, str) for name in recipes)
            or len(set(recipes)) != len(recipes) or not set(recipes) <= known):
        raise ValueError("recipes must contain distinct known recipe identifiers.")
    # Preserve the predeclared order even if CLI identifiers are reordered.
    return [row for row in plan if row["id"] in recipes]


def run_experiment(task_path: Path, *, output_dir: Path | None = None,
                   fold_index=0, recipes=None) -> dict:
    started = time.monotonic()
    if type(fold_index) is not int or not 0 <= fold_index <= 4:
        raise ValueError("fold-index must be an integer from 0 to 4.")
    plan = select_recipes(recipes)
    task_path = _plain(task_path)
    if not task_path.is_file() or task_path.suffix.lower() != ".md":
        raise ValueError("Pass an existing task.md file.")
    task_dir = task_path.parent
    data_dir = _plain(find_data(task_dir))
    hashes = _input_hashes(data_dir, task_path)
    task_text = task_path.read_text(encoding="utf-8-sig")
    validate_task_metric(task_text, True)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "delivery" / "classification-experiments" / f"{stamp}-{uuid4().hex[:8]}"
    output_dir = _plain(output_dir)
    if output_dir.is_relative_to(task_dir):
        raise ValueError("The output directory must be outside the task directory.")
    # No existing output is silently replaced, including a file or source path.
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "format": "classification-weight-experiment", "version": 1, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "api": {"used": False, "calls": 0}, "input_sha256": hashes,
        "candidates": [], "output_dir": str(output_dir),
    }
    try:
        snapshot = output_dir / "inputs"
        snapshot.mkdir()
        for name in INPUT_NAMES:
            shutil.copyfile(_plain(data_dir / name), snapshot / name)
        (output_dir / "task.md").write_bytes(task_path.read_bytes())
        if (_input_hashes(snapshot, output_dir / "task.md") != hashes
                or _input_hashes(data_dir, task_path) != hashes):
            raise ValueError("Task inputs changed while preparing the experiment.")
        (train, test, columns, id_column, target, y, features,
         numeric, categorical, classification) = load_data(snapshot)
        if not classification:
            raise ValueError("This experiment supports classification only.")
        x = prepare_features(train, features, numeric, categorical)
        fit_rows, held_rows, split = grouped_fold(train, y, features, fold_index)
        split["validation_hash"] = _json_hash({
            "id_column": id_column, "target_column": target,
            "ids": train[id_column].iloc[held_rows].tolist(),
            "actuals": train[target].iloc[held_rows].tolist(),
        })
        # This research identity includes the exact five input file hashes.
        # validation_hash is identical to the app's for the same held-out rows.
        dataset_fingerprint = _json_hash(hashes)
        comparison_key = _json_hash({"dataset_fingerprint": dataset_fingerprint,
                                    "metric": "Macro F1", "higher_is_better": True, "split": split})
        report.update(task_type="classification", dataset_fingerprint=dataset_fingerprint,
                      comparison_key=comparison_key, split=split,
                      task_rows={"train": len(train), "test": len(test)},
                      features={"numeric": numeric, "categorical": categorical},
                      id_column=id_column, target_column=target)
        report["planned"] = len(plan)
        report["selected_recipes"] = [row["id"] for row in plan]
        _atomic_json(output_dir / "report.json", report)
        for index, candidate in enumerate(plan, start=1):
            candidate_started = time.monotonic()
            config = candidate["model_config"]
            pipeline = make_pipeline(numeric, categorical, True)
            if candidate["action"] == "extra_trees":
                pipeline.set_params(model=ExtraTreesClassifier(random_state=SEED, n_jobs=8))
            pipeline.named_steps["model"].set_params(**config["parameters"])
            print(f"[{index}/{len(plan)}] {candidate['label']}", flush=True)
            fit_started = time.monotonic()
            with threadpool_limits(limits=8):
                pipeline.fit(x.iloc[fit_rows], y.iloc[fit_rows])
            fit_seconds = time.monotonic() - fit_started
            predict_started = time.monotonic()
            with threadpool_limits(limits=8):
                predicted = pipeline.predict(x.iloc[held_rows])
            prediction_seconds = time.monotonic() - predict_started
            actual = train[target].iloc[held_rows].tolist()
            score = float(f1_score(actual, predicted, average="macro", zero_division=0))
            diagnostics = classification_diagnostics(actual, predicted)
            candidate_dir = output_dir / candidate["id"]
            candidate_dir.mkdir()
            held_path = candidate_dir / "holdout_predictions.csv"
            pd.DataFrame({"id": train[id_column].iloc[held_rows].to_numpy(),
                          "actual": actual, "prediction": predicted}).to_csv(
                held_path, index=False, encoding="utf-8",
            )
            result = {
                "id": candidate["id"], "label": candidate["label"], "action": candidate["action"],
                "model_config": config,
                "metric": {"name": "Macro F1", "value": score, "higher_is_better": True},
                "comparison_key": comparison_key,
                "holdout_predictions_path": held_path.relative_to(output_dir).as_posix(),
                "timing_seconds": {"fit": fit_seconds, "prediction": prediction_seconds,
                                   "total": time.monotonic() - candidate_started},
            }
            if diagnostics is not None:
                result["classification_diagnostics"] = diagnostics
            report["candidates"].append(result)
            report["elapsed_seconds"] = time.monotonic() - started
            _atomic_json(output_dir / "report.json", report)
            print(f"  Macro F1: {score:.6f}", flush=True)
        if _input_hashes(data_dir, task_path) != hashes:
            raise ValueError("Source inputs changed during the experiment; comparison is invalid.")
        report.update(status="completed", source_inputs_unchanged=True,
                      elapsed_seconds=time.monotonic() - started,
                      best_candidate=max(report["candidates"], key=lambda row: row["metric"]["value"])["id"])
        _atomic_json(output_dir / "report.json", report)
        print(f"Report: {output_dir / 'report.json'}. API calls: 0.", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error), elapsed_seconds=time.monotonic() - started)
        try:
            _atomic_json(output_dir / "report.json", report)
        except (OSError, ValueError):
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="Path to the classification task.md")
    parser.add_argument("--output-dir", type=Path, help="New output directory, outside the task tree")
    parser.add_argument("--fold-index", type=int, choices=range(5), default=0,
                        help="Explicit grouped fold (0 reproduces the public reference)")
    parser.add_argument("--recipes", nargs="+", choices=[row["id"] for row in candidate_plan()],
                        help="Run only these predeclared recipes; omitted means all five")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.task, output_dir=args.output_dir,
                       fold_index=args.fold_index, recipes=args.recipes)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
