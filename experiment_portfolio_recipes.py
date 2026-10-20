"""Bounded offline comparison of frozen recipe portfolios, plus synthetic smoke.

Example: python experiment_portfolio_recipes.py --task data/classification/task.md
--out-dir data/experiments/tanya-model-recipes/classification --search-seconds 600
--attempt-seconds 180 --refit-seconds 180 --memory-gib 8 --seeds 1729 2718.
Use --builtin breast_cancer|wine|diabetes for paired transfer checks, or
--task-type regression|classification for the original synthetic smoke API.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from agent.metrics import get_metric, score_predictions
from agent.model_config import ACTIONS
from agent.portfolio_recipes import portfolio_candidates

ROOT = Path(__file__).resolve().parent
SEED = 1729


def smoke_data(task_type, binary=False):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(SEED)
    number = rng.normal(size=160)
    frame = pd.DataFrame({"number": number, "empty": np.full(160, np.nan),
                          "category": ["NA", "blue", "red", ""] * 40})
    frame.loc[::11, "number"] = np.nan
    frame.loc[128:, "category"] = "holdout_only"
    if task_type == "classification":
        labels = np.asarray(["001", "NA"] if binary else ["001", "NA", "Attack"])
        target = labels[np.arange(len(frame)) % len(labels)]
    else:
        target = number * 2 + rng.normal(scale=0.1, size=len(frame))
    return frame.iloc[:128].copy(), target[:128], frame.iloc[128:].copy(), target[128:]


def _worker(candidate, task_type, binary, directory):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agent.models import build_pipeline, predict_for_metric

    fit, y_fit, held, actual = smoke_data(task_type, binary)
    model = build_pipeline(candidate["action"], task_type, ["number", "empty"], ["category"],
                           candidate["model_config"], candidate["feature_config"])
    with threadpool_limits(limits=8):
        model.fit(fit, y_fit)
        prediction = predict_for_metric(model, held, "LogLoss" if task_type == "classification" else "RMSE")
    table = pd.DataFrame({"id": [f"{i:05d}" for i in range(len(held))], "actual": actual})
    classes = tuple(map(str, model.classes_)) if task_type == "classification" else ()
    if classes:
        for i in range(len(classes)):
            table[f"probability_{i}"] = prediction[:, i]
    else:
        table["prediction"] = np.asarray(prediction)
    directory = Path(directory)
    table.to_csv(directory / "predictions.csv", index=False)
    (directory / "classes.json").write_text(json.dumps(classes), encoding="utf-8")


def score_saved(directory, task_type, metric, binary=False):
    """Re-read persisted predictions and verify IDs/answers against the fixture."""
    import numpy as np
    import pandas as pd

    _, y_fit, held, actual = smoke_data(task_type, binary)
    table = pd.read_csv(Path(directory) / "predictions.csv", dtype=str, keep_default_na=False)
    if table["id"].tolist() != [f"{i:05d}" for i in range(len(held))]:
        raise ValueError("Smoke prediction IDs do not match the frozen holdout")
    classes = tuple(json.loads((Path(directory) / "classes.json").read_text(encoding="utf-8")))
    if task_type == "classification":
        if table["actual"].tolist() != actual.tolist() or set(classes) != set(y_fit) or len(classes) != len(set(classes)):
            raise ValueError("Smoke class evidence does not match the fixture")
        probabilities = table[[f"probability_{i}" for i in range(len(classes))]].to_numpy(float)
        # Validate the matrix even when the selected metric consumes labels.
        from agent.metrics import probability_matrix
        probabilities = probability_matrix(probabilities, classes=classes, rows=len(held))
        prediction = (np.asarray(classes)[probabilities.argmax(axis=1)]
                      if get_metric(metric).prediction_kind == "labels" else probabilities)
        return score_predictions(metric, actual, prediction, classes=classes,
                                 prediction_classes=classes, positive_class="001" if binary else None)
    if classes or not np.array_equal(table["actual"].to_numpy(float), actual):
        raise ValueError("Smoke actual values do not match the fixture")
    return score_predictions(metric, actual, table["prediction"].to_numpy(float))


def run_smoke(profile, output, *, available_actions=ACTIONS, limit=8, binary=False,
              attempt_seconds=30.0, total_seconds=120.0):
    from agent.execution import run_python

    candidates = portfolio_candidates(profile, available_actions=available_actions, limit=limit)
    for value in (attempt_seconds, total_seconds):
        try:
            valid = type(value) in {int, float} and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Smoke time limits must be positive finite numbers")
    if type(binary) is not bool or (binary and profile["task_type"] != "classification"):
        raise ValueError("binary is supported only for classification")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "synthetic_execution_smoke", "seed": SEED, "profile": dict(profile),
              "binary": binary, "attempt_seconds": attempt_seconds, "total_seconds": total_seconds,
              "api_calls": 0, "candidates": candidates, "results": []}
    deadline = time.monotonic() + total_seconds
    for candidate in candidates:
        remaining = deadline - time.monotonic()
        item = {"id": candidate["id"], "status": "budget_skipped"}
        if remaining > 0:
            directory = output / candidate["id"]
            directory.mkdir()
            code = (f"import sys, json; sys.path.insert(0, {str(ROOT)!r})\n"
                    "from experiment_portfolio_recipes import _worker\n"
                    f"_worker(json.loads({json.dumps(candidate)!r}), {profile['task_type']!r}, "
                    f"{binary!r}, {str(directory)!r})\n")
            execution = run_python(code, directory, directory, 1,
                                   timeout=min(attempt_seconds, remaining), max_memory_bytes=8 * 1024**3)
            item.update(status="failed", execution=execution)
            if (execution.get("exit_code") == 0 and not execution.get("error")
                    and not execution.get("timed_out") and not execution.get("memory_limit_exceeded")):
                try:
                    item.update(score=score_saved(directory, profile["task_type"], profile["metric"], binary),
                                status="completed")
                except (OSError, ValueError, KeyError) as error:
                    item["error"] = str(error)
        report["results"].append(item)
        (output / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    if not candidates:
        (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _json(path, value):
    """Atomic incremental evidence; no NaN or implicit object conversions."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_dataset(task_path=None, *, builtin=None):
    """Load raw features once; only the host retains validation truths."""
    import numpy as np
    import pandas as pd
    from agent.task_spec import load_task, load_task_data

    if (task_path is None) == (builtin is None):
        raise ValueError("Provide exactly one task path or builtin dataset")
    if builtin is not None:
        from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
        loaders = {"breast_cancer": load_breast_cancer, "wine": load_wine, "diabetes": load_diabetes}
        if builtin not in loaders:
            raise ValueError("Unsupported builtin dataset")
        bunch = loaders[builtin](as_frame=True)
        X = bunch.data.copy()
        classification = builtin != "diabetes"
        y = np.asarray(bunch.target, dtype=str if classification else float)
        ids = [f"{builtin}_{i:06d}" for i in range(len(X))]
        features, numeric, categorical = list(X.columns), list(X.columns), []
        metric = "Macro F1" if classification else "RMSE"
        classes = tuple(sorted(set(y))) if classification else ()
        positive = "1" if len(classes) == 2 else None
        provenance, fingerprints = {"builtin": builtin}, {}
    else:
        spec = load_task(Path(task_path))
        train, _, _, _, _, target, features, numeric, categorical, classification = load_task_data(spec)
        # Grouping must use these exact raw columns, before any feature coercion.
        X = train.loc[:, list(features)].copy()
        y = np.asarray(target, dtype=str if classification else float)
        ids = train[spec.id_column].astype(str).tolist()
        metric, classes, positive = spec.metric.name, spec.classes, spec.positive_class
        paths = [spec.task_path, *sorted(spec.data_dir.glob("*.csv"))]
        fingerprints = {str(path.resolve()): _file_hash(path) for path in paths}
        provenance = {"task": str(spec.task_path), "task_spec": spec.to_dict(),
                      "prior_exposure": "public train already researched; regression periods 3-5 and classification seed42 seen"}
    if len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
        raise ValueError("Dataset IDs must be unique and nonempty")
    if len(y) != len(X) or not len(X) or (not classification and not np.isfinite(y).all()):
        raise ValueError("Dataset target is empty, nonfinite, or has the wrong length")
    frame_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
    group_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).values.tobytes()).hexdigest() if classification else None
    fingerprint = _digest({"features": list(features), "dtypes": list(map(str, X.dtypes)), "raw_rows": frame_hash,
                           "ids": ids, "target": y.tolist()})
    return {"X": X, "y": y, "ids": ids, "features": list(features), "numeric": list(numeric),
            "categorical": list(categorical), "profile": {"task_type": "classification" if classification else "regression",
            "metric": metric}, "classes": tuple(classes), "positive_class": positive,
            "time_column": "Year_Factor" if not classification and builtin is None else None,
            "fingerprint": fingerprint, "raw_feature_groups_sha256": group_hash,
            "fingerprints": fingerprints, "provenance": provenance}


def freeze_splits(dataset, *, seeds=(1729, 2718), outer_folds=(0,), outer_years=(3, 4, 5)):
    """Fixed SGKF(5) folds, rolling past-only years, or builtin regression holdouts.

    Impossible requested outer splits are recorded. An impossible inner split
    retains the predeclared first candidate; no other seed/fold is searched.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    for values, name in ((seeds, "seeds"), (outer_folds, "outer_folds")):
        if not isinstance(values, (tuple, list)) or not values or any(type(v) is not int for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain distinct integers")
    if any(seed < 0 or seed >= 2**32 for seed in seeds) or any(fold not in range(5) for fold in outer_folds):
        raise ValueError("Seeds must fit uint32; outer folds must be between zero and four")
    if (not isinstance(outer_years, (tuple, list)) or not outer_years
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in outer_years)
            or len(set(outer_years)) != len(outer_years)):
        raise ValueError("outer_years must contain distinct finite numeric periods")
    X, y = dataset["X"], dataset["y"]
    rows, result = np.arange(len(X)), []
    classification = dataset["profile"]["task_type"] == "classification"
    groups = pd.util.hash_pandas_object(X.loc[:, dataset["features"]], index=False).to_numpy() if classification else None
    public_time = not classification and "task" in dataset["provenance"]
    if public_time:
        if "Year_Factor" not in dataset["features"]:
            raise ValueError("Public regression requires raw feature Year_Factor")
        periods = pd.to_numeric(X["Year_Factor"], errors="coerce").to_numpy(float)
        if not np.isfinite(periods).all():
            raise ValueError("Year_Factor must be finite numeric periods")
        requests = [(f"year_{year:g}", None, None, year) for year in outer_years]
    else:
        requests = [(f"seed_{seed}_fold_{fold}", seed, fold, None) for seed in seeds
                    for fold in (outer_folds if classification else (0,))]
    for name, seed, fold, year in requests:
        item = {"id": name, "seed": seed, "fold": fold, "year": year, "status": "ready", "inner_status": "ready",
                "outer_fit": [], "outer_held": [], "inner_fit": [], "inner_held": []}
        try:
            if classification:
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                fit, held = list(splitter.split(rows, y, groups))[fold]
            elif public_time:
                fit, held = rows[periods < year], rows[periods == year]
            else:
                fit, held = train_test_split(rows, test_size=.2, random_state=seed)
            if not len(fit) or not len(held):
                raise ValueError("Requested outer partition is empty")
            if classification:
                if set(y[fit]) != set(dataset["classes"]) or set(y[held]) != set(dataset["classes"]):
                    raise ValueError("Requested outer fold does not contain every class in both parts")
                if set(groups[fit]) & set(groups[held]):
                    raise ValueError("Raw feature groups overlap in outer split")
                item["outer_group_overlap"] = 0
                item["outer_class_support"] = {label: int(sum(y[held] == label)) for label in dataset["classes"]}
            item.update(outer_fit=fit.tolist(), outer_held=held.tolist())
            try:
                if classification:
                    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    a, b = list(splitter.split(fit, y[fit], groups[fit]))[0]
                    a, b = fit[a], fit[b]
                    if set(y[a]) != set(dataset["classes"]) or set(y[b]) != set(dataset["classes"]):
                        raise ValueError("Fixed inner fold does not contain every class in both parts")
                    if set(groups[a]) & set(groups[b]):
                        raise ValueError("Raw feature groups overlap in inner split")
                    item["inner_group_overlap"] = 0
                elif public_time:
                    latest = periods[fit].max()
                    a, b = fit[periods[fit] < latest], fit[periods[fit] == latest]
                    item["inner_year"] = float(latest)
                else:
                    a, b = train_test_split(fit, test_size=.2, random_state=seed)
                if len(a) < 2 or len(b) < 2:
                    raise ValueError("Insufficient history/rows for fixed inner split")
                item.update(inner_fit=a.tolist(), inner_held=b.tolist())
            except ValueError as error:
                item.update(inner_status="impossible", inner_error=str(error))
        except ValueError as error:
            item.update(status="impossible", error=str(error))
        item["split_hash"] = _digest({key: item[key] for key in ("seed", "fold", "year", "outer_fit", "outer_held", "inner_fit", "inner_held")})
        result.append(item)
    return result


def _settings(*, available_actions=ACTIONS, seeds=(1729, 2718), outer_folds=(0,), outer_years=(3, 4, 5),
              search_seconds=600, attempt_seconds=180, refit_seconds=180, memory_gib=8, diagnostics_seconds=600):
    from agent.resources import DEFAULT_MEMORY_LIMIT_BYTES
    values = locals().copy()
    for name in ("search_seconds", "attempt_seconds", "refit_seconds", "memory_gib", "diagnostics_seconds"):
        value = values[name]
        try:
            valid = type(value) in {int, float} and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a positive finite number")
    if int(memory_gib * 1024**3) < 1 or memory_gib * 1024**3 > DEFAULT_MEMORY_LIMIT_BYTES:
        raise ValueError("Memory limit exceeds existing resource helper default")
    return {key: value for key, value in values.items() if key != "DEFAULT_MEMORY_LIMIT_BYTES"}


def prepare_comparison(task_path, output, *, builtin=None, **settings):
    """Freeze a new manifest before any fit; returns (host dataset, manifest)."""
    from agent.features import normalize_feature_config
    from agent.search import exploration_plan

    config = _settings(**settings)
    dataset = comparison_dataset(task_path, builtin=builtin)
    # Validate caller availability even when all actions are unavailable.
    portfolio_candidates(dataset["profile"], available_actions=config["available_actions"])
    portfolios = {"exploration_plan": exploration_plan(dataset["profile"]["task_type"] == "classification"),
                  "portfolio_v1": portfolio_candidates(dataset["profile"], available_actions=ACTIONS)}
    splits = freeze_splits(dataset, seeds=config["seeds"], outer_folds=config["outer_folds"], outer_years=config["outer_years"])
    ablations = []
    for suffix in ("lgbm_conservative",):
        candidate = next(row for row in portfolios["portfolio_v1"] if row["id"].endswith(suffix))
        for label, options in (("raw", {}), ("missing_only", {"missing_indicators": True}),
                               ("rare_only", {"rare_min_count": 10}),
                               ("both", {"missing_indicators": True, "rare_min_count": 10})):
            ablations.append({**deepcopy(candidate), "id": candidate["id"] + "_ablation_" + label,
                              "feature_config": normalize_feature_config(options)})
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "threadpoolctl", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = [Path(__file__).resolve(), ROOT / "offline_baseline.py", *sorted((ROOT / "agent").glob("*.py"))]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"version": 2, "api_calls": 0, "purpose": "nested_frozen_portfolio_comparison_not_full_production_automl",
                "commit": commit, "python": sys.version, "packages": packages, "source_root": str(ROOT),
                "source_hashes": {str(path): _file_hash(path) for path in sources}, "settings": config,
                "provenance": dataset["provenance"], "input_fingerprints": dataset["fingerprints"],
                "dataset_fingerprint": dataset["fingerprint"], "raw_feature_groups_sha256": dataset["raw_feature_groups_sha256"],
                "profile": dataset["profile"],
                "features": dataset["features"], "numeric": dataset["numeric"], "categorical": dataset["categorical"],
                "class_order": list(dataset["classes"]), "positive_class": dataset["positive_class"],
                "portfolios": portfolios, "baseline_reference": deepcopy(portfolios["exploration_plan"][0]),
                "diagnostic_ablations": ablations, "splits": splits,
                "selection_rule": "Best host-rescored completed inner score; trusted metric direction; ties use predeclared candidate order. Exactly one outer refit, no fallback after refit failure.",
                "inner_impossible_rule": "Freeze first available predeclared member without search; explicitly label untuned fallback. No successful inner trial means no outer refit.",
                "budget_policy": "Sequential exploration search then portfolio search, independent equal wall-clock deadlines including subprocess/scoring overhead; attempt timeout=min(attempt,remaining), less host preparation time. Existing process-tree cleanup can overshoot the deadline and its full cost is recorded. One separately budgeted refit per winner; baseline reference separate refit cost. Diagnostics only after both choices frozen, separate budget, never change winner.",
                "resource_policy": {"model_n_jobs": 8, "blas_openmp_threads": 8, "logistic_n_jobs": 1,
                                    "max_memory_bytes": int(config["memory_gib"] * 1024**3)},
                "split_policy": "Classification raw-feature duplicate groups, SGKF5 requested outer fold and fixed inner fold0, same declared seed. Public regression Year_Factor rolling origin with latest earlier inner holdout. Builtin regression fixed seeded 80/20 outer and inner."}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for path in sources:
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    _json(output / "manifest.json", manifest)
    return dataset, manifest


def _bundle(dataset, fit, held, directory):
    """One shared fold-local bundle: worker never receives held-out targets."""
    import numpy as np
    directory.mkdir(parents=True)
    dataset["X"].iloc[fit].to_pickle(directory / "fit_features.pkl")
    dataset["X"].iloc[held].to_pickle(directory / "held_features.pkl")
    np.save(directory / "fit_target.npy", dataset["y"][fit], allow_pickle=False)
    payload = {"fit_features": str(directory / "fit_features.pkl"), "fit_target": str(directory / "fit_target.npy"),
               "held_features": str(directory / "held_features.pkl"), "fit_indices": fit, "held_indices": held,
               "held_ids": [dataset["ids"][i] for i in held], "numeric": dataset["numeric"],
               "categorical": dataset["categorical"], "task_type": dataset["profile"]["task_type"]}
    _json(directory / "input.json", payload)
    return payload


def _comparison_worker(payload_path):
    """Isolated single fit/predict, warning evidence survives ordinary failures."""
    import traceback
    import warnings
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agent.models import build_pipeline, predict_for_metric

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    directory, candidate = Path(payload_path).parent, payload["candidate"]
    started, cpu = time.monotonic(), time.process_time()
    evidence = {"fit_seconds": None, "predict_seconds": None, "warnings": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit, held = pd.read_pickle(payload["fit_features"]), pd.read_pickle(payload["held_features"])
            target = np.load(payload["fit_target"], allow_pickle=False)
            model = build_pipeline(candidate["action"], payload["task_type"], payload["numeric"], payload["categorical"],
                                   candidate["model_config"], candidate["feature_config"])
            with threadpool_limits(limits=8):
                mark = time.monotonic()
                model.fit(fit, target)
                evidence["fit_seconds"] = time.monotonic() - mark
                mark = time.monotonic()
                prediction = predict_for_metric(model, held, "LogLoss" if payload["task_type"] == "classification" else "RMSE")
                labels = model.predict(held) if payload["task_type"] == "classification" else None
                evidence["predict_seconds"] = time.monotonic() - mark
            classes = list(map(str, model.classes_)) if payload["task_type"] == "classification" else []
            table = pd.DataFrame({"id": payload["held_ids"], "position": payload["held_indices"]})
            if classes:
                table["label"] = labels
                for index in range(len(classes)):
                    table[f"probability_{index}"] = prediction[:, index]
            else:
                table["prediction"] = prediction
            table.to_csv(directory / "raw_predictions.csv", index=False)
            _json(directory / "classes.json", classes)
        except Exception:
            evidence["error"] = traceback.format_exc()
            raise
        finally:
            evidence.update(total_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu,
                            warnings=[{"category": item.category.__name__, "message": str(item.message)} for item in caught])
            _json(directory / "worker.json", evidence)


def score_trial(directory, dataset, held_indices):
    """Host rereads immutable IDs/truth, validates full probabilities, and scores."""
    import numpy as np
    import pandas as pd
    from agent.metrics import probability_matrix

    directory = Path(directory)
    table = pd.read_csv(directory / "predictions.csv", dtype=str, keep_default_na=False)
    expected_ids = [dataset["ids"][i] for i in held_indices]
    if table["id"].tolist() != expected_ids or table["position"].tolist() != list(map(str, held_indices)):
        raise ValueError("Prediction IDs/positions differ from frozen held rows")
    actual, classification = dataset["y"][held_indices], bool(dataset["classes"])
    classes = tuple(json.loads((directory / "classes.json").read_text(encoding="utf-8")))
    if classification:
        if table["actual"].tolist() != actual.tolist() or len(classes) != len(set(classes)) or set(classes) != set(dataset["classes"]):
            raise ValueError("Prediction actuals/class order differ from frozen task")
        probabilities = probability_matrix(table[[f"probability_{i}" for i in range(len(classes))]].to_numpy(float),
                                           classes=dataset["classes"], prediction_classes=classes, rows=len(actual))
        labels = table["label"].to_numpy()
        if not set(labels) <= set(dataset["classes"]):
            raise ValueError("Predicted labels contain an unknown class")
        prediction = labels if get_metric(dataset["profile"]["metric"]).prediction_kind == "labels" else probabilities
        return score_predictions(dataset["profile"]["metric"], actual, prediction, classes=dataset["classes"],
                                 prediction_classes=dataset["classes"], positive_class=dataset["positive_class"])
    if classes or not np.array_equal(table["actual"].to_numpy(float), actual):
        raise ValueError("Regression actuals differ from frozen held rows")
    return score_predictions(dataset["profile"]["metric"], actual, table["prediction"].to_numpy(float))


def _attempt(candidate, bundle, directory, dataset, *, timeout, manifest, split, phase):
    started = time.monotonic()
    from agent.execution import run_python
    import pandas as pd

    directory.mkdir(parents=True)
    payload = {**bundle, "candidate": candidate}
    _json(directory / "input.json", payload)
    item = {"id": candidate["id"], "candidate": candidate, "phase": phase, "split_hash": split["split_hash"],
            "seed": split["seed"], "fit_indices": bundle["fit_indices"], "held_indices": bundle["held_indices"],
            "directory": str(directory), "timeout_seconds": max(0, timeout), "status": "budget_skipped",
            "worker": {"fit_seconds": None, "predict_seconds": None, "total_seconds": None,
                       "process_cpu_seconds": None, "warnings": [], "unavailable_reason": "Worker did not complete"},
            "execution": None}
    if candidate["action"] not in manifest["settings"]["available_actions"]:
        item["status"] = "unavailable"
    elif timeout > 0:
        try:
            changed = [path for path, digest in manifest["source_hashes"].items() if _file_hash(path) != digest]
            if changed:
                raise ValueError("Frozen source changed: " + ", ".join(changed))
            source = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
                      "from experiment_portfolio_recipes import _comparison_worker\n"
                      f"_comparison_worker({str(directory / 'input.json')!r})\n")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                item["host_elapsed_seconds"] = time.monotonic() - started
                _json(directory / "trial.json", item)
                return item
            item["child_timeout_seconds"] = remaining
            execution = run_python(source, directory, directory, 1, timeout=remaining,
                                   max_memory_bytes=manifest["resource_policy"]["max_memory_bytes"])
            item.update(execution=execution, status="failed")
            if (directory / "worker.json").exists():
                item["worker"] = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
            if execution.get("timed_out"):
                item["status"] = "timeout"
            elif execution.get("memory_limit_exceeded"):
                item["status"] = "oom"
            elif execution.get("exit_code") == 0 and not execution.get("error"):
                table = pd.read_csv(directory / "raw_predictions.csv", dtype=str, keep_default_na=False)
                table.insert(2, "actual", dataset["y"][bundle["held_indices"]])
                table.to_csv(directory / "predictions.csv", index=False)
                value = score_trial(directory, dataset, bundle["held_indices"])
                # Reading and independently checking the CSV is part of the
                # attempt budget too. A late score remains evidence, not a
                # candidate eligible to win after the search deadline.
                item.update(score=value, status=("completed" if time.monotonic() - started <= timeout
                                                else "over_budget"))
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
    item["host_elapsed_seconds"] = time.monotonic() - started
    _json(directory / "trial.json", item)
    return item


def run_comparison(task_path, output, *, builtin=None, **settings):
    """Run fixed, paired searches sequentially; freeze both choices before outer."""
    dataset, manifest = prepare_comparison(task_path, output, builtin=builtin, **settings)
    output, config = Path(output).resolve(), manifest["settings"]
    report = {"version": 2, "manifest_sha256": _file_hash(output / "manifest.json"), "api_calls": 0,
              "profile": dataset["profile"], "status": "running", "splits": []}
    _json(output / "report.json", report)
    for split in manifest["splits"]:
        row = {"id": split["id"], "split_hash": split["split_hash"], "status": split["status"],
               "portfolios": {}, "baseline_reference": None, "diagnostics": []}
        report["splits"].append(row)
        _json(output / "report.json", report)
        if split["status"] != "ready":
            row["error"] = split["error"]
            continue
        directory = output / split["id"]
        preparation_started = time.monotonic()
        inner = (_bundle(dataset, split["inner_fit"], split["inner_held"], directory / "inner")
                 if split["inner_status"] == "ready" else None)
        outer = _bundle(dataset, split["outer_fit"], split["outer_held"], directory / "outer")
        row["shared_input_preparation_seconds"] = time.monotonic() - preparation_started
        frozen = {}
        for name, candidates in manifest["portfolios"].items():
            phase = {"search_budget_seconds": config["search_seconds"], "trials": [], "selected": None}
            row["portfolios"][name] = phase
            started = time.monotonic()
            if inner is not None:
                for candidate in candidates:
                    remaining = config["search_seconds"] - (time.monotonic() - started)
                    trial = _attempt(candidate, inner, directory / name / "search" / candidate["id"], dataset,
                                     timeout=min(config["attempt_seconds"], remaining), manifest=manifest, split=split, phase="inner_search")
                    phase["trials"].append(trial)
                    _json(output / "report.json", report)
                completed = [trial for trial in phase["trials"] if trial["status"] == "completed"]
                if completed:
                    direction = -1 if get_metric(dataset["profile"]["metric"]).higher_is_better else 1
                    winner = min(completed, key=lambda trial: direction * trial["score"])
                    phase["selected"] = winner["id"]
                    phase["selection_status"] = "inner_selected"
                    frozen[name] = deepcopy(winner["candidate"])
                else:
                    phase["selection_status"] = "no_completed_inner_trial_no_outer_refit"
            else:
                eligible = [candidate for candidate in candidates if candidate["action"] in config["available_actions"]]
                if eligible:
                    frozen[name] = deepcopy(eligible[0])
                    phase["selected"] = eligible[0]["id"]
                phase.update(selection_status="predeclared_untuned_inner_impossible", inner_error=split["inner_error"])
            phase["search_elapsed_seconds"] = time.monotonic() - started
        # Persist the irrevocable choice before diagnostics or any outer refit.
        _json(directory / "frozen_winners.json", {"split_hash": split["split_hash"], "winners": frozen,
                                                 "selection": {name: item["selection_status"] for name, item in row["portfolios"].items()}})
        _json(output / "report.json", report)
        for name, candidate in frozen.items():
            row["portfolios"][name]["outer"] = _attempt(candidate, outer, directory / name / "outer", dataset,
                timeout=config["refit_seconds"], manifest=manifest, split=split, phase="selected_outer_refit")
            _json(output / "report.json", report)
        row["baseline_reference"] = _attempt(manifest["baseline_reference"], outer, directory / "baseline_reference", dataset,
            timeout=config["refit_seconds"], manifest=manifest, split=split, phase="separate_baseline_reference")
        if inner is not None:
            started = time.monotonic()
            for candidate in manifest["diagnostic_ablations"]:
                remaining = config["diagnostics_seconds"] - (time.monotonic() - started)
                row["diagnostics"].append(_attempt(candidate, inner, directory / "diagnostics" / candidate["id"], dataset,
                    timeout=min(config["attempt_seconds"], remaining), manifest=manifest, split=split, phase="postfreeze_inner_diagnostic"))
                _json(output / "report.json", report)
            row["diagnostics_elapsed_seconds"] = time.monotonic() - started
        row["status"] = "finished"
        _json(output / "report.json", report)
    changed_inputs = [path for path, digest in manifest["input_fingerprints"].items() if _file_hash(path) != digest]
    changed_sources = [path for path, digest in manifest["source_hashes"].items() if _file_hash(path) != digest]
    report.update(status="finished" if not changed_inputs and not changed_sources else "invalidated",
                  inputs_unchanged=not changed_inputs, sources_unchanged=not changed_sources,
                  changed_inputs=changed_inputs, changed_sources=changed_sources)
    _json(output / "report.json", report)
    _json(output / "summary.json", summarize_comparison(output))
    return report


def summarize_comparison(output):
    """Derive costs and paired metrics from saved evidence without any fitting."""
    from collections import Counter
    import numpy as np
    import pandas as pd
    from agent.metrics import probability_matrix

    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    profile, classes = manifest["profile"], tuple(manifest["class_order"])
    metric = get_metric(profile["metric"])

    def trial_summary(item, secondary=False):
        if not item:
            return None
        worker, execution = item.get("worker") or {}, item.get("execution") or {}
        resources = execution.get("resources") or {}
        result = {"id": item["id"], "status": item["status"], "primary_score": item.get("score"),
                  "host_seconds": item.get("host_elapsed_seconds"), "fit_seconds": worker.get("fit_seconds"),
                  "predict_seconds": worker.get("predict_seconds"), "worker_total_seconds": worker.get("total_seconds"),
                  "worker_cpu_seconds": worker.get("process_cpu_seconds"), "warnings": worker.get("warnings", []),
                  "resources": {**resources, "available": bool(resources.get("available")),
                                "peak_rss_bytes": resources.get("peak_rss_bytes") if resources.get("available") else None,
                                "cpu_seconds": resources.get("cpu_seconds") if resources.get("available") else None}}
        if secondary and item["status"] in {"completed", "over_budget"}:
            try:
                directory = Path(item["directory"])
                table = pd.read_csv(directory / "predictions.csv", dtype=str, keep_default_na=False)
                native = tuple(json.loads((directory / "classes.json").read_text(encoding="utf-8")))
                if table["id"].duplicated().any() or table["id"].str.strip().eq("").any():
                    raise ValueError("Saved prediction IDs are empty or duplicated")
                if "held_indices" in item and table["position"].tolist() != list(map(str, item["held_indices"])):
                    raise ValueError("Saved positions differ from frozen trial rows")
                actual = table["actual"].to_numpy()
                if classes:
                    probabilities = probability_matrix(table[[f"probability_{i}" for i in range(len(native))]].to_numpy(float),
                        classes=classes, prediction_classes=native, rows=len(table))
                    prediction = table["label"].to_numpy()
                    names = ("Macro F1", "Weighted F1", "Accuracy", "ROC AUC", "LogLoss")
                    result["class_order"], result["native_class_order"] = list(classes), list(native)
                    per_class = {}
                    for label in classes:
                        tp = int(np.sum((actual == label) & (prediction == label)))
                        fp = int(np.sum((actual != label) & (prediction == label)))
                        fn = int(np.sum((actual == label) & (prediction != label)))
                        per_class[label] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                            "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
                    result["per_class"] = per_class
                else:
                    if native:
                        raise ValueError("Regression artifact unexpectedly declares classes")
                    actual, prediction = actual.astype(float), table["prediction"].to_numpy(float)
                    names = ("RMSE", "MAE", "MSE", "R2")
                scores, errors = {}, {}
                for name in names:
                    try:
                        values = probabilities if classes and get_metric(name).prediction_kind == "probabilities" else prediction
                        scores[name] = score_predictions(name, actual, values, classes=classes,
                            prediction_classes=classes if classes else None, positive_class=manifest.get("positive_class"))
                    except ValueError as error:
                        errors[name] = str(error)
                result.update(secondary_metrics=scores, secondary_errors=errors,
                              predictions_sha256=_file_hash(directory / "predictions.csv"))
                if metric.name in scores:
                    result["recomputed_primary_score"] = scores[metric.name]
                    if item.get("score") is not None and not math.isclose(scores[metric.name], item["score"], rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Saved predictions no longer reproduce the reported primary score")
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["artifact_error"] = f"{type(error).__name__}: {error}"
        return result

    def search_stats(trials):
        counts = dict(Counter(item["status"] for item in trials))
        workers = [item.get("worker") or {} for item in trials]
        stats = {"status_counts": counts, "completed": counts.get("completed", 0),
                 "attempted": sum(item["status"] not in {"budget_skipped", "unavailable"} for item in trials),
                 "launched": sum(bool(item.get("execution")) for item in trials), "planned": len(trials),
                 "warnings": [warning for worker in workers for warning in worker.get("warnings", [])],
                 "resource_available_trials": sum(bool((item.get("execution") or {}).get("resources", {}).get("available")) for item in trials)}
        for name in ("fit_seconds", "predict_seconds", "total_seconds", "process_cpu_seconds"):
            values = [worker[name] for worker in workers if worker.get(name) is not None]
            stats[name] = sum(values) if values else None
            stats[name + "_available_trials"] = len(values)
        return stats

    summary = {"version": 1, "status": report["status"], "profile": profile,
               "provenance": manifest["provenance"], "dataset_fingerprint": manifest["dataset_fingerprint"],
               "manifest_sha256": _file_hash(output / "manifest.json"), "splits": [],
               "paired_improvement_rule": "Positive favors portfolio_v1: new-old for higher-is-better, old-new otherwise; completed paired outer refits only.",
               "resource_note": "Configured memory is distinct from measured RSS. Unavailable process-tree telemetry remains null; worker CPU measures the root worker only."}
    for row in report["splits"]:
        result = {"id": row["id"], "status": row["status"], "portfolios": {}, "paired_improvement": None}
        for name, phase in row["portfolios"].items():
            selected = next((trial for trial in phase["trials"] if trial["id"] == phase.get("selected")), None)
            result["portfolios"][name] = {"selected": phase.get("selected"), "selection_status": phase.get("selection_status"),
                "inner_score": selected.get("score") if selected else None, "search_seconds": phase.get("search_elapsed_seconds"),
                "search_stats": search_stats(phase["trials"]), "outer": trial_summary(phase.get("outer"), secondary=True)}
        result["baseline_reference"] = trial_summary(row.get("baseline_reference"), secondary=True)
        result["diagnostics"] = {"scope": "postfreeze_inner_diagnostic_only_not_selection", "seconds": row.get("diagnostics_elapsed_seconds"),
                                 "trials": [trial_summary(item) for item in row.get("diagnostics", [])]}
        old, new = (result["portfolios"].get(name, {}).get("outer") for name in ("exploration_plan", "portfolio_v1"))
        if all(item and item["status"] == "completed" and item["primary_score"] is not None and not item.get("artifact_error") for item in (old, new)):
            result["paired_improvement"] = (new["primary_score"] - old["primary_score"]) * (1 if metric.higher_is_better else -1)
        summary["splits"].append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-type", choices=("regression", "classification"), help="Synthetic execution smoke")
    source.add_argument("--task", type=Path, help="Public task.md; grouped classification or Year_Factor regression")
    source.add_argument("--builtin", choices=("breast_cancer", "wine", "diabetes"), help="Paired sklearn transfer check")
    parser.add_argument("--metric", help="Default: RMSE for regression, LogLoss for classification")
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=list(ACTIONS))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--attempt-seconds", type=float, default=None)
    parser.add_argument("--total-seconds", type=float, default=120)
    parser.add_argument("--search-seconds", type=float, default=600)
    parser.add_argument("--refit-seconds", type=float, default=180)
    parser.add_argument("--diagnostics-seconds", type=float, default=600)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--outer-years", type=float, nargs="+", default=[3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718])
    parser.add_argument("--outer-folds", type=int, nargs="+", default=[0], help="Fixed SGKF5 fold numbers, without retry")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze manifest without any training")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    try:
        if not args.task_type:
            runner = prepare_comparison if args.prepare_only else run_comparison
            result = runner(args.task, args.out_dir, builtin=args.builtin, available_actions=tuple(args.actions),
                seeds=tuple(args.seeds), outer_folds=tuple(args.outer_folds), outer_years=tuple(args.outer_years),
                search_seconds=args.search_seconds, attempt_seconds=180 if args.attempt_seconds is None else args.attempt_seconds,
                refit_seconds=args.refit_seconds, memory_gib=args.memory_gib, diagnostics_seconds=args.diagnostics_seconds)
            print(json.dumps({"output": str(args.out_dir.resolve()), "status": "prepared" if args.prepare_only else result["status"]}))
            return 0 if args.prepare_only or result["status"] == "finished" else 1
        report = run_smoke({"task_type": args.task_type,
                            "metric": args.metric or ("RMSE" if args.task_type == "regression" else "LogLoss")},
                           args.out_dir, available_actions=tuple(args.actions), limit=args.limit, binary=args.binary,
                           attempt_seconds=30 if args.attempt_seconds is None else args.attempt_seconds, total_seconds=args.total_seconds)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for item in report["results"]:
        print(item["id"], item["status"], item.get("score", ""))
    return 0 if report["results"] and all(item["status"] == "completed" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
