"""Bounded v3 classification candidate: fit-only features and guarded OOF.

All model calls, including the final fresh fit, run sequentially through the
existing subprocess supervisor. Nested callbacks cannot extend the original
host deadline. The public output remains Macro F1 labels.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from agent import project_root
from agent.classification_decision import DEFAULT_GUARD_OPTIONS, fit_guarded_decision_rule, predict_with_rule
from agent.execution import run_python
from agent.model_config import make_model_config
from agent.metrics import get_metric
from agent.oof_classification import fit_oof_guarded_rule
from agent.round4_classification_probability import make_probability_recipe, fit_predict_probabilities
from agent.splits import split_rows

VARIANTS = ("identity", "oof", "frequency", "asymmetry", "group_weight", "extra_trees_asymmetry")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def make_recipe(base_recipe, classes, *, variant="identity", asymmetry_pairs=()):
    from agent.round4_classification_features import normalize_options
    if variant not in VARIANTS:
        raise ValueError("Unknown round4 variant")
    pairs = deepcopy(list(asymmetry_pairs)) if variant in {"asymmetry", "extra_trees_asymmetry"} else []
    if variant in {"asymmetry", "extra_trees_asymmetry"} and not pairs:
        raise ValueError("Asymmetry requires explicit schema pairs")
    if variant == "extra_trees_asymmetry":
        base_recipe = {"action": "extra_trees", "model_config": make_model_config(
            "extra_trees", "classification", {"n_estimators": 300, "max_depth": 18,
            "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced"})}
    probability = make_probability_recipe(base_recipe, classes)
    recipe = {"version": 1, "family": "round4_classification_v1", "variant": variant,
              "probability_recipe": probability,
              "features": normalize_options({"frequency": variant == "frequency", "asymmetry_pairs": pairs,
                           "group_weight": variant == "group_weight"}),
              "split": {"strategy": "grouped", "inner_seed": 42, "oof_folds": 3, "oof_seed": 42},
              "guard_options": deepcopy(DEFAULT_GUARD_OPTIONS),
              "caps": {"trial_seconds": 600 if variant == "oof" else 300,
                       "callback_seconds": 90, "final_reserve_seconds": 120,
                       "max_physical_fits": len(probability["base_recipe"]["components"]) * (5 if variant == "oof" else 1),
                       "memory_limit_bytes": 6 * 1024 ** 3, "max_threads": 8}}
    return json.loads(json.dumps(recipe, allow_nan=False))


def checked_recipe(value):
    if not isinstance(value, dict) or type(value.get("version")) is not int:
        raise ValueError("Invalid round4 recipe")
    try:
        normalized = make_recipe(value["probability_recipe"]["base_recipe"], value["probability_recipe"]["classes"],
                                 variant=value["variant"], asymmetry_pairs=value["features"]["asymmetry_pairs"])
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed round4 recipe") from error
    if value != normalized:
        raise ValueError("Recipe differs from fixed round4 policy")
    return normalized


def coverage(X, y, context):
    """Map raw groups, row support and independent-group support before fitting."""
    from sklearn.model_selection import StratifiedGroupKFold
    features, classes = list(context["features"]), list(context["classes"])
    if list(X.columns) != features or len(X) != len(y):
        raise ValueError("Coverage requires exact feature rows and aligned fit labels")
    labels = np.asarray(y, dtype=object)
    if not set(labels) <= set(classes):
        raise ValueError("Coverage labels are undeclared")
    groups = pd.util.hash_pandas_object(X.loc[:, features], index=False).to_numpy()
    def support(rows):
        return {c: {"rows": int(np.count_nonzero(labels[rows] == c)),
                    "groups": len(np.unique(groups[rows][labels[rows] == c]))} for c in classes}
    result = {"available": False, "reason": None, "groups": [int(v) for v in groups],
              "tuning_rows": [], "selection_rows": [], "folds": [],
              "support": {"allowed": support(np.arange(len(X)))}}
    try:
        A, B, metadata = split_rows(X, labels, features, True, {"strategy": "grouped"})
        # Runtime seed is fixed by the shared policy; never mutate host globals.
        if metadata["seed"] != 42:
            raise ValueError("The fixed inner policy requires seed 42")
        folds = []
        for fitting, held in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
                np.arange(len(A)), labels[A], groups[A]):
            fit_rows, held_rows = A[fitting], A[held]
            if len(np.intersect1d(groups[fit_rows], groups[held_rows])):
                raise ValueError("OOF group overlap")
            folds.append({"fit_rows": fit_rows.tolist(), "held_rows": held_rows.tolist()})
        result.update(available=True, reason="structured_grouped_split", tuning_rows=A.tolist(),
                      selection_rows=B.tolist(), folds=folds, metadata=metadata)
        result["support"].update(A=support(A), B=support(B), folds=[{
            "fit": support(np.asarray(f["fit_rows"])), "held": support(np.asarray(f["held_rows"]))} for f in folds])
    except ValueError as error:
        result["reason"] = "grouped_split_unavailable"
        result["error_type"] = type(error).__name__
    result["split_sha256"] = json_hash(result)
    return result


def worker_fit(request_path):
    """Trusted bounded child. Its payload contains no prediction labels."""
    import joblib
    from agent.round4_classification_features import augment_features
    request_path = Path(request_path)
    request = joblib.load(request_path)
    if set(request) != {"X_fit", "y_fit", "X_predict", "context", "recipe"}:
        raise ValueError("Unexpected child request")
    started = time.monotonic()
    directory = request_path.parent
    recipe = checked_recipe(request["recipe"])
    fitting, predicting, context, features, weights = augment_features(
        request["X_fit"], request["X_predict"], request["context"], recipe["features"])
    def observe(event):
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "elapsed_seconds": time.monotonic() - started}) + "\n")
            stream.flush()
    result = fit_predict_probabilities(fitting, request["y_fit"], predicting, context,
                                      recipe=recipe["probability_recipe"], sample_weight=weights, fit_observer=observe)
    np.save(directory / "probabilities.npy", result["probabilities"], allow_pickle=False)
    memory = {"peak_rss_bytes": None, "scope": "child_process_only_not_process_tree", "method": None}
    try:
        import resource
        import sys
        memory.update(peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                      * (1 if sys.platform == "darwin" else 1024), method="getrusage_RUSAGE_SELF")
    except (ImportError, OSError, ValueError):
        pass
    weight_support = None
    if weights is not None:
        labels = np.asarray(request["y_fit"])
        weight_support = {label: {"rows": int(np.count_nonzero(labels == label)),
                                 "total_weight": float(weights[labels == label].sum())} for label in result["classes"]}
    write_json(directory / "metadata.json", {"classes": result["classes"], "evidence": result["evidence"],
               "features": features, "weight_class_support": weight_support, "memory": memory,
               "elapsed_seconds": time.monotonic() - started, "fit_rows": len(fitting), "predict_rows": len(predicting),
               "fit_indices": request["X_fit"].index.tolist(), "predict_indices": request["X_predict"].index.tolist(),
               "request_sha256": file_hash(request_path), "probabilities_sha256": file_hash(directory / "probabilities.npy")})


class BoundedProbabilityFits:
    def __init__(self, directory, context, recipe, deadline):
        self.directory, self.context, self.recipe = Path(directory), deepcopy(context), recipe
        self.deadline, self.records = deadline, []

    def counts(self):
        events = []
        for path in sorted(self.directory.glob("*/events.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                events.append(json.loads(line))
        return {"physical_fits_started": sum(e["status"] == "started" for e in events)
                - sum(e["status"] == "cancelled_before_fit" for e in events),
                "physical_fits_completed": sum(e["status"] == "completed" for e in events)}

    def __call__(self, X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
        import joblib
        started = time.monotonic()
        expires = min(started + timeout_seconds, self.deadline - 5)
        if not math.isfinite(expires) or expires <= started:
            raise TimeoutError("No bounded child allowance remains")
        expected_fits = len(self.recipe["probability_recipe"]["base_recipe"]["components"])
        if self.counts()["physical_fits_started"] + expected_fits > self.recipe["caps"]["max_physical_fits"]:
            raise TimeoutError("Physical fit cap exhausted")
        directory = self.directory / ("f" + str(fit_index))
        directory.mkdir(parents=True, exist_ok=False)
        context = deepcopy(self.context)
        context["deadline_monotonic"] = expires
        request = directory / "request.joblib"
        joblib.dump({"X_fit": X_fit.copy(deep=True), "y_fit": np.asarray(y_fit).copy(),
                     "X_predict": X_predict.copy(deep=True), "context": context, "recipe": self.recipe}, request)
        request_digest = file_hash(request)
        remaining = expires - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Staging exhausted callback deadline")
        code = ("import sys\n" + f"sys.path.insert(0, {str(project_root())!r})\n"
                + "from agent.round4_classification_candidate import worker_fit\n"
                + f"worker_fit({str(request.resolve())!r})\n")
        execution = run_python(code, directory, directory / "log", 0, timeout=remaining,
                               max_memory_bytes=self.recipe["caps"]["memory_limit_bytes"])
        record = {"fit_index": fit_index, "execution": execution, "metadata": None,
                  "elapsed_seconds": time.monotonic() - started}
        self.records.append(record)
        write_json(directory / "execution.json", record)
        if file_hash(request) != request_digest:
            raise ValueError("Model child changed its request")
        if execution.get("timed_out"):
            raise TimeoutError("Supervised model callback timed out")
        if execution.get("exit_code") != 0 or execution.get("error") or execution.get("memory_limit_exceeded"):
            raise RuntimeError("Supervised model callback failed")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        record["metadata"] = metadata
        if (metadata["request_sha256"] != request_digest
                or metadata["probabilities_sha256"] != file_hash(directory / "probabilities.npy")
                or metadata["fit_rows"] != len(X_fit) or metadata["predict_rows"] != len(X_predict)):
            raise ValueError("Child probability receipt does not match this fit")
        matrix = np.load(directory / "probabilities.npy", allow_pickle=False)
        if time.monotonic() >= expires:
            raise TimeoutError("Callback returned beyond its original allowance")
        return {"probabilities": matrix, "classes": metadata["classes"]}


def execute_candidate(X_fit, y_fit, X_predict, context):
    recipe = checked_recipe(context["recipe"])
    classes = recipe["probability_recipe"]["classes"]
    if list(context["classes"]) != classes:
        raise ValueError("Host class order changed")
    if (context.get("task_type") != "classification" or context.get("prediction_kind") != "labels"
            or get_metric(context.get("metric")).name != "Macro F1"):
        raise ValueError("Round4 candidate only supports label classification")
    started = time.monotonic()
    absolute = context.get("deadline_monotonic")
    if type(absolute) not in (int, float) or not math.isfinite(absolute):
        raise ValueError("Finite original host deadline required")
    deadline = min(float(absolute), started + recipe["caps"]["trial_seconds"])
    fits = BoundedProbabilityFits(Path.cwd() / "round4_work", context, recipe, deadline)
    evidence = {"version": 1, "family": recipe["family"], "variant": recipe["variant"], "status": "started",
                "classes": classes, "physical_fit_count_basis": "adapter_start_complete_events", "oof": None,
                "decision": {"rules_trained": 0, "rules_accepted": 0, "rules_applied": 0},
                "fresh_full_refit": False, "full_fit_rows": len(X_fit), "predict_rows": len(X_predict),
                **{key: context.get(key) for key in ("nonce", "phase", "code_hash", "recipe_hash")}}
    try:
        if time.monotonic() >= deadline - 5:
            raise TimeoutError("Original host deadline exhausted before preparation")
        identity = fit_guarded_decision_rule(np.empty((0, len(classes))), [], classes,
                                            tuning_rows=[], selection_rows=[], options=recipe["guard_options"])["rule"]
        rule = identity
        if recipe["variant"] == "oof":
            partition = coverage(X_fit, y_fit, context)
            write_json(Path.cwd() / "round4_split.json", partition)
            evidence["coverage"] = {key: value for key, value in partition.items()
                                    if key not in {"groups", "tuning_rows", "selection_rows", "folds"}}
            calibration_deadline = deadline - recipe["caps"]["final_reserve_seconds"]
            if partition["available"]:
                oof = fit_oof_guarded_rule(X_fit, y_fit, classes, groups=partition["groups"], split_policy="grouped",
                    tuning_rows=partition["tuning_rows"], selection_rows=partition["selection_rows"], folds=partition["folds"],
                    fit_predict=fits, fold_timeout_seconds=recipe["caps"]["callback_seconds"],
                    deadline=calibration_deadline, clock=time.monotonic, guard_options=recipe["guard_options"])
                evidence["oof"] = oof
                evidence["decision"]["rules_trained"] = int(oof["guarded"] is not None)
                evidence["decision"]["rules_accepted"] = int(oof["status"] == "accepted")
                rule = oof["rule"]
            else:
                evidence["oof"] = {"status": "identity", "reason": partition["reason"], "completed_fits": 0, "guarded": None}
        if time.monotonic() >= deadline - 5:
            raise TimeoutError("No time left for fresh final fit")
        result = fits(X_fit, y_fit, X_predict, timeout_seconds=deadline - time.monotonic() - 5, fit_index="final")
        labels = predict_with_rule(result["probabilities"], result["classes"], rule)
        evidence["decision"]["rules_applied"] = int(evidence["decision"]["rules_accepted"] == 1
                                                    and any(v != 1.0 for v in rule["multipliers"]))
        if time.monotonic() >= deadline:
            raise TimeoutError("Final prediction exceeded original deadline")
        evidence.update(status="completed", fresh_full_refit=True, rule=rule,
                        rule_sha256=json_hash(rule), output_label_sha256=json_hash(labels.tolist()))
        return {"predictions": labels}
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        evidence.update(fits.counts(), child_runs=fits.records, elapsed_seconds=time.monotonic() - started)
        write_json(Path.cwd() / "round4_evidence.json", evidence)


def candidate_source():
    return ("from agent.round4_classification_candidate import execute_candidate\n\n"
            "def fit_predict(X_fit, y_fit, X_predict, context):\n"
            "    return execute_candidate(X_fit, y_fit, X_predict, context)\n")
