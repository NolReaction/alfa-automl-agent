"""Read-only error analysis of prepared public TRAIN benchmark predictions.

This never trains models and never evaluates original competition test labels.
The manifest and host-audit files must come from experiment_night_public.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(root: Path) -> dict:
    root = root.resolve()
    manifest_path, report_path = root / "manifest.json", root / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "night_public_prepared":
        raise ValueError("Expected an experiment_night_public prepared manifest")
    if report.get("kind") != "night_public_comparison":
        raise ValueError("Expected the matching public TRAIN comparison report")
    if report["manifest_sha256"] != _hash(manifest_path):
        raise ValueError("Benchmark manifest changed after execution")
    seal = json.loads((root / "seal.json").read_text(encoding="utf-8"))
    if seal != {"manifest_sha256": _hash(manifest_path)}:
        raise ValueError("Prepared manifest does not match its seal")
    rows = []
    for trial in report["trials"]:
        if trial.get("status") != "completed" or not trial["audit"]["valid"]:
            continue
        case = next(c for c in manifest["cases"]
                    if c["name"] == trial["case"] and c["seed"] == trial["seed"])
        audit_path = root / "host_audit" / f"{case['name']}_{case['seed']}.json"
        if _hash(audit_path) != case["audit_file_sha256"]:
            raise ValueError("Retained host answers changed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        folder = Path(trial["task"]).parent
        submission = folder / "submission.csv"
        if _hash(submission) != trial["audit"]["submission_sha256"]:
            raise ValueError("Predictions changed after their audit")
        predicted = pd.read_csv(submission, dtype=str, keep_default_na=False)
        inputs = pd.read_csv(folder / "inputs" / "test.csv", dtype=str,
                             keep_default_na=False)
        actual = np.asarray(audit["actuals"])
        if list(predicted.columns) != ["row_key", "outcome"]:
            raise ValueError("Unexpected public benchmark prediction schema")
        if (len(predicted) != len(actual)
                or predicted["row_key"].tolist() != inputs["row_key"].tolist()):
            raise ValueError("Prediction IDs do not match frozen outer inputs")
        row = {"case": trial["case"], "arm": trial["arm"],
               "metric": trial["audit"]["metric"],
               "selected": trial["solver_report"]["best_candidate"],
               "seconds": trial["elapsed_seconds"],
               "submission_sha256": _hash(submission),
               "audit_file_sha256": _hash(audit_path), "rows": len(actual)}
        if case["dataset"] == "classification":
            labels = manifest["datasets"]["classification"]["classes"]
            values = predicted["outcome"].to_numpy()
            if not set(actual) <= set(labels) or not set(values) <= set(labels):
                raise ValueError("Unknown literal class in answers/predictions")
            metrics = classification_report(actual, values, labels=labels,
                                            output_dict=True, zero_division=0)
            confusion = confusion_matrix(actual, values, labels=labels)
            pairs = [(i, j) for i in range(len(labels)) for j in range(len(labels))
                     if i != j]
            row.update(
                metric_recomputed=metrics["macro avg"]["f1-score"],
                accuracy=float((actual == values).mean()),
                weighted_f1=metrics["weighted avg"]["f1-score"],
                per_class={label: metrics[label] for label in labels},
                confusions=[{"actual": labels[i], "predicted": labels[j],
                             "count": int(confusion[i, j])}
                            for i, j in sorted(pairs, key=lambda z: -confusion[z])[:10]],
            )
        else:
            actual = actual.astype(float)
            values = predicted["outcome"].to_numpy(dtype=float)
            if not np.isfinite(actual).all() or not np.isfinite(values).all():
                raise ValueError("Nonfinite regression value")
            squared = (actual - values) ** 2
            total = squared.sum()
            top_count = max(1, math.ceil(len(squared) * .01))
            largest_errors = np.argsort(squared)[-top_count:]
            tail = actual > np.quantile(actual, .99)
            row.update(
                metric_recomputed=float(np.sqrt(squared.mean())),
                actual_quantiles={str(q): float(np.quantile(actual, q))
                                  for q in (.5, .9, .99, 1)},
                prediction_quantiles={str(q): float(np.quantile(values, q))
                                      for q in (.5, .9, .99, 1)},
                top_1pct_squared_error_share=(float(squared[largest_errors].sum() / total)
                                             if total else 0.0),
                top_1pct_target_squared_error_share=(float(squared[tail].sum() / total)
                                                    if total else 0.0),
                tail_target_rmse=float(np.sqrt(squared[tail].mean())) if tail.any() else None,
                non_tail_target_rmse=float(np.sqrt(squared[~tail].mean())) if (~tail).any() else None,
            )
        if not math.isclose(row["metric_recomputed"], row["metric"]["value"],
                            rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Recomputed metric differs from recorded public audit")
        rows.append(row)
    if not rows:
        raise ValueError("No completed predictions were available for re-audit")
    return {"version": 1, "scope": "Read-only re-audit of previously exposed public TRAIN holdouts; "
            "not platform scores or fresh independent confirmation",
            "fits": 0, "api_calls": 0, "manifest_sha256": _hash(manifest_path),
            "report_sha256": _hash(report_path), "trials": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Re-audited {len(result['trials'])} public TRAIN predictions; fits/API calls: 0. "
          f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
