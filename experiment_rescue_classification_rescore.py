"""Independently rescore a portable rescue evidence export; no ML packages/API.

The export contains evidence_manifest.json, SHA-256-pinned prediction CSVs and
host-only development answers. Competition TEST predictions are format-checked
only, never assigned a quality score. Original training data is not required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Invalid CSV header: " + str(path))
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("Malformed CSV row: " + str(path))
        return reader.fieldnames, rows


def macro_f1(actual, predicted, classes):
    if (not actual or len(actual) != len(predicted) or not classes
            or len(classes) != len(set(classes))
            or not set(actual + predicted) <= set(classes)):
        raise ValueError("Invalid label vectors or class order")
    result = {}
    for label in classes:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        result[label] = {"precision": tp / (tp + fp) if tp + fp else 0.,
                         "recall": tp / (tp + fn) if tp + fn else 0.,
                         "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                         "support": tp + fn}
    return {"macro_f1": sum(row["f1"] for row in result.values()) / len(classes),
            "per_class": result}


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "evidence_manifest.json").read_text(encoding="utf-8"))
    def checked_path(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or path.is_symlink() or not path.is_file():
            raise ValueError("Invalid artifact path: " + name)
        expected = manifest["files"].get(name)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Artifact hash mismatch: " + name)
        return path
    for name in manifest["files"]:
        checked_path(name)
    identifier, target = manifest["id_column"], manifest["target_column"]
    output = []
    for row in manifest["scored_predictions"]:
        header, predictions = read_csv(checked_path(row["predictions"]))
        answer_header, answers = read_csv(checked_path(row["answers"]))
        if header != [identifier, target] or answer_header != header:
            raise ValueError("Unexpected prediction/answer columns")
        ids = [value[identifier] for value in predictions]
        if (not ids or len(ids) != len(set(ids)) or any(not value for value in ids)
                or ids != [value[identifier] for value in answers]):
            raise ValueError("Prediction IDs differ from development answers")
        scores = macro_f1([value[target] for value in answers],
                          [value[target] for value in predictions], manifest["classes"])
        expected = row["macro_f1"]
        if (not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or not math.isclose(scores["macro_f1"], expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Saved Macro F1 does not reproduce")
        output.append({"candidate": row["candidate"], "seed": row["seed"], **scores})
    unscored = 0
    for row in manifest.get("unscored_predictions", []):
        header, predictions = read_csv(checked_path(row["predictions"]))
        id_header, expected_ids = read_csv(checked_path(row["expected_ids"]))
        ids = [value.get(identifier) for value in predictions]
        if (header != [identifier, target] or id_header != [identifier] or not ids
                or any(not value for value in ids) or len(ids) != len(set(ids))
                or ids != [value[identifier] for value in expected_ids]
                or not {value[target] for value in predictions} <= set(manifest["classes"])):
            raise ValueError("Invalid unscored competition TEST predictions")
        unscored += 1
    return {"scope": "public_train_development_only", "verified_files": len(manifest["files"]),
            "unscored_test_csvs_format_verified": unscored,
            "api_attempts": 0, "api_responses": 0, "results": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Extracted evidence folder")
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
