"""Export completed round4 evidence to a portable ZIP, using only the stdlib.

Run after screen, confirmation, full-refit, final reports and the host audit.
No model fitting, pickle loading, provider calls or edits to source results occur.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = "round4-classification-evidence"
SAFE_SUFFIXES = {".json", ".jsonl", ".py", ".txt", ".log", ".md"}
VALIDATION_LOGS = ("round4_contract_tests.log", "round4_harness_final_tests.log",
                   "round4_prepare.log", "round4_export_tests.log")
README = r"""# Round 4 classification: portable evidence

This archive contains saved predictions, public-TRAIN development answers,
ID-only competition TEST rows, sealed plans/recipes, execution receipts, child
metadata, probability arrays, logs, final reports, and a source/test snapshot.
Original datasets, feature CSVs and serialized requests/models are excluded.
Absolute paths in original receipts are preserved to retain their exact bytes;
`evidence_manifest.json` supplies the portable paths and SHA-256 checksums.

## Independent verification, without ML packages or original datasets

Python 3.10 or newer is sufficient. `rescore.py` is the unchanged stdlib scorer
from `experiment_rescue_classification_rescore.py`. It checks every manifest
hash, prediction/answer schema, ordered unique IDs, declared classes and saved
development Macro F1 (absolute tolerance 1e-12), then prints per-class scores.
Competition TEST predictions are format-checked only; their quality is unknown.

Linux/macOS, from the folder containing the downloaded ZIP:

```bash
python3 -m zipfile -e round4-classification-evidence.zip extracted
cd extracted/round4-classification-evidence
python3 rescore.py . > independent_rescore.json
```

PowerShell (replace the ZIP filename if it was renamed): use a fresh, short
extraction root such as `C:\r4e`. Deep execution-receipt paths can exceed Windows
path limits when extracted below a long Downloads path. These commands are
provided for portability; this export has not been tested on Windows.

```powershell
Expand-Archive -LiteralPath .\round4-classification-evidence.zip -DestinationPath C:\r4e
Set-Location C:\r4e\round4-classification-evidence
py -3 .\rescore.py . | Set-Content -Encoding utf8 .\independent_rescore.json
```

`nested_child_audit.json` records export-time checks of each bound successful
child: request and probability SHA-256, exact disk/embedded metadata equality,
execution receipt agreement, probability-array shape, and exact worker-local
fit/predict indices. Development OOF partitions must match sealed preflight;
final refits must use every supplied row. Requests are hashed without loading
pickle/joblib, then excluded from the archive. Consequently request-byte checks
cannot be repeated from this deliberately dataset-free archive. The unchanged
host audit covers the mapping from worker-local rows to original TRAIN/TEST.
Failed/unbound children are identified separately and are not claimed verified.

`source/` is a snapshot for review, not a turnkey training environment. Running
the experiment requires the original task files and pinned ML dependencies;
see `source/docs/round4_classification_plan.md` and the final report. Rescoring
above performs no training, API calls, platform submission or integration.
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_path(path, root):
    path, root = Path(path), Path(root).resolve()
    require(not path.is_symlink(), "Evidence symlink: " + str(path))
    resolved = path.resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(),
            "Missing file or path outside evidence root: " + str(path))
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), "Evidence directory symlink: " + str(parent))
    return resolved


def pinned(path, expected, root=None):
    path = checked_path(path, root) if root is not None else Path(path)
    require(sha256(path) == expected, "SHA-256 mismatch: " + str(path))
    return path


def portable(path, root):
    return (Path("evidence") / checked_path(path, root).relative_to(root)).as_posix()


def verify_npy(path, rows, columns):
    """Read only the documented NPY header; never deserialize Python objects."""
    with Path(path).open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Invalid NPY magic")
        version = tuple(stream.read(2))
        require(version in {(1, 0), (2, 0), (3, 0)}, "Unsupported NPY version")
        size_bytes = 2 if version == (1, 0) else 4
        raw = stream.read(size_bytes)
        require(len(raw) == size_bytes, "Truncated NPY header length")
        size = struct.unpack("<H" if size_bytes == 2 else "<I", raw)[0]
        require(size <= 65536, "Unexpectedly large NPY header")
        header = ast.literal_eval(stream.read(size).decode("utf-8" if version == (3, 0) else "latin1"))
        require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
                "Invalid NPY header schema")
        require(header["shape"] == (rows, columns) and type(header["fortran_order"]) is bool,
                "Probability NPY shape/order differs from child metadata")
        require(isinstance(header["descr"], str) and re.fullmatch(r"[<>=]f[48]", header["descr"]),
                "Probability NPY must contain plain floating-point values")
        require(Path(path).stat().st_size == stream.tell() + rows * columns * int(header["descr"][-1]),
                "Probability NPY payload length differs from header")


def checked_indices(values, size):
    require(isinstance(values, list) and all(type(v) is int and 0 <= v < size for v in values)
            and len(set(values)) == len(values), "Invalid local split indices")
    return set(values)


def verify_partition(path, record, evidence, coverage):
    split = read_json(path)
    require(split["split_sha256"] == json_hash({k: v for k, v in split.items() if k != "split_sha256"}),
            "OOF split digest differs from saved content")
    require({k: v for k, v in split.items() if k not in {"groups", "tuning_rows", "selection_rows", "folds"}}
            == evidence["coverage"], "OOF split differs from embedded coverage")
    if record["seed"] is not None:
        expected = next(row["coverage"] for row in coverage if row["seed"] == record["seed"])
        require(split == expected, "Development OOF split differs from sealed preflight")
    size = record["fit_rows"]
    groups = split["groups"]
    require(len(groups) == size and all(type(v) is int for v in groups), "Invalid raw group vector")
    if split["available"]:
        a, b = (checked_indices(split[name], size) for name in ("tuning_rows", "selection_rows"))
        require(a.isdisjoint(b) and a | b == set(range(size)), "OOF A/B do not partition all fit rows")
        require({groups[i] for i in a}.isdisjoint(groups[i] for i in b), "OOF A/B raw groups overlap")
        require(len(split["folds"]) == 3, "Expected three OOF folds")
        held_all = []
        for fold in split["folds"]:
            fit, held = (checked_indices(fold[name], size) for name in ("fit_rows", "held_rows"))
            require(fit.isdisjoint(held) and fit | held == a, "OOF fold does not partition A")
            require({groups[i] for i in fit}.isdisjoint(groups[i] for i in held), "OOF fold raw groups overlap")
            held_all.extend(held)
        require(len(held_all) == len(a) and set(held_all) == a, "OOF held folds do not cover A exactly once")
    return split


def verify_children(root, record, coverage, classes):
    results = []
    for item in record.get("candidate_evidence", []):
        if not item["path"].endswith("round4_evidence.json"):
            continue
        evidence = item["content"]
        invocation = checked_path(item["path"], root).parent
        partition = None
        if evidence["variant"] == "oof" and "coverage" in evidence:
            partition = verify_partition(checked_path(invocation / "round4_split.json", root),
                                         record, evidence, coverage)
        seen = set()
        for child in evidence["child_runs"]:
            index = child["fit_index"]
            require(index == "final" or type(index) is int and 0 <= index <= 3, "Unexpected child fit index")
            require(index not in seen, "Duplicate child fit index")
            seen.add(index)
            directory = invocation / "round4_work" / ("f" + str(index))
            execution_path = checked_path(directory / "execution.json", root)
            execution = read_json(execution_path)
            require({k: v for k, v in execution.items() if k != "metadata"}
                    == {k: v for k, v in child.items() if k != "metadata"},
                    "Disk execution differs from embedded child execution")
            require(execution.get("metadata") is None or execution["metadata"] == child["metadata"],
                    "Disk execution carries conflicting metadata")
            result = {"candidate": record["candidate"], "seed": record["seed"], "fit_index": index,
                      "execution": portable(execution_path, root), "execution_sha256": sha256(execution_path)}
            metadata = child["metadata"]
            if metadata is None:
                result.update(status="unbound_or_failed", reason="No metadata bound by parent adapter")
                results.append(result)
                continue
            metadata_path = checked_path(directory / "metadata.json", root)
            require(read_json(metadata_path) == metadata, "Disk metadata differs from embedded child metadata")
            require(child["execution"]["exit_code"] == 0 and not child["execution"]["timed_out"]
                    and not child["execution"]["memory_limit_exceeded"], "Bound child did not finish successfully")
            request = pinned(directory / "request.joblib", metadata["request_sha256"], root)
            probabilities = pinned(directory / "probabilities.npy", metadata["probabilities_sha256"], root)
            if index == "final":
                fit_indices, predict_indices = list(range(record["fit_rows"])), list(range(record["predict_rows"]))
            else:
                require(partition is not None and partition["available"], "OOF child has no available partition")
                if index == 3:
                    fit_indices, predict_indices = partition["tuning_rows"], partition["selection_rows"]
                else:
                    fold = partition["folds"][index]
                    fit_indices, predict_indices = fold["fit_rows"], fold["held_rows"]
            require(metadata["fit_indices"] == fit_indices and metadata["predict_indices"] == predict_indices
                    and metadata["fit_rows"] == len(fit_indices) and metadata["predict_rows"] == len(predict_indices),
                    "Child rows/indices differ from exact worker-local fit/predict partition")
            require(metadata["classes"] == classes, "Child probability class order differs from plan")
            verify_npy(probabilities, len(predict_indices), len(classes))
            result.update(status="verified", metadata=portable(metadata_path, root), metadata_sha256=sha256(metadata_path),
                          request_sha256=sha256(request), request_excluded=True,
                          probabilities=portable(probabilities, root), probabilities_sha256=sha256(probabilities),
                          fit_rows=len(fit_indices), predict_rows=len(predict_indices),
                          fit_indices_sha256=json_hash(fit_indices), predict_indices_sha256=json_hash(predict_indices))
            results.append(result)
        if record["status"] == "completed":
            require(any(r["fit_index"] == "final" and r["status"] == "verified" for r in results),
                    "Successful adapter has no verified full supplied-row final fit")
    return results


def verify_completed(root, repo=ROOT):
    """Read-only validation. Return all records and the supplemental nested audit."""
    plan_path = root / "plan.json"
    plan, seal = read_json(plan_path), read_json(root / "seal.json")
    require(sha256(plan_path) == seal["plan_sha256"], "Plan differs from seal")
    for name, expected in plan["source_hashes"].items():
        pinned(repo / name, expected, repo)
    for name, expected in plan["input_sha256"].items():
        pinned(name, expected)
    for name, expected in plan["host_only_answer_hashes"].items():
        pinned(name, expected, root)
    coverage = read_json(pinned(root / "coverage_preflight.json", plan["coverage_preflight_sha256"], root))
    for item in plan["recipes"].values():
        pinned(root / item["source"], item["source_sha256"], root)
        require(json_hash(item["recipe"]) == item["recipe_sha256"], "Plan recipe changed")
    splits = {row["seed"]: row for row in plan["splits"]}
    for split in splits.values():
        require(json_hash({k: split[k] for k in ("fit", "held")}) == split["sha256"], "Outer split changed")
    rows = []
    for directory in sorted((root / "trials").iterdir()):
        if directory.is_dir():
            rows.append(read_json(checked_path(directory / "record.json", root)))
    require(rows, "No recorded outcomes")
    keys = [(r["stage"], r["candidate"], r["seed"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate outcome identity")
    expected = {("screen", name, plan["seeds"][0]) for name in plan["screen_order"]}
    selection_path = root / "confirmation_selection.json"
    selection = read_json(pinned(selection_path, read_json(root / "confirmation_selection.sha256.json")["sha256"], root))
    require(selection["plan_sha256"] == sha256(plan_path) and selection["seeds"] == plan["seeds"][1:],
            "Confirmation selection differs from plan")
    for name, expected_hash in selection["screen_record_hashes"].items():
        pinned(name, expected_hash, root)
    if selection["candidate"] is not None:
        require(selection["candidate"] in plan["screen_order"], "Unplanned finalist")
        expected.update(("confirm", name, seed) for seed in selection["seeds"]
                        for name in ("native_fixed", selection["candidate"]))
        expected.add(("full-refit", selection["candidate"], None))
    require(set(keys) == expected, "Screen, confirmation or full-refit outcomes are incomplete/unexpected")
    report = read_json(checked_path(root / "audit.json", root))
    require(report["official_inputs_unchanged"] is True and report["recorded_outcomes"] == len(rows)
            and report["verified_completed_invocations"] == sum(r["status"] == "completed" for r in rows)
            and report["audit_harness_sha256"] == plan["source_hashes"]["experiment_round4_classification.py"],
            "Final host audit is missing, stale or inconsistent")
    table = read_json(checked_path(root / "experiments.json", root))
    require(table["plan_sha256"] == sha256(plan_path) and table["outcomes"] == rows,
            "Final experiment table differs from recorded outcomes")
    all_children, columns = [], None
    for row in rows:
        require(row["plan_sha256"] == sha256(plan_path), "Record belongs to another plan")
        for name, expected_hash in row.get("invocation_inputs", {}).items():
            pinned(name, expected_hash, root)
        for item in row.get("candidate_evidence", []):
            path = pinned(item["path"], item["sha256"], root)
            if "content" in item:
                actual = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                          if path.suffix == ".jsonl" else read_json(path))
                require(actual == item["content"], "Candidate evidence differs from embedded content")
        if row["status"] != "skipped":
            split = splits.get(row["seed"])
            fit_rows = len(split["fit"]) if split else plan["diagnostics"]["rows"]
            predict_rows = len(split["held"]) if split else plan["test_rows"]
            require(row["fit_rows"] == fit_rows and row["predict_rows"] == predict_rows,
                    "Outer row counts differ from sealed plan")
            all_children.extend(verify_children(root, row, coverage, plan["classes"]))
        if row["status"] != "completed":
            continue
        output = pinned(row["output_path"], row["output_sha256"], root)
        receipt = read_json(pinned(row["receipt_path"], row["receipt_sha256"], root))
        context = read_json(output.parent / "context.json")
        recipe = plan["recipes"][row["candidate"]]
        require(receipt == {"version": 3, "nonce": context["nonce"], "phase": row["phase"],
                "code_hash": recipe["source_sha256"], "recipe_hash": recipe["recipe_sha256"],
                "output_hash": row["output_sha256"]}, "Host receipt differs from invocation identity")
        actual_columns = (context["task_spec"]["id_column"], context["task_spec"]["target_column"])
        require(columns is None or columns == actual_columns, "Task schema differs across completed invocations")
        columns = actual_columns
        if row["phase"] == "full_refit":
            pinned(row["published_path"], row["published_sha256"], root)
            require(row["published_sha256"] == row["output_sha256"], "Published full-refit labels differ")
    require(columns is not None, "No completed predictions to export")
    return plan, rows, columns, {"version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "disk bytes and embedded receipts; no pickle loading or model fitting",
        "index_space": "worker-local RangeIndex; exact OOF positional lists or every supplied final row",
        "outer_original_row_mapping": "verified by separate unchanged host audit",
        "verified_successful_children": sum(r["status"] == "verified" for r in all_children),
        "unbound_or_failed_children": sum(r["status"] != "verified" for r in all_children),
        "children": all_children}


def inventory(root, plan, rows, repo=ROOT):
    selected, excluded = [], []
    allowed_csv = {Path(name).resolve() for name in plan["host_only_answer_hashes"]}
    for row in rows:
        if row["status"] == "completed":
            allowed_csv.add(Path(row["output_path"]).resolve())
            if row.get("published_path"):
                allowed_csv.add(Path(row["published_path"]).resolve())
    for source in sorted(root.rglob("*")):
        require(not source.is_symlink(), "Evidence symlink: " + str(source))
        if not source.is_file():
            continue
        if (source.suffix in SAFE_SUFFIXES or source.resolve() in allowed_csv
                or source.name == "probabilities.npy"):
            selected.append((source, Path("evidence") / source.relative_to(root)))
        else:
            excluded.append({"path": source.relative_to(root).as_posix(), "bytes": source.stat().st_size,
                             "reason": "raw_feature_csv_or_serialized_request_or_non_evidence_format"})
    source_names = set(plan["source_hashes"])
    source_names.update(path.relative_to(repo).as_posix() for path in (repo / "tests").glob("*.py"))
    source_names.update(path.relative_to(repo).as_posix() for path in (repo / "docs").glob("round4_classification*.md"))
    source_names.update({Path(__file__).name, "experiment_rescue_classification_rescore.py"})
    require({"docs/round4_classification_plan.md", "docs/round4_classification_report.md",
             "docs/round4_classification_integration.md"} <= source_names, "Final documents missing")
    for name in sorted(source_names):
        selected.append((checked_path(repo / name, repo), Path("source") / name))
    selected.append((repo / "experiment_rescue_classification_rescore.py", Path("rescore.py")))
    for name in VALIDATION_LOGS:
        source = repo / "data" / name
        if source.is_file():
            selected.append((checked_path(source, repo), Path("validation") / name))
    return selected, excluded


def write_test_ids(plan, identifier, destination):
    candidates = [Path(name) for name in plan["input_sha256"] if Path(name).name == "test.csv"]
    require(len(candidates) == 1, "Expected one sealed TEST input")
    source = candidates[0]
    pinned(source, plan["input_sha256"][str(source)])
    name = "answers/competition_test_ids.csv"
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8-sig", newline="") as stream, path.open("x", encoding="utf-8", newline="") as output:
        reader, writer = csv.DictReader(stream), csv.writer(output, lineterminator="\n")
        require(identifier in reader.fieldnames, "Missing TEST identifier")
        writer.writerow([identifier])
        writer.writerows([row[identifier]] for row in reader)
    return name


def build(input_directory, output, repo=ROOT):
    require(Path(output).is_absolute() and Path(output).suffix == ".zip", "--output must be an absolute .zip path")
    root, output, repo = Path(input_directory).resolve(), Path(output).resolve(), Path(repo).resolve()
    require(not output.exists() and output.parent.is_dir(), "Output exists or its parent directory is missing")
    require(not output.is_relative_to(root), "ZIP output must be outside its input evidence directory")
    plan, rows, columns, nested_audit = verify_completed(root, repo)
    selected, excluded = inventory(root, plan, rows, repo)
    with tempfile.TemporaryDirectory(prefix="round4-evidence-", dir=output.parent) as temporary:
        destination = Path(temporary) / ARCHIVE_ROOT
        destination.mkdir()
        for source, relative in selected:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = sha256(source)
            shutil.copyfile(source, target)
            require(before == sha256(target) == sha256(source), "Source changed during copy: " + str(source))
        identifier, target_column = columns
        scored, unscored = [], []
        answers = {split["seed"]: portable(root / "host_only" / ("answers_seed" + str(split["seed"]) + ".csv"), root)
                   for split in plan["splits"]}
        test_ids = write_test_ids(plan, identifier, destination)
        for row in rows:
            if row["status"] != "completed":
                continue
            if row["phase"] == "inner_selection":
                scored.append({"candidate": row["candidate"], "seed": row["seed"], "macro_f1": row["macro_f1"],
                               "predictions": portable(row["output_path"], root), "answers": answers[row["seed"]]})
            else:
                unscored.append({"predictions": portable(row["published_path"], root), "expected_ids": test_ids})
        write_json(destination / "nested_child_audit.json", nested_audit)
        write_json(destination / "export_provenance.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "base_sha": plan["base_sha"],
            "plan_sha256": sha256(root / "plan.json"), "scope": plan["scope"],
            "input_root": str(root), "original_inputs_not_copied": plan["input_sha256"], "excluded": excluded,
            "recorded_outcomes": len(rows), "scored_outputs": len(scored), "unscored_outputs": len(unscored),
            "source_checkout": git_state(repo),
            "serialized_requests_loaded": False, "api_attempts": 0, "api_responses": 0,
            "note": "Original receipt paths preserved; evidence_manifest.json provides portable artifact paths."})
        (destination / "README.md").write_text(README, encoding="utf-8")
        write_json(destination / "evidence_manifest.json", {
            "version": 1, "id_column": identifier, "target_column": target_column, "classes": plan["classes"],
            "files": {p.relative_to(destination).as_posix(): sha256(p)
                      for p in sorted(destination.rglob("*")) if p.is_file()},
            "scored_predictions": scored, "unscored_predictions": unscored})
        spec = importlib.util.spec_from_file_location("round4_independent_rescore",
                                                     repo / "experiment_rescue_classification_rescore.py")
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        rescored = scorer.audit(destination)
        # Reject a source that changed after its individual copy/verification.
        for source, relative in selected:
            require(sha256(source) == sha256(destination / relative), "Source changed before archive completion")
        archive_path = Path(temporary) / "evidence.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path(ARCHIVE_ROOT) / path.relative_to(destination)))
        with archive_path.open("rb") as source, output.open("xb") as final:
            shutil.copyfileobj(source, final)
    return {"output": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
            "recorded_outcomes": len(rows), "verified_successful_children": nested_audit["verified_successful_children"],
            "independent_rescore": rescored}


def git_state(repo):
    def run(*arguments):
        return subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    return {"head": run("rev-parse", "HEAD"),
            "changed_paths_from_head": run("diff", "--name-only", "HEAD").splitlines(),
            "untracked_paths": run("ls-files", "--others", "--exclude-standard").splitlines()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/r4c"), help="Finished experiment evidence directory")
    parser.add_argument("--output", type=Path, required=True, help="New absolute .zip path outside --input")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.input, args.output), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
