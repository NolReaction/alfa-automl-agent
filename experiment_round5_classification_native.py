"""One preregistered OFF/ON native classification pair; no automatic retries.

prepare freezes a clean checkout, inputs, grouped splits and one flat package.
smoke runs a separate tiny contract fixture; run executes OFF then ON, each with
3600 solver seconds. audit is read-only apart from its own derived report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from research_lab.integrity import file_hash, json_hash, read_json, require, safe_path, under, write_json
from research_lab.process import coordinator_lock, machine_lock_path, supervise


ROOT = Path(__file__).resolve().parent
BASE_SHA = "66e9481cb61cb83cfecab53be6984d7d2bf829b3"
EXPECTED_RECIPE_HASH = "cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc"
INPUTS = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
ORDER = ("off", "on")
NATIVE_SECONDS = 3600
WATCHDOG_SECONDS = 3630
MEMORY_BYTES = 6 * 1024 ** 3
PLAN_NAME = "docs/round5_classification_host_plan.md"
COMPARISON = "model_plus_scheduling_plus_paid_probe"


class IdentityMismatch(ValueError):
    def __init__(self, evidence):
        super().__init__("Frozen author recipe/class-order identity mismatch; do not run native")
        self.evidence = evidence


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*arguments):
    return subprocess.run(["git", "-C", str(ROOT), *arguments], check=True, capture_output=True,
                          text=True, timeout=20).stdout.strip()


def source_snapshot():
    require(not git("status", "--porcelain"), "Commit code and plan in a clean checkout before preparation")
    require(git("merge-base", BASE_SHA, "HEAD") == BASE_SHA, "Checkout does not descend from the declared base")
    names = git("ls-files", "--", "*.py", "requirements*.txt", PLAN_NAME,
                "docs/round5_classification_host_recipe.json").splitlines()
    require(PLAN_NAME in names and Path(__file__).name in names, "Harness and plan must be committed")
    return {"sha": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "base_sha": BASE_SHA, "files": {name: file_hash(ROOT / name) for name in names}}


def environment(python):
    from research_lab.integrity import environment_versions
    value = environment_versions(python)
    value.update(executable=str(Path(python).absolute()), host_cpu_count=os.cpu_count(), threads=8,
                 cuda_visible_devices="", api_credentials="absent in sanitized solver environment",
                 cgroup={name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                         for name in ("memory.max", "cpu.max") if (Path("/sys/fs/cgroup") / name).is_file()})
    return value


def input_hashes(task):
    from agent.task_spec import load_task
    task = safe_path(task)
    spec = load_task(task)
    require(spec.data_dir.is_relative_to(task.parent), "Task inputs must be below its task directory")
    return {str(task): file_hash(task), **{str(spec.data_dir / name): file_hash(spec.data_dir / name) for name in INPUTS}}


def hash_index(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Linked package/task artifact")
        if path.is_file() and "__pycache__" not in path.parts:
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def verify_files(root, values):
    for name, digest in values.items():
        require(file_hash(under(root, Path(root) / name)) == digest, "Artifact drift: " + name)


def packaged_dependencies(original, package, package_files, source_root=ROOT):
    """Bind logical dependencies to actual flat bytes, including transformed agent.py."""
    sources = {}
    for logical, item in original["sources"].items():
        relative = under(source_root, item["path"]).relative_to(source_root)
        if relative.as_posix() == "agent/__init__.py":
            filename = "agent.py"
        elif relative.parts[0] == "agent":
            filename = "_alfa_" + "__".join(relative.with_suffix("").parts) + ".py"
        else:
            filename = relative.name
        require(filename in package_files, "Dependency absent from flat package: " + logical)
        sources[logical] = {"path": str(package / filename), "sha256": package_files[filename]}
    return {"sources": sources, "versions": original["versions"]}


def frozen_identity(classes):
    from agent.round4_classification_candidate import make_recipe, candidate_source, checked_recipe
    from agent.round4_classification_probability import fixed_incumbent_recipe
    recipe = checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant="oof"))
    source = candidate_source()
    result = {"classes": list(classes), "recipe": recipe, "recipe_sha256": json_hash(recipe),
              "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "identity_probe_recipe": checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant="identity"))}
    if result["recipe_sha256"] != EXPECTED_RECIPE_HASH:
        raise IdentityMismatch({**result, "expected_recipe_sha256": EXPECTED_RECIPE_HASH})
    from agent.round5_classification_host import dependency_snapshot
    result["dependency_snapshot"] = dependency_snapshot()
    return result


def split_evidence(spec):
    import numpy as np
    import pandas as pd
    from agent.automl import choose_validation
    from agent.search import inner_split_rows
    from agent.splits import split_rows
    from agent.task_spec import load_task_data
    train, test, _, _, _, y, features, numeric, _, classification = load_task_data(spec)
    choice, reason = choose_validation(train, test, features, numeric, classification)
    require(choice == {"strategy": "grouped"}, "Expected the unchanged native grouped validation policy")
    fitting, held, metadata = split_rows(train, y, features, classification, choice)
    local = train.iloc[fitting].reset_index(drop=True)
    inner_fit, inner_held, inner_metadata = inner_split_rows(
        local, y.iloc[fitting].reset_index(drop=True), features, classification, choice, np.arange(len(local)))
    groups = pd.util.hash_pandas_object(train.loc[:, features], index=False).to_numpy()
    require(not set(groups[fitting]) & set(groups[held]), "Outer raw groups overlap")
    require(not set(groups[fitting[inner_fit]]) & set(groups[fitting[inner_held]]), "INNER raw groups overlap")
    require(metadata["seed"] == inner_metadata["seed"] == 42, "Native split seed changed")
    partition = {"outer_fit": fitting.tolist(), "outer_held": held.tolist(), "inner_fit": inner_fit.tolist(),
                 "inner_held": inner_held.tolist(), "choice": choice, "outer_metadata": metadata,
                 "inner_metadata": inner_metadata, "validation_reason": reason,
                 "raw_groups_sha256": json_hash([int(v) for v in groups]), "group_overlap": 0,
                 "train_rows": len(train), "test_rows": len(test)}
    return train, test, partition


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def prepare(output, public_task, workspace, python):
    from agent.task_spec import load_task
    import build_submission
    started = time.monotonic()
    output, workspace, public_task = map(safe_path, (output, workspace, public_task))
    require(Path(python).expanduser().absolute() == Path(sys.executable).absolute(),
            "Invoke this harness with the selected Python so preparation and native use one environment")
    require(not output.exists() and not workspace.exists(), "Preparation requires fresh evidence and workspace directories")
    require(not workspace.is_relative_to(ROOT), "Flat package and solver tasks must be outside the checkout")
    require(not output.is_relative_to(workspace) and not workspace.is_relative_to(output), "Separate evidence and runtime roots required")
    snapshot = source_snapshot()
    inputs = input_hashes(public_task)
    spec = load_task(public_task)
    require(spec.classification and spec.metric.name == "Macro F1" and spec.prediction_kind == "labels",
            "Only the declared Macro F1 label task is allowed")
    try:
        identity = frozen_identity(list(spec.classes))
    except IdentityMismatch as error:
        write_json(output / "identity_mismatch.json", {"status": "blocked", "utc": utc(), **error.evidence})
        raise
    env = environment(python)
    train, test, partition = split_evidence(spec)
    require(len(train) == 87670 and len(test) == 41166, "This preregistered experiment requires the original public row counts")
    output.mkdir(parents=True)
    workspace.mkdir(parents=True)
    write_json(output / "host_only/split.json", partition)
    write_csv(output / "host_only/outer_answers.csv", [spec.id_column, spec.target_column],
              train.iloc[partition["outer_held"]][[spec.id_column, spec.target_column]].itertuples(index=False, name=None))
    write_csv(output / "host_only/test_ids.csv", [spec.id_column], ((value,) for value in test[spec.id_column]))
    write_json(output / "frozen_identity.json", identity)
    (output / "frozen_oof.py").write_text(identity["source"], encoding="utf-8")
    build_submission.build_archive(output / "flat.zip")
    package = workspace / "package"
    package.mkdir()
    with zipfile.ZipFile(output / "flat.zip") as archive:
        require(all(Path(name).name == name and (name.endswith(".py") or name == "requirements.txt")
                    for name in archive.namelist()), "Packer did not produce a root-only package")
        archive.extractall(package)
    package_files = hash_index(package)
    arms = {}
    relative_data = spec.data_dir.relative_to(public_task.parent)
    for mode in ORDER:
        task = workspace / mode / "task"
        (task / relative_data).mkdir(parents=True)
        shutil.copyfile(public_task, task / "task.md")
        for name in INPUTS:
            shutil.copyfile(spec.data_dir / name, task / relative_data / name)
        staged = input_hashes(task / "task.md")
        require(list(staged.values()) == list(inputs.values()), "Staged task differs from original five inputs")
        arms[mode] = {"task": str(task / "task.md"), "inputs": staged,
                      "requested_mode": mode, "native_budget_seconds": NATIVE_SECONDS}
    require(input_hashes(public_task) == inputs and source_snapshot() == snapshot, "Inputs/source changed during preparation")
    require(time.monotonic() - started <= 600, "Preparation exceeded its separate 600-second cap; do not run")
    manifest = {"version": 1, "created_utc": utc(), "comparison_type": COMPARISON,
                "source": snapshot, "plan_sha256": file_hash(ROOT / PLAN_NAME), "source_root": str(ROOT),
                "public_task": str(public_task), "public_inputs": inputs, "workspace": str(workspace),
                "environment": env, "flat_zip_sha256": file_hash(output / "flat.zip"),
                "package": str(package), "package_files": package_files,
                "package_dependencies": packaged_dependencies(identity["dependency_snapshot"], package, package_files),
                "identity_sha256": file_hash(output / "frozen_identity.json"),
                "host_only_files": hash_index(output / "host_only"), "contract": {**spec.to_dict(), "prediction_kind": spec.prediction_kind},
                "order": list(ORDER), "arms": arms, "time_budget_seconds": NATIVE_SECONDS,
                "watchdog_seconds": WATCHDOG_SECONDS, "startup_cleanup_allowance_seconds": 30,
                "preparation_elapsed_seconds": time.monotonic() - started, "memory_limit_bytes": MEMORY_BYTES,
                "machine_lock": str(machine_lock_path()), "minimum_absolute_gain": .002,
                "source_runtime_hashes": {name: digest for name, digest in snapshot["files"].items() if name.startswith("agent/")},
                "stop_policy": "OFF invalid, source/input/receipt drift, any outer or nested execution timeout/memory stop, or cleanup false stops pair; no retries",
                "resource_policy": "Unavailable RSS/CPU/cleanup stays unknown; successful model evidence may be resource-provisional",
                "scope": "known_public_train_outer_development; competition_test_unscored; default_off"}
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "manifest_seal.json", {"sha256": file_hash(output / "run_manifest.json")})
    return manifest


def load_manifest(output, *, execution=False):
    output = safe_path(output)
    manifest = read_json(output / "run_manifest.json")
    require(file_hash(output / "run_manifest.json") == read_json(output / "manifest_seal.json")["sha256"], "Run manifest drift")
    require(manifest["order"] == list(ORDER) and manifest["time_budget_seconds"] == NATIVE_SECONDS
            and manifest["watchdog_seconds"] == WATCHDOG_SECONDS and manifest["memory_limit_bytes"] == MEMORY_BYTES,
            "Preregistered bounds changed")
    require(file_hash(output / "flat.zip") == manifest["flat_zip_sha256"], "Flat ZIP drift")
    verify_files(Path(manifest["package"]), manifest["package_files"])
    require(hash_index(Path(manifest["package"])) == manifest["package_files"], "Flat package file set changed")
    verify_files(output / "host_only", manifest["host_only_files"])
    require(file_hash(output / "frozen_identity.json") == manifest["identity_sha256"], "Frozen identity drift")
    for values in [manifest["public_inputs"], *(arm["inputs"] for arm in manifest["arms"].values())]:
        for name, digest in values.items():
            require(file_hash(name) == digest, "Original/staged input drift: " + name)
    verify_files(ROOT, manifest["source"]["files"])
    if execution:
        require(source_snapshot() == manifest["source"], "Native arms require the same clean integration SHA/tree")
        require(environment(manifest["environment"]["executable"]) == manifest["environment"], "Execution environment drift")
    return manifest


def arm_command(manifest, mode):
    require(mode in ORDER, "Unknown arm")
    return [manifest["environment"]["executable"], str(Path(manifest["package"]) / "solution.py"),
            manifest["arms"][mode]["task"], "--mode", "offline", "--research-search", "adaptive",
            "--search-profile", "legacy", "--time-budget", str(NATIVE_SECONDS),
            "--classification-round4-mode", mode]


def execution_ok(execution):
    return (type(execution.get("exit_code")) is int and execution.get("exit_code") == 0 and not execution.get("external_timeout")
            and not execution.get("memory_limit_exceeded") and execution.get("cleanup", {}).get("complete") is not False)


def execution_stops(value, prefix=""):
    """Any saved internal timeout/memory failure stops the pair, even after recovery."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = prefix + "/" + key
            if key in {"timed_out", "external_timeout", "host_deadline_exceeded", "memory_limit_exceeded"} and item is True:
                found.append(location)
            elif key == "error_type" and isinstance(item, str) and item in {"TimeoutError", "MemoryError"}:
                found.append(location + ":" + item)
            found.extend(execution_stops(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(execution_stops(item, prefix + "/" + str(index)))
    return found


def inspect_execution_evidence(run_dir, task):
    stops = []
    for path in sorted(run_dir.rglob("*")):
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        require(under(task, path).stat().st_size <= 32 * 1024 ** 2, "Oversized execution evidence")
        raw = path.read_text(encoding="utf-8", errors="replace")
        documents = raw.splitlines() if path.suffix == ".jsonl" else [raw]
        for index, text in enumerate(documents):
            if not text.strip():
                continue
            location = str(path.relative_to(task)) + (":" + str(index + 1) if path.suffix == ".jsonl" else "")
            try:
                stops.extend(execution_stops(json.loads(text), location))
            except ValueError:
                stops.append(location + ":incomplete_json_evidence")
    return stops


def score_labels(path, answers, contract):
    from experiment_rescue_classification_rescore import read_csv, macro_f1
    header, predicted = read_csv(path)
    answer_header, actual = read_csv(answers)
    identifier, target = contract["id_column"], contract["target_column"]
    require(header == answer_header == [identifier, target], "Prediction/answer schema differs")
    ids = [row[identifier] for row in predicted]
    require(ids and all(ids) and len(ids) == len(set(ids)) and ids == [row[identifier] for row in actual],
            "Prediction IDs/order differ from sealed answers")
    a, p = [row[target] for row in actual], [row[target] for row in predicted]
    scores = macro_f1(a, p, contract["classes"])
    classes = contract["classes"]
    scores["class_order"] = classes
    scores["confusion_matrix"] = [[sum(x == c and y == d for x, y in zip(a, p)) for d in classes] for c in classes]
    return scores


def verify_custom_identity(directory, context, evidence, identity):
    require(evidence["variant"] in {"oof", "identity"}, "Unplanned custom variant")
    expected_recipe = identity["recipe"] if evidence["variant"] == "oof" else identity["identity_probe_recipe"]
    require(context["code_hash"] == evidence["code_hash"] == identity["source_sha256"]
            == file_hash(directory / "candidate.py"), "Custom source differs from sealed author source")
    require(context["recipe"] == expected_recipe and context["recipe_hash"] == evidence["recipe_hash"]
            == json_hash(expected_recipe) == file_hash(directory / "recipe.json"), "Custom recipe differs from sealed variant")
    require(context["classes"] == evidence["classes"] == identity["classes"], "Custom class order differs from frozen identity")


def verify_host_manifest(run_dir, report, manifest, identity):
    section = report.get("classification_round4", {})
    if not section.get("manifest"):
        return None
    path = under(run_dir, section["manifest"])
    helper = read_json(path)
    require(helper["source_hash"] == identity["source_sha256"] and helper["classes"] == identity["classes"]
            and helper["oof_recipe"] == identity["recipe"] and helper["oof_recipe_hash"] == identity["recipe_sha256"]
            and helper["identity_recipe"] == identity["identity_probe_recipe"]
            and helper["identity_recipe_hash"] == json_hash(identity["identity_probe_recipe"]), "Host helper manifest identity drift")
    dependencies, expected = helper["dependencies"], manifest["package_dependencies"]
    require(dependencies["versions"] == expected["versions"] and set(dependencies["sources"]) == set(expected["sources"]),
            "Host helper dependency set/versions drift")
    package = Path(manifest["package"])
    for name, item in dependencies["sources"].items():
        source = under(package, item["path"])
        require(source.parent == package and str(source) == expected["sources"][name]["path"]
                and item["sha256"] == expected["sources"][name]["sha256"]
                == manifest["package_files"].get(source.name) == file_hash(source), "Host dependency did not originate from sealed flat package: " + name)
    preflight = section.get("preflight") or {}
    proof = {"path": str(path), "sha256": file_hash(path), "dependency_count": len(dependencies["sources"])}
    if preflight.get("result_path") and Path(preflight["result_path"]).is_file():
        result_path = under(run_dir, preflight["result_path"])
        result = read_json(result_path)
        require(file_hash(result_path.parent / "request.joblib") == preflight["request_sha256"] == result["request_sha256"],
                "Preflight request hash differs from saved receipt")
        partition = result["partition"]
        require(partition["split_sha256"] == json_hash({key: value for key, value in partition.items() if key != "split_sha256"}),
                "Preflight partition hash differs")
        proof["preflight_request_sha256"] = preflight["request_sha256"]
        proof["preflight_result_sha256"] = file_hash(result_path)
    return proof


def audit_arm(output, mode, manifest=None):
    """Verify exact current-run phase inputs, frozen identities and publication."""
    from research_lab.audit import _phase, _csv, _project
    from experiment_rescue_classification_rescore import read_csv
    manifest = load_manifest(output) if manifest is None else manifest
    arm = output / mode
    execution = read_json(arm / "execution.json")
    require(execution_ok(execution), "Native arm failed, timed out, exceeded memory, or has failed cleanup")
    require(execution["argv"] == arm_command(manifest, mode), "Execution argv differs from sealed arm")
    task = Path(manifest["arms"][mode]["task"]).parent
    reports = list((task / ".agent_runs").glob("*/run.json"))
    require(len(reports) == 1, "Expected exactly one current native run report")
    report_path, report = reports[0], read_json(reports[0])
    run_dir = report_path.parent
    stops = inspect_execution_evidence(run_dir, task)
    require(report.get("research_protocol_version") == 3 and report.get("mode") == "local_only"
            and report.get("stage") == "result" and report.get("status") == "completed", "Native v3 did not complete")
    require(report.get("publication", {}).get("new_result_published") is True
            and report.get("output_check", {}).get("valid") is True, "No verified native model publication")
    require(all(type(report.get("api", {}).get(key)) is int and report["api"][key] == 0
                for key in ("attempts", "responses")), "Native API counters are not zero")
    require(report.get("quality_options", {}).get("classification_round4_mode") == mode,
            "Requested and native recorded classification modes differ")
    require(report.get("coordinator", {}).get("active") is True, "Native adaptive coordinator was not reached")
    require(isinstance(report.get("time_budget_seconds"), (int, float))
            and 0 < report["time_budget_seconds"] <= NATIVE_SECONDS
            and isinstance(report.get("elapsed_seconds"), (int, float))
            and 0 <= report["elapsed_seconds"] <= report["time_budget_seconds"] + .1,
            "Native timing exceeds original solver budget")
    partition = read_json(output / "host_only/split.json")
    require(report.get("host_validation", {}).get("split") == partition["outer_metadata"], "Native outer split differs from sealed policy")
    spec = manifest["contract"]
    identity = read_json(output / "frozen_identity.json")
    helper_proof = verify_host_manifest(run_dir, report, manifest, identity)
    from agent.task_spec import load_task
    staged_spec = load_task(task / "task.md")
    train_header, train = _csv(staged_spec.data_dir / "train.csv")
    test_header, test = _csv(staged_spec.data_dir / "test.csv")
    require(spec["target_column"] not in test_header, "TEST contains target labels")
    fit_columns = [spec["id_column"], *spec["features"], spec["target_column"]]
    predict_columns = [spec["id_column"], *spec["features"]]
    train_fit, train_predict, test_predict = (_project(h, rows, columns) for h, rows, columns in
        ((train_header, train, fit_columns), (train_header, train, predict_columns), (test_header, test, predict_columns)))
    selected = report.get("published_recipe")
    require(selected in {"baseline", "candidate"}, "Published candidate identity is missing")
    frozen, receipts = report.get("frozen_" + selected), report.get("candidate_receipts")
    require(isinstance(frozen, dict) and isinstance(receipts, dict)
            and all(receipts.get(key) == frozen.get(key) for key in ("code_hash", "recipe_hash")), "Frozen/publication identity differs")
    originals = [path for path in run_dir.rglob("frozen_candidate.json")
                 if read_json(path) == {"version": 3, **frozen}]
    require(originals, "Original frozen publication snapshot is missing")
    for path in originals:
        require(file_hash(path.parent / "research_candidate.py") == frozen["code_hash"]
                and file_hash(path.parent / "research_recipe.json") == frozen["recipe_hash"], "Frozen source/recipe bytes drifted")
    proof = {}
    outer = _phase(task, run_dir, receipts.get("outer"), "outer", "outer_validation", "outer_evaluations", frozen, spec,
                   [train_fit[i] for i in partition["outer_fit"]], [train_predict[i] for i in partition["outer_held"]], proof)
    refit = _phase(task, run_dir, receipts.get("refit"), "refit", "full_refit", "full_refit", frozen, spec,
                   train_fit, test_predict, proof)
    require(proof["outer"]["nonce"] != proof["refit"]["nonce"], "Outer/refit nonce reused")
    require(file_hash(refit) == file_hash(task / "submission.csv") == file_hash(run_dir / "last_valid_submission.csv"),
            "Published TEST, full refit and last-valid checkpoint differ")
    header, predictions = read_csv(task / "submission.csv")
    require(header == [spec["id_column"], spec["target_column"]]
            and [row[spec["id_column"]] for row in predictions] == [row[0] for row in test_predict]
            and {row[spec["target_column"]] for row in predictions} <= set(spec["classes"]), "Published TEST schema/IDs/classes invalid")
    scores = score_labels(outer, output / "host_only/outer_answers.csv", spec)
    claimed = report.get("evaluations", {}).get("final_check", {}).get(selected, {})
    require(claimed.get("name") == "Macro F1" and isinstance(claimed.get("value"), (int, float))
            and math.isclose(claimed["value"], scores["macro_f1"], rel_tol=0, abs_tol=1e-12), "Claimed outer Macro F1 does not reproduce")
    # All completed INNER invocations, including a paid probe, must use the same
    # permitted local rows. Reuse the pinned stdlib phase auditor, never pickle.
    inner_proof = {}
    outer_rows = partition["outer_fit"]
    inner_fit = [train_fit[outer_rows[i]] for i in partition["inner_fit"]]
    inner_predict = [train_predict[outer_rows[i]] for i in partition["inner_held"]]
    for context_path in sorted(run_dir.rglob("context.json")):
        context = read_json(context_path)
        receipt_path = context_path.parent / "receipt.json"
        if context.get("phase") == "inner_selection" and receipt_path.is_file():
            inner_identity = {key: context[key] for key in ("code_hash", "recipe_hash")}
            _phase(task, run_dir, receipt_path, str(len(inner_proof)), "inner_selection", "", inner_identity, spec,
                   inner_fit, inner_predict, inner_proof)
    custom = audit_custom_children(task, run_dir, spec["classes"], identity)
    require(not custom or helper_proof is not None, "Custom invocation has no sealed host dependency manifest")
    incomplete_custom = any(row.get("evidence_complete") is not True for row in custom)
    return {"version": 1, "status": "incomplete" if stops or incomplete_custom else "completed", "arm": mode, "comparison_type": COMPARISON,
            "internal_execution_stops": stops, "incomplete_custom_evidence": incomplete_custom,
            "requested_mode": mode, "recorded_mode": report.get("quality_options", {}).get("classification_round4_mode", "off"),
            "source_sha": manifest["source"]["sha"], "source_tree": manifest["source"]["tree"],
            "run_report": str(report_path), "run_report_sha256": file_hash(report_path), "execution": execution,
            "scores": scores, "published_recipe": selected, "published_identity": frozen,
            "published_recipe_content": read_json(refit.parent / "recipe.json"),
            "outer_predictions": str(outer), "test_predictions": str(task / "submission.csv"),
            "test_predictions_sha256": file_hash(task / "submission.csv"), "test_quality": "unknown",
            "phase_proof": proof, "inner_invocations": inner_proof, "custom_invocations": custom,
            "helper_manifest_proof": helper_proof,
            "oof_published": frozen["recipe_hash"] == EXPECTED_RECIPE_HASH and frozen["code_hash"] == identity["source_sha256"],
            "native_summary": {key: report.get(key) for key in ("local_experiments", "local_successful_experiments", "local_research",
                "baseline_search", "evaluations", "refit_execution", "time_allocation", "quality_options", "classification_round4", "events")},
            "resources": execution.get("resources"), "cleanup": execution.get("cleanup"),
            "resource_qualification": "observed" if execution.get("resources", {}).get("available") is True
                and execution.get("cleanup", {}).get("complete") is True else "provisional_unknown_telemetry"}


def audit_custom_children(task, run_dir, classes, identity):
    from experiment_round4_classification_export import verify_children
    results = []
    for context_path in sorted(run_dir.rglob("context.json")):
        context = read_json(context_path)
        recipe = context.get("recipe") or {}
        if recipe.get("family") != "round4_classification_v1":
            continue
        path = context_path.parent / "round4_evidence.json"
        summary_present = path.is_file()
        evidence = read_json(path) if summary_present else {
            "variant": recipe["variant"], "status": "incomplete_summary",
            **{key: context[key] for key in ("nonce", "phase", "code_hash", "recipe_hash", "classes")}}
        require(all(evidence.get(key) == context.get(key) for key in ("nonce", "phase", "code_hash", "recipe_hash")),
                "Custom evidence differs from current invocation")
        verify_custom_identity(path.parent, context, evidence, identity)
        from agent.round4_classification_candidate import checked_recipe
        checked_recipe(recipe)
        require(json_hash(recipe) == context["recipe_hash"], "Custom recipe hash differs")
        if evidence["variant"] == "oof":
            require(context["recipe_hash"] == EXPECTED_RECIPE_HASH, "Reached OOF identity differs from author finalist")
        children = []
        if summary_present:
            record = {"candidate": evidence["variant"], "seed": None, "status": evidence["status"],
                      "fit_rows": evidence["full_fit_rows"], "predict_rows": evidence["predict_rows"],
                      "candidate_evidence": [{"path": str(path), "content": evidence}]}
            children = verify_children(task, record, [], classes)
        events, parse_errors = [], []
        for event_path in sorted((path.parent / "round4_work").rglob("events.jsonl")):
            for number, line in enumerate(event_path.read_text(errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    require(isinstance(event, dict) and event.get("status") in {"started", "completed", "cancelled_before_fit"},
                            "Invalid physical-fit event")
                    events.append({**event, "path": str(event_path)})
                except (ValueError, TypeError):
                    parse_errors.append({"path": str(event_path), "line": number})
        started = sum(event.get("status") == "started" for event in events)
        cancelled = sum(event.get("status") == "cancelled_before_fit" for event in events)
        completed = sum(event.get("status") == "completed" for event in events)
        if summary_present and not parse_errors:
            require(started - cancelled == evidence["physical_fits_started"]
                    and completed == evidence["physical_fits_completed"], "Custom physical-fit counts differ from child events")
        if evidence["status"] == "completed":
            require(not parse_errors, "Successful custom invocation has corrupt fit events")
            require(2 <= completed <= started - cancelled <= recipe["caps"]["max_physical_fits"],
                    "Successful custom physical-fit totals exceed the frozen cap or lack completed fits")
        # Preserve original events even if a future adapter adds fields.
        results.append({"path": str(path) if summary_present else None, "sha256": file_hash(path) if summary_present else None,
                        "context": str(context_path), "context_sha256": file_hash(context_path),
                        "evidence_complete": summary_present and not parse_errors,
                        "event_parse_errors": parse_errors, "observed_counts_are_lower_bounds": not summary_present or bool(parse_errors),
                        "phase": context["phase"],
                        "variant": evidence["variant"], "status": evidence["status"], "decision": evidence.get("decision"),
                        "oof": evidence.get("oof"), "elapsed_seconds": evidence.get("elapsed_seconds"),
                        "physical_fits_started": evidence.get("physical_fits_started") if not parse_errors else None,
                        "physical_fits_completed": evidence.get("physical_fits_completed") if not parse_errors else None,
                        "events": events, "event_counts": {"started": started, "cancelled_before_fit": cancelled,
                            "actual_started": started - cancelled, "completed": completed}, "children": children})
    return results


def run_prepared(output):
    output = safe_path(output)
    with coordinator_lock(output / "coordinator.lock"), coordinator_lock(machine_lock_path()):
        manifest = load_manifest(output, execution=True)
        require(not (output / "pair_started.json").exists(), "Native pair is single-use; no automatic resume or retry")
        smoke = read_json(output / "smoke/report.json")
        require(smoke.get("status") == "completed" and smoke.get("flat_zip_sha256") == manifest["flat_zip_sha256"],
                "A successful separate flat-package contract smoke is required")
        write_json(output / "pair_started.json", {"utc": utc(), "manifest_sha256": file_hash(output / "run_manifest.json"),
                                                 "machine_lock": str(machine_lock_path())})
        results = []
        try:
            for mode in ORDER:
                manifest = load_manifest(output, execution=True)
                task = Path(manifest["arms"][mode]["task"]).parent
                require(not (task / ".agent_runs").exists() and not (task / "submission.csv").exists(), "Arm task is not fresh")
                arm = output / mode
                arm.mkdir()
                command = arm_command(manifest, mode)
                write_json(arm / "started.json", {"utc": utc(), "argv": command, "manifest_sha256": file_hash(output / "run_manifest.json")})
                execution = supervise(command, task, arm / "execution", WATCHDOG_SECONDS, MEMORY_BYTES, threads=8)
                write_json(arm / "execution.json", execution)
                load_manifest(output, execution=True)
                result = audit_arm(output, mode, manifest)
                write_json(arm / "report.json", result)
                results.append(result)
                require(result["status"] == "completed", "Internal execution timeout/memory stop or incomplete custom evidence; series stopped")
        except Exception as error:
            write_json(output / "stop.json", {"utc": utc(), "status": "incomplete", "completed_arms": [r["arm"] for r in results],
                                              "error_type": type(error).__name__, "reason": str(error), "retry_allowed": False})
            raise
        return audit(output)


def audit(output):
    output = safe_path(output)
    manifest = load_manifest(output)
    results = []
    for mode in ORDER:
        if (output / mode / "execution.json").is_file():
            results.append(audit_arm(output, mode, manifest))
    complete = len(results) == 2 and all(row["status"] == "completed" for row in results) and not (output / "stop.json").exists()
    delta = results[1]["scores"]["macro_f1"] - results[0]["scores"]["macro_f1"] if complete else None
    reached = bool(complete and any(row["variant"] == "oof" for row in results[1]["custom_invocations"]))
    status = ("incomplete" if not complete else "not_reached" if not reached else
              "promising_opt_in" if delta >= manifest["minimum_absolute_gain"] else "rejected")
    result = {"version": 1, "status": status, "comparison_type": COMPARISON, "manifest_sha256": file_hash(output / "run_manifest.json"),
              "source_sha": manifest["source"]["sha"], "source_tree": manifest["source"]["tree"], "arms": results,
              "on_minus_off_macro_f1": delta, "engineering_threshold": manifest["minimum_absolute_gain"],
              "quality_attribution": "model_plus_scheduling_plus_paid_probe_bundle",
              "oof_model_published": results[1]["oof_published"] if complete else None,
              "oof_model_quality_claim": bool(complete and results[1]["oof_published"]),
              "default_mode": "off", "test_quality": "unknown", "public_outer_is_independent_test": False}
    write_json(output / "report.json", result)
    return result


SMOKE_CODE = r'''import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pandas as pd
import agent
from agent.metrics import get_metric
from agent.task_spec import TaskSpec
from agent.research_candidate import freeze_candidate, run_candidate
from agent.round4_classification_candidate import make_recipe, candidate_source
from agent.round4_classification_probability import fixed_incumbent_recipe
from agent.round5_classification_host import dependency_snapshot, DEPENDENCIES
dependencies = dependency_snapshot()
assert set(dependencies["sources"]) == set(DEPENDENCIES)
assert all(Path(item["path"]).parent == Path(sys.argv[1]) for item in dependencies["sources"].values())
root = Path.cwd()
source = root / "candidate.py"
source.write_text(candidate_source(), encoding="utf-8")
classes = ("a", "b")
spec = TaskSpec(root / "task.md", root, "row", "label", ("x", "z"), ("x", "z"), (), classes,
                ("row", "label"), (), None, get_metric("Macro F1"))
fit = pd.DataFrame({"row": [str(i) for i in range(400)], "x": [str(i) for i in range(400)],
                    "z": [str((i * 17) % 29) for i in range(400)], "label": [classes[i % 2] for i in range(400)]})
predict = fit.iloc[:12].drop(columns="label").copy()
frozen = freeze_candidate(source, make_recipe(fixed_incumbent_recipe(), classes, variant="oof"), root / "frozen")
result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=predict, output_dir=root / "worker",
                        phase="full_refit", timeout=120, deadline=time.monotonic() + 120)
evidence = json.loads((result.output_path.parent / "round4_evidence.json").read_text())
assert evidence["status"] == "completed" and evidence["fresh_full_refit"] is True
assert evidence["full_fit_rows"] == 400 and evidence["predict_rows"] == 12
assert evidence["child_runs"][-1]["metadata"]["fit_indices"] == list(range(400))
assert Path(agent.__file__).parent == Path(sys.argv[1])
summary = {"status": "completed", "fixture": "synthetic_400_rows_two_classes", "quality_admission": False,
           "dependency_snapshot": dependencies, "all_dependency_origins_are_flat_package": True,
           "source_root": str(Path(agent.__file__).parent), "receipt": str(result.receipt_path),
           "receipt_sha256": hashlib.sha256(result.receipt_path.read_bytes()).hexdigest(),
           "predictions": str(result.output_path), "predictions_sha256": hashlib.sha256(result.output_path.read_bytes()).hexdigest(),
           "physical_fits_started": evidence["physical_fits_started"], "physical_fits_completed": evidence["physical_fits_completed"]}
(root / "result.json").write_text(json.dumps(summary, indent=2))
'''


def smoke(output):
    output = safe_path(output)
    with coordinator_lock(output / "coordinator.lock"), coordinator_lock(machine_lock_path()):
        manifest = load_manifest(output, execution=True)
        require(not (output / "pair_started.json").exists(), "Contract smoke must precede the native pair")
        directory = output / "smoke"
        require(not directory.exists(), "Contract smoke is single-use")
        directory.mkdir()
        work = Path(manifest["workspace"]) / "smoke"
        work.mkdir()
        script = work / "contract_smoke.py"
        script.write_text(SMOKE_CODE, encoding="utf-8")
        command = [manifest["environment"]["executable"], str(script), manifest["package"]]
        write_json(directory / "started.json", {"utc": utc(), "command": command, "separate_contract_smoke": True})
        execution = supervise(command, work, directory / "execution", 150, MEMORY_BYTES, threads=8)
        write_json(directory / "execution.json", execution)
        require(execution_ok(execution), "Flat ZIP contract smoke failed; native pair forbidden")
        result = read_json(work / "result.json")
        require(result.get("status") == "completed" and result.get("quality_admission") is False
                and result.get("source_root") == manifest["package"], "Invalid contract smoke proof")
        observed = result["dependency_snapshot"]
        require(result.get("all_dependency_origins_are_flat_package") is True
                and observed == manifest["package_dependencies"], "Flat dependency snapshot differs from sealed package bytes")
        for name in ("receipt", "predictions"):
            require(file_hash(under(work, result[name])) == result[name + "_sha256"], "Smoke artifact drift")
        load_manifest(output, execution=True)
        result.update(flat_zip_sha256=manifest["flat_zip_sha256"], execution=execution,
                      counted_separately_from_native=True, source_sha=manifest["source"]["sha"])
        write_json(directory / "report.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "run", "audit"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/round5-oof")
    parser.add_argument("--public-task", type=Path, default=ROOT / "data/classification/task.md")
    parser.add_argument("--workspace", type=Path, default=ROOT.parent / "round5-native-runtime")
    parser.add_argument("--python", default=sys.executable, help="Absolute Python 3.12 executable; preserve venv path")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, args.public_task, args.workspace, args.python)
        result = {"status": "prepared", "source": result["source"]["sha"], "manifest": str(args.output / "run_manifest.json")}
    else:
        result = {"smoke": smoke, "run": run_prepared, "audit": audit}[args.command](args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
