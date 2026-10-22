"""Read-only, independent audit of saved lab runs.

Only the separately pinned auditor checkout is imported. Candidate source and
recipes are bytes throughout this module. Re-audit never launches a solver,
downloads a source, modifies a receipt, or promotes an interrupted run.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import re

from .integrity import (file_hash, is_transport_temp, json_hash, read_json, require,
                        safe_path, transient_artifact, under)


_REPORT = re.compile(r"\.agent_runs/[0-9a-f]{32}/run\.json")
_AUTOML_REPORT = re.compile(r"\.agent_runs/[0-9a-f]{32}/automl_report\.json")
_HEX = re.compile(r"[0-9a-f]{64}")
_NONCE = re.compile(r"[0-9a-f]{32}")
_TERMINAL = {"finished", "failed", "censored", "interrupted", "completed"}


class EvidenceDrift(ValueError):
    """Frozen input or identity no longer matches its independent seal."""


def _sealed(condition, message):
    if not condition:
        raise EvidenceDrift(message)


def _path(root, value):
    require(isinstance(value, (str, Path)) and bool(str(value)), "Missing evidence path")
    value = Path(value)
    path = under(root, value if value.is_absolute() else Path(root) / value)
    require(not is_transport_temp(path.relative_to(safe_path(root))),
            "Referenced evidence is inside reserved transport staging")
    return path


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _csv(path):
    with safe_path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header and len(header) == len(set(header)), "Missing/duplicate CSV columns")
        rows = list(reader)
    require(all(len(row) == len(header) for row in rows), "Ragged CSV evidence")
    return header, rows


def _project(header, rows, columns):
    require(set(columns) <= set(header), "Required input columns are absent")
    indices = [header.index(column) for column in columns]
    return [[row[index] for index in indices] for row in rows]


def _same_rows(path, columns, expected):
    header, rows = _csv(path)
    require(header == columns and rows == expected,
            "Phase input rows/values/order differ from frozen task: " + str(path.name))


def _index(directory, output, excluded=()):
    """Index every available file, including failure logs and unused receipts."""
    directory, output = safe_path(directory), safe_path(output)
    excluded = {safe_path(path) for path in excluded}
    result = {}
    if directory.exists():
        for path in sorted(directory.rglob("*")):
            # Check links even when they point at directories or missing files.
            require(not path.is_symlink(), "Linked artifact in arm evidence")
            if (path.is_file() and safe_path(path) not in excluded
                    and not transient_artifact(path.relative_to(directory), directory)):
                result[_path(output, path).relative_to(output).as_posix()] = file_hash(path)
    return result


def _verify_index(output, expected):
    require(isinstance(expected, dict), "Missing artifact hash index")
    for relative, digest in expected.items():
        require(isinstance(digest, str) and _HEX.fullmatch(digest), "Invalid artifact SHA256")
        require(file_hash(_path(output, relative)) == digest, "Artifact drift: " + str(relative))


def _workspace_split(evidence, run_dir, task):
    matches = [item for item in evidence.get("observations", [])
               if isinstance(item, dict) and item.get("phase") == "prepare_research"
               and item.get("run_dir") == run_dir.relative_to(task).as_posix()]
    require(len(matches) == 1, "Missing/ambiguous current-run host split observation")
    split = matches[0].get("frozen_split")
    require(isinstance(split, dict), "Missing observed host split")
    return split


def _phase(task, run_dir, raw, name, phase, folder, frozen, spec,
           fit_rows, predict_rows, proof):
    receipt_path = _path(task, raw)
    require(receipt_path.is_relative_to(run_dir / folder) and receipt_path.name == "receipt.json",
            "Receipt is outside the current run/phase")
    directory = receipt_path.parent
    receipt = read_json(receipt_path)
    context = read_json(directory / "context.json")
    recipe = read_json(directory / "recipe.json")
    require(context.get("version") == 3 and context.get("protocol_version") == 3
            and context.get("phase") == phase, "Current phase context is invalid")
    nonce = context.get("nonce")
    require(isinstance(nonce, str) and _NONCE.fullmatch(nonce), "Missing invocation nonce")
    for key, filename in (("code_hash", "candidate.py"), ("recipe_hash", "recipe.json")):
        require(context.get(key) == frozen[key] and file_hash(directory / filename) == frozen[key],
                "Frozen source/recipe does not match invocation")
    require(context.get("recipe") == recipe and json_hash(recipe) == frozen["recipe_hash"],
            "Context recipe differs from canonical frozen recipe")
    local_spec = context.get("task_spec")
    require(isinstance(local_spec, dict), "Missing current invocation TaskSpec")
    for key in ("version", "id_column", "target_column", "features", "numeric", "categorical",
                "classes", "output_columns", "probability_columns", "positive_class", "metric"):
        require(local_spec.get(key) == spec.get(key), "Invocation TaskSpec changed: " + key)
    require(safe_path(local_spec.get("data_dir", "")) == directory
            and safe_path(local_spec.get("task_path", "")) == directory / "task.md",
            "Invocation TaskSpec directory mismatch")
    for key in ("features", "numeric", "categorical", "classes", "positive_class", "metric"):
        require(context.get(key) == spec.get(key), "Invocation context changed: " + key)
    require(context.get("prediction_kind") == spec["prediction_kind"]
            and context.get("task_type") == spec["metric"]["task_type"], "Prediction semantics changed")
    fit_columns = [spec["id_column"], *spec["features"], spec["target_column"]]
    predict_columns = [spec["id_column"], *spec["features"]]
    _same_rows(directory / "fit.csv", fit_columns, fit_rows)
    _same_rows(directory / "predict.csv", predict_columns, predict_rows)
    output_hash = file_hash(directory / "predictions.csv")
    expected = {"version": 3, "nonce": nonce, "phase": phase,
                "code_hash": frozen["code_hash"], "recipe_hash": frozen["recipe_hash"],
                "output_hash": output_hash}
    require(receipt == expected, "Receipt is not bound to the current context/output")
    proof[name] = {"receipt": receipt_path.relative_to(task).as_posix(),
                   "receipt_sha256": file_hash(receipt_path), "nonce": nonce,
                   "context_sha256": file_hash(directory / "context.json"),
                   "fit_rows": len(fit_rows), "predict_rows": len(predict_rows),
                   "output_sha256": output_hash}
    return directory / "predictions.csv"


def _quality_chain(task, report_path, report, evidence, audit):
    """Strengthen the trusted benchmark reader with context and exact input proof."""
    from benchmark_agent import _research_summary
    summary = _research_summary(task)
    require(isinstance(summary, dict) and summary.get("receipt_checks_valid") is True,
            "Trusted current-report publication/receipt audit failed")
    run_dir = report_path.parent
    pointer = summary.get("benchmark_evidence") or {}
    require(pointer.get("run_report") == report_path.relative_to(task).as_posix()
            and pointer.get("run_report_sha256") == file_hash(report_path),
            "Trusted reader points to a different run")
    selected = report.get("published_recipe")
    require(selected in {"baseline", "candidate"}, "No fixed published recipe")
    frozen = report.get("frozen_" + selected)
    require(isinstance(frozen, dict) and all(isinstance(frozen.get(key), str)
            and _HEX.fullmatch(frozen[key]) for key in ("code_hash", "recipe_hash")),
            "Missing frozen candidate identity")
    receipts = report.get("candidate_receipts")
    require(isinstance(receipts, dict) and all(receipts.get(key) == frozen[key]
            for key in ("code_hash", "recipe_hash")), "Published identity differs from frozen identity")
    frozen_observations = [item for item in evidence.get("observations", [])
                           if isinstance(item, dict) and item.get("phase") == "frozen_publication"
                           and item.get("run_dir") == run_dir.relative_to(task).as_posix()
                           and item.get("published_recipe") == selected]
    require(len(frozen_observations) == 1, "Missing/ambiguous original frozen-publication observation")
    original = frozen_observations[0].get("frozen_source")
    require(isinstance(original, dict), "Missing original frozen-source paths")
    code = _path(task, original.get("code_path"))
    recipe_path = _path(task, original.get("recipe_path"))
    require(code.is_relative_to(run_dir) and code.parent == recipe_path.parent
            and code.name == "research_candidate.py" and recipe_path.name == "research_recipe.json",
            "Original frozen source/recipe is outside current run")
    for key, path in (("code_hash", code), ("recipe_hash", recipe_path)):
        require(file_hash(path) == original.get(key) == original.get("declared_" + key) == frozen[key],
                "Original frozen bytes differ from declared/published identity")
    require(read_json(code.parent / "frozen_candidate.json") == {"version": 3, **frozen},
            "Original frozen manifest differs from selected identity")
    spec = audit["contract"]
    # The original task is separate from the invocation-local TaskSpec paths.
    from agent.task_spec import load_task
    loaded = load_task(task / "task.md").to_dict()
    for key in ("version", "id_column", "target_column", "features", "numeric", "categorical",
                "classes", "output_columns", "probability_columns", "positive_class", "metric"):
        require(loaded.get(key) == spec.get(key), "Staged task differs from sealed contract: " + key)
    data_dir = _path(task, loaded["data_dir"])
    train_header, train = _csv(data_dir / "train.csv")
    test_header, test = _csv(data_dir / "test.csv")
    fit_columns = [spec["id_column"], *spec["features"], spec["target_column"]]
    predict_columns = [spec["id_column"], *spec["features"]]
    train_fit = _project(train_header, train, fit_columns)
    train_predict = _project(train_header, train, predict_columns)
    test_predict = _project(test_header, test, predict_columns)
    require(spec["target_column"] not in test_header, "External audit answers leaked into staged test")
    require([row[0] for row in test_predict] == audit["test_ids"], "Test literal IDs changed")
    split = _workspace_split(evidence, run_dir, task)
    fitting, held = split.get("fit_rows"), split.get("validation_rows")
    require(isinstance(fitting, list) and isinstance(held, list) and fitting and held
            and all(type(index) is int and 0 <= index < len(train) for index in fitting + held)
            and len(set(fitting)) == len(fitting) and len(set(held)) == len(held)
            and not set(fitting) & set(held) and set(fitting + held) == set(range(len(train))),
            "Observed outer split does not partition original training rows")
    require(report.get("host_validation", {}).get("split") == split.get("split"),
            "Report outer split differs from observed host split")
    proof = {"published_recipe": selected, "code_sha256": frozen["code_hash"],
             "recipe_sha256": frozen["recipe_hash"], "outer_split_sha256": json_hash(split),
             "frozen_source": original}
    outer = _phase(task, run_dir, receipts.get("outer"), "outer", "outer_validation", "outer_evaluations",
                   frozen, spec, [train_fit[i] for i in fitting], [train_predict[i] for i in held], proof)
    refit = _phase(task, run_dir, receipts.get("refit"), "refit", "full_refit", "full_refit",
                   frozen, spec, train_fit, test_predict, proof)
    require(proof["outer"]["nonce"] != proof["refit"]["nonce"], "Outer/refit invocation nonce reused")
    published = file_hash(task / "submission.csv")
    require(published == file_hash(run_dir / "last_valid_submission.csv") == file_hash(refit),
            "Published CSV/checkpoint/refit output differ")
    # Validate the outer output independently too; its labels come only from train.
    from qualify_unseen_tasks import audit_submission
    outer_audit = {"contract": spec, "metric": audit["metric"],
                   "test_ids": [train_predict[i][0] for i in held],
                   "audit_actuals": [train_fit[i][-1] for i in held]}
    outer_result = audit_submission(outer, outer_audit)
    require(outer_result.get("valid") is True, "Independent outer output validation failed")
    claimed = report.get("evaluations", {}).get("final_check", {}).get(selected)
    require(isinstance(claimed, dict) and claimed.get("name") == outer_result["metric"]["name"]
            and _number(claimed.get("value")) and math.isclose(claimed["value"],
                outer_result["metric"]["value"], rel_tol=1e-9, abs_tol=1e-10),
            "Claimed outer metric differs from saved predictions")
    proof["outer_metric"] = outer_result["metric"]
    return summary, proof


def _resource(execution, report):
    from benchmark_maximize import resource_observation
    execution = dict(execution or {})
    resources = dict(execution.get("resources") or {})
    cleanup = execution.get("cleanup") or {}
    if isinstance(cleanup, dict) and "complete" in cleanup:
        resources["cleanup_complete"] = cleanup["complete"]
    execution["resources"] = resources
    execution["timed_out"] = bool(execution.get("external_timeout") or execution.get("timed_out"))
    return resource_observation({"launcher": execution, "research": report or {}})


def _automl_fixture_chain(task, report_path, report, audit):
    """Verify the exact native short-route model chain, never v3 quality.

    Native AutoML has no v3 receipts. Its typed checkpoint plus deterministic
    selected phase paths must prove a real full-data model publication. No
    other run/report can supply missing evidence and no candidate is imported.
    """
    from agent.task_spec import load_task, score_holdout
    from agent.automl import data_fingerprint

    run_dir = report_path.parent
    require(safe_path(report.get("report_path", "")) == report_path,
            "Native AutoML report path differs from exact worker pointer")
    require(report.get("mode") == "offline" and report.get("output_valid") is True,
            "Native AutoML has no accepted model output")
    search = report.get("search") or {}
    require(search.get("phase") == "finished" and search.get("status") in
            {"completed", "time_limit", "fallback"}, "Native AutoML did not finish model publication")
    selected = search.get("best_candidate")
    require(isinstance(selected, str) and re.fullmatch(r"[A-Za-z0-9_-]+", selected),
            "Native AutoML selected candidate is invalid")
    spec = load_task(task / "task.md")
    contract = spec.to_dict()
    for key in ("version", "id_column", "target_column", "features", "numeric", "categorical",
                "classes", "output_columns", "probability_columns", "positive_class", "metric"):
        require(contract.get(key) == audit["contract"].get(key),
                "Native AutoML task differs from sealed contract: " + key)
    source_inputs = safe_path(spec.data_dir)
    names = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
    snapshot = run_dir / "input_snapshot"
    for name in names:
        require(file_hash(snapshot / name) == file_hash(source_inputs / name),
                "Native AutoML input snapshot drift: " + name)
    require(report.get("input_fingerprint") == data_fingerprint(snapshot),
            "Native AutoML snapshot fingerprint differs from its report")
    checkpoint_pointer = run_dir / "checkpoint.json"
    checkpoint = read_json(checkpoint_pointer)
    relative = checkpoint.get("directory")
    require(checkpoint.get("version") == 1 and isinstance(relative, str)
            and re.fullmatch(r"checkpoints/[0-9a-f]{32}", relative),
            "Native AutoML checkpoint pointer is invalid")
    checkpoint_dir = _path(task, run_dir / relative)
    published = file_hash(task / "submission.csv")
    require(checkpoint.get("submission_sha256") == published
            == file_hash(checkpoint_dir / "submission.csv"),
            "Native AutoML checkpoint/publication CSV differ")
    held_hash = file_hash(checkpoint_dir / "holdout_predictions.csv")
    require(checkpoint.get("holdout_sha256") == held_hash
            == file_hash(run_dir / "holdout_predictions.csv"),
            "Native AutoML checkpoint/holdout export differ")
    # These two exact paths are the only native phases capable of full fitting.
    # Do not glob for a report, checkpoint or another successful candidate.
    phase_dirs = [run_dir / selected / phase for phase in ("seed", "refit")]
    matches = [path for path in phase_dirs if (path / "submission.csv").is_file()
               and file_hash(path / "submission.csv") == published]
    require(len(matches) == 1, "Missing/ambiguous exact native AutoML full-fit phase")
    phase_dir = matches[0]
    config = read_json(phase_dir / "config.json")
    require(config.get("id") == selected and config.get("phase") == phase_dir.name
            and safe_path(config.get("candidate_dir", "")) == phase_dir,
            "Native AutoML full-fit phase/config identity differs")
    require(config.get("model_config") == report.get("candidate_configs", {}).get(selected)
            and config.get("feature_config") == report.get("candidate_feature_configs", {}).get(selected),
            "Native AutoML selected recipe differs from phase config")
    local_spec = config.get("task_spec") or {}
    for key in ("version", "id_column", "target_column", "features", "numeric", "categorical",
                "classes", "output_columns", "probability_columns", "positive_class", "metric"):
        require(local_spec.get(key) == contract.get(key), "Native AutoML phase TaskSpec differs: " + key)
    # Native read_text/write_text round-trips LF as CRLF on Windows. Compare
    # UTF-8 text with universal newlines only; preserve all other whitespace.
    # The original task's raw bytes remain protected by case input hashes.
    require(safe_path(local_spec.get("data_dir", "")) == phase_dir / "inputs"
            and safe_path(local_spec.get("task_path", "")) == phase_dir / "task.md"
            and safe_path(phase_dir / "task.md").read_text(encoding="utf-8")
                == safe_path(task / "task.md").read_text(encoding="utf-8"),
            "Native AutoML phase task paths/content differ")
    for name in names:
        require(file_hash(phase_dir / "inputs" / name) == file_hash(source_inputs / name),
                "Native AutoML full-fit staged input drift: " + name)
    header, train = _csv(source_inputs / "train.csv")
    fitting, held = config.get("fit_rows"), config.get("validation_rows")
    require(isinstance(fitting, list) and isinstance(held, list) and fitting and held
            and all(type(i) is int and 0 <= i < len(train) for i in fitting + held)
            and len(set(fitting)) == len(fitting) and len(set(held)) == len(held)
            and not set(fitting) & set(held) and set(fitting + held) == set(range(len(train))),
            "Native AutoML phase split does not partition original TRAIN")
    durations = read_json(phase_dir / "durations.json")
    require(all(_number(durations.get(key)) and durations[key] >= 0
                for key in ("fit_seconds", "predict_seconds", "total_seconds"))
            and durations["fit_seconds"] > 0, "Native AutoML has no real fit duration evidence")
    execution_dir = phase_dir / "execution"
    marker = f"refitting on all {len(train)} rows"
    require(execution_dir.is_dir() and any(path.is_file() and path.name.endswith(".stdout.txt")
            and marker in safe_path(path).read_text(encoding="utf-8", errors="replace")
            for path in execution_dir.iterdir()), "Native AutoML full-data worker marker is absent")
    outer_phase = "seed" if phase_dir.name == "seed" else "blend" if selected == "blend" else "verification"
    outer_path = run_dir / selected / outer_phase / "holdout_predictions.csv"
    require(file_hash(outer_path) == held_hash, "Native AutoML selected outer/checkpoint output differ")
    ids = [row[header.index(spec.id_column)] for row in train]
    actuals = [row[header.index(spec.target_column)] for row in train]
    value = score_holdout(outer_path, spec, [ids[i] for i in held], [actuals[i] for i in held])
    claimed = report.get("metric") or {}
    require(claimed.get("name") == spec.metric.name and _number(claimed.get("value"))
            and claimed.get("higher_is_better") is spec.metric.higher_is_better
            and math.isclose(claimed["value"], value, rel_tol=1e-9, abs_tol=1e-10),
            "Native AutoML claimed outer metric differs from saved predictions")
    return {"kind": "automl_fixture", "selected_candidate": selected,
            "checkpoint": checkpoint_pointer.relative_to(task).as_posix(),
            "checkpoint_sha256": file_hash(checkpoint_pointer),
            "outer": {"path": outer_path.relative_to(task).as_posix(), "output_sha256": held_hash,
                      "fit_rows": len(fitting), "predict_rows": len(held), "metric": claimed},
            "refit": {"config": (phase_dir / "config.json").relative_to(task).as_posix(),
                      "config_sha256": file_hash(phase_dir / "config.json"),
                      "phase": phase_dir.name, "fit_rows": len(train),
                      "predict_rows": len(audit["test_ids"]), "output_sha256": published}}


def _fixture_contract(task, audit, case):
    """Join the fixture's output-only audit contract to its sealed TaskSpec.

    unseen_cases.materialize_case deliberately supplies an output schema, not a
    complete native TaskSpec. The plan separately freezes the full TaskSpec;
    missing optional schema fields must not be guessed or treated as conflicts.
    """
    from agent.task_spec import load_task

    native = load_task(task / "task.md")
    contract, frozen = native.to_dict(), case.get("spec")
    require(isinstance(frozen, dict), "Fixture has no separately sealed full TaskSpec")
    for key in ("version", "id_column", "target_column", "features", "numeric", "categorical",
                "classes", "output_columns", "probability_columns", "positive_class", "metric"):
        require(contract.get(key) == frozen.get(key), "Fixture TaskSpec differs from sealed plan: " + key)
    contract["prediction_kind"] = native.prediction_kind
    declared = audit.get("contract")
    required = {"id_column", "target_column", "classes", "output_columns",
                "probability_columns", "positive_class", "prediction_kind"}
    require(isinstance(declared, dict) and required <= declared.keys(),
            "Fixture external audit output contract is incomplete")
    for key, value in declared.items():
        # Full real-data contracts can also carry template-local absolute paths.
        if key not in {"task_path", "data_dir"}:
            require(key in contract and contract[key] == value,
                    "Fixture external audit contract differs: " + key)
    require(audit.get("metric") == native.metric.name,
            "Fixture external audit metric differs from native TaskSpec")
    return contract


def _automl_fixture_resources(execution, report, evidence, task, report_path):
    """Use observed full child executions, not lossy AutoML row aggregates."""
    executions, observed, observed_dirs = [], set(), set()
    for item in evidence.get("observations", []):
        if not isinstance(item, dict) or item.get("phase") != "automl_execution":
            continue
        directory = _path(task, item.get("directory"))
        require(directory.is_relative_to(report_path.parent),
                "Observed AutoML child is outside the exact current run")
        config = read_json(directory / "config.json")
        identifier = config.get("id")
        require(config.get("phase") in {"seed", "selection", "verification", "refit"}
                and safe_path(config.get("candidate_dir", "")) == directory
                and isinstance(identifier, str) and directory.parent.name == identifier,
                "Observed AutoML child directory/config differ")
        result = item.get("execution")
        require(isinstance(result, dict), "Observed AutoML child has no execution")
        observed.add(identifier)
        require(directory not in observed_dirs, "Duplicate observed AutoML child invocation")
        observed_dirs.add(directory)
        executions.append(result)
    unknown = []
    for row in report.get("search", {}).get("candidates", []):
        if isinstance(row, dict) and (row.get("resources") or row.get("error")) and row.get("id") not in observed:
            unknown.append({"resources": {"available": False},
                            "error": row.get("error")})
        if isinstance(row, dict) and isinstance(row.get("id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            candidate_dir = report_path.parent / row["id"]
            if candidate_dir.is_dir():
                for phase_dir in candidate_dir.iterdir():
                    if (phase_dir.is_dir() and (phase_dir / "config.json").is_file()
                            and (phase_dir / "execution").is_dir() and phase_dir not in observed_dirs):
                        unknown.append({"resources": {"available": False}, "error": row.get("error")})
    if not executions:
        unknown.append({"resources": {"available": False}})
    # A fully observed outer tree alone does not prove every nested attempt's
    # inventory/cleanup. Missing old-worker observations remain provisional,
    # or censored when the corresponding native trial failed.
    return _resource(execution, {"executions": executions, "unobserved": unknown})


def _load(output, *, execution=True):
    from .planning import load_plan
    return load_plan(output, execution=execution)


def audit_arm(output, case, arm, worker_evidence, plan=None):
    """Audit one started arm without changing any saved file or arm state."""
    output = safe_path(output)
    plan = _load(output) if plan is None else plan
    entry = case["arms"][arm]
    task = _path(output, entry["task_dir"])
    state_path, config_path = (_path(output, entry[key]) for key in ("state_path", "config_path"))
    state = read_json(state_path)
    config = read_json(config_path)
    state_before = file_hash(state_path)
    arm_dir = config_path.parent
    # state and its future terminal seal are intentionally mutable coordinator files.
    exclusions = (state_path, state_path.with_suffix(".sha256"), arm_dir / "audit.json")
    before = _index(arm_dir, output, exclusions)
    errors, report, summary, proof = [], None, None, None
    current = None
    actual_api = None
    mechanism = False
    fixture = plan.get("profile") == "fixture-smoke"
    automl_fixture = False
    integrity = True
    external = {"valid": False, "error": "No saved output", "metric": None}
    evidence_path = None
    evidence_hash = None
    evidence = None
    try:
        if isinstance(worker_evidence, (str, Path)):
            evidence_path = _path(output, worker_evidence)
            evidence = read_json(evidence_path)
            evidence_hash = file_hash(evidence_path)
        else:
            require(isinstance(worker_evidence, dict), "Missing worker evidence")
            evidence = worker_evidence
            evidence_hash = json_hash(evidence)
        _sealed(evidence.get("version") == 1 and evidence.get("arm") == arm
                and evidence.get("run_id") == config.get("run_id") == state.get("run_id"),
                "Worker/state/config identity differs")
        _sealed(evidence.get("config_sha256") == file_hash(config_path), "Worker config hash changed")
        _sealed(safe_path(config.get("task_dir", "")) == task, "Worker task directory changed")
        _sealed(config.get("case_key") == case.get("key", case.get("case_key")), "Worker case key changed")
        for relative, digest in case["input_hashes"].items():
            _sealed(file_hash(_path(task, relative)) == digest, "Staged input drift: " + relative)
        audit_path = _path(output, case["audit_path"])
        _sealed(not audit_path.is_relative_to(arm_dir), "External audit answers are inside solver arm")
        _sealed(file_hash(audit_path) == case["audit_sha256"], "External audit labels/contract changed")
        host_audit = read_json(audit_path)
        if fixture:
            host_audit = {**host_audit, "contract": _fixture_contract(task, host_audit, case)}
        from qualify_unseen_tasks import audit_submission
        external = audit_submission(task / "submission.csv", host_audit)
        pointer = evidence.get("run_report")
        require(isinstance(pointer, dict), "Missing exact current-run report pointer")
        current = pointer
        relative = pointer.get("path")
        automl_fixture = (fixture and config.get("require_adaptive") is False
                          and pointer.get("kind") == "automl")
        pattern = _AUTOML_REPORT if automl_fixture else _REPORT
        require(isinstance(relative, str) and pattern.fullmatch(relative)
                and pointer.get("kind") == ("automl" if automl_fixture else "research_v3"),
                "Current report is not an allowed typed native route")
        report_path = _path(task, relative)
        require(file_hash(report_path) == pointer.get("sha256"), "Current run report hash changed")
        report = read_json(report_path)
        api = report.get("api")
        if isinstance(api, dict):
            actual_api = {key: api.get(key) if type(api.get(key)) is int and api[key] >= 0 else None
                          for key in ("attempts", "responses")}
            actual_api["source_report_sha256"] = file_hash(report_path)
        require(type(evidence.get("solver_returned")) is bool
                and pointer.get("origin") == ("returned" if evidence["solver_returned"]
                                               else "observed_state_after_failure"),
                "Returned/failure report provenance differs")
        require(evidence.get("solver_returned") is True and evidence.get("cli_exit_code") == 0
                and type(evidence.get("cli_exit_code")) is int and not evidence.get("terminal_error"),
                "Native solver/main did not return successfully")
        original, passed = evidence.get("original_time_budget_seconds"), evidence.get("cli_time_budget_seconds")
        require(_number(original) and _number(passed) and 0 < passed <= original
                and original == config.get("time_budget_seconds") == plan.get("time_budget_seconds"),
                "Native original/remaining budget contract changed")
        expected_argv = [str(safe_path(config["solver_root"]) / "solution.py"), str(task / "task.md"),
                         *config["solver_args"], "--time-budget", str(passed)]
        require(evidence.get("cli_argv") == expected_argv, "Observed CLI arguments differ from sealed config")
        if automl_fixture:
            require(passed <= 300 and report.get("mode") == "offline",
                    "Native AutoML fixture is not the short offline CLI route")
            require(isinstance(api, dict) and api.get("used") is False
                    and type(api.get("calls")) is int and api["calls"] == 0,
                    "Native AutoML fixture does not prove zero API calls")
            require(not any(isinstance(item, dict) and item.get("phase") == "run_agent"
                            for item in evidence.get("observations", [])),
                    "Native AutoML fixture contains a conflicting v3 invocation")
            native_budget = report.get("time_budget_seconds")
            require(_number(native_budget) and 0 < native_budget <= passed
                    and _number(report.get("elapsed_seconds"))
                    and 0 <= report["elapsed_seconds"] <= native_budget + .1,
                    "Native AutoML fixture exceeds its original remaining budget")
            proof = _automl_fixture_chain(task, report_path, report, host_audit)
            actual_api = {"attempts": 0, "responses": 0,
                          "kind": "native_offline_zero_calls", "native_calls": api["calls"],
                          "source_report_sha256": file_hash(report_path)}
            summary = {key: report.get(key) for key in ("mode", "api", "search", "metric",
                       "time_budget_seconds", "elapsed_seconds", "output_valid")}
            summary["route"] = "automl_fixture"
            mechanism = True
        else:
            require(report.get("research_protocol_version") == 3, "Expected native research protocol v3")
            summary = {key: report.get(key) for key in (
                "research_protocol_version", "status", "reason", "stage", "api", "mode", "coordinator",
                "experiments", "successful_experiments", "local_experiments", "local_successful_experiments",
                "local_research", "baseline_search", "evaluations", "publication", "output_check",
                "time_budget_seconds", "elapsed_seconds", "time_allocation", "published_recipe", "quality_options")}
            require(report.get("mode") == "local_only" and report.get("api", {}).get("attempts") == 0
                    and report.get("api", {}).get("responses") == 0
                    and type(report.get("api", {}).get("attempts")) is int
                    and type(report.get("api", {}).get("responses")) is int,
                    "Actual report does not prove zero-API local-only mode")
            starts = [item for item in evidence.get("observations", [])
                      if isinstance(item, dict) and item.get("phase") == "run_agent"]
            require(len(starts) == 1, "Missing/ambiguous native invocation observation")
            native_budget = starts[0].get("time_budget_seconds")
            require(_number(native_budget) and 0 < native_budget <= passed
                    and _number(report.get("time_budget_seconds")) and math.isclose(
                        report["time_budget_seconds"], native_budget, rel_tol=0, abs_tol=.001),
                    "Native report budget differs from observed run_agent invocation")
            require(_number(report.get("elapsed_seconds")) and 0 <= report["elapsed_seconds"] <= native_budget + .1,
                    "Native elapsed time is missing or exceeds its remaining budget")
            require(starts[0].get("local_only") is True and starts[0].get("adaptive_search") is True,
                    "Native invocation changed default local/adaptive policy")
            if native_budget > 300:
                require(report.get("coordinator", {}).get("active") is True,
                        "Adaptive coordinator was inactive despite more than 300 seconds remaining")
            if config.get("require_adaptive"):
                require(native_budget > 300, "Pilot did not retain more than 300 seconds for adaptive research")
            require(report.get("stage") == "result" and report.get("status") in {"completed", "fallback"},
                    "Native research did not reach a terminal result")
            require(report.get("publication", {}).get("new_result_published") is True
                    and report.get("output_check", {}).get("valid") is True,
                    "Fallback/prior output has no new model-quality proof")
            require(report.get("status") == "completed", "Fallback output remains quality-unverified")
            trusted_summary, proof = _quality_chain(task, report_path, report, evidence, host_audit)
            summary.update(trusted_summary)
            mechanism = fixture
    except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error, UnicodeError) as error:
        if isinstance(error, EvidenceDrift):
            integrity = False
        errors.append(type(error).__name__ + ": " + str(error)[:700])
    after = _index(arm_dir, output, exclusions)
    state_after = file_hash(state_path)
    if before != after or state_before != state_after:
        integrity = False
        errors.append("Evidence changed while auditing")
    resources = _resource(state.get("execution"), report)
    if automl_fixture and report is not None:
        try:
            resources = _automl_fixture_resources(state.get("execution"), report, evidence, task, report_path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(type(error).__name__ + ": " + str(error)[:700])
    execution = state.get("execution") or {}
    failed = (type(execution.get("exit_code")) is not int or execution.get("exit_code") != 0
              or execution.get("external_timeout") or execution.get("timed_out")
              or execution.get("memory_limit_exceeded") or execution.get("error") or execution.get("error_type"))
    wall = execution.get("worker_wall_seconds", execution.get("elapsed_seconds"))
    budget = plan.get("time_budget_seconds")
    if not failed and not (_number(wall) and wall >= 0):
        errors.append("Missing finite observed worker wall time")
    interrupted = state.get("status") in {"interrupted", "censored"} and not execution
    complete = bool(integrity and not errors and external.get("valid") is True and not failed and not interrupted)
    mechanism = bool(fixture and mechanism and complete)
    quality = bool(not fixture and complete)
    if not integrity:
        status = "integrity_failed"
    elif resources["qualification"] == "censored" or interrupted:
        status, quality, mechanism = "censored", False, False
    elif failed:
        status = "solver_failed"
    elif _number(wall) and _number(budget) and wall > budget + .25:
        status, quality, mechanism = "deadline_exceeded", False, False
    elif errors:
        status = "unverified_contract"
    elif external.get("valid") is not True:
        status = "invalid_output"
    else:
        status = "completed"
    return {"version": 1, "case": case.get("key", case.get("case_key")), "arm": arm,
            "status": status, "quality_verified": quality, "mechanism_verified": mechanism, "quality_admission": False,
            "integrity_unchanged": integrity, "contract_errors": errors, "audit": external,
            "resources": resources, "research": summary, "provenance": proof,
            "current_report": current, "actual_api": actual_api, "artifact_hashes": after,
            "state_sha256_before": state_before, "state_sha256_after": state_after,
            "worker_evidence_sha256": evidence_hash, "worker_evidence_snapshot": evidence}


def re_audit(output):
    """Validate sealed plan and every started arm; recompute metrics read-only."""
    output = safe_path(output)
    errors, rows = [], []
    try:
        plan = _load(output, execution=False)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        # An invalid seal cannot establish quality, but keep discoverable arms
        # in the failure denominator instead of returning an empty success set.
        try:
            untrusted = read_json(output / "plan.json")
            for case in untrusted.get("cases", []):
                for arm, entry in case.get("arms", {}).items():
                    state = read_json(_path(output, entry["state_path"]))
                    rows.append({"case": case.get("key"), "arm": arm, "valid": False,
                                 "status": "integrity_failed", "recorded_status": state.get("status"),
                                 "quality_verified": False})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        return {"version": 1, "valid": False, "status": "integrity_failed", "arms": rows,
                "errors": [type(error).__name__ + ": " + str(error)[:700]], "quality_admission": False}
    for case in plan["cases"]:
        for arm, entry in case["arms"].items():
            row = {"case": case.get("key", case.get("case_key")), "arm": arm}
            try:
                state_path = _path(output, entry["state_path"])
                state = read_json(state_path)
                if state.get("status") == "prepared":
                    rows.append({**row, "status": "not_started", "valid": True})
                    continue
                require(state.get("status") in _TERMINAL | {"started"}, "Unknown arm state")
                before = file_hash(state_path)
                index = state.get("artifact_hashes")
                if index is None and isinstance(state.get("audit"), dict):
                    index = state["audit"].get("artifact_hashes")
                if state.get("status") == "started":
                    # A crash has no terminal seal. Keep all discovered evidence,
                    # mark incomplete, and do not infer a completed experiment.
                    config = _path(output, entry["config_path"])
                    rows.append({**row, "status": "censored", "valid": True,
                                 "reason": "interrupted_started_arm", "quality_verified": False,
                                 "artifact_hashes": _index(config.parent, output, (state_path,))})
                    continue
                require(isinstance(index, dict) and bool(index), "Terminal arm has no artifact index")
                marker = state_path.with_suffix(".sha256")
                require(marker.is_file() and safe_path(marker).read_text(encoding="ascii").strip() == before,
                        "Terminal mutable arm state differs from its external digest")
                _verify_index(output, index)
                config = read_json(_path(output, entry["config_path"]))
                worker = config.get("worker_output")
                require(worker, "Missing worker evidence location")
                worker_path = _path(output, worker)
                saved = state.get("audit")
                require(isinstance(saved, dict), "Missing terminal audit result")
                if worker_path.is_file():
                    worker = worker_path
                else:
                    worker = saved.get("worker_evidence_snapshot")
                    require(isinstance(worker, dict) and json_hash(worker) == saved.get("worker_evidence_sha256"),
                            "Missing worker file and unsealed supervisor failure evidence")
                audit = audit_arm(output, case, arm, worker, plan)
                require(audit["artifact_hashes"] == index, "Artifact files added/removed after terminal audit")
                for field in ("status", "quality_verified", "mechanism_verified", "audit", "provenance", "current_report", "actual_api"):
                    require(saved.get(field) == audit.get(field), "Recomputed saved audit differs: " + field)
                require(file_hash(state_path) == before, "Mutable state changed during re-audit")
                rows.append({**row, "valid": True, **audit})
            except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as error:
                errors.append({**row, "error": type(error).__name__ + ": " + str(error)[:700]})
                rows.append({**row, "valid": False, "status": "integrity_failed", "quality_verified": False})
    return {"version": 1, "valid": not errors, "status": "verified" if not errors else "integrity_failed",
            "arms": rows, "errors": errors, "quality_admission": False}
