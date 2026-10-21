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
        from qualify_unseen_tasks import audit_submission
        external = audit_submission(task / "submission.csv", host_audit)
        pointer = evidence.get("run_report")
        require(isinstance(pointer, dict), "Missing exact current-run report pointer")
        current = pointer
        relative = pointer.get("path")
        require(isinstance(relative, str) and _REPORT.fullmatch(relative)
                and pointer.get("kind") == "research_v3", "Current report is not native v3 research")
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
        original, passed = evidence.get("original_time_budget_seconds"), evidence.get("cli_time_budget_seconds")
        require(_number(original) and _number(passed) and 0 < passed <= original
                and original == config.get("time_budget_seconds") == plan.get("time_budget_seconds"),
                "Native original/remaining budget contract changed")
        expected_argv = [str(safe_path(config["solver_root"]) / "solution.py"), str(task / "task.md"),
                         *config["solver_args"], "--time-budget", str(passed)]
        require(evidence.get("cli_argv") == expected_argv, "Observed CLI arguments differ from sealed config")
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
    execution = state.get("execution") or {}
    failed = (type(execution.get("exit_code")) is not int or execution.get("exit_code") != 0
              or execution.get("external_timeout") or execution.get("timed_out")
              or execution.get("memory_limit_exceeded") or execution.get("error") or execution.get("error_type"))
    wall = execution.get("worker_wall_seconds", execution.get("elapsed_seconds"))
    budget = plan.get("time_budget_seconds")
    if not failed and not (_number(wall) and wall >= 0):
        errors.append("Missing finite observed worker wall time")
    interrupted = state.get("status") in {"interrupted", "censored"} and not execution
    quality = bool(integrity and not errors and external.get("valid") is True and not failed and not interrupted)
    if not integrity:
        status = "integrity_failed"
    elif resources["qualification"] == "censored" or interrupted:
        status, quality = "censored", False
    elif failed:
        status = "solver_failed"
    elif _number(wall) and _number(budget) and wall > budget + .25:
        status, quality = "deadline_exceeded", False
    elif errors:
        status = "unverified_contract"
    elif external.get("valid") is not True:
        status = "invalid_output"
    else:
        status = "completed"
    return {"version": 1, "case": case.get("key", case.get("case_key")), "arm": arm,
            "status": status, "quality_verified": quality, "quality_admission": False,
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
                for field in ("status", "quality_verified", "audit", "provenance", "current_report", "actual_api"):
                    require(saved.get(field) == audit.get(field), "Recomputed saved audit differs: " + field)
                require(file_hash(state_path) == before, "Mutable state changed during re-audit")
                rows.append({**row, "valid": True, **audit})
            except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as error:
                errors.append({**row, "error": type(error).__name__ + ": " + str(error)[:700]})
                rows.append({**row, "valid": False, "status": "integrity_failed", "quality_verified": False})
    return {"version": 1, "valid": not errors, "status": "verified" if not errors else "integrity_failed",
            "arms": rows, "errors": errors, "quality_admission": False}
