"""Compact CPU reports from freshly re-audited saved evidence, never from fits."""
from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import tempfile

from .integrity import file_hash, json_hash, read_json, safe_path, under, write_json


ARMS = ("incumbent", "candidate")
OUTCOMES = ("win", "tie", "loss", "provisional_resource_observation", "failed", "censored",
            "integrity_failed", "skipped", "not_completed", "metric_contract_mismatch",
            "fixture_smoke", "self_check")


def _load_plan(output):
    from .planning import load_plan
    return load_plan(output, execution=False)


def _re_audit(output):
    from .audit import re_audit
    return re_audit(output)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metric(value):
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    if not _number(value.get("value")) or type(value.get("higher_is_better")) is not bool:
        return None
    return {key: value[key] for key in ("name", "value", "higher_is_better")}


def _reasons(*values):
    result = []
    for value in values:
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item and item not in result:
                result.append(item)
    return result


def _arm(output, entry, audited, *, fixture=False):
    state, state_error = {}, None
    try:
        state_path = under(output, output / entry["state_path"])
        state = read_json(state_path)
        pinned = audited.get("state_sha256_after")
        if pinned is not None and file_hash(state_path) != pinned:
            raise ValueError("State changed after independent re-audit")
    except (OSError, ValueError, TypeError, KeyError) as error:
        state_error = type(error).__name__ + ": " + str(error)
    execution = state.get("execution") or {}
    if not isinstance(execution, dict):
        execution = {}
    external = audited.get("audit") or {}
    if not isinstance(external, dict):
        external = {}
    resources = audited.get("resources") or {}
    if not isinstance(resources, dict):
        resources = {}
    metric = _metric(external.get("metric"))
    complete = (audited.get("valid") is True
                and audited.get("status") == "completed" and external.get("valid") is True
                and metric is not None and state_error is None)
    verified = complete and not fixture and audited.get("quality_verified") is True
    mechanism = complete and fixture and audited.get("mechanism_verified") is True
    qualified = verified and resources.get("qualification") == "observed" and resources.get("fully_observed") is True
    observed_mechanism = (mechanism and resources.get("qualification") == "observed"
                          and resources.get("fully_observed") is True)
    cleanup = execution.get("cleanup") or {}
    raw_resources = execution.get("resources") or {}
    cleanup_complete = cleanup.get("complete") if isinstance(cleanup, dict) else None
    if type(cleanup_complete) is not bool:
        cleanup_complete = raw_resources.get("cleanup_complete") if isinstance(raw_resources, dict) else None
    if type(cleanup_complete) is not bool:
        cleanup_complete = None
    provenance = audited.get("provenance")
    finalization = bool((verified or mechanism) and isinstance(provenance, dict)
                        and provenance.get("outer") and provenance.get("refit"))
    timing = {name: execution.get(name) if _number(execution.get(name)) and execution[name] >= 0 else None
              for name in ("worker_wall_seconds", "cleanup_elapsed_seconds", "supervisor_elapsed_seconds", "elapsed_seconds")}
    timing.update({name: state.get(name) if isinstance(state.get(name), str) else None
                   for name in ("started_at_utc", "finished_at_utc")})
    raw_api = audited.get("actual_api") or {}
    actual_api = {name: raw_api.get(name) if isinstance(raw_api, dict)
                  and type(raw_api.get(name)) is int and raw_api[name] >= 0 else None
                  for name in ("attempts", "responses")}
    return {"status": "integrity_failed" if state_error else audited.get("status", "missing_audit"),
            "state_status": state.get("status", "missing"), "audit_valid": audited.get("valid") is True and state_error is None,
            "quality_verified": verified, "qualified_cpu": qualified,
            "quality_admission": False,
            "mechanism_verified": mechanism, "observed_mechanism": observed_mechanism, "metric": metric,
            "resource_qualification": resources.get("qualification", "unavailable"),
            "resource_observation": resources, "cleanup_complete": cleanup_complete,
            "csv_valid": external.get("valid") is True, "outer_and_full_refit_verified": finalization,
            "current_report": audited.get("current_report"), "provenance": provenance,
            "per_class": external.get("per_class", []), "timing": timing,
            "actual_api": actual_api, "actual_api_source_report_sha256": raw_api.get("source_report_sha256")
            if isinstance(raw_api, dict) else None, "actual_api_verified": audited.get("valid") is True
            and state_error is None and all(value is not None for value in actual_api.values()),
            "actual_api_provenance": {key: raw_api.get(key) for key in ("kind", "native_calls")}
            if isinstance(raw_api, dict) and raw_api.get("kind") else None,
            "reason": _reasons(state_error, audited.get("reason"), audited.get("contract_errors"),
                               external.get("error"), state.get("reason"), state.get("error"))}


def _pair(case, arms, *, policy_comparison, fixture, self_check, tolerance):
    metadata = case.get("metadata") or {}
    row = {"case": case.get("key", case.get("case_key")), "dataset_id": case.get("dataset_id"),
           "seed": case.get("seed"), "tier": case.get("tier", "unspecified"),
           "metric": case.get("metric"), "direction": case.get("direction"),
           "role": metadata.get("role", case.get("role", "unspecified")),
           "prior_exposure": metadata.get("prior_exposure", case.get("prior_exposure")),
           "arm_order": case.get("arm_order"), "arms": arms,
           "fingerprints": {key: case.get(key) for key in ("audit_sha256", "split_hashes", "input_hashes")},
           "signed_difference": None, "provisional_signed_difference": None, "outcome": "not_completed"}
    left, right = (arms[name] for name in ARMS)
    measured = all(arm["quality_verified"] for arm in (left, right))
    fixture_measured = fixture and all(arm["mechanism_verified"] for arm in (left, right))
    qualified = all(arm["qualified_cpu"] for arm in (left, right))
    if measured or fixture_measured:
        a, b = left["metric"], right["metric"]
        expected = row["metric"].get("name") if isinstance(row["metric"], dict) else row["metric"]
        same = (a["name"] == b["name"] == expected and a["higher_is_better"] == b["higher_is_better"]
                and row["direction"] in ("minimize", "maximize")
                and a["higher_is_better"] == (row["direction"] == "maximize"))
        if not same:
            row["outcome"] = "metric_contract_mismatch"
            return row
        difference = b["value"] - a["value"] if a["higher_is_better"] else a["value"] - b["value"]
        if not math.isfinite(difference):
            row["outcome"] = "metric_contract_mismatch"
            return row
        if fixture:
            row["outcome"] = "fixture_smoke"
        elif self_check:
            row["outcome"] = "self_check"
        elif qualified and policy_comparison:
            row["signed_difference"] = difference
            row["outcome"] = "win" if difference > tolerance else "loss" if difference < -tolerance else "tie"
        elif (policy_comparison and all(arm["resource_qualification"] in
              ("observed", "provisional_resource_observation") for arm in (left, right))):
            row["outcome"] = "provisional_resource_observation"
            row["provisional_signed_difference"] = difference
        return row
    statuses = {arm["status"] for arm in (left, right)}
    if "integrity_failed" in statuses:
        row["outcome"] = "integrity_failed"
    elif "censored" in statuses or "interrupted" in statuses:
        row["outcome"] = "censored"
    elif statuses & {"skipped", "unavailable", "ineligible", "unsupported"}:
        row["outcome"] = "skipped"
    elif statuses & {"solver_failed", "deadline_exceeded", "unverified_contract", "invalid_output", "failed"}:
        row["outcome"] = "failed"
    return row


def _summary(rows, denominator):
    counts = Counter(row["outcome"] for row in rows)
    counts = {name: counts[name] for name in OUTCOMES}
    counts["missing"] = max(0, denominator - len(rows))
    started = [arm for row in rows for arm in row["arms"].values()
               if arm["status"] != "not_started" and arm["state_status"] != "prepared"]
    api_known = [arm for arm in started if arm["actual_api_verified"]]
    api = {name: sum(arm["actual_api"][name] for arm in api_known)
           if started and len(api_known) == len(started) else None for name in ("attempts", "responses")}
    return {"denominator_pairs": denominator, "planned_pairs": len(rows), "outcomes": counts,
            "qualified_pairs": sum(counts[name] for name in ("win", "tie", "loss")),
            "candidate_win_rate": counts["win"] / denominator if denominator else None,
            "verified_quality_arms": sum(arm["quality_verified"] for row in rows for arm in row["arms"].values()),
            "qualified_cpu_arms": sum(arm["qualified_cpu"] for row in rows for arm in row["arms"].values()),
            "verified_mechanism_arms": sum(arm["mechanism_verified"] for row in rows for arm in row["arms"].values()),
            "observed_mechanism_arms": sum(arm["observed_mechanism"] for row in rows for arm in row["arms"].values()),
            "actual_api": {**api, "verified_arms": len(api_known), "started_arms": len(started)}}


def _identity(value):
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in ("sha", "source_fingerprint", "root", "dirty")}


def _cell(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, ".9g")
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _markdown(result):
    summary = result["summary"]
    lines = ["# External benchmark laboratory report", "", "Status: **" + result["status"] + "**. Profile: `" +
             _cell(result["profile"]) + "`. Scope: " + result["scope"] + ".", "",
             "Saved evidence was independently re-audited. This report starts no training or downloads.", "",
             "Candidate wins: **" + str(summary["outcomes"]["win"]) + "/" + str(summary["denominator_pairs"]) +
             " planned pairs**; ties: " + str(summary["outcomes"]["tie"]) + "; losses: " + str(summary["outcomes"]["loss"]) +
             ". Provisional pairs: " + str(summary["outcomes"]["provisional_resource_observation"]) + ".", "",
             "Positive signed difference means candidate better: incumbent − candidate for minimize, candidate − incumbent for maximize. "
             "Only two freshly verified, fully observed CPU arms count; tie tolerance: " + _cell(result["tie_tolerance"]) +
             ". Provisional differences are separate and never wins. Different metrics are not averaged; budget tiers remain separate.", "",
             "| Dataset / seed / tier | Metric / direction | Incumbent | Candidate | Signed difference | Provisional difference | Outcome |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in result["pairs"]:
        arms = row["arms"]
        values = [" / ".join(_cell(row[key]) for key in ("dataset_id", "seed", "tier")),
                  _cell(row["metric"]) + " / " + _cell(row["direction"])]
        values += [_cell(arms[name]["metric"]["value"] if arms[name]["metric"] else None) for name in ARMS]
        values += [_cell(row["signed_difference"]), _cell(row["provisional_signed_difference"]), row["outcome"]]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "| Case | Arm | Outcome / state | Worker s | Cleanup s | CSV valid | Outer + full refit | Cleanup complete | Resources | Reason |",
              "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    for row in result["pairs"]:
        for name in ARMS:
            arm = row["arms"][name]
            values = [row["case"], name, arm["status"] + " / " + arm["state_status"], arm["timing"]["worker_wall_seconds"],
                      arm["timing"]["cleanup_elapsed_seconds"], arm["csv_valid"], arm["outer_and_full_refit_verified"],
                      arm["cleanup_complete"], arm["resource_qualification"], "; ".join(arm["reason"]) or "—"]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "| Case | Arm | UTC start | UTC finish | Actual API attempts / responses |",
              "| --- | --- | --- | --- | --- |"]
    for row in result["pairs"]:
        for name in ARMS:
            arm = row["arms"][name]
            values = [row["case"], name, arm["timing"]["started_at_utc"], arm["timing"]["finished_at_utc"],
                      _cell(arm["actual_api"]["attempts"]) + " / " + _cell(arm["actual_api"]["responses"])]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "All pair outcomes: " + ", ".join(name + "=" + str(value) for name, value in summary["outcomes"].items()) + ".", "",
              "Actual report API attempts / responses: " + _cell(summary["actual_api"]["attempts"]) + " / " +
              _cell(summary["actual_api"]["responses"]) + "; verified report coverage " + str(summary["actual_api"]["verified_arms"]) +
              "/" + str(summary["actual_api"]["started_arms"]) + " started arms. Missing observations remain unknown.", "",
              "Incumbent: `" + _cell((result["solvers"].get("incumbent") or {}).get("sha")) + "`; candidate: `" +
              _cell((result["solvers"].get("candidate") or {}).get("sha")) + "`.", "",
              "Plan SHA-256: `" + _cell(result["plan_sha256"]) + "`. Manifest SHA-256: `" + _cell(result["manifest_sha256"]) +
              "`. Method SHA-256: `" + _cell(result["method_sha256"]) + "`.", "",
              "Evidence directory: `" + _cell(result["evidence_directory"]) + "`. JSON and Markdown file hashes are saved in their `.sha256` sidecars.", "",
              "Windows-native execution and GPU experiments are not established by this CPU report. Synthetic fixture smoke is mechanism evidence only. "
              "Local quality does not establish a contest-score improvement."]
    if result["errors"]:
        lines += ["", "Re-audit / report errors:", ""] + ["- " + _cell(error) for error in result["errors"]]
    if result["fixture_gate"]["applicable"]:
        gate = result["fixture_gate"]
        lines += ["", "Fixture mechanism gate: **" + ("passed" if gate["passed"] else "not passed") +
                  "**; verified mechanism arms " + str(summary["verified_mechanism_arms"]) +
                  "; fully observed mechanism arms " + str(summary["observed_mechanism_arms"]) +
                  ". Quality admission: false; qualified CPU quality arms: 0."]
    return "\n".join(lines) + "\n"


def _write_text(path, value):
    path = safe_path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lab-report-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value.encode("utf-8"))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def report(output, write=True):
    """Return JSON-ready outcomes and optionally write report.json/report.md.

    A four-pair pilot keeps denominator four even when arms are not started,
    failed, censored or provisional. No saved score can bypass fresh re-audit.
    ``write=False`` performs no writes, environment subprocess, training or I/O
    beyond saved source/evidence verification.
    """
    output = safe_path(output)
    errors, plan_trusted = [], True
    try:
        plan = _load_plan(output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        plan_trusted = False
        errors.append("Plan validation failed: " + type(error).__name__ + ": " + str(error))
        try:
            plan = read_json(output / "plan.json")
        except (OSError, ValueError, TypeError, KeyError):
            plan = {"cases": [], "profile": "unknown"}
    try:
        fresh = _re_audit(output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        fresh = {"valid": False, "arms": [], "errors": [type(error).__name__ + ": " + str(error)]}
    errors.extend(fresh.get("errors", []))
    lookup, duplicates = {}, set()
    for arm in fresh.get("arms", []):
        key = (arm.get("case"), arm.get("arm"))
        if key in lookup:
            duplicates.add(key)
            errors.append("Duplicate independent arm audit: " + str(key))
        lookup[key] = arm
    profile = plan.get("profile", "unknown")
    cases = plan.get("cases", [])
    if not isinstance(cases, list):
        cases, plan_trusted = [], False
        errors.append("Plan cases are malformed")
    if profile == "pilot" and len(cases) > 4:
        plan_trusted = False
        errors.append("Pilot contains more than the frozen four pairs")
    tolerance = plan.get("tie_tolerance", 0.0)
    if not _number(tolerance) or tolerance < 0:
        tolerance, plan_trusted = 0.0, False
        errors.append("Invalid frozen tie tolerance")
    rows = []
    for case in cases:
        key = case.get("key", case.get("case_key"))
        arms = {}
        for name in ARMS:
            audited = lookup.get((key, name), {"status": "missing_audit", "valid": False})
            if not plan_trusted or (key, name) in duplicates:
                audited = {**audited, "status": "integrity_failed", "valid": False, "quality_verified": False}
            arms[name] = _arm(output, case.get("arms", {}).get(name, {}), audited,
                              fixture=profile == "fixture-smoke")
        rows.append(_pair(case, arms, policy_comparison=plan_trusted and plan.get("policy_comparison") is True,
                          fixture=profile == "fixture-smoke", self_check=plan.get("self_check") is True, tolerance=tolerance))
    denominator = 4 if profile == "pilot" else len(rows)
    summary = _summary(rows, denominator)
    tiers = {tier: _summary([row for row in rows if row["tier"] == tier], sum(row["tier"] == tier for row in rows))
             for tier in sorted({row["tier"] for row in rows})}
    if len(tiers) > 1:
        summary["candidate_win_rate"] = None
    integrity_ok = (plan_trusted and fresh.get("valid") is True and not duplicates and not errors
                    and not any(row["outcome"] == "integrity_failed" for row in rows))
    incomplete = any(row["outcome"] not in ("win", "tie", "loss", "fixture_smoke", "self_check") for row in rows)
    status = "integrity_failed" if not integrity_ok else "incomplete" if incomplete or len(rows) < denominator else "completed"
    expected_arms = denominator * len(ARMS)
    fixture_complete = (profile == "fixture-smoke" and integrity_ok and status == "completed"
                        and expected_arms > 0 and summary["verified_mechanism_arms"] == expected_arms)
    fixture_observed = fixture_complete and summary["observed_mechanism_arms"] == expected_arms
    fixture_zero_api = summary["actual_api"]["attempts"] == 0 and summary["actual_api"]["responses"] == 0
    result = {"schema_version": 1, "status": status, "profile": profile,
              "quality_admission": False,
              "scope": "mechanism_fixture" if profile == "fixture-smoke" else "same_solver_self_check" if plan.get("self_check") else "paired_cpu",
              "quality_evidence": bool(plan.get("policy_comparison") is True and summary["qualified_pairs"] > 0 and plan_trusted),
              "fixture_gate": {"applicable": profile == "fixture-smoke",
                               "mechanism_complete": fixture_complete,
                               "resources_observed": fixture_observed,
                               "zero_api_verified": fixture_zero_api if profile == "fixture-smoke" else False,
                               "passed": bool(fixture_complete and fixture_observed and fixture_zero_api),
                               "quality_admission": False},
              "plan_sha256": file_hash(output / "plan.json") if (output / "plan.json").is_file() else None,
              "manifest_sha256": plan.get("manifest_copy_sha256"), "method_sha256": plan.get("method_sha256"),
              "plan_id": plan.get("plan_id"), "tie_tolerance": tolerance, "comparison_rule": plan.get("comparison_rule"),
              "environment": plan.get("environment"), "hardware": plan.get("hardware"),
              "configured_limits": {key: plan.get(key) for key in ("threads", "memory_limit_bytes", "time_budget_seconds")},
              "solvers": {name: _identity(plan.get("solvers", {}).get(name)) for name in ARMS},
              "auditor": _identity(plan.get("auditor")), "summary": summary, "tiers": tiers, "pairs": rows,
              "re_audit_valid": fresh.get("valid") is True, "errors": errors, "evidence_directory": str(output),
              "report_files": {"json": "report.json", "markdown": "report.md", "hashes": ["report.sha256", "report.md.sha256"]}}
    result["content_sha256"] = json_hash(result)
    if write:
        write_json(output / "report.json", result)
        _write_text(output / "report.md", _markdown(result))
        _write_text(output / "report.sha256", file_hash(output / "report.json") + "\n")
        _write_text(output / "report.md.sha256", file_hash(output / "report.md") + "\n")
    return result


build_report = report
