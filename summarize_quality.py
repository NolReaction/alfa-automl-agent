"""Read sealed paired reports without running solvers or modifying evidence.

This is a descriptive audit. It never averages unlike metrics and never treats
public train holdouts as independent confirmation or a competition score.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

LOSS = {"RMSE", "MAE", "MSE", "LogLoss"}
BOUNDED = {"Macro F1", "Weighted F1", "Accuracy", "ROC AUC"}
EPS = 1e-12
CONFIRM_FAMILIES = {
    "linear_regression", "mae_signed_nonlinear", "mae_heavy_tailed",
    "imbalanced_binary", "multiclass_duplicates", "categorical_shift",
}
PAIRED_FIELDS = ("case", "seed", "mode", "input_sha256", "audit_sha256", "metric",
                 "validation", "time_budget_seconds", "memory_limit_bytes")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def valid_output(row):
    if row is None:
        return False
    ex, audit = row.get("execution") or {}, row.get("audit") or {}
    return (row.get("status") == "completed" and audit.get("valid") is True
            and ex.get("exit_code") == 0 and not ex.get("timed_out")
            and not ex.get("memory_limit_exceeded") and not ex.get("error")
            and row.get("inputs_unchanged") is True and row.get("sources_unchanged") is True)


def compare(before, after, *, pair, planned=None):
    metadata = before or after or planned or {}
    result = {"pair": pair, "comparable": False,
              "case": metadata.get("case"),
              "seed": metadata.get("seed"),
              "mode": metadata.get("mode"),
              "time_budget_seconds": metadata.get("time_budget_seconds", metadata.get("budget")),
              "baseline_status": (before or {}).get("status", "not_started"),
              "candidate_status": (after or {}).get("status", "not_started")}
    if before is None or after is None:
        return dict(result, reason="missing_trial", outcome="unpaired")
    if any(before.get(k) != after.get(k) for k in PAIRED_FIELDS):
        return dict(result, reason="contract_or_budget_mismatch", outcome="unpaired")
    if not valid_output(before) or not valid_output(after):
        return dict(result, reason="invalid_or_incomplete_output", outcome="unpaired")
    b, a = before["audit"].get("metric") or {}, after["audit"].get("metric") or {}
    name = b.get("name")
    if (name != a.get("name") or b.get("higher_is_better") != a.get("higher_is_better")
            or not finite(b.get("value")) or not finite(a.get("value"))):
        return dict(result, reason="invalid_or_mismatched_metric", outcome="unpaired")
    if name not in LOSS | BOUNDED or b["higher_is_better"] != (name in BOUNDED):
        return dict(result, reason="metric_outside_locked_protocol", outcome="unpaired")
    delta = a["value"] - b["value"]
    improvement = delta if name in BOUNDED else -delta
    return dict(result, comparable=True, metric=name, baseline=b["value"], candidate=a["value"],
                absolute_delta=delta, improvement=improvement,
                relative_loss_gain=improvement / max(abs(b["value"]), EPS) if name in LOSS else None,
                bounded_score_gain=improvement if name in BOUNDED else None,
                outcome="win" if improvement > EPS else "loss" if improvement < -EPS else "tie",
                baseline_seconds=before["elapsed_seconds"], candidate_seconds=after["elapsed_seconds"],
                elapsed_delta_seconds=after["elapsed_seconds"] - before["elapsed_seconds"],
                input_sha256=before["input_sha256"], audit_sha256=before["audit_sha256"])


def family_summaries(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["case"], pair["mode"], pair["time_budget_seconds"])].append(pair)
    rows = []
    for (case, mode, budget), values in sorted(groups.items()):
        valid = [p for p in values if p["comparable"]]
        metrics = {p["metric"] for p in valid}
        if len(metrics) > 1:
            raise ValueError("A family group has unlike metrics")
        metric = next(iter(metrics), None)
        complete = len(valid) == len(values)
        gains = [(p["relative_loss_gain"] if metric in LOSS else p["bounded_score_gain"]) for p in valid]
        median = statistics.median(gains) if complete and gains else None
        rows.append({"case": case, "mode": mode, "time_budget_seconds": budget, "metric": metric,
                     "planned_pairs": len(values), "comparable_pairs": len(valid),
                     "outcomes": dict(Counter(p["outcome"] for p in values)),
                     "gain_kind": "relative_loss" if metric in LOSS else "absolute_score" if metric in BOUNDED else None,
                     "median_gain": median,
                     "median_absolute_delta": statistics.median(p["absolute_delta"] for p in valid) if complete and valid else None,
                     "median_baseline": statistics.median(p["baseline"] for p in valid) if complete and valid else None,
                     "median_candidate": statistics.median(p["candidate"] for p in valid) if complete and valid else None,
                     "baseline_seconds": sum(p.get("baseline_seconds", 0) for p in valid),
                     "candidate_seconds": sum(p.get("candidate_seconds", 0) for p in valid),
                     "regression_tolerance_passed": None if median is None else median >= (-0.03 if metric in LOSS else -0.01),
                     "practical_improvement": None if median is None else median >= (0.05 if metric in LOSS else 0.005)})
    return rows


def numeric_gate(report, manifest, pairs, families, arms):
    if report.get("purpose") != "confirmation":
        return {"assessed": False, "reason": "Not independent confirmation"}
    locked_matrix = {(case, seed, arm) for case in CONFIRM_FAMILIES for seed in (211, 307)
                     for arm in ("baseline", "candidate")}
    locked = (len(manifest["trial_plan"]) == 24
              and {(t.get("case"), t.get("seed"), t.get("arm")) for t in manifest["trial_plan"]} == locked_matrix
              and {c.get("case") for c in manifest["cases"]} == CONFIRM_FAMILIES
              and {c.get("seed") for c in manifest["cases"]} == {211, 307}
              and {t.get("budget") for t in manifest["trial_plan"]} == {180}
              and {t.get("mode") for t in manifest["trial_plan"]} == {"automl"})
    measured = len(pairs) == 12 and all(p["comparable"] for p in pairs)
    checks = {
        "locked_confirmation_shape": locked,
        "series_finished": report.get("status") == "completed",
        "all_planned_candidate_csvs_normal_exit": arms["candidate"]["valid_outputs"] == 12,
        "all_pairs_comparable": measured,
        "all_family_medians_within_tolerance": len(families) == 6 and all(f["regression_tolerance_passed"] is True for f in families),
        "at_least_two_practical_improvement_families": sum(f["practical_improvement"] is True for f in families) >= 2,
        "all_candidate_refits_v1_verified": arms["candidate"]["full_refit_v1_verified"] == 12,
    }
    quality_checks = {k: v for k, v in checks.items() if k != "all_candidate_refits_v1_verified"}
    return {"assessed": True, "checks": checks, "quality_gate_passed": all(quality_checks.values()),
            "default_eligibility": "not_established_by_this_descriptive_audit",
            "remaining_evidence": ["strict full-refit re-audit when v1 lacks stdout",
                                   "budget/seed/finalization review across both budgets",
                                   "Research v3 API-failure full-path evidence",
                                   "public nested sanity contract/budget review"]}


def summarize_report(path, reaudit=None):
    path = Path(path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path, seal_path = path.parent / "manifest.json", path.parent / "seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest_hash = digest(manifest_path)
    if report.get("manifest_sha256") != manifest_hash or seal.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest seal or report binding changed")
    if report.get("sources") != manifest.get("sources"):
        raise ValueError("Report and manifest sources differ")
    planned = {t["index"]: t for t in manifest["trial_plan"]}
    if len(planned) != len(manifest["trial_plan"]):
        raise ValueError("Duplicate planned trial index")
    frozen_cases = {(c.get("case", c.get("name")), c["seed"]): c for c in manifest["cases"]}
    actual = {}
    for row in report["trials"]:
        index = row["trial_index"]
        if index in actual or index not in planned:
            raise ValueError("Duplicate or unplanned trial index")
        item = planned[index]
        if row["arm"] != item["arm"] or row["pair"] != item["pair"]:
            raise ValueError("Trial index no longer matches plan")
        if "case" in item:
            expected = {"case": item["case"], "seed": item["seed"], "mode": item["mode"],
                        "time_budget_seconds": item["budget"]}
        else:
            frozen = manifest["cases"][item["pair"]]
            expected = {"case": frozen["name"], "seed": frozen["seed"], "mode": "automl",
                        "time_budget_seconds": manifest["budget_seconds"]}
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Actual trial contract differs from its planned case or budget")
        frozen = frozen_cases.get((row["case"], row["seed"]))
        if frozen is None or any(row.get(k) != frozen.get(k) for k in ("input_sha256", "audit_sha256")):
            raise ValueError("Trial input/audit hashes differ from frozen inputs")
        actual[index] = row
    pair_rows = defaultdict(dict)
    arms = {}
    for index, item in planned.items():
        if item["arm"] in pair_rows[item["pair"]]:
            raise ValueError("Duplicate planned pair arm")
        pair_rows[item["pair"]][item["arm"]] = actual.get(index)
    pair_metadata = {}
    for item in planned.values():
        meta = dict(item)
        if "case" not in meta:
            frozen_case = manifest["cases"][item["pair"]]
            meta.update(case=frozen_case["name"], seed=frozen_case["seed"], mode="automl",
                        budget=manifest["budget_seconds"])
        pair_metadata[item["pair"]] = meta
    pairs = [compare(rows.get("baseline"), rows.get("candidate"), pair=p, planned=pair_metadata[p]) for p, rows in sorted(pair_rows.items())]
    reaudited = {}
    reaudit_info = None
    if reaudit is not None:
        reaudit_path = Path(reaudit).resolve()
        proof = json.loads(reaudit_path.read_text(encoding="utf-8"))
        if (proof.get("kind") != "night_quality_reaudit" or proof.get("version") != 2
                or proof.get("original_report_sha256") != digest(path)
                or proof.get("original_manifest_sha256") != manifest_hash):
            raise ValueError("Re-audit is not bound to this original report and manifest")
        for row in proof["trials"]:
            index = row["trial_index"]
            if index in reaudited or index not in actual or row["arm"] != actual[index]["arm"]:
                raise ValueError("Re-audit contains a duplicate or unknown trial")
            if row.get("v1_refit_evidence") != actual[index].get("refit_evidence"):
                raise ValueError("Re-audit rewrote the original v1 evidence")
            if row.get("valid") is True and (row.get("independent_audit") != actual[index].get("audit")
                    or row.get("independent_audit_matches_original") is not True
                    or row.get("full_refit", {}).get("valid") is not True):
                raise ValueError("Valid re-audit row does not corroborate original audit and full refit")
            reaudited[index] = row
        reaudit_info = {"path": str(reaudit_path), "sha256": digest(reaudit_path), "auditor": proof.get("auditor")}
    for arm in ("baseline", "candidate"):
        rows = [actual.get(i) for i, item in planned.items() if item["arm"] == arm]
        arms[arm] = {
            "planned_trials": len(rows), "recorded_trials": sum(r is not None for r in rows),
            "statuses": dict(Counter((r or {}).get("status", "not_started") for r in rows)),
            "valid_outputs": sum(valid_output(r) for r in rows),
            "full_refit_v1_verified": sum((r or {}).get("refit_evidence", {}).get("valid") is True for r in rows),
            "full_refit_v2_verified": sum(reaudited.get(i, {}).get("valid") is True
                                           and reaudited[i].get("full_refit", {}).get("valid") is True
                                           and reaudited[i].get("independent_audit_matches_original") is True
                                           and reaudited[i].get("independent_audit", {}).get("valid") is True
                                           for i, item in planned.items() if item["arm"] == arm) if reaudit_info else None,
            "elapsed_seconds": sum((r or {}).get("elapsed_seconds", 0) for r in rows),
            "telemetry_available_trials": sum(((r or {}).get("execution", {}).get("resources") or {}).get("available") is True for r in rows),
            "timed_out_trials": sum((r or {}).get("execution", {}).get("timed_out") is True for r in rows),
            "failures": [{"trial_index": i, "status": (actual.get(i) or {}).get("status", "not_started")}
                         for i, item in planned.items() if item["arm"] == arm and not valid_output(actual.get(i))],
            "refit_v1_unverified": [{"trial_index": i, "reason": (actual.get(i) or {}).get("refit_evidence")}
                                    for i, item in planned.items() if item["arm"] == arm
                                    and (actual.get(i) or {}).get("refit_evidence", {}).get("valid") is not True],
        }
    families = family_summaries(pairs)
    return {"report_path": str(path), "report_sha256": digest(path), "manifest_sha256": manifest_hash,
            "kind": report.get("kind"), "purpose": report.get("purpose", "public_sanity"),
            "status": report.get("status"), "abort_reason": report.get("abort_reason"),
            "sources": {k: {s: v.get(s) for s in ("commit", "source_sha256", "worktree_dirty")} for k, v in report["sources"].items()},
            "environment": report.get("environment", manifest.get("environment")),
            "harness_sha256": manifest.get("harness_sha256"),
            "planned_trial_wall_caps_seconds": manifest.get("planned_trial_wall_caps_seconds"),
            "external_api_calls": report.get("external_api_calls"),
            "reaudit_v2": reaudit_info,
            "arms": arms, "counts": dict(Counter(p["outcome"] for p in pairs)),
            "pairs": pairs, "families": families,
            "confirmation_gate": numeric_gate(report, manifest, pairs, families, arms)}


def markdown(summary):
    lines = ["# Paired quality audit", "", "Metric deltas are candidate minus baseline. Positive loss gain or score gain is better.",
             "Unlike raw metrics are never averaged. Public train holdouts are sanity checks, not competition scores.", ""]
    for report in summary["reports"]:
        lines += [f"## {report['purpose']}: {report['status']}", "", f"Original report SHA-256: `{report['report_sha256']}`.", "",
                  "| Family | Budget | Metric | Baseline median | Candidate median | Median gain | W/T/L/unpaired | Seconds B/C |", "|---|---:|---|---:|---:|---:|---|---:|"]
        for family in report["families"]:
            num = lambda x: "—" if x is None else f"{x:.6g}"
            gain = family["median_gain"]
            formatted = (f"{100 * gain:+.2f}%" if family["gain_kind"] == "relative_loss" else f"{gain:+.6f}") if gain is not None else "unverified"
            counts = family["outcomes"]
            wtl = "/".join(str(counts.get(k, 0)) for k in ("win", "tie", "loss", "unpaired"))
            lines.append(f"| {family['case']} | {num(family['time_budget_seconds'])} | {family['metric']} | {num(family['median_baseline'])} | {num(family['median_candidate'])} | {formatted} | {wtl} | {family['baseline_seconds']:.1f}/{family['candidate_seconds']:.1f} |")
        lines += ["", "| Arm | Valid/planned outputs | v1/v2 full refits | Seconds | Timeouts | Telemetry available |", "|---|---:|---:|---:|---:|---:|"]
        for arm, data in report["arms"].items():
            v2 = data['full_refit_v2_verified'] if data['full_refit_v2_verified'] is not None else 'pending'
            lines.append(f"| {arm} | {data['valid_outputs']}/{data['planned_trials']} | {data['full_refit_v1_verified']}/{v2} | {data['elapsed_seconds']:.3f} | {data['timed_out_trials']} | {data['telemetry_available_trials']} |")
        gate = report["confirmation_gate"]
        if gate["assessed"]:
            lines += ["", f"Fixed numerical confirmation quality gate: **{'PASS' if gate['quality_gate_passed'] else 'FAIL'}**.", ""]
            lines += [f"- {key}: {value}" for key, value in gate["checks"].items()]
            lines += ["", "This numerical check alone does not establish default eligibility; remaining host-path/refit/budget/public evidence must be reviewed."]
        lines.append("")
    lines += ["Small fixed synthetic samples do not establish private-score improvement. Re-audit v2 must remain separate from the original v1 flags. Missing telemetry is not a measured zero RSS/CPU.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--reaudit", action="append", type=Path, default=[],
                        help="Optional source-bound v2 proofs; matched by original report SHA")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args(argv)
    reaudits = {}
    for path in args.reaudit:
        proof = json.loads(path.read_text(encoding="utf-8"))
        original = proof["original_report_sha256"]
        if original in reaudits:
            parser.error("Duplicate re-audit for the same original report")
        reaudits[original] = path
    report_hashes = {digest(path) for path in args.report}
    if not set(reaudits) <= report_hashes:
        parser.error("A re-audit has no matching report")
    summary = {"version": 1, "kind": "independent_descriptive_paired_audit",
               "auditor_sha256": digest(__file__),
               "reports": [summarize_report(p, reaudits.get(digest(p))) for p in args.report]}
    originals = {p.resolve() for p in args.report + args.reaudit}
    outputs = {args.json_output.resolve(), args.markdown_output.resolve()}
    if (len(outputs) != 2 or any(output in originals or output == Path(__file__).resolve()
            or any(output.is_relative_to(path.parent) for path in args.report)
            for output in outputs)):
        parser.error("Audit must write distinct outputs outside original report directories")
    for output in (args.json_output, args.markdown_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
