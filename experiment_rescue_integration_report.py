"""Summarize saved native reports without importing the solver or running work.

Only the explicit output is written. Missing evidence is null, not zero or a
successful qualification. This is a report index, not a receipt/resource audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


def _object(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(value):
    return value[:1000] if isinstance(value, str) else None


def _number(value, *, minimum=0):
    return value if (type(value) in {int, float} and math.isfinite(value)
                     and (minimum is None or value >= minimum)) else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _flag(value):
    return value if type(value) is bool else None


def _metric(value):
    value = _object(value)
    if not value:
        return None
    return {"name": _text(value.get("name")),
            "value": _number(value.get("value"), minimum=None),
            "higher_is_better": _flag(value.get("higher_is_better"))}


def _walk(value, path="$"):
    if isinstance(value, dict):
        yield path, value
        for name, item in value.items():
            yield from _walk(item, path + "." + name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _trial(row):
    feedback = _object(row.get("feedback"))
    return {**{name: _text(row.get(name)) for name in
               ("kind", "family", "phase", "status", "reason", "error_type")},
            **{name: _text(row.get(name)) or _text(feedback.get(name)) for name in ("code_hash", "recipe_hash")},
            "attempted": _flag(row.get("attempted")),
            "verified": _flag(feedback.get("verified", row.get("verified"))),
            "accepted": _flag(feedback.get("accepted", row.get("accepted"))),
            "feedback_reason": _text(feedback.get("reason")),
            "metric": _metric(feedback.get("metric", row.get("metric"))),
            "elapsed_seconds": _number(feedback.get("elapsed_seconds", row.get("elapsed_seconds")))}


def _search(value):
    value = _object(value)
    return {**{name: _text(value.get(name)) for name in ("status", "phase", "best_candidate")},
            **{name: _count(value.get(name)) for name in
               ("planned", "completed", "selection_completed", "verified")},
            "candidates": [{**{name: _text(row.get(name)) for name in
                               ("id", "model", "status", "phase", "rejection_reason", "error")},
                            "selected": _flag(row.get("selected")),
                            **{name: _number(row.get(name), minimum=None) for name in
                               ("metric", "selection_metric", "verification_metric")},
                            "elapsed_seconds": _number(row.get("elapsed_seconds")),
                            "durations": {name: _number(_object(row.get("durations")).get(name))
                                          for name in ("fit_seconds", "predict_seconds", "total_seconds")}}
                           for row in _rows(value.get("candidates"))]}


def summarize_report(path):
    """Read exactly one supplied JSON file; do not follow its artifact paths."""
    path = Path(path).expanduser().resolve()
    result = {"source": {"path": str(path), "sha256": None, "bytes": None}, "load_status": "error"}
    try:
        raw = path.read_bytes()
        result["source"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        report = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("Native report must be a JSON object")
    except (OSError, UnicodeError, ValueError) as error:
        result["error_type"] = type(error).__name__
        return result
    api, local = _object(report.get("api")), _object(report.get("local_research"))
    extensions = _object(report.get("research_extensions"))
    final = _object(_object(report.get("evaluations")).get("final_check"))
    publication = _object(report.get("publication"))
    allocation = _object(report.get("time_allocation"))
    baseline = _object(report.get("baseline_search"))
    adaptive = _object(local.get("adaptive"))
    stages = []
    for row in _rows(report.get("stages")):
        elapsed = _number(row.get("elapsed_seconds"))
        stages.append({**{name: _text(row.get(name)) for name in ("id", "status", "reason")},
                       "elapsed_seconds": elapsed,
                       "timing_status": "reported" if elapsed is not None else "unknown"})
    observations, execution_times = [], []
    for location, row in _walk(report):
        resources = row.get("resources")
        if isinstance(resources, dict):
            observations.append({"source": location + ".resources",
                "available": _flag(resources.get("available")),
                **{name: _number(resources.get(name)) for name in
                   ("peak_rss_bytes", "cpu_seconds", "memory_limit_bytes")},
                "unavailable_reason": _text(resources.get("unavailable_reason")),
                "cleanup_complete": _flag(_object(row.get("cleanup")).get("complete")),
                "timed_out": _flag(row.get("timed_out", resources.get("timed_out"))),
                "memory_limit_exceeded": _flag(row.get("memory_limit_exceeded",
                                                       resources.get("memory_exceeded")))})
        if any(name in row for name in ("exit_code", "host_elapsed_seconds")):
            elapsed, host = _number(row.get("elapsed_seconds")), _number(row.get("host_elapsed_seconds"))
            if elapsed is not None or host is not None:
                execution_times.append({"source": location, "elapsed_seconds": elapsed,
                                        "host_elapsed_seconds": host})
    errors = []
    for event in _rows(report.get("events")):
        if event.get("kind") == "api_error":
            details = _object(event.get("details"))
            errors.append({**{name: _text(details.get(name)) for name in
                              ("error_type", "cause_type", "transport_reason", "code")},
                           "status_code": _count(details.get("status_code"))})
    result.update(load_status="loaded",
        schema=("research_v3" if report.get("research_protocol_version") == 3 else
                "automl" if report.get("mode") == "offline" and isinstance(report.get("search"), dict)
                else "unknown"),
        run={**{name: _text(report.get(name)) for name in ("status", "reason", "mode", "stage", "started_at")},
             "research_protocol_version": _count(report.get("research_protocol_version")),
             "elapsed_seconds": _number(report.get("elapsed_seconds")),
             "time_budget_seconds": _number(report.get("time_budget_seconds"))},
        timings={"stages": stages, "execution_times": execution_times,
                 "estimated_stage_seconds": {name: _number(value) for name, value in
                                              _object(allocation.get("stage_seconds")).items()},
                 "estimated_finalization_seconds": _number(allocation.get("estimated_finalization_seconds")),
                 "note": "Stage estimates are not measured durations; absent stage timings are unknown."},
        api={**{name: _count(api.get(name)) for name in
                ("attempts", "responses", "calls", "usage_reported_responses",
                 "prompt_tokens", "completion_tokens", "total_tokens")},
             "used": _flag(api.get("used")), "errors": errors},
        local_research={**{name: _text(local.get(name)) for name in ("status", "reason", "policy")},
                        **{name: _count(local.get(name)) for name in
                           ("attempted", "verified", "improvements", "failed", "warmup_attempted", "adaptive_attempted")},
                        "extension_hook_called": _flag(local.get("extension_hook_called")),
                        "extension_schedule": _text(local.get("extension_schedule")),
                        **{name: _number(local.get(name)) for name in
                           ("extension_hook_started_remaining_seconds", "extension_hook_ended_remaining_seconds")},
                        "extension_hook_error_type": _text(local.get("extension_hook_error_type")),
                        "adaptive_status": _text(adaptive.get("status")),
                        "adaptive_stop_reason": _text(adaptive.get("stop_reason")),
                        "trials": [_trial(row) for row in _rows(local.get("trials"))]},
        research_extensions={**{name: _text(extensions.get(name)) for name in ("status", "reason", "policy")},
                             "schedule": _text(extensions.get("schedule")),
                             "order": ([_text(item) for item in extensions["order"]]
                                       if isinstance(extensions.get("order"), list) else None),
                             **{name: _count(extensions.get(name)) for name in ("attempted", "verified", "accepted")},
                             "trials": [_trial(row) for row in _rows(extensions.get("trials"))]},
        baseline_search={"status": _text(baseline.get("status")), "reason": _text(baseline.get("reason")),
                         "search": _search(baseline.get("search"))},
        search=_search(report.get("search")),
        final_check={"baseline": _metric(final.get("baseline")), "candidate": _metric(final.get("candidate")),
                     **{name: _text(final.get(name)) for name in ("selected", "candidate_acceptance", "scope")},
                     "selection_uses_outer": _flag(final.get("selection_uses_outer")),
                     "unbiased_estimate": _flag(final.get("unbiased_estimate")),
                     "errors": {name: _text(value) for name, value in _object(final.get("errors")).items()}},
        publication={"new_result_published": _flag(publication.get("new_result_published")),
                     **{name: _text(publication.get(name)) for name in ("recipe", "reason")},
                     "published_recipe": _text(report.get("published_recipe")),
                     "output_check_valid": _flag(_object(report.get("output_check")).get("valid")),
                     "output_valid": _flag(report.get("output_valid")),
                     "reported_best_metric": _metric(report.get("best_metric", report.get("metric"))),
                     "refit_reported_recipes": list(_object(report.get("refit_execution")))},
        resources={"qualification": "unknown", "observations": observations,
                   "note": "Native observations only; no complete process-tree, receipt or refit audit is performed."})
    return result


def build_summary(paths):
    reports = [summarize_report(path) for path in paths]
    return {"schema_version": 1, "kind": "rescue_native_report_summary",
            "status": "completed" if all(row["load_status"] == "loaded" for row in reports) else "incomplete",
            "missing_values": "null means absent or unusable evidence; counters and qualification are not inferred.",
            "reports": reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if any(output == path.expanduser().resolve() for path in args.reports):
        print("Error: output must not overwrite an input report.", file=sys.stderr)
        return 2
    temporary = None
    try:
        summary = build_summary(args.reports)
        content = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".rescue-summary-", suffix=".tmp",
                                         mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(output)
    except (OSError, ValueError) as error:
        print(f"Error: could not write summary ({type(error).__name__}).", file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Native reports: {len(summary['reports'])}; {summary['status']}. Summary: {output}")
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
