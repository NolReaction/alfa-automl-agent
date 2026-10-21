"""Audit of the history schema proposer on the synthetic catalog (history_schema_cases.py); no models, no API.

    python experiment_history_schema.py --mode audit      # frozen series, seeds 211 907 4003
    python experiment_history_schema.py --mode confirm    # after the code freeze, new tables: seeds 8081 12007
    python experiment_history_schema.py --mode scaling    # 1000 / 10000 / 100000 rows with the default caps
    python experiment_history_schema.py --mode exploratory --seeds 1 2   # any other seeds: reported as exploratory

For every case and seed the proposer is called once on the case table; the run checks the hard contract (shape,
HistoryConfig.from_json, allowed columns, caps, JSON, unchanged inputs), compares every proposal's evidence with
the brute-force oracle and with the accepted HistoryEncoder, repeats the call with shuffled rows, columns and lists,
and counts with the hidden true object id how many history rows match another object. The truth is used only here,
after the call. Results go to data/experiments/history_schema/<mode> (ignored by Git): results.json, cases.csv,
proposals.csv, manifest.json with the code hashes, the caps and the git commit.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import tracemalloc

from agent.history_candidate import HistoryConfig
from agent.history_schema_proposer import DEFAULT_LIMITS, propose_history_schemas
import history_schema_cases as cases


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "experiments" / "history_schema"
SEEDS = {"audit": (211, 907, 4003), "confirm": (8081, 12007)}
SCALING_ROWS = (1000, 10_000, 100_000)
SCALING_SEED = 211
CODE = ("agent/history_schema_proposer.py", "history_schema_cases.py", "experiment_history_schema.py")
TOP_FIELDS = {"version", "status", "reason", "scope", "auto_activate", "proposals", "limits"}
PROPOSAL_FIELDS = {"config", "evidence", "identity_verified", "reason"}
FORBIDDEN = ("accepted", "prediction", "weight")


def write_file(path: Path, write) -> None:
    """Write through a temporary file and replace the target in one step; a failure keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            write(file)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def write_json(path: Path, value) -> None:
    write_file(path, lambda file: file.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))


def write_csv(path: Path, rows: list[dict]) -> None:
    def write(file):
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    write_file(path, write)


def manifest(mode: str, seeds) -> dict:
    def git(*arguments):
        completed = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    status = git("status", "--porcelain", "--", *CODE)
    return {"mode": mode, "series": "exploratory" if mode == "exploratory" else mode, "seeds": list(seeds),
            "commit": git("rev-parse", "HEAD"), "code_uncommitted": bool(status),
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE},
            "caps": dict(DEFAULT_LIMITS), "python": platform.python_version(), "platform": platform.platform(),
            "api_calls": 0}


# ---------------------------------------------------------------- checks of one call

def permuted(case: cases.Case, seed: int) -> tuple[dict, dict]:
    rng = random.Random(f"{seed}:order:{case.name}")
    order = list(range(len(case.truth)))
    rng.shuffle(order)
    names = list(case.raw_columns)
    rng.shuffle(names)
    raw = {name: [case.raw_columns[name][index] for index in order] for name in names}
    call = copy.deepcopy(case.call)
    for field in ("feature_columns", "allowed_period_columns", "allowed_key_columns", "explicit_key_sets",
                  "numeric_key_columns", "positive_key_columns"):
        rng.shuffle(call[field])
    return raw, call


def violations(case: cases.Case, result: dict, before: tuple, after: tuple) -> list[str]:
    found = []
    call, caps = case.call, DEFAULT_LIMITS
    if set(result) != TOP_FIELDS:
        found.append("top_level_fields")
    if (result.get("version"), result.get("scope"), result.get("auto_activate")) != (1, "inner_fit_features_only",
                                                                                     False):
        found.append("version_scope_or_auto_activate")
    if result.get("status") not in ("proposed", "abstained", "limited"):
        found.append("status")
    if (result.get("reason") is None) != (result.get("status") == "proposed"):
        found.append("reason")
    proposals = result.get("proposals", [])
    if len(proposals) > caps["max_proposals"] or (result.get("status") == "abstained" and proposals):
        found.append("proposal_count")
    if result["limits"]["counts"]["checked_configs"] > caps["max_checked_configs"]:
        found.append("check_cap")
    try:
        text = json.dumps(result, allow_nan=False)
        if any(word in text for word in FORBIDDEN):
            found.append("forbidden_field")
    except (TypeError, ValueError):
        found.append("json")
    if before != after:
        found.append("input_changed")
    for proposal in proposals:
        config = proposal.get("config", {})
        if set(proposal) != PROPOSAL_FIELDS or proposal.get("identity_verified") is not False:
            found.append("proposal_fields_or_identity")
        try:
            if HistoryConfig.from_json(config).to_json() != config:
                found.append("config_round_trip")
        except ValueError:
            found.append("config_rejected_by_adapter")
            continue
        if config["ambiguity_policy"] != "past_unique_only":
            found.append("policy")
        if (config["period_column"] not in call["allowed_period_columns"]
                or not set(config["key_columns"]) <= set(call["allowed_key_columns"])
                or not set(config["numeric_columns"]) <= set(call["numeric_key_columns"])
                or not set(config["positive_columns"]) <= set(call["positive_key_columns"])):
            found.append("column_not_allowed")
        if proposal["evidence"] != cases.oracle_evidence(case.raw_columns, config):
            found.append("evidence_differs_from_oracle")
        if proposal["evidence"] != cases.adapter_evidence(case.raw_columns, config):
            found.append("evidence_differs_from_adapter")
    return found


def run_case(case: cases.Case, seed: int) -> dict:
    before = copy.deepcopy((case.raw_columns, case.call))
    started = time.perf_counter()
    result = propose_history_schemas(case.raw_columns, **case.call)
    seconds = time.perf_counter() - started
    after = (case.raw_columns, case.call)
    problems = violations(case, result, before, after)
    raw, call = permuted(case, seed)
    if propose_history_schemas(raw, **call) != result:
        problems.append("order_dependent")
    admissible = {(period, frozenset(key)) for period, key in case.admissible}
    rows, hit_rank = [], None
    for rank, proposal in enumerate(result["proposals"], 1):
        config = proposal["config"]
        audit = cases.oracle_evidence(case.raw_columns, config, case.truth)
        known = (config["period_column"], frozenset(config["key_columns"])) in admissible
        hit_rank = hit_rank or (rank if known else None)
        rows.append({"rank": rank, "period": config["period_column"], "key": "+".join(config["key_columns"]),
                     "known_schema": known, **proposal["evidence"],
                     **{name: audit[name] for name in ("false_match_rows", "false_applied_rows",
                                                       "correct_applied_rows", "true_past_rows")}})
    proposed = bool(result["proposals"])
    if case.expect == "abstain":
        expectation = result["status"] == "abstained" and result["reason"] == case.reason
    else:
        expectation = hit_rank is not None
    return {"case": case.name, "family": case.family, "seed": seed, "expect": case.expect,
            "status": result["status"], "reason": result["reason"], "proposals": len(result["proposals"]),
            "hit_rank": hit_rank, "expectation_met": expectation,
            "false_abstention": case.expect == "propose" and not proposed,
            "false_proposal": case.expect == "abstain" and proposed,
            "violations": problems, "seconds": seconds, "counts": result["limits"]["counts"],
            "configs": [proposal["config"] for proposal in result["proposals"]], "proposal_rows": rows}


def rename_check(records: list[dict], catalog: list[cases.Case]) -> dict:
    """Is the renamed clean panel proposed exactly like the original once its names are mapped back?"""
    named = {case.name: case for case in catalog}
    by_name = {record["case"]: record for record in records}
    mapping = named["clean_panel_renamed"].renamed_from
    mapped = [{**config, "key_columns": [mapping[name] for name in config["key_columns"]],
               "period_column": mapping[config["period_column"]],
               "numeric_columns": [mapping[name] for name in config["numeric_columns"]],
               "positive_columns": [mapping[name] for name in config["positive_columns"]]}
              for config in by_name["clean_panel_renamed"]["configs"]]
    original = by_name["clean_panel"]["configs"]
    same_sets = sorted(map(_config_id, mapped)) == sorted(map(_config_id, original))
    # automatic keys list their columns by name, so only the column sets per rank are compared
    return {"identical_order": list(map(_config_id, mapped)) == list(map(_config_id, original)), "same_set": same_sets,
            "same_top1": bool(mapped and original) and _config_id(mapped[0]) == _config_id(original[0])}


def _config_id(config: dict) -> tuple:
    return config["period_column"], tuple(sorted(config["key_columns"]))


def deception_check(records: list[dict]) -> dict:
    by_name = {record["case"]: record for record in records}
    one, two = by_name["deceptive_one_object"], by_name["deceptive_two_objects"]
    return {"same_proposals": one["configs"] == two["configs"],
            "false_match_rows_one_object": sum(row["false_match_rows"] for row in one["proposal_rows"]),
            "false_match_rows_two_objects": sum(row["false_match_rows"] for row in two["proposal_rows"])}


def run_series(mode: str, seeds, out_dir: Path, log=print) -> dict:
    records, checks = [], []
    for seed in seeds:
        catalog = cases.build_catalog(seed)
        seed_records = [run_case(case, seed) for case in catalog]
        records += seed_records
        checks.append({"seed": seed, "rename": rename_check(seed_records, catalog),
                       "deception": deception_check(seed_records)})
        log(f"seed {seed}: {sum(r['expectation_met'] for r in seed_records)}/{len(seed_records)} expectations met")
    summary = summarize(records, checks)
    result = {"manifest": manifest(mode, seeds), "summary": summary, "checks": checks, "records": records}
    write_json(out_dir / "results.json", result)
    write_json(out_dir / "manifest.json", result["manifest"])
    write_csv(out_dir / "cases.csv", [{key: (json.dumps(value) if isinstance(value, (list, dict)) else value)
                                       for key, value in record.items() if key not in ("configs", "proposal_rows")}
                                      for record in records])
    write_csv(out_dir / "proposals.csv", [{"case": record["case"], "seed": record["seed"], **row}
                                          for record in records for row in record["proposal_rows"]])
    return result


def summarize(records: list[dict], checks: list[dict]) -> dict:
    propose = [record for record in records if record["expect"] == "propose"]
    abstain = [record for record in records if record["expect"] == "abstain"]
    rows = [row for record in records for row in record["proposal_rows"]]
    return {"runs": len(records), "propose_cases": len(propose),
            "known_schema_in_top4": sum(record["hit_rank"] is not None for record in propose),
            "known_schema_top1": sum(record["hit_rank"] == 1 for record in propose),
            "false_abstentions": sum(record["false_abstention"] for record in propose),
            "abstain_cases": len(abstain), "correct_abstentions": sum(record["expectation_met"] for record in abstain),
            "false_proposals": sum(record["false_proposal"] for record in abstain),
            "contract_violations": sum(len(record["violations"]) for record in records),
            "proposals": len(rows), "proposals_with_false_matches": sum(row["false_match_rows"] > 0 for row in rows),
            "false_match_rows": sum(row["false_match_rows"] for row in rows),
            "false_match_rows_top1": sum(record["proposal_rows"][0]["false_match_rows"] for record in records
                                         if record["proposal_rows"]),
            "false_applied_rows": sum(row["false_applied_rows"] for row in rows),
            "limited_runs": sum(record["status"] == "limited" for record in records),
            "max_checked_configs": max(record["counts"]["checked_configs"] for record in records),
            "max_seconds": max(record["seconds"] for record in records),
            "rename_identical": all(check["rename"]["identical_order"] for check in checks),
            "deception_same_proposals": all(check["deception"]["same_proposals"] for check in checks)}


# ---------------------------------------------------------------- scaling

def run_scaling(out_dir: Path, log=print) -> dict:
    records = []
    for rows in SCALING_ROWS:
        case = cases.scaling_case(rows, SCALING_SEED)
        started = time.perf_counter()
        result = propose_history_schemas(case.raw_columns, **case.call)
        seconds = time.perf_counter() - started
        tracemalloc.start()
        propose_history_schemas(case.raw_columns, **case.call)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        counts = result["limits"]["counts"]
        top = result["proposals"][0]["config"] if result["proposals"] else None
        known = {(period, frozenset(key)) for period, key in case.admissible}
        records.append({"rows": rows, "status": result["status"], "reason": result["reason"],
                        "seconds": seconds, "traced_peak_mib": peak / 2**20, "work_planned": counts["work_planned"],
                        "screened_columns": counts["screened_columns"], "checked_configs": counts["checked_configs"],
                        "search_complete": counts["search_complete"],
                        "top1": "+".join(top["key_columns"]) if top else None,
                        "known_schema_in_top4": any((p["config"]["period_column"],
                                                     frozenset(p["config"]["key_columns"])) in known
                                                    for p in result["proposals"])})
        log(f"{rows} rows: {seconds:.2f} s, {result['status']}")
    result = {"manifest": manifest("scaling", [SCALING_SEED]), "records": records,
              "note": "traced_peak_mib is the tracemalloc peak of Python allocations during the call, not process RSS"}
    write_json(out_dir / "results.json", result)
    write_json(out_dir / "manifest.json", result["manifest"])
    write_csv(out_dir / "scaling.csv", records)
    return result


# ---------------------------------------------------------------- report tables

def render(result: dict) -> str:
    summary, checks = result["summary"], result["checks"]
    lines = [f"series {result['manifest']['series']}, seeds {result['manifest']['seeds']}, "
             f"commit {result['manifest']['commit']}", "",
             "| case | family | status per seed | top-4 known | top-1 key | unambiguous / ambiguous (top-1) | "
             "false match rows top-1 / all | checked max | ms max | violations |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    by_case = {}
    for record in result["records"]:
        by_case.setdefault(record["case"], []).append(record)
    for name, records in by_case.items():
        tops = [record["proposal_rows"][0] for record in records if record["proposal_rows"]]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {:.1f} | {} |".format(
            name, records[0]["family"],
            ", ".join(record["status"] + (f" ({record['reason']})" if record["reason"] else "") for record in records),
            f"{sum(r['hit_rank'] is not None for r in records)}/{len(records)}" if records[0]["expect"] == "propose"
            else "n/a", ", ".join(sorted({row["key"] for row in tops})) or "-",
            ", ".join(f"{row['past_unambiguous_history_rows']}/{row['past_ambiguous_history_rows']}" for row in tops)
            or "-", "{}/{}".format(sum(row["false_match_rows"] for row in tops),
                                   sum(row["false_match_rows"] for r in records for row in r["proposal_rows"])),
            max(r["counts"]["checked_configs"] for r in records), 1000 * max(r["seconds"] for r in records),
            sum(len(r["violations"]) for r in records)))
    lines += ["", "| case | period | key | ranks | unambiguous | ambiguous | eligible | false match | false applied |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups = {}
    for record in result["records"]:
        for row in record["proposal_rows"]:
            groups.setdefault((record["case"], row["period"], row["key"]), []).append(row)
    span = lambda rows, name: "{}..{}".format(min(r[name] for r in rows), max(r[name] for r in rows))
    for (name, period, key), rows in groups.items():
        lines.append(f"| {name} | {period} | {key} | {','.join(str(r['rank']) for r in rows)} | "
                     f"{span(rows, 'past_unambiguous_history_rows')} | {span(rows, 'past_ambiguous_history_rows')} | "
                     f"{span(rows, 'eligible_rows')} | {span(rows, 'false_match_rows')} | "
                     f"{span(rows, 'false_applied_rows')} |")
    lines += ["", json.dumps(summary, ensure_ascii=False), "",
              json.dumps([{"seed": c["seed"], **c["rename"], **c["deception"]} for c in checks], ensure_ascii=False)]
    return "\n".join(lines)


def render_scaling(result: dict) -> str:
    lines = ["| rows | status | seconds | traced peak MiB | work planned | screened | checked | complete | top-1 | "
             "known in top-4 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["records"]:
        lines.append(f"| {r['rows']} | {r['status']} | {r['seconds']:.2f} | {r['traced_peak_mib']:.1f} | "
                     f"{r['work_planned']} | {r['screened_columns']} | {r['checked_configs']} | "
                     f"{r['search_complete']} | {r['top1']} | {r['known_schema_in_top4']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("audit", "confirm", "scaling", "exploratory"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="exploratory mode only")
    parser.add_argument("--out-dir", type=Path, help=f"default: {OUT_DIR}/<mode>")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if (args.mode == "exploratory") != bool(args.seeds):
        parser.error("--seeds is required for and allowed only with --mode exploratory")
    out_dir = args.out_dir or OUT_DIR / args.mode
    if args.mode == "scaling":
        print(render_scaling(run_scaling(out_dir)))
    else:
        print(render(run_series(args.mode, args.seeds or SEEDS[args.mode], out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
