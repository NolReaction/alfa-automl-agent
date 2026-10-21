"""Development CLI. Only the explicitly named fetch command uses the network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from research_lab.integrity import ROOT, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("catalog", "fetch", "estimate", "prepare"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, default=ROOT / "research_lab/datasets.json")
        if name != "catalog":
            profiles = ("pilot", "suite", "all", "confirmation") if name == "fetch" else ("pilot", "suite", "confirmation", "final", "fixture-smoke")
            sub.add_argument("--profile", choices=profiles, default="pilot")
            sub.add_argument("--cache", type=Path, default=Path("data/lab-cache"))
        if name in {"prepare", "estimate"}:
            sub.add_argument("--time-budget", type=int, default=360)
            sub.add_argument("--threads", type=int, default=8)
            sub.add_argument("--memory-gib", type=int, default=24)
            sub.add_argument("--seeds", type=int, nargs="+")
        if name == "estimate":
            sub.add_argument("--dry-run", action="store_true", required=True)
        if name == "prepare":
            from research_lab.planning import CANDIDATE_SHA
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--incumbent-root", type=Path, required=True)
            sub.add_argument("--candidate-root", type=Path, required=True)
            sub.add_argument("--candidate-commit", default=CANDIDATE_SHA)
            sub.add_argument("--python", type=Path, default=Path(sys.executable))
            sub.add_argument("--self-check", action="store_true")
    for name in ("run", "resume", "re-audit", "gpu-re-audit", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
    gpu = commands.add_parser("gpu")
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--cpu-plan", type=Path, required=True)
    gpu.add_argument("--enable", action="store_true")
    gpu.add_argument("--time-budget", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            from research_lab.catalog import load_manifest
            result = load_manifest(args.manifest)
        elif args.command == "fetch":
            from research_lab.catalog import load_manifest
            from research_lab.fetch import fetch_manifest
            result = fetch_manifest(load_manifest(args.manifest), args.cache, args.profile)
            # Evidence of every unavailable/skipped source, not only successful
            # cache objects. Never included in the immutable cache itself.
            write_json(Path("data/lab-fetch-results") / (args.profile + "-latest.json"), result)
        elif args.command in {"estimate", "prepare"}:
            from research_lab.planning import estimate, prepare
            options = {"manifest": args.manifest, "profile": args.profile, "cache": args.cache,
                       "time_budget_seconds": args.time_budget, "threads": args.threads,
                       "memory_gib": args.memory_gib, "seeds": args.seeds}
            if args.command == "estimate":
                result = estimate(**options)
            else:
                result = prepare(**options, output=args.output, incumbent_root=args.incumbent_root,
                                 candidate_root=args.candidate_root, python=args.python,
                                 candidate_commit=args.candidate_commit, self_check=args.self_check)
                result = {"plan_id": result["plan_id"], "output": str(args.output), "estimate": result["estimate"]}
        elif args.command in {"run", "resume"}:
            from research_lab.runner import run_plan
            result = run_plan(args.output, resume=args.command == "resume")
        elif args.command == "re-audit":
            from research_lab.audit import re_audit
            result = re_audit(args.output)
        elif args.command == "gpu-re-audit":
            from research_lab.gpu import re_audit_gpu
            result = re_audit_gpu(args.output)
        elif args.command == "report":
            from research_lab.report import report
            result = report(args.output)
        else:
            from research_lab.gpu import run_gpu
            result = run_gpu(args.output, args.cpu_plan, time_budget_seconds=args.time_budget, enabled=args.enable)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and (result.get("valid") is False or result.get("status") == "integrity_failed") else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__, "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
