"""Run the optional official TabM model locally; never modifies task submission.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_lab.tabular_neural import NeuralConfig, availability, run_neural_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, nargs="?", help="Path to task.md")
    parser.add_argument("--output", type=Path, help="New experiment directory (must not exist)")
    parser.add_argument("--check", action="store_true", help="Check optional packages without training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--time-budget", type=float, default=3600, help="Seconds, maximum 43200")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validation", choices=("auto", "random", "grouped", "time"), default="auto")
    parser.add_argument("--time-column")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(availability(), indent=2))
        return 0 if availability()["available"] else 2
    if args.task is None or args.output is None:
        parser.error("task and --output are required unless --check is used")
    if (args.validation == "time") != (args.time_column is not None):
        parser.error("--time-column is required only with --validation time")
    validation = {"strategy": args.validation}
    if args.time_column is not None:
        validation["time_column"] = args.time_column
    config = NeuralConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                          k=args.members, hidden_size=args.hidden_size, seed=args.seed, threads=args.threads)
    def progress(event):
        metric = event.get("inner_metric")
        print(f"{event['phase']}: epoch {event['epoch']}" +
              (f", inner metric {metric:.6f}" if metric is not None else ""), flush=True)
    try:
        report = run_neural_experiment(args.task, args.output, device=args.device,
            time_budget=args.time_budget, validation=validation, config=config, progress=progress)
    except Exception as error:
        print(f"TabM experiment stopped: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"status": report["status"], "metric": report["metric"],
                      "fit": report["fit"], "report": str(args.output / "report.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
