"""Local, time-budgeted continuation of the existing INNER-only TPE search.

The larger trial ceiling is a safety bound for long workstation jobs, not a
request to repeat fits until it is reached. The host retains the clock, row
splits and finalization reserve. Native contest callers keep their own cap.
"""
from __future__ import annotations

import math

from agent.adaptive_search import AdaptiveSearch


class AppAdaptiveSearch(AdaptiveSearch):
    MAX_TRIALS = 512

    def __init__(self, classification: bool, metric_name: str, *, seed=42, max_trials=256):
        super().__init__(classification, metric_name, seed=seed, max_trials=max_trials)

    def trial_timeout(self, recipe, *, remaining, selection_window):
        """Spend a measured share, bounded by the existing selection deadline.

        A slow failed family cannot consume the entire search. Once observed,
        same-family fits estimate cost, scaling a promoted trial's tree count.
        This is a timeout estimate, never evidence that the fit will finish.
        """
        for value in (remaining, selection_window):
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("Adaptive budgets must be finite numbers")
        if remaining <= 1 or selection_window <= 0:
            return 0.0
        report = self.report()
        ladder = report["resource_ladders"][recipe["action"]]
        parameter = ladder["parameter"]
        resource = recipe["model_config"]["parameters"].get(parameter) if parameter else None
        estimates = []
        for item in report["trials"]:
            if (item["family"] != recipe["action"] or item["status"] != "verified"
                    or item["elapsed_seconds"] is None):
                continue
            scale = max(1.0, resource / item["resource"]) if resource and item["resource"] else 1.0
            estimates.append(item["elapsed_seconds"] * scale)
        # Cold families receive a substantial but finite share. Subsequent
        # trials use the slowest observed family fit plus a startup margin.
        allowance = max(30.0, min(600.0, selection_window / 5))
        if estimates:
            allowance = max(15.0, max(estimates) * 2 + 5.0)
        return max(0.0, min(1800.0, remaining, allowance))

    def summary(self, *, stop_reason=None):
        report = self.report()
        return {
            "strategy": "adaptive", "max_trials": self.max_trials,
            "proposed": report["proposed"], "observed": report["observed"],
            "stop_reason": stop_reason or report["stop_reason"],
            "optimizer": report["optimizer"]["backend"],
            # Reports/checkpoints survive a kill, but no claim is made that
            # the optimizer or fitted model state can yet resume across jobs.
            "resumable": False,
        }
