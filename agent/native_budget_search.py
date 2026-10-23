"""Explicit long-run search without the legacy native 64-trial bottleneck.

The host still owns the original deadline, finalization reserve and admission.
This only raises a finite safety ceiling; it never starts another time budget,
reuses outer feedback, or repeats fits to keep the machine busy.
"""

from agent.adaptive_search import AdaptiveSearch


class NativeBudgetSearch(AdaptiveSearch):
    MAX_TRIALS = 256

    def __init__(self, classification, metric_name, *, seed=42, max_trials=256):
        super().__init__(classification, metric_name, seed=seed, max_trials=max_trials)
