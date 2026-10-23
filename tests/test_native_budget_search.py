"""Native long-run search: useful promotions beyond 64 fits, unchanged deadlines."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from agent.adaptive_search import AdaptiveSearch
from agent.native_budget_search import NativeBudgetSearch
from agent.offline_research import run_deadline_continuation
from agent.research import run_agent
from agent.research_v3 import _track_local_candidate, run_agent_v3
from tests import test_adaptive_cli as cli_tests
from tests.test_adaptive_search import measured
from tests.test_offline_research import CODE_HASH, State, recipe
from tests.test_research import make_regression


class NativeBudgetSearchTests(unittest.TestCase):
    def test_original_proposer_and_coordinator_bounds_stay_explicit(self):
        with self.assertRaises(ValueError):
            AdaptiveSearch(False, "RMSE", max_trials=97)
        with self.assertRaises(ValueError):
            NativeBudgetSearch(False, "RMSE", max_trials=257)
        with self.assertRaises(ValueError):
            run_deadline_continuation(State(), [], adaptive=None,
                evaluate=lambda item: measured(), code_hash=CODE_HASH, max_trials=97)
        for value in (None, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_deadline_continuation(State(), [], adaptive=None,
                    evaluate=lambda item: measured(), code_hash=CODE_HASH, budget_mode=value)

    def test_native_queue_reaches_1200_tree_promotions_despite_16_warmup_fits(self):
        for classification, metric in ((False, "RMSE"), (True, "Macro F1")):
            with self.subTest(classification=classification), \
                    patch("agent.adaptive_search._load_optuna", return_value=None):
                state = State()
                # Warmup consumes native trial slots just as in the real archive.
                # It is not optimizer evidence; adaptive.observe ignores it.
                warmup = [recipe(index + 20) for index in range(16)]
                proposer = NativeBudgetSearch(classification, metric)
                before_api = deepcopy(state.report["api"])
                result = run_deadline_continuation(state, warmup, adaptive=proposer,
                    evaluate=lambda item: measured(metric, .6 if classification else 40),
                    code_hash=CODE_HASH, max_trials=256, budget_mode=True)
                self.assertEqual(result["attempted"], 256)
                self.assertEqual(result["warmup_attempted"], 16)
                self.assertEqual(result["reason"], "trial_limit")
                self.assertEqual(result["trial_limit_kind"], "safety")
                self.assertEqual(result["improvements"], 0)
                high = {row["family"] for row in result["adaptive"]["trials"]
                        if row["rung"] == 2 and row["status"] == "verified"}
                self.assertTrue({"lightgbm", "catboost"} <= high, high)
                self.assertEqual(state.report["api"], before_api)

    def test_measured_reserve_stops_budget_mode_before_the_ceiling(self):
        state = State()
        proposer = NativeBudgetSearch(False, "RMSE")
        evaluated = []

        def evaluate(item):
            evaluated.append(item)
            if len(evaluated) == 7:
                state.timeout = 0
            return measured()

        with patch("agent.adaptive_search._load_optuna", return_value=None):
            result = run_deadline_continuation(state, [], adaptive=proposer,
                evaluate=evaluate, code_hash=CODE_HASH, max_trials=256, budget_mode=True)
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertEqual(result["attempted"], 7)
        self.assertIsNone(result["adaptive"]["pending"])

    def test_frozen_candidate_retention_cap_depends_on_explicit_policy(self):
        for enabled, limit in ((False, 70), (True, 262)):
            with self.subTest(enabled=enabled):
                state = SimpleNamespace(adaptive_budget=enabled, _local_frozen_attempts=[])
                retained = []
                for _ in range(limit):
                    candidate = object()
                    retained.append(candidate)
                    _track_local_candidate(state, candidate)
                    _track_local_candidate(state, candidate)
                self.assertEqual(len(state._local_frozen_attempts), limit)
                with self.assertRaisesRegex(RuntimeError, "frozen-attempt limit"):
                    _track_local_candidate(state, object())

    def test_router_forwards_budget_policy_without_changing_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as execute:
                run_agent(task, None, local_only=True, adaptive_search=True,
                          adaptive_budget=True, time_budget=360)
            self.assertTrue(execute.call_args.kwargs["adaptive_budget"])
            with patch("agent.research_v3.run_agent_v3", return_value=Path(directory)) as execute:
                run_agent(task, None, adaptive_search=True, time_budget=360)
            self.assertNotIn("adaptive_budget", execute.call_args.kwargs)

    def test_invalid_native_budget_requests_fail_before_reading_the_task(self):
        for function in (run_agent, run_agent_v3):
            for options in ({"adaptive_budget": True}, {"adaptive_budget": 1},
                            {"adaptive_budget": "true"}):
                with self.subTest(function=function.__name__, options=options), \
                        self.assertRaisesRegex(ValueError, "adaptive_budget"):
                    function(Path("missing-task.md"), None, **options)
        with self.assertRaisesRegex(ValueError, "greater than 300"):
            run_agent_v3(Path("missing-task.md"), None, adaptive_search=True,
                         adaptive_budget=True, time_budget=300)

    def test_cli_explicit_budget_mode_has_distinct_quality_evidence(self):
        harness = cli_tests.AdaptiveCliTests()
        options = harness.run_offline("--research-search", "adaptive-budget")
        self.assertTrue(options["quality_options"]["adaptive_search"])
        self.assertTrue(options["quality_options"]["adaptive_budget"])
        self.assertNotIn("adaptive_budget", harness.run_offline()["quality_options"])


if __name__ == "__main__":
    unittest.main()
