"""Adaptive local scheduling retains the existing host research boundaries."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import adaptive_local
from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import make_model_config
from tests import test_offline_research as fixtures


CODE_HASH = "a" * 64


def candidate(identifier="linear", action="linear", parameters=None):
    return {"id": identifier, "label": identifier, "action": action,
            "model_config": make_model_config(action, "regression", parameters),
            "feature_config": normalize_feature_config()}


def state():
    result = fixtures.State()
    gate = result.gate
    gate.spec = SimpleNamespace(classification=False, metric=get_metric("RMSE"), target_column="y")
    gate.research_train = pd.DataFrame({"x": np.arange(80), "y": np.arange(80) / 2})
    gate.fit_rows, gate.held_rows = list(range(64)), list(range(64, 80))
    gate.outer_fit_frame = pd.DataFrame({"x": np.arange(80)})
    gate.outer_holdout_frame = pd.DataFrame({"x": np.arange(20)})
    gate.train, gate.test = pd.DataFrame({"x": np.arange(100)}), pd.DataFrame({"x": [1, 2]})
    result.remaining = lambda: 180.0
    return result


class AdaptiveLocalIntegrationTests(unittest.TestCase):
    def invoke(self, current, pool, evaluate, *, seen=(), ranked=(), refinement=None):
        with patch.object(adaptive_local, "exploration_candidates", return_value=pool), \
                patch.object(adaptive_local, "refinement_candidates", side_effect=refinement or (lambda *args, **kwargs: [])):
            return adaptive_local.run_adaptive_continuation(current, evaluate=evaluate, seen=seen,
                                                            code_hash=CODE_HASH, ranked=ranked)

    def test_code_aware_seen_suppresses_only_the_same_implementation(self):
        first, second = candidate(), candidate("trees", "extra_trees")
        alias = {**first, "id": "old", "label": "Old presentation"}
        alias.pop("action")
        for implementation, expected in ((CODE_HASH, ["trees"]), ("d" * 64, ["linear", "trees"])):
            called, current = [], state()

            def evaluate(item, *, maximum):
                called.append(item["id"])
                self.assertGreater(maximum, 0)
                self.assertLessEqual(maximum, current.timeout)
                return {"verified": True, "accepted": True, "metric": {"value": 1}}

            self.invoke(current, [first, second], evaluate, seen=[{"recipe": alias, "code_hash": implementation}])
            self.assertEqual(called, expected)
            self.assertEqual(current.report["api"], {"attempts": 1, "responses": 0})
            self.assertEqual(current.report["successful_experiments"], 7)

    def test_current_inner_scores_precede_startup_prior_on_another_split(self):
        current = state()
        baseline, linear = candidate("baseline", "baseline"), candidate()
        startup = candidate("startup-best", "baseline", {"max_iter": 200})
        rankings = []

        def evaluate(item, *, maximum):
            return {"verified": True, "accepted": True,
                    "metric": {"value": 100 if item["action"] == "baseline" else 1}}

        def refine(classification, metric, ranked, **kwargs):
            rankings.append(deepcopy(ranked))
            return []

        self.invoke(current, [baseline, linear], evaluate, ranked=[startup], refinement=refine)
        self.assertEqual(len(rankings), 1)
        self.assertEqual([item["id"] for item in rankings[0]], ["linear", "baseline", "startup-best"])

    def test_terminal_and_unavailable_states_never_evaluate_candidates(self):
        for condition in ("closed", "outer_stage", "semantics", "no_inner", "no_baseline"):
            with self.subTest(condition=condition):
                current = state()
                if condition == "closed":
                    current.dialogue_closed = True
                elif condition == "outer_stage":
                    current.report["stage"] = "outer_validation"
                elif condition == "semantics":
                    current.gate.manifest["task_semantics"]["adaptive_research_allowed"] = False
                elif condition == "no_inner":
                    current.gate.held_rows = []
                else:
                    current.baseline = None
                result = self.invoke(current, [candidate()], lambda *args, **kwargs: self.fail("Unexpected fit"))
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(result["attempted"], 0)

    def test_existing_source_input_and_baseline_guards_still_propagate_mutation(self):
        for mutation in ("source", "input", "baseline", "closed"):
            with self.subTest(mutation=mutation):
                current = state()

                def evaluate(item, *, maximum):
                    if mutation == "source":
                        current.source_changed = True
                    elif mutation == "input":
                        current.gate.changed = True
                    elif mutation == "baseline":
                        current.baseline = fixtures.Baseline()
                    else:
                        current.dialogue_closed = True
                    raise ValueError("Training failure must not hide mutation")

                with self.assertRaises((ValueError, RuntimeError)):
                    self.invoke(current, [candidate()], evaluate)

    def test_reserved_time_stops_before_any_evaluation(self):
        current = state()
        current.timeout = 1
        result = self.invoke(current, [candidate()], lambda *args, **kwargs: self.fail("Unexpected fit"))
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertEqual(result["attempted"], 0)

    def test_numpy_row_views_do_not_require_ambiguous_array_truthiness(self):
        current, called = state(), []
        current.gate.fit_rows = np.asarray(current.gate.fit_rows)
        current.gate.held_rows = np.asarray(current.gate.held_rows)

        def evaluate(item, *, maximum):
            called.append(item["id"])
            return {"verified": True, "accepted": True, "metric": {"value": 1}}

        result = self.invoke(current, [candidate()], evaluate)
        self.assertEqual(called, ["linear"])
        self.assertEqual(result["verified"], 1)

    def test_trial_limit_and_real_evaluator_maximum_remain_bounded(self):
        current, limits = state(), []
        pool = [candidate(str(alpha), parameters={"alpha": alpha}) for alpha in range(1, 21)]

        def evaluate(item, *, maximum):
            limits.append(maximum)
            return {"verified": True, "accepted": True, "metric": {"value": 1}}

        result = self.invoke(current, pool, evaluate)
        self.assertEqual(result["attempted"], 12)
        self.assertEqual(len(limits), 12)
        self.assertTrue(all(0 < value <= 10 for value in limits))
        self.assertEqual(current.report["local_search_policy"]["dispatched"], 12)
        self.assertEqual(current.report["api"], {"attempts": 1, "responses": 0})

    def test_training_errors_are_bounded_redacted_and_do_not_replace_baseline(self):
        current = state()
        original_baseline = current.baseline
        pool = [candidate(str(alpha), parameters={"alpha": alpha}) for alpha in range(1, 10)]

        def evaluate(item, *, maximum):
            raise OSError("private-provider-body and secret-key")

        result = self.invoke(current, pool, evaluate)
        self.assertEqual((result["reason"], result["attempted"], result["failed"]), ("patience", 6, 6))
        self.assertIs(current.baseline, original_baseline)
        self.assertNotIn("secret-key", str(result) + str(current.report["local_search_policy"]))

    def test_callback_recipe_mutation_does_not_change_queue_or_timing_identity(self):
        current = state()
        pool = [candidate()]
        original = deepcopy(pool)

        def evaluate(item, *, maximum):
            item["model_config"]["parameters"]["alpha"] = 999
            return {"verified": True, "accepted": True, "metric": {"value": 1}}

        self.invoke(current, pool, evaluate)
        self.assertEqual(pool, original)
        observation = current.report["local_search_policy"]["observations"][0]
        self.assertNotIn("999", observation["key"])


if __name__ == "__main__":
    unittest.main()
