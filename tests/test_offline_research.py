"""Host-local research limits, recipe identity and inner-view invariants."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.offline_research import canonical_recipe_key, run_local_continuation, startup_seen
from agent.research_workspace import ResearchWorkspace
from tests.test_research import make_regression, write_csv


CODE_HASH = "a" * 64


def recipe(alpha=1.0, *, label="Local fit"):
    return {"id": f"local-{alpha}", "label": label, "action": "linear",
            "model_config": make_model_config("linear", "regression", {"alpha": alpha}),
            "feature_config": normalize_feature_config()}


class Baseline:
    code_hash = "b" * 64
    recipe_hash = "c" * 64

    def verify(self):
        pass


class Gate:
    def __init__(self):
        self.held_rows = [1]
        self.manifest = {"task_semantics": {"adaptive_research_allowed": True}}
        self.changed = False

    def check_inputs(self):
        if self.changed:
            raise ValueError("Research inputs changed")

    @property
    def expected_actuals(self):
        raise AssertionError("Local coordinator accessed outer answers")


class State:
    def __init__(self, gate=None):
        self.dialogue_closed = False
        self.gate = gate or Gate()
        self.baseline = Baseline()
        self.report = {"stage": "inner_research", "successful_experiments": 7,
                       "api": {"attempts": 1, "responses": 0}}
        self.timeout = 120
        self.source_changed = False
        self.saved = 0

    def check_source(self):
        if self.source_changed:
            raise ValueError("Original source changed")
        if isinstance(self.gate, ResearchWorkspace):
            self.gate.check_source()

    def research_timeout(self):
        return self.timeout

    def save(self):
        self.saved += 1


class OfflineResearchTests(unittest.TestCase):
    def test_canonical_identity_ignores_presentation_and_separates_implementations(self):
        first, second = recipe(), recipe(label="Different UI label")
        second["id"] = "different-id"
        second.pop("feature_config")
        second.pop("action")
        self.assertEqual(canonical_recipe_key(first, CODE_HASH), canonical_recipe_key(second, CODE_HASH))
        self.assertNotEqual(canonical_recipe_key(first, CODE_HASH), canonical_recipe_key(first, "d" * 64))
        second["target_transform"] = "log1p"
        self.assertNotEqual(canonical_recipe_key(first, CODE_HASH), canonical_recipe_key(second, CODE_HASH))
        self.assertNotEqual(canonical_recipe_key(recipe(1), CODE_HASH), canonical_recipe_key(recipe(2), CODE_HASH))

    def test_custom_recipe_fields_are_not_discarded(self):
        left = {"family": "custom", "id": "used-in-custom-code"}
        right = {"family": "custom", "id": "different-semantic-setting"}
        self.assertNotEqual(canonical_recipe_key(left, CODE_HASH), canonical_recipe_key(right, CODE_HASH))

    def test_startup_dedup_uses_measured_recipes_and_fixed_blends(self):
        first, second = recipe(1), recipe(2)
        rows = [
            {**first, "status": "completed", "selection_metric": 0.0},
            {**recipe(3), "status": "skipped", "selection_metric": None},
            {**recipe(4), "status": "failed", "selection_metric": None},
            {**recipe(5), "status": "completed", "selection_metric": None},
            {"status": "completed", "metric": 0.1,
             "model_config": {"model": "FixedBlend", "parameters": {"weight": .5,
                               "components": [first["model_config"], second["model_config"]]}},
             "component_feature_configs": [first["feature_config"], second["feature_config"]]},
        ]
        previous = startup_seen({"search": {"candidates": rows}}, code_hash=CODE_HASH)
        self.assertEqual(len(previous), 2)
        keys = {canonical_recipe_key(item["recipe"], item["code_hash"]) for item in previous}
        self.assertIn(canonical_recipe_key(first, CODE_HASH), keys)
        self.assertIn(canonical_recipe_key({"components": [second, first]}, CODE_HASH), keys)

    def test_startup_selection_fit_is_seen_when_later_verification_fails_or_is_skipped(self):
        rows = [{**recipe(alpha), "status": status, "selection_metric": .1,
                 "metric": None} for alpha, status in ((1, "failed"), (2, "skipped"), (3, "timed_out"))]
        rows.extend({**recipe(alpha), "status": "completed", "selection_metric": value,
                     "metric": None} for alpha, value in ((4, float("nan")), (5, True)))
        previous = startup_seen({"candidates": rows}, code_hash=CODE_HASH)
        self.assertEqual(len(previous), 3)
        keys = {canonical_recipe_key(item["recipe"], item["code_hash"]) for item in previous}
        for alpha in (1, 2, 3):
            self.assertIn(canonical_recipe_key(recipe(alpha), CODE_HASH), keys)

    def test_unique_trials_are_bounded_and_do_not_change_llm_or_api_counts(self):
        state, called = State(), []
        initial_baseline = state.baseline

        def evaluate(item):
            called.append(item)
            item["label"] = "callback mutation cannot edit caller queue"
            return {"verified": True, "accepted": len(called) == 2,
                    "reason": "improved" if len(called) == 2 else "not_better"}

        items = [recipe(value) for value in (1, 1, 2, 3, 4)]
        original = deepcopy(items)
        result = run_local_continuation(state, items, evaluate=evaluate, code_hash=CODE_HASH, max_trials=3)
        self.assertEqual(result["reason"], "trial_limit")
        self.assertEqual((result["attempted"], result["duplicates"], result["verified"], result["improvements"]), (3, 1, 3, 1))
        self.assertEqual(state.report["local_experiments"], 3)
        self.assertEqual(state.report["successful_experiments"], 7)
        self.assertEqual(state.report["api"], {"attempts": 1, "responses": 0})
        self.assertIs(state.baseline, initial_baseline)
        self.assertEqual(items, original)

    def test_completed_startup_and_same_code_trials_are_deduplicated(self):
        state, called = State(), []
        seen = [{"recipe": recipe(1), "code_hash": CODE_HASH},
                {"recipe": recipe(2), "code_hash": "d" * 64}]
        result = run_local_continuation(state, [recipe(1), recipe(2)],
            evaluate=lambda item: called.append(item) or {"verified": True, "accepted": False},
            seen=seen, code_hash=CODE_HASH)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["model_config"], recipe(2)["model_config"])
        self.assertEqual(result["duplicates"], 1)

    def test_patience_and_inspection_limits_bound_failure_and_duplicate_generators(self):
        state = State()

        def failure(item):
            raise ValueError("secret-and-arbitrary-error-text")

        result = run_local_continuation(state, (recipe(value) for value in range(1, 20)),
            evaluate=failure, code_hash=CODE_HASH, patience=2)
        self.assertEqual((result["reason"], result["failed"], result["attempted"]), ("patience", 2, 2))
        self.assertNotIn("secret-and-arbitrary-error-text", json.dumps(result))
        state = State()

        def duplicates():
            while True:
                yield recipe()

        result = run_local_continuation(state, duplicates(),
            evaluate=lambda item: {"verified": True, "accepted": True},
            code_hash=CODE_HASH, max_inspected=5)
        self.assertEqual((result["reason"], result["inspected"], result["attempted"]), ("inspection_limit", 5, 1))

    def test_finalization_reserve_stops_before_consuming_another_recipe(self):
        state, consumed = State(), []

        def items():
            for value in (1, 2):
                consumed.append(value)
                yield recipe(value)

        def evaluate(item):
            state.timeout = 1
            return {"verified": True, "accepted": True, "reason": "improved"}

        result = run_local_continuation(state, items(), evaluate=evaluate, code_hash=CODE_HASH)
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertEqual(consumed, [1])

    def test_closed_incomplete_and_unavailable_inner_never_consume_queue(self):
        for condition in ("closed", "outer_stage", "incomplete", "no_inner", "no_baseline"):
            with self.subTest(condition=condition):
                state = State()
                if condition == "closed":
                    state.dialogue_closed = True
                elif condition == "outer_stage":
                    state.report["stage"] = "outer_validation"
                elif condition == "incomplete":
                    state.gate.manifest["task_semantics"]["adaptive_research_allowed"] = False
                elif condition == "no_inner":
                    state.gate.held_rows = []
                else:
                    state.baseline = None

                def forbidden():
                    raise AssertionError("An ineligible continuation consumed the queue")
                    yield

                result = run_local_continuation(state, forbidden(), evaluate=lambda item: None, code_hash=CODE_HASH)
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(result["attempted"], 0)

    def test_input_mutation_is_not_hidden_by_a_training_exception(self):
        for mutation in ("source", "view", "baseline", "closed"):
            with self.subTest(mutation=mutation):
                state = State()

                def evaluate(item):
                    if mutation == "source":
                        state.source_changed = True
                    elif mutation == "view":
                        state.gate.changed = True
                    elif mutation == "baseline":
                        state.baseline = Baseline()
                    else:
                        state.dialogue_closed = True
                    raise ValueError("Training failed after mutation")

                with self.assertRaises((ValueError, RuntimeError)):
                    run_local_continuation(state, [recipe()], evaluate=evaluate, code_hash=CODE_HASH)

    def test_outer_target_intervention_keeps_inner_trials_and_selected_recipe_unchanged(self):
        split = {"fit_rows": list(range(80)), "validation_rows": list(range(80, 100)),
                 "choice": {"strategy": "random"}, "split": {"strategy": "random"}}
        evidence = []
        with tempfile.TemporaryDirectory() as directory:
            for name, change_outer in (("original", False), ("intervened", True)):
                root = Path(directory) / name
                task = make_regression(root)
                if change_outer:
                    write_csv(task.parent / "regression/participants/train.csv",
                        [["row_key", "measurement", "outcome"]] + [
                            [f"train-{index:03d}", index, 900000 + index if index >= 80 else 3 * index + 2]
                            for index in range(100)])
                gate = ResearchWorkspace(task, root / "research-view", frozen_split=split)
                state = State(gate)
                observed, best = [], {"value": float("inf"), "recipe": None}

                def evaluate(item):
                    fitting = gate.research_train.iloc[gate.fit_rows]
                    held = gate.research_train.iloc[gate.held_rows]
                    x = fitting["measurement"].astype(float).to_numpy()
                    y = fitting["outcome"].astype(float).to_numpy()
                    alpha = item["model_config"]["parameters"]["alpha"]
                    slope = np.dot(x - x.mean(), y - y.mean()) / (np.dot(x - x.mean(), x - x.mean()) + alpha)
                    predictions = slope * (held["measurement"].astype(float).to_numpy() - x.mean()) + y.mean()
                    score = float(np.sqrt(np.mean((predictions - held["outcome"].astype(float).to_numpy()) ** 2)))
                    accepted = score < best["value"]
                    if accepted:
                        best.update(value=score, recipe=deepcopy(item))
                    observed.append(score)
                    return {"verified": True, "accepted": accepted, "reason": "improved" if accepted else "not_better"}

                report = run_local_continuation(state, [recipe(10), recipe(1), recipe(.1)],
                    evaluate=evaluate, code_hash=CODE_HASH)
                evidence.append((report, observed, best))
        self.assertEqual(evidence[0], evidence[1])
        self.assertEqual(evidence[0][0]["attempted"], 3)
        self.assertEqual(evidence[0][2]["recipe"]["model_config"]["parameters"]["alpha"], .1)


if __name__ == "__main__":
    unittest.main()
