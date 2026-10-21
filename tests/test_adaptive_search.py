"""Pure contracts for bounded, INNER-only adaptive proposals; no model fitting."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent import adaptive_search
from agent.adaptive_search import AdaptiveSearch
from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import ACTIONS, make_model_config, validate_model_config


def measured(name="RMSE", value=1.0, *, accepted=False):
    metric = get_metric(name)
    return {"verified": True, "accepted": accepted, "scope": "inner_selection",
            "reason": "improved" if accepted else "not_better",
            "metric": {"name": metric.name, "value": value,
                       "higher_is_better": metric.higher_is_better}}


def executable(recipe):
    return {"action": recipe["action"], "model_config": recipe["model_config"],
            "feature_config": recipe["feature_config"]}


class AdaptiveSearchTests(unittest.TestCase):
    def test_constructor_rejects_task_metric_mismatch_and_unbounded_controls(self):
        invalid = ((False, "Macro F1", {}), (True, "RMSE", {}),
                   ("false", "RMSE", {}), (False, "unknown", {}),
                   (False, "RMSE", {"seed": True}),
                   (False, "RMSE", {"seed": -1}),
                   (False, "RMSE", {"seed": 2 ** 32}),
                   (False, "RMSE", {"max_trials": True}),
                   (False, "RMSE", {"max_trials": 0}),
                   (False, "RMSE", {"max_trials": 97}))
        for classification, metric, kwargs in invalid:
            with self.subTest(classification=classification, metric=metric, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AdaptiveSearch(classification, metric, **kwargs)

    def test_constructor_and_fixed_warmup_do_not_import_optional_optimizer(self):
        with patch.object(adaptive_search, "_load_optuna", side_effect=AssertionError("eager optional import")):
            search = AdaptiveSearch(False, "RMSE")
            actions = []
            for index in range(5):
                recipe = search.suggest()
                actions.append(recipe["action"])
                search.observe(recipe, measured(value=10 - index))
            self.assertEqual(set(actions), set(ACTIONS))
            self.assertEqual(len(actions), len(set(actions)))

    def test_pending_suggestion_is_idempotent_and_detached(self):
        search = AdaptiveSearch(False, "RMSE")
        original = search.suggest()
        again = search.suggest()
        self.assertEqual(original, again)
        self.assertIsNot(original, again)
        first_parameter = next(iter(again["model_config"]["parameters"]))
        again["model_config"]["parameters"][first_parameter] = "caller mutation"
        again["feature_config"]["missing_indicators"] = True
        self.assertEqual(search.suggest(), original)
        self.assertEqual(search.report()["proposed"], 1)
        self.assertTrue(search.observe(original, measured()))
        self.assertNotEqual(search.suggest()["id"], original["id"])

    def test_every_suggestion_is_valid_normal_recipe_and_deduplicated(self):
        for classification, metric in ((False, "RMSE"), (True, "Macro F1")):
            with self.subTest(classification=classification), patch.object(adaptive_search, "_load_optuna", return_value=None):
                search = AdaptiveSearch(classification, metric, seed=2031, max_trials=40)
                identities = set()
                while (recipe := search.suggest()) is not None:
                    self.assertIn(recipe["action"], ACTIONS)
                    self.assertEqual(recipe["model_config"], validate_model_config(recipe["model_config"], action=recipe["action"]))
                    self.assertEqual(recipe["feature_config"], normalize_feature_config(recipe["feature_config"]))
                    identity = json.dumps(executable(recipe), sort_keys=True, allow_nan=False)
                    self.assertNotIn(identity, identities)
                    identities.add(identity)
                    search.observe(recipe, measured(metric, 0.5 if classification else 2.0))
                self.assertEqual(len(identities), 40)
                self.assertEqual(search.report()["proposed"], 40)

    def test_fallback_is_deterministic_for_seed_and_feedback(self):
        def run(seed):
            with patch.object(adaptive_search, "_load_optuna", return_value=None):
                search = AdaptiveSearch(False, "MAE", seed=seed, max_trials=28)
                proposals = []
                while (recipe := search.suggest()) is not None:
                    proposals.append(recipe)
                    search.observe(recipe, measured("MAE", 2.0 + len(proposals) % 7))
                return proposals, search.report()

        left, left_report = run(312)
        right, right_report = run(312)
        other, _ = run(313)
        self.assertEqual(left, right)
        self.assertEqual(left_report, right_report)
        self.assertNotEqual(left, other)
        self.assertFalse(left_report["optimizer"]["available"])
        self.assertIn("random", left_report["optimizer"]["backend"])

    def test_optimizer_import_failure_falls_back_without_exposing_error_text(self):
        secret = "secret-in-import-failure"
        with patch.object(adaptive_search, "_load_optuna", side_effect=RuntimeError(secret)) as loader:
            search = AdaptiveSearch(False, "RMSE", max_trials=12)
            while (recipe := search.suggest()) is not None:
                search.observe(recipe, measured())
        report = search.report()
        self.assertEqual(report["proposed"], 12)
        self.assertEqual(report["observed"], 12)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(report["optimizer"]["backend"], "deterministic_random")
        self.assertFalse(report["optimizer"]["available"])
        self.assertEqual(report["optimizer"]["error_type"], "RuntimeError")
        self.assertNotIn(secret, json.dumps(report))

    def test_study_creation_failure_uses_the_same_bounded_fallback(self):
        secret = "secret-in-study-creation"
        create_study = Mock(side_effect=ValueError(secret))
        optimizer = SimpleNamespace(create_study=create_study,
                                    samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()))
        with patch.object(adaptive_search, "_load_optuna", return_value=optimizer):
            search = AdaptiveSearch(False, "RMSE", max_trials=12)
            while (recipe := search.suggest()) is not None:
                search.observe(recipe, measured())
        report = search.report()
        self.assertEqual(create_study.call_count, 1)
        self.assertEqual(report["observed"], 12)
        self.assertEqual(report["optimizer"]["backend"], "deterministic_random")
        self.assertFalse(report["optimizer"]["available"])
        self.assertEqual(report["optimizer"]["error_type"], "ValueError")
        self.assertNotIn(secret, json.dumps(report))

    def test_finite_cap_includes_failed_and_skipped_proposals(self):
        with patch.object(adaptive_search, "_load_optuna", return_value=None):
            for cap in (1, 5, 9):
                with self.subTest(cap=cap):
                    search = AdaptiveSearch(False, "RMSE", max_trials=cap)
                    for _ in range(cap):
                        recipe = search.suggest()
                        self.assertIsNotNone(recipe)
                        search.observe(recipe, {"verified": False, "accepted": False, "reason": "duplicate"})
                    self.assertIsNone(search.suggest())
                    report = search.report()
                    self.assertEqual((report["proposed"], report["observed"]), (cap, cap))
                    self.assertEqual(report["skipped"], cap)
                    self.assertIsNotNone(report["stop_reason"])

    def test_verified_feedback_requires_finite_metric_with_trusted_identity(self):
        mutations = ({"value": math.nan}, {"value": math.inf}, {"value": -math.inf},
                     {"value": True}, {"value": "1.0"}, {"name": "MAE"},
                     {"higher_is_better": True}, {"higher_is_better": 0})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                search = AdaptiveSearch(False, "RMSE")
                recipe = search.suggest()
                feedback = measured()
                feedback["metric"].update(mutation)
                before = search.report()
                with self.assertRaises(ValueError):
                    search.observe(recipe, feedback)
                self.assertEqual(search.report(), before)

    def test_outer_feedback_and_contradictory_acceptance_are_rejected(self):
        invalid = (dict(measured(), scope="outer_validation"),
                   dict(measured(), scope="final_check"),
                   dict(measured(), verified="true"),
                   dict(measured(), accepted="true"),
                   {"verified": False, "accepted": True, "reason": "improved"})
        for feedback in invalid:
            with self.subTest(feedback=feedback):
                search = AdaptiveSearch(False, "RMSE")
                recipe = search.suggest()
                before = search.report()
                with self.assertRaises(ValueError):
                    search.observe(recipe, feedback)
                self.assertEqual(search.report(), before)

    def test_verified_zero_metric_is_observed_even_when_not_accepted(self):
        search = AdaptiveSearch(True, "Macro F1")
        recipe = search.suggest()
        self.assertTrue(search.observe(recipe, measured("Macro F1", 0.0, accepted=False)))
        report = search.report()
        self.assertEqual(report["verified"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["trials"][0]["metric"]["value"], 0.0)
        self.assertFalse(report["trials"][0]["accepted"])

    def test_external_warmup_is_ignored_without_learning_or_clearing_pending(self):
        search = AdaptiveSearch(False, "RMSE")
        pending = search.suggest()
        external = {"id": "external", "action": "linear", "label": "Host warmup",
                    "model_config": make_model_config("linear", "regression", {"alpha": 71}),
                    "feature_config": normalize_feature_config()}
        before = search.report()
        self.assertFalse(search.observe(external, measured(value=0.00001, accepted=True)))
        after = search.report()
        self.assertEqual(after["proposed"], before["proposed"])
        self.assertEqual(after["observed"], before["observed"])
        self.assertEqual(after["verified"], before["verified"])
        self.assertEqual(after["family_failures"], before["family_failures"])
        self.assertEqual(search.suggest(), pending)
        self.assertEqual(after["ignored_external_observations"], before["ignored_external_observations"] + 1)

    def test_cosmetic_recipe_changes_preserve_issued_identity(self):
        search = AdaptiveSearch(False, "RMSE")
        recipe = search.suggest()
        cosmetic = deepcopy(recipe)
        cosmetic["id"] = "different-presentation-id"
        cosmetic["label"] = "Different label"
        self.assertTrue(search.observe(cosmetic, measured()))
        self.assertEqual(search.report()["observed"], 1)

    def test_a_completed_observation_cannot_be_replayed_into_the_optimizer(self):
        search = AdaptiveSearch(False, "RMSE")
        recipe = search.suggest()
        search.observe(recipe, measured())
        before = search.report()
        with self.assertRaises(ValueError):
            search.observe(recipe, measured(value=0.0, accepted=True))
        self.assertEqual(search.report(), before)

    def test_mutated_issued_configuration_is_rejected(self):
        search = AdaptiveSearch(False, "RMSE")
        recipe = search.suggest()
        changed = deepcopy(recipe)
        parameters = changed["model_config"]["parameters"]
        numeric_parameter = next(name for name, value in parameters.items()
                                 if type(value) in (int, float))
        parameters[numeric_parameter] += 1
        before = search.report()
        with self.assertRaises(ValueError):
            search.observe(changed, measured())
        self.assertEqual(search.report(), before)

    def test_repeated_execution_failures_disable_each_family_and_terminate(self):
        with patch.object(adaptive_search, "_load_optuna", return_value=None):
            search = AdaptiveSearch(False, "RMSE", max_trials=96)
            counts = Counter()
            while (recipe := search.suggest()) is not None:
                counts[recipe["action"]] += 1
                search.observe(recipe, {"verified": False, "accepted": False, "reason": "execution_failed"})
            self.assertEqual(set(counts), set(ACTIONS))
            self.assertTrue(all(count == 3 for count in counts.values()))
            report = search.report()
            self.assertEqual(set(report["disabled_families"]), set(ACTIONS))
            self.assertEqual(report["failed"], 15)
            self.assertLess(report["proposed"], report["max_trials"])

    def test_neutral_skips_do_not_accumulate_execution_failures(self):
        search = AdaptiveSearch(False, "RMSE")
        for reason in ("duplicate", "insufficient_budget", "finalization_reserve"):
            recipe = search.suggest()
            search.observe(recipe, {"verified": False, "accepted": False, "reason": reason})
        report = search.report()
        self.assertEqual(report["skipped"], 3)
        self.assertEqual(report["failed"], 0)
        self.assertTrue(all(value == 0 for value in report["family_failures"].values()))
        self.assertEqual(report["disabled_families"], [])

    def test_a_verified_result_resets_only_its_family_failure_streak(self):
        with patch.object(adaptive_search, "_load_optuna", return_value=None):
            search = AdaptiveSearch(False, "RMSE", max_trials=30)
            counts = Counter()
            for _ in range(15):
                recipe = search.suggest()
                action = recipe["action"]
                counts[action] += 1
                if counts[action] == 2:
                    search.observe(recipe, measured(accepted=False))
                else:
                    search.observe(recipe, {"verified": False, "accepted": False, "reason": "execution_failed"})
            self.assertEqual(set(counts.values()), {3})
            report = search.report()
            self.assertEqual(set(report["family_failures"].values()), {1})
            self.assertEqual(report["disabled_families"], [])

    def test_explicit_unavailable_family_is_disabled_without_execution_failure(self):
        search = AdaptiveSearch(False, "RMSE")
        recipe = search.suggest()
        search.observe(recipe, {"verified": False, "accepted": False, "reason": "family_blocked"})
        report = search.report()
        self.assertIn(recipe["action"], report["disabled_families"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["skipped"], 1)

    def test_promotions_preserve_full_recipe_except_iteration_resource(self):
        cases = ((False, "RMSE"), (True, "Macro F1"),
                 (False, "R2"), (True, "LogLoss"))
        for classification, metric in cases:
            with self.subTest(classification=classification, metric=metric), patch.object(adaptive_search, "_load_optuna", return_value=None):
                search = AdaptiveSearch(classification, metric, seed=42, max_trials=80)
                before_by_id = {}
                promotion_count = 0
                while (recipe := search.suggest()) is not None:
                    record = search.report()["trials"][-1]
                    if record["origin"] == "promotion":
                        promotion_count += 1
                        parent = before_by_id[record["parent_id"]]
                        self.assertEqual(record["family"], parent["family"])
                        self.assertEqual(record["rung"], parent["rung"] + 1)
                        self.assertGreater(record["resource"], parent["resource"])
                        old, new = deepcopy(parent["recipe"]), deepcopy(recipe)
                        for value in (old, new):
                            value.pop("id")
                            value.pop("label")
                        differences = [name for name in old["model_config"]["parameters"]
                                       if old["model_config"]["parameters"][name] != new["model_config"]["parameters"][name]]
                        self.assertEqual(len(differences), 1)
                        resource_parameter = differences[0]
                        self.assertIn(resource_parameter, ("iterations", "n_estimators", "max_iter"))
                        new["model_config"]["parameters"][resource_parameter] = old["model_config"]["parameters"][resource_parameter]
                        self.assertEqual(old, new)
                        eligible = [row for row in before_by_id.values()
                                    if row["family"] == parent["family"] and row["rung"] == parent["rung"] and row["status"] == "verified"]
                        self.assertGreaterEqual(len(eligible), 3)
                        eligible.sort(key=lambda row: row["metric"]["value"],
                                      reverse=get_metric(metric).higher_is_better)
                        self.assertIn(parent["id"], {row["id"] for row in eligible[:math.ceil(len(eligible) / 3)]})
                    # All results are measured, while admission is deliberately
                    # false: promotion must use quality rather than accepted.
                    score = (len(before_by_id) % 11) / 11
                    search.observe(recipe, measured(metric, score, accepted=False))
                    observed = search.report()["trials"][-1]
                    before_by_id[observed["id"]] = observed
                self.assertGreater(promotion_count, 0)

    def test_report_is_json_finite_and_deeply_detached(self):
        search = AdaptiveSearch(False, "RMSE")
        recipe = search.suggest()
        search.observe(recipe, measured(value=0.0))
        original = search.report()
        external = search.report()
        external["trials"][0]["recipe"]["feature_config"]["missing_indicators"] = True
        external["family_failures"]["linear"] = 100
        external["optimizer"]["backend"] = "mutated"
        self.assertEqual(search.report(), original)
        self.assertEqual(json.loads(json.dumps(original, allow_nan=False)), original)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optional Optuna not installed")
    def test_real_optuna_sequence_is_deterministic_without_training(self):
        def run():
            search = AdaptiveSearch(False, "RMSE", seed=716, max_trials=40)
            recipes = []
            while (recipe := search.suggest()) is not None:
                recipes.append(recipe)
                search.observe(recipe, measured(value=float((len(recipes) * 7) % 13)))
            return recipes, search.report()

        first, report = run()
        second, other = run()
        self.assertEqual(first, second)
        self.assertEqual(report, other)
        self.assertTrue(report["optimizer"]["available"])
        self.assertIn("tpe", report["optimizer"]["backend"])
        # At least one family reaches sampling after its three startup
        # observations; this exercises more than TPE's random cold start.
        adaptive_counts = Counter(row["family"] for row in report["trials"]
                                  if row["origin"] == "tpe")
        self.assertGreaterEqual(max(adaptive_counts.values()), 3)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optional Optuna not installed")
    def test_real_tpe_proposals_respond_to_verified_feedback_after_cold_start(self):
        def run(target):
            search = AdaptiveSearch(False, "RMSE", seed=716, max_trials=24)
            alphas = []
            while (recipe := search.suggest()) is not None:
                if recipe["action"] != "linear":
                    search.observe(recipe, {"verified": False, "accepted": False, "reason": "family_blocked"})
                    continue
                alpha = recipe["model_config"]["parameters"]["alpha"]
                alphas.append(alpha)
                # A deterministic proxy objective tests the optimizer, without
                # fitting a model or claiming predictive improvement.
                search.observe(recipe, measured(value=(math.log(alpha) - target) ** 2))
            return alphas

        positive, negative = run(3.0), run(-3.0)
        self.assertEqual(positive[:3], negative[:3])
        self.assertGreater(len(positive), 6)
        self.assertNotEqual(positive[3:], negative[3:])

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optional Optuna not installed")
    def test_tell_failure_keeps_observation_and_continues_without_replay(self):
        import optuna

        search = AdaptiveSearch(False, "RMSE", seed=716, max_trials=12)
        identities = set()
        for _ in range(5):
            recipe = search.suggest()
            identities.add(json.dumps(executable(recipe), sort_keys=True))
            search.observe(recipe, measured())
        issued = search.suggest()
        self.assertEqual(search.report()["trials"][-1]["origin"], "tpe")
        identities.add(json.dumps(executable(issued), sort_keys=True))
        secret = "secret-in-observation-tell"
        with patch.object(optuna.study.Study, "tell", side_effect=RuntimeError(secret)) as tell:
            self.assertTrue(search.observe(issued, measured(value=0.0, accepted=True)))
        self.assertEqual(tell.call_count, 1)
        report = search.report()
        self.assertEqual(report["observed"], 6)
        self.assertEqual(report["verified"], 6)
        self.assertIsNone(report["pending"])
        self.assertEqual(report["trials"][-1]["status"], "verified")
        self.assertEqual(report["trials"][-1]["metric"]["value"], 0.0)
        self.assertEqual(report["optimizer"]["backend"], "deterministic_random")
        self.assertFalse(report["optimizer"]["available"])
        self.assertEqual(report["optimizer"]["error_type"], "RuntimeError")
        self.assertNotIn(secret, json.dumps(report))
        with self.assertRaises(ValueError):
            search.observe(issued, measured(value=0.0, accepted=True))
        self.assertEqual(search.report(), report)
        while (recipe := search.suggest()) is not None:
            identity = json.dumps(executable(recipe), sort_keys=True)
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertNotEqual(search.report()["trials"][-1]["origin"], "tpe")
            search.observe(recipe, measured())
        self.assertEqual(search.report()["observed"], 12)


if __name__ == "__main__":
    unittest.main()
