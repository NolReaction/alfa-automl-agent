"""The expanded bounded plan remains valid across application report boundaries."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from app.model_config import make_model_config
from app.search_summary import validate_search_summary, MAX_SEARCH_CANDIDATES, normalize_search_strategy


def completed_search(count):
    rows = [{"id": f"candidate-{index}", "label": f"Recipe {index}", "model": "Ridge",
             "model_config": make_model_config("linear", "regression", {"alpha": index + 1}),
             "status": "completed", "metric": float(index), "selected": index == 0,
             "elapsed_seconds": 1.0, "error": None} for index in range(count)]
    return {"status": "completed", "planned": count, "completed": count,
            "best_candidate": "candidate-0", "validation_reason": "Fixed holdout", "candidates": rows}


class SearchSummaryBoundTests(unittest.TestCase):
    def test_full_twenty_two_recipe_plan_is_accepted_without_truncation(self):
        summary = completed_search(22)
        validated = validate_search_summary(summary, metric={"name": "MAE", "value": 0.0,
                                            "higher_is_better": False}, model="Ridge")
        self.assertEqual(validated, summary)
        self.assertEqual(len(validated["candidates"]), 22)
        self.assertEqual(validated["candidates"][-1]["id"], "candidate-21")

    def test_extended_adaptive_plan_is_accepted_without_truncation(self):
        summary = completed_search(258)
        summary["controller"] = {"strategy": "adaptive", "max_trials": 256, "proposed": 256,
            "observed": 256, "stop_reason": "trial_limit", "optimizer": "optuna_tpe", "resumable": False}
        validated = validate_search_summary(summary)
        self.assertEqual(len(validated["candidates"]), 258)
        self.assertEqual(validated["controller"], summary["controller"])

    def test_plan_beyond_safety_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "число попыток"):
            validate_search_summary(completed_search(MAX_SEARCH_CANDIDATES + 1))

    def test_controller_cannot_claim_resume_or_more_observations_than_proposals(self):
        controller = {"strategy": "adaptive", "max_trials": 256, "proposed": 2,
            "observed": 2, "stop_reason": "selection_budget", "optimizer": "optuna_tpe", "resumable": False}
        for change in ({"resumable": True}, {"observed": 3}, {"max_trials": 513},
                       {"optimizer": "fake"}, {"stop_reason": "mystery"}, {"proposed": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_search_summary({**completed_search(2), "controller": {**controller, **change}})

    def test_strategy_is_explicit_and_only_applies_to_automl(self):
        self.assertEqual(normalize_search_strategy(), "adaptive")
        self.assertEqual(normalize_search_strategy("legacy"), "legacy")
        self.assertIsNone(normalize_search_strategy(None, action="catboost"))
        for value, action in (("other", "automl"), ({}, "automl"), ("adaptive", "catboost")):
            with self.subTest(value=value, action=action), self.assertRaises(ValueError):
                normalize_search_strategy(value, action=action)

    def test_long_adaptive_report_survives_export_import_and_retains_strategy(self):
        import json
        from agent.features import normalize_feature_config
        from app.sharing import ReportStore
        from app.tests.test_sharing import FINGERPRINT, job

        summary = completed_search(514)
        summary["controller"] = {"strategy": "adaptive", "max_trials": 512, "proposed": 512,
            "observed": 512, "stop_reason": "trial_limit", "optimizer": "optuna_tpe", "resumable": False}
        for row in summary["candidates"]:
            row["feature_config"] = normalize_feature_config()
        original = job(action="automl", model="Ridge", search_strategy="adaptive",
            model_config=summary["candidates"][0]["model_config"], search=summary,
            feature_config=normalize_feature_config(),
            metric={"name": "RMSE", "value": 0.0, "higher_is_better": False})
        with tempfile.TemporaryDirectory() as directory:
            store = ReportStore(Path(directory) / ".workspace")
            envelope = store.export("regression", FINGERPRINT, [original])
            self.assertGreater(len(json.dumps(envelope).encode()), 128 * 1024)
            self.assertEqual(store.import_reports("regression", FINGERPRINT, envelope), {"imported": 1, "skipped": 0})
            imported, = store.list("regression")
            self.assertEqual(len(imported["search"]["candidates"]), 514)
            self.assertEqual(imported["search_strategy"], "adaptive")

    def test_weighted_blend_requires_matching_component_feature_count(self):
        from agent.features import normalize_feature_config
        from app.model_config import feature_evidence

        config = {"model": "WeightedBlend", "parameters": {"weights": [.2, .3, .5], "components": [
            make_model_config("linear", "regression", {"alpha": value}) for value in (1, 2, 3)]}}
        evidence = {"model_config": config, "component_feature_configs": [normalize_feature_config()] * 3}
        self.assertEqual(feature_evidence(evidence, model="WeightedBlend"),
                         {"component_feature_configs": evidence["component_feature_configs"]})
        with self.assertRaisesRegex(ValueError, "соответствовать"):
            feature_evidence({**evidence, "component_feature_configs": evidence["component_feature_configs"][:2]}, model="WeightedBlend")


class RecoveredSearchSummaryTests(unittest.TestCase):
    def fixture(self):
        summary = completed_search(2)
        summary.update(status="fallback", phase="finished", planned=3)
        summary["candidates"].append({
            "id": "interrupted", "label": "Interrupted recipe", "model": "Ridge",
            "status": "failed", "metric": None, "selected": False,
            "elapsed_seconds": .5, "error": "Search interrupted after an accepted result",
        })
        metric = {"name": "MAE", "value": 0.0, "higher_is_better": False}
        return summary, metric

    def test_fallback_preserves_the_verified_winner_metric_and_model(self):
        summary, metric = self.fixture()
        validated = validate_search_summary(summary, metric=metric, model="Ridge")
        self.assertEqual(validated, summary)
        winner, = [row for row in validated["candidates"] if row["selected"]]
        self.assertEqual(winner["id"], validated["best_candidate"])
        self.assertEqual(winner["metric"], metric["value"])
        self.assertEqual(winner["model"], "Ridge")
        self.assertEqual(validated["candidates"][-1]["status"], "failed")

    def test_fallback_without_an_accepted_winner_is_rejected_even_as_progress(self):
        summary, metric = self.fixture()
        summary["best_candidate"] = None
        for row in summary["candidates"]:
            row["selected"] = False
        for allow_running in (False, True):
            with self.subTest(allow_running=allow_running):
                with self.assertRaisesRegex(ValueError, "Не указан победитель"):
                    validate_search_summary(summary, metric=metric, model="Ridge",
                                            allow_running=allow_running)

    def test_fallback_does_not_accept_selection_only_evidence_or_a_forged_best_metric(self):
        summary, metric = self.fixture()
        selection_only = deepcopy(summary)
        selection_only["candidates"][0].update(metric=None, selection_metric=0.0)
        with self.assertRaisesRegex(ValueError, "внешняя проверка"):
            validate_search_summary(selection_only, metric=metric, model="Ridge")
        with self.assertRaisesRegex(ValueError, "Метрика победителя"):
            validate_search_summary(summary, metric={**metric, "value": -1.0}, model="Ridge")
        with self.assertRaisesRegex(ValueError, "Модель победителя"):
            validate_search_summary(summary, metric=metric, model="ExtraTreesRegressor")

    def test_recovered_search_survives_sharing_without_becoming_locally_verified(self):
        from app.sharing import ReportStore
        from app.tests.test_sharing import FINGERPRINT, job

        summary, metric = self.fixture()
        original = job(action="automl", model="Ridge", metric=metric,
                       model_config=summary["candidates"][0]["model_config"], search=summary)
        with tempfile.TemporaryDirectory() as directory:
            store = ReportStore(Path(directory) / ".workspace")
            envelope = store.export("regression", FINGERPRINT, [original])
            self.assertEqual(store.import_reports("regression", FINGERPRINT, envelope),
                             {"imported": 1, "skipped": 0})
            imported, = store.list("regression")
            self.assertEqual(imported["search"]["status"], "fallback")
            self.assertEqual(imported["search"]["best_candidate"], "candidate-0")
            self.assertEqual(imported["model"], "Ridge")
            self.assertEqual(imported["metric"], metric)
            self.assertFalse(imported["verified"])
            self.assertFalse(imported["published"])
            self.assertEqual(store.export("regression", FINGERPRINT, [imported])["reports"],
                             envelope["reports"])


if __name__ == "__main__":
    unittest.main()
