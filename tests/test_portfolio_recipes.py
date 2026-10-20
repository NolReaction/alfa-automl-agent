"""Contracts and execution probes for the opt-in experimental recipes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agent.features import normalize_feature_config
from agent.metrics import get_metric, registered_metrics, score_predictions
from agent.model_config import ACTIONS, validate_model_config
from agent.models import build_pipeline, predict_for_metric
from agent.portfolio_recipes import portfolio_candidates
from experiment_portfolio_recipes import run_smoke, score_saved, smoke_data


class RecipeContractTests(unittest.TestCase):
    def test_all_metrics_return_eight_unique_valid_json_recipes(self):
        for metric in registered_metrics():
            with self.subTest(metric=metric.name):
                result = portfolio_candidates({"task_type": metric.task_type, "metric": metric.name},
                                              available_actions=ACTIONS)
                self.assertEqual(len(result), 8)
                self.assertEqual(len({item["id"] for item in result}), 8)
                identities = set()
                for item in result:
                    self.assertEqual(set(item), {"id", "action", "label", "model_config", "feature_config"})
                    self.assertTrue(item["id"].startswith("portfolio_v1_"))
                    self.assertEqual(validate_model_config(item["model_config"], item["action"]), item["model_config"])
                    self.assertEqual(normalize_feature_config(item["feature_config"]), item["feature_config"])
                    identities.add(json.dumps([item["model_config"], item["feature_config"]], sort_keys=True))
                self.assertEqual(len(identities), 8)
                self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_filtering_precedes_limit_and_preserves_stable_order(self):
        profile = {"task_type": "regression", "metric": "RMSE"}
        all_items = portfolio_candidates(profile, available_actions=ACTIONS)
        filtered = [item for item in all_items if item["action"] == "lightgbm"]
        self.assertEqual(portfolio_candidates(profile, available_actions=("lightgbm",), limit=2), filtered[:2])
        self.assertEqual(portfolio_candidates(profile, available_actions=()), [])
        with patch.dict("sys.modules", {"catboost": None, "lightgbm": None}):
            self.assertEqual(len(portfolio_candidates(profile, available_actions=("baseline", "extra_trees", "linear"))), 3)

    def test_nested_results_and_inputs_are_independent(self):
        profile = {"task_type": "classification", "metric": "Macro F1"}
        original = deepcopy(profile)
        expected = portfolio_candidates(profile, available_actions=ACTIONS)
        result = portfolio_candidates(profile, available_actions=ACTIONS)
        result[0]["model_config"]["parameters"]["C"] = 999
        result[0]["feature_config"]["date_columns"].append("injected")
        self.assertEqual(portfolio_candidates(profile, available_actions=ACTIONS), expected)
        self.assertEqual(profile, original)
        self.assertEqual(result[1]["feature_config"]["date_columns"], [])

    def test_invalid_profiles_are_rejected_even_without_available_models(self):
        valid = {"task_type": "regression", "metric": "RMSE"}
        for profile in (None, [], {}, {**valid, "answers": [1]}, {"task_type": True, "metric": "RMSE"},
                        {"task_type": "other", "metric": "RMSE"}, {"task_type": "classification", "metric": "RMSE"},
                        {"task_type": "regression", "metric": "unknown"},
                        {"task_type": "regression", "metric": {"name": "RMSE"}}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=())

    def test_runtime_controls_and_malformed_arguments_are_rejected(self):
        profile = {"task_type": "regression", "metric": "RMSE"}
        for limit in (True, False, 0, 9, 1.0, "1", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=(), limit=limit)
        for actions in (None, [], "linear", ("linear", "linear"), ("unknown",), (True,), ([],)):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                portfolio_candidates(profile, available_actions=actions)


class RecipeExecutionTests(unittest.TestCase):
    def test_real_recipes_fit_regression_binary_and_multiclass_mixed_inputs(self):
        for task_type, binary in (("regression", False), ("classification", True), ("classification", False)):
            metric = "RMSE" if task_type == "regression" else "LogLoss"
            fit, y_fit, held, actual = smoke_data(task_type, binary)
            before = fit.copy(deep=True)
            # Actual recipes, including the representation variant; small fixed fixture.
            for candidate in portfolio_candidates({"task_type": task_type, "metric": metric}, available_actions=ACTIONS):
                with self.subTest(task_type=task_type, binary=binary, candidate=candidate["id"]):
                    model = build_pipeline(candidate["action"], task_type, ["number", "empty"], ["category"],
                                           candidate["model_config"], candidate["feature_config"])
                    with threadpool_limits(limits=8):
                        model.fit(fit, y_fit)
                        prediction = predict_for_metric(model, held, metric)
                    classes = tuple(map(str, model.classes_)) if task_type == "classification" else ()
                    value = score_predictions(metric, actual, prediction, classes=classes, prediction_classes=classes)
                    self.assertTrue(np.isfinite(value))
                    self.assertEqual(len(prediction), len(held))
                    pd.testing.assert_frame_equal(fit, before)

    def test_linear_preprocessing_learns_category_vocabulary_from_fit_only(self):
        fit, y, held, _ = smoke_data("classification", True)
        candidate = portfolio_candidates({"task_type": "classification", "metric": "LogLoss"},
                                         available_actions=("linear",))[0]
        model = build_pipeline("linear", "classification", ["number", "empty"], ["category"],
                               candidate["model_config"], candidate["feature_config"])
        with threadpool_limits(limits=8):
            model.fit(fit, y)
        categories = model.named_steps["prepare"].named_transformers_["category"].named_steps["encode"].categories_[0]
        self.assertNotIn("holdout_only", categories)
        self.assertEqual(predict_for_metric(model, held, "LogLoss").shape, (32, 2))

    def test_subprocess_smoke_rescores_saved_predictions_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            profile = {"task_type": "classification", "metric": "ROC AUC"}
            report = run_smoke(profile, output, available_actions=("linear",), binary=True,
                               attempt_seconds=30, total_seconds=40)
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(report["results"][0]["status"], "completed", report)
            item = report["results"][0]
            self.assertEqual(score_saved(output / item["id"], "classification", "ROC AUC", True), item["score"])
            with self.assertRaises(FileExistsError):
                run_smoke(profile, output, available_actions=("linear",), binary=True)
            prediction_path = output / item["id"] / "predictions.csv"
            table = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
            table.loc[0, "id"] = "wrong"
            table.to_csv(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "IDs"):
                score_saved(prediction_path.parent, "classification", "ROC AUC", True)

    def test_invalid_smoke_limits_are_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for bad in (True, 0, -1, float("nan"), float("inf"), 10**1000):
                with self.subTest(limit=bad), self.assertRaises(ValueError):
                    run_smoke({"task_type": "regression", "metric": "RMSE"}, output, attempt_seconds=bad)
            self.assertFalse(output.exists())


class ComparisonProtocolTests(unittest.TestCase):
    @staticmethod
    def dataset(classification=True):
        import experiment_portfolio_recipes as runner

        dataset = runner.comparison_dataset(builtin="wine" if classification else "diabetes")
        if classification:
            # Each pair has the same raw features but conflicting labels. Neither
            # targets nor unique IDs may break that duplicate-feature group.
            dataset.update(X=pd.DataFrame({"number": np.repeat(np.arange(60), 2),
                                           "kind": ["same"] * 120}),
                           y=np.asarray(["001", "NA"] * 60),
                           ids=[f"{i:05d}" for i in range(120)], features=["number", "kind"],
                           numeric=["number"], categorical=["kind"], classes=("001", "NA"),
                           positive_class="001")
        else:
            dataset.update(X=pd.DataFrame({"Year_Factor": np.repeat([1, 2, 3, 4, 5], 12),
                                           "number": np.arange(60)}),
                           y=np.arange(60, dtype=float), ids=[f"{i:05d}" for i in range(60)],
                           features=["Year_Factor", "number"], numeric=["Year_Factor", "number"],
                           categorical=[], provenance={"task": "synthetic temporal fixture"},
                           time_column="Year_Factor")
        return dataset

    def test_nested_group_splits_are_frozen_and_keep_conflicting_duplicates_together(self):
        from experiment_portfolio_recipes import freeze_splits

        dataset = self.dataset()
        first = freeze_splits(dataset)
        self.assertEqual(first, freeze_splits(dataset))
        self.assertEqual(len(first), 2)
        groups = pd.util.hash_pandas_object(dataset["X"], index=False).to_numpy()
        for split in first:
            self.assertEqual(split["status"], "ready")
            self.assertEqual(split["inner_status"], "ready")
            fit, held = set(split["outer_fit"]), set(split["outer_held"])
            inner_fit, inner_held = set(split["inner_fit"]), set(split["inner_held"])
            self.assertEqual(fit | held, set(range(120)))
            self.assertEqual(inner_fit | inner_held, fit)
            self.assertFalse(fit & held or inner_fit & inner_held)
            for a, b in ((fit, held), (inner_fit, inner_held)):
                self.assertFalse(set(groups[list(a)]) & set(groups[list(b)]))
                self.assertEqual(set(dataset["y"][list(a)]), {"001", "NA"})
                self.assertEqual(set(dataset["y"][list(b)]), {"001", "NA"})

    def test_temporal_inner_and_outer_use_only_strictly_past_periods(self):
        from experiment_portfolio_recipes import freeze_splits

        dataset = self.dataset(False)
        splits = freeze_splits(dataset)
        self.assertEqual([item["year"] for item in splits], [3, 4, 5])
        periods = dataset["X"]["Year_Factor"].to_numpy()
        for split in splits:
            self.assertEqual(split["status"], "ready")
            self.assertEqual(split["inner_status"], "ready")
            self.assertLess(periods[split["inner_fit"]].max(), periods[split["inner_held"]].min())
            self.assertLess(periods[split["outer_fit"]].max(), periods[split["outer_held"]].min())
            self.assertEqual(set(split["inner_fit"]) | set(split["inner_held"]), set(split["outer_fit"]))
        self.assertEqual([len(item["inner_fit"]) for item in splits], [12, 24, 36])

    def test_impossible_group_fold_is_recorded_without_replacement(self):
        from experiment_portfolio_recipes import freeze_splits

        dataset = self.dataset()
        dataset["X"].loc[:, "number"] = 0
        splits = freeze_splits(dataset, seeds=(1729,), outer_folds=(0,))
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0]["seed"], 1729)
        self.assertEqual(splits[0]["fold"], 0)
        self.assertEqual(splits[0]["status"], "impossible")

    def test_manifest_is_written_without_fits_and_cannot_overwrite_an_existing_run(self):
        import experiment_portfolio_recipes as runner

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            with patch.object(runner, "_attempt", side_effect=AssertionError("Unexpected fit")):
                dataset, manifest = runner.prepare_comparison(None, output, builtin="wine", seeds=(1729,))
            self.assertEqual(json.loads((output / "manifest.json").read_text()), json.loads(json.dumps(manifest)))
            self.assertEqual(set(manifest["portfolios"]), {"exploration_plan", "portfolio_v1"})
            self.assertTrue(manifest["source_hashes"])
            self.assertEqual(len(manifest["diagnostic_ablations"]), 4)
            self.assertEqual(len(dataset["X"]), 178)
            with self.assertRaises(FileExistsError):
                runner.prepare_comparison(None, output, builtin="wine", seeds=(1729,))

    def test_choices_freeze_from_inner_scores_before_outer_or_diagnostic_calls(self):
        import experiment_portfolio_recipes as runner

        calls = []

        def attempt(candidate, bundle, directory, dataset, *, timeout, manifest, split, phase):
            calls.append((phase, candidate["id"], timeout))
            if phase != "inner_search":
                frozen_path = Path(directory)
                while not (frozen_path / "frozen_winners.json").exists():
                    if frozen_path == frozen_path.parent:
                        self.fail("Outer work started before winner freeze")
                    frozen_path = frozen_path.parent
            # A diagnostic can look perfect without changing the frozen choice.
            value = 0.95 if candidate["id"] in {"candidate_8", "portfolio_v1_linear_regularized"} else 0.1
            if phase == "postfreeze_inner_diagnostic":
                value = 1.0
            return {"id": candidate["id"], "candidate": candidate, "status": "completed", "score": value,
                    "phase": phase, "directory": str(directory)}

        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "_attempt", side_effect=attempt):
            report = runner.run_comparison(None, Path(temporary) / "experiment", builtin="wine", seeds=(1729,))
            selected = report["splits"][0]["portfolios"]
            self.assertEqual(selected["exploration_plan"]["selected"], "candidate_8")
            self.assertEqual(selected["portfolio_v1"]["selected"], "portfolio_v1_linear_regularized")
            self.assertTrue(all(phase == "inner_search" for phase, _, _ in calls[:16]))
            self.assertEqual(sum(phase == "selected_outer_refit" for phase, _, _ in calls), 2)
            self.assertEqual(sum(phase == "postfreeze_inner_diagnostic" for phase, _, _ in calls), 4)
            self.assertTrue(report["inputs_unchanged"] and report["sources_unchanged"])

    def test_comparison_saved_predictions_preserve_class_order_and_reject_tampering(self):
        from experiment_portfolio_recipes import score_trial

        dataset = self.dataset()
        held = [0, 1, 2, 3]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Reverse native probability columns deliberately; class labels stay literal.
            (directory / "classes.json").write_text(json.dumps(["NA", "001"]))
            table = pd.DataFrame({"id": [dataset["ids"][i] for i in held], "position": held,
                                  "actual": dataset["y"][held], "label": dataset["y"][held],
                                  "probability_0": [0.1, 0.9, 0.1, 0.9],
                                  "probability_1": [0.9, 0.1, 0.9, 0.1]})
            table.to_csv(directory / "predictions.csv", index=False)
            dataset["profile"] = {"task_type": "classification", "metric": "LogLoss"}
            self.assertAlmostEqual(score_trial(directory, dataset, held), -np.log(0.9))
            table.loc[0, "probability_0"] = float("nan")
            table.to_csv(directory / "predictions.csv", index=False)
            with self.assertRaises(ValueError):
                score_trial(directory, dataset, held)

    def test_score_finishing_after_attempt_deadline_is_not_eligible(self):
        import experiment_portfolio_recipes as runner

        dataset = self.dataset()
        candidate = portfolio_candidates(dataset["profile"], available_actions=("linear",))[0]
        manifest = {"settings": {"available_actions": ("linear",)}, "source_hashes": {},
                    "resource_policy": {"max_memory_bytes": 1024**3}}
        split = {"seed": 1729, "split_hash": "frozen"}
        clock = [0.0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            bundle = runner._bundle(dataset, list(range(4, 100)), [0, 1, 2, 3], Path(temporary) / "bundle")

            def execution(*args, **kwargs):
                pd.DataFrame({"id": bundle["held_ids"], "position": bundle["held_indices"]}).to_csv(
                    directory / "raw_predictions.csv", index=False)
                return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}

            def late_score(*args):
                clock[0] = 1.1
                return 0.99

            with patch("agent.execution.run_python", side_effect=execution), \
                    patch.object(runner.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(runner, "score_trial", side_effect=late_score):
                result = runner._attempt(candidate, bundle, directory, dataset, timeout=1.0,
                                         manifest=manifest, split=split, phase="inner_search")
            self.assertEqual(result["status"], "over_budget")
            self.assertEqual(result["score"], 0.99)

    def test_worker_bundle_contains_only_fit_targets(self):
        import experiment_portfolio_recipes as runner

        dataset = self.dataset(False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            payload = runner._bundle(dataset, [0, 1, 2], [3, 4], directory)
            np.testing.assert_array_equal(np.load(payload["fit_target"], allow_pickle=False), dataset["y"][:3])
            self.assertEqual(len(pd.read_pickle(payload["held_features"])), 2)
            self.assertNotIn("actual", payload)
            self.assertNotIn("held_target", payload)
            self.assertEqual(set(path.name for path in directory.iterdir()),
                             {"input.json", "fit_target.npy", "fit_features.pkl", "held_features.pkl"})

    def test_invalid_comparison_budgets_fail_before_creating_run_directory(self):
        from experiment_portfolio_recipes import prepare_comparison

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unused"
            for value in (True, None, 0, -1, float("nan"), float("inf"), 10**1000):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prepare_comparison(None, output, builtin="wine", search_seconds=value)
            with self.assertRaises(ValueError):
                prepare_comparison(None, output, builtin="wine", memory_gib=1000)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
