"""Adaptive policy integration with genuine host evidence and mocked workers."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import solution
from agent import automl, automl_worker
from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.task_spec import load_task
from tests import test_automl as fixtures
from validation import validate_predictions


def candidate(identifier, action, parameters=None):
    return {"id": identifier, "label": identifier, "action": action,
            "model_config": make_model_config(action, "regression", parameters),
            "feature_config": normalize_feature_config()}


class AdaptiveAutoMLIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids rediscovering or rerunning the fixture's tests.
        self.fixture = fixtures.AutoMLTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_adaptive(self, outcomes=None, *, pool=None, refine=None, callback=None, progress=None):
        options = {"search_policy": "adaptive_v1"}
        if progress is not None:
            options["progress"] = progress
        with ExitStack() as stack:
            if pool is not None:
                stack.enter_context(patch.object(automl, "exploration_candidates", return_value=pool))
            if refine is not None:
                stack.enter_context(patch.object(automl, "refinement_candidates", side_effect=refine))
            return self.fixture.run_search(outcomes, full_plan=True, callback=callback, **options)

    def test_seed_published_before_selection_and_outer_score_never_ranks_refinements(self):
        pool = [candidate("linear", "linear"), candidate("baseline", "baseline"),
                candidate("trees", "extra_trees"), candidate("cat", "catboost"),
                candidate("light", "lightgbm"), candidate("trees_more", "extra_trees", {"n_estimators": 200})]
        outcomes = {"seed": {"error": 0}, "selection": {"error": 9}}
        inner = {"linear": 8, "baseline": 7, "extra_trees": 6, "catboost": 1, "lightgbm": 2}
        observed_rankings = []

        def callback(config):
            if config["phase"] == "selection":
                self.assertTrue(validate_predictions(self.fixture.root)["valid"])
                outcomes[(config["id"], "selection")] = {"error": inner[config["action"]]}

        def refine(classification, metric, ranked, **kwargs):
            observed_rankings.append(deepcopy(ranked))
            return []

        result = self.run_adaptive(outcomes, pool=pool, refine=refine, callback=callback)
        self.assertEqual(result["search_policy"]["name"], "adaptive_v1")
        self.assertEqual(result["search"]["best_candidate"], "seed")
        self.assertEqual(len(observed_rankings), 1)
        self.assertEqual([item["action"] for item in observed_rankings[0]][:2], ["catboost", "lightgbm"])
        self.assertNotIn("seed", [item["id"] for item in observed_rankings[0]])
        seed = next(call for call in self.fixture.calls if call["phase"] == "seed")
        for call in self.fixture.calls:
            if call["phase"] == "selection":
                self.assertFalse(set(call["fit_rows"]) & set(call["validation_rows"]))
                self.assertTrue(set(call["fit_rows"]) | set(call["validation_rows"]) <= set(seed["fit_rows"]))
                self.assertFalse(set(call["validation_rows"]) & set(seed["validation_rows"]))
        observations = result["search_policy"]["schedule"]["observations"]
        self.assertEqual(len(observations), len(pool))
        self.assertTrue(all(item["elapsed_seconds"] > 0 for item in observations))
        self.assertEqual(result["api"], {"used": False, "calls": 0})

    def test_real_policy_filters_missing_optional_packages_and_keeps_seed(self):
        with patch.object(automl.importlib.util, "find_spec", return_value=None):
            result = self.run_adaptive({"seed": {"error": 0}})
        selected = [item for item in self.fixture.calls if item["phase"] == "selection"]
        self.assertTrue(selected)
        self.assertEqual({item["action"] for item in selected}, {"linear", "baseline", "extra_trees"})
        self.assertLessEqual(len(selected), 12)
        self.assertTrue(all(item["timeout"] >= 5 for item in selected))
        self.assertTrue(validate_predictions(self.fixture.root)["valid"])
        self.assertEqual(result["search"]["best_candidate"], "seed")

    def test_inner_winner_is_rechecked_on_outer_and_only_improving_winner_refits(self):
        pool = [candidate("tree", "extra_trees"), candidate("boost", "baseline")]
        outcomes = {"seed": {"error": 5}, ("tree", "selection"): {"error": 0},
                    ("boost", "selection"): {"error": 1}, ("tree", "verification"): {"error": 4},
                    ("boost", "verification"): {"error": 1}}
        result = self.run_adaptive(outcomes, pool=pool, refine=lambda *args, **kwargs: [])
        self.assertEqual(result["search"]["best_candidate"], "boost")
        self.assertEqual(result["metric"]["value"], 1)
        refits = [call for call in self.fixture.calls if call["phase"] == "refit"]
        self.assertEqual([item["id"] for item in refits], ["boost"])
        self.assertTrue(validate_predictions(self.fixture.root)["valid"])

    def test_full_refit_worker_uses_every_training_row_and_original_test(self):
        folder = self.fixture.root / "refit-worker"
        folder.mkdir()
        config = {**candidate("full", "linear"), "candidate_dir": str(folder), "phase": "refit",
                  "task_spec": load_task(self.fixture.task).to_dict(),
                  "fit_rows": [0], "validation_rows": [1]}
        path = folder / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        seen = {}

        class Pipeline:
            def fit(self, x, y):
                seen["fit"] = (x.copy(), y.copy())
                return self

            def predict(self, x):
                seen["predict"] = x.copy()
                return np.arange(len(x), dtype=float)

        with patch.object(automl_worker, "build_pipeline", return_value=Pipeline()), redirect_stdout(io.StringIO()):
            automl_worker.run_candidate(path)
        self.assertEqual(len(seen["fit"][0]), 80)
        self.assertEqual(len(seen["fit"][1]), 80)
        self.assertEqual(seen["fit"][0]["x"].astype(float).tolist(), list(range(80)))
        self.assertEqual(seen["predict"]["x"].astype(float).tolist(), [5, 9])
        final = automl.read_csv(folder / "submission.csv")
        self.assertEqual(final["id"].tolist(), ["0001", "NA"])

    def test_post_fit_host_callback_is_charged_to_selection_deadline(self):
        clock, advanced = [0.0], [False]
        pool = [candidate("late", "baseline"), candidate("unused", "extra_trees")]

        def progress(event):
            if (not advanced[0] and event["search"]["phase"] == "selection"
                    and any(item["selection_metric"] is not None for item in event["search"]["candidates"])):
                clock[0] = 73.0  # The 120-second run's selection deadline is 72.
                advanced[0] = True

        with patch.object(automl.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.run_adaptive({"seed": {"error": 5}, "selection": {"error": 0}},
                                       pool=pool, refine=lambda *args, **kwargs: [], progress=progress)
        self.assertTrue(advanced[0])
        row = next(item for item in result["search"]["candidates"] if item["id"] == "late")
        self.assertIsNone(row["selection_metric"])
        self.assertEqual(row["rejection_reason"], "insufficient_budget")
        self.assertEqual(result["search_policy"]["schedule"]["observations"][0]["elapsed_seconds"], 73)
        self.assertEqual(result["search"]["best_candidate"], "seed")
        self.assertFalse(any(item["phase"] == "verification" for item in self.fixture.calls))

    def test_failed_optional_family_is_removed_without_repeating_import_failures(self):
        pool = [candidate("missing", "catboost"), candidate("missing_more", "catboost", {"iterations": 200}),
                candidate("works", "baseline")]
        execute = self.fixture.fake_execution({"seed": {"error": 0}})

        def fail_optional(code, task_dir, run_dir, step, timeout):
            result = execute(code, task_dir, run_dir, step, timeout)
            config = self.fixture.calls[-1]
            if config["phase"] == "selection" and config["action"] == "catboost":
                result.update(exit_code=1, stderr="ModuleNotFoundError: No module named catboost")
            return result

        with patch.object(automl, "run_python", side_effect=fail_optional), \
                patch.object(automl, "exploration_candidates", return_value=pool), \
                patch.object(automl, "refinement_candidates", return_value=[]), redirect_stdout(io.StringIO()):
            result = automl.run_automl(self.fixture.task, time_budget=120, search_policy="adaptive_v1")
        attempts = [item for item in self.fixture.calls if item["phase"] == "selection"]
        self.assertEqual([item["id"] for item in attempts], ["missing", "works"])
        skipped = next(item for item in result["search"]["candidates"] if item["id"] == "missing_more")
        self.assertEqual(skipped["rejection_reason"], "optional_dependency_unavailable")
        self.assertTrue(result["output_valid"])

    def test_unknown_policy_is_rejected_before_any_run_directory(self):
        with self.assertRaisesRegex(ValueError, "search_policy"):
            automl.run_automl(self.fixture.task, search_policy="unknown")
        self.assertFalse((self.fixture.root / ".agent_runs").exists())

    def test_legacy_default_has_same_behavior_as_explicit_legacy(self):
        with patch.object(automl.time, "monotonic", return_value=0):
            implicit = self.fixture.run_search({"seed": {"error": 0}})
            first = [(item["id"], item["phase"], item["model_config"]) for item in self.fixture.calls]
            self.fixture.calls.clear()
            explicit = self.fixture.run_search({"seed": {"error": 0}}, search_policy="legacy")
            second = [(item["id"], item["phase"], item["model_config"]) for item in self.fixture.calls]
        self.assertEqual(first, second)
        self.assertEqual(implicit["search"], explicit["search"])
        self.assertEqual(implicit["metric"], explicit["metric"])
        self.assertNotIn("schedule", explicit["search_policy"])

    def test_cli_forwards_only_explicit_nonlegacy_policy(self):
        result = {"model": "Ridge", "metric": {"name": "RMSE", "value": 1}, "report_path": "report.json"}
        for option, expected in (([], None), (["--search-policy", "legacy"], None),
                                 (["--search-policy", "adaptive_v1"], "adaptive_v1")):
            with self.subTest(option=option), patch.dict(os.environ, {}, clear=True), \
                    patch.object(solution, "_prepare_emergency", return_value=None), \
                    patch.object(solution, "make_client", side_effect=AssertionError("Unexpected API")), \
                    patch("agent.automl.run_automl", return_value=result) as run, \
                    patch("sys.argv", ["solution.py", str(self.fixture.task), "--mode", "offline", *option]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(solution.main(), 0)
            options = run.call_args.kwargs
            if expected is None:
                self.assertNotIn("search_policy", options)
            else:
                self.assertEqual(options["search_policy"], expected)
