"""AutoGluon wrapper contracts without importing the optional backend."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import autogluon_candidate as candidate


CLASSES = ["01", "NA", "class c"]


def frame(count=90):
    return pd.DataFrame({"x": [str(index) for index in range(count)],
                         "kind": ["NA", "0007", "7"] * (count // 3)})


def context(*, classification=True, metric=None, policy=None, families=("RF",)):
    metric = metric or ("LogLoss" if classification else "RMSE")
    recipe = candidate.make_autogluon_recipe("classification" if classification else "regression", metric,
        validation=policy or {"strategy": "random"}, model_families=families)
    spec = candidate.get_metric(metric)
    return {"protocol_version": 3, "version": 3, "phase": "inner_selection",
            "recipe": recipe, "features": ["x", "kind"], "numeric": ["x"], "categorical": ["kind"],
            "classes": list(CLASSES) if classification else [], "positive_class": None,
            "metric": spec.to_dict(), "task_type": spec.task_type,
            "prediction_kind": spec.prediction_kind}


class FakeGenerator:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakePredictor:
    instances = []
    refit_result = None
    probability_columns = ["NA", "class c", "01"]

    def __init__(self, **kwargs):
        self.constructor = deepcopy(kwargs)
        self.model_best = "FitModel"
        self.fit_options = None
        self.refit_options = None
        self.predict_options = None
        self.__class__.instances.append(self)

    def fit(self, **kwargs):
        self.fit_options = deepcopy(kwargs)
        self.all_refit_data = pd.concat([kwargs["train_data"], kwargs["tuning_data"]]).sort_index()
        return self

    def refit_full(self, **kwargs):
        self.refit_options = deepcopy(kwargs)
        return deepcopy(self.refit_result) if self.refit_result is not None else {"FitModel": "FitModel_FULL"}

    def model_info(self, model):
        if model != "FitModel_FULL":
            raise AssertionError("Metadata must describe the actual full-data refit")
        return {"num_samples": len(self.all_refit_data)}

    def predict_proba(self, data, **kwargs):
        self.predict_options = deepcopy(kwargs)
        self.prediction_frame = data.copy(deep=True)
        return pd.DataFrame(np.tile([0.2, 0.3, 0.5], (len(data), 1)),
                            columns=self.probability_columns, index=data.index)

    def predict(self, data, **kwargs):
        self.predict_options = deepcopy(kwargs)
        self.prediction_frame = data.copy(deep=True)
        return pd.Series(np.arange(len(data), dtype=float), index=data.index)


class RecipeTests(unittest.TestCase):
    def test_recipe_is_detached_explicit_and_cpu_only(self):
        policy = {"strategy": "time", "time_column": "period"}
        models = ["CAT", "GBM"]
        recipe = candidate.make_recipe("regression", "RMSE", validation=policy, model_families=models)
        policy["time_column"] = "changed"
        models.append("RF")
        self.assertEqual(recipe["validation"]["time_column"], "period")
        self.assertEqual(recipe["model_families"], ["CAT", "GBM"])
        self.assertEqual(candidate._recipe(recipe), recipe)
        json.dumps(recipe, allow_nan=False)

    def test_validation_must_be_explicit_and_resolved(self):
        for policy in (None, {"strategy": "auto"}, {"strategy": "time"},
                       {"strategy": "random", "time_column": "x"}):
            with self.subTest(policy=policy), self.assertRaises(candidate.AutoGluonCandidateError):
                candidate.make_recipe("regression", "RMSE", validation=policy)

    def test_task_metric_and_supported_policy_must_agree(self):
        for task, metric, policy in (("regression", "Macro F1", {"strategy": "random"}),
                                    ("classification", "RMSE", {"strategy": "random"}),
                                    ("classification", "Macro F1", {"strategy": "time", "time_column": "x"}),
                                    ("regression", "RMSE", {"strategy": "grouped"})):
            with self.subTest(task=task, metric=metric), self.assertRaises(candidate.AutoGluonCandidateError):
                candidate.make_recipe(task, metric, validation=policy)

    def test_resource_and_model_allowlists_reject_bypass(self):
        for changes in ({"time_limit": True}, {"time_limit": float("inf")}, {"time_limit": 1801},
                        {"time_limit": 0}, {"num_cpus": 9}, {"num_cpus": True},
                        {"memory_limit": 25}, {"memory_limit": 0},
                        {"model_families": ["TABPFN"]}, {"model_families": ["RF", "RF"]},
                        {"model_families": []}):
            with self.subTest(changes=changes), self.assertRaises(candidate.AutoGluonCandidateError):
                candidate.make_recipe("regression", "RMSE", validation={"strategy": "random"}, **changes)

    def test_saved_recipe_rejects_unknown_fields_and_version_aliases(self):
        for changes in ({"version": True}, {"version": 2}, {"family": "unknown"}, {"num_gpus": 1}):
            recipe = context()["recipe"]
            recipe.update(changes)
            with self.subTest(changes=changes), self.assertRaises(candidate.AutoGluonCandidateError):
                candidate._recipe(recipe)

    def test_availability_distinguishes_missing_and_wrong_version_without_importing_backend(self):
        with patch.object(candidate.importlib.util, "find_spec", return_value=None):
            self.assertEqual(candidate.availability()["reason"], "package_unavailable")
        with patch.object(candidate.importlib.util, "find_spec", return_value=object()), \
             patch.object(candidate.importlib.metadata, "version", return_value="9.0"):
            self.assertEqual(candidate.availability()["reason"], "version_mismatch")
        with patch.object(candidate.importlib.util, "find_spec", return_value=object()), \
             patch.object(candidate.importlib.metadata, "version", return_value="1.3.1"), \
             patch.object(candidate, "_process_probe", return_value=True):
            self.assertTrue(candidate.availability()["available"])

    def test_missing_process_inventory_is_an_explicit_skip_not_fabricated_memory(self):
        with patch.object(candidate.importlib.util, "find_spec", return_value=object()), \
             patch.object(candidate.importlib.metadata, "version", return_value="1.3.1"), \
             patch.object(candidate, "_process_probe", return_value=False):
            result = candidate.availability()
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "process_inventory_unavailable")
        self.assertFalse(result["process_inventory_available"])

    def test_process_probe_uses_actual_pid_and_rejects_os_metadata_failure(self):
        import psutil
        with patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(321)) as process, \
             patch.object(candidate.os, "getpid", return_value=321), \
             patch("agent.resources._namespace_matches", return_value=True):
            self.assertFalse(candidate._process_probe())
        process.assert_called_once_with(321)

    def test_process_probe_rejects_namespace_mismatch_before_inspecting_colliding_host_pid(self):
        import psutil
        with patch.object(psutil, "Process") as process, \
             patch("agent.resources._namespace_matches", return_value=False):
            self.assertFalse(candidate._process_probe())
        process.assert_not_called()

    def test_generated_source_uses_the_wrapper_from_flat_compatible_package(self):
        namespace = {}
        exec(compile(candidate.candidate_source(), "candidate.py", "exec"), namespace)
        self.assertTrue(callable(namespace["fit_predict"]))
        self.assertNotIn("pip", candidate.candidate_source())


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        FakePredictor.instances = []
        FakePredictor.refit_result = None
        FakePredictor.probability_columns = ["NA", "class c", "01"]

    def invoke(self, fitting=None, labels=None, prediction=None, ctx=None, clock=None):
        fitting = frame() if fitting is None else fitting
        labels = pd.Series(CLASSES * 30) if labels is None else labels
        prediction = frame(6) if prediction is None else prediction
        ctx = context() if ctx is None else ctx
        with patch.object(candidate, "_load_backend", return_value=(FakePredictor, FakeGenerator)), \
             patch.object(candidate.time, "monotonic", side_effect=clock or (lambda: 100.0)), \
             patch("pathlib.Path.write_text", autospec=True, return_value=0) as writer:
            result = candidate.fit_predict(fitting, labels, prediction, ctx)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(writer.call_args.args[0].name, candidate.EVIDENCE_NAME)
        self.evidence = json.loads(writer.call_args.args[1])
        self.predictor = FakePredictor.instances[-1]
        return result

    def test_class_probabilities_align_exact_literals_and_full_refit_is_required(self):
        result = self.invoke()
        self.assertEqual(result["classes"], CLASSES)
        np.testing.assert_allclose(result["predictions"], np.tile([0.5, 0.2, 0.3], (6, 1)))
        self.assertEqual(self.predictor.predict_options["model"], "FitModel_FULL")
        self.assertEqual(self.predictor.refit_options["model"], "FitModel")
        self.assertEqual(len(self.predictor.all_refit_data), 90)
        self.assertEqual(self.evidence["full_fit_rows"], 90)
        self.assertTrue(self.evidence["full_refit"])
        self.assertFalse(Path(self.predictor.constructor["path"]).exists())

    def test_labels_are_selected_in_declared_class_order_and_no_threshold_calibration_occurs(self):
        result = self.invoke(ctx=context(metric="Macro F1"))
        np.testing.assert_array_equal(result["predictions"], ["01"] * 6)
        self.assertFalse(self.predictor.fit_options["calibrate_decision_threshold"])

    def test_fit_options_disable_download_models_stacking_bagging_and_parallel_ray(self):
        self.invoke(ctx=context(families=candidate.MODEL_FAMILIES))
        options = self.predictor.fit_options
        self.assertEqual(set(options["hyperparameters"]), set(candidate.MODEL_FAMILIES))
        self.assertEqual(options["num_cpus"], 8)
        self.assertEqual(options["num_gpus"], 0)
        self.assertEqual(options["fit_strategy"], "sequential")
        self.assertEqual(options["num_bag_folds"], 0)
        self.assertEqual(options["num_stack_levels"], 0)
        self.assertFalse(options["auto_stack"])
        self.assertFalse(options["dynamic_stacking"])
        self.assertFalse(options["fit_weighted_ensemble"])
        self.assertEqual(options["memory_limit"], 24.0)
        self.assertNotIn("label_count_threshold", options)
        self.assertEqual(self.predictor.constructor["learner_kwargs"], {"label_count_threshold": 1})
        self.assertTrue(all(value is False for value in options["feature_generator"].options.values()))

    def test_predict_features_never_enter_fit_or_tuning_and_inputs_are_unchanged(self):
        fitting, prediction = frame(), frame(6)
        prediction["x"] = ["9000"] * 6
        prediction["kind"] = ["never-in-fit"] * 6
        before_fit, before_predict = fitting.copy(deep=True), prediction.copy(deep=True)
        labels = pd.Series(CLASSES * 30)
        before_labels = labels.copy(deep=True)
        self.invoke(fitting, labels, prediction)
        trained = self.predictor.all_refit_data
        self.assertNotIn("never-in-fit", trained["kind"].tolist())
        self.assertLess(trained["x"].max(), 9000)
        pd.testing.assert_frame_equal(fitting, before_fit)
        pd.testing.assert_frame_equal(prediction, before_predict)
        pd.testing.assert_series_equal(labels, before_labels)

    def test_phase_does_not_change_training_or_recipe(self):
        snapshots = []
        for phase in ("inner_selection", "outer_validation", "full_refit"):
            ctx = context()
            ctx["phase"] = phase
            result = self.invoke(ctx=ctx)
            snapshots.append((self.predictor.all_refit_data, result["predictions"]))
        for data, values in snapshots[1:]:
            pd.testing.assert_frame_equal(data, snapshots[0][0])
            np.testing.assert_array_equal(values, snapshots[0][1])

    def test_grouped_internal_split_never_splits_duplicate_raw_groups(self):
        fitting = pd.DataFrame({"x": [str(index // 2) for index in range(120)],
                                "kind": [CLASSES[(index // 2) % 3] for index in range(120)]})
        labels = pd.Series([CLASSES[(index // 2) % 3] for index in range(120)])
        self.invoke(fitting, labels, ctx=context(policy={"strategy": "grouped"}))
        options = self.predictor.fit_options
        training = set(map(tuple, options["train_data"][["x", "kind"]].to_numpy()))
        tuning = set(map(tuple, options["tuning_data"][["x", "kind"]].to_numpy()))
        self.assertFalse(training & tuning)
        self.assertEqual(self.evidence["internal_split"]["group_overlap"], 0)

    def test_time_internal_split_preserves_complete_periods(self):
        fitting = pd.DataFrame({"x": [str(index // 12) for index in range(72)],
                                "kind": ["NA", "0007", "7"] * 24})
        labels = pd.Series(np.arange(72, dtype=float))
        ctx = context(classification=False, policy={"strategy": "time", "time_column": "x"})
        result = self.invoke(fitting, labels, ctx=ctx)
        options = self.predictor.fit_options
        self.assertLess(options["train_data"]["x"].max(), options["tuning_data"]["x"].min())
        self.assertEqual(options["tuning_data"]["x"].nunique(), 1)
        self.assertEqual(self.evidence["full_fit_rows"], 72)
        self.assertNotIn("classes", result)
        np.testing.assert_array_equal(result["predictions"], np.arange(6, dtype=float))

    def test_impossible_time_policy_is_rejected_without_random_fallback(self):
        fitting = frame()
        fitting["x"] = "1"
        ctx = context(classification=False, policy={"strategy": "time", "time_column": "x"})
        with self.assertRaisesRegex(candidate.AutoGluonCandidateError, "Fit-only tuning split"):
            self.invoke(fitting, pd.Series(np.arange(90, dtype=float)), ctx=ctx)
        self.assertFalse(FakePredictor.instances)

    def test_refit_missing_or_copy_of_old_model_cannot_claim_full_data_training(self):
        for value in ({}, {"FitModel": "FitModel"}, {"FitModel": None}):
            FakePredictor.refit_result = value
            with self.subTest(value=value), self.assertRaisesRegex(candidate.AutoGluonCandidateError, "full-data refit"):
                self.invoke()
            self.assertIsNone(FakePredictor.instances[-1].predict_options)

    def test_backend_full_refit_must_report_all_supplied_fit_rows(self):
        for count in (None, 72, True):
            with self.subTest(count=count), patch.object(FakePredictor, "model_info", return_value={"num_samples": count}):
                with self.assertRaisesRegex(candidate.AutoGluonCandidateError, "refit row count"):
                    self.invoke()
                self.assertIsNone(FakePredictor.instances[-1].predict_options)

    def test_host_deadline_and_time_budget_clamp_the_soft_initial_fit_limit(self):
        ctx = context()
        ctx.update(deadline_monotonic=110.0, time_budget_seconds=3.0)
        self.invoke(ctx=ctx)
        self.assertAlmostEqual(self.predictor.fit_options["time_limit"], 3.0 * 0.45)
        self.assertEqual(self.evidence["effective_time_limit"], 3.0)

    def test_expired_host_deadline_starts_no_model_fit(self):
        ctx = context()
        ctx["deadline_monotonic"] = 99.0
        with self.assertRaises(TimeoutError):
            self.invoke(ctx=ctx)
        self.assertTrue(all(model.fit_options is None for model in FakePredictor.instances))

    def test_expiring_refit_cannot_publish_unverified_or_incomplete_result(self):
        # start, loading-budget, initial-start, initial-budget, initial-end, full-budget.
        with self.assertRaises(TimeoutError):
            self.invoke(clock=iter([100.0, 100.0, 100.0, 100.0, 279.8, 279.8]).__next__)
        self.assertIsNone(FakePredictor.instances[-1].refit_options)

    def test_malformed_schema_targets_and_task_identity_are_rejected(self):
        cases = []
        ctx = context(); ctx["numeric"] = ["x", "kind"]; cases.append((frame(), pd.Series(CLASSES * 30), ctx))
        ctx = context(); ctx["features"] = ["kind", "x"]; cases.append((frame(), pd.Series(CLASSES * 30), ctx))
        ctx = context(); ctx["task_type"] = "regression"; cases.append((frame(), pd.Series(CLASSES * 30), ctx))
        ctx = context(); ctx["metric"] = candidate.get_metric("Accuracy").to_dict(); cases.append((frame(), pd.Series(CLASSES * 30), ctx))
        ctx = context(); cases.append((frame(), pd.Series(["01"] * 90), ctx))
        ctx = context(); labels = pd.Series(CLASSES * 30); labels.iloc[0] = None; cases.append((frame(), labels, ctx))
        for fitting, labels, ctx in cases:
            with self.subTest(ctx=ctx), self.assertRaises(candidate.AutoGluonCandidateError):
                self.invoke(fitting, labels, ctx=ctx)
        self.assertFalse(FakePredictor.instances)

    def test_unknown_probability_class_columns_are_rejected(self):
        FakePredictor.probability_columns = ["NA", "class c", "not-a-class"]
        with self.assertRaises(ValueError):
            self.invoke()

    def test_generated_target_name_does_not_overwrite_a_feature(self):
        fitting, prediction = frame(), frame(6)
        fitting = fitting.rename(columns={"kind": "__alfa_autogluon_target__"})
        prediction = prediction.rename(columns={"kind": "__alfa_autogluon_target__"})
        ctx = context()
        ctx["features"] = list(fitting.columns)
        ctx["categorical"] = ["__alfa_autogluon_target__"]
        self.invoke(fitting, prediction=prediction, ctx=ctx)
        self.assertEqual(self.predictor.constructor["label"], "__alfa_autogluon_target___")
        self.assertEqual(self.predictor.all_refit_data["__alfa_autogluon_target__"].tolist(), fitting.iloc[:, 1].tolist())


if __name__ == "__main__":
    unittest.main()
