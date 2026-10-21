"""Native early-stopping contracts using fit spies; no estimator is trained."""

from contextlib import contextmanager
from copy import deepcopy
import json
from io import StringIO
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import boosting_candidate as boosting
from agent.features import FeatureTransformer, normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import make_model_config
from agent.models import _LightGBMFrame
from agent.splits import split_rows


CLASSES = ["NA", "class c", "01"]


def ordinary_recipe(action="catboost", *, classification=False, features=None):
    return {"id": "base", "label": "Original boosting candidate", "action": action,
            "model_config": make_model_config(action, "classification" if classification else "regression"),
            "feature_config": normalize_feature_config(features)}


def data(*, classification=False, grouped=False):
    groups = np.repeat(np.arange(60), 3) if grouped else np.arange(160)
    count = len(groups)
    periods = (groups % 3 + 1 if grouped else np.where(groups < 60, 1, np.where(groups < 120, 2, 3)))
    frame = pd.DataFrame({"x": [str(value) for value in groups],
                          "period": [str(value) for value in periods],
                          "kind": [["NA", "0007", "7"][value % 3] for value in groups]},
                         index=["fit_" + str(index) for index in range(count)])
    if not grouped:
        frame.loc[frame["period"].eq("3"), "kind"] = "inner_unknown"
    labels = ([CLASSES[value % len(CLASSES)] for value in groups] if classification
              else [float(value * 2 + 1) for value in groups])
    target = pd.Series(labels, index=frame.index)
    prediction = pd.DataFrame({"x": ["901", "902", "903"], "period": ["4"] * 3,
                               "kind": ["test_only", "NA", "0007"]},
                              index=["predict_0", "predict_1", "predict_2"])
    return frame, target, prediction


def context(recipe, *, classification=False, metric=None, phase="inner_selection", deadline=None):
    metric = get_metric(metric or ("LogLoss" if classification else "RMSE"))
    return {"version": 3, "protocol_version": 3, "phase": phase,
            "nonce": "boosting-unit", "recipe": recipe,
            "code_hash": "a" * 64, "recipe_hash": "b" * 64,
            "features": ["x", "period", "kind"], "numeric": ["x", "period"],
            "categorical": ["kind"], "classes": list(CLASSES) if classification else [],
            "positive_class": None, "metric": metric.to_dict(), "task_type": metric.task_type,
            "prediction_kind": metric.prediction_kind,
            "deadline_monotonic": time.monotonic() + 120 if deadline is None else deadline}


class EarlyEstimator:
    def __init__(self, action, best_iteration=137, on_fit=None):
        self.action = action
        self.best_iteration_ = best_iteration
        self.on_fit = on_fit
        self.fit_calls = []

    def fit(self, frame, target, **kwargs):
        self.fit_calls.append((frame.copy(deep=True), np.asarray(target).copy(), kwargs))
        if self.on_fit is not None:
            self.on_fit()
        return self

    def get_best_iteration(self):
        return self.best_iteration_


class RefitPipeline:
    def __init__(self):
        self.fit_calls = []
        self.predict_calls = []
        self.classes_ = np.asarray(["class c", "01", "NA"])

    def fit(self, frame, target):
        self.fit_calls.append((frame.copy(deep=True), np.asarray(target).copy()))
        return self

    def predict(self, frame):
        self.predict_calls.append(frame.copy(deep=True))
        return np.full(len(frame), 7.0)

    def predict_proba(self, frame):
        self.predict_calls.append(frame.copy(deep=True))
        return np.tile([0.6, 0.1, 0.3], (len(frame), 1))


@contextmanager
def spies(*, best_iteration=137, on_early_fit=None):
    calls = {"early": [], "refit": [], "feature_fit": [], "category_fit": []}

    class RecordingFeatures(FeatureTransformer):
        def fit(self, frame, target=None):
            calls["feature_fit"].append(frame.copy(deep=True))
            return super().fit(frame, target)

    class RecordingCategories(_LightGBMFrame):
        def fit(self, frame, target=None):
            result = super().fit(frame, target)
            calls["category_fit"].append({"frame": frame.copy(deep=True),
                                          "categories": deepcopy(self.categories_)})
            return result

    def estimator(action, task_type, parameters, categorical, early_metric):
        model = EarlyEstimator(action, best_iteration, on_early_fit)
        calls["early"].append({"action": action, "task_type": task_type,
                               "parameters": deepcopy(parameters), "categorical": list(categorical),
                               "early_metric": early_metric, "model": model})
        return model

    def pipeline(*args, **kwargs):
        model = RefitPipeline()
        calls["refit"].append({"args": deepcopy(args), "kwargs": deepcopy(kwargs), "model": model})
        return model

    with patch.object(boosting, "_make_estimator", side_effect=estimator), \
            patch.object(boosting, "_lightgbm_callbacks", return_value=["native_callback_spy"]), \
            patch.object(boosting, "FeatureTransformer", RecordingFeatures), \
            patch.object(boosting, "_LightGBMFrame", RecordingCategories), \
            patch.object(boosting, "build_pipeline", side_effect=pipeline):
        yield calls


class BoostingRecipeTests(unittest.TestCase):
    def test_factory_accepts_explicit_policy_and_returns_detached_recipe(self):
        base = ordinary_recipe()
        validation = {"strategy": "time", "time_column": "period"}
        before = deepcopy((base, validation))
        recipe = boosting.make_boosting_recipe(base, validation, max_iterations=300, stopping_rounds=20)
        self.assertEqual(recipe["family"], "early_stopping_boosting_v1")
        self.assertEqual((base, validation), before)
        recipe["base_recipe"]["model_config"]["parameters"]["depth"] = 9
        self.assertEqual(base, before[0])

    def test_factory_requires_a_single_supported_model(self):
        invalid = []
        with_components = ordinary_recipe()
        with_components["components"] = [ordinary_recipe()]
        invalid.append(with_components)
        invalid.append(ordinary_recipe("linear"))
        extra = ordinary_recipe()
        extra["arbitrary_setting"] = "unsupported"
        invalid.append(extra)
        transformed = ordinary_recipe()
        transformed["model_config"]["target_transform"] = "log1p"
        invalid.append(transformed)
        for base in invalid:
            with self.subTest(base=base), self.assertRaises(ValueError):
                boosting.make_boosting_recipe(base, {"strategy": "random"})

    def test_factory_rejects_auto_or_missing_validation(self):
        for validation in (None, {"strategy": "auto"}, {"strategy": "time"},
                           {"strategy": "random", "time_column": "period"}):
            with self.subTest(validation=validation), self.assertRaises(ValueError):
                boosting.make_boosting_recipe(ordinary_recipe(), validation)

    def test_factory_enforces_finite_iteration_and_stopping_bounds(self):
        invalid = ((49, 10), (2001, 10), (True, 10), (300, 0),
                   (300, 201), (300, True), (50, 51), (300.0, 10))
        for iterations, stopping in invalid:
            with self.subTest(iterations=iterations, stopping=stopping), self.assertRaises(ValueError):
                boosting.make_boosting_recipe(ordinary_recipe(), {"strategy": "random"},
                                              max_iterations=iterations, stopping_rounds=stopping)

    def test_generated_source_compiles_without_materializing_files(self):
        source = boosting.candidate_source()
        self.assertIsInstance(source, str)
        compile(source, "boosting_generated.py", "exec")


class BoostingExecutionTests(unittest.TestCase):
    def invoke(self, action="catboost", *, classification=False, metric=None,
               validation=None, max_iterations=250, best_iteration=137, phase="inner_selection"):
        training, target, prediction = data(classification=classification)
        validation = validation or ({"strategy": "random"} if classification
                                     else {"strategy": "time", "time_column": "period"})
        recipe = boosting.make_boosting_recipe(ordinary_recipe(action, classification=classification), validation,
                                              max_iterations=max_iterations, stopping_rounds=20)
        with spies(best_iteration=best_iteration) as calls:
            result, evidence = boosting.execute_boosting_candidate(
                training, target, prediction,
                context(recipe, classification=classification, metric=metric, phase=phase),
                return_evidence=True)
        return result, evidence, calls, training, target, prediction, recipe

    def test_catboost_selects_zero_based_best_iteration_then_refits_all_training_rows(self):
        result, _, calls, training, target, prediction, _ = self.invoke()
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(len(calls["refit"]), 1)
        early = calls["early"][0]["model"]
        self.assertEqual(len(early.fit_calls), 1)
        self.assertLess(len(early.fit_calls[0][0]), len(training))
        refit = calls["refit"][0]
        config = refit["args"][4] if len(refit["args"]) > 4 else refit["kwargs"]["model_config"]
        self.assertEqual(config["parameters"]["iterations"], 138)
        pd.testing.assert_frame_equal(refit["model"].fit_calls[0][0], training.reset_index(drop=True))
        np.testing.assert_array_equal(refit["model"].fit_calls[0][1], target.to_numpy())
        pd.testing.assert_frame_equal(refit["model"].predict_calls[0], prediction)
        np.testing.assert_array_equal(result["predictions"], np.full(len(prediction), 7.0))

    def test_lightgbm_uses_one_based_best_iteration(self):
        _, _, calls, *_ = self.invoke("lightgbm")
        refit = calls["refit"][0]
        config = refit["args"][4] if len(refit["args"]) > 4 else refit["kwargs"]["model_config"]
        self.assertEqual(config["parameters"]["n_estimators"], 137)

    def test_small_valid_best_iteration_is_clamped_to_minimum_fifty(self):
        for action in ("catboost", "lightgbm"):
            for observed, expected in ((2, 50), (249, 250 if action == "catboost" else 249)):
                with self.subTest(action=action, observed=observed):
                    _, _, calls, *_ = self.invoke(action, best_iteration=observed)
                    refit = calls["refit"][0]
                    config = refit["args"][4] if len(refit["args"]) > 4 else refit["kwargs"]["model_config"]
                    name = "iterations" if action == "catboost" else "n_estimators"
                    self.assertEqual(config["parameters"][name], expected)

    def test_impossible_native_best_iteration_is_a_typed_skip(self):
        for action in ("catboost", "lightgbm"):
            with self.subTest(action=action), self.assertRaises(boosting.BoostingSkipError) as caught:
                self.invoke(action, best_iteration=9999)
            self.assertEqual(caught.exception.reason, "best_iteration_unavailable")

    def test_time_policy_and_preprocessing_use_only_internal_training_rows(self):
        _, _, calls, training, target, prediction, _ = self.invoke("lightgbm")
        fitting, held, _ = split_rows(training, target, list(training.columns), False,
                                     {"strategy": "time", "time_column": "period"})
        self.assertEqual(len(calls["feature_fit"]), 1)
        pd.testing.assert_frame_equal(calls["feature_fit"][0], training.reset_index(drop=True).iloc[fitting])
        self.assertEqual(len(calls["category_fit"]), 1)
        category_fit = calls["category_fit"][0]
        self.assertEqual(category_fit["frame"].index.tolist(), fitting.tolist())
        self.assertNotIn("inner_unknown", category_fit["categories"]["kind"])
        self.assertNotIn("test_only", category_fit["categories"]["kind"])
        early_frame, early_y, kwargs = calls["early"][0]["model"].fit_calls[0]
        self.assertEqual(early_frame.index.tolist(), fitting.tolist())
        np.testing.assert_array_equal(early_y, target.iloc[fitting].to_numpy())
        evaluation = kwargs["eval_set"]
        evaluation = evaluation[0] if isinstance(evaluation, list) else evaluation
        self.assertEqual(evaluation[0].index.tolist(), held.tolist())
        np.testing.assert_array_equal(evaluation[1], target.iloc[held].to_numpy())
        self.assertEqual(set(calls["feature_fit"][0]["x"]) & set(prediction["x"]), set())

    def test_native_catboost_preserves_literal_categories(self):
        _, _, calls, *_ = self.invoke("catboost")
        frame, _, kwargs = calls["early"][0]["model"].fit_calls[0]
        self.assertEqual(set(frame["kind"]), {"NA", "0007", "7"})
        evaluation = kwargs["eval_set"]
        evaluation = evaluation[0] if isinstance(evaluation, list) else evaluation
        self.assertEqual(set(evaluation[0]["kind"]), {"inner_unknown"})

    def test_class_probabilities_are_aligned_to_literal_context_order(self):
        result, _, calls, *_ = self.invoke("lightgbm", classification=True)
        self.assertEqual(result["classes"], CLASSES)
        np.testing.assert_allclose(result["predictions"], np.tile([0.3, 0.6, 0.1], (3, 1)), rtol=0, atol=0)
        self.assertEqual(calls["refit"][0]["model"].classes_.tolist(), ["class c", "01", "NA"])

    def test_label_predictions_follow_aligned_probabilities(self):
        result, *_ = self.invoke("catboost", classification=True, metric="Macro F1")
        self.assertEqual(list(result["predictions"]), ["class c"] * 3)

    def test_lightgbm_internal_class_encoding_uses_context_order(self):
        _, _, calls, training, target, *_ = self.invoke("lightgbm", classification=True)
        frame, actual, _ = calls["early"][0]["model"].fit_calls[0]
        positions = {label: index for index, label in enumerate(CLASSES)}
        expected = [positions[label] for label in target.iloc[frame.index]]
        np.testing.assert_array_equal(actual, expected)

    def test_phase_is_not_a_training_or_prediction_switch(self):
        first = self.invoke("catboost", phase="inner_selection")
        second = self.invoke("catboost", phase="full_refit")
        np.testing.assert_array_equal(first[0]["predictions"], second[0]["predictions"])
        for key in ("parameters", "categorical", "early_metric"):
            self.assertEqual(first[2]["early"][0][key], second[2]["early"][0][key])
        pd.testing.assert_frame_equal(first[2]["early"][0]["model"].fit_calls[0][0],
                                      second[2]["early"][0]["model"].fit_calls[0][0])

    def test_grouped_policy_keeps_raw_duplicate_groups_disjoint(self):
        training, target, prediction = data(classification=True, grouped=True)
        recipe = boosting.make_boosting_recipe(ordinary_recipe("lightgbm", classification=True),
                                              {"strategy": "grouped"}, max_iterations=250, stopping_rounds=20)
        with spies() as calls:
            _, evidence = boosting.execute_boosting_candidate(training, target, prediction,
                context(recipe, classification=True), return_evidence=True)
        fitting, _, kwargs = calls["early"][0]["model"].fit_calls[0]
        held = kwargs["eval_set"][0][0]
        raw = training.reset_index(drop=True)
        fit_groups = set(pd.util.hash_pandas_object(raw.iloc[fitting.index], index=False))
        held_groups = set(pd.util.hash_pandas_object(raw.iloc[held.index], index=False))
        self.assertFalse(fit_groups & held_groups)
        self.assertEqual(evidence["validation"]["strategy"], "grouped")
        self.assertEqual(evidence["validation"]["group_overlap"], 0)

    def test_lightgbm_callback_uses_host_metric_and_literal_class_encoding(self):
        probabilities = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        for metric, expected in (("Macro F1", 1.0), ("LogLoss", -np.log(0.9))):
            with self.subTest(metric=metric):
                _, _, calls, *_ = self.invoke("lightgbm", classification=True, metric=metric)
                evaluator = calls["early"][0]["early_metric"]
                name, value, higher = evaluator(np.asarray([0, 1, 2]), probabilities)
                self.assertEqual(name, "host_" + metric.replace(" ", "_"))
                self.assertAlmostEqual(value, expected, places=12)
                self.assertEqual(higher, get_metric(metric).higher_is_better)

    def test_catboost_macro_f1_early_metric_is_explicit_and_unweighted(self):
        _, _, calls, *_ = self.invoke("catboost", classification=True, metric="Macro F1")
        self.assertEqual(calls["early"][0]["early_metric"], "TotalF1:average=Macro;use_weights=false")

    def test_native_category_budget_uses_full_fit_rows_and_never_ordinal_fallback(self):
        training, target, prediction = data()
        recipe = boosting.make_boosting_recipe(ordinary_recipe(),
                                              {"strategy": "time", "time_column": "period"})
        # Internal fitting contains three categories; the fourth appears only
        # in internal validation. The full-refit guard must still count it.
        for limit, value in (("MAX_NATIVE_CATEGORIES", 3),
                             ("MAX_NATIVE_CATEGORY_CELLS", len(training) - 1)):
            with self.subTest(limit=limit), patch.object(boosting, limit, value), spies() as calls:
                with self.assertRaises(boosting.BoostingSkipError) as caught:
                    boosting.execute_boosting_candidate(training, target, prediction, context(recipe))
                self.assertEqual(caught.exception.reason, "native_category_budget_exceeded")
                self.assertEqual(calls["early"], [])
                self.assertEqual(calls["refit"], [])

    def test_impossible_grouped_split_is_not_replaced_by_random_validation(self):
        training, target, prediction = data(classification=True)
        training.loc[:, "x"] = "one_group"
        training.loc[:, "period"] = "1"
        training.loc[:, "kind"] = "same"
        recipe = boosting.make_boosting_recipe(ordinary_recipe("lightgbm", classification=True),
                                              {"strategy": "grouped"})
        with spies() as calls, self.assertRaises(boosting.BoostingSkipError) as caught:
            boosting.execute_boosting_candidate(training, target, prediction, context(recipe, classification=True))
        self.assertEqual(caught.exception.reason, "internal_split_unavailable")
        self.assertEqual(calls["early"], [])
        self.assertEqual(calls["refit"], [])

    def test_prediction_rows_and_phase_cannot_select_iterations(self):
        training, target, prediction = data()
        original_training, original_target = training.copy(deep=True), target.copy(deep=True)
        recipe = boosting.make_boosting_recipe(ordinary_recipe(),
                                              {"strategy": "time", "time_column": "period"},
                                              max_iterations=250, stopping_rounds=20)
        evidence = []
        early_inputs = []
        for query in (prediction, prediction.assign(kind="completely_different", x="-100000")):
            with spies() as calls:
                _, trace = boosting.execute_boosting_candidate(training, target, query, context(recipe), return_evidence=True)
            evidence.append(trace)
            early_inputs.append(calls["early"][0]["model"].fit_calls[0][0])
        self.assertEqual(evidence[0]["selected_iterations"], evidence[1]["selected_iterations"])
        pd.testing.assert_frame_equal(early_inputs[0], early_inputs[1])
        pd.testing.assert_frame_equal(training, original_training)
        pd.testing.assert_series_equal(target, original_target)

    def test_expired_deadline_prevents_any_model_fit(self):
        training, target, prediction = data()
        recipe = boosting.make_boosting_recipe(ordinary_recipe(), {"strategy": "random"})
        with spies() as calls, self.assertRaises((TimeoutError, boosting.BoostingSkipError)):
            boosting.execute_boosting_candidate(training, target, prediction, context(recipe, deadline=time.monotonic() - 1))
        self.assertEqual(calls["early"], [])
        self.assertEqual(calls["refit"], [])

    def test_deadline_expiring_during_early_fit_prevents_full_refit(self):
        training, target, prediction = data()
        recipe = boosting.make_boosting_recipe(ordinary_recipe(), {"strategy": "random"})
        clock = [1000.0]
        def expire():
            clock[0] = 1011.0
        with patch.object(boosting.time, "monotonic", side_effect=lambda: clock[0]), \
                spies(on_early_fit=expire) as calls, self.assertRaises((TimeoutError, boosting.BoostingSkipError)):
            boosting.execute_boosting_candidate(training, target, prediction, context(recipe, deadline=1010.0))
        self.assertEqual(len(calls["early"]), 1)
        self.assertEqual(calls["refit"], [])

    def test_execute_has_no_diagnostic_file_io(self):
        with patch.object(Path, "write_text", side_effect=AssertionError("Direct execution wrote a file")), \
                patch.object(Path, "open", side_effect=AssertionError("Direct execution opened a file")):
            self.invoke("catboost")

    def test_generated_source_emits_receipt_and_preserves_strict_result(self):
        class Capture(StringIO):
            def __exit__(self, *args):
                self.flush()
                return False

        expected = {"predictions": [7.0, 8.0]}
        evidence = {"version": 1, "status": "completed", "fresh_full_refit": True}
        stream = Capture()
        with patch.object(boosting, "execute_boosting_candidate", return_value=(expected, evidence)), \
                patch.object(Path, "open", autospec=True, return_value=stream) as writer:
            namespace = {}
            exec(compile(boosting.candidate_source(), "boosting_generated.py", "exec"), namespace)
            actual = namespace["fit_predict"](None, None, None, {})
        self.assertIs(actual, expected)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(writer.call_args.args[0].name, boosting.EVIDENCE_NAME)
        self.assertEqual(writer.call_args.args[1], "x")
        self.assertEqual(json.loads(stream.getvalue()), evidence)
        stream.close()


if __name__ == "__main__":
    unittest.main()
