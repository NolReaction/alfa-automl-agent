"""Probability identity, class alignment, fit-only behavior and deadline evidence."""
from copy import deepcopy
import importlib.util
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import round4_classification_probability as adapter
from agent.features import normalize_feature_config
from agent.model_config import make_model_config


CLASSES = ["z", "01", "NA"]


def data():
    fitting = pd.DataFrame({"number": ["0", "1", "2", "3", "4", "5"],
                            "kind": ["a", "b", "a", "b", "a", "b"]}, index=[10, 20, 30, 40, 50, 60])
    labels = pd.Series(["z", "01", "NA", "z", "01", "NA"], index=fitting.index)
    predicting = pd.DataFrame({"number": ["100", "2"], "kind": ["unseen", "a"]}, index=[99, 100])
    return fitting, labels, predicting


def small_base():
    return {"components": [
        {"action": "baseline", "model_config": make_model_config("baseline", "classification", {"n_estimators": 50}),
         "feature_config": normalize_feature_config()},
        {"action": "lightgbm", "model_config": make_model_config("lightgbm", "classification", {"n_estimators": 50}),
         "feature_config": normalize_feature_config()},
    ]}


def context(base=None, *, phase="inner_selection"):
    recipe = adapter.make_probability_recipe(base or small_base(), CLASSES)
    return {"recipe": recipe, "task_type": "classification", "metric": "Macro F1", "prediction_kind": "labels",
            "classes": list(CLASSES), "features": ["number", "kind"], "numeric": ["number"],
            "categorical": ["kind"], "deadline_monotonic": time.monotonic() + 30, "phase": phase,
            "positive_class": None}


class FakeModel:
    def __init__(self, classes, probabilities, *, mutate=False, fit_error=False):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)
        self.mutate, self.fit_error = mutate, fit_error
        self.fit_frames, self.targets, self.predict_frames = [], [], []
        self.fit_options = []

    def fit(self, frame, labels, **kwargs):
        self.fit_frames.append(frame.copy(deep=True))
        self.targets.append(np.asarray(labels).copy())
        self.fit_options.append(deepcopy(kwargs))
        if self.mutate:
            frame.iloc[0, 0] = "changed_by_model"
            labels[0] = "changed_by_model"
        if self.fit_error:
            raise RuntimeError("expected test failure")
        return self

    def predict_proba(self, frame):
        self.predict_frames.append(frame.copy(deep=True))
        if self.mutate:
            frame.iloc[0, 0] = "changed_by_model"
        return self.probabilities.copy()


class ProbabilityAdapterTests(unittest.TestCase):
    def test_equal_blend_aligns_real_columns_and_leaves_task_and_inputs_unchanged(self):
        fit, labels, predict = data()
        original = fit.copy(), labels.copy(), predict.copy()
        ctx = context()
        original_context = deepcopy(ctx)
        first = FakeModel(["NA", "z", "01"], [[.2, .6, .2], [.3, .1, .6]], mutate=True)
        second = FakeModel(["01", "NA", "z"], [[.4, .2, .4], [.2, .2, .6]], mutate=True)
        with patch.object(adapter, "build_pipeline", side_effect=[first, second]) as factory:
            result = adapter.fit_predict_probabilities(fit, labels, predict, ctx)
        np.testing.assert_allclose(result["probabilities"], [[.5, .3, .2], [.35, .4, .25]])
        self.assertEqual(result["classes"], CLASSES)
        self.assertEqual(result["evidence"]["physical_fits_started"], 2)
        self.assertEqual(result["evidence"]["physical_fits_completed"], 2)
        self.assertEqual(factory.call_count, 2)
        for model in (first, second):
            pd.testing.assert_frame_equal(model.fit_frames[0], original[0])
            pd.testing.assert_frame_equal(model.predict_frames[0], original[2])
            np.testing.assert_array_equal(model.targets[0], original[1].to_numpy())
        pd.testing.assert_frame_equal(fit, original[0])
        pd.testing.assert_series_equal(labels, original[1])
        pd.testing.assert_frame_equal(predict, original[2])
        self.assertEqual(ctx, original_context)

    def test_exact_equal_maxima_keep_host_tie_order(self):
        fit, labels, predict = data()
        base = small_base()["components"][0]
        fake = FakeModel(["NA", "01", "z"], [[0., .5, .5], [0., .5, .5]])
        with patch.object(adapter, "build_pipeline", return_value=fake):
            result = adapter.fit_predict_probabilities(fit, labels, predict, context(base))
        np.testing.assert_array_equal(np.asarray(CLASSES)[result["probabilities"].argmax(axis=1)], ["z", "z"])

    def test_missing_fit_class_and_expired_deadline_skip_before_model_construction(self):
        fit, labels, predict = data()
        for selected, error in (([0, 1, 3, 4], adapter.ProbabilitySkipError), (None, adapter.ProbabilityDeadlineError)):
            ctx = context()
            if selected is None:
                ctx["deadline_monotonic"] = time.monotonic() - 1
                X, y = fit, labels
            else:
                X, y = fit.iloc[selected], labels.iloc[selected]
            with patch.object(adapter, "build_pipeline") as factory, self.assertRaises(error) as caught:
                adapter.fit_predict_probabilities(X, y, predict, ctx)
            factory.assert_not_called()
            self.assertEqual(caught.exception.evidence["physical_fits_started"], 0)

    def test_no_probability_fabrication_for_missing_class_nan_or_zero_sum(self):
        fit, labels, predict = data()
        variants = [(["z", "01"], [[.5, .5], [.5, .5]]),
                    (CLASSES, [[np.nan, 0., 1.], [.5, .5, 0.]]),
                    (CLASSES, [[0., 0., 0.], [0., 0., 0.]]),
                    (CLASSES, ["z", "01"])]
        for classes, raw in variants:
            with self.subTest(classes=classes, raw=raw):
                fake = FakeModel(classes, raw)
                with patch.object(adapter, "build_pipeline", return_value=fake), self.assertRaises(ValueError):
                    adapter.fit_predict_probabilities(fit, labels, predict, context(small_base()["components"][0]))

    def test_late_fit_cannot_return_partial_probabilities_or_start_next_component(self):
        fit, labels, predict = data()
        fake = FakeModel(CLASSES, [[1., 0., 0.], [0., 1., 0.]])
        ctx = context()
        # Stage clocks: start, start deadline, preparation, fit deadline,
        # fit timing, elapsed after fit, then after-fit deadline is exhausted.
        with patch.object(adapter.time, "monotonic", side_effect=[0., 0., 0., 0., 0., 2., 2., 2.]), \
                patch.object(adapter, "build_pipeline", return_value=fake) as factory:
            ctx["deadline_monotonic"] = 1.
            with self.assertRaises(adapter.ProbabilityDeadlineError) as caught:
                adapter.fit_predict_probabilities(fit, labels, predict, ctx)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(fake.predict_frames), 0)
        self.assertEqual(caught.exception.evidence["physical_fits_completed"], 1)

    def test_failure_keeps_physical_fit_attempt_count(self):
        fit, labels, predict = data()
        fake = FakeModel(CLASSES, [], fit_error=True)
        with patch.object(adapter, "build_pipeline", return_value=fake), self.assertRaises(adapter.ProbabilityModelError) as caught:
            adapter.fit_predict_probabilities(fit, labels, predict, context())
        self.assertEqual(caught.exception.evidence["physical_fits_started"], 1)
        self.assertEqual(caught.exception.evidence["physical_fits_completed"], 0)

    def test_metric_substitution_and_runtime_recipe_tampering_rejected_before_fit(self):
        fit, labels, predict = data()
        variants = []
        ctx = context(); ctx.update(metric="LogLoss", prediction_kind="probabilities"); variants.append(ctx)
        ctx = context(); ctx["recipe"]["caps"]["max_threads"] = 16; variants.append(ctx)
        ctx = context(); ctx["classes"] = list(reversed(CLASSES)); variants.append(ctx)
        for ctx in variants:
            with patch.object(adapter, "build_pipeline") as factory, self.assertRaises(ValueError):
                adapter.fit_predict_probabilities(fit, labels, predict, ctx)
            factory.assert_not_called()

    def test_explicit_recipe_override_weights_and_observer_order(self):
        fit, labels, predict = data()
        ctx = context()
        recipe = ctx.pop("recipe")
        ctx["recipe"] = {"outer_wrapper": "unchanged"}
        weights = np.asarray([1., .5, 1., .5, 1., .5])
        original_weights = weights.copy()
        models = [FakeModel(CLASSES, [[1., 0., 0.], [0., 1., 0.]]) for _ in range(2)]
        events = []
        def observe(event):
            index = event["component_index"]
            expected_fits = 0 if event["status"] == "started" else 1
            self.assertEqual(len(models[index].fit_frames), expected_fits)
            events.append(event)
        with patch.object(adapter, "build_pipeline", side_effect=models):
            result = adapter.fit_predict_probabilities(fit, labels, predict, ctx, recipe=recipe,
                                                       sample_weight=weights, fit_observer=observe)
        self.assertEqual([(row["component_index"], row["status"]) for row in events],
                         [(0, "started"), (0, "completed"), (1, "started"), (1, "completed")])
        for model in models:
            np.testing.assert_array_equal(model.fit_options[0]["model__sample_weight"], weights)
        np.testing.assert_array_equal(weights, original_weights)
        self.assertTrue(result["evidence"]["sample_weight_provided"])
        self.assertEqual(ctx["recipe"], {"outer_wrapper": "unchanged"})

    def test_invalid_weights_rejected_before_physical_fit(self):
        fit, labels, predict = data()
        for weights in ([1., 2.], [1., 1., 1., 1., 1., np.inf], [0., 1., 1., 1., 1., 1.],
                        [True, 1., 1., 1., 1., 1.]):
            with patch.object(adapter, "build_pipeline") as factory, self.assertRaises(ValueError):
                adapter.fit_predict_probabilities(fit, labels, predict, context(), sample_weight=weights)
            factory.assert_not_called()

    def test_observer_deadline_cancels_started_event_without_a_physical_fit(self):
        fit, labels, predict = data()
        ctx = context()
        ctx["deadline_monotonic"] = 1.
        now, events = [0.], []
        model = FakeModel(CLASSES, [[1., 0., 0.], [0., 1., 0.]])
        def observe(event):
            events.append(event)
            if event["status"] == "started":
                now[0] = 2.
        with patch.object(adapter.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(adapter, "build_pipeline", return_value=model) as factory, \
                self.assertRaises(adapter.ProbabilityDeadlineError) as caught:
            adapter.fit_predict_probabilities(fit, labels, predict, ctx, fit_observer=observe)
        factory.assert_called_once()
        self.assertEqual([event["status"] for event in events], ["started", "cancelled_before_fit"])
        self.assertEqual(events[0]["component_index"], events[1]["component_index"])
        self.assertEqual(model.fit_frames, [])
        self.assertEqual(caught.exception.evidence["physical_fits_started"], 0)
        self.assertEqual(caught.exception.evidence["physical_fits_completed"], 0)
        self.assertEqual(caught.exception.evidence["components"][0]["status"], "cancelled_before_fit")


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "Pinned LightGBM is not installed")
class NativeProbabilityIdentityTests(unittest.TestCase):
    def test_real_probabilities_argmax_matches_current_baseline_source(self):
        from agent.research_v3 import BASELINE_SOURCE
        rng = np.random.default_rng(7)
        rows = 90
        fitting = pd.DataFrame({"number": rng.normal(size=rows).astype(str),
                                "kind": np.asarray(["0007", "7", "NA"])[np.arange(rows) % 3]})
        labels = pd.Series(np.asarray(CLASSES)[np.arange(rows) % 3])
        predicting = pd.DataFrame({"number": ["0", "1", "-1"], "kind": ["unknown", "0007", "NA"]})
        base = small_base()
        ctx = context(base)
        namespace = {}
        exec(compile(BASELINE_SOURCE, "current_baseline_source", "exec"), namespace)
        original_context = deepcopy(ctx)
        original_context["recipe"] = base
        expected = namespace["fit_predict"](fitting, labels, predicting, original_context)
        actual = adapter.fit_predict_probabilities(fitting, labels, predicting, ctx)
        np.testing.assert_array_equal(np.asarray(actual["classes"])[actual["probabilities"].argmax(axis=1)],
                                      expected["predictions"])
        self.assertGreater(np.count_nonzero((actual["probabilities"] > 0) & (actual["probabilities"] < 1)), 0)
        self.assertEqual(actual["evidence"]["physical_fits_completed"], 2)


if __name__ == "__main__":
    unittest.main()
