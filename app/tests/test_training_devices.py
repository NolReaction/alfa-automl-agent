"""CPU defaults, bounded capability checks and explicit GPU-only execution."""

from copy import deepcopy
import json
import os
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from agent.model_config import make_model_config
from agent.models import build_pipeline, CatBoostClassificationAdapter
from app import training_devices as devices


def capability():
    return {"available": True, "label": "GPU · CatBoost", "reason": None,
            "device_count": 1, "device_name": "NVIDIA GeForce RTX 3080",
            "driver_version": "555.00", "total_memory_mib": 10240.0,
            "catboost_version": "1.2.8", "training_verified": False}


class FakeCatBoost:
    calls = []

    def __init__(self, **kwargs):
        self.parameters = kwargs
        type(self).calls.append(self)

    def fit(self, X, y, **kwargs):
        self.fit_frame = X.copy(deep=True)
        self.classes_ = np.asarray(sorted(set(y)))
        return self

    def get_param(self, name):
        return self.parameters.get(name)

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


class TrainingDeviceTests(unittest.TestCase):
    def setUp(self):
        FakeCatBoost.calls = []

    def _pipeline(self, *, task_type="classification", options=None):
        config = make_model_config("catboost", task_type, {"iterations": 50}, **(options or {}))
        return build_pipeline("catboost", task_type, ["x"], ["category"], model_config=config)

    def test_cpu_default_and_gpu_only_for_fixed_catboost(self):
        self.assertEqual(devices.normalize_training_device(), "cpu")
        self.assertEqual(devices.normalize_training_device("gpu", action="catboost"), "gpu")
        for value in (True, "cuda", "GPU", 0, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                devices.normalize_training_device(value, action="catboost")
        for action in (None, "automl", "agent", "lightgbm", "linear"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                devices.normalize_training_device("gpu", action=action)

    def test_capabilities_are_cached_copied_and_never_claim_training(self):
        with patch.object(devices, "_capability_cache", None), \
                patch.object(devices, "probe_gpu_capability", return_value=capability()) as probe:
            first = devices.training_device_capabilities()
            self.assertEqual(first["default"], "cpu")
            self.assertEqual(first["supported_gpu_actions"], ["catboost"])
            self.assertFalse(first["gpu"]["training_verified"])
            first["gpu"]["available"] = False
            second = devices.training_device_capabilities()
            self.assertTrue(second["gpu"]["available"])
            self.assertEqual(probe.call_count, 1)
            devices.training_device_capabilities(force=True)
            self.assertEqual(probe.call_count, 2)

    def test_probe_has_fixed_timeout_and_strips_credentials(self):
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(capability()), stderr="")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-key", "SECRET_TOKEN": "private-token",
                                     "CUDA_VISIBLE_DEVICES": "7"}), \
                patch.object(devices.subprocess, "run", return_value=completed) as runner:
            result = devices.probe_gpu_capability()
        kwargs = runner.call_args.kwargs
        self.assertTrue(result["available"])
        self.assertEqual(kwargs["timeout"], 10)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("SECRET_TOKEN", kwargs["env"])
        self.assertNotIn("private-key", runner.call_args.args[0][-1])
        self.assertFalse(result["training_verified"])

    def test_probe_timeout_failure_and_malformed_reply_are_explicit_unavailable(self):
        replies = [SimpleNamespace(returncode=1, stdout="", stderr="secret"),
                   SimpleNamespace(returncode=0, stdout="broken", stderr=""),
                   SimpleNamespace(returncode=0, stdout=json.dumps({**capability(), "training_verified": True}), stderr="")]
        for reply in replies:
            with self.subTest(reply=reply), patch.object(devices.subprocess, "run", return_value=reply):
                result = devices.probe_gpu_capability()
                self.assertFalse(result["available"])
                self.assertFalse(result["training_verified"])
                self.assertNotIn("secret", result["reason"])
        with patch.object(devices.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 10)):
            result = devices.probe_gpu_capability()
        self.assertFalse(result["available"])
        self.assertIn("10", result["reason"])

    def test_library_probe_zero_devices_never_fits_anything(self):
        with patch("catboost.utils.get_gpu_device_count", return_value=0), \
                patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("must not fit")), \
                patch("shutil.which", return_value=None):
            result = devices._probe_in_process()
        self.assertFalse(result["available"])
        self.assertEqual(result["device_count"], 0)
        self.assertFalse(result["training_verified"])

    def test_cpu_pipeline_is_unchanged_and_confirmation_requires_fit(self):
        pipeline = build_pipeline("linear", "regression", ["x"], [])
        original = pipeline.named_steps["model"]
        with patch.object(devices, "probe_gpu_capability", side_effect=AssertionError("CPU must not probe GPU")):
            actual, evidence = devices.prepare_training_pipeline(pipeline, action="linear")
        self.assertIs(actual, pipeline)
        self.assertIs(actual.named_steps["model"], original)
        self.assertFalse(evidence["backend_verified"])
        with self.assertRaises(ValueError):
            devices.confirm_training_device(pipeline, evidence)
        X = pd.DataFrame({"x": np.arange(12, dtype=float)})
        pipeline.fit(X, np.arange(12, dtype=float) * 2)
        final = devices.confirm_training_device(pipeline, evidence)
        self.assertTrue(final["fit_completed"])
        self.assertTrue(final["backend_verified"])
        self.assertEqual(final["backend"], "CPU")
        self.assertFalse(final["exploratory"])
        self.assertFalse(evidence["fit_completed"])

    def test_gpu_unavailable_stops_before_fit_or_pipeline_replacement(self):
        pipeline = self._pipeline()
        original = pipeline.named_steps["model"]
        unavailable = devices._empty_gpu("Нет видеокарты")
        with patch.object(devices, "probe_gpu_capability", return_value=unavailable), \
                patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("must not fit")), \
                self.assertRaisesRegex(ValueError, "Нет видеокарты"):
            devices.prepare_training_pipeline(pipeline, action="catboost", training_device="gpu")
        self.assertIs(pipeline.named_steps["model"], original)
        self.assertIsInstance(original, CatBoostClassificationAdapter)

    def test_gpu_pipeline_clone_has_explicit_runtime_and_fold_local_categories(self):
        pipeline = self._pipeline()
        with patch.object(devices, "probe_gpu_capability", return_value=capability()):
            pipeline, evidence = devices.prepare_training_pipeline(pipeline, action="catboost", training_device="gpu")
        self.assertFalse(evidence["fit_completed"])
        X = pd.DataFrame({"x": [1., 2., 3., 4.], "category": ["a", "b", "a", "b"]})
        with patch("catboost.CatBoostClassifier", FakeCatBoost):
            pipeline.fit(X, np.asarray(["no", "yes", "no", "yes"]))
            final = clone(pipeline)
            final.fit(X, np.asarray(["no", "yes", "no", "yes"]))
        self.assertEqual(len(FakeCatBoost.calls), 2)
        runtime = FakeCatBoost.calls[-1].parameters
        self.assertEqual(runtime["task_type"], "GPU")
        self.assertEqual(runtime["devices"], "0")
        self.assertEqual(runtime["thread_count"], 8)
        self.assertEqual(runtime["gpu_ram_part"], .75)
        self.assertFalse(runtime["allow_writing_files"])
        observed = devices.confirm_training_device(final, evidence)
        self.assertEqual(observed["device_name"], "NVIDIA GeForce RTX 3080")
        self.assertTrue(observed["backend_verified"])
        self.assertTrue(observed["exploratory"])
        self.assertEqual(final.predict(X).tolist(), ["no"] * 4)
        self.assertNotIn("task_type", final.named_steps["model"].parameters)

    def test_gpu_category_fallback_and_target_wrapper_preserve_transforms(self):
        pipeline = self._pipeline(task_type="regression", options={"target_transform": "log1p"})
        with patch.object(devices, "probe_gpu_capability", return_value=capability()):
            pipeline, evidence = devices.prepare_training_pipeline(pipeline, action="catboost", training_device="gpu")
        X = pd.DataFrame({"x": [1., 2., 3., 4.], "category": ["a", "b", "a", "b"]})
        with patch("catboost.CatBoostRegressor", FakeCatBoost), patch.object(devices, "MAX_NATIVE_CATEGORIES", 0):
            pipeline.fit(X, np.asarray([1., 2., 3., 4.]))
        fitted = pipeline.named_steps["model"].regressor_
        self.assertIsInstance(fitted, devices.LocalGPUCatBoostRegressor)
        self.assertEqual(fitted.category_encoding_, "ordinal")
        self.assertEqual(FakeCatBoost.calls[0].parameters["cat_features"], [])
        self.assertTrue(devices.confirm_training_device(pipeline, evidence)["fit_completed"])

    def test_failed_gpu_fit_is_not_reported_as_success_or_retried_on_cpu(self):
        pipeline = self._pipeline()
        with patch.object(devices, "probe_gpu_capability", return_value=capability()):
            pipeline, evidence = devices.prepare_training_pipeline(pipeline, action="catboost", training_device="gpu")
        X = pd.DataFrame({"x": [1., 2.], "category": ["a", "b"]})
        with patch("catboost.CatBoostClassifier", FakeCatBoost), \
                patch.object(FakeCatBoost, "fit", side_effect=RuntimeError("GPU OOM")) as fit, \
                self.assertRaisesRegex(RuntimeError, "GPU OOM"):
            pipeline.fit(X, np.asarray(["no", "yes"]))
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(FakeCatBoost.calls[0].parameters["task_type"], "GPU")
        with self.assertRaises(ValueError):
            devices.confirm_training_device(pipeline, evidence)

    def test_evidence_rejects_backend_mismatch_unfitted_and_runtime_injection(self):
        pipeline = self._pipeline()
        _, evidence = devices.prepare_training_pipeline(pipeline, action="catboost", training_device="cpu")
        devices.validate_training_device_evidence(evidence, training_device="cpu")
        with self.assertRaises(ValueError):
            devices.validate_training_device_evidence(evidence, require_fit=True)
        for name, bad in (("backend", "GPU"), ("device_index", 0), ("exploratory", True),
                          ("device_name", "bad\nname"), ("backend_verified", True)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                devices.validate_training_device_evidence({**evidence, name: bad})
        with self.assertRaises(ValueError):
            devices.validate_training_device_evidence({**evidence, "command": "run me"})
        with self.assertRaises(ValueError):
            devices.validate_training_device_evidence(evidence, training_device="gpu")


if __name__ == "__main__":
    unittest.main()
