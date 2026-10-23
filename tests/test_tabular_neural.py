"""Optional TabM experiment: train-only preprocessing and honest nested evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.tabular_neural import (FoldPreprocessor, NeuralConfig, TabMModel, availability,
                                   run_neural_experiment)


class PreprocessingTests(unittest.TestCase):
    def test_validation_only_values_do_not_change_stats_or_vocabulary(self):
        train = pd.DataFrame({"x": [0, 2, 4, np.nan], "kind": ["a", "a", "NA", None]})
        encoder = FoldPreprocessor(["x"], ["kind"], max_categories=4).fit(train)
        before = json.dumps(encoder.to_dict(), sort_keys=True)
        number, category = encoder.transform(pd.DataFrame({"x": [1e30, None], "kind": ["unseen", None]}))
        self.assertTrue(np.isfinite(number).all())
        self.assertEqual(number.shape, (2, 2))
        self.assertEqual(number[1, 1], 1)
        self.assertEqual(category[0, 0], 0)
        self.assertNotEqual(category[1, 0], 0)
        self.assertEqual(before, json.dumps(encoder.to_dict(), sort_keys=True))
        self.assertEqual(encoder.medians_, [2])
        extreme = FoldPreprocessor(["x"], []).fit(pd.DataFrame({"x": [-1e308, 1e308]}))
        self.assertTrue(np.isfinite(extreme.transform(pd.DataFrame({"x": [1e308]}))[0]).all())
        json.dumps(extreme.to_dict(), allow_nan=False)

    def test_missing_literal_and_unknown_are_distinct(self):
        encoder = FoldPreprocessor([], ["kind"]).fit(pd.DataFrame({"kind": [None, "missing:", "NA"]}))
        _, result = encoder.transform(pd.DataFrame({"kind": [None, "missing:", "NA", "unseen"]}))
        self.assertEqual(len(set(result[:, 0])), 4)

    def test_category_order_and_cap_are_deterministic(self):
        frame = pd.DataFrame({"kind": ["a", "b", "c", "a", "b"]})
        a = FoldPreprocessor([], ["kind"], max_categories=3).fit(frame)
        b = FoldPreprocessor([], ["kind"], max_categories=3).fit(frame.iloc[::-1])
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(a.cardinalities_, [3])
        self.assertEqual(a.transform(frame)[1][2, 0], 0)

    def test_invalid_budgets_and_schema_fail(self):
        for change in ({"epochs": True}, {"k": 0}, {"learning_rate": float("nan")},
                       {"threads": 9}, {"batch_size": 100000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                NeuralConfig(**change)
        with self.assertRaises(ValueError):
            FoldPreprocessor(["x"], ["x"])
        with self.assertRaises(ValueError):
            TabMModel(numeric=["x"], categorical=[], device="gpu")

    def test_package_probe_does_not_import_torch(self):
        with patch("agent.tabular_neural.importlib.util.find_spec", return_value=None):
            state = availability()
        self.assertFalse(state["available"])
        self.assertEqual(state["missing"], ["torch", "tabm"])


def make_task(root, classification=False):
    inputs = root / "inputs"
    inputs.mkdir()
    task = root / "task.md"
    task.write_text("Primary metric: " + ("Macro F1" if classification else "RMSE"), encoding="utf-8")
    n = 160
    frame = pd.DataFrame({"id": [str(i) for i in range(n)], "x": [i % 20 for i in range(n)],
                          "period": [i // 32 + 1 for i in range(n)],
                          "kind": ["a" if i % 2 else "b" for i in range(n)],
                          "y": ["NA" if i % 2 else "class A" for i in range(n)] if classification
                          else [float(i % 20) for i in range(n)]})
    frame.to_csv(inputs / "train.csv", index=False)
    frame.iloc[:4].drop(columns="y").assign(id=["t0", "t1", "t2", "t3"]).to_csv(inputs / "test.csv", index=False)
    pd.DataFrame({"id": ["t0", "t1", "t2", "t3"], "y": ["placeholder"] * 4}).to_csv(inputs / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["id", "x", "period", "kind", "y"],
                  "role": ["id", "feature", "feature", "feature", "target"],
                  "dtype": ["numeric", "numeric", "numeric", "category", "category" if classification else "numeric"]}).to_csv(inputs / "data_dictionary.csv", index=False)
    return task


class ProtocolTests(unittest.TestCase):
    def test_outer_holdout_never_reaches_epoch_selection_and_task_submission_unchanged(self):
        fitted, predicted = [], []
        class FakeModel:
            def __init__(self, **kwargs):
                pass
            def fit(self, frame, y, **kwargs):
                fitted.append((frame.copy(), kwargs.copy()))
                self.evidence_ = {"best_epoch": 2}
                self.history_ = []
                return self
            def predict_raw(self, frame, **kwargs):
                predicted.append(frame.copy())
                return frame["x"].astype(float).to_numpy()
            def save(self, directory):
                directory.mkdir()
                return {"weights_sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = make_task(root)
            (root / "submission.csv").write_text("existing submission", encoding="utf-8")
            with patch("agent.tabular_neural._backend"), patch("agent.tabular_neural.TabMModel", FakeModel):
                report = run_neural_experiment(task, root / "run", time_budget=60,
                                               validation={"strategy": "time", "time_column": "period"})
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["metric"]["value"], 0)
            self.assertLess(fitted[0][0]["period"].astype(int).max(), 4)
            self.assertEqual(set(fitted[0][1]["tuning_frame"]["period"].astype(int)), {4})
            self.assertEqual(set(fitted[1][0]["period"].astype(int)), {1, 2, 3, 4})
            self.assertNotIn("tuning_frame", fitted[1][1])
            self.assertEqual(fitted[1][1]["fixed_epochs"], 2)
            self.assertEqual(set(predicted[0]["period"].astype(int)), {5})
            self.assertEqual((root / "submission.csv").read_text(encoding="utf-8"), "existing submission")
            self.assertEqual(report["model_artifact"]["training_scope"], "outer_fit_only")

    def test_backend_failure_leaves_honest_report_without_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("agent.tabular_neural._backend", side_effect=RuntimeError("CUDA unavailable")):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    run_neural_experiment(root / "missing.md", root / "run", device="cuda")
            report = json.loads((root / "run" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["comparison_eligible"])
            self.assertNotIn("metric", report)


@unittest.skipUnless(availability()["available"], "Optional PyTorch/TabM backend is not installed")
class TorchSmokeTests(unittest.TestCase):
    def test_real_cpu_classification_probabilities_and_repeatability(self):
        frame = pd.DataFrame({"x": np.linspace(-2, 2, 80), "cat": ["a", "b"] * 40})
        y = pd.Series(["left"] * 40 + ["right"] * 40)
        options = dict(numeric=["x"], categorical=["cat"], metric="Macro F1", classes=["left", "right"],
                       config=NeuralConfig(epochs=3, patience=3, hidden_size=16, k=2, batch_size=32, threads=1))
        models = [TabMModel(**options).fit(frame, y, fixed_epochs=3) for _ in range(2)]
        a, b = [model.predict_raw(frame.iloc[:5]) for model in models]
        np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(models[0].evidence_["device"], "cpu")
        self.assertEqual(models[0].evidence_["tuning_rows"], 0)

    def test_real_nested_regression_run_exports_loadable_cpu_weights(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = make_task(root)
            report = run_neural_experiment(task, root / "run", time_budget=60,
                validation={"strategy": "time", "time_column": "period"},
                config=NeuralConfig(epochs=3, patience=3, hidden_size=16, k=2, batch_size=32, threads=1))
            self.assertEqual(report["status"], "completed")
            self.assertTrue(np.isfinite(report["metric"]["value"]))
            weights = torch.load(root / "run" / "model" / "weights.pt", weights_only=True, map_location="cpu")
            self.assertTrue(weights)
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in weights.values()))
            self.assertEqual(report["fit"]["training_rows"], 128)
            self.assertEqual(report["fit"]["tuning_rows"], 0)
            restored = TabMModel.load(root / "run" / "model")
            test_frame = pd.read_csv(root / "inputs" / "test.csv")
            saved = pd.read_csv(root / "run" / "test_predictions.csv")
            np.testing.assert_allclose(restored.predict_raw(test_frame), saved["y"], rtol=1e-6)
            weights_path = root / "run" / "model" / "weights.pt"
            weights_path.write_bytes(weights_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                TabMModel.load(root / "run" / "model")

    def test_explicit_cuda_does_not_fallback(self):
        import torch
        from agent.tabular_neural import _backend
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                _backend("cuda")


if __name__ == "__main__":
    unittest.main()
