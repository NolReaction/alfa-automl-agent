"""Worker phase boundaries must actually avoid repeated full-data refits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.automl_worker import run_candidate
from agent.model_config import make_model_config
from agent.task_spec import load_task


class WorkerPhaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        inputs = self.directory / "inputs"
        inputs.mkdir()
        (self.directory / "task.md").write_text("Metric: MAE", encoding="utf-8")
        pd.DataFrame({"id": [f"t{i}" for i in range(40)], "x": range(40), "y": range(40)}).to_csv(inputs / "train.csv", index=False)
        pd.DataFrame({"id": ["001", "NA"], "x": [41, 42]}).to_csv(inputs / "test.csv", index=False)
        pd.DataFrame({"id": ["001", "NA"], "y": [0, 0]}).to_csv(inputs / "sample_submission.csv", index=False)
        pd.DataFrame({"feature": ["id", "x", "y"], "role": ["id", "feature", "target"],
                      "dtype": ["category", "numeric", "numeric"]}).to_csv(inputs / "data_dictionary.csv", index=False)
        self.spec = load_task(self.directory / "task.md")
        self.fits = []
        self.predictions = []
        owner = self
        class RecordingModel:
            def fit(self, X, y):
                owner.fits.append((len(X), list(y)))
                return self
            def predict(self, X):
                owner.predictions.append(len(X))
                return np.zeros(len(X))
        self.factory = RecordingModel

    def run_phase(self, phase, components=None):
        config = {"candidate_dir": str(self.directory), "action": "linear", "phase": phase,
                  "task_spec": self.spec.to_dict(), "model_config": make_model_config("linear", "regression"),
                  "fit_rows": list(range(20)), "validation_rows": list(range(20, 25))}
        if components:
            component = {"action": "linear", "model_config": config["model_config"]}
            config["components"] = [component, component]
            config["model_config"] = {"model": "FixedBlend", "parameters": {
                "weight": 0.5, "components": [component["model_config"], component["model_config"]]}}
        path = self.directory / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with patch("agent.automl_worker.build_pipeline", side_effect=lambda *args, **kwargs: self.factory()):
            run_candidate(path)

    def test_selection_and_verification_fit_only_frozen_training_subset(self):
        for phase in ("selection", "verification"):
            with self.subTest(phase=phase):
                self.fits.clear()
                self.predictions.clear()
                self.run_phase(phase)
                self.assertEqual(self.fits, [(20, list(range(20)))])
                self.assertEqual(self.predictions, [5])
                self.assertFalse((self.directory / "submission.csv").exists())
                self.assertTrue((self.directory / "holdout_predictions.csv").is_file())
                with np.load(self.directory / "holdout_raw.npz", allow_pickle=False) as bundle:
                    self.assertEqual(bundle["ids"].tolist(), [f"t{i}" for i in range(20, 25)])

    def test_seed_performs_one_validation_fit_then_one_full_refit(self):
        self.run_phase("seed")
        self.assertEqual([size for size, _ in self.fits], [20, 40])
        self.assertEqual(self.predictions, [5, 2])
        self.assertTrue((self.directory / "submission.csv").is_file())
        durations = json.loads((self.directory / "durations.json").read_text())
        self.assertGreaterEqual(durations["total_seconds"], durations["fit_seconds"] + durations["predict_seconds"])

    def test_final_refit_has_no_new_holdout_fit(self):
        self.run_phase("refit")
        self.assertEqual([size for size, _ in self.fits], [40])
        self.assertEqual(self.predictions, [2])
        self.assertFalse((self.directory / "holdout_predictions.csv").exists())

    def test_final_blend_refits_exactly_two_components_once(self):
        self.run_phase("refit", components=True)
        self.assertEqual([size for size, _ in self.fits], [40, 40])
        self.assertEqual(self.predictions, [2, 2])
        self.assertFalse((self.directory / "holdout_predictions.csv").exists())
        self.assertTrue((self.directory / "submission.csv").is_file())


if __name__ == "__main__":
    unittest.main()
