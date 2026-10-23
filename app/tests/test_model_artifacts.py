"""Fitted exports reproduce raw-input predictions and reject damaged bundles."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from agent.features import normalize_feature_config
from agent.model_config import make_model_config
from agent.models import build_pipeline
from app import model_artifacts as artifacts


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fitted(self):
        frame = pd.DataFrame({"number": [str(index) for index in range(40)],
                              "category": ["__MISSING__", "", "__ALFA_VALUE__x", "plain"] * 10})
        labels = np.asarray(["A", "B", "A", "B"] * 10)
        options = normalize_feature_config({"signed_log": True, "rare_min_count": 2,
                                            "missing_indicators": True})
        config = make_model_config("extra_trees", "classification", {"n_estimators": 50})
        pipeline = build_pipeline("extra_trees", "classification", ["number"], ["category"],
                                  model_config=config, feature_config=options).fit(frame, labels)
        recipe = {"version": 1, "action": "extra_trees", "task_type": "classification",
                  "model_config": config, "feature_config": options,
                  "validation": {"strategy": "random"}, "training_device": "cpu"}
        schema = {"features": ["number", "category"], "numeric": ["number"],
                  "categorical": ["category"], "id_column": "id", "target_column": "answer",
                  "classes": ["A", "B"]}
        arguments = {"recipe": recipe, "schema": schema, "train_rows": len(frame),
                     "split": {"train_rows": 32, "validation_rows": 8, "validation_hash": "c" * 64},
                     "dataset_fingerprint": "a" * 64, "comparison_key": "b" * 64}
        return frame, pipeline, arguments

    def test_full_pipeline_round_trip_on_raw_strings_and_reserved_categories(self):
        frame, pipeline, arguments = self.fitted()
        expected = pipeline.predict(frame)
        summary = artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(artifacts.validate_artifact_summary(summary), summary)
        path = artifacts.artifact_download_path(self.root, "model")
        restored = joblib.load(path)  # Only our own just-produced trusted artifact.
        np.testing.assert_array_equal(restored.predict(frame), expected)
        manifest = json.loads(artifacts.artifact_download_path(self.root, "manifest").read_text())
        recipe = json.loads(artifacts.artifact_download_path(self.root, "recipe").read_text())
        self.assertEqual(recipe, arguments["recipe"])
        self.assertEqual(manifest["fit"]["rows"], 40)
        self.assertEqual(manifest["fit"]["validation_split"]["train_rows"], 32)
        self.assertEqual(manifest["fit"]["validation_metric_source"], "separate_holdout_fit")
        self.assertEqual(manifest["schema"]["features"], list(frame.columns))
        self.assertIn("scikit-learn", manifest["libraries"])
        self.assertEqual(manifest["artifact"]["size_bytes"], path.stat().st_size)
        self.assertFalse(list(self.root.glob(".model-export-*")))

    def test_download_never_unpickles_and_checks_both_artifact_and_recipe(self):
        _, pipeline, arguments = self.fitted()
        artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        with patch("joblib.load", side_effect=AssertionError("Download must never execute pickle")):
            self.assertTrue(artifacts.artifact_download_path(self.root, "model").is_file())
            for kind in ("../model", "model.joblib", "unknown"):
                with self.assertRaises(ValueError):
                    artifacts.artifact_download_path(self.root, kind)
        path = self.root / artifacts.MODEL_FILENAME
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt")
        with self.assertRaises(ValueError):
            artifacts.artifact_download_path(self.root, "model")
        path.write_bytes(original)
        recipe_path = self.root / artifacts.RECIPE_FILENAME
        recipe = json.loads(recipe_path.read_text())
        recipe["model_config"]["parameters"]["n_estimators"] += 1
        recipe_path.write_text(json.dumps(recipe))
        with self.assertRaises(ValueError):
            artifacts.artifact_download_path(self.root, "recipe")

    def test_summary_cannot_supply_paths_or_claim_a_validation_model(self):
        _, pipeline, arguments = self.fitted()
        summary = artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        for change in ({"filename": "../../other.joblib"}, {"trained_on": "validation"},
                       {"validation_model_is_distinct": False}, {"size_bytes": True},
                       {"sha256": "0" * 63}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                artifacts.validate_artifact_summary({**summary, **change})
        self.assertEqual(artifacts.validate_artifact_summary({"available": False, "reason": "Not saved"}),
                         {"available": False, "reason": "Not saved"})

    def test_serialization_failure_or_disk_limit_leaves_no_partial_bundle(self):
        _, pipeline, arguments = self.fitted()

        def fail_after_writing(value, stream, **kwargs):
            stream.write(b"partial")
            raise RuntimeError("simulated serialization failure")

        with patch("joblib.dump", side_effect=fail_after_writing), self.assertRaises(RuntimeError):
            artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch.object(artifacts, "MAX_MODEL_BYTES", 8), self.assertRaises(ValueError):
            artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch.object(artifacts.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()), self.assertRaises(ValueError):
            artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_files_and_symlinks_are_not_replaced(self):
        _, pipeline, arguments = self.fitted()
        target = self.root / artifacts.MODEL_FILENAME
        target.write_bytes(b"keep")
        with self.assertRaises(ValueError):
            artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(target.read_bytes(), b"keep")
        target.unlink()
        outside = self.root / "outside.joblib"
        outside.write_bytes(b"private")
        try:
            target.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unavailable")
        with self.assertRaises(ValueError):
            artifacts.save_model_artifacts(self.root, pipeline, **arguments)
        self.assertEqual(outside.read_bytes(), b"private")


if __name__ == "__main__":
    unittest.main()
