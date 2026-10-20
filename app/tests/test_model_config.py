"""Bounds, exact defaults and provenance for editable offline model settings."""

from copy import deepcopy
import unittest

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from app.model_config import (
    make_model_config, model_options, normalize_model_params, validate_model_config,
)
from offline_baseline import SEED, make_pipeline


class ModelConfigTests(unittest.TestCase):
    def test_defaults_preserve_existing_estimators_and_fixed_runtime_limits(self):
        old = {
            ("baseline", "classification"): make_pipeline(["x"], [], True).named_steps["model"],
            ("baseline", "regression"): make_pipeline(["x"], [], False).named_steps["model"],
            ("extra_trees", "classification"): ExtraTreesClassifier(
                n_estimators=160, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", n_jobs=8, random_state=SEED),
            ("extra_trees", "regression"): ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=2, n_jobs=8, random_state=SEED),
        }
        for (action, task_type), estimator in old.items():
            with self.subTest(action=action, task_type=task_type):
                config = make_model_config(action, task_type)
                parameters = estimator.get_params()
                self.assertEqual(config["model"], type(estimator).__name__)
                self.assertEqual(config["parameters"],
                                 {key: parameters[key] for key in config["parameters"]})
                self.assertEqual(parameters["random_state"], 42)
                if "n_jobs" in parameters:
                    self.assertEqual(parameters["n_jobs"], 8)
                self.assertNotIn("random_state", config["parameters"])
                self.assertNotIn("n_jobs", config["parameters"])

    def test_form_options_have_typed_values_and_are_independent_copies(self):
        options = model_options("classification")
        self.assertEqual(set(options), {"baseline", "extra_trees"})
        for definition in options.values():
            self.assertEqual(set(definition["defaults"]), {field["name"] for field in definition["fields"]})
            self.assertTrue(all(type(field["nullable"]) is bool for field in definition["fields"]))
        max_features = next(field for field in options["baseline"]["fields"] if field["name"] == "max_features")
        self.assertEqual([option["value"] for option in max_features["options"]], ["sqrt", "log2", 0.5, 1.0])
        options["baseline"]["defaults"]["n_estimators"] = 1
        options["baseline"]["fields"].clear()
        self.assertEqual(model_options("classification")["baseline"]["defaults"]["n_estimators"], 120)
        self.assertTrue(model_options("classification")["baseline"]["fields"])
        self.assertEqual(model_options("unsupported"), {})

    def test_partial_settings_fill_defaults_and_accept_exact_boundaries(self):
        low = normalize_model_params("baseline", "classification", {
            "n_estimators": 50, "max_depth": 2, "min_samples_leaf": 1,
            "max_features": 0.5, "class_weight": None,
        })
        self.assertEqual(low["n_estimators"], 50)
        self.assertIsNone(low["class_weight"])
        high = normalize_model_params("extra_trees", "classification", {
            "n_estimators": 600, "max_depth": 64, "min_samples_leaf": 100,
            "max_features": 1, "class_weight": "balanced",
        })
        self.assertEqual(high["max_features"], 1.0)
        self.assertIs(type(high["max_features"]), float)
        self.assertIsNone(normalize_model_params("baseline", "classification", {"max_depth": None})["max_depth"])
        self.assertEqual(normalize_model_params("baseline", "classification", {"max_depth": 10})["n_estimators"], 120)
        self.assertEqual(normalize_model_params("baseline", "regression", {
            "max_iter": 50, "max_leaf_nodes": 4, "min_samples_leaf": 1,
            "learning_rate": 0.01, "l2_regularization": 0,
        })["l2_regularization"], 0.0)
        self.assertEqual(normalize_model_params("baseline", "regression", {
            "max_iter": 600, "max_leaf_nodes": 128, "min_samples_leaf": 100,
            "learning_rate": 0.3, "l2_regularization": 10,
        })["l2_regularization"], 10.0)

    def test_unknown_types_nonfinite_values_and_runtime_controls_are_rejected(self):
        bad_tree = [
            {"n_estimators": True}, {"n_estimators": "120"}, {"n_estimators": 120.0},
            {"n_estimators": 49}, {"n_estimators": 601}, {"max_depth": 1}, {"max_depth": 65},
            {"min_samples_leaf": 0}, {"min_samples_leaf": 101}, {"max_features": True},
            {"max_features": "1.0"}, {"max_features": 0.7}, {"max_features": float("nan")},
            {"max_features": float("inf")}, {"max_features": 10**1000},
            {"class_weight": {}}, {"class_weight": False}, {"n_jobs": 100},
            {"random_state": 1}, {"criterion": "gini"},
        ]
        for value in bad_tree:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_model_params("baseline", "classification", value)
        bad_boost = [
            {"max_iter": 49}, {"max_iter": 601}, {"max_iter": False}, {"max_leaf_nodes": 3},
            {"max_leaf_nodes": 129}, {"learning_rate": 0.009}, {"learning_rate": 0.301},
            {"learning_rate": "0.1"}, {"learning_rate": float("nan")},
            {"l2_regularization": -0.1}, {"l2_regularization": 10.1},
            {"l2_regularization": float("inf")}, {"l2_regularization": 10**1000},
        ]
        for value in bad_boost:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_model_params("baseline", "regression", value)
        for action, task_type, value in (
            ("agent", "classification", {}), ("baseline", "other", {}),
            ("extra_trees", "regression", {"class_weight": "balanced"}),
            ("extra_trees", "classification", {"class_weight": "balanced_subsample"}),
            ("baseline", "classification", []),
        ):
            with self.subTest(action=action, task_type=task_type, value=value), self.assertRaises(ValueError):
                normalize_model_params(action, task_type, value)

    def test_saved_configs_require_all_fields_and_match_the_action(self):
        config = make_model_config("extra_trees", "regression", {"max_depth": 8})
        self.assertEqual(validate_model_config(config), config)
        self.assertEqual(validate_model_config(config, "extra_trees"), config)
        with self.assertRaises(ValueError):
            validate_model_config(config, "baseline")
        with self.assertRaises(ValueError):
            validate_model_config(config, "agent")
        invalid = [None, {}, {"model": "something", "parameters": {}},
                   {**config, "extra": 1}, {"model": config["model"], "parameters": {}}]
        missing = deepcopy(config)
        missing["parameters"].pop("max_depth")
        invalid.append(missing)
        unknown = deepcopy(config)
        unknown["parameters"]["n_jobs"] = 8
        invalid.append(unknown)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_model_config(value)


if __name__ == "__main__":
    unittest.main()
