"""Fixed history recipe options and the real generated fit/predict wrapper."""

import ast
import copy
from decimal import Decimal
import hashlib
import json
import math
import unittest

import numpy as np
import pandas as pd

from agent.history_runtime import (MAX_RECIPE_BYTES, history_candidate_source,
                                   make_history_recipe, normalize_history_options)
from agent.research_candidate import _json_bytes


CONFIG = {"key_columns": ["signature"], "period_column": "period",
          "numeric_columns": ["signature"]}
BASE_SOURCE = '''import numpy as np

def fit_predict(X_fit, y_fit, X_predict, context):
    return {"predictions": np.full(len(X_predict), context["recipe"]["constant"], dtype=float)}
'''


def frame(*rows):
    return pd.DataFrame(rows, columns=["signature", "period"], dtype=object)


def context(recipe, *, phase="inner_selection"):
    return {"recipe": recipe, "features": ["signature", "period"], "task_type": "regression",
            "metric": {"name": "RMSE"}, "phase": phase}


def wrapped(source=BASE_SOURCE):
    namespace = {}
    exec(history_candidate_source(source), namespace)
    return namespace["fit_predict"], namespace


class HistoryOptionsTests(unittest.TestCase):
    def test_disabled_config_still_validates_weight(self):
        self.assertIsNone(normalize_history_options(None))
        self.assertIsNone(normalize_history_options(None, 0))
        invalid = (True, False, "0.5", None, -0.1, 1.1, math.nan, math.inf, -math.inf,
                   Decimal("0.5"), 10 ** 1000)
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "blend_weight"):
                normalize_history_options(None, value)

    def test_defaults_and_order_come_from_adapter_and_containers_are_detached(self):
        config = {"key_columns": ["site", "signature"], "period_column": "period",
                  "numeric_columns": ["signature"]}
        result = normalize_history_options(config, 1)
        self.assertEqual(result, {"history_config": {**config, "positive_columns": [],
                                                     "ambiguity_policy": "allow"}, "blend_weight": 1.0})
        config["key_columns"].append("later")
        result["history_config"]["numeric_columns"].clear()
        self.assertEqual(result["history_config"]["key_columns"], ["site", "signature"])
        self.assertEqual(config["numeric_columns"], ["signature"])
        self.assertEqual(normalize_history_options({**CONFIG, "key_columns": ("signature",)})[
            "history_config"]["key_columns"], ["signature"])

    def test_malformed_explicit_configs_are_rejected(self):
        invalid = [[], "signature", {}, {**CONFIG, "unknown": 1}, {**CONFIG, "key_columns": []},
                   {**CONFIG, "key_columns": ["signature", "signature"]},
                   {**CONFIG, "period_column": "signature"},
                   {**CONFIG, "numeric_columns": ["other"]},
                   {**CONFIG, "positive_columns": ["other"]},
                   {**CONFIG, "ambiguity_policy": "guess"},
                   {**CONFIG, 7: "bad key"}, {**CONFIG, "extra": float("nan")},
                   {**CONFIG, "key_columns": {"signature"}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_history_options(value)

    def test_recipe_preserves_components_and_detaches_nested_model_settings(self):
        base = {"components": [{"action": "fixture", "model_config": {"depth": 3},
                                 "feature_config": {"columns": ["signature"]}},
                                {"action": "second", "model_config": {"scale": 0.25}}]}
        original = copy.deepcopy(base)
        config = copy.deepcopy(CONFIG)
        recipe = make_history_recipe(base, config, 0.25)
        self.assertEqual(recipe["version"], 1)
        self.assertEqual(recipe["family"], "history_blend")
        self.assertEqual(recipe["base_recipe"], original)
        self.assertEqual(recipe["blend_weight"], 0.25)
        base["components"][0]["model_config"]["depth"] = 99
        config["numeric_columns"].clear()
        self.assertEqual(recipe["base_recipe"], original)
        self.assertEqual(recipe["history_config"]["numeric_columns"], ["signature"])
        recipe["base_recipe"]["components"][0]["feature_config"]["columns"].append("period")
        self.assertEqual(base["components"][0]["feature_config"]["columns"], ["signature"])

    def test_recipe_hash_changes_with_weight_and_matches_host_json_contract(self):
        low = make_history_recipe({"constant": 5}, CONFIG, 0.25)
        high = make_history_recipe({"constant": 5}, CONFIG, 0.75)
        self.assertNotEqual(hashlib.sha256(_json_bytes(low)).hexdigest(),
                            hashlib.sha256(_json_bytes(high)).hexdigest())
        self.assertEqual({key: value for key, value in low.items() if key != "blend_weight"},
                         {key: value for key, value in high.items() if key != "blend_weight"})
        self.assertEqual(json.loads(_json_bytes(low)), low)

    def test_recipe_rejects_non_json_nonfinite_recursive_and_disabled_settings(self):
        recursive = {}
        recursive["self"] = recursive
        invalid = [[], None, {1: "non-string key"}, {"nested": {"value": math.inf}},
                   {"nested": [math.nan]}, {"nested": object()}, recursive]
        for base in invalid:
            with self.subTest(base=base), self.assertRaises(ValueError):
                make_history_recipe(base, CONFIG)
        with self.assertRaisesRegex(ValueError, "explicit history config"):
            make_history_recipe({}, None)

    def test_recipe_limit_counts_final_utf8_bytes_including_wrapper(self):
        # The base alone fits. Adding the normalized config must exceed the
        # same byte limit that freeze_candidate applies to the final recipe.
        base = {"label": "я" * ((MAX_RECIPE_BYTES - 100) // 2)}
        self.assertLess(len(_json_bytes(base)), MAX_RECIPE_BYTES)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            make_history_recipe(base, CONFIG)
        small = make_history_recipe({"label": "я" * 100}, CONFIG)
        self.assertLess(len(_json_bytes(small)), MAX_RECIPE_BYTES)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            normalize_history_options({**CONFIG, "period_column": "п" * MAX_RECIPE_BYTES})


class HistorySourceTests(unittest.TestCase):
    def test_source_preserves_original_prefix_and_has_no_phase_or_file_logic(self):
        source = history_candidate_source(BASE_SOURCE)
        self.assertTrue(source.startswith(BASE_SOURCE))
        tree = ast.parse(source[len(BASE_SOURCE):])
        wrappers = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual([node.name for node in wrappers], ["fit_predict"])
        self.assertNotIn("phase", {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                                  and isinstance(node.value, str)})
        self.assertFalse(any(isinstance(node, (ast.For, ast.While, ast.With)) for node in ast.walk(tree)))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue({"open", "Path", "read_csv", "eval", "exec"}.isdisjoint(names))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.asname.startswith("_history_runtime_") for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertTrue(all(alias.asname.startswith("_history_runtime_") for alias in node.names))
        for invalid in (None, "", "  ", "def broken(:"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                history_candidate_source(invalid)

    def test_raw_close_numeric_keys_and_strict_past_only_history(self):
        function, _ = wrapped()
        first, second = "61242.123456789012345", "61242.123456789012346"
        X = frame((first, "1"), (second, "1"), (first, "2"), (first, "4"))
        y = pd.Series([10.0, 90.0, 30.0, 9999.0])
        query = frame((first, "1"), (first, "2"), (first, "3"), (second, "3"),
                      ("unseen", "3"), ("", "3"), (first, "not-a-period"))
        recipe = make_history_recipe({"constant": 5}, CONFIG, 1)
        result = function(X, y, query, context(recipe))
        self.assertEqual(set(result), {"predictions"})
        np.testing.assert_array_equal(result["predictions"], [5, 10, 20, 90, 5, 5, 5])

    def test_zero_and_fixed_weights_use_base_recipe_and_leave_missing_keys_exact(self):
        function, _ = wrapped()
        X, y = frame(("1", "1")), pd.Series([20.0])
        query = frame(("1", "2"), ("missing", "2"))
        for weight, expected in ((0, [8.125, 8.125]), (0.25, [11.09375, 8.125]), (1, [20, 8.125])):
            with self.subTest(weight=weight):
                recipe = make_history_recipe({"constant": 8.125}, CONFIG, weight)
                values = function(X, y, query, context(recipe))["predictions"]
                np.testing.assert_array_equal(values, expected)

    def test_base_mutations_cannot_change_raw_inputs_answers_or_outer_context(self):
        source = '''import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    assert context["recipe"] == {"constant": 7, "nested": {"value": [1]}}
    X_fit["signature"] = X_fit["signature"].astype(float)
    X_predict["signature"] = X_predict["signature"].astype(float)
    y_fit.iloc[:] = 1000000
    context["features"].clear()
    context["recipe"]["nested"]["value"].append(2)
    return {"predictions": np.full(len(X_predict), context["recipe"]["constant"])}
'''
        function, _ = wrapped(source)
        X = frame(("9007199254740992", "1"), ("9007199254740993", "1"))
        y = pd.Series([10.0, 90.0])
        query = frame(("9007199254740992", "2"), ("9007199254740993", "2"))
        invocation = context(make_history_recipe({"constant": 7, "nested": {"value": [1]}}, CONFIG, 1))
        snapshots = (X.copy(deep=True), y.copy(deep=True), query.copy(deep=True), copy.deepcopy(invocation))
        np.testing.assert_array_equal(function(X, y, query, invocation)["predictions"], [10, 90])
        pd.testing.assert_frame_equal(X, snapshots[0])
        pd.testing.assert_series_equal(y, snapshots[1])
        pd.testing.assert_frame_equal(query, snapshots[2])
        self.assertEqual(invocation, snapshots[3])

    def test_classification_is_rejected_before_calling_base(self):
        source = 'def fit_predict(*args):\n    raise AssertionError("base must not be fitted")\n'
        function, _ = wrapped(source)
        invocation = context(make_history_recipe({"constant": 0}, CONFIG))
        invocation["task_type"] = "classification"
        with self.assertRaisesRegex(ValueError, "regression only"):
            function(frame(("1", "1")), pd.Series(["label"]), frame(("1", "2")), invocation)

    def test_identical_raw_arguments_and_recipe_have_identical_output_for_every_phase(self):
        function, namespace = wrapped()
        X, y, query = frame(("1", "1")), pd.Series([20.0]), frame(("1", "2"))
        recipe = make_history_recipe({"constant": 8}, CONFIG, 0.5)
        results = [function(X, y, query, context(recipe, phase=phase))["predictions"]
                   for phase in ("inner_selection", "outer_validation", "full_refit")]
        for result in results:
            np.testing.assert_array_equal(result, [14.0])
        self.assertIs(namespace["np"], np)

    def test_existing_baseline_source_executes_all_components_with_original_configs(self):
        from agent.research_v3 import BASELINE_SOURCE

        calls = []

        class FixtureModel:
            def __init__(self, constant):
                self.constant = constant

            def fit(self, X, y):
                self.fitted_rows = len(X)

            def predict(self, X):
                return np.full(len(X), self.constant, dtype=float)

        def build(action, task_type, numeric, categorical, model_config, feature_config):
            calls.append(copy.deepcopy((action, task_type, numeric, categorical, model_config, feature_config)))
            return FixtureModel(model_config["constant"])

        base = {"components": [{"action": "first", "model_config": {"constant": 8},
                                 "feature_config": {"marker": [1]}},
                                {"action": "second", "model_config": {"constant": 12},
                                 "feature_config": {"marker": [2]}}]}
        original = copy.deepcopy(base)
        function, namespace = wrapped(BASELINE_SOURCE)
        namespace["build_pipeline"] = build
        invocation = context(make_history_recipe(base, CONFIG, 0.5))
        invocation.update(numeric=["signature", "period"], categorical=[])
        result = function(frame(("1", "1")), pd.Series([20.0]), frame(("1", "2")), invocation)
        np.testing.assert_array_equal(result["predictions"], [15.0])
        self.assertEqual([item[0] for item in calls], ["first", "second"])
        for call, component in zip(calls, original["components"]):
            self.assertEqual(call[1:4], ("regression", ["signature", "period"], []))
            self.assertEqual(call[4:], (component["model_config"], component["feature_config"]))
        self.assertEqual(invocation["recipe"]["base_recipe"], original)
        self.assertEqual(base, original)

    def test_base_programming_errors_are_not_hidden(self):
        function, _ = wrapped('def fit_predict(*args):\n    return [1]\n')
        with self.assertRaisesRegex(ValueError, "predictions dictionary"):
            function(frame(("1", "1")), pd.Series([1]), frame(("1", "2")),
                     context(make_history_recipe({}, CONFIG)))


if __name__ == "__main__":
    unittest.main()
