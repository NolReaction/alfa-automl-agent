"""Frequency representation checks, independent of any audit quality result."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.frequency_candidate import (
    FrequencyContractError,
    FrequencyLimitError,
    FrequencySkipError,
    frame_fingerprint,
    fit_frequency_transform,
    frequency_candidate_source,
    frequency_feature_name,
    make_frequency_recipe,
)


CLASSES = ["NA", "class c", "01"]


def features(categories, *, index=None):
    return pd.DataFrame({"x": [str(i) for i in range(len(categories))],
                         "kind": pd.Series(categories, dtype=object).to_numpy()}, index=index)


def fit(frame, columns=("kind",)):
    return fit_frequency_transform(frame, categorical_columns=list(columns),
                                   feature_columns=list(frame.columns))


def context(recipe, *, task_type="classification", prediction_kind="probabilities"):
    metric = ("Macro F1" if prediction_kind == "labels" else "LogLoss") if task_type == "classification" else "RMSE"
    return {"version": 3, "protocol_version": 3, "phase": "inner_selection",
            "nonce": "fixed-test-nonce", "recipe": recipe,
            "code_hash": "a" * 64, "recipe_hash": "b" * 64,
            "features": ["x", "kind"], "numeric": ["x"], "categorical": ["kind"],
            "classes": list(CLASSES) if task_type == "classification" else [],
            "positive_class": None,
            "metric": {"name": metric,
                       "higher_is_better": metric == "Macro F1", "task_type": task_type,
                       "prediction_kind": prediction_kind},
            "task_type": task_type, "prediction_kind": prediction_kind}


def executable(base_source):
    namespace = {}
    source = frequency_candidate_source(base_source)
    exec(compile(source, "frequency_candidate_generated.py", "exec"), namespace)
    raw = namespace["fit_predict"]

    def invoke(*args, **kwargs):
        # Trace files are real in the bounded host. Unit invocation intercepts
        # only this write so generated source never creates a repository file.
        with patch("pathlib.Path.write_text", autospec=True, return_value=0) as writer:
            result = raw(*args, **kwargs)
        if writer.call_count != 1 or writer.call_args.args[0].name != "frequency_evidence.json":
            raise AssertionError("Successful generated source must emit one frequency trace")
        invoke.last_evidence = json.loads(writer.call_args.args[1])
        return result

    invoke.last_evidence = None
    return invoke


class FrequencyTransformTests(unittest.TestCase):
    def test_frame_fingerprint_matches_independent_canonical_typed_json(self):
        frame = pd.DataFrame({
            "text": pd.Series(["NA", "", 'лес\n"'], dtype=object),
            "number": pd.Series([1.0, -0.0, np.nan], dtype=object),
            "integer": pd.Series([7, np.int64(8), 9], dtype=object),
            "flag": pd.Series([True, np.bool_(False), None], dtype=object),
            "missing": pd.Series([pd.NA, pd.NaT, None], dtype=object),
        })
        payload = {"columns": ["text", "number", "integer", "flag", "missing"],
                   "rows": [
                       [["str", "NA"], ["float", "0x1.0000000000000p+0"], ["int", "7"],
                        ["bool", True], ["pd.NA"]],
                       [["str", ""], ["float", "-0x0.0p+0"], ["int", "8"],
                        ["bool", False], ["pd.NaT"]],
                       [["str", 'лес\n"'], ["float", "nan"], ["int", "9"],
                        ["none"], ["none"]],
                   ]}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(frame_fingerprint(frame), expected)
        self.assertEqual(frame_fingerprint(frame.set_axis([7, 7, -1], axis=0)), expected)
        self.assertNotEqual(frame_fingerprint(frame.iloc[::-1]), expected)
        self.assertNotEqual(frame_fingerprint(frame.loc[:, list(reversed(frame.columns))]), expected)

    def test_exact_literals_empty_strings_and_true_missing_are_distinct(self):
        train = features(["NA", "NA", "0007", "7", "", "", None, np.nan,
                          pd.NA, pd.NaT, " NA ", "NA "])
        query = features(["NA", "0007", "7", "", None, pd.NA, np.nan,
                          pd.NaT, "unknown", " NA ", "NA "])
        transformer = fit(train)
        result = transformer.transform(query)

        expected = [math.log1p(2), math.log1p(1), math.log1p(1), math.log1p(2),
                    math.log1p(4), math.log1p(4), math.log1p(4), math.log1p(4),
                    0.0, math.log1p(1), math.log1p(1)]
        self.assertEqual(transformer.added_columns, (frequency_feature_name("kind"),))
        np.testing.assert_allclose(result[transformer.added_columns[0]], expected, rtol=0, atol=1e-15)
        pd.testing.assert_frame_equal(result.loc[:, list(query.columns)], query)

    def test_unknown_missing_has_zero_frequency_when_missing_was_not_fitted(self):
        transformer = fit(features(["NA", "", "0007", "7"]))
        query = features([None, np.nan, pd.NA, pd.NaT, "new"])
        result = transformer.transform(query)
        np.testing.assert_array_equal(result[transformer.added_columns[0]], np.zeros(len(query)))

    def test_frequency_counts_only_fit_rows_and_never_update_on_predict(self):
        transformer = fit(features(["oak", "oak", "river"]))
        snapshot = deepcopy(transformer.to_json())
        single = transformer.transform(features(["oak", "new"]))
        repeated = transformer.transform(features(["new"] * 400 + ["oak"] * 300))
        after = transformer.transform(features(["oak", "new"]))
        name = transformer.added_columns[0]

        self.assertAlmostEqual(single[name].iloc[0], math.log1p(2))
        self.assertEqual(single[name].iloc[1], 0.0)
        self.assertTrue((repeated[name].iloc[:400] == 0.0).all())
        self.assertTrue(np.allclose(repeated[name].iloc[400:], math.log1p(2)))
        pd.testing.assert_frame_equal(single, after)
        self.assertEqual(transformer.to_json(), snapshot)

    def test_row_order_batch_order_and_duplicate_index_do_not_change_values(self):
        train = features(["a", "a", "b", ""], index=[9, 9, 4, 1])
        query = features(["b", "new", "a", "", "a"], index=[8, 8, 3, 2, 3])
        transformer = fit(train)
        together = transformer.transform(query)
        order = [4, 1, 0, 3, 2]
        shuffled = transformer.transform(query.iloc[order])
        pd.testing.assert_frame_equal(shuffled, together.iloc[order])
        for position in range(len(query)):
            pd.testing.assert_frame_equal(transformer.transform(query.iloc[[position]]),
                                          together.iloc[[position]])
        permuted_fit = fit(train.iloc[[2, 0, 3, 1]])
        pd.testing.assert_frame_equal(permuted_fit.transform(query), together)

    def test_independent_columns_use_no_pairs_or_target_information(self):
        frame = pd.DataFrame({"first": ["a", "a", "b"], "second": ["x", "y", "x"],
                              "numeric": ["1", "2", "3"]})
        transformer = fit(frame, ("first", "second"))
        query = pd.DataFrame({"first": ["a", "a"], "second": ["x", "new"], "numeric": ["900", "-9"]})
        result = transformer.transform(query)
        self.assertEqual(len(transformer.added_columns), 2)
        np.testing.assert_allclose(result[frequency_feature_name("first")], [math.log1p(2)] * 2)
        np.testing.assert_allclose(result[frequency_feature_name("second")], [math.log1p(2), 0.0])
        self.assertNotIn(frequency_feature_name("numeric"), result)

    def test_inner_outer_and_refit_build_different_dictionaries_from_their_fit(self):
        query = features(["a", "b", "c", None])
        phases = [features(["a", "b"]), features(["a", "a", "b", None]),
                  features(["a", "a", "a", "b", "c", None, None])]
        expected = [[1, 1, 0, 0], [2, 1, 0, 1], [3, 1, 1, 2]]
        outputs = []
        for frame, counts in zip(phases, expected):
            transformer = fit(frame)
            result = transformer.transform(query)
            outputs.append(result[transformer.added_columns[0]].to_numpy())
            np.testing.assert_allclose(outputs[-1], np.log1p(counts), rtol=0, atol=1e-15)
        # A later fit cannot replace state belonging to the first fitted object.
        first = fit(phases[0])
        fit(phases[-1])
        np.testing.assert_allclose(first.transform(query)[first.added_columns[0]], outputs[0])

    def test_fitted_state_and_returned_json_are_detached_from_inputs(self):
        frame = features(["a", "a", None])
        original = frame.copy(deep=True)
        transformer = fit(frame)
        pd.testing.assert_frame_equal(frame, original)
        before = deepcopy(transformer.to_json())
        detached = transformer.to_json()
        detached["caller_added"] = ["not fitted state"]
        diagnostics = transformer.diagnostics
        diagnostics["caller_added"] = True
        frame.loc[0, "kind"] = "changed"
        query = features(["a", None], index=[10, 11])
        query_snapshot = query.copy(deep=True)
        output = transformer.transform(query)
        output.iloc[0, 0] = "caller mutation"
        self.assertEqual(transformer.to_json(), before)
        self.assertNotIn("caller_added", transformer.diagnostics)
        pd.testing.assert_frame_equal(query, query_snapshot)
        self.assertEqual(original["kind"].tolist(), ["a", "a", None])
        json.dumps(before, ensure_ascii=False, allow_nan=False)
        np.testing.assert_allclose(transformer.transform(query)[transformer.added_columns[0]],
                                   [math.log1p(2), math.log1p(1)])

    def test_feature_names_are_deterministic_and_generated_collisions_fail(self):
        for column in ("kind", "NA", "0007", "a b", "лес", "kind_2"):
            expected = "__frequency_log1p_" + hashlib.sha256(column.encode("utf-8")).hexdigest()[:16]
            self.assertEqual(frequency_feature_name(column), expected)
        frame = features(["a", "b"])
        frame[frequency_feature_name("kind")] = "existing original feature"
        with self.assertRaises(FrequencyContractError):
            fit(frame)

    def test_exact_sixteen_columns_allowed_and_seventeen_refused(self):
        frame = pd.DataFrame({f"category_{index}": ["a", "b"] for index in range(16)})
        transformer = fit(frame, tuple(frame.columns))
        self.assertEqual(len(transformer.added_columns), 16)
        self.assertEqual(len(transformer.transform(frame).columns), 32)
        frame["category_16"] = "another"
        with self.assertRaises(FrequencyLimitError):
            fit(frame, tuple(frame.columns))

    def test_category_limit_counts_real_missing_as_one_distinct_value(self):
        frame = features([f"category-{index}" for index in range(4095)] + [None, np.nan, pd.NA])
        transformer = fit(frame)
        result = transformer.transform(features([None, "unseen", "category-0000", "category-0"]))
        np.testing.assert_allclose(result[transformer.added_columns[0]], [math.log1p(3), 0, 0, math.log1p(1)])
        over = pd.concat((frame, features(["one-too-many"])), ignore_index=True)
        with self.assertRaises(FrequencyLimitError):
            fit(over)

    def test_no_categories_is_explicit_skip_instead_of_redundant_fit(self):
        with self.assertRaises(FrequencySkipError):
            fit(features(["a", "b"]), ())
        with self.assertRaises(FrequencySkipError):
            make_frequency_recipe({"action": "linear"}, [])

    def test_invalid_schema_and_nonstring_categories_are_contract_errors(self):
        good = features(["a", "b"])
        cases = [
            (good, ["missing"], list(good.columns)),
            (good, ["kind", "kind"], list(good.columns)),
            (good, "kind", list(good.columns)),
            (good, ["kind"], ["kind", "x"]),
            (good, ["kind"], ["x"]),
            (good, ["kind"], ["x", "kind", "id"]),
            (good.set_axis(["kind", "kind"], axis=1), ["kind"], ["kind", "kind"]),
            (features([7, "7"]), ["kind"], ["x", "kind"]),
            (features([True, "True"]), ["kind"], ["x", "kind"]),
            (features([np.inf, "inf"]), ["kind"], ["x", "kind"]),
        ]
        for frame, categories, declared in cases:
            with self.subTest(categories=categories, declared=declared):
                with self.assertRaises(FrequencyContractError):
                    fit_frequency_transform(frame, categorical_columns=categories, feature_columns=declared)

    def test_transform_requires_exact_original_schema_order(self):
        transformer = fit(features(["a", "b"]))
        for query in (features(["a"])[["kind", "x"]], features(["a"]).drop(columns="x"),
                      features(["a"]).assign(id="forbidden extra")):
            with self.subTest(columns=query.columns.tolist()), self.assertRaises(FrequencyContractError):
                transformer.transform(query)


class FrequencyRecipeTests(unittest.TestCase):
    def test_base_recipe_is_exact_json_snapshot_including_components_and_features(self):
        base = {"label": "baseline with components", "components": [
            {"action": "linear", "model_config": {"model": "first", "parameters": {"C": 1}},
             "feature_config": {"missing_indicators": True}},
            {"action": "linear", "model_config": {"model": "second", "parameters": {"C": 2}},
             "feature_config": None}], "custom": {"exact": ["NA", "0007", "7"]}}
        snapshot = deepcopy(base)
        recipe = make_frequency_recipe(base, ["kind"])
        self.assertEqual(recipe["base_recipe"], snapshot)
        self.assertEqual(recipe["frequency"]["categorical_columns"], ["kind"])
        self.assertEqual(recipe["frequency"]["max_columns"], 16)
        self.assertEqual(recipe["frequency"]["max_categories"], 4096)
        self.assertEqual(recipe["frequency"]["missing_policy"], "genuine_missing_own_category")
        json.dumps(recipe, ensure_ascii=False, allow_nan=False)
        base["components"][0]["model_config"]["parameters"]["C"] = 999
        self.assertEqual(recipe["base_recipe"], snapshot)
        recipe["base_recipe"]["custom"]["exact"].append("caller")
        self.assertEqual(base["custom"], snapshot["custom"])

    def test_recipe_rejects_nonjson_base_and_invalid_category_declarations(self):
        for base in (None, [], {"not_json": np.array([1])}, {"bad": float("nan")}):
            with self.subTest(base=type(base).__name__), self.assertRaises(FrequencyContractError):
                make_frequency_recipe(base, ["kind"])
        for columns in ("kind", ["kind", "kind"], [7], [""], [f"c{i}" for i in range(17)]):
            with self.subTest(columns=columns), self.assertRaises(FrequencyContractError):
                make_frequency_recipe({}, columns)

    def test_builder_is_deterministic_and_invalid_python_is_refused(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    return {"predictions": [0.0] * len(X_predict)}\n'
        self.assertEqual(frequency_candidate_source(source), frequency_candidate_source(source))
        with self.assertRaises(FrequencyContractError):
            frequency_candidate_source("def broken(:\n")


class FrequencySourceTests(unittest.TestCase):
    def test_generated_trace_hashes_exact_crlf_baseline_source_bytes(self):
        source = ('def fit_predict(X_fit, y_fit, X_predict, context):\r\n'
                  '    return {"predictions": ["01"] * len(X_predict)}\r\n')
        function = executable(source)
        ctx = context(make_frequency_recipe({"fixed": True}, ["kind"]), prediction_kind="labels")
        function(features(["a", "b"]), pd.Series(["01", "NA"]), features(["a"]), ctx)
        self.assertEqual(function.last_evidence["base_source_hash"],
                         hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.assertNotEqual(function.last_evidence["base_source_hash"],
                            hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest())

    def test_custom_baseline_output_and_noncanonical_probability_order_are_preserved(self):
        source = '''def fit_predict(X_fit, y_fit, X_predict, context):
    assert context["recipe"] == {"keep": ["exact", 7]}
    return {"predictions": [[0.8, 0.2] for _ in range(len(X_predict))], "classes": ["NA", "01"]}
'''
        function = executable(source)
        recipe = make_frequency_recipe({"keep": ["exact", 7]}, ["kind"])
        ctx = context(recipe)
        ctx["classes"] = ["01", "NA"]
        result = function(features(["a", "a", "b"]), pd.Series(["01", "NA", "01"]),
                          features(["new", "a"]), ctx)
        self.assertEqual(result, {"predictions": [[0.8, 0.2], [0.8, 0.2]], "classes": ["NA", "01"]})

    def test_actual_baseline_source_components_receive_updated_numeric_metadata_and_unchanged_recipe(self):
        from agent.research_v3 import BASELINE_SOURCE

        calls, fits = [], []
        base = {"components": [
            {"action": "first", "model_config": {"marker": 1}, "feature_config": {"missing_indicators": True}},
            {"action": "second", "model_config": {"marker": 2}, "feature_config": None}]}
        recipe = make_frequency_recipe(base, ["kind"])
        ctx = context(recipe)
        ctx_snapshot = deepcopy(ctx)
        train = features(["NA", "NA", "other"])
        query = features(["NA", "unseen"])
        y = pd.Series(["01", "NA", "class c"])
        added = frequency_feature_name("kind")

        class ModelSpy:
            classes_ = np.asarray(["class c", "01", "NA"])

            def __init__(self, marker):
                self.marker = marker

            def fit(self, frame, labels):
                fits.append((self.marker, frame.copy(deep=True), np.asarray(labels).copy()))
                return self

            def predict_proba(self, frame):
                np.testing.assert_allclose(frame[added], [math.log1p(2), 0.0])
                row = [0.1, 0.2, 0.7] if self.marker == 1 else [0.3, 0.4, 0.3]
                return np.tile(row, (len(frame), 1))

        def build(action, task_type, numeric, categorical, model_config, feature_config):
            calls.append((action, task_type, list(numeric), list(categorical),
                          deepcopy(model_config), deepcopy(feature_config)))
            return ModelSpy(model_config["marker"])

        with patch("agent.models.build_pipeline", side_effect=build):
            function = executable(BASELINE_SOURCE)
            result = function(train, y, query, ctx)

        self.assertEqual([row[0] for row in calls], ["first", "second"])
        for record, component in zip(calls, base["components"]):
            self.assertEqual(record[1:4], ("classification", ["x", added], ["kind"]))
            self.assertEqual(record[4], component["model_config"])
            self.assertEqual(record[5], component["feature_config"])
        for marker, fitted, actual in fits:
            self.assertEqual(fitted.columns.tolist(), ["x", "kind", added])
            np.testing.assert_allclose(fitted[added], [math.log1p(2), math.log1p(2), math.log1p(1)])
            np.testing.assert_array_equal(actual, y.to_numpy())
        self.assertEqual(result["classes"], CLASSES)
        # Each model emits c/01/NA. BASELINE_SOURCE aligns before averaging.
        np.testing.assert_allclose(result["predictions"], [[0.5, 0.2, 0.3]] * 2, rtol=0, atol=1e-15)
        self.assertEqual(ctx, ctx_snapshot)
        self.assertEqual(ctx["recipe"]["base_recipe"], base)

    def test_label_and_regression_prediction_shapes_pass_through(self):
        for task_type, kind, values in (("classification", "labels", ["NA", "01"]),
                                        ("regression", "regression", [-1.5, 9.25])):
            with self.subTest(task_type=task_type):
                source = ("def fit_predict(X_fit, y_fit, X_predict, context):\n"
                          + f"    return {{'predictions': {values!r}}}\n")
                ctx = context(make_frequency_recipe({"fixed": True}, ["kind"]),
                              task_type=task_type, prediction_kind=kind)
                result = executable(source)(features(["a", "b"]), np.asarray([1, 2]),
                                            features(["a", "new"]), ctx)
                self.assertEqual(result, {"predictions": values})

    def test_each_source_call_refits_counts_even_with_same_function_object(self):
        source = '''def fit_predict(X_fit, y_fit, X_predict, context):
    name = context["numeric"][-1]
    return {"predictions": X_predict[name].to_numpy()}
'''
        function = executable(source)
        ctx = context(make_frequency_recipe({"fixed": True}, ["kind"]),
                      task_type="regression", prediction_kind="regression")
        query = features(["a", "b", "new"])
        first = function(features(["a", "b"]), np.array([0., 1.]), query, ctx)
        second = function(features(["a", "a", "a", "b"]), np.arange(4.), query, ctx)
        third = function(features(["a", "b"]), np.array([999., -999.]), query, ctx)
        np.testing.assert_allclose(first["predictions"], [math.log1p(1), math.log1p(1), 0])
        np.testing.assert_allclose(second["predictions"], [math.log1p(3), math.log1p(1), 0])
        np.testing.assert_array_equal(first["predictions"], third["predictions"])

    def test_generated_trace_pins_fit_only_transform_and_host_identity(self):
        from agent.frequency_candidate import frame_fingerprint

        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    return {"predictions": ["01"] * len(X_predict)}\n'
        function = executable(source)
        train, query = features(["a", "a", None]), features(["a", "new"])
        base = {"components": [{"fixed": ["do not rewrite", 1]}]}
        ctx = context(make_frequency_recipe(base, ["kind"]), prediction_kind="labels")
        function(train, pd.Series(["01", "NA", "class c"]), query, ctx)
        evidence = function.last_evidence
        transformer = fit(train)
        self.assertEqual(evidence["base_source_hash"], hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.assertEqual(evidence["transform"], transformer.to_json())
        self.assertEqual(evidence["phase"], ctx["phase"])
        self.assertEqual(evidence["nonce"], ctx["nonce"])
        self.assertEqual(evidence["code_hash"], ctx["code_hash"])
        self.assertEqual(evidence["recipe_hash"], ctx["recipe_hash"])
        self.assertEqual(evidence["fit_rows"], len(train))
        self.assertEqual(evidence["predict_rows"], len(query))
        self.assertEqual(evidence["features"], ["x", "kind", *transformer.added_columns])
        self.assertEqual(evidence["numeric"], ["x", *transformer.added_columns])
        self.assertEqual(evidence["categorical"], ["kind"])
        self.assertEqual(evidence["augmented_fit_sha256"], frame_fingerprint(transformer.transform(train)))
        self.assertEqual(evidence["augmented_predict_sha256"], frame_fingerprint(transformer.transform(query)))
        changed_batch = features(["new"] * 100 + ["a"])
        function(train, pd.Series(["01", "NA", "class c"]), changed_batch, ctx)
        self.assertEqual(function.last_evidence["transform"], evidence["transform"])
        self.assertNotEqual(function.last_evidence["augmented_predict_sha256"], evidence["augmented_predict_sha256"])

    def test_baseline_mutation_of_its_local_arguments_does_not_mutate_callers(self):
        source = '''def fit_predict(X_fit, y_fit, X_predict, context):
    X_fit.iloc[0, 0] = "mutated fit"
    X_predict.iloc[0, 0] = "mutated predict"
    y_fit.iloc[0] = "mutated target"
    context["recipe"]["nested"]["x"].append(99)
    context["features"].append("mutated metadata")
    return {"predictions": ["01"] * len(X_predict)}
'''
        train, query = features(["a", "a", "b"]), features(["a", "new"])
        y = pd.Series(["01", "NA", "class c"])
        ctx = context(make_frequency_recipe({"nested": {"x": [1]}}, ["kind"]), prediction_kind="labels")
        snapshots = train.copy(deep=True), query.copy(deep=True), y.copy(deep=True), deepcopy(ctx)
        executable(source)(train, y, query, ctx)
        pd.testing.assert_frame_equal(train, snapshots[0])
        pd.testing.assert_frame_equal(query, snapshots[1])
        pd.testing.assert_series_equal(y, snapshots[2])
        self.assertEqual(ctx, snapshots[3])

    def test_numeric_or_technical_columns_cannot_be_selected_outside_declared_categories(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    raise AssertionError("must refuse before baseline fit")\n'
        function = executable(source)
        for columns in (["x"], ["id"], ["target"]):
            with self.subTest(columns=columns):
                ctx = context(make_frequency_recipe({}, columns))
                with self.assertRaises(FrequencyContractError):
                    function(features(["a", "b"]), np.asarray(["01", "NA"]), features(["a"]), ctx)

    def test_source_limit_failure_never_calls_baseline(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    raise AssertionError("limit must refuse before any model fit")\n'
        function = executable(source)
        train = features([f"category-{index}" for index in range(4097)])
        ctx = context(make_frequency_recipe({}, ["kind"]))
        with self.assertRaises(FrequencyLimitError):
            function(train, np.repeat("01", len(train)), features(["new"]), ctx)

    def test_stored_wrapper_recipe_tampering_is_rejected_before_baseline(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    raise AssertionError("invalid recipe reached baseline")\n'
        function = executable(source)
        original = make_frequency_recipe({"keep": {"exact": True}}, ["kind"])
        corrupted = []
        for key, value in (("version", True), ("version", 2), ("family", "different"),
                           ("base_recipe", []), ("unknown_field", "extra")):
            recipe = deepcopy(original)
            recipe[key] = value
            corrupted.append(recipe)
        for key, value in (("max_columns", 17), ("max_categories", 4097),
                           ("max_categories", True), ("missing_policy", "stringify_missing"),
                           ("feature_name_prefix", "different_"), ("new_option", 1)):
            recipe = deepcopy(original)
            recipe["frequency"][key] = value
            corrupted.append(recipe)
        for recipe in corrupted:
            with self.subTest(recipe=recipe), self.assertRaises(FrequencyContractError):
                function(features(["a", "b"]), np.asarray(["01", "NA"]),
                         features(["a"]), context(recipe))
        absent = deepcopy(original)
        absent["frequency"]["categorical_columns"] = []
        with self.assertRaises(FrequencySkipError):
            function(features(["a", "b"]), np.asarray(["01", "NA"]),
                     features(["a"]), context(absent))

    def test_host_schema_metadata_must_be_an_exact_partition(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    raise AssertionError("invalid schema reached baseline")\n'
        function = executable(source)
        ctx = context(make_frequency_recipe({}, ["kind"]))
        mutations = [{"features": ["kind", "x"]}, {"features": ["x", "kind", "id"]},
                     {"numeric": ["x", "kind"]}, {"numeric": ["x", "x"]},
                     {"numeric": []}, {"categorical": ["kind", "x"]}]
        for mutation in mutations:
            altered = {**deepcopy(ctx), **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(FrequencyContractError):
                function(features(["a", "b"]), np.asarray(["01", "NA"]), features(["a"]), altered)

    def test_trace_write_failure_propagates_instead_of_returning_success(self):
        source = 'def fit_predict(X_fit, y_fit, X_predict, context):\n    return {"predictions": ["01"] * len(X_predict)}\n'
        namespace = {}
        exec(compile(frequency_candidate_source(source), "frequency_generated.py", "exec"), namespace)
        ctx = context(make_frequency_recipe({}, ["kind"]), prediction_kind="labels")
        with patch("pathlib.Path.write_text", side_effect=OSError("trace unavailable")), \
                self.assertRaisesRegex(OSError, "trace unavailable"):
            namespace["fit_predict"](features(["a", "b"]), np.asarray(["01", "NA"]), features(["a"]), ctx)


class FrequencyNativeCandidateTests(unittest.TestCase):
    def test_real_baseline_components_pass_native_host_and_emit_bound_refit_trace(self):
        """One tiny real subprocess fit proves integration, not experiment quality."""
        from agent.metrics import get_metric
        from agent.model_config import make_model_config
        from agent.research_candidate import freeze_candidate, run_candidate
        from agent.research_v3 import BASELINE_SOURCE
        from agent.task_spec import TaskSpec

        base = {"components": [
            {"action": "linear", "model_config": make_model_config(
                "linear", "classification", {"C": 0.5, "max_iter": 100}),
             "feature_config": None},
            {"action": "linear", "model_config": make_model_config(
                "linear", "classification", {"C": 1.0, "max_iter": 100}),
             "feature_config": None},
        ]}
        recipe = make_frequency_recipe(base, ["kind"])
        fit_frame = pd.DataFrame({
            "row_key": [f"fit-{index:03d}" for index in range(30)],
            "x": [str(index % 7) for index in range(30)],
            "kind": ["NA", "", "0007", "7", "oak"] * 6,
            "answer": [CLASSES[index % 3] for index in range(30)],
        })
        predict = pd.DataFrame({"row_key": ["0005", "NA"], "x": ["9", "10"],
                                "kind": ["NA", "unseen"]})
        snapshots = fit_frame.copy(deep=True), predict.copy(deep=True), deepcopy(recipe)
        with tempfile.TemporaryDirectory(prefix="frequency-native-test-") as directory:
            root = Path(directory)
            source = frequency_candidate_source(BASELINE_SOURCE)
            proposal = root / "proposal.py"
            proposal.write_bytes(source.encode("utf-8"))
            candidate = freeze_candidate(proposal, recipe, root / "frozen")
            output_classes = ("01", "class c", "NA")
            spec = TaskSpec(root / "task.md", root / "data", "row_key", "answer",
                            ("x", "kind"), ("x",), ("kind",), tuple(CLASSES),
                            ("row_key", *output_classes),
                            tuple((label, label) for label in output_classes), None, get_metric("LogLoss"))
            result = run_candidate(candidate, spec=spec, fit_frame=fit_frame, predict_frame=predict,
                                   output_dir=root / "executions", phase="full_refit", timeout=30)
            result.verify()
            candidate.verify()
            self.assertEqual(result.execution["exit_code"], 0)
            self.assertFalse(result.execution.get("timed_out"))
            self.assertFalse(result.execution.get("memory_limit_exceeded"))
            self.assertIsNone(result.metric)
            output = pd.read_csv(result.output_path, dtype=str, keep_default_na=False)
            self.assertEqual(output.columns.tolist(), list(spec.output_columns))
            self.assertEqual(output["row_key"].tolist(), ["0005", "NA"])
            probabilities = output.loc[:, list(output_classes)].astype(float).to_numpy()
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0], rtol=0, atol=1e-12)
            invocation = result.output_path.parent
            native_context = json.loads((invocation / "context.json").read_text(encoding="utf-8"))
            receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
            evidence = json.loads((invocation / "frequency_evidence.json").read_text(encoding="utf-8"))
            for name in ("phase", "nonce", "code_hash", "recipe_hash"):
                self.assertEqual(evidence[name], native_context[name])
                self.assertEqual(evidence[name], receipt[name])
            self.assertEqual(evidence["phase"], "full_refit")
            self.assertEqual(evidence["code_hash"], candidate.code_hash)
            self.assertEqual(evidence["recipe_hash"], candidate.recipe_hash)
            self.assertEqual(evidence["base_source_hash"], hashlib.sha256(BASELINE_SOURCE.encode("utf-8")).hexdigest())
            base_bytes = json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")
            self.assertEqual(evidence["base_recipe_hash"], hashlib.sha256(base_bytes).hexdigest())
            self.assertEqual(candidate.recipe["base_recipe"], base)
            self.assertEqual(evidence["fit_rows"], 30)
            self.assertEqual(evidence["predict_rows"], 2)
            transformer = fit(fit_frame.loc[:, ["x", "kind"]])
            self.assertEqual(evidence["transform"], transformer.to_json())
            self.assertEqual(evidence["augmented_fit_sha256"],
                             frame_fingerprint(transformer.transform(fit_frame.loc[:, ["x", "kind"]])))
            self.assertEqual(evidence["augmented_predict_sha256"],
                             frame_fingerprint(transformer.transform(predict.loc[:, ["x", "kind"]])))
            self.assertEqual(proposal.read_bytes(), source.encode("utf-8"))
        pd.testing.assert_frame_equal(fit_frame, snapshots[0])
        pd.testing.assert_frame_equal(predict, snapshots[1])
        self.assertEqual(recipe, snapshots[2])


if __name__ == "__main__":
    unittest.main()
