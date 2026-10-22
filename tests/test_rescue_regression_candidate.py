"""Candidate wiring, leakage boundaries and executable-source evidence, no ML fits."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import regression_candidate as candidate
from agent.metrics import get_metric
from agent.regression_history import HistorySignals


def fixture(keys=100, policy="calibrated", history=True):
    rows = [{"entity": f"entity_{key:03}", "period": str(period), "x": str(key + period)}
            for period in (1, 2, 3) for key in range(keys)]
    X = pd.DataFrame(rows, index=[f"arbitrary_{i}" for i in range(len(rows))])
    y = pd.Series(np.repeat([1., 3., 9.], keys), index=X.index)
    query = pd.DataFrame({"entity": ["entity_000", "unseen", None], "period": ["4", "4", "4"], "x": ["7", "8", "9"]})
    config = {"key_columns": ["entity"], "period_column": "period", "numeric_columns": [],
              "positive_columns": [], "ambiguity_policy": "past_unique_only"}
    recipe = candidate.make_regression_recipe("period", history_config=config if history else None,
                                               history_policy=policy)
    metric = get_metric("RMSE")
    context = {"features": list(X.columns), "numeric": ["period", "x"], "categorical": ["entity"],
               "task_type": "regression", "prediction_kind": metric.prediction_kind, "metric": metric.to_dict(),
               "recipe": recipe, "deadline_monotonic": time.monotonic() + 120}
    return X, y, query, context


@contextmanager
def base_spy(*, calibration=True, positions=None):
    calls = []

    def execute(X, y, query, context, recipe):
        calls.append({"X": X.copy(deep=True), "y": np.asarray(y).copy(), "query": query.copy(deep=True),
                      "context": deepcopy(context), "recipe": deepcopy(recipe)})
        fitting = np.flatnonzero(X["period"].isin(["1", "2"]).to_numpy())
        held = np.flatnonzero(X["period"].eq("3").to_numpy())
        if positions is not None:
            fitting, held = positions
        cal = {"fit_positions": np.asarray(fitting, dtype=int), "validation_positions": np.asarray(held, dtype=int),
               "predictions": np.zeros(len(held))} if calibration else None
        return np.full(len(query), 100.), {"fresh_full_refit": True, "full_fit_rows": len(X), "api_requests": 0}, cal

    with patch.object(candidate, "fit_base", side_effect=execute):
        yield calls


@contextmanager
def history_spy():
    fits, queries = [], []

    class Recording(HistorySignals):
        def fit(self, X, y):
            fits.append((X.copy(deep=True), np.asarray(y).copy()))
            return super().fit(X, y)

        def transform(self, X):
            queries.append(X.copy(deep=True))
            return super().transform(X)

    with patch.object(candidate, "HistorySignals", Recording):
        yield fits, queries


@contextmanager
def current_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class RescueRegressionCandidateTests(unittest.TestCase):
    def test_recipe_normalizes_detaches_and_forbids_wrong_period_or_ambiguity(self):
        *_, context = fixture()
        recipe = context["recipe"]
        self.assertEqual(candidate._recipe(recipe), recipe)
        self.assertEqual(recipe["base"]["engine"], "catboost")
        self.assertEqual(recipe["minimum_calibration_history_rows"], 64)
        self.assertNotIn("iterations", recipe["base"]["parameters"])
        changed = deepcopy(recipe)
        changed["base"]["parameters"].pop("learning_rate")
        with self.assertRaises(ValueError):
            candidate._recipe(changed)
        for config in ({**recipe["history_config"], "period_column": "elsewhere"},
                       {**recipe["history_config"], "ambiguity_policy": "allow"}):
            with self.assertRaises(ValueError):
                candidate.make_regression_recipe("period", history_config=config)
        config = deepcopy(recipe["history_config"])
        result = candidate.make_regression_recipe("period", engine="lightgbm", history_config=config)
        config["key_columns"].append("mutated")
        self.assertEqual(result["history_config"]["key_columns"], ["entity"])
        self.assertEqual(result["base"]["engine"], "lightgbm")

    def test_recipe_rejects_extra_fields_and_modified_frozen_threshold(self):
        *_, context = fixture()
        recipe = context["recipe"]
        for changed in ({**recipe, "runtime_file": "anything"}, {**recipe, "minimum_calibration_history_rows": 1},
                        {**recipe, "version": True}, {**recipe, "history_policy": "unknown"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                candidate._recipe(changed)

    def test_history_schema_roles_rejected_before_base_execution(self):
        X, y, query, context = fixture()
        for config in ({**context["recipe"]["history_config"], "key_columns": ["technical_id"]},
                       {**context["recipe"]["history_config"], "numeric_columns": ["entity"]}):
            changed = deepcopy(context)
            changed["recipe"] = candidate.make_regression_recipe("period", history_config=config)
            with base_spy() as calls, self.assertRaisesRegex(ValueError, "declared feature roles"):
                candidate.execute_regression_candidate(X, y, query, changed)
            self.assertFalse(calls)

    def test_real_base_validation_rejects_undeclared_columns_and_categorical_period_without_fits(self):
        X, y, query, context = fixture()
        cases = [(X.assign(technical_id="secret"), query, context),
                 (X, query.assign(target=999), context)]
        changed = deepcopy(context)
        changed["numeric"], changed["categorical"] = ["x"], ["entity", "period"]
        cases.append((X, query, changed))
        with patch("agent.regression_base._make_estimator", side_effect=AssertionError("A fit was attempted")), \
                patch("agent.regression_base.build_pipeline", side_effect=AssertionError("A refit was attempted")):
            for fitting, prediction, ctx in cases:
                with self.subTest(columns=fitting.columns.tolist()), self.assertRaises(ValueError):
                    candidate.execute_regression_candidate(fitting, y, prediction, ctx)

    def test_one_base_refit_produces_base_mean_and_calibrated_outputs(self):
        X, y, query, context = fixture()
        with base_spy() as calls:
            result, evidence, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(result), {"predictions"})
        self.assertEqual(set(alternatives), {"base", "mean", "calibrated"})
        np.testing.assert_array_equal(alternatives["base"], [100., 100., 100.])
        self.assertAlmostEqual(alternatives["mean"][0], 50. + (1. + 3. + 9.) / 6.)
        self.assertAlmostEqual(alternatives["calibrated"][0], 25. + 9. * .75)
        np.testing.assert_array_equal(result["predictions"], alternatives["calibrated"])
        self.assertEqual(evidence["history"]["reason"], "fit_only_calibrated")
        self.assertEqual(evidence["history"]["selected"], {"mode": "last", "weight": .75, "shrinkage": 0})
        self.assertEqual(evidence["history"]["eligible_rows"], 1)
        self.assertEqual(evidence["history"]["no_history_rows"], 2)
        self.assertTrue(evidence["fresh_full_refit"])

    def test_calibration_history_receives_only_original_fit_positions_and_labels(self):
        X, y, query, context = fixture()
        # A deliberately sparse internal fit proves no other permitted fit rows
        # are silently added to the calibration history before selection.
        fitting, held = np.arange(100)[::-1], np.arange(200, 300)[::-1]
        y.iloc[held] = np.arange(len(held), dtype=float) + 9.
        with base_spy(positions=(fitting, held)), history_spy() as (fits, queries), \
                patch.object(candidate, "select_history", wraps=candidate.select_history) as select:
            candidate.execute_regression_candidate(X, y, query, context)
        np.testing.assert_array_equal(select.call_args.args[1], y.iloc[held].to_numpy())
        np.testing.assert_array_equal(select.call_args.args[2], np.zeros(len(held)))
        self.assertEqual(len(fits), 2)
        pd.testing.assert_frame_equal(fits[0][0], X.iloc[fitting])
        np.testing.assert_array_equal(fits[0][1], y.iloc[fitting].to_numpy())
        pd.testing.assert_frame_equal(queries[0], X.iloc[held])
        self.assertTrue(set(fits[0][0].index).isdisjoint(X.iloc[held].index))
        pd.testing.assert_frame_equal(fits[1][0], X)
        np.testing.assert_array_equal(fits[1][1], y.to_numpy())
        pd.testing.assert_frame_equal(queries[1], query)

    def test_minimum_calibration_history_rows_threshold_is_exact(self):
        for keys, expected in ((63, "insufficient_calibration_history_mean_fallback"), (64, "fit_only_calibrated")):
            X, y, query, context = fixture(keys=keys)
            with base_spy(), patch.object(candidate, "select_history", wraps=candidate.select_history) as select:
                _, evidence, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
            self.assertEqual(evidence["history"]["reason"], expected)
            self.assertEqual(select.call_count, int(keys >= 64))
            if keys < 64:
                np.testing.assert_array_equal(alternatives["mean"], alternatives["calibrated"])

    def test_calibration_threshold_counts_eligible_history_not_all_validation_rows(self):
        X, y, query, context = fixture()
        # Keep 100 calibration rows but remove earlier matches for 37 entities.
        mask = X["period"].isin(["1", "2"]) & X["entity"].isin([f"entity_{key:03}" for key in range(37)])
        X.loc[mask, "entity"] = X.loc[mask, "entity"].map(lambda value: "earlier_only_" + value)
        with base_spy(), patch.object(candidate, "select_history", side_effect=AssertionError("Only 63 rows have history")):
            _, evidence, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(evidence["history"]["reason"], "insufficient_calibration_history_mean_fallback")
        np.testing.assert_array_equal(alternatives["mean"], alternatives["calibrated"])

    def test_base_content_sort_cannot_move_later_targets_into_internal_training(self):
        X, y, query, context = fixture(keys=3, history=False, policy="off")
        # Exact feature ties in the early period exercise the target tie-break.
        X.loc[X["period"].eq("1"), ["entity", "x"]] = ["same", "same"]
        y.iloc[:3] = [3., 1., 2.]
        altered_y = y.copy()
        altered_y.iloc[6:] = [999., -999., 0.]
        observed = []
        def internal(frame, target, fitting, held, *_args):
            observed.append((frame.iloc[fitting].reset_index(drop=True), target.iloc[fitting].to_numpy()))
            return 50, np.zeros(len(held)), {}
        class Refit:
            def fit(self, _frame, _target, **_kwargs):
                return self
            def predict(self, frame):
                return np.zeros(len(frame))
        with patch("agent.regression_base._internal_fit", side_effect=internal), \
                patch("agent.regression_base.build_pipeline", side_effect=lambda *_args: Refit()):
            candidate.execute_regression_candidate(X, y, query, context)
            candidate.execute_regression_candidate(X, altered_y, query, context)
        pd.testing.assert_frame_equal(observed[0][0], observed[1][0])
        np.testing.assert_array_equal(observed[0][1], observed[1][1])
        self.assertTrue(observed[0][0]["period"].isin(["1", "2"]).all())

    def test_missing_calibration_uses_mean_without_selecting(self):
        X, y, query, context = fixture()
        with base_spy(calibration=False), patch.object(candidate, "select_history", side_effect=AssertionError("Unexpected tuning")):
            _, evidence, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(evidence["history"]["reason"], "no_internal_period_mean_fallback")
        np.testing.assert_array_equal(alternatives["mean"], alternatives["calibrated"])

    def test_no_schema_has_exact_base_fallback_and_never_builds_history(self):
        X, y, query, context = fixture(history=False)
        with base_spy(), patch.object(candidate, "HistorySignals", side_effect=AssertionError("Unexpected history")):
            result, evidence, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(evidence["history"]["reason"], "no_frozen_history_schema")
        for values in alternatives.values():
            np.testing.assert_array_equal(values, [100., 100., 100.])
        np.testing.assert_array_equal(result["predictions"], alternatives["base"])

    def test_mean_and_off_policies_do_not_tune(self):
        for policy, selected in (("off", "base"), ("mean", "mean")):
            X, y, query, context = fixture(policy=policy)
            with base_spy(), patch.object(candidate, "select_history", side_effect=AssertionError("Unexpected tuning")):
                result, _, alternatives = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
            np.testing.assert_array_equal(result["predictions"], alternatives[selected])

    def test_same_period_and_future_targets_never_enter_history_prediction(self):
        X, y, _, context = fixture(policy="mean")
        query = pd.DataFrame({"entity": ["entity_000"] * 5, "period": ["1", "2", "3", "4", "3"], "x": ["0"] * 5})
        future = pd.DataFrame({"entity": ["entity_000"], "period": ["9"], "x": ["99"]}, index=["future"])
        augmented = pd.concat([X, future])
        labels = pd.concat([y, pd.Series([1e9], index=future.index)])
        with base_spy():
            result = candidate.execute_regression_candidate(augmented, labels, query, context)
        np.testing.assert_allclose(result["predictions"], [100., 50.5, 51., 50. + 13. / 6., 51.])
        # Mutating current-period answers cannot change that period's history.
        changed = labels.copy()
        changed.loc[augmented["period"].eq("3")] = -1e6
        with base_spy():
            altered = candidate.execute_regression_candidate(augmented, changed, query, context)
        np.testing.assert_array_equal(result["predictions"][[0, 1, 2, 4]], altered["predictions"][[0, 1, 2, 4]])

    def test_ambiguous_prior_key_stays_at_base(self):
        X, y, query, context = fixture(policy="mean")
        duplicate = X.iloc[[0]].copy()
        duplicate.index = ["duplicate"]
        X = pd.concat([X, duplicate])
        y = pd.concat([y, pd.Series([1.], index=duplicate.index)])
        with base_spy():
            result, evidence, _ = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        np.testing.assert_array_equal(result["predictions"], [100., 100., 100.])
        self.assertEqual(evidence["history"]["ambiguous_rows"], 1)

    def test_query_composition_order_and_index_do_not_change_calibration(self):
        X, y, query, context = fixture()
        extreme = pd.DataFrame({"entity": ["query_only"], "period": ["9999"], "x": ["1e100"]})
        changed = pd.concat([query.iloc[::-1], extreme], ignore_index=True)
        with base_spy(), patch.object(candidate, "select_history", wraps=candidate.select_history) as select1:
            result, evidence, _ = candidate.execute_regression_candidate(X, y, query, context, return_evidence=True)
        with base_spy(), patch.object(candidate, "select_history", wraps=candidate.select_history) as select2:
            altered, evidence2, _ = candidate.execute_regression_candidate(X, y, changed, context, return_evidence=True)
        pd.testing.assert_frame_equal(select1.call_args.args[0], select2.call_args.args[0])
        np.testing.assert_array_equal(select1.call_args.args[1], select2.call_args.args[1])
        np.testing.assert_array_equal(select1.call_args.args[2], select2.call_args.args[2])
        self.assertEqual(evidence["history"]["selected"], evidence2["history"]["selected"])
        np.testing.assert_array_equal(result["predictions"], altered["predictions"][:3][::-1])

    def test_shared_recipe_context_and_input_frames_remain_unchanged(self):
        X, y, query, context = fixture()
        original = deepcopy((X, y, query, context))
        with base_spy():
            candidate.execute_regression_candidate(X, y, query, context)
        pd.testing.assert_frame_equal(X, original[0])
        pd.testing.assert_series_equal(y, original[1])
        pd.testing.assert_frame_equal(query, original[2])
        self.assertEqual(context, original[3])

    def test_base_receives_reduced_original_deadline_and_expiry_stops_before_history(self):
        X, y, query, context = fixture()
        context["deadline_monotonic"] = 100.
        with patch.object(candidate.time, "monotonic", return_value=0.), base_spy() as calls:
            candidate.execute_regression_candidate(X, y, query, context)
        self.assertEqual(calls[0]["context"]["deadline_monotonic"], 95.)
        self.assertEqual(context["deadline_monotonic"], 100.)
        context["deadline_monotonic"] = 1000.
        with patch.object(candidate.time, "monotonic", return_value=0.), base_spy() as calls:
            candidate.execute_regression_candidate(X, y, query, context)
        self.assertEqual(calls[0]["context"]["deadline_monotonic"], 992.)
        clock = [0.]
        def exhausted(*_args, **_kwargs):
            clock[0] = 1001.
            return np.full(len(query), 100.), {}, None
        with patch.object(candidate.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(candidate, "fit_base", side_effect=exhausted), \
                patch.object(candidate, "HistorySignals", side_effect=AssertionError("History started after expiry")), \
                self.assertRaises(TimeoutError):
            candidate.execute_regression_candidate(X, y, query, context)

    def test_public_entrypoint_returns_only_protocol_predictions(self):
        X, y, query, context = fixture()
        with base_spy():
            result = candidate.fit_predict(X, y, query, context)
        self.assertEqual(set(result), {"predictions"})
        self.assertEqual(result["predictions"].shape, (len(query),))
        self.assertTrue(np.isfinite(result["predictions"]).all())

    def test_generated_source_executes_with_stub_and_writes_verifiable_sidecars(self):
        source = candidate.candidate_source()
        namespace = {}
        exec(compile(source, "isolated_regression_candidate.py", "exec"), namespace)
        X, y, query, context = fixture()
        alternatives = {"base": np.array([5., 6., 7.]), "mean": np.array([4., 5., 6.]), "calibrated": np.array([3., 4., 5.])}
        expected = {"predictions": alternatives["calibrated"]}
        def execute(*args, **kwargs):
            self.assertTrue(kwargs["return_evidence"])
            self.assertIs(args[0], X)
            self.assertIs(args[1], y)
            self.assertIs(args[2], query)
            self.assertIs(args[3], context)
            return expected, {"status": "completed", "api_requests": 0}, alternatives
        with tempfile.TemporaryDirectory() as directory, current_directory(directory), \
                patch.dict(namespace, {"execute_regression_candidate": execute}):
            result = namespace["fit_predict"](X, y, query, context)
            self.assertEqual(set(result), {"predictions"})
            np.testing.assert_array_equal(result["predictions"], expected["predictions"])
            table = pd.read_csv("regression_ablation.csv")
            self.assertEqual(table.columns.tolist(), ["base", "mean", "calibrated"])
            self.assertEqual(len(table), len(query))
            for column, values in alternatives.items():
                np.testing.assert_array_equal(table[column], values)
            evidence = json.loads(Path("regression_evidence.json").read_text())
            self.assertEqual(evidence["ablation_sha256"], hashlib.sha256(Path("regression_ablation.csv").read_bytes()).hexdigest())
            self.assertEqual(evidence["api_requests"], 0)
            previous = Path("regression_ablation.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                namespace["fit_predict"](X, y, query, context)
            self.assertEqual(Path("regression_ablation.csv").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
