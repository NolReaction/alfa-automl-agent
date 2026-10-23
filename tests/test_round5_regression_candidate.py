"""Risks introduced by mixing staged bases; no physical models fit here."""
from copy import deepcopy
from types import SimpleNamespace
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import round5_regression_base as staged_base
from agent import round5_regression_candidate as candidate
from agent.metrics import get_metric
from agent.regression_candidate import make_regression_recipe, _recipe as validate_incumbent
from agent.regression_history import HistorySignals
from tests.test_rescue_regression_base import fixture as base_fixture, spies


def fixture(arm="N25", *, history=True, keys=64):
    rows = [{"entity": f"entity_{key:03}", "period": str(period), "value": str(key + period)}
            for period in (1, 2, 3) for key in range(keys)]
    frame = pd.DataFrame(rows, index=[1000 + i * 3 for i in range(len(rows))])
    target = pd.Series([float(int(row["period"]) * 10) for row in rows], index=frame.index)
    query = pd.DataFrame({"entity": ["entity_000"] * 3 + ["unseen"],
                          "period": ["2", "3", "4", "4"], "value": ["9", "8", "7", "6"]})
    schema = {"key_columns": ["entity"], "period_column": "period", "numeric_columns": [],
              "positive_columns": [], "ambiguity_policy": "past_unique_only"}
    recipe = candidate.make_round5_recipe("period", arm, history_config=schema if history else None)
    metric = get_metric("RMSE")
    context = {"features": list(frame), "numeric": ["period", "value"], "categorical": ["entity"],
               "task_type": "regression", "metric": metric.to_dict(),
               "prediction_kind": metric.prediction_kind, "recipe": recipe,
               "deadline_monotonic": time.monotonic() + 120}
    return frame, target, query, context


def fake_stage(frame, target, query, context, component, *, quota_seconds, **_budgets):
    fitting = np.flatnonzero(frame["period"].isin(["1", "2"]).to_numpy())
    held = np.flatnonzero(frame["period"].eq("3").to_numpy())
    # Different component order is deliberate: sorting on a different feature
    # subset does not preserve the position order, only the exact row sets.
    if component["feature_subset"] == "declared_numeric":
        fitting, held = fitting[::-1], held[::-1]
    offset = 30. if component["feature_subset"] == "declared_numeric" else 10.
    calibration = {"fit_positions": fitting, "validation_positions": held,
                   "predictions": held.astype(float) + offset}
    return SimpleNamespace(component=deepcopy(component), X=frame.copy(), y=target.copy(),
        predict=query.copy(), context=deepcopy(context), calibration=calibration,
        quota_seconds=quota_seconds, consumed_seconds=0.,
        evidence={"physical_fits_expected_completed": 1})


def fake_refit(stage, **_budgets):
    offset = 30. if stage.component["feature_subset"] == "declared_numeric" else 10.
    return np.full(len(stage.predict), offset), {"physical_fits_expected_completed": 2}


class Round5RegressionContractTests(unittest.TestCase):
    def test_frozen_family_rejects_recipe_changes_and_preserves_incumbent(self):
        old = make_regression_recipe("period")
        self.assertEqual(validate_incumbent(old), old)
        with self.assertRaises(ValueError):
            candidate._recipe(old)
        with self.assertRaises(ValueError):
            candidate.make_round5_recipe("period", "I")
        recipe = candidate.make_round5_recipe("period", "N25")
        self.assertEqual(candidate._recipe(json.loads(json.dumps(recipe))), recipe)
        copied = candidate._recipe(recipe)
        copied["components"][0]["base"]["parameters"]["depth"] = 1
        self.assertEqual(recipe["components"][0]["base"]["parameters"]["depth"], 6)
        variants = []
        for key, value in (("version", True), ("runtime_path", "private"), ("arm", "E25")):
            altered = deepcopy(recipe); altered[key] = value; variants.append(altered)
        altered = deepcopy(recipe); altered["components"][0]["weight"] = .5; variants.append(altered)
        altered = deepcopy(recipe); altered["components"][1]["base"]["parameters"]["depth"] = 7; variants.append(altered)
        altered = deepcopy(recipe); altered["calibration"]["tree_alignment"] = "refit_count"; variants.append(altered)
        altered = deepcopy(recipe); altered["resources"]["component_time_shares"] = [.9, .1]; variants.append(altered)
        for altered in variants:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                candidate._recipe(altered)

    def test_alignment_uses_position_identity_not_component_order(self):
        first = {"fit_positions": [3, 1], "validation_positions": [4, 2], "predictions": [40., 20.]}
        second = {"fit_positions": [1, 3], "validation_positions": [2, 4], "predictions": [200., 400.]}
        original = deepcopy((first, second))
        actual = candidate.align_calibrations([first, second], [.75, .25])
        np.testing.assert_array_equal(actual["fit_positions"], [1, 3])
        np.testing.assert_array_equal(actual["validation_positions"], [2, 4])
        np.testing.assert_array_equal(actual["predictions"], [65., 130.])
        self.assertEqual((first, second), original)
        for changed in (None, {**second, "validation_positions": [2, 5]},
                        {**second, "fit_positions": [1, 2]},
                        {**second, "validation_positions": [2, 2]},
                        {**second, "predictions": [float("nan"), 400.]},
                        {**second, "validation_positions": [2., 4.]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                candidate.align_calibrations([first, changed], [.75, .25])
        self.assertIsNone(candidate.align_calibrations([None, None], [.75, .25]))

    def test_numeric_projection_is_copied_but_history_uses_original_raw_rows(self):
        X, y, query, context = fixture()
        originals = deepcopy((X, y, query, context))
        seen, history_fits, events = [], [], []

        def stage(frame, target, predict, ctx, component, **budgets):
            seen.append((frame.copy(), predict.copy(), deepcopy(ctx), deepcopy(budgets)))
            events.append("stage:" + component["id"])
            result = fake_stage(frame, target, predict, ctx, component, **budgets)
            # Mutation of the component-owned copy must not affect the source or
            # the full-feature frame later supplied to HistorySignals.
            frame.iloc[0, 0] = "component-local-change"
            ctx["features"].append("component-local-name")
            return result

        class History(HistorySignals):
            def fit(self, frame, target):
                history_fits.append((frame.copy(), np.asarray(target).copy()))
                return super().fit(frame, target)

        def select(signals, labels, predictions):
            events.append("history_selection")
            held = np.flatnonzero(X["period"].eq("3").to_numpy())
            np.testing.assert_array_equal(labels, y.iloc[held])
            np.testing.assert_array_equal(predictions, held + 15.)
            return {"mode": "mean", "weight": .5, "shrinkage": 0}

        def refit(stage, **budgets):
            events.append("refit:" + stage.component["id"])
            return fake_refit(stage, **budgets)

        with patch.object(candidate, "stage_component", side_effect=stage), \
                patch.object(candidate, "refit_component", side_effect=refit), \
                patch.object(candidate, "HistorySignals", History), \
                patch.object(candidate, "select_history", side_effect=select):
            result, evidence, _ = candidate.execute_round5_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual(events, ["stage:catboost_all", "stage:catboost_numeric", "history_selection",
                                  "refit:catboost_all", "refit:catboost_numeric"])
        self.assertEqual(seen[0][0].columns.tolist(), list(X))
        self.assertEqual(seen[1][0].columns.tolist(), ["period", "value"])
        self.assertEqual(seen[1][2]["categorical"], [])
        self.assertEqual(seen[1][2]["features"], seen[1][2]["numeric"])
        earlier = np.flatnonzero(X["period"].isin(["1", "2"]).to_numpy())
        pd.testing.assert_frame_equal(history_fits[0][0], X.iloc[earlier])
        np.testing.assert_array_equal(history_fits[0][1], y.iloc[earlier])
        pd.testing.assert_frame_equal(history_fits[1][0], X)
        # t=2 sees period1 only, t=3 sees periods1+2, and the absent entity has no
        # history. Same-period and future labels never enter these history terms.
        np.testing.assert_array_equal(result["predictions"], [12.5, 15., 17.5, 15.])
        self.assertEqual(evidence["history"]["reason"], "fit_only_calibrated")
        pd.testing.assert_frame_equal(X, originals[0])
        pd.testing.assert_series_equal(y, originals[1])
        pd.testing.assert_frame_equal(query, originals[2])
        self.assertEqual(context, originals[3])

    def test_no_schema_stays_base_without_inventing_all_cold_or_tuning(self):
        X, y, query, context = fixture(history=False)
        with patch.object(candidate, "stage_component", side_effect=fake_stage), \
                patch.object(candidate, "refit_component", side_effect=fake_refit), \
                patch.object(candidate, "HistorySignals", side_effect=AssertionError("No schema")), \
                patch.object(candidate, "select_history", side_effect=AssertionError("No schema")):
            result, evidence, alternatives = candidate.execute_round5_candidate(X, y, query, context, return_evidence=True)
        np.testing.assert_array_equal(result["predictions"], np.full(len(query), 15.))
        for name in ("base", "mean", "calibrated"):
            np.testing.assert_array_equal(alternatives[name], result["predictions"])
        self.assertIsNone(evidence["history"]["no_history_rows"])
        self.assertFalse(evidence["history"]["schema_available"])

    def test_component_failure_and_misalignment_do_not_publish_partial_mixture(self):
        X, y, query, context = fixture(history=False)
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(args[4]["id"])
            if len(calls) == 2:
                raise TimeoutError("component quota exhausted")
            return fake_stage(*args, **kwargs)

        with patch.object(candidate, "stage_component", side_effect=fail_second), \
                patch.object(candidate, "refit_component") as refit, self.assertRaises(TimeoutError):
            candidate.fit_predict(X, y, query, context)
        refit.assert_not_called()
        self.assertEqual(calls, ["catboost_all", "catboost_numeric"])

        def mismatched(*args, **kwargs):
            stage = fake_stage(*args, **kwargs)
            if args[4]["feature_subset"] == "declared_numeric":
                stage.calibration["validation_positions"][0] += 100
            return stage

        with patch.object(candidate, "stage_component", side_effect=mismatched), \
                patch.object(candidate, "refit_component") as refit, self.assertRaises(ValueError):
            candidate.fit_predict(X, y, query, context)
        refit.assert_not_called()

    def test_one_absolute_budget_and_fixed_nontransferable_component_shares(self):
        X, y, query, context = fixture(history=False)
        context["deadline_monotonic"] = 100.
        seen = []

        def stage(*args, **kwargs):
            seen.append(deepcopy(kwargs))
            return fake_stage(*args, **kwargs)

        with patch.object(candidate.time, "monotonic", return_value=0.), \
                patch.object(candidate, "stage_component", side_effect=stage), \
                patch.object(candidate, "refit_component", side_effect=fake_refit):
            _, evidence, _ = candidate.execute_round5_candidate(X, y, query, context, return_evidence=True)
        self.assertEqual([row["quota_seconds"] for row in seen], [47.5, 47.5])
        self.assertEqual([row["base_deadline"] for row in seen], [95., 95.])
        self.assertTrue(all(row["original_deadline"] == 100. for row in seen))
        self.assertEqual(evidence["history_reserve_seconds"], 5.)
        context["deadline_monotonic"] = time.monotonic() - 1
        with patch.object(candidate, "stage_component") as stage, self.assertRaises(TimeoutError):
            candidate.fit_predict(X, y, query, context)
        stage.assert_not_called()

    def test_staging_fits_only_earlier_preprocessing_then_fresh_all_row_models(self):
        X, y, query, context, _ = base_fixture()
        recipe = candidate.make_round5_recipe("period", "E50")
        deadline = context["deadline_monotonic"]
        with spies(best=10) as calls:
            stages = [staged_base.stage_component(X, y, query, context, component,
                quota_seconds=50., base_deadline=deadline - 5, original_deadline=deadline)
                for component in recipe["components"]]
            self.assertEqual(len(calls["early"]), 2)
            self.assertEqual(calls["refit"], [])
            aligned = candidate.align_calibrations([stage.calibration for stage in stages], [.5, .5])
            self.assertEqual(set(aligned["fit_positions"]), {1, 2, 4, 5})
            self.assertEqual(set(aligned["validation_positions"]), {0, 3})
            for fitted in calls["feature_fit"]:
                self.assertFalse(fitted["kind"].eq("held_only").any())
                self.assertFalse(fitted["kind"].eq("query_only").any())
            self.assertNotIn("held_only", calls["category_fit"][0]["kind"])
            self.assertNotIn("query_only", calls["category_fit"][0]["kind"])
            for stage in stages:
                self.assertFalse(any(name in vars(stage) for name in ("model", "transformer", "estimator")))
                staged_base.refit_component(stage, base_deadline=deadline - 5, original_deadline=deadline)
            self.assertEqual(len(calls["refit"]), 2)
            self.assertIsNot(calls["refit"][0]["model"], calls["refit"][1]["model"])
            for fitted in calls["refit"]:
                self.assertEqual(len(fitted["model"].train), len(X))
                np.testing.assert_array_equal(np.sort(fitted["model"].target), np.sort(y))
        # Legacy alignment is intentional: best 11/10, final counts50; no hidden
        # calibration recheck was added and the mismatch remains observable.
        self.assertEqual([stage.evidence["calibration_iterations"] for stage in stages], [11, 10])
        self.assertEqual([stage.selected_iterations for stage in stages], [50, 50])

    def test_refit_cannot_reset_consumed_quota_or_original_deadline(self):
        X, y, query, context, _ = base_fixture()
        component = candidate.make_round5_recipe("period", "LGB")["components"][0]
        deadline = context["deadline_monotonic"]
        with spies() as calls:
            stage = staged_base.stage_component(X, y, query, context, component,
                quota_seconds=50., base_deadline=deadline - 5, original_deadline=deadline)
            stage.consumed_seconds = stage.quota_seconds
            with self.assertRaises(TimeoutError):
                staged_base.refit_component(stage, base_deadline=deadline - 5, original_deadline=deadline)
            self.assertEqual(calls["refit"], [])
        with patch.object(staged_base.time, "monotonic", return_value=3.):
            self.assertEqual(staged_base._phase_deadline(100., 90., 7.), 10.)
            self.assertEqual(staged_base._phase_deadline(8., 7., 20.), 7.)


if __name__ == "__main__":
    unittest.main()
