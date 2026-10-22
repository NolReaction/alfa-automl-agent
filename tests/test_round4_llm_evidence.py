"""Boundary and provenance checks for bounded INNER evidence, without API/fits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.round4_llm_evidence import (profile_inner_fit, validate_inner_snapshot,
                                     validate_recipe_identity)


def recipe_identity(recipe=None):
    recipe = {"action": "ridge", "model_config": {"alpha": 1.}} if recipe is None else recipe
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return {"code_hash": "a" * 64, "recipe_hash": hashlib.sha256(encoded).hexdigest(), "recipe": recipe}


def snapshot():
    return {"version": 1, "scope": "inner_selection",
        "task": {"task_type": "regression", "metric": {"name": "RMSE", "higher_is_better": False},
                 "prediction_kind": "regression"},
        "allowed_inputs": ["research_validation.json", "data/train.csv"],
        "features": [{"name": "measurement", "role": "numeric"}, {"name": "group", "role": "categorical"}],
        "shape": {"fit_rows": 3, "validation_rows": 2, "feature_count": 2},
        "split": {"strategy": "grouped", "seed": None}, "baseline": recipe_identity(),
        "current_metric": {"name": "RMSE", "higher_is_better": False, "value": 2.5},
        "timing": {"remaining_seconds": 45, "final_reserve_seconds": 20, "last_fit_seconds": None,
                   "cpu_threads": 8, "memory_limit_bytes": None},
        "installed_tools": ["numpy", "pandas", "sklearn"]}


class EvidenceBoundaryTests(unittest.TestCase):
    def test_pure_copy_preserves_unknowns_and_literal_probability_class_order(self):
        original = snapshot()
        before = deepcopy(original)
        result = validate_inner_snapshot(original)
        result["features"][0]["name"] = "changed"
        self.assertEqual(original, before)
        self.assertIsNone(result["diagnostics"])
        self.assertIsNone(result["timing"]["last_fit_seconds"])
        original["task"] = {"task_type": "classification", "metric": {"name": "LogLoss", "higher_is_better": False},
                            "prediction_kind": "probabilities", "classes": ["01", "NA", "class c"],
                            "probability_classes": ["class c", "01", "NA"], "positive_class": None}
        original["current_metric"] = None
        result = validate_inner_snapshot(original)
        self.assertEqual(result["task"]["probability_classes"], ["class c", "01", "NA"])

    def test_scope_claim_does_not_authorize_outer_fields_or_row_data(self):
        for field, value in (("outer_metric", 1.2), ("row_ids", ["private-id"]), ("actuals", [1, 2]),
                             ("events", []), ("data_directory", "C:\\private\\inputs")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_inner_snapshot({**snapshot(), field: value})
        with self.assertRaises(ValueError):
            validate_inner_snapshot({**snapshot(), "scope": "outer_validation"})
        for feature in ({"name": "id", "role": "id"}, {"name": "target", "role": "target"}):
            with self.assertRaises(ValueError):
                validate_inner_snapshot({**snapshot(), "features": [feature]})
        with self.assertRaises(ValueError):
            validate_inner_snapshot({**snapshot(), "allowed_inputs": ["C:\\private\\train.csv"]})

    def test_normalized_recipe_identity_distinguishes_settings_and_rejects_tampering(self):
        first = recipe_identity({"model_config": {"alpha": 1.}, "action": "ridge"})
        reordered = recipe_identity({"action": "ridge", "model_config": {"alpha": 1.}})
        changed = recipe_identity({"action": "ridge", "model_config": {"alpha": 2.}})
        self.assertEqual(validate_recipe_identity(first)["recipe_hash"], reordered["recipe_hash"])
        self.assertNotEqual(first["recipe_hash"], changed["recipe_hash"])
        first["recipe"]["model_config"]["alpha"] = 99
        with self.assertRaises(ValueError):
            validate_recipe_identity(first)
        self.assertEqual(validate_recipe_identity({"code_hash": "a"*64, "recipe_hash": "b"*64}),
                         {"code_hash": "a"*64, "recipe_hash": "b"*64})

    def test_recipe_rejects_obvious_private_payloads_without_echoing_them(self):
        sentinel = "private-secret-provider-body"
        for recipe in ({"api_key": sentinel}, {"outer_score": 12}, {"file": "C:\\private\\train.csv"},
                       {"code": "import os\nprint('private')"}, {"huge": [1]*65}):
            with self.subTest(keys=list(recipe)):
                try:
                    validate_recipe_identity(recipe_identity(recipe))
                except ValueError as error:
                    self.assertNotIn(sentinel, str(error))
                else:
                    self.fail("Private/oversized recipe was accepted")

    def test_wrong_metric_probability_contract_and_nonfinite_values_are_rejected(self):
        for change in ({"current_metric": {"name": "MAE", "higher_is_better": False, "value": 2}},
                       {"current_metric": {"name": "RMSE", "higher_is_better": False, "value": float('nan')}},
                       {"timing": {"cpu_threads": 9}}, {"features": [{"name": "x", "role": "numeric"}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_inner_snapshot({**snapshot(), **change})
        data = snapshot()
        data["task"] = {"task_type": "classification", "metric": {"name": "LogLoss", "higher_is_better": False},
                        "prediction_kind": "probabilities", "classes": ["01", "NA", "class c"],
                        "probability_classes": ["01", "NA"]}
        data["current_metric"] = None
        with self.assertRaises(ValueError):
            validate_inner_snapshot(data)
        data["task"]["probability_classes"] = None
        with self.assertRaises(ValueError):
            validate_inner_snapshot(data)
        data["task"]["classes"] = None
        with self.assertRaises(ValueError):
            validate_inner_snapshot(data)
        data = snapshot()
        data["task"]["metric"] = None
        with self.assertRaises(ValueError):
            validate_inner_snapshot(data)

    def test_diagnostics_accept_only_bounded_matching_host_inner_aggregates(self):
        data = snapshot()
        data["diagnostics"] = {"version": 1, "scope": "inner_selection", "rows": 2,
                               "metric": data["current_metric"], "regression": {"mean_bias": -.4, "mae": 1.2,
                               "absolute_error_p90": 2.}}
        self.assertEqual(validate_inner_snapshot(data)["diagnostics"]["regression"]["mean_bias"], -.4)
        for change in ({"scope": "outer_validation"}, {"rows_actuals": [1, 2]},
                       {"regression": {"errors_by_row": [1, -1]}}):
            changed = deepcopy(data)
            changed["diagnostics"].update(change)
            with self.assertRaises(ValueError):
                validate_inner_snapshot(changed)

    def test_feature_profile_uses_only_explicit_inner_fit_and_emits_no_values(self):
        roles = snapshot()["features"]
        frame = pd.DataFrame({"measurement": ["1", "", "3"], "group": ["private-A", "private-B", "private-A"]})
        profile = profile_inner_fit(frame, roles)
        self.assertEqual(profile["columns"][0]["missing_count"], 1)
        self.assertEqual(profile["columns"][1]["distinct_count"], 2)
        self.assertNotIn("private-A", json.dumps(profile))
        self.assertNotIn("private-B", json.dumps(profile))
        data = snapshot()
        data["profile"] = profile
        self.assertEqual(validate_inner_snapshot(data)["profile"]["sample_rows"], 3)
        with self.assertRaises(ValueError):
            profile_inner_fit(frame.assign(target=[1, 2, 3]), roles)
        again = profile_inner_fit(frame, roles)
        self.assertEqual(profile["columns"], again["columns"])

    def test_real_frozen_outer_target_intervention_cannot_change_projected_evidence(self):
        from agent.research_workspace import ResearchWorkspace
        from offline_baseline import read_csv
        from tests.test_research import make_regression

        def packet(gate):
            roles = [{"name": name, "role": "numeric" if name in gate.numeric else "categorical"}
                     for name in gate.features]
            profile = profile_inner_fit(gate.research_train.iloc[gate.fit_rows].loc[:, list(gate.features)], roles)
            data = {"version": 1, "scope": "inner_selection", "task": {
                "task_type": gate.spec.metric.task_type,
                "metric": {"name": gate.spec.metric.name, "higher_is_better": gate.spec.metric.higher_is_better},
                "prediction_kind": gate.spec.prediction_kind,
                "classes": list(gate.spec.classes), "probability_classes": [], "positive_class": gate.spec.positive_class},
                "features": roles, "allowed_inputs": gate.manifest["allowed_inputs"],
                "shape": {"fit_rows": len(gate.fit_rows), "validation_rows": len(gate.held_rows), "feature_count": len(roles)},
                "split": {"strategy": gate.inner_split["strategy"], "seed": gate.inner_split["seed"]},
                "profile": profile}
            result = validate_inner_snapshot(data)
            del result["profile"]["elapsed_seconds"]
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = make_regression(root / "case")
            first = ResearchWorkspace(task, root / "view1")
            original = packet(first)
            train_path = first.spec.data_dir / "train.csv"
            changed = read_csv(train_path)
            changed.loc[first.outer_holdout_rows, first.spec.target_column] = "987654321.125"
            changed.to_csv(train_path, index=False)
            second = ResearchWorkspace(task, root / "view2", frozen_split=first.frozen_split)
            self.assertNotEqual(first.expected_actuals, second.expected_actuals)
            self.assertEqual(first.expected_inner_actuals, second.expected_inner_actuals)
            self.assertEqual(first.fit_rows, second.fit_rows)
            self.assertEqual(original, packet(second))

    def test_profile_explicit_cell_time_and_large_value_limits(self):
        roles = snapshot()["features"]
        frame = pd.DataFrame({"measurement": ["1"] * 1000, "group": ["a"] * 1000})
        limited = profile_inner_fit(frame, roles, max_cells=20)
        self.assertEqual(limited["sample_rows"], 10)
        self.assertTrue(limited["limited"])
        with patch("agent.round4_llm_evidence.time.monotonic", side_effect=[0., 1., 1.]):
            expired = profile_inner_fit(frame, roles)
        self.assertEqual(expired["columns"], [])
        self.assertEqual(expired["omitted_columns"], 2)
        with patch("agent.round4_llm_evidence.time.monotonic", side_effect=[0., 0., 0., 1., 1.]):
            partial = profile_inner_fit(frame, roles)
        self.assertEqual(partial["columns"], [])
        self.assertTrue(partial["limited"])
        oversized = frame.iloc[:1].copy()
        oversized["group"] = "private" * 1000
        result = profile_inner_fit(oversized, roles)
        self.assertEqual(len(result["columns"]), 1)
        self.assertTrue(result["limited"])


if __name__ == "__main__":
    unittest.main()
