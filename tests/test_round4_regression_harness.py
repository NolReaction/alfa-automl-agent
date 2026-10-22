"""Evidence, isolation and lifecycle checks without fitting real estimators."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment_round4_regression as lab


CONFIG = {"key_columns": ["key"], "period_column": "period", "numeric_columns": [],
          "positive_columns": [], "ambiguity_policy": "past_unique_only"}


@contextmanager
def cwd(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class Round4HarnessTests(unittest.TestCase):
    def test_recipe_requires_exact_same_schema_and_period(self):
        frozen = {"period_column": "period"}
        discovered = {"period_column": "period", "selected_config": CONFIG}
        recipe = {"base": {"period_column": "period"}, "history_config": deepcopy(CONFIG)}
        lab.check_recipe(recipe, frozen, discovered)
        for value in ({**recipe, "history_config": None},
                      {**recipe, "base": {"period_column": "other"}}):
            with self.assertRaises(ValueError):
                lab.check_recipe(value, frozen, discovered)
        before = deepcopy(recipe)
        removed = lab.algorithm(recipe)
        self.assertIsNone(removed["history_config"])
        self.assertEqual(recipe, before)

    def test_frozen_discovery_rejects_tampering_and_extended_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "discovery/y4"
            directory.mkdir(parents=True)
            report = {"selected_config": CONFIG, "max_cell_visits": 8_000_000,
                      "max_checked_configs": 48, "fit_input_sha256": None}
            lab.write(directory / "discovery_report.json", report)
            lab.write(directory / "host_schema.json", {
                "selected_config": CONFIG,
                "discovery_report_sha256": lab.digest(directory / "discovery_report.json")})
            self.assertEqual(lab.schema(tmp, 4)["selected_config"], CONFIG)
            (directory / "discovery_report.json").write_text(json.dumps({**report, "max_checked_configs": 192}))
            with self.assertRaisesRegex(ValueError, "changed"):
                lab.schema(tmp, 4)
            record = lab.read(directory / "host_schema.json")
            record["discovery_report_sha256"] = lab.digest(directory / "discovery_report.json")
            (directory / "host_schema.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "original host"):
                lab.schema(tmp, 4)

    def test_known_year5_is_blocked_before_any_discovery_or_fit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lab, "plan", return_value={}):
            with patch.object(lab, "schema") as schema:
                with self.assertRaisesRegex(ValueError, "Freeze"):
                    lab.run(tmp, 5, "incumbent")
                schema.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Freeze"):
                lab.discover(tmp, 5)

    def test_discovery_uses_exact_earlier_host_inner_features(self):
        frame = pd.DataFrame({"id": ["a", "b", "c"], "period": ["1", "2", "3"],
                              "x": ["7", "8", "9"], "target": ["11", "12", "13"]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "inputs/y4/data"
            data.mkdir(parents=True)
            frame.to_csv(data / "train.csv", index=False)
            spec = SimpleNamespace(data_dir=data, features=("period", "x"), numeric=("period", "x"), id_column="id")
            def discover(frame, **kwargs):
                self.assertEqual(frame.to_dict("list"), {"period": ["1", "2"], "x": ["7", "8"]})
                self.assertFalse(kwargs["extended"])
                report = {"selected_config": None, "status": "abstained", "reason": "single_period",
                          "fit_input_sha256": None}
                kwargs["output_dir"].mkdir(parents=True)
                lab.write(kwargs["output_dir"] / "discovery_report.json", report)
                return report
            with patch.object(lab, "plan", return_value={"period_column": "period"}), \
                    patch.object(lab, "load_task", return_value=spec), \
                    patch("agent.regression_discovery.discover_regression_schema_bounded", side_effect=discover):
                record = lab.discover(root, 4)
            self.assertEqual(record["outer_fit_rows"], 3)
            self.assertEqual(record["host_inner_fit_rows"], 2)

    def test_physical_fit_observer_records_failures_and_restores_methods(self):
        class Cat:
            def fit(self, X, y):
                return self
        class Lgb:
            def fit(self, X, y):
                raise ValueError("deliberate")
        original_cat, original_lgb = Cat.fit, Lgb.fit
        code = '''
def fit_predict(X_fit, y_fit, X_predict, context):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    CatBoostRegressor().fit(X_fit, y_fit)
    LGBMRegressor().fit(X_fit, y_fit)
'''
        with tempfile.TemporaryDirectory() as tmp, cwd(tmp), patch.dict("sys.modules", {
                "catboost": SimpleNamespace(CatBoostRegressor=Cat),
                "lightgbm": SimpleNamespace(LGBMRegressor=Lgb)}):
            namespace = {}
            exec(lab.instrument_source(code), namespace)
            X = pd.DataFrame({"x": [1, 2, 3]})
            y = np.array([4., 5., 6.])
            with self.assertRaisesRegex(ValueError, "deliberate"):
                namespace["fit_predict"](X, y, X, {})
            ledger = lab.read(Path("physical_fits.json"))["fits"]
            self.assertEqual([x["rows"] for x in ledger], [3, 3])
            self.assertEqual([x["completed"] for x in ledger], [True, False])
            self.assertEqual(ledger[1]["exception_type"], "ValueError")
            self.assertTrue(all(x["elapsed_seconds"] >= 0 for x in ledger))
        self.assertIs(Cat.fit, original_cat)
        self.assertIs(Lgb.fit, original_lgb)
        self.assertEqual(X.to_dict("list"), {"x": [1, 2, 3]})
        np.testing.assert_array_equal(y, [4., 5., 6.])

    def test_four_way_history_groups_conserve_errors_and_unknown_schema_is_explicit(self):
        fit = pd.DataFrame({"key": ["a", "b", "b"], "period": ["1", "1", "1"],
                            "size": ["2", "3", "4"], "target": ["1", "2", "3"]})
        held = pd.DataFrame({"key": ["a", "b", "new", None, "a"],
                             "period": ["2", "2", "2", "2", "bad"], "size": ["5"] * 5})
        answers = pd.DataFrame({"target": ["1", "2", "3", "4", "5"]})
        spec = SimpleNamespace(features=("key", "period", "size"), categorical=("key",), target_column="target")
        frozen = {"period_column": "period", "size_column": "size"}
        predictions = {"candidate": np.zeros(5)}
        rows = lab.diagnostics(fit, held, answers, predictions, spec, frozen, CONFIG)
        history = {r["segment"]: r for r in rows if r["partition"] == "history"}
        self.assertEqual({k: v["n"] for k, v in history.items()}, {
            "eligible": 1, "ambiguous": 1, "valid_without_past": 1,
            "incomplete_key_or_invalid_period": 2})
        self.assertEqual(sum(v["sse"] for v in history.values()), 55.)
        unknown = lab.diagnostics(fit, held, answers, predictions, spec, frozen, None)
        history = [r for r in unknown if r["partition"] == "history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["segment"], "schema_unavailable")
        self.assertFalse(history[0]["schema_available"])

    def test_freeze_requires_complete_same_source_results_and_at_most_two_finalists(self):
        identity = {"head": "measured", "dirty": False, "root": "root"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(lab, "plan", return_value={}), \
                patch.object(lab, "_clean_identity", return_value=identity):
            with self.assertRaisesRegex(ValueError, "at most two"):
                lab.freeze(tmp, "a", ["a", "b", "c"])
            path = Path(tmp) / "runs/y2/incumbent/summary.json"
            lab.write(path, {"status": "completed", "source": {**identity, "head": "different"}})
            with self.assertRaisesRegex(ValueError, "this clean source"):
                lab.freeze(tmp, "a", ["a"])


if __name__ == "__main__":
    unittest.main()
