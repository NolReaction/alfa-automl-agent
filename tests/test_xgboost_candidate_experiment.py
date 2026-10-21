"""Catalog, sealing and evidence handling of the XGBoost smoke; one tiny native run."""

import ast
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import xgboost_candidate as candidate
from agent.metrics import score_predictions
from agent.research_candidate import CandidateExecutionError
import experiment_xgboost_candidate as experiment
import xgboost_candidate_cases as cases


ROOT = Path(__file__).resolve().parents[1]
NATIVE = importlib.util.find_spec("xgboost") is not None
NEW_FILES = ("agent/xgboost_candidate.py", "xgboost_candidate_cases.py", "experiment_xgboost_candidate.py")


@contextmanager
def raw_root():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "xgboost_candidate"
        with patch.object(experiment, "RAW_ROOT", root):
            yield root


def feature_hashes(frame):
    return set(pd.util.hash_pandas_object(frame.loc[:, list(cases.FEATURES)], index=False))


class XGBoostCaseTests(unittest.TestCase):
    def test_generation_is_deterministic_and_answers_are_host_only(self):
        for name in cases.CASES:
            with self.subTest(name=name):
                first, second = cases.generate(name), cases.generate(name)
                for label in ("fit", "holdout", "test"):
                    pd.testing.assert_frame_equal(first[label], second[label])
                self.assertEqual(first["holdout_answers"], second["holdout_answers"])
                self.assertIn(cases.TARGET_COLUMN, first["fit"].columns)
                self.assertNotIn(cases.TARGET_COLUMN, first["holdout"].columns)
                self.assertNotIn(cases.TARGET_COLUMN, first["test"].columns)
                ids = [value for label in ("fit", "holdout", "test") for value in first[label][cases.ID_COLUMN]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(len(first["holdout"]), len(first["holdout_answers"]))

    def test_cases_contain_missing_values_literal_and_unseen_categories(self):
        for name in cases.CASES:
            with self.subTest(name=name):
                generated = cases.generate(name)
                train = pd.concat([generated["fit"], generated["holdout"]])
                self.assertTrue(train["x1"].eq("").any())
                self.assertTrue({"NA", "0007", "7", "", " Ω "} <= set(train["kind"]))
                unseen = set(generated["test"]["city"]) - set(train["city"])
                self.assertTrue(unseen and all(value.startswith("new_") for value in unseen))

    def test_host_holdouts_respect_time_and_group_boundaries(self):
        regression = cases.generate("regression_time_rmse")
        fit_periods = pd.to_numeric(regression["fit"]["period"])
        held_periods = pd.to_numeric(regression["holdout"]["period"])
        self.assertLess(fit_periods.max(), held_periods.min())
        self.assertEqual(set(held_periods), {6})
        self.assertEqual(set(regression["test"]["period"]), {"7"})
        grouped = cases.generate("multiclass_grouped_macro_f1")
        self.assertFalse(feature_hashes(grouped["fit"]) & feature_hashes(grouped["holdout"]))
        self.assertEqual(len(grouped["fit"]) % 2, 0)
        binary = cases.generate("binary_random_auc")
        self.assertEqual(set(binary["holdout_answers"]), {"NA", "01"})


class XGBoostPlanTests(unittest.TestCase):
    def test_prepare_seals_catalog_inputs_and_recipes_with_zero_fits(self):
        with raw_root() as root, \
                patch.object(experiment, "run_candidate", side_effect=AssertionError("trained")), \
                patch.object(candidate, "execute_xgboost_candidate", side_effect=AssertionError("trained")):
            output = root / "plan"
            plan = experiment.prepare(output)
            self.assertEqual([entry["name"] for entry in plan["cases"]], list(cases.CASES))
            self.assertFalse((output / "runs").exists())
            self.assertEqual(plan["base_sha"], "184408390c83ef951e5a570cf9d2d6c3e9e72f98")
            self.assertEqual(plan["environment"]["external_api_calls"], 0)
            for entry in plan["cases"]:
                self.assertEqual(candidate._checked_recipe(entry["recipe"]), entry["recipe"])
                self.assertEqual(entry["recipe"]["parameters"],
                                 candidate.FIXED_CONFIGURATIONS[entry["definition"]["configuration"]])
                answers = experiment.read_json(output / "host_only" / entry["name"] / "holdout_answers.json")
                holdout = experiment._read_csv(output / "inputs" / entry["name"] / "holdout.csv")
                self.assertEqual(len(answers["answers"]), len(holdout))
                self.assertNotIn(cases.TARGET_COLUMN, holdout.columns)
            self.assertEqual(experiment.load_plan(output)["cases"], plan["cases"])
            with self.assertRaises(ValueError):
                experiment.prepare(output)

    def test_outputs_must_stay_below_the_raw_root(self):
        with raw_root() as root:
            for path in (root, root.parent / "elsewhere"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    experiment.prepare(path)

    def test_changed_inputs_sources_or_recipes_are_rejected_before_running(self):
        with raw_root() as root:
            output = root / "sealed"
            experiment.prepare(output, ["regression_random_mae"])
            fit = output / "inputs" / "regression_random_mae" / "fit.csv"
            original = fit.read_bytes()
            fit.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                experiment.load_plan(output)
            fit.write_bytes(original)
            changed = dict(experiment.source_hashes(), **{"agent/xgboost_candidate.py": "0" * 64})
            with patch.object(experiment, "source_hashes", return_value=changed), \
                    self.assertRaisesRegex(ValueError, "Sources changed"):
                experiment.run(output)
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            plan["cases"][0]["recipe"]["max_iterations"] = 60
            (output / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Recipe changed"):
                experiment.load_plan(output)


class XGBoostEvidenceTests(unittest.TestCase):
    def test_failures_and_timeouts_are_saved_not_retried(self):
        execution = {"exit_code": None, "timed_out": True, "memory_limit_exceeded": False,
                     "stdout": "o" * 5000, "stderr": "boom", "elapsed_seconds": 600.0}
        failures = (CandidateExecutionError("child timed out", execution), OSError("disk"))
        with raw_root() as root:
            output = root / "failure"
            output.mkdir(parents=True)
            for error, status in zip(failures, ("failed", "error")):
                with self.subTest(status=status), \
                        patch.object(experiment, "run_candidate", side_effect=error) as runner:
                    record, result = experiment.invoke(output, None, None, pd.DataFrame({"a": [1]}),
                                                       pd.DataFrame({"a": [2]}), output / status, "full_refit")
                    self.assertEqual(runner.call_count, 1)
                    self.assertIsNone(result)
                    self.assertEqual(record["status"], status)
                    self.assertEqual(record["evidence"], {"count": 0})
            self.assertTrue(record["error"].startswith("OSError"))

    def test_saved_timeout_tail_is_bounded(self):
        execution = {"timed_out": True, "stdout": "o" * 5000, "stderr": "boom"}
        saved = experiment._execution(execution, failed=True)
        self.assertTrue(saved["timed_out"])
        self.assertEqual((len(saved["stdout_tail"]), saved["stderr_tail"]), (4000, "boom"))
        self.assertNotIn("stdout_tail", experiment._execution(execution, failed=False))

    def test_run_records_every_case_once_and_refuses_a_second_run(self):
        def failed(output, frozen, spec, fit, predict, directory, phase, actuals=None):
            return {"phase": phase, "status": "failed", "evidence": {"count": 0}}, None

        with raw_root() as root:
            output = root / "sequence"
            experiment.prepare(output, ["regression_random_mae", "binary_random_auc"])
            with patch.object(experiment, "invoke", side_effect=failed) as calls:
                results = experiment.run(output)
            self.assertEqual(calls.call_count, 4)
            self.assertEqual([call.args[6] for call in calls.call_args_list[:2]],
                             ["outer_validation", "full_refit"])
            self.assertEqual(results["summary"], {"cases": 2, "completed_phases": 0,
                                                  "unsuccessful_phases": 4, "all_checks_passed": False})
            self.assertFalse((output / "results.partial.json").exists())
            self.assertEqual(experiment.read_json(output / "results.json"), results)
            with self.assertRaises(ValueError):
                experiment.run(output)

    def test_rescore_uses_saved_columns_by_literal_class(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = experiment.make_spec(Path(directory), "binary", cases.case_definition("binary_random_auc"))
            output = b"row_id,NA,01\na,0.9,0.1\nb,0.2,0.8\nc,0.6,0.4\n"
            answers = ["NA", "01", "01"]
            expected = score_predictions(spec.metric, answers, np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
                                         classes=("NA", "01"), positive_class="01")
            self.assertEqual(experiment.rescore(output, spec, answers), expected)
            self.assertEqual(experiment.constant_reference(spec, ["NA", "01", "01", "01"], answers), 0.5)

    def test_new_runtime_files_import_no_network_or_api_modules(self):
        forbidden = {"openai", "httpx", "requests", "urllib", "socket", "http", "dotenv",
                     "agent.llm", "agent.gateways", "agent.api_diagnostics"}
        for name in NEW_FILES:
            with self.subTest(name=name):
                tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module)
                self.assertFalse({module for module in imported
                                  if module in forbidden or module.split(".")[0] in forbidden})
                environment = [node for node in ast.walk(tree)
                               if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}
                               or isinstance(node, ast.Constant) and isinstance(node.value, str)
                               and node.value.strip().endswith(".env")]
                self.assertEqual(environment, [], "environment or .env access")


@unittest.skipUnless(NATIVE, "Optional xgboost is not installed")
class XGBoostTinyNativeSmokeTests(unittest.TestCase):
    def test_tiny_native_run_and_audit_without_mocks(self):
        small = {name: dict(cases.CASES[name], train_rows=360, test_rows=40)
                 for name in ("regression_time_rmse", "multiclass_grouped_macro_f1")}
        with raw_root() as root, patch.dict(cases.CASES, small):
            output = root / "tiny"
            experiment.prepare(output, list(small))
            results = experiment.run(output)
            self.assertEqual(results["summary"], {"cases": 2, "completed_phases": 4,
                                                  "unsuccessful_phases": 0, "all_checks_passed": True})
            for case in results["cases"]:
                self.assertEqual(case["outer_rescore"], case["phases"][0]["metric"]["value"])
                for phase in case["phases"]:
                    self.assertEqual(phase["evidence"]["content"]["status"], "completed")
                    self.assertEqual(phase["evidence"]["content"]["package"]["version"], "2.1.4")
            boundary = results["cases"][0]["checks"]["outer_validation_internal_boundary"]
            self.assertIs(boundary, True)
            self.assertIs(results["cases"][1]["checks"]["full_refit_internal_boundary"], True)
            self.assertFalse(list((output / "runs").rglob("holdout_answers.json")))
            self.assertTrue(experiment.audit(output)["all_matched"])


if __name__ == "__main__":
    unittest.main()
