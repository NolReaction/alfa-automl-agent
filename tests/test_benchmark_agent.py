"""Developer audit boundaries, checkout selection and paired score contracts."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark_agent as benchmark
from benchmark_cases import CASE_NAMES, make_case, write_case
from agent.task_spec import load_task


FAKE_SOLVER = '''from pathlib import Path
import json
import os
import shutil
import time

candidate_plan = lambda classification: [{{"id": "fake_candidate"}}]
metric_candidates = lambda *args, **kwargs: [{{"id": "metric_candidate"}}]

def run_automl(task, *, time_budget, validation, output_dir, **options):
    assert not any(name in os.environ for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")), "Credentials leaked into offline worker"
    {extra}
    shutil.copyfile(task.parent / "inputs" / "sample_submission.csv", task.parent / "submission.csv")
    output_dir.mkdir()
    report = {{"api": {{"used": False, "calls": 0}}, "output_valid": True,
              "model": "Selected fake checkout", "options_received": options,
              "search": {{"status": "completed", "best_candidate": "seed",
                          "candidates": [{{"id": "seed"}}, *candidate_plan(False),
                                         *(metric_candidates() if options.get("metric_recipes") else [])]}}}}
    (output_dir / "automl_report.json").write_text(json.dumps(report), encoding="utf-8")
'''


def fake_checkout(directory: Path, extra="pass") -> Path:
    directory.mkdir()
    (directory / "agent").mkdir()
    for filename in ("solution.py", "offline_baseline.py", "validation.py", "agent/__init__.py"):
        (directory / filename).write_text("# Fake fixed solver checkout\n", encoding="utf-8")
    (directory / "agent" / "automl.py").write_text(FAKE_SOLVER.format(extra=extra), encoding="utf-8")
    return directory


def quiet_benchmark(*args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return benchmark.benchmark(*args, **kwargs)


class BenchmarkAgentTests(unittest.TestCase):
    def test_published_audit_scores_are_independently_recomputed_for_all_case_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in CASE_NAMES:
                with self.subTest(case=name):
                    case = make_case(name, 42)
                    task = write_case(case, Path(directory) / name)
                    spec = load_task(task)
                    if spec.prediction_kind == "probabilities":
                        predictions = np.asarray([[float(actual == label) for label in spec.classes]
                                                  for actual in case.audit_actuals])
                    else:
                        predictions = case.audit_actuals
                    spec.prediction_frame(case.test["row_key"], predictions,
                                          prediction_classes=spec.classes if spec.classes else None).to_csv(
                        task.parent / "submission.csv", index=False)
                    audit = benchmark.audit_submission(task, case.audit_actuals)
                    self.assertTrue(audit["valid"], audit)
                    self.assertEqual(audit["rows"], 150)
                    self.assertEqual(len(audit["submission_sha256"]), 64)
                    expected = 1 if spec.metric.name == "Macro F1" else 0
                    self.assertAlmostEqual(audit["metric"]["value"], expected, places=10)

    def test_audit_answer_intervention_changes_only_host_score_not_input_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            case = make_case("linear_regression", 42)
            task = write_case(case, Path(directory) / "task")
            spec = load_task(task)
            spec.prediction_frame(case.test["row_key"], case.audit_actuals).to_csv(
                task.parent / "submission.csv", index=False)
            before = benchmark.input_fingerprint(task)
            first = benchmark.audit_submission(task, case.audit_actuals)
            altered = benchmark.audit_submission(task, tuple(value + 5 for value in case.audit_actuals))
            self.assertEqual(benchmark.input_fingerprint(task), before)
            self.assertAlmostEqual(first["metric"]["value"], 0)
            self.assertAlmostEqual(altered["metric"]["value"], 5)

    def test_invalid_published_schema_is_rejected_even_when_numeric_predictions_look_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            case = make_case("linear_regression", 42)
            task = write_case(case, Path(directory) / "task")
            (task.parent / "submission.csv").write_text("row_key,outcome\nwrong-row,0\n", encoding="utf-8")
            audit = benchmark.audit_submission(task, case.audit_actuals)
            self.assertFalse(audit["valid"])
            self.assertIsNone(audit["metric"])

    def test_selected_checkout_runs_both_modes_without_api_credentials_or_audit_arguments(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "OPENAI_API_KEY": "do-not-inherit-secret", "OPENAI_BASE_URL": "https://unused.invalid/v1",
                "OPENAI_MODEL": "do-not-inherit-model"}):
            root = Path(directory)
            solver = fake_checkout(root / "solver")
            report = quiet_benchmark(root / "results", solver_root=solver,
                                     cases=("linear_regression",), seeds=(42,), time_budget=5)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertEqual(report["source"]["root"], str(solver.resolve()))
            self.assertEqual([row["solver_report"]["candidate_count"] for row in report["trials"]], [1, 2])
            self.assertEqual(report["within_solver"]["counts"], {"tie": 1})
            for row in report["trials"]:
                self.assertEqual(row["solver_report"]["model"], "Selected fake checkout")
                driver = (Path(row["task"]).parent / "execution" / "experiment_000.py").read_text(encoding="utf-8")
                self.assertNotIn("host_audit", driver)
                self.assertNotIn("audit_actuals", driver)
                self.assertNotIn("do-not-inherit-secret", driver)
                self.assertTrue(row["inputs_unchanged"])
                self.assertTrue(row["sources_unchanged"])
                self.assertEqual(row["execution"]["resources"]["memory_limit_bytes"], 4096 * 1024 ** 2)
            stored = json.loads((root / "results" / "host_audit" / "linear_regression_42.json").read_text())
            self.assertEqual(len(stored["actuals"]), 150)
            self.assertNotIn("do-not-inherit-secret", json.dumps(report))

    def test_optional_new_features_are_omitted_by_default_and_forwarded_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = fake_checkout(root / "solver")
            for name, options in (("default", {}), ("new", {"stability_splits": 3, "metric_recipes": True})):
                report = quiet_benchmark(root / name, solver_root=solver, cases=("linear_regression",),
                                         modes=("seed",), time_budget=5, solver_options=options)
                self.assertEqual(report["status"], "completed")
                worker = json.loads((Path(report["trials"][0]["task"]).parent / "solver_output" /
                                     "automl_report.json").read_text())
                self.assertEqual(worker["options_received"], options)
                self.assertEqual(report["trials"][0]["solver_report"]["candidate_count"], 1)
                self.assertTrue(report["trials"][0]["search_disabled_by_harness"])

    def test_source_fingerprint_tracks_selected_solver_bytes_without_loading_its_code(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = fake_checkout(Path(directory) / "solver")
            first = benchmark.source_provenance(solver)
            (solver / "agent" / "automl.py").write_text("raise AssertionError('Do not import')\n")
            second = benchmark.source_provenance(solver)
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            self.assertNotIn(".env", [row["path"] for row in first["files"]])

    def test_worker_timeout_is_reported_and_does_not_publish_false_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = fake_checkout(root / "solver", "time.sleep(5)")
            report = quiet_benchmark(root / "results", solver_root=solver,
                                     cases=("linear_regression",), modes=("automl",), time_budget=.25)
            row = report["trials"][0]
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(row["status"], "timeout")
            self.assertTrue(row["execution"]["timed_out"])
            self.assertFalse(row["audit"]["valid"])
            self.assertIsNone(row["audit"]["metric"])

    def test_mutated_task_inputs_invalidate_quality_and_stop_later_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = fake_checkout(root / "solver", "(task.parent / 'inputs' / 'train.csv').unlink()")
            report = quiet_benchmark(root / "results", solver_root=solver,
                                     cases=("linear_regression", "nonlinear_regression"), time_budget=5)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(report["planned_trials"], 4)
            self.assertEqual(report["trials"][0]["status"], "input_changed")
            self.assertIsNone(report["trials"][0]["audit"]["metric"])

    def test_reference_pairing_requires_identical_data_metric_and_resources(self):
        control = {"case": "linear_regression", "seed": 42, "mode": "automl", "status": "completed",
                   "input_sha256": "a" * 64, "audit_sha256": "b" * 64, "metric": "RMSE",
                   "validation": {"strategy": "random"},
                   "time_budget_seconds": 45, "memory_limit_bytes": 100, "elapsed_seconds": 5,
                   "audit": {"metric": {"name": "RMSE", "value": 3, "higher_is_better": False}}}
        candidate = deepcopy(control)
        candidate["audit"]["metric"]["value"] = 2
        result = benchmark.paired_comparisons([control], [candidate], kind="reference")
        self.assertEqual(result["pairs"][0]["improvement"], 1)
        self.assertEqual(result["counts"], {"win": 1})
        self.assertIsNone(result["aggregate_metric"])
        for key in ("input_sha256", "audit_sha256", "metric", "validation", "time_budget_seconds", "memory_limit_bytes"):
            altered = deepcopy(candidate)
            altered[key] = "different"
            result = benchmark.paired_comparisons([control], [altered], kind="reference")
            self.assertFalse(result["pairs"][0]["comparable"])
            self.assertEqual(result["pairs"][0]["reason"], "contract_or_budget_mismatch")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            benchmark.paired_comparisons([control, control], [candidate], kind="reference")

    def test_malformed_solver_report_does_not_crash_the_host_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "solver_output"
            output.mkdir()
            for report in ([], {"search": ["wrong"]}, {"search": {"candidates": "wrong"}}):
                (output / "automl_report.json").write_text(json.dumps(report), encoding="utf-8")
                self.assertIsNone(benchmark._solver_summary(root))

    def test_previous_output_and_invalid_flags_are_rejected_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = fake_checkout(root / "solver")
            output = root / "existing"
            output.mkdir()
            (output / "canary").write_text("keep")
            with self.assertRaises(FileExistsError):
                quiet_benchmark(output, solver_root=solver, cases=("linear_regression",), modes=("seed",))
            self.assertEqual((output / "canary").read_text(), "keep")
            for options in ({"stability_splits": True}, {"stability_splits": 4}, {"metric_recipes": 1}, {"unknown": 2}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    quiet_benchmark(root / "unused", solver_root=solver, solver_options=options)
                self.assertFalse((root / "unused").exists())

    def test_reference_report_is_compared_without_mixing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = fake_checkout(root / "solver")
            previous = quiet_benchmark(root / "control", solver_root=solver,
                                       cases=("linear_regression", "imbalanced_binary"),
                                       modes=("seed",), time_budget=5)
            current = quiet_benchmark(root / "candidate", solver_root=solver,
                                      cases=("linear_regression", "imbalanced_binary"), modes=("seed",),
                                      time_budget=5, compare_report=root / "control" / "report.json")
            self.assertEqual(current["reference"]["counts"], {"tie": 2})
            self.assertTrue(current["reference"]["environment_matches"])
            self.assertIsNone(current["reference"]["aggregate_metric"])
            self.assertEqual(previous["completed_trials"], 2)


if __name__ == "__main__":
    unittest.main()
