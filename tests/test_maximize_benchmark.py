"""Paired audit contracts use a fake worker, never model training or an API."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_maximize as benchmark
from agent.task_spec import load_task


def fake_checkout(path):
    path.mkdir()
    (path / "agent").mkdir()
    for name in ("solution.py", "offline_baseline.py", "validation.py", "agent/__init__.py",
                 "agent/automl.py", "agent/research_v3.py", "agent/research_candidate.py"):
        (path / name).write_text("# sealed fake worker source, not imported\n", encoding="utf-8")
    return path


def fake_execution(task, *, arm, budget, available=True, cleanup=True, api_attempts=0,
                   adaptive_active=True, nested_timeout=False):
    """Create a complete fixture receipt chain without executing any fit."""
    directory = task.parent
    spec = load_task(task)
    shutil.copyfile(directory / "inputs/sample_submission.csv", directory / "submission.csv")
    root = directory / ".agent_runs" / ("a" * 32)
    root.mkdir(parents=True)
    recipe = {"action": "linear", "model_config": {"model": "fixture"}}
    recipe_bytes = benchmark.json_bytes(recipe)
    code_hash, recipe_hash = "b" * 64, benchmark.hashlib.sha256(recipe_bytes).hexdigest()
    receipts = {"code_hash": code_hash, "recipe_hash": recipe_hash}
    for key, phase, parent in (("outer", "outer_validation", "outer_evaluations"),
                               ("refit", "full_refit", "full_refit")):
        folder = root / parent / (phase + "_fixture")
        folder.mkdir(parents=True)
        shutil.copyfile(directory / "submission.csv", folder / "predictions.csv")
        (folder / "recipe.json").write_bytes(recipe_bytes)
        receipt = {"version": 3, "nonce": "fixture", "phase": phase,
                   "code_hash": code_hash, "recipe_hash": recipe_hash,
                   "output_hash": benchmark.file_hash(folder / "predictions.csv")}
        benchmark.write_json(folder / "receipt.json", receipt)
        receipts[key] = str(folder / "receipt.json")
    resources = {"available": available, "cleanup_complete": cleanup, "peak_rss_bytes": None}
    execution = {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False,
                 "elapsed_seconds": 0.01, "resources": resources}
    nested = deepcopy(execution)
    nested["timed_out"] = nested_timeout
    report = {"research_protocol_version": 3, "status": "completed", "reason": "local_only",
              "stage": "result", "mode": "local_only", "api": {"attempts": api_attempts, "responses": 0},
              "time_budget_seconds": budget - 2, "elapsed_seconds": 1.0,
              "published_recipe": "baseline", "frozen_baseline": {"code_hash": code_hash, "recipe_hash": recipe_hash},
              "candidate_receipts": receipts, "output_check": {"valid": True},
              "publication": {"new_result_published": True, "recipe": "baseline"},
              "coordinator": {"active": arm == "adaptive" and adaptive_active},
              "local_research": {"attempted": 8, "verified": 8, "reason": "queue_exhausted",
                                 "adaptive_attempted": 4 if arm == "adaptive" else 0,
                                 "trials": [{"family": "catboost"}, {"family": "linear"}]},
              "research_extensions": {"attempted": 1 if arm == "adaptive" else 0},
              "time_allocation": {"remaining_seconds": budget - 3, "measurements": {
                  "recipe": [{"elapsed_seconds": .1}, {"elapsed_seconds": .2}]}},
              "baseline_search": {"execution": nested},
              "evaluations": {"selection": {"metric": {"name": spec.metric.name, "value": 0.5,
                                                        "higher_is_better": spec.metric.higher_is_better}}}}
    benchmark.write_json(root / "run.json", report)
    evidence = {"version": 2, "mode": "paired_maximize", "arm": arm, "external_api_calls": 0,
                "requested_time_budget_seconds": budget - 2,
                "solver_returned": True, "run_report_origin": "returned", "terminal_error_type": None,
                "run_report": (root / "run.json").relative_to(directory).as_posix(),
                "run_report_sha256": benchmark.file_hash(root / "run.json"),
                "worker_elapsed_seconds": 1.1, "observations": [
                    {"phase": "prepare_baseline", "elapsed_seconds": .4},
                    {"phase": "candidate_full_refit", "elapsed_seconds": .2, "execution": execution},
                    {"phase": "finalize", "elapsed_seconds": .3}]}
    (directory / "solver_output").mkdir()
    benchmark.write_json(directory / "solver_output/research_benchmark.json", evidence)
    return execution


class MaximizeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.solver = fake_checkout(self.root / "solver")

    def prepare(self, **options):
        output = self.root / options.pop("name", "output")
        plan = benchmark.prepare(output, solver_root=self.solver, **options)
        return output, plan

    def fake_runner(self, **options):
        calls = []

        def run(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
            calls.append({"code": code, "timeout": timeout, "memory": max_memory_bytes})
            return fake_execution(Path(task_dir) / "task.md", arm=Path(task_dir).name,
                                  budget=timeout, **options)
        return calls, run

    def test_prepare_freezes_identical_arm_inputs_and_external_audit_labels(self):
        output, plan = self.prepare()
        self.assertEqual(len(plan["entries"]), 2)
        self.assertEqual(plan["method"]["arms"], ["control", "adaptive"])
        self.assertEqual(plan["method"]["time_budget_seconds"], 360)
        self.assertFalse(plan["method"]["quality_admission"])
        self.assertEqual(plan["entries"][0]["arm_order"], ["control", "adaptive"])
        self.assertEqual(plan["entries"][1]["arm_order"], ["adaptive", "control"])
        for entry in plan["entries"]:
            audit = benchmark.read_json(output / entry["audit"])
            self.assertEqual(audit["ids"][:3], ["000005", "NA", "audit c"])
            for relative in entry["tasks"].values():
                task = output / relative
                self.assertNotIn(task.parent, (output / entry["audit"]).parents)
                self.assertNotIn("host_audit", task.read_text())
                self.assertEqual(benchmark.input_fingerprint(task), entry["input_sha256"])
        self.assertEqual(benchmark.load_plan(output)[1], plan)

    def test_invalid_configuration_never_creates_an_output(self):
        for options in ({"time_budget": 300}, {"time_budget": True}, {"time_budget": float("nan")},
                        {"time_budget": 10501}, {"seeds": (True,)}, {"seeds": (42, 42)},
                        {"cases": ("missing",)}, {"memory_mib": True}, {"memory_mib": 64}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)
            self.assertFalse((self.root / "output").exists())

    def test_existing_evidence_is_not_overwritten(self):
        output, _ = self.prepare()
        before = (output / "plan.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((output / "plan.json").read_bytes(), before)

    def test_input_or_audit_mutation_is_rejected_before_execution(self):
        for target in ("task", "audit"):
            with self.subTest(target=target):
                output, plan = self.prepare(name=target)
                entry = plan["entries"][0]
                path = output / (entry["tasks"]["control"] if target == "task" else entry["audit"])
                path.write_text(path.read_text() + " ", encoding="utf-8")
                with patch.object(benchmark, "run_python") as worker, self.assertRaises(ValueError):
                    benchmark.run_prepared(output)
                worker.assert_not_called()

    def test_source_drift_is_rejected_before_a_fit(self):
        output, _ = self.prepare()
        (self.solver / "agent/research_v3.py").write_text("raise AssertionError('changed')\n")
        with patch.object(benchmark, "run_python") as worker, self.assertRaisesRegex(ValueError, "source bytes"):
            benchmark.run_prepared(output)
        worker.assert_not_called()

    def test_preregistration_and_host_method_tampering_are_rejected(self):
        output, plan = self.prepare()
        plan["method"]["time_budget_seconds"] = 900
        benchmark.write_json(output / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "seal"):
            benchmark.load_plan(output)
        output, _ = self.prepare(name="host_drift")
        with patch.object(benchmark, "host_sources", return_value={"modified": "0" * 64}), \
                self.assertRaisesRegex(ValueError, "Host method"):
            benchmark.load_plan(output)

    def test_driver_contains_only_task_inputs_and_one_original_budget(self):
        task = self.root / "task/task.md"
        for arm in benchmark.ARMS:
            code = benchmark.driver(self.solver, task, arm, 360)
            compile(code, "worker.py", "exec")
            self.assertIn("local_only=True", code)
            self.assertIn("adaptive_search=" + str(arm == "adaptive"), code)
            self.assertIn("360 - (time.monotonic() - started) - 2.0", code)
            for forbidden in ("host_audit", "audit_actuals", "OPENAI_API_KEY", "make_client", "load_dotenv"):
                self.assertNotIn(forbidden, code)
            self.assertNotIn(".glob(", code)
            self.assertNotIn(".rglob(", code)

    def test_fake_native_receipts_are_audited_and_equal_caps_retained(self):
        output, _ = self.prepare()
        calls, runner = self.fake_runner()
        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["executed_arms"], 4)
        self.assertEqual(report["outcomes"], {"tie": 2})
        self.assertEqual([call["timeout"] for call in calls], [360] * 4)
        self.assertEqual([call["memory"] for call in calls], [4096 * 1024 ** 2] * 4)
        self.assertFalse(report["quality_admission"])
        self.assertIsNone(report["aggregate_metric"])
        for pair in report["pairs"]:
            self.assertTrue(pair["functionally_comparable"])
            for row in pair["arms"].values():
                self.assertEqual(row["contract_errors"], [])
                self.assertTrue(row["research"]["receipt_checks_valid"])
                self.assertEqual(row["research"]["recorded_recipe_fit_measurements"], 2)
                self.assertEqual(row["research"]["published_family"], "linear")
                self.assertEqual(row["research"]["observed_phase_wall_seconds"]["finalize"], .3)
        self.assertTrue((output / "report.md").is_file())
        self.assertTrue((output / "completion_seal.json").is_file())
        with self.assertRaises(FileExistsError):
            benchmark.run_prepared(output)

    def test_unknown_resources_are_provisional_and_never_quality_admission(self):
        output, _ = self.prepare()
        _, runner = self.fake_runner(available=False, cleanup=None)
        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(report["status"], "completed")
        for pair in report["pairs"]:
            self.assertEqual(pair["qualification"], "provisional_resource_observation")
            self.assertFalse(pair["quality_admission"])
            for row in pair["arms"].values():
                self.assertFalse(row["resources"]["fully_observed"])
                self.assertIsNone(row["resources"]["execution_records"][0]["peak_rss_bytes"])

    def test_unknown_cleanup_after_nested_timeout_censors_and_stops_series(self):
        output, _ = self.prepare()
        calls, runner = self.fake_runner(available=False, cleanup=None, nested_timeout=True)
        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["executed_arms"], 1)
        self.assertEqual(report["planned_arms"], 4)
        pair = report["pairs"][0]
        self.assertFalse(pair["functionally_comparable"])
        self.assertIsNone(pair["outcome"])
        self.assertEqual(pair["arms"]["control"]["status"], "censored")

    def test_api_activity_and_inactive_adaptive_arm_do_not_count_as_success(self):
        for name, options in (("api", {"api_attempts": 1}), ("inactive", {"adaptive_active": False})):
            with self.subTest(name=name):
                output, _ = self.prepare(name=name)
                _, runner = self.fake_runner(**options)
                with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
                    report = benchmark.run_prepared(output)
                self.assertEqual(report["status"], "incomplete")
                self.assertFalse(any(pair["functionally_comparable"] for pair in report["pairs"]))

    def test_nonzero_launcher_and_missing_execution_resources_stop_the_series(self):
        failures = (
            {"exit_code": 1, "resources": {"available": False, "cleanup_complete": None}},
            {"exit_code": None, "error_type": "OSError", "resources": {"available": False}},
            {"exit_code": 1},
            {"error_type": "OSError"},
            {"error": "fake launcher error"},
            {},
            {"exit_code": 0, "resources": {"available": True, "cleanup_complete": False}},
        )
        for index, execution in enumerate(failures):
            with self.subTest(execution=execution):
                output, _ = self.prepare(name="failure_" + str(index))
                calls = []

                def runner(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
                    calls.append(code)
                    fake_execution(Path(task_dir) / "task.md", arm=Path(task_dir).name, budget=timeout)
                    return execution

                with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
                    report = benchmark.run_prepared(output)
                self.assertEqual(len(calls), 1)
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["pairs"][0]["arms"]["control"]["status"], "censored")

    def test_candidate_error_without_an_execution_receipt_is_censored(self):
        output, _ = self.prepare()
        calls = []

        def runner(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
            calls.append(code)
            execution = fake_execution(Path(task_dir) / "task.md", arm=Path(task_dir).name, budget=timeout)
            path = Path(task_dir) / "solver_output/research_benchmark.json"
            evidence = benchmark.read_json(path)
            evidence["observations"].append({"phase": "candidate_inner_selection", "error_type": "OSError",
                                              "elapsed_seconds": .1})
            benchmark.write_json(path, evidence)
            return execution

        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["pairs"][0]["arms"]["control"]["status"], "censored")

    def test_driver_pins_failed_current_state_without_admitting_an_old_valid_csv(self):
        output, _ = self.prepare()
        calls = []

        def runner(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
            calls.append(code)
            task_dir = Path(task_dir)
            shutil.copyfile(task_dir / "inputs/sample_submission.csv", task_dir / "submission.csv")
            decoy = task_dir / ".agent_runs" / ("a" * 32)
            current = task_dir / ".agent_runs" / ("c" * 32)
            decoy.mkdir(parents=True)
            current.mkdir()
            benchmark.write_json(decoy / "run.json", {"status": "completed", "stage": "result"})
            research = ModuleType("agent.research_v3")

            def prepare_research(state):
                benchmark.write_json(state.run_dir / "run.json", {
                    "research_protocol_version": 3, "status": "failed", "stage": "inner_research",
                    "mode": "local_only", "api": {"attempts": 0, "responses": 0},
                    "time_budget_seconds": state.budget, "elapsed_seconds": .1,
                    "evaluations": None,
                    "coordinator": {"active": False},
                    "local_research": {"attempted": 10, "verified": 9, "trials": []}})

            def run_agent(task, client, **options):
                state = SimpleNamespace(run_dir=current, budget=options["time_budget"])
                research.prepare_research(state)
                raise KeyError("label")

            research.prepare_research = prepare_research
            research.run_agent_v3 = run_agent
            package = ModuleType("agent")
            package.research_v3 = research
            with patch.dict(sys.modules, {"agent": package, "agent.research_v3": research}), \
                    patch.object(sys, "path", list(sys.path)), self.assertRaises(KeyError):
                exec(compile(code, "fake_failed_worker.py", "exec"), {})
            return {"exit_code": 1, "resources": {"available": False, "cleanup_complete": None}}

        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["executed_arms"], 1)
        pair = report["pairs"][0]
        self.assertFalse(pair["functionally_comparable"])
        self.assertIsNone(pair["outcome"])
        row = pair["arms"]["control"]
        self.assertEqual(row["status"], "censored")
        self.assertTrue(row["audit"]["valid"])
        self.assertFalse(row["quality_admission"])
        self.assertIn("solver_did_not_return", row["contract_errors"])
        self.assertIn("verified_full_refit_contract", row["contract_errors"])
        summary = row["research"]
        self.assertEqual(summary["raw_report"], ".agent_runs/" + "c" * 32 + "/run.json")
        self.assertEqual(summary["raw_report_sha256"], benchmark.file_hash(
            output / "runs/nonlinear_regression_42/control" / summary["raw_report"]))
        self.assertFalse(summary["solver_returned"])
        self.assertEqual(summary["run_report_origin"], "observed_state_after_failure")
        self.assertEqual(summary["terminal_error_type"], "KeyError")
        self.assertEqual(summary["stage"], "inner_research")
        self.assertEqual((summary["local_attempted"], summary["local_verified"]), (10, 9))
        self.assertIsNone(summary["published_family"])
        self.assertFalse(summary["receipt_checks_valid"])
        self.assertEqual(row["resources"]["censored_cleanup_paths"], ["root/launcher"])
        self.assertIsNone(row["resources"]["execution_records"][0]["cleanup_complete"])
        self.assertIn("KeyError", (output / "report.md").read_text(encoding="utf-8"))

    def test_exception_before_observed_state_never_selects_an_existing_report(self):
        output, plan = self.prepare()
        task = output / plan["entries"][0]["tasks"]["control"]
        decoy = task.parent / ".agent_runs" / ("d" * 32)
        decoy.mkdir(parents=True)
        benchmark.write_json(decoy / "run.json", {"status": "completed"})
        research = ModuleType("agent.research_v3")

        def run_agent(*args, **kwargs):
            raise RuntimeError("before prepare")

        research.run_agent_v3 = run_agent
        package = ModuleType("agent")
        package.research_v3 = research
        with patch.dict(sys.modules, {"agent": package, "agent.research_v3": research}), \
                patch.object(sys, "path", list(sys.path)), self.assertRaises(RuntimeError):
            exec(compile(benchmark.driver(self.solver, task, "control", 360), "fake_worker.py", "exec"), {})
        evidence = benchmark.read_json(task.parent / "solver_output/research_benchmark.json")
        self.assertIsNone(evidence["run_report"])
        self.assertIsNone(evidence["run_report_origin"])
        self.assertFalse(evidence["solver_returned"])
        self.assertEqual(evidence["terminal_error_type"], "RuntimeError")
        self.assertEqual(benchmark.summarize_research(task.parent, "control", 360),
                         (None, ["missing_or_invalid_current_report"]))

    def test_failed_return_cannot_admit_even_a_complete_valid_receipt_chain(self):
        output, _ = self.prepare()

        def runner(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
            execution = fake_execution(Path(task_dir) / "task.md", arm=Path(task_dir).name, budget=timeout)
            path = Path(task_dir) / "solver_output/research_benchmark.json"
            evidence = benchmark.read_json(path)
            evidence.update(solver_returned=False, run_report_origin="observed_state_after_failure",
                            terminal_error_type="KeyError")
            benchmark.write_json(path, evidence)
            return execution

        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(report["status"], "incomplete")
        for pair in report["pairs"]:
            self.assertFalse(pair["functionally_comparable"])
            for row in pair["arms"].values():
                self.assertEqual(row["status"], "unverified_contract")
                self.assertTrue(row["audit"]["valid"])
                self.assertFalse(row["research"]["receipt_checks_valid"])
                self.assertIn("solver_did_not_return", row["contract_errors"])
                self.assertFalse(row["quality_admission"])

    def test_source_mutation_during_worker_removes_metric_and_stops_later_arms(self):
        output, _ = self.prepare()
        calls = []

        def runner(code, task_dir, run_dir, step, *, timeout, max_memory_bytes):
            calls.append(code)
            result = fake_execution(Path(task_dir) / "task.md", arm=Path(task_dir).name, budget=timeout)
            (self.solver / "agent/automl.py").write_text("# changed during worker\n")
            return result

        with patch.object(benchmark, "run_python", side_effect=runner), redirect_stdout(io.StringIO()):
            report = benchmark.run_prepared(output)
        self.assertEqual(len(calls), 1)
        row = report["pairs"][0]["arms"]["control"]
        self.assertEqual(row["status"], "integrity_failed")
        self.assertIsNone(row["audit"]["metric"])

    def test_postrun_submission_or_report_hash_drift_invalidates_receipts(self):
        output, plan = self.prepare()
        task = output / plan["entries"][0]["tasks"]["adaptive"]
        fake_execution(task, arm="adaptive", budget=360)
        summary, errors = benchmark.summarize_research(task.parent, "adaptive", 360)
        self.assertEqual(errors, [])
        self.assertTrue(summary["receipt_checks_valid"])
        (task.parent / "submission.csv").write_text("row_key,outcome\nwrong,0\n")
        summary, errors = benchmark.summarize_research(task.parent, "adaptive", 360)
        self.assertIn("verified_full_refit_contract", errors)
        evidence_path = task.parent / "solver_output/research_benchmark.json"
        evidence = benchmark.read_json(evidence_path)
        evidence["run_report_sha256"] = "0" * 64
        benchmark.write_json(evidence_path, evidence)
        summary, errors = benchmark.summarize_research(task.parent, "adaptive", 360)
        self.assertIsNone(summary)
        self.assertEqual(errors, ["missing_or_invalid_current_report"])

    def test_published_recipe_and_time_accounting_are_independently_checked(self):
        for name in ("recipe", "clock"):
            with self.subTest(name=name):
                output, plan = self.prepare(name=name)
                task = output / plan["entries"][0]["tasks"]["adaptive"]
                fake_execution(task, arm="adaptive", budget=360)
                if name == "recipe":
                    recipe_path = task.parent / ".agent_runs" / ("a" * 32) / "full_refit/full_refit_fixture/recipe.json"
                    recipe_path.write_text('{}\n')
                else:
                    evidence_path = task.parent / "solver_output/research_benchmark.json"
                    evidence = benchmark.read_json(evidence_path)
                    report_path = task.parent / evidence["run_report"]
                    report = benchmark.read_json(report_path)
                    report["elapsed_seconds"] = 1000
                    benchmark.write_json(report_path, report)
                    evidence["run_report_sha256"] = benchmark.file_hash(report_path)
                    benchmark.write_json(evidence_path, evidence)
                _, errors = benchmark.summarize_research(task.parent, "adaptive", 360)
                expected = "published_recipe_bytes_changed" if name == "recipe" else "invalid_or_exhausted_host_time_accounting"
                self.assertIn(expected, errors)


if __name__ == "__main__":
    unittest.main()
