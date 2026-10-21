"""Frozen comparison plans, independent audits and bounded sequential execution."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import benchmark_agent as benchmark
from benchmark_cases import make_case, write_case
import experiment_night_quality as night


FAKE_AUTOML = '''import json
import os
import shutil

def run_automl(task, *, time_budget, validation, output_dir, **options):
    assert not any(name in os.environ for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'))
    output_dir.mkdir()
    shutil.copyfile(task.parent / 'inputs' / 'sample_submission.csv', task.parent / 'submission.csv')
    report = {'api': {'used': False, 'calls': 0}, 'output_valid': True,
              'model': 'unit-test-source', 'options_received': options,
              'search': {'status': 'completed', 'best_candidate': 'seed',
                         'candidates': [{'id': 'seed', 'selected': True, 'status': 'completed', 'phase': 'seed'}]}}
    (output_dir / 'automl_report.json').write_text(json.dumps(report), encoding='utf-8')
'''


def fake_checkout(path):
    path.mkdir()
    (path / "agent").mkdir()
    for name in ("solution.py", "validation.py", "offline_baseline.py", "agent/__init__.py"):
        (path / name).write_text("# Unit-test checkout, never imported by host\n", encoding="utf-8")
    (path / "agent/automl.py").write_text(FAKE_AUTOML, encoding="utf-8")
    return path


def comparison_row(*, arm="baseline", value=2.0, higher=False, metric="RMSE", budget=10, status="completed"):
    return {"case": "linear_regression", "seed": 42, "mode": "automl", "arm": arm,
            "input_sha256": "same-input", "audit_sha256": "same-audit", "metric": metric,
            "validation": {"strategy": "random"}, "time_budget_seconds": budget,
            "memory_limit_bytes": 100, "status": status, "elapsed_seconds": 1,
            "audit": {"valid": status == "completed", "metric": {
                "name": metric, "value": value, "higher_is_better": higher}},
            "execution": {"resources": {"available": False}}, "refit_evidence": {"valid": False}}


class NightQualityRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = fake_checkout(self.root / "baseline")
        self.candidate = fake_checkout(self.root / "candidate")
        self.output = self.root / "results"

    def prepare(self, **options):
        arguments = {"baseline_root": self.baseline, "candidate_root": self.candidate,
                     "cases": ("linear_regression",), "seeds": (42,), "budgets": (10,),
                     "allow_dirty": True}
        arguments.update(options)
        return night.prepare(self.output, **arguments)

    def execute(self):
        with redirect_stdout(io.StringIO()):
            return night.execute_prepared(self.output)

    def test_prepare_freezes_inputs_sources_and_alternating_order_without_running(self):
        with patch.object(benchmark, "run_trial", side_effect=AssertionError("Preflight trained")):
            manifest = self.prepare(seeds=(42, 73), budgets=(10, 20))
        self.assertEqual(manifest["planned_trials"], 8)
        self.assertEqual(manifest["planned_trial_wall_caps_seconds"], 120)
        self.assertEqual([r["arm"] for r in manifest["trial_plan"][:4]],
                         ["baseline", "candidate", "candidate", "baseline"])
        self.assertEqual(manifest["solver_options"], {"baseline": {}, "candidate": {"search_policy": "adaptive_v1"}})
        self.assertEqual(len(manifest["cases"]), 2)
        self.assertEqual(night._read_plan(self.output), manifest)
        self.assertFalse((self.output / "execution_started.json").exists())
        self.assertEqual(list((self.output / "runs").iterdir()), [])
        for case in manifest["cases"]:
            task = self.output / "prepared_inputs" / f"{case['case']}_{case['seed']}" / "task.md"
            self.assertNotIn("host_audit", task.read_text(encoding="utf-8"))
            self.assertEqual(benchmark.input_fingerprint(task), case["input_sha256"])

    def test_fresh_output_and_explicit_total_budget_are_enforced(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        shutil.rmtree(self.output)
        with self.assertRaisesRegex(ValueError, "cover all trial"):
            self.prepare(max_total_seconds=19)
        self.assertFalse(self.output.exists())

    def test_legacy_candidate_option_is_explicit_for_future_default_changes(self):
        manifest = self.prepare(candidate_policy="legacy")
        self.assertEqual(manifest["solver_options"], {"baseline": {}, "candidate": {"search_policy": "legacy"}})

    def test_invalid_options_and_dirty_confirmation_are_rejected_before_creating_output(self):
        for options in ({"budgets": (True,)}, {"budgets": (float("nan"),)}, {"seeds": (True,)},
                        {"cases": ("unknown",)}, {"modes": ("seed",)}, {"memory_mib": True},
                        {"budgets": (10, 10)}, {"purpose": "confirmation"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)
            self.assertFalse(self.output.exists())

    def test_prepared_source_input_audit_manifest_and_host_copy_tampering_are_rejected(self):
        changes = (
            lambda: (self.candidate / "agent/automl.py").write_text("raise RuntimeError('Do not import')\n"),
            lambda: (self.output / "prepared_inputs/linear_regression_42/inputs/train.csv").write_text("changed"),
            lambda: (self.output / "host_audit/linear_regression_42.json").write_text("{}"),
            lambda: (self.output / "manifest.json").write_text("{}"),
            lambda: (self.output / "host_source/benchmark_cases.py").write_text("changed"),
        )
        for change in changes:
            with self.subTest(change=change):
                original = (self.candidate / "agent/automl.py").read_bytes()
                self.prepare()
                change()
                with patch.object(benchmark, "run_trial", side_effect=AssertionError("Changed plan executed")):
                    with self.assertRaises((ValueError, KeyError)):
                        self.execute()
                self.assertFalse((self.output / "execution_started.json").exists())
                shutil.rmtree(self.output)
                (self.candidate / "agent/automl.py").write_bytes(original)

    def test_auditor_runtime_is_frozen_even_when_candidate_root_is_another_checkout(self):
        original = benchmark.source_provenance
        changed = [False]

        def sources(path):
            if Path(path).resolve() == night.ROOT:
                return {"root": str(night.ROOT), "commit": "host", "worktree_dirty": False,
                        "source_sha256": "changed" if changed[0] else "original", "files": []}
            return original(path)

        with patch.object(benchmark, "source_provenance", side_effect=sources):
            manifest = self.prepare()
            self.assertEqual(manifest["host_source"]["source_sha256"], "original")
            changed[0] = True
            with self.assertRaisesRegex(ValueError, "source_sha256 changed"):
                night._read_plan(self.output)

    def test_actual_fresh_workers_forward_candidate_option_and_never_inherit_credentials(self):
        self.prepare()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "night-secret", "OPENAI_BASE_URL": "https://unused.invalid",
                                     "OPENAI_MODEL": "night-secret-model"}):
            report = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(report["external_api_calls"], 0)
        self.assertEqual(report["summary"]["comparisons"][0]["counts"], {"tie": 1})
        for row in report["trials"]:
            self.assertTrue(row["inputs_unchanged"])
            self.assertTrue(row["sources_unchanged"])
            self.assertTrue(row["audit"]["valid"])
            # Copying a template passes the CSV schema; it is not refit evidence.
            self.assertFalse(row["refit_evidence"]["valid"])
            path = Path(row["task"]).parent / "solver_output/automl_report.json"
            worker = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(worker["options_received"], {} if row["arm"] == "baseline"
                             else {"search_policy": "adaptive_v1"})
            driver = (Path(row["task"]).parent / "execution/experiment_000.py").read_text(encoding="utf-8")
            self.assertNotIn("host_audit", driver)
            self.assertNotIn("night-secret", driver)
        self.assertNotIn("night-secret", json.dumps(report))
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_changed_other_checkout_during_execution_aborts_before_next_trial(self):
        self.prepare()
        original = benchmark.run_trial

        def intervention(*args, **kwargs):
            row = original(*args, **kwargs)
            (self.candidate / "agent/automl.py").write_text("# Changed after first trial\n", encoding="utf-8")
            return row

        with patch.object(benchmark, "run_trial", side_effect=intervention) as run:
            report = self.execute()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(report["status"], "aborted")
        self.assertIn("source_sha256 changed", report["abort_reason"])
        self.assertEqual(len(report["trials"]), 1)

    def test_series_budget_never_starts_a_shortened_unfair_second_arm(self):
        self.prepare(max_total_seconds=20)
        clock = [0.0]

        def fake_run(case, mode, directory, **kwargs):
            task = write_case(case, directory)
            clock[0] += 11
            row = comparison_row()
            row.update(task=str(task), input_sha256=benchmark.input_fingerprint(task),
                       audit_sha256=night._hash(night._bytes(case.audit_actuals)))
            return row

        with patch.object(night.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(benchmark, "run_trial", side_effect=fake_run) as run:
            report = self.execute()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["abort_reason"], "series_budget_exhausted")
        self.assertEqual(run.call_args.kwargs["budget"], 10)
        self.assertEqual(report["summary"]["recorded_trials"], 1)
        self.assertEqual(report["summary"]["comparisons"][0]["counts"], {"unpaired": 1})
        self.assertEqual(report["summary"]["comparisons"][0]["pairs"][0]["reason"], "missing_candidate_trial")

    def test_comparisons_are_grouped_by_budget_and_keep_metric_directions_and_failures(self):
        low = comparison_row(value=2)
        good = comparison_row(arm="candidate", value=1)
        high = comparison_row(metric="Macro F1", higher=True, value=.5, budget=20)
        high_candidate = comparison_row(arm="candidate", metric="Macro F1", higher=True, value=.6, budget=20)
        result = night.summarize([low, good, high, high_candidate], 4)
        self.assertIsNone(result["aggregate_metric"])
        first, second = result["comparisons"]
        self.assertEqual(first["counts"], {"win": 1})
        self.assertEqual(first["pairs"][0]["relative_loss_improvement"], .5)
        self.assertEqual(second["counts"], {"win": 1})
        self.assertNotIn("relative_loss_improvement", second["pairs"][0])
        failed = deepcopy(good)
        failed["status"] = "timeout"
        failed["audit"]["valid"] = False
        result = night.summarize([low, failed], 2)
        self.assertEqual(result["arms"]["candidate"]["status_counts"], {"timeout": 1})
        self.assertEqual(result["comparisons"][0]["pairs"][0]["reason"], "incomplete_trial")
        self.assertEqual(result["arms"]["candidate"]["verified_full_refits"], 0)

    def test_refit_evidence_requires_exact_selected_checkpoint_config_original_inputs_and_completion(self):
        case = make_case("linear_regression", 42)
        task = write_case(case, self.root / "refit_task")
        phase_dir = task.parent / ".agent_runs" / ("1" * 32) / "seed" / "seed"
        phase_dir.mkdir(parents=True)
        shutil.copytree(task.parent / "inputs", phase_dir / "inputs")
        shutil.copyfile(task, phase_dir / "task.md")
        shutil.copyfile(task.parent / "inputs/sample_submission.csv", phase_dir / "submission.csv")
        spec = benchmark.load_task(phase_dir / "task.md")
        config = {"id": "seed", "phase": "seed", "task_spec": spec.to_dict(),
                  "candidate_dir": str(phase_dir), "model_config": {"model": "fixed-test"}}
        (phase_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (phase_dir / "durations.json").write_text(json.dumps({"fit_seconds": .1, "predict_seconds": .1,
                                                              "total_seconds": .3}), encoding="utf-8")
        (phase_dir / "execution").mkdir()
        log = phase_dir / "execution/experiment_001.stdout.txt"
        log.write_text("seed: refitting on all 600 rows\nseed completed in 0.30s; API calls: 0\n", encoding="utf-8")
        output = task.parent / "solver_output"
        checkpoint = output / "checkpoints" / ("2" * 32)
        checkpoint.mkdir(parents=True)
        shutil.copyfile(phase_dir / "submission.csv", checkpoint / "submission.csv")
        shutil.copyfile(phase_dir / "submission.csv", task.parent / "submission.csv")
        (checkpoint / "holdout_predictions.csv").write_text("id,actual,prediction\na,1,1\n", encoding="utf-8")
        pointer = {"version": 1, "directory": "checkpoints/" + "2" * 32,
                   "submission_sha256": night._hash((checkpoint / "submission.csv").read_bytes()),
                   "holdout_sha256": night._hash((checkpoint / "holdout_predictions.csv").read_bytes())}
        (output / "checkpoint.json").write_text(json.dumps(pointer), encoding="utf-8")
        report = {"search": {"best_candidate": "seed", "candidates": [{
            "id": "seed", "selected": True, "status": "completed", "phase": "seed",
            "model_config": config["model_config"]}]}}
        (output / "automl_report.json").write_text(json.dumps(report), encoding="utf-8")
        row = {"mode": "automl", "task": str(task), "input_sha256": benchmark.input_fingerprint(task),
               "status": "completed", "sources_unchanged": True, "inputs_unchanged": True}
        evidence = night.refit_evidence(row, 600)
        self.assertTrue(evidence["valid"], evidence)
        self.assertEqual(evidence["kind"], "source_bound_worker_artifacts")
        # Native Windows rewriting changes only task.md line endings.
        phase_text = (phase_dir / "task.md").read_text(encoding="utf-8")
        (phase_dir / "task.md").write_bytes(phase_text.replace("\n", "\r\n").encode("utf-8"))
        self.assertTrue(night.refit_evidence(row, 600)["valid"])
        for field, value in (("status", "source_changed"), ("sources_unchanged", False),
                             ("inputs_unchanged", False)):
            invalid = {**row, field: value}
            self.assertFalse(night.refit_evidence(invalid, 600)["valid"])
        log.write_text("seed: fitting on subset only\n", encoding="utf-8")
        self.assertFalse(night.refit_evidence(row, 600)["valid"])
        log.write_text("seed: refitting on all 600 rows\nseed completed in 0.30s\n", encoding="utf-8")
        (phase_dir / "inputs/train.csv").write_text("changed", encoding="utf-8")
        self.assertFalse(night.refit_evidence(row, 600)["valid"])

    def test_cli_defaults_to_prepare_and_requires_explicit_execute(self):
        arguments = ["--output", str(self.output), "--baseline-root", str(self.baseline),
                     "--candidate-root", str(self.candidate), "--allow-dirty", "--cases", "linear_regression",
                     "--seeds", "42", "--budgets", "10"]
        with redirect_stdout(io.StringIO()), \
                patch.object(night, "execute_prepared", side_effect=AssertionError("CLI executed without opt-in")):
            self.assertEqual(night.main(arguments), 0)
        self.assertTrue((self.output / "manifest.json").is_file())


class NightQualityReauditV2Tests(unittest.TestCase):
    """Artifact-only fixtures exercise the proof; no model is trained."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = fake_checkout(self.root / "baseline")
        self.candidate = fake_checkout(self.root / "candidate")
        trusted_worker = (night.ROOT / "agent/automl_worker.py").read_bytes()
        for root in (self.baseline, self.candidate):
            (root / "agent/automl_worker.py").write_bytes(trusted_worker)
        self.output = self.root / "prepared"
        self.manifest = night.prepare(self.output, baseline_root=self.baseline, candidate_root=self.candidate,
                                      cases=("linear_regression",), seeds=(42,), budgets=(10,), allow_dirty=True)
        self.rows = [self.completed_artifact_fixture(index, arm)
                     for index, arm in ((0, "baseline"), (1, "candidate"))]
        self.report = {"version": 1, "kind": "night_quality_comparison", "status": "completed",
                       "manifest_sha256": night._hash((self.output / "manifest.json").read_bytes()),
                       "sources": self.manifest["sources"], "finished_at": "fixture-finished",
                       "external_api_calls": 0, "trials": self.rows}
        night._write(self.output / "report.json", self.report)

    def completed_artifact_fixture(self, index, arm):
        case = make_case("linear_regression", 42)
        task = write_case(case, self.output / "runs" / f"trial_{index:04d}_{arm}")
        spec = benchmark.load_task(task)
        phase = task.parent / ".agent_runs" / (str(index + 1) * 32) / "seed/seed"
        phase.mkdir(parents=True)
        shutil.copytree(task.parent / "inputs", phase / "inputs")
        shutil.copyfile(task, phase / "task.md")
        local_spec = benchmark.load_task(phase / "task.md")
        model = {"model": "Ridge", "parameters": {"alpha": 1.0}}
        features = {"missing_indicators": False, "signed_log": False, "rare_min_count": 0,
                    "date_columns": [], "text_columns": []}
        fitting, held = list(range(480)), list(range(480, 600))
        config = {"id": "seed", "phase": "seed", "action": "linear", "model_config": model,
                  "feature_config": features, "candidate_dir": str(phase), "task_spec": local_spec.to_dict(),
                  "fit_rows": fitting, "validation_rows": held}
        night._write(phase / "config.json", config)
        prediction = [1.25] * len(case.test)
        spec.prediction_frame(case.test["row_key"], prediction).to_csv(phase / "submission.csv", index=False)
        spec.holdout_frame(case.train["row_key"].iloc[held], case.train["outcome"].iloc[held],
                           [1.25] * len(held)).to_csv(phase / "holdout_predictions.csv", index=False)
        durations = {"fit_seconds": .1, "predict_seconds": .1, "total_seconds": .3}
        night._write(phase / "durations.json", durations)
        execution_dir = phase / "execution"
        execution_dir.mkdir()
        solver_root = Path(self.manifest["sources"][arm]["root"])
        launcher = ("import sys\n" f"sys.path.insert(0, {str(solver_root)!r})\n"
                    "from pathlib import Path\nfrom agent.automl_worker import run_candidate\n"
                    f"run_candidate(Path({str(phase / 'config.json')!r}))\n")
        (execution_dir / "experiment_001.py").write_text(launcher, encoding="utf-8")
        (execution_dir / "experiment_001.stdout.txt").write_text("", encoding="utf-8")
        output = task.parent / "solver_output"
        checkpoint = output / "checkpoints" / ("a" * 32)
        checkpoint.mkdir(parents=True)
        for filename in ("submission.csv", "holdout_predictions.csv"):
            shutil.copyfile(phase / filename, checkpoint / filename)
        shutil.copyfile(phase / "submission.csv", task.parent / "submission.csv")
        pointer = {"version": 1, "directory": "checkpoints/" + "a" * 32,
                   "submission_sha256": night._hash((checkpoint / "submission.csv").read_bytes()),
                   "holdout_sha256": night._hash((checkpoint / "holdout_predictions.csv").read_bytes())}
        night._write(output / "checkpoint.json", pointer)
        selected = {"id": "seed", "selected": True, "status": "completed", "phase": "seed",
                    "model_config": model, "feature_config": features, "durations": durations,
                    "resources": {"exit_code": 0, "timed_out": False, "memory_exceeded": False}}
        night._write(output / "automl_report.json", {"api": {"used": False, "calls": 0}, "output_valid": True,
                                                     "search": {"best_candidate": "seed", "candidates": [selected]}})
        row = {"case": case.name, "seed": case.seed, "mode": "automl", "arm": arm, "trial_index": index,
               "pair": 0, "task": str(task), "input_sha256": benchmark.input_fingerprint(task),
               "audit_sha256": night._hash(night._bytes(case.audit_actuals)), "metric": case.metric,
               "validation": case.validation, "time_budget_seconds": 10.0,
               "memory_limit_bytes": self.manifest["memory_mib"] * 1024 ** 2,
               "solver_options": self.manifest["solver_options"][arm], "status": "completed",
               "sources_unchanged": True, "inputs_unchanged": True, "elapsed_seconds": .5,
               "execution": {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False},
               "audit": benchmark.audit_submission(task, case.audit_actuals)}
        row["refit_evidence"] = night.refit_evidence(row, 600)
        self.assertFalse(row["refit_evidence"]["valid"])
        return row

    def phase(self):
        return next(Path(self.rows[1]["task"]).parent.glob(".agent_runs/*/seed/seed"))

    def test_missing_stdout_has_separate_v2_proof_without_changing_v1(self):
        row = self.rows[1]
        before = deepcopy(row["refit_evidence"])
        evidence = night.refit_evidence_v2(row, 600, self.manifest["sources"]["candidate"])
        self.assertTrue(evidence["valid"], evidence)
        self.assertFalse(evidence["stdout_corroborated"])
        self.assertEqual(evidence["kind"], "source_bound_worker_v2")
        self.assertEqual(row["refit_evidence"], before)
        self.assertFalse(night.refit_evidence(row, 600)["valid"])

    def test_missing_duration_modified_inputs_untrusted_launcher_and_incomplete_phase_reject(self):
        phase = self.phase()
        selected_report = Path(self.rows[1]["task"]).parent / "solver_output/automl_report.json"
        files = [phase / "durations.json", phase / "inputs/train.csv",
                 phase / "execution/experiment_001.py", phase / "config.json", selected_report]
        backups = {path: path.read_bytes() for path in files}
        changes = [lambda: files[0].unlink(),
                   lambda: files[1].write_text("changed", encoding="utf-8"),
                   lambda: files[2].write_text("print('forged launcher')\n", encoding="utf-8"),
                   lambda: files[3].write_text("{}", encoding="utf-8"),
                   lambda: files[4].write_text(json.dumps({"api": {"used": False, "calls": 0}, "output_valid": True,
                       "search": {"best_candidate": "seed", "candidates": [{"id": "seed", "selected": True,
                       "status": "running", "phase": "seed"}]}}), encoding="utf-8")]
        for change in changes:
            with self.subTest(change=change):
                change()
                evidence = night.refit_evidence_v2(self.rows[1], 600, self.manifest["sources"]["candidate"])
                self.assertFalse(evidence["valid"], evidence)
                for path, content in backups.items():
                    path.write_bytes(content)

    def test_template_only_untrusted_worker_and_unsuccessful_trial_do_not_gain_proof(self):
        row = deepcopy(self.rows[1])
        for key, value in (("status", "timeout"), ("sources_unchanged", False), ("inputs_unchanged", False)):
            invalid = {**row, key: value}
            self.assertFalse(night.refit_evidence_v2(invalid, 600, self.manifest["sources"]["candidate"])["valid"])
        worker = self.candidate / "agent/automl_worker.py"
        original_worker = worker.read_bytes()
        worker.write_text("# Untrusted version that could fabricate completion\n", encoding="utf-8")
        changed_source = benchmark.source_provenance(self.candidate)
        self.assertFalse(night.refit_evidence_v2(row, 600, changed_source)["valid"])
        worker.write_bytes(original_worker)
        # A schema-valid copied template has no completed worker artifacts.
        shutil.rmtree(self.phase())
        task = Path(row["task"])
        shutil.copyfile(task.parent / "inputs/sample_submission.csv", task.parent / "submission.csv")
        self.assertTrue(benchmark.audit_submission(task, make_case("linear_regression", 42).audit_actuals)["valid"])
        self.assertFalse(night.refit_evidence_v2(row, 600, self.manifest["sources"]["candidate"])["valid"])

    def test_reaudit_independently_scores_and_writes_new_bound_evidence_without_mutating_old_files(self):
        old = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        destination = self.output / "reaudit_v2.json"
        with patch.object(benchmark, "run_trial", side_effect=AssertionError("Re-audit trained a model")):
            result = night.reaudit_prepared(self.output, destination)
        self.assertTrue(result["all_trials_verified"])
        self.assertEqual(result["original_report_sha256"], night._hash(old[self.output / "report.json"]))
        self.assertEqual(result["summary"]["candidate"]["verified_full_refits"], 1)
        for row in result["trials"]:
            self.assertTrue(row["valid"])
            self.assertTrue(row["independent_audit_matches_original"])
            self.assertTrue(row["full_refit"]["valid"])
            self.assertFalse(row["v1_refit_evidence"]["valid"])
        for path, content in old.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        with self.assertRaisesRegex(ValueError, "new .json"):
            night.reaudit_prepared(self.output, destination)
        self.assertNotIn("OPENAI_API_KEY", destination.read_text(encoding="utf-8"))

    def test_finished_report_audit_truth_and_row_plan_cannot_be_changed_silently(self):
        report_path = self.output / "report.json"
        report = deepcopy(self.report)
        report["status"] = "running"
        night._write(report_path, report)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            night.reaudit_prepared(self.output, self.output / "audit1.json")
        night._write(report_path, self.report)
        report = deepcopy(self.report)
        report["trials"][1]["time_budget_seconds"] = 20
        night._write(report_path, report)
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            night.reaudit_prepared(self.output, self.output / "audit2.json")
        night._write(report_path, self.report)
        truth = next((self.output / "host_audit").glob("*.json"))
        truth.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit answers changed"):
            night.reaudit_prepared(self.output, self.output / "audit3.json")

    def test_public_finished_report_uses_same_reaudit_and_immutable_v1_flags(self):
        # Convert only the outer manifest/report envelope. Artifact proof and
        # independent scorer are the same for public and synthetic case inputs.
        manifest = deepcopy(self.manifest)
        manifest["kind"] = "night_public_prepared"
        manifest["budget_seconds"] = 10
        original_fingerprint = next((self.output / "prepared_inputs").glob("*/task.md"))
        manifest["datasets"] = {"regression": {"fingerprints": {
            str(original_fingerprint): night._hash(original_fingerprint.read_bytes())}}}
        manifest["script_sources"] = {str(night.ROOT / name): night._hash((night.ROOT / name).read_bytes())
                                      for name in night.HARNESS_FILES}
        for item in manifest["cases"]:
            item["name"] = item.pop("case")
            item["split"] = {"split_hash": "public-fixture-split"}
        manifest["trial_plan"] = [{"index": item["index"], "pair": item["pair"], "arm": item["arm"]}
                                   for item in manifest["trial_plan"]]
        night._write(self.output / "manifest.json", manifest)
        night._write(self.output / "seal.json", {"manifest_sha256": night._hash((self.output / "manifest.json").read_bytes())})
        report = deepcopy(self.report)
        report["kind"] = "night_public_comparison"
        report["manifest_sha256"] = night._hash((self.output / "manifest.json").read_bytes())
        for row in report["trials"]:
            row["public_outer_split_hash"] = "public-fixture-split"
        night._write(self.output / "report.json", report)
        result = night.reaudit_prepared(self.output, self.output / "public_reaudit.json")
        self.assertTrue(result["all_trials_verified"])
        self.assertEqual(len(result["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
