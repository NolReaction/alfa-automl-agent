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


if __name__ == "__main__":
    unittest.main()
