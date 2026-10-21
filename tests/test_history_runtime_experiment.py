"""Fake-worker protocol tests, not ML training or evidence of model quality."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import benchmark_agent as benchmark
from agent.metrics import score_predictions
from agent.research_candidate import freeze_candidate
import experiment_history_runtime as experiment
from history_runtime_cases import CASE_NAMES, explicit_history_config, make_history_case


SOURCE = "def fit_predict(X_fit, y_fit, X_predict, context):\n    return {'predictions': [0.] * len(X_predict)}\n"


class HistoryExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.solver = self.root / "fake_solver"
        (self.solver / "agent").mkdir(parents=True)
        for name in ("solution.py", "offline_baseline.py", "validation.py", "agent/__init__.py", "agent/automl.py"):
            (self.solver / name).write_text("# Unexecuted fake solver source for protocol tests only.\n")
        (self.solver / "agent/research_v3.py").write_text(
            "def run_agent_v3(task, client, *, time_budget, local_only, history_config, history_weight):\n"
            "    raise RuntimeError('This fixture must never train')\n")
        self.source = benchmark.source_provenance(self.solver)
        self.preflight = patch.object(experiment, "preflight", return_value={"ready": True, "source": self.source})
        self.preflight.start(); self.addCleanup(self.preflight.stop)
        self.worker = patch.object(benchmark, "run_python", side_effect=self.fake_worker)
        self.worker.start(); self.addCleanup(self.worker.stop)
        self.mismatch_baseline = False

    def prepare(self, name="plan", **kwargs):
        output = self.root / name
        experiment.prepare(output, solver_root=self.solver, **kwargs)
        return output

    def invocation(self, parent, frozen, split, fit_rows, held_rows, phase, *, full=False):
        directory = parent / (phase + "_" + "b" * 32)
        directory.mkdir(parents=True)
        spec, train, test = split["spec"], split["train"], split["test"]
        fitting = train if full else train.iloc[fit_rows]
        predicting = test if full else train.iloc[held_rows]
        shutil.copyfile(frozen.code_path, directory / "candidate.py")
        shutil.copyfile(frozen.recipe_path, directory / "recipe.json")
        fitting.loc[:, [spec.id_column, *spec.features, spec.target_column]].to_csv(directory / "fit.csv", index=False)
        predicting.loc[:, [spec.id_column, *spec.features]].to_csv(directory / "predict.csv", index=False)
        spec.prediction_frame(predicting[spec.id_column].tolist(), np.zeros(len(predicting))).to_csv(directory / "predictions.csv", index=False)
        context = {"nonce": "test-nonce", "phase": phase, "code_hash": frozen.code_hash,
                   "recipe_hash": frozen.recipe_hash, "recipe": frozen.recipe, "features": list(spec.features)}
        experiment._write(directory / "context.json", context)
        experiment._write(directory / "receipt.json", {"version": 3, "nonce": context["nonce"], "phase": phase,
            "code_hash": frozen.code_hash, "recipe_hash": frozen.recipe_hash, "output_hash": experiment._hash(directory / "predictions.csv")})
        metric = None if full else {"name": "RMSE", "higher_is_better": False,
            "value": score_predictions("RMSE", train[spec.target_column].iloc[held_rows].tolist(), np.zeros(len(held_rows)))}
        return directory / "receipt.json", metric

    def fake_worker(self, code, task_dir, run_dir, step, timeout, **kwargs):
        # Deliberately fabricate protocol artifacts. This does not run generated
        # code, train a baseline, or establish any expected quality improvement.
        self.assertIn("run_agent_v3(task,None", code)
        task = Path(task_dir) / "task.md"
        request = experiment._read(task.parent / "driver_request.json")
        split = experiment._split(task)
        directory = task.parent / ".agent_runs" / ("a" * 32)
        directory.mkdir(parents=True)
        source_path = directory / "fake_source.py"; source_path.write_text(SOURCE)
        recipe = {"family": "test_fixture", "variant": int(self.mismatch_baseline and request["history_config"] is not None)}
        baseline = freeze_candidate(source_path, recipe, directory / "baseline_frozen")
        identity = {"code_hash": baseline.code_hash, "recipe_hash": baseline.recipe_hash}
        outer, metric = self.invocation(directory / "outer_evaluations/baseline", baseline, split,
            split["outer_fit"], split["outer_held"], "outer_validation")
        refit, _ = self.invocation(directory / "full_refit/baseline", baseline, split, [], [], "full_refit", full=True)
        shutil.copyfile(refit.parent / "predictions.csv", task.parent / "submission.csv")
        shutil.copyfile(refit.parent / "predictions.csv", directory / "last_valid_submission.csv")
        requested = request["history_config"] is not None
        eligible = split["choice"]["strategy"] == "time"
        history = {"version": 1, "policy": "explicit_v1", "origin": "host_history", "scope": "inner_selection",
            "requested": requested, "active": requested and eligible, "status": "disabled" if not requested else "skipped",
            "attempted": 0, "history_config": request["history_config"], "blend_weight": 0.5,
            "baseline": None, "candidate": None, "feedback": None, "reason": "test_fixture"}
        if requested and eligible:
            history_source = directory / "fake_history.py"; history_source.write_text(SOURCE + "\n# Fake history receipt fixture\n")
            candidate = freeze_candidate(history_source, {"base_recipe": recipe, "history_config": request["history_config"]}, directory / "history_frozen")
            _, inner_metric = self.invocation(directory / "inner_evaluations", candidate, split,
                split["inner_fit"], split["inner_held"], "inner_selection")
            history.update(status="completed", attempted=1, baseline=identity,
                candidate={"code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash},
                feedback={"verified": True, "accepted": False, "metric": inner_metric})
        report = {"research_protocol_version": 3, "mode": "local_only", "api": {"attempts": 0, "responses": 0},
            "baseline_search": {"status": "searched", "scope": "outer_fit_only"}, "host_validation": {"split": split["metadata"]},
            "publication": {"new_result_published": True, "recipe": "baseline"}, "published_recipe": "baseline",
            "frozen_baseline": identity, "frozen_candidate": None, "history_research": history,
            "candidate_receipts": {"outer": str(outer), "refit": str(refit), **identity},
            "best_metric": metric, "evaluations": {"final_check": {"baseline": metric}}, "reason": "local_only"}
        experiment._write(directory / "run.json", report)
        experiment._write(task.parent / "driver_evidence.json", {"version": 1, "mode": "local_only", "api_calls": 0,
            "run_report": ".agent_runs/" + "a" * 32 + "/run.json", "run_report_sha256": experiment._hash(directory / "run.json")})
        return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False,
                "resources": {"available": False, "unavailable_reason": "Fake worker; no ML telemetry"}}

    def test_case_determinism_and_no_audit_target_in_solver_input(self):
        for name in CASE_NAMES:
            left, right = make_history_case(name, 131), make_history_case(name, 131)
            self.assertTrue(left.train.equals(right.train)); self.assertEqual(left.audit_actuals, right.audit_actuals)
            self.assertNotIn("outcome", left.test)
            self.assertEqual((240, 60), (len(left.train), len(left.test)))
            config = explicit_history_config(name); config["key_columns"].append("changed")
            self.assertEqual(["entity_key"], explicit_history_config(name)["key_columns"])
        output = self.prepare()
        plan = experiment.read_plan(output)
        self.assertEqual(8, len(plan["pairs"]))
        self.assertEqual({"development": 4, "confirmation": 4}, {stage: sum(p["stage"] == stage for p in plan["pairs"]) for stage in ("development", "confirmation")})
        for case in plan["cases"]:
            self.assertNotIn("outcome", pd.read_csv(output / Path(case["task"]).parent / "inputs/test.csv").columns)
            self.assertFalse((output / Path(case["task"]).parent).is_relative_to(output / "host_audit"))

    def test_fake_worker_dev_confirmation_and_method_freeze_protocol(self):
        output = self.prepare()
        with self.assertRaises(ValueError): experiment.run_stage(output, "confirmation")
        report = experiment.run_stage(output)
        self.assertEqual("completed", report["status"])
        audit = experiment.reaudit(output)
        self.assertEqual("completed", audit["status"]); self.assertEqual(8, len(audit["rows"]))
        iid = next(row for row in audit["rows"] if row["case"].startswith("iid_regression") and row["arm"] == "history")
        self.assertEqual("skipped", iid["history_status"])
        self.assertTrue(all(pair["baseline_recipe_match"] for pair in audit["pairs"]))
        experiment.freeze_confirmation(output)
        with self.assertRaises(ValueError): experiment.freeze_confirmation(output)
        self.assertEqual("completed", experiment.run_stage(output, "confirmation")["status"])
        self.assertEqual("completed", experiment.reaudit(output, "confirmation")["status"])
        freeze = output / "confirmation_freeze.json"
        original = freeze.read_bytes()
        value = experiment._read(freeze); value["method_sha256"] = "changed"; experiment._write(freeze, value)
        with self.assertRaisesRegex(ValueError, "after freeze"): experiment.reaudit(output, "confirmation")
        freeze.write_bytes(original)
        freeze.unlink()
        with self.assertRaises(ValueError): experiment.reaudit(output, "confirmation")

    def test_baseline_mismatch_remains_end_to_end_comparable(self):
        self.mismatch_baseline = True
        output = self.prepare(smoke=True)
        experiment.run_stage(output)
        pair = experiment.reaudit(output)["pairs"][0]
        self.assertTrue(pair["comparable_end_to_end"])
        self.assertFalse(pair["baseline_recipe_match"])

    def test_tampered_receipt_and_publication_fail_reaudit(self):
        for artifact in ("receipt", "output"):
            output = self.prepare(name=artifact, smoke=True)
            report = experiment.run_stage(output)
            row = report["trials"][0]; task = output / row["task"]
            run = task.parent / ".agent_runs" / ("a" * 32)
            if artifact == "receipt":
                raw = experiment._read(run / "run.json")
                path = Path(raw["candidate_receipts"]["refit"])
                value = experiment._read(path); value["nonce"] = "other-invocation"; experiment._write(path, value)
            else:
                path = task.parent / "submission.csv"; path.write_text(path.read_text().replace(",0.0", ",123.0"))
            self.assertEqual("incomplete", experiment.reaudit(output)["status"])

    def test_prepared_source_or_audit_changes_stop_before_worker(self):
        output = self.prepare(smoke=True)
        path = self.solver / "solution.py"; original = path.read_bytes(); path.write_text("# mutated\n")
        with self.assertRaisesRegex(ValueError, "source changed"): experiment.read_plan(output)
        path.write_bytes(original)
        plan = experiment.read_plan(output)
        audit = output / plan["cases"][0]["audit"]; audit.write_text("{}")
        with self.assertRaisesRegex(ValueError, "audit changed"): experiment.run_stage(output)

    def test_coordinator_budget_censors_whole_pairs_without_shortening_one_arm(self):
        output = self.prepare(total_seconds=50, smoke=True)
        report = experiment.run_stage(output)
        self.assertEqual("incomplete", report["status"])
        self.assertEqual([], report["trials"])
        self.assertEqual("coordinator_budget", report["censored_pairs"][0]["reason"])

    def test_runtime_version_change_stops_before_worker(self):
        output = self.prepare(smoke=True)
        with patch.object(benchmark, "_versions", return_value={"python": "changed"}):
            with self.assertRaisesRegex(ValueError, "versions changed"): experiment.read_plan(output)


if __name__ == "__main__":
    unittest.main()
