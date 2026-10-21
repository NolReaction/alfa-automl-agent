"""Mocked host/protocol checks; these fixtures never train an ML model.

Successful fake receipts establish audit mechanics only, never model quality.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from agent import research_v3 as host
from agent.research_candidate import freeze_candidate
from agent.metrics import get_metric
from agent.task_spec import TaskSpec
from agent.search import seed_candidate
from agent.frequency_candidate import execute_frequency_candidate, make_frequency_recipe
import experiment_frequency_candidate as experiment
from frequency_candidate_cases import suite_cases


SOURCE = ("def fit_predict(X_fit,y_fit,X_predict,context):\n"
          "    return {'predictions': [0.] * len(X_predict)}\n")


class TemporaryEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def state(self, categories=None):
        source = self.root / "baseline.py"
        source.write_text(SOURCE, encoding="utf-8")
        baseline = freeze_candidate(source, {"family": "fixture", "components": [
            {"action": "fixture", "model_config": {"depth": 2}}]}, self.root / "baseline")
        if categories is None:
            categories = ["a", "a", "b", "held-only"]
        frame = pd.DataFrame({"x": [str(i) for i in range(len(categories))], "kind": categories})
        run_dir = self.root / "run"
        run_dir.mkdir()
        return SimpleNamespace(
            baseline=baseline, baseline_inner=SimpleNamespace(metric={"value": 0.5}),
            selected_candidate=baseline, best_inner={"value": 0.5}, dialogue_closed=False,
            gate=SimpleNamespace(task_semantics={"adaptive_research_allowed": True},
                                 fit_rows=list(range(len(frame) - 1)), held_rows=[len(frame) - 1],
                                 inner_split={"policy": "fixture"}, features=["x", "kind"],
                                 categorical=["kind"], research_train=frame, check_inputs=Mock()),
            report={"stage": "inner_research"}, run_dir=run_dir, research_timeout=Mock(return_value=20.0),
            check_source=Mock(), save=Mock(), event=Mock())

    @staticmethod
    def feedback(candidate, accepted=False):
        return {"accepted": accepted, "verified": True, "reason": "improved" if accepted else "not_better",
                "scope": "inner_selection", "origin": "frequency", "code_hash": candidate.code_hash,
                "recipe_hash": candidate.recipe_hash, "metric": {"value": 0.4}}


class NativeHookTests(TemporaryEvidence):
    def test_control_calls_original_queue_once_without_frequency_attempt(self):
        state = self.state()
        marker = object()
        baseline_bytes = state.baseline.code_path.read_bytes()
        recipe_bytes = state.baseline.recipe_path.read_bytes()
        continuation = Mock(return_value=marker)
        with patch.object(host, "continue_locally", continuation), patch.object(experiment, "frequency_attempt") as attempt:
            with experiment.installed_hooks("control", {}, self.root / "observation.json") as observation:
                self.assertIs(host.continue_locally(state), marker)
        continuation.assert_called_once_with(state)
        attempt.assert_not_called()
        self.assertNotIn("frequency_experiment", state.report)
        self.assertEqual(observation["continuations"], 1)
        self.assertEqual(observation["baseline_before"], observation["baseline_after"])
        self.assertEqual(state.baseline.code_path.read_bytes(), baseline_bytes)
        self.assertEqual(state.baseline.recipe_path.read_bytes(), recipe_bytes)

    def test_frequency_delegates_native_rejection_then_unchanged_continuation(self):
        state = self.state()
        baseline = state.baseline
        order = []
        feedbacks = []
        def assess(actual_state, candidate, hypothesis, *, origin):
            self.assertIs(actual_state, state)
            self.assertEqual(origin, "frequency")
            self.assertIn("hypothesis", hypothesis)
            self.assertEqual(candidate.recipe["base_recipe"], baseline.recipe)
            self.assertEqual(candidate.recipe["frequency"]["categorical_columns"], ["kind"])
            order.append("native_assessment")
            feedbacks.append(self.feedback(candidate))
            return feedbacks[-1]
        def continuation(actual_state):
            order.append("ordinary_queue")
            self.assertIs(actual_state.selected_candidate, baseline)
            return "native queue result"
        with patch.object(host, "_assess_candidate", side_effect=assess) as native_assess, \
                patch.object(host, "continue_locally", side_effect=continuation):
            with experiment.installed_hooks("frequency", {}, self.root / "observation.json") as observation:
                result = host.continue_locally(state)
        self.assertEqual(result, "native queue result")
        self.assertEqual(order, ["native_assessment", "ordinary_queue"])
        native_assess.assert_called_once()
        self.assertIs(state.report["frequency_experiment"]["feedback"], feedbacks[0])
        self.assertIs(observation["admissions"][0]["feedback"], feedbacks[0])
        self.assertFalse(state.report["frequency_experiment"]["feedback"]["accepted"])
        self.assertIs(state.selected_candidate, baseline)

    def test_native_acceptance_mutation_is_preserved_without_extra_assessment(self):
        state = self.state()
        seen = []
        def assess(actual_state, candidate, hypothesis, *, origin):
            seen.append(candidate)
            actual_state.selected_candidate = candidate
            return self.feedback(candidate, accepted=True)
        with patch.object(host, "_assess_candidate", side_effect=assess) as native_assess:
            result = experiment.frequency_attempt(state)
        native_assess.assert_called_once()
        self.assertIs(state.selected_candidate, seen[0])
        self.assertTrue(result["feedback"]["accepted"])
        self.assertEqual(result["attempted"], 1)

    def test_no_categorical_columns_skip_before_transform_freeze_or_fit(self):
        state = self.state()
        state.gate.categorical = []
        with patch("agent.frequency_candidate.fit_frequency_transform") as transform, \
                patch("agent.research_candidate.freeze_candidate") as freeze, \
                patch.object(host, "_assess_candidate") as assess:
            result = experiment.frequency_attempt(state)
        self.assertEqual(result["reason"], "no_categorical_features")
        self.assertEqual(result["attempted"], 0)
        transform.assert_not_called(); freeze.assert_not_called(); assess.assert_not_called()
        self.assertFalse((state.run_dir / "frequency_source.py").exists())

    def test_column_and_cardinality_limits_refuse_before_native_model_call(self):
        for kind in ("columns", "cardinality"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                old_root = self.root
                self.root = Path(directory)
                state = self.state(categories=[str(i) for i in range(4098)] if kind == "cardinality" else None)
                self.root = old_root
                if kind == "columns":
                    state.gate.research_train = pd.DataFrame({f"c{i}": ["a", "b", "c", "d"] for i in range(17)})
                    state.gate.features = list(state.gate.research_train.columns)
                    state.gate.categorical = list(state.gate.features)
                baseline = state.selected_candidate
                with patch.object(host, "_assess_candidate") as assess:
                    result = experiment.frequency_attempt(state)
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(result["reason"], "frequency_contract_refusal")
                self.assertEqual(result["attempted"], 0)
                self.assertIs(state.selected_candidate, baseline)
                assess.assert_not_called()
                self.assertFalse((state.run_dir / "frequency_source.py").exists())

    def test_eligibility_reads_only_inner_fitting_features(self):
        # Held values are malformed for this transform and must never be fitted.
        state = self.state(categories=["a", "a", "b", 31])
        with patch.object(host, "_assess_candidate", side_effect=lambda s, c, h, **k: self.feedback(c)) as assess:
            result = experiment.frequency_attempt(state)
        self.assertEqual(result["status"], "completed")
        assess.assert_called_once()

    def test_native_reserve_skip_does_not_build_candidate(self):
        state = self.state()
        state.research_timeout.return_value = 2
        with patch.object(host, "_assess_candidate") as assess:
            result = experiment.frequency_attempt(state)
        self.assertEqual(result["reason"], "finalization_reserve")
        self.assertFalse((state.run_dir / "frequency_frozen").exists())
        assess.assert_not_called()

    def test_failed_attempt_keeps_baseline_and_continues_native_queue(self):
        state = self.state()
        baseline = state.baseline
        marker = object()
        continuation = Mock(return_value=marker)
        with patch.object(host, "_assess_candidate", side_effect=RuntimeError("fixture worker failed")), \
                patch.object(host, "continue_locally", continuation):
            with experiment.installed_hooks("frequency", {}, self.root / "observation.json"):
                self.assertIs(host.continue_locally(state), marker)
        self.assertEqual(state.report["frequency_experiment"]["status"], "failed")
        self.assertIs(state.baseline, baseline)
        self.assertIs(state.selected_candidate, baseline)
        baseline.verify()
        continuation.assert_called_once_with(state)

    def test_baseline_mutation_is_rejected_by_final_guard(self):
        state = self.state()
        def corrupt(actual_state, candidate, hypothesis, *, origin):
            actual_state.baseline.recipe_path.write_text('{"changed":true}', encoding="utf-8")
            return self.feedback(candidate)
        with patch.object(host, "_assess_candidate", side_effect=corrupt):
            with self.assertRaises(ValueError):
                experiment.frequency_attempt(state)

    def test_all_patches_restore_after_original_queue_exception(self):
        state = self.state()
        continuation = Mock(side_effect=KeyboardInterrupt("fixture interruption"))
        with patch.object(host, "continue_locally", continuation):
            originals = tuple(getattr(host, key) for key in ("ResearchWorkspace", "run_candidate", "continue_locally", "_assess_candidate"))
            with self.assertRaises(KeyboardInterrupt):
                with experiment.installed_hooks("control", {}, self.root / "observation.json"):
                    host.continue_locally(state)
            self.assertEqual(tuple(getattr(host, key) for key in ("ResearchWorkspace", "run_candidate", "continue_locally", "_assess_candidate")), originals)
        self.assertEqual(experiment.read_json(self.root / "observation.json")["continuations"], 1)

    def test_native_result_is_verified_and_returned_by_identity(self):
        state = self.state()
        result = SimpleNamespace(verify=Mock(), metric={"value": 1.0}, execution={"exit_code": 0},
                                 code_hash=state.baseline.code_hash, recipe_hash=state.baseline.recipe_hash,
                                 phase="inner_selection", output_path=self.root / "predictions.csv",
                                 receipt_path=self.root / "receipt.json", output_hash="a" * 64,
                                 receipt_hash="b" * 64, output_size=40)
        output = self.root / "outputs"
        with patch.object(host, "run_candidate", return_value=result) as native:
            with experiment.installed_hooks("control", {}, self.root / "observation.json") as observation:
                returned = host.run_candidate(state.baseline, output_dir=output, phase="inner_selection", timeout=7.5,
                                               fit_frame=pd.DataFrame({"x": [1, 2]}), predict_frame=pd.DataFrame({"x": [3]}))
        self.assertIs(returned, result)
        result.verify.assert_called_once_with()
        native.assert_called_once()
        self.assertTrue(observation["invocations"][0]["verified"])
        self.assertEqual(observation["invocations"][0]["timeout_seconds"], 7.5)

    def test_invalid_or_forged_native_admission_feedback_is_rejected(self):
        state = self.state()
        valid = self.feedback(state.baseline)
        for mutation in ({"accepted": 1}, {"verified": False}, {"scope": "outer_validation"},
                         {"origin": "llm"}, {"code_hash": "0" * 64}, {"recipe_hash": "0" * 64}):
            with self.subTest(mutation=mutation):
                with patch.object(host, "_assess_candidate", return_value={**valid, **mutation}):
                    with experiment.installed_hooks("control", {}, self.root / "observation.json"):
                        with self.assertRaises(ValueError):
                            host._assess_candidate(state, state.baseline, {}, origin="frequency")


class WorkerBudgetTests(TemporaryEvidence):
    def config(self):
        self.environment = {"python": "fixture"}
        self.sources = {"fixture.py": "a" * 64}
        self.snapshot = {"frozen_split": {"fit_rows": [0], "validation_rows": [1]}}
        config = {"arm": "control", "arm_cap_seconds": 90, "environment": self.environment,
                  "sources": self.sources, "snapshot": self.snapshot, "task": str(self.root / "task.md")}
        path = self.root / "config.json"
        experiment.write_json(path, config)
        return path

    def fake_hooks(self, arm, split, path):
        @contextmanager
        def hooks():
            self.assertEqual(arm, "control")
            self.assertEqual(split, self.snapshot["frozen_split"])
            experiment.write_json(path, {"gate": self.snapshot})
            yield {"gate": self.snapshot}
        return hooks()

    def test_bootstrap_clock_precedes_driver_and_host_imports(self):
        code = experiment.child_bootstrap(self.root / "config.json")
        self.assertLess(code.index("native_started=time.monotonic()"), code.index("import sys"))
        self.assertLess(code.index("native_started=time.monotonic()"), code.index("from experiment_frequency_candidate"))
        compile(code, "bootstrap.py", "exec")

    def test_worker_passes_only_remaining_native_clock_budget_and_real_v3_flags(self):
        config = self.config()
        run_dir = self.root / "native_run"
        experiment.write_json(run_dir / "run.json", {"fixture": "no fit"})
        with patch.object(experiment, "environment_versions", return_value=self.environment), \
                patch.object(experiment, "source_hashes", return_value=self.sources), \
                patch.object(experiment, "installed_hooks", side_effect=self.fake_hooks), \
                patch.object(experiment.time, "monotonic", side_effect=[14.0, 20.0]), \
                patch.object(host, "run_agent_v3", return_value=run_dir) as native:
            experiment.worker_run(config, native_started=10.0)
        native.assert_called_once_with(self.root / "task.md", client=None, local_only=True, time_budget=86.0)
        summary = experiment.read_json(self.root / "worker_summary.json")
        self.assertEqual(summary["native_elapsed_seconds"], 10.0)
        self.assertEqual(summary["host_budget_seconds"], 86.0)
        self.assertEqual(summary["status"], "returned")
        self.assertEqual(summary["host_run_hash"], experiment.file_hash(run_dir / "run.json"))

    def test_import_time_exhaustion_prevents_native_run(self):
        config = self.config()
        with patch.object(experiment, "environment_versions", return_value=self.environment), \
                patch.object(experiment, "source_hashes", return_value=self.sources), \
                patch.object(experiment, "installed_hooks", side_effect=self.fake_hooks), \
                patch.object(experiment.time, "monotonic", side_effect=[101.0, 102.0]), \
                patch.object(host, "run_agent_v3") as native:
            with self.assertRaisesRegex(ValueError, "imports exhausted"):
                experiment.worker_run(config, native_started=10.0)
        native.assert_not_called()
        summary = experiment.read_json(self.root / "worker_summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["native_elapsed_seconds"], 92.0)

    def test_child_source_and_environment_drift_fail_before_host_run(self):
        for kind in ("sources", "environment"):
            with self.subTest(kind=kind):
                config = self.config()
                with patch.object(experiment, "environment_versions", return_value={"drift": True} if kind == "environment" else self.environment), \
                        patch.object(experiment, "source_hashes", return_value={"drift": True} if kind == "sources" else self.sources), \
                        patch.object(host, "run_agent_v3") as native:
                    with self.assertRaises(ValueError):
                        experiment.worker_run(config, native_started=0.0)
                native.assert_not_called()
                self.assertEqual(experiment.read_json(self.root / "worker_summary.json")["status"], "failed")


class SealedPlanTests(TemporaryEvidence):
    def setUp(self):
        super().setUp()
        self.source = {"fixture.py": "a" * 64}
        self.environment = {"python": "pinned-fixture"}
        self.inputs = {"train.csv": "b" * 64}
        self.plan = {"version": 1, "base_sha": experiment.BASE_SHA, "sources": self.source,
                     "environment": self.environment, "method": {"version": 1},
                     "case_definitions": {"fixture": True}, "entries": [], "public": {"entries": []}}
        for index, descriptor in enumerate(suite_cases()):
            audit = self.root / "audit" / f"{index}.json"
            experiment.write_json(audit, {"fixture_answers": ["01", "NA"]})
            snapshot = {"frozen_split": {"fit_rows": [0], "validation_rows": [1]}}
            self.plan["entries"].append({**descriptor, "key": f"fixture_{index}", "template": f"cases/{index}",
                                         "snapshot": snapshot, "snapshot_hash": experiment.json_hash(snapshot),
                                         "input_hashes": self.inputs, "audit": str(audit.relative_to(self.root)),
                                         "audit_hash": experiment.file_hash(audit)})
        self.seal()
        for patcher in (patch.object(experiment, "_output_root", side_effect=Path),
                        patch.object(experiment, "source_hashes", return_value=self.source),
                        patch.object(experiment, "environment_versions", return_value=self.environment),
                        patch.object(experiment, "_input_hashes", return_value=self.inputs)):
            patcher.start(); self.addCleanup(patcher.stop)

    def seal(self):
        self.plan["method_sha"] = experiment.json_hash({"sources": self.plan["sources"], "method": self.plan["method"],
                                                       "cases": self.plan["case_definitions"], "entries": self.plan["entries"],
                                                       "public": self.plan["public"]})
        experiment.write_json(self.root / "plan.json", self.plan)
        experiment.write_json(self.root / "preflight.json", {"plan_hash": experiment.file_hash(self.root / "plan.json"),
                                                            "method_sha": self.plan["method_sha"]})

    def test_sealed_plan_loads_with_no_training(self):
        with patch.object(host, "run_agent_v3") as native:
            self.assertEqual(experiment.load_plan(self.root), self.plan)
        native.assert_not_called()

    def test_raw_plan_method_source_environment_and_input_drift_rejected(self):
        for kind in ("raw_plan", "method", "source", "environment", "input", "audit", "snapshot"):
            with self.subTest(kind=kind):
                original = deepcopy(self.plan)
                self.seal()
                with patch.object(experiment, "source_hashes", return_value={"drift": True} if kind == "source" else self.source), \
                        patch.object(experiment, "environment_versions", return_value={"drift": True} if kind == "environment" else self.environment), \
                        patch.object(experiment, "_input_hashes", return_value={"drift": True} if kind == "input" else self.inputs):
                    if kind == "raw_plan":
                        (self.root / "plan.json").write_bytes((self.root / "plan.json").read_bytes() + b" ")
                    elif kind == "method":
                        self.plan["method"]["version"] = 9
                        experiment.write_json(self.root / "plan.json", self.plan)
                        preflight = experiment.read_json(self.root / "preflight.json")
                        preflight["plan_hash"] = experiment.file_hash(self.root / "plan.json")
                        experiment.write_json(self.root / "preflight.json", preflight)
                    elif kind in ("audit", "snapshot"):
                        entry = self.plan["entries"][0]
                        if kind == "audit":
                            entry["audit_hash"] = "0" * 64
                        else:
                            entry["snapshot_hash"] = "0" * 64
                        self.seal()
                    with self.assertRaises(ValueError):
                        experiment.load_plan(self.root)
                self.plan = original

    def test_missing_or_reordered_pair_plan_rejected_even_when_resealed(self):
        for kind in ("missing", "reordered"):
            with self.subTest(kind=kind):
                original = deepcopy(self.plan)
                if kind == "missing": self.plan["entries"].pop()
                else: self.plan["entries"] = list(reversed(self.plan["entries"]))
                self.seal()
                with self.assertRaisesRegex(ValueError, "Incomplete or reordered"):
                    experiment.load_plan(self.root)
                self.plan = original


class StrictEvidenceTests(TemporaryEvidence):
    def test_duplicate_and_nonfinite_json_are_rejected(self):
        path = self.root / "evidence.json"
        for raw in ('{"nonce":"first","nonce":"second"}', '{"value":NaN}', '{"value":Infinity}'):
            with self.subTest(raw=raw):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError): experiment.read_json(path)

    def test_regular_file_hash_rejects_missing_or_symlink_evidence(self):
        path = self.root / "file"
        with self.assertRaises(ValueError): experiment.file_hash(path)
        path.write_bytes(b"fixture")
        link = self.root / "link"
        try:
            link.symlink_to(path)
        except OSError as error:
            self.skipTest("This environment cannot create symlinks: " + str(error))
        with self.assertRaises(ValueError): experiment.file_hash(link)

    def test_require_and_strict_json_fail_closed_under_optimized_python(self):
        # Spawn no worker and train nothing: -O must not remove evidence guards.
        path = self.root / "duplicate.json"
        path.write_text('{"nonce":"a","nonce":"b"}', encoding="utf-8")
        code = ("import experiment_frequency_candidate as e\n"
                "checks = [lambda: e.require(False, 'guard'), lambda: e.read_json(" + repr(str(path)) + ")]\n"
                "for check in checks:\n"
                "    try: check()\n"
                "    except ValueError: pass\n"
                "    else: raise SystemExit('Guard disabled under -O')\n")
        result = subprocess.run([sys.executable, "-O", "-c", code], cwd=experiment.ROOT,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


class CompletedArmAuditTests(TemporaryEvidence):
    """Complete artificial protocol evidence; no generated source is executed."""
    def setUp(self):
        super().setUp()
        self.arm = self.root / "runs" / "fixture" / "control"
        self.task = self.arm / "task" / "task.md"
        self.data = self.task.parent / "inputs"
        self.data.mkdir(parents=True)
        self.task.write_text("Fake protocol fixture. No solver or model may run.\n", encoding="utf-8")
        self.spec = TaskSpec(task_path=self.task, data_dir=self.data, id_column="row", target_column="answer",
                             features=("x", "kind"), numeric=("x",), categorical=("kind",),
                             classes=("01", "NA", "class c"), output_columns=("row", "answer"),
                             probability_columns=(), positive_class=None, metric=get_metric("Macro F1"))
        self.schema = self.spec.to_dict()
        self.schema.pop("task_path"); self.schema.pop("data_dir")
        self.train = pd.DataFrame({"row": ["fit0", "fit1", "fit2", "fit3"], "x": ["0", "1", "2", "3"],
                                   "kind": ["a", "a", "b", ""], "answer": ["01", "NA", "01", "NA"]})
        self.test = pd.DataFrame({"row": ["0007", "NA", "audit c"], "x": ["9", "8", "7"],
                                  "kind": ["a", "new", ""]})
        self.train.to_csv(self.data / "train.csv", index=False)
        self.test.to_csv(self.data / "test.csv", index=False)
        pd.DataFrame({"name": ["x", "kind"], "type": ["numeric", "categorical"]}).to_csv(self.data / "data_dictionary.csv", index=False)
        self.spec.prediction_frame(self.test["row"].tolist(), ["01"] * 3).to_csv(self.data / "sample_submission.csv", index=False)
        self.input_hashes = {"task.md": experiment.file_hash(self.task), **{
            name: experiment.file_hash(self.data / name) for name in experiment.INPUT_NAMES}}
        self.snapshot = {"schema": self.schema, "frozen_split": {"fit_rows": [0, 1, 2], "validation_rows": [3]},
                         "inner_fit_rows": [0, 1], "inner_validation_rows": [2], "inner_split": {"policy": "fixture"},
                         "inner_fit_ids": ["fit0", "fit1"], "inner_validation_ids": ["fit2"],
                         "outer_fit_ids": ["fit0", "fit1", "fit2"], "outer_validation_ids": ["fit3"],
                         "full_fit_ids": self.train["row"].tolist(), "test_ids": self.test["row"].tolist(),
                         "semantics_allowed": True}
        audit_path = self.root / "audit" / "fixture.json"
        experiment.write_json(audit_path, {"version": 1, "case": "fixture", "ids": self.test["row"].tolist(),
                                            "actuals": ["01", "NA", "class c"], "schema": self.schema, "scope": "external_audit"})
        self.entry = {"key": "fixture", "snapshot": self.snapshot, "input_hashes": self.input_hashes,
                      "arm_cap_seconds": 90, "audit": "audit/fixture.json", "audit_hash": experiment.file_hash(audit_path)}
        self.plan = {"sources": {"fixture.py": "a" * 64}, "environment": {"python": "fake-pinned"}, "method_sha": "b" * 64}
        self.run_dir = self.task.parent / ".agent_runs" / "fixture"
        self.run_dir.mkdir(parents=True)
        source = self.run_dir / "source.py"
        source.write_text(host.BASELINE_SOURCE, encoding="utf-8")
        self.baseline = freeze_candidate(source, seed_candidate(self.spec.classification), self.run_dir / "baseline")
        self.identity = experiment._candidate_identity(self.baseline)
        self.manifest_path = self.run_dir / "research" / "research_validation.json"
        experiment.write_json(self.manifest_path, {"fit_rows": [0, 1], "validation_rows": [2], "split": {"policy": "fixture"}})
        self.records = [self.invocation("inner_selection", [0, 1], [2], 1.0),
                        self.invocation("outer_validation", [0, 1, 2], [3], 0.0),
                        self.invocation("full_refit", [0, 1, 2, 3], [0, 1, 2], None)]
        self.observation = {"version": 1, "arm": "control", "gate": self.snapshot, "continuations": 1,
                            "baseline_before": self.identity, "baseline_after": self.identity,
                            "baseline_inner": self.records[0]["metric"], "admissions": [], "invocations": self.records,
                            "manifest": {"path": str(self.manifest_path), "sha256": experiment.file_hash(self.manifest_path)}}
        simple_identity = {key: self.identity[key] for key in ("code_hash", "recipe_hash")}
        self.host_report = {"research_protocol_version": 3, "status": "completed", "time_budget_seconds": 89,
                            "publication": {"new_result_published": True, "recipe": "baseline"}, "published_recipe": "baseline",
                            "api": {"attempts": 0, "responses": 0}, "frozen_baseline": simple_identity,
                            "baseline_search": {"scope": "outer_fit_only", "status": "seed_only"},
                            "candidate_receipts": {**simple_identity, "outer": self.records[1]["receipt_path"],
                                                   "refit": self.records[2]["receipt_path"]},
                            "evaluations": {"final_check": {"scope": "final_check", "selection_uses_outer": True,
                                                               "unbiased_estimate": False, "baseline": self.records[1]["metric"]}}}
        self.worker = {"version": 1, "arm": "control", "status": "returned", "sources": self.plan["sources"],
                       "environment": self.plan["environment"], "native_started": 10.0, "host_started": 11.0,
                       "native_ended": 15.0, "native_elapsed_seconds": 5.0, "host_budget_seconds": 89,
                       "external_model_api_calls": 0, "platform_submissions": 0, "run_dir": str(self.run_dir)}
        self.config = {"version": 1, "arm": "control", "task": str(self.task), "arm_cap_seconds": 90,
                       "snapshot": self.snapshot, "sources": self.plan["sources"], "environment": self.plan["environment"],
                       "method_sha": self.plan["method_sha"]}
        experiment.write_json(self.arm / "config.json", self.config)
        experiment.write_json(self.arm / "launcher.json", {"exit_code": 0, "timed_out": False,
                                                             "memory_limit_exceeded": False, "elapsed_seconds": 6.0})
        shutil.copyfile(self.records[2]["output_path"], self.task.parent / "submission.csv")
        shutil.copyfile(self.records[2]["output_path"], self.run_dir / "last_valid_submission.csv")
        self.summary = {"status": "completed", "arm": "control", "case": "fixture",
                        "directory": str(self.arm.relative_to(self.root))}
        self.seal_artifacts()
        for patcher in (patch.object(experiment, "_output_root", side_effect=Path),
                        patch.object(experiment, "_input_hashes", return_value=self.input_hashes)):
            patcher.start(); self.addCleanup(patcher.stop)

    def invocation(self, phase, fit_rows, predict_rows, metric_value):
        directory = self.run_dir / (phase + "_fake")
        directory.mkdir()
        fitting = self.train.iloc[fit_rows]
        predicting = (self.test if phase == "full_refit" else self.train).iloc[predict_rows]
        shutil.copyfile(self.baseline.code_path, directory / "candidate.py")
        shutil.copyfile(self.baseline.recipe_path, directory / "recipe.json")
        fitting.loc[:, ["row", "x", "kind", "answer"]].to_csv(directory / "fit.csv", index=False)
        predicting.loc[:, ["row", "x", "kind"]].to_csv(directory / "predict.csv", index=False)
        self.spec.prediction_frame(predicting["row"].tolist(), ["01"] * len(predicting)).to_csv(directory / "predictions.csv", index=False)
        context = {"version": 3, "protocol_version": 3, "phase": phase, "nonce": "fake-" + phase,
                   "code_hash": self.baseline.code_hash, "recipe_hash": self.baseline.recipe_hash,
                   "recipe": self.baseline.recipe, "task_type": self.spec.metric.task_type,
                   "prediction_kind": self.spec.prediction_kind, "task_spec": self.spec.to_dict(),
                   **{key: self.schema[key] for key in ("features", "numeric", "categorical", "classes", "positive_class", "metric")}}
        experiment.write_json(directory / "context.json", context)
        experiment.write_json(directory / "receipt.json", {"version": 3, "phase": phase, "nonce": context["nonce"],
            "code_hash": self.baseline.code_hash, "recipe_hash": self.baseline.recipe_hash,
            "output_hash": experiment.file_hash(directory / "predictions.csv")})
        return {"index": {"inner_selection": 0, "outer_validation": 1, "full_refit": 2}[phase], "phase": phase,
                "candidate": self.identity, "verified": True, "fit_rows": len(fitting), "predict_rows": len(predicting),
                "metric": None if metric_value is None else {"name": "Macro F1", "value": metric_value, "higher_is_better": True},
                "execution": {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False},
                "code_hash": self.baseline.code_hash, "recipe_hash": self.baseline.recipe_hash,
                "output_path": str(directory / "predictions.csv"), "receipt_path": str(directory / "receipt.json"),
                "output_hash": experiment.file_hash(directory / "predictions.csv"),
                "receipt_hash": experiment.file_hash(directory / "receipt.json"), "output_size": (directory / "predictions.csv").stat().st_size,
                "invocations": [{"directory": str(directory), "files": experiment._invocation_files(directory)}]}

    def seal_artifacts(self):
        experiment.write_json(self.arm / "observation.json", self.observation)
        experiment.write_json(self.run_dir / "run.json", self.host_report)
        self.worker.update(config_hash=experiment.file_hash(self.arm / "config.json"),
                           observation_hash=experiment.file_hash(self.arm / "observation.json"),
                           host_run_hash=experiment.file_hash(self.run_dir / "run.json"))
        experiment.write_json(self.arm / "worker_summary.json", self.worker)
        self.summary.update({key: experiment.file_hash(path) for key, path in (
            ("config_hash", self.arm / "config.json"), ("launcher_hash", self.arm / "launcher.json"),
            ("worker_summary_hash", self.arm / "worker_summary.json"), ("observation_hash", self.arm / "observation.json"),
            ("host_run_hash", self.run_dir / "run.json"))})

    def audit(self):
        return experiment.audit_arm(self.root, self.entry, "control", self.summary, self.plan)

    def repin_invocation(self, record):
        invocation = record["invocations"][0]
        invocation["files"] = experiment._invocation_files(Path(invocation["directory"]))
        self.seal_artifacts()

    def test_complete_fake_control_receipts_are_reauditable_without_fit_or_repair(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(host, "run_agent_v3", side_effect=AssertionError("Re-audit must not fit")), \
                patch.object(experiment, "write_json", side_effect=AssertionError("Re-audit must not repair")):
            result = self.audit()
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["metric"]["value"], 1 / 6)
        self.assertEqual(result["per_class"]["NA"]["support"], 1)
        self.assertEqual(result["per_class"]["class c"]["f1"], 0.0)
        self.assertIsNone(result["frequency"])
        self.assertGreater(len(result["proof_hashes"]), 20)

    def test_canonical_artifacts_accept_an_existing_noncanonical_output_alias(self):
        # _regular_under returns canonical paths. An output alias must resolve
        # to the same root before relative proof keys are generated (including
        # Windows short/long temp aliases in the ordinary fixture above).
        expected = self.audit()
        alias_step = self.root / "existing_alias_step"
        alias_step.mkdir()
        output_alias = alias_step / ".."
        actual = experiment.audit_arm(output_alias, self.entry, "control", self.summary, self.plan)
        self.assertEqual(actual, expected)
        self.assertTrue(all(not Path(key).is_absolute() for key in actual["proof_hashes"]))

    def test_missing_full_refit_cannot_be_replaced_by_published_csv_claim(self):
        self.observation["invocations"].pop()
        self.seal_artifacts()
        with self.assertRaisesRegex(ValueError, "outer/full-refit"):
            self.audit()

    def test_reused_nonce_rejected_even_after_all_journal_hashes_are_updated(self):
        record = self.records[1]
        context_path = Path(record["invocations"][0]["directory"]) / "context.json"
        context = experiment.read_json(context_path)
        context["nonce"] = "fake-inner_selection"
        experiment.write_json(context_path, context)
        self.repin_invocation(record)
        with self.assertRaisesRegex(ValueError, "nonce"):
            self.audit()

    def test_receipt_nonce_tamper_rejected_even_with_matching_journal_receipt_hash(self):
        record = self.records[0]
        path = Path(record["receipt_path"])
        receipt = experiment.read_json(path)
        receipt["nonce"] = "forged"
        experiment.write_json(path, receipt)
        record["receipt_hash"] = experiment.file_hash(path)
        self.repin_invocation(record)
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            self.audit()

    def test_mutated_context_schema_rejected_with_repinned_artifact(self):
        record = self.records[0]
        path = Path(record["invocations"][0]["directory"]) / "context.json"
        context = experiment.read_json(path)
        context["features"] = ["kind", "x"]
        experiment.write_json(path, context)
        self.repin_invocation(record)
        with self.assertRaisesRegex(ValueError, "schema changed"):
            self.audit()

    def test_mutated_fit_partition_rejected_with_repinned_artifact(self):
        record = self.records[0]
        path = Path(record["invocations"][0]["directory"]) / "fit.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "kind"] = "held-only-leak"
        frame.to_csv(path, index=False)
        self.repin_invocation(record)
        with self.assertRaisesRegex(ValueError, "fit/predict rows or values"):
            self.audit()

    def test_code_hash_and_published_checkpoint_tampering_fail_closed(self):
        code = self.baseline.code_path
        original = code.read_bytes()
        code.write_bytes(original + b"\n# forged source\n")
        with self.assertRaisesRegex(ValueError, "hash drift"):
            self.audit()
        code.write_bytes(original)
        checkpoint = self.run_dir / "last_valid_submission.csv"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "CSV/checkpoint"):
            self.audit()

    def test_forged_printed_metric_does_not_override_saved_csv(self):
        self.records[0]["metric"]["value"] = 0.99
        self.observation["baseline_inner"] = self.records[0]["metric"]
        self.seal_artifacts()
        with self.assertRaisesRegex(ValueError, "differs from saved CSV"):
            self.audit()

    def test_missing_refit_receipt_rejected_under_optimized_python(self):
        self.observation["invocations"].pop()
        self.seal_artifacts()
        fixture = self.root / "audit_request.json"
        experiment.write_json(fixture, {"entry": self.entry, "summary": self.summary, "plan": self.plan})
        code = ("from pathlib import Path\nimport experiment_frequency_candidate as e\n"
                "root=Path(" + repr(str(self.root)) + ")\n"
                "request=e.read_json(root/'audit_request.json')\n"
                "e._output_root=lambda value: Path(value)\n"
                "e._input_hashes=lambda task: request['entry']['input_hashes']\n"
                "try: e.audit_arm(root,request['entry'],'control',request['summary'],request['plan'])\n"
                "except ValueError as error:\n"
                "    if 'outer/full-refit' not in str(error): raise\n"
                "else: raise SystemExit('Missing full refit accepted under -O')\n")
        result = subprocess.run([sys.executable, "-O", "-c", code], cwd=experiment.ROOT,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_frequency_trace_counts_and_augmented_fingerprints_are_independently_checked(self):
        fit = pd.DataFrame({"x": ["0", "1", "2"], "kind": ["a", "a", "NA"]})
        predict = pd.DataFrame({"x": ["9", "8"], "kind": ["a", "unknown"]})
        recipe = make_frequency_recipe(self.baseline.recipe, ["kind"])
        context = {"version": 3, "protocol_version": 3, "phase": "inner_selection", "nonce": "trace-fixture",
                   "features": ["x", "kind"], "numeric": ["x"], "categorical": ["kind"],
                   "recipe": recipe, "code_hash": "c" * 64, "recipe_hash": experiment.json_hash(recipe)}
        def fake_predict(*args):
            return {"predictions": ["01", "01"]}
        _, trace = execute_frequency_candidate(fake_predict, fit, ["01", "NA", "01"], predict, context, return_evidence=True)
        trace["base_source_hash"] = self.baseline.code_hash
        result = experiment._independent_frequency_trace(trace, fit, predict, context, self.identity)
        self.assertEqual(result["fit_rows"], 3)
        self.assertEqual(result["predict_rows"], 2)
        for kind in ("count", "fit_fingerprint", "predict_fingerprint", "base_recipe", "phase"):
            with self.subTest(kind=kind):
                tampered = deepcopy(trace)
                if kind == "count": tampered["transform"]["columns"][0]["counts"][0]["count"] += 1
                elif kind == "fit_fingerprint": tampered["augmented_fit_sha256"] = "0" * 64
                elif kind == "predict_fingerprint": tampered["augmented_predict_sha256"] = "0" * 64
                elif kind == "base_recipe": tampered["base_recipe_hash"] = "0" * 64
                else: tampered["phase"] = "full_refit"
                with self.assertRaises(ValueError):
                    experiment._independent_frequency_trace(tampered, fit, predict, context, self.identity)


class ControllerProtocolTests(TemporaryEvidence):
    """Fake run_arm records exercise pairing/seals, not solver correctness."""
    def setUp(self):
        super().setUp()
        self.plan = {"base_sha": experiment.BASE_SHA, "executable_sha": "c" * 40, "method_sha": "d" * 64,
                     "sources": {"fixture.py": "e" * 64}, "environment": {"python": "fake-pinned"},
                     "entries": [], "public": {"status": "skipped_missing_data", "entries": [],
                                               "checked_paths": ["/explicit/missing/task.md"]}}
        for index, descriptor in enumerate(suite_cases()):
            self.plan["entries"].append({**descriptor, "key": descriptor["name"] + "__" + str(descriptor["seed"]),
                                         "arm_cap_seconds": 90, "order": list(experiment.ARMS if index % 2 == 0 else reversed(experiment.ARMS))})
        experiment.write_json(self.root / "plan.json", self.plan)
        self.calls = []
        for patcher in (patch.object(experiment, "_output_root", side_effect=Path),
                        patch.object(experiment, "load_plan", return_value=self.plan),
                        patch.object(experiment, "source_hashes", return_value=self.plan["sources"]),
                        patch.object(experiment, "environment_versions", return_value=self.plan["environment"]),
                        patch.object(experiment.time, "monotonic", return_value=0.0),
                        patch.object(experiment, "run_arm", side_effect=self.fake_arm),
                        patch.object(experiment, "audit_arm", side_effect=self.fake_audit)):
            patcher.start(); self.addCleanup(patcher.stop)

    def fake_arm(self, output, entry, arm, plan):
        self.calls.append((entry["key"], arm))
        if entry["stage"] == "confirmation":
            self.assertTrue((output / "confirmation_seal.json").is_file())
        directory = Path("runs") / entry["key"] / arm
        audit = {"status": "completed", "metric": {"name": "Macro F1", "value": .5 if arm == "control" else .6,
                                                       "higher_is_better": True},
                 "baseline": {"code_hash": "a" * 64, "recipe_hash": "b" * 64},
                 "per_class": {"01": {"f1": .5 if arm == "control" else .8},
                               "NA": {"f1": .4 if arm == "control" else .3}, "class c": {"f1": .2}}}
        summary = {"version": 1, "case": entry["key"], "arm": arm, "status": "completed",
                   "directory": str(directory), "audit": audit, "artifact_hashes": {}}
        experiment.write_json(output / directory / "arm_summary.json", summary)
        return summary

    @staticmethod
    def fake_audit(output, entry, arm, summary, plan):
        # Only controller structure is exercised here. Full artifact audit has
        # its own complete fixture and tamper checks above.
        if not isinstance(summary.get("audit"), dict):
            raise ValueError("Fake completed arm lacks its audit")
        return summary["audit"]

    def reseal_report(self, report):
        report_path = self.root / "report.json"
        experiment.write_json(report_path, report)
        seal_path = self.root / "synthetic_completion_seal.json"
        seal = experiment.read_json(seal_path)
        seal.update(report_hash=experiment.file_hash(report_path), terminal_status=report["status"])
        experiment.write_json(seal_path, seal)

    def test_alternating_complete_pairs_and_separate_confirmation_seal(self):
        report = experiment.run_prepared(self.root)
        expected_calls = [(entry["key"], arm) for entry in self.plan["entries"] for arm in entry["order"]]
        self.assertEqual(self.calls, expected_calls)
        self.assertEqual(report["status"], "completed")
        self.assertEqual((report["comparable_pairs"], report["completed_arms"]), (8, 16))
        confirmation = experiment.read_json(self.root / "confirmation_seal.json")
        self.assertFalse(confirmation["method_changed_after_development"])
        self.assertEqual(confirmation["development_pairs_hash"], experiment.json_hash(report["pairs"][:4]))
        self.assertEqual(len(confirmation["development_summary_hashes"]), 8)
        self.assertAlmostEqual(report["pairs"][0]["signed_improvement"], .1)
        self.assertAlmostEqual(report["pairs"][0]["per_class_harm"]["NA"], -.1)
        with patch.object(experiment, "run_arm", side_effect=AssertionError("Re-audit cannot run an arm")), \
                patch.object(experiment, "write_json", side_effect=AssertionError("Re-audit cannot repair evidence")):
            result = experiment.re_audit(self.root)
        self.assertTrue(result["verified"])
        self.assertEqual(result["fits"], 0)

    def test_development_summary_paths_use_pair_structure_with_native_separators(self):
        # The Windows spelling must work even when this regression runs on
        # Linux. Selection never searches for POSIX substrings in saved keys.
        for path_type in (PurePosixPath, PureWindowsPath):
            with self.subTest(path_type=path_type.__name__):
                pairs = [{"arms": {arm: {"directory": str(path_type("runs") / case / arm)}
                                    for arm in experiment.ARMS}}
                         for case in ("development_case", "confirmation_case")]
                pairs[0]["arms"]["not_started"] = {"status": "not_started"}
                paths = experiment._arm_summary_paths(path_type("evidence_root"), pairs[:1])
                self.assertEqual(paths, [str(path_type("runs") / "development_case" / arm / "arm_summary.json")
                                         for arm in experiment.ARMS])

    def test_development_summary_seal_drift_is_rejected_after_outer_resealing(self):
        report = experiment.run_prepared(self.root)
        path = self.root / "confirmation_seal.json"
        seal = experiment.read_json(path)
        first = next(iter(seal["development_summary_hashes"]))
        seal["development_summary_hashes"][first] = "0" * 64
        experiment.write_json(path, seal)
        report["confirmation_seal_hash"] = experiment.file_hash(path)
        self.reseal_report(report)
        with self.assertRaisesRegex(ValueError, "Development evidence changed"):
            experiment.re_audit(self.root)

    def test_insufficient_coordinator_reserve_starts_neither_arm(self):
        with patch.object(experiment.time, "monotonic", side_effect=[0.0] + [1621.0] * 40):
            report = experiment.run_prepared(self.root)
        self.assertEqual(self.calls, [])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["abort_reason"], "coordinator_cannot_reserve_two_full_arms")
        self.assertTrue(all(arm["status"] == "not_started" for pair in report["pairs"] for arm in pair["arms"].values()))
        self.assertFalse((self.root / "confirmation_seal.json").exists())
        self.assertEqual(experiment.re_audit(self.root)["completed_arms"], 0)

    def test_exact_two_arm_reserve_allows_full_pair_without_shortening_arm_caps(self):
        with patch.object(experiment.time, "monotonic", side_effect=[0.0, 1620.0] + [1621.0] * 40):
            report = experiment.run_prepared(self.root)
        first = self.plan["entries"][0]
        self.assertEqual(self.calls, [(first["key"], arm) for arm in first["order"]])
        self.assertEqual(report["completed_arms"], 2)
        self.assertEqual(report["comparable_pairs"], 1)
        self.assertTrue(all(entry["arm_cap_seconds"] == 90 for entry in self.plan["entries"]))
        self.assertEqual(experiment.re_audit(self.root)["status"], "incomplete")

    def test_uncertain_child_cleanup_aborts_other_arm_and_all_remaining_pairs(self):
        def censored(output, entry, arm, plan):
            summary = self.fake_arm(output, entry, arm, plan)
            summary.update(status="censored", abort_series=True)
            summary.pop("audit")
            experiment.write_json(output / summary["directory"] / "arm_summary.json", summary)
            return summary
        with patch.object(experiment, "run_arm", side_effect=censored):
            report = experiment.run_prepared(self.root)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["abort_reason"], "censored_process_cleanup_unverified")
        self.assertEqual(experiment.re_audit(self.root)["status"], "incomplete")

    def test_replay_is_refused_without_launch_or_repair(self):
        experiment.run_prepared(self.root)
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "cannot be rerun or repaired"):
            experiment.run_prepared(self.root)
        self.assertEqual(self.calls, [])
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_partial_pair_and_missing_pair_rejected_even_with_new_report_hash(self):
        report = experiment.run_prepared(self.root)
        for kind in ("arm", "pair"):
            with self.subTest(kind=kind):
                tampered = deepcopy(report)
                if kind == "arm": tampered["pairs"][0]["arms"].pop("frequency")
                else: tampered["pairs"].pop()
                self.reseal_report(tampered)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    experiment.re_audit(self.root)

    def test_confirmation_method_drift_rejected_even_with_updated_outer_hashes(self):
        report = experiment.run_prepared(self.root)
        path = self.root / "confirmation_seal.json"
        seal = experiment.read_json(path)
        seal["environment"] = {"python": "changed-after-development"}
        experiment.write_json(path, seal)
        report["confirmation_seal_hash"] = experiment.file_hash(path)
        self.reseal_report(report)
        with self.assertRaisesRegex(ValueError, "before confirmation"):
            experiment.re_audit(self.root)

    def test_public_missing_data_preserves_exact_checked_paths_and_fits_zero(self):
        report = experiment.run_prepared(self.root, suite="public")
        self.assertEqual(self.calls, [])
        self.assertEqual(report["status"], "skipped_missing_data")
        result = experiment.re_audit(self.root, suite="public")
        self.assertEqual(result["checked_paths"], ["/explicit/missing/task.md"])
        self.assertEqual(result["fits"], 0)

    def test_failed_artifact_drift_is_detected_without_promoting_partial_result(self):
        report = experiment.run_prepared(self.root)
        summary = report["pairs"][0]["arms"]["control"]
        summary.update(status="failed", reason="fake partial worker")
        summary.pop("audit")
        directory = self.root / summary["directory"]
        partial = directory / "partial.log"
        partial.write_text("first failure bytes", encoding="utf-8")
        summary["artifact_hashes"] = {"partial.log": experiment.file_hash(partial)}
        experiment.write_json(directory / "arm_summary.json", summary)
        report["pairs"][0] = experiment._pair_summary(self.plan["entries"][0], report["pairs"][0]["arms"])
        # Update the outer seals as an adversary might; the pinned failure log
        # still must be checked independently when re-auditing an incomplete run.
        report["status"] = "incomplete"
        report["comparable_pairs"] = 7
        report["completed_arms"] = 15
        relative = str((directory / "arm_summary.json").relative_to(self.root))
        report["arm_summary_hashes"][relative] = experiment.file_hash(directory / "arm_summary.json")
        self.reseal_report(report)
        partial.write_text("changed failure bytes", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact index drift"):
            experiment.re_audit(self.root)


class ArmLauncherProtocolTests(TemporaryEvidence):
    def setUp(self):
        super().setUp()
        template = self.root / "template"
        template.mkdir()
        (template / "task.md").write_text("Fake worker fixture; no actual model execution.\n", encoding="utf-8")
        self.entry = {"key": "fixture", "template": "template", "input_hashes": {"task.md": "fixture"},
                      "snapshot": {"frozen_split": {}}, "arm_cap_seconds": 90}
        self.plan = {"sources": {}, "environment": {}, "method_sha": "a" * 64}
        for patcher in (patch.object(experiment, "_output_root", side_effect=Path),
                        patch.object(experiment, "_input_hashes", return_value=self.entry["input_hashes"])):
            patcher.start(); self.addCleanup(patcher.stop)

    def test_hard_timeout_is_censored_pinned_and_aborts_when_cleanup_unknown(self):
        launcher = {"exit_code": -9, "timed_out": True, "memory_limit_exceeded": False,
                    "resources": {"cleanup_complete": False}}
        with patch("agent.execution.run_python", return_value=launcher) as worker, patch.object(experiment, "audit_arm") as audit:
            summary = experiment.run_arm(self.root, self.entry, "frequency", self.plan)
        self.assertEqual(summary["status"], "censored")
        self.assertTrue(summary["abort_series"])
        self.assertEqual(summary["reason"], "external_hard_timeout")
        self.assertIn("launcher.json", summary["artifact_hashes"])
        self.assertIn("config.json", summary["artifact_hashes"])
        self.assertNotIn("arm_summary.json", summary["artifact_hashes"])
        self.assertEqual(worker.call_args.kwargs["timeout"], 90)
        self.assertNotIn("audit", experiment.read_json(self.root / summary["directory"] / "config.json"))
        audit.assert_not_called()

    def test_missing_resource_telemetry_cannot_prove_cleanup(self):
        launcher = {"exit_code": -9, "timed_out": True, "memory_limit_exceeded": False}
        with patch("agent.execution.run_python", return_value=launcher):
            summary = experiment.run_arm(self.root, self.entry, "control", self.plan)
        self.assertEqual(summary["status"], "censored")
        self.assertTrue(summary["abort_series"])

    def test_nested_native_timeout_censors_even_when_outer_launcher_exits_zero(self):
        launcher = {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False, "resources": {"cleanup_complete": True}}
        def fake_worker(code, task_dir, launcher_dir, step, *, timeout):
            directory = Path(launcher_dir).parent
            partial = {"invocations": [{"execution": {"exit_code": -9, "timed_out": True,
                        "memory_limit_exceeded": False, "resources": {"cleanup_complete": False}}}]}
            experiment.write_json(directory / "observation.json", partial)
            return launcher
        with patch("agent.execution.run_python", side_effect=fake_worker), patch.object(experiment, "audit_arm") as audit:
            summary = experiment.run_arm(self.root, self.entry, "control", self.plan)
        self.assertEqual(summary["status"], "censored")
        self.assertEqual(summary["reason"], "nested_censored_cleanup_unverified")
        self.assertTrue(summary["abort_series"])
        self.assertTrue(any("observation.invocations[0].execution" in path for path in summary["uncertain_censor_paths"]))
        self.assertIn("observation.json", summary["artifact_hashes"])
        audit.assert_not_called()

    def test_worker_exit_failure_retains_artifacts_and_never_uses_stdout_as_success(self):
        def fake_worker(code, task_dir, launcher_dir, step, *, timeout):
            directory = Path(launcher_dir).parent
            (directory / "fake_stdout.log").write_text('{"accepted":true,"metric":1.0}\n', encoding="utf-8")
            return {"exit_code": 2, "timed_out": False, "memory_limit_exceeded": False,
                    "resources": {"cleanup_complete": True}}
        with patch("agent.execution.run_python", side_effect=fake_worker), patch.object(experiment, "audit_arm") as audit:
            summary = experiment.run_arm(self.root, self.entry, "control", self.plan)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["reason"], "native_process_failed")
        self.assertIn("fake_stdout.log", summary["artifact_hashes"])
        self.assertNotIn("audit", summary)
        audit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
