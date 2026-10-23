"""No-ML native pair ordering, stops, immutable identity and CSV/receipt checks."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import experiment_round5_classification_native as native


def canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.manifest = {"environment": {"executable": "/selected/python"}, "package": "/flat/package",
                         "flat_zip_sha256": "f" * 64, "arms": {}}
        for mode in native.ORDER:
            task = self.output / "runtime" / mode / "task.md"
            task.parent.mkdir(parents=True)
            task.write_text("task")
            self.manifest["arms"][mode] = {"task": str(task)}
        native.write_json(self.output / "run_manifest.json", self.manifest)
        native.write_json(self.output / "smoke/report.json", {"status": "completed", "flat_zip_sha256": "f" * 64})
        self.sequence, self.locks = [], []
        self.addCleanup(patch.stopall)
        patch.object(native, "load_manifest", return_value=self.manifest).start()
        patch.object(native, "coordinator_lock", self.lock).start()
        patch.object(native, "supervise", side_effect=self.supervise).start()
        patch.object(native, "audit_arm", side_effect=self.audit_arm).start()
        patch.object(native, "audit", return_value={"status": "rejected"}).start()

    @contextmanager
    def lock(self, path):
        self.locks.append(str(path))
        self.sequence.append("lock")
        yield
        self.sequence.append("unlock")

    def supervise(self, argv, cwd, output, timeout, memory, *, threads):
        mode = argv[-1]
        self.sequence.append("run_" + mode)
        self.assertEqual(argv[-4:], ["--time-budget", "3600", "--classification-round4-mode", mode])
        self.assertEqual((timeout, memory, threads), (3630, 6 * 1024 ** 3, 8))
        return {"argv": argv, "exit_code": 0, "external_timeout": False,
                "memory_limit_exceeded": False, "cleanup": {"complete": None}}

    def audit_arm(self, output, mode, manifest):
        self.sequence.append("audit_" + mode)
        return {"status": "completed", "arm": mode}

    def test_off_is_verified_before_on_and_global_lock_covers_both(self):
        native.run_prepared(self.output)
        self.assertEqual(self.sequence, ["lock", "lock", "run_off", "audit_off", "run_on", "audit_on", "unlock", "unlock"])
        self.assertEqual(self.locks[-1], str(native.machine_lock_path()))
        self.assertTrue((self.output / "pair_started.json").is_file())

    def test_invalid_off_stops_on_and_retry_is_forbidden(self):
        with patch.object(native, "audit_arm", side_effect=ValueError("OFF receipt mismatch")):
            with self.assertRaisesRegex(ValueError, "OFF receipt mismatch"):
                native.run_prepared(self.output)
        self.assertNotIn("run_on", self.sequence)
        stop = native.read_json(self.output / "stop.json")
        self.assertFalse(stop["retry_allowed"])
        with self.assertRaisesRegex(ValueError, "single-use"):
            native.run_prepared(self.output)

    def test_drift_before_second_arm_prevents_spawn(self):
        with patch.object(native, "load_manifest", side_effect=[self.manifest, self.manifest,
                         self.manifest, ValueError("source drift")]):
            with self.assertRaisesRegex(ValueError, "source drift"):
                native.run_prepared(self.output)
        self.assertIn("run_off", self.sequence)
        self.assertNotIn("run_on", self.sequence)

    def test_internal_timeout_preserves_off_report_but_prevents_on(self):
        with patch.object(native, "audit_arm", return_value={"status": "incomplete", "arm": "off",
                          "internal_execution_stops": ["nested/timed_out"]}):
            with self.assertRaisesRegex(ValueError, "Internal execution timeout"):
                native.run_prepared(self.output)
        self.assertNotIn("run_on", self.sequence)
        self.assertEqual(native.read_json(self.output / "off/report.json")["status"], "incomplete")

    def test_missing_flat_smoke_prevents_any_native_spawn(self):
        (self.output / "smoke/report.json").unlink()
        with self.assertRaises(ValueError):
            native.run_prepared(self.output)
        self.assertFalse(any(value.startswith("run_") for value in self.sequence))
        self.assertFalse((self.output / "pair_started.json").exists())

    def test_common_command_differs_only_in_mode_and_task_path(self):
        off, on = (native.arm_command(self.manifest, mode) for mode in native.ORDER)
        self.assertEqual([i for i, (a, b) in enumerate(zip(off, on)) if a != b], [2, len(off) - 1])

    def test_external_timeout_memory_and_failed_cleanup_rejected_unknown_is_preserved(self):
        good = {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                "cleanup": {"complete": None}}
        self.assertTrue(native.execution_ok(good))
        for update in ({"external_timeout": True}, {"memory_limit_exceeded": True},
                       {"exit_code": 1}, {"cleanup": {"complete": False}}):
            self.assertFalse(native.execution_ok({**good, **update}))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.task = self.output / "runtime" / "task"
        self.data = self.task / "inputs"
        self.run = self.task / ".agent_runs" / ("c" * 32)
        self.run.mkdir(parents=True)
        self.data.mkdir()
        (self.task / "task.md").write_text("fixture")
        (self.data / "train.csv").write_text("row,x,label\n0,10,a\n1,20,b\n2,30,a\n3,40,b\n")
        (self.data / "test.csv").write_text("row,x\n4,50\n5,60\n")
        self.metric = {"name": "Macro F1", "higher_is_better": True, "task_type": "classification", "prediction_kind": "labels"}
        self.spec = {"version": 1, "id_column": "row", "target_column": "label", "features": ["x"],
                     "numeric": ["x"], "categorical": [], "classes": ["a", "b"], "output_columns": ["row", "label"],
                     "probability_columns": [], "positive_class": None, "metric": self.metric, "prediction_kind": "labels"}
        self.manifest = {"contract": self.spec, "environment": {"executable": "/selected/python"},
                         "package": "/flat/package", "source": {"sha": "s", "tree": "t"},
                         "arms": {"off": {"task": str(self.task / "task.md")}}}
        native.write_json(self.output / "frozen_identity.json", {"source_sha256": "f" * 64})
        self.partition = {"outer_fit": [0, 1], "outer_held": [2, 3], "inner_fit": [0], "inner_held": [1],
                          "outer_metadata": {"strategy": "grouped", "seed": 42}}
        native.write_json(self.output / "host_only/split.json", self.partition)
        (self.output / "host_only/outer_answers.csv").write_text("row,label\n2,a\n3,b\n")
        self.recipe = {"action": "fixture"}
        source = "def fit_predict(*args): return {}\n"
        self.frozen = self.run / "baseline_frozen"
        self.frozen.mkdir()
        (self.frozen / "research_candidate.py").write_text(source)
        canonical(self.frozen / "research_recipe.json", self.recipe)
        self.identity = {"code_hash": native.file_hash(self.frozen / "research_candidate.py"),
                         "recipe_hash": native.file_hash(self.frozen / "research_recipe.json")}
        native.write_json(self.frozen / "frozen_candidate.json", {"version": 3, **self.identity})
        self.outer = self.phase("outer_evaluations", "outer_validation", "a" * 32,
                               "row,x,label\n0,10,a\n1,20,b\n", "row,x\n2,30\n3,40\n", "row,label\n2,a\n3,b\n")
        self.refit = self.phase("full_refit", "full_refit", "b" * 32,
                               (self.data / "train.csv").read_text(), (self.data / "test.csv").read_text(), "row,label\n4,a\n5,b\n")
        (self.task / "submission.csv").write_bytes((self.refit / "predictions.csv").read_bytes())
        (self.run / "last_valid_submission.csv").write_bytes((self.refit / "predictions.csv").read_bytes())
        self.report = {"research_protocol_version": 3, "mode": "local_only", "stage": "result", "status": "completed",
                       "publication": {"new_result_published": True}, "output_check": {"valid": True},
                       "api": {"attempts": 0, "responses": 0}, "quality_options": {"classification_round4_mode": "off"},
                       "coordinator": {"active": True}, "time_budget_seconds": 3500, "elapsed_seconds": 30,
                       "host_validation": {"split": self.partition["outer_metadata"]}, "published_recipe": "baseline",
                       "frozen_baseline": self.identity, "candidate_receipts": {**self.identity,
                           "outer": str(self.outer / "receipt.json"), "refit": str(self.refit / "receipt.json")},
                       "evaluations": {"final_check": {"baseline": {"name": "Macro F1", "value": 1.0}}}}
        self.save_report()
        native.write_json(self.output / "off/execution.json", {"argv": native.arm_command(self.manifest, "off"),
                          "exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                          "cleanup": {"complete": None}, "resources": {"available": False, "peak_rss_bytes": None}})
        self.addCleanup(patch.stopall)
        patch("agent.task_spec.load_task", return_value=SimpleNamespace(data_dir=self.data)).start()

    def phase(self, folder, phase, nonce, fit_csv, predict_csv, predictions):
        directory = self.run / folder / "baseline" / phase
        directory.mkdir(parents=True)
        (directory / "candidate.py").write_bytes((self.frozen / "research_candidate.py").read_bytes())
        (directory / "recipe.json").write_bytes((self.frozen / "research_recipe.json").read_bytes())
        for name, content in (("fit.csv", fit_csv), ("predict.csv", predict_csv), ("predictions.csv", predictions)):
            (directory / name).write_text(content)
        context = {"version": 3, "protocol_version": 3, "nonce": nonce, "phase": phase,
                   **self.identity, "recipe": self.recipe, "task_spec": {**self.spec,
                      "data_dir": str(directory), "task_path": str(directory / "task.md")},
                   **{key: self.spec[key] for key in ("features", "numeric", "categorical", "classes", "positive_class", "metric")},
                   "prediction_kind": "labels", "task_type": "classification"}
        native.write_json(directory / "context.json", context)
        native.write_json(directory / "receipt.json", {"version": 3, "nonce": nonce, "phase": phase,
                          **self.identity, "output_hash": native.file_hash(directory / "predictions.csv")})
        return directory

    def save_report(self):
        native.write_json(self.run / "run.json", self.report)

    def audit(self):
        return native.audit_arm(self.output, "off", self.manifest)

    def test_full_publication_chain_and_independent_outer_macro_f1(self):
        result = self.audit()
        self.assertEqual(result["scores"]["macro_f1"], 1.)
        self.assertEqual(result["scores"]["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(result["phase_proof"]["refit"]["fit_rows"], 4)
        self.assertEqual(result["test_quality"], "unknown")
        self.assertEqual(result["resource_qualification"], "provisional_unknown_telemetry")

    def test_completed_inner_preserves_full_frozen_identity_for_custom_audit(self):
        import hashlib
        from agent.round4_classification_candidate import candidate_source, make_recipe
        from agent.round4_classification_probability import fixed_incumbent_recipe
        from agent.round5_classification_host import dependency_snapshot

        classes, source = self.spec["classes"], candidate_source()
        recipe = make_recipe(fixed_incumbent_recipe(), classes, variant="oof")
        frozen_identity = {"classes": classes, "source": source,
                           "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                           "recipe": recipe, "recipe_sha256": native.json_hash(recipe),
                           "identity_probe_recipe": make_recipe(fixed_incumbent_recipe(), classes, variant="identity"),
                           "dependency_snapshot": dependency_snapshot()}
        native.write_json(self.output / "frozen_identity.json", frozen_identity)
        self.phase("inner_evaluations", "inner_selection", "d" * 32,
                   "row,x,label\n0,10,a\n", "row,x\n1,20\n", "row,label\n1,b\n")
        with patch.object(native, "audit_custom_children", wraps=native.audit_custom_children) as custom_audit:
            result = self.audit()
        self.assertEqual(len(result["inner_invocations"]), 1)
        self.assertEqual(result["inner_invocations"]["0"]["nonce"], "d" * 32)
        self.assertEqual(custom_audit.call_args.args[3], frozen_identity)

    def test_wrong_outer_rows_rejected_even_with_valid_output_receipt(self):
        (self.outer / "fit.csv").write_text("row,x,label\n1,20,b\n0,10,a\n")
        with self.assertRaisesRegex(ValueError, "input rows/values/order"):
            self.audit()

    def test_receipt_nonce_tampering_rejected(self):
        receipt = native.read_json(self.refit / "receipt.json")
        receipt["nonce"] = "f" * 32
        native.write_json(self.refit / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Receipt is not bound"):
            self.audit()

    def test_outer_refit_nonce_reuse_rejected(self):
        for filename in ("context.json", "receipt.json"):
            value = native.read_json(self.refit / filename)
            value["nonce"] = "a" * 32
            native.write_json(self.refit / filename, value)
        with self.assertRaisesRegex(ValueError, "nonce reused"):
            self.audit()

    def test_claimed_quality_is_recomputed(self):
        self.report["evaluations"]["final_check"]["baseline"]["value"] = .9
        self.save_report()
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            self.audit()

    def test_prior_output_and_missing_api_counters_not_qualified(self):
        for change in ({"status": "fallback"}, {"api": {"attempts": 0}}, {"published_recipe": "prior"}):
            original = deepcopy(self.report)
            self.report.update(change)
            self.save_report()
            with self.assertRaises(ValueError):
                self.audit()
            self.report = original

    def test_changed_full_test_publication_rejected(self):
        (self.task / "submission.csv").write_text("row,label\n4,b\n5,a\n")
        with self.assertRaisesRegex(ValueError, "checkpoint differ"):
            self.audit()

    def test_wrong_requested_mode_and_split_are_rejected(self):
        for change in ({"quality_options": {"classification_round4_mode": "on"}},
                       {"host_validation": {"split": {"strategy": "random", "seed": 42}}}):
            original = deepcopy(self.report)
            self.report.update(change)
            self.save_report()
            with self.assertRaises(ValueError):
                self.audit()
            self.report = original

    def test_nested_timeout_stops_series_despite_valid_outer_and_refit_publication(self):
        native.write_json(self.run / "failed_attempt/execution.json", {
            "fit_index": 0, "execution": {"exit_code": -9, "timed_out": True}, "metadata": None})
        result = self.audit()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["scores"]["macro_f1"], 1.)
        self.assertTrue(any("timed_out" in value for value in result["internal_execution_stops"]))

    def test_nested_memory_stop_in_jsonl_is_not_lost(self):
        (self.run / "events.jsonl").write_text(json.dumps({"execution": {"memory_limit_exceeded": True}}) + "\n")
        self.assertEqual(self.audit()["status"], "incomplete")


class PreparationTests(unittest.TestCase):
    def test_dirty_checkout_blocks_before_any_packaging_or_data_read(self):
        with patch.object(native, "git", return_value=" M solution.py") as git:
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                native.source_snapshot()
        self.assertEqual(git.call_count, 1)

    def test_wrong_base_is_rejected(self):
        with patch.object(native, "git", side_effect=["", "0" * 40]):
            with self.assertRaisesRegex(ValueError, "declared base"):
                native.source_snapshot()

    def test_recipe_class_order_mismatch_blocks_native(self):
        with self.assertRaisesRegex(ValueError, "recipe/class-order identity"):
            native.frozen_identity(["a", "b"])

    def test_environment_preflight_uses_existing_public_helper_signature(self):
        with patch("research_lab.integrity.environment_versions", return_value={"packages": {}, "python": "3.12"}) as versions:
            result = native.environment("/selected/python")
        versions.assert_called_once_with("/selected/python")
        self.assertEqual(result["threads"], 8)
        self.assertEqual(result["cuda_visible_devices"], "")


class FrozenCustomTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "candidate.py"
        self.source.write_text("# original source\n")
        self.recipe = {"family": "round4_classification_v1", "variant": "identity"}
        canonical(self.root / "recipe.json", self.recipe)
        self.identity = {"source_sha256": native.file_hash(self.source), "recipe": {},
                         "identity_probe_recipe": self.recipe, "classes": ["a", "b"]}
        self.context = {"code_hash": self.identity["source_sha256"], "recipe": self.recipe,
                        "recipe_hash": native.json_hash(self.recipe), "classes": ["a", "b"]}
        self.evidence = {**self.context, "variant": "identity"}

    def test_actual_custom_source_must_match_author_bytes_not_only_self_consistent_receipts(self):
        native.verify_custom_identity(self.root, self.context, self.evidence, self.identity)
        self.source.write_text("# another source with internally consistent hashes\n")
        self.context["code_hash"] = self.evidence["code_hash"] = native.file_hash(self.source)
        with self.assertRaisesRegex(ValueError, "sealed author source"):
            native.verify_custom_identity(self.root, self.context, self.evidence, self.identity)

    def test_identity_recipe_is_sealed_separately(self):
        self.context["recipe"] = {**self.recipe, "extra": True}
        canonical(self.root / "recipe.json", self.context["recipe"])
        self.context["recipe_hash"] = self.evidence["recipe_hash"] = native.json_hash(self.context["recipe"])
        with self.assertRaisesRegex(ValueError, "sealed variant"):
            native.verify_custom_identity(self.root, self.context, self.evidence, self.identity)

    def test_missing_summary_keeps_partial_fit_attempt_and_unknown_totals(self):
        self.context.update(nonce="n", phase="inner_selection")
        native.write_json(self.root / "context.json", self.context)
        events = self.root / "round4_work/final/events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text('{"status":"started","component_index":0}\n{"status":')
        with patch("agent.round4_classification_candidate.checked_recipe", return_value=self.recipe):
            result = native.audit_custom_children(self.root, self.root, ["a", "b"], self.identity)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "incomplete_summary")
        self.assertIsNone(result[0]["physical_fits_started"])
        self.assertEqual(result[0]["event_counts"]["actual_started"], 1)
        self.assertTrue(result[0]["observed_counts_are_lower_bounds"])
        self.assertEqual(len(result[0]["event_parse_errors"]), 1)

    def test_successful_custom_summary_cannot_hide_corrupt_events_or_over_cap_fits(self):
        self.recipe["caps"] = {"max_physical_fits": 2}
        canonical(self.root / "recipe.json", self.recipe)
        self.context.update(nonce="n", phase="inner_selection", recipe_hash=native.json_hash(self.recipe))
        native.write_json(self.root / "context.json", self.context)
        events = self.root / "round4_work/final/events.jsonl"
        events.parent.mkdir(parents=True)
        evidence = {**self.context, "variant": "identity", "status": "completed", "full_fit_rows": 2,
                    "predict_rows": 1, "physical_fits_started": 3, "physical_fits_completed": 3}
        native.write_json(self.root / "round4_evidence.json", evidence)
        with patch("agent.round4_classification_candidate.checked_recipe", return_value=self.recipe), \
                patch("experiment_round4_classification_export.verify_children", return_value=[{"status": "verified"}]):
            events.write_text('{"status":"started"}\n{"status":')
            with self.assertRaisesRegex(ValueError, "corrupt fit events"):
                native.audit_custom_children(self.root, self.root, ["a", "b"], self.identity)
            events.write_text('\n'.join(json.dumps({"status": status}) for status in
                                       ("started", "completed") * 3) + '\n')
            with self.assertRaisesRegex(ValueError, "frozen cap"):
                native.audit_custom_children(self.root, self.root, ["a", "b"], self.identity)

    def test_helper_dependency_must_resolve_inside_exact_flat_package(self):
        package = self.root / "package"
        package.mkdir()
        source = package / "_alfa_agent__fixture.py"
        source.write_text("# dependency\n")
        digest = native.file_hash(source)
        deps = {"versions": {}, "sources": {"agent.fixture": {"path": str(source), "sha256": digest}}}
        identity = {**self.identity, "recipe_sha256": native.json_hash({}), "dependency_snapshot": deepcopy(deps)}
        helper = {"dependencies": deepcopy(deps), "classes": identity["classes"], "source_hash": identity["source_sha256"],
                  "oof_recipe": {}, "oof_recipe_hash": identity["recipe_sha256"], "identity_recipe": self.recipe,
                  "identity_recipe_hash": native.json_hash(self.recipe)}
        path = self.root / "manifest.json"
        native.write_json(path, helper)
        report = {"classification_round4": {"manifest": str(path)}}
        manifest = {"package": str(package), "package_files": {source.name: digest}, "package_dependencies": deepcopy(deps)}
        native.verify_host_manifest(self.root, report, manifest, identity)
        outside = self.root / source.name
        outside.write_bytes(source.read_bytes())
        helper["dependencies"]["sources"]["agent.fixture"]["path"] = str(outside)
        native.write_json(path, helper)
        with self.assertRaisesRegex(ValueError, "escapes its root"):
            native.verify_host_manifest(self.root, report, manifest, identity)

    def test_transformed_package_initializer_has_its_own_hash(self):
        source_root, package = self.root / "source", self.root / "flat"
        (source_root / "agent").mkdir(parents=True)
        package.mkdir()
        original = source_root / "agent/__init__.py"
        original.write_text("# original initializer\n")
        flat = package / "agent.py"
        flat.write_text("# original initializer plus generated loader\n")
        deps = {"sources": {"agent": {"path": str(original), "sha256": native.file_hash(original)}}, "versions": {}}
        result = native.packaged_dependencies(deps, package, {"agent.py": native.file_hash(flat)}, source_root)
        self.assertEqual(result["sources"]["agent"]["sha256"], native.file_hash(flat))
        self.assertNotEqual(result["sources"]["agent"]["sha256"], deps["sources"]["agent"]["sha256"])


class ConclusionTests(unittest.TestCase):
    def test_positive_bundle_delta_with_ordinary_winner_has_no_oof_model_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            native.write_json(output / "run_manifest.json", {})
            for mode in native.ORDER:
                native.write_json(output / mode / "execution.json", {})
            manifest = {"source": {"sha": "s", "tree": "t"}, "minimum_absolute_gain": .002}
            rows = [{"status": "completed", "scores": {"macro_f1": .6}, "oof_published": False, "custom_invocations": []},
                    {"status": "completed", "scores": {"macro_f1": .603}, "oof_published": False,
                     "custom_invocations": [{"variant": "oof"}]}]
            with patch.object(native, "load_manifest", return_value=manifest), patch.object(native, "audit_arm", side_effect=rows):
                report = native.audit(output)
            self.assertEqual(report["status"], "promising_opt_in")
            self.assertFalse(report["oof_model_quality_claim"])
            self.assertEqual(report["quality_attribution"], "model_plus_scheduling_plus_paid_probe_bundle")


if __name__ == "__main__":
    unittest.main()
