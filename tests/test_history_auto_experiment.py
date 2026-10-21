"""Auto-history experiment proof tests using fabricated, nontraining runs.

The pure schema helper is real. Candidate receipts are protocol fixtures and
are never evidence of model quality, API availability, or a trained baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from agent.history_discovery import discover_history_schema
from agent.history_discovery_host import _source
import benchmark_agent as benchmark
import experiment_history_runtime as experiment
import tests.test_history_runtime_experiment as fixtures


class HistoryAutoExperimentTests(unittest.TestCase):
    # Reuse only the protocol fixture implementation, not its test methods.
    setUp = fixtures.HistoryExperimentTests.setUp
    prepare = fixtures.HistoryExperimentTests.prepare
    invocation = fixtures.HistoryExperimentTests.invocation

    def fake_worker(self, code, task_dir, run_dir, step, timeout, **kwargs):
        task = Path(task_dir) / "task.md"
        request_path = task.parent / "driver_request.json"
        request = experiment._read(request_path)
        if not request.get("history_auto", False):
            return fixtures.HistoryExperimentTests.fake_worker(self, code, task_dir, run_dir, step, timeout, **kwargs)

        self.assertIsNone(request["history_config"])
        split = experiment._split(task)
        spec, train = split["spec"], split["train"]
        directory = task.parent / ".agent_runs" / ("a" * 32)
        report = {"version": 1, "policy": "auto_schema_v1", "requested": True,
                  "scope": "inner_fit_features_only", "identity_verified": False,
                  "status": "skipped", "reason": "time_validation_required",
                  "blend_weight": 0.5, "selected_config": None, "fit_rows": 0,
                  "feature_columns": [], "period_column": None, "fit_input_sha256": None,
                  "fit_ids_sha256": None, "worker_source_sha256": None,
                  "worker_source_text_sha256": None, "execution": None, "result": None}
        selected = None
        if split["choice"]["strategy"] == "time":
            # The reused base fixture creates its run directory itself.
            discovery = task.parent / "fake_discovery_staging"
            (discovery / "execution").mkdir(parents=True)
            features, numeric, period = list(spec.features), list(spec.numeric), split["choice"]["time_column"]
            fitting = train.iloc[split["inner_fit"]]
            fitting.loc[:, features].to_csv(discovery / "fit_features.csv", index=False)
            experiment._write(discovery / "request.json", {
                "feature_columns": features, "numeric_columns": numeric, "period_column": period})
            # The canonical worker source is an artifact only: it is not run.
            # Mirror the real driver: its import root comes from the sealed
            # provenance, not a Windows short/long alias passed to prepare.
            plan = experiment._read(Path(task_dir).parents[1] / "manifest.json")
            source = _source(Path(plan["source"]["root"]))
            with (discovery / "execution/experiment_000.py").open(
                    "w", encoding="utf-8", newline=getattr(self, "worker_source_newline", None)) as stream:
                stream.write(source)
            frame = pd.read_csv(discovery / "fit_features.csv", dtype=object, keep_default_na=False, na_filter=False)
            result = discover_history_schema(frame, feature_columns=features,
                                             numeric_columns=numeric, period_column=period)
            experiment._write(discovery / "discovery.json", result)
            selected = result["selected_config"]
            ids = fitting[spec.id_column].tolist()
            report.update(status="selected" if result["status"] == "proposed" else result["status"],
                          reason="one_schema_frozen" if selected else result["reason"],
                          selected_config=selected, feature_columns=features, period_column=period,
                          fit_rows=len(fitting), result=result,
                          fit_input_sha256=experiment._hash(discovery / "fit_features.csv"),
                          fit_ids_sha256=hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":"),
                                                                  allow_nan=False).encode("utf-8")).hexdigest(),
                          worker_source_sha256=experiment._hash(discovery / "execution/experiment_000.py"),
                          worker_source_text_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                          execution={"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False})

        # The old fixture accepts explicit recipes. Translate privately while
        # fabricating its receipts, then restore the actual frozen auto request.
        translated = {**request, "history_config": selected}
        experiment._write(request_path, translated)
        try:
            execution = fixtures.HistoryExperimentTests.fake_worker(self, code, task_dir, run_dir, step, timeout, **kwargs)
        finally:
            experiment._write(request_path, request)
        if split["choice"]["strategy"] == "time":
            shutil.move(str(discovery), directory / "history_discovery")
        run_path = directory / "run.json"
        run = experiment._read(run_path)
        run["history_discovery"] = report
        experiment._write(run_path, run)
        driver_path = task.parent / "driver_evidence.json"
        driver = experiment._read(driver_path)
        driver["run_report_sha256"] = experiment._hash(run_path)
        experiment._write(driver_path, driver)
        return execution

    def test_explicit_mode_default_keeps_old_seeds_and_driver_interface(self):
        output = self.prepare(smoke=True)
        plan = experiment.read_plan(output)
        self.assertEqual(plan["method"]["history_mode"], "explicit")
        self.assertEqual(plan["method"]["development_seeds"], [131])
        self.assertEqual(plan["method"]["confirmation_seeds"], [2203, 4001])
        self.assertEqual(plan["cases"][0]["seed"], 131)
        code = experiment._driver(self.solver, output / plan["cases"][0]["task"])
        self.assertNotIn("history_auto", code)
        report = experiment.run_stage(output)
        self.assertEqual(report["status"], "completed")
        for row in report["trials"]:
            self.assertNotIn("history_auto", row["request"])
            self.assertEqual(row["request"]["history_config"] is not None, row["arm"] == "history")

    def test_auto_plan_freezes_new_seeds_and_has_no_host_selected_config(self):
        output = self.prepare(history_mode="auto")
        plan = experiment.read_plan(output)
        self.assertEqual(plan["method"]["history_mode"], "auto")
        self.assertEqual(plan["method"]["development_seeds"], [313])
        self.assertEqual(plan["method"]["confirmation_seeds"], [8117, 16417])
        self.assertEqual({case["seed"] for case in plan["cases"] if case["stage"] == "development"}, {313})
        self.assertEqual({case["seed"] for case in plan["cases"] if case["stage"] == "confirmation"}, {8117, 16417})
        self.assertTrue(all(config is None for config in plan["method"]["history_configs"].values()))
        code = experiment._driver(self.solver, output / plan["cases"][0]["task"], history_mode="auto")
        self.assertIn("history_auto", code)
        smoke = self.prepare(name="auto_smoke", smoke=True, history_mode="auto")
        smoke_plan = experiment.read_plan(smoke)
        self.assertEqual([(case["name"], case["seed"]) for case in smoke_plan["cases"]], [("entity_signal", 313)])

    def test_invalid_mode_rejected_before_creating_plan(self):
        for value in ("unknown", None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(name="bad_" + repr(value), history_mode=value)

    def test_fake_auto_pair_proves_selected_schema_and_preserves_control(self):
        output = self.prepare(smoke=True, history_mode="auto")
        report = experiment.run_stage(output)
        self.assertEqual(report["status"], "completed", [row.get("verification_error") for row in report["trials"]])
        audit = experiment.reaudit(output)
        self.assertEqual(audit["status"], "completed")
        self.assertTrue(audit["pairs"][0]["baseline_recipe_match"])
        self.assertTrue(audit["pairs"][0]["comparable_end_to_end"])
        for row in report["trials"]:
            self.assertIsNone(row["request"]["history_config"])
            self.assertEqual(row["request"]["history_auto"], row["arm"] == "history")
            verified = row["verified"]
            self.assertTrue(verified["full_refit_verified"])
            self.assertEqual(verified["discovery_verified"], row["arm"] == "history")
            if row["arm"] == "control":
                self.assertIsNone(verified["discovery"])
                self.assertEqual(verified["history"]["status"], "disabled")
                continue
            discovery = verified["discovery"]
            self.assertEqual(discovery["status"], "selected")
            self.assertIs(discovery["identity_verified"], False)
            self.assertEqual(discovery["selected_config"]["key_columns"], ["entity_key"])
            self.assertEqual(discovery["selected_config"]["ambiguity_policy"], "past_unique_only")
            self.assertEqual(verified["history"]["history_config"], discovery["selected_config"])
            self.assertTrue(verified["history_verified"])
            task = output / row["task"]
            frame = pd.read_csv(task.parent / ".agent_runs" / ("a" * 32) / "history_discovery/fit_features.csv",
                                dtype=object, keep_default_na=False, na_filter=False)
            self.assertNotIn("row_key", frame.columns)
            self.assertNotIn("outcome", frame.columns)
            self.assertEqual(list(frame.columns), discovery["feature_columns"])
            self.assertEqual(len(frame), discovery["fit_rows"])
            for name in ("fit_features.csv", "request.json", "discovery.json", "execution/experiment_000.py"):
                self.assertTrue(any(Path(path).as_posix().endswith("history_discovery/" + name)
                                    for path in verified["artifacts"]))

    def test_no_coverage_abstains_and_iid_skips_with_valid_normal_outputs(self):
        output = self.prepare(history_mode="auto")
        report = experiment.run_stage(output)
        self.assertEqual(report["status"], "completed", [row.get("verification_error") for row in report["trials"]])
        history = {row["case"].rsplit("_", 1)[0]: row["verified"] for row in report["trials"] if row["arm"] == "history"}
        no_history = history["no_history_coverage"]
        self.assertEqual(no_history["discovery"]["status"], "abstained")
        self.assertTrue(no_history["discovery_verified"])
        self.assertIsNone(no_history["discovery"]["selected_config"])
        iid = history["iid_regression"]
        self.assertEqual(iid["discovery"]["status"], "skipped")
        self.assertFalse(iid["discovery_verified"])
        for verified in (no_history, iid):
            self.assertEqual(verified["history"]["status"], "disabled")
            self.assertIs(verified["history"]["requested"], False)
            self.assertFalse(verified["history_verified"])
            self.assertTrue(verified["full_refit_verified"])
            self.assertTrue(verified["audit"]["valid"])
        self.assertEqual(experiment.reaudit(output)["status"], "completed")

    def test_tampering_any_discovery_artifact_fails_reaudit(self):
        for index, artifact in enumerate(("fit_features.csv", "request.json", "discovery.json", "execution/experiment_000.py")):
            with self.subTest(artifact=artifact):
                output = self.prepare(name=f"tamper_{index}", smoke=True, history_mode="auto")
                report = experiment.run_stage(output)
                self.assertEqual(report["status"], "completed")
                row = next(row for row in report["trials"] if row["arm"] == "history")
                path = (output / row["task"]).parent / ".agent_runs" / ("a" * 32) / "history_discovery" / artifact
                path.write_bytes(path.read_bytes() + b"\n")
                audit = experiment.reaudit(output)
                self.assertEqual(audit["status"], "incomplete")
                self.assertTrue(any(not row["verified"] for row in audit["rows"] if row["arm"] == "history"))

    def test_feature_only_inner_rows_are_checked_even_if_hashes_are_repaired(self):
        output = self.prepare(smoke=True, history_mode="auto")
        report = experiment.run_stage(output)
        self.assertEqual(report["status"], "completed")
        row = next(row for row in report["trials"] if row["arm"] == "history")
        task = output / row["task"]
        run_path = task.parent / ".agent_runs" / ("a" * 32) / "run.json"
        run = experiment._read(run_path)
        path = run_path.parent / "history_discovery/fit_features.csv"
        frame = pd.read_csv(path, dtype=object, keep_default_na=False, na_filter=False)
        frame["outcome"] = "forbidden"
        frame.to_csv(path, index=False)
        run["history_discovery"]["fit_input_sha256"] = experiment._hash(path)
        experiment._write(run_path, run)
        driver_path = task.parent / "driver_evidence.json"
        driver = experiment._read(driver_path)
        driver["run_report_sha256"] = experiment._hash(run_path)
        experiment._write(driver_path, driver)
        plan = experiment.read_plan(output)
        case = next(case for case in plan["cases"] if case["key"] == row["case"])
        actuals = experiment._read(output / case["audit"])["actuals"]
        with self.assertRaises(ValueError):
            experiment.inspect_trial(task, case, row["request"], actuals, solver_root=self.solver)

    def test_preflight_requires_auto_interface_only_in_auto_mode(self):
        self.preflight.stop()
        source = {**self.source, "worktree_dirty": False}
        with patch.object(benchmark, "source_provenance", return_value=source), \
                patch.object(experiment.importlib.util, "find_spec", return_value=object()):
            explicit = experiment.preflight(self.solver)
            auto = experiment.preflight(self.solver, history_mode="auto")
        self.assertTrue(explicit["ready"])
        self.assertFalse(auto["ready"])
        self.assertIn("history_auto", auto["missing_interface"])

    def test_canonical_solver_root_and_crlf_worker_keep_distinct_byte_and_text_hashes(self):
        # A noncanonical spelling reproduces Windows short/long temp aliases
        # on every OS without requiring platform-specific symlink privileges.
        alias_step = self.solver.parent / "existing_alias_step"
        alias_step.mkdir()
        self.solver = alias_step / ".." / self.solver.name
        self.worker_source_newline = "\r\n"
        output = self.prepare(name="crlf_alias", smoke=True, history_mode="auto")
        plan = experiment.read_plan(output)
        self.assertNotEqual(str(self.solver), plan["source"]["root"])
        report = experiment.run_stage(output)
        self.assertEqual(report["status"], "completed", [row.get("verification_error") for row in report["trials"]])
        row = next(row for row in report["trials"] if row["arm"] == "history")
        task = output / row["task"]
        script = task.parent / ".agent_runs" / ("a" * 32) / "history_discovery/execution/experiment_000.py"
        raw = script.read_bytes()
        source = _source(Path(plan["source"]["root"]))
        discovery = row["verified"]["discovery"]
        self.assertIn(b"\r\n", raw)
        self.assertEqual(script.read_text(encoding="utf-8"), source)
        self.assertEqual(discovery["worker_source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(discovery["worker_source_text_sha256"], hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.assertNotEqual(discovery["worker_source_sha256"], discovery["worker_source_text_sha256"])
        self.assertEqual(experiment.reaudit(output)["status"], "completed")
        # A newline-only edit preserves text semantics but invalidates physical
        # source evidence. Keep the auditor's byte-integrity requirement.
        script.write_bytes(source.encode("utf-8"))
        self.assertEqual(experiment.reaudit(output)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
