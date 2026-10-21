"""Independent qualification provenance, isolation and failure classifications."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import qualify_unseen_tasks as qualification
from unseen_cases import materialize_case


class QualificationTests(unittest.TestCase):
    def materialize(self, directory, name="flow_rmse"):
        metadata = materialize_case(name, Path(directory) / "task")
        metadata["input_sha256"] = qualification.input_hashes(Path(metadata["task_path"]).parent,
                                                              metadata["input_paths"])
        return metadata

    def write_submission(self, metadata, predictions=None, *, destination=None):
        contract = metadata["contract"]
        path = Path(destination or Path(metadata["task_path"]).parent / "submission.csv")
        actual = metadata["audit_actuals"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=contract["output_columns"])
            writer.writeheader()
            for index, row_id in enumerate(metadata["test_ids"]):
                row = {contract["id_column"]: row_id}
                if contract["prediction_kind"] == "probabilities":
                    for column, label in contract["probability_columns"]:
                        row[column] = .98 if actual[index] == label else (
                            .02 if len(contract["classes"]) == 2 else .02/(len(contract["classes"])-1))
                else:
                    row[contract["target_column"]] = actual[index] if predictions is None else predictions[index]
                writer.writerow(row)
        return path

    def test_independent_scorer_covers_nine_metrics_and_literal_schema(self):
        names = ("flow_rmse", "inventory_mae", "fatigue_mse", "binding_r2", "logic_accuracy",
                 "rings_macro_f1", "cascade_weighted_f1", "survival_binary_auc", "count_binary_logloss",
                 "checker_multiclass_logloss", "checker_multiclass_auc", "reverse_positive_binary_auc",
                 "signed_flow_unicode_schema", "probability_columns_permuted_logloss", "literal_labels_accuracy")
        checked_metrics = set()
        with tempfile.TemporaryDirectory() as temporary:
            for name in names:
                with self.subTest(case=name):
                    directory = Path(temporary) / name
                    directory.mkdir()
                    metadata = self.materialize(directory, name)
                    checked = qualification.audit_submission(self.write_submission(metadata), metadata)
                    self.assertTrue(checked["valid"], checked)
                    checked_metrics.add(checked["metric"]["name"])
                    self.assertEqual(checked["metric"]["higher_is_better"],
                                     metadata["metric"] in {"Accuracy", "Macro F1", "Weighted F1", "ROC AUC", "R2"})
                    if metadata["metric"] == "R2":
                        self.assertEqual(checked["metric"]["value"], 1.)
        self.assertEqual(checked_metrics, set(qualification.METRIC_DIRECTIONS))

    def test_scorer_rejects_wrong_ids_order_width_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary)
            path = self.write_submission(metadata)
            original = path.read_text()
            header, rows = qualification.read_csv(path)
            mutations = []
            reordered = list(reversed(rows))
            for body in (reordered, [rows[0], *rows[:-1]]):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=header)
                    writer.writeheader(); writer.writerows(body)
                mutations.append(path.read_text())
            mutations.extend((original.replace("\n", ",extra\n", 1),
                              original.replace(str(metadata["audit_actuals"][0]), "nan", 1)))
            for text in mutations:
                path.write_text(text, encoding="utf-8")
                self.assertFalse(qualification.audit_submission(path, metadata)["valid"])

    def test_probability_sum_range_and_label_errors_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary, "checker_multiclass_logloss")
            path = self.write_submission(metadata)
            header, rows = qualification.read_csv(path)
            probabilities = [column for column, _ in metadata["contract"]["probability_columns"]]
            for value in ("NaN", "Inf", "-0.1", "1.1", "0.4"):
                changed = copy.deepcopy(rows)
                changed[0].update({column: value for column in probabilities})
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=header)
                    writer.writeheader(); writer.writerows(changed)
                self.assertFalse(qualification.audit_submission(path, metadata)["valid"], value)

    def test_binary_probability_positive_mapping_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary, "reverse_positive_binary_auc")
            self.assertEqual(metadata["contract"]["positive_class"], "NA")
            path = self.write_submission(metadata)
            self.assertEqual(qualification.audit_submission(path, metadata)["metric"]["value"], 1.)
            metadata["contract"]["positive_class"] = None
            self.assertFalse(qualification.audit_submission(path, metadata)["valid"])

    def test_json_and_csv_links_and_malformed_quotes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "broken.csv"
            csv_path.write_text('id,target\na,"unterminated\n', encoding="utf-8")
            with self.assertRaises(csv.Error):
                qualification.read_csv(csv_path)
            path = root / "evidence.json"
            path.write_text("{}", encoding="utf-8")
            if os.name != "nt":
                linked = root / "linked.json"
                linked.symlink_to(path)
                with self.assertRaises(ValueError):
                    qualification.read_json(linked)

    def test_environment_strips_credentials_and_caps_threads(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret", "OPENAI_BASE_URL": "https://secret.invalid",
                                     "AWS_SECRET_ACCESS_KEY": "private", "PYTHONPATH": "/untrusted", "OMP_NUM_THREADS": "99"}):
            environment = qualification.sanitized_environment()
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("OPENAI_BASE_URL", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["OMP_NUM_THREADS"], "8")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "8")

    def test_diagnostics_remove_secrets_and_provider_urls(self):
        text = qualification.safe_text("api_key=abc Authorization: Bearer private sk-example https://secret.invalid/path?key=q")
        for forbidden in ("abc", "private", "sk-example", "secret.invalid", "key=q"):
            self.assertNotIn(forbidden, text)

    def test_real_child_has_no_competition_environment_and_times_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-private"}):
                execution = qualification.run_solver(
                    [sys.executable, "-c", "import os;print('OPENAI_API_KEY' in os.environ)"], root,
                    root / "clean", 5, 1024**3)
            self.assertEqual(execution["exit_code"], 0)
            self.assertIn("False", execution["stdout"])
            execution = qualification.run_solver([sys.executable, "-c", "import time;time.sleep(10)"], root,
                                                   root / "timeout", .1, 1024**3)
            self.assertTrue(execution["external_timeout"])
            self.assertNotEqual(execution["exit_code"], 0)

    def test_statuses_distinguish_refusal_prior_censoring_and_full_model(self):
        entry = {"definition": {"expected_status": "model_or_prior"}, "mode": "offline"}
        execution = {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                     "stderr": ""}
        checked = {"valid": True}
        self.assertEqual(qualification._classify(entry, execution, checked, {"kind": "target_prior"}, True),
                         ("valid_prior_fallback", True))
        self.assertEqual(qualification._classify(entry, execution, checked, {"kind": "full_model_result"}, True),
                         ("full_result", True))
        self.assertEqual(qualification._classify(entry, execution, checked, {"kind": "unverified_output"}, True),
                         ("valid_output_unverified", False))
        failed = {**execution, "exit_code": 1, "stderr": "Explicit error"}
        self.assertEqual(qualification._classify(entry, failed, checked, {"kind": "target_prior"}, True)[0], "solver_error")
        self.assertEqual(qualification._classify(entry, {**failed, "external_timeout": True}, checked,
                                                {"kind": "target_prior"}, True)[0], "external_timeout")
        negative = {**entry, "case_id": "missing_metric", "definition": {"expected_status": "refusal"}}
        semantic_failure = {**failed, "stderr": "Error: No supported primary metric declared"}
        self.assertEqual(qualification._classify(negative, semantic_failure, {}, {"kind": "none"}, True),
                         ("expected_refusal", True))
        self.assertEqual(qualification._classify(negative, execution, {}, {"kind": "none"}, True)[0], "unexpected_acceptance")
        infrastructure = {**failed, "stderr": "ModuleNotFoundError: No module named 'openai'"}
        self.assertEqual(qualification._classify(negative, infrastructure, {}, {"kind": "none"}, True),
                         ("solver_error", False))
        contract = {**entry, "mode": "contract"}
        self.assertEqual(qualification._classify(contract, execution, {"valid": False}, {"kind": "none"}, True),
                         ("contract_mismatch", False))
        self.assertEqual(execution["exit_code"], 0)
        duplicate = {**entry, "case_id": "duplicate_test_ids", "definition": {"expected_status": "refusal"}}
        duplicate_error = {**failed, "stderr": json.dumps({"contract_error": {
            "type": "ValueError", "message": "test.csv: IDs must be nonempty and unique"}})}
        self.assertEqual(qualification._classify(duplicate, duplicate_error, {}, {"kind": "none"}, True),
                         ("expected_refusal", True))
        malformed = {**entry, "case_id": "malformed_csv", "definition": {"expected_status": "refusal"}}
        unrelated = {**failed, "stderr": 'Traceback (most recent call last):\n'
                     '  File "/tasks/malformed_csv--contract/internal.py", line 2, in main\n'
                     'ValueError: unrelated internal configuration failure\n'}
        self.assertEqual(qualification._classify(malformed, unrelated, {}, {"kind": "none"}, True),
                         ("solver_error", False))

    def fake_identity(self, root):
        return {"root": str(root), "sha": "a"*40, "dirty": False,
                "source_hashes": {"solution.py": "b"*64}, "source_fingerprint": "c"*64}

    def prepare(self, root, output, *, mode="offline", name="flow_rmse"):
        identity = self.fake_identity(root)
        with patch.object(qualification, "solver_identity", return_value=identity), \
                patch.object(qualification, "environment_versions", return_value={"python": "unit"}):
            plan = qualification.prepare_run(root, output, python=sys.executable, cases=[name], modes=[mode])
        return plan, identity

    def test_preflight_keeps_private_answers_outside_solver_tree_and_does_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(qualification, "run_solver") as run:
                plan, identity = self.prepare(root, root / "run")
            run.assert_not_called()
            entry = plan["entries"][0]
            self.assertFalse(Path(entry["audit_path"]).is_relative_to(Path(entry["task_dir"])))
            self.assertEqual(sorted(path.name for path in Path(entry["task_dir"]).iterdir()), ["inputs", "task.md"])
            with self.assertRaises(ValueError):
                self.prepare(root, root / "run")
            with patch.object(qualification, "solver_identity", return_value={**identity, "source_fingerprint": "changed"}), \
                    patch.object(qualification, "environment_versions", return_value={"python": "unit"}):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    qualification.load_plan(root / "run", execution=True)

    def test_frozen_audit_and_inputs_tampering_are_detected(self):
        for which in ("audit", "input", "manifest"):
            with self.subTest(which=which), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan, _ = self.prepare(root, root / "run")
                entry = plan["entries"][0]
                if which == "audit":
                    path = Path(entry["audit_path"])
                elif which == "input":
                    path = Path(entry["task_path"])
                else:
                    path = root / "run/manifest.json"
                if which == "manifest":
                    changed = json.loads(path.read_text())
                    changed["seed"] += 1
                    path.write_text(json.dumps(changed), encoding="utf-8")
                else:
                    path.write_text(path.read_text() + " ", encoding="utf-8")
                with self.assertRaises(ValueError):
                    qualification.load_plan(root / "run")

    def test_default_argv_has_no_extra_flags_and_contract_probe_is_separate(self):
        plan = {"python": "python", "solver": {"root": "/solver"}, "budgets": {"solver_seconds": 30}}
        entry = {"mode": "default", "task_path": "/task/task.md"}
        self.assertEqual(qualification._argv(plan, entry, {}), ["python", "/solver/solution.py", "/task/task.md"])
        args = qualification._argv(plan, {**entry, "mode": "offline"}, {"validation": {"strategy": "time", "time_column": "period"}})
        self.assertEqual(args[-4:], ["--validation", "time", "--time-column", "period"])
        self.assertNotIn("audit", " ".join(args))
        self.assertIn("-c", qualification._argv(plan, {**entry, "mode": "contract"}, {}))

    def test_runner_saves_predictions_reaudits_and_forbids_second_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            plan, identity = self.prepare(root, output)

            def fake_run(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes):
                metadata = qualification.read_json(plan["entries"][0]["audit_path"])
                metadata["task_path"] = str(Path(cwd)/"task.md")
                self.write_submission(metadata)
                return {"argv": argv, "exit_code": 0, "external_timeout": False,
                        "memory_limit_exceeded": False, "elapsed_seconds": .01,
                        "stderr": "", "stdout": "completed", "resources": {"available": False}}

            with patch.object(qualification, "solver_identity", return_value=identity), \
                    patch.object(qualification, "environment_versions", return_value={"python": "unit"}), \
                    patch.object(qualification, "run_solver", side_effect=fake_run), \
                    patch.object(qualification, "inspect_model_evidence", return_value={
                        "kind": "full_model_result", "verified": True, "artifacts": []}):
                result = qualification.execute_run(output)
                self.assertEqual(result["counts"], {"full_result": 1})
                with self.assertRaises(FileExistsError):
                    qualification.execute_run(output)
            reaudit = qualification.re_audit(output)
            self.assertEqual(reaudit["verified_predictions"], 1)
            self.assertEqual(reaudit["models_refitted"], 0)
            raw = output / result["results"][0]["raw_predictions"]["path"]
            raw.write_text(raw.read_text()+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw predictions"):
                qualification.re_audit(output)

    def test_prior_receipt_does_not_claim_a_model_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary)
            path = self.write_submission(metadata)
            task_dir = path.parent
            prior_dir = task_dir / ".agent_runs" / ("a"*32)
            prior_dir.mkdir(parents=True)
            shutil.copyfile(path, prior_dir / "prior.csv")
            receipt = {"kind": "target_prior", "status": "ready", "quality_verified": False,
                       "submission_sha256": qualification.file_hash(path),
                       "inputs": {Path(name).name: value for name, value in metadata["input_sha256"].items()}}
            qualification.write_json(prior_dir / "emergency.json", receipt)
            evidence = qualification.inspect_model_evidence(task_dir, qualification.file_hash(path), metadata)
            self.assertEqual(evidence["kind"], "target_prior")
            self.assertFalse(evidence["quality_verified"])
            receipt["inputs"]["train.csv"] = "0"*64
            qualification.write_json(prior_dir / "emergency.json", receipt)
            self.assertEqual(qualification.inspect_model_evidence(task_dir, qualification.file_hash(path), metadata)["kind"],
                             "unverified_output")

    def test_contract_metric_mismatch_keeps_factual_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, identity = self.prepare(root, root / "run", mode="contract", name="inventory_mae")
            audit = qualification.read_json(plan["entries"][0]["audit_path"])
            returned = dict(audit["contract"])
            returned["metric"] = {"name": "RMSE", "higher_is_better": False,
                                  "task_type": "regression", "prediction_kind": "regression"}
            execution = {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                         "stderr": "", "stdout": json.dumps(returned), "resources": {"available": False}}
            with patch.object(qualification, "solver_identity", return_value=identity), \
                    patch.object(qualification, "environment_versions", return_value={"python": "unit"}), \
                    patch.object(qualification, "run_solver", return_value=execution):
                report = qualification.execute_run(root / "run")
            result = report["results"][0]
            self.assertEqual(result["status"], "contract_mismatch")
            self.assertFalse(result["qualified"])
            self.assertEqual(result["execution"]["exit_code"], 0)

    def test_deleted_input_is_recorded_and_remaining_cases_are_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = self.fake_identity(root)
            with patch.object(qualification, "solver_identity", return_value=identity), \
                    patch.object(qualification, "environment_versions", return_value={"python": "unit"}):
                plan = qualification.prepare_run(root, root / "run", python=sys.executable,
                                                 cases=["flow_rmse", "inventory_mae"])

            def delete_input(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes):
                Path(cwd, "task.md").unlink()
                return {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                        "stderr": "", "stdout": "", "resources": {"available": False}}

            with patch.object(qualification, "solver_identity", return_value=identity), \
                    patch.object(qualification, "environment_versions", return_value={"python": "unit"}), \
                    patch.object(qualification, "run_solver", side_effect=delete_input) as launch:
                report = qualification.execute_run(root / "run")
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(report["status"], "aborted_input_mutation")
            self.assertEqual([row["status"] for row in report["results"]],
                             ["input_mutation", "skipped_input_mutation"])
            self.assertFalse(report["qualification_passed"])
            self.assertEqual(report["results"][0]["execution"]["exit_code"], 0)

    def test_nonobject_contract_probe_json_is_a_mismatch_without_changing_exit_code(self):
        for returned in ([], None, 0, "not a contract"):
            with self.subTest(returned=returned), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, identity = self.prepare(root, root / "run", mode="contract")
                execution = {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                             "stderr": "", "stdout": json.dumps(returned), "resources": {"available": False}}
                with patch.object(qualification, "solver_identity", return_value=identity), \
                        patch.object(qualification, "environment_versions", return_value={"python": "unit"}), \
                        patch.object(qualification, "run_solver", return_value=execution):
                    report = qualification.execute_run(root / "run")
                result = report["results"][0]
                self.assertEqual(result["status"], "contract_mismatch")
                self.assertFalse(result["qualified"])
                self.assertEqual(result["execution"]["exit_code"], 0)

    def test_malformed_json_receipts_are_unverified_not_runner_crashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary)
            path = self.write_submission(metadata)
            directory = path.parent / ".agent_runs" / ("a"*32)
            directory.mkdir(parents=True)
            for name in ("checkpoint.json", "emergency.json"):
                (directory / name).write_text("[]", encoding="utf-8")
            result = qualification.inspect_model_evidence(path.parent, qualification.file_hash(path), metadata)
            self.assertEqual(result["kind"], "unverified_output")
            self.assertEqual(len(result["errors"]), 2)
            with self.assertRaisesRegex(ValueError, "object"):
                qualification.read_json(directory / "checkpoint.json")

    def test_full_refit_evidence_requires_actual_phase_and_complete_original_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.materialize(temporary)
            path = self.write_submission(metadata)
            task_dir = path.parent
            run = task_dir / ".agent_runs" / ("a"*32)
            checkpoint = run / "checkpoints" / ("b"*32)
            phase = run / "seed/seed"
            staged = phase / "inputs"
            staged.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (phase / "execution").mkdir()
            for name in metadata["input_paths"]:
                if Path(name).suffix == ".csv":
                    shutil.copyfile(task_dir / name, staged / Path(name).name)
            for dest in (checkpoint / "submission.csv", phase / "submission.csv"):
                shutil.copyfile(path, dest)
            held = checkpoint / "holdout_predictions.csv"
            held.write_text("id,actual,prediction\nx,1,1\n", encoding="utf-8")
            qualification.write_json(run / "checkpoint.json", {"version": 1,
                "directory": "checkpoints/"+"b"*32,
                "submission_sha256": qualification.file_hash(path), "holdout_sha256": qualification.file_hash(held)})
            qualification.write_json(run / "automl_report.json", {"search": {"best_candidate": "seed"},
                                                                   "output_valid": True, "metric": {"value": .5}})
            qualification.write_json(phase / "config.json", {"phase": "seed", "task_spec": {"data_dir": str(staged)}})
            qualification.write_json(phase / "durations.json", {"fit_seconds": .1, "predict_seconds": .01, "total_seconds": .11})
            log = phase / "execution/001.stdout.txt"
            log.write_text("seed: refitting on all 300 rows\n", encoding="utf-8")
            evidence = qualification.inspect_model_evidence(task_dir, qualification.file_hash(path), metadata)
            self.assertEqual(evidence["kind"], "full_model_result", evidence)
            self.assertEqual(evidence["full_train_rows"], 300)
            evidence_paths = {item["path"] for item in evidence["artifacts"]}
            for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
                self.assertIn((staged / name).relative_to(task_dir).as_posix(), evidence_paths)
            for corrupted in ({"search": [], "metric": {}}, {"search": {"best_candidate": "seed"}, "metric": None}):
                qualification.write_json(run / "automl_report.json", corrupted)
                self.assertEqual(qualification.inspect_model_evidence(task_dir, qualification.file_hash(path), metadata)["kind"],
                                 "unverified_output")
            qualification.write_json(run / "automl_report.json", {"search": {"best_candidate": "seed"},
                                                                   "output_valid": True, "metric": {"value": .5}})
            log.write_text("selection: fitting on 180 rows\n", encoding="utf-8")
            self.assertEqual(qualification.inspect_model_evidence(task_dir, qualification.file_hash(path), metadata)["kind"],
                             "unverified_output")


if __name__ == "__main__":
    unittest.main()
