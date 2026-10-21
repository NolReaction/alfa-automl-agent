"""Synthetic saved artifacts test the audit protocol, never ML quality."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent.task_spec import load_task
from research_lab import audit
from research_lab.integrity import file_hash, file_index, json_bytes, write_json
from tests.test_task_spec import make_task, write_csv


def fixture(root, *, probabilities=False):
    """A positive native receipt chain without importing/fitting its candidate."""
    arm_dir = root / "cases" / "literal-1" / "incumbent"
    task = arm_dir / "task"
    task.mkdir(parents=True)
    extra = ""
    outputs = ("answer",)
    labels = ("00", "nan")
    if probabilities:
        outputs = ("nan", "00")
    task_path = make_task(task, metric="LogLoss" if probabilities else "Accuracy",
                          labels=labels, dtype="category", outputs=outputs, extra=extra)
    spec = load_task(task_path)
    contract = spec.to_dict()
    contract["prediction_kind"] = spec.prediction_kind
    host = root / "audit" / "literal.json"
    answers = {"contract": contract, "metric": spec.metric.name,
               "audit_actuals": ["00", "nan"], "test_ids": ["0005", "NA"]}
    write_json(host, answers)
    config_path, state_path, worker_path = (arm_dir / name for name in ("config.json", "state.json", "worker.json"))
    config = {"version": 1, "run_id": "fixture-run", "case_key": "literal-1", "arm": "incumbent",
              "task_dir": str(task), "task_path": str(task_path), "time_budget_seconds": 360,
              "solver_root": str(root / "solver"), "solver_args": ["--mode", "offline"],
              "require_adaptive": True, "worker_output": str(worker_path)}
    write_json(config_path, config)
    execution = {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False,
                 "elapsed_seconds": 12, "worker_wall_seconds": 11, "cleanup": {"complete": True},
                 "resources": {"available": True, "peak_rss_bytes": 100000}}
    state = {"version": 1, "run_id": "fixture-run", "status": "finished", "execution": execution}
    write_json(state_path, state)
    run = task / ".agent_runs" / ("a" * 32)
    run.mkdir(parents=True)
    code = b'raise RuntimeError("The auditor must never execute candidate source")\n'
    recipe = {"model": "protocol-fixture-only"}
    frozen_dir = run / "baseline_frozen"
    frozen_dir.mkdir()
    (frozen_dir / "research_candidate.py").write_bytes(code)
    (frozen_dir / "research_recipe.json").write_bytes(json_bytes(recipe))
    frozen = {"code_hash": file_hash(frozen_dir / "research_candidate.py"),
              "recipe_hash": file_hash(frozen_dir / "research_recipe.json")}
    write_json(frozen_dir / "frozen_candidate.json", {"version": 3, **frozen})
    with (spec.data_dir / "train.csv").open(newline="", encoding="utf-8") as stream:
        train = list(csv.reader(stream))[1:]
    with (spec.data_dir / "test.csv").open(newline="", encoding="utf-8") as stream:
        test = list(csv.reader(stream))[1:]
    fitting, held = list(range(40)), list(range(40, 60))
    split = {"fit_rows": fitting, "validation_rows": held, "choice": {"strategy": "random"},
             "split": {"strategy": "random", "train_rows": 40, "validation_rows": 20}}
    receipts = dict(frozen)
    outer_score = None
    for name, phase, parent, fit, predict, actual, nonce in (
            ("outer", "outer_validation", "outer_evaluations", [train[i] for i in fitting],
             [train[i][:-1] for i in held], [train[i][-1] for i in held], "b" * 32),
            ("refit", "full_refit", "full_refit", train, test, ["00", "nan"], "c" * 32)):
        directory = run / parent / "baseline" / (phase + "_" + nonce)
        directory.mkdir(parents=True)
        (directory / "candidate.py").write_bytes(code)
        (directory / "recipe.json").write_bytes(json_bytes(recipe))
        write_csv(directory / "fit.csv", [[spec.id_column, *spec.features, spec.target_column], *fit])
        write_csv(directory / "predict.csv", [[spec.id_column, *spec.features], *predict])
        local_spec = spec.to_dict()
        local_spec.update(task_path=str(directory / "task.md"), data_dir=str(directory))
        context = {"version": 3, "protocol_version": 3, "phase": phase, "nonce": nonce,
                   "recipe": recipe, **frozen, "task_spec": local_spec,
                   **{key: contract[key] for key in ("features", "numeric", "categorical", "classes",
                                                     "positive_class", "metric")},
                   "task_type": spec.metric.task_type, "prediction_kind": spec.prediction_kind}
        write_json(directory / "context.json", context)
        if probabilities:
            probability = [[.9 if label == column_label else .1 for _, column_label in spec.probability_columns]
                           for label in actual]
            predictions = [[row[0], *value] for row, value in zip(predict, probability)]
        else:
            predictions = [[row[0], label] for row, label in zip(predict, actual)]
        write_csv(directory / "predictions.csv", [list(spec.output_columns), *predictions])
        receipt = {"version": 3, "nonce": nonce, "phase": phase, **frozen,
                   "output_hash": file_hash(directory / "predictions.csv")}
        write_json(directory / "receipt.json", receipt)
        receipts[name] = str(directory / "receipt.json")
        if name == "refit":
            raw = (directory / "predictions.csv").read_bytes()
            (task / "submission.csv").write_bytes(raw)
            (run / "last_valid_submission.csv").write_bytes(raw)
        else:
            from qualify_unseen_tasks import audit_submission
            outer_score = audit_submission(directory / "predictions.csv",
                    {"contract": contract, "metric": spec.metric.name,
                     "test_ids": [row[0] for row in predict], "audit_actuals": actual})["metric"]
    report = {"research_protocol_version": 3, "mode": "local_only", "api": {"attempts": 0, "responses": 0},
              "coordinator": {"active": True}, "status": "completed", "stage": "result",
              "time_budget_seconds": 350, "elapsed_seconds": 10, "published_recipe": "baseline",
              "frozen_baseline": frozen, "candidate_receipts": receipts,
              "host_validation": {"split": split["split"]},
              "evaluations": {"final_check": {"baseline": outer_score}},
              "publication": {"new_result_published": True}, "output_check": {"valid": True},
              "refit_execution": {"baseline": {"exit_code": 0,
                   "resources": {"available": True, "cleanup_complete": True, "peak_rss_bytes": 100}}}}
    write_json(run / "run.json", report)
    pointer = {"path": (run / "run.json").relative_to(task).as_posix(),
               "sha256": file_hash(run / "run.json"), "origin": "returned", "kind": "research_v3"}
    write_json(task / "solver_output" / "research_benchmark.json",
               {"run_report": pointer["path"], "run_report_sha256": pointer["sha256"]})
    worker = {"version": 1, "run_id": "fixture-run", "arm": "incumbent",
              "config_sha256": file_hash(config_path), "solver_returned": True, "cli_exit_code": 0,
              "original_time_budget_seconds": 360, "cli_time_budget_seconds": 358,
              "cli_argv": [str(root / "solver" / "solution.py"), str(task_path),
                           *config["solver_args"], "--time-budget", "358"],
              "run_report": pointer,
              "observations": [{"phase": "run_agent", "time_budget_seconds": 350,
                                "local_only": True, "adaptive_search": True},
                               {"phase": "prepare_research", "run_dir": run.relative_to(task).as_posix(),
                                "frozen_split": split},
                               {"phase": "frozen_publication", "run_dir": run.relative_to(task).as_posix(),
                                "published_recipe": "baseline", "frozen_source": {
                                    "code_path": (frozen_dir / "research_candidate.py").relative_to(task).as_posix(),
                                    "recipe_path": (frozen_dir / "research_recipe.json").relative_to(task).as_posix(),
                                    **frozen, "declared_code_hash": frozen["code_hash"],
                                    "declared_recipe_hash": frozen["recipe_hash"]}}]}
    write_json(worker_path, worker)
    inputs = {path.relative_to(task).as_posix(): file_hash(path)
              for path in [task_path, *(spec.data_dir / name for name in
                  ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"))]}
    case = {"key": "literal-1", "audit_path": str(host), "audit_sha256": file_hash(host),
            "input_hashes": inputs,
            "arms": {"incumbent": {"task_dir": str(task), "config_path": str(config_path),
                                    "state_path": str(state_path)}}}
    plan = {"time_budget_seconds": 360, "cases": [case]}
    write_json(root / "plan.json", plan)
    return case, plan, worker_path, run


def terminal(root, case, plan, worker):
    result = audit.audit_arm(root, case, "incumbent", worker, plan)
    state_path = Path(case["arms"]["incumbent"]["state_path"])
    state = json.loads(state_path.read_text())
    state.update(audit=result, artifact_hashes=result["artifact_hashes"])
    write_json(state_path, state)
    state_path.with_suffix(".sha256").write_text(file_hash(state_path) + "\n", encoding="ascii")
    return result


class ResearchLabAuditTests(unittest.TestCase):
    def test_complete_chain_and_reaudit_are_read_only_and_keep_literal_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, _ = fixture(root)
            result = terminal(root, case, plan, worker)
            self.assertEqual(result["contract_errors"], [])
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["quality_verified"])
            self.assertEqual(result["audit"]["metric"]["value"], 1)
            self.assertEqual([row["class"] for row in result["audit"]["per_class"]], ["00", "nan"])
            self.assertEqual(result["provenance"]["refit"]["fit_rows"], 60)
            self.assertFalse(result["quality_admission"])
            before = {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}
            with patch.object(audit, "_load", return_value=plan):
                report = audit.re_audit(root)
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["arms"][0]["status"], "completed")
            self.assertEqual(before, {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()})

    def test_nonce_and_exact_feature_values_are_semantically_checked(self):
        for mutation in ("nonce", "fit_value", "predict_value", "checkpoint", "outer_metric"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, run = fixture(root)
                report = json.loads((run / "run.json").read_text())
                refit = Path(report["candidate_receipts"]["refit"]).parent
                if mutation == "nonce":
                    receipt = json.loads((refit / "receipt.json").read_text())
                    receipt["nonce"] = "d" * 32
                    write_json(refit / "receipt.json", receipt)
                elif mutation in {"fit_value", "predict_value"}:
                    path = refit / ("fit.csv" if mutation == "fit_value" else "predict.csv")
                    with path.open(newline="") as stream:
                        rows = list(csv.reader(stream))
                    rows[1][1] = "changed-feature-value"
                    write_csv(path, rows)
                elif mutation == "checkpoint":
                    (run / "last_valid_submission.csv").write_bytes(b"wrong checkpoint\n")
                else:
                    report["evaluations"]["final_check"]["baseline"]["value"] = .125
                    write_json(run / "run.json", report)
                    value = json.loads(worker.read_text())
                    value["run_report"]["sha256"] = file_hash(run / "run.json")
                    write_json(worker, value)
                    write_json(Path(case["arms"]["incumbent"]["task_dir"]) / "solver_output" /
                               "research_benchmark.json", {"run_report": value["run_report"]["path"],
                                   "run_report_sha256": value["run_report"]["sha256"]})
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertFalse(result["quality_verified"], result)
                self.assertEqual(result["status"], "unverified_contract")

    def test_probability_columns_follow_literal_class_mapping_not_output_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, _ = fixture(root, probabilities=True)
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertEqual(result["contract_errors"], [])
            self.assertEqual(result["status"], "completed")
            self.assertAlmostEqual(result["audit"]["metric"]["value"], -math.log(.9))
            self.assertEqual([row["class"] for row in result["audit"]["per_class"]], ["00", "nan"])

    def test_report_selection_never_uses_a_valid_old_glob_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, _ = fixture(root)
            value = json.loads(worker.read_text())
            value["run_report"]["path"] = ".agent_runs/" + "e" * 32 + "/run.json"
            write_json(worker, value)
            result = audit.audit_arm(root, case, "incumbent", worker, plan)
            self.assertFalse(result["quality_verified"])
            self.assertIsNone(result["research"])

    def test_terminal_state_csv_receipt_and_failed_log_drift_are_rejected(self):
        for mutation in ("state", "csv", "receipt", "failed_log", "extra_file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, run = fixture(root)
                state_path = Path(case["arms"]["incumbent"]["state_path"])
                if mutation == "failed_log":
                    state = json.loads(state_path.read_text())
                    state["status"] = "failed"
                    state["execution"]["exit_code"] = 1
                    write_json(state_path, state)
                    log = worker.parent / "failed.stderr.txt"
                    log.write_text("original failure")
                terminal(root, case, plan, worker)
                if mutation == "state":
                    value = json.loads(state_path.read_text())
                    value["execution"]["elapsed_seconds"] = .1
                    write_json(state_path, value)
                elif mutation == "csv":
                    (run / "last_valid_submission.csv").write_text("tampered")
                elif mutation == "receipt":
                    report = json.loads((run / "run.json").read_text())
                    Path(report["candidate_receipts"]["outer"]).write_text("{}")
                elif mutation == "failed_log":
                    log.write_text("failure hidden")
                else:
                    (worker.parent / "new.csv").write_text("unexpected file")
                with patch.object(audit, "_load", return_value=plan):
                    result = audit.re_audit(root)
                self.assertFalse(result["valid"], result)
                self.assertEqual(len(result["arms"]), 1)

    def test_fallback_and_unknown_cleanup_never_become_quality_wins(self):
        for mutation in ("prior", "unknown_failed_cleanup", "unknown_success_inventory"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, run = fixture(root)
                state_path = Path(case["arms"]["incumbent"]["state_path"])
                state = json.loads(state_path.read_text())
                if mutation == "prior":
                    report = json.loads((run / "run.json").read_text())
                    report["publication"]["new_result_published"] = False
                    report["status"] = "fallback"
                    write_json(run / "run.json", report)
                    value = json.loads(worker.read_text())
                    value["run_report"]["sha256"] = file_hash(run / "run.json")
                    write_json(worker, value)
                else:
                    state["execution"]["resources"] = {"available": False}
                    state["execution"].pop("cleanup")
                    if mutation == "unknown_failed_cleanup":
                        state["execution"]["exit_code"] = 1
                    write_json(state_path, state)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                if mutation == "unknown_success_inventory":
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["resources"]["qualification"], "provisional_resource_observation")
                else:
                    self.assertFalse(result["quality_verified"])
                if mutation == "unknown_failed_cleanup":
                    self.assertEqual(result["status"], "censored")

    def test_interrupted_started_run_is_censored_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker, _ = fixture(root)
            state_path = Path(case["arms"]["incumbent"]["state_path"])
            write_json(state_path, {"status": "started", "run_id": "fixture-run"})
            before = file_hash(state_path)
            with patch.object(audit, "_load", return_value=plan):
                result = audit.re_audit(root)
            self.assertEqual(result["arms"][0]["status"], "censored")
            self.assertFalse(result["arms"][0]["quality_verified"])
            self.assertEqual(file_hash(state_path), before)

    def test_missing_child_file_retains_supervisor_failure_and_available_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, plan, worker_path, _ = fixture(root)
            worker_path.unlink()
            state_path = Path(case["arms"]["incumbent"]["state_path"])
            state = json.loads(state_path.read_text())
            state["execution"]["exit_code"] = 1
            state["status"] = "failed"
            write_json(state_path, state)
            config_path = Path(case["arms"]["incumbent"]["config_path"])
            evidence = {"version": 1, "run_id": "fixture-run", "arm": "incumbent",
                        "config_sha256": file_hash(config_path), "solver_returned": False,
                        "run_report": None, "observations": []}
            result = terminal(root, case, plan, evidence)
            self.assertEqual(result["status"], "solver_failed")
            self.assertFalse(result["quality_verified"])
            self.assertTrue(result["artifact_hashes"])
            with patch.object(audit, "_load", return_value=plan):
                report = audit.re_audit(root)
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["arms"][0]["status"], "solver_failed")
            self.assertFalse(worker_path.exists())

    def test_sealed_inputs_and_answers_are_not_silently_repaired(self):
        for mutation in ("answers", "input"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case, plan, worker, _ = fixture(root)
                if mutation == "answers":
                    path = Path(case["audit_path"])
                    value = json.loads(path.read_text())
                    value["audit_actuals"].reverse()
                    write_json(path, value)
                else:
                    task = Path(case["arms"]["incumbent"]["task_dir"])
                    path = task / next(name for name in case["input_hashes"] if name.endswith("train.csv"))
                    path.write_text(path.read_text().replace("train-000", "changed-id"))
                digest = file_hash(path)
                result = audit.audit_arm(root, case, "incumbent", worker, plan)
                self.assertEqual(result["status"], "integrity_failed")
                self.assertFalse(result["quality_verified"])
                self.assertEqual(file_hash(path), digest)

    def test_guards_remain_active_under_python_optimized_mode(self):
        command = [sys.executable, "-O", "-c", (
            "from research_lab.audit import _verify_index\n"
            "try: _verify_index('.', {'missing':'invalid'})\n"
            "except ValueError: print('guard-active')\n"
            "else: raise SystemExit('guard-disabled')\n")]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "guard-active")

    def test_only_published_transport_spools_are_excluded_and_failure_logs_remain_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "arm"
            contents = {
                "execution/stdout.txt": "published bounded stdout",
                "execution/stderr.txt": "published bounded failure stderr",
                "probe_execution/stdout.txt": "published probe stdout",
                "probe_execution/.stderr.raw": "no published stderr tail: preserve failure proof",
                "other/.stdout.raw": "raw file outside known supervisor directory",
                "execution/failed.stderr.txt": "real failed child log",
            }
            for name, content in contents.items():
                path = arm / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            before = audit._index(arm, root)
            for name in ("execution/.stdout.raw", "execution/.stderr.raw", "probe_execution/.stdout.raw",
                         "execution/.rsync-tmp/.stderr.raw", "nested/.rsync-tmp/receipt.json"):
                path = arm / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("late transport delivery")
            after = audit._index(arm, root)
            self.assertEqual(before, after)
            self.assertEqual({"arm/" + name: value for name, value in file_index(arm).items()}, after)
            for name in contents:
                self.assertIn("arm/" + name, after)
            for name in ("execution/stdout.txt", "execution/stderr.txt", "execution/failed.stderr.txt",
                         "probe_execution/.stderr.raw", "other/.stdout.raw"):
                path = arm / name
                original = path.read_text()
                path.write_text("changed proof")
                with self.assertRaisesRegex(ValueError, "Artifact drift"):
                    audit._verify_index(root, before)
                path.write_text(original)
            with self.assertRaisesRegex(ValueError, "reserved transport"):
                audit._path(arm, "nested/.rsync-tmp/receipt.json")


if __name__ == "__main__":
    unittest.main()
