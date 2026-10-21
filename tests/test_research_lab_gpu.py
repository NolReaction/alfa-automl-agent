"""GPU exploration protocol tests. All model calls are fakes; no CUDA fit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.metadata
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agent.metrics import get_metric
from agent.task_spec import TaskSpec
from research_lab import gpu
from research_lab.integrity import file_hash, read_json, write_json


class GPUFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu lab Ω ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def spec(self, metric="LogLoss"):
        classification = metric != "RMSE"
        probabilities = metric == "LogLoss"
        return TaskSpec(self.root / "task.md", self.root / "inputs", "row", "answer",
                        ("x", "kind"), ("x",), ("kind",), ("01", "NA", "class c") if classification else (),
                        ("row", "pc", "pna", "p01") if probabilities else ("row", "answer"),
                        (("pc", "class c"), ("pna", "NA"), ("p01", "01")) if probabilities else (),
                        None, get_metric(metric))


class GPUModelContractTests(GPUFixture):
    def test_static_conversion_keeps_literal_categories_and_never_fits_predict_statistics(self):
        spec = self.spec()
        fit = pd.DataFrame({"x": ["", "1", "2"], "kind": ["NA", "0007", ""]})
        predict = pd.DataFrame({"x": ["9000", ""], "kind": ["7", "unseen"]})
        before = fit.copy(deep=True)
        transformed = gpu._features(fit, spec)
        future = gpu._features(predict, spec)
        self.assertTrue(math.isnan(transformed.x.iloc[0]))
        self.assertEqual(future.x.iloc[0], 9000.0)
        self.assertTrue(math.isnan(future.x.iloc[1]))
        self.assertEqual(transformed.kind.tolist(), ["NA", "0007", ""])
        self.assertEqual(future.kind.tolist(), ["7", "unseen"])
        pd.testing.assert_frame_equal(fit, before)
        with self.assertRaises(ValueError): gpu._features(fit[["kind", "x"]], spec)
        with self.assertRaises(ValueError): gpu._features(pd.DataFrame({"x": ["inf"], "kind": ["NA"]}), spec)

    def test_one_fixed_gpu_fit_preserves_probability_class_order_and_exact_ids(self):
        spec = self.spec()
        train = pd.DataFrame({"row": ["a", "b", "c"], "x": ["1", "2", ""],
                              "kind": ["NA", "0007", ""], "answer": ["01", "NA", "class c"]})
        test = pd.DataFrame({"row": ["0007", "NA"], "x": ["8", "9"], "kind": ["new", ""]})
        model = Mock(classes_=np.asarray(["class c", "NA", "01"]))
        model.predict_proba.return_value = np.asarray([[.2, .3, .5], [.1, .6, .3]])
        factory = Mock(return_value=model)
        fake = SimpleNamespace(CatBoostClassifier=factory, CatBoostRegressor=Mock(side_effect=AssertionError("Wrong model")))
        with patch.dict(sys.modules, {"catboost": fake}):
            output, parameters, classes = gpu._fit_predict(train, test, spec, 711)
        factory.assert_called_once()
        model.fit.assert_called_once()
        self.assertEqual(model.fit.call_args.kwargs, {"cat_features": ["kind"]})
        self.assertEqual(model.fit.call_args.args[1], ["01", "NA", "class c"])
        self.assertEqual(parameters["task_type"], "GPU")
        self.assertEqual(parameters["devices"], "0")
        self.assertEqual(parameters["thread_count"], 8)
        self.assertEqual(parameters["random_seed"], 711)
        self.assertEqual(parameters["iterations"], 256)
        self.assertEqual(classes, ["class c", "NA", "01"])
        self.assertEqual(output.row.tolist(), ["0007", "NA"])
        np.testing.assert_array_equal(output[["pc", "pna", "p01"]], [[.2, .3, .5], [.1, .6, .3]])
        self.assertNotIn("eval_set", model.fit.call_args.kwargs)

    def test_regression_uses_numeric_train_target_and_no_audit_labels(self):
        spec = self.spec("RMSE")
        train = pd.DataFrame({"row": ["a", "b"], "x": ["1", "2"], "kind": ["NA", ""], "answer": ["-1.2", "7"]})
        test = pd.DataFrame({"row": ["0007"], "x": ["8"], "kind": ["new"]})
        model = Mock()
        model.predict.return_value = np.asarray([2.5])
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"catboost": SimpleNamespace(CatBoostClassifier=Mock(), CatBoostRegressor=factory)}):
            output, parameters, classes = gpu._fit_predict(train, test, spec, 8)
        np.testing.assert_array_equal(model.fit.call_args.args[1], [-1.2, 7.0])
        self.assertEqual(parameters["loss_function"], "RMSE")
        self.assertIsNone(classes)
        self.assertEqual(output.answer.tolist(), [2.5])


class GPUProbeTests(GPUFixture):
    def test_missing_library_and_driver_are_explicit_skips_not_install_attempts(self):
        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError), \
                patch.object(gpu.shutil, "which", return_value=None), patch.object(gpu.subprocess, "run") as shell:
            value = gpu.probe_worker(self.root / "capability.json")
        self.assertFalse(value["available"])
        self.assertEqual(value["reason"], "catboost_not_installed")
        self.assertIsNone(value["libraries"]["lightgbm"])
        self.assertFalse(value["lightgbm"]["gpu_verified"])
        shell.assert_not_called()
        self.assertEqual(read_json(self.root / "capability.json"), value)

    def test_actual_catboost_device_api_is_used_without_model_fit(self):
        device_count = Mock(return_value=0)
        with patch.object(importlib.metadata, "version", return_value="1.2.8"), \
                patch.object(gpu.shutil, "which", return_value=None), \
                patch.dict(sys.modules, {"catboost.utils": SimpleNamespace(get_gpu_device_count=device_count)}):
            value = gpu.probe_worker(self.root / "capability.json")
        device_count.assert_called_once_with()
        self.assertFalse(value["available"])
        self.assertEqual(value["reason"], "no_usable_catboost_cuda_device")
        self.assertEqual(value["catboost_gpu_devices"], 0)

    def test_device_driver_and_library_evidence_are_retained(self):
        device_count = Mock(return_value=1)
        process = SimpleNamespace(returncode=0, stdout="0, NVIDIA GeForce RTX 3080, 555.85, 10240 MiB\n", stderr="")
        with patch.object(importlib.metadata, "version", return_value="fixture-version"), \
                patch.object(gpu.shutil, "which", return_value="nvidia-smi"), \
                patch.object(gpu.subprocess, "run", return_value=process) as shell, \
                patch.dict(sys.modules, {"catboost.utils": SimpleNamespace(get_gpu_device_count=device_count)}):
            value = gpu.probe_worker(self.root / "capability.json")
        self.assertTrue(value["available"])
        self.assertIn("RTX 3080", value["hardware"][0])
        self.assertEqual(shell.call_args.args[0][0], "nvidia-smi")
        self.assertEqual(shell.call_args.kwargs["timeout"], 8)
        self.assertFalse(value["lightgbm"]["gpu_verified"])


class GPUEnvironmentTests(GPUFixture):
    def test_gpu_environment_is_sanitized_bounded_and_restored_on_failure(self):
        import qualify_unseen_tasks as trusted
        original = trusted.sanitized_environment
        with patch.dict(os.environ, {"OPENAI_API_KEY": "never-child", "HTTPS_PROXY": "never-child", "PYTHONPATH": "never-child"}):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with gpu._gpu_environment():
                    environment = trusted.sanitized_environment()
                    self.assertNotIn("OPENAI_API_KEY", environment)
                    self.assertNotIn("HTTPS_PROXY", environment)
                    self.assertNotIn("PYTHONPATH", environment)
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
                    self.assertEqual(environment["OMP_NUM_THREADS"], "8")
                    self.assertEqual(environment["MKL_NUM_THREADS"], "8")
                    raise RuntimeError("fixture failure")
        self.assertIs(trusted.sanitized_environment, original)

    def test_same_trusted_supervisor_enforces_host_ram_and_original_timeout(self):
        import qualify_unseen_tasks as trusted
        marker = {"elapsed_seconds": 3.0}
        with patch.object(trusted, "run_solver", return_value=marker) as supervise:
            result = gpu._supervise(["python", "worker.py"], self.root, self.root / "execution", 17.5)
        self.assertIs(result, marker)
        self.assertEqual(result["threads"], 8)
        self.assertEqual(result["memory_limit_bytes"], 24 * 1024**3)
        supervise.assert_called_once_with(["python", "worker.py"], self.root, self.root / "execution", 17.5, 24 * 1024**3)
        for invalid in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): gpu._supervise([], self.root, self.root, invalid)


class GPURunnerTests(GPUFixture):
    def setUp(self):
        super().setUp()
        self.cpu = self.root / "CPU pilot"
        self.cpu.mkdir()
        self.plan = {"plan_id": "cpu-fixture", "python": sys.executable, "environment": {}, "profile": "pilot",
                     "cases": [{"key": "case" + str(i), "seed": 17 + i} for i in range(4)]}
        write_json(self.cpu / "plan.json", self.plan)
        self.capability = {"version": 1, "available": False, "reason": "no_usable_catboost_cuda_device"}
        self.gate = {"allowed": True, "reason": "cpu_pilot_terminal_and_reaudited", "terminal_arms": 8}

    def fake_probe(self, argv, cwd, output, timeout):
        self.assertEqual(timeout, gpu.PROBE_SECONDS)
        self.assertNotIn("audit", " ".join(argv).lower())
        write_json(Path(cwd) / "capability.json", self.capability)
        return {"exit_code": 0, "external_timeout": False, "memory_limit_exceeded": False, "cleanup": {"complete": True}}

    def run_gpu(self, output="GPU run", **kwargs):
        with patch("research_lab.planning.load_plan", return_value=self.plan), \
                patch.object(gpu, "_pilot_gate", return_value=self.gate), \
                patch.object(gpu, "_supervise", side_effect=self.fake_probe):
            return gpu.run_gpu(self.root / output, self.cpu, enabled=True, **kwargs)

    def test_disabled_default_never_probes_loads_cpu_plan_or_trains(self):
        with patch.object(gpu, "_supervise") as probe, patch.object(gpu, "_run_case") as fit, \
                patch("research_lab.planning.load_plan") as load:
            value = gpu.run_gpu(self.root / "disabled", self.root / "missing CPU")
        self.assertEqual(value["status"], "skipped")
        self.assertEqual(value["reason"], "explicit_opt_in_required")
        self.assertFalse(value["cpu_qualification"])
        probe.assert_not_called(); fit.assert_not_called(); load.assert_not_called()
        with patch.object(gpu, "_supervise", side_effect=AssertionError("Re-audit cannot run a child")):
            self.assertTrue(gpu.re_audit_gpu(self.root / "disabled")["verified"])

    def test_unavailable_gpu_keeps_every_cpu_case_in_skip_denominator(self):
        with patch.object(gpu, "_run_case") as fit:
            result = self.run_gpu()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(result["cases"]), 4)
        self.assertTrue(all(row["reason"] == "no_usable_catboost_cuda_device" and row["metric"] is None for row in result["cases"]))
        fit.assert_not_called()
        self.assertTrue((self.root / "GPU run/gpu_seal.json").is_file())
        with patch("research_lab.planning.load_plan", return_value=self.plan), \
                patch.object(gpu, "_supervise", side_effect=AssertionError("Re-audit cannot probe hardware")), \
                patch.object(gpu, "write_json", side_effect=AssertionError("Re-audit cannot repair files")):
            self.assertEqual(gpu.re_audit_gpu(self.root / "GPU run")["cases"], 4)
        path = self.root / "GPU run/capability.json"
        path.write_bytes(path.read_bytes() + b" ")
        with patch("research_lab.planning.load_plan", return_value=self.plan), self.assertRaisesRegex(ValueError, "capability drift"):
            gpu.re_audit_gpu(self.root / "GPU run")

    def test_incomplete_cpu_pilot_blocks_gpu_fit_even_with_usable_device(self):
        self.gate.update(allowed=False, reason="cpu_pilot_incomplete_or_censored", terminal_arms=3)
        self.capability.update(available=True, reason="catboost_cuda_device_available")
        with patch.object(gpu, "_run_case") as fit:
            result = self.run_gpu()
        self.assertEqual(result["reason"], "cpu_pilot_incomplete_or_censored")
        self.assertEqual(len(result["cases"]), 4)
        fit.assert_not_called()

    def test_unknown_probe_cleanup_blocks_any_gpu_training(self):
        self.capability.update(available=True, reason="catboost_cuda_device_available")
        def probe(*args):
            value = self.fake_probe(*args)
            value["cleanup"]["complete"] = None
            return value
        with patch("research_lab.planning.load_plan", return_value=self.plan), patch.object(gpu, "_pilot_gate", return_value=self.gate), \
                patch.object(gpu, "_supervise", side_effect=probe), patch.object(gpu, "_run_case") as fit:
            result = gpu.run_gpu(self.root / "unclean", self.cpu, enabled=True)
        self.assertEqual(result["reason"], "capability_probe_cleanup_unverified")
        fit.assert_not_called()

    def test_failure_stops_series_and_never_retries_started_case(self):
        self.capability.update(available=True, reason="catboost_cuda_device_available")
        fake = {"case": "case0", "status": "censored", "reason": "gpu_cleanup_unverified", "stop_series": True, "metric": None}
        with patch.object(gpu, "_run_case", return_value=fake) as fit:
            result = self.run_gpu()
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                self.run_gpu()
        fit.assert_called_once()
        self.assertEqual(result["status"], "censored")
        self.assertEqual(result["cases"][0]["status"], "censored")
        self.assertTrue(all(row["status"] == "skipped" for row in result["cases"][1:]))

    def test_opt_in_and_budget_types_refuse_before_any_child(self):
        for arguments in ({"enabled": 1}, {"time_budget_seconds": 0}, {"time_budget_seconds": 601}, {"time_budget_seconds": True}):
            with self.subTest(arguments=arguments), patch.object(gpu, "_supervise") as child:
                with self.assertRaises(ValueError): gpu.run_gpu(self.root / "invalid", self.cpu, **arguments)
                child.assert_not_called()


class GPUPilotGateTests(GPUFixture):
    def test_real_cpu_state_seals_and_cleanup_are_required_before_fit(self):
        plan = {"profile": "pilot", "cases": []}
        for index in range(4):
            case = {"arms": {}}
            for arm in ("incumbent", "candidate"):
                relative = f"cases/{index}/{arm}/state.json"
                path = self.root / relative
                write_json(path, {"status": "finished", "execution": {"cleanup": {"complete": True}}})
                path.with_suffix(".sha256").write_text(file_hash(path) + "\n", encoding="ascii")
                case["arms"][arm] = {"state_path": relative}
            plan["cases"].append(case)
        with patch("research_lab.audit.re_audit", return_value={"valid": True}):
            gate = gpu._pilot_gate(self.root, plan)
            self.assertTrue(gate["allowed"])
            self.assertEqual(gate["terminal_arms"], 8)
            self.assertEqual(len(gate["state_sha256"]), 8)
            path = self.root / plan["cases"][0]["arms"]["candidate"]["state_path"]
            state = read_json(path)
            state["execution"]["cleanup"]["complete"] = None
            write_json(path, state)
            path.with_suffix(".sha256").write_text(file_hash(path) + "\n", encoding="ascii")
            self.assertEqual(gpu._pilot_gate(self.root, plan)["reason"], "cpu_pilot_cleanup_unverified")
            write_json(path, {"status": "started"})
            self.assertEqual(gpu._pilot_gate(self.root, plan)["reason"], "cpu_pilot_incomplete_or_censored")


class GPUWorkerReceiptTests(GPUFixture):
    """Actual worker/CSV/auditor wiring with exactly one fake model call."""
    def setUp(self):
        super().setUp()
        self.directory = self.root / "GPU case"
        self.task = self.directory / "task"
        (self.task / "inputs").mkdir(parents=True)
        self.contract = replace(self.spec("Macro F1"), task_path=self.task / "task.md", data_dir=self.task / "inputs")
        (self.task / "task.md").write_text("Protocol fixture; no real model fits.\n", encoding="utf-8")
        pd.DataFrame({"row": ["a", "b", "c"], "x": ["1", "2", "3"], "kind": ["NA", "0007", ""],
                      "answer": ["01", "NA", "class c"]}).to_csv(self.task / "inputs/train.csv", index=False)
        pd.DataFrame({"row": ["0007", "NA"], "x": ["8", "9"], "kind": ["new", ""]}).to_csv(self.task / "inputs/test.csv", index=False)
        pd.DataFrame({"feature": ["x", "kind"], "dtype": ["numeric", "category"]}).to_csv(self.task / "inputs/data_dictionary.csv", index=False)
        pd.DataFrame({"row": ["0007", "NA"], "answer": ["01", "01"]}).to_csv(self.task / "inputs/sample_submission.csv", index=False)
        self.hashes = gpu._input_hashes(self.task)
        self.config = {"version": 1, "run_id": "fake-current-run", "task_dir": str(self.task), "seed": 19,
                       "source_sha256": file_hash(gpu.__file__), "input_hashes": self.hashes, "parameters": gpu.FIXED_PARAMETERS}
        self.config_path = self.directory / "config.json"
        write_json(self.config_path, self.config)
        write_json(self.directory / "state.json", {"run_id": self.config["run_id"], "config_sha256": file_hash(self.config_path),
                                                     "started_monotonic": 0.0, "deadline": 120.0})
        self.plan = {"source_sha256": self.config["source_sha256"], "parameters": gpu.FIXED_PARAMETERS, "time_budget_seconds": 120}
        self.cpu = self.root / "CPU"
        audit = {"contract": {**self.contract.to_dict(), "prediction_kind": "labels"}, "metric": "Macro F1",
                 "test_ids": ["0007", "NA"], "audit_actuals": ["NA", "01"]}
        write_json(self.cpu / "audit.json", audit)
        self.case = {"key": "fixture", "seed": 19, "input_hashes": self.hashes, "audit_path": "audit.json",
                     "audit_sha256": file_hash(self.cpu / "audit.json")}
        self.prediction = pd.DataFrame({"row": ["0007", "NA"], "answer": ["NA", "01"]})
        self.parameters = {**gpu.FIXED_PARAMETERS, "random_seed": 19, "loss_function": "MultiClass"}

    def worker(self, deadline=120.0):
        with patch("agent.task_spec.load_task", return_value=self.contract), patch.object(gpu, "_NATIVE_STARTED", 1.0), \
                patch.object(gpu.time, "monotonic", return_value=2.0), \
                patch.object(gpu, "_fit_predict", return_value=(self.prediction, self.parameters, list(self.contract.classes))) as fit:
            receipt = gpu.worker(self.config_path, deadline)
        return receipt, fit

    def test_worker_receipt_binds_source_config_deadline_inputs_and_actual_csv_score(self):
        receipt, fit = self.worker()
        fit.assert_called_once()
        self.assertEqual(receipt["api_attempts"], 0)
        self.assertEqual(receipt["deadline"], 120.0)
        self.assertEqual(receipt["input_hashes"], self.hashes)
        self.assertEqual(receipt["config_sha256"], file_hash(self.config_path))
        self.assertNotIn("audit", read_json(self.config_path))
        self.assertEqual(gpu._input_hashes(self.task), self.hashes)
        result = gpu._audit_case(self.directory, self.cpu, self.case, self.plan)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["metric"]["value"], 1.0)
        receipt["deadline"] = 240.0
        write_json(self.directory / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "deadline"):
            gpu._audit_case(self.directory, self.cpu, self.case, self.plan)

    def test_expired_original_budget_stale_csv_and_input_drift_fail_without_retry(self):
        with patch.object(gpu.time, "monotonic", return_value=120.0), patch.object(gpu, "_fit_predict") as fit:
            with self.assertRaisesRegex(ValueError, "deadline"):
                gpu.worker(self.config_path, 120.0)
        fit.assert_not_called()
        (self.task / "submission.csv").write_text("stale\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.worker()
        (self.task / "submission.csv").unlink()
        (self.task / "inputs/test.csv").write_bytes((self.task / "inputs/test.csv").read_bytes() + b"\n")
        with patch.object(gpu, "_fit_predict") as fit:
            with self.assertRaisesRegex(ValueError, "input drift"):
                gpu.worker(self.config_path, time_budget_deadline())
        fit.assert_not_called()

    def test_audit_truth_fields_are_not_accepted_by_worker_config(self):
        write_json(self.config_path, {**self.config, "audit_actuals": ["NA", "01"]})
        with patch.object(gpu, "_fit_predict") as fit:
            with self.assertRaisesRegex(ValueError, "fields"):
                gpu.worker(self.config_path, 999999999999.0)
        fit.assert_not_called()


def time_budget_deadline():
    # Monotonic rather than epoch time; this helper never adds solver budget.
    import time
    return time.monotonic() + 120


if __name__ == "__main__":
    unittest.main()
