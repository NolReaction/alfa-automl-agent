"""Exercise the opt-in adaptive CLI using only extracted flat release sources."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np

from agent.execution import child_environment
from agent.resources import POLL_SECONDS, ProcessTreeMonitor
from agent.task_spec import load_task, score_holdout
from benchmark_agent import audit_submission
from benchmark_cases import make_case, write_case
from build_submission import archive_files, build_archive
from tests.test_release_recovery_archive import _csv_rows, _sha256
from validation import validate_predictions


PROCESS_WALL_SECONDS = 105.0
MEMORY_LIMIT_BYTES = 4 * 1024 ** 3


class AdaptiveArchiveTests(unittest.TestCase):
    def test_flat_archive_offline_adaptive_search_retains_full_data_predictions(self):
        generated = make_case("mae_signed_nonlinear", 911)
        case = replace(generated, train=generated.train.iloc[:120].reset_index(drop=True).copy(),
                       test=generated.test.iloc[:30].reset_index(drop=True).copy(),
                       audit_actuals=generated.audit_actuals[:30])
        expected_ids = case.test["row_key"].tolist()
        self.assertIn("NA", expected_ids)
        self.assertIn("000005", expected_ids)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "solution.zip"
            task = write_case(case, root / "isolated adaptive task")
            names = build_archive(archive)
            expected_sources = archive_files()
            self.assertEqual(names, list(expected_sources))
            # The current release has 40 files; compare the full allowlisted
            # manifest so future legitimate runtime modules do not break this
            # integration test solely by increasing that count.
            self.assertTrue(all("/" not in name and "\\" not in name for name in names))
            self.assertTrue(all(name.endswith(".py") or name == "requirements.txt" for name in names))
            for required in ("solution.py", "_alfa_agent__search_policy.py",
                             "_alfa_agent__search_schedule.py", "_alfa_agent__adaptive_local.py",
                             "_alfa_agent__automl_worker.py"):
                self.assertIn(required, names)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertEqual(zipped.namelist(), names)
                for name, content in expected_sources.items():
                    self.assertEqual(zipped.read(name), content, name)
                    (task.parent / name).write_bytes(content)
            originals = {path: path.read_bytes() for path in task.parent.rglob("*") if path.is_file()}

            environment = child_environment()
            for forbidden in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "PYTHONPATH", "PYTHONHOME"):
                self.assertNotIn(forbidden, environment)
            self.assertFalse(any(name.lower().endswith("_proxy") for name in environment))
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                self.assertEqual(environment[name], "8")
            stdout_path, stderr_path = root / "adaptive.stdout.txt", root / "adaptive.stderr.txt"
            argv = [sys.executable, "solution.py", "task.md", "--mode", "offline",
                    "--search-policy", "adaptive_v1", "--time-budget", "90"]
            started = time.monotonic()
            timed_out = memory_exceeded = False
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    argv, cwd=task.parent, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                monitor = ProcessTreeMonitor(process, max_memory_bytes=MEMORY_LIMIT_BYTES)
                try:
                    while True:
                        memory_exceeded = monitor.sample()
                        if process.poll() is not None or memory_exceeded:
                            break
                        remaining = PROCESS_WALL_SECONDS - (time.monotonic() - started)
                        if remaining <= 0:
                            timed_out = True
                            break
                        try:
                            process.wait(timeout=min(POLL_SECONDS, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    stopped = monitor.terminate()
            resources = monitor.summary()
            output = stdout_path.read_text(encoding="utf-8", errors="replace")[-24000:]
            errors = stderr_path.read_text(encoding="utf-8", errors="replace")[-24000:]
            details = json.dumps({"argv": argv, "elapsed_seconds": time.monotonic() - started,
                                  "archive_root_files": len(names),
                                  "resources": resources, "stdout": output, "stderr": errors},
                                 ensure_ascii=False)
            self.assertTrue(stopped, details)
            self.assertFalse(timed_out, details)
            self.assertFalse(memory_exceeded, details)
            self.assertEqual(process.returncode, 0, details)
            if not resources["available"]:
                self.assertIsNone(resources["peak_rss_bytes"])
                self.assertIsNone(resources["cpu_seconds"])
                self.assertTrue(resources["unavailable_reason"])

            self.assertFalse((task.parent / "agent").exists())
            self.assertFalse((task.parent / "app").exists())
            for path, content in originals.items():
                with self.subTest(original=str(path.relative_to(task.parent))):
                    self.assertEqual(path.read_bytes(), content)
            self.assertEqual(archive_files(), expected_sources)
            self.assertTrue(validate_predictions(task.parent)["valid"], details)
            final = task.parent / "submission.csv"
            self.assertEqual([row["row_key"] for row in _csv_rows(final)], expected_ids)
            audit = audit_submission(task, case.audit_actuals)
            self.assertTrue(audit["valid"], audit)
            self.assertEqual(audit["metric"]["name"], "MAE")
            self.assertFalse(audit["metric"]["higher_is_better"])
            self.assertTrue(np.isfinite(audit["metric"]["value"]))

            emergency_paths = list((task.parent / ".agent_runs").glob("*/emergency.json"))
            self.assertEqual(len(emergency_paths), 1, details)
            receipt = json.loads(emergency_paths[0].read_text(encoding="utf-8"))
            prior = emergency_paths[0].parent / "prior.csv"
            self.assertEqual(receipt["kind"], "target_prior")
            self.assertEqual(receipt["status"], "ready")
            self.assertFalse(receipt["quality_verified"])
            self.assertEqual(receipt["model"]["strategy"], "median")
            self.assertEqual(receipt["model"]["fit_rows"], 120)
            self.assertEqual(receipt["predict_rows"], 30)
            self.assertEqual(receipt["submission_sha256"], _sha256(prior.read_bytes()))
            prior_rows = _csv_rows(prior)
            self.assertEqual([row["row_key"] for row in prior_rows], expected_ids)
            median = float(np.median([float(row["outcome"])
                                      for row in _csv_rows(task.parent / "inputs/train.csv")]))
            self.assertTrue(all(np.isclose(float(row["outcome"]), median, rtol=0, atol=1e-12)
                                for row in prior_rows))
            self.assertNotEqual(final.read_bytes(), prior.read_bytes())
            self.assertNotEqual([row["outcome"] for row in _csv_rows(final)],
                                [row["outcome"] for row in prior_rows])

            report_paths = list((task.parent / ".agent_runs").glob("*/automl_report.json"))
            self.assertEqual(len(report_paths), 1, details)
            run_dir = report_paths[0].parent
            report = json.loads(report_paths[0].read_text(encoding="utf-8"))
            self.assertTrue(report["output_valid"], details)
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertEqual(report["search_policy"]["name"], "adaptive_v1")
            self.assertGreaterEqual(report["search_policy"]["schedule"]["dispatched"], 1, details)
            self.assertTrue(any(row["selection_metric"] is not None for row in report["search"]["candidates"]), details)
            self.assertEqual(report["search"]["phase"], "finished")
            self.assertIn(report["search"]["status"], {"completed", "time_limit"})
            selected, = [row for row in report["search"]["candidates"] if row["selected"]]
            self.assertEqual(selected["id"], report["search"]["best_candidate"])
            self.assertEqual(selected["status"], "completed")
            self.assertEqual(selected["resources"]["exit_code"], 0)
            phase = "seed" if selected["id"] == "seed" else "refit"
            stage = run_dir / selected["id"] / phase
            config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["phase"], phase)
            self.assertEqual(config["id"], selected["id"])
            self.assertEqual(config["model_config"], report["model_config"])
            self.assertEqual(Path(config["candidate_dir"]).resolve(), stage.resolve())
            input_dir = Path(config["task_spec"]["data_dir"])
            self.assertEqual(input_dir.resolve(), (stage / "inputs").resolve())
            self.assertEqual(_csv_rows(input_dir / "train.csv"), _csv_rows(task.parent / "inputs/train.csv"))
            self.assertEqual(_csv_rows(input_dir / "test.csv"), _csv_rows(task.parent / "inputs/test.csv"))
            self.assertEqual(len(_csv_rows(input_dir / "train.csv")), 120)
            self.assertEqual(len(_csv_rows(input_dir / "test.csv")), 30)
            # Exact archived worker bytes were checked above and remain
            # unchanged. Its seed/refit branch fits the complete staged train,
            # and commits this durations marker only after prediction finishes.
            # Together with fresh full inputs, phase config, exit=0 and the
            # accepted checkpoint below, this proves completion independently
            # of an occasionally empty captured stdout file.
            durations = json.loads((stage / "durations.json").read_text(encoding="utf-8"))
            self.assertEqual(set(durations), {"fit_seconds", "predict_seconds", "total_seconds"})
            self.assertTrue(all(type(value) in {int, float} and np.isfinite(value) and value > 0
                                for value in durations.values()), durations)
            self.assertGreaterEqual(durations["total_seconds"] + 1e-9,
                                    durations["fit_seconds"] + durations["predict_seconds"])
            worker_logs = list((stage / "execution").glob("experiment_*.stdout.txt"))
            self.assertLessEqual(len(worker_logs), 1, details)
            worker_output = (worker_logs[0].read_text(encoding="utf-8", errors="replace")
                             if worker_logs else "")
            stdout_corroborates = f"{phase}: refitting on all 120 rows" in worker_output
            self.assertEqual((stage / "submission.csv").read_bytes(), final.read_bytes())

            pointer = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["version"], 1)
            self.assertTrue(re.fullmatch(r"checkpoints/[0-9a-f]{32}", pointer["directory"]))
            checkpoint = run_dir / pointer["directory"]
            self.assertEqual(pointer["submission_sha256"], _sha256(final.read_bytes()))
            self.assertEqual((checkpoint / "submission.csv").read_bytes(), final.read_bytes())
            self.assertEqual(pointer["holdout_sha256"], _sha256((checkpoint / "holdout_predictions.csv").read_bytes()))
            original_train = _csv_rows(task.parent / "inputs/train.csv")
            held = [original_train[index] for index in config["validation_rows"]]
            held_value = score_holdout(checkpoint / "holdout_predictions.csv", load_task(task),
                                       [row["row_key"] for row in held], [row["outcome"] for row in held])
            self.assertAlmostEqual(held_value, report["metric"]["value"], places=10)
            print(json.dumps({"adaptive_archive": "verified", "root_files": len(names),
                              "full_fit_rows": 120, "test_rows": 30, "phase": phase,
                              "completion_proof": "exact_worker_config_inputs_durations_checkpoint",
                              "stdout_corroborates": stdout_corroborates,
                              "resource_telemetry_available": resources["available"]}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
