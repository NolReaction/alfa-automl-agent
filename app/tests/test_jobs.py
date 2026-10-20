"""Isolated training, explicit publication and process lifecycle checks."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from app import jobs
from app.jobs import JobManager
from app.profiling import DataCatalog


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.task = self.data / "example"
        self.task.mkdir(parents=True)
        (self.task / "task.md").write_text("Основная метрика: RMSE", encoding="utf-8")
        inputs = self.task / "nested" / "participants"
        csv_file(inputs / "data_dictionary.csv", [
            ["feature", "role", "dtype"], ["id", "id", "category"],
            ["x", "feature", "numeric"], ["answer", "target", "numeric"],
        ])
        csv_file(inputs / "train.csv", [["id", "x", "answer"]] + [
            [f"train-{n:03}", n, n * 2] for n in range(80)])
        csv_file(inputs / "test.csv", [["id", "x"], ["NA", 1], ["001", 2]])
        csv_file(inputs / "sample_submission.csv", [["id", "answer"], ["NA", 0], ["001", 0]])
        csv_file(self.task / "submission.csv", [["id", "answer"], ["NA", -1], ["001", -2]])
        self.state = self.root / "state"
        self.catalog = DataCatalog(self.data)
        self.manager = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(self.manager.close)

    def wait(self, job, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.manager.detail(job["task_id"], job["id"])["job"]
            if current["status"] not in {"queued", "running"}:
                return current
            time.sleep(0.02)
        self.fail("Job did not terminate within test timeout")

    def test_training_is_isolated_and_publication_preserves_backup(self):
        original = {str(path.relative_to(self.task)): path.read_bytes()
                    for path in self.task.rglob("*") if path.is_file()}
        results = []
        for action in ("baseline", "extra_trees"):
            job = self.manager.start("example", action, author="Команда")
            result = self.wait(job)
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertTrue(result["output_valid"])
            self.assertFalse(result["published"])
            self.assertFalse(result["api_used"])
            self.assertEqual(result["metric"]["name"], "RMSE")
            self.assertEqual(self.manager.artifact("example", job["id"]).name, "submission.csv")
            results.append(result)
        self.assertEqual(results[0]["comparison_key"], results[1]["comparison_key"])
        current = {str(path.relative_to(self.task)): path.read_bytes()
                   for path in self.task.rglob("*") if path.is_file()}
        self.assertEqual(current, original)
        published = self.manager.publish("example", results[1]["id"])
        self.assertTrue(published["published"])
        backups = list((self.state / "backups" / results[1]["id"]).glob("*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original["submission.csv"])
        self.assertFalse((self.task / "baseline_metrics.json").exists())
        self.assertEqual((self.task / "submission.csv").read_bytes(),
                         self.manager.artifact("example", results[1]["id"]).read_bytes())
        # A terminal run replacing the file must clear the *current* published indicator.
        (self.task / "submission.csv").write_bytes(original["submission.csv"])
        self.assertFalse(self.manager.detail("example", results[1]["id"])["job"]["published"])

    def test_automl_selects_best_without_api_and_survives_restore(self):
        original = (self.task / "submission.csv").read_bytes()
        with patch.object(jobs, "_settings_values", side_effect=AssertionError("AutoML must not read API settings")):
            # Redaction may read settings, so patch it independently: the launch
            # itself must never request or inject credentials into the worker.
            with patch.object(self.manager, "redact", side_effect=lambda task, job, value: value):
                job = self.manager.start("example", "automl", minutes=1)
                result = self.wait(job, timeout=60)
        self.assertEqual(result["status"], "completed", self.manager.detail("example", job["id"]))
        self.assertEqual(result["validation"], {"strategy": "auto"})
        self.assertEqual(result["split"]["strategy"], "random")
        self.assertFalse(result["api_used"])
        self.assertFalse(result["published"])
        self.assertEqual((self.task / "submission.csv").read_bytes(), original)
        search = result["search"]
        completed = [row for row in search["candidates"] if row["status"] == "completed"]
        self.assertEqual(search["completed"], len(completed))
        self.assertGreaterEqual(len(completed), 2)
        self.assertEqual(result["metric"]["value"], min(row["metric"] for row in completed))
        selected = next(row for row in completed if row["selected"])
        self.assertEqual(selected["id"], search["best_candidate"])
        self.assertEqual(selected["model"], result["model_config"]["model"])
        folder = self.state / "jobs" / job["id"]
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["time_budget"], 60)
        self.assertNotIn("model_params", config)
        reference = self.wait(self.manager.start("example", "extra_trees",
                                                 model_params={"n_estimators": 50}), timeout=60)
        self.assertEqual(reference["status"], "completed")
        self.assertEqual(result["comparison_key"], reference["comparison_key"])
        self.manager.close()
        restored = JobManager(self.data, self.state, self.catalog)
        try:
            saved = restored.detail("example", job["id"])["job"]
            self.assertEqual(saved["search"], search)
            self.assertEqual(saved["model_config"], result["model_config"])
            self.assertTrue(restored.publish("example", job["id"])["published"])
        finally:
            restored.close()

    def test_qwen_command_keeps_explicit_llm_mode(self):
        command = self.manager._command({"action": "agent"}, self.task / "task.md", 2)
        self.assertEqual(command[command.index("--mode") + 1], "llm")
        self.assertEqual(command[command.index("--time-budget") + 1], "120")

    def test_changed_data_and_unfinished_jobs_cannot_be_published(self):
        job = self.manager.start("example", "baseline")
        with self.assertRaises(ValueError):
            self.manager.publish("example", job["id"])
        result = self.wait(job)
        self.assertEqual(result["status"], "completed", result.get("error"))
        artifact = self.manager.artifact("example", job["id"])
        original_artifact = artifact.read_bytes()
        csv_file(artifact, [["id", "answer"], ["NA", 999], ["001", 999]])
        with self.assertRaisesRegex(ValueError, "изменён"):
            self.manager.artifact("example", job["id"])
        artifact.write_bytes(original_artifact)
        with (self.task / "task.md").open("a", encoding="utf-8") as stream:
            stream.write("\nChanged task")
        with self.assertRaises(ValueError):
            self.manager.publish("example", job["id"])

    def test_time_validation_reaches_both_recipes_and_survives_restart(self):
        validation = {"strategy": "time", "time_column": "x"}
        results = []
        for action in ("baseline", "extra_trees"):
            job = self.manager.start("example", action, validation=validation)
            result = self.wait(job)
            self.assertEqual(result["status"], "completed", self.manager.detail("example", job["id"]))
            self.assertEqual(result["validation"], validation)
            self.assertEqual(result["split"]["strategy"], "time")
            self.assertEqual(result["split"]["time_column"], "x")
            self.assertFalse(result["split"]["shuffle"])
            self.assertFalse(result["split"]["stratified"])
            config = json.loads((self.state / "jobs" / job["id"] / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["validation"], validation)
            results.append(result)
        self.assertEqual(results[0]["comparison_key"], results[1]["comparison_key"])
        self.manager.close()
        restored = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(restored.close)
        for result in results:
            saved = restored.detail("example", result["id"])["job"]
            self.assertEqual(saved["validation"], validation)
            self.assertEqual(saved["split"], result["split"])
        restored.close()
        path = self.state / "jobs" / results[0]["id"] / "job.json"
        wrong_column = json.loads(path.read_text(encoding="utf-8"))
        wrong_column["split"]["time_column"] = "another_column"
        path.write_text(json.dumps(wrong_column), encoding="utf-8")
        restored = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(restored.close)
        self.assertEqual([row["id"] for row in restored.list()], [results[1]["id"]])

    def test_invalid_validation_does_not_create_jobs_or_touch_source(self):
        original = (self.task / "submission.csv").read_bytes()
        invalid = [True, [], "time", {}, {"strategy": []}, {"strategy": "unknown"},
                   {"strategy": "random", "command": "anything"}, {"strategy": "time"},
                   {"strategy": "time", "time_column": "x", "path": "../../.env"},
                   {"strategy": "time", "time_column": "../../.env"},
                   {"strategy": "time", "time_column": "answer"}, {"strategy": "grouped"}]
        for validation in invalid:
            with self.subTest(validation=validation), self.assertRaises(ValueError):
                self.manager.start("example", "baseline", validation=validation)
        for action, task_id in (("agent", "example"), ("prepare", None)):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "валидации"):
                self.manager.start(task_id, action, validation={"strategy": "random"})
        self.assertEqual(self.manager.list(), [])
        self.assertEqual(list(self.manager.jobs_root.iterdir()), [])
        self.assertEqual((self.task / "submission.csv").read_bytes(), original)

    def test_custom_model_parameters_reach_worker_and_survive_restore(self):
        requests = [("baseline", {"max_iter": 50, "max_leaf_nodes": 8}),
                    ("extra_trees", {"n_estimators": 50, "max_depth": 4})]
        results = []
        for action, parameters in requests:
            expected = deepcopy(parameters)
            job = self.manager.start("example", action, model_params=parameters)
            parameters["max_iter" if action == "baseline" else "n_estimators"] = 51
            result = self.wait(job)
            self.assertEqual(result["status"], "completed", self.manager.detail("example", job["id"]))
            self.assertEqual(result["model_config"]["model"], result["model"])
            for name, value in expected.items():
                self.assertEqual(result["model_config"]["parameters"][name], value)
            folder = self.state / "jobs" / job["id"]
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            evidence = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model_params"], result["model_config"]["parameters"])
            self.assertEqual(evidence["model_config"], result["model_config"])
            results.append(result)
        self.manager.close()
        restored = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(restored.close)
        for result in results:
            self.assertEqual(restored.detail("example", result["id"])["job"]["model_config"],
                             result["model_config"])

    def test_invalid_model_parameters_create_no_jobs(self):
        original = (self.task / "submission.csv").read_bytes()
        invalid = [True, [], "anything", {"command": "anything"}, {"n_jobs": 1},
                   {"random_state": 0}, {"n_estimators": 50}, {"max_iter": True},
                   {"max_iter": 49}, {"max_iter": 601}, {"max_iter": {"path": "../../.env"}},
                   {"learning_rate": float("nan")}, {"learning_rate": float("inf")}]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.manager.start("example", "baseline", model_params=parameters)
        for action, task_id in (("agent", "example"), ("prepare", None)):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "модели"):
                self.manager.start(task_id, action, model_params={"max_iter": 50})
        self.assertEqual(self.manager.list(), [])
        self.assertEqual(list(self.manager.jobs_root.iterdir()), [])
        self.assertEqual((self.task / "submission.csv").read_bytes(), original)

    def test_collected_model_config_must_match_requested_parameters(self):
        job = self.manager.start("example", "baseline", model_params={"max_iter": 50})
        result = self.wait(job)
        self.assertEqual(result["status"], "completed", self.manager.detail("example", job["id"]))
        path = self.state / "jobs" / job["id"] / "result.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        changed = deepcopy(original)
        changed["model_config"]["parameters"]["max_iter"] = 51
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "настройками"):
            self.manager._collect(self.manager._jobs[job["id"]])
        changed = deepcopy(original)
        changed["model_config"]["parameters"]["command"] = "must not leak"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.manager._collect(self.manager._jobs[job["id"]])
        changed = deepcopy(original)
        changed.pop("model_config")
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.manager._collect(self.manager._jobs[job["id"]])
        self.manager._jobs[job["id"]].pop("model_config")
        self.manager._collect(self.manager._jobs[job["id"]])

    def test_restore_accepts_legacy_split_and_rejects_invalid_validation_metadata(self):
        job = self.manager.start("example", "baseline")
        self.assertEqual(self.wait(job)["status"], "completed")
        self.manager.close()
        path = self.state / "jobs" / job["id"] / "job.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        legacy = deepcopy(original)
        legacy.pop("validation", None)
        legacy.pop("model_config", None)
        legacy["split"].pop("strategy", None)
        legacy["split"]["unknown_private_value"] = "hidden"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        restored = JobManager(self.data, self.state, self.catalog)
        try:
            saved = restored.detail("example", job["id"])["job"]
            self.assertNotIn("validation", saved)
            self.assertNotIn("model_config", saved)
            self.assertNotIn("strategy", saved["split"])
            self.assertNotIn("unknown_private_value", saved["split"])
        finally:
            restored.close()
        mutations = [
            lambda row: row.update(validation={"strategy": "random", "path": "../../.env"}),
            lambda row: row.update(validation={"strategy": "time", "time_column": []}),
            lambda row: row.update(validation={"strategy": "grouped"}),
            lambda row: row["split"].update(strategy="unknown"),
            lambda row: row["split"].update(shuffle=1),
            lambda row: row["split"].update(group_count=True),
            lambda row: row["split"].update(group_count=1 << 31),
            lambda row: row["split"].update(train_period_min="x" * 101),
            lambda row: row["split"].update(validation_fraction=True),
            lambda row: row["model_config"].update(path="../../.env"),
            lambda row: row["model_config"]["parameters"].update(command="anything"),
            lambda row: row["model_config"]["parameters"].update(max_iter=True),
            lambda row: row.update(model="ExtraTreesRegressor"),
        ]
        for mutate in mutations:
            row = deepcopy(original)
            mutate(row)
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.subTest(record=row):
                restored = JobManager(self.data, self.state, self.catalog)
                try:
                    self.assertEqual(restored.list(), [])
                finally:
                    restored.close()

    @unittest.skipUnless(os.name == "nt", "Windows short path aliases are checked here")
    def test_windows_short_data_path_supports_training_and_preserves_workspace_boundary(self):
        import ctypes
        from ctypes import wintypes

        self.manager.close()
        long_data = self.root / "data directory with a long name"
        self.data.rename(long_data)
        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD
        required = get_short_path(str(long_data), None, 0)
        if not required:
            self.skipTest("Short path names are unavailable on this Windows filesystem")
        buffer = ctypes.create_unicode_buffer(required)
        length = get_short_path(str(long_data), buffer, len(buffer))
        if not length or length >= len(buffer):
            self.skipTest("Could not obtain a short path name on this Windows filesystem")
        short_data = Path(buffer.value)
        if short_data == long_data.resolve():
            self.skipTest("This Windows filesystem does not create short path aliases")
        self.assertEqual(short_data.resolve(), long_data.resolve())
        catalog = DataCatalog(short_data)
        self.manager = JobManager(short_data, self.state, catalog)
        self.addCleanup(self.manager.close)

        task = long_data / "example"
        original_csv = (task / "submission.csv").read_bytes()
        job = self.manager.start("example", "baseline")
        result = self.wait(job)
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertTrue(result["output_valid"])
        self.assertFalse(result["api_used"])
        self.assertEqual((task / "submission.csv").read_bytes(), original_csv)
        self.assertTrue(self.manager.publish("example", job["id"])["published"])
        self.assertEqual((task / "submission.csv").read_bytes(),
                         self.manager.artifact("example", job["id"]).read_bytes())

        nested = JobManager(short_data, task / ".workspace", catalog)
        self.addCleanup(nested.close)
        with self.assertRaisesRegex(ValueError, "отдельно"):
            nested.start("example", "baseline")

    def test_one_job_at_a_time_cancel_and_close_are_bounded(self):
        if os.name != "nt" and not Path("/proc").is_dir():
            self.skipTest("Arbitrary subprocess fixtures require OS process inventory")
        command = [sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(60)"]
        with patch.object(self.manager, "_command", return_value=command):
            job = self.manager.start("example", "baseline")
            with self.assertRaises(RuntimeError):
                self.manager.start("example", "extra_trees")
            self.assertEqual(self.manager.active_job()["id"], job["id"])
            began = time.monotonic()
            stopped = self.manager.stop("example", job["id"])
            self.assertEqual(stopped["status"], "cancelled")
            self.assertLess(time.monotonic() - began, 5)
            self.assertIsNone(self.manager.active_job())
            second = self.manager.start("example", "baseline")
            began = time.monotonic()
            self.manager.close()
            self.assertLess(time.monotonic() - began, 5)
            self.assertEqual(self.manager.detail("example", second["id"])["job"]["status"], "cancelled")

    def test_fixed_recipe_busy_cancel_and_timeout_work_without_process_inventory(self):
        job = self.manager.start("example", "baseline")
        with self.assertRaises(RuntimeError):
            self.manager.start("example", "extra_trees")
        deadline = time.monotonic() + 10
        while self.manager.detail("example", job["id"])["job"]["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.manager.stop("example", job["id"])["status"], "cancelled")
        self.assertIsNone(self.manager.active_job())
        with patch.object(jobs, "LOCAL_TIMEOUT", 0):
            result = self.wait(self.manager.start("example", "baseline"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("лимит времени", result["error"])

    @unittest.skipIf(os.name == "nt", "POSIX sessions and /proc are checked here")
    def test_cancel_kills_descendants_even_in_new_sessions(self):
        if not Path("/proc").is_dir():
            self.skipTest("Detached descendant fixture requires OS process inventory")
        child_pid = self.root / "child.pid"
        grandchild = "import time; time.sleep(60)"
        code = ("import subprocess,sys,time,pathlib; "
                f"child=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True); "
                f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); time.sleep(60)")
        with patch.object(self.manager, "_command", return_value=[sys.executable, "-c", code]):
            job = self.manager.start("example", "baseline")
            deadline = time.monotonic() + 10
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(child_pid.exists())
            pid = int(child_pid.read_text())
            self.manager.stop("example", job["id"])
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                text = stat.read_text()
                self.assertEqual(text[text.rfind(")") + 2:].split()[0], "Z")

    def test_timeout_and_oversized_logs_fail_without_touching_source(self):
        if os.name != "nt" and not Path("/proc").is_dir():
            self.skipTest("Arbitrary subprocess fixtures require OS process inventory")
        before = (self.task / "submission.csv").read_bytes()
        command = [sys.executable, "-u", "-c", "import time; print('waiting',flush=True); time.sleep(60)"]
        with patch.object(self.manager, "_command", return_value=command), patch.object(jobs, "LOCAL_TIMEOUT", 0.1):
            result = self.wait(self.manager.start("example", "baseline"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("лимит времени", result["error"])
        loud = [sys.executable, "-u", "-c", "import time; print('x'*10000,flush=True); time.sleep(60)"]
        with patch.object(self.manager, "_command", return_value=loud), patch.object(jobs, "MAX_LOG_BYTES", 100):
            result = self.wait(self.manager.start("example", "baseline"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("журнал", result["error"])
        self.assertEqual((self.task / "submission.csv").read_bytes(), before)

    def test_missing_api_settings_and_invalid_actions_do_not_launch(self):
        with patch.object(jobs, "_settings_values", return_value={name: "" for name in jobs.API_FIELDS}):
            with self.assertRaises(ValueError):
                self.manager.start("example", "agent")
        with self.assertRaises(ValueError):
            self.manager.start("example", "arbitrary_shell")
        with self.assertRaises(ValueError):
            self.manager.detail("example", "../private")
        self.assertEqual(self.manager.list(), [])

    def test_second_manager_is_refused_and_old_active_job_is_interrupted(self):
        with self.assertRaises(ValueError):
            JobManager(self.data, self.state, self.catalog)
        job = self.manager.start("example", "baseline")
        self.wait(job)
        self.manager.close()
        path = self.state / "jobs" / job["id"] / "job.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["status"] = "running"
        path.write_text(json.dumps(report), encoding="utf-8")
        restored = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(restored.close)
        self.assertEqual(restored.detail("example", job["id"])["job"]["status"], "interrupted")

    def test_workspace_inside_task_is_refused(self):
        other = JobManager(self.data, self.task / ".workspace", self.catalog)
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            other.start("example", "baseline")

    def test_recipe_refuses_a_different_declared_metric(self):
        (self.task / "task.md").write_text("Основная метрика: MAE", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "RMSE"):
            self.manager.start("example", "baseline")
        self.assertEqual(self.manager.list(), [])

    def test_agent_hypothesis_is_appended_only_to_the_copied_task(self):
        import threading
        original = (self.task / "task.md").read_bytes()
        job_id = "b" * 32
        (self.state / "jobs" / job_id).mkdir()
        job = {"id": job_id, "action": "agent", "task_id": "example",
               "dataset_fingerprint": self.catalog.fingerprint("example"),
               "hypothesis": "Проверить новый числовой признак"}
        task_path = self.manager._stage(job, threading.Event())
        self.assertIn(job["hypothesis"], task_path.read_text(encoding="utf-8"))
        self.assertEqual((self.task / "task.md").read_bytes(), original)

    def test_failed_agent_retains_usage_and_redacts_nested_logs_without_real_api(self):
        settings = dict(OPENAI_API_KEY="secret-for-test", OPENAI_BASE_URL="http://unused.invalid", OPENAI_MODEL="fake")
        job = self.manager.start("example", "baseline")
        self.assertEqual(self.wait(job)["status"], "completed")
        stored = self.manager._jobs[job["id"]]
        stored.update(action="agent", status="failed", metric=None, split=None, comparison_key=None)
        run = self.state / "jobs" / job["id"] / "task" / ".agent_runs" / ("a" * 32)
        run.mkdir(parents=True)
        report = {"status": "failed", "events": [], "error": "API failed", "model": "fake",
                  "api": {"attempts": 1, "responses": 0, "usage_reported_responses": 0,
                          "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        (run / "run.json").write_text(json.dumps(report), encoding="utf-8")
        (run / "experiment_101.stdout.txt").write_text("secret-for-test", encoding="utf-8")
        (self.state / "jobs" / job["id"] / "stdout.txt").write_text("secret-for-test", encoding="utf-8")
        with patch.object(jobs, "_settings_values", return_value=settings):
            self.manager._collect_agent(stored)
            result = self.manager.detail("example", job["id"])["job"]
            self.assertTrue(result["api_used"])
            self.assertEqual(result["api"]["state"], "unknown")
            self.assertIsNone(result["metric"])
            detail = self.manager.detail("example", job["id"])
            self.assertNotIn("secret-for-test", detail["stdout"])
            raw = self.manager.agent_reader("example", job["id"]).file("a" * 32, "experiment_101.stdout.txt")
            clean = self.manager.redact("example", job["id"], raw)
            self.assertEqual(clean["text"], "[REDACTED]")

    def test_corrupt_persisted_jobs_are_skipped_and_unknown_fields_are_hidden(self):
        job = self.manager.start("example", "baseline")
        result = self.wait(job)
        self.assertEqual(
            result["status"], "completed",
            json.dumps(self.manager.detail("example", job["id"]), ensure_ascii=False, indent=2),
        )
        self.manager.close()
        path = self.state / "jobs" / job["id"] / "job.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for label, change in (("bad date", {"started_at": []}), ("NaN", {"elapsed_seconds": float("nan")}),
                              ("missing task", {"task_id": None})):
            with self.subTest(case=label):
                path.write_text(json.dumps({**original, **change}), encoding="utf-8")
                restored = JobManager(self.data, self.state, self.catalog)
                try:
                    self.assertEqual(restored.list(), [])
                finally:
                    restored.close()
        path.write_text(json.dumps({**original, "unknown_private_value": "hidden"}), encoding="utf-8")
        restored = JobManager(self.data, self.state, self.catalog)
        self.addCleanup(restored.close)
        self.assertNotIn("unknown_private_value", restored.list()[0])


if __name__ == "__main__":
    unittest.main()
