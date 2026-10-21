"""The exact platform CLI recovers using only root files from the release ZIP."""

from __future__ import annotations

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import zipfile

from agent.execution import child_environment
from agent.llm import DEFAULT_MODEL
from agent.resources import POLL_SECONDS, ProcessTreeMonitor
from build_submission import build_archive
from tests.test_research import make_regression
from validation import validate_predictions


PLACEHOLDER_KEY = "local-release-recovery-placeholder"


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class ReleaseRecoveryArchiveTests(unittest.TestCase):
    def prepare(self, temporary):
        root = Path(temporary).resolve()
        archive, task_dir = root / "solution.zip", root / "isolated task"
        task_dir.mkdir()
        names = build_archive(archive)
        self.assertIn("_alfa_agent__emergency.py", names)
        self.assertFalse(any("/" in name for name in names))
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if name.endswith(".py") or name == "requirements.txt":
                    (task_dir / name).write_bytes(zipped.read(name))
        task = make_regression(task_dir)
        originals = {path: path.read_bytes() for path in task_dir.rglob("*") if path.is_file()}
        return task, originals

    def environment(self, base_url):
        # Positive allowlist removes proxies, developer API settings and import
        # paths. The extracted archive supplies every project module itself.
        environment = child_environment()
        self.assertFalse(any(name.lower().endswith("_proxy") for name in environment))
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        environment.update(OPENAI_API_KEY=PLACEHOLDER_KEY, OPENAI_BASE_URL=base_url)
        return environment

    def execute(self, task, environment, observe=None):
        stdout_path, stderr_path = task.parent / "release.stdout.txt", task.parent / "release.stderr.txt"
        started = time.monotonic()
        exceeded = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "solution.py", "task.md"],
                cwd=task.parent, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            monitor = ProcessTreeMonitor(process, max_memory_bytes=4 * 1024 ** 3)
            try:
                while True:
                    memory_exceeded = monitor.sample()
                    if observe is not None:
                        observe()
                    if process.poll() is not None:
                        break
                    remaining = 180 - (time.monotonic() - started)
                    if remaining <= 0 or memory_exceeded:
                        exceeded = True
                        break
                    try:
                        process.wait(timeout=min(POLL_SECONDS, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                stopped = monitor.terminate()
        result = SimpleNamespace(
            returncode=process.returncode,
            stdout=stdout_path.read_text(encoding="utf-8", errors="replace")[-20000:],
            stderr=stderr_path.read_text(encoding="utf-8", errors="replace")[-20000:],
        )
        detail = result.stdout + result.stderr
        self.assertTrue(stopped, detail)
        self.assertFalse(exceeded, detail)
        self.assertEqual(result.returncode, 0, detail)
        self.assertNotIn(PLACEHOLDER_KEY, detail)
        return result

    def assert_originals_unchanged(self, task, originals):
        self.assertFalse((task.parent / "agent").exists())
        self.assertFalse((task.parent / "app").exists())
        for path, content in originals.items():
            with self.subTest(original=path.name):
                self.assertEqual(path.read_bytes(), content)

    def assert_emergency_checkpoint(self, task):
        paths = list((task.parent / ".agent_runs").glob("*/emergency.json"))
        self.assertEqual(len(paths), 1)
        path = paths[0]
        receipt = json.loads(path.read_text(encoding="utf-8"))
        prior = path.parent / "prior.csv"
        self.assertEqual(receipt["version"], 1)
        self.assertEqual(receipt["kind"], "target_prior")
        self.assertEqual(receipt["status"], "ready")
        self.assertFalse(receipt["quality_verified"])
        self.assertEqual(receipt["model"]["strategy"], "mean")
        self.assertEqual(receipt["model"]["fit_rows"], 100)
        self.assertEqual(receipt["predict_rows"], 2)
        self.assertEqual(receipt["submission_sha256"], _sha256(prior.read_bytes()))
        self.assertEqual([row["row_key"] for row in _csv_rows(prior)], ["0005", "NA"])
        self.assertEqual([float(row["outcome"]) for row in _csv_rows(prior)], [150.5, 150.5])
        self.assertFalse((path.parent / "run.json").exists())
        return prior

    def assert_final_predictions_improve_prior(self, task, prior):
        self.assertTrue(validate_predictions(task.parent)["valid"])
        final = task.parent / "submission.csv"
        rows, prior_rows = _csv_rows(final), _csv_rows(prior)
        self.assertEqual([row["row_key"] for row in rows], ["0005", "NA"])
        self.assertNotEqual(final.read_bytes(), prior.read_bytes())
        actual = [17, 20]
        final_error = sum((float(row["outcome"]) - value) ** 2 for row, value in zip(rows, actual))
        prior_error = sum((float(row["outcome"]) - value) ** 2 for row, value in zip(prior_rows, actual))
        self.assertLess(final_error, prior_error)

    def test_default_cli_retries_local_503_once_then_finishes_verified_full_refit(self):
        requests, early_publication = [], []
        first_request = threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            task, originals = self.prepare(temporary)

            def observe():
                if first_request.is_set() or early_publication:
                    return
                paths = list((task.parent / ".agent_runs").glob("*/emergency.json"))
                if not paths:
                    return
                try:
                    receipt = json.loads(paths[0].read_text(encoding="utf-8"))
                    content = (task.parent / "submission.csv").read_bytes()
                    if _sha256(content) == receipt["submission_sha256"] and not first_request.is_set():
                        early_publication.append(content)
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    # The receipt is committed before the public CSV; a later
                    # poll sees the completed atomic publication.
                    return

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    first_request.set()
                    observation = {}
                    requests.append(observation)
                    try:
                        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        reports = list((task.parent / ".agent_runs").glob("*/run.json"))
                        checkpoint_paths = list((task.parent / ".agent_runs").glob("*/emergency.json"))
                        observation.update(model=payload["model"], research_reports=len(reports),
                                           emergency_reports=len(checkpoint_paths))
                        report = json.loads(reports[0].read_text(encoding="utf-8"))
                        receipt = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
                        prior = checkpoint_paths[0].parent / "prior.csv"
                        observation.update(stage=report["stage"], final_check=report.get("evaluations", {}).get("final_check"),
                                           attempts=report["api"]["attempts"], retries=report["api_recovery"]["retries"],
                                           prior_ready=receipt["status"] == "ready",
                                           prior_hash_valid=_sha256(prior.read_bytes()) == receipt["submission_sha256"],
                                           public_csv_valid=validate_predictions(task.parent)["valid"])
                    except Exception as error:
                        observation["error"] = f"{type(error).__name__}: {error}"
                    body = json.dumps({"error": {"type": "server_error", "message": "Local simulated overload"}}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
            thread.start()
            try:
                self.execute(task, self.environment(f"http://127.0.0.1:{server.server_port}/v1"), observe)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(len(requests), 2, requests)
            self.assertEqual(len(early_publication), 1, "Target-prior publication was not observed before HTTP")
            for index, request in enumerate(requests, 1):
                with self.subTest(request=index):
                    self.assertNotIn("error", request)
                    self.assertEqual(request["model"], DEFAULT_MODEL)
                    self.assertEqual(request["research_reports"], 1)
                    self.assertEqual(request["emergency_reports"], 1)
                    self.assertEqual(request["stage"], "inner_research")
                    self.assertIsNone(request["final_check"])
                    self.assertEqual(request["attempts"], index)
                    self.assertEqual(request["retries"], index - 1)
                    self.assertTrue(request["prior_ready"] and request["prior_hash_valid"] and request["public_csv_valid"])
            prior = self.assert_emergency_checkpoint(task)
            self.assertEqual(early_publication[0], prior.read_bytes())
            self.assert_final_predictions_improve_prior(task, prior)
            self.assert_originals_unchanged(task, originals)
            reports = list((task.parent / ".agent_runs").glob("*/run.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["research_protocol_version"], 3)
            self.assertEqual(report["status"], "fallback")
            self.assertEqual(report["reason"], "api_error")
            self.assertEqual(report["api"]["attempts"], 2)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertEqual(report["api_recovery"], {"maximum_retries": 1, "retries": 1})
            self.assertTrue(all(item["status"] == "completed" for item in report["stages"]))
            self.assertEqual(report["baseline_search"]["status"], "searched")
            self.assertGreaterEqual(report["local_research"]["verified"], 1)
            self.assertTrue(report["publication"]["new_result_published"])
            receipts = report["candidate_receipts"]
            outer_path, refit_path = Path(receipts["outer"]), Path(receipts["refit"])
            self.assertTrue(outer_path.resolve().is_relative_to(reports[0].parent / "outer_evaluations"))
            self.assertTrue(refit_path.resolve().is_relative_to(reports[0].parent / "full_refit"))
            outer = json.loads(outer_path.read_text(encoding="utf-8"))
            refit = json.loads(refit_path.read_text(encoding="utf-8"))
            self.assertEqual(outer["phase"], "outer_validation")
            self.assertEqual(refit["phase"], "full_refit")
            self.assertEqual(refit["output_hash"], _sha256((task.parent / "submission.csv").read_bytes()))
            source = task.parent / "regression" / "participants"
            self.assertEqual(_csv_rows(refit_path.parent / "fit.csv"), _csv_rows(source / "train.csv"))
            self.assertEqual(_csv_rows(refit_path.parent / "predict.csv"), _csv_rows(source / "test.csv"))
            self.assertEqual(refit["code_hash"], _sha256((refit_path.parent / "candidate.py").read_bytes()))
            for name in ("code_hash", "recipe_hash"):
                self.assertEqual(outer[name], refit[name])
                self.assertEqual(refit[name], receipts[name])

    def test_default_cli_uses_offline_training_when_sdk_constructor_rejects_base_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, originals = self.prepare(temporary)
            # HTTPX rejects a nonnumeric port during OpenAI construction. No
            # socket or real endpoint is involved in this failure condition.
            result = self.execute(task, self.environment("http://127.0.0.1:not-a-port/v1"))
            self.assertIn("API client unavailable; starting offline fallback", result.stderr)
            self.assertIn("API calls: 0", result.stdout)
            self.assertEqual(list((task.parent / ".agent_runs").glob("*/run.json")), [])
            prior = self.assert_emergency_checkpoint(task)
            self.assert_final_predictions_improve_prior(task, prior)
            self.assert_originals_unchanged(task, originals)
            paths = list((task.parent / ".agent_runs").glob("*/automl_report.json"))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertTrue(report["output_valid"])
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertEqual(report["search"]["phase"], "finished")
            pointer = json.loads((paths[0].parent / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["submission_sha256"], _sha256((task.parent / "submission.csv").read_bytes()))


if __name__ == "__main__":
    unittest.main()
