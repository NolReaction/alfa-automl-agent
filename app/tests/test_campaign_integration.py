"""Tiny real campaign, process ownership, resource evidence and report integration.

Windows CI runs both fixed recipes. A Linux host with a mismatched /proc mount
cannot honestly qualify resource evidence, so it skips before any model starts.
Source identity, dependencies, workers and ProcessTreeMonitor are never mocked.
"""
import csv
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import time
import unittest

from app.campaign_reports import build_campaign_report
from app.server import WorkspaceServer
from app.supervisor import process_inventory_matches


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class CampaignIntegrationTests(unittest.TestCase):
    def test_two_real_fixed_jobs_qualify_and_form_one_comparison(self):
        if sys.platform == "linux" and not process_inventory_matches():
            self.skipTest("Linux process inventory does not match this PID namespace")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            task = data / "tiny_regression"
            task.mkdir(parents=True)
            (task / "task.md").write_text(
                "Predict answer.\nPrimary metric: RMSE\n", encoding="utf-8")
            _csv(task / "data_dictionary.csv", [
                ["feature", "role", "dtype"], ["id", "id", "category"],
                ["x", "feature", "numeric"], ["answer", "target", "numeric"],
            ])
            # Unique numeric features avoid automatic duplicate-group policies.
            _csv(task / "train.csv", [["id", "x", "answer"]] + [
                [f"train-{index:03d}", index, 2 * index + index % 3]
                for index in range(40)
            ])
            _csv(task / "test.csv", [["id", "x"], ["NA", 4], ["001", 17]])
            _csv(task / "sample_submission.csv", [
                ["id", "answer"], ["NA", 0], ["001", 0],
            ])
            original = {path.name: path.read_bytes() for path in task.iterdir()}
            # The real server wires the actual checkout, dependency seals,
            # managers and shared execution gate. No HTTP thread is necessary.
            server = WorkspaceServer(data, port=0, state_root=root / "state")
            identifier = None
            try:
                manager = server.campaigns
                body = {
                    "preset": "custom", "task_ids": ["tiny_regression"],
                    "name": "Two tiny fixed CPU recipes", "max_hours": 2,
                    "continue_on_error": False,
                    "validation_by_task": {"tiny_regression": {"strategy": "random"}},
                    "steps": [
                        {"task_id": "tiny_regression", "action": "baseline", "minutes": 1,
                         "model_params": {"max_iter": 50, "max_leaf_nodes": 4}},
                        {"task_id": "tiny_regression", "action": "extra_trees", "minutes": 1,
                         "model_params": {"n_estimators": 50, "max_depth": 2}},
                    ],
                }
                preview = manager.preview(body)
                self.assertTrue(preview["ready"], preview)
                started = manager.start({**body, "preview_token": preview["plan_sha256"]})
                identifier = started["id"]
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    detail = manager.detail(identifier)
                    if detail["status"] in {"completed", "failed", "interrupted", "stopped", "cleanup_pending"}:
                        break
                    time.sleep(.05)
                else:
                    self.fail("Tiny campaign exceeded 60 seconds: " + str(manager.detail(identifier)))

                self.assertEqual(detail["status"], "completed", detail)
                self.assertIsNone(server.jobs.active_job())
                self.assertEqual(len(server.jobs.list()), 2)
                self.assertEqual([step["status"] for step in detail["steps"]], ["completed", "completed"])
                completed = []
                for step in detail["steps"]:
                    self.assertTrue(step["admissible"], step)
                    capsule = server.jobs.campaign_job(identifier, step["id"])
                    self.assertIsNotNone(capsule)
                    self.assertIs(capsule["cleanup_confirmed"], True, capsule)
                    self.assertIs(capsule["resources_ok"], True, capsule)
                    job = capsule["job"]
                    self.assertEqual(job["id"], step["job_id"])
                    self.assertIs(job["api_used"], False)
                    self.assertIs(job["output_valid"], True)
                    self.assertEqual(job["split"]["strategy"], "random")
                    resources = job["resources"]
                    self.assertIs(resources["available"], True, resources)
                    self.assertIs(resources["cleanup_complete"], True, resources)
                    self.assertGreater(resources["sample_count"], 0)
                    self.assertGreater(resources["peak_rss_bytes"], 0)
                    self.assertEqual(resources["memory_limit_bytes"], 28 * 1024**3)
                    with server.jobs.artifact("tiny_regression", job["id"]).open(
                            encoding="utf-8", newline="") as stream:
                        self.assertEqual([row["id"] for row in csv.DictReader(stream)], ["NA", "001"])
                    completed.append(job)

                self.assertNotEqual(completed[0]["id"], completed[1]["id"])
                self.assertGreaterEqual(datetime.fromisoformat(completed[1]["started_at"]),
                                        datetime.fromisoformat(completed[0]["finished_at"]))
                self.assertEqual(completed[0]["split"], completed[1]["split"])
                self.assertEqual(completed[0]["comparison_key"], completed[1]["comparison_key"])
                report = build_campaign_report(detail)
                self.assertEqual(report["summary"]["completed"], 2, report)
                self.assertEqual(report["summary"]["comparison_eligible"], 2, report)
                self.assertEqual(len(report["groups"]), 1, report)
                self.assertEqual(report["groups"][0]["count"], 2)
                self.assertIs(report["groups"][0]["comparison_available"], True)
                self.assertEqual({path.name: path.read_bytes() for path in task.iterdir()}, original)
            finally:
                try:
                    if identifier is not None:
                        server.campaigns.stop(identifier)
                finally:
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
