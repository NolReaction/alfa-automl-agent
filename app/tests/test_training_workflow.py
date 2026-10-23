"""Real HTTP training/export/restart coverage on tiny, known local inputs.

The direct-model workflow works without qualifying process-tree observations.
The optional serial study never mocks source/runtime/resource evidence and is
skipped before fitting when this host cannot honestly qualify a campaign.
"""

from contextlib import closing, nullcontext
import csv
from datetime import datetime
import hashlib
import http.client
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from agent.features import normalize_feature_config
from app.server import WorkspaceServer
from app.supervisor import process_inventory_matches
from research_lab.integrity import source_identity


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.task = self.data_root / "tiny_attacks"
        self.task.mkdir(parents=True)
        (self.task / "task.md").write_text(
            "Classify the connection.\nPrimary metric: Macro F1\n", encoding="utf-8")
        self._csv("data_dictionary.csv", [
            ["feature", "role", "dtype"], ["row_key", "id", "category"],
            ["x", "feature", "numeric"], ["protocol", "feature", "category"],
            ["label", "target", "category"],
        ])
        categories = ("tcp", "", "__MISSING__", "__ALFA_MISSING__")
        self._csv("train.csv", [["row_key", "x", "protocol", "label"]] + [
            [f"train-{index:03d}", index, categories[index % 4],
             "NA" if (index // 4) % 2 == 0 else "attack"]
            for index in range(96)
        ])
        self._csv("test.csv", [
            ["row_key", "x", "protocol"], ["NA", 8, "__MISSING__"],
            ["0001", 21, ""], ["0002", 45, "unseen"],
        ])
        self._csv("sample_submission.csv", [
            ["row_key", "label"], ["NA", "NA"], ["0001", "NA"], ["0002", "NA"],
        ])
        # Experiments must remain isolated until a separate explicit Publish.
        (self.task / "submission.csv").write_text(
            "row_key,label\nNA,NA\n0001,NA\n0002,NA\n", encoding="utf-8")
        self.original = self._input_snapshot()
        self.state_root = self.root / "state"
        self.server = None
        self.thread = None
        self._open_server()
        self.addCleanup(self._close_server)

    def _csv(self, name, rows):
        with (self.task / name).open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def _input_snapshot(self):
        return {path.name: path.read_bytes() for path in self.task.iterdir() if path.is_file()}

    def _open_server(self):
        self.server = WorkspaceServer(self.data_root, port=0, state_root=self.state_root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _close_server(self):
        if self.server is None:
            return
        server, thread = self.server, self.thread
        self.server = self.thread = None
        try:
            server.shutdown()
        finally:
            server.server_close()
            if thread is not None:
                thread.join(timeout=3)

    def _request(self, path, *, method="GET", value=None):
        headers = {}
        body = None
        if value is not None:
            headers = {"Content-Type": "application/json", "X-Alfa-Token": self.server.token}
            body = json.dumps(value).encode("utf-8")
        with closing(http.client.HTTPConnection(
                "127.0.0.1", self.server.server_port, timeout=15)) as client:
            client.request(method, path, body=body, headers=headers)
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def _post(self, path, value):
        return self._request(path, method="POST", value=value)

    def _job(self, identifier):
        status, _, raw = self._request(f"/api/jobs/{identifier}?task_id=tiny_attacks")
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    def _wait_job(self, identifier):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            detail = self._job(identifier)
            if detail["job"]["status"] not in {"queued", "running"}:
                self.assertEqual(detail["job"]["status"], "completed", detail)
                return detail["job"]
            time.sleep(.05)
        self.fail("Tiny HTTP model did not finish within 90 seconds: " + str(self._job(identifier)))

    def _download(self, identifier, kind):
        status, headers, raw = self._request(
            f"/api/jobs/{identifier}/{kind}?task_id=tiny_attacks")
        self.assertEqual(status, 200, raw[:1000])
        self.assertEqual(int(headers["Content-Length"]), len(raw))
        self.assertIn("attachment", headers["Content-Disposition"])
        return headers, raw

    def _preview_identity_context(self):
        # These checks inspect only preview/replay boundaries. A dirty working
        # tree may use a fixed source seal here, never to qualify real fits or
        # start a campaign. The separate real-campaign test mocks no evidence.
        identity = source_identity(self.server.campaigns.source_root)
        if not identity.get("dirty"):
            return nullcontext()
        return patch.object(self.server.campaigns, "_identity", return_value={
            "sha": identity["sha"], "source_fingerprint": identity["source_fingerprint"],
            "valid": True, "checked_at": "2026-10-23T00:00:00+00:00",
        })

    def test_three_http_models_preserve_raw_predictions_and_replay_after_restart(self):
        feature_config = normalize_feature_config({
            "missing_indicators": True, "rare_min_count": 2, "signed_log": True,
        })
        trained, downloads = [], {}
        for count in (500, 750, 1000):
            status, _, raw = self._post("/api/jobs", {
                "task_id": "tiny_attacks", "action": "lightgbm", "minutes": 1,
                "validation": {"strategy": "random"}, "feature_config": feature_config,
                "model_params": {"n_estimators": count, "num_leaves": 7, "min_child_samples": 5},
                "hypothesis": f"Only increase the tree count to {count}",
            })
            self.assertEqual(status, 202, raw)
            identifier = json.loads(raw)["job"]["id"]
            job = self._wait_job(identifier)
            self.assertTrue(job["output_valid"])
            self.assertFalse(job["api_used"])
            self.assertEqual(job["training_device"], "cpu")
            self.assertEqual(job["worker_timeout_seconds"], 60)
            self.assertEqual(job["model_config"]["parameters"]["n_estimators"], count)
            self.assertEqual(job["feature_config"], feature_config)
            summary = job["model_artifact"]
            self.assertTrue(summary["available"], summary)
            self.assertEqual(summary["trained_on"], "full_train")
            self.assertTrue(summary["validation_model_is_distinct"])

            model_headers, model_bytes = self._download(identifier, "model")
            manifest_headers, manifest_bytes = self._download(identifier, "model-manifest")
            recipe_headers, recipe_bytes = self._download(identifier, "recipe")
            self.assertEqual(model_headers["Content-Type"], "application/octet-stream")
            self.assertTrue(manifest_headers["Content-Type"].startswith("application/json"))
            self.assertTrue(recipe_headers["Content-Type"].startswith("application/json"))
            for headers, name in ((model_headers, "model.joblib"),
                                  (manifest_headers, "model_manifest.json"),
                                  (recipe_headers, "training_recipe.json")):
                self.assertIn(name, headers["Content-Disposition"])
            self.assertEqual(hashlib.sha256(model_bytes).hexdigest(), summary["sha256"])
            self.assertEqual(len(model_bytes), summary["size_bytes"])
            manifest, recipe = json.loads(manifest_bytes), json.loads(recipe_bytes)
            self.assertEqual(manifest["artifact"], summary)
            self.assertEqual(manifest["recipe"], recipe)
            self.assertEqual(hashlib.sha256(recipe_bytes).hexdigest(), manifest["recipe_sha256"])
            self.assertEqual(recipe["model_config"], job["model_config"])
            self.assertEqual(recipe["feature_config"], feature_config)
            self.assertEqual(manifest["dataset_fingerprint"], job["dataset_fingerprint"])
            self.assertEqual(manifest["comparison_key"], job["comparison_key"])
            self.assertEqual(manifest["fit"]["scope"], "full_train")
            self.assertEqual(manifest["fit"]["rows"], 96)
            self.assertEqual(manifest["fit"]["validation_metric_source"], "separate_holdout_fit")
            self.assertEqual(manifest["fit"]["validation_split"], job["split"])

            # The test deliberately loads only an artifact produced by its own
            # just-finished trusted worker, never a report/user-uploaded pickle.
            pipeline = joblib.load(BytesIO(model_bytes))
            raw_test = pd.read_csv(self.task / "test.csv", dtype=str, keep_default_na=False)
            predicted = pipeline.predict(raw_test.loc[:, manifest["schema"]["features"]])
            _, submission_bytes = self._download(identifier, "submission")
            submission = pd.read_csv(StringIO(submission_bytes.decode("utf-8-sig")),
                                     dtype=str, keep_default_na=False)
            np.testing.assert_array_equal(np.asarray(predicted).astype(str), submission["label"].to_numpy())
            self.assertEqual(submission["row_key"].tolist(), ["NA", "0001", "0002"])
            downloads[identifier] = (model_bytes, manifest_bytes, recipe_bytes)
            trained.append(job)
            self.assertEqual(self._input_snapshot(), self.original)
            self.assertIsNone(self.server.jobs.active_job())

        self.assertEqual(len({row["id"] for row in trained}), 3)
        self.assertEqual(len({row["comparison_key"] for row in trained}), 1)
        self.assertEqual(len({row["dataset_fingerprint"] for row in trained}), 1)
        self.assertTrue(all(row["split"] == trained[0]["split"] for row in trained))
        base_parameters = {key: value for key, value in trained[0]["model_config"]["parameters"].items()
                           if key != "n_estimators"}
        for previous, following in zip(trained, trained[1:]):
            self.assertGreaterEqual(datetime.fromisoformat(following["started_at"]),
                                    datetime.fromisoformat(previous["finished_at"]))
        for row in trained[1:]:
            self.assertEqual({key: value for key, value in row["model_config"]["parameters"].items()
                              if key != "n_estimators"}, base_parameters)

        source_id = trained[0]["id"]
        body = {"preset": "lightgbm_trees", "task_ids": ["tiny_attacks"],
                "source_job_id": source_id, "minutes_per_step": 1, "max_hours": 1}
        with self._preview_identity_context():
            status, _, raw = self._post("/api/campaigns/preview", body)
            self.assertEqual(status, 200, raw)
            preview = json.loads(raw)
            self.assertTrue(preview["ready"], preview)
            self.assertEqual(preview["source_job_id"], source_id)
            self.assertEqual([step["model_config"]["parameters"]["n_estimators"]
                              for step in preview["steps"]], [500, 750, 1000])
            self.assertEqual([step["is_control"] for step in preview["steps"]], [True, False, False])
            for step in preview["steps"]:
                self.assertEqual(step["comparison_key"], trained[0]["comparison_key"])
                self.assertEqual(step["split"], trained[0]["split"])
                self.assertEqual(step["feature_config"], feature_config)
                self.assertEqual(step["worker_timeout_seconds"], 60)
            original_description = (self.task / "task.md").read_bytes()
            try:
                (self.task / "task.md").write_bytes(original_description + b"\nChanged source task.\n")
                status, _, raw = self._post("/api/campaigns/preview", body)
                self.assertEqual(status, 200, raw)
                changed = json.loads(raw)
                self.assertFalse(changed["ready"], changed)
                self.assertTrue(changed["errors"], changed)
            finally:
                (self.task / "task.md").write_bytes(original_description)

        self._close_server()
        self._open_server()
        self.assertIsNone(self.server.jobs.active_job())
        for job in trained:
            restored = self._job(job["id"])["job"]
            self.assertEqual(restored["status"], "completed", restored)
            self.assertEqual(restored["feature_config"], feature_config)
            self.assertEqual(restored["model_artifact"], job["model_artifact"])
            for kind, expected in zip(("model", "model-manifest", "recipe"), downloads[job["id"]]):
                self.assertEqual(self._download(job["id"], kind)[1], expected)
        self.assertEqual(self._input_snapshot(), self.original)

    def test_gpu_capability_is_explicit_and_wrong_action_fails_before_launch(self):
        from app import training_devices

        with patch.object(training_devices, "_capability_cache", None), \
                patch.object(training_devices, "probe_gpu_capability", return_value=
                             training_devices._empty_gpu("No GPU in this deterministic test")) as probe:
            status, _, raw = self._request("/api/training-devices")
            self.assertEqual(status, 200, raw)
            capability = json.loads(raw)
            self.assertFalse(capability["gpu"]["available"])
            self.assertFalse(capability["gpu"]["training_verified"])
            probe.assert_called_once_with()
        status, _, raw = self._post("/api/jobs", {
            "task_id": "tiny_attacks", "action": "lightgbm", "training_device": "gpu",
        })
        self.assertEqual(status, 400, raw)
        self.assertIn("CatBoost", json.loads(raw)["error"])
        self.assertEqual(self.server.jobs.list(), [])
        self.assertIsNone(self.server.jobs.active_job())

    def test_real_http_study_qualifies_three_serial_jobs_when_inventory_is_available(self):
        if sys.platform == "linux" and not process_inventory_matches():
            self.skipTest("Linux process inventory does not match this PID namespace")
        if source_identity(self.server.campaigns.source_root).get("dirty"):
            self.skipTest("Real campaign qualification requires a clean committed checkout")
        body = {"preset": "lightgbm_trees", "task_ids": ["tiny_attacks"],
                "minutes_per_step": 1, "max_hours": 1, "continue_on_error": False,
                "validation_by_task": {"tiny_attacks": {"strategy": "random"}}}
        status, _, raw = self._post("/api/campaigns/preview", body)
        self.assertEqual(status, 200, raw)
        preview = json.loads(raw)
        self.assertTrue(preview["ready"], preview)
        status, _, raw = self._post("/api/campaigns", {**body, "preview_token": preview["plan_sha256"]})
        self.assertEqual(status, 202, raw)
        identifier = json.loads(raw)["campaign"]["id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status, _, raw = self._request(f"/api/campaigns/{identifier}")
            self.assertEqual(status, 200, raw)
            campaign = json.loads(raw)["campaign"]
            if campaign["status"] in {"completed", "failed", "stopped", "interrupted", "cleanup_pending"}:
                break
            time.sleep(.05)
        else:
            self.fail("Tiny serial HTTP campaign did not finish within 120 seconds")
        self.assertEqual(campaign["status"], "completed", campaign)
        self.assertEqual([step["status"] for step in campaign["steps"]], ["completed"] * 3)
        self.assertTrue(all(step["admissible"] for step in campaign["steps"]))
        status, _, raw = self._request(f"/api/campaigns/{identifier}/report.json")
        self.assertEqual(status, 200, raw)
        report = json.loads(raw)
        self.assertEqual(report["summary"]["comparison_eligible"], 3)
        self.assertEqual(len(report["groups"]), 1)
        group = report["groups"][0]
        self.assertTrue(group["comparison_available"], group)
        self.assertEqual(group["control"]["step_id"], campaign["steps"][0]["id"])
        self.assertEqual(group["count"], 3)
        self.assertEqual(self._input_snapshot(), self.original)


if __name__ == "__main__":
    unittest.main()
