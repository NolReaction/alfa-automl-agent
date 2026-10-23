"""Exercise real HTTP boundaries for the serial-workbench APIs without fitting."""
import json
import unittest
from unittest.mock import patch

from app.tests import test_server as http_fixture


class CampaignServerTests(unittest.TestCase):
    setUp = http_fixture.ServerTests.setUp
    tearDown = http_fixture.ServerTests.tearDown
    request = http_fixture.ServerTests.request
    post = http_fixture.ServerTests.post

    def test_overview_and_system_are_local_read_only_and_no_fit(self):
        with patch.object(self.server.jobs, "start") as start:
            for path in ("/api/dashboard", "/api/system", "/api/campaign-presets", "/api/campaigns"):
                status, _, body = self.request(path)
                self.assertEqual(status, 200, body)
                self.assertIsInstance(json.loads(body), dict)
                self.assertEqual(self.request(path, headers={"Origin": "https://foreign.example"})[0], 403)
                self.assertEqual(self.request(path + "?unexpected=1")[0], 400)
            start.assert_not_called()
        overview = json.loads(self.request("/api/dashboard")[2])
        self.assertEqual(overview["counts"]["tasks_ready"], 1)
        self.assertEqual(overview["counts"]["jobs_total"], 0)
        self.assertEqual(overview["server"]["data_dir"], str(self.root))
        self.assertIsNone(overview["active_campaign"])

    def test_campaign_actions_need_token_and_reject_query_or_extra_body(self):
        identifier = "a" * 32
        with patch.object(self.server.campaigns, "start") as start, patch.object(self.server.campaigns, "stop") as stop:
            self.assertEqual(self.post("/api/campaigns", {}, token=False)[0], 403)
            self.assertEqual(self.post("/api/campaigns/preview", {}, token=False)[0], 403)
            self.assertEqual(self.post(f"/api/campaigns/{identifier}/stop", {}, token=False)[0], 403)
            self.assertEqual(self.post("/api/campaigns?mode=override", {})[0], 400)
            self.assertEqual(self.post(f"/api/campaigns/{identifier}/stop", {"kill_all": True})[0], 400)
            start.assert_not_called()
            stop.assert_not_called()

    def test_paused_campaign_reserves_both_other_launch_paths(self):
        with patch.object(self.server.campaigns, "active_campaign", return_value={"id": "a" * 32, "status": "paused"}), \
             patch.object(self.server.jobs, "start") as start, patch.object(self.server.native_queues, "start") as native:
            self.assertEqual(self.post("/api/jobs", {"task_id": "case", "action": "automl"})[0], 409)
            self.assertEqual(self.post("/api/native-queues", {
                "regression_task_id": "case", "classification_task_id": "case"})[0], 409)
            start.assert_not_called()
            native.assert_not_called()

    def test_owned_child_stop_directs_user_to_whole_series(self):
        identifier = "b" * 32
        campaign = {"id": "a" * 32, "status": "running", "steps": [{"job_id": identifier}]}
        with patch.object(self.server.campaigns, "active_campaign", return_value=campaign), \
             patch.object(self.server.jobs, "stop") as stop:
            status, _, body = self.post(f"/api/jobs/{identifier}/stop", {"task_id": "case"})
            self.assertEqual(status, 409, body)
            self.assertIn("Остановить всю серию", json.loads(body)["error"])
            stop.assert_not_called()

    def test_report_downloads_have_bounded_fixed_formats_and_no_paths(self):
        identifier = "a" * 32
        detail = {"id": identifier, "name": "Тест", "status": "completed", "steps": [],
                  "source_seal": {"valid": False}, "_private": "never-export-this"}
        with patch.object(self.server.campaigns, "detail", return_value=detail):
            for filename, mime in (("report.json", "application/json"), ("report.csv", "text/csv"), ("bundle.zip", "application/zip")):
                status, headers, body = self.request(f"/api/campaigns/{identifier}/{filename}")
                self.assertEqual(status, 200, body)
                self.assertTrue(headers["Content-Type"].startswith(mime))
                self.assertIn("attachment", headers["Content-Disposition"])
                self.assertNotIn(b"never-export-this", body)
            self.assertEqual(self.request(f"/api/campaigns/{identifier}/source.py")[0], 404)
            self.assertEqual(self.request(f"/api/campaigns/{identifier}/report.json?path=secret")[0], 400)

    def test_paused_plan_alone_does_not_hold_sleep_request(self):
        with patch.object(self.server.jobs, "active_job", return_value=None), \
             patch.object(self.server.native_queues, "active_queue", return_value=None), \
             patch.object(self.server.campaigns, "active_campaign", return_value={"status": "paused"}):
            self.assertFalse(self.server._work_needs_awake())
        with patch.object(self.server.campaigns, "active_campaign", return_value={"status": "cleanup_pending"}):
            self.assertTrue(self.server._work_needs_awake())


if __name__ == "__main__":
    unittest.main()
