"""Exercise the HTTP boundary, not just the routing implementation."""

import csv
from contextlib import closing
import http.client
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import threading
import unittest

from app.server import WorkspaceServer


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        task = self.root / "case"
        task.mkdir()
        (task / "task.md").write_text("Classification. Metric: Macro F1", encoding="utf-8")
        records = {
            "data_dictionary.csv": [["feature", "role", "dtype"], ["id", "id", "category"],
                                    ["feature", "feature", "binary"], ["label", "target", "category"]],
            "train.csv": [["id", "feature", "label"], ["0001", "0", "NA"], ["0002", "1", "yes"]],
            "test.csv": [["id", "feature"], ["0010", "1"]],
            "sample_submission.csv": [["id", "label"], ["0010", "yes"]],
        }
        for name, rows in records.items():
            with (task / name).open("w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(rows)
        (task / ".env").write_text("SECRET=must-never-be-served", encoding="utf-8")
        self.server = WorkspaceServer(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = f"127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, method="GET", headers=None):
        with closing(http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)) as client:
            client.request(method, path, headers=headers or {})
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def test_actual_data_flows_through_json_without_losing_na_or_ids(self):
        status, headers, body = self.request("/api/tasks")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["tasks"][0]["id"], "case")
        status, headers, body = self.request("/api/tasks/case/profile")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["preview"]["rows"][0], ["0001", "0", "NA"])
        self.assertEqual(data["target"], "label")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_page_assets_are_local_and_served_with_correct_types(self):
        class Assets(HTMLParser):
            def __init__(self):
                super().__init__()
                self.paths = []

            def handle_starttag(self, tag, attributes):
                attrs = dict(attributes)
                if tag == "script" and "src" in attrs:
                    self.paths.append((attrs["src"], "javascript"))
                if tag == "link" and attrs.get("rel") == "stylesheet":
                    self.paths.append((attrs["href"], "text/css"))

        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        assets = Assets()
        assets.feed(body.decode("utf-8"))
        self.assertEqual(len(assets.paths), 2)
        for path, mime in assets.paths:
            self.assertTrue(path.startswith("/") and not path.startswith("//"))
            status, headers, content = self.request(path)
            self.assertEqual(status, 200, path)
            self.assertIn(mime, headers["Content-Type"])
            self.assertTrue(content)
        status, _, body = self.request("/", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_remote_origins_dns_rebinding_and_mutations_are_rejected(self):
        for headers in [{"Host": "evil.example"}, {"Origin": "https://evil.example"},
                        {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}]:
            self.assertEqual(self.request("/api/tasks", headers=headers)[0], 403)
        self.assertEqual(self.request("/api/tasks", headers={"Origin": "http://" + self.host})[0], 200)
        for method in ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"]:
            self.assertEqual(self.request("/api/tasks", method=method)[0], 405)

    def test_secret_and_arbitrary_files_cannot_be_addressed(self):
        paths = ["/.env", "/../.env", "/%2e%2e/.env", "/api/tasks/case/.env",
                 "/api/tasks/case/runs/../../.env", "/api/tasks/case/description/extra",
                 "/api/tasks/%2e%2e/profile", "/app/server.py", "/README.md"]
        for path in paths:
            status, _, body = self.request(path)
            self.assertIn(status, {400, 403, 404}, path)
            self.assertNotIn(b"must-never-be-served", body)

    def test_missing_result_and_empty_runs_are_normal_states(self):
        status, _, body = self.request("/api/tasks/case/runs")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["runs"], [])
        status, _, body = self.request("/api/tasks/case/result")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["submission"]["exists"])
        self.assertIsNone(json.loads(body)["baseline"])

    def test_download_serves_only_submission_and_is_an_attachment(self):
        file = self.root / "case" / "submission.csv"
        file.write_text("id,label\n0010,NA\n", encoding="utf-8")
        status, headers, body = self.request("/api/tasks/case/submission")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertEqual(body, file.read_bytes())
        before = file.read_bytes()
        self.request("/api/tasks/case/submission", method="DELETE")
        self.assertEqual(file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
