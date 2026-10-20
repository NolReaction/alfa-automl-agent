"""Exercise the HTTP boundary, not just the routing implementation."""

import csv
from contextlib import closing
import http.client
from io import BytesIO
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

import build_submission

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
            "train.csv": [["id", "feature", "label"]] + [
                [f"{index + 1:04d}", str(index % 2), "yes" if index % 2 else "NA"]
                for index in range(40)
            ],
            "test.csv": [["id", "feature"], ["0010", "1"]],
            "sample_submission.csv": [["id", "label"], ["0010", "yes"]],
        }
        for name, rows in records.items():
            with (task / name).open("w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(rows)
        (task / ".env").write_text("SECRET=must-never-be-served", encoding="utf-8")
        self.server = WorkspaceServer(self.root, port=0, state_root=self.root / "workspace")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = f"127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, method="GET", headers=None, body=None):
        with closing(http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)) as client:
            client.request(method, path, headers=headers or {}, body=body)
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def post(self, path, value, *, token=True, headers=None):
        request_headers = {"Content-Type": "application/json"}
        if token:
            request_headers["X-Alfa-Token"] = self.server.token
        request_headers.update(headers or {})
        return self.request(path, method="POST", headers=request_headers,
                            body=json.dumps(value).encode("utf-8"))

    def test_mutations_require_session_token_and_same_origin(self):
        body = {"task_id": "case", "action": "baseline"}
        self.assertEqual(self.post("/api/jobs", body, token=False)[0], 403)
        self.assertEqual(self.post("/api/jobs", body, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.post("/api/jobs", {**body, "action": "run_shell"})[0], 400)
        self.assertEqual(self.post("/api/jobs", {**body, "command": "anything"})[0], 400)
        self.assertIsNone(self.server.jobs.active_job())
        status, _, response = self.request("/api/session")
        self.assertEqual(status, 200)
        session = json.loads(response)
        self.assertEqual(session["token"], self.server.token)
        self.assertEqual(set(session["api"]), {"ready", "missing", "model_configured"})

    def test_browser_job_is_isolated_until_explicit_publish_and_report_roundtrip(self):
        task = self.root / "case"
        original = b"id,label\n0010,NA\n"
        (task / "submission.csv").write_bytes(original)
        status, _, response = self.post("/api/jobs", {
            "task_id": "case", "action": "baseline", "author": "Test teammate",
            "hypothesis": "Use the baseline on fixed validation rows",
        })
        self.assertEqual(status, 202, response)
        job_id = json.loads(response)["job"]["id"]
        for _ in range(200):
            status, _, response = self.request(f"/api/jobs/{job_id}?task_id=case")
            self.assertEqual(status, 200, response)
            detail = json.loads(response)
            if detail["job"]["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        self.assertEqual(detail["job"]["status"], "completed", detail)
        self.assertTrue(detail["job"]["output_valid"])
        self.assertEqual(detail["job"]["metric"]["name"], "Macro F1")
        self.assertTrue(detail["stdout"])
        self.assertEqual((task / "submission.csv").read_bytes(), original)
        status, headers, csv_bytes = self.request(f"/api/jobs/{job_id}/submission?task_id=case")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertNotEqual(csv_bytes, original)
        status, headers, package = self.request(f"/api/jobs/{job_id}/bundle?task_id=case")
        self.assertEqual(status, 200, package)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with ZipFile(BytesIO(package)) as archive:
            self.assertEqual(set(archive.namelist()), {"submission.csv", "reports.json", "report.html"})
            self.assertEqual(archive.read("submission.csv"), csv_bytes)
            envelope = json.loads(archive.read("reports.json"))
            self.assertEqual([row["id"] for row in envelope["reports"]], [job_id])
            self.assertEqual(envelope["reports"][0]["metric"], detail["job"]["metric"])
            self.assertNotIn(b"must-never-be-served", b"".join(archive.read(name) for name in archive.namelist()))
        self.assertEqual(self.request(f"/api/jobs/{job_id}/bundle")[0], 400)
        self.assertEqual(self.request(f"/api/jobs/{job_id}/bundle?task_id=missing")[0], 400)
        status, _, response = self.post(f"/api/jobs/{job_id}/publish", {"task_id": "case"})
        self.assertEqual(status, 200, response)
        self.assertTrue(json.loads(response)["job"]["published"])
        self.assertEqual((task / "submission.csv").read_bytes(), csv_bytes)
        status, _, response = self.request("/api/tasks/case/workflow")
        self.assertEqual(status, 200, response)
        self.assertEqual(json.loads(response)["published_job_id"], job_id)
        status, headers, report = self.request("/api/tasks/case/export")
        self.assertEqual(status, 200, report)
        self.assertIn("attachment", headers["Content-Disposition"])
        envelope = json.loads(report)
        self.assertEqual(len(envelope["reports"]), 1)
        status, _, response = self.request("/api/tasks/case/import", method="POST", headers={
            "Content-Type": "application/zip", "X-Alfa-Token": self.server.token,
        }, body=package)
        self.assertEqual(status, 200, response)
        self.assertEqual(json.loads(response), {"imported": 1, "skipped": 0})
        status, _, response = self.post("/api/tasks/case/import", envelope)
        self.assertEqual(status, 200, response)
        self.assertEqual(json.loads(response), {"imported": 0, "skipped": 1})
        status, _, response = self.request("/api/tasks/case/experiments")
        self.assertEqual(status, 200, response)
        rows = json.loads(response)["experiments"]
        self.assertEqual(len(rows), 2)
        imported = next(item for item in rows if item["source"] == "imported")
        self.assertFalse(imported["published"])
        artifact = self.server.jobs.artifact("case", job_id)
        saved = artifact.read_bytes()
        artifact.write_bytes(original)
        self.assertEqual(self.request(f"/api/jobs/{job_id}/bundle?task_id=case")[0], 400)
        artifact.write_bytes(saved)
        with (task / "task.md").open("a", encoding="utf-8") as stream:
            stream.write("\nChanged data description")
        self.assertEqual(self.request(f"/api/jobs/{job_id}/bundle?task_id=case")[0], 400)

    def test_automl_http_preserves_auto_validation_and_budget_without_model_controls(self):
        body = {"task_id": "case", "action": "automl", "validation": {"strategy": "auto"}}
        for model_params in (None, {}, {"n_estimators": 50}):
            self.assertEqual(self.post("/api/jobs", {**body, "model_params": model_params})[0], 400)
        for minutes in (True, 0, 176, "10"):
            self.assertEqual(self.post("/api/jobs", {**body, "minutes": minutes})[0], 400)
        self.assertEqual(self.post("/api/jobs", {**body, "validation": {"strategy": "auto", "command": "x"}})[0], 400)
        self.assertIsNone(self.server.jobs.active_job())
        with patch.object(self.server.jobs, "start", return_value={"id": "example", "action": "automl"}) as start:
            status, _, response = self.post("/api/jobs", body)
        self.assertEqual(status, 202, response)
        self.assertEqual(start.call_args.kwargs["validation"], {"strategy": "auto"})
        self.assertEqual(start.call_args.kwargs["minutes"], 10)
        self.assertIsNone(start.call_args.kwargs["model_params"])

    def test_invalid_validation_is_rejected_before_starting_or_creating_artifacts(self):
        body = {"task_id": "case", "action": "baseline"}
        invalid = [True, [], "random", {}, {"strategy": []}, {"strategy": "unknown"},
                   {"strategy": "random", "command": "anything"},
                   {"strategy": "grouped", "path": "../../.env"},
                   {"strategy": "time", "time_column": "feature"}]
        for validation in invalid:
            with self.subTest(validation=validation):
                status, _, response = self.post("/api/jobs", {**body, "validation": validation})
                self.assertEqual(status, 400, response)
        for action in ("agent", "prepare"):
            with self.subTest(action=action):
                status, _, response = self.post("/api/jobs", {
                    "task_id": None if action == "prepare" else "case",
                    "action": action, "validation": {"strategy": "random"},
                })
                self.assertEqual(status, 400, response)
                self.assertIn("валидации", json.loads(response)["error"])
        self.assertEqual(self.server.jobs.list(), [])
        self.assertEqual(list(self.server.jobs.jobs_root.iterdir()), [])
        self.assertFalse((self.root / "case" / "submission.csv").exists())

    def test_browser_grouped_validation_reaches_worker_and_returns_group_evidence(self):
        task = self.root / "case"
        with (task / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([["id", "feature", "label"]] + [
                [f"train-{index}", index // 4, "yes" if index % 2 else "NA"] for index in range(40)])
        validation = {"strategy": "grouped"}
        status, _, response = self.post("/api/jobs", {
            "task_id": "case", "action": "extra_trees", "validation": validation,
            "model_params": {"n_estimators": 50, "max_depth": 4},
        })
        self.assertEqual(status, 202, response)
        job_id = json.loads(response)["job"]["id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status, _, response = self.request(f"/api/jobs/{job_id}?task_id=case")
            self.assertEqual(status, 200, response)
            detail = json.loads(response)
            if detail["job"]["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        self.assertEqual(detail["job"]["status"], "completed", detail)
        self.assertEqual(detail["job"]["validation"], validation)
        self.assertEqual(detail["job"]["split"]["strategy"], "grouped")
        self.assertEqual(detail["job"]["split"]["group_overlap"], 0)
        self.assertEqual(detail["job"]["split"]["group_count"], 10)
        self.assertEqual(detail["job"]["model_config"]["parameters"]["n_estimators"], 50)
        self.assertEqual(detail["job"]["model_config"]["parameters"]["max_depth"], 4)
        self.assertFalse((task / "submission.csv").exists())

    def test_profile_exposes_model_options_without_changing_cached_profile(self):
        profile = self.server.catalog.profile("case")
        self.assertNotIn("model_options", profile)
        with patch.object(self.server.catalog, "profile", return_value=profile):
            status, _, response = self.request("/api/tasks/case/profile")
        self.assertEqual(status, 200, response)
        options = json.loads(response)["model_options"]
        self.assertEqual(set(options), {"baseline", "extra_trees"})
        self.assertEqual(options["baseline"]["model"], "RandomForestClassifier")
        self.assertEqual(options["extra_trees"]["model"], "ExtraTreesClassifier")
        self.assertIn("n_estimators", options["baseline"]["defaults"])
        self.assertTrue(options["baseline"]["fields"])
        self.assertNotIn("model_options", profile)
        self.assertNotIn("model_options", self.server.catalog.profile("case"))

    def test_invalid_model_parameters_are_rejected_without_artifacts(self):
        body = {"task_id": "case", "action": "baseline"}
        invalid = [True, [], "anything", {"command": "anything"}, {"n_jobs": 1},
                   {"random_state": 0}, {"max_iter": 50}, {"n_estimators": True},
                   {"n_estimators": 49}, {"n_estimators": 601}, {"min_samples_leaf": 0},
                   {"max_depth": {"path": "../../.env"}}]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                status, _, response = self.post("/api/jobs", {**body, "model_params": parameters})
                self.assertEqual(status, 400, response)
        for action in ("agent", "prepare"):
            for parameters in (None, {}, {"n_estimators": 50}):
                with self.subTest(action=action, parameters=parameters):
                    status, _, response = self.post("/api/jobs", {
                        "task_id": None if action == "prepare" else "case",
                        "action": action, "model_params": parameters,
                    })
                    self.assertEqual(status, 400, response)
                    self.assertIn("модели", json.loads(response)["error"])
        self.assertEqual(self.server.jobs.list(), [])
        self.assertEqual(list(self.server.jobs.jobs_root.iterdir()), [])
        self.assertFalse((self.root / "case" / "submission.csv").exists())

    def test_delivery_is_token_protected_and_downloads_only_competition_sources(self):
        self.assertEqual(self.post("/api/delivery", {}, token=False)[0], 403)
        self.assertEqual(self.post("/api/delivery", {}, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.post("/api/delivery", {"path": "../.env"})[0], 400)
        self.assertEqual(self.request("/api/delivery")[0], 404)
        status, headers, package = self.post("/api/delivery", {})
        self.assertEqual(status, 200, package)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertIn('filename="solution.zip"', headers["Content-Disposition"])
        with ZipFile(BytesIO(package)) as archive:
            expected = {path.relative_to(build_submission.ROOT).as_posix()
                        for path in build_submission.source_files()}
            self.assertEqual(set(archive.namelist()), expected)
            self.assertIn("solution.py", archive.namelist())
            self.assertFalse(any(name.startswith(("app/", "data/")) or name.endswith(".env")
                                 for name in archive.namelist()))

    def test_zip_report_import_is_bounded_and_never_extracts_other_files(self):
        def package(entries):
            stream = BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                    for name, content in entries:
                        archive.writestr(name, content)
            return stream.getvalue()

        headers = {"Content-Type": "application/zip", "X-Alfa-Token": self.server.token}
        invalid = [
            b"not a ZIP",
            package([("nested/reports.json", b"{}")]),
            package([("reports.json", b"{}"), ("reports.json", b"{}")]),
            package([("reports.json", b" " * (1024 * 1024 + 1))]),
            package([("reports.json", b"{\"unsafe\":NaN}")]),
            package([("reports.json", b"{}"), *[(f"ignored-{n}", b"") for n in range(32)]]),
        ]
        for content in invalid:
            with self.subTest(size=len(content)):
                self.assertEqual(self.request("/api/tasks/case/import", method="POST",
                                              headers=headers, body=content)[0], 400)
        self.assertEqual(self.request("/api/tasks/case/import", method="POST", headers={
            **headers, "Content-Length": str(20 * 1024 * 1024 + 1),
        }, body=b"")[0], 400)
        self.assertEqual(self.request("/api/tasks/case/import", method="POST", headers={
            "Content-Type": "application/zip",
        }, body=invalid[0])[0], 403)
        self.assertEqual(self.server.reports.list("case"), [])
        envelope = self.server.reports.export("case", self.server.catalog.fingerprint("case"), [])
        safe = package([("reports.json", json.dumps(envelope).encode()),
                        ("../must-not-extract.txt", b"do not write"),
                        ("submission.csv", b"untrusted predictions"),
                        ("report.html", b"<script>never execute</script>")])
        status, _, response = self.request("/api/tasks/case/import", method="POST",
                                           headers=headers, body=safe)
        self.assertEqual(status, 200, response)
        self.assertEqual(json.loads(response), {"imported": 0, "skipped": 0})
        self.assertFalse((self.root / "must-not-extract.txt").exists())
        self.assertFalse((self.root / "case" / "submission.csv").exists())

    def test_import_cannot_replace_dataset_or_execute_unknown_content(self):
        response = self.post("/api/tasks/case/import", {
            "format": "alfa-lab-reports", "version": 1,
            "dataset_fingerprint": "0" * 64, "reports": [],
        })
        self.assertEqual(response[0], 400)
        self.assertIsNone(self.server.jobs.active_job())

    def test_isolated_agent_code_is_accessible_only_through_its_job_and_redacted(self):
        job_id, run_id = "f" * 32, "a" * 32
        folder = self.server.jobs.jobs_root / job_id / "task" / ".agent_runs" / run_id
        folder.mkdir(parents=True)
        self.server.jobs._jobs[job_id] = {"id": job_id, "task_id": "case", "action": "agent"}
        (folder / "run.json").write_text(json.dumps({
            "status": "fallback", "events": [], "api": {
                "attempts": 1, "responses": 0, "usage_reported_responses": 0,
            },
        }), encoding="utf-8")
        (folder / "experiment_101.py").write_text("print('fixture-secret')", encoding="utf-8")
        prefix = f"/api/jobs/{job_id}/agent-runs"
        status, _, body = self.request(prefix + "?task_id=case")
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["runs"][0]["id"], run_id)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-secret"}):
            status, _, body = self.request(prefix + f"/{run_id}/file?task_id=case&name=experiment_101.py")
        self.assertEqual(status, 200, body)
        self.assertNotIn(b"fixture-secret", body)
        self.assertIn(b"[REDACTED]", body)
        self.assertEqual(self.request(prefix + f"/{run_id}/file?task_id=case&name=.env")[0], 400)
        self.assertEqual(self.request(prefix + f"/{run_id}?task_id=missing")[0], 400)

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
