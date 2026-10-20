"""The flat submission runs v3 with real SDK transport and unmocked host fits."""

from __future__ import annotations

import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile

from agent.llm import DEFAULT_MODEL
from build_submission import build_archive
from tests.test_research import make_regression
from validation import validate_predictions


LINEAR_CANDIDATE = """import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    column = context['numeric'][0]
    slope, intercept = np.polyfit(np.asarray(X_fit[column], dtype=float),
                                 np.asarray(y_fit, dtype=float), 1)
    return {'predictions': slope * np.asarray(X_predict[column], dtype=float) + intercept}
"""


def _candidate_tool():
    return (
        "import json\nfrom pathlib import Path\n"
        f"Path('research_candidate.py').write_text({LINEAR_CANDIDATE!r}, encoding='utf-8')\n"
        "Path('research_recipe.json').write_text(json.dumps({'family': 'linear', 'degree': 1}), encoding='utf-8')\n"
        "Path('research_hypothesis.json').write_text(json.dumps({'hypothesis': "
        "'Measure a linear fit on the host-approved inner split.'}), encoding='utf-8')\n"
    )


def _completion(*, tool=False):
    message = {"role": "assistant", "content": "Measured one inner hypothesis."}
    if tool:
        message["content"] = None
        message["tool_calls"] = [{
            "id": "call-flat-v3", "type": "function", "function": {
                "name": "run_python", "arguments": json.dumps({"code": _candidate_tool()}),
            },
        }]
    return {
        "id": "chatcmpl-flat-v3", "object": "chat.completion", "created": 1,
        "model": DEFAULT_MODEL,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool else "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class ResearchV3ArchiveTests(unittest.TestCase):
    def test_flat_archive_uses_default_v3_and_host_finalization_without_repository(self):
        requests, request_stages, request_final_checks = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, task_dir = root / "solution.zip", root / "task"
            task_dir.mkdir()
            names = build_archive(archive)
            self.assertIn("_alfa_agent__research_v3.py", names)
            self.assertFalse(any("/" in name or name.startswith("app") for name in names))
            with zipfile.ZipFile(archive) as zipped:
                # Match the platform: copy root Python sources and requirements,
                # with no access to an extracted agent/ package or the app.
                for name in zipped.namelist():
                    if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                        (task_dir / name).write_bytes(zipped.read(name))
            task = make_regression(task_dir)
            source_dir = task_dir / "regression" / "participants"
            source_bytes = {path.name: path.read_bytes() for path in source_dir.iterdir()}
            task_bytes = task.read_bytes()

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    requests.append(payload)
                    reports = list((task_dir / ".agent_runs").glob("*/run.json"))
                    current = json.loads(reports[0].read_text(encoding="utf-8")) if reports else {}
                    request_stages.append(current.get("stage"))
                    request_final_checks.append(current.get("evaluations", {}).get("final_check"))
                    body = json.dumps(_completion(tool=len(requests) == 1)).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
            thread.start()
            try:
                env = {name: value for name, value in os.environ.items()
                       if not name.lower().endswith("_proxy")
                       and name not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_MODEL"}}
                env.update(OPENAI_API_KEY="local-flat-v3-placeholder",
                           OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                           PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
                result = subprocess.run(
                    [sys.executable, "solution.py", "task.md", "--mode", "auto",
                     "--time-budget", "90", "--max-steps", "2"],
                    cwd=task_dir, env=env, capture_output=True, text=True,
                    encoding="utf-8", timeout=115,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(requests), 2, result.stdout + result.stderr)
            self.assertEqual(request_stages, ["inner_research", "inner_research"])
            self.assertEqual(request_final_checks, [None, None])
            self.assertTrue(all(request["model"] == DEFAULT_MODEL for request in requests))
            self.assertTrue(validate_predictions(task_dir)["valid"])
            self.assertFalse((task_dir / "agent").exists())
            self.assertFalse((task_dir / "app").exists())
            self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in source_dir.iterdir()})
            self.assertEqual(task.read_bytes(), task_bytes)
            with (task_dir / "submission.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["row_key"] for row in rows], ["0005", "NA"])
            for row, expected in zip(rows, [17, 20]):
                self.assertAlmostEqual(float(row["outcome"]), expected, places=7)

            reports = list((task_dir / ".agent_runs").glob("*/run.json"))
            self.assertEqual(len(reports), 1)
            run_dir = reports[0].parent
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["research_protocol_version"], 3)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["baseline_search"]["scope"], "outer_fit_only")
            self.assertEqual(report["baseline_search"]["status"], "searched")
            self.assertTrue((run_dir / "startup" / "automl_report.json").is_file())
            self.assertEqual(report["evaluations"]["selection"]["scope"], "inner_selection")
            final_check = report["evaluations"]["final_check"]
            self.assertIsNotNone(final_check["baseline"])
            self.assertTrue(final_check["selection_uses_outer"])
            self.assertFalse(final_check["unbiased_estimate"])
            self.assertEqual(report["api"]["attempts"], 2)
            self.assertEqual(report["api"]["responses"], 2)
            self.assertEqual(report["api"]["total_tokens"], 60)
            receipts = report["candidate_receipts"]
            outer = json.loads(Path(receipts["outer"]).read_text(encoding="utf-8"))
            refit = json.loads(Path(receipts["refit"]).read_text(encoding="utf-8"))
            for field in ("code_hash", "recipe_hash"):
                self.assertEqual(outer[field], refit[field])
                self.assertEqual(refit[field], receipts[field])
            self.assertEqual(outer["phase"], "outer_validation")
            self.assertEqual(refit["phase"], "full_refit")
            self.assertTrue((run_dir / "last_valid_submission.csv").is_file())


if __name__ == "__main__":
    unittest.main()
