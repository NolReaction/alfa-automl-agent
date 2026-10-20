"""Real SDK and child-process round trip against a local, fake API server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


HAS_PACKAGES = all(importlib.util.find_spec(name) is not None
                   for name in ("openai", "pandas", "sklearn"))


@unittest.skipUnless(HAS_PACKAGES, "Install requirements.txt to run the real-SDK integration test")
class LocalApiTests(unittest.TestCase):
    def test_cli_to_api_to_training_to_csv(self):
        requests = []
        restored_outputs = []
        candidate = """from sklearn.linear_model import LinearRegression
def fit_predict(X_fit, y_fit, X_predict, context):
    model = LinearRegression().fit(X_fit[['feature']].astype(float), y_fit)
    return {'predictions': model.predict(X_predict[['feature']].astype(float))}
"""
        experiment_template = """from pathlib import Path
import json
manifest = json.loads(Path('research_validation.json').read_text(encoding='utf-8'))
assert manifest['version'] == 3
assert manifest['training_data_scope'] == 'outer_fit_only'
Path('research_candidate.py').write_text(__CANDIDATE__, encoding='utf-8')
Path('research_recipe.json').write_text(json.dumps({'family': 'linear'}), encoding='utf-8')
Path('research_hypothesis.json').write_text(json.dumps({
    'hypothesis': 'A linear model should improve on trees for this linear target.'
}), encoding='utf-8')
Path(__PUBLISHED_PATH__).write_text('row_id,outcome\\n20,987654\\n10,987654\\n', encoding='utf-8')
print('candidate-written')
"""

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, payload))
                if len(requests) > 1:
                    restored_outputs.append((task_dir / "submission.csv").read_bytes())
                if len(requests) == 1:
                    message = {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call-local", "type": "function", "function": {
                            "name": "run_python", "arguments": json.dumps({"code": experiment}),
                        }}],
                    }
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "Training complete."}
                    finish = "stop"
                body = json.dumps({
                    "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                    "model": "provided-test-model", "choices": [{
                        "index": 0, "message": message, "finish_reason": finish,
                    }],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory)
                inputs = task_dir / "inputs"
                inputs.mkdir()
                (task_dir / "task.md").write_text(
                    "Train a regression model using inputs/*.csv. Write submission.csv.\nMetric: RMSE\n")
                (inputs / "train.csv").write_text(
                    "row_id,feature,outcome\n" + "".join(
                        f"{index},{index},{3 * index + 1}\n" for index in range(100)))
                (inputs / "test.csv").write_text("row_id,feature\n20,4\n10,5\n")
                (inputs / "sample_submission.csv").write_text("row_id,outcome\n20,0\n10,0\n")
                (inputs / "data_dictionary.csv").write_text(
                    "feature,role,dtype\nrow_id,id,numeric\nfeature,feature,numeric\noutcome,target,numeric\n"
                )
                original_output = b"row_id,outcome\r\n20,111.0000\r\n10,222.0000\r\n"
                (task_dir / "submission.csv").write_bytes(original_output)
                experiment = experiment_template.replace("__CANDIDATE__", repr(candidate)).replace(
                    "__PUBLISHED_PATH__", repr(str(task_dir / "submission.csv")))
                original_inputs = {path: path.read_bytes() for path in inputs.iterdir()}
                # The local fixture must not inherit a workstation's proxy.
                env = {key: value for key, value in os.environ.items()
                       if not key.lower().endswith("_proxy")}
                env.update({
                    "OPENAI_API_KEY": "local-test-placeholder",
                    "OPENAI_MODEL": "provided-test-model",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                })
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve().parents[1] / "solution.py"),
                     str(task_dir / "task.md"), "--mode", "llm", "--max-steps", "3", "--time-budget", "90"],
                    env=env, capture_output=True, text=True, timeout=105,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                import pandas as pd

                output = pd.read_csv(task_dir / "submission.csv")
                self.assertEqual(output["row_id"].tolist(), [20, 10])
                self.assertAlmostEqual(output["outcome"].iloc[0], 13)
                self.assertAlmostEqual(output["outcome"].iloc[1], 16)
                self.assertEqual(len(requests), 2)
                self.assertEqual(restored_outputs, [original_output])
                self.assertEqual(requests[0][0], "/v1/chat/completions")
                tool_reply = json.loads(next(item["content"] for item in reversed(requests[1][1]["messages"])
                                             if item["role"] == "tool"))
                self.assertEqual(tool_reply["exit_code"], 0)
                self.assertIn("candidate-written", tool_reply["stdout"])
                self.assertTrue(tool_reply["selection"]["verified"])
                self.assertTrue(tool_reply["selection"]["accepted"])
                self.assertLess(tool_reply["selection"]["metric"]["value"], 1e-10)
                self.assertEqual(tool_reply["selection"]["scope"], "inner_selection")
                reports = list((task_dir / ".agent_runs").glob("*/run.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertEqual(report["research_protocol_version"], 3)
                self.assertEqual((report["status"], report["reason"]), ("completed", "model_finished"))
                self.assertTrue(report["evaluations"]["final_check"]["selection_uses_outer"])
                self.assertFalse(report["evaluations"]["final_check"]["unbiased_estimate"])
                receipts = report["candidate_receipts"]
                outer_receipt = json.loads(Path(receipts["outer"]).read_text(encoding="utf-8"))
                refit_receipt = json.loads(Path(receipts["refit"]).read_text(encoding="utf-8"))
                self.assertEqual(outer_receipt["code_hash"], refit_receipt["code_hash"])
                self.assertEqual(outer_receipt["recipe_hash"], refit_receipt["recipe_hash"])
                self.assertEqual(outer_receipt["code_hash"], report["frozen_candidate"]["code_hash"])
                for path, content in original_inputs.items():
                    self.assertEqual(path.read_bytes(), content)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
