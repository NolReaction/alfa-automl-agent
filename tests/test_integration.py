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
        experiment = """from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
manifest = json.loads(Path('research_validation.json').read_text(encoding='utf-8'))
train = pd.read_csv('inputs/train.csv', dtype=str, keep_default_na=False)
test = pd.read_csv('inputs/test.csv', dtype=str, keep_default_na=False)
x = train[['feature']].astype(float)
y = train['outcome'].astype(float)
fit, held = manifest['fit_rows'], manifest['validation_rows']
model = LinearRegression().fit(x.iloc[fit], y.iloc[fit])
pd.DataFrame({
    'id': train['row_id'].iloc[held].to_numpy(),
    'actual': train['outcome'].iloc[held].to_numpy(),
    'prediction': model.predict(x.iloc[held]),
}).to_csv('holdout_predictions.csv', index=False)
Path('research_hypothesis.json').write_text(json.dumps({
    'hypothesis': 'A linear model should improve on trees for this linear target.'
}), encoding='utf-8')
model.fit(x, y)
result = pd.DataFrame({'row_id': test['row_id'], 'outcome': model.predict(test[['feature']].astype(float))})
result.to_csv('submission.tmp.csv', index=False)
Path('submission.tmp.csv').replace('submission.csv')
print('predictions-written')
"""

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, payload))
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
                self.assertEqual(requests[0][0], "/v1/chat/completions")
                tool_reply = json.loads(next(item["content"] for item in reversed(requests[1][1]["messages"])
                                             if item["role"] == "tool"))
                self.assertEqual(tool_reply["exit_code"], 0)
                self.assertIn("predictions-written", tool_reply["stdout"])
                self.assertTrue(tool_reply["selection"]["verified"])
                self.assertTrue(tool_reply["selection"]["accepted"])
                self.assertLess(tool_reply["selection"]["metric"]["value"], 1e-10)
                for path, content in original_inputs.items():
                    self.assertEqual(path.read_bytes(), content)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
