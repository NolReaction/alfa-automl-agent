import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.research import run_agent
from validation import validate_predictions


class FakeCall:
    def __init__(self, code):
        self.id = "local-research-call"
        self.function = SimpleNamespace(name="run_python", arguments=json.dumps({"code": code}))

    def model_dump(self, **kwargs):
        return {"id": self.id, "type": "function", "function": vars(self.function)}


def response(*, code=None, content="Done", usage=None):
    result = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if code is not None else "stop",
        message=SimpleNamespace(content=content, tool_calls=[FakeCall(code)] if code is not None else None),
    )])
    # Omitting this field simulates providers that do not report token usage.
    if usage is not None:
        result.usage = SimpleNamespace(**usage)
    return result


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.received_messages = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.received_messages.append(copy.deepcopy(kwargs["messages"]))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def make_regression(task_dir: Path) -> Path:
    inputs = task_dir / "regression" / "participants"
    write_csv(inputs / "data_dictionary.csv", [
        ["feature", "role", "dtype"],
        ["row_key", "id", "category"],
        ["measurement", "feature", "numeric"],
        ["outcome", "target", "numeric"],
    ])
    write_csv(inputs / "train.csv", [["row_key", "measurement", "outcome"]] + [
        [f"train-{index:03d}", index, 3 * index + 2] for index in range(100)
    ])
    write_csv(inputs / "test.csv", [
        ["row_key", "measurement"], ["0005", 5], ["NA", 6],
    ])
    write_csv(inputs / "sample_submission.csv", [
        ["row_key", "outcome"], ["0005", 0], ["NA", 0],
    ])
    task = task_dir / "task.md"
    task.write_text(
        "Regression: use regression/participants/*.csv and predict outcome. "
        "Optimize RMSE on a training holdout. Save submission.csv here.",
        encoding="utf-8",
    )
    return task


def existing_output(task_dir: Path) -> bytes:
    # Noncanonical formatting makes a byte-for-byte recovery check meaningful.
    original = b"row_key,outcome\r\n0005,17.0000\r\nNA,20.0000\r\n"
    (task_dir / "submission.csv").write_bytes(original)
    return original


class ResearchTests(unittest.TestCase):
    def run_quietly(self, task, client, max_steps):
        with patch.dict(os.environ, {"OPENAI_MODEL": "provided-test-model"}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_agent(task, client, max_steps=max_steps, time_budget=90)

    def test_real_starting_model_survives_api_failure_and_redacts_key(self):
        secret = 'unit-test-credential-"quoted\\suffix'
        client = FakeClient([ConnectionError(f"Request failed with key {secret}")])
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
                run_dir = self.run_quietly(task, client, max_steps=2)
            self.assertEqual(validate_predictions(task_dir), {"valid": True, "errors": []})
            self.assertTrue((run_dir / "last_valid_submission.csv").is_file())
            self.assertTrue((task_dir / "baseline_metrics.json").is_file())
            raw_report = (run_dir / "run.json").read_text(encoding="utf-8")
            self.assertNotIn(secret, raw_report)
            self.assertNotIn(json.dumps(secret)[1:-1], raw_report)
            self.assertIn("[REDACTED]", raw_report)
            report = json.loads(raw_report)
            api_errors = [event for event in report["events"] if event["kind"] == "api_error"]
            self.assertEqual(len(api_errors), 1)
            self.assertNotIn(secret, api_errors[0]["error"])
            self.assertEqual(api_errors[0]["error"], "Request failed with key [REDACTED]")
            self.assertEqual((report["status"], report["reason"]), ("fallback", "api_error"))
            self.assertEqual(report["baseline"], "trained")
            self.assertEqual(report["experiments"], 0)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["api"]["attempts"], 1)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertEqual(report["api"]["usage_reported_responses"], 0)
            self.assertEqual(len(client.received_messages), 1)

    def test_failed_experiments_restore_existing_output_without_retraining(self):
        failures = {
            "deleted_output": "from pathlib import Path; Path('submission.csv').unlink(); raise RuntimeError('failed experiment')",
            "valid_but_failed_output": (
                "from pathlib import Path; "
                "Path('submission.csv').write_text('row_key,outcome\\n0005,999\\nNA,999\\n'); "
                "raise RuntimeError('failed experiment')"
            ),
        }
        for name, code in failures.items():
            with self.subTest(failure=name), tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory)
                task = make_regression(task_dir)
                original = existing_output(task_dir)
                client = FakeClient([response(code=code, content=None), response()])
                run_dir = self.run_quietly(task, client, max_steps=2)
                self.assertEqual((task_dir / "submission.csv").read_bytes(), original)
                self.assertEqual((run_dir / "last_valid_submission.csv").read_bytes(), original)
                self.assertTrue(validate_predictions(task_dir)["valid"])
                self.assertFalse((task_dir / "baseline_metrics.json").exists())
                report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(report["baseline"], "existing_output")
                self.assertEqual(report["experiments"], 1)
                self.assertEqual(report["successful_experiments"], 0)
                self.assertEqual((report["status"], report["reason"]), ("fallback", "step_limit"))
                tool_messages = [message for message in client.received_messages[1]
                                 if message["role"] == "tool"]
                self.assertEqual(len(tool_messages), 1)
                tool_result = json.loads(tool_messages[0]["content"])
                self.assertNotEqual(tool_result["exit_code"], 0)
                self.assertTrue(tool_result["previous_output_restored"])
                self.assertEqual(report["api"]["usage_reported_responses"], 0)
                self.assertEqual(report["api"]["total_tokens"], 0)

    def test_done_requires_experiment_and_usage_counts_only_reported_responses(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        client = FakeClient([
            response(),
            response(code="print('Inspected a local research hypothesis')", content=None, usage=usage),
        ])
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            original = existing_output(task_dir)
            run_dir = self.run_quietly(task, client, max_steps=2)
            self.assertEqual((task_dir / "submission.csv").read_bytes(), original)
            self.assertEqual(len(client.received_messages), 2)
            follow_up = client.received_messages[1][-1]
            self.assertEqual(follow_up["role"], "user")
            self.assertIn("run_python", follow_up["content"])
            self.assertIn("hypothesis", follow_up["content"])
            self.assertIn("validation", follow_up["content"])
            report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(report["baseline"], "existing_output")
            self.assertEqual(report["experiments"], 1)
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["api"], {
                "attempts": 2, "responses": 2, "usage_reported_responses": 1,
                **usage,
            })
            self.assertEqual((report["status"], report["reason"]), ("fallback", "step_limit"))


if __name__ == "__main__":
    unittest.main()
