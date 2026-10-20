from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import solution


class FakeCall:
    def __init__(self, code):
        self.id = "test-call"
        self.function = SimpleNamespace(name="run_python", arguments=json.dumps({"code": code}))

    def model_dump(self, **kwargs):
        return {"id": self.id, "type": "function", "function": vars(self.function)}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.chat = SimpleNamespace(completions=self)
        self.received_messages = []

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.received_messages.append(list(kwargs["messages"]))
        message = next(self.responses)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class ExecutionTests(unittest.TestCase):
    def test_generated_code_reads_utf8_files_independently_of_host_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / "manifest.json").write_text(
                json.dumps({"reason": "Проверка гипотезы", "label": "网络"}, ensure_ascii=False),
                encoding="utf-8",
            )
            code = (
                "import json; from pathlib import Path; "
                "value = json.loads(Path('manifest.json').read_text()); "
                "Path('copy.json').write_text(json.dumps(value, ensure_ascii=False))"
            )
            with patch.dict(os.environ, {"PYTHONUTF8": "0"}):
                result = solution.run_python(code, task, task / "logs", 1, 10)
            self.assertEqual(result["exit_code"], 0, result)
            self.assertEqual((task / "copy.json").read_bytes(), (task / "manifest.json").read_bytes())

    def test_generated_modules_can_be_imported_from_the_task_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / "helper.py").write_text("VALUE = 42\n")
            result = solution.run_python(
                "import helper; print(helper.VALUE)", task, task / "logs", 1, 10,
            )
            self.assertEqual(result["exit_code"], 0, result)
            self.assertEqual(result["stdout"].strip(), "42")

    def test_relative_paths_and_api_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            code = (
                "import os; from pathlib import Path; "
                "assert 'OPENAI_API_KEY' not in os.environ; "
                "Path('answer.txt').write_text('ok'); print('done')"
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}):
                result = solution.run_python(code, task, task / "logs", 1, timeout=10)
            self.assertEqual(result["exit_code"], 0, result)
            self.assertEqual((task / "answer.txt").read_text(encoding="utf-8"), "ok")
            self.assertIn("done", result["stdout"])

    def test_timed_out_experiment_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            output = task / "submission.csv"
            output.write_text("id,prediction\n1,2\n")
            result = solution.run_python(
                "import time; time.sleep(20)", task, task / "logs", 1, timeout=0.2,
            )
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "id,prediction\n1,2\n")

    def test_large_output_is_returned_as_a_bounded_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            result = solution.run_python("print('x' * 50000)", task, task / "logs", 1, 10)
            self.assertEqual(result["exit_code"], 0)
            self.assertLessEqual(len(result["stdout"].encode()), 12000)


class AgentTests(unittest.TestCase):
    def make_task(self, root):
        folder = root / "inputs"
        folder.mkdir()
        (folder / "sample_submission.csv").write_text("id,y\n1,0\n")
        (folder / "test.csv").write_text("id,x\n1,4\n")
        (folder / "train.csv").write_text("id,x,y\n0,1,2\n")
        (folder / "data_dictionary.csv").write_text(
            "feature,role,dtype\nid,id,numeric\nx,feature,numeric\ny,target,numeric\n"
        )
        task = root / "task.md"
        task.write_text("Write submission.csv using inputs/*.csv.")
        return task

    def test_tool_failure_is_returned_to_model_and_can_be_corrected(self):
        messages = [
            SimpleNamespace(content=None, tool_calls=[FakeCall("raise ValueError('try again')")]),
            SimpleNamespace(content=None, tool_calls=[FakeCall(
                "from pathlib import Path; import json; "
                "Path('submission.csv').write_text('id,y\\n1,2\\n'); "
                "Path('research_hypothesis.json').write_text(json.dumps({'hypothesis': 'Use the observed training value as a constant prediction.'}))"
            )]),
            SimpleNamespace(content="Done", tool_calls=None),
        ]
        client = FakeClient(messages)
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            with patch.dict(os.environ, {"OPENAI_MODEL": "provided-test-model"}):
                logs = solution.run_agent(task, client, max_steps=3, time_budget=30)
            self.assertEqual((task.parent / "submission.csv").read_text(encoding="utf-8"), "id,y\n1,2\n")
            failure = json.loads(client.received_messages[1][-1]["content"])
            self.assertNotEqual(failure["exit_code"], 0)
            self.assertIn("try again", failure["stderr"])
            self.assertTrue(logs.is_dir())
            report = json.loads((logs / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((report["status"], report["reason"]),
                             ("fallback", "host_metric_unsupported"))
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["unverified_experiments"], 1)

    def test_step_limit_does_not_delete_existing_predictions(self):
        client = FakeClient([SimpleNamespace(content=None, tool_calls=[FakeCall(
            "from pathlib import Path; import json; "
            "Path('submission.csv').write_text('id,y\\n1,2\\n'); "
            "Path('research_hypothesis.json').write_text(json.dumps({'hypothesis': 'Use the observed training value as a constant prediction.'}))"
        )])])
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            with patch.dict(os.environ, {"OPENAI_MODEL": "provided-test-model"}):
                solution.run_agent(task, client, max_steps=1, time_budget=30)
            self.assertTrue((task.parent / "submission.csv").is_file())

    def test_model_saying_done_without_predictions_does_not_count_as_success(self):
        client = FakeClient([SimpleNamespace(content="Done", tool_calls=None)])
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            with patch.dict(os.environ, {"OPENAI_MODEL": "provided-test-model"}):
                with self.assertRaisesRegex(RuntimeError, "Step limit"):
                    solution.run_agent(task, client, max_steps=1, time_budget=30)


if __name__ == "__main__":
    unittest.main()
