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
        "Regression: use regression/participants/*.csv and predict outcome.\n"
        "Metric: RMSE\nSave submission.csv here.", encoding="utf-8",
    )
    return task


def existing_output(task_dir: Path) -> bytes:
    original = b"row_key,outcome\r\n0005,17.0000\r\nNA,20.0000\r\n"
    (task_dir / "submission.csv").write_bytes(original)
    return original


def known_start(state):
    """Controlled checked starter; tests below exercise real generated processes."""
    if state.validator is None:
        state.report["baseline"] = "unavailable"
        return
    original = existing_output(state.workspace)
    gate = state.validator
    holdout = state.run_dir / "starter_holdout.csv"
    write_csv(holdout, [["id", "actual", "prediction"]] + [
        [row_id, actual, float(actual) + 10]
        for row_id, actual in zip(gate.expected_ids, gate.expected_actuals)
    ])
    metric = gate.evaluate(holdout, state.workspace)
    state.checkpoint(state.workspace, metric)
    state.report["baseline"] = "trained"
    assert (state.task_dir / "submission.csv").read_bytes() == original


# Real child code uses the fixed split and never reads validation answers to fit.
EXPERIMENT = """import csv, json
from pathlib import Path
spec = json.loads(Path('research_validation.json').read_text(encoding="utf-8"))
with (Path(spec['data_directory']) / 'train.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
held = [rows[index] for index in spec['validation_rows']]
with Path('holdout_predictions.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['id', 'actual', 'prediction'])
    writer.writerows([[row[spec['id_column']], row[spec['target_column']],
                       3 * float(row['measurement']) + 2 + OFFSET] for row in held])
Path('submission.csv').write_text('row_key,outcome\\n0005,17\\nNA,20\\n')
Path('research_hypothesis.json').write_text(json.dumps({'hypothesis': 'Test a linear relation'}))
"""


class ResearchTests(unittest.TestCase):
    def run_quietly(self, task, client, max_steps, *, real_start=False):
        with patch.dict(os.environ, {"OPENAI_MODEL": "provided-test-model"}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if real_start:
                return run_agent(task, client, max_steps=max_steps, time_budget=90)
            with patch("agent.research.prepare_starting_output", side_effect=known_start):
                return run_agent(task, client, max_steps=max_steps, time_budget=90)

    def test_real_starting_model_survives_api_failure_and_redacts_key(self):
        secret = 'unit-test-credential-"quoted\\suffix'
        client = FakeClient([ConnectionError(f"Request failed with key {secret}")])
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            before = {path.name: path.read_bytes() for path in (task_dir / "regression/participants").iterdir()}
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
                run_dir = self.run_quietly(task, client, max_steps=2, real_start=True)
            self.assertEqual(validate_predictions(task_dir), {"valid": True, "errors": []})
            self.assertTrue((run_dir / "last_valid_submission.csv").is_file())
            self.assertTrue((run_dir / "best_holdout_predictions.csv").is_file())
            self.assertTrue((run_dir / "startup/automl_report.json").is_file())
            raw = (run_dir / "run.json").read_text(encoding="utf-8")
            self.assertNotIn(secret, raw)
            self.assertNotIn(json.dumps(secret)[1:-1], raw)
            self.assertIn("[REDACTED]", raw)
            report = json.loads(raw)
            self.assertEqual((report["status"], report["reason"]), ("fallback", "api_error"))
            self.assertEqual(report["baseline"], "trained")
            self.assertEqual(report["best_metric"]["name"], "RMSE")
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["api"]["attempts"], 1)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in (task_dir / "regression/participants").iterdir()})

    def test_worse_and_tied_valid_experiments_keep_checked_best_bytewise(self):
        for offset in (1000, 10):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory)
                task = make_regression(task_dir)
                expected = existing_output(task_dir)
                client = FakeClient([response(code=EXPERIMENT.replace("OFFSET", str(offset))), response()])
                run_dir = self.run_quietly(task, client, 2)
                report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual((task_dir / "submission.csv").read_bytes(), expected)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["successful_experiments"], 1)
                self.assertEqual(report["improvements"], 0)
                self.assertEqual(report["best_metric"]["value"], 10)
                selection = report["events"][-1]["result"]["selection"]
                self.assertEqual(selection["reason"], "not_better")
                self.assertTrue(selection["verified"])
                self.assertFalse(selection["accepted"])

    def test_improvement_is_published_and_stale_evidence_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            client = FakeClient([
                response(code=EXPERIMENT.replace("OFFSET", "0")),
                response(code="from pathlib import Path; Path('submission.csv').write_text('row_key,outcome\\n0005,999\\nNA,999\\n')"),
                response(),
            ])
            run_dir = self.run_quietly(task, client, 3)
            report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(report["best_metric"]["value"], 0)
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["improvements"], 1)
            self.assertEqual((task_dir / "submission.csv").read_text(encoding="utf-8"), "row_key,outcome\n0005,17\nNA,20\n")
            self.assertTrue((run_dir / "best_holdout_predictions.csv").is_file())
            selections = [event["result"]["selection"] for event in report["events"] if event["kind"] == "experiment"]
            self.assertTrue(selections[0]["accepted"])
            self.assertFalse(selections[1]["accepted"])

    def test_print_only_does_not_complete_research_and_usage_counts_reported_responses(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            expected = existing_output(task_dir)
            client = FakeClient([response(), response(code="print('hypothesis inspected')", usage=usage)])
            run_dir = self.run_quietly(task, client, 2)
            report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((task_dir / "submission.csv").read_bytes(), expected)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual((report["status"], report["reason"]), ("fallback", "step_limit"))
            self.assertEqual(report["api"], {"attempts": 2, "responses": 2, "usage_reported_responses": 1, **usage})
            instruction = client.received_messages[1][-1]["content"]
            self.assertIn("hypothesis", instruction)
            self.assertIn("validation", instruction)

    def test_failed_or_misaligned_experiment_cannot_replace_best(self):
        codes = [
            EXPERIMENT.replace("OFFSET", "0") + "\nraise RuntimeError('experiment crashed')",
            EXPERIMENT.replace("OFFSET", "0").replace("row[spec['id_column']]", "'wrong-id'"),
            EXPERIMENT.replace("OFFSET", "0").replace("row[spec['target_column']]", "'false-answer'"),
        ]
        for code in codes:
            with self.subTest(code=code[-50:]), tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory)
                task = make_regression(task_dir)
                expected = existing_output(task_dir)
                client = FakeClient([response(code=code), response()])
                run_dir = self.run_quietly(task, client, 2)
                report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual((task_dir / "submission.csv").read_bytes(), expected)
                self.assertEqual(report["successful_experiments"], 0)
                self.assertEqual(report["improvements"], 0)

    def test_mutating_workspace_input_rejects_candidate_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            source = task_dir / "regression/participants/train.csv"
            original = source.read_bytes()
            code = EXPERIMENT.replace("OFFSET", "0") + "\nPath('regression/participants/train.csv').write_text('changed')"
            run_dir = self.run_quietly(task, FakeClient([response(code=code), response()]), 2)
            report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["best_metric"]["value"], 10)
            self.assertIn("changed", report["events"][-1]["result"]["selection"]["reason"])

    def test_source_mutation_after_good_experiment_cannot_finish_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            source = task_dir / "regression/participants/train.csv"
            code = f"from pathlib import Path; Path({str(source)!r}).write_text('changed')"
            client = FakeClient([response(code=EXPERIMENT.replace("OFFSET", "0")), response(code=code), response()])
            with self.assertRaisesRegex(ValueError, "Original task inputs changed"):
                self.run_quietly(task, client, 3)
            runs = list((task_dir / ".agent_runs").glob("*/run.json"))
            report = json.loads(runs[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["best_metric"]["value"], 0)
            self.assertEqual((task_dir / "submission.csv").read_text(encoding="utf-8"), "row_key,outcome\n0005,17\nNA,20\n")

    def test_unknown_metric_preserves_llm_route_but_marks_quality_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = make_regression(task_dir)
            task.write_text(task.read_text(encoding="utf-8").replace("Metric: RMSE", "Metric: SMAPE"))
            code = (
                "from pathlib import Path\n"
                "Path('submission.csv').write_text('row_key,outcome\\n0005,17\\nNA,20\\n')\n"
                "Path('research_hypothesis.json').write_text('{\"hypothesis\":\"A linear model\"}')\n"
            )
            run_dir = self.run_quietly(task, FakeClient([response(code=code), response()]), 2)
            report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((report["status"], report["reason"]), ("fallback", "host_metric_unsupported"))
            self.assertEqual(report["host_validation"]["status"], "unsupported")
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["unverified_experiments"], 1)
            self.assertIsNone(report["best_metric"])
            self.assertTrue(validate_predictions(task_dir)["valid"])


if __name__ == "__main__":
    unittest.main()
