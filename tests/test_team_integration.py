"""Default v3 lifecycle across generated tools, host validation and publication."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import solution
from agent.automl import _atomic_copy
from agent.research import run_agent
from agent.research_candidate import freeze_candidate, run_candidate
from agent.research_workspace import ResearchWorkspace
from tests.test_research import FakeClient, make_regression, response, write_csv
from validation import validate_predictions


CONSTANT_CODE = """import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    return {'predictions': np.full(len(X_predict), np.asarray(y_fit, dtype=float).mean())}
"""

LINEAR_CODE = """import numpy as np
def fit_predict(X_fit, y_fit, X_predict, context):
    slope, intercept = np.polyfit(np.asarray(X_fit['measurement'], dtype=float),
                                 np.asarray(y_fit, dtype=float), 1)
    return {'predictions': slope * np.asarray(X_predict['measurement'], dtype=float) + intercept}
"""


def candidate_tool(code=LINEAR_CODE, recipe=None):
    recipe = recipe if recipe is not None else {"family": "linear", "degree": 1}
    return (
        "import json\nfrom pathlib import Path\n"
        f"Path('research_candidate.py').write_text({code!r}, encoding='utf-8')\n"
        f"Path('research_recipe.json').write_text({json.dumps(recipe)!r}, encoding='utf-8')\n"
        "Path('research_hypothesis.json').write_text(json.dumps("
        "{'hypothesis': 'Fit a linear relationship using only the inner fit rows.'}), encoding='utf-8')\n"
    )


def fixture_baseline(state):
    """Skip portfolio search while retaining real subprocess fitting/scoring."""
    code = state.run_dir / "baseline_source.py"
    code.write_text(CONSTANT_CODE, encoding="utf-8")
    state.baseline = freeze_candidate(code, {"family": "constant"},
                                      state.run_dir / "baseline_frozen")
    state.report["baseline"] = "trained"


class LifecycleClient(FakeClient):
    def __init__(self, replies, terminal):
        super().__init__(replies)
        self.terminal = terminal

    def create(self, **kwargs):
        if self.terminal["started"]:
            self.terminal["late_requests"] += 1
            raise AssertionError("The model was requested after external validation started")
        return super().create(**kwargs)


class TeamIntegrationTests(unittest.TestCase):
    def run_v3(self, task, replies, *, frozen_split=None, baseline=fixture_baseline,
               fail_final=False):
        terminal = {"started": False, "late_requests": 0}
        executions, views = [], []
        client = LifecycleClient(replies, terminal)

        def workspace(*args, **kwargs):
            if frozen_split is not None:
                kwargs["frozen_split"] = frozen_split
            gate = ResearchWorkspace(*args, **kwargs)
            views.append({name: (gate.workspace / name).read_bytes()
                          for name in gate.manifest["allowed_inputs"]})
            return gate

        def execute(candidate, **kwargs):
            if kwargs["phase"] in {"outer_validation", "final_check", "full_refit", "refit"}:
                terminal["started"] = True
                if fail_final:
                    raise ValueError("Fixture final evaluation could not complete")
            executions.append({"phase": kwargs["phase"], "code_hash": candidate.code_hash,
                               "recipe_hash": candidate.recipe_hash})
            return run_candidate(candidate, **kwargs)

        with patch.dict(os.environ, {"OPENAI_MODEL": "v3-fixture-model"}), \
                patch("agent.research_v3.prepare_baseline", side_effect=baseline), \
                patch("agent.research_v3.ResearchWorkspace", side_effect=workspace), \
                patch("agent.research_v3.run_candidate", side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_dir = run_agent(task, client, max_steps=len(replies), time_budget=90)
        self.assertEqual(terminal["late_requests"], 0)
        report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        return run_dir, report, client, executions, views

    def test_unsupported_metric_uses_explicit_legacy_route(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            task.write_text(task.read_text(encoding="utf-8").replace("Metric: RMSE", "Metric: SMAPE"),
                            encoding="utf-8")
            expected = Path(directory) / "legacy-report"
            client = object()
            with patch("agent.research.run_agent_v2", return_value=expected) as legacy:
                self.assertEqual(run_agent(task, client, max_steps=1, time_budget=30), expected)
            legacy.assert_called_once_with(task, client, max_steps=1, time_budget=30)

    def test_default_route_freezes_inner_winner_before_outer_and_refits_same_code(self):
        self.assertIs(solution.run_agent, run_agent)
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            run_dir, report, client, executions, _ = self.run_v3(
                task, [response(code=candidate_tool()), response()])
            self.assertEqual(report["research_protocol_version"], 3)
            self.assertEqual(report["evaluations"]["selection"]["scope"], "inner_selection")
            self.assertEqual(len(client.received_messages), 2)
            self.assertTrue(validate_predictions(task.parent)["valid"])
            final_check = report["evaluations"]["final_check"]
            self.assertTrue(final_check["selection_uses_outer"])
            self.assertFalse(final_check["unbiased_estimate"])
            frozen = report["frozen_candidate"]
            candidate_calls = [entry for entry in executions if entry["code_hash"] == frozen["code_hash"]]
            self.assertTrue(any(entry["phase"] in {"outer_validation", "final_check"}
                                for entry in candidate_calls))
            self.assertTrue(any(entry["phase"] in {"full_refit", "refit"} for entry in candidate_calls))
            self.assertEqual({entry["recipe_hash"] for entry in candidate_calls}, {frozen["recipe_hash"]})
            self.assertTrue((run_dir / "last_valid_submission.csv").is_file())
            values = np.genfromtxt(task.parent / "submission.csv", delimiter=",", skip_header=1,
                                   usecols=1)
            np.testing.assert_allclose(values, [17, 20], rtol=0, atol=1e-10)

    def test_outer_answer_intervention_preserves_research_view_and_frozen_recipe(self):
        split = {"fit_rows": list(range(80)), "validation_rows": list(range(80, 100)),
                 "choice": {"strategy": "random"},
                 "split": {"strategy": "random"}}
        evidence = []
        with tempfile.TemporaryDirectory() as directory:
            for label, intervention in (("original", False), ("intervened", True)):
                task = make_regression(Path(directory) / label)
                if intervention:
                    write_csv(task.parent / "regression/participants/train.csv",
                              [["row_key", "measurement", "outcome"]] + [
                                  [f"train-{index:03d}", index,
                                   910000 + index if index >= 80 else 3 * index + 2]
                                  for index in range(100)])
                result = self.run_v3(task, [response(code=candidate_tool()), response()],
                                     frozen_split=split)
                run_dir, report, client, _, views = result
                frozen = report["frozen_candidate"]
                self.assertTrue(frozen)
                self.assertNotIn("910080", json.dumps(client.received_messages))
                evidence.append((views, frozen, report["evaluations"]["selection"]))
        self.assertEqual(evidence[0], evidence[1])

    def test_failed_candidate_keeps_baseline_and_does_not_publish_generated_csv(self):
        code = candidate_tool("def fit_predict(*args):\n    raise ValueError('bad recipe')\n")
        code += "Path('submission.csv').write_text('row_key,outcome\\n0005,999\\nNA,999\\n')\n"
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            _, report, _, _, _ = self.run_v3(task, [response(code=code), response()])
            self.assertTrue(validate_predictions(task.parent)["valid"])
            result = (task.parent / "submission.csv").read_text(encoding="utf-8")
            self.assertNotIn(",999", result)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["improvements"], 0)

    def test_api_failure_stops_requests_and_keeps_real_host_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            _, report, client, _, _ = self.run_v3(
                task, [ConnectionError("fixture failure"), response(code=candidate_tool())])
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["api"]["attempts"], 1)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertTrue(validate_predictions(task.parent)["valid"])
            self.assertEqual(report["status"], "fallback")

    def test_malformed_reply_without_verified_final_output_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with self.assertRaises((RuntimeError, ValueError)):
                self.run_v3(task, [SimpleNamespace(choices=[])], fail_final=True)
            self.assertFalse((task.parent / "submission.csv").exists())
            reports = list((task.parent / ".agent_runs").glob("*/run.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["api"]["attempts"], 1)

    def test_selected_candidate_refit_failure_keeps_verified_baseline_bytes(self):
        # A diagnostic failure injection, not a suggested model recipe: the
        # exact same code must be frozen for inner, outer and failing refit.
        code = LINEAR_CODE.replace(
            "    slope, intercept", "    if context['phase'] == 'full_refit':\n"
            "        raise RuntimeError('fixture refit failure')\n    slope, intercept")
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            run_dir, report, _, executions, _ = self.run_v3(
                task, [response(code=candidate_tool(code)), response()])
            self.assertEqual(report["evaluations"]["final_check"]["selected"], "candidate")
            self.assertEqual(report["published_recipe"], "baseline")
            self.assertEqual((report["status"], report["reason"]), ("fallback", "refit_failed"))
            values = np.genfromtxt(task.parent / "submission.csv", delimiter=",", skip_header=1,
                                   usecols=1)
            np.testing.assert_allclose(values, [150.5, 150.5], rtol=0, atol=1e-10)
            full = [entry["code_hash"] for entry in executions if entry["phase"] == "full_refit"]
            self.assertEqual(full, [report["frozen_baseline"]["code_hash"],
                                    report["frozen_candidate"]["code_hash"]])
            self.assertEqual(report["best_metric"], report["evaluations"]["final_check"]["baseline"])
            baseline_outputs = list((run_dir / "full_refit" / "baseline").glob("*/predictions.csv"))
            self.assertEqual(len(baseline_outputs), 1)
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             baseline_outputs[0].read_bytes())

    def test_optional_evidence_export_failure_keeps_published_candidate_metadata(self):
        failed_exports = []

        def copy(source, destination):
            if Path(destination).name == "best_holdout_predictions.csv":
                failed_exports.append(Path(destination))
                raise OSError("Fixture convenience evidence export failure")
            return _atomic_copy(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3._atomic_copy", side_effect=copy):
                run_dir, report, _, _, _ = self.run_v3(
                    task, [response(code=candidate_tool()), response()])
            self.assertTrue(failed_exports)
            self.assertFalse((run_dir / "best_holdout_predictions.csv").exists())
            self.assertEqual((report["status"], report["reason"]), ("completed", "model_finished"))
            self.assertEqual(report["published_recipe"], "candidate")
            self.assertEqual(report["best_metric"], report["evaluations"]["final_check"]["candidate"])
            receipts = report["candidate_receipts"]
            self.assertEqual(receipts["code_hash"], report["frozen_candidate"]["code_hash"])
            self.assertEqual(receipts["recipe_hash"], report["frozen_candidate"]["recipe_hash"])
            refit = json.loads(Path(receipts["refit"]).read_text(encoding="utf-8"))
            self.assertEqual(refit["code_hash"], receipts["code_hash"])
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             (Path(receipts["refit"]).parent / "predictions.csv").read_bytes())
            provenance = report["best_metric_provenance"]
            self.assertEqual(provenance["scope"], "final_check")
            self.assertTrue(provenance["selection_uses_outer"])
            self.assertFalse(provenance["unbiased_estimate"])
            values = np.genfromtxt(task.parent / "submission.csv", delimiter=",", skip_header=1,
                                   usecols=1)
            np.testing.assert_allclose(values, [17, 20], rtol=0, atol=1e-10)

    def test_unknown_usage_and_malformed_paid_reply_have_distinct_accounting(self):
        valid = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        for paid_malformed in (False, True):
            with self.subTest(paid_malformed=paid_malformed), tempfile.TemporaryDirectory() as directory:
                task = make_regression(Path(directory))
                reply = (SimpleNamespace(choices=[], usage=SimpleNamespace(**valid))
                         if paid_malformed else response())
                _, report, client, _, _ = self.run_v3(task, [reply])
                self.assertEqual(len(client.received_messages), 1)
                self.assertEqual(report["api"]["responses"], 1)
                self.assertEqual(report["api"]["usage_reported_responses"], int(paid_malformed))
                for field, value in valid.items():
                    self.assertEqual(report["api"][field], value if paid_malformed else 0)
                self.assertEqual(report["reason"], "api_error" if paid_malformed else "step_limit")
                self.assertTrue(validate_predictions(task.parent)["valid"])

    def test_three_failed_candidate_executions_end_research_and_keep_host_fallback(self):
        replies = [response(code=candidate_tool(
            f"def fit_predict(*args):\n    raise RuntimeError('fixture failure {index}')\n"))
                   for index in range(3)]
        replies.append(response(code=candidate_tool()))
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            _, report, client, _, _ = self.run_v3(task, replies)
            self.assertEqual(len(client.received_messages), 3)
            self.assertEqual(report["api"]["attempts"], 3)
            self.assertEqual((report["status"], report["reason"]), ("fallback", "repeated_failures"))
            self.assertEqual(report["successful_experiments"], 0)
            self.assertTrue(validate_predictions(task.parent)["valid"])


if __name__ == "__main__":
    unittest.main()
