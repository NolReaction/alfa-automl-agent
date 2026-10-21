"""Early dialogue stops still permit bounded, host-scored inner research."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from agent.model_config import make_model_config
from agent.research_v3 import (
    V3State, _assess_candidate, _request_completion, continue_locally,
    prepare_baseline, prepare_research, run_agent_v3,
)
from agent.search import seed_candidate
from tests import test_team_integration as lifecycle
from tests.test_research import FakeClient, make_regression, response, write_csv
from tests.test_team_integration import (
    LINEAR_CODE, candidate_tool, fixture_baseline,
)
from validation import validate_predictions


def truncated_response():
    reply = response(content="Incomplete recipe must never be executed")
    reply.choices[0].finish_reason = "length"
    return reply


class ResearchContinuationTests(unittest.TestCase):
    def test_strong_startup_receives_profile_and_larger_cap_inside_the_same_budget(self):
        for profile, cap in (("legacy", 600.0), ("strong_v1", 1800.0)):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                task = make_regression(Path(directory))
                state = V3State(task, 1, 10500)
                state.report["quality_options"] = {"search_profile": profile}
                prepare_research(state)
                deadline = state.budget.deadline
                observed = []

                def execute(code, *args, **kwargs):
                    observed.append((code, kwargs["timeout"]))
                    return {"exit_code": 1, "stderr": "Injected startup failure"}

                with patch("agent.execution.run_python", side_effect=execute):
                    prepare_baseline(state)
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0][1], cap)
                self.assertLessEqual(cap, (state.remaining() - state.final_reserve) * .40)
                if profile == "strong_v1":
                    self.assertIn("search_profile='strong_v1'", observed[0][0])
                else:
                    self.assertNotIn("search_profile=", observed[0][0])
                self.assertEqual(state.budget.deadline, deadline)
                self.assertIsNotNone(state.baseline)

    def test_invalid_profile_is_rejected_before_creating_run_or_reading_inputs(self):
        for value in ("unknown", [], None, 1):
            with self.subTest(profile=value), self.assertRaisesRegex(ValueError, "search profile"):
                run_agent_v3(Path("nonexistent-task.md"), object(), search_profile=value)

    def test_strong_continuation_preserves_local_refinements_and_uses_only_inner_fit_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            state = V3State(task, 1, 90)
            state.report["quality_options"] = {"search_profile": "strong_v1"}
            prepare_research(state)
            fixture_baseline(state)
            state.transition("inner_research")
            strong_recipe, legacy_recipe = seed_candidate(False), seed_candidate(False)
            strong_recipe["id"] = "strong-test"
            legacy_recipe["id"] = "local-refinement-test"
            strong = Mock(return_value=[strong_recipe])
            module = SimpleNamespace(strong_candidate_plan=strong)
            with patch.dict("sys.modules", {"agent.tabular_backend": module}), \
                    patch("agent.research_v3.local_candidate_plan", return_value=[legacy_recipe]), \
                    patch("agent.offline_research.run_local_continuation",
                          return_value={"status": "completed"}) as local:
                continue_locally(state)
            strong.assert_called_once()
            self.assertEqual(strong.call_args.args[:2], (False, "RMSE"))
            expected = state.gate.research_train.iloc[state.gate.fit_rows][state.gate.spec.target_column]
            self.assertEqual(strong.call_args.args[2].tolist(), expected.tolist())
            self.assertEqual(local.call_args.args[1], [strong_recipe, legacy_recipe])
            self.assertEqual(state.report["api"]["attempts"], 0)

    def test_guard_and_response_stops_train_locally_without_further_paid_requests(self):
        cases = (
            ("step_limit", [response()], 1),
            ("truncated_response", [truncated_response()], 1),
            ("context_budget", [response()], 0),
            ("repeated_failures", [response(code=f"raise ValueError('injected {index}')")
                                   for index in range(3)] + [response()], 3),
        )
        for reason, replies, requests in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                task = make_regression(Path(directory))
                initial = []

                def baseline(state):
                    initial.append((state, state.budget.deadline))
                    fixture_baseline(state)

                # Exercise the actual UTF-8 payload guard, not a mocked stop.
                context = (patch("agent.research_v3._context", return_value="x" * 70000)
                           if reason == "context_budget" else nullcontext())
                with context, patch("agent.research_v3.local_candidate_plan",
                                    return_value=[seed_candidate(False)]):
                    run_dir, report, client, executions, _ = lifecycle.TeamIntegrationTests().run_v3(
                        task, replies, baseline=baseline)
                state, deadline = initial[0]
                self.assertEqual(state.budget.deadline, deadline)
                self.assertEqual(state.deadline, deadline)
                self.assertEqual(report["time_budget_seconds"], 90)
                self.assertEqual(report["research_stop"], {
                    "reason": reason, "local_continuation_requested": True,
                })
                self.assertEqual((report["status"], report["reason"]), ("fallback", reason))
                self.assertEqual(len(client.received_messages), requests)
                self.assertEqual(report["api"]["attempts"], requests)
                self.assertEqual(report["api"]["responses"], requests)
                self.assertEqual(report["successful_experiments"], 0)
                self.assertEqual(report["local_experiments"], 1)
                self.assertEqual(report["local_successful_experiments"], 1)
                self.assertEqual(report["local_research"]["verified"], 1)
                self.assertEqual(report["local_research"]["api_requests"], 0)
                self.assertEqual(report["published_recipe"], "candidate")
                self.assertTrue(report["publication"]["new_result_published"])
                self.assertTrue(validate_predictions(task.parent)["valid"])
                self.assertEqual((task.parent / "submission.csv").read_bytes(),
                                 (run_dir / "last_valid_submission.csv").read_bytes())
                first_outer = next(index for index, item in enumerate(executions)
                                   if item["phase"] == "outer_validation")
                self.assertTrue(all(item["phase"] != "inner_selection"
                                    for item in executions[first_outer:]))

    def test_step_limit_keeps_better_measured_llm_recipe_when_local_trial_is_worse(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.local_candidate_plan", return_value=[seed_candidate(False)]):
                _, report, client, _, _ = lifecycle.TeamIntegrationTests().run_v3(
                    task, [response(code=candidate_tool())])
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["reason"], "step_limit")
            self.assertEqual(report["successful_experiments"], 1)
            self.assertEqual(report["local_research"]["verified"], 1)
            self.assertEqual(report["local_research"]["improvements"], 0)
            self.assertEqual(report["frozen_candidate"]["code_hash"],
                             hashlib.sha256(LINEAR_CODE.encode("utf-8")).hexdigest())
            values = np.genfromtxt(task.parent / "submission.csv", delimiter=",", skip_header=1,
                                   usecols=1)
            np.testing.assert_allclose(values, [17, 20], rtol=0, atol=1e-10)

    def test_truncated_response_and_failed_local_fit_still_publish_verified_baseline_bytes(self):
        local_recipe = seed_candidate(False)
        local_recipe["model_config"] = make_model_config("linear", "regression", target_transform="log1p")
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            write_csv(task.parent / "regression/participants/train.csv",
                      [["row_key", "measurement", "outcome"]] + [
                          [f"train-{index:03d}", index, -(3 * index + 2)] for index in range(100)])
            with patch("agent.research_v3.local_candidate_plan", return_value=[local_recipe]):
                _, report, client, _, _ = lifecycle.TeamIntegrationTests().run_v3(task, [truncated_response()])
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["reason"], "truncated_response")
            self.assertEqual(report["local_research"]["failed"], 1)
            self.assertEqual(report["local_research"]["verified"], 0)
            self.assertIsNone(report["frozen_candidate"])
            self.assertEqual(report["published_recipe"], "baseline")
            receipt = Path(report["candidate_receipts"]["refit"])
            self.assertEqual((task.parent / "submission.csv").read_bytes(),
                             (receipt.parent / "predictions.csv").read_bytes())
            self.assertTrue(validate_predictions(task.parent)["valid"])

    def test_original_input_mutation_during_new_continuation_cannot_publish(self):
        def mutate(state, candidate, hypothesis, *, origin):
            feedback = _assess_candidate(state, candidate, hypothesis, origin=origin)
            (state.gate.source_dir / "train.csv").write_text("changed source", encoding="utf-8")
            return feedback

        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.prepare_baseline", side_effect=fixture_baseline), \
                    patch("agent.research_v3.local_candidate_plan", return_value=[seed_candidate(False)]), \
                    patch("agent.research_v3._assess_candidate", side_effect=mutate), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Original task inputs changed"):
                    run_agent_v3(task, FakeClient([truncated_response()]), max_steps=1, time_budget=90)
            report_path, = (task.parent / ".agent_runs").glob("*/run.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["research_stop"]["reason"], "truncated_response")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["recovery"]["status"], "rejected")
            self.assertFalse((task.parent / "submission.csv").exists())

    def test_truncation_does_not_spend_the_original_finalization_reserve(self):
        observed = []

        def reserve_only(state, client, messages):
            result = _request_completion(state, client, messages)
            offset = (state.budget.deadline - state.budget.minimum_final_reserve
                      - time.monotonic() + .1)
            state.budget._clock = lambda: time.monotonic() + offset
            observed.append((state, state.budget.deadline))
            return result

        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3._request_completion", side_effect=reserve_only), \
                    patch("agent.research_v3.local_candidate_plan", return_value=[seed_candidate(False)]):
                _, report, client, _, _ = lifecycle.TeamIntegrationTests().run_v3(task, [truncated_response()])
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["reason"], "truncated_response")
            self.assertEqual(report["local_research"]["reason"], "finalization_reserve")
            self.assertEqual(report["local_research"]["attempted"], 0)
            self.assertEqual(report["published_recipe"], "baseline")
            self.assertEqual(observed[0][0].budget.deadline, observed[0][1])
            self.assertTrue(validate_predictions(task.parent)["valid"])

    def test_time_limit_goes_directly_to_finalization_without_api_or_local_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            with patch("agent.research_v3.V3State.request_allowed", return_value="time_limit"), \
                    patch("agent.research_v3.continue_locally") as local:
                _, report, client, _, _ = lifecycle.TeamIntegrationTests().run_v3(task, [response()])
            local.assert_not_called()
            self.assertEqual(client.received_messages, [])
            self.assertEqual(report["research_stop"], {
                "reason": "time_limit", "local_continuation_requested": False,
            })
            self.assertEqual(report["reason"], "time_limit")
            self.assertTrue(validate_predictions(task.parent)["valid"])


if __name__ == "__main__":
    unittest.main()
