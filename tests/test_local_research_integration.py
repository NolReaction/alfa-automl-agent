"""Real v3 local fallback: outer isolation and recovery after a failed fit."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.model_config import make_model_config
from agent.search import seed_candidate
from tests import test_team_integration as lifecycle_fixture
from tests.test_research import make_regression, write_csv
from validation import validate_predictions


class LocalResearchIntegrationTests(unittest.TestCase):
    def test_api_failure_local_research_is_unchanged_by_outer_answer_intervention(self):
        # Reuse the fixture runner without inheriting its complete test suite.
        runner = lifecycle_fixture.TeamIntegrationTests()
        split = {"fit_rows": list(range(80)), "validation_rows": list(range(80, 100)),
                 "choice": {"strategy": "random"}, "split": {"strategy": "random"}}
        evidence = []
        with tempfile.TemporaryDirectory() as directory:
            for name, change_outer in (("original", False), ("intervened", True)):
                task = make_regression(Path(directory) / name)
                if change_outer:
                    write_csv(task.parent / "regression/participants/train.csv",
                        [["row_key", "measurement", "outcome"]] + [
                            [f"train-{index:03d}", index,
                             910000 + index if index >= 80 else 3 * index + 2]
                            for index in range(100)])
                with patch("agent.research_v3.local_candidate_plan", return_value=[seed_candidate(False)]):
                    _, report, client, executions, views = runner.run_v3(
                        task, [ConnectionError("Offline fixture: API unavailable")], frozen_split=split)
                self.assertEqual(len(client.received_messages), 1)
                self.assertEqual(report["api"]["attempts"], 1)
                self.assertEqual(report["api"]["responses"], 0)
                self.assertEqual(report["successful_experiments"], 0)
                self.assertEqual(report["local_experiments"], 1)
                self.assertEqual(report["local_successful_experiments"], 1)
                self.assertEqual(report["local_research"]["verified"], 1)
                self.assertIsNotNone(report["frozen_candidate"])
                self.assertTrue(validate_predictions(task.parent)["valid"])
                first_outer = next(index for index, item in enumerate(executions)
                                   if item["phase"] == "outer_validation")
                self.assertTrue(all(item["phase"] != "inner_selection"
                                    for item in executions[first_outer:]))
                # Final outer metrics deliberately differ; research inputs,
                # inner choices and frozen code/recipes must remain identical.
                evidence.append((views, report["frozen_baseline"], report["frozen_candidate"],
                                 report["evaluations"]["selection"], report["local_research"]))
        self.assertEqual(evidence[0], evidence[1])

    def test_real_local_log_target_fit_failure_preserves_constant_baseline_publication(self):
        runner = lifecycle_fixture.TeamIntegrationTests()
        local_recipe = seed_candidate(False)
        local_recipe["model_config"] = make_model_config("linear", "regression", target_transform="log1p")
        local_recipe["label"] = "Failure injection: log1p requires nonnegative fit targets"
        with tempfile.TemporaryDirectory() as directory:
            task = make_regression(Path(directory))
            write_csv(task.parent / "regression/participants/train.csv",
                [["row_key", "measurement", "outcome"]] + [
                    [f"train-{index:03d}", index, -(3 * index + 2)] for index in range(100)])
            with patch("agent.research_v3.local_candidate_plan", return_value=[local_recipe]):
                run_dir, report, client, executions, _ = runner.run_v3(
                    task, [ConnectionError("Offline fixture: API unavailable")])
            self.assertEqual(len(client.received_messages), 1)
            self.assertEqual(report["api"]["attempts"], 1)
            self.assertEqual(report["api"]["responses"], 0)
            self.assertEqual(report["successful_experiments"], 0)
            self.assertEqual(report["local_experiments"], 1)
            self.assertEqual(report.get("local_successful_experiments", 0), 0)
            self.assertEqual(report["local_research"]["failed"], 1)
            self.assertEqual(report["local_research"]["verified"], 0)
            self.assertIsNone(report["frozen_candidate"])
            self.assertEqual(report["published_recipe"], "baseline")
            self.assertEqual(report["reason"], "api_error")
            self.assertTrue(validate_predictions(task.parent)["valid"])
            self.assertTrue(report["publication"]["new_result_published"])

            # Confirm a genuine log-target model-fit failure in the child,
            # rather than replacing fitting with a mocked exception.
            local_errors = []
            for context_path in (run_dir / "inner_evaluations").glob("inner_selection_*/context.json"):
                context = json.loads(context_path.read_text(encoding="utf-8"))
                if context["recipe"].get("model_config", {}).get("target_transform") == "log1p":
                    local_errors.extend(path.read_text(encoding="utf-8", errors="replace")
                                        for path in (context_path.parent / "logs").glob("*.stderr.txt"))
            self.assertTrue(local_errors)
            self.assertIn("targets must contain finite nonnegative", "\n".join(local_errors))

            # The full-data constant refit is the expected arithmetic mean.
            output = task.parent / "submission.csv"
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["row_key"] for row in rows], ["0005", "NA"])
            self.assertEqual([float(row["outcome"]) for row in rows], [-150.5, -150.5])
            baseline_refit = Path(report["candidate_receipts"]["refit"]).parent / "predictions.csv"
            self.assertEqual(output.read_bytes(), baseline_refit.read_bytes())
            baseline_hash = report["frozen_baseline"]["code_hash"]
            outer_and_refit = [item for item in executions
                               if item["phase"] in {"outer_validation", "full_refit"}]
            self.assertTrue(outer_and_refit)
            self.assertEqual({item["code_hash"] for item in outer_and_refit}, {baseline_hash})


if __name__ == "__main__":
    unittest.main()
