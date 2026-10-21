"""Strong search keeps host validation, bounded workers and accepted output."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import automl
from agent.research import run_agent
from tests import test_automl as fixtures
from tests import test_solution_cli as cli_fixtures


class StrongSearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AutoMLTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_strong_plan_is_measured_on_inner_rows_before_bounded_outer_check(self):
        result = self.fixture.run_search(search_profile="strong_v1")
        calls = self.fixture.calls
        seed = next(call for call in calls if call["phase"] == "seed")
        trials = [call for call in calls if call["phase"] == "selection"]
        self.assertEqual(len(trials), 8)
        self.assertTrue(all(call["id"].startswith("strong_") for call in trials))
        self.assertTrue(any(call["model_config"]["parameters"].get("n_estimators", 0) >= 1000
                            for call in trials))
        for trial in trials:
            self.assertFalse(set(trial["fit_rows"]) & set(trial["validation_rows"]))
            self.assertFalse(set(trial["validation_rows"]) & set(seed["validation_rows"]))
            self.assertTrue(set(trial["fit_rows"]) | set(trial["validation_rows"])
                            <= set(seed["fit_rows"]))
        self.assertLessEqual(sum(call["phase"] == "verification" for call in calls), 3)
        self.assertEqual(result["search_policy"]["search_profile"], "strong_v1")
        self.assertEqual(result["search"]["best_candidate"], "seed")

    def test_failed_strong_workers_preserve_the_full_data_seed_checkpoint(self):
        result = self.fixture.run_search({"selection": {"exit_code": 1}}, search_profile="strong_v1")
        self.assertTrue(result["output_valid"])
        self.assertEqual(result["search"]["best_candidate"], "seed")
        self.assertTrue((self.fixture.root / "submission.csv").is_file())

    def test_unknown_search_profile_is_rejected_before_reading_data(self):
        for value in ("all_models", "", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "search_profile"):
                automl.run_automl(Path("missing-task.md"), search_profile=value)

    def test_research_router_forwards_strong_profile_to_v3(self):
        with patch("agent.research_v3.run_agent_v3") as research:
            run_agent(self.fixture.task, object(), search_profile="strong_v1")
        self.assertEqual(research.call_args.kwargs["search_profile"], "strong_v1")


class QualityCliTests(unittest.TestCase):
    def test_explicit_short_budget_and_profile_reach_the_offline_solver(self):
        # Reuse the CLI driver, without inheriting and rerunning its full suite.
        driver = cli_fixtures.ModeTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.md"
            task.write_text("Metric: RMSE", encoding="utf-8")
            result = {"model": "seed", "metric": {"name": "RMSE", "value": 1.0},
                      "report_path": str(Path(directory) / "report.json")}
            with patch("agent.automl.run_automl", return_value=result) as offline:
                status, _, errors = driver.invoke([str(task), "--mode", "offline",
                    "--time-budget", "90", "--search-profile", "strong_v1"])
            self.assertEqual(status, 0, errors)
            driver.assert_budgeted_call(offline, task, budget=90, validation={"strategy": "auto"},
                                        search_profile="strong_v1")


if __name__ == "__main__":
    unittest.main()
