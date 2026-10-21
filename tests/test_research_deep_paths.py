"""Exercise the production research lifecycle in nested Windows-sized paths."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_agent import benchmark


class DeepPathResearchTests(unittest.TestCase):
    def test_api_failure_still_publishes_verified_full_refit_in_deep_directory(self):
        # This is deliberately the real solver, not the fake checkout used by
        # benchmark contract unit tests. Its nested invocation directories cross
        # CreateProcessW's cwd limit even when the initial task path is short.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            while len(str(output)) < 180:
                output /= "research путь with spaces"
            output /= "benchmark"
            with redirect_stdout(io.StringIO()):
                report = benchmark(
                    output, cases=("nonlinear_regression",), seeds=(42,),
                    modes=("research_api_failure",), time_budget=180,
                    solver_options={"metric_recipes": True},
                )
            trial = report["trials"][0]
            details = json.dumps({
                "status": trial["status"],
                "execution": trial["execution"],
                "contract_errors": trial.get("research_contract_errors"),
            }, ensure_ascii=False, default=str)
            self.assertEqual(report["status"], "completed", details)
            self.assertEqual(trial["status"], "completed", details)
            self.assertEqual(trial.get("research_contract_errors"), [], details)
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertEqual(report["simulation"]["external_api_calls"], 0)
            self.assertEqual(report["simulation"]["recorded_simulated_api_calls"], 2)
            summary = trial["solver_report"]
            self.assertTrue(summary["receipt_checks_valid"], details)
            self.assertTrue(summary["publication"]["new_result_published"], details)
            self.assertGreaterEqual(summary["local_research"]["verified"], 1, details)
            receipt = summary["benchmark_evidence"]["run_report"]
            run_dir = Path(trial["task"]).parent / Path(receipt).parent
            invocation_dirs = list((run_dir / "outer_evaluations").glob("*/*"))
            self.assertTrue(any(len(str(path)) > 260 for path in invocation_dirs),
                            f"Test did not reach a long invocation cwd: {invocation_dirs}")


if __name__ == "__main__":
    unittest.main()
