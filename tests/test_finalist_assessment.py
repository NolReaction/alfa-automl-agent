"""Only feasible host-scored observations enter the optional finalist pool."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.research_candidate import freeze_candidate
from agent.research_finalists import FinalistPool
from agent.research_v3 import _assess_candidate


class FinalistAssessmentTests(unittest.TestCase):
    def assess(self, directory, recipe, *, estimate=10, stage=5):
        root = Path(directory)
        source = root / "model.py"
        source.write_text("def fit_predict(*args):\n    return {'predictions': []}\n", encoding="utf-8")
        baseline = freeze_candidate(source, {"family": "baseline"}, root / "baseline")
        candidate = freeze_candidate(source, recipe, root / "candidate")
        metric = {"name": "RMSE", "value": 2.0, "higher_is_better": False}
        baseline_metric = dict(metric, value=3.0)
        state = SimpleNamespace(
            gate=SimpleNamespace(check_inputs=Mock(), inner_split={}, research_spec=None,
                                 expected_inner_actuals=None),
            finalist_pool=FinalistPool(), tried_recipes=[], selected_candidate=None,
            best_inner=baseline_metric, report={"improvements": 0, "evaluations": {}},
            measure_inner=Mock(return_value=SimpleNamespace(metric=metric)),
            finalization_plan=Mock(return_value=SimpleNamespace(
                estimated_seconds=estimate, stage_seconds={"candidate_full": stage})),
            remaining=lambda: 100,
        )
        with patch("agent.research_v3.inner_feedback", return_value={}):
            feedback = _assess_candidate(state, candidate, "measured trial", origin="local")
        return state, baseline, baseline_metric, candidate, metric, feedback

    def test_arbitrary_recipe_family_cannot_reject_a_verified_feasible_model(self):
        for family, expected in (({"arbitrary": [1, 2]}, "custom"),
                                 ("x" * 500, "x" * 120), ("   ", "custom")):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                state, baseline, base_metric, candidate, metric, feedback = self.assess(
                    directory, {"family": family, "metric": {"value": -999}})
                self.assertTrue(feedback["accepted"])
                self.assertIs(state.selected_candidate, candidate)
                entries = state.finalist_pool.shortlist(baseline, base_metric, candidate, metric)
                self.assertEqual(entries[1]["family"], expected)
                self.assertEqual(entries[1]["metric"], metric)
                state.measure_inner.assert_called_once_with(candidate)
                state.gate.check_inputs.assert_called_once_with()

    def test_verified_but_unaffordable_observation_never_enters_pool(self):
        for estimate, stage in ((101, 5), (10, 1801)):
            with self.subTest(estimate=estimate, stage=stage), tempfile.TemporaryDirectory() as directory:
                state, _, _, _, _, feedback = self.assess(
                    directory, {"family": "trees"}, estimate=estimate, stage=stage)
                self.assertTrue(feedback["verified"])
                self.assertFalse(feedback["accepted"])
                self.assertEqual(len(state.finalist_pool), 0)
                self.assertIsNone(state.selected_candidate)


if __name__ == "__main__":
    unittest.main()
