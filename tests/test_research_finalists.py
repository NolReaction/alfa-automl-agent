"""Shortlists retain only bounded, immutable canonical INNER observations."""

from pathlib import Path
import tempfile
import unittest

from agent.research_candidate import freeze_candidate
from agent.research_finalists import FinalistPool, MAX_POOL_ENTRIES


def score(value, name="RMSE", higher=False):
    return {"name": name, "value": value, "higher_is_better": higher}


class ResearchFinalistsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.py"
        self.source.write_text("def fit_predict(*args):\n    return {'predictions': []}\n", encoding="utf-8")
        self.serial = 0

    def candidate(self, family="trees", *, recipe=None):
        self.serial += 1
        if recipe is None:
            recipe = {"action": family, "variant": self.serial}
        return freeze_candidate(self.source, recipe, self.root / str(self.serial))

    def test_baseline_and_incumbent_precede_the_strongest_distinct_runner_up(self):
        baseline, selected = self.candidate(), self.candidate()
        weaker, strongest = self.candidate("linear"), self.candidate()
        pool = FinalistPool()
        for candidate, value in ((baseline, 5.0), (selected, 1.0), (weaker, 3.0), (strongest, 2.0)):
            pool.record(candidate, score(value))
        result = pool.shortlist(baseline, score(5.0), selected, score(1.0))
        self.assertEqual([entry["id"] for entry in result], ["baseline", "candidate", "alternative"])
        self.assertEqual([entry["candidate"] for entry in result], [baseline, selected, strongest])
        self.assertEqual(result[-1]["metric"], score(2.0))

    def test_higher_is_better_uses_the_same_registered_metric_orientation(self):
        baseline, selected, first, second = [self.candidate() for _ in range(4)]
        pool = FinalistPool()
        pool.record(first, score(.8, "Accuracy", True))
        pool.record(second, score(.85, "Accuracy", True))
        result = pool.shortlist(baseline, score(.7, "Accuracy", True), selected, score(.9, "Accuracy", True))
        self.assertIs(result[-1]["candidate"], second)

    def test_equal_score_prefers_a_distinct_family_but_never_a_worse_score(self):
        baseline, selected = self.candidate(), self.candidate()
        same_family, diverse = self.candidate(), self.candidate("linear")
        for same_score, expected in ((2.0, diverse), (1.5, same_family)):
            with self.subTest(same_score=same_score):
                pool = FinalistPool()
                pool.record(same_family, score(same_score))
                pool.record(diverse, score(2.0))
                result = pool.shortlist(baseline, score(4.0), selected, score(1.0))
                self.assertIs(result[-1]["candidate"], expected)

    def test_identity_breaks_equal_score_family_ties_independently_of_arrival_order(self):
        baseline, selected = self.candidate(), self.candidate()
        candidates = [self.candidate("linear") for _ in range(4)]
        expected = min(candidates, key=lambda candidate: candidate.code_hash + ":" + candidate.recipe_hash)
        for order in (candidates, list(reversed(candidates))):
            pool = FinalistPool()
            for candidate in order:
                pool.record(candidate, score(2.0))
            result = pool.shortlist(baseline, score(5.0), selected, score(1.0))
            self.assertIs(result[-1]["candidate"], expected)

    def test_one_hundred_trials_remain_bounded_and_eviction_deletes_no_files(self):
        pool = FinalistPool()
        baseline, incumbent = self.candidate(), self.candidate()
        pool.record(baseline, score(1000.0))
        pool.record(incumbent, score(500.0))
        strongest = None
        for value in reversed(range(120)):
            strongest = self.candidate()
            pool.record(strongest, score(float(value)))
            self.assertLessEqual(len(pool), MAX_POOL_ENTRIES)
        self.assertEqual(len(pool), MAX_POOL_ENTRIES)
        self.assertTrue(baseline.code_path.is_file())
        self.assertTrue(incumbent.recipe_path.is_file())
        result = pool.shortlist(baseline, score(1000.0), incumbent, score(500.0))
        self.assertEqual([entry["candidate"] for entry in result], [baseline, incumbent, strongest])

    def test_equal_cutoff_scores_keep_a_diverse_family_when_possible(self):
        baseline, selected = self.candidate(), self.candidate()
        trees = [self.candidate() for _ in range(MAX_POOL_ENTRIES + 2)]
        linear = self.candidate("linear")
        for order in ([*trees, linear], [linear, *reversed(trees)]):
            pool = FinalistPool()
            for candidate in order:
                pool.record(candidate, score(2.0))
            result = pool.shortlist(baseline, score(4.0), selected, score(1.0))
            self.assertIs(result[-1]["candidate"], linear)
            self.assertEqual(len(pool), MAX_POOL_ENTRIES)

    def test_same_frozen_identity_is_deduplicated_across_different_paths(self):
        baseline = self.candidate()
        same = self.candidate(recipe=baseline.recipe)
        alternative = self.candidate()
        pool = FinalistPool()
        self.assertTrue(pool.record(baseline, score(1.0)))
        self.assertFalse(pool.record(same, score(1.0)))
        pool.record(alternative, score(2.0))
        self.assertEqual(len(pool), 2)
        result = pool.shortlist(baseline, score(1.0), same, score(1.0))
        self.assertEqual([entry["id"] for entry in result], ["baseline", "alternative"])
        self.assertEqual([entry["candidate"] for entry in result], [baseline, alternative])

    def test_canonical_score_cannot_be_replaced_by_another_split_score(self):
        candidate = self.candidate()
        pool = FinalistPool()
        pool.record(candidate, score(2.0))
        with self.assertRaisesRegex(ValueError, "canonical score cannot be replaced"):
            pool.record(candidate, score(1.0))
        with self.assertRaisesRegex(ValueError, "canonical score"):
            pool.shortlist(candidate, score(1.0))
        self.assertEqual(pool.shortlist(candidate, score(2.0))[0]["metric"], score(2.0))

    def test_input_and_returned_metrics_are_independent_copies(self):
        baseline, alternative = self.candidate(), self.candidate()
        original = score(2)
        pool = FinalistPool()
        pool.record(alternative, original)
        original["value"] = 999
        result = pool.shortlist(baseline, score(1))
        self.assertIs(result[-1]["candidate"], alternative)
        self.assertEqual(result[-1]["metric"], score(2))
        self.assertIs(type(result[-1]["metric"]["value"]), int)
        result[-1]["metric"]["value"] = -10
        self.assertEqual(pool.shortlist(baseline, score(1))[-1]["metric"], score(2))

    def test_unmeasured_or_noncanonical_metrics_cannot_enter_the_pool(self):
        invalid = [None, {}, score(float("nan")), score(float("inf")), score(float("-inf")),
                   score(True), score("2"), score(2, "rmse"), score(2, "unknown"),
                   score(2, higher=True), score(2, higher=0),
                   {**score(2), "scope": "outer_validation"}, score(10 ** 1000)]
        candidate = self.candidate()
        for metric in invalid:
            with self.subTest(metric=repr(metric)[:100]):
                pool = FinalistPool()
                with self.assertRaises(ValueError):
                    pool.record(candidate, metric)
                self.assertEqual(len(pool), 0)

    def test_a_pool_or_shortlist_cannot_mix_registered_metrics(self):
        baseline, alternative = self.candidate(), self.candidate()
        pool = FinalistPool()
        pool.record(alternative, score(2))
        with self.assertRaisesRegex(ValueError, "same canonical metric"):
            pool.record(baseline, score(3, "MAE"))
        with self.assertRaisesRegex(ValueError, "same canonical metric"):
            pool.shortlist(baseline, score(3, "MAE"))
        with self.assertRaisesRegex(ValueError, "same canonical metric"):
            FinalistPool().shortlist(baseline, score(3), alternative, score(2, "MAE"))

    def test_missing_measurements_use_no_shortlist_instead_of_inventing_scores(self):
        pool = FinalistPool()
        baseline = self.candidate()
        self.assertEqual(pool.shortlist(None, None), [])
        for arguments in ((baseline, None), (None, score(1)),
                          (None, None, baseline, score(1)), (baseline, score(1), None, score(2))):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    pool.shortlist(*arguments)

    def test_two_slot_mode_keeps_baseline_and_incumbent_without_a_third_model(self):
        baseline, selected, alternative = [self.candidate() for _ in range(3)]
        pool = FinalistPool(max_candidates=2)
        pool.record(alternative, score(2))
        result = pool.shortlist(baseline, score(3), selected, score(1))
        self.assertEqual([entry["candidate"] for entry in result], [baseline, selected])
        for value in (None, True, False, 0, 1, 4, 3.0, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FinalistPool(max_candidates=value)

    def test_explicit_family_and_recipe_family_are_validated(self):
        candidate = self.candidate(recipe={"family": "native", "action": "catboost"})
        pool = FinalistPool()
        pool.record(candidate, score(1))
        self.assertEqual(pool.shortlist(candidate, score(1))[0]["family"], "native")
        for family in ("", " ", " padded", "x" * 121, 1, True):
            with self.subTest(family=family), self.assertRaises(ValueError):
                FinalistPool().record(candidate, score(1), family=family)
        with self.assertRaisesRegex(ValueError, "family cannot be replaced"):
            pool.record(candidate, score(1), family="changed")

    def test_a_mutated_frozen_source_is_rejected_before_recording(self):
        candidate = self.candidate()
        candidate.code_path.write_text("raise RuntimeError('changed')\n", encoding="utf-8")
        pool = FinalistPool()
        with self.assertRaisesRegex(ValueError, "Frozen candidate code or recipe changed"):
            pool.record(candidate, score(1))
        self.assertEqual(len(pool), 0)

    def test_a_retained_recipe_mutation_fails_reads_even_if_not_the_runner_up(self):
        baseline, selected, best, other = [self.candidate() for _ in range(4)]
        pool = FinalistPool()
        pool.record(best, score(2))
        pool.record(other, score(3))
        other.recipe_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen candidate code or recipe changed"):
            pool.shortlist(baseline, score(4), selected, score(1))
        with self.assertRaisesRegex(ValueError, "Frozen candidate code or recipe changed"):
            pool.record(self.candidate(), score(.5))

    def test_external_incumbent_integrity_is_verified_after_pool_eviction(self):
        baseline, selected = self.candidate(), self.candidate()
        pool = FinalistPool()
        for value in range(MAX_POOL_ENTRIES):
            pool.record(self.candidate(), score(float(value)))
        selected.code_path.write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen candidate code or recipe changed"):
            pool.shortlist(baseline, score(100), selected, score(50))


if __name__ == "__main__":
    unittest.main()
