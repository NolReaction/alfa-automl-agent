"""Research reports preserve score purpose without exposing host artifacts."""

from copy import deepcopy
import unittest

from app.research_summary import research_summary, validate_research_summary


def v3_report():
    metric = {"name": "R2", "value": -0.25, "higher_is_better": True}
    return {
        "research_protocol_version": 3, "stage": "outer_validation",
        "stages": [
            {"id": "inner_research", "status": "completed"},
            {"id": "outer_validation", "status": "running"},
        ],
        "evaluations": {
            "selection": {**metric, "scope": "inner_selection", "split": {
                "strategy": "time", "train_rows": 40, "validation_rows": 10,
                "fit_rows": [1, 2], "path": "/private/train.csv",
            }},
            "final_check": {"baseline": metric, "candidate": None, "selected": "baseline",
                            "selection_uses_outer": True, "unbiased_estimate": False},
        },
        "frozen_candidate": {"code_hash": "a" * 64, "recipe_hash": "b" * 64,
                             "path": "/private/candidate.py"},
    }


class ResearchSummaryTests(unittest.TestCase):
    def test_v3_keeps_negative_metric_and_score_origin_without_private_artifacts(self):
        source = v3_report()
        summary = research_summary(source)
        self.assertEqual(summary["evaluations"]["selection"]["metric"]["value"], -0.25)
        self.assertEqual(summary["evaluations"]["selection"]["scope"], "inner_selection")
        self.assertFalse(summary["evaluations"]["final_check"]["unbiased_estimate"])
        self.assertNotIn("/private", str(summary))
        self.assertNotIn("fit_rows", str(summary))
        self.assertEqual(validate_research_summary(summary), summary)
        self.assertEqual(source, v3_report())

    def test_old_reports_do_not_acquire_v3_provenance_from_extra_fields(self):
        old = v3_report()
        old.pop("research_protocol_version")
        self.assertEqual(research_summary(old), {"protocol_version": 2})
        self.assertEqual(research_summary({}), {"protocol_version": 2})

    def test_pending_v3_is_readable_and_wrong_scope_or_duplicate_stages_are_rejected(self):
        self.assertEqual(research_summary({"research_protocol_version": 3}), {
            "protocol_version": 3, "stages": [], "evaluations": {},
        })
        for mutation in (
            lambda item: item["evaluations"]["selection"].update(scope="outer"),
            lambda item: item["stages"].append(deepcopy(item["stages"][0])),
            lambda item: item["stages"][0].update(status=[]),
            lambda item: item["evaluations"]["selection"]["split"].update(strategy=[]),
            lambda item: item["evaluations"]["final_check"].update(selected=[]),
            lambda item: item["frozen_candidate"].update(code_hash="/private/code.py"),
            lambda item: item["evaluations"]["final_check"]["baseline"].update(value=float("nan")),
        ):
            item = v3_report()
            mutation(item)
            with self.assertRaises(ValueError):
                research_summary(item)


if __name__ == "__main__":
    unittest.main()
