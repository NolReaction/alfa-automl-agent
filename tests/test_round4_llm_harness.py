"""Two real-host adapter risks; the five comparison pairs run only via the CLI."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_round4_llm as experiment


class Round4HarnessTests(unittest.TestCase):
    def test_useful_candidate_has_real_fit_receipts_and_guarded_payload(self):
        with tempfile.TemporaryDirectory(prefix="round4-useful-") as temporary:
            directory = Path(temporary) / "arm"
            row = experiment.run_arm(directory, "useful", "round4_context")
            self.assertIsNone(row["failure"])
            self.assertEqual(row["request_count"], 2)
            self.assertTrue(row["guards_match_transmitted_payloads"])
            self.assertTrue(row["inputs_unchanged"])
            self.assertEqual(row["network_requests"], 0)
            self.assertEqual(row["verified_inner_trials"], 1)
            self.assertEqual(row["unique_verified_inner_trials"], 1)
            self.assertEqual(row["candidate_physical_fits"], 3)
            self.assertEqual(row["physical_fits"], 6)
            self.assertEqual(row["candidate_invocations"], 3)
            self.assertIsNotNone(row["time_to_first_verified_inner_seconds"])
            self.assertTrue(row["smoke"]["coverage"]["generated_candidate_published"])
            self.assertEqual(row["smoke"]["coverage"]["published_output_origin"], "llm")
            self.assertTrue(row["smoke"]["coverage"]["qwen_full_refit_verified"])
            self.assertFalse(row["helper_active_in_production"])
            self.assertIsNone(row["provider_usage"])
            self.assertEqual(row["usage_reason"], "mock_responses_without_usage")
            self.assertEqual(row["smoke"]["api"]["responses_without_usage"], 2)
            self.assertLess(row["estimated_tokens_reserved"], row["estimated_token_limit"])
            self.assertGreater(row["transport_estimated_tokens"], row["estimated_tokens_reserved"])
            identity = row["trials"][0]
            executed = [item for item in row["executions"]
                        if item["code_hash"] == identity["code_hash"]
                        and item["recipe_hash"] == identity["recipe_hash"]]
            self.assertEqual({item["phase"] for item in executed},
                             {"inner_selection", "outer_validation", "full_refit"})
            public = (directory / "public_summary.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(public), row)
            for private in (temporary, experiment.FIXTURE_KEY, experiment.FIT_MARKER,
                            "def fit_predict", "research_hypothesis.json", "fixture-call-"):
                self.assertNotIn(private, public)

    def test_invalid_scope_never_sends_and_finalizes_native_baseline(self):
        original_snapshot = experiment._snapshot

        def wrong_scope(state):
            snapshot = original_snapshot(state)
            snapshot["scope"] = "outer_validation"
            return snapshot

        with tempfile.TemporaryDirectory(prefix="round4-scope-") as temporary:
            with patch.object(experiment, "_snapshot", side_effect=wrong_scope):
                row = experiment.run_arm(Path(temporary) / "arm", "useful", "round4_context")
            self.assertEqual(row["failure"]["stage"], "pre_guard_context")
            self.assertEqual(row["request_count"], 0)
            self.assertEqual(row["network_requests"], 0)
            self.assertEqual(row["verified_inner_trials"], 0)
            self.assertEqual(row["candidate_invocations"], 0)
            self.assertEqual(row["candidate_physical_fits"], 0)
            self.assertTrue(row["inputs_unchanged"])
            self.assertTrue(row["smoke"]["coverage"]["finalization_verified"])
            self.assertEqual(row["smoke"]["coverage"]["published_output_origin"], "baseline")
            self.assertFalse(row["smoke"]["coverage"]["generated_candidate_published"])
            self.assertEqual(row["smoke"]["reason"], "context_budget")
            self.assertEqual(row["physical_fits"], 3)


if __name__ == "__main__":
    unittest.main()
