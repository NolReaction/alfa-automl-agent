"""Target-free schema screening: semantic examples and resource admission.

These tests run the real, deterministic proposer unless a dependency response
or its call boundary is the behavior under test. They perform no model fits.
"""

import copy
from decimal import Inexact, ROUND_FLOOR, Rounded, localcontext
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import history_discovery as discovery
from agent.history_schema_proposer import propose_history_schemas


def panel():
    return pd.DataFrame({"entity": ["a", "b", "a", "b", "a", "b"],
                         "period": ["1", "1", "2", "2", "3", "3"]}, dtype=object)


def discover(frame, **options):
    arguments = {"feature_columns": list(frame.columns), "numeric_columns": ["period"],
                 "period_column": "period"}
    arguments.update(options)
    return discovery.discover_history_schema(frame, **arguments)


def evidence(result, name):
    return next(item for item in result["screening"]["columns"] if item["column"] == name)


def dependency_result():
    """A valid response independent of the helper's internal implementation."""
    return {"version": 1, "scope": "inner_fit_features_only", "auto_activate": False,
            "status": "proposed", "reason": None,
            "proposals": [{"config": {"key_columns": ["entity"], "period_column": "period",
                                      "numeric_columns": [], "positive_columns": [],
                                      "ambiguity_policy": "past_unique_only"},
                           "identity_verified": False,
                           "evidence": {"past_unambiguous_history_rows": 4}}],
            "limits": {"counts": {"checked_configs": 1}}}


class HistoryDiscoverySemanticsTests(unittest.TestCase):
    def test_proposal_is_only_structural_and_has_finite_json_contract(self):
        result = discover(panel())
        self.assertEqual(set(result), {"version", "scope", "status", "reason", "selected_config",
                                       "screening", "proposer_result", "identity_verified"})
        self.assertEqual((result["version"], result["scope"], result["status"]),
                         (1, "inner_fit_features_only", "proposed"))
        self.assertEqual(result["reason"], "structurally_eligible_for_inner_measurement")
        self.assertIs(result["identity_verified"], False)
        self.assertIs(result["proposer_result"]["auto_activate"], False)
        self.assertIs(result["proposer_result"]["proposals"][0]["identity_verified"], False)
        self.assertEqual(result["selected_config"], {
            "key_columns": ["entity"], "period_column": "period", "numeric_columns": [],
            "positive_columns": [], "ambiguity_policy": "past_unique_only"})
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        entry = evidence(result, "entity")
        self.assertEqual((entry["eligible_rows"], entry["distinct_values"],
                          entry["clean_recurrent_rows"], entry["cross_period_recurrent_rows"]),
                         (6, 2, 4, 6))
        self.assertIsNone(entry["excluded_reason"])

    def test_numeric_neighbors_above_float_precision_stay_distinct(self):
        frame = pd.DataFrame({
            "signature": ["9007199254740992.0", "9007199254740993", "9007199254740992",
                          "9007199254740993.00", "9.007199254740992e15", "9007199254740993.0"],
            "period": ["1", "1", "2.0", "2", "3", "3.0"]}, dtype=object)
        with localcontext() as context:
            context.prec, context.rounding = 3, ROUND_FLOOR
            context.traps[Inexact] = context.traps[Rounded] = True
            context.clear_flags()
            result = discover(frame, numeric_columns=["signature", "period"])
            self.assertFalse(any(context.flags.values()))
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["selected_config"]["numeric_columns"], ["signature"])
        entry = evidence(result, "signature")
        self.assertEqual((entry["distinct_values"], entry["clean_recurrent_rows"],
                          entry["cross_period_recurrent_rows"]), (2, 4, 6))

    def test_exact_period_neighbors_and_equivalent_decimal_periods(self):
        frame = pd.DataFrame({"entity": ["a", "b", "a", "b"],
                              "period": ["9007199254740992", "9007199254740992.00",
                                         "9007199254740993", "9007199254740993.0"]}, dtype=object)
        result = discover(frame)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["screening"]["counts"]["distinct_periods"], 2)
        self.assertEqual(evidence(result, "entity")["clean_recurrent_rows"], 2)
        # The two representations of period 1 are the SAME period: a is
        # ambiguous there, while b has one clean observation in each period.
        repeated = pd.DataFrame({"entity": ["a", "a", "a", "b", "b"],
                                 "period": ["1", "1.0", "2", "1", "2"]}, dtype=object)
        result = discover(repeated)
        self.assertEqual(result["screening"]["counts"]["distinct_periods"], 2)
        self.assertEqual(evidence(result, "entity")["clean_recurrent_rows"], 1)
        self.assertEqual(evidence(result, "entity")["cross_period_recurrent_rows"], 5)

    def test_text_na_and_leading_zeroes_are_literal_but_whitespace_is_stripped(self):
        frame = pd.DataFrame({"entity": ["NA", "0007", "7", " NA ", "0007 ", " 7"],
                              "period": ["1", "1", "1", "2", "2", "2"]}, dtype=object)
        result = discover(frame)
        entry = evidence(result, "entity")
        self.assertEqual(result["status"], "proposed")
        self.assertEqual((entry["eligible_rows"], entry["distinct_values"],
                          entry["clean_recurrent_rows"]), (6, 3, 3))
        self.assertEqual(result["selected_config"]["numeric_columns"], [])

    def test_invalid_numeric_strings_and_missing_keys_do_not_create_history(self):
        frame = pd.DataFrame({"entity": ["a", "b", "a", "b", None, " ", "a", "b"],
                              "period": ["1", "1", "2", "2", "3", "3", "NaN", "Infinity"]},
                             dtype=object)
        result = discover(frame)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["screening"]["counts"]["valid_period_rows"], 6)
        self.assertEqual(evidence(result, "entity")["eligible_rows"], 4)
        self.assertEqual(evidence(result, "entity")["clean_recurrent_rows"], 2)

    def test_result_is_invariant_to_row_column_and_whitelist_order(self):
        frame = panel()
        frame["alias"] = frame["entity"].tolist()
        frame["constant"] = "same"
        frame["unique"] = [str(i) for i in range(len(frame))]
        original = discover(frame, numeric_columns=("period",))
        shuffled = frame.sample(frac=1, random_state=379).iloc[:, ::-1]
        changed = discover(shuffled, feature_columns=list(reversed(frame.columns)), numeric_columns=("period",))
        self.assertEqual(changed, original)

    def test_coarse_recurrent_keys_abstain_without_unique_past(self):
        frame = pd.DataFrame({"entity": ["a", "a", "b", "b"] * 3,
                              "period": [str(p) for p in (1, 2, 3) for _ in range(4)]}, dtype=object)
        with patch.object(discovery, "propose_history_schemas", wraps=propose_history_schemas) as proposer:
            result = discover(frame)
        proposer.assert_called_once()
        self.assertEqual((result["status"], result["reason"]), ("abstained", "proposer_no_past_coverage"))
        self.assertIsNone(result["selected_config"])
        self.assertEqual(result["screening"]["selected_columns"], ["entity"])
        self.assertEqual(evidence(result, "entity")["clean_recurrent_rows"], 0)
        self.assertEqual(evidence(result, "entity")["cross_period_recurrent_rows"], 12)

    def test_ambiguous_single_parts_can_form_a_clean_composite_key(self):
        rows = [(a, b, str(period)) for period in (1, 2, 3)
                for a in ("x", "y") for b in ("u", "v")]
        frame = pd.DataFrame(rows, columns=["a", "b", "period"], dtype=object)
        result = discover(frame)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["selected_config"]["key_columns"], ["a", "b"])
        self.assertEqual(result["proposer_result"]["proposals"][0]["evidence"]["past_unambiguous_history_rows"], 8)
        for name in ("a", "b"):
            entry = evidence(result, name)
            self.assertEqual((entry["clean_recurrent_rows"], entry["cross_period_recurrent_rows"]), (0, 12))
            self.assertIsNone(entry["excluded_reason"])

    def test_all_excluded_columns_and_zero_coverage_never_call_proposer(self):
        frame = pd.DataFrame({"period": ["1", "1", "2", "2", "3", "3"],
                              "constant": ["same"] * 6,
                              "unique": [str(i) for i in range(6)],
                              "within_period": ["a", "a", "b", "b", "c", "c"]}, dtype=object)
        with patch.object(discovery, "propose_history_schemas") as proposer:
            result = discover(frame)
        proposer.assert_not_called()
        self.assertEqual((result["status"], result["reason"]), ("abstained", "no_recurrent_key_columns"))
        self.assertIsNone(result["selected_config"])
        self.assertIsNone(result["proposer_result"])
        self.assertEqual({item["column"]: item["excluded_reason"] for item in result["screening"]["columns"]},
                         {"constant": "constant", "unique": "all_unique",
                          "within_period": "no_cross_period_recurrence"})
        counts = result["screening"]["counts"]
        self.assertEqual((counts["excluded_constant"], counts["excluded_all_unique"],
                          counts["excluded_no_cross_period_recurrence"], counts["recurrent_columns"]),
                         (1, 1, 1, 0))

    def test_no_rows_no_valid_period_and_single_period_abstain_before_proposer(self):
        for frame, reason in ((panel().iloc[:0], "no_rows"),
                              (panel().assign(period=[None, "", " ", "NaN", "inf", "x"]), "no_valid_period"),
                              (panel().assign(period=["2", "2.0", "2E0", " 2 ", None, "x"]), "single_period")):
            with self.subTest(reason=reason), patch.object(discovery, "propose_history_schemas") as proposer:
                result = discover(frame)
                proposer.assert_not_called()
                self.assertEqual((result["status"], result["reason"]), ("abstained", reason))
                self.assertIsNone(result["selected_config"])


class HistoryDiscoveryRankingTests(unittest.TestCase):
    def test_screening_keeps_twelve_deterministic_columns_and_excludes_period(self):
        frame = pd.DataFrame({f"key_{i:02d}": ["a", "b", "c"] * 3 for i in range(16)}, dtype=object)
        frame["period"] = [str(p) for p in (1, 2, 3) for _ in range(3)]
        frame["constant"] = "same"
        frame["unique"] = [str(i) for i in range(len(frame))]
        frame["within_period"] = frame["period"].tolist()
        with patch.object(discovery, "propose_history_schemas", wraps=propose_history_schemas) as proposer:
            result = discover(frame)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["screening"]["selected_columns"], [f"key_{i:02d}" for i in range(12)])
        self.assertNotIn("period", [item["column"] for item in result["screening"]["columns"]])
        counts = result["screening"]["counts"]
        self.assertEqual((counts["screened_columns"], counts["recurrent_columns"], counts["selected_columns"]),
                         (19, 16, 12))
        raw, = proposer.call_args.args
        self.assertEqual(set(raw), {"period", *(f"key_{i:02d}" for i in range(12))})
        self.assertEqual(proposer.call_args.kwargs["allowed_key_columns"], [f"key_{i:02d}" for i in range(12)])

    def test_rank_prefers_clean_history_then_total_recurrence_then_name(self):
        periods = [str(p) for p in (1, 2, 3) for _ in range(4)]
        frame = pd.DataFrame({
            "period": periods,
            "z_clean": ["a", "b", None, None] * 3,
            "a_clean": ["a", "b", None, None] * 3,
            "z_collision": ["x", "x", "y", "y"] * 3,
            "a_collision": ["x", "x", "y", "y"] * 3,
            "short_collision": ["x", "x", "y", "y"] * 2 + [None] * 4}, dtype=object)
        result = discover(frame)
        self.assertEqual(result["screening"]["selected_columns"],
                         ["a_clean", "z_clean", "a_collision", "z_collision", "short_collision"])
        self.assertEqual(evidence(result, "a_clean")["clean_recurrent_rows"], 4)
        self.assertEqual(evidence(result, "a_collision")["cross_period_recurrent_rows"], 12)
        self.assertEqual(evidence(result, "short_collision")["cross_period_recurrent_rows"], 8)


class HistoryDiscoveryContractTests(unittest.TestCase):
    def test_proposer_called_once_with_fixed_policy_limits_and_only_screened_raw_columns(self):
        frame = panel()
        frame["constant"] = "same"
        with patch.object(discovery, "propose_history_schemas", wraps=propose_history_schemas) as proposer:
            result = discover(frame)
        proposer.assert_called_once()
        raw, = proposer.call_args.args
        self.assertEqual(raw, {"entity": panel()["entity"].tolist(), "period": panel()["period"].tolist()})
        arguments = proposer.call_args.kwargs
        self.assertEqual(arguments["task_type"], "regression")
        self.assertEqual(arguments["feature_columns"], ["constant", "entity", "period"])
        self.assertEqual(arguments["allowed_period_columns"], ["period"])
        self.assertEqual(arguments["allowed_key_columns"], ["entity"])
        self.assertEqual(arguments["numeric_key_columns"], [])
        self.assertEqual(arguments["positive_key_columns"], ())
        self.assertEqual(arguments["explicit_key_sets"], ())
        self.assertEqual(arguments["limits"], {
            "max_rows": 250_000, "max_work": 8_000_000 - 36, "max_period_columns": 1,
            "max_key_columns": 12, "max_auto_key_width": 3, "max_explicit_key_width": 3,
            "max_checked_configs": 48, "max_proposals": 1})
        counts = result["screening"]["counts"]
        self.assertEqual((counts["raw_cells"], counts["raw_cells_scanned"], counts["proposer_calls"]), (18, 18, 1))
        self.assertEqual(counts["screening_cell_visits_planned"], 36)
        self.assertEqual(counts["proposer_cell_visits_planned"], 6 * 195)
        self.assertEqual(counts["total_cell_visits_planned"], 36 + 6 * 195)

    def test_inputs_are_unchanged_and_output_is_detached_from_dependency_containers(self):
        frame = panel()
        before = frame.copy(deep=True)
        features, numeric = ["entity", "period"], ["period"]
        source = dependency_result()
        source_before = copy.deepcopy(source)
        with patch.object(discovery, "propose_history_schemas", return_value=source):
            result = discover(frame, feature_columns=features, numeric_columns=numeric)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual((features, numeric), (["entity", "period"], ["period"]))
        self.assertEqual(source, source_before)
        result["selected_config"]["key_columns"].append("changed")
        result["proposer_result"]["limits"]["counts"]["checked_configs"] = 99
        result["screening"]["selected_columns"].append("changed")
        self.assertEqual(source, source_before)
        self.assertEqual(result["proposer_result"]["proposals"][0]["config"]["key_columns"], ["entity"])

    def test_search_needing_more_than_24_checks_still_completes_and_proposes_the_entity(self):
        # Five random binary columns repeat across periods. Their coverage bounds
        # exceed the entity's unique history, so proving the entity best takes
        # 26 full checks, as on public TRAIN periods 1-3 (37 checks).
        random = np.random.default_rng(0)
        frame = pd.DataFrame([
            {"period": str(period), "building": f"b{entity}" if entity < 20 else f"p{period}_{entity}",
             **{f"d{index}": f"v{random.integers(2)}" for index in range(5)}}
            for period in (1, 2, 3) for entity in range(40)], dtype=object)
        result = discover(frame)
        counts = result["proposer_result"]["limits"]["counts"]
        self.assertEqual((result["status"], result["selected_config"]["key_columns"]), ("proposed", ["building"]))
        self.assertEqual(counts["checked_configs"], 26)
        self.assertIs(counts["search_complete"], True)
        with patch.object(discovery, "MAX_CHECKED_CONFIGS", 24):
            limited = discover(frame)
        self.assertEqual((limited["status"], limited["reason"]), ("limited", "proposer_limited"))
        self.assertIsNone(limited["selected_config"])

    def test_limited_dependency_never_selects_even_a_present_proposal(self):
        response = dependency_result()
        response["status"], response["reason"] = "limited", "work_limit"
        with patch.object(discovery, "propose_history_schemas", return_value=response) as proposer:
            result = discover(panel())
        proposer.assert_called_once()
        self.assertEqual((result["status"], result["reason"]), ("limited", "proposer_limited"))
        self.assertIsNone(result["selected_config"])
        self.assertEqual(result["proposer_result"]["proposals"], response["proposals"])

    def test_bad_dependency_contract_is_rejected(self):
        changes = [lambda value: value.update(version=2),
                   lambda value: value.update(scope="all_rows"),
                   lambda value: value.update(auto_activate=True),
                   lambda value: value.update(status="unknown"),
                   lambda value: value.update(proposals=[]),
                   lambda value: value["proposals"][0].update(identity_verified=True),
                   lambda value: value["proposals"][0]["config"].update(ambiguity_policy="allow"),
                   lambda value: value["proposals"][0]["config"].update(period_column="other"),
                   lambda value: value["proposals"][0]["config"].update(key_columns=["unknown"]),
                   lambda value: value["proposals"][0]["config"].update(key_columns=["entity", "entity"]),
                   lambda value: value["proposals"][0]["config"].update(numeric_columns=["entity"]),
                   lambda value: value["proposals"][0]["config"].update(positive_columns=["entity"]),
                   lambda value: value["proposals"][0]["config"].update(extra="forbidden")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                response = dependency_result()
                change(response)
                with patch.object(discovery, "propose_history_schemas", return_value=response), self.assertRaises(ValueError):
                    discover(panel())

    def test_nonfinite_dependency_evidence_is_rejected_by_json_boundary(self):
        response = dependency_result()
        response["limits"]["counts"]["invalid"] = float("nan")
        with patch.object(discovery, "propose_history_schemas", return_value=response), self.assertRaises(ValueError):
            discover(panel())

    def test_frame_must_match_exactly_the_declared_feature_whitelist(self):
        frame = panel()
        bad_options = [{"feature_columns": ["entity"]},
                       {"feature_columns": ["entity", "period", "missing"]},
                       {"feature_columns": ["entity", "entity", "period"]},
                       {"feature_columns": "entity"},
                       {"feature_columns": []},
                       {"numeric_columns": ["target"]},
                       {"numeric_columns": "period"},
                       {"numeric_columns": ["period", "period"]},
                       {"period_column": "missing"},
                       {"period_column": None}]
        with patch.object(discovery, "propose_history_schemas") as proposer:
            for options in bad_options:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    discover(frame, **options)
            with self.assertRaises(ValueError):
                discover(frame.assign(target=["answer"] * len(frame)), feature_columns=["entity", "period"])
        proposer.assert_not_called()

    def test_malformed_frames_names_and_cells_are_not_coerced(self):
        with self.assertRaises(ValueError):
            discovery.discover_history_schema({"entity": ["a"], "period": ["1"]},
                                             feature_columns=["entity", "period"],
                                             numeric_columns=["period"], period_column="period")
        duplicate = pd.DataFrame([["a", "b", "1"]], columns=["entity", "entity", "period"], dtype=object)
        invalid_name = pd.DataFrame([["a", "1"]], columns=[1, "period"], dtype=object)
        for frame in (duplicate, invalid_name, panel().rename(columns={"entity": " "})):
            with self.subTest(columns=list(frame.columns)), self.assertRaises(ValueError):
                discover(frame)
        # Even a column which screening would otherwise discard is validated.
        for value in (True, False, 1, 1.0, np.nan, pd.NA, np.bool_(True), b"raw"):
            with self.subTest(value=repr(value)):
                frame = panel()
                frame["constant"] = pd.Series([value] * len(frame), dtype=object)
                with patch.object(discovery, "propose_history_schemas") as proposer, self.assertRaises(ValueError):
                    discover(frame)
                proposer.assert_not_called()
        bad_utf8 = panel()
        bad_utf8.loc[0, "entity"] = "\ud800"
        with self.assertRaises(ValueError):
            discover(bad_utf8)


class HistoryDiscoveryResourceTests(unittest.TestCase):
    def assert_limited(self, frame, reason, **options):
        with patch.object(discovery, "propose_history_schemas") as proposer:
            result = discover(frame, **options)
        proposer.assert_not_called()
        self.assertEqual((result["status"], result["reason"]), ("limited", reason))
        self.assertIsNone(result["selected_config"])
        self.assertIsNone(result["proposer_result"])
        self.assertEqual(result["screening"]["counts"]["proposer_calls"], 0)
        return result

    def test_real_row_limit_is_checked_without_reading_or_sampling_cells(self):
        frame = pd.DataFrame({"entity": [None] * 250_001, "period": [None] * 250_001}, dtype=object)
        result = self.assert_limited(frame, "row_limit")
        self.assertEqual(result["screening"]["counts"]["rows"], 250_001)
        self.assertEqual(result["screening"]["counts"]["raw_cells_scanned"], 0)
        self.assertEqual(result["screening"]["selected_columns"], [])

    def test_real_feature_limit_is_checked_before_cell_validation(self):
        frame = pd.DataFrame({"period": ["1"], **{f"k{i:03d}": [True] for i in range(128)}}, dtype=object)
        result = self.assert_limited(frame, "feature_limit")
        self.assertEqual(result["screening"]["counts"]["features"], 129)
        self.assertEqual(result["screening"]["counts"]["raw_cells_scanned"], 0)

    def test_screening_work_is_admitted_before_any_raw_cell_is_read(self):
        with patch.object(discovery, "MAX_CELL_VISITS", 23):
            result = self.assert_limited(panel(), "work_limit")
        self.assertEqual(result["screening"]["counts"]["screening_cell_visits_planned"], 24)
        self.assertEqual(result["screening"]["counts"]["raw_cells_scanned"], 0)

    def test_full_proposer_work_is_admitted_before_call_without_partial_proposals(self):
        with patch.object(discovery, "MAX_CELL_VISITS", 24):
            result = self.assert_limited(panel(), "work_limit")
        counts = result["screening"]["counts"]
        self.assertEqual(counts["raw_cells_scanned"], 12)
        self.assertEqual(counts["proposer_cell_visits_planned"], 1170)
        self.assertEqual(counts["total_cell_visits_planned"], 1194)
        self.assertEqual(result["screening"]["selected_columns"], ["entity"])
        with patch.object(discovery, "MAX_CELL_VISITS", 1194):
            accepted = discover(panel())
        self.assertEqual(accepted["status"], "proposed")
        self.assertEqual(accepted["screening"]["counts"]["total_cell_visits_planned"], 1194)

    def test_value_length_limit_uses_original_raw_length_without_truncation(self):
        frame = panel()
        frame.loc[0, "entity"] = " " * 513
        result = self.assert_limited(frame, "value_length_limit")
        self.assertEqual(result["screening"]["counts"]["max_value_chars_observed"], 513)
        self.assertLess(result["screening"]["counts"]["raw_cells_scanned"], len(frame) * len(frame.columns))

    def test_raw_byte_limit_counts_utf8_bytes_and_accepts_exact_bound(self):
        frame = panel().replace({"a": "яя", "b": "бб"})
        # Six two-character Cyrillic keys take 24 bytes, six periods take 6.
        with patch.object(discovery, "MAX_RAW_BYTES", 29):
            result = self.assert_limited(frame, "raw_bytes_limit")
        self.assertEqual(result["screening"]["counts"]["raw_value_bytes_observed"], 30)
        self.assertEqual(result["screening"]["counts"]["max_value_chars_observed"], 2)
        with patch.object(discovery, "MAX_RAW_BYTES", 30):
            accepted = discover(frame)
        self.assertEqual(accepted["status"], "proposed")
        self.assertEqual(accepted["screening"]["counts"]["raw_value_bytes_observed"], 30)


if __name__ == "__main__":
    unittest.main()
