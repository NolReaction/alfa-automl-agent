import ast
import copy
from decimal import Inexact, ROUND_FLOOR, Rounded, localcontext
from itertools import combinations
import inspect
import json
from pathlib import Path
import random
import unittest

import pandas as pd

from agent.building_history import history_keys
from agent.history_candidate import HistoryConfig
from agent.history_schema_proposer import DEFAULT_LIMITS, propose_history_schemas
import history_schema_cases as cases


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ["rows", "eligible_rows", "strict_past_history_rows", "past_unambiguous_history_rows",
            "past_ambiguous_history_rows", "distinct_keys", "distinct_periods"]


def table(*rows, columns=("obj", "zone", "period")) -> dict:
    return {name: [row[index] for row in rows] for index, name in enumerate(columns)}


def propose(raw, *, keys=("obj", "zone"), periods=("period",), **options):
    arguments = {"task_type": "regression", "feature_columns": list(raw), "allowed_period_columns": list(periods),
                 "allowed_key_columns": list(keys)}
    arguments.update(options)
    return propose_history_schemas(raw, **arguments)


def configs(result) -> list:
    return [(proposal["config"]["period_column"], proposal["config"]["key_columns"]) for proposal in result["proposals"]]


PANEL = table(("a", "z", "1"), ("b", "z", "1"), ("a", "z", "2"), ("b", "z", "2"), ("c", "y", "2"), ("a", "z", "3"))


def random_table(seed: int) -> dict:
    rng = random.Random(seed)
    rows = rng.randint(30, 80)
    raw = {}
    for name, size in (("c1", 2), ("c2", 3), ("c3", 5), ("c4", 7), ("c5", 4)):
        raw[name] = [rng.choice([None, *(f"v{value}" for value in range(size))]) if rng.random() < 0.1
                     else f"v{rng.randrange(size)}" for _ in range(rows)]
    raw["period"] = [str(rng.randint(1, 4)) for _ in range(rows)]
    raw["alt"] = [str(rng.randint(1, 2)) for _ in range(rows)]
    return raw


def exhaustive(raw, keys, periods, top, width) -> list:
    """All automatic keys checked by the oracle and ranked by the documented rule."""
    found = []
    for period in periods:
        usable = [name for name in keys if name != period]
        for size in range(1, width + 1):
            for key in combinations(sorted(usable), size):
                config = {"key_columns": list(key), "period_column": period, "numeric_columns": [],
                          "positive_columns": []}
                found.append((period, key, cases.oracle_evidence(raw, config)))
    found.sort(key=lambda item: (-item[2]["past_unambiguous_history_rows"], item[2]["past_ambiguous_history_rows"],
                                 len(item[1]), item[0], item[1]))
    kept = []
    for period, key, evidence in found:
        if len(kept) == top or evidence["past_unambiguous_history_rows"] == 0:
            break
        same = any(p == period and set(k) < set(key) and e["eligible_rows"] == evidence["eligible_rows"]
                   and e["distinct_keys"] == evidence["distinct_keys"] for p, k, e in kept)
        if not same:
            kept.append((period, key, evidence))
    return [(period, list(key)) for period, key, _ in kept]


class ContractTests(unittest.TestCase):
    def test_signature_matches_the_agreed_interface(self):
        parameters = inspect.signature(propose_history_schemas).parameters
        self.assertEqual(list(parameters), ["raw_columns", "task_type", "feature_columns", "allowed_period_columns",
                                            "allowed_key_columns", "explicit_key_sets", "numeric_key_columns",
                                            "positive_key_columns", "limits"])
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in parameters.items()
                            if name != "raw_columns"))

    def test_result_shape_is_json_and_never_activates_or_verifies_identity(self):
        result = propose(PANEL)
        self.assertEqual(set(result), {"version", "status", "reason", "scope", "auto_activate", "proposals", "limits"})
        self.assertEqual((result["version"], result["status"], result["reason"]), (1, "proposed", None))
        self.assertEqual((result["scope"], result["auto_activate"]), ("inner_fit_features_only", False))
        json.dumps(result, allow_nan=False)
        for proposal in result["proposals"]:
            self.assertEqual(set(proposal), {"config", "evidence", "identity_verified", "reason"})
            self.assertIs(proposal["identity_verified"], False)
            self.assertEqual(proposal["reason"], "structurally_eligible_for_inner_measurement")
            self.assertEqual(list(proposal["evidence"]), EVIDENCE)
            self.assertEqual(proposal["config"]["ambiguity_policy"], "past_unique_only")
            self.assertEqual(HistoryConfig.from_json(proposal["config"]).to_json(), proposal["config"])
        text = json.dumps(result)
        for word in ("accepted", "prediction", "weight", "improve"):
            self.assertNotIn(word, text)

    def test_every_status_keeps_the_same_shape(self):
        results = [propose(PANEL), propose(PANEL, task_type="classification"), propose(PANEL, periods=()),
                   propose(PANEL, limits={"max_rows": 2})]
        self.assertEqual([r["status"] for r in results], ["proposed", "abstained", "abstained", "limited"])
        for result in results:
            self.assertEqual(set(result), {"version", "status", "reason", "scope", "auto_activate", "proposals",
                                           "limits"})
            self.assertEqual(set(result["limits"]), {"caps", "counts"})
            self.assertFalse(result["auto_activate"])

    def test_module_uses_only_the_standard_library_and_no_public_names(self):
        source = (ROOT / "agent" / "history_schema_proposer.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        # pure computation only: no files, environment, network, clock or third-party packages
        self.assertLessEqual(imported, {"__future__", "collections", "decimal", "itertools", "operator"})
        for name in ("Year_Factor", "State_Factor", "building_class", "facility_type", "floor_area", "year_built",
                     "ELEVATION", "site_eui", "open("):
            self.assertNotIn(name, source)


class InputTests(unittest.TestCase):
    def check_error(self, raw=PANEL, **options):
        with self.assertRaises(ValueError):
            propose(raw, **options)

    def test_illegal_or_unknown_columns_are_rejected(self):
        self.check_error(keys=("obj", "missing"))
        self.check_error(periods=("missing",))
        self.check_error(feature_columns=["obj", "zone"])                    # raw has a column outside features
        self.check_error(feature_columns=["obj", "zone", "period", "extra"], keys=("obj", "extra"))
        self.check_error(explicit_key_sets=[["obj", "period"]], keys=("obj", "zone"))  # explicit outside allowed
        self.check_error(numeric_key_columns=["period"])                     # numeric part outside allowed keys
        self.check_error(positive_key_columns=["obj"])                       # positive part must be numeric
        self.check_error(keys=("obj", "obj"))
        self.check_error(feature_columns=["obj", "zone", "period", "obj"])
        self.check_error(explicit_key_sets=[["obj", "zone"], ["zone", "obj"]])
        self.check_error(explicit_key_sets=[[]])
        self.check_error(explicit_key_sets="obj")
        self.check_error(keys="obj")
        self.check_error(keys=("obj", " "))

    def test_malformed_tables_and_values_are_rejected(self):
        self.check_error(raw={})
        self.check_error(raw=[["a"]])
        self.check_error(raw={"obj": ["a", "b"], "zone": ["z"], "period": ["1", "2"]})
        self.check_error(raw={"obj": "ab", "zone": ["z", "z"], "period": ["1", "2"]})
        self.check_error(raw={"obj": ["a", 1], "zone": ["z", "z"], "period": ["1", "2"]})
        self.check_error(raw={"obj": ["a", "b"], "zone": ["z", "z"], "period": ["1", 2.0]})
        self.check_error(raw={1: ["a"], "zone": ["z"], "period": ["1"]}, feature_columns=["zone", "period"])
        self.check_error(task_type=None)

    def test_wrong_limits_are_rejected(self):
        for limits in ({"max_rows": True}, {"max_rows": 0}, {"max_rows": -1}, {"max_rows": 2.0}, {"max_rows": "9"},
                       {"unknown": 1}, ["max_rows", 1], "max_rows"):
            with self.subTest(limits=limits):
                self.check_error(limits=limits)

    def test_inputs_are_not_changed(self):
        raw = copy.deepcopy(PANEL)
        explicit, limits = [["obj", "zone"]], {"max_proposals": 3}
        before = copy.deepcopy((raw, explicit, limits))
        result = propose(raw, explicit_key_sets=explicit, limits=limits)
        self.assertEqual((raw, explicit, limits), before)
        result["proposals"][0]["config"]["key_columns"].append("x")
        self.assertEqual(raw, before[0])

    def test_tuples_are_accepted_like_lists(self):
        raw = {name: tuple(values) for name, values in PANEL.items()}
        self.assertEqual(propose(raw), propose(PANEL))


class AbstentionTests(unittest.TestCase):
    def test_classification_and_unknown_task_types_abstain(self):
        self.assertEqual(propose(PANEL, task_type="classification")["reason"], "classification_task")
        self.assertEqual(propose(PANEL, task_type="ranking")["reason"], "unknown_task_type")

    def test_missing_permissions_abstain(self):
        self.assertEqual(propose(PANEL, periods=())["reason"], "no_allowed_period_column")
        self.assertEqual(propose(PANEL, keys=())["reason"], "no_allowed_key_columns")
        self.assertEqual(propose({name: [] for name in PANEL})["reason"], "no_rows")

    def test_one_period_or_no_valid_period_abstains(self):
        one = table(("a", "z", "5"), ("a", "z", "5.0"), ("b", "z", " 5 "))
        self.assertEqual(propose(one)["reason"], "single_period")
        none = table(("a", "z", None), ("a", "z", "x"), ("b", "z", "nan"), ("b", "z", "inf"))
        self.assertEqual(propose(none)["reason"], "no_valid_period")

    def test_unique_objects_and_same_period_repeats_give_no_history(self):
        unique = table(("a", "z1", "1"), ("b", "z2", "2"), ("c", "z3", "3"))
        self.assertEqual(propose(unique)["reason"], "no_past_coverage")
        same_period = table(("a", "z1", "2"), ("a", "z1", "2"), ("b", "z2", "1"))
        self.assertEqual(propose(same_period)["reason"], "no_past_coverage")

    def test_only_repeated_past_keys_abstain_under_the_fixed_policy(self):
        raw = table(("a", "z", "1"), ("a", "z", "1"), ("a", "z", "2"))
        result = propose(raw, keys=("obj",))
        self.assertEqual((result["status"], result["reason"]), ("abstained", "no_past_coverage"))


class EvidenceTests(unittest.TestCase):
    def evidence(self, raw, key, **options):
        result = propose(raw, keys=key, limits={"max_proposals": 1}, **options)
        self.assertEqual(result["status"], "proposed")
        return result["proposals"][0]["evidence"]

    def test_only_strictly_earlier_periods_give_history(self):
        raw = table(("a", "z", "1"), ("a", "z", "2"), ("a", "z", "2"), ("a", "z", "3"), ("b", "z", "3"),
                    ("b", "z", "3"))
        evidence = self.evidence(raw, ("obj",))
        self.assertEqual(evidence["strict_past_history_rows"], 3)       # the two rows of 2 and the row of 3
        self.assertEqual(evidence["past_unambiguous_history_rows"], 2)  # period 2 sees only period 1
        self.assertEqual(evidence["past_ambiguous_history_rows"], 1)    # period 3 sees the repeat inside period 2
        self.assertEqual((evidence["eligible_rows"], evidence["distinct_keys"], evidence["distinct_periods"]),
                         (6, 2, 3))

    def test_period_is_exact_without_float_or_context_rounding(self):
        close = ["1." + "0" * 25 + "1", "1." + "0" * 25 + "2"]
        raw = table(("a", "z", close[0]), ("a", "z", close[1]), ("b", "z", "1E+30"),
                    ("b", "z", "1" + "0" * 29 + "1"))
        with localcontext() as context:
            context.prec, context.rounding = 3, ROUND_FLOOR
            context.traps[Inexact] = context.traps[Rounded] = True
            context.clear_flags()
            evidence = self.evidence(raw, ("obj",))
            self.assertFalse(any(context.flags.values()))
        self.assertEqual((evidence["strict_past_history_rows"], evidence["distinct_periods"]), (2, 4))
        same = table(("a", "z", "5"), ("b", "z", "5.000"), ("c", "z", "5E0"), ("a", "z", "6"))
        self.assertEqual(self.evidence(same, ("obj",))["distinct_periods"], 2)

    def test_missing_semantics_follow_the_adapter(self):
        raw = table(("NA", "z", "1"), ("NA", "z", "2"), ("?", "z", "1"), ("?", "z", "2"), (None, "z", "1"),
                    (None, "z", "2"), ("", "z", "1"), ("  ", "z", "2"))
        evidence = self.evidence(raw, ("obj",))
        self.assertEqual((evidence["eligible_rows"], evidence["strict_past_history_rows"]), (4, 2))
        numbers = table(("61242", "0", "1"), ("61242.0", "-3", "2"), (" 61242.00 ", "7", "3"), ("n/a", "7", "4"),
                        columns=("code", "built", "period"))
        evidence = self.evidence(numbers, ("code", "built"), numeric_key_columns=["code", "built"],
                                 positive_key_columns=["built"], explicit_key_sets=[["code"]])
        self.assertEqual((evidence["eligible_rows"], evidence["strict_past_history_rows"]), (3, 2))

    def test_numeric_parsing_only_for_explicit_numeric_columns(self):
        raw = table(("1", "z", "1"), ("1.0", "z", "2"))
        self.assertEqual(propose(raw, keys=("obj",))["reason"], "no_past_coverage")
        self.assertEqual(propose(raw, keys=("obj",), numeric_key_columns=["obj"])["status"], "proposed")

    def test_values_match_the_adapter_parsing(self):
        values = ["61242", "61242.0", " 61242 ", "-0", "0", "1e3", "1000", "NA", "?", "", "  ", None, "nan", "inf",
                  "x", "0.1", "0.10", "1_000", "-5"]
        frame = pd.DataFrame({"text": values, "number": values, "positive": values}, dtype=object)
        adapter = history_keys(frame, ["text", "number", "positive"], ["number", "positive"], ["positive"],
                               allow_missing=True)
        for index, value in enumerate(values):
            with self.subTest(value=value):
                key = tuple(cases._oracle_value(value, name != "text", name == "positive")
                            for name in ("text", "number", "positive"))
                self.assertEqual(key, adapter[index])
                one = {"obj": [value, value], "zone": ["z", "z"], "period": ["1", "2"]}
                for numeric, positive, part in ((False, False, 0), (True, False, 1), (True, True, 2)):
                    result = propose(one, keys=("obj",), numeric_key_columns=["obj"] if numeric else [],
                                     positive_key_columns=["obj"] if positive else [])
                    self.assertEqual(result["status"] == "proposed", adapter[index][part] is not None)

    def test_evidence_matches_the_oracle_and_the_adapter_encoder_on_the_catalog(self):
        for case in cases.build_catalog(5):
            result = propose_history_schemas(case.raw_columns, **case.call)
            for proposal in result["proposals"]:
                with self.subTest(case=case.name, config=proposal["config"]):
                    config = proposal["config"]
                    self.assertEqual(proposal["evidence"], cases.oracle_evidence(case.raw_columns, config))
                    self.assertEqual(proposal["evidence"], cases.adapter_evidence(case.raw_columns, config))


class SearchTests(unittest.TestCase):
    def test_complete_search_equals_the_exhaustive_ranking(self):
        compared = 0
        for seed in range(40):
            raw = random_table(seed)
            keys = ["c1", "c2", "c3", "c4", "c5"]
            for periods in (["period"], ["period", "alt"]):
                with self.subTest(seed=seed, periods=periods):
                    result = propose(raw, keys=keys, periods=periods)
                    if result["limits"]["counts"]["search_complete"]:
                        compared += 1
                        self.assertEqual(configs(result), exhaustive(raw, keys, periods, 4, 3))
                    else:
                        self.assertEqual(result["status"], "limited")
                    full = propose(raw, keys=keys, periods=periods, limits={"max_checked_configs": 500})
                    self.assertTrue(full["limits"]["counts"]["search_complete"])
                    self.assertEqual(configs(full), exhaustive(raw, keys, periods, 4, 3))
        self.assertGreaterEqual(compared, 10)  # 15 of 80 complete within the default cap of 24

    def test_row_column_and_list_order_do_not_change_the_result(self):
        for case in cases.build_catalog(5):
            with self.subTest(case=case.name):
                expected = propose_history_schemas(case.raw_columns, **case.call)
                rng = random.Random(case.name)
                order = list(range(len(case.truth)))
                rng.shuffle(order)
                names = list(case.raw_columns)
                rng.shuffle(names)
                raw = {name: [case.raw_columns[name][index] for index in order] for name in names}
                call = copy.deepcopy(case.call)
                for field in ("feature_columns", "allowed_period_columns", "allowed_key_columns", "explicit_key_sets",
                              "numeric_key_columns", "positive_key_columns"):
                    rng.shuffle(call[field])
                self.assertEqual(propose_history_schemas(raw, **call), expected)

    def test_checks_never_exceed_the_cap_and_a_cut_search_is_limited(self):
        rng = random.Random(3)
        rows = 300
        raw = {f"k{index}": [f"v{rng.randrange(3 + index)}" for _ in range(rows)] for index in range(12)}
        raw["period"] = [str(rng.randint(1, 6)) for _ in range(rows)]
        keys = [f"k{index}" for index in range(12)]
        for cap in (1, 5, 24):
            result = propose(raw, keys=keys, limits={"max_checked_configs": cap})
            counts = result["limits"]["counts"]
            self.assertLessEqual(counts["checked_configs"], cap)
            if not counts["search_complete"]:
                self.assertEqual((result["status"], result["reason"]), ("limited", "check_limit"))
                self.assertGreater(counts["unchecked_pool"], 0)
            self.assertLessEqual(len(result["proposals"]), 4)

    def test_explicit_sets_are_checked_first_and_kept_as_given(self):
        raw = cases.long_explicit_key(random.Random(1)).raw_columns
        call = cases.long_explicit_key(random.Random(1)).call
        call["explicit_key_sets"] = [["p5", "p3", "p1", "p2", "p4"]]
        result = propose_history_schemas(raw, **call, limits={"max_checked_configs": 1})
        self.assertEqual(result["limits"]["counts"]["checked_configs"], 1)
        self.assertEqual(configs(result)[0], ("period", ["p5", "p3", "p1", "p2", "p4"]))
        self.assertEqual(result["status"], "limited")
        complete = propose_history_schemas(raw, **call)
        self.assertEqual(configs(complete)[0], ("period", ["p5", "p3", "p1", "p2", "p4"]))

    def test_a_wider_key_with_the_same_groups_is_not_proposed_twice(self):
        raw = table(("a", "z1", "1"), ("b", "z2", "1"), ("a", "z1", "2"), ("b", "z2", "2"))
        self.assertEqual(configs(propose(raw)), [("period", ["obj"]), ("period", ["zone"])])

    def test_over_caps_nothing_is_read(self):
        poisoned = {"obj": [1, 2, 3], "zone": ["z", "z", "z"], "period": ["1", "2", "3"]}  # int would raise if read
        for limits, reason in (({"max_rows": 2}, "row_limit"), ({"max_work": 10}, "work_limit"),
                               ({"max_key_columns": 1}, "key_columns_over_limit"),
                               ({"max_period_columns": 1}, "period_columns_over_limit"),
                               ({"max_explicit_key_width": 1}, "explicit_key_width_over_limit")):
            with self.subTest(reason=reason):
                result = propose(poisoned, limits=limits, explicit_key_sets=[["obj", "zone"]],
                                 periods=("period", "zone") if reason == "period_columns_over_limit" else ("period",))
                self.assertEqual((result["status"], result["reason"], result["proposals"]), ("limited", reason, []))
        self.assertEqual(DEFAULT_LIMITS["max_checked_configs"], 24)

    def test_same_x_with_different_hidden_truth_gives_the_same_proposals(self):
        one, two = cases.deceptive_one_object(random.Random(9)), cases.deceptive_two_objects(random.Random(9))
        self.assertEqual(one.raw_columns, two.raw_columns)
        self.assertNotEqual(one.truth, two.truth)
        first = propose_history_schemas(one.raw_columns, **one.call)
        self.assertEqual(first, propose_history_schemas(two.raw_columns, **two.call))
        config = first["proposals"][0]["config"]
        self.assertEqual(cases.oracle_evidence(one.raw_columns, config, one.truth)["false_match_rows"], 0)
        self.assertGreater(cases.oracle_evidence(two.raw_columns, config, two.truth)["false_match_rows"], 0)


class CatalogTests(unittest.TestCase):
    def test_catalog_is_deterministic_and_the_renamed_copy_is_the_same_table(self):
        first, second = cases.build_catalog(211), cases.build_catalog(211)
        self.assertEqual([c.raw_columns for c in first], [c.raw_columns for c in second])
        named = {case.name: case for case in first}
        renamed = named["clean_panel_renamed"]
        for new, old in renamed.renamed_from.items():
            self.assertEqual(renamed.raw_columns[new], named["clean_panel"].raw_columns[old])

    def test_truth_is_never_part_of_the_proposer_call(self):
        for case in cases.build_catalog(211):
            self.assertNotIn("truth", json.dumps(case.call))
            self.assertEqual(len(case.truth), len(next(iter(case.raw_columns.values()))))

    def test_scaling_case_has_the_requested_rows(self):
        case = cases.scaling_case(1000, 211)
        self.assertEqual(len(case.raw_columns["period"]), 1000)


if __name__ == "__main__":
    unittest.main()
