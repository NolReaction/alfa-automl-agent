"""Host diagnostics use frozen feature masks and fitting-only boundaries."""

import copy
import json
import math
import unittest

import numpy as np
import pandas as pd

from experiment_rescue_regression_diagnostics import segment_rows


REFERENCE = {"key_columns": ["entity"], "period_column": "period", "numeric_columns": [],
             "positive_columns": [], "ambiguity_policy": "past_unique_only"}


def fixture():
    """Small raw frames cover every history status without fitting a model."""
    fit = pd.DataFrame({
        "entity": ["steady"] * 3 + ["repeated"] * 2 + [f"fit-{i}" for i in range(19)],
        "period": ["1", "2", "3", "2", "2"] + ["1"] * 19,
        "size": [str(i) for i in range(1, 25)],
        "color": ["common"] * 20 + ["rare"] * 4,
        "zone": ["zcommon"] * 22 + ["zrare"] * 2,
    })
    held = pd.DataFrame({
        "entity": ["steady", "steady", "repeated", "new-key", "steady", None, "steady", "steady"],
        "period": ["4", "5", "3", "4", "1", "4", "invalid", "2"],
        "size": ["1", "12.5", "12.6", "21.7", "21.8", "100", None, "invalid"],
        "color": ["common", "rare", "unseen", "rare", None, "common", "common", "common"],
        "zone": ["zcommon", "zcommon", "zrare", "znew", None, "zcommon", "zcommon", "zcommon"],
    })
    for number in range(6):
        fit[f"feature_{number}"] = "present"
        held[f"feature_{number}"] = "present"
        held.loc[6, f"feature_{number}"] = None
    held.loc[7, "feature_0"] = " "
    y_fit = np.arange(1.0, 25.0)
    y_held = np.array([1.0, 12.5, 12.6, 21.7, 21.8, 23.77, 23.78, 100.0])
    predictions = y_held - np.arange(1.0, 9.0)
    return fit, y_fit, held, y_held, predictions


def report(fit, y_fit, held, y_held, predictions, *, reference=REFERENCE):
    return segment_rows(fit, y_fit, held, y_held, predictions, period_column="period", size_column="size",
                        categorical=("color", "zone"), reference_history_config=reference)


def groups(rows, partition):
    selected = [row for row in rows if row["partition"] == partition]
    return {row["segment"]: row for row in selected}


def membership_summary(rows):
    return {(row["partition"], row["segment"]): (row["n"], row["row_share"]) for row in rows}


class RescueRegressionDiagnosticsTests(unittest.TestCase):
    def assert_group(self, rows, partition, segment, positions):
        row = groups(rows, partition)[segment]
        errors = np.asarray([position + 1.0 for position in positions])
        self.assertEqual(row["n"], len(positions))
        self.assertAlmostEqual(row["row_share"], len(positions) / 8)
        self.assertAlmostEqual(row["sse"], float(np.dot(errors, errors)))
        if positions:
            self.assertAlmostEqual(row["rmse"], math.sqrt(float(np.mean(errors ** 2))))
            self.assertAlmostEqual(row["mae"], float(np.mean(errors)))
        else:
            self.assertIsNone(row["rmse"])
            self.assertIsNone(row["mae"])

    def test_strict_reference_history_and_recency_masks(self):
        rows = report(*fixture())
        self.assert_group(rows, "history", "eligible", [0, 1, 7])
        self.assert_group(rows, "history", "ambiguous", [2])
        self.assert_group(rows, "history", "absent", [3, 4, 5, 6])
        for segment, positions in (("0", [3, 4, 5, 6]), ("1", [7]), ("2", [2]), ("3+", [0, 1])):
            self.assert_group(rows, "past_rows", segment, positions)
        for segment, positions in (("none", [3, 4, 5, 6]), ("one", [0, 2, 7]), ("two_or_more", [1])):
            self.assert_group(rows, "recency", segment, positions)

    def test_fit_quantile_bands_are_right_closed_and_size_invalid_is_missing(self):
        rows = report(*fixture())
        for segment, positions in (("<=p50", [0, 1]), ("(p50,p90]", [2, 3]),
                                   ("(p90,p99]", [4, 5]), (">p99", [6, 7])):
            self.assert_group(rows, "target", segment, positions)
        for segment, positions in (("<=p50", [0, 1]), ("(p50,p90]", [2, 3]),
                                   (">p90", [4, 5]), ("missing", [6, 7])):
            self.assert_group(rows, "size", segment, positions)

    def test_category_novelty_priority_rare_fit_threshold_and_missing_values(self):
        rows = report(*fixture())
        # Fit count exactly 20 is common; unseen wins over a rare other column.
        # Missing both declared categorical values is not evidence of novelty.
        self.assert_group(rows, "category_novelty", "new", [2, 3])
        self.assert_group(rows, "category_novelty", "rare", [1])
        self.assert_group(rows, "category_novelty", "common", [0, 4, 5, 6, 7])
        self.assert_group(rows, "category:color", "level:common", [0, 5, 6, 7])
        self.assert_group(rows, "category:color", "level:rare", [1, 3])
        self.assert_group(rows, "category:color", "other", [2, 4])
        self.assert_group(rows, "missing_features", "0", [0, 1, 2, 3])
        self.assert_group(rows, "missing_features", "1-5", [4, 5, 7])
        self.assert_group(rows, "missing_features", "6+", [6])

    def test_each_partition_conserves_rows_error_and_shares(self):
        rows = report(*fixture())
        required = {"overall", "history", "past_rows", "recency", "target", "size", "missing_features",
                    "category_novelty", "period", "category:color", "category:zone"}
        self.assertEqual({row["partition"] for row in rows}, required)
        overall = groups(rows, "overall")["all"]
        self.assertEqual(overall["n"], 8)
        self.assertEqual(overall["sse"], 204.0)
        for partition in required:
            with self.subTest(partition=partition):
                partition_rows = [row for row in rows if row["partition"] == partition]
                self.assertEqual(len({row["segment"] for row in partition_rows}), len(partition_rows))
                self.assertEqual(sum(row["n"] for row in partition_rows), overall["n"])
                self.assertAlmostEqual(sum(row["row_share"] for row in partition_rows), 1.0)
                self.assertAlmostEqual(sum(row["sse"] for row in partition_rows), overall["sse"])
                self.assertAlmostEqual(sum(row["sse_share"] for row in partition_rows), 1.0)
        self.assert_group(rows, "period", "missing", [6])
        self.assert_group(rows, "period", "4", [0, 3, 5])
        json.dumps(rows, allow_nan=False)

    def test_held_target_changes_cannot_change_feature_partition_masks(self):
        fit, y_fit, held, y_held, predictions = fixture()
        before = report(fit, y_fit, held, y_held, predictions)
        changed_y = np.full(len(held), -500.0)
        # Keep each row's error fixed as labels change: metrics now also expose
        # a mistaken reassignment between equal-sized feature segments.
        after = report(fit, y_fit, held, changed_y, changed_y - np.arange(1.0, 9.0))
        before_features = [row for row in before if row["partition"] != "target"]
        after_features = [row for row in after if row["partition"] != "target"]
        self.assertEqual(membership_summary(before_features), membership_summary(after_features))
        self.assertEqual(before_features, after_features)
        self.assertNotEqual(membership_summary([row for row in before if row["partition"] == "target"]),
                            membership_summary([row for row in after if row["partition"] == "target"]))

    def test_held_distribution_does_not_recompute_fit_target_or_size_boundaries(self):
        fit, y_fit, held, y_held, predictions = fixture()
        before = report(fit, y_fit, held, y_held, predictions)
        extra = held.iloc[[0]].copy()
        extra["size"] = "1000000000"
        larger = pd.concat([held, extra], ignore_index=True)
        after = report(fit, y_fit, larger, np.r_[y_held, 1e9], np.r_[predictions, 1e9])
        for partition, last_segment in (("target", ">p99"), ("size", ">p90")):
            for segment, old in groups(before, partition).items():
                new = groups(after, partition)[segment]
                self.assertEqual(new["n"], old["n"] + (segment == last_segment))
                self.assertEqual(new["sse"], old["sse"])

    def test_mapping_predictions_use_identical_masks_for_every_arm(self):
        fit, y_fit, held, y_held, predictions = fixture()
        rows = report(fit, y_fit, held, y_held, {"base": predictions, "alternative": y_held})
        self.assertEqual({row["arm"] for row in rows}, {"base", "alternative"})
        base = [row for row in rows if row["arm"] == "base"]
        alternative = [row for row in rows if row["arm"] == "alternative"]
        self.assertEqual(membership_summary(base), membership_summary(alternative))
        self.assertEqual(membership_summary(base), membership_summary(report(*fixture())))
        self.assertTrue(all(row["sse"] == 0.0 for row in alternative))
        json.dumps(rows, allow_nan=False)

    def test_none_reference_is_absent_for_every_row_and_preserves_other_masks(self):
        args = fixture()
        rows = report(*args, reference=None)
        self.assert_group(rows, "history", "absent", list(range(8)))
        self.assert_group(rows, "history", "eligible", [])
        self.assert_group(rows, "history", "ambiguous", [])
        self.assert_group(rows, "past_rows", "0", list(range(8)))
        self.assert_group(rows, "recency", "none", list(range(8)))
        feature_partitions = {"target", "size", "missing_features", "category_novelty", "period",
                              "category:color", "category:zone"}
        self.assertEqual(membership_summary([row for row in rows if row["partition"] in feature_partitions]),
                         membership_summary([row for row in report(*args) if row["partition"] in feature_partitions]))

    def test_duplicate_indices_are_positional_and_inputs_are_unchanged(self):
        fit, y_fit, held, y_held, predictions = fixture()
        expected = report(fit, y_fit, held, y_held, predictions)
        fit.index = ["same"] * len(fit)
        held.index = ["repeated", "repeated", "other", "other"] * 2
        y_fit = pd.Series(y_fit, index=fit.index)
        y_held = pd.Series(y_held, index=held.index)
        predictions = pd.Series(predictions, index=held.index)
        saved = copy.deepcopy((fit, y_fit, held, y_held, predictions, REFERENCE))
        actual = report(fit, y_fit, held, y_held, predictions)
        self.assertEqual(actual, expected)
        pd.testing.assert_frame_equal(fit, saved[0])
        pd.testing.assert_series_equal(y_fit, saved[1])
        pd.testing.assert_frame_equal(held, saved[2])
        pd.testing.assert_series_equal(y_held, saved[3])
        pd.testing.assert_series_equal(predictions, saved[4])
        self.assertEqual(REFERENCE, saved[5])

    def test_empty_groups_are_retained_and_json_finite(self):
        fit, y_fit, held, y_held, predictions = fixture()
        rows = report(fit, y_fit, held.iloc[[0]], y_held[:1], predictions[:1])
        self.assertEqual(set(groups(rows, "history")), {"eligible", "absent", "ambiguous"})
        self.assertEqual(set(groups(rows, "past_rows")), {"0", "1", "2", "3+"})
        self.assertEqual(set(groups(rows, "target")), {"<=p50", "(p50,p90]", "(p90,p99]", ">p99"})
        self.assertTrue(any(row["n"] == 0 for row in rows))
        for row in rows:
            if row["n"] == 0:
                self.assertEqual(row["sse"], 0.0)
                self.assertEqual(row["row_share"], 0.0)
                self.assertIsNone(row["rmse"])
                self.assertIsNone(row["mae"])
        json.dumps(rows, allow_nan=False)

    def test_empty_held_frame_has_null_error_metrics_and_row_shares(self):
        fit, y_fit, held, _, _ = fixture()
        rows = report(fit, y_fit, held.iloc[:0], np.array([]), np.array([]))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["n"], 0)
            self.assertEqual(row["sse"], 0.0)
            self.assertIsNone(row["row_share"])
            self.assertIsNone(row["rmse"])
            self.assertIsNone(row["mae"])
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
