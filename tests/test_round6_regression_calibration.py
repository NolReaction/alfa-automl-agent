"""Model-free checks of group boundaries, fixed residual policies and B vetoes."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from agent import round6_regression_calibration as calibration


class Round6CalibrationTests(unittest.TestCase):
    def test_raw_groups_ignore_labels_ids_indices_and_row_or_column_order(self):
        frame = pd.DataFrame({"value": np.repeat(np.arange(500), 2),
            "kind": np.repeat(["kind_" + str(i % 5) for i in range(500)], 2),
            "id": np.arange(1000), "untrusted_target": np.arange(1000) * 100})
        frame.index = np.arange(1000) * 17
        a, b, evidence = calibration.split_calibration(frame, ["value", "kind"])
        self.assertTrue(evidence["supported"])
        assigned = np.zeros(len(frame), dtype=bool); assigned[a] = True
        np.testing.assert_array_equal(assigned[::2], assigned[1::2])
        np.testing.assert_array_equal(np.sort(np.concatenate([a, b])), np.arange(len(frame)))
        changed = frame.iloc[::-1].copy()
        changed["id"] = "unrelated"
        changed["untrusted_target"] = np.nan
        changed.index = ["duplicate-index"] * len(changed)
        a2, b2, evidence2 = calibration.split_calibration(changed, ["kind", "value"])
        assigned2 = np.zeros(len(changed), dtype=bool); assigned2[a2] = True
        np.testing.assert_array_equal(assigned, assigned2[::-1])
        self.assertEqual(evidence, evidence2)
        self.assertFalse(evidence["labels_id_index_used"])
        self.assertFalse(evidence["row_groups_exported"])
        self.assertFalse(any(isinstance(value, (list, dict)) for value in evidence.values()))
        with self.assertRaises(ValueError):
            calibration.split_calibration(frame, ["value", "value"])

    def test_fixed_shift_factors_target_std_clipping_and_order_invariance(self):
        predictions = np.linspace(-100., 100., 200)
        truth = predictions + 1.
        for arm, factor in (("S20", .2), ("S50", .5)):
            with self.subTest(arm=arm):
                policy = calibration.fit_correction(arm, truth, predictions)
                self.assertEqual(policy["status"], "fitted")
                self.assertAlmostEqual(policy["offset"], factor)
                self.assertEqual(policy["slope"], 1.)
                self.assertEqual(policy, calibration.fit_correction(arm, truth[::-1], predictions[::-1]))
                np.testing.assert_allclose(calibration.apply_correction(predictions, policy), predictions + factor)
                clipped = calibration.fit_correction(arm, predictions + 1000., predictions)
                self.assertAlmostEqual(clipped["offset"], .1 * np.std(predictions, ddof=0))
                negative = calibration.fit_correction(arm, predictions - 1000., predictions)
                self.assertAlmostEqual(negative["offset"], -.1 * np.std(predictions, ddof=0))

    def test_affine_ridge_penalizes_intercept_and_clips_effective_slope(self):
        predictions = np.arange(200, dtype=float)
        centered = predictions - predictions.mean()
        truth = predictions + 2. + .3 * centered
        policy = calibration.fit_correction("A20", truth, predictions)
        self.assertAlmostEqual(policy["ridge_intercept"], 2. / 5.)
        self.assertAlmostEqual(policy["ridge_normalized_slope"], .3 * predictions.std(ddof=0) / 5.)
        self.assertAlmostEqual(policy["offset"], .4)
        self.assertAlmostEqual(policy["slope"], 1.06)
        expected = predictions + .4 + .06 * centered
        np.testing.assert_allclose(calibration.apply_correction(predictions, policy), expected)
        clipped = calibration.fit_correction("A20", 6. * predictions + 1000., predictions)
        self.assertEqual(clipped["slope"], 1.1)
        self.assertAlmostEqual(clipped["offset"], .1 * np.std(6. * predictions + 1000.))

    def test_insufficient_constant_nonfinite_and_unstable_values_return_identity(self):
        predictions = np.arange(200, dtype=float)
        cases = [("S20", np.ones(200), predictions),
                 ("S50", predictions[:99], predictions[:99]),
                 ("A20", predictions, np.ones(200)),
                 ("S20", np.r_[np.nan, predictions[1:]], predictions),
                 ("A20", predictions, np.r_[np.inf, predictions[1:]]),
                 ("A20", np.full(200, 1e308), predictions)]
        for arm, truth, estimate in cases:
            with self.subTest(arm=arm, first=truth[0]):
                policy = calibration.fit_correction(arm, truth, estimate)
                self.assertEqual(policy["status"], "identity")
                np.testing.assert_array_equal(calibration.apply_correction(predictions, policy), predictions)
        for arm in ("I", "S25", None):
            with self.assertRaises(ValueError):
                calibration.fit_correction(arm, predictions, predictions)
        with self.assertRaises(ValueError):
            calibration.apply_correction([np.nan], {"status": "identity"})

    def test_B_category_veto_overrides_total_gain_and_does_not_refit_A(self):
        earlier = pd.DataFrame({"period": ["1"] * 400,
            "kind": ["known_bad"] * 200 + ["known_good"] * 200,
            "value": np.arange(400).astype(str)})
        held = pd.DataFrame({"period": ["2"] * 400,
            "kind": ["known_bad"] * 100 + ["known_good"] * 300,
            "value": np.arange(400).astype(str)})
        earlier_y = np.arange(400, dtype=float)
        prediction_A = np.linspace(-100., 100., 200)
        policy = calibration.fit_correction("S20", prediction_A + 5., prediction_A)
        before = deepcopy(policy)
        prediction_B = np.zeros(400)
        target_B = np.r_[np.full(100, -1.), np.full(300, 5.)]
        result = calibration.gate_correction(earlier, earlier_y, held, target_B,
            prediction_B, policy, period_column="period", categorical=["kind"])
        self.assertGreater(result["sse_gain"], 0.)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "significant_segment_veto")
        bad = next(row for row in result["violations"] if row["segment"] == "level:known_bad")
        self.assertEqual(bad["rows"], 100)
        self.assertEqual(bad["identity_sse"], 100.)
        self.assertEqual(bad["corrected_sse"], 400.)
        self.assertEqual(result["history_gate"], "unknown")
        self.assertTrue(result["unknown_segments"])
        accepted = calibration.gate_correction(earlier, earlier_y, held, np.full(400, 5.),
            prediction_B, policy, period_column="period", categorical=["kind"])
        self.assertTrue(accepted["accepted"])
        self.assertFalse(accepted["parameter_refit_after_B"])
        self.assertEqual(policy, before)
        cuts1 = next(row["cutpoints"] for row in result["segments"] if row["partition"] == "target")
        cuts2 = next(row["cutpoints"] for row in accepted["segments"] if row["partition"] == "target")
        self.assertEqual(cuts1, cuts2)
        self.assertEqual(cuts1["p50"], float(np.quantile(earlier_y, .5)))
        for row in accepted["segments"]:
            if row["veto_partition"] and row["rows"] < 100:
                self.assertEqual(row["status"], "unknown")

    def test_B_ties_and_insufficient_support_keep_identity(self):
        frame = pd.DataFrame({"period": ["1"] * 200, "value": np.arange(200).astype(str)})
        policy = {"status": "fitted", "center": 0., "offset": 0., "slope": 1.}
        result = calibration.gate_correction(frame, np.arange(200.), frame,
            np.arange(200.), np.zeros(200), policy, period_column="period", categorical=[])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "B_practical_tie_identity")
        small = calibration.gate_correction(frame, np.arange(200.), frame.iloc[:99],
            np.arange(99.), np.zeros(99), policy, period_column="period", categorical=[])
        self.assertFalse(small["accepted"])
        self.assertEqual(small["status"], "insufficient_support")
        self.assertEqual(small["history_gate"], "unknown")


if __name__ == "__main__":
    unittest.main()
