import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import building_history_check as bh
from agent.building_history import HistoryEncoder


def frame(rows) -> pd.DataFrame:
    """rows: (year, name, y, floor_area, year_built) -> text frame as read from CSV, then typed."""
    data = [{"State_Factor": "State_1", "building_class": "Commercial", "facility_type": "Office",
             "floor_area": str(area), "year_built": str(built), "ELEVATION": "2.4",
             bh.YEAR: str(year), bh.TARGET: str(y), "name": name}
            for year, name, y, area, built in rows]
    return bh.typed(pd.DataFrame(data))


def make_task(root: Path) -> Path:
    """A task folder with the files offline_baseline expects: 5 train years, test = year 6."""
    rng = np.random.default_rng(0)
    n = 300
    buildings = pd.DataFrame({
        "State_Factor": rng.choice(["State_1", "State_2"], n), "building_class": rng.choice(["Commercial", "Residential"], n),
        "facility_type": rng.choice(["Office", "Retail", "School"], n), "floor_area": 10000.0 + 137 * np.arange(n),
        "year_built": (1950 + np.arange(n) % 60).astype(float), "ELEVATION": (np.arange(n) % 7) * 1.5,
        "base": rng.uniform(20, 200, n)})
    buildings.loc[[5, 6], ["year_built"]] = np.nan  # unknown year_built
    buildings.loc[7, "year_built"] = 0.0
    buildings.loc[11, ["State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION"]] = \
        buildings.loc[10, ["State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION"]].to_numpy()
    rows = []
    for year in range(1, 7):
        present = buildings[rng.random(n) < 0.6]
        rows.append(present.assign(**{bh.YEAR: float(year), "temp": rng.normal(size=len(present)),
                                      bh.TARGET: present["base"] + rng.normal(scale=2, size=len(present))}))
    everything = pd.concat(rows, ignore_index=True).drop(columns="base")
    everything.insert(0, "id", np.arange(len(everything)))
    train, test = everything[everything[bh.YEAR] < 6], everything[everything[bh.YEAR] == 6].drop(columns=bh.TARGET)
    inputs = root / "regression" / "participants"
    inputs.mkdir(parents=True)
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], bh.TARGET: 0.0}).to_csv(inputs / "sample_submission.csv", index=False)
    dictionary = [("id", "id", "numeric"), (bh.TARGET, "target", "numeric"), (bh.YEAR, "feature", "numeric"),
                  ("State_Factor", "feature", "category"), ("building_class", "feature", "category"),
                  ("facility_type", "feature", "category"), ("floor_area", "feature", "numeric"),
                  ("year_built", "feature", "numeric"), ("ELEVATION", "feature", "numeric"), ("temp", "feature", "numeric")]
    pd.DataFrame(dictionary, columns=["feature", "role", "dtype"]).to_csv(inputs / "data_dictionary.csv", index=False)
    return inputs


class KeyTests(unittest.TestCase):
    def test_exact_structured_keys_match_the_encoder_without_rounding_or_delimiter_collisions(self):
        data = frame([(1, "a", 10, "61242.123456789012345", "1950"),
                      (2, "b", 20, "61242.123456789012346", "1950"),
                      (1, "c", 30, "9007199254740992", "1950"),
                      (2, "d", 40, "9007199254740993", "1950"),
                      (1, "e", 50, "100", "1950"), (2, "f", 60, "100", "1950")])
        data.loc[4, ["State_Factor", "building_class"]] = ["a|b", "c"]
        data.loc[5, ["State_Factor", "building_class"]] = ["a", "b|c"]
        strict = bh.make_keys(data, "strict")
        self.assertEqual(strict.nunique(), len(data))
        encoder = HistoryEncoder(bh.KEY_COLUMNS, bh.YEAR, bh.NUMERIC_KEY_COLUMNS, ("year_built",))
        self.assertEqual(strict.tolist(), encoder._rows(data)[0])
        self.assertEqual(encoder.fit_transform(data, data[bh.TARGET])["history_present"].tolist(), [0] * len(data))

    def test_missing_marker_never_collides_with_literal_question_mark(self):
        data = frame([(1, "a", 10, "100", "1950"), (2, "b", 20, "100", "1950"),
                      (3, "c", 30, "100", "1950"), (4, "d", 40, "100", "1950")])
        data["State_Factor"] = ["?", "", None, np.nan]
        strict, lenient = bh.make_keys(data, "strict"), bh.make_keys(data, "lenient")
        self.assertIsNotNone(strict.iloc[0])
        self.assertTrue(all(value is None for value in strict.iloc[1:]))
        self.assertNotEqual(lenient.iloc[0], lenient.iloc[1])
        self.assertEqual(lenient.iloc[1], lenient.iloc[2])
        self.assertEqual(lenient.iloc[2], lenient.iloc[3])

    def test_numbers_are_compared_as_numbers_and_unknown_year_built_follows_the_variant(self):
        rows = [(1, "a", 10, "61242.0", "1942.0"), (2, "a", 11, "61242", "1942"),   # same building, different formatting
                (1, "b", 20, "5000", ""), (2, "c", 21, "5000", "0"),                 # unknown year_built: empty and zero
                (1, "d", 30, "7000", "1950")]
        data = frame(rows)

        strict, lenient = bh.make_keys(data, "strict"), bh.make_keys(data, "lenient")

        self.assertEqual(strict.iloc[0], strict.iloc[1])
        self.assertTrue(strict.iloc[2] is None and strict.iloc[3] is None)
        self.assertEqual(lenient.iloc[2], lenient.iloc[3])  # unknown year_built matches itself only when lenient
        self.assertNotEqual(lenient.iloc[2], lenient.iloc[4])
        self.assertTrue(bh.unknown_year_built(data).tolist() == [False, False, True, True, False])
        with self.assertRaises(ValueError):
            bh.make_keys(data, "other")

    def test_year_table_counts_repeats_inside_one_year(self):
        data = frame([(1, "a", 10, 100, 1950), (1, "b", 20, 100, 1950), (1, "c", 30, 100, 1950),  # one key, three rows
                      (1, "d", 40, 200, 1960), (2, "d", 41, 200, 1960), (2, "e", 50, 300, "")])
        table = {row["year"]: row for row in bh.year_table(data, bh.make_keys(data, "strict"))}

        self.assertEqual(table[1]["keys_repeated"], 1)
        self.assertEqual(table[1]["rows_in_repeated_keys"], 3)
        self.assertAlmostEqual(table[1]["share_in_repeated_keys"], 0.75)
        self.assertEqual(table[1]["repeated_keys_with_identical_target"], 0)
        self.assertEqual(table[2]["keys_repeated"], 0)
        self.assertEqual((table[2]["without_key"], table[2]["unknown_year_built"]), (1, 1))


class HistoryMeanTests(unittest.TestCase):
    def setUp(self):
        # year 1: A, B, D, and E twice with one key; year 2: A, B, C (new), D twice with one key, E.
        self.train = frame([(1, "A", 10, 1, 1950), (1, "B", 100, 2, 1950), (1, "D", 40, 4, 1950),
                            (1, "E", 60, 5, 1950), (1, "E", 80, 5, 1950),
                            (2, "A", 12, 1, 1950), (2, "B", 90, 2, 1950), (2, "C", 50, 3, 1950),
                            (2, "D", 44, 4, 1950), (2, "D", 30, 4, 1950), (2, "E", 70, 5, 1950)])
        self.key = bh.make_keys(self.train, "strict")

    def test_rmse_overall_and_by_match_quality(self):
        fold = bh.evaluate_fold(self.train, self.key, 2, bootstrap=0)

        self.assertEqual((fold["fit_rows"], fold["constant"]), (5, 58.0))  # mean of 10, 100, 40, 60, 80
        # predictions: A 10, B 100, C 58 (no history), D 40 and 40, E 70 (mean of its two rows in year 1)
        self.assertAlmostEqual(fold["all_rows"]["rmse_history_mean"], math.sqrt((4 + 100 + 64 + 16 + 100 + 0) / 6))
        self.assertAlmostEqual(fold["all_rows"]["rmse_constant"], math.sqrt((2116 + 1024 + 64 + 196 + 784 + 144) / 6))
        self.assertEqual(fold["with_history"]["rows"], 5)
        self.assertAlmostEqual(fold["with_history"]["rmse_history_mean"], math.sqrt(220 / 5))
        self.assertAlmostEqual(fold["with_history"]["rmse_constant"], math.sqrt(4264 / 5))
        self.assertEqual(fold["clean"]["rows"], 2)  # A and B: one row per year on both sides
        self.assertAlmostEqual(fold["clean"]["rmse_history_mean"], math.sqrt((4 + 100) / 2))
        self.assertEqual(fold["ambiguous"]["rows"], 3)  # D (two rows in year 2) and E (two rows in year 1)
        self.assertAlmostEqual(fold["ambiguous"]["rmse_history_mean"], math.sqrt((16 + 100 + 0) / 3))
        self.assertEqual(fold["no_history"]["rows"], 1)
        self.assertEqual(fold["no_history"]["rmse_history_mean"], fold["no_history"]["rmse_constant"])  # constant is used
        self.assertAlmostEqual(fold["rmse_history_mean_only_when_clean"], math.sqrt((4 + 100 + 64 + 196 + 784 + 144) / 6))
        self.assertTrue(all(math.isnan(v) for v in fold["bootstrap_gain_all_rows"]))

    def test_validation_year_never_leaks_into_the_history(self):
        changed = self.train.copy()
        changed.loc[changed[bh.YEAR] == 2, bh.TARGET] += 1000  # validation answers must not change the predictions

        first = bh.evaluate_fold(self.train, self.key, 2, bootstrap=0)["all_rows"]
        second = bh.evaluate_fold(changed, self.key, 2, bootstrap=0)["all_rows"]

        self.assertNotAlmostEqual(first["rmse_history_mean"], second["rmse_history_mean"])  # only the answers moved
        self.assertEqual(bh.evaluate_fold(changed, self.key, 2, bootstrap=0)["constant"], 58.0)

    def test_bootstrap_interval_is_positive_when_history_is_clearly_better(self):
        rng = np.random.default_rng(1)
        constant_error, history_error = rng.normal(50, 5, 400) ** 2, rng.normal(10, 2, 400) ** 2
        low, high = bh.paired_bootstrap(constant_error, history_error, 1, 200)
        self.assertTrue(0 < low < high)

    def test_coverage_of_validation_and_test_rows(self):
        test = frame([(6, "A", 0, 1, 1950), (6, "C", 0, 3, 1950), (6, "Z", 0, 99, 1950), (6, "D", 0, 4, 1950), (6, "D", 0, 4, 1950)])
        test_key = bh.make_keys(test, "strict")

        coverage = bh.test_coverage(self.train, test, self.key, test_key)

        self.assertAlmostEqual(coverage["share_with_history"], 4 / 5)  # A, C, D, D; only Z has no history
        self.assertAlmostEqual(coverage["share_with_key_in_year"][1], 3 / 5)
        self.assertAlmostEqual(coverage["share_with_key_in_year"][2], 4 / 5)
        self.assertAlmostEqual(coverage["share_clean"], 2 / 5)  # A and C; both D rows are ambiguous
        self.assertAlmostEqual(coverage["share_by_number_of_years_with_history"][0], 1 / 5)
        valid = bh.validation_coverage(self.train, self.key, 2)
        self.assertEqual((valid["rows"], valid["with_history"]), (6, 5))


class EvidenceTests(unittest.TestCase):
    def test_consecutive_year_correlation_separates_same_key_from_control(self):
        rng = np.random.default_rng(2)
        rows = []
        for i in range(200):
            level = rng.uniform(10, 200)
            kind = "Office" if i % 2 else "Retail"
            for year, factor in ((1, 1.0), (2, 1.1)):
                rows.append({"State_Factor": "State_1", "building_class": "Commercial", "facility_type": kind,
                             "floor_area": str(1000 + i), "year_built": "1950", "ELEVATION": "1",
                             bh.YEAR: str(year), bh.TARGET: str(level * factor)})
        data = bh.typed(pd.DataFrame(rows))

        result = bh.consecutive_year_correlation(data, bh.make_keys(data, "strict"))

        self.assertEqual(result["pairs"], 200)
        self.assertGreater(result["spearman_same_key"], 0.99)
        self.assertLess(result["spearman_control_same_facility_type"], 0.5)


class ScriptTests(unittest.TestCase):
    def test_baseline_reference_limits_previously_initialized_pools_during_fit_and_predict(self):
        import offline_baseline as baseline
        from threadpoolctl import threadpool_info, threadpool_limits

        observed = []

        class CheckingModel:
            def fit(self, X, y):
                observed.append(("fit", threadpool_info()))
                return self

            def predict(self, X):
                observed.append(("predict", threadpool_info()))
                return np.zeros(len(X))

        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            train, _ = bh.load(inputs)
            with threadpool_limits(limits=12), mock.patch.object(baseline, "make_pipeline", return_value=CheckingModel()):
                bh.baseline_reference(inputs, train, bh.make_keys(train, "strict"), bootstrap=0)
        self.assertEqual([stage for stage, _ in observed], ["fit", "predict"] * 2)
        for stage, pools in observed:
            self.assertTrue(pools, stage)
            self.assertTrue(all(pool["num_threads"] <= 8 for pool in pools), (stage, pools))

    def test_full_report_on_a_synthetic_task_and_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            train, test = bh.load(inputs)
            report = bh.analyze(train, test, bootstrap=20)
            output, path = io.StringIO(), Path(directory) / "report.json"
            with mock.patch("sys.argv", ["bh", "--data-dir", str(inputs), "--json", str(path), "--bootstrap", "20"]), \
                    contextlib.redirect_stdout(output):
                code = bh.main()
            data = json.loads(path.read_text(encoding="utf-8"))  # allow_nan=False: the file is strict JSON
            with mock.patch("sys.argv", ["bh", "--data-dir", str(Path(directory) / "missing")]), \
                    contextlib.redirect_stderr(io.StringIO()):
                missing_code = bh.main()

        self.assertEqual(report["years"], {"train": [1, 2, 3, 4, 5], "test": [6]})
        strict, lenient = report["variants"]["strict"], report["variants"]["lenient"]
        self.assertLess(strict["key_summary"]["keyed_rows"], lenient["key_summary"]["keyed_rows"])  # unknown year_built
        self.assertTrue(any(row["keys_repeated"] for row in strict["train_years"]))  # planted twin building
        fold = strict["folds"][-1]
        self.assertEqual(fold["validation_year"], 5)
        self.assertLess(fold["with_history"]["rmse_history_mean"], fold["with_history"]["rmse_constant"] / 2)  # planted signal
        self.assertGreater(strict["test_coverage"]["share_with_history"], 0.5)
        self.assertEqual(code, 0)
        self.assertIn("main check", output.getvalue())
        self.assertEqual(data["variants"]["strict"]["folds"][-1]["validation_year"], 5)
        self.assertEqual(missing_code, 1)

    def test_baseline_reference_runs_on_the_project_model(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            train, _ = bh.load(inputs)

            results = bh.baseline_reference(inputs, train, bh.make_keys(train, "strict"), bootstrap=20)

        self.assertEqual([r["features"].split(",")[0].split(" ")[0] for r in results], ["all", "without"])
        for ref in results:
            self.assertEqual(ref["validation_year"], 5)
            self.assertGreater(ref["rows_with_history"], 0)
            self.assertTrue(all(math.isfinite(ref[name]) for name in (
                "rmse_model_all", "rmse_model_with_history", "rmse_history_with_history", "rmse_blend_with_history")))
            self.assertLess(ref["rmse_history_with_history"], ref["rmse_model_with_history"])  # planted building effect


class ReportFileTests(unittest.TestCase):
    def run_main(self, inputs: Path, target: Path) -> tuple[int, str]:
        errors = io.StringIO()
        with mock.patch("sys.argv", ["bh", "--data-dir", str(inputs), "--json", str(target), "--bootstrap", "20"]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            return bh.main(), errors.getvalue()

    def test_json_over_train_csv_fails_and_keeps_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            before = {path.name: path.read_bytes() for path in inputs.iterdir()}

            code, message = self.run_main(inputs, inputs / "train.csv")

            self.assertEqual(code, 1)
            self.assertIn("train.csv", message)
            self.assertEqual({path.name: path.read_bytes() for path in inputs.iterdir()}, before)

    def test_report_through_a_hard_link_or_symlink_to_an_input_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            train_bytes = (inputs / "train.csv").read_bytes()
            hard = Path(directory) / "hard.json"
            os.link(inputs / "train.csv", hard)
            self.assertEqual(self.run_main(inputs, hard)[0], 1)
            self.assertEqual(hard.read_bytes(), train_bytes)

            soft = Path(directory) / "soft.json"
            try:
                soft.symlink_to(inputs / "train.csv")
            except (OSError, NotImplementedError):
                pass  # symbolic links need a privilege on Windows
            else:
                self.assertEqual(self.run_main(inputs, soft)[0], 1)
            self.assertEqual((inputs / "train.csv").read_bytes(), train_bytes)

    def test_report_needs_the_json_extension_and_a_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            for name in ("report.txt", "report.csv", "report"):
                target = Path(directory) / name
                self.assertEqual(self.run_main(inputs, target)[0], 1, name)
                self.assertFalse(target.exists(), name)
            self.assertEqual(self.run_main(inputs, Path(directory))[0], 1)

    def test_failed_write_keeps_the_previous_report_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            target.write_text('{"old": true}\n', encoding="utf-8")

            with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                bh.write_json_atomically(target, {"new": 1})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            with mock.patch("json.dumps", side_effect=ValueError("not JSON")), self.assertRaises(ValueError):
                bh.write_json_atomically(target, {"new": 1})

            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["report.json"])

    def test_successful_write_replaces_the_report_in_one_step(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_task(Path(directory))
            target = Path(directory) / "report.json"
            target.write_text("old", encoding="utf-8")

            code, _ = self.run_main(inputs, target)

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["years"], {"train": [1, 2, 3, 4, 5], "test": [6]})
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["regression", "report.json"])


if __name__ == "__main__":
    unittest.main()
