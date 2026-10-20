import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from agent import profiling


def write_task(root: Path, name: str, folder: str, dictionary, train, test, sample, with_dictionary=True) -> Path:
    inputs = root / name / folder / "participants"
    inputs.mkdir(parents=True)
    if with_dictionary:
        pd.DataFrame(dictionary, columns=["feature", "role", "dtype"]).to_csv(inputs / "data_dictionary.csv", index=False)
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    sample.to_csv(inputs / "sample_submission.csv", index=False)
    (root / name / "task.md").write_text("# task\n", encoding="utf-8")
    return root / name / "task.md"


def regression_task(root: Path, with_dictionary=True) -> Path:
    rng = np.random.default_rng(0)
    n, n_test = 600, 150
    total = n + n_test
    signal = rng.normal(size=total)
    frame = pd.DataFrame({
        "row_id": np.arange(1, total + 1),
        "x_signal": signal,
        "x_missing": np.where(rng.random(total) < 0.25, "", rng.normal(size=total).round(3).astype(str)),
        "x_text": rng.normal(size=total).round(3).astype(str),
        "x_cat": rng.choice(["a", "b", "c"], size=total),
        "x_const": 7,
        "x_year": np.r_[rng.integers(1, 4, size=n), np.full(n_test, 4)],  # test is a later period than train
    })
    frame["x_dup"] = frame["x_signal"]
    frame.loc[:2, "x_text"] = "abc"
    frame["target"] = 3 * signal + rng.standard_t(2, size=total)
    frame["x_leak"] = frame["target"] * 2 + rng.normal(scale=0.01, size=total)
    train, test = frame.iloc[:n].copy(), frame.iloc[n:].drop(columns="target").reset_index(drop=True)
    features = [c for c in test.columns if c != "row_id"]
    dictionary = [("row_id", "id", "numeric"), ("target", "target", "numeric")] + [
        (c, "feature", "category" if c == "x_cat" else "numeric") for c in features]
    sample = pd.DataFrame({"row_id": test["row_id"], "target": 0.0})
    return write_task(root, "regression", "regression", dictionary, train, test, sample, with_dictionary)


def classification_task(root: Path) -> Path:
    rng = np.random.default_rng(1)
    labels = rng.permutation(np.array(["a"] * 800 + ["NA"] * 94 + ["c"] * 6))
    n, n_test = len(labels), 100
    # x_noise takes three values, so many rows repeat exactly, as in the real network-flow data.
    train = pd.DataFrame({"id": np.arange(n), "x_noise": rng.integers(0, 3, size=n),
                          "x_flag": (labels == "c").astype(int), "x_flag_copy": (labels == "c").astype(int),
                          "protocol": rng.choice(["tcp", "udp"], size=n), "verdict": labels})
    test = pd.DataFrame({"id": np.arange(n_test), "x_noise": rng.integers(0, 3, size=n_test), "x_flag": 0,
                         "x_flag_copy": 0, "protocol": rng.choice(["tcp", "udp"], size=n_test)})
    dictionary = [("id", "id", "numeric"), ("verdict", "target", "category"), ("x_noise", "feature", "numeric"),
                  ("x_flag", "feature", "binary"), ("x_flag_copy", "feature", "numeric"),
                  ("protocol", "feature", "category")]
    sample = pd.DataFrame({"id": test["id"], "verdict": "a"})
    return write_task(root, "classification", "multiclass", dictionary, train, test, sample)


def found(profile, area, columns=(), level=None):
    return [item for item in profile["findings"]
            if item["area"] == area and (level is None or item["level"] == level)
            and set(columns) <= set(item["columns"])]


class ProfilingTests(unittest.TestCase):
    def test_regression_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = profiling.profile_task(regression_task(Path(directory)))

        self.assertEqual(profile["target"]["kind"], "regression")
        self.assertEqual(profile["rows"], {"train": 600, "test": 150})
        self.assertTrue(found(profile, "shift", ["x_year"], "high"))  # period never seen in train
        self.assertIn("x_year", profile["validation_advice"])
        self.assertTrue(found(profile, "leakage", ["x_leak"], "high"))
        self.assertTrue(found(profile, "missing", ["x_missing"], "low"))
        self.assertTrue(found(profile, "types", ["x_text"], "medium"))  # numeric column with a text cell
        self.assertTrue(found(profile, "types", ["x_const"], "low"))
        self.assertTrue(any(set(item["columns"]) == {"x_signal", "x_dup"} for item in found(profile, "types")))
        self.assertGreater(profile["target"]["top_1pct_variance_share"], 0.05)
        self.assertEqual(profile["id"]["row_counter"], [False, False])
        scores = profile["leakage"]["single_feature_scores"].set_index("feature")["score"]
        self.assertGreater(scores["x_leak"], 0.9)
        self.assertAlmostEqual(profile["feature_table"].set_index("feature").loc["x_missing", "missing_train"], 0.25, delta=0.06)

    def test_classification_imbalance_ids_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = profiling.profile_task(classification_task(Path(directory)).parent)  # a directory works too

        target = profile["target"]
        self.assertEqual(target["kind"], "classification")
        self.assertEqual(target["counts"], {"a": 800, "NA": 94, "c": 6})  # "NA" stays a class, not a missing value
        self.assertAlmostEqual(target["majority_macro_f1"], 2 * (800 / 900) / (1 + 800 / 900) / 3)
        self.assertTrue(found(profile, "target", level="high"))
        self.assertTrue(found(profile, "target", ["c"], "medium"))
        self.assertTrue(found(profile, "leakage", ["x_flag"]))
        self.assertTrue(any(set(item["columns"]) == {"x_flag", "x_flag_copy"} for item in found(profile, "types")))
        self.assertEqual(profile["id"]["row_counter"], [True, True])
        self.assertEqual(profile["id"]["overlap"], 100)
        self.assertFalse(found(profile, "id", level="high"))  # counter IDs overlap by construction
        self.assertGreater(profile["duplicates"]["random_80_20_holdout_rows_with_twin_in_fit"], 0.1)
        self.assertTrue(found(profile, "duplicates", level="medium"))
        self.assertIn("identical feature rows", profile["validation_advice"])

    def test_text_is_bounded_and_keeps_high_priority_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = profiling.profile_task(regression_task(Path(directory)))

        full = profiling.format_profile(profile, 10_000)
        short = profiling.format_profile(profile, 900)
        self.assertLessEqual(len(short), 900)
        self.assertIn("[HIGH]", short)
        self.assertIn("omitted", short)
        self.assertNotIn("omitted", full)
        self.assertTrue(full.startswith("DATA PROFILE"))

    def test_infers_types_without_a_data_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = profiling.profile_task(regression_task(Path(directory), with_dictionary=False))

        self.assertEqual(profile["id_column"], "row_id")  # first column of sample_submission.csv
        self.assertEqual(profile["target"]["kind"], "regression")
        kinds = profile["feature_table"].set_index("feature")["kind"]
        self.assertEqual(kinds["x_signal"], "numeric")
        self.assertEqual(kinds["x_cat"], "category")

    def test_profile_text_never_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            message = profiling.profile_text(Path(directory) / "missing" / "task.md")
            ok = profiling.profile_text(regression_task(Path(directory)))

        self.assertTrue(message.startswith("DATA PROFILE unavailable"))
        self.assertTrue(ok.startswith("DATA PROFILE (computed"))

    def test_command_line_writes_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = regression_task(root)
            report = root / "profile.json"
            with mock.patch("sys.argv", ["profiling", str(task), "--json", str(report)]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(profiling.main(), 0)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertIn("findings", data)
            self.assertIn("DATA PROFILE", output.getvalue())
            with mock.patch("sys.argv", ["profiling", str(root / "nope.md")]), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(profiling.main(), 1)


if __name__ == "__main__":
    unittest.main()
