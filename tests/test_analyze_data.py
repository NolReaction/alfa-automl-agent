import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import analyze_data


def write_task(root: Path, name: str, folder: str, dictionary, train, test, sample) -> Path:
    inputs = root / name / folder / "participants"
    inputs.mkdir(parents=True)
    pd.DataFrame(dictionary, columns=["feature", "role", "dtype", "description"]).to_csv(
        inputs / "data_dictionary.csv", index=False)
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    sample.to_csv(inputs / "sample_submission.csv", index=False)
    return root / name


def regression_task(root: Path) -> Path:
    rng = np.random.default_rng(0)
    n, n_test = 600, 150
    signal = rng.normal(size=n + n_test)
    frame = pd.DataFrame({
        "row_id": np.arange(1, n + n_test + 1),
        "x_signal": signal,
        "x_missing": np.where(rng.random(n + n_test) < 0.25, "", rng.normal(size=n + n_test).round(3).astype(str)),
        "x_text": rng.normal(size=n + n_test).round(3).astype(str),
        "x_cat": rng.choice(["a", "b", "c"], size=n + n_test),
        "x_const": 7,
        "x_outlier": rng.normal(size=n + n_test),
        "x_shift": rng.normal(size=n + n_test) + np.r_[np.zeros(n), np.full(n_test, 3.0)],
        "x_year": np.r_[rng.integers(1, 4, size=n), np.full(n_test, 4)],  # test is a later period than train
    })
    frame["x_dup"] = frame["x_signal"]
    frame.loc[:2, "x_text"] = "abc"
    frame.loc[5, "x_outlier"] = 1e6
    frame["target"] = 3 * frame["x_signal"] + rng.normal(size=n + n_test)
    frame["x_leak"] = frame["target"] * 2 + rng.normal(scale=0.01, size=n + n_test)
    train, test = frame.iloc[:n].copy(), frame.iloc[n:].drop(columns="target").reset_index(drop=True)
    # Five train rows repeated with a different answer, and three test rows identical to train rows.
    repeats = train.iloc[:5].copy()
    repeats["row_id"] = np.arange(-5, 0)  # keep the train ID range disjoint from the test IDs
    repeats["target"] += 1
    train = pd.concat([train, repeats], ignore_index=True)
    features = [c for c in test.columns if c != "row_id"]
    test.loc[:2, features] = train.loc[10:12, features].to_numpy()
    dictionary = [("row_id", "id", "numeric", ""), ("target", "target", "numeric", "")] + [
        (c, "feature", "category" if c == "x_cat" else "numeric", "") for c in features]
    sample = pd.DataFrame({"row_id": test["row_id"], "target": 0.0})
    return write_task(root, "regression", "regression", dictionary, train, test, sample)


def classification_task(root: Path, counter_ids: bool = False) -> Path:
    rng = np.random.default_rng(1)
    labels = np.array(["a"] * 800 + ["NA"] * 94 + ["c"] * 6)  # sorted on purpose: the file is ordered by class
    n, n_test = len(labels), 100
    train = pd.DataFrame({"row_key": [str(i) if counter_ids else f"{i:04d}" for i in range(n)], "x_noise": rng.normal(size=n),
                          "x_flag": (labels == "c").astype(int), "x_code": pd.factorize(labels)[0],
                          "protocol": rng.choice(["tcp", "udp"], size=n), "verdict": labels})
    test = pd.DataFrame({"row_key": [str(i) if counter_ids else f"t{i:03d}" for i in range(n_test)], "x_noise": rng.normal(size=n_test),
                         "x_flag": 0, "x_code": 0, "protocol": rng.choice(["tcp", "udp", "icmp"], size=n_test)})
    dictionary = [("row_key", "id", "category", ""), ("verdict", "target", "category", ""),
                  ("x_noise", "feature", "numeric", ""), ("x_flag", "feature", "binary", ""),
                  ("x_code", "feature", "numeric", ""), ("protocol", "feature", "category", "")]
    sample = pd.DataFrame({"row_key": test["row_key"], "verdict": "a"})
    return write_task(root, "classification", "multiclass", dictionary, train, test, sample)


def find(report, area, columns=(), level=None):
    return [item for item in report["findings"]
            if item["area"] == area and (level is None or item["level"] == level)
            and set(columns) <= set(item["columns"])]


class AnalyzeDataTests(unittest.TestCase):
    def test_regression_problems_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            report, profile = analyze_data.analyze(regression_task(Path(directory)))

        self.assertEqual(report["schema"]["task_type"], "regression")
        self.assertTrue(find(report, "утечка", ["x_leak"], "critical"))
        self.assertTrue(find(report, "пропуски", ["x_missing"], "info"))
        self.assertTrue(find(report, "типы", ["x_text"], "warning"))
        self.assertTrue(find(report, "типы", ["x_const"], "warning"))
        self.assertTrue(any({"x_signal", "x_dup"} == set(item["columns"])
                            for item in find(report, "типы", level="warning")))  # identical columns
        self.assertTrue(find(report, "сдвиг", ["x_shift"], "critical"))
        self.assertTrue(any("отделён" in item["message"]  # test period never seen in train
                            for item in find(report, "сдвиг", ["x_year"], "critical")))
        self.assertTrue(find(report, "выбросы", ["x_outlier"]))
        self.assertTrue(find(report, "ID", level="info"))
        self.assertGreaterEqual(report["leakage"]["duplicates"]["conflicting_groups"], 1)
        self.assertGreaterEqual(report["leakage"]["duplicates"]["test_rows_with_unambiguous_train_match"], 0.01)
        scores = report["leakage"]["single_feature_scores"].set_index("feature")["score"]
        self.assertGreater(scores["x_leak"], 0.9)
        self.assertLess(abs(scores["x_const"]), 0.05)
        by_name = profile.set_index("feature")
        self.assertAlmostEqual(by_name.loc["x_missing", "missing_train"], 0.25, delta=0.06)
        self.assertGreater(by_name.loc["x_signal", "corr_spearman"], 0.8)

    def test_classification_imbalance_and_label_hygiene(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = analyze_data.analyze(classification_task(Path(directory)))

        target = report["target"]
        self.assertEqual(report["schema"]["task_type"], "classification")
        self.assertEqual(target["counts"], {"a": 800, "NA": 94, "c": 6})  # "NA" stays a class, not a missing value
        self.assertEqual(target["empty"], 0)
        self.assertGreater(target["imbalance_ratio"], 100)
        self.assertTrue(find(report, "дисбаланс", level="critical"))
        self.assertTrue(find(report, "дисбаланс", ["c"], "warning"))
        self.assertTrue(find(report, "утечка", ["x_code"], "critical"))  # copy of the class label
        self.assertTrue(find(report, "утечка", ["x_flag"]))  # identifies only the rarest class
        self.assertTrue(any("порядок строк" in item["message"].lower()  # sorted file: row order predicts the class
                            for item in find(report, "утечка", level="warning")))
        self.assertTrue(find(report, "сдвиг", ["protocol"]) or find(report, "типы", ["protocol"]))
        self.assertAlmostEqual(target["majority_macro_f1"], 2 * (800 / 900) / (1 + 800 / 900) / 3)

    def test_random_macro_f1_accounts_for_class_imbalance(self):
        def random_macro_f1(counts):
            labels = pd.Series([label for label, n in counts.items() for _ in range(n)], name="verdict")
            codes, classes = pd.factorize(labels, sort=True)
            task = SimpleNamespace(train=labels.to_frame(), target="verdict", classification=True)
            return analyze_data.analyze_target(task, codes.astype(int), [str(c) for c in classes], [])[
                "uniform_random_macro_f1"]

        real = {"Normal": 28000, "Generic": 20000, "Exploits": 16696, "Fuzzers": 9092, "DoS": 6132,
                "Reconnaissance": 5245, "Analysis": 1000, "Backdoor": 873, "Shellcode": 567, "Worms": 65}
        self.assertAlmostEqual(random_macro_f1(real), 0.0734, places=3)  # not 1 / 10
        self.assertAlmostEqual(random_macro_f1({"a": 50, "b": 50}), 0.5)  # balanced classes: 1 / k

    def test_ids_that_are_row_counters_are_not_a_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = analyze_data.analyze(classification_task(Path(directory), counter_ids=True))

        self.assertEqual(report["ids"]["train_test_overlap"], 100)
        self.assertTrue(report["ids"]["train_is_row_counter"] and report["ids"]["test_is_row_counter"])
        self.assertFalse(find(report, "ID", level="critical"))
        self.assertTrue(find(report, "ID", level="info"))

    def test_reports_are_written_and_json_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, profile = analyze_data.analyze(regression_task(root))
            out = analyze_data.write_reports("regression", report, profile, root / "analysis" / "regression")
            text = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIn("findings", text)
            self.assertIn("Найденные проблемы", (out / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(list(pd.read_csv(out / "features.csv").columns), analyze_data.PROFILE_COLUMNS)

    def test_main_runs_both_tasks_and_fails_cleanly_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regression_task(root)
            classification_task(root)
            with mock.patch("sys.argv", ["analyze_data.py", "all", "--data-dir", str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(analyze_data.main(), 0)
            self.assertIn("regression", output.getvalue())
            self.assertTrue((root / "analysis" / "classification" / "report.md").is_file())
            with mock.patch("sys.argv", ["analyze_data.py", "all", "--data-dir", str(root / "missing")]), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analyze_data.main(), 1)


if __name__ == "__main__":
    unittest.main()
