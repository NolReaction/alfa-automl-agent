"""Round 5 data readiness: the full shift table and the fit-only OOF audit are correct and read-only."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import experiment_round5_data_readiness as readiness
from agent import round4_classification_probability as probability
from agent.round4_classification_candidate import coverage, make_recipe
from agent.round4_classification_probability import fixed_incumbent_recipe
from tests.test_round4_feature_passport import tree


CLASSES = ["alpha", "beta", "gamma"]
FEATURES = ["x1", "x2", "proto"]
# Keys that would carry row-level positions, group keys or category examples if a list ever reached a report.
ROW_KEYS = {"groups", "group_codes", "tuning_rows", "selection_rows", "fit_rows", "held_rows", "examples"}


def write_task(root: Path, name: str, dictionary, train: pd.DataFrame, test: pd.DataFrame, metric: str) -> Path:
    inputs = root / name / "inputs"
    inputs.mkdir(parents=True)
    pd.DataFrame(dictionary, columns=["feature", "role", "dtype", "description_ru"]).to_csv(
        inputs / "data_dictionary.csv", index=False)
    id_column = next(row[0] for row in dictionary if row[1] == "id")
    target = next(row[0] for row in dictionary if row[1] == "target")
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    pd.DataFrame({id_column: test[id_column], target: train[target].iloc[0]}).to_csv(
        inputs / "sample_submission.csv", index=False)
    (root / name / "task.md").write_text(f"Основная метрика: {metric}\n", encoding="utf-8")
    return root / name


def shift_task(root: Path, shifted: int = 25, name: str = "regression") -> Path:
    """More than 20 shifted numeric features, one spread-only change and categories with blanks."""
    rng = np.random.default_rng(3)
    n, n_test = 300, 120
    train = pd.DataFrame({"row_id": [f"IDSENTINEL-{i}" for i in range(n)]})
    test = pd.DataFrame({"row_id": [f"IDSENTINEL-t{i}" for i in range(n_test)]})
    for index in range(shifted):
        train[f"x{index:02d}"] = rng.normal(size=n).round(4)
        test[f"x{index:02d}"] = (rng.normal(size=n_test) + 1 + index / 10).round(4)
    # Same median, wider spread: a median shift cannot see it, KS can.
    train["x_spread"] = np.resize([-1, 0, 1], n)
    test["x_spread"] = np.resize([-5, 0, 5], n_test)
    train["proto"] = np.where(np.arange(n) < 15, "", rng.choice(["CATSENTINEL-a", "CATSENTINEL-b"], n))
    test["proto"] = np.where(np.arange(n_test) < 6, " ", rng.choice(["CATSENTINEL-a", "CATSENTINEL-new"], n_test))
    train["target"] = rng.normal(size=n).round(3)
    dictionary = [("row_id", "id", "category", ""), ("target", "target", "numeric", ""),
                  *[(f"x{index:02d}", "feature", "numeric", "") for index in range(shifted)],
                  ("x_spread", "feature", "numeric", ""), ("proto", "feature", "category", "")]
    return write_task(root, name, dictionary, train, test, "RMSE")


def fit_frame(counts=(300, 200, 100), classes=CLASSES, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(classes[:len(counts)], counts).astype(object)
    n = len(labels)
    frame = pd.DataFrame({"x1": rng.normal(size=n).round(4).astype(str),
                          "x2": rng.integers(0, 100_000, n).astype(str),
                          "proto": rng.choice(["CATSENTINEL-a", "CATSENTINEL-b"], n)})
    return frame, labels


def context(classes=CLASSES, variant="oof") -> dict:
    return {"features": list(FEATURES), "numeric": ["x1", "x2"], "categorical": ["proto"], "classes": list(classes),
            "task_type": "classification", "prediction_kind": "labels", "metric": {"name": "Macro F1"},
            "id_column": "row_id", "target_column": "label",
            "recipe": make_recipe(fixed_incumbent_recipe(), list(classes), variant=variant)}


def classification_task(root: Path, counts=(300, 200, 100), name: str = "classification") -> Path:
    frame, labels = fit_frame(counts)
    train = pd.concat([pd.DataFrame({"row_id": [f"IDSENTINEL-{i}" for i in range(len(labels))]}), frame,
                       pd.DataFrame({"label": labels})], axis=1)
    test, _ = fit_frame((40,), seed=9)
    test.loc[:5, "proto"] = "CATSENTINEL-new"
    test.insert(0, "row_id", [f"IDSENTINEL-t{i}" for i in range(len(test))])
    dictionary = [("row_id", "id", "category", ""), ("label", "target", "category", ""), ("x1", "feature", "numeric", ""),
                  ("x2", "feature", "numeric", ""), ("proto", "feature", "category", "")]
    return write_task(root, name, dictionary, train, test, "Macro F1")


def keys(value) -> set[str]:
    """Every key whose value is a list, at any depth."""
    if isinstance(value, dict):
        return ({key for key, item in value.items() if isinstance(item, list)}
                | {key for item in value.values() for key in keys(item)})
    if isinstance(value, list):
        return {key for item in value for key in keys(item)}
    return set()


def statuses(audit: dict) -> dict[str, dict]:
    return {check["id"]: check for check in audit["checks"]}


class ReadinessCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.root = self.base / "data"
        self.root.mkdir()

    def cli(self, *args: str) -> tuple[int, str]:
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = readiness.main(list(args))
        return code, errors.getvalue()

    def shift(self, task: str, output: Path, root: Path | None = None) -> dict:
        code, errors = self.cli("shift", "--data-dir", str(root or self.root), "--task", task, "--output-dir", str(output))
        self.assertEqual(code, 0, errors)
        return json.loads((output / "shift_report.json").read_text(encoding="utf-8"))

    def fit(self, task_dir: Path, output: Path, *extra: str) -> dict:
        inputs = task_dir / "inputs"
        code, errors = self.cli("fit", "--task-md", str(task_dir / "task.md"), "--dictionary",
                                str(inputs / "data_dictionary.csv"), "--fit-csv", str(inputs / "train.csv"),
                                "--output-dir", str(output), *extra)
        self.assertEqual(code, 0, errors)
        return json.loads((output / "oof_fit_report.json").read_text(encoding="utf-8"))

    @staticmethod
    def texts(output: Path, names) -> list[str]:
        return [(output / name).read_text(encoding="utf-8") for name in names]


class ShiftTableTests(ReadinessCase):
    def test_more_than_twenty_shifted_numeric_features_all_stay_in_the_table(self) -> None:
        shift_task(self.root)
        report = self.shift("regression", self.base / "out")
        rows = {row["name"]: row for row in report["features"]}
        self.assertEqual(report["feature_count"], 27)
        self.assertEqual(len(rows), 27)
        numeric = [row for row in rows.values() if row["dtype"] == "numeric"]
        self.assertEqual(len(numeric), 26)
        self.assertTrue(all(row["median_shift_score"] is not None and row["ks_train_test"] is not None for row in numeric))
        summary = report["summary"]
        self.assertEqual((summary["median_shift_in_profile"], summary["median_shift_outside_profile"]), (20, 6))
        self.assertEqual(sum(row["median_shift_in_profile"] is False for row in numeric), 6)
        self.assertEqual(report["source_caps"]["numeric_shift"], {"cap": 20, "returned": 20, "may_be_truncated": True})
        # The spread-only change keeps its median, so the median shift alone cannot report it.
        spread = rows["x_spread"]
        self.assertEqual((spread["train_median"], spread["test_median"], spread["median_shift_score"]), (0.0, 0.0, 0.0))
        self.assertGreaterEqual(spread["ks_train_test"], 0.3)
        self.assertIn("x_spread", summary["ks_warning_with_unchanged_median"])
        self.assertIn("x_spread", summary["ks_warning_outside_profile"])
        self.assertEqual(len((self.base / "out" / "shift_features.csv").read_text(encoding="utf-8").splitlines()), 28)

    def test_definitions_stay_separate_and_unknowns_carry_reasons(self) -> None:
        shift_task(self.root)
        rows = {row["name"]: row for row in self.shift("regression", self.base / "out")["features"]}
        proto = rows["proto"]
        # Empty cells: 5% in TRAIN, none in TEST. Blank-only cells: 6 TEST rows in 120 are a space.
        self.assertEqual((proto["train_missing_percent_empty"], proto["test_missing_percent_empty"]), (5.0, 0.0))
        self.assertEqual((proto["train_missing_share_blank"], proto["test_missing_share_blank"]), (0.05, 0.05))
        self.assertEqual(proto["new_category_count"], 2)  # " " and the new value, counted as raw distinct values
        self.assertGreater(proto["new_category_row_share"], 0)
        self.assertLess(proto["new_category_row_share"], 1)
        self.assertEqual(proto["null_reasons"]["ks_train_test"], "not_numeric_dtype")
        x00 = rows["x00"]
        self.assertNotEqual(x00["train_std_ddof0"], x00["train_std_ddof1"])
        self.assertEqual(x00["null_reasons"]["test_nonfinite"], "not_provided_by_sources")
        self.assertEqual(x00["null_reasons"]["numeric_code_unseen_share"], "more_than_20_train_values")
        self.assertEqual(x00["null_reasons"]["test_missing_percent_empty"], "change_below_1pp_not_reported")
        self.assertAlmostEqual(rows["x_spread"]["numeric_code_unseen_share"], 2 / 3)  # -5 and 5 are unseen codes
        for row in rows.values():
            self.assertEqual({field for field in readiness.SHIFT_FIELDS if row[field] is None}, set(row["null_reasons"]))

    def test_shift_reuses_the_profile_and_skips_the_heavy_analysis(self) -> None:
        shift_task(self.root)
        with mock.patch("analyze_data.analyze", side_effect=AssertionError("full analysis")), \
                mock.patch("analyze_data.analyze_leakage", side_effect=AssertionError("leakage")), \
                mock.patch("analyze_data.correlated_pairs", side_effect=AssertionError("correlations")):
            report = self.shift("regression", self.base / "out")
        self.assertTrue(report["crosscheck"]["matched_full_calculation"])
        drifted = mock.patch.object(readiness, "_numeric_shift",
                                    side_effect=lambda name, train, test: {"name": name, "train_median": 0.0,
                                                                          "test_median": 0.0, "score": 0.0})
        with drifted:
            code, errors = self.cli("shift", "--data-dir", str(self.root), "--task", "regression",
                                    "--output-dir", str(self.base / "out-2"))
        self.assertEqual(code, 1)
        self.assertIn("не совпал", errors)


class FitAuditTests(ReadinessCase):
    def test_compatible_fit_reports_support_folds_and_unknown_acceptance(self) -> None:
        X, y = fit_frame()
        audit = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(audit["decision"], "structurally_compatible", audit["reasons"])
        self.assertTrue(all(check["status"] == "passed" for check in audit["checks"]))
        self.assertEqual(audit["support"]["allowed"]["gamma"], {"rows": 100, "groups": 100})
        layout = audit["layout"]
        self.assertEqual(layout["A"]["rows"] + layout["B"]["rows"], 600)
        self.assertEqual(len(audit["support"]["folds"]), 3)
        self.assertEqual(layout["held_coverage"]["exactly_once"], layout["A"]["rows"])
        self.assertEqual(audit["model_acceptance"], "not_evaluated")
        self.assertEqual({item["status"] for item in audit["not_evaluated"]}, {"unknown"})
        self.assertIn("selection_macro_f1_gain", {item["id"] for item in audit["not_evaluated"]})
        self.assertNotIn("passed", json.dumps(audit["not_evaluated"]))
        self.assertFalse(keys(audit) & ROW_KEYS)

    def test_rare_or_absent_class_is_never_admitted(self) -> None:
        X, y = fit_frame((300, 200, 6))
        rare = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(rare["decision"], "ineligible")
        self.assertEqual(statuses(rare)["oof_support_gates"]["status"], "violated")
        self.assertEqual(rare["support"]["allowed"]["gamma"], {"rows": 6, "groups": 6})
        self.assertIn("gamma", next(item for item in rare["constraints"] if item["id"] == "B_min_class_rows")["classes_below"])

        X, y = fit_frame((300, 200, 100))
        absent = readiness.audit_oof_fit(X, y, context([*CLASSES, "delta"]))
        self.assertEqual(absent["decision"], "ineligible")
        self.assertEqual(statuses(absent)["oof_support_gates"]["reason"], "tuning_missing_class")
        self.assertIn("final_fit_side=fit_missing_class", statuses(absent)["child_input_caps"]["reason"])
        self.assertEqual(absent["support"]["allowed"]["delta"], {"rows": 0, "groups": 0})

        X, y = fit_frame((300, 200, 10))
        X.loc[y == "gamma", FEATURES] = ["1.0", "7", "CATSENTINEL-a"]  # the whole class is one raw-feature group
        single = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(single["decision"], "ineligible")
        self.assertEqual(statuses(single)["grouped_split"]["status"], "violated")
        self.assertIn("одной группе", single["split"]["refusal"])
        self.assertEqual(statuses(single)["oof_support_gates"]["status"], "not_reached")
        self.assertEqual(single["support"]["allowed"]["gamma"], {"rows": 10, "groups": 1})

    def test_duplicate_groups_with_conflicting_labels_never_cross_a_boundary(self) -> None:
        X, y = fit_frame()
        # Thirty copies of alpha rows relabelled beta, and ten exact alpha copies.
        X = pd.concat([X, X.iloc[:30], X.iloc[30:40]], ignore_index=True)
        y = np.concatenate([y, np.full(30, "beta", dtype=object), y[30:40]])
        audit = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(audit["decision"], "structurally_compatible", audit["reasons"])
        duplicates = audit["duplicates"]
        self.assertEqual((duplicates["duplicate_groups"], duplicates["extra_copies"]), (40, 40))
        self.assertEqual((duplicates["conflicting_label_groups"], duplicates["rows_in_conflicting_groups"]), (30, 60))
        self.assertEqual(audit["raw_feature_groups"], 600)
        layout = audit["layout"]
        self.assertEqual((layout["ab_group_overlap"], layout["ab_row_overlap"], layout["rows_outside_ab"]), (0, 0, 0))
        self.assertTrue(all(fold["group_overlap"] == 0 for fold in layout["folds"]))
        # Independent check on the canonical partition itself: every copy sits on one side of each boundary.
        partition = coverage(X, y, context())
        groups = np.asarray(partition["groups"])
        side = {}
        for part, rows in (("A", partition["tuning_rows"]), ("B", partition["selection_rows"])):
            for group in groups[rows]:
                self.assertEqual(side.setdefault(group, part), part)

    def test_a_leaking_partition_is_detected_not_trusted(self) -> None:
        X, y = fit_frame()
        X = pd.concat([X, X.iloc[:5]], ignore_index=True)
        y = np.concatenate([y, y[:5]])
        real = coverage(X, y, context())
        groups = np.asarray(real["groups"])
        A, B = list(real["tuning_rows"]), list(real["selection_rows"])
        copy = next(row for row in A if (groups == groups[row]).sum() > 1 and row >= 600)
        A.remove(copy)
        B.append(copy)  # one copy crosses to B while its twin stays in A
        leaking = {**real, "tuning_rows": A, "selection_rows": B,
                   "folds": [{"fit_rows": [r for r in fold["fit_rows"] if r != copy],
                              "held_rows": [r for r in fold["held_rows"] if r != copy]} for fold in real["folds"]]}
        with mock.patch.object(readiness, "coverage", return_value=leaking):
            audit = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(audit["decision"], "ineligible")
        self.assertEqual(audit["layout"]["ab_group_overlap"], 1)
        self.assertEqual(statuses(audit)["ab_disjoint"]["status"], "violated")
        self.assertIn("oof_contract_error", statuses(audit)["oof_support_gates"]["reason"])

    def test_wrong_schema_labels_class_order_and_size_give_clear_refusals(self) -> None:
        X, y = fit_frame()
        cases = {
            "порядке": (X.loc[:, ["x2", "x1", "proto"]], y, context()),
            "необъявленных": (X, np.where(y == "gamma", "omega", y), context()),
            "порядок классов": (X, y, {**context(), "classes": ["beta", "alpha", "gamma"]}),
            "variant": (X, y, context(variant="identity")),
            "Macro F1": (X, y, {**context(), "metric": {"name": "RMSE"}}),
            "непустыми": (X, np.where(y == "gamma", " ", y), context()),
            "ID и target": (X.rename(columns={"x1": "label"}), y,
                            {**context(), "features": ["label", "x2", "proto"], "numeric": ["label", "x2"]}),
        }
        for message, (frame, labels, declared) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                readiness.audit_oof_fit(frame, labels, declared)
        with self.assertRaisesRegex(ValueError, "recipe"):
            readiness.audit_oof_fit(X, y, {**context(), "recipe": {"version": 1}})
        with self.assertRaisesRegex(ValueError, "Too many declared classes"):
            context([f"c{i}" for i in range(51)])
        with mock.patch.dict(probability.CAPS, {"max_fit_rows": 500}):
            audit = readiness.audit_oof_fit(X, y, context())
        self.assertEqual(audit["decision"], "ineligible")
        self.assertIn("final_fit_side=row_cap_exceeded", statuses(audit)["child_input_caps"]["reason"])
        self.assertFalse(next(item for item in audit["constraints"] if item["id"] == "fit_rows")["satisfied"])

    def test_pure_audit_opens_no_file(self) -> None:
        X, y = fit_frame()
        declared = context()
        readiness.audit_oof_fit(X, y, declared)  # lazy imports load their modules before files are blocked
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")), \
                mock.patch.object(os, "open", side_effect=AssertionError("file opened")):
            audit = readiness.audit_oof_fit(X, y, declared)
        self.assertEqual(audit["decision"], "structurally_compatible")


class ExportTests(ReadinessCase):
    def test_test_changes_or_deletion_do_not_affect_fit_mode(self) -> None:
        task = classification_task(self.root)
        first = self.fit(task, self.base / "out-1")
        test_csv = task / "inputs" / "test.csv"
        test_csv.write_text(test_csv.read_text(encoding="utf-8").replace("CATSENTINEL-new", "CATSENTINEL-zzz"),
                            encoding="utf-8")
        changed = self.fit(task, self.base / "out-2")
        test_csv.unlink()
        deleted = self.fit(task, self.base / "out-3")
        for report in (changed, deleted):
            self.assertEqual(report["content_sha256"], first["content_sha256"])
            self.assertEqual(report["files"], first["files"])
        self.assertEqual(set(first["inputs"]), {"task.md", "data_dictionary.csv", "fit.csv"})
        self.assertEqual(first["audit"]["decision"], "structurally_compatible")

    def test_inputs_stay_byte_identical_and_outputs_hold_no_rows(self) -> None:
        shift_task(self.root)
        task = classification_task(self.root)
        before = tree(self.root)
        shift = self.shift("classification", self.base / "shift")
        fit = self.fit(task, self.base / "fit")
        self.assertEqual(tree(self.root), before)
        self.assertEqual(sorted(path.name for path in (self.base / "shift").iterdir()), sorted(readiness.SHIFT_FILES))
        self.assertEqual(sorted(path.name for path in (self.base / "fit").iterdir()), sorted(readiness.FIT_FILES))
        texts = self.texts(self.base / "shift", readiness.SHIFT_FILES) + self.texts(self.base / "fit", readiness.FIT_FILES)
        for text in texts:
            for forbidden in ("IDSENTINEL", "CATSENTINEL", str(self.base), self.base.name):
                self.assertNotIn(forbidden, text)
        self.assertFalse(keys(shift) & ROW_KEYS)
        self.assertFalse(keys(fit) & ROW_KEYS)
        self.assertEqual((shift["scope"], fit["scope"]), (readiness.SHIFT_SCOPE, readiness.FIT_SCOPE))
        self.assertIn("не является INNER-контекстом", fit["statement"])
        self.assertIn("не INNER-контекст", shift["statement"])

    def test_copies_in_another_folder_keep_content_hashes(self) -> None:
        shift_task(self.root)
        task = classification_task(self.root)
        elsewhere = self.base / "another" / "place" / "data"
        shutil.copytree(self.root, elsewhere)
        first, second = self.shift("regression", self.base / "s1"), self.shift("regression", self.base / "s2", elsewhere)
        for key in ("inputs", "config_sha256", "features_sha256", "content_sha256", "files"):
            self.assertEqual(first[key], second[key], key)
        fit_first = self.fit(task, self.base / "f1")
        fit_second = self.fit(elsewhere / "classification", self.base / "f2")
        for key in ("inputs", "config_sha256", "content_sha256", "files"):
            self.assertEqual(fit_first[key], fit_second[key], key)

    def test_no_training_no_training_subprocess_and_no_network(self) -> None:
        shift_task(self.root)
        task = classification_task(self.root)
        real_popen = subprocess.Popen

        def only_git(args, *rest, **options):
            if not (isinstance(args, (list, tuple)) and Path(str(args[0])).stem.lower() == "git"):
                raise AssertionError("unexpected subprocess")
            return real_popen(args, *rest, **options)

        blocked = AssertionError("training or network")
        with mock.patch("subprocess.Popen", only_git), \
                mock.patch("agent.execution.run_python", side_effect=blocked), \
                mock.patch("agent.round4_classification_probability.fit_predict_probabilities", side_effect=blocked), \
                mock.patch("agent.round4_classification_candidate.BoundedProbabilityFits.__call__", side_effect=blocked), \
                mock.patch("agent.models.build_pipeline", side_effect=blocked), \
                mock.patch.object(socket.socket, "connect", side_effect=blocked), \
                mock.patch("socket.create_connection", side_effect=blocked):
            self.shift("regression", self.base / "shift")
            report = self.fit(task, self.base / "fit")
        self.assertEqual(report["audit"]["decision"], "structurally_compatible")

    def test_output_inside_inputs_is_refused_and_interrupted_write_is_marked(self) -> None:
        task = classification_task(self.root)
        inputs = task / "inputs"
        for args in (("shift", "--data-dir", str(self.root), "--task", "classification", "--output-dir", str(self.root / "o")),
                     ("fit", "--task-md", str(task / "task.md"), "--dictionary", str(inputs / "data_dictionary.csv"),
                      "--fit-csv", str(inputs / "train.csv"), "--output-dir", str(task / "o"))):
            code, errors = self.cli(*args)
            self.assertEqual(code, 1)
            self.assertIn("вне папок с входными данными", errors)
        output = self.base / "fit"
        self.fit(task, output)
        original = readiness.passport.write_text

        def failing(path, text):
            if path.name == "oof_fit_summary.md":
                raise OSError("disk full")
            original(path, text)

        with mock.patch.object(readiness.passport, "write_text", side_effect=failing):
            code, _ = self.cli("fit", "--task-md", str(task / "task.md"), "--dictionary", str(inputs / "data_dictionary.csv"),
                               "--fit-csv", str(inputs / "train.csv"), "--output-dir", str(output))
        self.assertEqual(code, 1)
        failed = json.loads((output / "oof_fit_report.json").read_text(encoding="utf-8"))
        self.assertEqual((failed["status"], failed["mode"]), ("error", "fit"))
        self.assertNotIn("audit", failed)

    def test_class_order_comes_from_the_declaration_or_the_command_line(self) -> None:
        task = classification_task(self.root)
        report = self.fit(task, self.base / "sorted")
        self.assertEqual((report["class_order_source"], report["audit"]["classes"]), ("sorted_fit_labels", CLASSES))
        self.assertTrue(report["warnings"])
        declared = self.fit(task, self.base / "declared", "--classes", "alpha,beta,gamma,delta")
        self.assertEqual(declared["class_order_source"], "command_line")
        self.assertEqual(declared["audit"]["decision"], "ineligible")
        with (task / "task.md").open("a", encoding="utf-8") as stream:
            stream.write('Порядок классов: ["gamma", "beta", "alpha"]\n')
        code, errors = self.cli("fit", "--task-md", str(task / "task.md"), "--dictionary",
                                str(task / "inputs" / "data_dictionary.csv"), "--fit-csv",
                                str(task / "inputs" / "train.csv"), "--output-dir", str(self.base / "conflict"),
                                "--classes", "alpha,beta,gamma")
        self.assertEqual(code, 1)
        self.assertIn("противоречит", errors)


if __name__ == "__main__":
    unittest.main()
