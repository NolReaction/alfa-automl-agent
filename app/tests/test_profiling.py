"""Behavioral checks for local, read-only dataset inspection."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.profiling import DataCatalog


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def make_task(data_root: Path, name: str = "classification") -> Path:
    task = data_root / name
    inputs = task / "nested" / "participants"
    inputs.mkdir(parents=True)
    (task / "task.md").write_text("# Test task\nPredict the verdict.\n", encoding="utf-8")
    write_csv(inputs / "data_dictionary.csv", [
        ["feature", "role", "dtype", "description_ru"],
        ["row_key", "id", "category", "Identifier"],
        ["measure", "feature", "numeric", "Measurement"],
        ["protocol", "feature", "category", "Protocol"],
        ["flag", "feature", "binary", "Flag"],
        ["verdict", "target", "category", "Verdict"],
    ])
    write_csv(inputs / "train.csv", [
        ["row_key", "measure", "protocol", "flag", "verdict"],
        ["001", "3", "NA", "1", "NA"],
        ["NA", "", "", "0", "Normal"],
        ["003", "inf", "http", "1", "NA"],
        ["004", "invalid", "https", "", "Normal"],
        ["005", "5", " ", "0", "NA"],
    ])
    write_csv(inputs / "test.csv", [
        ["row_key", "measure", "protocol", "flag"],
        ["010", "7", "NA", "0"],
        ["011", "9", "https", "1"],
    ])
    write_csv(inputs / "sample_submission.csv", [
        ["row_key", "verdict"], ["010", "NA"], ["011", "Normal"],
    ])
    return task


class ProfilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "data"
        self.root.mkdir()

    def test_discovery_and_preview_preserve_ids_and_na_labels(self) -> None:
        make_task(self.root)
        catalog = DataCatalog(self.root)
        tasks = catalog.discover()
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]["available"], tasks[0].get("error"))
        task_id = tasks[0]["id"]
        profile = catalog.profile(task_id)
        self.assertEqual(profile["task_type"], "classification")
        self.assertEqual(profile["target"], "verdict")
        self.assertEqual((profile["train_rows"], profile["test_rows"]), (5, 2))
        self.assertEqual(profile["feature_count"], 3)
        self.assertEqual(profile["type_counts"]["binary"], 1)
        preview = profile["preview"]
        id_index = preview["columns"].index("row_key")
        target_index = preview["columns"].index("verdict")
        self.assertEqual([row[id_index] for row in preview["rows"]],
                         ["001", "NA", "003", "004", "005"])
        self.assertEqual(preview["rows"][0][target_index], "NA")
        counts = {item["label"]: item["count"]
                  for item in profile["target_distribution"]["bins"]}
        self.assertEqual(counts, {"NA": 3, "Normal": 2})

    def test_empty_cells_only_are_missing_and_numeric_bins_exclude_nonfinite(self) -> None:
        make_task(self.root)
        catalog = DataCatalog(self.root)
        profile = catalog.profile("classification")
        columns = {item["name"]: item for item in profile["columns"]}
        self.assertEqual(columns["protocol"]["missing_count"], 1)
        self.assertEqual(columns["measure"]["missing_count"], 1)
        self.assertEqual(columns["verdict"]["missing_count"], 0)
        measure = catalog.column("classification", "measure")
        self.assertEqual(measure["stats"]["min"], 3)
        self.assertEqual(measure["stats"]["max"], 5)
        self.assertEqual(measure["stats"]["mean"], 4)
        self.assertEqual(sum(item["count"] for item in measure["distribution"]["bins"]), 2)
        self.assertTrue(profile["warnings"], "Invalid numeric strings should be disclosed")
        flag = catalog.column("classification", "flag")
        self.assertEqual(flag["dtype"], "binary")
        self.assertEqual(flag["stats"]["min"], 0)
        self.assertEqual(flag["stats"]["max"], 1)
        self.assertEqual(sum(item["count"] for item in flag["distribution"]["bins"]), 4)

    def test_preview_is_bounded_and_cache_tracks_file_updates(self) -> None:
        task = make_task(self.root)
        train = task / "nested" / "participants" / "train.csv"
        rows = [["row_key", "measure", "protocol", "flag", "verdict"]] + [
            [f"id-{index}", 3, "NA", index % 2, "Normal"] for index in range(120)
        ]
        write_csv(train, rows)
        catalog = DataCatalog(self.root)
        first = catalog.profile("classification")
        self.assertEqual(first["train_rows"], 120)
        self.assertEqual(len(first["preview"]["rows"]), 100)
        self.assertEqual(catalog.column("classification", "measure")["stats"]["mean"], 3)
        timestamp = train.stat().st_mtime_ns
        rows[1][1] = 123
        write_csv(train, rows)
        os.utime(train, ns=(timestamp + 1_000_000, timestamp + 1_000_000))
        self.assertEqual(catalog.column("classification", "measure")["stats"]["mean"], 4)
        second = catalog.profile("classification")
        measure_index = second["preview"]["columns"].index("measure")
        self.assertEqual(second["preview"]["rows"][0][measure_index], "123")

    def test_missing_input_is_visible_as_unavailable_task(self) -> None:
        task = make_task(self.root)
        (task / "nested" / "participants" / "train.csv").unlink()
        catalog = DataCatalog(self.root)
        tasks = catalog.discover()
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]["available"])
        self.assertTrue(tasks[0]["error"])
        with self.assertRaises((ValueError, OSError)):
            catalog.profile(tasks[0]["id"])

    def test_duplicate_headers_are_rejected(self) -> None:
        task = make_task(self.root)
        write_csv(task / "nested" / "participants" / "train.csv", [
            ["row_key", "measure", "measure", "flag", "verdict"],
            ["a", "3", "4", "1", "NA"],
        ])
        catalog = DataCatalog(self.root)
        with self.assertRaises((ValueError, OSError)):
            catalog.profile("classification")

    def test_malformed_row_is_rejected_and_discovery_remembers_the_problem(self) -> None:
        task = make_task(self.root)
        train = task / "nested" / "participants" / "train.csv"
        with train.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["006", "7", "http", "0", "Normal", "extra"])
        catalog = DataCatalog(self.root)
        with self.assertRaises((ValueError, OSError)):
            catalog.profile("classification")
        tasks = catalog.discover()
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]["available"])
        self.assertTrue(tasks[0]["error"])

    def test_metric_is_read_from_task_text_instead_of_guessed_from_target(self) -> None:
        task = make_task(self.root)
        task_file = task / "task.md"
        catalog = DataCatalog(self.root)
        for text, name, higher_is_better in [
            ("# Generic classification\nPredict the verdict.\n", "Не определена", None),
            ("# Classification\nОсновная метрика: Macro F1\n", "Macro F1", True),
            ("# Metric declaration\nМетрика: RMSE\n", "RMSE", False),
            ("Reference metric: Accuracy\nОсновная метрика: Macro F1\n", "Macro F1", True),
            ("Reference metric: Accuracy\nОсновная метрика: MAP@K\n", "Не определена", None),
        ]:
            with self.subTest(metric=name):
                task_file.write_text(text, encoding="utf-8")
                metric = catalog.profile("classification")["metric"]
                self.assertEqual(metric["name"], name)
                self.assertIs(metric["higher_is_better"], higher_is_better)

    def test_empty_root_unknown_task_and_task_text(self) -> None:
        catalog = DataCatalog(self.root)
        self.assertEqual(catalog.discover(), [])
        self.assertEqual(DataCatalog(self.root / "missing").discover(), [])
        for candidate in ["unknown", "../private", "/etc", "a/b"]:
            with self.assertRaises(ValueError, msg=candidate):
                catalog.resolve(candidate)
        task = make_task(self.root)
        self.assertEqual(catalog.resolve("classification"), task)
        self.assertIn("Predict the verdict", catalog.task_text("classification"))

    def test_long_ascii_directory_name_uses_a_short_stable_task_id(self) -> None:
        task = make_task(self.root, "a" * 129)
        catalog = DataCatalog(self.root)
        summary = catalog.discover()[0]
        task_id = summary["id"]
        self.assertTrue(summary["available"], summary.get("error"))
        self.assertLessEqual(len(task_id), 128)
        self.assertRegex(task_id, r"^[A-Za-z0-9_-]+$")
        self.assertEqual(catalog.resolve(task_id), task)
        self.assertEqual(catalog.profile(task_id)["train_rows"], 5)
        self.assertEqual(DataCatalog(self.root).discover()[0]["id"], task_id)

    def test_symlinked_input_cannot_expose_outside_file(self) -> None:
        task = make_task(self.root)
        train = task / "nested" / "participants" / "train.csv"
        external = Path(self.directory.name) / "private.csv"
        external.write_text("row_key,secret\na,do-not-expose\n", encoding="utf-8")
        train.unlink()
        try:
            train.symlink_to(external)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable on this platform")
        catalog = DataCatalog(self.root)
        tasks = catalog.discover()
        self.assertFalse(tasks[0]["available"])
        with self.assertRaises((ValueError, OSError)):
            catalog.profile("classification")

    def test_discovery_and_nested_scan_have_directory_limits(self) -> None:
        task = make_task(self.root)
        for name in ("extra-a", "extra-b", "extra-c"):
            (self.root / name).mkdir()
        with patch("app.profiling.MAX_FOLDERS", 3):
            with self.assertRaisesRegex(ValueError, "более 3"):
                DataCatalog(self.root).discover()
        for name in ("extra-a", "extra-b", "extra-c"):
            (self.root / name).rmdir()
            (task / name).mkdir()
        with patch("app.profiling.MAX_FOLDERS", 3):
            catalog = DataCatalog(self.root)
            self.assertFalse(catalog.discover()[0]["available"])
            with self.assertRaisesRegex(ValueError, "более 3"):
                catalog.profile("classification")

    @unittest.skipUnless(os.name == "nt", "Windows junction behavior")
    def test_windows_junctions_are_not_followed_even_inside_the_data_root(self) -> None:
        task = make_task(self.root)
        original = task / "nested"
        saved = task / "saved-inputs"
        original.rename(saved)
        # The target stays inside the task: resolve().is_relative_to() alone
        # would accept this junction, but the dashboard deliberately rejects it.
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(original), str(saved)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            self.skipTest("This Windows filesystem cannot create junctions")
        self.addCleanup(lambda: original.rmdir() if original.is_junction() else None)
        self.assertTrue(original.is_junction())
        catalog = DataCatalog(self.root)
        # A second junction to an ordinary task must not create another task.
        alias = self.root / "alias"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(task)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            self.skipTest("This Windows filesystem cannot create a task junction")
        self.addCleanup(lambda: alias.rmdir() if alias.is_junction() else None)
        self.assertEqual([item["id"] for item in catalog.discover()], ["classification"])
        # The real saved-inputs directory is still discovered, while the
        # junction is ignored; following it would incorrectly find two samples.
        self.assertTrue(catalog.discover()[0]["available"])
        with self.assertRaises(ValueError):
            catalog.resolve("alias")


if __name__ == "__main__":
    unittest.main()
