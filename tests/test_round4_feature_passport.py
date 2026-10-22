"""The feature passport is correct, bounded and leaves the task data untouched."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import experiment_round4_feature_passport as passport
from app.tests.test_profiling import make_task, write_csv
from tests.test_analyze_data import classification_task, regression_task


SENTINELS = ("IDSENTINEL", "TARGETSENTINEL", "RAWTRAIN", "RAWTEST", "DESCSENTINEL", "FUTURESENTINEL")


def tree(root: Path) -> dict[str, str | None]:
    """Every file and folder under root with its content hash (None for folders)."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in sorted(root.rglob("*"))}


def write_generic_task(root: Path, name: str, dictionary: list[list[str]], train: list[list[object]],
                       test: list[list[object]], sample: list[list[object]], metric: str) -> Path:
    task = root / name
    inputs = task / "inputs"
    write_csv(inputs / "data_dictionary.csv", [["feature", "role", "dtype", "description_ru"], *dictionary])
    write_csv(inputs / "train.csv", train)
    write_csv(inputs / "test.csv", test)
    write_csv(inputs / "sample_submission.csv", sample)
    (task / "task.md").write_text(f"Основная метрика: {metric}\n", encoding="utf-8")
    return task


class FeaturePassportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / "data"
        self.root.mkdir()

    def run_cli(self, task: str, output: Path, root: Path | None = None) -> tuple[int, str]:
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = passport.main(["--data-dir", str(root or self.root), "--task", task, "--output-dir", str(output)])
        return code, errors.getvalue()

    def export(self, task: str, output: Path | None = None, root: Path | None = None) -> tuple[dict, Path]:
        output = output or self.base / "out"
        code, errors = self.run_cli(task, output, root)
        self.assertEqual(code, 0, errors)
        return json.loads((output / "passport.json").read_text(encoding="utf-8")), output

    @staticmethod
    def texts(output: Path) -> dict[str, str]:
        return {name: (output / name).read_text(encoding="utf-8") for name in passport.OUTPUT_FILES}

    @staticmethod
    def rows(data: dict) -> dict[str, dict]:
        return {row["name"]: row for row in data["features"]}

    def test_inputs_stay_byte_identical_and_only_three_files_are_written(self) -> None:
        make_task(self.root)
        before = tree(self.root)
        data, output = self.export("classification")
        self.assertEqual(tree(self.root), before)  # no submission.csv, model or .agent_runs
        self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(passport.OUTPUT_FILES))
        texts = self.texts(output)
        for name in ("features.csv", "README.md"):
            self.assertEqual(data["files"][name], hashlib.sha256(texts[name].encode("utf-8")).hexdigest())
        self.assertEqual((data["status"], data["train_rows"], data["test_rows"]), ("complete", 5, 2))
        self.assertEqual(list(self.rows(data)), ["measure", "protocol", "flag"])

    def test_fingerprint_is_portable_and_feature_data_ignore_the_train_target(self) -> None:
        make_task(self.root)
        elsewhere = self.base / "another" / "place" / "data"
        make_task(elsewhere)
        first, first_output = self.export("classification")
        second, second_output = self.export("classification", self.base / "out-2", elsewhere)
        for key in ("dataset_fingerprint", "features_sha256", "content_sha256", "files"):
            self.assertEqual(first[key], second[key], key)
        for text in [*self.texts(first_output).values(), *self.texts(second_output).values()]:
            self.assertNotIn(str(self.base), text)
            self.assertNotIn(self.base.name, text)

        train = self.root / "classification" / "nested" / "participants" / "train.csv"
        with train.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        target = rows[0].index("verdict")
        for row in rows[1:]:
            row[target] = "Changed" if row[target] == "NA" else "NA"
        write_csv(train, rows)
        changed, _ = self.export("classification", self.base / "out-3")
        self.assertNotEqual(changed["dataset_fingerprint"], first["dataset_fingerprint"])
        self.assertEqual(changed["features"], first["features"])
        self.assertEqual(changed["features_sha256"], first["features_sha256"])

    def test_roles_not_names_exclude_target_and_id_and_raw_values_never_leak(self) -> None:
        # The target comes first and looks like a feature; the ID is last and numeric-looking.
        write_generic_task(
            self.root, "custom",
            [["signal", "target", "category", "DESCSENTINEL target"], ["kind", "feature", "category", "DESCSENTINEL"],
             ["amount", "feature", "numeric", "DESCSENTINEL"], ["value_7", "id", "category", "DESCSENTINEL id"]],
            [["signal", "kind", "amount", "value_7"],
             *[[f"TARGETSENTINEL-{index % 2}", f"RAWTRAIN-{index % 3}", index, f"IDSENTINEL-{index}"] for index in range(6)]],
            [["kind", "amount", "value_7"], ["RAWTEST-unseen", 3, "IDSENTINEL-t1"], ["RAWTRAIN-1", 4, "IDSENTINEL-t2"]],
            [["value_7", "signal"], ["IDSENTINEL-t1", "TARGETSENTINEL-0"], ["IDSENTINEL-t2", "TARGETSENTINEL-1"]],
            "Macro F1")
        original = passport.DataCatalog.profile

        def with_future_fields(catalog, task_id):
            # A newer profiler may add fields anywhere; only the whitelist may reach the files.
            profile = original(catalog, task_id)
            profile["future_field"] = "FUTURESENTINEL"
            for column in profile["columns"]:
                column["future_field"] = "FUTURESENTINEL"
            for item in profile["quality"]["new_categories"]:
                item["future_field"] = "FUTURESENTINEL"
            return profile

        with mock.patch.object(passport.DataCatalog, "profile", with_future_fields):
            data, output = self.export("custom")
        self.assertEqual(list(self.rows(data)), ["kind", "amount"])
        self.assertEqual(self.rows(data)["kind"]["new_category_count"], 1)
        for row in data["features"]:
            self.assertEqual(set(row), set(passport.CSV_COLUMNS))
        self.assertLessEqual(set(data), {
            "schema_version", "task_id", "scope", "source", "source_commit", "dataset_fingerprint", "train_rows",
            "test_rows", "feature_count", "status", "limitations", "warnings", "source_caps", "field_sources",
            "features", "features_sha256", "content_sha256", "files", "generated"})
        for name, text in self.texts(output).items():
            for sentinel in SENTINELS:
                self.assertNotIn(sentinel, text, f"{sentinel} leaked into {name}")
            self.assertNotIn("signal", text, name)
            self.assertNotIn("value_7", text, name)

    def test_missing_semantics_follow_the_profiler_and_bad_inputs_fail_without_output(self) -> None:
        make_task(self.root)
        data, _ = self.export("classification")
        rows = self.rows(data)
        # Literal NA and a lone space are values; inf and "invalid" are counted, never zeros.
        self.assertEqual((rows["measure"]["train_missing_count"], rows["measure"]["train_nonfinite_count"]), (1, 2))
        self.assertEqual((rows["measure"]["train_min"], rows["measure"]["train_max"]), (3.0, 5.0))
        self.assertEqual((rows["protocol"]["train_missing_count"], rows["protocol"]["train_unique_count"]), (1, 4))
        self.assertIsNone(rows["protocol"]["train_nonfinite_count"])
        self.assertEqual(rows["protocol"]["null_reasons"]["train_nonfinite_count"], "not_numeric_dtype")

        def malformed(inputs: Path) -> None:
            with (inputs / "train.csv").open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["006", "7", "http", "0", "Normal", "extra"])

        def missing_column(inputs: Path) -> None:
            write_csv(inputs / "test.csv", [["row_key", "measure", "protocol"], ["010", "7", "tcp"]])

        def missing_file(inputs: Path) -> None:
            (inputs / "train.csv").unlink()

        for damage in (malformed, missing_column, missing_file):
            with self.subTest(damage=damage.__name__):
                root = self.base / damage.__name__
                make_task(root)
                damage(root / "classification" / "nested" / "participants")
                output = self.base / f"out-{damage.__name__}"
                code, errors = self.run_cli("classification", output, root)
                self.assertEqual(code, 1)
                self.assertIn("Экспорт паспорта не выполнен", errors)
                self.assertFalse(output.exists())
        code, errors = self.run_cli("missing-task", self.base / "out-unknown")
        self.assertEqual(code, 1)
        self.assertIn("classification", errors)

    def test_source_caps_mark_partial_rows_and_the_output_stays_bounded(self) -> None:
        count = 24
        numeric = [f"n{index:02d}" for index in range(count)]
        category = [f"c{index:02d}" for index in range(count)]
        dictionary = [["row_id", "id", "numeric", ""], ["y", "target", "numeric", ""],
                      *[[name, "feature", "numeric", ""] for name in numeric],
                      *[[name, "feature", "category", ""] for name in category]]
        train = [["row_id", "y", *numeric, *category],
                 *[[row, row, *([row] * count), *(["a", "b"][row % 2] for _ in category)] for row in range(10)]]
        # Every numeric feature drifts by a different amount, 12 of them lose TEST values,
        # and every category gets an unseen TEST value: more than the profiler keeps.
        test = [["row_id", *numeric, *category],
                *[[100 + row, *("" if index < 12 and row < 2 else 10 + index + row for index in range(count)),
                   *(f"new-{index}" for index in range(count))] for row in range(4)]]
        sample = [["row_id", "y"], *[[100 + row, 0] for row in range(4)]]
        write_generic_task(self.root, "wide", dictionary, train, test, sample, "RMSE")
        data, output = self.export("wide")
        caps = data["source_caps"]
        self.assertEqual(data["status"], "partial")
        self.assertEqual((caps["new_categories"]["returned"], caps["new_categories"]["features_outside_cap"]), (20, 4))
        self.assertEqual((caps["numeric_shift"]["returned"], caps["numeric_shift"]["features_outside_cap"]), (20, 4))
        self.assertEqual((caps["missing_changes"]["returned"], caps["missing_changes"]["features_outside_cap"]),
                         (10, count * 2 - 10))
        outside = [row for row in data["features"] if row["null_reasons"].get("new_category_count") == "outside_source_cap"]
        self.assertEqual(len(outside), 4)
        self.assertTrue(all(row["new_category_count"] is None and row["status"] == "partial" for row in outside))
        self.assertTrue(data["warnings"])
        with (output / "features.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), data["feature_count"])
        table_rows = [line for line in self.texts(output)["README.md"].splitlines()
                      if line.startswith("| ") and not line.startswith(("| ---", "| Признак", "| Поле"))]
        self.assertLessEqual(len(table_rows), 6 + 4 * passport.README_TOP)

    def test_names_with_commas_newlines_and_markdown_survive_csv_and_markdown(self) -> None:
        names = ["a,b", "line\nbreak", "**bold**|pipe", "`tick` <b>"]
        write_generic_task(
            self.root, "names",
            [["row_id", "id", "numeric", ""], ["y", "target", "numeric", ""], *[[name, "feature", "category", ""] for name in names]],
            [["row_id", "y", *names], *[[row, row, *(["x", "z"][row % 2] for _ in names)] for row in range(4)]],
            [["row_id", *names], [10, *(f"unseen-{index}" for index in range(len(names)))]],
            [["row_id", "y"], [10, 0]], "RMSE")
        data, output = self.export("names")
        self.assertEqual(list(self.rows(data)), names)
        with (output / "features.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([row["name"] for row in csv.DictReader(stream)], names)
        readme = self.texts(output)["README.md"]
        section = readme.split("### Новые категории в TEST", 1)[1].split("###", 1)[0]
        rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line][1:]
        self.assertEqual(len(rows), len(names))
        for row in rows:
            self.assertEqual(len(re.split(r"(?<!\\)\|", row)), 4, row)  # exactly two cells
        self.assertIn("line ↵ break", readme)
        self.assertIn(r"\*\*bold\*\*\|pipe", readme)
        self.assertNotIn("<b>", readme)

    def test_output_inside_data_or_on_a_file_is_refused_before_writing(self) -> None:
        task = make_task(self.root)
        inputs = task / "nested" / "participants"
        outside_file = self.base / "existing.txt"
        outside_file.write_text("keep", encoding="utf-8")
        before = tree(self.root)
        for output in (self.root, task, inputs, inputs / "train.csv", inputs / "new-folder", outside_file):
            with self.subTest(output=output.name):
                code, errors = self.run_cli("classification", output)
                self.assertEqual(code, 1)
                self.assertIn("--output-dir", errors)
                self.assertEqual(tree(self.root), before)
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "keep")

    def test_reexport_keeps_foreign_files_and_a_failed_write_is_never_complete(self) -> None:
        make_task(self.root)
        _, output = self.export("classification")
        (output / "notes.txt").write_text("keep", encoding="utf-8")
        self.export("classification", output)
        self.assertEqual(sorted(path.name for path in output.iterdir()), sorted([*passport.OUTPUT_FILES, "notes.txt"]))
        original = passport.write_text

        def failing(names):
            def write(path, text):
                if path.name in names:
                    raise OSError("disk full")
                return original(path, text)
            return write

        with mock.patch.object(passport, "write_text", failing({"README.md"})):
            code, _ = self.run_cli("classification", output)
        self.assertEqual(code, 1)
        failed = json.loads((output / "passport.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["status"], "error")
        self.export("classification", output)
        with mock.patch.object(passport, "write_text", failing({"passport.json"})):
            code, _ = self.run_cli("classification", output)
        self.assertEqual(code, 1)
        self.assertFalse((output / "passport.json").exists())  # the old complete passport is gone
        self.assertEqual((output / "notes.txt").read_text(encoding="utf-8"), "keep")
        self.assertFalse([path.name for path in output.iterdir() if path.name.endswith(".tmp")])

    def test_analyzer_fixtures_show_constants_shift_unparseable_and_new_categories(self) -> None:
        for task in (regression_task(self.root), classification_task(self.root)):
            (task / "task.md").write_text("Основная метрика: RMSE\n" if task.name == "regression"
                                          else "Основная метрика: Macro F1\n", encoding="utf-8")
        regression, _ = self.export("regression", self.base / "out-regression")
        rows = self.rows(regression)
        self.assertNotIn("target", rows)
        self.assertNotIn("row_id", rows)
        self.assertTrue(rows["x_const"]["is_constant"])
        self.assertEqual(rows["x_text"]["train_nonfinite_count"], 6)  # three "abc" rows, repeated once in TRAIN
        shifted = max((row for row in rows.values() if row["shift_score"] is not None), key=lambda row: row["shift_score"])
        self.assertEqual(shifted["name"], "x_shift")
        classification, output = self.export("classification", self.base / "out-classification")
        rows = self.rows(classification)
        self.assertEqual(set(rows), {"x_noise", "x_flag", "x_code", "protocol"})
        self.assertEqual(rows["protocol"]["new_category_count"], 1)
        self.assertIn("unknown fallback", self.texts(output)["README.md"])


if __name__ == "__main__":
    unittest.main()
