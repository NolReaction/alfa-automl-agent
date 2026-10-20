"""Verify archive boundaries and the human-readable report's escaping."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import build_submission
from app.exports import delivery_archive, result_archive, _readable_report
from app.model_config import make_model_config
from app.sharing import ReportStore
from app.tests.test_sharing import search_summary


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # Match the builder's canonical ROOT, including Windows 8.3 aliases.
        self.root = Path(self.directory.name).resolve()
        for name in build_submission.REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# source " + name, encoding="utf-8")
        for name in ("agent/sub/module.py", "agent/__pycache__/ignored.py", "app/private.py",
                     "data/task/train.csv", "data/task/submission.csv", ".env", "agent/key.txt"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("private fixture", encoding="utf-8")

    def test_delivery_uses_original_builder_allowlist_without_writing_output(self):
        with patch.object(build_submission, "ROOT", self.root):
            expected = {path.relative_to(self.root).as_posix() for path in build_submission.source_files()}
            package = delivery_archive()
        with ZipFile(BytesIO(package)) as archive:
            self.assertEqual(set(archive.namelist()), expected)
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read("solution.py"), (self.root / "solution.py").read_bytes())
            self.assertIn("agent/sub/module.py", archive.namelist())
            self.assertFalse(any(name.startswith(("app/", "data/")) or "__pycache__" in name
                                 or name.endswith(".env") for name in archive.namelist()))
        self.assertFalse((self.root / "delivery").exists())

    def test_required_missing_or_linked_source_and_size_limit_are_rejected(self):
        with patch.object(build_submission, "ROOT", self.root):
            with patch("app.exports.MAX_SOURCE_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "лимит"):
                    delivery_archive()
            required = self.root / "solution.py"
            required.unlink()
            with self.assertRaisesRegex(ValueError, "Missing regular"):
                delivery_archive()
            outside = self.root / "outside.txt"
            outside.write_text("secret")
            try:
                required.symlink_to(outside)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(ValueError):
                delivery_archive()

    def test_report_has_no_script_injection_and_labels_fallback_without_metric(self):
        report = {"id": "a" * 32, "action": "agent", "agent_status": "fallback",
                  "author": '<script>alert("author")</script>', "hypothesis": '<img src=x onerror="evil()">',
                  "dataset_fingerprint": "b" * 64, "metric": None}
        html = _readable_report(report).decode("utf-8")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img ", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Резервный CSV", html)
        self.assertIn("Качество не измерено", html)
        self.assertIn("reports.json", html)

    def test_readable_report_identifies_validation_scheme_and_escapes_period_column(self):
        report = {"id": "a" * 32, "action": "baseline", "dataset_fingerprint": "b" * 64,
                  "metric": {"name": "RMSE", "value": 50.0, "higher_is_better": False},
                  "split": {"strategy": "time", "time_column": "<script>Year</script>",
                            "train_period_min": "1", "train_period_max": "4", "validation_period": "5",
                            "validation_fraction": 0.34, "validation_rows": 34}}
        html = _readable_report(report).decode("utf-8")
        self.assertIn("По последнему периоду", html)
        self.assertIn("1–4 → 5", html)
        self.assertIn("34% данных", html)
        self.assertIn("&lt;script&gt;Year&lt;/script&gt;", html)
        self.assertNotIn("<script>", html)
        report["split"] = {"strategy": "grouped", "train_groups": 12, "validation_groups": 3,
                           "group_overlap": 0, "validation_fraction": 0.25, "validation_rows": 4}
        html = _readable_report(report).decode("utf-8")
        self.assertIn("По группам одинаковых признаков", html)
        self.assertIn("12 / 3", html)
        self.assertIn("Пересечение групп</dt><dd>0", html)

    def test_result_package_keeps_selected_model_and_validation_settings(self):
        config = make_model_config("extra_trees", "regression", {"n_estimators": 200, "max_depth": 12})
        report = {"id": "a" * 32, "action": "extra_trees", "status": "completed",
                  "model": config["model"], "model_config": config, "dataset_fingerprint": "b" * 64,
                  "metric": {"name": "RMSE", "value": 50.0, "higher_is_better": False},
                  "output_valid": True, "api_used": False,
                  "split": {"strategy": "grouped", "train_groups": 12, "validation_groups": 3,
                            "group_overlap": 0, "n_splits": 5, "fold": 2,
                            "validation_fraction": 0.2, "validation_rows": 3}}
        csv_bytes = b"id,answer\n001,1.5\n"
        artifact = self.root / "predictions.csv"
        artifact.write_bytes(csv_bytes)
        jobs = Mock()
        jobs.artifact.return_value = artifact
        jobs.list.return_value = [report]
        jobs._file_hash.return_value = hashlib.sha256(csv_bytes).hexdigest()
        catalog = Mock()
        catalog.fingerprint.return_value = report["dataset_fingerprint"]
        package = result_archive("example", report["id"], jobs=jobs, catalog=catalog,
                                 reports=ReportStore(self.root / "reports"))
        with ZipFile(BytesIO(package)) as archive:
            self.assertEqual(set(archive.namelist()), {"submission.csv", "reports.json", "report.html"})
            self.assertEqual(archive.read("submission.csv"), csv_bytes)
            exported, = json.loads(archive.read("reports.json"))["reports"]
            self.assertEqual(exported["model_config"], config)
            self.assertEqual(exported["split"], report["split"])
            html = archive.read("report.html").decode("utf-8")
        self.assertIn("ExtraTreesRegressor", html)
        self.assertIn("Количество деревьев: 200", html)
        self.assertIn("Глубина дерева: 12", html)
        self.assertIn("По группам одинаковых признаков", html)
        self.assertIn("Разбиение групп</dt><dd>3 из 5", html)

    def test_readable_model_settings_are_escaped_and_not_invented_for_legacy_runs(self):
        report = {"id": "a" * 32, "action": "baseline", "model": "RandomForestClassifier",
                  "dataset_fingerprint": "b" * 64, "metric": None}
        html = _readable_report(report).decode("utf-8")
        self.assertNotIn("Параметры модели", html)
        config = make_model_config("baseline", "classification")
        report["model_config"] = config
        html = _readable_report(report).decode("utf-8")
        self.assertIn("Количество деревьев: 120", html)
        self.assertIn("Веса классов: Балансировать в каждом дереве", html)
        self.assertIn("Признаки при разбиении: Квадратный корень", html)
        # Even a direct renderer call with untrusted text cannot add markup.
        config["parameters"]["n_estimators"] = '<img src=x onerror="evil()">'
        html = _readable_report(report).decode("utf-8")
        self.assertNotIn("<img ", html)
        self.assertIn("&lt;img src=x onerror=&quot;evil()&quot;&gt;", html)

    def test_automl_report_shows_attempts_winner_and_escapes_text(self):
        search = search_summary(status="time_limit")
        search["validation_reason"] = "<script>validation</script>"
        search["candidates"][0]["label"] = '<img src=x onerror="evil()">'
        search["candidates"][2]["error"] = "<script>error</script>"
        report = {"id": "a" * 32, "action": "automl", "model": "ExtraTreesRegressor",
                  "dataset_fingerprint": "b" * 64, "search": search,
                  "metric": {"name": "RMSE", "value": 43.25, "higher_is_better": False},
                  "model_config": make_model_config("extra_trees", "regression")}
        html = _readable_report(report).decode("utf-8")
        self.assertIn("Автоподбор моделей", html)
        self.assertIn("Попытки автоподбора", html)
        self.assertIn("Победитель", html)
        self.assertIn("Лимит времени: сохранён лучший завершённый результат.", html)
        self.assertIn("43.25", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img ", html)
        self.assertIn("&lt;script&gt;validation&lt;/script&gt;", html)
        self.assertIn("&lt;script&gt;error&lt;/script&gt;", html)


if __name__ == "__main__":
    unittest.main()
