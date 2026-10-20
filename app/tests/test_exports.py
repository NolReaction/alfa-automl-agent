"""Verify archive boundaries and the human-readable report's escaping."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import build_submission
from app.exports import delivery_archive, _readable_report


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


if __name__ == "__main__":
    unittest.main()
