"""The CLI says plainly when no verified model was published and when a Windows path is too deep."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solution


PLATFORM_ENV = {"OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}


def run_report(directory: Path, published: bool, verified: bool = False) -> Path:
    report = {"status": "completed" if published else "fallback",
              "publication": {"new_result_published": published, "reason": "model_finished" if published else "refit_failed"},
              "best_metric_provenance": ({"scope": "final_check"} if published
                                         else {"scope": "existing_output", "quality_verified": verified})}
    (directory / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def captured(function, *args, **kwargs) -> str:
    errors = io.StringIO()
    with redirect_stderr(errors):
        function(*args, **kwargs)
    return errors.getvalue()


class UnverifiedOutputTests(unittest.TestCase):
    def test_kept_earlier_output_is_reported_with_its_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            text = captured(solution._warn_unverified_output, run_report(Path(directory), published=False))
        self.assertIn("no newly verified model was published", text)
        self.assertIn("refit_failed", text)
        self.assertIn("quality is unverified", text)

    def test_recovered_accepted_checkpoint_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as directory:
            text = captured(solution._warn_unverified_output, run_report(Path(directory), False, verified=True))
        self.assertIn("earlier accepted checkpoint", text)

    def test_published_or_unreadable_reports_stay_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(captured(solution._warn_unverified_output, run_report(folder, published=True)), "")
            (folder / "run.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(captured(solution._warn_unverified_output, folder), "")
            self.assertEqual(captured(solution._warn_unverified_output, folder / "missing"), "")
            self.assertEqual(captured(solution._warn_unverified_output, None), "")

    def test_research_cli_reports_an_unpublished_run(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = folder / "task.md"
            task.write_text("Research this task", encoding="utf-8")
            for published, expected in ((False, True), (True, False)):
                run_report(folder, published)
                output, errors = io.StringIO(), io.StringIO()
                with self.subTest(published=published), patch.dict(os.environ, PLATFORM_ENV, clear=True), \
                        patch("sys.argv", ["solution.py", "--research-search", "legacy", str(task)]), \
                        patch.object(solution, "make_client"), \
                        patch.object(solution, "run_agent", return_value=folder), \
                        redirect_stdout(output), redirect_stderr(errors):
                    status = solution.main()
                self.assertEqual(status, 0, errors.getvalue())
                self.assertEqual("no newly verified model" in errors.getvalue(), expected, errors.getvalue())


class LongPathTests(unittest.TestCase):
    def deep_task(self, root: Path) -> Path:
        depth = solution.WINDOWS_PATH_LIMIT - solution.RUN_PATH_DEPTH - len(str(root.resolve())) + 5
        folder = root / ("d" * max(1, depth))
        folder.mkdir()
        return folder / "task.md"

    def test_deep_windows_folder_without_long_paths_is_warned(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.deep_task(Path(directory))
            with patch.object(solution, "_windows_long_paths_enabled", return_value=False):
                text = captured(solution._warn_long_task_path, task, windows=True)
        self.assertIn("WinError 206", text)
        self.assertIn("shorter folder", text)

    def test_long_paths_enabled_short_folders_and_other_systems_stay_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = self.deep_task(root)
            for enabled in (True, None):
                with self.subTest(enabled=enabled), \
                        patch.object(solution, "_windows_long_paths_enabled", return_value=enabled):
                    text = captured(solution._warn_long_task_path, task, windows=True)
                self.assertEqual(text == "", enabled is True, text)
            with patch.object(solution, "_windows_long_paths_enabled", return_value=False):
                self.assertEqual(captured(solution._warn_long_task_path, task, windows=False), "")
                if len(str(root.resolve())) + solution.RUN_PATH_DEPTH <= solution.WINDOWS_PATH_LIMIT:
                    self.assertEqual(captured(solution._warn_long_task_path, root / "task.md", windows=True), "")


if __name__ == "__main__":
    unittest.main()
