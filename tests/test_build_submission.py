"""The organizer copies only root sources; package helpers must survive that step."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import build_submission


ROOT = Path(__file__).resolve().parents[1]


def platform_copy(archive: Path, task: Path) -> list[str]:
    """Model the organizer copy rule instead of preserving ZIP directories."""
    task.mkdir(parents=True, exist_ok=True)
    copied = []
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                (task / name).write_bytes(zipped.read(name))
                copied.append(name)
    return copied


def clean_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items()
            if name != "PYTHONPATH" and not name.startswith("OPENAI_")}


class FlatArchiveTests(unittest.TestCase):
    def test_archive_has_only_root_sources_and_exact_official_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "submission.zip"
            names = build_submission.build_archive(archive)
            self.assertLessEqual(archive.stat().st_size, 100_000_000)
            self.assertEqual(len(names), len({name.casefold() for name in names}))
            self.assertIn("agent.py", names)
            self.assertIn("_alfa_agent__automl.py", names)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.read("requirements.txt"),
                                 (ROOT / "requirements.platform.txt").read_bytes())
                for name in names:
                    self.assertNotIn("/", name)
                    self.assertTrue(name.endswith(".py") or name == "requirements.txt", name)
                    if name.startswith("_alfa_agent__"):
                        source = name.removeprefix("_alfa_agent__")
                        self.assertEqual(zipped.read(name), (ROOT / "agent" / source).read_bytes())

    def test_root_only_copy_preserves_source_inspection_pickling_and_child_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive, task = root / "solution.zip", root / "task data"
            names = build_submission.build_archive(archive)
            self.assertEqual(platform_copy(archive, task), names)
            # An ordinary same-name data directory must not be changed by loading.
            (task / "agent").mkdir()
            sentinel = task / "agent" / "untouched.txt"
            sentinel.write_text("task data", encoding="utf-8")
            probe = '''
import importlib, inspect, json, pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import agent
from agent import automl, llm
from agent.execution import run_python
root = Path(sys.argv[1])
assert agent.project_root() == root
assert automl.ROOT == root and llm.ROOT == root
for name in agent._alfa_flat_manifest:
    module = importlib.import_module(name)
    if module.__file__:
        assert Path(module.__file__).is_file(), module.__file__
assert 'def run_automl(' in inspect.getsource(automl)
assert pickle.loads(pickle.dumps(agent.project_root)) is agent.project_root
assert importlib.reload(automl).ROOT == root
code = ('import sys; sys.path.insert(0, ' + repr(str(root)) + '); '
        'from agent import project_root; from agent.automl_worker import run_candidate; '
        'from agent.research_validation import ResearchValidation; '
        'print(project_root())')
result = run_python(code, root, root / 'child runs', 1, 30)
assert result['exit_code'] == 0, result
assert str(root) in result['stdout'], result
print(json.dumps({'root': str(root), 'child_imports': True}))
'''
            # -I excludes both repository imports and ambient Python settings.
            result = subprocess.run(
                [sys.executable, "-I", "-c", probe, str(task)], cwd=root,
                env=clean_environment(), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["child_imports"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "task data")
            self.assertEqual(list((task / "agent").iterdir()), [sentinel])

    def test_root_only_copy_passes_setup_from_task_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, task = root / "solution.zip", root / "task"
            build_submission.build_archive(archive)
            platform_copy(archive, task)
            result = subprocess.run(
                [sys.executable, "solution.py", "--check"], cwd=task,
                env=clean_environment(), capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["python_child"])
            self.assertFalse(report["api_called"])


class BuilderBoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        for name in build_submission.REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        (self.root / "agent/__init__.py").write_bytes((ROOT / "agent/__init__.py").read_bytes())
        patched = patch.object(build_submission, "ROOT", self.root)
        patched.start()
        self.addCleanup(patched.stop)

    def write_source(self, name, text="VALUE = 7\n"):
        path = self.root / "agent" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_nested_regular_and_namespace_packages_keep_relative_imports(self):
        self.write_source("nested/__init__.py", "from .module import VALUE\n")
        self.write_source("nested/module.py")
        self.write_source("namespace/module.py", "from ..nested import VALUE\n")
        target = self.root / "copied"
        archive = self.root / "solution.zip"
        build_submission.build_archive(archive)
        platform_copy(archive, target)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from agent.nested import VALUE; from agent.namespace.module import VALUE as V; "
                "assert VALUE == V == 7")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(target)], cwd=target,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ambiguous_flat_names_fail_before_writing_output(self):
        self.write_source("nested__module.py")
        self.write_source("nested/module.py")
        with self.assertRaisesRegex(ValueError, "Colliding flat archive filename"):
            build_submission.build_archive(self.root / "solution.zip")
        self.assertFalse((self.root / "solution.zip").exists())

    def test_linked_source_file_and_directory_are_rejected(self):
        source = self.root / "outside.py"
        source.write_text("private = True\n", encoding="utf-8")
        for name, destination in (("linked.py", source), ("linked", self.root)):
            link = self.root / "agent" / name
            try:
                link.symlink_to(destination, target_is_directory=destination.is_dir())
            except OSError:
                self.skipTest("Creating symlinks is unavailable")
            try:
                with self.assertRaises(ValueError):
                    build_submission.archive_files()
            finally:
                link.unlink()

    def test_oversized_archive_does_not_replace_previous_archive(self):
        target = self.root / "solution.zip"
        target.write_bytes(b"previous archive")
        with patch.object(build_submission, "MAX_ARCHIVE_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                build_submission.build_archive(target)
        self.assertEqual(target.read_bytes(), b"previous archive")
        self.assertEqual(list(self.root.glob(".solution.zip.*.tmp")), [])

    def test_output_cannot_overwrite_source_or_development_requirements(self):
        paths = [self.root / "solution.py", self.root / "requirements.platform.txt",
                 self.root / "requirements.txt"]
        paths[-1].write_bytes(b"development dependencies")
        for target in paths:
            original = target.read_bytes()
            with self.subTest(target=target.name), self.assertRaises(ValueError):
                build_submission.build_archive(target)
            self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
