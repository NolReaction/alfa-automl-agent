"""Contract checks against the organizer files supplied on 2026-10-20.

The original baseline accepts an optional task.md argument and resolves data
paths relative to that file. Its MCP server is an internal implementation detail;
MCP is not among the packages listed in the supplied requirements.txt.
"""

import ast
import csv
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

from build_submission import build_archive
from validation import validate_predictions


ROOT = Path(__file__).resolve().parents[1]
# Frozen public platform package contract, not the subset installed for our app.
PLATFORM_REQUIREMENTS = """
numpy==1.26.4
scipy==1.12.0
pandas==2.2.3
polars==1.26.0
pyarrow==19.0.1
joblib==1.4.2
threadpoolctl==3.6.0
scikit-learn==1.5.2
catboost==1.2.8
lightgbm==4.6.0
xgboost==2.1.4
imbalanced-learn==0.12.4
category-encoders==2.6.4
feature-engine==1.8.3
statsmodels==0.14.0
optuna==4.2.1
torch==2.5.1
autogluon.tabular[lightgbm,catboost,xgboost]==1.3.1
lightautoml==0.4.2
flaml[automl]==2.3.6
openai==1.68.2
httpx==0.28.1
tenacity==9.0.0
langchain==0.3.27
langchain-openai==0.3.23
langgraph==0.6.7
pydantic==2.10.6
python-dotenv==1.0.1
PyYAML==6.0.2
orjson==3.10.15
jsonschema==4.23.0
matplotlib==3.9.4
seaborn==0.13.2
tqdm==4.67.1
psutil==6.1.1
cloudpickle==3.1.1
setuptools==80.9.0
"""
IMPORT_ALIASES = {
    "scikit-learn": "sklearn", "imbalanced-learn": "imblearn",
    "python-dotenv": "dotenv", "pyyaml": "yaml",
    "autogluon-tabular": "autogluon",
}


def pinned_packages(text):
    result = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, version = line.split("==")
        name = re.sub(r"[-_.]+", "-", name.split("[", 1)[0].lower())
        if name in result:
            raise ValueError(f"Duplicate package: {name}")
        result[name] = version
    return result


def write_task(task):
    inputs = task / "данные с пробелами"
    inputs.mkdir(parents=True)
    (task / "task.md").write_text(
        "Predict outcome using CSV files in данные с пробелами/.\n"
        "Metric: RMSE. Write submission.csv beside this task.md.\n", encoding="utf-8",
    )
    (inputs / "data_dictionary.csv").write_text(
        "feature,role,dtype\nrow_id,id,category\nx,feature,numeric\n"
        "kind,feature,category\noutcome,target,numeric\n", encoding="utf-8",
    )
    with (inputs / "train.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "x", "kind", "outcome"])
        writer.writerows((f"train-{i}", i, "а" if i % 2 else "б", 3 * i + i % 2)
                         for i in range(80))
    (inputs / "test.csv").write_text(
        "row_id,x,kind\n003,81,новое\n001,82,а\n002,83,б\n", encoding="utf-8",
    )
    (inputs / "sample_submission.csv").write_text(
        "row_id,outcome\n003,0\n001,0\n002,0\n", encoding="utf-8",
    )
    return {path: path.read_bytes() for path in task.rglob("*") if path.is_file()}


class PlatformContractTests(unittest.TestCase):
    def test_project_pins_are_available_at_the_exact_platform_versions(self):
        available = pinned_packages(PLATFORM_REQUIREMENTS)
        required = pinned_packages((ROOT / "requirements.txt").read_text(encoding="utf-8"))
        self.assertTrue(required)
        for name, version in required.items():
            with self.subTest(package=name):
                self.assertIn(name, available)
                self.assertEqual(version, available[name])

    def test_archive_contains_only_sources_and_available_static_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "solution.zip"
            build_archive(archive)
            self.assertLessEqual(archive.stat().st_size, 100_000_000)
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn("solution.py", names)
                self.assertEqual(len(names), len(set(names)))
                local = {PurePosixPath(name).parts[0].removesuffix(".py") for name in names}
                packages = pinned_packages(PLATFORM_REQUIREMENTS)
                imports = {IMPORT_ALIASES.get(name, name.replace("-", "_")) for name in packages}
                allowed = sys.stdlib_module_names | local | imports
                for name in names:
                    with self.subTest(path=name):
                        path = PurePosixPath(name)
                        self.assertFalse(path.is_absolute())
                        self.assertNotIn("..", path.parts)
                        self.assertTrue(name == "requirements.txt" or (
                            path.suffix == ".py" and len(path.parts) == 1
                        ))
                        if path.suffix != ".py":
                            continue
                        tree = ast.parse(zipped.read(name).decode("utf-8"), filename=name)
                        for node in ast.walk(tree):
                            modules = []
                            if isinstance(node, ast.Import):
                                modules = [alias.name for alias in node.names]
                            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                                modules = [node.module]
                            for module in modules:
                                self.assertIn(module.split(".")[0], allowed, f"{name}: {module}")

    def test_extracted_entrypoint_resolves_default_and_relative_task_paths(self):
        # No API credentials or repository PYTHONPATH may leak into the archive.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("OPENAI_") and key != "PYTHONPATH"}
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "solution.zip"
            build_archive(archive)
            package = root / "package"
            package.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                        (package / name).write_bytes(zipped.read(name))
            for default_task in (True, False):
                with self.subTest(default_task=default_task):
                    task = root / ("default task" if default_task else "relative task")
                    originals = write_task(task)
                    cwd = task if default_task else root
                    arguments = [] if default_task else [str(Path(task.name) / "task.md")]
                    completed = subprocess.run(
                        [sys.executable, str(package / "solution.py"), *arguments,
                         "--mode", "offline", "--time-budget", "60"],
                        cwd=cwd, env=env, capture_output=True, text=True,
                        encoding="utf-8", timeout=85,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertTrue(validate_predictions(task)["valid"])
                    self.assertFalse((package / "submission.csv").exists())
                    self.assertFalse((root / "submission.csv").exists())
                    self.assertIn("API calls: 0", completed.stdout)
                    with (task / "submission.csv").open(encoding="utf-8", newline="") as stream:
                        self.assertEqual([row["row_id"] for row in csv.DictReader(stream)], ["003", "001", "002"])
                    for path, content in originals.items():
                        self.assertEqual(path.read_bytes(), content, str(path))


if __name__ == "__main__":
    unittest.main()
