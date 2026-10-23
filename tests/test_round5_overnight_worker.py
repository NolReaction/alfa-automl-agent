"""Small native protocol fixtures; these tests never import ML or fit a model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from build_submission import _FLAT_LOADER


WORKER = Path(__file__).resolve().parents[1] / "experiment_round5_overnight_worker.py"

RESEARCH = '''
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

def prepare_research(state):
    state.gate = SimpleNamespace(
        frozen_split={"fit_rows": [0, 1], "validation_rows": [2],
                      "choice": {"strategy": "random"}, "split": {"strategy": "random"}},
        inner_split={"strategy": "random", "fixture": "original"})
    return "original-prepare-result"

def run_agent_v3(task_path, **kwargs):
    task = Path(task_path).parent
    run = task / ".agent_runs" / ("a" * 32)
    run.mkdir(parents=True)
    state = SimpleNamespace(run_dir=run, report={"published_recipe": "baseline"})
    assert prepare_research(state) == "original-prepare-result"
    code, recipe = run / "research_candidate.py", run / "research_recipe.json"
    code.write_text("# fixture only\\n", encoding="utf-8")
    recipe.write_text('{"fixture":true}', encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    state.baseline = SimpleNamespace(code_path=code, recipe_path=recipe,
                                    code_hash=digest(code), recipe_hash=digest(recipe))
    state.report.update(kwargs)
    (run / "run.json").write_text(json.dumps(state.report), encoding="utf-8")
    decoy = task / ".agent_runs" / ("b" * 32)
    decoy.mkdir()
    (decoy / "run.json").write_text('{"decoy":true}', encoding="utf-8")
    return run
'''

SOLUTION = '''
from pathlib import Path
import sys
from agent.research_v3 import run_agent_v3 as run_agent

def main():
    run_agent(Path(sys.argv[1]), time_budget=float(sys.argv[-1]), cli_args=sys.argv[2:])
    return 0
'''


class OvernightWorkerTests(unittest.TestCase):
    def fixture(self, base, profile="regression_off"):
        root, job = base / "flat solver", base / "job"
        task = job / "task"
        root.mkdir(parents=True)
        task.mkdir(parents=True)
        (root / "agent.py").write_text(_FLAT_LOADER.replace("__MANIFEST__", repr({
            "agent.research_v3": ("_alfa_agent__research_v3.py", False)})), encoding="utf-8")
        (root / "_alfa_agent__research_v3.py").write_text(RESEARCH, encoding="utf-8")
        (root / "solution.py").write_text(SOLUTION, encoding="utf-8")
        (task / "task.md").write_text("protocol fixture", encoding="utf-8")
        config = {"version": 1, "job_id": profile, "profile": profile,
                  "task_dir": str(task), "task_path": str(task / "task.md"),
                  "solver_root": str(root), "worker_output": str(job / "worker_evidence.json"),
                  "time_budget_seconds": 600}
        path = job / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return root, job, task, path, config

    def launch(self, config, deadline=None):
        return subprocess.run([sys.executable, "-I", str(WORKER), str(config), "--deadline",
                               str(time.monotonic() + 599 if deadline is None else deadline)],
                              capture_output=True, text=True, timeout=15)

    def test_profiles_retain_original_objects_and_exact_returned_pointer(self):
        for profile, flag in (("regression_off", "--no-rescue-regression"),
                              ("regression_on", "--rescue-regression"),
                              ("classification", "--no-rescue-regression")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root, job, task, config_path, config = self.fixture(Path(temporary), profile)
                result = self.launch(config_path)
                self.assertEqual(result.returncode, 0, result.stderr)
                evidence = json.loads((job / "worker_evidence.json").read_text(encoding="utf-8"))
                self.assertTrue(evidence["solver_returned"])
                self.assertIsNone(evidence["terminal_error"])
                self.assertEqual(evidence["run_report"]["origin"], "returned")
                self.assertEqual(evidence["run_report"]["path"], ".agent_runs/" + "a" * 32 + "/run.json")
                self.assertIn(flag, evidence["cli_argv"])
                self.assertLess(evidence["cli_time_budget_seconds"], 600)
                self.assertLessEqual(evidence["native_ended_monotonic"], evidence["original_deadline_monotonic"])
                self.assertEqual(evidence["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
                observations = evidence["observations"]
                self.assertEqual([item["phase"] for item in observations],
                                 ["run_agent", "prepare_research", "frozen_publication"])
                self.assertEqual(observations[0]["time_budget_seconds"], evidence["cli_time_budget_seconds"])
                self.assertEqual(observations[1]["frozen_split"]["validation_rows"], [2])
                self.assertEqual(observations[1]["inner_split"]["fixture"], "original")
                original = observations[2]["frozen_source"]
                self.assertEqual(original["code_hash"], original["declared_code_hash"])
                self.assertEqual(original["recipe_hash"], original["declared_recipe_hash"])
                pointer = json.loads((task / "solver_output" / "research_benchmark.json").read_text(encoding="utf-8"))
                self.assertEqual(pointer["run_report"], evidence["run_report"]["path"])
                self.assertEqual(pointer["run_report_sha256"], evidence["run_report"]["sha256"])

    def test_unknown_or_extra_config_fields_are_rejected_before_execution(self):
        for extra in ({"oracle_actuals": [1, 2]}, {"profile": "arbitrary_cli"}, {"version": True},
                      {"research_search": "unbounded"}, {"research_search": True}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                root, job, task, path, config = self.fixture(Path(temporary))
                config.update(extra)
                path.write_text(json.dumps(config), encoding="utf-8")
                result = self.launch(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((task / ".agent_runs").exists())

    def test_explicit_budget_search_reaches_packaged_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, job, task, path, config = self.fixture(Path(temporary))
            config["research_search"] = "adaptive-budget"
            path.write_text(json.dumps(config), encoding="utf-8")
            result = self.launch(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads((job / "worker_evidence.json").read_text(encoding="utf-8"))
            argv = evidence["cli_argv"]
            self.assertEqual(argv[argv.index("--research-search") + 1], "adaptive-budget")

    def test_existing_run_evidence_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, job, task, path, config = self.fixture(Path(temporary))
            (task / ".agent_runs").mkdir()
            result = self.launch(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((job / "worker_evidence.json").exists())

    def test_expired_or_extended_deadline_cannot_launch_solver(self):
        for delta in (-1, 1000):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as temporary:
                root, job, task, path, config = self.fixture(Path(temporary))
                result = self.launch(path, time.monotonic() + delta)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((task / ".agent_runs").exists())

    def test_native_error_is_reported_and_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, job, task, path, config = self.fixture(Path(temporary))
            (root / "solution.py").write_text(SOLUTION.replace("return 0", 'raise RuntimeError("original-failure")'),
                                               encoding="utf-8")
            result = self.launch(path)
            self.assertNotEqual(result.returncode, 0)
            evidence = json.loads((job / "worker_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["cli_exit_code"], 1)
            self.assertEqual(evidence["terminal_error"], {"type": "RuntimeError", "message": "original-failure"})


if __name__ == "__main__":
    unittest.main()
