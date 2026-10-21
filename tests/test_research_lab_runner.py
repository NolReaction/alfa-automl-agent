"""Process/coordination contracts using tiny programs; these tests never fit ML."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from research_lab import runner
from research_lab.integrity import file_hash, read_json, write_json
from research_lab.process import coordinator_lock, supervise


WORKER = Path(runner.__file__).with_name("worker.py")
COMMON = ["--mode", "offline", "--research-search", "adaptive", "--search-profile", "legacy"]


def worker_fixture(tmp_path, *, failure=False, short=False):
    root, arm = tmp_path / "solver", tmp_path / "arm"
    task = arm / "task"
    task.mkdir(parents=True)
    (task / "task.md").write_text("Tiny process fixture, no model training\n")
    (root / "agent").mkdir(parents=True)
    (root / "agent" / "__init__.py").write_text("")
    (root / "agent" / "research_v3.py").write_text(
        "import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "def prepare_research(state):\n"
        " state.gate=SimpleNamespace(frozen_split={'fit_rows':[0],'validation_rows':[1],"
        "'choice':{},'split':{}},inner_split={'strategy':'random'})\n"
        "def run_agent(task,client,**kwargs):\n"
        " directory=Path(task).parent/'.agent_runs'/('a'*32)\n"
        " directory.mkdir(parents=True)\n"
        " (directory/'run.json').write_text(json.dumps({'marker':'current','budget':kwargs['time_budget']}))\n"
        " decoy=directory.parent/('f'*32)\n decoy.mkdir()\n"
        " (decoy/'run.json').write_text('{}')\n"
        " state=SimpleNamespace(run_dir=directory,gate=None)\n prepare_research(state)\n"
        + (" raise RuntimeError('native failure')\n" if failure else " return directory\n"))
    (root / "agent" / "automl.py").write_text(
        "import json\nfrom pathlib import Path\nRESULT={}\n"
        "def run_automl(task,**kwargs):\n"
        " report=Path(task).parent/'.agent_runs'/('b'*32)/'automl_report.json'\n"
        " report.parent.mkdir(parents=True)\n report.write_text('{}')\n"
        " RESULT['report_path']=str(report)\n return RESULT\n")
    body = (
        "import time\ntime.sleep(.12)\nimport sys,json\nfrom pathlib import Path\n"
        "from agent.research_v3 import run_agent\n"
        "def main():\n"
        " args=sys.argv[2:]\n"
        " expected=['--mode','offline','--research-search','adaptive','--search-profile','legacy']\n"
        " assert args[:len(expected)]==expected\n"
        " budget=float(args[args.index('--time-budget')+1])\n"
        " task=Path(sys.argv[1])\n"
        " (task.parent/'argv.json').write_text(json.dumps(sys.argv))\n")
    if short:
        body += (" from agent import automl\n result=automl.run_automl(task,time_budget=budget)\n"
                 " assert result is automl.RESULT\n return 0\n")
    elif failure:
        body += (" try: run_agent(task,None,time_budget=budget,local_only=True,adaptive_search=True)\n"
                 " except RuntimeError: return 1\n return 0\n")
    else:
        body += (" result=run_agent(task,None,time_budget=budget,local_only=True,adaptive_search=True)\n"
                 " assert isinstance(result,Path)\n return 0\n")
    (root / "solution.py").write_text(body)
    config = {"version": 1, "run_id": "fixture-run", "case_key": "fixture", "arm": "incumbent",
              "task_dir": str(task), "task_path": str(task / "task.md"), "solver_root": str(root),
              "time_budget_seconds": 10, "threads": 8, "memory_limit_bytes": 24 * 1024**3,
              "require_adaptive": False, "worker_output": str(arm / "worker_evidence.json"),
              "solver_args": COMMON}
    config_path = arm / "config.json"
    write_json(config_path, config)
    return config_path, task, arm


def launch_worker(config, *, seconds=10):
    return subprocess.run([sys.executable, "-I", str(WORKER), str(config),
                           "--deadline", str(time.monotonic() + seconds)],
                          capture_output=True, text=True, timeout=15)
















def sealed_run_fixture(tmp_path, monkeypatch):
    from research_lab import audit, planning

    output = tmp_path / "run"
    output.mkdir()
    plan = {"python": sys.executable, "time_budget_seconds": 360, "threads": 8,
            "memory_limit_bytes": 24 * 1024**3, "solvers": {}, "cases": []}
    case = {"key": "fixture__1", "arms": {}, "arm_order": ["incumbent", "candidate"]}
    for arm in case["arm_order"]:
        arm_dir, run_id = output / arm, arm + "-run"
        task = arm_dir / "task"
        task.mkdir(parents=True)
        (task / "task.md").write_text("No model training")
        config = {"version": 1, "arm": arm, "case_key": case["key"], "run_id": run_id,
                  "task_dir": str(task), "worker_output": str(arm_dir / "worker_evidence.json"),
                  "solver_identity": {}, "time_budget_seconds": 360, "threads": 8,
                  "memory_limit_bytes": 24 * 1024**3}
        write_json(arm_dir / "config.json", config)
        write_json(arm_dir / "state.json", {"status": "prepared", "run_id": run_id})
        case["arms"][arm] = {"run_id": run_id, "task_dir": arm + "/task",
                             "config_path": arm + "/config.json", "state_path": arm + "/state.json"}
        plan["solvers"][arm] = {}
    plan["cases"].append(case)
    monkeypatch.setattr(planning, "load_plan", lambda path, execution=True: plan)
    monkeypatch.setattr(runner, "check_identity", lambda identity: identity)
    monkeypatch.setattr(runner, "machine_lock_path", lambda: tmp_path / "machine.lock")

    def fake_audit(output, case, arm, evidence, plan=None):
        entry = case["arms"][arm]
        state = read_json(output / entry["state_path"])
        execution = state.get("execution") or {}
        failed = state["status"] in {"failed", "interrupted"}
        qualification = ("censored" if failed and execution.get("cleanup", {}).get("complete") is not True
                         else "observed" if execution.get("resources", {}).get("available") else "provisional_resource_observation")
        index = {}
        directory = (output / entry["config_path"]).parent
        for path in directory.rglob("*"):
            if path.is_file() and path.name not in {"state.json", "state.sha256", "audit.json"}:
                index[path.relative_to(output).as_posix()] = file_hash(path)
        return {"status": "censored" if qualification == "censored" else "solver_failed" if failed else "completed",
                "quality_verified": not failed, "resources": {"qualification": qualification}, "artifact_hashes": index}

    monkeypatch.setattr(audit, "audit_arm", fake_audit)
    return output, plan


def fake_supervisor(calls, *, failed=False, observed=True):
    def run(argv, cwd, output_dir, timeout_seconds, memory_limit_bytes, *, threads):
        arm_dir = output_dir.parent
        unittest.TestCase().assertTrue(read_json(arm_dir / "state.json")["status"] == "started")
        unittest.TestCase().assertTrue(not output_dir.exists())
        unittest.TestCase().assertTrue(0 < timeout_seconds <= 360)
        unittest.TestCase().assertTrue(memory_limit_bytes == 24 * 1024**3 and threads == 8)
        calls.append(arm_dir.name)
        output_dir.mkdir()
        (output_dir / "stdout.txt").write_text("")
        write_json(arm_dir / "worker_evidence.json", {"version": 1})
        return {"exit_code": 1 if failed else 0, "external_timeout": failed,
                "memory_limit_exceeded": False, "elapsed_seconds": .02, "cleanup_elapsed_seconds": .001,
                "cleanup": {"complete": True if observed else None}, "resources": {"available": observed}}
    return run














def _fail(message):
    raise AssertionError(message)


class _MonkeyPatch:
    """Small stdlib patch stack shared by process-contract fixtures."""
    def __init__(self):
        self.stack = ExitStack()

    def setattr(self, target, name, value):
        self.stack.enter_context(patch.object(target, name, value))

    def setenv(self, name, value):
        self.stack.enter_context(patch.dict(os.environ, {name: value}))

    def close(self):
        self.stack.close()


class ResearchLabRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.monkeypatch = _MonkeyPatch()
        self.addCleanup(self.monkeypatch.close)

    def test_worker_charges_imports_and_uses_exact_returned_current_run(self):
        tmp_path = self.tmp_path
        config, task, arm = worker_fixture(tmp_path)
        process = launch_worker(config)
        self.assertTrue(process.returncode == 0, process.stderr)
        evidence = read_json(arm / "worker_evidence.json")
        self.assertTrue(0 < evidence["cli_time_budget_seconds"] < 10 - 2 - .12)
        self.assertTrue(evidence["original_time_budget_seconds"] == 10)
        self.assertTrue(evidence["cli_argv"] == json.loads((task / "argv.json").read_text()))
        self.assertTrue(evidence["config_sha256"] == file_hash(config))
        self.assertTrue(evidence["solver_returned"] is True)
        self.assertTrue(evidence["run_report"]["path"] == ".agent_runs/" + "a" * 32 + "/run.json")
        self.assertTrue(evidence["run_report"]["origin"] == "returned")
        self.assertTrue(evidence["run_report"]["sha256"] == file_hash(task / evidence["run_report"]["path"]))
        pointer = read_json(task / "solver_output/research_benchmark.json")
        self.assertTrue(pointer["run_report"] == evidence["run_report"]["path"])
        self.assertTrue(pointer["run_report_sha256"] == evidence["run_report"]["sha256"])
        self.assertTrue(evidence["observations"][1]["frozen_split"]["fit_rows"] == [0])
        self.assertTrue(len(evidence["observations"]) == 2)


    def test_worker_captures_failed_native_state_without_claiming_return(self):
        tmp_path = self.tmp_path
        config, _, arm = worker_fixture(tmp_path, failure=True)
        self.assertTrue(launch_worker(config).returncode == 1)
        evidence = read_json(arm / "worker_evidence.json")
        self.assertTrue(evidence["solver_returned"] is False)
        self.assertTrue(evidence["cli_exit_code"] == 1)
        self.assertTrue(evidence["run_report"]["origin"] == "observed_state_after_failure")
        self.assertTrue(evidence["run_report"]["path"] == ".agent_runs/" + "a" * 32 + "/run.json")


    def test_worker_short_automl_hook_preserves_returned_dictionary(self):
        tmp_path = self.tmp_path
        config, task, arm = worker_fixture(tmp_path, short=True)
        process = launch_worker(config)
        self.assertTrue(process.returncode == 0, process.stderr)
        evidence = read_json(arm / "worker_evidence.json")
        self.assertTrue(evidence["solver_returned"] is True)
        self.assertTrue(evidence["run_report"]["kind"] == "automl")
        self.assertTrue(evidence["run_report"]["path"].endswith("automl_report.json"))
        self.assertTrue(not (task / "solver_output/research_benchmark.json").exists())


    def test_worker_refuses_oracle_fields_and_second_attempt(self):
        tmp_path = self.tmp_path
        config, _, arm = worker_fixture(tmp_path)
        value = read_json(config)
        value["audit_path"] = "private_answers.json"
        write_json(config, value)
        self.assertTrue(launch_worker(config).returncode != 0)
        self.assertTrue(not (arm / "worker_evidence.json").exists())
        del value["audit_path"]
        write_json(config, value)
        self.assertTrue(launch_worker(config).returncode == 0)
        previous = file_hash(arm / "worker_evidence.json")
        self.assertTrue(launch_worker(config).returncode != 0)
        self.assertTrue(file_hash(arm / "worker_evidence.json") == previous)


    def test_worker_does_not_silently_degrade_adaptive_after_slow_bootstrap(self):
        tmp_path = self.tmp_path
        config, task, arm = worker_fixture(tmp_path)
        value = read_json(config)
        value.update(time_budget_seconds=360, require_adaptive=True)
        write_json(config, value)
        self.assertTrue(launch_worker(config, seconds=299).returncode == 1)
        evidence = read_json(arm / "worker_evidence.json")
        self.assertTrue(evidence["terminal_error"]["type"] == "TimeoutError")
        self.assertTrue(not (task / "argv.json").exists())


    def test_supervisor_adapter_keeps_cleanup_separate_and_environment_private(self):
        tmp_path = self.tmp_path
        monkeypatch = self.monkeypatch
        import qualify_unseen_tasks as trusted
        original_environment = trusted.sanitized_environment

        def stop(owned):
            time.sleep(.01)
            return {"complete": None}

        def run(*args):
            environment = trusted.sanitized_environment()
            self.assertTrue(environment["OMP_NUM_THREADS"] == "3")
            self.assertTrue(environment["CUDA_VISIBLE_DEVICES"] == "")
            self.assertTrue(not any(name in environment for name in ("OPENAI_API_KEY", "HTTP_PROXY", "PYTHONPATH")))
            cleanup = trusted._OwnedProcesses.stop(None)
            return {"elapsed_seconds": .5, "cleanup": cleanup, "resources": {"available": False}}

        monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
        monkeypatch.setenv("HTTP_PROXY", "not-a-proxy")
        monkeypatch.setenv("PYTHONPATH", "not-an-import-path")
        monkeypatch.setattr(trusted._OwnedProcesses, "stop", stop)
        monkeypatch.setattr(trusted, "run_solver", run)
        result = supervise(["python"], tmp_path, tmp_path / "logs", 10, 24 * 1024**3, threads=3)
        self.assertTrue(result["cleanup_elapsed_seconds"] >= .01)
        self.assertTrue(result["worker_wall_seconds"] < result["elapsed_seconds"])
        self.assertTrue(result["cleanup"]["complete"] is None)
        self.assertTrue(trusted.sanitized_environment is original_environment)
        self.assertTrue(trusted._OwnedProcesses.stop is stop)


    def test_coordinator_lock_excludes_another_coordinator(self):
        tmp_path = self.tmp_path
        path = tmp_path / "lane.lock"
        with coordinator_lock(path):
            with self.assertRaisesRegex(RuntimeError, "coordinator"):
                with coordinator_lock(path):
                    _fail("Second coordinator entered the same CPU lane")
        with coordinator_lock(path):
            pass


    def test_runner_durable_started_before_spawn_and_resume_never_repeats(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        calls = []
        monkeypatch.setattr(runner, "supervise", fake_supervisor(calls))
        result = runner.run_plan(output)
        self.assertTrue(calls == ["incumbent", "candidate"])
        self.assertTrue(result["planned_arms"] == 2 and result["status"] == "completed")
        for arm in calls:
            self.assertTrue((output / arm / "state.sha256").read_text().strip() == file_hash(output / arm / "state.json"))
        self.assertTrue(runner.run_plan(output, resume=True)["counts"] == {"finished": 2})
        self.assertTrue(calls == ["incumbent", "candidate"])


    def test_runner_success_without_resource_observation_remains_provisional(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        calls = []
        monkeypatch.setattr(runner, "supervise", fake_supervisor(calls, observed=False))
        result = runner.run_plan(output)
        self.assertTrue(len(calls) == 2)
        self.assertTrue(all(row["outcome"] == "provisional_resource_observation" for row in result["arms"]))
        self.assertTrue(result["quality_admission"] is False)


    def test_runner_failed_unknown_cleanup_stops_and_preserves_denominator(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        calls = []
        monkeypatch.setattr(runner, "supervise", fake_supervisor(calls, failed=True, observed=False))
        result = runner.run_plan(output)
        self.assertTrue(calls == ["incumbent"])
        self.assertTrue(result["planned_arms"] == 2)
        self.assertTrue(result["counts"] == {"censored": 1, "not_started": 1})
        self.assertTrue(runner.run_plan(output, resume=True)["status"] == "stopped")
        self.assertTrue(calls == ["incumbent"])


    def test_runner_crashed_started_arm_cannot_receive_a_fresh_budget(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        state_path = output / "incumbent/state.json"
        state = read_json(state_path)
        state["status"] = "started"
        write_json(state_path, state)
        monkeypatch.setattr(runner, "supervise", lambda *a, **kw: _fail("Interrupted arm relaunched"))
        result = runner.run_plan(output, resume=True)
        self.assertTrue(result["status"] == "stopped")
        self.assertTrue(result["counts"] == {"censored": 1, "not_started": 1})
        self.assertTrue(read_json(output / "candidate/state.json")["status"] == "prepared")


    def test_runner_spawn_error_is_consumed_and_censored(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise OSError("Could not spawn")

        monkeypatch.setattr(runner, "supervise", fail)
        result = runner.run_plan(output)
        self.assertTrue(len(calls) == 1)
        self.assertTrue(result["counts"] == {"censored": 1, "not_started": 1})
        self.assertTrue(read_json(output / "incumbent/state.json")["execution"]["cleanup"]["complete"] is None)


    def test_runner_detects_terminal_state_tampering(self):
        monkeypatch = self.monkeypatch
        sealed_run = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        output, _ = sealed_run
        monkeypatch.setattr(runner, "supervise", fake_supervisor([]))
        runner.run_plan(output)
        state_path = output / "incumbent/state.json"
        state = read_json(state_path)
        state["outcome"] = "tampered"
        write_json(state_path, state)
        with self.assertRaisesRegex(ValueError, "state seal"):
            runner.run_plan(output, resume=True)

    def test_runner_rechecks_full_plan_between_arms_and_stops_on_drift(self):
        from research_lab import planning
        output, plan = sealed_run_fixture(self.tmp_path, self.monkeypatch)
        calls, checks = [], []
        self.monkeypatch.setattr(runner, "supervise", fake_supervisor(calls))

        def load(path, execution=True):
            checks.append(execution)
            if len(checks) == 3:
                raise ValueError("Arm input drift")
            return plan

        self.monkeypatch.setattr(planning, "load_plan", load)
        with self.assertRaisesRegex(ValueError, "drift"):
            runner.run_plan(output)
        self.assertTrue(calls == ["incumbent"])
        self.assertTrue(checks == [True, True, True])
        self.assertTrue(read_json(output / "candidate/state.json")["status"] == "prepared")
        self.assertTrue(read_json(output / "run_report.json")["counts"] == {"finished": 1, "not_started": 1})



if __name__ == "__main__":
    unittest.main()
