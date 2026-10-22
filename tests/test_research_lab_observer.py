"""Exercise observation across the actual native baseline process boundary."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from tests.test_research_lab_runner import WORKER, worker_fixture


class BaselineObserverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def fixture(self, *, interrupted=False, observer_failure=False):
        config, task, arm = worker_fixture(self.root)
        solver = self.root / "solver"
        (solver / "agent" / "execution.py").write_text(
            "import subprocess,sys\nfrom pathlib import Path\n"
            "def _python_launch(script,task_dir):\n"
            " return [sys.executable,'-u','-c',"
            "\"import runpy,sys; runpy.run_path(sys.argv[1],run_name='__main__')\",str(script)],task_dir\n"
            "def run_python(code,task_dir,run_dir,step,timeout):\n"
            " assert timeout==2\n run_dir=Path(run_dir)\n run_dir.mkdir(parents=True)\n"
            " script=run_dir/f'experiment_{step:03d}.py'\n script.write_text(code)\n"
            " command,cwd=_python_launch(script,task_dir)\n"
            " result=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=8)\n"
            " return {'exit_code':result.returncode,'stderr':result.stderr}\n")
        (solver / "agent" / "automl.py").write_text(
            "import json,os\nfrom pathlib import Path\n"
            "RESULT={}\nEXECUTION={'exit_code':15,'timed_out':True,'memory_limit_exceeded':False,"
            "'elapsed_seconds':.25,'resources':{'available':True,'cleanup_complete':True,"
            "'peak_rss_bytes':1234,'cpu_seconds':.1,'process_id':os.getpid()}}\n"
            "def run_python(code,task_dir,run_dir,step,timeout):\n"
            " assert code=='protocol-no-fit' and step==1 and timeout==.5\n"
            + (" os._exit(7)\n" if interrupted else " return EXECUTION\n") +
            "def run_automl(task,**kwargs):\n"
            " report=Path(task).parent/'.agent_runs'/('b'*32)/'automl_report.json'\n"
            " report.parent.mkdir(parents=True)\n report.write_text('{}')\n"
            " directory=report.parent/'seed'/'seed'\n directory.mkdir(parents=True)\n"
            " result=run_python('protocol-no-fit',directory,directory/'execution',1,timeout=.5)\n"
            " assert result is EXECUTION\n"
            " RESULT['report_path']=str(report)\n return RESULT\n")
        baseline_code = (
            f"import sys\nsys.path.insert(0,{str(solver)!r})\n"
            "from pathlib import Path\nfrom agent.automl import run_automl\n"
            f"run_automl(Path({str(task / 'task.md')!r}),time_budget=1)\n")
        unrelated_code = (
            f"import sys\nsys.path.insert(0,{str(solver)!r})\n"
            "from agent import automl\nassert not hasattr(automl.run_python,'__wrapped__')\n")
        (solver / "agent" / "research_v3.py").write_text(
            "import json,os\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
            "def prepare_research(state):\n state.gate=None\n"
            "def run_agent(task,client,**kwargs):\n"
            " task=Path(task)\n directory=task.parent/'.agent_runs'/('a'*32)\n"
            " directory.mkdir(parents=True)\n"
            " (directory/'run.json').write_text('{}')\n"
            " (task.parent/'worker_pid.json').write_text(json.dumps(os.getpid()))\n"
            " state=SimpleNamespace(run_dir=directory,gate=None)\n prepare_research(state)\n"
            " from agent.execution import run_python\n"
            f" result=run_python({baseline_code!r},task.parent,directory/'startup_execution',0,timeout=2)\n"
            " (task.parent/'baseline_execution.json').write_text(json.dumps(result))\n"
            f" other=run_python({unrelated_code!r},task.parent,directory/'other_execution',0,timeout=2)\n"
            " assert other['exit_code']==0,other\n return directory\n")
        worker = WORKER
        if observer_failure:
            # Copying only the worker deliberately leaves its sibling bootstrap
            # absent. Failure must be recorded without breaking native routing.
            worker = self.root / "isolated_worker" / "worker.py"
            worker.parent.mkdir()
            shutil.copy2(WORKER, worker)
        return config, task, arm, worker, baseline_code

    def launch(self, config, worker):
        return subprocess.run([sys.executable, "-I", str(worker), str(config),
                               "--deadline", str(time.monotonic() + 10)],
                              capture_output=True, text=True, timeout=15)

    def test_nested_timeout_observation_preserves_cleanup_and_native_script(self):
        config, task, arm, worker, original_code = self.fixture()
        result = self.launch(config, worker)
        evidence = json.loads((arm / "worker_evidence.json").read_text())
        self.assertEqual(result.returncode, 0, evidence.get("terminal_error"))
        self.assertIsNone(evidence["terminal_error"])
        observed = [item for item in evidence["observations"] if item["phase"] == "automl_execution"]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["source"], "baseline_subprocess")
        self.assertEqual(observed[0]["directory"], ".agent_runs/" + "b" * 32 + "/seed/seed")
        execution = observed[0]["execution"]
        self.assertEqual(execution["exit_code"], 15)
        self.assertTrue(execution["timed_out"])
        self.assertFalse(execution["memory_limit_exceeded"])
        self.assertEqual(execution["elapsed_seconds"], .25)
        self.assertTrue(execution["resources"]["cleanup_complete"])
        self.assertEqual(execution["resources"]["peak_rss_bytes"], 1234)
        worker_pid = json.loads((task / "worker_pid.json").read_text())
        self.assertNotEqual(execution["resources"]["process_id"], worker_pid)
        self.assertNotEqual(execution["resources"]["process_id"], os.getpid())
        outer = task / ".agent_runs" / ("a" * 32)
        self.assertEqual((outer / "startup_execution" / "experiment_000.py").read_text(), original_code)
        sidecar = json.loads((outer / "startup" / "automl_observations.json").read_text())
        self.assertEqual(sidecar["errors"], [])
        self.assertEqual(sidecar["pending"], [])
        self.assertFalse(any(item["phase"] == "automl_observer_error" for item in evidence["observations"]))

    def test_interrupted_child_has_no_invented_execution(self):
        config, _, arm, worker, _ = self.fixture(interrupted=True)
        result = self.launch(config, worker)
        evidence = json.loads((arm / "worker_evidence.json").read_text())
        self.assertEqual(result.returncode, 0, evidence.get("terminal_error"))
        self.assertFalse(any(item["phase"] == "automl_execution" for item in evidence["observations"]))
        errors = [item for item in evidence["observations"] if item["phase"] == "automl_observer_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"], "AutoML child has no observed returned execution")

    def test_missing_observer_is_explicit_and_does_not_break_native_return(self):
        config, task, arm, worker, _ = self.fixture(observer_failure=True)
        result = self.launch(config, worker)
        evidence = json.loads((arm / "worker_evidence.json").read_text())
        self.assertEqual(result.returncode, 0, evidence.get("terminal_error"))
        self.assertTrue(evidence["solver_returned"])
        self.assertFalse(any(item["phase"] == "automl_execution" for item in evidence["observations"]))
        self.assertTrue(any(item["phase"] == "automl_observer_error" for item in evidence["observations"]))
        native = json.loads((task / "baseline_execution.json").read_text())
        self.assertEqual(native["exit_code"], 0)
        self.assertIn("AutoML observation unavailable", native["stderr"])


if __name__ == "__main__":
    unittest.main()
