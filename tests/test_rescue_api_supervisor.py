"""A hung probe and its child must not outlive the shared deadline."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import psutil

from experiment_rescue_api import _supervise


class ProbeSupervisorTests(unittest.TestCase):
    def test_hung_worker_and_spawned_child_are_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            code = ("import subprocess,sys,time,json; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    "print(json.dumps({'pid':p.pid}),flush=True); time.sleep(60)")
            started = time.monotonic()
            status, reason = _supervise([sys.executable, "-u", "-c", code], started + .6, path)
            self.assertEqual(reason, "deadline")
            self.assertNotEqual(status, 0)
            self.assertLess(time.monotonic() - started, 5)
            child = json.loads(path.read_text())["pid"]
            try:
                process = psutil.Process(child)
                self.assertEqual(process.status(), psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                pass


if __name__ == "__main__":
    unittest.main()
