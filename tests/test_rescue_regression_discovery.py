"""Opt-in discovery stays bounded and never publishes partial schemas; no fits."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from build_submission import archive_files
from agent import history_discovery as original
from agent import regression_discovery as rescue


def panel():
    return pd.DataFrame({"entity": ["a", "b", "a", "b", "a", "b"],
                         "period": ["1", "1", "2", "2", "3", "3"]})


def arguments(frame):
    return {"feature_columns": list(frame.columns), "numeric_columns": ["period"],
            "period_column": "period"}


class AdaptiveDiscoveryTests(unittest.TestCase):
    def test_ordinary_default_policy_and_output_are_unchanged(self):
        frame = panel()
        before = original.discover_history_schema(frame, **arguments(frame))
        extended = rescue.discover_regression_schema(frame, **arguments(frame))
        after = original.discover_history_schema(frame, **arguments(frame))
        self.assertEqual(before, after)
        self.assertNotIn("check_budget", before["screening"])
        self.assertEqual(before["screening"]["caps"]["max_cell_visits"], 8_000_000)
        self.assertEqual(before["proposer_result"]["limits"]["caps"]["max_checked_configs"], 48)
        self.assertEqual(extended["screening"]["check_budget"],
                         {"requested": 192, "chosen": 192, "adaptive": True})
        self.assertEqual(extended["selected_config"], before["selected_config"])

    def test_dynamic_work_allocation_can_reduce_checks_but_not_exceed_total_cap(self):
        frame = panel()
        result = original.discover_history_schema(frame, **arguments(frame),
            max_cell_visits=200, max_checked_configs=192, adaptive_checks=True)
        # Six rows: 24 screening +18 parsing/bounds +6 checks*24 visits=186.
        self.assertEqual(result["screening"]["check_budget"]["chosen"], 6)
        self.assertEqual(result["screening"]["counts"]["total_cell_visits_planned"], 186)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["proposer_result"]["limits"]["caps"]["max_checked_configs"], 6)

    def test_no_budget_for_any_check_never_calls_proposer(self):
        frame = panel()
        with patch.object(original, "propose_history_schemas") as proposer:
            result = original.discover_history_schema(frame, **arguments(frame),
                max_cell_visits=60, max_checked_configs=192, adaptive_checks=True)
        proposer.assert_not_called()
        self.assertEqual(result["status"], "limited")
        self.assertIsNone(result["selected_config"])
        self.assertEqual(result["screening"]["check_budget"]["chosen"], 0)

    def test_large_requested_limits_cannot_disable_hard_caps(self):
        frame = panel()
        for options in ({"max_cell_visits": 12_000_001}, {"max_checked_configs": 193},
                        {"max_cell_visits": True}, {"max_checked_configs": 1.5},
                        {"max_checked_configs": 0}, {"adaptive_checks": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                original.discover_history_schema(frame, **arguments(frame), **options)

    def test_partial_result_with_a_proposal_still_never_selects(self):
        frame = panel()
        candidate = original.discover_history_schema(frame, **arguments(frame))["proposer_result"]
        candidate = deepcopy(candidate)
        candidate.update(status="limited", reason="check_limit")
        with patch.object(original, "propose_history_schemas", return_value=candidate):
            result = rescue.discover_regression_schema(frame, **arguments(frame))
        self.assertEqual(result["status"], "limited")
        self.assertIsNone(result["selected_config"])

    def test_extended_search_can_certify_a_schema_beyond_the_default_48_checks(self):
        random = np.random.default_rng(0)
        frame = pd.DataFrame([
            {"period": str(period), "entity": f"e{entity}" if entity < 20 else f"new{period}_{entity}",
             **{f"d{index}": f"v{random.integers(2)}" for index in range(8)}}
            for period in (1, 2, 3) for entity in range(40)])
        small = original.discover_history_schema(frame, **arguments(frame))
        large = rescue.discover_regression_schema(frame, **arguments(frame))
        self.assertEqual(small["status"], "limited")
        self.assertIsNone(small["selected_config"])
        self.assertEqual(large["status"], "proposed")
        self.assertGreater(large["proposer_result"]["limits"]["counts"]["checked_configs"], 48)
        self.assertTrue(large["proposer_result"]["limits"]["counts"]["search_complete"])

    def test_rows_columns_and_feature_order_do_not_change_schema(self):
        frame = panel()
        first = rescue.discover_regression_schema(frame, **arguments(frame))
        shuffled = frame.sample(frac=1, random_state=9).iloc[:, ::-1]
        second = rescue.discover_regression_schema(shuffled, **arguments(shuffled))
        self.assertEqual(first, second)


class BoundedDiscoveryTests(unittest.TestCase):
    @staticmethod
    def worker(source, directory, run_dir, step, timeout, max_memory_bytes):
        request = json.loads((directory / "request.json").read_text())
        frame = pd.read_csv(directory / "fit_features.csv", dtype=str, keep_default_na=False)
        discover = (original.discover_history_schema if "from agent.history_discovery import " in source
                    else rescue.discover_regression_schema)
        result = discover(frame, **request)
        (directory / "discovery.json").write_text(json.dumps(result))
        return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False,
                "resources": {"available": False, "peak_rss_bytes": None}}

    def call(self, frame, directory, **extra):
        return rescue.discover_regression_schema_bounded(frame, **arguments(frame), output_dir=directory, **extra)

    def test_worker_gets_only_features_fixed_budget_and_frozen_config(self):
        frame = panel()
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python", side_effect=self.worker) as run:
            directory = Path(temporary) / "discovery"
            report = self.call(frame, directory)
            self.assertEqual(report["status"], "proposed")
            self.assertEqual(report["selected_config"]["key_columns"], ["entity"])
            self.assertEqual(set(pd.read_csv(directory / "fit_features.csv").columns), set(frame.columns))
            self.assertEqual(set(json.loads((directory / "request.json").read_text())),
                             {"feature_columns", "numeric_columns", "period_column"})
            self.assertEqual(run.call_args.kwargs["max_memory_bytes"], 1024**3)
            self.assertGreater(run.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(run.call_args.kwargs["timeout"], 8)
            self.assertFalse(report["execution"]["resources"]["available"])
            self.assertIsNone(report["execution"]["resources"]["peak_rss_bytes"])
            for key in ("fit_input_sha256", "request_sha256", "worker_source_sha256", "result_sha256"):
                self.assertEqual(len(report[key]), 64)

    def test_default_policy_ablation_uses_the_original_worker_limits(self):
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python", side_effect=self.worker) as run:
            report = self.call(panel(), Path(temporary) / "d", extended=False)
        self.assertIn("from agent.history_discovery import discover_history_schema", run.call_args.args[0])
        self.assertEqual(report["max_cell_visits"], 8_000_000)
        self.assertEqual(report["max_checked_configs"], 48)
        self.assertNotIn("check_budget", report["result"]["screening"])
        self.assertEqual(report["result"]["proposer_result"]["limits"]["caps"]["max_checked_configs"], 48)

    def test_expired_parent_deadline_abstains_without_worker(self):
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python") as run:
            report = self.call(panel(), Path(temporary) / "d", deadline=time.monotonic() - 1)
        run.assert_not_called()
        self.assertEqual((report["status"], report["reason"]), ("limited", "preparation_time_limit"))
        self.assertIsNone(report["selected_config"])

    def test_timeout_and_memory_limit_never_return_selected_schema(self):
        for key, reason in (("timed_out", "worker_time_limit"), ("memory_limit_exceeded", "worker_memory_limit")):
            execution = {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False, key: True}
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary, patch(
                    "agent.execution.run_python", return_value=execution):
                report = self.call(panel(), Path(temporary) / "d")
            self.assertEqual((report["status"], report["reason"]), ("limited", reason))
            self.assertIsNone(report["selected_config"])

    def test_changed_worker_inputs_are_rejected(self):
        def worker(*args, **kwargs):
            result = self.worker(*args, **kwargs)
            (args[1] / "fit_features.csv").write_text("changed")
            return result
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python", side_effect=worker):
            report = self.call(panel(), Path(temporary) / "d")
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["selected_config"])

    def test_worker_invalid_or_partial_result_is_not_accepted(self):
        def worker(*args, **kwargs):
            result = self.worker(*args, **kwargs)
            path = args[1] / "discovery.json"
            value = json.loads(path.read_text())
            value["status"] = "limited"  # Illegal non-null selected_config.
            path.write_text(json.dumps(value))
            return result
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python", side_effect=worker):
            report = self.call(panel(), Path(temporary) / "d")
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["selected_config"])

    def test_undeclared_columns_never_reach_worker(self):
        frame = panel().assign(target="not permitted")
        with tempfile.TemporaryDirectory() as temporary, patch("agent.execution.run_python") as run:
            with self.assertRaises(ValueError):
                rescue.discover_regression_schema_bounded(frame, feature_columns=["entity", "period"],
                    numeric_columns=["period"], period_column="period", output_dir=Path(temporary) / "d")
        run.assert_not_called()


class FlatDiscoveryTests(unittest.TestCase):
    def test_root_only_copy_runs_both_bounded_discovery_workers_outside_source_tree(self):
        # Actual worker subprocesses search six feature-only rows; no model is
        # fitted. The parent and child must both import the flat archive, while
        # all request/result files live in a different working directory.
        with tempfile.TemporaryDirectory(prefix="regression-discovery-flat-") as temporary:
            root = Path(temporary)
            flat, working = root / "flat", root / "task"
            flat.mkdir()
            working.mkdir()
            sources = archive_files()
            self.assertIn("_alfa_agent__regression_discovery.py", sources)
            self.assertTrue(all("/" not in name for name in sources))
            for name, content in sources.items():
                (flat / name).write_bytes(content)
            probe = working / "probe.py"
            probe.write_text('''import hashlib,json,sys
from pathlib import Path
flat=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(flat))
import agent
import pandas as pd
from agent.regression_discovery import discover_regression_schema_bounded
assert Path(agent.__file__).resolve()==flat/"agent.py"
frame=pd.DataFrame({"entity":["a","b","a","b","a","b"],
                    "period":["1","1","2","2","3","3"]})
expected={name:hashlib.sha256((flat/("_alfa_agent__"+name)).read_bytes()).hexdigest()
          for name in ("regression_discovery.py","history_discovery.py","history_schema_proposer.py")}
reports=[]
for extended in (False,True):
    report=discover_regression_schema_bounded(frame,feature_columns=list(frame.columns),
        numeric_columns=["period"],period_column="period",
        output_dir=Path("extended" if extended else "original"),extended=extended)
    assert report["status"]=="proposed",json.dumps(report)
    assert report["execution"]["exit_code"]==0,json.dumps(report)
    assert report["selected_config"]["key_columns"]==["entity"],json.dumps(report)
    assert report["implementation_sha256"]==expected,json.dumps(report)
    reports.append({"extended":extended,"status":report["status"],
        "max_checked_configs":report["max_checked_configs"]})
print(json.dumps(reports))
''', encoding="utf-8")
            env = {name: value for name, value in os.environ.items()
                   if name not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY",
                                   "OPENAI_BASE_URL", "OPENAI_MODEL"}}
            env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            result = subprocess.run([sys.executable, str(probe), str(flat)], cwd=working,
                env=env, capture_output=True, text=True, encoding="utf-8", timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = json.loads(result.stdout)
            self.assertEqual(reports, [
                {"extended": False, "status": "proposed", "max_checked_configs": 48},
                {"extended": True, "status": "proposed", "max_checked_configs": 192},
            ])
            self.assertFalse((flat / "agent").exists())
            self.assertFalse((working / "agent").exists())
            self.assertEqual(sources, {name: (flat / name).read_bytes() for name in sources})


if __name__ == "__main__":
    unittest.main()
