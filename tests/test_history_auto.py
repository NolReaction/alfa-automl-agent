"""Opt-in discovery crosses the v3 boundary using only INNER-fit features.

The baseline is a constant test fixture. Candidate fitting/scoring and the
discovery worker use real bounded subprocesses; there is no portfolio or API.
"""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agent import history_discovery_host as discovery_host
from agent.execution import run_python
from agent.research_candidate import freeze_candidate, run_candidate
from agent.research_v3 import run_agent_v3
from agent.research_workspace import ResearchWorkspace
from tests.test_history_research import ENTITIES, TEST_IDS, make_panel
from tests.test_team_integration import CONSTANT_CODE, fixture_baseline
from validation import validate_predictions


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def ids_hash(ids):
    return sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


class AutoHistoryV3Tests(unittest.TestCase):
    def run_panel(self, task, *, discovery_failure=None):
        executions, workers, gates, baselines = [], [], [], []

        def baseline(state):
            fixture_baseline(state)
            baselines.append((state.baseline, state.baseline.code_path.read_bytes(),
                              state.baseline.recipe_path.read_bytes()))

        def workspace(*args, **kwargs):
            gate = ResearchWorkspace(*args, **kwargs)
            gates.append(gate)
            return gate

        def discover(source, directory, logs, number, **kwargs):
            directory = Path(directory)
            payload = (directory / "fit_features.csv").read_bytes()
            workers.append({"source": source, "payload": payload, "directory": directory,
                            "request": json.loads((directory / "request.json").read_text(encoding="utf-8")),
                            "timeout": kwargs["timeout"], "memory": kwargs["max_memory_bytes"]})
            if discovery_failure == "timeout":
                return {"exit_code": None, "timed_out": True}
            if discovery_failure == "worker_error":
                return {"exit_code": 1, "error": "Fixture discovery worker error"}
            return run_python(source, directory, logs, number, **kwargs)

        def execute(candidate, **kwargs):
            result = run_candidate(candidate, **kwargs)
            executions.append({"phase": kwargs["phase"], "code_hash": candidate.code_hash,
                               "recipe_hash": candidate.recipe_hash, "recipe": copy.deepcopy(candidate.recipe),
                               "output": result.output_path, "receipt": result.receipt_path})
            return result

        with patch("agent.research_v3.prepare_baseline", side_effect=baseline), \
                patch("agent.research_v3.ResearchWorkspace", side_effect=workspace), \
                patch("agent.research_v3.local_candidate_plan", return_value=[]), \
                patch("agent.research_v3.run_candidate", side_effect=execute), \
                patch.object(discovery_host, "run_python", side_effect=discover), \
                patch("agent.research_v3._request_completion", side_effect=AssertionError("External API")) as api, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_dir = run_agent_v3(task, None, local_only=True, history_auto=True, time_budget=90)
        api.assert_not_called()
        self.assertEqual((len(gates), len(baselines), len(workers)), (1, 1, 1))
        baseline_candidate, baseline_code, baseline_recipe = baselines[0]
        baseline_candidate.verify()
        self.assertEqual(baseline_candidate.code_path.read_bytes(), baseline_code)
        self.assertEqual(baseline_candidate.recipe_path.read_bytes(), baseline_recipe)
        report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["frozen_baseline"], {"code_hash": baseline_candidate.code_hash,
                                                   "recipe_hash": baseline_candidate.recipe_hash})
        self.assertEqual(report["api"]["attempts"], 0)
        self.assertEqual(report["api"]["responses"], 0)
        self.assertTrue(report["quality_options"]["history_auto"])
        self.assertEqual(report["local_research"]["attempted"], 0)
        return run_dir, report, executions, workers[0], gates[0]

    def assert_discovery_payload(self, report, worker, gate):
        discovery = report["history_discovery"]
        self.assertEqual(discovery["scope"], "inner_fit_features_only")
        self.assertFalse(discovery["identity_verified"])
        self.assertEqual(discovery["blend_weight"], 0.5)
        self.assertEqual(discovery["fit_rows"], len(gate.fit_rows))
        self.assertEqual(discovery["fit_input_sha256"], sha256(worker["payload"]))
        self.assertEqual(discovery["worker_source_text_sha256"], sha256(worker["source"].encode("utf-8")))
        self.assertEqual(discovery["worker_source_sha256"],
                         sha256((worker["directory"] / "execution" / "experiment_000.py").read_bytes()))
        approved = gate.research_train.iloc[gate.fit_rows]
        self.assertEqual(discovery["fit_ids_sha256"], ids_hash(approved[gate.spec.id_column].tolist()))
        inputs = pd.read_csv(io.BytesIO(worker["payload"]), dtype=str, keep_default_na=False)
        self.assertEqual(inputs.columns.tolist(), list(gate.spec.features))
        self.assertTrue({gate.spec.id_column, gate.spec.target_column}.isdisjoint(inputs.columns))
        pd.testing.assert_frame_equal(inputs, approved.loc[:, list(gate.spec.features)].reset_index(drop=True))
        self.assertEqual(set(inputs["period"]), {"1", "2"})
        self.assertEqual(set(inputs["entity"]), set(ENTITIES))
        self.assertNotIn("unseen", set(inputs["entity"]))
        self.assertEqual(worker["request"], {"feature_columns": list(gate.spec.features),
            "numeric_columns": list(gate.spec.numeric), "period_column": "period"})
        self.assertGreater(worker["timeout"], 0)
        self.assertLessEqual(worker["timeout"], discovery_host.DISCOVERY_SECONDS)
        self.assertEqual(worker["memory"], discovery_host.DISCOVERY_MEMORY_BYTES)

    def test_real_discovery_receives_exact_inner_features_then_freezes_one_measured_history_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            task = make_panel(directory)
            _, report, calls, worker, gate = self.run_panel(task)
            self.assert_discovery_payload(report, worker, gate)
            discovery = report["history_discovery"]
            self.assertEqual((discovery["status"], discovery["reason"]), ("selected", "one_schema_frozen"))
            config = discovery["selected_config"]
            self.assertEqual(config, {"key_columns": ["entity"], "period_column": "period",
                "numeric_columns": [], "positive_columns": [], "ambiguity_policy": "past_unique_only"})
            self.assertEqual(discovery["result"]["screening"]["counts"]["proposer_calls"], 1)
            self.assertLessEqual(discovery["result"]["proposer_result"]["limits"]["counts"]["checked_configs"], 48)
            self.assertEqual(report["history_research"]["attempted"], 1)
            self.assertEqual(report["history_successful_experiments"], 1)
            self.assertEqual(report["history_research"]["history_config"], config)
            frozen = report["frozen_candidate"]
            self.assertEqual(frozen, report["history_research"]["candidate"])
            candidate_calls = [row for row in calls if row["code_hash"] == frozen["code_hash"]]
            self.assertEqual([row["phase"] for row in candidate_calls],
                             ["inner_selection", "outer_validation", "full_refit"])
            for row in candidate_calls:
                self.assertEqual(row["recipe_hash"], frozen["recipe_hash"])
                self.assertEqual(row["recipe"]["history_config"], config)
                self.assertEqual(row["recipe"]["blend_weight"], 0.5)
                receipt = json.loads(row["receipt"].read_text(encoding="utf-8"))
                self.assertEqual((receipt["code_hash"], receipt["recipe_hash"]),
                                 (frozen["code_hash"], frozen["recipe_hash"]))
            self.assertEqual(report["published_recipe"], "candidate")
            self.assertEqual((report["status"], report["reason"]), ("completed", "local_only"))
            self.assertEqual(validate_predictions(task.parent), {"valid": True, "errors": []})
            submission = pd.read_csv(task.parent / "submission.csv", dtype=str, keep_default_na=False)
            self.assertEqual(submission["row_key"].tolist(), TEST_IDS)
            np.testing.assert_allclose(submission["outcome"].astype(float), [22.5, 27.5, 35], rtol=0, atol=1e-12)

    def test_outer_target_intervention_cannot_change_discovery_input_config_or_inner_selection(self):
        evidence, outer = [], []
        with tempfile.TemporaryDirectory() as directory:
            for name, intervention in (("original", False), ("intervened", True)):
                task = make_panel(Path(directory) / name, intervene_outer=intervention)
                _, report, _, worker, gate = self.run_panel(task)
                self.assert_discovery_payload(report, worker, gate)
                discovery = report["history_discovery"]
                self.assertEqual(discovery["status"], "selected")
                evidence.append((worker["payload"], discovery["fit_input_sha256"], discovery["fit_ids_sha256"],
                    discovery["selected_config"], report["frozen_candidate"], report["frozen_baseline"],
                    report["evaluations"]["selection"], report["history_research"]["feedback"]["metric"]))
                outer.append(report["evaluations"]["final_check"]["candidate"]["value"])
        self.assertEqual(evidence[0], evidence[1])
        self.assertGreater(outer[1], outer[0] + 100000)

    def test_discovery_timeout_or_failure_still_refits_and_publishes_verified_baseline(self):
        for failure, status, reason in (("timeout", "limited", "worker_time_limit"),
                                        ("worker_error", "failed", "discovery_failed")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                task = make_panel(directory)
                _, report, calls, _, _ = self.run_panel(task, discovery_failure=failure)
                self.assertEqual((report["history_discovery"]["status"], report["history_discovery"]["reason"]),
                                 (status, reason))
                self.assertIsNone(report["history_discovery"]["selected_config"])
                self.assertEqual(report["history_research"]["attempted"], 0)
                self.assertEqual(report["published_recipe"], "baseline")
                self.assertEqual([row["phase"] for row in calls],
                                 ["inner_selection", "outer_validation", "full_refit"])
                self.assertTrue(all(row["code_hash"] == report["frozen_baseline"]["code_hash"] for row in calls))
                self.assertEqual(validate_predictions(task.parent), {"valid": True, "errors": []})
                self.assertEqual((task.parent / "submission.csv").read_bytes(), calls[-1]["output"].read_bytes())


class DiscoveryHostGuardTests(unittest.TestCase):
    def make_state(self, directory):
        directory = Path(directory)
        source = directory / "baseline.py"
        source.write_text(CONSTANT_CODE, encoding="utf-8")
        baseline = freeze_candidate(source, {"family": "constant"}, directory / "baseline_frozen")
        marker = directory / "source_input.csv"
        marker.write_bytes(b"original\n")

        def check_source():
            if marker.read_bytes() != b"original\n":
                raise ValueError("Original source input changed")

        frame = pd.DataFrame({"row_key": [f"000{number}" for number in range(6)],
            "entity": ["a", "b"] * 3, "period": ["1", "1", "2", "2", "3", "3"],
            "outcome": [10, 20] * 3})
        gate = SimpleNamespace(manifest={}, spec=SimpleNamespace(metric=SimpleNamespace(task_type="regression"),
            features=("entity", "period"),
            numeric=("period",), id_column="row_key", target_column="outcome"),
            choice={"strategy": "time", "time_column": "period"},
            inner_split={"strategy": "time", "time_column": "period"}, fit_rows=[0, 1, 2, 3], held_rows=[4, 5],
            research_train=frame, check_inputs=Mock(side_effect=check_source))
        state = SimpleNamespace(run_dir=directory, gate=gate, dialogue_closed=False, baseline=baseline,
            baseline_inner={"name": "RMSE", "value": 5.0, "higher_is_better": False},
            report={"stage": "inner_research", "extensions": {}}, event=Mock(),
            research_timeout=Mock(return_value=90), check_source=Mock(side_effect=check_source))
        return state, marker

    def test_ineligible_states_do_not_start_worker_or_modify_baseline(self):
        cases = {"iid": "time_validation_required", "classification": "regression_required",
            "no_baseline": "measured_baseline_required", "no_baseline_metric": "measured_baseline_required",
            "reserve": "finalization_reserve", "closed": "research_not_open",
            "wrong_inner": "matching_inner_time_required", "no_inner": "inner_validation_unavailable",
            "missing_period": "period_not_in_features", "semantics": "task_semantics_unavailable"}
        for case, reason in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                state, _ = self.make_state(directory)
                if case == "iid":
                    state.gate.choice = {"strategy": "random"}
                elif case == "classification":
                    state.gate.spec.metric.task_type = "classification"
                elif case == "no_baseline":
                    state.baseline = None
                elif case == "no_baseline_metric":
                    state.baseline_inner = None
                elif case == "reserve":
                    state.research_timeout.return_value = 2
                elif case == "closed":
                    state.dialogue_closed = True
                elif case == "wrong_inner":
                    state.gate.inner_split["time_column"] = "other"
                elif case == "no_inner":
                    state.gate.held_rows = []
                elif case == "missing_period":
                    state.gate.spec.features = ("entity",)
                elif case == "semantics":
                    state.gate.manifest = {"task_semantics": {"adaptive_research_allowed": False}}
                baseline = state.baseline
                with patch.object(discovery_host, "run_python") as worker:
                    self.assertIsNone(discovery_host.discover_history_options(state))
                report = state.report["history_discovery"]
                self.assertEqual((report["status"], report["reason"]), ("skipped", reason))
                self.assertIs(state.baseline, baseline)
                if baseline is not None:
                    baseline.verify()
                worker.assert_not_called()
                self.assertFalse((Path(directory) / "history_discovery").exists())

    def test_shape_and_value_limits_abstain_before_worker_without_sampling_or_truncation(self):
        for kind, reason in (("shape", "input_shape_limit"), ("value", "raw_value_limit"), ("bytes", "input_byte_limit")):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                state, _ = self.make_state(directory)
                context = contextlib.nullcontext()
                if kind == "shape":
                    context = patch.object(discovery_host, "MAX_ROWS", 3)
                elif kind == "value":
                    state.gate.research_train.loc[0, "entity"] = "x" * (discovery_host.MAX_VALUE_CHARS + 1)
                else:
                    context = patch.object(discovery_host, "MAX_CSV_BYTES", 1)
                with context, patch.object(discovery_host, "run_python") as worker:
                    self.assertIsNone(discovery_host.discover_history_options(state))
                self.assertEqual((state.report["history_discovery"]["status"],
                                  state.report["history_discovery"]["reason"]), ("limited", reason))
                worker.assert_not_called()
                self.assertIsNone(state.report["history_discovery"]["selected_config"])
                state.baseline.verify()

    def test_input_preparation_uses_discovery_deadline_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.make_state(directory)
            late = discovery_host.DISCOVERY_SECONDS + 1
            readings = iter((0.0, 0.0, 0.0, late, late))
            with patch.object(discovery_host, "_now", side_effect=lambda: next(readings)), \
                    patch.object(discovery_host, "run_python") as worker:
                self.assertIsNone(discovery_host.discover_history_options(state))
            self.assertEqual((state.report["history_discovery"]["status"], state.report["history_discovery"]["reason"]),
                             ("limited", "preparation_time_limit"))
            worker.assert_not_called()

    def test_worker_allowance_covers_a_cold_start_but_never_the_host_reserve(self):
        # A cold Python + pandas start alone took 11 s on a loaded laptop; 8 s lost the schema.
        for research_seconds, low, high in ((90, 30, discovery_host.DISCOVERY_SECONDS), (10, 0, 10)):
            with self.subTest(research_seconds=research_seconds), tempfile.TemporaryDirectory() as directory:
                state, _ = self.make_state(directory)
                state.research_timeout.return_value = research_seconds
                with patch.object(discovery_host, "run_python",
                                  return_value={"exit_code": None, "timed_out": True}) as worker:
                    self.assertIsNone(discovery_host.discover_history_options(state))
                timeout = worker.call_args.kwargs["timeout"]
                self.assertGreater(timeout, low)
                self.assertLessEqual(timeout, high)
                self.assertEqual(state.report["history_discovery"]["reason"], "worker_time_limit")

    def test_worker_error_does_not_swallow_source_or_baseline_integrity_failure(self):
        for target in ("source", "baseline"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                state, marker = self.make_state(directory)

                def mutate(*args, **kwargs):
                    if target == "source":
                        marker.write_bytes(b"changed\n")
                    else:
                        state.baseline.code_path.write_text("changed baseline", encoding="utf-8")
                    return {"exit_code": 1, "error": "Fixture failed after mutation"}

                message = "Original source input changed" if target == "source" else "Frozen candidate code or recipe changed"
                with patch.object(discovery_host, "run_python", side_effect=mutate), self.assertRaisesRegex(ValueError, message):
                    discovery_host.discover_history_options(state)


if __name__ == "__main__":
    unittest.main()
