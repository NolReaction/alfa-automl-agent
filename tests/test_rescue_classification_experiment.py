"""Safety properties of the fixed classification screen; no model fitting."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import pandas as pd

import experiment_rescue_classification as experiment
from agent.metrics import get_metric
from agent.task_spec import TaskSpec


class RescueExperimentTests(unittest.TestCase):
    def test_new_experiment_commit_keeps_exact_frozen_incumbent_runtime_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "agent").mkdir()
            (root / "agent/model.py").write_text("# unchanged model\n")
            (root / "offline_baseline.py").write_text("# unchanged baseline\n")
            hashes = {name: experiment.digest(root / name)
                      for name in ("agent/model.py", "offline_baseline.py")}
            native = {"source": {"sha": experiment.INCUMBENT_SHA, "files": hashes}}
            newer_experiment = {"source_sha": "f" * 40, "source_hashes": dict(hashes)}
            self.assertNotEqual(newer_experiment["source_sha"], native["source"]["sha"])
            with patch.object(experiment, "ROOT", root):
                experiment.check_incumbent_runtime(native, newer_experiment)
                (root / "agent/model.py").write_text("# modified model\n")
                with self.assertRaisesRegex(ValueError, "Control runtime differs"):
                    experiment.check_incumbent_runtime(native, newer_experiment)

    def test_control_replay_rejects_wrong_incumbent_sha_and_frozen_runtime_drift(self):
        native = {"source": {"sha": "0" * 40, "files": {}}}
        with self.assertRaisesRegex(ValueError, "preregistered incumbent SHA"):
            experiment.check_incumbent_runtime(native, {"source_sha": "f" * 40, "source_hashes": {}})
        native["source"] = {"sha": experiment.INCUMBENT_SHA,
                            "files": {"agent/model.py": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "Control runtime differs"):
            experiment.check_incumbent_runtime(native, {"source_sha": "f" * 40,
                                                        "source_hashes": {"agent/model.py": "b" * 64}})

    def test_seed_is_restored_after_error(self):
        import offline_baseline
        original = offline_baseline.SEED
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with experiment.outer_seed(123):
                self.assertEqual(offline_baseline.SEED, 123)
                raise RuntimeError("stop")
        self.assertEqual(offline_baseline.SEED, original)

    def test_duplicate_groups_exclude_id_and_target_and_count_conflicts(self):
        spec = SimpleNamespace(features=("x", "copy"), numeric=("x", "copy"), categorical=(),
                               target_column="label", classes=("A", "B"))
        frame = pd.DataFrame({"id": ["01", "02", "03", "04"], "x": ["1", "1", "2", "2"],
                              "copy": ["1", "1", "2", "2"], "label": ["A", "B", "A", "A"]})
        result = experiment.diagnostics(frame, spec)
        self.assertEqual(result["unique_feature_groups"], 2)
        self.assertEqual(result["conflicting_groups"], 1)
        self.assertEqual(result["rows_in_conflicting_groups"], 2)
        self.assertEqual(result["exact_duplicate_columns"], [["x", "copy"]])

    def test_finalists_require_failed_outcomes_and_use_fixed_tie_break(self):
        rows = [{"candidate": name, "phase": "inner_selection", "seed": 42,
                 "status": "completed", "macro_f1": .5} for name in reversed(experiment.CANDIDATES)]
        with self.assertRaisesRegex(ValueError, "All five"):
            experiment.finalists(rows[:-1])
        rows[0].update(status="failed", macro_f1=None)
        self.assertEqual(experiment.finalists(rows), ["catboost_none", "catboost_sqrt"])

    def test_saved_prediction_rescore_preserves_literal_labels_and_ids(self):
        spec = TaskSpec(Path("task.md"), Path("."), "id", "target", ("x",), ("x",), (),
                        ("01", "NA"), ("id", "target"), (), None, get_metric("Macro F1"))
        raw = b"id,target\n001,01\n002,NA\n"
        result = experiment.score_saved(raw, spec, ["001", "002"], ["01", "NA"])
        self.assertEqual(result["macro_f1"], 1.)
        self.assertEqual(result["confusion_matrix"], [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            experiment.score_saved(raw, spec, ["002", "001"], ["01", "NA"])

    def test_help_is_available_without_reading_data_or_starting_workers(self):
        with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
            experiment.main(["--help"])
        self.assertEqual(error.exception.code, 0)

    def test_screen_dispatches_exactly_five_fixed_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), command="screen")
            plan = {"splits": [{"seed": 42}]}
            with patch.object(experiment, "load_plan", return_value=(plan, None, None, None)), \
                    patch.object(experiment, "invoke") as invoke, patch.object(experiment, "export_table"):
                experiment.run(args)
            self.assertEqual([call.args[5] for call in invoke.call_args_list], list(experiment.CANDIDATES))
            self.assertTrue(all(call.kwargs == {"split": {"seed": 42}} for call in invoke.call_args_list))


if __name__ == "__main__":
    unittest.main()
