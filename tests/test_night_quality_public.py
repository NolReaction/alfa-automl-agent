"""Public TRAIN outer conversion, frozen provenance and cheap worker checks."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment_night_public as public
import experiment_night_quality as night
from tests.test_night_quality_runner import fake_checkout


def small_dataset(task, classification):
    count = 150
    if classification:
        groups = np.arange(count) // 2
        frame = pd.DataFrame({"number": groups.astype(float), "category": (groups % 4).astype(str)})
        target = np.asarray(["01", "NA", "class c"])[groups % 3]
        numeric, categorical = ["number"], ["category"]
    else:
        frame = pd.DataFrame({"Year_Factor": np.repeat(np.arange(1, 6), 30),
                              "number": np.linspace(-2, 3, count)})
        target = frame["number"].to_numpy() * 2
        numeric, categorical = list(frame.columns), []
    ids = [f"source-{index:05d}" for index in range(count)]
    return {"X": frame, "y": target, "ids": ids, "features": list(frame.columns),
            "numeric": numeric, "categorical": categorical,
            "profile": {"task_type": "classification" if classification else "regression",
                        "metric": "Macro F1" if classification else "RMSE"},
            "classes": ("01", "NA", "class c") if classification else (),
            "positive_class": None, "time_column": None if classification else "Year_Factor",
            "fingerprint": night._hash(pd.util.hash_pandas_object(frame).to_numpy().tobytes()
                                       + night._bytes(target.tolist())),
            "fingerprints": {str(task): night._hash(task.read_bytes())},
            "provenance": {"task": str(task), "prior_exposure": "unit fixture"}}


class PublicQualityRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = fake_checkout(self.root / "baseline")
        self.candidate = fake_checkout(self.root / "candidate")
        self.tasks = {}
        self.datasets = {}
        for label in ("regression", "classification"):
            path = self.root / f"{label}.md"
            path.write_text("Frozen public unit fixture\n", encoding="utf-8")
            self.tasks[label] = path
            self.datasets[label] = small_dataset(path, label == "classification")
        self.output = self.root / "result"

    def load(self, path):
        return deepcopy(self.datasets[Path(path).stem])

    def prepare(self, **options):
        arguments = {"baseline_root": self.baseline, "candidate_root": self.candidate,
                     "regression_task": self.tasks["regression"],
                     "classification_task": self.tasks["classification"],
                     "classification_seeds": (1729,), "regression_years": (5,),
                     "budget": 10, "allow_dirty": True}
        arguments.update(options)
        with patch.object(public, "comparison_dataset", side_effect=self.load):
            return public.prepare(self.output, **arguments)

    def execute(self):
        with patch.object(public, "comparison_dataset", side_effect=self.load), redirect_stdout(io.StringIO()):
            return public.execute_prepared(self.output)

    def test_frozen_public_conversion_preserves_raw_features_and_outer_truth_boundary(self):
        manifest = self.prepare()
        self.assertEqual(manifest["planned_trials"], 4)
        self.assertEqual([row["index"] for row in manifest["trial_plan"]], [0, 1, 2, 3])
        self.assertEqual([row["arm"] for row in manifest["trial_plan"]],
                         ["baseline", "candidate", "candidate", "baseline"])
        for frozen in manifest["cases"]:
            dataset = self.datasets[frozen["dataset"]]
            case = public.case_from_split(dataset, frozen["split"], frozen["dataset"])
            self.assertEqual(case.test.columns.tolist(), ["row_key", *dataset["features"]])
            self.assertNotIn("outcome", case.test.columns)
            self.assertEqual(case.audit_actuals, tuple(dataset["y"][frozen["split"]["outer_held"]].tolist()))
            self.assertEqual(case.train["row_key"].tolist(),
                             [dataset["ids"][i] for i in frozen["split"]["outer_fit"]])
            if frozen["dataset"] == "regression":
                self.assertLess(case.train["Year_Factor"].max(), case.test["Year_Factor"].min())
                self.assertEqual(case.validation, {"strategy": "time", "time_column": "Year_Factor"})
            else:
                fit_groups = set(pd.util.hash_pandas_object(case.train[dataset["features"]], index=False))
                held_groups = set(pd.util.hash_pandas_object(case.test[dataset["features"]], index=False))
                self.assertFalse(fit_groups & held_groups)
                self.assertEqual(case.validation, {"strategy": "grouped"})
        self.assertIn("public TRAIN", manifest["scope"])
        self.assertEqual(public.read_plan(self.output), json.loads((self.output / "manifest.json").read_text()))

    def test_new_directory_total_budget_and_unsupported_task_type_are_checked(self):
        with self.assertRaisesRegex(ValueError, "Total allowance"):
            self.prepare(max_total_seconds=39)
        self.assertFalse(self.output.exists())
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_impossible_requested_outer_period_is_reported_without_replacement(self):
        manifest = self.prepare(regression_years=(5, 99))
        impossible = [row for row in manifest["requested_splits"] if row["split"]["status"] != "ready"]
        self.assertEqual(len(impossible), 1)
        self.assertEqual(impossible[0]["split"]["year"], 99)
        self.assertEqual(len(manifest["cases"]), 2)
        self.assertEqual(manifest["planned_trials"], 4)

    def test_original_and_prepared_truth_tampering_rejects_before_any_solver_run(self):
        self.prepare()
        self.tasks["classification"].write_text("changed", encoding="utf-8")
        with patch.object(public.benchmark, "run_trial", side_effect=AssertionError("Changed source trained")):
            with self.assertRaisesRegex(ValueError, "Original public input changed"):
                self.execute()
        self.assertFalse((self.output / "execution_started.json").exists())

    def test_actual_fresh_fake_workers_are_paired_and_never_count_template_as_refit(self):
        self.prepare()
        report = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["trials"]), 4)
        self.assertEqual(report["summary"]["comparisons"][0]["counts"], {"tie": 2})
        for row in report["trials"]:
            self.assertTrue(row["audit"]["valid"])
            self.assertTrue(row["inputs_unchanged"])
            self.assertTrue(row["sources_unchanged"])
            self.assertFalse(row["refit_evidence"]["valid"])
            driver = (Path(row["task"]).parent / "execution/experiment_000.py").read_text(encoding="utf-8")
            self.assertNotIn("host_audit", driver)
            self.assertNotIn("audit_actuals", driver)
        with self.assertRaises(ValueError):
            self.execute()

    def test_changed_dataset_in_memory_cannot_override_prepared_metadata(self):
        self.prepare()
        self.datasets["classification"]["fingerprint"] = "changed"
        with self.assertRaisesRegex(ValueError, "Reloaded public dataset differs"):
            self.execute()


if __name__ == "__main__":
    unittest.main()
