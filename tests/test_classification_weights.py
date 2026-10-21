"""Research weights stay isolated, comparable and read-only to source data."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import experiment_classification_weights as experiment
from agent.diagnostics import validate_classification_diagnostics


class ClassificationWeightTests(unittest.TestCase):
    def make_task(self, root):
        task = root / "task"
        data = task / "inputs"
        data.mkdir(parents=True)
        (task / "task.md").write_text("Predict class.\nPrimary metric: Macro F1\n", encoding="utf-8")
        (data / "data_dictionary.csv").write_text(
            "feature,role,dtype\nid,id,category\nx,feature,numeric\nprotocol,feature,category\n"
            "target,target,category\n", encoding="utf-8",
        )
        rows = [{"id": f"{index:04}", "x": index // 2, "protocol": "tcp",
                 "target": "NA" if index // 2 % 2 == 0 else "attack"}
                for index in range(80)]
        pd.DataFrame(rows).to_csv(data / "train.csv", index=False)
        (data / "test.csv").write_text("id,x,protocol\n0000,200,unseen\n", encoding="utf-8")
        (data / "sample_submission.csv").write_text("id,target\n0000,NA\n", encoding="utf-8")
        return task / "task.md", data

    def test_predeclared_ablations_change_class_weight_only(self):
        plan = experiment.candidate_plan()
        self.assertEqual([row["id"] for row in plan],
                         ["rf_control", "et_control", "rf_none", "rf_balanced", "et_none"])
        by_id = {row["id"]: row["model_config"] for row in plan}
        for control, candidate, weight in (("rf_control", "rf_none", None),
                                           ("rf_control", "rf_balanced", "balanced"),
                                           ("et_control", "et_none", None)):
            with self.subTest(candidate=candidate):
                old, new = by_id[control], by_id[candidate]
                self.assertEqual(old["model"], new["model"])
                self.assertEqual({key for key in old["parameters"]
                                  if old["parameters"][key] != new["parameters"][key]}, {"class_weight"})
                self.assertEqual(new["parameters"]["class_weight"], weight)

    def test_explicit_folds_have_disjoint_groups_and_different_validation_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = self.make_task(Path(directory))
            train, _, _, _, _, y, features, _, _, _ = experiment.load_data(data)
            default_fit, default_held, default_split = experiment.split_rows(
                train, y, features, True, {"strategy": "grouped"})
            first_fit, first_held, first_split = experiment.grouped_fold(train, y, features, 0)
            self.assertEqual(first_fit.tolist(), default_fit.tolist())
            self.assertEqual(first_held.tolist(), default_held.tolist())
            self.assertEqual(first_split, default_split)
            held_sets = []
            for fold_index in (0, 1, 2):
                fit, held, split = experiment.grouped_fold(train, y, features, fold_index)
                self.assertEqual(split["fold"], fold_index)
                self.assertEqual(split["group_overlap"], 0)
                self.assertEqual(split["train_rows"], len(fit))
                self.assertEqual(split["validation_rows"], len(held))
                self.assertAlmostEqual(split["validation_fraction"], len(held) / len(train))
                self.assertEqual(set(y.iloc[fit]), set(y))
                self.assertEqual(set(y.iloc[held]), set(y))
                fitting_keys = set(map(tuple, train.iloc[fit][features].to_numpy()))
                held_keys = set(map(tuple, train.iloc[held][features].to_numpy()))
                self.assertFalse(fitting_keys & held_keys)
                held_sets.append(set(train.iloc[held]["id"]))
            self.assertFalse(held_sets[0] & held_sets[1])
            self.assertFalse(held_sets[0] & held_sets[2])
            self.assertFalse(held_sets[1] & held_sets[2])

    def test_requested_fold_missing_a_class_is_rejected_without_substituting_another_fold(self):
        rows = []
        for label in ("a", "b"):
            for group in range(10):
                rows.extend([{"x": f"{label}-{group}", "target": label}] * 2)
        rows.extend([{"x": "c-1", "target": "c"}] * 3)
        rows.extend([{"x": "c-2", "target": "c"}] * 2)
        train = pd.DataFrame(rows)
        y = train["target"]
        groups = pd.util.hash_pandas_object(train[["x"]], index=False).to_numpy()
        _, _, metadata = experiment.split_rows(train, y, ["x"], True, {"strategy": "grouped"})
        splitter = StratifiedGroupKFold(n_splits=metadata["n_splits"], shuffle=True, random_state=42)
        bad = [fold for fold, (fit, held) in enumerate(splitter.split(train, y, groups))
               if set(y.iloc[held]) != set(y) or set(y.iloc[fit]) != set(y)]
        self.assertTrue(bad)
        for fold_index in bad:
            with self.subTest(fold_index=fold_index), self.assertRaisesRegex(ValueError, "all classes"):
                experiment.grouped_fold(train, y, ["x"], fold_index)

    def test_recipe_and_fold_requests_are_bounded_and_do_not_create_outputs_on_error(self):
        self.assertEqual([row["id"] for row in experiment.select_recipes(["rf_balanced", "rf_control"])],
                         ["rf_control", "rf_balanced"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self.make_task(root)
            for recipes in ([], ["unknown"], ["rf_control", "rf_control"], "rf_control"):
                with self.subTest(recipes=recipes), self.assertRaisesRegex(ValueError, "recipes"):
                    experiment.run_experiment(task, output_dir=root / "output", recipes=recipes)
            for fold in (-1, 5, True, 0.5):
                with self.subTest(fold=fold), self.assertRaisesRegex(ValueError, "fold-index"):
                    experiment.run_experiment(task, output_dir=root / "output", fold_index=fold)
            self.assertFalse((root / "output").exists())

    def test_real_grouped_experiment_preserves_sources_and_saves_recomputable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self.make_task(root)
            originals = {name: (data / name).read_bytes() for name in experiment.INPUT_NAMES}
            output = root / "experiment"
            with contextlib.redirect_stdout(io.StringIO()):
                report = experiment.run_experiment(task, output_dir=output)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["api"], {"used": False, "calls": 0})
            self.assertTrue(report["source_inputs_unchanged"])
            self.assertEqual(report["planned"], 5)
            self.assertEqual(report["split"]["strategy"], "grouped")
            self.assertEqual(report["split"]["group_overlap"], 0)
            self.assertEqual(len(report["candidates"]), 5)
            signatures, identities = set(), []
            for row in report["candidates"]:
                held = pd.read_csv(output / row["holdout_predictions_path"], dtype=str, keep_default_na=False)
                self.assertEqual(set(held["actual"]), {"NA", "attack"})
                identities.append(list(zip(held["id"], held["actual"])))
                signatures.add(row["comparison_key"])
                self.assertEqual(list(held.columns), ["id", "actual", "prediction"])
                self.assertAlmostEqual(row["metric"]["value"],
                                       f1_score(held["actual"], held["prediction"], average="macro"), places=12)
                validate_classification_diagnostics(row["classification_diagnostics"],
                                                    metric=row["metric"], validation_rows=len(held))
            self.assertTrue(all(identity == identities[0] for identity in identities))
            self.assertEqual(signatures, {report["comparison_key"]})
            self.assertEqual(originals, {name: (data / name).read_bytes() for name in experiment.INPUT_NAMES})
            self.assertFalse((task.parent / "submission.csv").exists())
            self.assertFalse((output / "submission.csv").exists())
            self.assertEqual(report, json.loads((output / "report.json").read_text(encoding="utf-8")))

    def test_recipe_subset_and_nonzero_fold_preserve_source_and_recompute_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self.make_task(root)
            original = {name: (data / name).read_bytes() for name in experiment.INPUT_NAMES}
            output = root / "experiment"
            with contextlib.redirect_stdout(io.StringIO()):
                report = experiment.run_experiment(task, output_dir=output, fold_index=1,
                                                   recipes=["rf_control", "rf_balanced"])
            self.assertEqual(report["split"]["fold"], 1)
            self.assertEqual(report["planned"], 2)
            self.assertEqual(report["selected_recipes"], ["rf_control", "rf_balanced"])
            self.assertEqual(len(report["candidates"]), 2)
            self.assertEqual(report["split"]["group_overlap"], 0)
            self.assertEqual(original, {name: (data / name).read_bytes() for name in experiment.INPUT_NAMES})
            held = pd.read_csv(output / report["candidates"][0]["holdout_predictions_path"],
                               dtype=str, keep_default_na=False)
            self.assertEqual(len(held), report["split"]["validation_rows"])
            actual_hash = experiment._json_hash({"id_column": "id", "target_column": "target",
                                                "ids": held["id"].tolist(), "actuals": held["actual"].tolist()})
            self.assertEqual(actual_hash, report["split"]["validation_hash"])

    def test_source_directory_and_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self.make_task(root)
            original = (data / "train.csv").read_bytes()
            existing = root / "already-there"
            existing.mkdir()
            (existing / "report.json").write_text("previous", encoding="utf-8")
            for output in (task.parent / "experiment", data / "train.csv", existing):
                with self.subTest(output=output):
                    with self.assertRaises((ValueError, FileExistsError)):
                        experiment.run_experiment(task, output_dir=output)
            self.assertEqual((data / "train.csv").read_bytes(), original)
            self.assertEqual((existing / "report.json").read_text(), "previous")
            self.assertFalse((task.parent / "experiment").exists())

    def test_changed_source_invalidates_the_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self.make_task(root)
            output = root / "experiment"
            real_hashes = experiment._input_hashes
            calls = 0

            def changed_after_training(folder, task_path):
                nonlocal calls
                hashes = real_hashes(folder, task_path)
                calls += 1
                if calls == 4:
                    hashes["train.csv"] = "f" * 64
                return hashes

            # One real recipe suffices to exercise the final integrity guard.
            with patch.object(experiment, "candidate_plan", return_value=experiment.candidate_plan()[:1]), \
                    patch.object(experiment, "_input_hashes", side_effect=changed_after_training), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Source inputs changed"):
                    experiment.run_experiment(task, output_dir=output)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("best_candidate", report)


if __name__ == "__main__":
    unittest.main()
