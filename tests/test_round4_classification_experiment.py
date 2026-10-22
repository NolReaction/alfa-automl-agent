"""Experiment provenance, leakage and finite-budget failures; no model fits."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import experiment_round4_classification as experiment
from agent.metrics import get_metric
from agent.task_spec import TaskSpec


def spec():
    return TaskSpec(Path("task.md"), Path("."), "id", "target", ("x",), ("x",), (),
                    ("A", "B"), ("id", "target"), (), None, get_metric("Macro F1"))


class Round4ExperimentTests(unittest.TestCase):
    def test_help_never_initializes_workers_or_loads_data(self):
        with patch.object(experiment, "prepare") as prepare, patch.object(experiment, "run") as run, \
                redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
            experiment.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        prepare.assert_not_called()
        run.assert_not_called()

    def test_literal_reference_matches_actual_frozen_recipe_and_current_factory(self):
        from agent.round4_classification_probability import fixed_incumbent_recipe
        self.assertEqual(experiment.BASE_RECIPE, fixed_incumbent_recipe())
        self.assertEqual(experiment.BASE_SHA, "e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb")
        self.assertEqual(sum(experiment.PLANNED_FITS.values()), 21)
        self.assertEqual(experiment.SEEDS, (42, 43, 44, 137))

    def test_conflicting_labels_and_ids_do_not_split_identical_feature_groups(self):
        frame = pd.DataFrame({"id": [str(i) for i in range(160)],
                              "x": [str(i // 2) for i in range(160)], "target": ["A", "B"] * 80})
        partitions = experiment.grouped_splits(frame, spec())
        for partition in partitions:
            fit = frame.iloc[partition["fit"]]
            held = frame.iloc[partition["held"]]
            self.assertFalse(set(fit.x) & set(held.x))
            self.assertEqual(set(partition["fit"]) | set(partition["held"]), set(range(len(frame))))

    def test_cached_coverage_rejects_same_size_but_different_feature_groups_or_labels(self):
        frame = pd.DataFrame({"x": ["1", "2"], "target": ["A", "B"]})
        value = {"available": False, "groups": [int(v) for v in pd.util.hash_pandas_object(frame[["x"]], index=False)],
                 "support": {"allowed": {"A": {"rows": 1, "groups": 1}, "B": {"rows": 1, "groups": 1}}}}
        value["split_sha256"] = experiment.json_hash(value)
        experiment.validate_coverage(value, frame, spec())
        changed = frame.copy()
        changed.loc[0, "x"] = "changed"
        with self.assertRaisesRegex(ValueError, "raw feature groups"):
            experiment.validate_coverage(value, changed, spec())
        changed = frame.copy()
        changed.loc[0, "target"] = "B"
        with self.assertRaisesRegex(ValueError, "fitting labels"):
            experiment.validate_coverage(value, changed, spec())

    def test_score_saved_exposes_tp_fp_fn_and_exact_macro_contributions(self):
        result = experiment.metrics_saved(b"id,target\n001,A\n002,A\n003,B\n", spec(),
                                           ["001", "002", "003"], ["A", "B", "B"])
        self.assertEqual(result["class_counts"]["A"],
                         {"TP": 1, "FP": 1, "FN": 0, "support": 1, "f1": 2/3, "macro_contribution": 1/3})
        self.assertEqual(result["class_counts"]["B"]["FN"], 1)
        self.assertAlmostEqual(sum(x["macro_contribution"] for x in result["class_counts"].values()), result["macro_f1"])
        with self.assertRaises(ValueError):
            experiment.metrics_saved(b"id,target\n001,UNKNOWN\n", spec(), ["001"], ["A"])

    def test_cost_admission_includes_four_calibration_fits_and_final_reserve(self):
        self.assertTrue(experiment.admit_cost(60))
        self.assertTrue(experiment.admit_cost(75))
        self.assertFalse(experiment.admit_cost(75.001))
        self.assertFalse(experiment.admit_cost(float("nan")))

    def test_failed_fit_events_count_and_cancelled_before_fit_is_not_physical_training(self):
        events = [{"status": "started"}, {"status": "completed"}, {"status": "started"},
                  {"status": "cancelled_before_fit"}]
        evidence = [{"path": "f0/events.jsonl", "content": events},
                    {"path": "round4_evidence.json", "content": {
                        "physical_fits_started": 1, "physical_fits_completed": 1, "status": "failed"}}]
        counts = experiment.fit_accounting("identity", "failed", evidence)
        self.assertEqual((counts["actual_started"], counts["actual_completed"], counts["budget_charge"]), (1, 1, 2))
        evidence[1]["content"]["physical_fits_started"] = 2
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            experiment.fit_accounting("identity", "failed", evidence)

    def test_killed_parent_has_unknown_actual_total_and_counts_observed_starts(self):
        evidence = [{"path": "f0/events.jsonl", "content": [
            {"status": "started"}, {"status": "completed"}, {"status": "started"}]}]
        result = experiment.fit_accounting("oof", "failed", evidence)
        self.assertIsNone(result["actual_started"])
        self.assertEqual(result["observed_started"], 2)
        self.assertEqual(result["budget_charge"], 4)
        native = experiment.fit_accounting("native_fixed", "completed", [])
        self.assertIn("inferred_fixed_recipe", native["basis"])
        self.assertEqual(native["actual_started"], 2)

    def test_partial_event_file_preserves_prior_started_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "worker/invocation/round4_work/f0/events.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text('{"status":"started"}\n{"status":')
            evidence = experiment.collect_evidence(root)
            self.assertIn("parse_error", evidence[0])
            result = experiment.fit_accounting("oof", "failed", evidence)
            self.assertEqual(result["observed_started"], 1)
            self.assertEqual(result["budget_charge"], 3)

    def screen_rows(self):
        return [{"candidate": name, "stage": "screen", "seed": 42, "status": "completed",
                 "macro_f1": .5, "decision": {"rules_applied": 0}} for name in experiment.SCREEN]

    def test_finalist_excludes_identity_oof_and_confirms_best_even_if_gain_negative(self):
        rows = self.screen_rows()
        rows[0]["macro_f1"] = .9
        rows[2]["macro_f1"] = .99  # OOF identity is not a challenger.
        with patch.object(experiment, "identity_agreement", return_value=True):
            self.assertEqual(experiment.choose_finalist(rows), "frequency")
            rows[2]["decision"]["rules_applied"] = 1
            self.assertEqual(experiment.choose_finalist(rows), "oof")
            with self.assertRaisesRegex(ValueError, "All seven"):
                experiment.choose_finalist(rows[:-1])
        with patch.object(experiment, "identity_agreement", return_value=False):
            self.assertIsNone(experiment.choose_finalist(rows))

    def test_identity_contract_checks_labels_not_only_equal_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for name, raw in (("native_fixed", "id,target\n1,A\n2,B\n"), ("identity", "id,target\n1,B\n2,A\n")):
                path = Path(temporary) / (name + ".csv")
                path.write_text(raw)
                rows.append({"candidate": name, "stage": "screen", "status": "completed",
                             "output_path": str(path), "output_sha256": experiment.digest(path), "macro_f1": .5})
            self.assertFalse(experiment.identity_agreement(rows))

    def test_screen_contract_failure_records_every_hypothesis_without_fitting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            experiment.write_new(output / "plan.json", {})
            plan = {"splits": [{"seed": 42}], "physical_fit_caps": experiment.CAPS}
            rows = []
            def invoke(*args, **kwargs):
                row = {"candidate": args[5], "stage": kwargs["stage"], "status": "failed",
                       "physical_fits": {"budget_charge": 2}}
                rows.append(row)
                return row
            with patch.object(experiment, "load_plan", return_value=(plan, None, None, None)), \
                    patch.object(experiment, "records", side_effect=lambda _: list(rows)), \
                    patch.object(experiment, "invoke", side_effect=invoke) as mocked, \
                    patch.object(experiment, "export_table"):
                experiment.run(SimpleNamespace(output=output, command="screen"))
            self.assertEqual([call.args[5] for call in mocked.call_args_list], ["native_fixed", "identity"])
            for name in experiment.HYPOTHESES:
                row = experiment.read_json(output / "trials" / (name + "_seed42") / "record.json")
                self.assertEqual(row["reason"], "identity_contract_failed")

    def test_confirmation_gain_only_from_rarest_class_cannot_recommend_default(self):
        plan = {"classes": ["A", "B"], "rarest_class": "B", "acceptance": {
            "mean_gain": .003, "minimum_slice_gain": -.01, "minimum_class_gain": -.05}}
        rows = []
        for seed in experiment.SEEDS[1:]:
            for name, f1, b in (("native_fixed", .5, .5), ("frequency", .51, .52)):
                rows.append({"candidate": name, "seed": seed, "stage": "confirm", "status": "completed", "macro_f1": f1,
                             "class_counts": {"A": {"f1": .5}, "B": {"f1": b}}})
        result = experiment.paired_table(rows, plan)["confirmation"]
        self.assertGreater(result["mean_macro_delta"], .003)
        self.assertEqual(result["recommendation"], "retain_incumbent")

    def test_input_mutation_rejected_before_loading_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "train.csv"
            path.write_text("original")
            plan = {"base_sha": experiment.BASE_SHA, "input_sha256": {str(path): experiment.digest(path)}}
            experiment.write_new(output / "plan.json", plan)
            experiment.write_new(output / "seal.json", {"plan_sha256": experiment.digest(output / "plan.json")})
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Official input changed"):
                experiment.load_plan(output, execution=False)


class Round4AuditTests(unittest.TestCase):
    def fixture(self, output):
        task = spec()
        train = pd.DataFrame({"id": ["001", "002"], "x": ["1", "2"], "target": ["A", "B"]})
        test = pd.DataFrame({"id": ["003"], "x": ["3"]})
        invocation = output / "trials/native_fixed_seed42/worker/inner_selection_nonce"
        invocation.mkdir(parents=True)
        (invocation / "candidate.py").write_text("# frozen\n")
        (invocation / "recipe.json").write_text('{}')
        (invocation / "predictions.csv").write_text("id,target\n002,B\n")
        train.iloc[[0]].to_csv(invocation / "fit.csv", index=False)
        train.iloc[[1]].drop(columns="target").to_csv(invocation / "predict.csv", index=False)
        context = {"nonce": "nonce", "phase": "inner_selection", "code_hash": experiment.digest(invocation / "candidate.py"),
                   "recipe_hash": experiment.json_hash({})}
        experiment.write_new(invocation / "context.json", context)
        receipt = {"version": 3, **context, "output_hash": experiment.digest(invocation / "predictions.csv")}
        experiment.write_new(invocation / "receipt.json", receipt)
        plan = {"classes": ["A", "B"], "scope": "test", "rarest_class": "A", "physical_fit_caps": experiment.CAPS,
                "splits": [{"seed": 42, "fit": [0], "held": [1]}],
                "recipes": {"native_fixed": {"source_sha256": context["code_hash"], "recipe_sha256": context["recipe_hash"]}}}
        experiment.write_new(output / "plan.json", plan)
        row = {"candidate": "native_fixed", "seed": 42, "stage": "screen", "status": "completed", **context,
               "plan_sha256": experiment.digest(output / "plan.json"), "candidate_evidence": [],
               "physical_fits": experiment.fit_accounting("native_fixed", "completed", []),
               "output_path": str(invocation / "predictions.csv"), "output_sha256": receipt["output_hash"],
               "receipt_path": str(invocation / "receipt.json"), "receipt_sha256": experiment.digest(invocation / "receipt.json"),
               "invocation_inputs": {str(invocation / name): experiment.digest(invocation / name) for name in
                                     ("context.json", "candidate.py", "recipe.json", "fit.csv", "predict.csv")},
               **experiment.metrics_saved((invocation / "predictions.csv").read_bytes(), task, ["002"], ["B"])}
        rows = [row] + [{"candidate": name, "seed": 42, "stage": "screen", "status": "skipped", "reason": "test",
                        "plan_sha256": row["plan_sha256"], "physical_fits": {"budget_charge": 0}} for name in experiment.SCREEN[1:]]
        return plan, task, train, test, row, rows, invocation

    def check(self, output, fixture):
        with patch.object(experiment, "load_plan", return_value=fixture[:4]), \
                patch.object(experiment, "records", return_value=fixture[5]), redirect_stdout(StringIO()):
            experiment.audit(SimpleNamespace(output=output))

    def test_complete_receipt_passes_and_semantic_identity_tampering_fails(self):
        for field in ("nonce", "phase", "code_hash", "recipe_hash"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                fixture = self.fixture(output)
                self.check(output, fixture)
                row, invocation = fixture[4], fixture[6]
                receipt = experiment.read_json(invocation / "receipt.json")
                receipt[field] = "wrong"
                (invocation / "receipt.json").write_text(__import__("json").dumps(receipt))
                row["receipt_sha256"] = experiment.digest(invocation / "receipt.json")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.check(output, fixture)

    def test_resealed_csv_with_wrong_ids_or_unknown_labels_still_rejected(self):
        for raw in ("id,target\n999,B\n", "id,target\n002,UNKNOWN\n"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                fixture = self.fixture(output)
                row, invocation = fixture[4], fixture[6]
                (invocation / "predictions.csv").write_text(raw)
                row["output_sha256"] = experiment.digest(invocation / "predictions.csv")
                receipt = experiment.read_json(invocation / "receipt.json")
                receipt["output_hash"] = row["output_sha256"]
                (invocation / "receipt.json").write_text(__import__("json").dumps(receipt))
                row["receipt_sha256"] = experiment.digest(invocation / "receipt.json")
                with self.assertRaises(ValueError):
                    self.check(output, fixture)

    def test_resealed_fit_rows_cannot_include_held_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixture = self.fixture(output)
            row, invocation = fixture[4], fixture[6]
            fixture[2].to_csv(invocation / "fit.csv", index=False)
            row["invocation_inputs"][str(invocation / "fit.csv")] = experiment.digest(invocation / "fit.csv")
            with self.assertRaisesRegex(ValueError, "rows differ"):
                self.check(output, fixture)


if __name__ == "__main__":
    unittest.main()
