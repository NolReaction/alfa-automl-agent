"""Round 6: the audited fit is the actual host INNER fit, its nested boundaries hold, nothing trains."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import experiment_round4_feature_passport as passport
import experiment_round5_data_readiness as readiness
import experiment_round6_inner_readiness as inner
from agent.research_workspace import ResearchWorkspace
from tests.test_round4_feature_passport import tree
from tests.test_round5_data_readiness import ROW_KEYS, keys


COUNTS = {"alpha": 300, "beta": 200, "gamma": 120}
DICTIONARY = [("row_id", "id", "category"), ("label", "target", "category"), ("x1", "feature", "numeric"),
              ("x2", "feature", "numeric"), ("proto", "feature", "category")]


def train_frame(counts=COUNTS, *, few_groups=None, copies=40, conflicts=0, seed=0) -> pd.DataFrame:
    """Many independent raw groups per class, exact copies (so the host picks grouped), optional extras.

    ``few_groups`` maps a class to (groups, copies): that class then exists only in those raw groups.
    ``conflicts`` adds copies of existing feature rows carrying a different label.
    """
    rng = np.random.default_rng(seed)
    rows = [{"x1": f"{rng.normal():.5f}", "x2": str(int(rng.integers(0, 1_000_000))),
             "proto": str(rng.choice(["CATSENTINEL-a", "CATSENTINEL-b"])), "label": label}
            for label, count in counts.items() for _ in range(count)]
    for index, (label, (groups, repeat)) in enumerate((few_groups or {}).items()):
        rows += [{"x1": f"{1000 + 10 * index + group}", "x2": "7", "proto": "CATSENTINEL-a", "label": label}
                 for group in range(groups) for _ in range(repeat)]
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.iloc[:copies]], ignore_index=True)
    if conflicts:
        other = frame.iloc[copies:copies + conflicts].copy()
        labels = list(counts)
        other["label"] = [labels[(labels.index(value) + 1) % len(labels)] for value in other["label"]]
        frame = pd.concat([frame, other], ignore_index=True)
    frame.insert(0, "row_id", [f"IDSENTINEL-{index}" for index in range(len(frame))])
    return frame.loc[:, ["row_id", "x1", "x2", "proto", "label"]]


def write_task(root: Path, train: pd.DataFrame, name: str = "classification") -> Path:
    task = root / name
    inputs = task / "inputs"
    inputs.mkdir(parents=True)
    pd.DataFrame(DICTIONARY, columns=["feature", "role", "dtype"]).to_csv(inputs / "data_dictionary.csv", index=False)
    train.to_csv(inputs / "train.csv", index=False)
    rng = np.random.default_rng(9)
    test = pd.DataFrame({"row_id": [f"IDSENTINEL-t{index}" for index in range(40)],
                         "x1": [f"{value:.5f}" for value in rng.normal(size=40)],
                         "x2": [str(int(value)) for value in rng.integers(0, 1_000_000, 40)],
                         "proto": ["CATSENTINEL-new"] * 5 + ["CATSENTINEL-a"] * 35})
    test.to_csv(inputs / "test.csv", index=False)
    pd.DataFrame({"row_id": test["row_id"], "label": "alpha"}).to_csv(inputs / "sample_submission.csv", index=False)
    (task / "task.md").write_text("Primary metric: Macro F1\n", encoding="utf-8")
    return task / "task.md"


def digest(value) -> str:
    return passport.digest(value)


class InnerCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.root = self.base / "data"
        self.root.mkdir()

    def run_cli(self, task_md: Path, output: Path, *extra: str) -> tuple[int, str]:
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = inner.main(["--task-md", str(task_md), "--output-dir", str(output), *extra])
        return code, errors.getvalue()

    def export(self, task_md: Path, output: Path, *extra: str) -> dict:
        code, errors = self.run_cli(task_md, output, *extra)
        self.assertEqual(code, 0, errors)
        return json.loads((output / inner.FILES[0]).read_text(encoding="utf-8"))

    def gate(self, task_md: Path, name: str = "workspace", **options) -> ResearchWorkspace:
        return ResearchWorkspace(task_md, self.base / name, **options)

    def audit(self, gate: ResearchWorkspace, contract: dict | None = None) -> dict:
        watch = inner.Stopwatch()
        if contract is None:
            contract = inner.fit_contract(gate, self.base, watch)
        return inner.audit_contract(gate, contract, watch)


class HostInnerTests(InnerCase):
    def test_compatible_host_inner_is_the_pr51_table_and_reaches_the_first_fit(self) -> None:
        task = write_task(self.root, train_frame())
        report = self.export(task, self.base / "out")
        self.assertIn(report["decision"], inner.DECISIONS)
        self.assertEqual(report["decision"], "structurally_compatible")
        self.assertEqual(report["first_fit"]["structural"], "starts")
        self.assertEqual(report["first_fit"]["deciding_check"], "oof_support_gates")
        self.assertEqual(report["first_fit"]["runtime"], "unknown")
        fit, host = report["fit_table"], report["host"]
        for key in ("same_features", "same_rows_order_and_values", "same_labels", "same_ids"):
            self.assertIs(fit[key], True, key)
        self.assertEqual((host["outer"]["strategy"], host["inner"]["split"]["strategy"]), ("grouped", "grouped"))
        # The audited fit is the INNER fit: neither the outer fit nor the full TRAIN.
        self.assertEqual(fit["rows"], host["inner"]["fit_rows"])
        self.assertEqual(report["audit"]["rows"], fit["rows"])
        self.assertLess(fit["rows"], host["outer"]["fit_rows"])
        self.assertLess(host["outer"]["fit_rows"], sum(COUNTS.values()) + 40)
        self.assertFalse([item for item in report["boundaries"] if item["status"] == "violated"])
        self.assertEqual({item["id"] for item in report["boundaries"]} >= {
            "outer_partition", "research_is_outer_fit", "outer_groups_disjoint", "inner_partition",
            "inner_outside_outer_holdout", "inner_groups_disjoint", "audit_input_is_inner_fit", "ab_partition",
            "fold_1", "fold_2", "fold_3", "folds_hold_A_once"}, True)
        self.assertFalse(report["recipe"]["matches_frozen_public"])
        self.assertIn("отличается от публичной", report["recipe"]["explanation"])
        parts = {row["part"] for row in report["support"]}
        self.assertEqual(parts, {"allowed", "A", "B", *(f"fold_{i}_{side}" for i in (1, 2, 3) for side in ("fit", "held"))})
        margin = report["minimum_margin"]
        self.assertEqual(margin["margin"], min(row["margin"] for row in report["support"] if row["margin"] is not None))
        self.assertTrue(all(item["status"] != "violated" for item in report["host_prerequisites"]))

    def test_rare_class_is_ineligible_by_the_selection_gate(self) -> None:
        task = write_task(self.root, train_frame({**COUNTS, "delta": 24}))
        report = self.export(task, self.base / "out")
        self.assertEqual(report["decision"], "ineligible")
        self.assertEqual(report["first_fit"]["structural"], "blocked")
        self.assertEqual(report["first_fit"]["deciding_check"], "oof_support_gates")
        self.assertIn("selection_insufficient_class_support", report["first_fit"]["reason"])
        self.assertLess(report["minimum_margin"]["margin"], 0)
        self.assertIn({"part": "B", "class": "delta"},
                      [{"part": item["part"], "class": item["class"]} for item in report["minimum_margin"]["at"]])

    def test_class_missing_from_inner_via_host_replay_is_ineligible(self) -> None:
        task = write_task(self.root, train_frame({**COUNTS, "delta": 80}))
        canonical = self.gate(task, "canonical")
        labels = canonical.train["label"].to_numpy()
        held = sorted(set(canonical.outer_holdout_rows.tolist()) | set(np.flatnonzero(labels == "delta").tolist()))
        fit = sorted(set(range(len(labels))) - set(held))
        replay = self.gate(task, "replay", frozen_split={"fit_rows": fit, "validation_rows": held,
                                                          "choice": canonical.choice, "split": canonical.split})
        result = self.audit(replay)
        self.assertEqual(result["decision"], "ineligible")
        self.assertIn("oof_support_gates:tuning_missing_class", result["reasons"])
        delta = [row for row in result["support"] if row["class"] == "delta" and row["part"] == "allowed"]
        self.assertEqual(delta[0]["rows"], 0)

    def test_class_left_in_one_raw_group_never_reaches_a_fit(self) -> None:
        two = self.export(write_task(self.root, train_frame(few_groups={"delta": (2, 8)}), "two"), self.base / "two")
        self.assertEqual(two["decision"], "inner_unavailable")
        self.assertEqual(two["first_fit"]["structural"], "not_reached")
        self.assertIn("одной группе", two["reasons"][0])
        self.assertIsNone(two["fit_table"])
        three = self.export(write_task(self.root, train_frame(few_groups={"delta": (3, 8)}), "three"),
                            self.base / "three")
        self.assertEqual(three["decision"], "ineligible")
        self.assertEqual(three["first_fit"]["deciding_check"], "grouped_split")
        self.assertIn("одной группе", three["first_fit"]["reason"])

    def test_duplicates_with_conflicting_labels_stay_one_group_inside_every_boundary(self) -> None:
        task = write_task(self.root, train_frame(conflicts=30))
        report = self.export(task, self.base / "out")
        self.assertEqual(report["decision"], "structurally_compatible")
        self.assertGreater(report["audit"]["duplicates"]["conflicting_label_groups"], 0)
        self.assertFalse([item for item in report["boundaries"] if item["status"] == "violated"])
        self.assertTrue(all(item["shared_groups"] == 0 for item in report["boundaries"] if "shared_groups" in item))


class RefusalTests(InnerCase):
    def setUp(self) -> None:
        super().setUp()
        self.task = write_task(self.root, train_frame())

    def test_wrong_class_order_or_feature_roles_are_refused(self) -> None:
        gate = self.gate(self.task)
        spec = gate.research_spec
        # TaskSpec itself refuses overlapping roles, so a valid but different role split is the reachable case.
        cases = {"порядок классов": replace(spec, classes=tuple(reversed(spec.classes))),
                 "роли": replace(spec, numeric=("x1",), categorical=("x2", "proto"))}
        for expected, tampered in cases.items():
            with self.subTest(expected=expected):
                gate.research_spec = tampered
                result = self.audit(gate)
                self.assertEqual(result["decision"], "refused")
                self.assertTrue(any(expected in reason for reason in result["reasons"]), result["reasons"])
        gate.research_spec = spec
        watch = inner.Stopwatch()
        contract = inner.fit_contract(gate, self.base, watch)
        contract["y"].iloc[0] = "omega"
        result = inner.audit_contract(gate, contract, watch)
        self.assertEqual(result["decision"], "refused")
        self.assertTrue(any(reason.startswith("contract:") and "необъявленных" in reason
                            for reason in result["reasons"]), result["reasons"])
        self.assertIn("boundary:audit_input_is_inner_fit", result["reasons"])

    def test_artificial_boundary_leaks_are_refused(self) -> None:
        gate = self.gate(self.task, "inner-leak")
        gate.fit_rows = [*gate.fit_rows, gate.held_rows[0]]
        result = self.audit(gate)
        self.assertEqual(result["decision"], "refused")
        self.assertIn("boundary:inner_partition", result["reasons"])

        gate = self.gate(self.task, "outer-leak")
        gate.outer_holdout_rows = np.append(gate.outer_holdout_rows, gate.outer_fit_rows[0])
        result = self.audit(gate)
        self.assertEqual(result["decision"], "refused")
        self.assertIn("boundary:outer_partition", result["reasons"])

        gate = self.gate(self.task, "ab-leak")
        real = readiness.coverage

        def leaky(X, y, context):
            partition = real(X, y, context)
            partition["tuning_rows"] = [*partition["tuning_rows"], partition["selection_rows"][0]]
            return partition

        with mock.patch.object(readiness, "coverage", leaky):
            result = self.audit(gate)
        self.assertEqual(result["decision"], "refused")
        self.assertIn("boundary:ab_partition", result["reasons"])


class InvarianceTests(InnerCase):
    def setUp(self) -> None:
        super().setUp()
        self.task = write_task(self.root, train_frame())

    def test_held_answers_changed_after_freezing_do_not_change_the_fit_audit(self) -> None:
        gate = self.gate(self.task)
        watch = inner.Stopwatch()
        contract = inner.fit_contract(gate, self.base, watch)
        first = inner.audit_contract(gate, contract, watch)
        classes = list(gate.spec.classes)

        def rotate(values):
            return [classes[(classes.index(value) + 1) % len(classes)] for value in values]

        target = gate.spec.target_column
        held_train = gate.outer_fit_rows[np.asarray(gate.held_rows)]
        column = gate.train.columns.get_loc(target)
        for rows in (gate.outer_holdout_rows, held_train):
            gate.train.iloc[rows, column] = rotate(gate.train.iloc[rows, column])
        gate.research_train.iloc[gate.held_rows, gate.research_train.columns.get_loc(target)] = (
            gate.train.iloc[held_train, column].tolist())
        self.assertEqual(digest(inner.audit_contract(gate, contract, watch)), digest(first))
        again = inner.fit_contract(gate, self.base, watch)
        self.assertEqual(again["record"], contract["record"])
        self.assertEqual(digest(inner.audit_contract(gate, again, watch)), digest(first))

        # From a file: only outer-holdout answers change, the host replays the frozen outer split.
        source = pd.read_csv(self.task.parent / "inputs" / "train.csv", dtype=str, keep_default_na=False)
        source.iloc[gate.outer_holdout_rows, column] = rotate(source.iloc[gate.outer_holdout_rows, column])
        changed = write_task(self.base / "changed", source)
        replay = self.gate(changed, "replay", frozen_split=gate.frozen_split)
        self.assertEqual(replay.fit_rows, gate.fit_rows)
        self.assertEqual(digest(self.audit(replay)), digest(first))

    def test_test_changes_or_deletion_after_the_fit_do_not_change_audit_oof_fit(self) -> None:
        gate = self.gate(self.task)
        watch = inner.Stopwatch()
        contract = inner.fit_contract(gate, self.base, watch)
        context = inner.fit_context(gate)
        first = readiness.audit_oof_fit(contract["X"], contract["y"], context)
        result = inner.audit_contract(gate, contract, watch)
        test_csv = self.task.parent / "inputs" / "test.csv"
        test_csv.write_text(test_csv.read_text(encoding="utf-8").replace("CATSENTINEL-new", "CATSENTINEL-zzz"),
                            encoding="utf-8")
        self.assertEqual(readiness.audit_oof_fit(contract["X"], contract["y"], context), first)
        test_csv.unlink()
        self.assertEqual(readiness.audit_oof_fit(contract["X"], contract["y"], context), first)
        self.assertEqual(digest(inner.audit_contract(gate, contract, watch)), digest(result))
        with self.assertRaises((OSError, ValueError)):
            gate.check_source()  # the host itself notices the change; the audit does not depend on it

    def test_sources_stay_byte_identical_and_a_copy_elsewhere_keeps_its_hashes(self) -> None:
        before = tree(self.root)
        first = self.export(self.task, self.base / "out-1")
        self.assertEqual(tree(self.root), before)
        self.assertTrue(first["inputs_unchanged"])
        self.assertTrue(first["copy_matches_inputs"])
        self.assertTrue(first["copy_unchanged_during_run"])
        self.assertEqual(set(first["inputs"]), {"task.md", "inputs/data_dictionary.csv", "inputs/sample_submission.csv",
                                                "inputs/test.csv", "inputs/train.csv"})
        elsewhere = self.base / "another" / "place"
        shutil.copytree(self.root, elsewhere)
        second = self.export(elsewhere / "classification" / "task.md", self.base / "out-2")
        for key in ("inputs", "config_sha256", "content_sha256"):
            self.assertEqual(first[key], second[key], key)


class ExportTests(InnerCase):
    def setUp(self) -> None:
        super().setUp()
        self.task = write_task(self.root, train_frame())

    def test_outputs_hold_no_ids_categories_paths_or_row_keys(self) -> None:
        output = self.base / "out"
        report = self.export(self.task, output)
        self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(inner.FILES))
        for name in inner.FILES:
            text = (output / name).read_text(encoding="utf-8")
            for forbidden in ("IDSENTINEL", "CATSENTINEL", str(self.base), self.base.name, inner.SCRATCH_PREFIX,
                              tempfile.gettempdir()):
                self.assertNotIn(forbidden, text, name)
        self.assertFalse(keys(report) & ROW_KEYS)
        self.assertIn("не INNER-контекст для Qwen", report["statement"])
        self.assertEqual(report["audit"]["model_acceptance"], "not_evaluated")
        unknown = {item["id"]: item["status"] for item in report["not_evaluated"]}
        for name in ("selection_macro_f1_gain", "selection_class_f1_drop", "time_admission", "memory",
                     "identity_probe_seconds", "cost_admission", "platform_admission"):
            self.assertIn(unknown[name], {"unknown", "not_evaluated"}, name)
        self.assertNotIn("host_inner_equivalence", unknown)

    def test_no_model_fit_no_training_subprocess_and_no_network(self) -> None:
        real_popen = subprocess.Popen

        def only_git(args, *rest, **options):
            if not (isinstance(args, (list, tuple)) and Path(str(args[0])).stem.lower() == "git"):
                raise AssertionError("unexpected subprocess")
            return real_popen(args, *rest, **options)

        blocked = AssertionError("training or network")
        with mock.patch("subprocess.Popen", only_git), \
                mock.patch("agent.execution.run_python", side_effect=blocked), \
                mock.patch("agent.research_candidate.run_candidate", side_effect=blocked), \
                mock.patch("agent.round4_classification_probability.fit_predict_probabilities", side_effect=blocked), \
                mock.patch("agent.round4_classification_candidate.BoundedProbabilityFits.__call__", side_effect=blocked), \
                mock.patch("agent.models.build_pipeline", side_effect=blocked), \
                mock.patch.object(socket.socket, "connect", side_effect=blocked), \
                mock.patch("socket.create_connection", side_effect=blocked):
            report = self.export(self.task, self.base / "out")
        self.assertEqual(report["decision"], "structurally_compatible")

    def test_export_survives_unavailable_memory_counters(self) -> None:
        import psutil

        process = mock.Mock()
        process.memory_info.side_effect = psutil.AccessDenied(123)
        with mock.patch("psutil.Process", return_value=process), mock.patch.dict(sys.modules, {"resource": None}):
            report = self.export(self.task, self.base / "out")
        self.assertEqual(report["generated"]["memory"], {"peak_bytes": None, "method": None, "scope": "unknown"})
        self.assertIn("unknown", (self.base / "out" / inner.FILES[2]).read_text(encoding="utf-8"))

    def test_stages_are_exclusive_and_the_control_repeat_keeps_the_first_result(self) -> None:
        report = self.export(self.task, self.base / "out", "--control-repeat")
        timing = report["generated"]["timing"]
        self.assertLessEqual(sum(timing["seconds"].values()), timing["total_seconds"] + 0.01)
        self.assertTrue(all(value >= 0 for value in timing["seconds"].values()))
        repeat = report["generated"]["control_repeat"]
        self.assertTrue(repeat["content_equal"])
        self.assertEqual(set(repeat["timing"]["seconds"]), set(timing["seconds"]))
        self.assertNotIn("control_repeat", {key for key in report if key != "generated"})

    def test_output_inside_the_task_folder_is_refused(self) -> None:
        code, errors = self.run_cli(self.task, self.task.parent / "out")
        self.assertEqual(code, 1)
        self.assertIn("вне папок", errors)
        self.assertFalse((self.task.parent / "out").exists())


if __name__ == "__main__":
    unittest.main()
