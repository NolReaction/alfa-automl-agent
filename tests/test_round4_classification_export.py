"""Portable evidence integrity checks on tiny stdlib-only fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import experiment_round4_classification_export as export


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def npy(path, *, shape=(1, 2), fortran=True, dtype="<f8"):
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("ascii")
    header += b" " * ((-(10 + len(header) + 1)) % 64) + b"\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 16)


class ExportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.invocation = self.root / "worker" / "invocation"
        self.child_dir = self.invocation / "round4_work" / "ffinal"
        self.child_dir.mkdir(parents=True)
        # Intentionally not a pickle: only byte hashes may be inspected.
        (self.child_dir / "request.joblib").write_bytes(b"never deserialize this fixture")
        npy(self.child_dir / "probabilities.npy")
        self.metadata = {
            "classes": ["a", "b"], "fit_rows": 2, "predict_rows": 1,
            "fit_indices": [0, 1], "predict_indices": [0],
            "request_sha256": export.sha256(self.child_dir / "request.joblib"),
            "probabilities_sha256": export.sha256(self.child_dir / "probabilities.npy")}
        self.child = {"fit_index": "final", "metadata": self.metadata,
                      "execution": {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False},
                      "elapsed_seconds": 0.1}
        self.evidence = {"variant": "identity", "child_runs": [self.child]}
        self.record = {"candidate": "identity", "seed": 42, "status": "completed", "fit_rows": 2,
                       "predict_rows": 1, "candidate_evidence": [
                           {"path": str(self.invocation / "round4_evidence.json"), "content": self.evidence}]}
        self.flush()

    def flush(self):
        save(self.child_dir / "metadata.json", self.metadata)
        disk_child = deepcopy(self.child)
        disk_child["metadata"] = None
        save(self.child_dir / "execution.json", disk_child)
        save(self.invocation / "round4_evidence.json", self.evidence)

    def verify(self):
        return export.verify_children(self.root, self.record, [], ["a", "b"])

    def test_valid_metadata_and_local_indices_without_unpickling(self):
        report = self.verify()
        self.assertEqual(report[0]["status"], "verified")
        self.assertEqual(report[0]["fit_indices_sha256"], export.json_hash([0, 1]))
        self.assertTrue(report[0]["request_excluded"])
        self.assertNotIn("fit_indices", report[0])

    def test_noncanonical_root_preserves_portable_paths_and_containment(self):
        # Windows short-name paths can resolve to a different spelling too.
        root = self.root / "worker" / ".."
        report = export.verify_children(root, self.record, [], ["a", "b"])
        self.assertEqual(report[0]["status"], "verified")
        self.assertEqual(export.portable(self.child_dir / "metadata.json", root),
                         "evidence/worker/invocation/round4_work/ffinal/metadata.json")
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "metadata.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside evidence root"):
                export.portable(outside, root)

    def test_disk_metadata_tampering_rejected(self):
        altered = deepcopy(self.metadata)
        altered["fit_indices"] = [1, 0]
        save(self.child_dir / "metadata.json", altered)
        with self.assertRaisesRegex(ValueError, "Disk metadata"):
            self.verify()

    def test_request_and_probability_byte_tampering_rejected(self):
        for name in ("request.joblib", "probabilities.npy"):
            with self.subTest(name=name):
                path = self.child_dir / name
                original = path.read_bytes()
                path.write_bytes(original + b"modified")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    self.verify()
                path.write_bytes(original)

    def test_execution_tampering_rejected(self):
        path = self.child_dir / "execution.json"
        disk = json.loads(path.read_text())
        disk["execution"]["exit_code"] = 1
        save(path, disk)
        with self.assertRaisesRegex(ValueError, "Disk execution differs"):
            self.verify()

    def test_global_or_reordered_indices_rejected_even_when_receipts_agree(self):
        for indices in ([100, 101], [1, 0]):
            with self.subTest(indices=indices):
                self.metadata["fit_indices"] = indices
                self.flush()
                with self.assertRaisesRegex(ValueError, "exact worker-local"):
                    self.verify()

    def test_unbound_child_not_reported_successful(self):
        self.child["metadata"] = None
        self.record["status"] = "failed"
        self.flush()
        self.assertEqual(self.verify()[0]["status"], "unbound_or_failed")
        self.record["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "no verified"):
            self.verify()

    def test_npy_header_shape_object_dtype_and_truncation_rejected(self):
        path = self.child_dir / "probabilities.npy"
        for kwargs in ({"shape": (2, 1)}, {"dtype": "|O"}):
            npy(path, **kwargs)
            with self.assertRaises(ValueError):
                export.verify_npy(path, 1, 2)
        npy(path, fortran=False)
        export.verify_npy(path, 1, 2)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload length"):
            export.verify_npy(path, 1, 2)

    def test_oof_split_bound_to_preflight_and_group_disjoint(self):
        split = {"available": True, "groups": list(range(6)), "tuning_rows": [0, 1, 2],
                 "selection_rows": [3, 4, 5], "folds": [
                     {"fit_rows": [1, 2], "held_rows": [0]},
                     {"fit_rows": [0, 2], "held_rows": [1]},
                     {"fit_rows": [0, 1], "held_rows": [2]}]}
        split["split_sha256"] = export.json_hash(split)
        preflight = [{"seed": 42, "coverage": deepcopy(split)}]
        path = self.invocation / "round4_split.json"
        record = {"seed": 42, "fit_rows": 6}

        def check():
            save(path, split)
            evidence = {"coverage": {k: v for k, v in split.items()
                                     if k not in {"groups", "tuning_rows", "selection_rows", "folds"}}}
            return export.verify_partition(path, record, evidence, preflight)

        self.assertEqual(check(), split)
        split["tuning_rows"] = [2, 1, 0]
        split["split_sha256"] = export.json_hash({k: v for k, v in split.items() if k != "split_sha256"})
        with self.assertRaisesRegex(ValueError, "sealed preflight"):
            check()
        record["seed"] = None
        split["groups"][3] = split["groups"][0]
        split["split_sha256"] = export.json_hash({k: v for k, v in split.items() if k != "split_sha256"})
        with self.assertRaisesRegex(ValueError, "A/B raw groups overlap"):
            check()

    def test_inventory_excludes_raw_csv_requests_and_accepts_only_bound_labels(self):
        (self.invocation / "fit.csv").write_text("feature\nsecret\n")
        (self.invocation / "unknown.csv").write_text("feature\nsecret\n")
        predictions = self.invocation / "predictions.csv"
        predictions.write_text("id,label\n1,a\n")
        repo = self.root / "repo"
        (repo / "docs").mkdir(parents=True)
        (repo / "tests").mkdir()
        for name in ("experiment_round4_classification_export.py", "experiment_rescue_classification_rescore.py",
                     "docs/round4_classification_plan.md", "docs/round4_classification_report.md",
                     "docs/round4_classification_integration.md"):
            (repo / name).write_text("snapshot")
        self.record["output_path"] = str(predictions)
        plan = {"host_only_answer_hashes": {}, "source_hashes": {}}
        selected, excluded = export.inventory(self.invocation, plan, [self.record], repo)
        names = {path.name for path, _ in selected}
        self.assertIn("predictions.csv", names)
        self.assertIn("probabilities.npy", names)
        self.assertTrue({"fit.csv", "unknown.csv", "request.joblib"}.isdisjoint(names))
        self.assertEqual({Path(row["path"]).name for row in excluded}, {"fit.csv", "unknown.csv", "request.joblib"})

    def test_powershell_readme_preserves_backslashes(self):
        self.assertNotIn("\r", export.README)
        self.assertIn(r".\rescore.py", export.README)


if __name__ == "__main__":
    unittest.main()
