"""Security and provenance checks for portable experiment summaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from app.sharing import ReportStore


FINGERPRINT = "a" * 64


def job(**overrides):
    value = {
        "id": "d37fd491d4f04fa7a235e31514a6a893", "action": "extra_trees",
        "status": "completed", "stage": "finished", "author": "Дима",
        "hypothesis": "Дополнительные деревья улучшат RMSE",
        "started_at": "2026-10-20T00:01:00+00:00", "finished_at": "2026-10-20T00:02:00+00:00",
        "elapsed_seconds": 60.0, "model": "ExtraTreesRegressor",
        "metric": {"name": "RMSE", "value": 43.25, "higher_is_better": False},
        "comparison_key": "b" * 64, "dataset_fingerprint": FINGERPRINT,
        "split": {"seed": 42, "validation_fraction": 0.2, "stratified": False,
                  "validation_rows": 10662, "validation_hash": "c" * 64},
        "error": "", "output_valid": True, "api_used": False, "exit_code": 0,
    }
    value.update(overrides)
    return value


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = ReportStore(self.root / ".workspace")

    def tearDown(self):
        self.directory.cleanup()

    def envelope(self, *jobs):
        return self.store.export("regression", FINGERPRINT, list(jobs) if jobs else [job()])

    def test_export_import_preserves_evidence_and_marks_claim_unverified(self):
        envelope = self.envelope()
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                         {"imported": 1, "skipped": 0})
        report, = self.store.list("regression")
        self.assertEqual(report["metric"], job()["metric"])
        self.assertEqual(report["split"], job()["split"])
        self.assertEqual(report["comparison_key"], "b" * 64)
        self.assertEqual(report["origin_id"], job()["id"])
        self.assertRegex(report["id"], r"^import-[0-9a-f]{64}$")
        self.assertEqual(report["source"], "imported")
        self.assertIs(report["verified"], False)
        self.assertIs(report["published"], False)

    def test_duplicate_import_and_reexport_are_idempotent(self):
        envelope = self.envelope()
        self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                         {"imported": 0, "skipped": 1})
        exported_again = self.store.export("regression", FINGERPRINT, self.store.list("regression"))
        self.assertEqual(exported_again["reports"], envelope["reports"])
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, exported_again),
                         {"imported": 0, "skipped": 1})
        self.assertEqual(len(self.store.list("regression")), 1)

    def test_validation_metadata_survives_import_and_reexport(self):
        common = {**job()["split"], "train_fraction": 0.8, "train_rows": 42648,
                  "shuffle": False}
        splits = [
            {**common, "strategy": "random", "shuffle": True},
            {**common, "strategy": "time", "time_column": "timestamp",
             "train_period_min": "2024-01-01", "train_period_max": "2024-06-30",
             "validation_period": "2024-07-01 — 2024-09-30"},
            {**common, "strategy": "grouped", "group_count": 10, "train_groups": 8,
             "validation_groups": 2, "group_overlap": 0, "n_splits": 5, "fold": 3},
        ]
        for split in splits:
            with self.subTest(strategy=split["strategy"]):
                envelope = self.envelope(job(split=split))
                self.assertEqual(envelope["reports"][0]["split"], split)
                self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                                 {"imported": 1, "skipped": 0})
                imported = next(row for row in self.store.list("regression")
                                if row["split"]["strategy"] == split["strategy"])
                self.assertEqual(imported["split"], split)
                self.assertIs(imported["verified"], False)
                self.assertIs(imported["published"], False)
                reexported = self.store.export("regression", FINGERPRINT, [imported])
                self.assertEqual(reexported["reports"], envelope["reports"])
                self.assertEqual(self.store.import_reports("regression", FINGERPRINT, reexported),
                                 {"imported": 0, "skipped": 1})

    def test_validation_metadata_is_part_of_imported_report_identity(self):
        common = job()["split"]
        variants = [
            {**common, "strategy": "time", "validation_period": "2024-01"},
            {**common, "strategy": "time", "validation_period": "2024-02"},
            {**common, "strategy": "grouped", "n_splits": 5, "fold": 0},
            {**common, "strategy": "grouped", "n_splits": 5, "fold": 1},
        ]
        for split in variants:
            self.assertEqual(self.store.import_reports("regression", FINGERPRINT,
                                                      self.envelope(job(split=split))),
                             {"imported": 1, "skipped": 0})
        reports = self.store.list("regression")
        self.assertEqual(len({row["id"] for row in reports}), len(variants))
        self.assertTrue(all(row["origin_id"] == job()["id"] for row in reports))

    def test_legacy_reports_do_not_gain_validation_metadata(self):
        legacy_split = job()["split"]
        envelope = self.envelope()
        self.assertEqual(envelope["version"], 1)
        self.assertEqual(envelope["reports"][0]["split"], legacy_split)
        self.store.import_reports("regression", FINGERPRINT, envelope)
        imported, = self.store.list("regression")
        self.assertEqual(imported["split"], legacy_split)
        reexported = self.store.export("regression", FINGERPRINT, [imported])
        self.assertEqual(reexported["reports"], envelope["reports"])

    def test_invalid_validation_metadata_is_rejected_before_import_writes(self):
        changes = [{"strategy": value} for value in (None, "", "unknown", [], True)]
        for field, limit in (("time_column", 500), ("train_period_min", 100),
                             ("train_period_max", 100), ("validation_period", 100)):
            changes.extend({field: value} for value in (None, "", " ", 1, "x\x00", "x" * (limit + 1)))
        for field in ("train_rows", "group_count", "train_groups", "validation_groups",
                      "group_overlap", "n_splits", "fold"):
            changes.extend({field: value} for value in (None, True, 0.5, -1, 1 << 31, "2"))
        changes.extend({field: 0} for field in ("train_rows", "group_count", "train_groups",
                                               "validation_groups", "n_splits"))
        changes.extend([
            {"n_splits": 1}, {"n_splits": 5, "fold": 5}, {"n_splits": 2, "fold": 3},
            {"train_fraction": True}, {"train_fraction": 0}, {"train_fraction": 1},
            {"shuffle": 0}, {"shuffle": "false"}, {"shuffle": None},
        ])
        for change in changes:
            with self.subTest(change=change):
                envelope = self.envelope()
                envelope["reports"][0]["split"].update(change)
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_qwen_fallback_survives_export_import_without_raw_reason_diagnostics(self):
        envelope = self.envelope(job(action="agent", metric=None, agent_status="fallback",
                                     agent_reason="api_error"))
        self.store.import_reports("regression", FINGERPRINT, envelope)
        imported, = self.store.list("regression")
        self.assertEqual(imported["agent_status"], "fallback")
        self.assertEqual(imported["agent_reason"], "api_error")
        self.assertIsNone(imported["metric"])
        again = self.store.export("regression", FINGERPRINT, [imported])
        self.assertEqual(again["reports"], envelope["reports"])
        private = self.envelope(job(action="agent", agent_status="fallback",
                                   agent_reason="secret-key from C:\\Users\\Dima\\.env"))
        self.assertNotIn("secret-key", json.dumps(private))
        self.assertEqual(private["reports"][0]["agent_reason"], "unknown")
        private["reports"][0]["agent_status"] = "invented"
        with self.assertRaises(ValueError):
            self.store.import_reports("regression", FINGERPRINT, private)

    def test_same_data_can_have_different_local_task_id(self):
        envelope = self.envelope()
        self.store.import_reports("a-different-folder", FINGERPRINT, envelope)
        self.assertEqual(len(self.store.list("a-different-folder")), 1)
        self.assertEqual(self.store.list("regression"), [])

    def test_mismatched_dataset_rejected_before_writes(self):
        envelope = self.envelope()
        with self.assertRaisesRegex(ValueError, "Данные задачи отличаются"):
            self.store.import_reports("regression", "f" * 64, envelope)
        envelope["reports"][0]["dataset_fingerprint"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "другим данным"):
            self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_unknown_sensitive_fields_dropped_and_payload_paths_never_read(self):
        original = job()
        original.update({"api_key": "private", "path": "../../outside.json",
                         "code": "raise RuntimeError('never execute')", "logs": ["private"],
                         "_runtime": {"pid": 123, "identity": "private"},
                         "raw_data": [["private"]], "model_file": "/etc/passwd"})
        original["metric"]["private"] = "private"
        original["split"]["path"] = "/etc/passwd"
        exported = self.envelope(original)
        self.assertNotIn("private", json.dumps(exported))
        exported["reports"][0]["path"] = "../../outside.json"
        exported["reports"][0]["published"] = True
        exported["reports"][0]["verified"] = True
        exported["path"] = "/etc/passwd"
        self.store.import_reports("../../some-task", FINGERPRINT, exported)
        result = self.store.list("../../some-task")
        serialized = json.dumps(result)
        self.assertNotIn("/etc/passwd", serialized)
        self.assertNotIn("outside.json", serialized)
        self.assertNotIn("raw_data", serialized)
        expected = self.root / ".workspace" / "imports" / hashlib.sha256(b"../../some-task").hexdigest()
        self.assertTrue(expected.is_dir())

    def test_machine_errors_replaced_and_user_hypothesis_preserved(self):
        hypothesis = "Сравнить sklearn/trees на holdout"
        exported = self.envelope(job(status="failed", hypothesis=hypothesis,
                                    error="unknown-private-key at C:\\Users\\Dima\\.env and /home/dima/.env"))
        text = json.dumps(exported, ensure_ascii=False)
        self.assertNotIn("unknown-private-key", text)
        self.assertNotIn("C:\\\\Users", text)
        self.assertNotIn("/home/dima", text)
        self.assertEqual(exported["reports"][0]["hypothesis"], hypothesis)
        self.assertEqual(exported["reports"][0]["error"], "Запуск завершился с ошибкой")
        # A third-party envelope with a raw error also normalizes before hashing.
        exported["reports"][0]["error"] = "another private machine path /root/.env"
        self.store.import_reports("regression", FINGERPRINT, exported)
        again = self.store.export("regression", FINGERPRINT, self.store.list("regression"))
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, again),
                         {"imported": 0, "skipped": 1})

    def test_invalid_numeric_and_boolean_values_rejected(self):
        mutations = [
            lambda row: row["metric"].update(value=True),
            lambda row: row["metric"].update(value=float("nan")),
            lambda row: row["metric"].update(value=float("inf")),
            lambda row: row["metric"].update(value=10**300),
            lambda row: row["metric"].update(higher_is_better=1),
            lambda row: row.update(elapsed_seconds=-1),
            lambda row: row.update(elapsed_seconds=False),
            lambda row: row.update(output_valid=1),
            lambda row: row.update(api_used="false"),
            lambda row: row["split"].update(seed=True),
            lambda row: row["split"].update(validation_fraction=1),
            lambda row: row["split"].update(validation_hash="not-a-hash"),
            lambda row: row.update(started_at=123),
            lambda row: row.update(finished_at="tomorrow"),
            lambda row: row.update(comparison_key="not-a-hash"),
            lambda row: row.update(action=[]),
            lambda row: row.update(status={}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                envelope = self.envelope()
                mutate(envelope["reports"][0])
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_malformed_envelope_and_oversized_fields_rejected(self):
        alterations = [
            {"format": "pickle"}, {"version": True}, {"version": 2},
            {"reports": "not-list"}, {"created_at": []},
            {"reports": [job()] * 201}, {"unused": "x" * (1024 * 1024)},
            {"unused": 10**100}, {"unused": float("nan")},
        ]
        for alteration in alterations:
            with self.subTest(alteration=list(alteration)):
                envelope = self.envelope()
                envelope.update(alteration)
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        for field, limit in [("author", 80), ("hypothesis", 500), ("model", 120), ("error", 1000)]:
            with self.subTest(field=field):
                envelope = self.envelope()
                envelope["reports"][0][field] = "x" * (limit + 1)
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)

    def test_import_is_validated_before_any_file_is_written(self):
        envelope = self.envelope()
        bad = job(id="another")
        bad["metric"]["value"] = "bad"
        envelope["reports"].append(bad)
        with self.assertRaises(ValueError):
            self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_export_omits_active_jobs(self):
        exported = self.envelope(job(status="running"), job(action="prepare"),
                                 job(dataset_fingerprint="f" * 64), job(status="completed"))
        self.assertEqual(len(exported["reports"]), 1)
        self.assertEqual(exported["reports"][0]["status"], "completed")

    def test_export_does_not_inspect_large_local_error_body(self):
        exported = self.envelope(job(status="failed", error="sensitive response " * 10_000))
        self.assertEqual(exported["reports"][0]["error"], "Запуск завершился с ошибкой")

    def test_empty_import_does_not_create_state(self):
        envelope = self.store.export("regression", FINGERPRINT, [])
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                         {"imported": 0, "skipped": 0})
        self.assertFalse((self.root / ".workspace").exists())

    def test_corrupt_or_tampered_records_do_not_break_list(self):
        self.store.import_reports("regression", FINGERPRINT, self.envelope())
        record_file = next((self.root / ".workspace" / "imports").rglob("*.json"))
        valid_record = record_file.read_text(encoding="utf-8")
        record_file.write_text("{broken", encoding="utf-8")
        self.assertEqual(self.store.list("regression"), [])
        # Re-import repairs the corrupted content atomically.
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, self.envelope()),
                         {"imported": 1, "skipped": 0})
        record = json.loads(valid_record)
        record["metric"]["value"] = 0.01
        record_file.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(self.store.list("regression"), [])

    def test_symlinked_state_or_record_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        linked = self.root / "linked"
        try:
            linked.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unavailable in this Windows configuration")
        with self.assertRaisesRegex(ValueError, "Ссылки"):
            ReportStore(linked / "state")
        state = self.root / ".workspace"
        state.mkdir()
        (state / "imports").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Ссылки"):
            self.store.import_reports("regression", FINGERPRINT, self.envelope())
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
