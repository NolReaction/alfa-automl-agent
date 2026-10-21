"""Read-only run inspection, usage accounting and output validation checks."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import runs as runs_module
from app.runs import RunReader


RUN_ID = "a" * 32


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def report(**updates) -> dict:
    value = {
        "started_at": "2026-10-20T00:00:00+00:00",
        "task": "data/example/task.md",
        "model": "test-model",
        "max_steps": 12,
        "time_budget_seconds": 10500,
        "status": "fallback",
        "baseline": "existing_output",
        "experiments": 1,
        "successful_experiments": 0,
        "elapsed_seconds": 12.5,
        "reason": "api_error",
        "api": {
            "attempts": 0,
            "responses": 0,
            "usage_reported_responses": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "events": [],
    }
    value.update(updates)
    return value


class RunReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "task"
        self.root.mkdir()
        self.reader = RunReader(self.root)

    def write_run(self, data=None, run_id=RUN_ID):
        directory = self.root / ".agent_runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps(report() if data is None else data), encoding="utf-8"
        )
        return directory

    def make_inputs(self):
        folder = self.root / "nested" / "participants"
        write_csv(folder / "data_dictionary.csv", [
            ["feature", "role", "dtype"],
            ["row_key", "id", "category"],
            ["x", "feature", "numeric"],
            ["answer", "target", "numeric"],
        ])
        write_csv(folder / "sample_submission.csv", [
            ["row_key", "answer"], ["001", "0"], ["NA", "0"],
        ])
        write_csv(folder / "test.csv", [
            ["row_key", "x"], ["001", "1"], ["NA", "2"],
        ])
        write_csv(folder / "train.csv", [
            ["row_key", "x", "answer"], ["train-a", "1", "2"],
            ["train-b", "2", "4"],
        ])
        return folder

    def write_submission(self, reverse=False):
        rows = [["001", "2.0"], ["NA", "4.0"]]
        write_csv(self.root / "submission.csv", [
            ["row_key", "answer"], *(reversed(rows) if reverse else rows),
        ])

    def make_symlink(self, destination: Path, source: Path, *, directory=False):
        try:
            destination.symlink_to(source, target_is_directory=directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlinks unavailable: {error}")

    def test_empty_task_has_no_runs_or_submission(self):
        self.assertEqual(self.reader.list_runs(), [])
        result = self.reader.result()
        self.assertIsNone(result["baseline"])
        self.assertFalse(result["submission"]["exists"])
        self.assertIsNone(result["submission"]["valid"])

    def test_corrupt_and_missing_reports_stay_visible_as_invalid(self):
        broken = self.write_run(run_id="b" * 32)
        (broken / "run.json").write_text("{broken", encoding="utf-8")
        missing = self.root / ".agent_runs" / ("c" * 32)
        missing.mkdir()
        rows = {row["id"]: row for row in self.reader.list_runs()}
        self.assertEqual(set(rows), {"b" * 32, "c" * 32})
        for row in rows.values():
            self.assertEqual(row["status"], "invalid")
            self.assertTrue(row["error"])
        with self.assertRaises(ValueError):
            self.reader.detail("b" * 32)

    def test_malformed_report_schema_and_overflow_are_rejected_gracefully(self):
        invalid_events = report(events=["not an event object"])
        invalid_counter = report()
        invalid_counter["api"]["attempts"] = -1
        overflow = json.dumps(report())[:-1] + ', "extra": {"nested": [1e999]}}'
        cases = [
            ("top-level array", "[]"),
            ("string event", json.dumps(invalid_events)),
            ("negative API counter", json.dumps(invalid_counter)),
            ("overflow in unknown field", overflow),
        ]
        directory = self.write_run()
        for label, content in cases:
            with self.subTest(case=label):
                (directory / "run.json").write_text(content, encoding="utf-8")
                rows = self.reader.list_runs()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "invalid")
                self.assertTrue(rows[0]["error"])
                with self.assertRaises(ValueError):
                    self.reader.detail(RUN_ID)

    def test_usage_distinguishes_no_requests_unknown_partial_and_reported(self):
        cases = [
            (0, 0, 0, "none"),
            (1, 0, 0, "unknown"),
            (2, 2, 0, "unknown"),
            (2, 2, 1, "partial"),
            # A failed request may have consumed tokens despite absent usage.
            (2, 1, 1, "partial"),
            (2, 2, 2, "reported"),
        ]
        for attempts, responses, reported, expected in cases:
            with self.subTest(expected=expected, attempts=attempts,
                              responses=responses, reported=reported):
                data = report()
                data["api"].update(
                    attempts=attempts, responses=responses,
                    usage_reported_responses=reported,
                    prompt_tokens=11 * reported,
                    completion_tokens=7 * reported,
                    total_tokens=18 * reported,
                )
                self.write_run(data)
                usage = self.reader.detail(RUN_ID)["usage"]
                self.assertEqual(usage["state"], expected)
                self.assertEqual(usage["attempts"], attempts)
                self.assertEqual(usage["reported_responses"], reported)
                if reported:
                    self.assertEqual(usage["total_tokens"], 18 * reported)

    def test_report_details_keep_legacy_origin_and_v3_presentation_separate(self):
        self.write_run()
        self.assertEqual(self.reader.detail(RUN_ID)["research"], {"protocol_version": 2})
        self.write_run(report(research_protocol_version=3, stage="candidate_frozen", stages=[
            {"id": "candidate_frozen", "status": "completed"},
        ], frozen_candidate={"code_hash": "b" * 64, "path": "/private/candidate.py"}))
        summary = self.reader.detail(RUN_ID)["research"]
        self.assertEqual(summary["stage"], "candidate_frozen")
        self.assertEqual(summary["frozen_candidate"], {"code_hash": "b" * 64})

    def test_huge_json_integers_are_reported_without_overflow(self):
        huge = 10 ** 1000
        for location in ("counter", "unknown nested field"):
            with self.subTest(location=location):
                data = report()
                if location == "counter":
                    data["api"]["attempts"] = huge
                else:
                    data["extra"] = {"value": huge}
                self.write_run(data)
                self.assertEqual(self.reader.list_runs()[0]["status"], "invalid")
                with self.assertRaises(ValueError):
                    self.reader.detail(RUN_ID)
        baseline = {"model": {"name": "Example"}, "split": {}, "timing_seconds": {},
                    "metric": {"name": "R2", "value": -huge, "higher_is_better": True}}
        path = self.root / "baseline_metrics.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        result = self.reader.result()
        self.assertIsNone(result["baseline"])
        self.assertTrue(result["baseline_error"])
        baseline["metric"]["value"] = -0.25
        path.write_text(json.dumps(baseline), encoding="utf-8")
        result = self.reader.result()
        self.assertIsNone(result["baseline_error"])
        self.assertEqual(result["baseline"]["metric"]["value"], -0.25)

    def test_failed_experiment_and_recovery_events_are_preserved(self):
        events = [
            {"kind": "output_restored", "valid": True},
            {"kind": "experiment", "step": 1, "tool_index": 1,
             "result": {"exit_code": 1, "timed_out": False,
                        "stdout": "<script>untrusted</script>",
                        "stderr": "training failed",
                        "previous_output_restored": True,
                        "output_check": {"valid": True, "errors": []}}},
        ]
        directory = self.write_run(report(events=events))
        for name in ("experiment_101.py", "experiment_101.stdout.txt",
                     "experiment_101.stderr.txt"):
            (directory / name).write_text("example", encoding="utf-8")
        (directory / ".env").write_text("secret", encoding="utf-8")
        detail = self.reader.detail(RUN_ID)
        self.assertEqual(detail["report"]["events"], events)
        self.assertEqual(detail["report"]["status"], "fallback")
        self.assertEqual({item["kind"] for item in detail["files"]},
                         {"code", "stdout", "stderr"})
        self.assertEqual({item["experiment"] for item in detail["files"]}, {101})

    def test_path_traversal_and_non_experiment_files_are_rejected(self):
        self.write_run()
        for run_id in ("../", "../" + RUN_ID, RUN_ID + "/..", "unknown"):
            with self.subTest(run_id=run_id):
                with self.assertRaises((ValueError, FileNotFoundError, OSError)):
                    self.reader.detail(run_id)
        for name in ("../run.json", ".env", "run.json", "experiment_101.py/..",
                     "experiment_101.py/../../.env"):
            with self.subTest(name=name):
                with self.assertRaises((ValueError, FileNotFoundError, OSError)):
                    self.reader.file(RUN_ID, name)

    def test_symlinked_experiment_is_never_read_or_listed(self):
        directory = self.write_run()
        outside = Path(self.temporary.name) / "private.txt"
        outside.write_text("DO-NOT-EXPOSE", encoding="utf-8")
        self.make_symlink(directory / "experiment_101.py", outside)
        self.assertEqual(self.reader.detail(RUN_ID)["files"], [])
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.reader.file(RUN_ID, "experiment_101.py")

    def test_symlinked_run_root_is_rejected(self):
        outside = Path(self.temporary.name) / "outside-runs"
        outside.mkdir()
        self.make_symlink(self.root / ".agent_runs", outside, directory=True)
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            RunReader(self.root).list_runs()

    def test_large_code_returns_head_and_large_log_returns_tail(self):
        directory = self.write_run()
        payload = "HEAD-MARKER\n" + "x" * (140 * 1024) + "\nTAIL-MARKER"
        (directory / "experiment_101.py").write_text(payload, encoding="utf-8")
        (directory / "experiment_101.stdout.txt").write_text(payload, encoding="utf-8")
        code = self.reader.file(RUN_ID, "experiment_101.py")
        log = self.reader.file(RUN_ID, "experiment_101.stdout.txt")
        self.assertTrue(code["truncated"])
        self.assertTrue(log["truncated"])
        self.assertTrue(code["text"].startswith("HEAD-MARKER"))
        self.assertNotIn("TAIL-MARKER", code["text"])
        self.assertTrue(log["text"].endswith("TAIL-MARKER"))
        self.assertNotIn("HEAD-MARKER", log["text"])

    def test_submission_validation_preserves_string_ids_and_detects_order(self):
        self.make_inputs()
        self.write_submission()
        result = self.reader.result()
        self.assertTrue(result["submission"]["exists"])
        self.assertTrue(result["submission"]["valid"])
        self.assertEqual(self.reader.submission_path(), self.root / "submission.csv")
        self.write_submission(reverse=True)
        result = self.reader.result()
        self.assertFalse(result["submission"]["valid"])
        self.assertTrue(result["submission"]["errors"])

    def test_unchanged_validation_is_cached_and_changed_output_is_rechecked(self):
        self.make_inputs()
        self.write_submission()
        actual_validator = runs_module.validate_predictions
        with patch.object(runs_module, "validate_predictions", wraps=actual_validator) as check:
            self.assertTrue(self.reader.result()["submission"]["valid"])
            self.assertTrue(self.reader.result()["submission"]["valid"])
            self.assertEqual(check.call_count, 1)
            output = self.root / "submission.csv"
            previous = output.stat()
            self.write_submission(reverse=True)
            os.utime(output, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
            self.assertFalse(self.reader.result()["submission"]["valid"])
            self.assertEqual(check.call_count, 2)

    def test_corrupt_baseline_is_reported_without_hiding_submission(self):
        self.make_inputs()
        self.write_submission()
        metrics = self.root / "baseline_metrics.json"
        for content in ("{bad", "[]", '{"metric": {"value": NaN}}'):
            with self.subTest(content=content):
                metrics.write_text(content, encoding="utf-8")
                result = self.reader.result()
                self.assertIsNone(result["baseline"])
                self.assertTrue(result["baseline_error"])
                self.assertTrue(result["submission"]["valid"])

    def test_symlinked_submission_cannot_be_downloaded(self):
        outside = Path(self.temporary.name) / "private.csv"
        outside.write_text("private", encoding="utf-8")
        self.make_symlink(self.root / "submission.csv", outside)
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.reader.submission_path()


if __name__ == "__main__":
    unittest.main()
