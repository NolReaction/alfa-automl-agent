"""Reproducible MOCK transport -> real v3 INNER/refit evidence, never live Qwen.

Run as a unittest, or retain local evidence with:
    python -m tests.test_rescue_api_evidence --output-dir data/experiments/rescue_mock

Only SDK transport and baseline preparation are fixtures. Tool processes,
INNER scoring, candidate freezing, outer admission, refit and CSV validation
use the actual host. Native artifacts remain local and are not public reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from openai import OpenAI

from agent.api_probe import timestamps
from agent.gateways import ALTERNATE_BASE_URL
from agent.llm import DEFAULT_MODEL, ROOT
import experiment_rescue_api as probe
from tests.test_team_integration import candidate_tool, fixture_baseline


FIXTURE_KEY = "rescue-evidence-local-fixture-key"


def _completion(request, *, code=None):
    message = {"role": "assistant", "content": "OK"}
    if code is not None:
        message["content"] = None
        message["tool_calls"] = [{"id": "rescue-evidence-call", "type": "function", "function": {
            "name": "run_python", "arguments": json.dumps({"code": code})}}]
    return httpx.Response(200, request=request, json={
        "id": "mock-response", "object": "chat.completion", "created": 1,
        "model": DEFAULT_MODEL, "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if code is not None else "stop"}],
    })


def run_evidence(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    calls = []

    def handler(request):
        # Record safe ordinals only, never payloads/headers in evidence output.
        calls.append(len(calls) + 1)
        index = len(calls)
        if index in {1, 5}:
            return _completion(request)
        if index == 2:
            return _completion(request, code=probe.TOOL_CODE)
        if index in {3, 4}:
            return _completion(request, code=candidate_tool())
        raise AssertionError("Unexpected extra mock request")

    report = {
        "version": 1, "kind": "rescue_api_mock_evidence", "mode": "mock",
        "synthetic_only": True, "live_qwen": False, "network_requests": 0,
        "transport": "httpx.MockTransport", "baseline_preparation": "fixed_constant_fixture",
        "purpose": "host integration evidence only; no live gateway reliability or quality claim",
        "base_commit": "d05572846369f3b1e9cfa33f4008dda028db4fe0",
        **probe.source_provenance(),
        "probe_sha256": hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
        "evidence_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": timestamps(), "limits": {
            "max_requests": 6, "max_estimated_tokens": 30000, "deadline_seconds": 90},
        "context_helper_active": False, "integration_passed": False,
        "full_candidate_path_passed": False,
        "accounting_note": "api_called/attempts count mock SDK invocations; actual network requests are zero",
    }
    sdk = OpenAI(api_key=FIXTURE_KEY, base_url=ALTERNATE_BASE_URL, max_retries=0,
                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    # No real client creation and no SDK network transport can be used here.
    with patch.dict(os.environ, {"OPENAI_API_KEY": FIXTURE_KEY, "OPENAI_MODEL": DEFAULT_MODEL,
                                 "OPENAI_BASE_URL": ALTERNATE_BASE_URL}), \
            patch.object(probe, "make_client", return_value=sdk), \
            patch("httpx.HTTPTransport.handle_request", side_effect=AssertionError("Network is forbidden in mock evidence")), \
            patch("agent.research_v3.prepare_baseline", side_effect=fixture_baseline):
        code = probe.execute_probe(directory, report, deadline=time.monotonic() + 90,
                                   max_requests=6, max_tokens=30000)
    report.update(mock_transport_calls=len(calls), exit_code=code)
    probe._write(directory / "probe_report.json", report)
    return code, report


class RescueApiEndToEndEvidenceTests(unittest.TestCase):
    def test_mock_stages_reach_actual_inner_frozen_recipe_and_same_full_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "evidence"
            code, report = run_evidence(directory)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "integration_passed")
            self.assertFalse(report["live_qwen"])
            self.assertEqual(report["network_requests"], 0)
            self.assertEqual(report["mock_transport_calls"], 5)
            self.assertEqual(report["budget"]["attempts"], 5)
            self.assertEqual(report["budget"]["responses"], 5)
            self.assertEqual(report["budget"]["responses_without_usage"], 5)
            self.assertFalse(report["budget"]["usage_complete"])
            self.assertLessEqual(report["budget"]["estimated_tokens_reserved"], 30000)
            self.assertIsNone(report["budget"]["blocked"])
            self.assertEqual([row["variant"] for row in report["attempts"]],
                             ["text", "tool_schema", "research_shape", "v3_smoke", "v3_smoke"])
            self.assertTrue(report["stages"][1]["deterministic_tool_executed"])
            self.assertFalse(report["stages"][2]["candidate_executed"])
            self.assertTrue(report["integration_passed"])
            self.assertTrue(report["full_candidate_path_passed"])
            coverage = report["smoke"]["coverage"]
            for field in ("api_response_received", "tool_call_valid", "candidate_executed",
                          "inner_trial_verified", "candidate_frozen", "full_refit_verified",
                          "qwen_full_refit_verified", "csv_valid", "candidate_published", "finalization_verified"):
                self.assertTrue(coverage[field], field)
            self.assertEqual(coverage["qwen_trials_with_identity"], 1)
            self.assertEqual(coverage["published_output_origin"], "llm")
            self.assertRegex(report["source_commit"], r"^[0-9a-f]{40}$")
            for field in ("source_files_sha256", "probe_sha256", "evidence_harness_sha256"):
                self.assertRegex(report[field], r"^[0-9a-f]{64}$")
            self.assertIs(type(report["source_worktree_dirty"]), bool)
            native_path = next((directory / "smoke" / "regression" / ".agent_runs").glob("*/run.json"))
            native = json.loads(native_path.read_text(encoding="utf-8"))
            self.assertEqual(native["api"]["attempts"], 2)
            self.assertEqual(native["successful_experiments"], 1)
            self.assertEqual(native["api_recovery"]["retries"], 0)
            receipts = native["candidate_receipts"]
            outer = json.loads(Path(receipts["outer"]).read_text(encoding="utf-8"))
            refit = json.loads(Path(receipts["refit"]).read_text(encoding="utf-8"))
            self.assertEqual(outer["phase"], "outer_validation")
            self.assertEqual(refit["phase"], "full_refit")
            for field in ("code_hash", "recipe_hash"):
                self.assertEqual(outer[field], refit[field])
                self.assertEqual(refit[field], native["frozen_candidate"][field])
            csv = directory / "smoke" / "regression" / "submission.csv"
            self.assertEqual(hashlib.sha256(csv.read_bytes()).hexdigest(), refit["output_hash"])
            public = (directory / "probe_report.json").read_text(encoding="utf-8")
            for forbidden in (FIXTURE_KEY, str(directory), candidate_tool(), "Authorization", "Bearer "):
                self.assertNotIn(forbidden, public)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()) and not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("Use gitignored data/ or an external directory for mock evidence")
    directory = output / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid4().hex[:8])
    code, report = run_evidence(directory)
    print(f"MOCK evidence: {report['status']}; network requests: 0; mock calls: {report['mock_transport_calls']}. "
          f"Report: {directory / 'probe_report.json'}")
    return code


if __name__ == "__main__":
    if "--output-dir" in sys.argv:
        raise SystemExit(main())
    unittest.main()
