"""Plan invariants without real downloads or model fits."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_lab import planning
from research_lab.integrity import file_hash, read_json, write_json, sanitized_environment


class PlanTests(unittest.TestCase):
    def test_estimate_bound_and_confirmation_denominator(self):
        pilot = planning.estimate()
        self.assertEqual((pilot["datasets"], pilot["arms"], pilot["training_upper_bound_seconds"]), (4, 8, 2880))
        suite = planning.estimate("suite", seeds=[1, 2, 3], time_budget_seconds=900)
        self.assertEqual((suite["arms"], suite["training_upper_bound_seconds"]), (60, 54000))
        self.assertEqual(planning.estimate("confirmation")["datasets"], 5)
        self.assertEqual(planning.estimate("final", time_budget_seconds=10500)["training_upper_bound_seconds"], 42000)
        self.assertEqual(pilot["direct_rental_cost"], 0)
        self.assertIsNone(pilot["electricity_cost"])

    def test_sanitized_worker_environment(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret", "HTTPS_PROXY": "secret-proxy", "PYTHONPATH": "/developer"}):
            env = sanitized_environment(4)
        self.assertFalse(any("OPENAI" in name or "PROXY" in name.upper() or name == "PYTHONPATH" for name in env))
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_fixture_plan_oracle_boundary_and_all_seals(self):
        with tempfile.TemporaryDirectory(prefix="lab план ") as temporary:
            root = Path(temporary)
            incumbent, candidate = root / "inc", root / "cand"
            incumbent.mkdir(); candidate.mkdir()
            identities = {str(incumbent): {"root": str(incumbent), "sha": planning.INCUMBENT_SHA, "dirty": False,
                                         "source_hashes": {}, "source_fingerprint": "i" * 64},
                          str(candidate): {"root": str(candidate), "sha": planning.CANDIDATE_SHA, "dirty": False,
                                           "source_hashes": {}, "source_fingerprint": "c" * 64},
                          str(planning.ROOT): {"root": str(planning.ROOT), "sha": "a" * 40, "dirty": False,
                                              "source_hashes": {}, "source_fingerprint": "a" * 64}}
            packages = {}
            for line in (planning.ROOT / "requirements.txt").read_text().splitlines():
                if "==" in line and not line.startswith("#"):
                    name, version = line.split("=="); packages[name] = version
            versions = {"python": "3.12.test", "packages": packages}
            def identity(path):
                return deepcopy(identities[str(path)])
            output = root / "run"
            with patch.object(planning, "source_identity", side_effect=identity), patch.object(planning, "check_identity"), \
                 patch.object(planning, "environment_versions", return_value=versions):
                plan = planning.prepare(output=output, manifest=planning.ROOT / "research_lab/datasets.json",
                    profile="fixture-smoke", incumbent_root=incumbent, candidate_root=candidate,
                    python="python", cache=root / "cache", time_budget_seconds=20)
                self.assertFalse(plan["policy_comparison"])
                self.assertEqual(plan["estimate"]["arms"], 2)
                case = plan["cases"][0]
                for arm in case["arms"].values():
                    config = read_json(output / arm["config_path"])
                    self.assertNotIn("audit", json.dumps(config))
                    self.assertNotIn(case["audit_path"], json.dumps(config))
                    self.assertEqual(config["time_budget_seconds"], 20)
                self.assertEqual(read_json(output / case["arms"]["candidate"]["config_path"])["solver_args"],
                                 planning.COMMON_ARGS + planning.CANDIDATE_ARGS)
                labels = output / case["audit_path"]
                old = labels.read_bytes(); labels.write_bytes(old + b" ")
                with self.assertRaisesRegex(ValueError, "Audit labels"):
                    planning.load_plan(output, execution=False)
                labels.write_bytes(old)
                for filename, message in ((case["task_template"] + "/inputs/test.csv", "Template"),
                        (case["arms"]["incumbent"]["task_dir"] + "/inputs/train.csv", "Arm input"),
                        (case["arms"]["candidate"]["config_path"], "Worker config"),
                        ("manifest.json", "Manifest"), ("plan.json", "Plan seal")):
                    path = output / filename
                    old = path.read_bytes(); path.write_bytes(old + b" ")
                    with self.assertRaisesRegex(ValueError, message):
                        planning.load_plan(output, execution=False)
                    path.write_bytes(old)
                planning.load_plan(output, execution=False)
                extra = output / case["arms"]["incumbent"]["task_dir"] / "inputs/extra.csv"
                extra.write_text("target\nsecret\n")
                with self.assertRaisesRegex(ValueError, "file set drift"):
                    planning.load_plan(output, execution=False)
                extra.unlink()

    def test_new_output_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "new output"):
                planning.prepare(output=temporary, manifest="missing", profile="pilot", incumbent_root="missing",
                                 candidate_root="missing", python="missing", cache="missing")


if __name__ == "__main__":
    unittest.main()
