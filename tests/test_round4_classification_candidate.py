"""Round4 supervision, fresh refit and the genuine v3 subprocess boundary.

All mocked checks are fit-free. The final smoke uses only 90 synthetic rows:
two small native reference fits plus two isolated candidate fits.
"""
from contextlib import chdir
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from agent import oof_classification
from agent import round4_classification_candidate as candidate
from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import make_model_config
from agent.research_candidate import freeze_candidate, run_candidate
from agent.task_spec import TaskSpec


CLASSES = ["z", "01", "NA"]


def base_recipe():
    return {"components": [
        {"action": action, "model_config": make_model_config(action, "classification", {"n_estimators": 50}),
         "feature_config": normalize_feature_config()} for action in ("baseline", "lightgbm")]}


def context(variant="identity", deadline=500., phase="inner_selection"):
    return {"recipe": candidate.make_recipe(base_recipe(), CLASSES, variant=variant),
            "classes": list(CLASSES), "features": ["number", "kind"], "numeric": ["number"],
            "categorical": ["kind"], "task_type": "classification", "metric": "Macro F1",
            "prediction_kind": "labels", "positive_class": None, "phase": phase,
            "deadline_monotonic": deadline, "nonce": "synthetic", "code_hash": "code", "recipe_hash": "recipe"}


def frames(rows=18):
    X = pd.DataFrame({"number": np.arange(rows).astype(str), "kind": np.asarray(["a", "b", "c"])[np.arange(rows) % 3]},
                     index=np.arange(1000, 1000 + rows))
    y = pd.Series(np.asarray(CLASSES)[np.arange(rows) % 3], index=X.index)
    P = pd.DataFrame({"number": ["100", "101"], "kind": ["unseen", "a"]}, index=[9998, 9999])
    return X, y, P


def partition(rows=300):
    A, B = np.arange(180), np.arange(180, rows)
    folds = []
    for i in range(3):
        held = A[i * 60:(i + 1) * 60]
        folds.append({"fit_rows": np.setdiff1d(A, held).tolist(), "held_rows": held.tolist()})
    return {"available": True, "reason": "synthetic_grouped_split", "groups": list(range(rows)),
            "tuning_rows": A.tolist(), "selection_rows": B.tolist(), "folds": folds, "support": {}}


class Clock:
    def __init__(self, now=100.):
        self.now = now

    def __call__(self):
        return self.now


class FakeFits:
    """Records callback inputs; never constructs a model or guesses fit labels."""
    def __init__(self, fail_at=None):
        self.calls, self.records = [], []
        self.fail_at = fail_at
        self.started = self.completed = 0

    def __call__(self, X_fit, y_fit, X_predict, *, timeout_seconds, fit_index):
        self.calls.append((fit_index, X_fit.copy(), np.asarray(y_fit).copy(), X_predict.copy(), timeout_seconds))
        self.started += 2
        if fit_index == self.fail_at:
            raise TimeoutError("synthetic callback timeout")
        self.completed += 2
        # Earlier OOF predictions differ deliberately from the final ones.
        row = [.45, .4, .15] if fit_index == "final" else [.05, .05, .9]
        return {"probabilities": np.tile(row, (len(X_predict), 1)), "classes": list(CLASSES)}

    def counts(self):
        return {"physical_fits_started": self.started, "physical_fits_completed": self.completed}


class Round4CandidateTests(unittest.TestCase):
    def test_callback_supervisor_receives_remaining_budget_and_only_current_fit_labels(self):
        X, y, P = frames()
        ctx, clock = context(), Clock()
        dump = joblib.dump
        seen = {}
        def slow_dump(value, path):
            result = dump(value, path)
            clock.now += 3.5
            return result
        def supervisor(code, work, logs, step, *, timeout, max_memory_bytes):
            request_path = work / "request.joblib"
            payload = joblib.load(request_path)
            seen.update(payload=payload, timeout=timeout, memory=max_memory_bytes, code=code)
            self.assertEqual(set(payload), {"X_fit", "y_fit", "X_predict", "context", "recipe"})
            self.assertNotIn("y_predict", payload)
            self.assertEqual(list(payload["X_predict"].columns), list(P.columns))
            np.testing.assert_array_equal(payload["y_fit"], y.iloc[:6])
            np.save(work / "probabilities.npy", np.tile([.5, .3, .2], (len(P), 1)), allow_pickle=False)
            metadata = {"classes": CLASSES, "request_sha256": candidate.file_hash(request_path),
                        "probabilities_sha256": candidate.file_hash(work / "probabilities.npy"),
                        "fit_rows": 6, "predict_rows": len(P)}
            (work / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}
        with tempfile.TemporaryDirectory() as root, patch.object(candidate.time, "monotonic", clock), \
                patch("joblib.dump", side_effect=slow_dump), patch.object(candidate, "run_python", side_effect=supervisor) as launch:
            fits = candidate.BoundedProbabilityFits(Path(root), ctx, ctx["recipe"], deadline=150.)
            result = fits(X.iloc[:6], y.iloc[:6], P, timeout_seconds=20., fit_index=0)
            launch.assert_called_once()
            self.assertEqual(seen["timeout"], 16.5)
            self.assertEqual(seen["payload"]["context"]["deadline_monotonic"], 120.)
            self.assertEqual(seen["memory"], 6 * 1024 ** 3)
            self.assertIn("worker_fit(", seen["code"])
            self.assertIn("from agent.round4_classification_candidate import worker_fit", seen["code"])
            self.assertEqual(result["classes"], CLASSES)
            self.assertEqual(fits.records[0]["metadata"]["fit_rows"], 6)
            self.assertTrue((Path(root) / "f0" / "execution.json").is_file())

    def test_expired_staging_and_fit_cap_cannot_launch_child(self):
        X, y, P = frames()
        ctx, clock, dump = context(), Clock(), joblib.dump
        with tempfile.TemporaryDirectory() as root, patch.object(candidate.time, "monotonic", clock), \
                patch.object(candidate, "run_python") as launch:
            fits = candidate.BoundedProbabilityFits(Path(root) / "expired", ctx, ctx["recipe"], deadline=104.)
            with self.assertRaises(TimeoutError):
                fits(X, y, P, timeout_seconds=30., fit_index=0)
            def exhausted_dump(value, path):
                result = dump(value, path)
                clock.now += 6.
                return result
            fits = candidate.BoundedProbabilityFits(Path(root) / "staging", ctx, ctx["recipe"], deadline=500.)
            with patch("joblib.dump", side_effect=exhausted_dump), self.assertRaisesRegex(TimeoutError, "Staging"):
                fits(X, y, P, timeout_seconds=5., fit_index=0)
            fits = candidate.BoundedProbabilityFits(Path(root) / "cap", ctx, ctx["recipe"], deadline=500.)
            with patch.object(fits, "counts", return_value={"physical_fits_started": 2}), self.assertRaisesRegex(TimeoutError, "fit cap"):
                fits(X, y, P, timeout_seconds=30., fit_index=0)
            launch.assert_not_called()

    def test_worker_emits_fit_events_and_binds_metadata_to_request_and_probabilities(self):
        X, y, P = frames()
        ctx = context(deadline=time.monotonic() + 30.)
        def adapter(fitting, labels, predicting, augmented_context, *, recipe, sample_weight, fit_observer):
            np.testing.assert_array_equal(labels, y)
            self.assertEqual(augmented_context["recipe"], ctx["recipe"])
            self.assertEqual(recipe, ctx["recipe"]["probability_recipe"])
            self.assertIsNone(sample_weight)
            for status in ("started", "completed"):
                fit_observer({"component_index": 0, "action": "baseline", "status": status,
                              "fit_rows": len(fitting), "predict_rows": len(predicting)})
            return {"probabilities": np.tile([.5, .3, .2], (len(P), 1)), "classes": CLASSES,
                    "evidence": {"physical_fits_started": 1, "physical_fits_completed": 1}}
        with tempfile.TemporaryDirectory() as root, patch.object(candidate, "fit_predict_probabilities", side_effect=adapter) as fitted:
            directory = Path(root)
            request = directory / "request.joblib"
            joblib.dump({"X_fit": X, "y_fit": y, "X_predict": P, "context": ctx, "recipe": ctx["recipe"]}, request)
            digest = candidate.file_hash(request)
            candidate.worker_fit(request)
            fitted.assert_called_once()
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["request_sha256"], digest)
            self.assertEqual(candidate.file_hash(request), digest)
            self.assertEqual(metadata["probabilities_sha256"], candidate.file_hash(directory / "probabilities.npy"))
            self.assertEqual(metadata["fit_indices"], X.index.tolist())
            self.assertEqual(metadata["predict_indices"], P.index.tolist())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in events], ["started", "completed"])

    def test_expired_original_deadline_stops_before_preparation_or_model_work(self):
        X, y, P = frames()
        for deadline in (99., 104.):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as root, chdir(root), \
                    patch.object(candidate.time, "monotonic", return_value=100.), \
                    patch.object(candidate, "coverage") as coverage, patch.object(candidate, "run_python") as launch:
                with self.assertRaises(TimeoutError):
                    candidate.execute_candidate(X, y, P, context("oof", deadline=deadline))
                coverage.assert_not_called()
                launch.assert_not_called()
                evidence = json.loads(Path("round4_evidence.json").read_text())
                self.assertEqual(evidence["status"], "failed")
                self.assertEqual(evidence["physical_fits_started"], 0)

    def test_final_reserve_prevents_any_oof_fit_and_still_uses_fresh_full_fit(self):
        X, y, P = frames(300)
        fits = FakeFits()
        with tempfile.TemporaryDirectory() as root, chdir(root), \
                patch.object(candidate.time, "monotonic", return_value=100.), \
                patch.object(candidate, "coverage", return_value=partition()), \
                patch.object(candidate, "BoundedProbabilityFits", return_value=fits):
            output = candidate.execute_candidate(X, y, P, context("oof", deadline=210.))
            self.assertEqual([call[0] for call in fits.calls], ["final"])
            pd.testing.assert_frame_equal(fits.calls[0][1], X)
            np.testing.assert_array_equal(fits.calls[0][2], y)
            np.testing.assert_array_equal(output["predictions"], ["z", "z"])
            evidence = json.loads(Path("round4_evidence.json").read_text())
            self.assertEqual(evidence["oof"]["reason"], "deadline_exhausted")
            self.assertEqual(evidence["oof"]["completed_fits"], 0)
            self.assertTrue(evidence["fresh_full_refit"])
            self.assertEqual(evidence["physical_fits_completed"], 2)

    def test_partial_oof_timeout_never_calibrates_or_mixes_predictions_into_final(self):
        X, y, P = frames(300)
        fits = FakeFits(fail_at=1)
        with tempfile.TemporaryDirectory() as root, chdir(root), \
                patch.object(candidate.time, "monotonic", return_value=100.), \
                patch.object(candidate, "coverage", return_value=partition()), \
                patch.object(candidate, "BoundedProbabilityFits", return_value=fits), \
                patch.object(oof_classification, "fit_guarded_decision_rule", wraps=oof_classification.fit_guarded_decision_rule) as guard:
            output = candidate.execute_candidate(X, y, P, context("oof"))
            self.assertEqual([call[0] for call in fits.calls], [0, 1, "final"])
            self.assertEqual(guard.call_count, 1)
            self.assertEqual(len(guard.call_args.args[0]), 0)  # only the identity-rule constructor
            self.assertTrue(set(fits.calls[0][1].index).isdisjoint(fits.calls[0][3].index))
            pd.testing.assert_frame_equal(fits.calls[-1][1], X)
            np.testing.assert_array_equal(fits.calls[-1][2], y)
            np.testing.assert_array_equal(output["predictions"], ["z", "z"])
            evidence = json.loads(Path("round4_evidence.json").read_text())
            self.assertEqual(evidence["oof"]["reason"], "callback_timeout")
            self.assertEqual(evidence["oof"]["completed_fits"], 1)
            self.assertEqual(evidence["decision"], {"rules_trained": 0, "rules_accepted": 0, "rules_applied": 0})
            self.assertEqual(evidence["rule"]["multipliers"], [1., 1., 1.])
            self.assertEqual(evidence["physical_fits_started"], 6)
            self.assertEqual(evidence["physical_fits_completed"], 4)

    def test_phase_and_working_path_do_not_change_identity_algorithm(self):
        X, y, P = frames()
        outputs = []
        for phase in ("inner_selection", "full_refit"):
            ctx, fits = context(phase=phase), FakeFits()
            original = deepcopy(ctx)
            with tempfile.TemporaryDirectory(prefix=phase) as root, chdir(root), \
                    patch.object(candidate.time, "monotonic", return_value=100.), \
                    patch.object(candidate, "BoundedProbabilityFits", return_value=fits), \
                    patch.object(candidate, "coverage") as coverage:
                outputs.append(candidate.execute_candidate(X, y, P, ctx)["predictions"])
                coverage.assert_not_called()
                self.assertEqual([call[0] for call in fits.calls], ["final"])
                self.assertEqual(ctx, original)
                evidence = json.loads(Path("round4_evidence.json").read_text())
                self.assertEqual(evidence["phase"], phase)
                self.assertEqual(evidence["physical_fits_completed"], 2)
        np.testing.assert_array_equal(outputs[0], outputs[1])


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "Pinned LightGBM is not installed")
class NativeRound4ProtocolTests(unittest.TestCase):
    def test_real_v3_identity_matches_native_reference_and_saves_verified_child_receipts(self):
        from agent.research_v3 import BASELINE_SOURCE
        X, y, P = frames(90)
        base = base_recipe()
        namespace = {}
        exec(compile(BASELINE_SOURCE, "native_reference", "exec"), namespace)
        ctx = context(deadline=time.monotonic() + 60.)
        ctx["recipe"] = base
        expected = namespace["fit_predict"](X, y, P, ctx)["predictions"]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "proposal.py"
            source.write_text(candidate.candidate_source(), encoding="utf-8")
            frozen = freeze_candidate(source, candidate.make_recipe(base, CLASSES), root / "frozen")
            spec = TaskSpec(root / "task.md", root / "private", "row_key", "answer",
                            ("number", "kind"), ("number",), ("kind",), tuple(CLASSES),
                            ("row_key", "answer"), (), None, get_metric("Macro F1"))
            fit, predict = X.copy(), P.copy()
            fit.insert(0, "row_key", [f"train-{i}" for i in range(len(fit))])
            fit["answer"] = y
            predict.insert(0, "row_key", ["0007", "NA"])
            result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=predict,
                                   output_dir=root / "executions", phase="outer_validation", timeout=45.,
                                   actuals=["z", "01"], max_memory_bytes=6 * 1024 ** 3)
            result.verify()
            output = pd.read_csv(result.output_path, dtype=str, keep_default_na=False)
            self.assertEqual(output["row_key"].tolist(), ["0007", "NA"])
            np.testing.assert_array_equal(output["answer"], expected)
            self.assertEqual(result.metric["name"], "Macro F1")
            work = result.output_path.parent
            evidence = json.loads((work / "round4_evidence.json").read_text())
            self.assertEqual(evidence["status"], "completed")
            self.assertTrue(evidence["fresh_full_refit"])
            self.assertEqual(evidence["physical_fits_started"], 2)
            self.assertEqual(evidence["physical_fits_completed"], 2)
            self.assertEqual(evidence["recipe_hash"], frozen.recipe_hash)
            self.assertEqual(len(evidence["child_runs"]), 1)
            child = work / "round4_work" / "ffinal"
            metadata = json.loads((child / "metadata.json").read_text())
            execution = json.loads((child / "execution.json").read_text())
            self.assertEqual(execution["execution"]["exit_code"], 0)
            self.assertFalse(execution["execution"]["timed_out"])
            self.assertEqual(metadata, evidence["child_runs"][0]["metadata"])
            self.assertEqual(metadata["classes"], CLASSES)
            self.assertEqual(metadata["request_sha256"], candidate.file_hash(child / "request.joblib"))
            self.assertEqual(metadata["probabilities_sha256"], candidate.file_hash(child / "probabilities.npy"))
            payload = joblib.load(child / "request.joblib")
            self.assertEqual(set(payload), {"X_fit", "y_fit", "X_predict", "context", "recipe"})
            self.assertEqual(list(payload["X_fit"].columns), ["number", "kind"])
            self.assertEqual(list(payload["X_predict"].columns), ["number", "kind"])
            self.assertEqual(payload["context"]["metric"]["name"], "Macro F1")
            self.assertEqual(payload["context"]["prediction_kind"], "labels")
            self.assertLessEqual(payload["context"]["deadline_monotonic"],
                                 json.loads((work / "context.json").read_text())["deadline_monotonic"])


if __name__ == "__main__":
    unittest.main()
