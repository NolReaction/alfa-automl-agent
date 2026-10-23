"""Fit-free Round5 admission, accounting, identity and OFF-route contracts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import joblib
import pandas as pd

from agent.features import normalize_feature_config
from agent.metrics import get_metric
from agent.model_config import make_model_config
from agent.offline_research import canonical_recipe_key
from agent.research_candidate import CandidateResult, freeze_candidate
from agent import research_v3
from tests.test_coordinator_v2 import Proposer


ORDINARY_CODE_HASH = hashlib.sha256(research_v3.BASELINE_SOURCE.encode()).hexdigest()


def ordinary(action="linear", **parameters):
    return {"id": "presentation-only", "label": "Arbitrary label", "action": action,
            "model_config": make_model_config(action, "classification", parameters),
            "feature_config": normalize_feature_config()}


def target_slot():
    return ordinary("catboost", iterations=400, depth=10, learning_rate=.03)


def feedback():
    return {"verified": True, "accepted": False, "reason": "not_better",
            "metric": {"name": "Macro F1", "value": .5, "higher_is_better": True}}


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class Round5HostTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "baseline.py"
        source.write_text(research_v3.BASELINE_SOURCE, encoding="utf-8")
        baseline = freeze_candidate(source, ordinary("baseline"), self.root / "baseline")
        spec = SimpleNamespace(metric=get_metric("Macro F1"), classification=True,
            prediction_kind="labels", features=("number", "kind"), numeric=("number",),
            categorical=("kind",), classes=("literal_z", "literal_a"),
            id_column="row_id", target_column="target")
        train = pd.DataFrame({"row_id": [str(i) for i in range(400)],
            "number": [str(i) for i in range(400)], "kind": ["a", "b"] * 200,
            "target": ["literal_z", "literal_a"] * 200})
        gate = SimpleNamespace(spec=spec, research_spec=spec, research_train=train,
            features=spec.features, numeric=spec.numeric, categorical=spec.categorical,
            fit_rows=list(range(320)), held_rows=list(range(320, 400)),
            choice={"strategy": "grouped"}, split={"strategy": "grouped"},
            inner_split={"strategy": "grouped"},
            manifest={"task_semantics": {"adaptive_research_allowed": True}},
            check_inputs=Mock(), check_source=Mock())
        # Eligibility may inspect lengths of eventual phases, never their values.
        gate.outer_fit_frame, gate.outer_holdout_frame = range(400), range(100)
        gate.train, gate.test = range(500), range(100)
        self.state = SimpleNamespace(gate=gate, run_dir=self.root, workspace=self.root,
            report={"stage": "inner_research", "quality_options": {"classification_round4_mode": "on"},
                    "api": {"attempts": 0}, "successful_experiments": 0},
            baseline=baseline, baseline_inner={"value": .5}, selected_candidate=None,
            best_inner={"value": .5}, tried_recipes=[], adaptive_search=True,
            dialogue_closed=False, check_source=Mock(), save=Mock(), event=Mock(),
            research_timeout=Mock(side_effect=lambda maximum=1800: min(2000., maximum)),
            remaining=Mock(return_value=3000.), _local_frozen_attempts=[])

    def test_off_and_default_keep_base_order_sources_caps_and_extension_schedule(self):
        """Golden calls are the base 66e9481 continuation contract, without fits."""
        queue = [ordinary("linear", C=.1), ordinary("extra_trees"), target_slot(),
                 ordinary("linear", C=10.)]
        original = deepcopy(queue)
        all_orders = []
        for mode in (None, "off"):
            with self.subTest(mode=mode):
                self.state.report["quality_options"] = {} if mode is None else {"classification_round4_mode": mode}
                order = []
                proposer = Proposer([ordinary("linear", C=100.)])

                def assess(state, candidate, hypothesis, *, origin):
                    candidate.verify()
                    self.assertEqual(candidate.code_hash, ORDINARY_CODE_HASH)
                    self.assertEqual(origin, "local")
                    order.append(canonical_recipe_key(candidate.recipe, candidate.code_hash))
                    return feedback()

                def extensions(state, assessor, *, max_trials, prioritize):
                    self.assertIs(assessor, assess)
                    self.assertEqual(max_trials, 6)
                    self.assertFalse(prioritize)
                    order.append("extensions")
                    return {"attempted": 0}

                from agent import round5_classification_host as host
                with patch.object(research_v3, "local_candidate_plan", return_value=queue), \
                        patch("agent.adaptive_search.AdaptiveSearch", return_value=proposer) as construct, \
                        patch("agent.research_extensions.run_research_extensions", side_effect=extensions), \
                        patch.object(research_v3, "_assess_candidate", new=assess), \
                        patch.object(host, "run_round5_host", side_effect=AssertionError("OFF invoked ON helper")):
                    result = research_v3._continue_locally(self.state)
                construct.assert_called_once_with(True, "Macro F1", max_trials=64)
                self.assertEqual(result["attempted"], 5)
                self.assertEqual(result["extension_schedule"], "after_warmup")
                self.assertEqual(order, [*(canonical_recipe_key(item, ORDINARY_CODE_HASH) for item in queue),
                    "extensions", canonical_recipe_key(ordinary("linear", C=100.), ORDINARY_CODE_HASH)])
                self.assertEqual(self.state.report["api"], {"attempts": 0})
                self.assertEqual(self.state.report["successful_experiments"], 0)
                all_orders.append(order)
        self.assertEqual(all_orders[0], all_orders[1])
        self.assertEqual(queue, original)

    def test_semantic_slot_does_not_depend_on_position_or_presentation(self):
        from agent import round5_classification_host as host
        target = target_slot()
        target.update(id="unrelated", label="No CatBoost in this label")
        for queue in ([target, ordinary()], [ordinary(), target, ordinary("extra_trees")]):
            with self.subTest(queue=queue):
                self.assertIs(host.semantic_slot(queue, [], ORDINARY_CODE_HASH), target)
        wrong_depth = ordinary("catboost", iterations=400, depth=4, learning_rate=.03)
        wrong_rate = ordinary("catboost", iterations=400, depth=10, learning_rate=.08)
        self.assertIsNone(host.semantic_slot([wrong_depth, wrong_rate], [], ORDINARY_CODE_HASH))

    def test_slot_seen_uses_canonical_recipe_and_actual_source_hash(self):
        from agent import round5_classification_host as host
        target, seen_target = target_slot(), target_slot()
        seen_target.update(id="another", label="Already measured under another label")
        same = [{"recipe": seen_target, "code_hash": ORDINARY_CODE_HASH}]
        other_source = [{"recipe": seen_target, "code_hash": "f" * 64}]
        self.assertIsNone(host.semantic_slot([target], same, ORDINARY_CODE_HASH))
        self.assertIs(host.semantic_slot([target], other_source, ORDINARY_CODE_HASH), target)

    def run_host(self, *, recipes=None, seen=None, assess=None, track=None):
        from agent import round5_classification_host as host
        return host.run_round5_host(self.state, recipes if recipes is not None else [target_slot()],
            seen if seen is not None else [], ORDINARY_CODE_HASH,
            assess if assess is not None else Mock(return_value=feedback()),
            track if track is not None else Mock())

    def test_cheap_ineligibility_never_reaches_preflight_or_fit(self):
        from agent import round5_classification_host as host
        mutations = {
            "adaptive_disabled": lambda: setattr(self.state, "adaptive_search", False),
            "research_closed": lambda: setattr(self.state, "dialogue_closed", True),
            "outer_stage": lambda: self.state.report.update(stage="outer_validation"),
            "unmeasured_baseline": lambda: setattr(self.state, "baseline_inner", None),
            "no_inner": lambda: setattr(self.state.gate, "held_rows", []),
            "semantics": lambda: self.state.gate.manifest["task_semantics"].update(adaptive_research_allowed=False),
            "random_policy": lambda: setattr(self.state.gate, "choice", {"strategy": "random"}),
            "one_class": lambda: setattr(self.state.gate.spec, "classes", ("literal_z",)),
            "repeated_classes": lambda: setattr(self.state.gate.spec, "classes", ("literal_z", "literal_z")),
            "too_many_classes": lambda: setattr(self.state.gate.spec, "classes", tuple(map(str, range(51)))),
            "fit_row_cap": lambda: setattr(self.state.gate, "fit_rows", range(100_001)),
            "predict_row_cap": lambda: setattr(self.state.gate, "held_rows", range(500_001)),
            "overlapping_roles": lambda: setattr(self.state.gate.spec, "numeric", ("number", "kind")),
            "missing_role": lambda: setattr(self.state.gate.spec, "categorical", ()),
            "id_as_feature": lambda: setattr(self.state.gate.spec, "features", ("row_id", "kind")),
            "target_as_feature": lambda: setattr(self.state.gate.spec, "features", ("target", "kind")),
        }
        for name, mutate in mutations.items():
            with self.subTest(condition=name):
                # Recreate only the small fixture; no large frame is allocated.
                if name != "adaptive_disabled":
                    self.setUp()
                mutate()
                queue, seen, assessor, track = [ordinary(), target_slot()], [], Mock(), Mock()
                self.state.measure_inner = Mock(side_effect=AssertionError("Ineligible state fitted"))
                with patch.object(host, "bounded_preflight", side_effect=AssertionError("Early bounds staged data")):
                    result_queue, result_seen, used = self.run_host(recipes=queue, seen=seen, assess=assessor, track=track)
                self.assertEqual(result_queue, queue)
                self.assertEqual(result_seen, seen)
                self.assertEqual(used, 0)
                self.state.measure_inner.assert_not_called()
                assessor.assert_not_called()
                track.assert_not_called()

    def test_missing_or_previously_measured_slot_does_not_pay_for_preflight(self):
        from agent import round5_classification_host as host
        for queue, seen in (([ordinary()], []), ([target_slot()],
                [{"recipe": target_slot(), "code_hash": ORDINARY_CODE_HASH}])):
            with self.subTest(queue=queue, seen=seen), \
                    patch.object(host, "bounded_preflight", side_effect=AssertionError("No replacement slot")):
                self.state.measure_inner = Mock()
                result_queue, result_seen, used = self.run_host(recipes=queue, seen=seen)
                self.assertEqual((result_queue, result_seen, used), (queue, seen, 0))
                self.state.measure_inner.assert_not_called()

    def test_shape_caps_are_checked_without_materializing_large_matrices(self):
        from agent import round5_classification_host as host
        for count, fit_rows, predict_rows in ((513, 320, 80), (101, 100_000, 80), (21, 320, 500_000)):
            with self.subTest(features=count, fit_rows=fit_rows, predict_rows=predict_rows):
                spec = self.state.gate.spec
                spec.features = tuple("feature_" + str(i) for i in range(count))
                spec.numeric, spec.categorical = spec.features, ()
                self.state.gate.features = spec.features
                self.state.gate.numeric, self.state.gate.categorical = spec.numeric, spec.categorical
                self.state.gate.fit_rows = range(fit_rows)
                self.state.gate.held_rows = range(predict_rows)
                self.state.measure_inner = Mock()
                with patch.object(host, "bounded_preflight", side_effect=AssertionError("Oversized matrix staged")):
                    queue, seen, used = self.run_host()
                self.assertEqual(used, 0)
                self.state.measure_inner.assert_not_called()
                self.assertEqual(self.state.report["classification_round4"]["reason"], "feature_matrix_cap_exceeded")

    def test_paid_early_invocations_reduce_ordinary_cap_without_adaptive_observation(self):
        from agent import round5_classification_host as host
        from agent import offline_research
        first, target, last = ordinary("linear", C=.1), target_slot(), ordinary("extra_trees")
        queue, filtered = [first, target, last], [first, last]
        custom = {"family": "round4_classification_v1", "variant": "oof"}
        added_seen = [{"recipe": custom, "code_hash": "c" * 64},
                      {"recipe": target, "code_hash": ORDINARY_CODE_HASH}]
        proposer, order = Proposer([ordinary("linear", C=100.)]), []

        def assess(state, candidate, hypothesis, *, origin):
            order.append(candidate.recipe)
            return feedback()

        with patch.object(research_v3, "local_candidate_plan", return_value=queue), \
                patch.object(host, "run_round5_host", return_value=(filtered, added_seen, 2)) as early, \
                patch("agent.adaptive_search.AdaptiveSearch", return_value=proposer), \
                patch("agent.research_extensions.run_research_extensions", return_value={"attempted": 0}), \
                patch.object(research_v3, "_assess_candidate", new=assess), \
                patch.object(offline_research, "run_deadline_continuation", wraps=offline_research.run_deadline_continuation) as continuation:
            research_v3._continue_locally(self.state)
        early.assert_called_once()
        self.assertEqual(continuation.call_args.kwargs["max_trials"], 62)
        self.assertEqual(order[:2], filtered)
        self.assertNotIn(target, order)
        self.assertEqual([item for item, _ in proposer.observations], [ordinary("linear", C=100.)])

    def test_actual_oof_attempt_is_frozen_custom_source_and_removes_only_semantic_slot(self):
        from agent import round5_classification_host as host
        from agent.round4_classification_candidate import checked_recipe
        clock, probes = Clock(), []
        initial_best, initial_baseline = self.state.best_inner, self.state.baseline
        target = target_slot()
        queue = [ordinary("linear", C=.1), target, ordinary("extra_trees")]
        original = deepcopy(queue)

        def measure(candidate, *, maximum):
            probes.append(candidate)
            self.assertEqual(maximum, 120)
            self.assertEqual(candidate.recipe["variant"], "identity")
            self.assertEqual(candidate.recipe["caps"]["trial_seconds"], 300)
            self.assertIs(self.state.best_inner, initial_best)
            clock.now += 20.
            return SimpleNamespace(receipt_path=self.root / "probe" / "receipt.json", receipt_hash="e" * 64)

        def assess(state, candidate, hypothesis, *, origin, maximum):
            self.assertEqual(origin, "local")
            self.assertEqual(maximum, 600)
            candidate.verify()
            self.assertNotEqual(candidate.code_hash, ORDINARY_CODE_HASH)
            self.assertEqual(candidate.recipe, checked_recipe(candidate.recipe))
            self.assertEqual(candidate.recipe["variant"], "oof")
            self.assertEqual(candidate.recipe["caps"]["trial_seconds"], 600)
            self.assertEqual(state._round5_snapshot["oof_recipe_hash"], candidate.recipe_hash)
            clock.now += 30.
            return feedback()

        self.state.measure_inner = Mock(side_effect=measure)
        track, assessor = Mock(), Mock(side_effect=assess)
        with patch.object(host, "bounded_preflight", return_value={"reason": None}), \
                patch.object(host.time, "monotonic", clock):
            result_queue, result_seen, used = self.run_host(recipes=queue, assess=assessor, track=track)
        self.assertEqual(used, 2)
        self.assertEqual(result_queue, [queue[0], queue[2]])
        self.assertEqual(queue, original)
        self.assertEqual(self.state.report["local_experiments"], 2)
        self.assertEqual(track.call_count, 2)
        self.state.measure_inner.assert_called_once()
        assessor.assert_called_once()
        self.assertIs(self.state.baseline, initial_baseline)
        self.assertIs(self.state.best_inner, initial_best)
        self.assertIsNone(self.state.selected_candidate)
        known = {canonical_recipe_key(row["recipe"], row["code_hash"]) for row in result_seen}
        self.assertIn(canonical_recipe_key(target, ORDINARY_CODE_HASH), known)
        oof = track.call_args_list[-1].args[1]
        self.assertIn(canonical_recipe_key(oof.recipe, oof.code_hash), known)
        report = self.state.report["classification_round4"]
        self.assertEqual(report["probe"]["elapsed_seconds"], 20.)
        self.assertEqual(report["admission"]["estimated_oof_seconds"], 240.)
        self.assertEqual(report["ordinary_remaining_trial_cap"], 62)
        self.assertTrue(report["ordinary_slot_removed"])
        self.assertEqual(report["reached_mode"], "on")
        manifest = json.loads(Path(report["manifest"]).read_text())
        self.assertEqual(manifest["classes"], list(self.state.gate.spec.classes))
        self.assertEqual(manifest["source_hash"], oof.code_hash)
        self.assertIn("agent.round4_classification_probability", manifest["dependencies"]["sources"])

    def test_failed_probe_preserves_spent_partial_fit_events_and_ordinary_queue(self):
        from agent import round5_classification_host as host
        clock, queue, assessor = Clock(), [ordinary(), target_slot()], Mock()
        initial_baseline, initial_best = self.state.baseline, self.state.best_inner

        def measure(candidate, *, maximum):
            directory = self.root / "inner_evaluations" / "failed-probe" / "round4_work" / "ffinal"
            directory.mkdir(parents=True)
            events = [{"component_index": 0, "status": "started"},
                      {"component_index": 0, "status": "completed"},
                      {"component_index": 1, "status": "started"}]
            (directory / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            clock.now += 120.
            raise TimeoutError("Bounded probe failed")

        self.state.measure_inner = Mock(side_effect=measure)
        with patch.object(host, "bounded_preflight", return_value={"reason": None}), \
                patch.object(host.time, "monotonic", clock):
            result_queue, result_seen, used = self.run_host(recipes=queue, assess=assessor)
        self.assertEqual(result_queue, queue)
        self.assertEqual(used, 1)
        self.assertEqual(self.state.report["local_experiments"], 1)
        assessor.assert_not_called()
        self.assertIs(self.state.baseline, initial_baseline)
        self.assertIs(self.state.best_inner, initial_best)
        report = self.state.report["classification_round4"]
        self.assertEqual(report["reason"], "identity_probe_failed")
        self.assertEqual(report["ordinary_remaining_trial_cap"], 63)
        probe = report["probe"]
        self.assertEqual(probe["elapsed_seconds"], 120.)
        self.assertEqual(probe["attempts"][0]["physical_fits_started"], 2)
        self.assertEqual(probe["attempts"][0]["physical_fits_completed"], 1)
        self.assertFalse(report["ordinary_slot_removed"])
        self.assertEqual(len(result_seen), 1)
        self.assertEqual(result_seen[0]["recipe"]["variant"], "identity")

    def test_post_probe_cost_or_remaining_budget_failure_is_still_paid(self):
        from agent import round5_classification_host as host
        for wall, after_allowance in ((80., 600.), (20., 200.)):
            with self.subTest(wall=wall, after_allowance=after_allowance):
                self.setUp()
                clock, assessor = Clock(), Mock()
                queue = [ordinary(), target_slot()]

                def measure(candidate, *, maximum):
                    clock.now += wall
                    self.state.research_timeout.side_effect = lambda maximum=1800: min(after_allowance, maximum)
                    return SimpleNamespace(receipt_path=self.root / "receipt.json", receipt_hash="e" * 64)

                self.state.measure_inner = Mock(side_effect=measure)
                with patch.object(host, "bounded_preflight", return_value={"reason": None}), \
                        patch.object(host.time, "monotonic", clock):
                    result_queue, result_seen, used = self.run_host(recipes=queue, assess=assessor)
                self.assertEqual(result_queue, queue)
                self.assertEqual(used, 1)
                self.assertEqual(self.state.report["local_experiments"], 1)
                self.assertEqual(len(result_seen), 1)
                assessor.assert_not_called()
                report = self.state.report["classification_round4"]
                self.assertEqual(report["reason"], "oof_cost_admission_failed")
                self.assertEqual(report["probe"]["elapsed_seconds"], wall)
                self.assertEqual(report["admission"]["available_research_seconds"], after_allowance)
                self.assertFalse(report["ordinary_slot_removed"])

    def test_budget_is_rechecked_after_preflight_before_paid_probe(self):
        from agent import round5_classification_host as host
        self.state.measure_inner = Mock()

        def preflight(state, context):
            state.research_timeout.side_effect = lambda maximum=1800: min(119., maximum)
            return {"reason": None, "elapsed_seconds": 59.}

        with patch.object(host, "bounded_preflight", side_effect=preflight):
            queue, seen, used = self.run_host()
        self.state.measure_inner.assert_not_called()
        self.assertEqual(used, 0)
        self.assertEqual(self.state.report["classification_round4"]["reason"], "probe_reserve_unavailable")

    def test_failed_oof_attempt_does_not_restore_spent_slot(self):
        from agent import round5_classification_host as host
        clock, track = Clock(), Mock()

        def measure(candidate, *, maximum):
            clock.now += 10.
            return SimpleNamespace(receipt_path=self.root / "receipt.json", receipt_hash="e" * 64)

        self.state.measure_inner = Mock(side_effect=measure)
        queue = [ordinary(), target_slot(), ordinary("extra_trees")]
        assessor = Mock(side_effect=TimeoutError("Synthetic OOF timeout"))
        with patch.object(host, "bounded_preflight", return_value={"reason": None}), \
                patch.object(host.time, "monotonic", clock):
            result_queue, seen, used = self.run_host(recipes=queue, assess=assessor, track=track)
        self.assertEqual(result_queue, [queue[0], queue[2]])
        self.assertEqual(used, 2)
        self.assertEqual(track.call_count, 2)
        self.assertEqual(self.state.report["classification_round4"]["status"], "failed")
        self.assertTrue(self.state.report["classification_round4"]["ordinary_slot_removed"])
        self.assertIsNone(self.state.selected_candidate)

    def receipt_fixture(self):
        """Pinned host bytes with a failed calibration and a fresh successful final.

        Child payloads are inert bytes: this tests receipt/hash verification,
        not a substitute for the separately supervised real contract smoke.
        """
        from agent import round5_classification_host as host
        directory = self.root / "receipt_fixture"
        directory.mkdir()
        classes = list(self.state.gate.spec.classes)
        recipe = host.make_recipe(host.fixed_incumbent_recipe(), classes, variant="oof")
        source = directory / "source.py"
        source.write_text(host.candidate_source(), encoding="utf-8")
        candidate = freeze_candidate(source, recipe, directory / "frozen")
        invocation = directory / "invocation"
        invocation.mkdir()
        context = {"nonce": "fresh-invocation-nonce", "phase": "inner_selection",
                   "code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash,
                   "classes": classes, "recipe": recipe}
        (invocation / "context.json").write_text(json.dumps(context))
        output = ("row_id,target\n" + "".join(f"{i},{classes[i % 2]}\n" for i in range(80))).encode()
        output_path = invocation / "predictions.csv"
        output_path.write_bytes(output)
        receipt = {key: context[key] for key in ("nonce", "phase", "code_hash", "recipe_hash")}
        receipt.update(version=3, output_hash=hashlib.sha256(output).hexdigest())
        receipt_path = invocation / "receipt.json"
        receipt_bytes = json.dumps(receipt).encode()
        receipt_path.write_bytes(receipt_bytes)
        failed, final = invocation / "round4_work" / "f0", invocation / "round4_work" / "ffinal"
        failed.mkdir(parents=True)
        final.mkdir()
        (failed / "events.jsonl").write_text(json.dumps({"component_index": 0, "status": "started"}) + "\n")
        events = [{"component_index": component, "status": status}
                  for component in (0, 1) for status in ("started", "completed")]
        (final / "events.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
        (final / "request.joblib").write_bytes(b"inert request hash fixture")
        (final / "probabilities.npy").write_bytes(b"inert probability hash fixture")
        metadata = {"request_sha256": host.file_hash(final / "request.joblib"),
                    "probabilities_sha256": host.file_hash(final / "probabilities.npy"),
                    "fit_rows": 320, "predict_rows": 80}
        evidence = {key: context[key] for key in ("nonce", "phase", "code_hash", "recipe_hash", "classes")}
        evidence.update(status="completed", fresh_full_refit=True, full_fit_rows=320, predict_rows=80,
            physical_fits_started=3, physical_fits_completed=2,
            decision={"rules_trained": 0, "rules_accepted": 0, "rules_applied": 0},
            child_runs=[{"fit_index": 0, "metadata": None,
                         "execution": {"exit_code": -9, "timed_out": True}},
                        {"fit_index": "final", "metadata": metadata,
                         "execution": {"exit_code": 0, "timed_out": False}}])
        (invocation / "round4_evidence.json").write_text(json.dumps(evidence))
        self.state._round5_snapshot = {"source_hash": candidate.code_hash,
            "recipe_hashes": (candidate.recipe_hash,), "oof_recipe_hash": candidate.recipe_hash,
            "classes": classes, "candidates": (candidate,), "dependencies": host.dependency_snapshot()}
        self.state.report["classification_round4"] = {}
        result = CandidateResult(output_path, feedback()["metric"], {"exit_code": 0},
            candidate.code_hash, candidate.recipe_hash, "inner_selection", receipt_path,
            hashlib.sha256(output).hexdigest(), len(output), hashlib.sha256(receipt_bytes).hexdigest())
        return candidate, result, context, evidence

    def test_valid_failed_calibration_identity_fallback_keeps_verified_fresh_final(self):
        from agent import round5_classification_host as host
        candidate, result, _, _ = self.receipt_fixture()
        host.verify_result(self.state, candidate, result)
        self.assertFalse(getattr(self.state, "_local_integrity_failed", False))
        record = self.state.report["classification_round4"]["invocations"][0]
        self.assertEqual(record["physical_fits_started"], 3)
        self.assertEqual(record["physical_fits_completed"], 2)
        self.assertTrue(record["fresh_full_refit"])
        self.assertEqual(record["decision"]["rules_applied"], 0)

    def test_invalid_child_final_or_identity_receipts_fail_closed(self):
        from agent import round5_classification_host as host
        cases = ("missing_final", "final_metadata_missing", "successful_child_metadata_missing",
                 "wrong_final_rows", "wrong_class_order", "wrong_identity", "wrong_counts",
                 "no_fresh_refit", "changed_receipt", "changed_child_request", "changed_frozen_recipe")
        for name in cases:
            with self.subTest(case=name):
                self.setUp()
                candidate, result, context, evidence = self.receipt_fixture()
                directory = result.receipt_path.parent
                if name == "missing_final":
                    evidence["child_runs"].pop()
                elif name == "final_metadata_missing":
                    evidence["child_runs"][-1]["metadata"] = None
                elif name == "successful_child_metadata_missing":
                    evidence["child_runs"][0]["execution"] = {"exit_code": 0, "timed_out": False}
                elif name == "wrong_final_rows":
                    evidence["child_runs"][-1]["metadata"]["fit_rows"] -= 1
                elif name == "wrong_class_order":
                    context["classes"] = list(reversed(context["classes"]))
                elif name == "wrong_identity":
                    evidence["recipe_hash"] = "0" * 64
                elif name == "wrong_counts":
                    evidence["physical_fits_started"] += 1
                elif name == "no_fresh_refit":
                    evidence["fresh_full_refit"] = False
                elif name == "changed_receipt":
                    result.receipt_path.write_text("{}")
                elif name == "changed_child_request":
                    (directory / "round4_work" / "ffinal" / "request.joblib").write_bytes(b"changed")
                elif name == "changed_frozen_recipe":
                    candidate.recipe_path.write_text("{}")
                (directory / "context.json").write_text(json.dumps(context))
                (directory / "round4_evidence.json").write_text(json.dumps(evidence))
                with self.assertRaises(ValueError):
                    host.verify_result(self.state, candidate, result)
                self.assertTrue(self.state._local_integrity_failed)
                self.assertNotIn("invocations", self.state.report["classification_round4"])

    def test_dependency_version_and_source_hash_drift_are_terminal(self):
        from agent import round5_classification_host as host
        for drift in ("version", "source"):
            with self.subTest(drift=drift):
                self.setUp()
                self.receipt_fixture()
                host.verify_dependencies(self.state)
                changed = deepcopy(self.state._round5_snapshot["dependencies"])
                if drift == "version":
                    changed["versions"]["lightgbm"] = "unexpected-version"
                else:
                    changed["sources"]["agent.round4_classification_candidate"]["sha256"] = "0" * 64
                with patch.object(host, "dependency_snapshot", return_value=changed), self.assertRaisesRegex(ValueError, "snapshot changed"):
                    host.verify_dependencies(self.state)
                self.assertTrue(self.state._local_integrity_failed)
                # A restored source does not reopen a compromised run.
                with self.assertRaisesRegex(ValueError, "previously failed"):
                    host.verify_dependencies(self.state)

    def test_reordered_research_classes_fail_before_preflight(self):
        from agent import round5_classification_host as host
        self.state.gate.research_spec = deepcopy(self.state.gate.spec)
        self.state.gate.research_spec.classes = tuple(reversed(self.state.gate.spec.classes))
        with patch.object(host, "bounded_preflight", side_effect=AssertionError("Class mismatch staged data")):
            _, _, used = self.run_host()
        self.assertEqual(used, 0)
        self.assertEqual(self.state.report["classification_round4"]["reason"], "class_order_invalid")

    def test_partial_cancelled_and_truncated_fit_events_remain_explicit(self):
        from agent import round5_classification_host as host
        directory = self.root / "events" / "round4_work" / "f1"
        directory.mkdir(parents=True)
        events = [{"status": "started"}, {"status": "completed"},
                  {"status": "started"}, {"status": "cancelled_before_fit"}]
        path = directory / "events.jsonl"
        path.write_text("\n".join(map(json.dumps, events)) + "\n")
        record = host.invocation_evidence(self.root / "events")
        self.assertEqual((record["physical_fits_started"], record["physical_fits_completed"],
                          record["physical_fits_cancelled_before_fit"]), (1, 1, 1))
        with path.open("a") as stream:
            stream.write('{"status":')
        record = host.invocation_evidence(self.root / "events")
        self.assertIsNone(record["physical_fits_started"])
        self.assertEqual(record["observed_fit_counts_lower_bounds"]["physical_fits_started"], 1)
        self.assertTrue(record["event_parse_errors"])

    def test_bounded_structural_preflight_uses_only_inner_fit_and_charges_staging(self):
        from agent import round5_classification_host as host
        clock = Clock()
        self.state.budget = SimpleNamespace(deadline=1000., publication_buffer=5.)
        spec = self.state.gate.research_spec
        context = {"classes": list(spec.classes), "features": list(spec.features),
                   "numeric": list(spec.numeric), "categorical": list(spec.categorical)}
        original_dump = joblib.dump

        def dump(value, path):
            self.assertEqual(set(value), {"X", "y", "context"})
            self.assertEqual(value["X"].index.tolist(), self.state.gate.fit_rows)
            self.assertEqual(value["y"].index.tolist(), self.state.gate.fit_rows)
            self.assertEqual(list(value["X"].columns), list(spec.features))
            self.assertNotIn(spec.target_column, value["X"].columns)
            original_dump(value, path)
            clock.now += 3.

        def supervisor(code, cwd, logs, step, *, timeout, max_memory_bytes):
            self.assertEqual(timeout, 57.)
            self.assertEqual(max_memory_bytes, 6 * 1024 ** 3)
            self.assertIn("preflight_worker", code)
            host.preflight_worker(cwd / "request.joblib", cwd / "result.json")
            clock.now += 2.
            return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}

        with patch("joblib.dump", side_effect=dump), \
                patch("agent.execution.run_python", side_effect=supervisor) as launch, \
                patch.object(host.time, "monotonic", clock):
            result = host.bounded_preflight(self.state, context)
        launch.assert_called_once()
        self.assertIsNone(result["reason"])
        self.assertEqual(result["elapsed_seconds"], 5.)
        self.assertEqual(sum(row["rows"] for row in result["support"]["allowed"].values()), 320)
        self.assertEqual(len(result["support"]["folds"]), 3)
        self.assertEqual(self.state.budget.deadline, 1000.)

    def test_preflight_staging_cannot_extend_deadline_or_launch_late_child(self):
        from agent import round5_classification_host as host
        clock = Clock()
        self.state.budget = SimpleNamespace(deadline=1000., publication_buffer=5.)
        spec = self.state.gate.spec
        context = {"classes": list(spec.classes), "features": list(spec.features)}
        original_dump = joblib.dump

        def slow_dump(value, path):
            original_dump(value, path)
            clock.now += 61.

        with patch("joblib.dump", side_effect=slow_dump), patch("agent.execution.run_python") as launch, \
                patch.object(host.time, "monotonic", clock):
            result = host.bounded_preflight(self.state, context)
        launch.assert_not_called()
        self.assertEqual(result["reason"], "preflight_staging_timeout")
        self.assertEqual(result["elapsed_seconds"], 61.)
        self.assertEqual(self.state.budget.deadline, 1000.)

    def test_structural_availability_does_not_relax_fixed_class_support(self):
        from agent import round5_classification_host as host
        from agent.round4_classification_candidate import coverage
        fitting = self.state.gate.research_train.iloc[self.state.gate.fit_rows]
        context = {"classes": list(self.state.gate.spec.classes), "features": list(self.state.gate.spec.features)}
        labels = fitting["target"].copy()
        labels[:] = "literal_z"
        labels.iloc[:4] = "literal_a"
        with self.assertWarnsRegex(UserWarning, "least populated class"):
            partition = coverage(fitting[list(self.state.gate.spec.features)], labels, context)
        self.assertTrue(partition["available"])
        self.assertEqual(host._support_reason(partition, context["classes"]), "ab_insufficient_class_support")

    def test_preflight_input_mutation_is_integrity_failure(self):
        from agent import round5_classification_host as host
        clock = Clock()
        self.state.budget = SimpleNamespace(deadline=1000., publication_buffer=5.)
        spec = self.state.gate.spec
        context = {"classes": list(spec.classes), "features": list(spec.features)}

        def supervisor(code, cwd, logs, step, *, timeout, max_memory_bytes):
            with (cwd / "request.joblib").open("ab") as stream:
                stream.write(b"changed input")
            return {"exit_code": 0, "timed_out": False, "memory_limit_exceeded": False}

        with patch("agent.execution.run_python", side_effect=supervisor), \
                patch.object(host.time, "monotonic", clock), self.assertRaisesRegex(ValueError, "request changed"):
            host.bounded_preflight(self.state, context)
        self.assertTrue(self.state._local_integrity_failed)


if __name__ == "__main__":
    unittest.main()
